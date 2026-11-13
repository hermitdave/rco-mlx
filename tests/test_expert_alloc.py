"""Tests for per-(layer, projection) expert allocation."""

import pytest

from rco_mlx.expert_alloc import (
    GGUF_DOWN_225_LAYERS,
    GGUF_DOWN_IQ4_LAYERS,
    apply_gguf_recipe,
    enforce_group_uniformity,
    expert_group_key,
    fill_missing_expert_groups,
    gguf_recipe_bits,
    is_expert_name,
    mask_values_below_min_bits,
    uniform_expert_bits,
)
from rco_mlx.search import budget_dp

import numpy as np

HF_GATE = "model.language_model.layers.{l}.mlp.experts.{e}.gate_up_proj"
HF_DOWN = "model.language_model.layers.{l}.mlp.experts.{e}.down_proj"
MLX_GATE = "language_model.model.layers.{l}.mlp.switch_mlp.gate_proj"
MLX_DOWN = "language_model.model.layers.{l}.mlp.switch_mlp.down_proj"


def test_group_key_hf_per_expert_names():
    assert expert_group_key(HF_GATE.format(l=3, e=17)) == (3, "gate_up_proj")
    assert expert_group_key(HF_DOWN.format(l=3, e=17)) == (3, "down_proj")


def test_group_key_lfm2_feed_forward_experts():
    """LFM2/LFM2.5 keep experts under feed_forward.experts with w1/w2/w3 names."""
    assert expert_group_key(
        "model.layers.2.feed_forward.experts.0.w1") == (2, "w1")
    assert expert_group_key(
        "model.layers.2.feed_forward.experts.7.w2") == (2, "w2")
    assert expert_group_key(
        "model.layers.21.feed_forward.experts.31.w3") == (21, "w3")
    # the mlx_lm stacked form must land on the same (layer, projection) family
    assert expert_group_key(
        "model.layers.2.feed_forward.switch_mlp.gate_proj") == (2, "gate_proj")
    # dense layers have no experts
    assert not is_expert_name("model.layers.0.feed_forward.w1")


def test_group_key_mlx_stacked_names():
    assert expert_group_key(MLX_GATE.format(l=11)) == (11, "gate_proj")
    assert expert_group_key(MLX_DOWN.format(l=11)) == (11, "down_proj")


def test_shared_expert_and_router_are_not_experts():
    assert not is_expert_name(
        "model.language_model.layers.0.mlp.shared_expert.gate_up_proj"
    )
    assert not is_expert_name("model.language_model.layers.0.mlp.gate")
    assert expert_group_key("model.language_model.layers.0.mlp.gate") is None


def test_uniformity_is_within_group_not_across_projections():
    names = [HF_GATE.format(l=0, e=e) for e in range(4)] + [
        HF_DOWN.format(l=0, e=e) for e in range(4)
    ]
    assignment = {n: 2 for n in names[:4]}
    assignment.update({n: 4 for n in names[4:]})
    out, n_overridden = enforce_group_uniformity(assignment, names)
    assert n_overridden == 0  # gate at 2 and down at 4 is already legal
    assert out[names[0]] == 2 and out[names[4]] == 4


def test_uniformity_collapses_mixed_bits_within_a_group():
    names = [HF_GATE.format(l=5, e=e) for e in range(4)]
    assignment = {names[0]: 2, names[1]: 2, names[2]: 2, names[3]: 4}
    out, n_overridden = enforce_group_uniformity(assignment, names)
    assert n_overridden == 1
    assert all(out[n] == 2 for n in names)


def test_fill_missing_uses_same_layer_majority():
    names = [HF_GATE.format(l=2, e=e) for e in range(2)] + [
        HF_DOWN.format(l=2, e=e) for e in range(2)
    ]
    assignment = {n: 3 for n in names[:2]}
    out, n_filled = fill_missing_expert_groups(assignment, names, fallback_bits=4)
    assert n_filled == 2
    assert all(out[n] == 3 for n in names[2:])


def test_fill_missing_falls_back_when_layer_unassigned():
    names = [HF_DOWN.format(l=9, e=0)]
    out, n_filled = fill_missing_expert_groups({}, names, fallback_bits=4)
    assert out[names[0]] == 4 and n_filled == 1


def test_uniform_expert_bits_thresholds_unchanged():
    assert [uniform_expert_bits(t) for t in (2.0, 3.0, 4.0, 5.0, 6.0)] == [2, 3, 4, 5, 6]


def test_gguf_recipe_tables_cover_all_48_layers():
    assert len(GGUF_DOWN_IQ4_LAYERS) == 24
    assert len(GGUF_DOWN_225_LAYERS) == 23
    assert GGUF_DOWN_IQ4_LAYERS.isdisjoint(GGUF_DOWN_225_LAYERS)
    assert GGUF_DOWN_IQ4_LAYERS | GGUF_DOWN_225_LAYERS | {4} == set(range(48))


def test_gguf_recipe_bits():
    assert gguf_recipe_bits(0, "gate_up_proj") == 2
    assert gguf_recipe_bits(47, "gate_proj") == 2
    assert gguf_recipe_bits(0, "down_proj") == 4   # IQ4_NL tier
    assert gguf_recipe_bits(1, "down_proj") == 2   # 2.25 bpw tier
    assert gguf_recipe_bits(4, "down_proj") == 6   # Q5_0 tier, rank preserved
    assert gguf_recipe_bits(47, "down_proj") == 4
    with pytest.raises(ValueError):
        gguf_recipe_bits(48, "down_proj")


def test_apply_gguf_recipe_overrides_only_experts():
    names = [
        HF_GATE.format(l=1, e=0),
        HF_DOWN.format(l=1, e=0),
        HF_DOWN.format(l=0, e=0),
        "model.language_model.layers.0.self_attn.q_proj",
    ]
    assignment = {n: 8 for n in names}
    out, _ = apply_gguf_recipe(assignment, names, model_type="qwen4_exp")
    assert out[names[0]] == 2  # gate/up
    assert out[names[1]] == 2  # layer 1 down: 2.25 bpw tier
    assert out[names[2]] == 4  # layer 0 down: IQ4_NL tier
    assert out[names[3]] == 8  # attention untouched
    with pytest.raises(ValueError):
        apply_gguf_recipe(assignment, names, model_type="olmoe")


def test_min_bits_mask_forbids_only_expert_groups():
    values = np.array([[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]])
    masked = mask_values_below_min_bits(values, [2, 3, 4], [True, False], min_bits=3)
    assert np.isneginf(masked[0, 0])
    assert masked[0, 1:] .tolist() == [-2.0, -3.0]
    assert masked[1].tolist() == [-1.0, -2.0, -3.0]  # non-expert untouched
    # no-op when the floor is at or below the smallest option
    assert mask_values_below_min_bits(values, [2, 3, 4], [True, True], 2).tolist() == values.tolist()


def test_min_bits_dp_compensates_via_attention():
    # One expert group (90% of params, loves 2-bit) + one attention group
    # (10%, hates 2-bit). Unconstrained DP at a tight budget must give the
    # expert 2-bit; with a 3-bit floor it must instead drop attention.
    bitwidths = np.array([2, 3, 4, 8])
    actual = bitwidths + 0.25  # gs64 effective, pipeline convention
    losses = np.array([
        [0.10, 0.11, 0.12, 0.13],   # expert: nearly flat -> wants 2-bit
        [10.0, 1.0, 0.1, 0.01],     # attention: steep -> wants 8-bit
    ])
    fracs = np.array([0.9, 0.1])
    values = -losses

    idx = budget_dp(values, fracs, actual, target_avg_bits=2.9, resolution=1000)
    assert bitwidths[idx[0]] == 2  # unconstrained: expert at 2-bit

    masked = mask_values_below_min_bits(values, bitwidths, [True, False], min_bits=3)
    idx_f = budget_dp(masked, fracs, actual, target_avg_bits=3.3, resolution=1000)
    assert bitwidths[idx_f[0]] >= 3  # floor respected
    assert bitwidths[idx_f[1]] < bitwidths[idx[1]]  # attention pays for it

    # Infeasible floor+target must surface as the DP's error, not a silent act
    with pytest.raises(ValueError):
        budget_dp(masked, fracs, actual, target_avg_bits=3.0, resolution=1000)
