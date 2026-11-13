"""Error-DB expert aggregation must match every adapter's key style."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_error_db_perblock.py"
_spec = importlib.util.spec_from_file_location("build_error_db_perblock", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


@pytest.mark.parametrize(
    "key,expected",
    [
        # OLMoE: model.layers.N.mlp.experts.E.{gate,up,down}_proj
        ("model.layers.0.mlp.experts.0.gate_proj.weight",
         "model.layers.0.mlp.experts.gate_proj"),
        ("model.layers.15.mlp.experts.63.down_proj.weight",
         "model.layers.15.mlp.experts.down_proj"),
        # Qwen3.5 / Qwen4-Exp: model.language_model.layers.N...
        ("model.language_model.layers.3.mlp.experts.17.gate_up_proj.weight",
         "model.language_model.layers.3.mlp.experts.gate_up_proj"),
        # LFM2/LFM2.5: feed_forward.experts with SwiGLU w1/w2/w3 names.
        # The projection class must include digits -- `w1` is not `[a-z_]+`.
        ("model.layers.2.feed_forward.experts.0.w1.weight",
         "model.layers.2.feed_forward.experts.w1"),
        ("model.layers.21.feed_forward.experts.31.w3.weight",
         "model.layers.21.feed_forward.experts.w3"),
    ],
)
def test_expert_row_key_aggregates(key, expected):
    assert _mod.expert_row_key(key) == expected


def test_expert_row_key_ignores_non_experts():
    assert _mod.expert_row_key("model.layers.0.self_attn.q_proj.weight") is None
    assert _mod.expert_row_key(
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight"
    ) is None
