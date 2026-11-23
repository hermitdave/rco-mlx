"""LossTable / objective tests.

The important checks: effective bits include scale overhead, the target budget
is respected, and the MLX width constraints are enforced at the boundary.
"""

import numpy as np
import pytest

from rco_mlx.objective import (
    MLX_SUPPORTED_BITS,
    MLX_SUPPORTED_GROUP_SIZES,
    LossTable,
    effective_bits,
)


def test_effective_bits_includes_scale_overhead():
    # 2-bit with bf16 scales at group 64: 2 + 16/64 = 2.25
    assert effective_bits(2, 64) == pytest.approx(2.25)
    assert effective_bits(4, 64) == pytest.approx(4.25)
    assert effective_bits(8, 128) == pytest.approx(8.125)


def test_effective_bits_rejects_unsupported_widths():
    with pytest.raises(ValueError, match="not supported by MLX"):
        effective_bits(1, 64)
    with pytest.raises(ValueError, match="not supported by MLX"):
        effective_bits(7, 64)
    with pytest.raises(ValueError, match="not supported by MLX"):
        effective_bits(4, 16)


def test_supported_constants_match_mlx():
    assert MLX_SUPPORTED_BITS == (2, 3, 4, 5, 6, 8)
    assert MLX_SUPPORTED_GROUP_SIZES == (32, 64, 128)


def _toy_table():
    """4 groups, 3 options, with a clear sensitivity ordering."""
    losses = np.array(
        [
            [0.50, 0.30, 0.10],  # group 0: very sensitive
            [0.45, 0.28, 0.12],  # group 1: very sensitive
            [0.20, 0.15, 0.10],  # group 2: insensitive
            [0.18, 0.14, 0.11],  # group 3: insensitive
        ]
    )
    fracs = np.array([0.25, 0.25, 0.25, 0.25])
    bitwidths = np.array([2, 4, 8])
    return LossTable(losses=losses, fracs=fracs, bitwidths=bitwidths, group_size=64)


def test_loss_table_assignment_respects_the_budget():
    table = _toy_table()
    target = 4.5
    assignment = table.assignment(target)

    achieved = table.achieved_bpw(assignment)
    assert achieved <= target + 0.26  # one group's worth of rounding slack
    assert achieved > target - 0.5


def test_loss_table_spends_bits_where_they_matter():
    """Sensitive groups should get more bits than insensitive ones."""
    table = _toy_table()
    assignment = table.assignment(target_avg_bits=5.0)

    sensitive = assignment[:2]
    insensitive = assignment[2:]
    assert sensitive.min() >= insensitive.max()


def test_loss_table_report_shape():
    table = _toy_table()
    report = table.report(target_avg_bits=5.0)

    assert report["n_groups"] == 4
    assert report["group_size"] == 64
    assert set(report["bit_histogram"]) <= set([2, 4, 8])
    assert abs(report["achieved_bpw"] - report["target_bpw"]) < 0.5


def test_loss_table_rejects_unsupported_bitwidth():
    with pytest.raises(ValueError, match="not supported by MLX"):
        LossTable(
            losses=np.zeros((2, 2)),
            fracs=np.array([0.5, 0.5]),
            bitwidths=np.array([1, 4]),
        )


def test_loss_table_rejects_bad_fracs():
    with pytest.raises(ValueError):
        LossTable(
            losses=np.zeros((2, 2)),
            fracs=np.array([0.0, 0.0]),
            bitwidths=np.array([2, 4]),
        )


def test_loss_table_rejects_non_finite_losses():
    with pytest.raises(ValueError, match="non-finite"):
        LossTable(
            losses=np.array([[np.nan, 0.1], [0.1, 0.2]]),
            fracs=np.array([0.5, 0.5]),
            bitwidths=np.array([2, 4]),
        )


def test_actual_bits_tracks_the_bitwidths():
    table = _toy_table()
    bits = table.actual_bits()
    assert np.allclose(bits, [2.25, 4.25, 8.25])


def test_affine_scale_bits_double_the_group_overhead():
    """affine stores a scale AND a bias per group; symmetric stores one scale.

    Charging 16 bits/group for an affine build under-counts the real cost by
    0.25 bpw at gs64, so the DP overspends and the artifact comes out larger
    than its reported target.
    """
    kw = dict(losses=np.array([[0.25, 0.05], [0.95, 0.05]]),
              fracs=np.array([0.9, 0.1]), bitwidths=np.array([2, 4]), group_size=64)
    sym = LossTable(scale_bits=16, **kw).actual_bits()
    aff = LossTable(scale_bits=32, **kw).actual_bits()
    assert np.allclose(sym, [2.25, 4.25])
    assert np.allclose(aff, [2.5, 4.5])
    assert np.allclose(aff - sym, 16 / 64)


def test_values_carry_the_parameter_fraction():
    """Regression: budget_dp sums values plain, so the caller must weight them.

    Without the fracs factor the DP minimises an unweighted mean over tensors
    while spending a parameter-weighted budget -- a 256-parameter norm gets the
    same vote as a 1.3 B-parameter embedding matrix.
    """
    table = _toy_table()
    assert np.allclose(table.values(), table.fracs[:, None] * (-table.losses))


def test_weighted_objective_spends_bits_on_the_big_group():
    """A 0.9-fraction group must win the budget over a tiny high-loss one.

    Unweighted, group 1 (loss 0.95 -> 0.05) looks far more deserving than group
    0 (0.25 -> 0.05). Parameter-weighted, group 0 carries 90% of the parameters
    and group 1's total contribution is 0.095 against group 0's 0.225.

    Budget arithmetic (gs64, scale_bits=16 -> 2-bit costs 2.25, 4-bit 4.25):
    both-2-bit 2.250, group0-only 4.050, group1-only 2.450, both-4-bit 4.250.
    A 4.15 target therefore affords exactly one upgrade.
    """
    table = LossTable(
        losses=np.array([[0.25, 0.05], [0.95, 0.05]]),
        fracs=np.array([0.9, 0.1]),
        bitwidths=np.array([2, 4]),
        group_size=64,
        scale_bits=16,
    )
    assignment = table.assignment(4.15)
    assert assignment[0] == 4, "the 0.9-fraction group must get the bits"
    assert assignment[1] == 2, "the tiny high-loss group must not"


def test_log_transform_changes_the_allocation():
    """The transform is a real modelling choice, not a no-op."""
    table_neg = LossTable(
        losses=np.array([[0.9, 0.1], [0.1, 0.05]]),
        fracs=np.array([0.5, 0.5]),
        bitwidths=np.array([2, 4]),
        group_size=64,
        transform="neg",
    )
    table_log = LossTable(
        losses=np.array([[0.9, 0.1], [0.1, 0.05]]),
        fracs=np.array([0.5, 0.5]),
        bitwidths=np.array([2, 4]),
        group_size=64,
        transform="log",
    )
    assert not np.array_equal(table_neg.values(), table_log.values())
