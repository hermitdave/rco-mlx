"""Budget DP tests.

The DP is checked against a brute-force enumeration of every assignment, so a
subtle indexing or cost bug cannot slip through. Budget feasibility is checked
independently, and the MLX width constraints are checked at the boundary.
"""

import itertools

import numpy as np
import pytest

from rco_mlx.search import ProjectedGumbelOptimizer, budget_dp, values_from_losses


def _brute_force(values, fracs, actual_bits, target, resolution=10000):
    """Replicate ``round_with_budget_dp``'s semantics exactly.

    The DP does *not* maximize value over all feasible budgets. It takes the
    largest reachable total cost ``best_j <= budget``, then the highest-value
    assignment among those costing exactly ``best_j``. The brute force has to
    match that, or it is checking a different objective.
    """
    n_groups, n_options = values.shape
    budget = int(round(target * resolution))
    costs = np.round(np.outer(fracs, actual_bits) * resolution).astype(np.int64)

    combos = list(itertools.product(range(n_options), repeat=n_groups))

    reachable = set()
    for combo in combos:
        total = sum(int(costs[g, combo[g]]) for g in range(n_groups))
        if total <= budget:
            reachable.add(total)
    best_j = max(reachable)

    best_value = -np.inf
    best_assignment = None
    for combo in combos:
        total = sum(int(costs[g, combo[g]]) for g in range(n_groups))
        if total != best_j:
            continue
        value = sum(values[g, combo[g]] for g in range(n_groups))
        if value > best_value:
            best_value = value
            best_assignment = combo

    return best_assignment, best_value


def test_budget_dp_matches_brute_force():
    rng = np.random.default_rng(0)
    n_groups, n_options = 4, 3
    values = rng.normal(size=(n_groups, n_options))
    fracs = rng.random(n_groups) + 0.1
    fracs /= fracs.sum()
    actual_bits = np.array([2.125, 3.125, 4.125])

    for target in [2.5, 3.0, 3.5, 4.0]:
        got = budget_dp(values, fracs, actual_bits, target)
        expected, _ = _brute_force(values, fracs, actual_bits, target)
        assert tuple(got) == tuple(expected), f"target={target}: {got} != {expected}"


def test_budget_dp_never_exceeds_the_budget():
    rng = np.random.default_rng(1)
    n_groups, n_options = 8, 4
    values = rng.normal(size=(n_groups, n_options))
    fracs = rng.random(n_groups) + 0.1
    fracs /= fracs.sum()
    actual_bits = np.array([2.125, 3.125, 4.125, 6.125])
    target = 3.5
    resolution = 10000

    assignment = budget_dp(values, fracs, actual_bits, target, resolution)

    costs = np.round(np.outer(fracs, actual_bits) * resolution).astype(np.int64)
    total = sum(int(costs[g, assignment[g]]) for g in range(n_groups))
    assert total <= int(round(target * resolution))


def test_budget_dp_prefers_the_full_budget_when_reachable():
    """If the budget is exactly reachable, the DP should use it."""
    fracs = np.array([0.5, 0.5])
    actual_bits = np.array([2.0, 4.0])
    values = np.array([[0.0, 0.0], [0.0, 0.0]])
    target = 3.0

    assignment = budget_dp(values, fracs, actual_bits, target, resolution=1000)
    # any assignment summing to 3.0 is optimal; all-4bit = 4.0 > budget, so
    # the answer must be one 2-bit and one 4-bit group
    assert sorted(assignment.tolist()) == [0, 1]


def test_budget_dp_raises_when_infeasible():
    fracs = np.array([1.0])
    actual_bits = np.array([4.0, 6.0, 8.0])
    values = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="No feasible assignment"):
        budget_dp(values, fracs, actual_bits, target_avg_bits=2.0)


def test_budget_dp_rejects_bad_shapes():
    with pytest.raises(ValueError, match="values must be 2-D"):
        budget_dp(np.zeros(4), np.array([1.0]), np.array([4.0]), 4.0)

    with pytest.raises(ValueError, match="fracs must be"):
        budget_dp(np.zeros((2, 3)), np.array([1.0, 1.0, 1.0]), np.array([4.0, 6.0, 8.0]), 4.0)


def test_budget_dp_rejects_non_finite_values():
    # NaN is rejected; -inf is the documented "forbidden option" marker and
    # must be accepted (see mask_values_below_min_bits in expert_alloc).
    values = np.array([[0.0, np.nan], [0.0, 1.0]])
    with pytest.raises(ValueError, match="NaN"):
        budget_dp(values, np.array([0.5, 0.5]), np.array([2.0, 4.0]), 3.0)

    idx = budget_dp(
        np.array([[0.0, -np.inf], [0.0, 1.0]]),
        np.array([0.5, 0.5]),
        np.array([2.0, 4.0]),
        3.0,
    )
    assert idx[0] == 0  # -inf option never chosen


def test_values_from_losses_sign_and_order():
    losses = np.array([[0.1, 0.5], [0.3, 0.2]])
    neg = values_from_losses(losses, "neg")
    assert np.allclose(neg, -losses)
    # lower loss -> higher value
    assert neg[0, 0] > neg[0, 1]
    assert neg[1, 1] > neg[1, 0]

    log = values_from_losses(losses, "log")
    assert np.allclose(log, -np.log(losses))


def test_values_from_losses_rejects_unknown_transform():
    with pytest.raises(ValueError, match="unknown transform"):
        values_from_losses(np.array([[0.1]]), "nope")


def test_optimizer_keeps_the_budget_every_step():
    """The optimizer's defining property: the budget holds by construction."""
    rng = np.random.default_rng(2)
    n_groups, n_options = 5, 3
    fracs = rng.random(n_groups) + 0.1
    fracs /= fracs.sum()
    actual_bits = np.array([2.125, 3.125, 4.125])
    target = 3.0

    opt = ProjectedGumbelOptimizer(
        n_groups, n_options, fracs, actual_bits, target, lr=0.05, seed=0
    )
    opt.init_to_bits()

    def loss_and_grad(probs):
        # a simple separable stand-in: prefer higher bit width
        grad = -actual_bits[None, :] * 0.01
        loss = float((probs * actual_bits[None, :]).sum())
        return loss, grad

    for _ in range(10):
        metrics = opt.step(loss_and_grad)
        # the budget holds after the retraction, which is the property that matters
        assert abs(metrics["avg_bits"] - target) < 1e-3
        # the pre-correction drift is a diagnostic, not a constraint
        assert metrics["budget_drift"] < 0.05


def test_optimizer_hard_assignment_respects_the_budget():
    rng = np.random.default_rng(3)
    n_groups, n_options = 6, 4
    fracs = rng.random(n_groups) + 0.1
    fracs /= fracs.sum()
    actual_bits = np.array([2.125, 3.125, 4.125, 6.125])
    target = 4.0

    opt = ProjectedGumbelOptimizer(
        n_groups, n_options, fracs, actual_bits, target, seed=0
    )
    opt.init_to_bits()

    assignment = opt.hard_assignment()
    costs = np.round(np.outer(fracs, actual_bits) * 10000).astype(np.int64)
    total = sum(int(costs[g, assignment[g]]) for g in range(n_groups))
    assert total <= int(round(target * 10000))


def test_optimizer_warm_start_from_values():
    n_groups, n_options = 3, 3
    fracs = np.array([0.2, 0.3, 0.5])
    actual_bits = np.array([2.0, 4.0, 8.0])
    values = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]])

    opt = ProjectedGumbelOptimizer(
        n_groups, n_options, fracs, actual_bits, target_avg_bits=4.0, seed=0
    )
    opt.init_from_values(values)
    assert abs(opt.expected_bits() - 4.0) < 1e-3
