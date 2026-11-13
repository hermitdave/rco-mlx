"""Budget-constrained discrete assignment: the DP solver and the optimizer.

Ported from ``src/search/quant.py`` in ISTA-DASLab/RCO (Apache-2.0):

    round_with_budget_dp       -> budget_dp
    budget_constrained_argmax  -> budget_dp (same knapsack DP, vectorized)
    optimize_projected_gumbel  -> ProjectedGumbelOptimizer

Two things are worth stating plainly, because they are the reason this module
is shaped the way it is.

1. ``budget_dp`` is a multiple-choice knapsack solver: pick exactly one option
   per group so the total bit cost hits ``target_avg_bits`` while maximizing a
   sum of per-option values. It is **exact** under a separable objective. This
   is the same DP the original applies to the optimizer's soft probabilities
   as its final rounding step, so it is a faithful port of that half of the
   algorithm.

2. ``ProjectedGumbelOptimizer`` is the half that handles the *non-decomposable*
   case -- where the objective (model loss) cannot be written as a sum over
   groups. The original couples it to a live PyTorch model through weight
   interpolation. This port instead takes a ``loss_and_grad`` callable that
   returns ``(loss, dloss/dprobs)``, so the same optimizer drives an MLX
   graph, a PyTorch graph, or anything else that can supply a gradient.

If you only have a per-group loss table (the practical MLX case, see
``objective.py``), skip the optimizer and call ``budget_dp`` directly: for a
separable objective the optimizer's answer is what the DP produces anyway.
"""

from __future__ import annotations

import math

import numpy as np

from .manifold import project_gradient, retraction, softmax, vector_transport

__all__ = [
    "budget_dp",
    "values_from_losses",
    "ProjectedGumbelOptimizer",
    "Adam",
]

_NEG_INF = -np.inf


def budget_dp(
    values: np.ndarray,
    fracs: np.ndarray,
    actual_bits: np.ndarray,
    target_avg_bits: float,
    resolution: int = 10000,
) -> np.ndarray:
    """Exact multiple-choice knapsack under a bit budget.

    Parameters
    ----------
    values : (n_groups, n_options)
        Proxy value of each option (higher is better). The original passes
        ``log(prob)`` from the optimizer; a loss table passes ``-loss``.
    fracs : (n_groups,)
        Parameter fraction of each group, ``group_params / total_params``.
        These weight both the cost and the objective, so a group's share of
        the model is how its bits are counted.
    actual_bits : (n_options,)
        Effective bits per weight for each option. Use the *effective* width
        (including scale/bias overhead), not the nominal one -- otherwise the
        budget and the real file size disagree.
    target_avg_bits : float
        Target weighted-average bits per weight across the whole model.
    resolution : int
        Discretization of the budget. The DP is O(n_groups * n_options *
        budget), so a coarse resolution is much cheaper and, at the sizes
        that matter, indistinguishable.

    Returns
    -------
    (n_groups,) int array of chosen option indices.
    """
    values = np.asarray(values, dtype=np.float64)
    fracs = np.asarray(fracs, dtype=np.float64)
    actual_bits = np.asarray(actual_bits, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError(f"values must be 2-D, got shape {values.shape}")
    n_groups, n_options = values.shape

    if fracs.shape != (n_groups,):
        raise ValueError(f"fracs must be ({n_groups},), got {fracs.shape}")
    if actual_bits.shape != (n_options,):
        raise ValueError(f"actual_bits must be ({n_options},), got {actual_bits.shape}")
    if np.isnan(values).any():
        raise ValueError("values contains NaN entries")
    # -inf is the documented way to forbid an option for a group (higher is
    # better, so -inf is never chosen while any finite option is reachable).

    budget = int(round(target_avg_bits * resolution))
    if budget <= 0:
        raise ValueError(f"target_avg_bits={target_avg_bits} gives a non-positive budget")

    costs = np.round(np.outer(fracs, actual_bits) * resolution).astype(np.int64)
    if (costs < 0).any():
        raise ValueError("costs must be non-negative")

    dp = np.full(budget + 1, _NEG_INF, dtype=np.float64)
    dp[0] = 0.0
    choice = np.full((n_groups, budget + 1), -1, dtype=np.int64)

    for g in range(n_groups):
        new_dp = np.full(budget + 1, _NEG_INF, dtype=np.float64)
        for k in range(n_options):
            c = int(costs[g, k])
            if c > budget:
                continue
            v = values[g, k]
            if not np.isfinite(v):
                continue  # forbidden option (masked with -inf)
            cand = dp[: budget + 1 - c] + v
            seg = new_dp[c:]
            better = cand > seg
            seg[better] = cand[better]
            choice[g, c:][better] = k
        dp = new_dp

    finite = np.isfinite(dp)
    if not finite.any():
        # Report the cheapest option set that respects forbidden (-inf)
        # options, not the raw minimum -- with an expert bit floor the raw
        # minimum can sit far below the budget while the true minimum is
        # above it, which reads as a contradiction.
        per_group_min = np.where(np.isfinite(values), costs, np.inf).min(axis=1)
        if not np.isfinite(per_group_min).all():
            raise ValueError(
                f"No feasible assignment at target_avg_bits={target_avg_bits}: "
                f"{int((~np.isfinite(per_group_min)).sum())} groups have no "
                f"allowed option (all masked)."
            )
        raise ValueError(
            f"No feasible assignment at target_avg_bits={target_avg_bits} "
            f"(budget={budget} units at resolution={resolution}). The cheapest "
            f"option set respecting forbidden options costs "
            f"{int(per_group_min.sum())} units."
        )

    # Pick the highest-value reachable cost at or under budget.
    #
    # The previous behaviour restricted the search to the LARGEST reachable
    # cost, on the theory that spending the full budget is always best. That
    # is false: the exact-cost constraint at the top of the budget can force a
    # bad combination. Measured on LFM2.5-8B-A1B at target 4.25, the DP chose
    # cost 42500 (total loss 2.20e-3) when cost 42419 gave 6.07e-4 -- 3.6x
    # better for 0.008 bpw less, and better than the 4.0 bpw run's 7.74e-4.
    # The knapsack optimum is the max over all j <= budget.
    finite = np.isfinite(dp)
    best_value = float(dp[finite].max())
    # Among the costs achieving the best value, take the largest: when two
    # assignments tie on quality, spend the budget rather than undershoot it.
    best_j = int(np.max(np.nonzero(finite & (dp >= best_value - 1e-12))[0]))

    assignment = np.zeros(n_groups, dtype=np.int64)
    j = best_j
    for g in range(n_groups - 1, -1, -1):
        k = int(choice[g, j])
        assignment[g] = k
        j -= int(costs[g, k])

    return assignment


def values_from_losses(losses: np.ndarray, transform: str = "neg") -> np.ndarray:
    """Convert a per-group loss table into a value table (higher is better).

    ``neg``  : ``-loss``. Minimizes the sum of losses. This is the additive
                proxy, and it is what the greedy allocator in
                ``streaming-mixed-precision-quantization`` does -- RCO's DP
                spends the same budget strictly better than greedy because it
                solves the knapsack exactly instead of taking the local best.
    ``log``  : ``-log(loss)``. Minimizes the product of losses, which weights
                balance more heavily than the sum does.

    Whichever you pick, the assumption is separability: the loss of a
    combination is the sum of its parts. That is exactly the assumption RCO's
    paper was written to avoid, so treat a table-driven result as a strong
    heuristic, not the true objective. The non-separable case needs
    ``ProjectedGumbelOptimizer``.
    """
    losses = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(losses).all():
        raise ValueError("losses contains non-finite entries")

    if transform == "neg":
        return -losses
    if transform == "log":
        return -np.log(np.maximum(losses, 1e-12))
    raise ValueError(f"unknown transform {transform!r}; expected 'neg' or 'log'")


class Adam:
    """Minimal Adam, matching the optimizer state RCO manipulates.

    Exists so ``vector_transport`` has an ``exp_avg`` with the same semantics
    as ``torch.optim.Adam``. Only the first moment is transported.
    """

    def __init__(self, shape: tuple[int, ...], lr: float = 0.05, betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.exp_avg = np.zeros(shape, dtype=np.float64)
        self.exp_avg_sq = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def step(self, grad: np.ndarray, alpha: np.ndarray) -> None:
        self.t += 1
        self.exp_avg = self.b1 * self.exp_avg + (1.0 - self.b1) * grad
        self.exp_avg_sq = self.b2 * self.exp_avg_sq + (1.0 - self.b2) * (grad * grad)
        m_hat = self.exp_avg / (1.0 - self.b1**self.t)
        v_hat = self.exp_avg_sq / (1.0 - self.b2**self.t)
        alpha -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ProjectedGumbelOptimizer:
    """Riemannian Adam over the budget manifold, with Gumbel-STE forward.

    Each step:

      1. Draw Gumbel noise, form ``noisy = (alpha + noise) / tau``.
      2. Solve the knapsack on ``noisy`` to get a budget-feasible hard
         assignment, and form the straight-through estimate
         ``probs = hard_onehot - soft.detach() + soft``.
      3. Forward pass through ``loss_and_grad(probs)``.
      4. Backprop through the STE to ``alpha``.
      5. Project the gradient onto the budget tangent plane.
      6. Adam step, retract to the manifold, transport the moment.

    Temperature anneals from ``tau_init`` to ``tau_min``, collapsing the soft
    distribution onto the hard assignment. No budget penalty and no Lagrangian
    multiplier are involved: the budget holds by construction.
    """

    def __init__(
        self,
        n_groups: int,
        n_options: int,
        fracs: np.ndarray,
        actual_bits: np.ndarray,
        target_avg_bits: float,
        lr: float = 0.05,
        tau_init: float = 1.0,
        tau_min: float = 0.05,
        resolution: int = 10000,
        seed: int | None = 0,
    ):
        self.n_groups = n_groups
        self.n_options = n_options
        self.fracs = np.asarray(fracs, dtype=np.float64)
        self.actual_bits = np.asarray(actual_bits, dtype=np.float64)
        self.target = float(target_avg_bits)
        self.resolution = resolution

        if self.fracs.shape != (n_groups,):
            raise ValueError(f"fracs must be ({n_groups},), got {self.fracs.shape}")
        if self.actual_bits.shape != (n_options,):
            raise ValueError(f"actual_bits must be ({n_options},), got {self.actual_bits.shape}")

        self.rng = np.random.default_rng(seed)
        self.alpha = np.zeros((n_groups, n_options), dtype=np.float64)
        self.opt = Adam(self.alpha.shape, lr=lr)
        self.tau_init = tau_init
        self.tau_min = tau_min

    # -- initialization -----------------------------------------------------

    def init_uniform(self) -> None:
        """Set every option's logit to 0, then retract onto the manifold."""
        self.alpha[:] = 0.0
        retraction(self.alpha, self.actual_bits, self.target, self.fracs, tol=1e-4)

    def init_from_values(self, values: np.ndarray) -> None:
        """Warm-start from a value table (e.g. ``-loss``), then retract."""
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self.alpha.shape:
            raise ValueError(f"values must be {self.alpha.shape}, got {values.shape}")
        self.alpha = values - values.mean(axis=-1, keepdims=True)
        retraction(self.alpha, self.actual_bits, self.target, self.fracs, tol=1e-4)

    def init_to_bits(self) -> None:
        """Binary-search a logit offset so the expected bits start on target."""
        lo, hi = -10.0, 10.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            logits = -mid * self.actual_bits
            probs = softmax(logits[None, :])
            if float((probs * self.actual_bits).sum()) > self.target:
                lo = mid
            else:
                hi = mid
        self.alpha[:] = (-0.5 * (lo + hi) * self.actual_bits)[None, :]
        retraction(self.alpha, self.actual_bits, self.target, self.fracs, tol=1e-4)

    # -- helpers ------------------------------------------------------------

    def expected_bits(self) -> float:
        probs = softmax(self.alpha, axis=-1)
        per_group = (probs * self.actual_bits[None, :]).sum(axis=1)
        return float((per_group * self.fracs).sum())

    def get_probs(self) -> np.ndarray:
        return softmax(self.alpha, axis=-1)

    def hard_assignment(self) -> np.ndarray:
        return budget_dp(
            np.log(np.maximum(self.get_probs(), 1e-10)),
            self.fracs,
            self.actual_bits,
            self.target,
            resolution=self.resolution,
        )

    # -- the step -----------------------------------------------------------

    def _gumbel(self) -> np.ndarray:
        u = self.rng.random(self.alpha.shape).clip(1e-20, 1.0 - 1e-20)
        return -np.log(-np.log(u) + 1e-20)

    def step(self, loss_and_grad) -> dict:
        """One projected-Gumbel step.

        ``loss_and_grad`` receives ``probs`` of shape ``(n_groups, n_options)``
        (a straight-through estimate, hard forward / soft backward) and returns
        ``(loss, dloss_dprobs)``. It is expected to be differentiable in the
        surrounding backend; the gradient supplied is used directly.
        """
        tau = self._tau
        noisy = (self.alpha + self._gumbel()) / tau
        soft = softmax(noisy, axis=-1)

        hard_idx = budget_dp(
            noisy, self.fracs, self.actual_bits, self.target, resolution=self.resolution
        )
        hard_onehot = np.eye(self.n_options, dtype=np.float64)[hard_idx]
        probs = hard_onehot - soft + soft  # forward hard, backward soft

        loss, grad_probs = loss_and_grad(probs)
        grad_probs = np.asarray(grad_probs, dtype=np.float64)

        # STE backward: dsoft/dalpha = (1/tau) * (diag(soft) - soft soft^T)
        weighted = (grad_probs * soft).sum(axis=-1, keepdims=True)
        grad_alpha = (grad_probs - weighted) / tau

        grad_alpha, coeff, raw_norm, proj_norm = project_gradient(
            self.alpha, grad_alpha, self.actual_bits, self.fracs
        )

        self.opt.step(grad_alpha, self.alpha)
        budget_before = self.expected_bits()
        achieved = retraction(self.alpha, self.actual_bits, self.target, self.fracs, tol=1e-4)
        vector_transport(self.opt.exp_avg, self.alpha, self.actual_bits, self.fracs)

        probs_now = self.get_probs()
        max_probs = probs_now.max(axis=1)
        entropy = float(-(probs_now * np.log(probs_now + 1e-10)).sum(axis=1).mean())

        return {
            "loss": float(loss),
            "avg_bits": achieved,
            "budget_drift": abs(budget_before - achieved),
            "tau": tau,
            "grad_norm": float(np.linalg.norm(grad_alpha)),
            "proj_coeff": coeff,
            "raw_norm": raw_norm,
            "proj_norm": proj_norm,
            "entropy": entropy,
            "n_decided": int((max_probs > 0.9).sum()),
            "n_hard": int((max_probs > 0.99).sum()),
        }

    def run(self, loss_and_grad, n_steps: int = 400, callback=None) -> tuple[np.ndarray, list[dict]]:
        """Optimize for ``n_steps`` and return ``(probs, history)``."""
        if callback is not None:
            callback("start", {"avg_bits": self.expected_bits(), "target": self.target})

        history = []
        for step in range(n_steps):
            progress = step / max(n_steps - 1, 1)
            self._tau = max(self.tau_min, self.tau_init * (self.tau_min / self.tau_init) ** progress)

            metrics = self.step(loss_and_grad)
            metrics["step"] = step
            history.append(metrics)

            if callback is not None:
                callback("step", metrics)

        return self.get_probs(), history

    _tau: float = 1.0
