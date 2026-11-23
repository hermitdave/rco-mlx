"""Objective construction for the RCO allocator.

RCO needs two things per (group, option) pair:

    * a **cost**, in bits -- how much of the budget this choice consumes, and
    * a **value** -- how good this choice is.

Costs come from the bit width and the group's parameter fraction, and are
framework-independent. Values come from a *measurement*, and this module is the
place where that measurement is shaped into the table ``budget_dp`` expects.

The honest caveat
-----------------
A loss table assumes the objective **separates**: the loss of a whole-model
assignment is the sum of the per-group losses. RCO's paper exists because that
is false for a real model -- the loss is non-decomposable, which is why the
original drives the optimizer from a live forward pass.

So treat ``LossTable`` as the practical MLX path, not the theoretically correct
one. It is still a strict improvement over the greedy allocator it replaces,
because for a fixed separable objective the DP is *exact* where greedy is not.
When a differentiable backend is available, use ``ProjectedGumbelOptimizer``
instead and leave this module out of the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .search import values_from_losses

__all__ = ["LossTable", "effective_bits", "MLX_SUPPORTED_BITS", "MLX_SUPPORTED_GROUP_SIZES"]

# Verified against mlx 0.32.3 by probing mx.quantize directly. MLX rejects
# bits=1 and bits=7; 2 is the floor. There is no 1-bit rung, so the lowest
# EXL3/GGUF low-bit targets are unreachable in MLX regardless of method.
MLX_SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)
MLX_SUPPORTED_GROUP_SIZES = (32, 64, 128)


def effective_bits(bits: int, group_size: int, scale_bits: int = 16) -> float:
    """Effective bits per weight including per-group scale overhead.

    ``bits + scale_bits / group_size``. This is the number to feed the budget
    constraint -- using the nominal width instead understates the real cost by
    ~0.13-0.5 bpw depending on group size, and the resulting file is larger
    than the target.

    ``scale_bits`` defaults to 16 because MLX stores one bf16 scale (and, in
    affine mode, one bf16 bias) per group. Pass 8 for a symmetric format with
    no bias.
    """
    if bits not in MLX_SUPPORTED_BITS:
        raise ValueError(
            f"bits={bits} is not supported by MLX (supported: {MLX_SUPPORTED_BITS})"
        )
    if group_size not in MLX_SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"group_size={group_size} is not supported by MLX "
            f"(supported: {MLX_SUPPORTED_GROUP_SIZES})"
        )
    return bits + scale_bits / group_size


@dataclass
class LossTable:
    """A per-group, per-option loss table ready for ``budget_dp``.

    Parameters
    ----------
    losses : (n_groups, n_options)
        Measured loss for each (group, option). Lower is better. The reference
        column (usually the highest bit width) should be the near-lossless
        baseline the other columns are compared against.
    fracs : (n_groups,)
        Parameter fraction per group.
    bitwidths : (n_options,)
        Nominal bit width of each option.
    group_size : int
        Group size the losses were measured at.
    """

    losses: np.ndarray
    fracs: np.ndarray
    bitwidths: np.ndarray
    group_size: int = 64
    transform: str = "neg"
    #: Bits of side-channel state per group. 16 for symmetric (one bf16 scale),
    #: 32 for affine (scale + bias). ``effective_bits`` defaulted to 16 for both,
    #: so every affine build under-charged the budget by 0.25 bpw at gs64 and the
    #: emitted file came out larger than the reported target.
    scale_bits: int = 16

    def __post_init__(self) -> None:
        self.losses = np.asarray(self.losses, dtype=np.float64)
        self.fracs = np.asarray(self.fracs, dtype=np.float64)
        self.bitwidths = np.asarray(self.bitwidths, dtype=np.int64)

        if self.losses.ndim != 2:
            raise ValueError(f"losses must be 2-D, got {self.losses.shape}")
        n_groups, n_options = self.losses.shape
        if self.fracs.shape != (n_groups,):
            raise ValueError(f"fracs must be ({n_groups},), got {self.fracs.shape}")
        if self.bitwidths.shape != (n_options,):
            raise ValueError(f"bitwidths must be ({n_options},), got {self.bitwidths.shape}")
        if not np.isfinite(self.losses).all():
            raise ValueError("losses contains non-finite entries")
        if (self.fracs < 0).any() or self.fracs.sum() <= 0:
            raise ValueError("fracs must be non-negative with a positive sum")

        for bw in self.bitwidths:
            if bw not in MLX_SUPPORTED_BITS:
                raise ValueError(
                    f"bitwidth {bw} is not supported by MLX "
                    f"(supported: {MLX_SUPPORTED_BITS})"
                )

    @property
    def n_groups(self) -> int:
        return self.losses.shape[0]

    @property
    def n_options(self) -> int:
        return self.losses.shape[1]

    def actual_bits(self) -> np.ndarray:
        """Effective bits per weight per option, including scale overhead."""
        return np.array(
            [effective_bits(int(b), self.group_size, self.scale_bits) for b in self.bitwidths],
            dtype=np.float64,
        )

    def values(self) -> np.ndarray:
        """Value table (higher is better) for ``budget_dp``.

        Parameter-weighted: ``budget_dp`` accumulates a plain sum over groups, so
        each group's value must carry its own parameter fraction for the
        objective to be the param-weighted error that the bit budget is
        denominated in (``costs = fracs x bits``).

        Without the weighting the DP minimises an UNWEIGHTED mean over tensors
        while spending a parameter-weighted budget, so small tensors get the
        same vote as 1.3 B-parameter ones. Measured on Qwen3.8-27B Coder390 at
        4.5 bpw: the unweighted optimum is 31.8% worse on the param-weighted
        reconstruction error at the same budget, and loses to a near-uniform
        4/5-bit allocation. ``budget_dp``'s own docstring already promises that
        ``fracs`` weight the objective as well as the cost; this makes the code
        match it.
        """
        return self.fracs[:, None] * values_from_losses(self.losses, self.transform)

    def solve(self, target_avg_bits: float, resolution: int = 10000) -> np.ndarray:
        """Run the knapsack DP and return the chosen option index per group."""
        from .search import budget_dp

        return budget_dp(
            self.values(), self.fracs, self.actual_bits(), target_avg_bits, resolution
        )

    def assignment(self, target_avg_bits: float, resolution: int = 10000) -> np.ndarray:
        """Chosen nominal bit width per group."""
        return self.bitwidths[self.solve(target_avg_bits, resolution)]

    def achieved_bpw(self, assignment: np.ndarray) -> float:
        """Weighted-average effective bpw of a concrete assignment.

        ``assignment`` is a per-group **bit width** (what ``assignment()``
        returns), not an option index.
        """
        widths = np.asarray(assignment, dtype=np.int64)
        if widths.shape != (self.n_groups,):
            raise ValueError(
                f"assignment must be ({self.n_groups},), got {widths.shape}"
            )

        actual = self.actual_bits()
        by_width = {int(b): actual[i] for i, b in enumerate(self.bitwidths)}
        unknown = sorted(set(widths.tolist()) - set(by_width))
        if unknown:
            raise ValueError(
                f"assignment contains widths not in this table: {unknown} "
                f"(available: {sorted(by_width)})"
            )

        chosen = np.array([by_width[int(w)] for w in widths], dtype=np.float64)
        return float((chosen * self.fracs).sum())

    def report(self, target_avg_bits: float, resolution: int = 10000) -> Dict:
        """Solve and return a human-readable summary of the allocation."""
        idx = self.solve(target_avg_bits, resolution)
        bits = self.bitwidths[idx]
        counts = {int(b): int((bits == b).sum()) for b in sorted(set(bits.tolist()))}
        return {
            "target_bpw": float(target_avg_bits),
            "achieved_bpw": self.achieved_bpw(bits),
            "n_groups": self.n_groups,
            "bit_histogram": counts,
            "group_size": self.group_size,
            "transform": self.transform,
        }
