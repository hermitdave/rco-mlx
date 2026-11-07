"""Per-(layer, projection) expert allocation for RCO.

The original pipeline forces *every* routed expert in the model to one
bitwidth (adapted from the target bpw). That constraint exists because the
MLX loaders stack experts into a single ``switch_mlp`` tensor per
(layer, projection), so all experts *within* one stacked tensor must share
a bitwidth -- ``ValueError: [stack] All arrays must have the same shape``
otherwise. It does **not** require gate, up and down projections, or
different layers, to share a bitwidth: those are separate stacked tensors.

This module implements the finer granularity:

* :func:`expert_group_key` -- the sanitize-relevant identity of an expert
  tensor: ``(layer, projection)``. Every expert index inside that group
  must use the same bits; nothing outside the group is constrained.
* :func:`enforce_group_uniformity` -- collapse a (possibly mixed) DP
  assignment to one bitwidth per group, by majority vote.
* :func:`fill_missing_expert_groups` -- expert groups absent from the
  assignment (e.g. DB rows clipped / params=0 fallbacks) inherit the
  majority bitwidth of their layer, else a caller-supplied fallback.
* :func:`apply_gguf_recipe` -- the measured UD-IQ2_XS recipe for
  Qwen3.8-Flash-Next (``qwen4_exp``, 48 layers), extracted from the GGUF
  tensor-info headers without downloading the weights. At the pipeline's
  single group size (64) the IQ2_XXS (2.06 bpw) and IQ2_S (2.56 bpw)
  gate/up tiers both map to 2-bit (2.5 bpw effective); the down-projection
  tiers map to 4-bit (IQ4_NL, 4.5 bpw), 2-bit (the 2.25 bpw tier) and
  6-bit for the single Q5_0 layer, preserving the GGUF's protection rank.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "is_expert_name",
    "expert_group_key",
    "group_expert_names",
    "enforce_group_uniformity",
    "fill_missing_expert_groups",
    "uniform_expert_bits",
    "gguf_recipe_bits",
    "apply_gguf_recipe",
    "mask_values_below_min_bits",
    "GGUF_GATE_UP_XXS_LAYERS",
    "GGUF_DOWN_IQ4_LAYERS",
    "GGUF_DOWN_225_LAYERS",
    "GGUF_DOWN_Q5_LAYERS",
]

# HF per-expert names: ...layers.{L}.mlp.experts.{E}.{projection}
_HF_EXPERT_RE = re.compile(
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.([A-Za-z0-9_]+)"
)
# MLX stacked names: ...switch_mlp.{projection} (single tensor per layer)
_MLX_STACKED_RE = re.compile(
    r"layers\.(\d+)\..*switch_(?:mlp|glu)\.([A-Za-z0-9_]+)"
)
# Some writers use feed_forward. instead of mlp. for the stacked tensor.
_MLX_FEED_FORWARD_RE = re.compile(
    r"layers\.(\d+)\.feed_forward\.([A-Za-z0-9_]+)"
)

# --- Measured UD-IQ2_XS recipe (Qwen3.8-Flash-Next, qwen4_exp) -----------
# Parsed from the GGUF tensor-info headers (shards 00002/00003):
#   ffn_gate_exps / ffn_up_exps: IQ2_XXS (2.06 bpw) on 8 layers,
#                                IQ2_S   (2.56 bpw) on the other 40
#   ffn_down_exps:               IQ4_NL  (4.50 bpw) on 24 layers,
#                                type-42 (2.25 bpw) on 23 layers,
#                                Q5_0    (5.50 bpw) on layer 4 only
GGUF_GATE_UP_XXS_LAYERS = frozenset({0, 1, 2, 7, 8, 9, 10, 11})
GGUF_DOWN_IQ4_LAYERS = frozenset(
    {0, 2, 3, 5, 6, 7, 12, 15, 16, 17, 18, 21, 22, 24, 25, 26, 27, 28,
     32, 37, 38, 39, 40, 47}
)
GGUF_DOWN_225_LAYERS = frozenset(
    {1, 8, 9, 10, 11, 13, 14, 19, 20, 23, 29, 30, 31, 33, 34, 35, 36,
     41, 42, 43, 44, 45, 46}
)
GGUF_DOWN_Q5_LAYERS = frozenset({4})
GGUF_NUM_LAYERS = 48


def is_expert_name(name: str) -> bool:
    """True for routed-expert tensors only.

    Excludes the shared expert (``shared_expert``) and the MoE router
    (``mlp.gate``) -- neither is stacked into ``switch_mlp``, so neither
    belongs in an expert group.
    """
    low = name.lower()
    if "shared_expert" in low or "shared" in low:
        return False
    if low.endswith(".mlp.gate") or low.endswith(".switch_mlp.gate"):
        return False
    if low.endswith(".mlp.gate.weight") or low.endswith(".switch_mlp.gate.weight"):
        return False
    return bool(
        _HF_EXPERT_RE.search(name)
        or _MLX_STACKED_RE.search(name)
        or _MLX_FEED_FORWARD_RE.search(name)
    )


def expert_group_key(name: str) -> Optional[Tuple[int, str]]:
    """Return ``(layer, projection)`` for an expert tensor, else ``None``.

    All experts inside one group must share a bitwidth (sanitize stacking);
    the group itself is the finest granularity the loader requires.
    """
    if not is_expert_name(name):
        return None
    m = _HF_EXPERT_RE.search(name)
    if m:
        return int(m.group(1)), m.group(3)
    m = _MLX_STACKED_RE.search(name)
    if m:
        return int(m.group(1)), m.group(2)
    m = _MLX_FEED_FORWARD_RE.search(name)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def group_expert_names(names: Iterable[str]) -> Dict[Tuple[int, str], List[str]]:
    """Bucket expert tensor names by their ``(layer, projection)`` group."""
    groups: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    for name in names:
        key = expert_group_key(name)
        if key is not None:
            groups[key].append(name)
    return dict(groups)


def _majority_bits(bits: List[int]) -> int:
    """Most common bitwidth; ties break toward the higher (safer) width."""
    counts = Counter(bits)
    top = max(counts.values())
    return max(b for b, c in counts.items() if c == top)


def enforce_group_uniformity(
    assignment: Dict[str, int],
    names: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, int], int]:
    """Force one bitwidth per (layer, projection) expert group.

    Returns ``(new_assignment, n_overridden)``. Non-expert entries are
    untouched. This is the sanitize-compatibility invariant: every expert
    index in a group gets the group's majority bitwidth.
    """
    source = list(names) if names is not None else list(assignment)
    groups = group_expert_names(source)
    new_assignment = dict(assignment)
    n_overridden = 0
    for members in groups.values():
        present = [new_assignment[m] for m in members if m in new_assignment]
        if not present:
            continue
        chosen = _majority_bits(present)
        for member in members:
            if member in new_assignment and new_assignment[member] != chosen:
                new_assignment[member] = chosen
                n_overridden += 1
    return new_assignment, n_overridden


def fill_missing_expert_groups(
    assignment: Dict[str, int],
    names: Iterable[str],
    fallback_bits: int,
) -> Tuple[Dict[str, int], int]:
    """Add expert groups that have no assignment entry yet.

    A group with no assigned member inherits the majority bitwidth already
    assigned to other expert projections in the same layer; if the layer
    has no assigned expert at all, it falls back to ``fallback_bits``
    (typically :func:`uniform_expert_bits` for the target bpw).
    Returns ``(new_assignment, n_filled)``.
    """
    new_assignment = dict(assignment)
    groups = group_expert_names(names)
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for (layer, _proj), members in groups.items():
        for member in members:
            if member in new_assignment:
                by_layer[layer].append(new_assignment[member])
    n_filled = 0
    for (layer, _proj), members in groups.items():
        if any(m in new_assignment for m in members):
            continue
        layer_bits = by_layer.get(layer, [])
        chosen = _majority_bits(layer_bits) if layer_bits else int(fallback_bits)
        for member in members:
            new_assignment[member] = chosen
        n_filled += len(members)
    return new_assignment, n_filled


def uniform_expert_bits(target_bpw: float) -> int:
    """The legacy single expert bitwidth for a target bpw."""
    if target_bpw <= 2.5:
        return 2
    if target_bpw <= 3.5:
        return 3
    if target_bpw <= 4.5:
        return 4
    if target_bpw <= 5.5:
        return 5
    return 6


def _normalise_projection(projection: str) -> str:
    proj = projection.lower()
    if proj in ("down_proj", "down", "w2", "o_proj"):
        return "down"
    return "gate_up"


def gguf_recipe_bits(layer: int, projection: str) -> int:
    """MLX bits (group size 64) replicating UD-IQ2_XS for one expert group.

    Gate/up: 2-bit on every layer (IQ2_XXS 2.06 and IQ2_S 2.56 both round
    to 2-bit at gs64, 2.5 bpw effective).
    Down: 4-bit on the 24 IQ4_NL layers, 2-bit on the 23 layers at
    2.25 bpw, and 6-bit on layer 4 (Q5_0) so its extra protection rank
    survives the mapping.
    """
    if not 0 <= layer < GGUF_NUM_LAYERS:
        raise ValueError(
            f"GGUF recipe is measured for {GGUF_NUM_LAYERS} layers, got layer {layer}"
        )
    if _normalise_projection(projection) == "gate_up":
        return 2
    if layer in GGUF_DOWN_Q5_LAYERS:
        return 6
    if layer in GGUF_DOWN_IQ4_LAYERS:
        return 4
    if layer in GGUF_DOWN_225_LAYERS:
        return 2
    raise AssertionError(f"layer {layer} missing from GGUF down-projection table")


def apply_gguf_recipe(
    assignment: Dict[str, int],
    names: Iterable[str],
    model_type: Optional[str] = None,
) -> Tuple[Dict[str, int], int]:
    """Override expert groups with the measured UD-IQ2_XS recipe.

    Only valid for the model it was measured on (``qwen4_exp``,
    48 layers). Returns ``(new_assignment, n_overridden)``.
    """
    if model_type is not None and model_type != "qwen4_exp":
        raise ValueError(
            "GGUF recipe was measured on qwen4_exp (Qwen3.8-Flash-Next); "
            f"refusing to apply it to model_type={model_type!r}"
        )
    new_assignment = dict(assignment)
    n_overridden = 0
    for name in names:
        key = expert_group_key(name)
        if key is None:
            continue
        layer, projection = key
        bits = gguf_recipe_bits(layer, projection)
        if new_assignment.get(name) != bits:
            new_assignment[name] = bits
            n_overridden += 1
    return new_assignment, n_overridden


def mask_values_below_min_bits(
    values: np.ndarray,
    bitwidths: Sequence[int] | np.ndarray,
    group_is_expert: Sequence[bool],
    min_bits: int,
) -> np.ndarray:
    """Forbid expert groups from choosing options below ``min_bits`` in the DP.

    Returns a copy of the value table with ``-inf`` in every disallowed
    (expert group, option) cell. Unlike a post-hoc override, masking before
    ``budget_dp`` keeps the assignment budget-feasible: the DP compensates
    for the more expensive experts by lowering non-expert widths. If even
    the cheapest allowed option set exceeds the target, ``budget_dp``
    raises its infeasibility error -- the correct signal that the floor
    and the target are incompatible.
    """
    values_arr = np.asarray(values, dtype=np.float64)
    widths = np.asarray(bitwidths)
    if values_arr.ndim != 2 or values_arr.shape[0] != len(group_is_expert):
        raise ValueError(
            f"values must be (n_groups, n_options) with n_groups == "
            f"len(group_is_expert); got {values_arr.shape} vs {len(group_is_expert)}"
        )
    masked = values_arr.copy()
    if min_bits <= int(widths.min()):
        return masked
    disallowed = widths < min_bits
    for g, is_expert in enumerate(group_is_expert):
        if is_expert:
            masked[g, disallowed] = -np.inf
    return masked
