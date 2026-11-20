"""Layer grouping for constraining layers to share a bitwidth.

Ported from ``src/grouping.py`` in ISTA-DASLab/RCO (Apache-2.0). This module
has no torch dependency in the original either, so the port is close to a
transcription.

Grouping is the knob that controls the *degrees of freedom* of the allocation.
More groups = more freedom = better fit to the budget, but a larger search
space. The default in RCO is one group per layer; the MoE variant makes one
group per expert.

Note the naming: RCO groups *layers* (``model.layers.N.mlp.gate_proj``). MLX
names modules differently (``backbone.layers.N.mixer.switch_mlp.fc1``). The
grouping logic is name-agnostic -- it matches whatever substring patterns you
give it -- so the same code works on both, but the patterns must be written
against the target naming scheme.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "LayerGroup",
    "parse_group_spec",
    "build_layer_groups",
    "build_moe_per_expert_groups",
    "get_layer_to_group_mapping",
    "get_group_param_count",
]


@dataclass
class LayerGroup:
    """A set of layers constrained to share the same bitwidth."""

    group_id: int
    layer_names: List[str]
    pattern: str
    param_count: int = 0


def parse_group_spec(group_spec: Optional[str]) -> List[List[str]]:
    """Parse a custom group specification string.

    Format: space-separated groups, each group is comma-separated patterns.

        'mlp.gate_proj,mlp.down_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj'

    Returns a list of groups, each a list of patterns.
    """
    if not group_spec or not group_spec.strip():
        return []

    groups: List[List[str]] = []
    for group_str in group_spec.strip().split():
        patterns = [p.strip() for p in group_str.split(",") if p.strip()]
        if patterns:
            groups.append(patterns)
    return groups


def build_layer_groups(
    layer_names: List[str],
    group_patterns: Optional[List[List[str]]] = None,
) -> List[LayerGroup]:
    """Build layer groups from substring patterns.

    For each transformer block, layers matching patterns in the same group are
    constrained to share a bitwidth. With no patterns, every layer is its own
    group.

    ``block_pattern`` is matched against the *layer name*, so it must be
    adapted to the naming scheme in use (``model.layers.N.`` for HF-style
    names, ``backbone.layers.N.`` for MLX-style).
    """
    if not group_patterns:
        return [LayerGroup(i, [name], name) for i, name in enumerate(layer_names)]

    block_pattern = re.compile(r"layers\.(\d+)\.")

    def _compile(pattern: str) -> re.Pattern:
        """`N` is the wildcard for an index (expert / projection).

        Without this the patterns are literal substrings, and a spec written
        against the documented `...experts.N.up_proj` form never matches --
        silently leaving one RCO group per tensor instead of one per
        (layer, projection).
        """
        parts = pattern.split("N")
        out = []
        for i, chunk in enumerate(parts):
            if i:
                out.append(r"\d+")
            out.append(re.escape(chunk))
        return re.compile("".join(out))

    compiled = [[_compile(p) for p in patterns] for patterns in group_patterns]

    block_groups: Dict[tuple, List[str]] = {}
    ungrouped: List[str] = []

    for name in layer_names:
        match = block_pattern.search(name)
        if not match:
            ungrouped.append(name)
            continue

        block_idx = int(match.group(1))
        suffix = name[match.end():]

        assigned = False
        for group_idx, pats in enumerate(compiled):
            for pat in pats:
                if pat.search(suffix):
                    block_groups.setdefault((block_idx, group_idx), []).append(name)
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            ungrouped.append(name)

    groups: List[LayerGroup] = []
    gid = 0
    for (block_idx, pattern_idx), layers in sorted(block_groups.items()):
        groups.append(
            LayerGroup(
                group_id=gid,
                layer_names=sorted(layers),
                pattern=f"block_{block_idx}:{','.join(group_patterns[pattern_idx])}",
            )
        )
        gid += 1

    for name in ungrouped:
        groups.append(LayerGroup(group_id=gid, layer_names=[name], pattern=name))
        gid += 1

    return groups


def build_moe_per_expert_groups(
    layer_names: List[str],
    extra_group_patterns: Optional[List[List[str]]] = None,
    expert_pattern: Optional[str] = None,
) -> List[LayerGroup]:
    """MoE-aware grouping: one group per expert, plus non-expert groups.

    Each expert (gate_proj + up_proj + down_proj across one (layer, expert))
    becomes a group. Non-expert layers fall through to ``build_layer_groups``.

    ``expert_pattern`` defaults to the HF-style ``layers.N.mlp.experts.E.``.
    Pass an MLX-style pattern (e.g. ``layers\\.(\\d+)\\.mixer\\.switch_mlp\\.fc1``)
    when grouping MLX module names -- but note that MLX stacks experts into a
    single tensor, so per-expert grouping is generally *not* expressible there.
    """
    pattern = expert_pattern or r"layers\.(\d+)\.mlp\.experts\.(\d+)\."
    expert_re = re.compile(pattern)

    expert_layers: Dict[tuple, List[str]] = {}
    non_expert: List[str] = []

    for name in layer_names:
        match = expert_re.search(name)
        if match:
            expert_layers.setdefault((int(match.group(1)), int(match.group(2))), []).append(name)
        else:
            non_expert.append(name)

    groups: List[LayerGroup] = []
    gid = 0
    for (layer_idx, expert_idx), names in sorted(expert_layers.items()):
        groups.append(
            LayerGroup(
                group_id=gid,
                layer_names=sorted(names),
                pattern=f"layer_{layer_idx}_expert_{expert_idx}",
            )
        )
        gid += 1

    for g in build_layer_groups(non_expert, extra_group_patterns or []):
        groups.append(LayerGroup(gid, g.layer_names, g.pattern))
        gid += 1

    return groups


def get_layer_to_group_mapping(groups: List[LayerGroup]) -> Dict[str, int]:
    """Map each layer name to its group id."""
    mapping: Dict[str, int] = {}
    for group in groups:
        for name in group.layer_names:
            mapping[name] = group.group_id
    return mapping


def get_group_param_count(group: LayerGroup, param_counts: Dict[str, int]) -> int:
    """Total parameter count for a group.

    ``param_counts`` is keyed by tensor name (``...weight``) while a group's
    ``layer_names`` are module paths, so try the bare name and the ``.weight``
    form.  The default of 1 for a miss is what let a whole model report a
    near-zero parameter total.
    """
    total = 0
    for name in group.layer_names:
        if name in param_counts:
            total += param_counts[name]
        elif name + ".weight" in param_counts:
            total += param_counts[name + ".weight"]
        else:
            total += 1
    return total


def attach_param_counts(
    groups: List[LayerGroup], param_counts: Dict[str, int]
) -> np.ndarray:
    """Resolve per-group parameter counts and return the model-wide fractions.

    Returns
    -------
    (n_groups,) float array of ``group_params / total_params``. These are the
    ``weights`` in the budget constraint: they are what make a group's bits
    count in proportion to how much of the model it is.
    """
    import numpy as np

    counts = np.array(
        [float(get_group_param_count(g, param_counts)) for g in groups], dtype=np.float64
    )
    for g, c in zip(groups, counts):
        g.param_count = int(c)

    total = counts.sum()
    if total <= 0:
        raise ValueError("total parameter count is zero; cannot form fractions")
    return counts / total
