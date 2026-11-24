# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
"""Ling / Bailing MoE V3 model_file for MLX (``trust_remote_code=True``).

Ships beside a quantized checkpoint and is named by ``config.json``::

    {"model_file": "bailing_hybrid.py", ...}

Load with::

    from mlx_lm import load
    model, tokenizer = load("<dir>", trust_remote_code=True)

Why this file exists
--------------------
``mlx_lm.models.bailing_moe_v3`` resolves for the ``bailing_hybrid`` /
``BailingMoeV3ForCausalLM`` architecture pair and its layer layout matches the
Ling checkpoints. Three things it does not do, all measured on this family:

1. **The trained SwiGLU clamp.** Ling trains with a per-layer clamped SwiGLU on
   its late layers; the limits ship in ``config.json`` as
   ``expert_swiglu_limit_list`` / ``share_expert_swiglu_limit_list``. The stock
   ``ModelArgs`` does not declare those keys, so ``BaseModelArgs.from_dict``
   silently drops them and the model runs unclamped -- worth **88.41% vs 71.34%
   on HumanEval (-17.07pp)** in oMLX's measurement, at zero runtime cost.
   Semantics follow vLLM: ``silu(gate).clamp(max=L) * up.clamp(-L, L)``.

2. **Stacked quantized experts.** MLX stores a MoE layer's routed experts as one
   ``switch_mlp.<proj>`` tensor. The stock ``sanitize`` stacks only ``.weight``,
   leaving every ``.scales`` / ``.biases`` sibling orphaned, which surfaces as
   ``Received N parameters not in model``. This file stacks all three suffixes.

3. Nothing else. Layer types, KDA/gated-MLA alternation, router, cache handling
   and RoPE are inherited unchanged from ``mlx_lm.models.bailing_moe_v3``.

Requires ``mlx_lm`` with ``bailing_moe_v3`` (2026 builds).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx import nn

import mlx_lm.models.bailing_moe_v3 as _bailing

__all__ = ["Model", "ModelArgs", "layer_swiglu_limit"]


class ModelArgs(_bailing.ModelArgs):
    """Stock args plus the two trained clamp tables.

    ``BaseModelArgs.from_dict`` filters to declared fields, so the stock class
    silently drops both limit lists. They are attached explicitly instead of
    being declared as new dataclass fields -- ``@dataclass`` on a subclass
    requires the module to be present in ``sys.modules``, which is not true
    when mlx-lm loads a ``model_file`` via ``spec_from_file_location``.
    """

    @classmethod
    def from_dict(cls, params):
        args = super().from_dict(params)
        args.expert_swiglu_limit_list = params.get("expert_swiglu_limit_list")
        args.share_expert_swiglu_limit_list = params.get(
            "share_expert_swiglu_limit_list"
        )
        return args


def layer_swiglu_limit(limit_list: Optional[List[float]], layer_idx: int) -> Optional[float]:
    """Per-layer clamp, or ``None`` when the list is absent, short, or zero."""
    if not limit_list or layer_idx >= len(limit_list):
        return None
    limit = limit_list[layer_idx]
    if limit in (None, 0):
        return None
    return float(limit)


def _clamped_swiglu(gate: mx.array, x: mx.array, limit: float) -> mx.array:
    """``silu(gate).clamp(max=L) * up.clamp(-L, L)`` -- vLLM's Ling clamp."""
    return mx.minimum(nn.silu(gate), limit) * mx.clip(x, -limit, limit)


class ClampedSwiGLU(nn.Module):
    """``SwitchGLU``-signature activation: called as ``(x_up, x_gate)``.

    Getting this order wrong silently clamps the wrong tensor; the stock
    ``SwiGLU`` is ``swiglu(gate, x)`` so silu always lands on the gate.
    """

    def __init__(self, limit: float):
        super().__init__()
        self.limit = float(limit)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _clamped_swiglu(gate, x, self.limit)


class ClampedMLP(nn.Module):
    """Wraps a dense/``shared_experts`` MLP with the same clamp."""

    def __init__(self, inner: nn.Module, limit: float):
        super().__init__()
        self.gate_proj = inner.gate_proj
        self.up_proj = inner.up_proj
        self.down_proj = inner.down_proj
        self.limit = float(limit)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit)
        )


def bind_clamps(model: nn.Module, args: ModelArgs) -> int:
    """Bind per-layer limits onto routed and shared experts.

    Returns the number of expert paths clamped, so a caller can log it and a
    missing table is visible rather than silent.
    """
    expert_limits = getattr(args, "expert_swiglu_limit_list", None)
    shared_limits = getattr(args, "share_expert_swiglu_limit_list", None)
    if not expert_limits and not shared_limits:
        return 0

    clamped = 0
    layers = getattr(getattr(model, "model", model), "layers", [])
    for idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        switch_mlp = getattr(mlp, "switch_mlp", None)
        limit = layer_swiglu_limit(expert_limits, idx)
        if switch_mlp is not None and limit:
            switch_mlp.activation = ClampedSwiGLU(limit)
            clamped += 1
        shared = getattr(mlp, "shared_experts", None)
        limit = layer_swiglu_limit(shared_limits, idx)
        if shared is not None and limit:
            mlp.shared_experts = ClampedMLP(shared, limit)
            clamped += 1
    return clamped


# ---------------------------------------------------------------------------
# sanitize: stock behaviour, then stack the quantized siblings
# ---------------------------------------------------------------------------

_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(\w+)\.(\w+)$"
)


def _stack_quant_siblings(weights: Dict[str, mx.array], num_experts: int) -> Dict[str, mx.array]:
    """Stack per-expert ``.scales`` / ``.biases`` into ``switch_mlp`` carriers.

    ``super().sanitize`` already stacked ``.weight``; without this the orphaned
    siblings make ``load_weights`` fail with "parameters not in model".
    Groups whose members are incomplete are left alone rather than guessed at.
    """
    groups: Dict[tuple, Dict[int, str]] = {}
    for key in list(weights):
        m = _EXPERT_RE.match(key)
        if not m:
            continue
        prefix, expert, proj, suffix = m.groups()
        groups.setdefault((prefix, proj, suffix), {})[int(expert)] = key

    for (prefix, proj, suffix), members in groups.items():
        if len(members) != num_experts:
            continue
        vals = [weights.pop(members[e]) for e in range(num_experts)]
        weights[f"{prefix}.switch_mlp.{proj}.{suffix}"] = mx.stack(vals)
    return weights


class Model(_bailing.Model):
    """Stock Ling model plus the trained clamp and complete expert stacking."""

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self._ling_clamped_paths = bind_clamps(self, args)

    @property
    def ling_clamped_paths(self) -> int:
        return self._ling_clamped_paths

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = super().sanitize(weights)
        return _stack_quant_siblings(weights, self.args.num_experts)
