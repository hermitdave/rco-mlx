#!/usr/bin/env python3
"""Adapter for SenseNova-Loop (dense Qwen3.5-text backbone + loop + two-stream mHC).

Subclasses the generic dense adapter: the checkpoint is not MoE, so allocation
is one RCO group per tensor.  Three tensor classes have no quantizable
counterpart in ``mlx_lm.models.sensenova_loop`` and must be kept out of the
quantized set:

* ``MHCRouter.weight`` / ``.bias`` -- plain ``mx.array`` parameters, not
  ``nn.Linear``.  Writing ``.scales``/``.biases`` for them makes the loader
  fail ("parameters not in model").
* ``linear_attn.conv1d.weight`` -- ``nn.Conv1d`` weight ``[C, 1, K]`` with
  ``K = linear_conv_kernel_dim = 4``; not divisible by any group size.
* ``model.visual.*`` / ``mtp.*`` / ``loop_gate.logits`` -- no counterpart in
  the text-only MLX model class (``sanitize`` drops them anyway, but writing
  them would leave dead weight in the artifact).
"""

from __future__ import annotations

from .base import register
from .default import DefaultAdapter

_EXTRA_BF16_PREFIXES = ("mhc_attn", "mhc_ffn", "conv1d")
_DROP_PREFIXES = ("model.visual.", "mtp.", "model.language_model.loop_gate.")


@register("sensenova_loop")
class SenseNovaLoopAdapter(DefaultAdapter):
    model_type = "sensenova_loop"

    # Emit every tensor (quantized or BF16) under its MLX name so the output
    # does not depend on the loader's HF->MLX sanitize pass.
    rename_bf16_keys = True

    def non_quantized_prefixes(self):
        prefixes = super().non_quantized_prefixes()
        for p in _EXTRA_BF16_PREFIXES:
            if p not in prefixes:
                prefixes.append(p)
        return prefixes

    def drop_key(self, key: str) -> bool:
        return key.startswith(_DROP_PREFIXES)
