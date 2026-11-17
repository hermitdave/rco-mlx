#!/usr/bin/env python3
"""Recursive Flint adapter (``model_type: flint`` / ``FlintForCausalLM``).

Upstream mlx-lm support was merged in PR #1998 (2026-10-24); the model class
lives in ``mlx_lm/models/flint.py`` and needs ``mlx_lm/kernels`` for
``expert_routing``.

Architecture: 49 layers -- 2 dense SwiGLU MLPs (``mlp_only_layers``) followed by
47 MoE layers with 256 routed experts, top-6, plus an always-on shared expert.
Attention alternates sliding-window (513) and full; only sliding layers carry
RoPE, full-attention layers have no positional encoding at all.

Three details no other adapter in this repo has to absorb:

- ``self_attn.softmax_offset`` -- a learned per-head attention sink, present in
  the checkpoint on *every* layer but created in mlx-lm as a plain
  ``mx.zeros((n_heads,))`` attribute rather than an ``nn.Parameter``, so it does
  not appear in ``parameters()``.  It is tiny (32 floats/layer) and stays BF16.
- Experts are stored **per expert** in the checkpoint
  (``mlp.experts.E.{up,down,gate}_proj.weight``) and stacked by mlx-lm's
  ``sanitize`` into ``mlp.switch_mlp.*``.  Like LFM2, the pipeline must stack
  them itself (``--model-type flint`` triggers the post-process).
- HF and MLX names otherwise coincide, so ``map_key_to_mlx`` is the identity
  apart from the BF16 exclusions -- no w1/w2/w3 renaming is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("flint")
class FlintAdapter(ModelAdapter):
    """Adapter for Recursive's Flint (39.8B-A2.5B)."""

    model_type = "flint"
    default_group_size = 64
    default_bits = 4

    # Kept in BF16: embeddings, final norm, the router, QK-norms, the attention
    # sinks, and the two layernorms.  The shared expert IS quantized -- it is
    # always active, so it is a good allocation target.
    _BF16_PREFIXES = (
        "embed_tokens",
        "lm_head",
        "model.norm",
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "self_attn.softmax_offset",
        "mlp.gate",
    )

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        num_layers = int(cfg.get("num_hidden_layers", 49))
        dense = set(cfg.get("mlp_only_layers") or [0, 1])
        layer_types = cfg.get("layer_types", [])

        shards = sorted(source_dir.glob("*.safetensors"))
        param_counts: Dict[str, int] = {}
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in param_counts:
                        param_counts[key] = int(np.prod(f.get_slice(key).get_shape()))

        layers = []
        for idx in range(num_layers):
            prefix = f"model.layers.{idx}."
            weight_keys = [k for k in param_counts if k.startswith(prefix)]
            layers.append({
                "index": idx,
                "type": ("dense" if idx in dense
                         else (layer_types[idx] if idx < len(layer_types) else "moe")),
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })
        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        return {
            "bits": self.default_bits,
            "group_size": self.default_group_size,
        }

    def grouping_spec(self) -> Optional[str]:
        return (
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
            " model.layers.N.mlp.experts.N.up_proj"
            " model.layers.N.mlp.experts.N.down_proj"
            " model.layers.N.mlp.experts.N.gate_proj"
            " model.layers.N.mlp.shared_expert.up_proj"
            " model.layers.N.mlp.shared_expert.down_proj"
            " model.layers.N.mlp.shared_expert.gate_proj"
            " model.layers.N.mlp.up_proj,model.layers.N.mlp.down_proj,"
            "model.layers.N.mlp.gate_proj"
        )

    def non_quantized_prefixes(self) -> List[str]:
        return list(self._BF16_PREFIXES)

    def load_calibration(
        self,
        source_dir: Path,
        n_samples: int = 512,
        seq_len: int = 2048,
    ) -> np.ndarray:
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        # Component-aligned, matching `_filter_bf16_prefixes`: `mlp.gate` must
        # exclude the router without also catching `mlp.gate_proj`.
        parts = key.split(".")
        for prefix in self._BF16_PREFIXES:
            pp = prefix.split(".")
            n = len(pp)
            for i in range(len(parts) - n + 1):
                if parts[i:i + n] == pp:
                    return None
        return key
