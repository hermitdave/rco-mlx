#!/usr/bin/env python3
"""
LFM2 / LFM2.5 MoE adapter (Liquid AI).

Covers LFM2-8B-A1B / LFM2.5-8B-A1B / LFM2.5-24B-A2B (``model_type:
lfm2_moe`` in HF; ``lfm2_moe`` in mlx_lm).

Architecture: a hybrid of gated short-convolution blocks and full-attention
blocks, with a routed MoE feed-forward on every layer except the first
``num_dense_layers`` (which are plain SwiGLU MLPs).

Naming differences from the Qwen/OLMoE adapters this has to absorb:
- HF stores SwiGLU as ``w1``/``w2``/``w3``; mlx_lm's ``sanitize`` renames them
  to ``gate_proj``/``down_proj``/``up_proj``. The adapter must apply the same
  rename so emitted config keys match what the loader expects.
- Experts live under ``feed_forward.experts.E.*`` (not ``mlp.experts.E.*``).
- ``conv.conv.weight`` is a depthwise ``[C,1,3]`` tensor -- 3-D, so it must
  stay BF16 (the pipeline's 3-D path assumes a stacked-expert layout).
- The router is ``feed_forward.gate`` and there is a per-expert
  ``feed_forward.expert_bias`` vector; both stay BF16.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("lfm2_moe")
class Lfm2MoeAdapter(ModelAdapter):
    """Adapter for Liquid AI's LFM2/LFM2.5 MoE models."""

    model_type = "lfm2_moe"
    default_group_size = 64
    default_bits = 4

    _BF16_PREFIXES = (
        "embed_tokens",
        "embedding_norm",
        "lm_head",
        "ffn_norm",
        "operator_norm",
        "q_layernorm",
        "k_layernorm",
        "feed_forward.gate",
        "expert_bias",
        "conv.conv",
    )

    # HF SwiGLU name -> mlx_lm sanitize name
    _RENAME = {
        "w1.weight": "gate_proj.weight",
        "w2.weight": "down_proj.weight",
        "w3.weight": "up_proj.weight",
    }

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        num_layers = cfg.get("num_hidden_layers", 24)
        layer_types = cfg.get("layer_types", [])

        shards = sorted(source_dir.glob("*.safetensors"))
        param_counts: Dict[str, int] = {}
        if shards:
            with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
                for key in f.keys():
                    param_counts[key] = int(f.get_tensor(key).numel())

        layers = []
        for idx in range(num_layers):
            prefix = f"model.layers.{idx}."
            weight_keys = [k for k in param_counts if k.startswith(prefix)]
            layers.append({
                "index": idx,
                "type": layer_types[idx] if idx < len(layer_types) else "block",
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
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj"
            " conv.in_proj,conv.out_proj"
            " model.layers.N.feed_forward.experts.N.w1"
            " model.layers.N.feed_forward.experts.N.w2"
            " model.layers.N.feed_forward.experts.N.w3"
            " model.layers.N.feed_forward.w1,model.layers.N.feed_forward.w2,"
            "model.layers.N.feed_forward.w3"
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
        for prefix in self._BF16_PREFIXES:
            if (
                key.startswith(f"model.{prefix}.")
                or key.startswith(f"{prefix}.")
                or key == f"model.{prefix}"
                or key == prefix
            ):
                return None
        for old, new in self._RENAME.items():
            if old in key:
                key = key.replace(old, new)
        return key
