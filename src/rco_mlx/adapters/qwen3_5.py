#!/usr/bin/env python3
"""
Qwen3.5 / Qwen3.8 DENSE hybrid adapter (GatedDeltaNet + full attention).

For dense checkpoints such as Qwen/Qwen3.8-27B: 64 layers laid out as
16 x (3 x GatedDeltaNet + 1 x full attention), a vision tower
(``model.visual.*``) and an MTP head (``mtp.*``), HF keys under
``model.language_model.layers.N.*``.

Distinct from ``qwen3_5_moe`` (routed experts): no experts exist here, so
this adapter is NOT in ``_MOE_ADAPTERS`` and every quantizable tensor is its
own RCO group.

Tensor classes:
  - quantizable: self_attn.{q,k,v,o}_proj (full-attention layers only),
    linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj},
    mlp.{gate,up,down}_proj -- all ordinary nn.Linear in mlx_lm's qwen3_5.
  - BF16: norms, embed_tokens, lm_head, linear_attn.{A_log,dt_bias} (plain
    parameters, not .weight), linear_attn.conv1d (depthwise [C,1,K]).
  - dropped: model.visual.* and mtp.* -- mlx_lm's qwen3_5 Model has no
    counterpart; graft them back from the BF16 source after quantization
    (vlm-quantization skill, scripts/graft_vision_mtp.py).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import register
from .default import DefaultAdapter


@register("qwen3_5")
class Qwen35DenseAdapter(DefaultAdapter):
    """Dense GatedDeltaNet hybrid (Qwen3.5/3.8 dense VLM checkpoints)."""

    model_type = "qwen3_5"
    default_group_size = 64
    default_bits = 4
    rename_bf16_keys = True

    _BF16_PREFIXES = (
        "embed_tokens",
        "lm_head",
        "norm",
        "layernorm",
        "input_layernorm",
        "post_attention_layernorm",
        "q_norm",
        "k_norm",
        "mtp",
        "visual",
        "conv1d",
        "A_log",
        "dt_bias",
    )

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        cfg = json.loads((source_dir / "config.json").read_text())
        text_cfg = cfg.get("text_config", cfg)
        num_layers = text_cfg.get("num_hidden_layers", 64)
        layer_types = text_cfg.get("layer_types", [])

        # Shapes straight from the safetensors headers -- no tensor payload
        # reads (the source may sit on a slow USB volume).
        param_counts: Dict[str, int] = {}
        for shard in sorted(source_dir.glob("*.safetensors")):
            with open(shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k == "__metadata__" or not k.endswith(".weight"):
                    continue
                param_counts[k] = int(np.prod(v["shape"]))

        layers = []
        for idx in range(num_layers):
            prefix = f"model.language_model.layers.{idx}."
            weight_keys = [k for k in param_counts if k.startswith(prefix)]
            layer_type = layer_types[idx] if idx < len(layer_types) else "dense"
            layers.append({
                "index": idx,
                "type": layer_type,
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })
        return layers

    def grouping_spec(self) -> Optional[str]:
        # One RCO group per tensor: finest granularity, no stacked experts.
        return None

    def drop_key(self, key: str) -> bool:
        return key.startswith("model.visual.") or key.startswith("mtp.")

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        # Idempotent: the pipeline pre-renames model.language_model.* before
        # calling the adapter in stream_quantize, but the config rewrite calls
        # the adapter with raw HF keys.
        if key.startswith("model.language_model."):
            return "language_model.model." + key[len("model.language_model."):]
        if key == "lm_head.weight":
            return "language_model.lm_head.weight"
        return key