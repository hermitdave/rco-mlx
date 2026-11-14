#!/usr/bin/env python3
"""
Qwen3.5 / Groot MoE adapter (next-gen Qwen MoE, e.g. Groot-19B-A3B).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("qwen3_5_moe")
class Qwen3NextMoEAdapter(ModelAdapter):
    """Adapter for Qwen3.5-style MoE models (e.g. Groot-19B-A3B-261023).

    Architecture: 40 layers (10 full-attention, 30 linear-attention),
    128 experts top-8, stacked gate_up_proj + down_proj.
    Model path: model.language_model.model.layers.{idx}.
    """

    model_type = "qwen3_5_moe"
    default_group_size = 64
    default_bits = 4


    _BF16_PREFIXES = (
        "mlp.gate",
        "conv.conv",
        "linear_attn.conv1d",
        "visual",
    )

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        text_cfg = cfg.get("text_config", cfg)
        num_layers = text_cfg.get("num_hidden_layers", 40)
        layer_types = text_cfg.get("layer_types", [])

        shards = sorted(source_dir.glob("*.safetensors"))
        param_counts: Dict[str, int] = {}
        for shard_path in shards:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith(".weight"):
                        param_counts[key] = int(f.get_tensor(key).numel())

        layers = []
        for idx in range(num_layers):
            prefix = f"model.language_model.layers.{idx}."
            weight_keys = [k for k in param_counts if k.startswith(prefix)]
            layer_type = layer_types[idx] if idx < len(layer_types) else "block"
            layers.append({
                "index": idx,
                "type": layer_type,
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
            " linear_attn.in_proj_a,linear_attn.in_proj_b,linear_attn.in_proj_qkv,linear_attn.in_proj_z,linear_attn.out_proj"
            " model.language_model.layers.N.mlp.experts.N.down_proj"
            " model.language_model.layers.N.mlp.experts.N.gate_up_proj"
        )

    def non_quantized_prefixes(self) -> List[str]:
        prefixes = list(self._BF16_PREFIXES)
        if getattr(self, 'quantize_mtp', False):
            prefixes = [p for p in prefixes if p != 'mtp']
        return prefixes

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
        mlx = key
        if mlx.startswith("model."):
            mlx = mlx[len("model."):]
        mlx = mlx.replace("language_model.layers", "language_model.model.layers")

        if ".mlp.experts.down_proj" in mlx:
            mlx = mlx.replace(".mlp.experts.down_proj", ".mlp.experts.down_proj")
        if ".mlp.experts.gate_up_proj" in mlx:
            mlx = mlx.replace(".mlp.experts.gate_up_proj", ".mlp.experts.gate_up_proj")
        if ".mlp.shared_expert." in mlx:
            mlx = mlx.replace(".mlp.shared_expert.down_proj.", ".mlp.shared_expert.down_proj.")
            mlx = mlx.replace(".mlp.shared_expert.gate_up_proj.", ".mlp.shared_expert.gate_proj.")
        return mlx
