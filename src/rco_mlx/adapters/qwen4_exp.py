#!/usr/bin/env python3
"""
Qwen4-Exp adapter for rco-mlx (e.g. Groot-65B-A6B-261023).

Architecture: 48 layers (12 full-attention, 36 linear-attention/Gated DeltaNet),
256 routed experts, top-10 activation, plus a 51B n-gram PLE lookup table
and a 1-layer MTP head.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("qwen4_exp")
class Qwen4ExpAdapter(ModelAdapter):
    """Adapter for Qwen4-Exp models (e.g. Groot-65B-A6B-261023).

    Key differences from qwen3_5_moe:
    - 48 layers (not 40)
    - 256 experts, top-10 (not 128, top-8)
    - N-gram PLE table at layer 1 (128 shards)
    - MTP head present
    - Mixed linear/full attention layers
    """

    model_type = "qwen4_exp"
    default_group_size = 64
    default_bits = 4

    # N-gram PLE table quantization (applied uniformly, not via RCO).
    ngram_quant_bits = 3
    ngram_group_size = 32

    _BF16_PREFIXES = (
        "mlp.gate",
        "conv.conv",
        "conv1d",
        "visual",
        "mtp",
    )

    def is_ngram_key(self, key: str) -> bool:
        return "ple.ple_embedding.ngram_embedding" in key

    def is_mtp_key(self, key: str) -> bool:
        return key.startswith("mtp.")

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        text_cfg = cfg.get("text_config", cfg)
        num_layers = text_cfg.get("num_hidden_layers", 48)
        layer_types = text_cfg.get("layer_types", [])

        # Use index file for fast enumeration (avoids reading all shards).
        idx_path = source_dir / "model.safetensors.index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            param_counts: Dict[str, int] = {}
            for key in weight_map:
                if key.endswith(".weight") and not key.endswith(".scales") and not key.endswith(".biases"):
                    # Estimate param count from key name only; actual shapes
                    # would require opening shards. Grouping uses key patterns,
                    # so counts are only used for reporting.
                    param_counts[key] = 0
        else:
            shards = sorted(source_dir.glob("*.safetensors"))
            param_counts = {}
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
