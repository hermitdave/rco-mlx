#!/usr/bin/env python3
"""
OLMoE adapter (AllenAI, 1B active / 7B total).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("olmoe")
class OLMoEAdapter(ModelAdapter):
    """Adapter for OLMoE-1B-7B (model_type: olmoe).

    Architecture: 16 identical MoE layers, 64 routed experts, top-8 routing.
    No shared expert, no sliding window, no MTP.
    Router is ``mlp.gate`` (not SwiGLU's ``gate_proj``).
    """

    model_type = "olmoe"
    default_group_size = 64
    default_bits = 4


    _BF16_PREFIXES = (
        "embed_tokens",
        "lm_head",
        "norm",
        "layernorm",
        "q_norm",
        "k_norm",
        "mlp.gate",
    )

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        num_layers = cfg.get("num_hidden_layers", 16)
        num_experts = cfg.get("num_experts", 64)

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
                "type": "moe",
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })

        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        # Default: apply tensor-type sensitivity if available
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
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)
        hidden_dim = cfg.get("hidden_size", 2048)
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        for prefix in self._BF16_PREFIXES:
            # Handle both "model.xxx" and bare "xxx" key styles
            if key.startswith(f"model.{prefix}.") or key.startswith(f"{prefix}.") or key == f"model.{prefix}" or key == prefix:
                return None

        if ".mlp.experts." in key:
            return key

        return key
