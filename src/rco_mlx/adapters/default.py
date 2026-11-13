#!/usr/bin/env python3
"""
Default adapter for dense transformer models (Llama/Qwen/LFM3-style).

Used when the pipeline runs on a non-MoE checkpoint. Quantizes every
``nn.Linear`` weight except norms, embeddings, ``lm_head`` and other
BF16-preserved classes; the RCO allocator decides the per-tensor bitwidth.

The adapter is deliberately architecture-agnostic: the error DB measures
per-weight reconstruction error directly from safetensors (no forward pass,
no model class), so any dense model mlx_lm can load is supported as long
as its checkpoint uses ``layers.N.*`` naming.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("default")
class DefaultAdapter(ModelAdapter):
    """Generic dense-transformer adapter (one RCO group per tensor)."""

    model_type = "default"
    default_group_size = 64
    default_bits = 4

    _BF16_PREFIXES = (
        "embed_tokens",
        "lm_head",
        "norm",
        "layernorm",
        "embedding_norm",
        "mtp",
        "visual",
    )

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        num_layers = cfg.get("num_hidden_layers", 0)
        if not num_layers:
            # Some configs nest the transformer under text_config.
            num_layers = cfg.get("text_config", {}).get("num_hidden_layers", 0)

        shards = sorted(source_dir.glob("*.safetensors"))
        param_counts: Dict[str, int] = {}
        if shards:
            with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
                for key in f.keys():
                    param_counts[key] = int(f.get_tensor(key).numel())

        layers = []
        for idx in range(num_layers):
            prefixes = (f"model.layers.{idx}.", f"layers.{idx}.")
            weight_keys = [k for k in param_counts if k.startswith(prefixes)]
            layers.append({
                "index": idx,
                "type": "dense",
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
        # One group per tensor: the finest granularity the allocator supports.
        return None

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
        # Nested language-model wrapper (some HF exports keep the multimodal
        # wrapper): model.language_model.X -> language_model.model.X, which is
        # how the MLX model class names those modules.
        if key.startswith("model.language_model."):
            return "language_model.model." + key[len("model.language_model."):]
        if key == "lm_head.weight":
            return "language_model.lm_head.weight"
        return key
