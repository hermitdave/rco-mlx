#!/usr/bin/env python3
"""
Granite-4.0-H (IBM) MoE hybrid adapter.

Covers ``granite-4.0-h-tiny`` / ``-micro`` / ``-small`` / ``-1b``
(``model_type: granitemoehybrid`` in HF; ``granitemoehybrid`` in mlx_lm).

Architecture: a hybrid of Mamba2 blocks and full-attention blocks, each
followed by a shared dense MLP and a routed MoE (64 local experts, top-k).

Naming differences from the Qwen/OLMoE adapters this has to absorb:
- HF stores the routed experts pre-stacked as
  ``block_sparse_moe.input_linear.weight`` ``[E, 2*hidden, in]`` and
  ``output_linear.weight`` ``[E, in, hidden]``. mlx_lm's ``sanitize`` splits
  ``input_linear`` into ``switch_mlp.gate_proj``/``up_proj`` (along dim 1)
  and renames ``output_linear`` to ``switch_mlp.down_proj``. The adapter maps
  ``input_linear`` to the ``gate_proj`` carrier; the post-process splits it.
- The shared MLP is ``shared_mlp.input_linear``/``output_linear`` (dense, 2D)
  and keeps its HF names.
- The Mamba2 depthwise ``conv1d`` is 3-D ``[C, 1, 4]`` -- its last dim is not
  divisible by any group size, so it must stay BF16.
- The router (``block_sparse_moe.router.layer``) is a 2-D gate; it stays BF16
  (the LFM2 precedent -- routers are routing-critical and small).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("granitemoehybrid")
class GraniteMoeHybridAdapter(ModelAdapter):
    """Adapter for IBM's Granite-4.0-H hybrid MoE models."""

    model_type = "granitemoehybrid"
    default_group_size = 64
    default_bits = 4

    _BF16_PREFIXES = (
        "embed_tokens",
        "lm_head",
        "conv1d",
        "router",
        "A_log",
        "D",
        "dt_bias",
    )

    # HF MoE name -> mlx_lm sanitize name. `input_linear` is the [E, 2H, I]
    # carrier; the post-process splits it into gate_proj + up_proj.
    _RENAME = {
        "block_sparse_moe.input_linear.weight": "block_sparse_moe.switch_mlp.gate_proj.weight",
        "block_sparse_moe.output_linear.weight": "block_sparse_moe.switch_mlp.down_proj.weight",
    }

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        num_layers = cfg.get("num_hidden_layers", 40)
        layer_types = cfg.get("layer_types", [])

        # Use the index for a complete key list (the weights are spread over
        # several shards; reading only shard 0 would truncate the layer list).
        index_path = source_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            param_counts = _index_param_counts(source_dir, weight_map)
        else:
            param_counts = {}
            for shard in sorted(source_dir.glob("*.safetensors")):
                with safe_open(str(shard), framework="pt", device="cpu") as f:
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
            " mamba.in_proj,mamba.out_proj"
            " block_sparse_moe.input_linear,block_sparse_moe.output_linear"
            " shared_mlp.input_linear,shared_mlp.output_linear"
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


def _index_param_counts(source_dir: Path, weight_map: Dict[str, str]) -> Dict[str, int]:
    """Param counts from safetensors headers, using the index key->shard map."""
    counts: Dict[str, int] = {}
    cache: Dict[str, Any] = {}
    for key, shard in weight_map.items():
        handle = cache.get(shard)
        if handle is None:
            handle = safe_open(str(source_dir / shard), framework="pt", device="cpu")
            cache[shard] = handle
        try:
            counts[key] = int(handle.get_tensor(key).numel())
        except Exception:
            counts[key] = 0
    for handle in cache.values():
        try:
            handle.__exit__(None, None, None)
        except Exception:
            pass
    return counts
