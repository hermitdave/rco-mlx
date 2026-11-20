#!/usr/bin/env python3
"""Adapter for Inkling-2 Mini (``inkling2_loop``): a looped MoE VLM.

Layout
------
The routed experts are *pre-stacked* in the checkpoint -- one fused tensor per
(layer, projection), ``mlp.experts.gate_up_proj`` ``[E, 2I, in]`` and
``mlp.experts.down_proj`` ``[E, out, in]`` -- so one RCO group per tensor is
already the finest granularity the loader permits, and no per-expert stacking
post-process is needed.

The source for this adapter is emitted in MLX layout (see the BF16 source
builder), so :meth:`map_key_to_mlx` is the identity and ``rename_bf16_keys``
keeps the BF16 fallbacks under the same names.

Kept in BF16
------------
Anything the MLX module tree cannot bind as a ``QuantizedLinear`` /
``SwitchLinear``: the RMSNorms, the token embedding and ``lm_head``, the MoE
router, the attention-residual / message queries and the L2-clip scales (plain
``mx.array``), the depthwise ``conv1d`` kernels (``[C, 1, 4]``, not
group-divisible), the relative-logits table, and the whole vision tower.
The four rank-64 message gates per looped layer are also left alone -- the
model's own ``quant_predicate`` excludes them and they are tiny.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register

# Whole dotted components (``_filter_bf16_prefixes`` is component-aligned).
_BF16_COMPONENTS = (
    "embed_tokens",
    "lm_head",
    "norm",
    "layernorm",
    "embedding_norm",
    "mtp",
    "visual",
    "vision_tower",
    # plain mx.array parameters, not nn.Linear
    "query",
    "scale",
    "proj",
    "static_weight",
    "e_score_correction_bias",
    # depthwise conv kernels: [C, 1, K] with K = sconv_kernel_size (4)
    "conv1d",
    # the MoE router is a plain mx.array, not an nn.Linear
    "mlp.gate",
    # rank-64 loop message mixers
    "q_message_gate",
    "k_message_gate",
    "v_message_gate",
    "mlp_message_gate",
)


@register("inkling2_loop")
class Inkling2LoopAdapter(ModelAdapter):
    """Looped Inkling-2 MoE VLM (24 physical layers, 48 visits)."""

    model_type = "inkling2_loop"
    default_group_size = 64
    default_bits = 4

    # The source is already in MLX layout; write BF16 fallbacks under the same
    # name so the output never depends on the loader's sanitize pass.
    rename_bf16_keys = True

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        cfg = json.loads((source_dir / "config.json").read_text())
        text = cfg.get("text_config", cfg)
        num_layers = int(text.get("num_hidden_layers", 0))
        core = set(range(text.get("loop_config", {}).get("core_start", 0),
                         text.get("loop_config", {}).get("core_end", 0)))

        counts: Dict[str, int] = {}
        for shard in sorted(source_dir.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith(".weight"):
                        counts[key] = int(f.get_tensor(key).numel())

        layers = []
        for idx in range(num_layers):
            prefix = f"model.language_model.layers.{idx}."
            keys = [k for k in counts if k.startswith(prefix)]
            layers.append({
                "index": idx,
                "type": "loop" if idx in core else "dense",
                "weight_keys": keys,
                "param_count": sum(counts[k] for k in keys),
            })
        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        return {"bits": self.default_bits, "group_size": self.default_group_size}

    def grouping_spec(self) -> Optional[str]:
        # One group per tensor: the experts are already stacked per
        # (layer, projection), which is exactly the uniformity the loader needs.
        return None

    def non_quantized_prefixes(self) -> List[str]:
        return list(_BF16_COMPONENTS)

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        # ``stream_quantize`` rewrites ``model.language_model.X`` to
        # ``language_model.model.X`` before consulting the adapter (a
        # ``default``-adapter assumption).  Inkling 2's module tree keeps the
        # ``model.`` prefix, so normalise both spellings back to one name.
        if key.startswith("language_model.model."):
            return "model.language_model." + key[len("language_model.model."):]
        return key

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
