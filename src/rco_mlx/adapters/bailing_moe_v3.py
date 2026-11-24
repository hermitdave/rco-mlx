#!/usr/bin/env python3
"""Ling / Bailing MoE V3 adapter (``bailing_hybrid`` -> ``bailing_moe_v3``).

Covers InclusionAI's Ling family: Ling-3.0-flash (text) and the trunks of
Ling-3.0-flash-VL / Ling-3.2-air (multimodal wrappers whose language backbone
is the same model MLX loads).

Architecture
------------
42 layers alternating Kimi Delta Attention (KDA) and gated Multi-head Latent
Attention (MLA) on a 5:1 schedule (``layer_group_size`` layers per group; the
last layer of each group is MLA), 512 routed experts with top-8 activation,
one shared expert, 2 dense layers, and a 1-layer MTP head stored at
``model.layers.<num_hidden_layers>`` -- i.e. layer 42, indexed independently of
the trunk and carrying no ``mtp.`` marker in its key names.

Naming
------
HF and MLX module paths coincide for this family, so ``map_key_to_mlx`` is the
identity.  What HF stores per-expert, MLX stacks at load time:
``model.layers.N.mlp.experts.{E}.{gate,up,down}_proj`` becomes the single
``model.layers.N.mlp.switch_mlp.{gate,up,down}_proj`` tensor, and
``attention.kv_b_proj`` is split into ``embed_q`` / ``unembed_out``.  Stacking
happens in the post-quantize step (see ``expert_stacking``), which also
rewrites the config entries to the stacked names -- mirroring the LFM2 path.

Per-expert emission is correct *provided* every expert of a (layer, projection)
shares a bitwidth; that is what ``grouping_spec`` enforces structurally and
what ``--expert-bits 2`` pins in practice.

Two traps this adapter exists to avoid
--------------------------------------
1. **The SwiGLU clamp.** Ling trains with a per-layer clamped SwiGLU on its late
   layers (``expert_swiglu_limit_list`` / ``share_expert_swiglu_limit_list``).
   ``BaseModelArgs.from_dict`` drops undeclared keys and
   ``mlx_lm.models.bailing_moe_v3.ModelArgs`` does not declare them, so a stock
   load silently runs *unclamped*.  Measured on this family: HumanEval 88.41%
   clamped vs 71.34% unclamped (-17.07pp), at no runtime cost.
   ``prepare_config`` re-injects both lists into the emitted config so the
   shipped model_file can bind them.
2. **FP8 sources.** Ling-3.2-air ships block-FP8 (E4M3, 128x128 blocks) plus a
   float32 ``weight_scale_inv`` grid.  Feeding those tensors to the quantizer
   casts the raw E4M3 codes to bf16 *without applying the block scales*, which
   silently produces meaningless weights.  ``enumerate_layers`` refuses an FP8
   source; run the streaming dequant prep first.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import ModelAdapter, register

_FP8_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ", "F8"}
_SIDECAR_SUFFIX = ".weight_scale_inv"


def _shard_headers(source_dir: Path) -> Dict[str, tuple]:
    """Parse every shard's safetensors header without loading any tensor.

    Returns ``key -> (shape, dtype)``.  At 255 GB a ``get_tensor``-based scan
    would materialize the whole model just to read metadata.
    """
    out: Dict[str, tuple] = {}
    for shard in sorted(source_dir.glob("*.safetensors")):
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            out[key] = (tuple(info["shape"]), info.get("dtype", ""))
    return out


@register("bailing_moe_v3")
class BailingMoeV3Adapter(ModelAdapter):
    """Adapter for Ling / Bailing MoE V3 (hybrid KDA + gated MLA + sparse MoE)."""

    model_type = "bailing_moe_v3"
    default_group_size = 64
    default_bits = 4

    # The text class MLX resolves.  ``bailing_hybrid`` +
    # ``BailingMoeV3ForCausalLM`` is the pair mlx-lm's architecture remapping
    # knows; the VL wrapper's ``bailing_moe_v3_vl`` pair is not registered.
    MLX_MODEL_TYPE = "bailing_hybrid"
    MLX_ARCHITECTURE = "BailingMoeV3ForCausalLM"

    # Ling trains with a per-layer clamped SwiGLU and stocks the limits in
    # config.json; stock mlx-lm drops them.  The bundled model_file binds them.
    model_file_name = "bailing_hybrid.py"

    # Set by the pipeline under --quantize-embeddings.  main() refuses that flag
    # unless the adapter declares the attribute, so it must exist even though
    # the default keeps embeddings/lm_head BF16.
    quantize_embeddings = False

    # Tensors that must never be quantized.  The pipeline wraps these as
    # ``.<prefix>.``, so ``mlp.gate`` hits the router while ``mlp.gate_proj``
    # (dense layers) is untouched.
    _BF16_PREFIXES = (
        "norm",
        "attention.A_log",
        "attention.dt_bias",
        "attention.q_conv1d",
        "attention.k_conv1d",
        "attention.v_conv1d",
        # kv_b_proj is split by sanitize into embed_q/unembed_out; the stock
        # sanitize raises if a `.scales` sibling exists, so it stays BF16.
        "attention.kv_b_proj",
        "mlp.gate",
        "visual",
        "vision_tower",
        "linear_proj",
        "eh_proj",
        "enorm",
        "hnorm",
    )

    # ------------------------------------------------------------------
    # config handling
    # ------------------------------------------------------------------
    @staticmethod
    def _read_config(source_dir: Path) -> dict:
        with open(source_dir / "config.json") as f:
            return json.load(f)

    @staticmethod
    def _text_cfg(cfg: dict) -> dict:
        """Language-backbone fields, hoisted out of a multimodal wrapper."""
        return cfg.get("text_config", cfg)

    def num_hidden_layers(self, source_dir: Path) -> int:
        return int(self._text_cfg(self._read_config(source_dir))["num_hidden_layers"])

    def prepare_config(self, cfg: dict) -> dict:
        """Emit a text-only config the MLX text class can consume.

        Hoists ``text_config`` to the top level (a VL wrapper nests every LLM
        field there) and drops the vision/projector blocks and the FP8
        ``quantization_config``.  The SwiGLU limit lists are carried through
        explicitly -- ``BaseModelArgs.from_dict`` discards undeclared keys.
        """
        text = dict(self._text_cfg(cfg))
        for key in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"):
            if key in text:
                cfg[key] = list(text[key])
        for key in (
            "vision_config",
            "quantization_config",
            "auto_map",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "text_config",
        ):
            cfg.pop(key, None)
        cfg.update(text)
        cfg["model_type"] = self.MLX_MODEL_TYPE
        cfg["architectures"] = [self.MLX_ARCHITECTURE]
        return cfg

    def mlx_model_type(self) -> Optional[str]:
        """Deliberately None -- ``prepare_config`` already wrote both fields.

        Returning a value here makes ``_rewrite_config_mlx_keys`` set
        ``model_type`` *and pop* ``architectures``.  The architecture remapping
        that resolves Ling is the PAIR
        ``("bailing_hybrid", "BailingMoeV3ForCausalLM") -> "bailing_moe_v3"``;
        with ``architectures`` gone, ``_get_classes`` falls through to the
        generic remap and imports ``mlx_lm.models.bailing_hybrid`` -- a
        different, older class whose modules match nothing in this checkpoint
        (measured: ``ValueError: Received 118 parameters not in model``).
        """
        return None

    # ------------------------------------------------------------------
    # layer enumeration
    # ------------------------------------------------------------------
    @staticmethod
    def _param_counts(headers: Dict[str, tuple]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for key, (shape, _dtype) in headers.items():
            n = 1
            for d in shape:
                n *= int(d)
            counts[key] = n
        return counts

    @staticmethod
    def _fp8_report(headers: Dict[str, tuple]) -> tuple:
        fp8 = sum(1 for _s, dt in headers.values() if dt.upper() in _FP8_DTYPES)
        sidecars = sum(1 for k in headers if k.endswith(_SIDECAR_SUFFIX))
        return fp8, sidecars

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        cfg = self._text_cfg(self._read_config(source_dir))
        num_layers = int(cfg["num_hidden_layers"])
        headers = _shard_headers(source_dir)
        counts = self._param_counts(headers)

        fp8, sidecars = self._fp8_report(headers)
        if fp8 or sidecars:
            raise RuntimeError(
                f"{source_dir.name}: {fp8} FP8 tensors and {sidecars} "
                f"'{_SIDECAR_SUFFIX}' scale sidecars detected. Quantizing raw "
                "FP8 codes casts them to bf16 WITHOUT the per-block scales, "
                "silently yielding garbage weights. Run the streaming dequant "
                "prep first (references/bailing-moe-v3.md)."
            )

        group_size = int(cfg.get("layer_group_size", 6))
        first_dense = int(cfg.get("first_k_dense_replace", 0))

        layers: List[Dict] = []
        for idx in range(num_layers):
            prefix = f"model.layers.{idx}."
            weight_keys = [
                k
                for k in counts
                if k.startswith(prefix) and k.endswith(".weight")
            ]
            is_linear = not (
                (idx + 1) % group_size == 0
                or idx >= num_layers // group_size * group_size
            )
            layers.append(
                {
                    "index": idx,
                    "type": (
                        "dense"
                        if idx < first_dense
                        else ("linear" if is_linear else "mla")
                    ),
                    "weight_keys": weight_keys,
                    "param_count": sum(counts[k] for k in weight_keys),
                }
            )
        return layers

    # ------------------------------------------------------------------
    # allocation hooks
    # ------------------------------------------------------------------
    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        return {"bits": self.default_bits, "group_size": self.default_group_size}

    def grouping_spec(self) -> Optional[str]:
        """One RCO group per (layer, projection).

        Two invariants meet here.  Every expert inside one (layer, projection)
        must share a bitwidth or MLX's loader fails with ``ValueError: [stack]
        All arrays must have the same shape`` -- one pattern per projection
        gives exactly that.  And each projection needs its OWN config entry:
        a stacked tensor's last dim is packed U32 columns, not the input width,
        so mlx-lm's ``infer_quant_config`` shape fallback cannot recover its
        bits/group_size (it works for plain 2-D linears, which is why LFM2's
        shared attention group gets away with one entry -- experts do not).
        """
        return (
            "attention.q_proj,attention.k_proj,attention.v_proj,attention.o_proj,"
            "attention.f_proj,attention.g_proj,attention.dense,"
            "attention.embed_q,attention.unembed_out "
            "mlp.experts.N.gate_proj "
            "mlp.experts.N.up_proj "
            "mlp.experts.N.down_proj "
            "mlp.shared_experts.gate_proj "
            "mlp.shared_experts.up_proj "
            "mlp.shared_experts.down_proj "
            "mlp.gate_proj "
            "mlp.up_proj "
            "mlp.down_proj"
        )

    def non_quantized_prefixes(self) -> List[str]:
        # Ling names its MTP head by layer index, not by an `mtp.` prefix, so
        # no "mtp" entry can match; `is_mtp_key` covers it instead.
        return list(self._BF16_PREFIXES)

    # ``_mtp_start`` is resolved from the source config by `bind_source`,
    # which the pipeline calls once after construction.
    _mtp_start: int = 0

    def bind_source(self, source_dir: Path) -> None:
        self._mtp_start = self.num_hidden_layers(source_dir)

    def is_mtp_key(self, key: str) -> bool:
        """Ling stores the MTP head at ``model.layers.<num_hidden_layers>``.

        Those indices are independent of the trunk and the keys carry no
        ``mtp`` marker, so the pipeline's generic ``"mtp" in key`` test misses
        them entirely.
        """
        m = re.match(r"model\.layers\.(\d+)\.", key)
        if not m or not self._mtp_start:
            return False
        return int(m.group(1)) >= self._mtp_start

    # ------------------------------------------------------------------
    # key mapping
    # ------------------------------------------------------------------
    def map_key_to_mlx(self, key: str) -> Optional[str]:
        # HF and MLX paths coincide for this family; stacking/splitting is the
        # post-quantize step's job.  Returning None here would mean "keep this
        # tensor BF16 under its source name", which is NOT what we want for
        # ordinary weights, so the identity is deliberate.
        return key

    def drop_key(self, key: str) -> bool:
        """Omit tensors the MLX text class has no counterpart for.

        Distinct from ``map_key_to_mlx`` returning None (keep, unquantized):
        these keys must not be written at all or ``load_weights`` fails with
        "parameters not in model".
        """
        if key.startswith(("model.visual.", "visual.")):
            return True
        if key.startswith("linear_proj"):
            return True
        if key.endswith(_SIDECAR_SUFFIX):
            # After the dequant prep this is gone; if it is still present the
            # run above has already refused to proceed.
            return True
        return False

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
