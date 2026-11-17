#!/usr/bin/env python3
"""Nemotron-H Puzzle / Bolt adapter (NVIDIA).

Covers ``NVIDIA-Nemotron-3.5-Bolt-74B-A9B`` and the
``Nemotron-Labs-3-Puzzle`` family (``model_type: nemotron_h_puzzle`` in MLX,
``nemotron_h_omni`` at the top level of the HF checkpoint with the text model
nested under ``llm_config``).

What makes this family different from the other MoE adapters:

- **Heterogeneous blocks.** 88 blocks: 40 ``mamba``, 40 ``moe``, 8
  ``attention``. Everything lives under one ``mixer`` stem, so attention is
  ``mixer.q_proj`` (not ``self_attn.q_proj``) and the SSM is
  ``mixer.in_proj``/``mixer.out_proj``.
- **Heterogeneous expert counts.** ``n_routed_experts`` varies per layer
  (240/304/368/432/496); the top-level ``num_experts: 512`` is NOT a per-layer
  count. ``ModelArgs.for_layer`` supplies the right one.
- **Latent MoE.** Experts map ``latent -> intermediate -> latent``; the
  projections are ``fc1_latent_proj`` / ``fc2_latent_proj`` and the routed
  experts are ``experts.E.{up,down}_proj``.
- **Individual experts in HF.** MLX ``SwitchLinear`` wants one stacked tensor
  per projection, so the pipeline's post-process stacks them into
  ``switch_mlp.fc1`` / ``switch_mlp.fc2``.
- **Skip the vision tower** (``vision_model.radio_model.*``) for text-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from safetensors import safe_open

from .base import ModelAdapter, register

HF_PREFIX = "language_model.backbone."


def _deserialize_specials(obj):
    """Replace HF's tagged-JSON stand-ins with real Python values.

    HF writes non-finite floats as ``{"__float__": "Infinity"}``; MLX passes
    ``time_step_limit`` straight into ``mx.clip``, which rejects a dict.
    """
    if isinstance(obj, dict):
        if set(obj) == {"__float__"}:
            return float(obj["__float__"])
        if set(obj) == {"__int__"}:
            return int(obj["__int__"])
        return {k: _deserialize_specials(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_specials(v) for v in obj]
    return obj


@register("nemotron_h_puzzle")
class NemotronHPuzzleAdapter(ModelAdapter):
    """Adapter for Nemotron-H Puzzle / Bolt."""

    model_type = "nemotron_h_puzzle"
    default_group_size = 64
    default_bits = 4

    # Kept in BF16 (matched against the *MLX* name, after language_model. is
    # stripped): embeddings, final norm, router gate, depthwise conv, the
    # mamba A_log/D/dt_bias buffers.  Norms and biases are already excluded
    # generically by the quantizer.
    _BF16_PREFIXES = (
        "backbone.embeddings",
        "backbone.norm_f",
        "lm_head",
        "mixer.gate",
        "mixer.conv1d",
        "mixer.dt_bias",
        "mixer.A_log",
        "mixer.D",
    )

    # No counterpart in MLX's nemotron_h model class -- writing them makes
    # load_weights fail with "parameters not in model".
    _DROP_PREFIXES = ("vision_model", "mlp1", "audio_model", "sound_model")

    rename_bf16_keys = True

    def drop_key(self, key: str) -> bool:
        return key.startswith(self._DROP_PREFIXES)

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)
        llm = cfg.get("llm_config", cfg)
        blocks = llm.get("block_configs") or []
        num_layers = len(blocks) if blocks else llm.get("num_hidden_layers", 88)

        # Bolt spreads 28k tensors over 63 shards -- every shard must be read.
        param_counts: Dict[str, int] = {}
        for shard_path in sorted(source_dir.glob("*.safetensors")):
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith(".weight"):
                        param_counts[key] = int(f.get_tensor(key).numel())

        layers = []
        for idx in range(num_layers):
            prefix = f"{HF_PREFIX}layers.{idx}."
            weight_keys = [k for k in param_counts if k.startswith(prefix)]
            layers.append({
                "index": idx,
                "type": blocks[idx].get("block_type") if idx < len(blocks) else "block",
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })
        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        return {"bits": self.default_bits, "group_size": self.default_group_size}

    def grouping_spec(self) -> Optional[str]:
        # One RCO group per (layer, projection) across all routed experts.
        # Patterns match the suffix after ``layers.<idx>.``, and ``N`` is the
        # index wildcard (see grouping.build_layer_groups).
        return "mixer.experts.N.up_proj mixer.experts.N.down_proj"

    def mlx_model_type(self) -> Optional[str]:
        # The HF checkpoint is an *omni* wrapper (vision + sound); the LLM
        # backbone is nemotron_h_puzzle, which is what MLX loads.
        return "nemotron_h_puzzle"

    def prepare_config(self, cfg: dict) -> dict:
        """Hoist ``llm_config`` so the emitted config is loadable by MLX.

        The omni checkpoint keeps every LLM field (hidden_size, num_layers,
        block_configs, ...) under ``llm_config`` and fills the top level with
        vision/sound fields; MLX's ``ModelArgs.from_dict`` reads one flat dict.
        """
        llm = cfg.get("llm_config")
        if not isinstance(llm, dict):
            return cfg
        out = dict(llm)
        out["model_type"] = "nemotron_h_puzzle"
        out.pop("architectures", None)
        # Carry over fields that live at the top level of the *quantized*
        # config rather than inside llm_config -- dropping them empties the
        # quantization map and yields a checkpoint that loads as dense.
        for k in (
            "quantization",
            "quantization_config",
            "eos_token_id",
            "bos_token_id",
            "pad_token_id",
            "tie_word_embeddings",
        ):
            if k in cfg and k not in out:
                out[k] = cfg[k]
        return _deserialize_specials(out)

    def non_quantized_prefixes(self) -> List[str]:
        return list(self._BF16_PREFIXES)

    def load_calibration(self, source_dir: Path, n_samples: int = 512,
                         seq_len: int = 2048):
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        """HF module path -> MLX module path.

        The HF checkpoint wraps the LLM in ``language_model.`` and keeps the
        backbone under ``backbone.``; MLX's nemotron_h model expects the
        backbone at the top level (``backbone.layers.N.mixer.*``).  BF16-ness
        is decided separately by ``non_quantized_prefixes``, so this always
        returns a name.
        """
        mlx = key[len("language_model."):] if key.startswith("language_model.") else key
        return mlx
