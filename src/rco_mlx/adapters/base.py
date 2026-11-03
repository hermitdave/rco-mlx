#!/usr/bin/env python3
"""
Base model adapter and registry for the RCO quantization pipeline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from safetensors import safe_open

__all__ = ["ModelAdapter", "register", "get"]

_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Decorator to register a model adapter."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get(name: str) -> type:
    """Look up a registered adapter by model type name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"No adapter registered for '{name}'. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


class ModelAdapter(ABC):
    """Base class for model adapters."""

    # Subclasses set these
    model_type: str = ""
    layer_type_key: str = "layer_types"  # config key for per-layer type list
    default_group_size: int = 64
    default_bits: int = 4

    # Tensor-type sensitivity hierarchy (from Unsloth Dynamic v3 methodology).
    # Keys are HF checkpoint key fragments; lower value = more sensitive.
    # The default is applied when no specific rule matches.

    @abstractmethod
    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        """Return a list of layer descriptors."""

    @abstractmethod
    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        """Return a quantization spec for this layer, or None to keep BF16."""


    @abstractmethod
    def grouping_spec(self) -> Optional[str]:
        """Return a grouping spec string for RCO, or None for per-layer groups."""

    @abstractmethod
    def load_calibration(
        self,
        source_dir: Path,
        n_samples: int = 512,
        seq_len: int = 2048,
    ) -> np.ndarray:
        """Load calibration activations for error measurement."""

    @abstractmethod
    def map_key_to_mlx(self, key: str) -> Optional[str]:
        """Map a source checkpoint key to MLX module path, or None to skip."""

    def build_layer_groups(self, layers: List[Dict]) -> List[Dict]:
        """Group layers for RCO allocation. Default: one group per layer."""
        return [
            {"group_id": i, "layer_indices": [l["index"]], "pattern": str(l["index"])}
            for i, l in enumerate(layers)
        ]

    def non_quantized_prefixes(self) -> List[str]:
        """Key prefixes that should always stay BF16."""
        return [
            "embed_tokens",
            "lm_head",
            "norm",
            "layernorm",
            "embedding_norm",
        ]
