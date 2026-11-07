"""Turn a RCO allocation into an MLX ``config["quantization"]`` block.

The on-disk MLX quantized format (see the ``mlx-quantized-checkpoint-format``
reference in the ``streaming-mixed-precision-quantization`` skill):

    {
      "quantization": {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
        "backbone.layers.1.mixer.switch_mlp.fc1": {"group_size": 64, "bits": 2, "mode": "affine"},
        "backbone.embeddings":                     {"group_size": 64, "bits": 8, "mode": "affine"}
      },
      "quantization_config": <same object>
    }

Keys are **module paths without ``.weight``**. The top-level ``group_size`` /
``bits`` are the default for any module not listed, and ``quantization_config``
must mirror ``quantization``.

Two loader behaviours make exactness matter:

  * a module with ``.scales`` on disk that is **absent** from the map is
    quantized at the *default* width -- so if the default is wrong, the shapes
    mismatch on load;
  * a module **listed** but with no ``.scales`` on disk is quantized at load
    and then fails to find its weights.

Hence the rule this module enforces: list every quantized path, list nothing you
did not quantize, and never rely on the default to carry an intended width.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .objective import MLX_SUPPORTED_BITS, MLX_SUPPORTED_GROUP_SIZES

__all__ = [
    "MlxQuantConfig",
    "allocation_to_config",
    "merge_with_defaults",
    "write_config",
    "read_config",
    "recover_bits_from_packed_shape",
]


class MlxQuantConfig:
    """A per-path MLX quantization config, with the invariants enforced."""

    def __init__(
        self,
        default_bits: int = 4,
        group_size: int = 64,
        mode: str = "affine",
    ):
        if default_bits not in MLX_SUPPORTED_BITS:
            raise ValueError(
                f"default_bits={default_bits} is not supported by MLX "
                f"(supported: {MLX_SUPPORTED_BITS})"
            )
        if group_size not in MLX_SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"group_size={group_size} is not supported by MLX "
                f"(supported: {MLX_SUPPORTED_GROUP_SIZES})"
            )
        self.default_bits = default_bits
        self.group_size = group_size
        self.mode = mode
        self.paths: Dict[str, Dict] = {}

    def set(self, module_path: str, bits: int, group_size: Optional[int] = None) -> None:
        """Declare one module as quantized at ``bits``."""
        module_path = module_path.removesuffix(".weight")
        if bits not in MLX_SUPPORTED_BITS:
            raise ValueError(
                f"bits={bits} is not supported by MLX (supported: {MLX_SUPPORTED_BITS})"
            )
        gs = self.group_size if group_size is None else group_size
        if gs not in MLX_SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"group_size={gs} is not supported by MLX "
                f"(supported: {MLX_SUPPORTED_GROUP_SIZES})"
            )
        self.paths[module_path] = {"group_size": gs, "bits": int(bits), "mode": self.mode}

    def to_dict(self) -> Dict:
        """The ``quantization`` block, including the top-level defaults."""
        block: Dict = {
            "group_size": self.group_size,
            "bits": self.default_bits,
            "mode": self.mode,
        }
        block.update(self.paths)
        return block

    def to_json(self) -> Dict:
        """The full pair MLX expects: ``quantization`` and its mirror."""
        block = self.to_dict()
        return {"quantization": block, "quantization_config": block}

    def bits_for(self, module_path: str) -> int:
        """Resolve the effective width for a module, default included."""
        module_path = module_path.removesuffix(".weight")
        return int(self.paths.get(module_path, {}).get("bits", self.default_bits))

    def group_size_for(self, module_path: str) -> int:
        module_path = module_path.removesuffix(".weight")
        return int(self.paths.get(module_path, {}).get("group_size", self.group_size))

    def __len__(self) -> int:
        return len(self.paths)

    def __contains__(self, module_path: str) -> bool:
        return module_path.removesuffix(".weight") in self.paths


def allocation_to_config(
    assignment: Mapping[str, int],
    default_bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
    group_size_map: Optional[Mapping[str, int]] = None,
) -> MlxQuantConfig:
    """Build an ``MlxQuantConfig`` from a ``{module_path: bits}`` assignment.

    Modules already at ``default_bits`` are still listed explicitly -- the
    loader's default fallback cannot distinguish "intentionally 4-bit" from
    "I forgot this one", so an explicit entry is the only unambiguous form.
    ``group_size_map`` optionally overrides the group size per module path
    (per-tensor group sizing: sensitive tensors get finer groups).
    """
    cfg = MlxQuantConfig(default_bits=default_bits, group_size=group_size, mode=mode)
    for module_path, bits in sorted(assignment.items()):
        gs = None
        if group_size_map:
            gs = group_size_map.get(module_path) or group_size_map.get(
                module_path.removesuffix(".weight")
            )
        cfg.set(module_path, bits, group_size=gs)
    return cfg


def merge_with_defaults(
    cfg: MlxQuantConfig,
    defaults: Optional[Mapping[str, int]] = None,
) -> MlxQuantConfig:
    """Apply a ``{path: bits}`` override map on top of an existing config."""
    if defaults:
        for path, bits in defaults.items():
            cfg.set(path, bits)
    return cfg


def write_config(
    cfg: MlxQuantConfig,
    path: Path | str,
    base_config: Optional[Mapping] = None,
) -> Dict:
    """Write ``config.json`` with the quantization block merged in.

    ``base_config`` is the model's existing config dict; pass it to preserve
    every other field. Returns the merged config that was written.
    """
    merged: Dict = dict(base_config) if base_config else {}
    merged.update(cfg.to_json())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=False) + "\n")
    return merged


def read_config(path: Path | str) -> Dict:
    """Read a ``config.json`` and return its quantization block."""
    cfg = json.loads(Path(path).read_text())
    block = cfg.get("quantization") or cfg.get("quantization_config")
    if block is None:
        raise ValueError(f"no quantization block in {path}")
    return block


def recover_bits_from_packed_shape(packed_cols: int, in_features: int) -> float:
    """Recover the real bit width from a packed tensor's shape.

    Config and weights drift apart; the packed shape never lies:

        bits = packed_cols * 32 / in_features

    A mismatch between the config's ``bits`` and the real packed width surfaces
    on load as ``Expected shape (8192, 512) but received shape (8192, 256)``.
    """
    if in_features <= 0:
        raise ValueError(f"in_features must be positive, got {in_features}")
    if packed_cols <= 0:
        raise ValueError(f"packed_cols must be positive, got {packed_cols}")
    return packed_cols * 32.0 / in_features
