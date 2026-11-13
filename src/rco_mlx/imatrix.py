"""Importance-matrix (imatrix) support for RCO.

An imatrix is the llama.cpp calibration artifact: for every weight tensor
it stores, per input column (and per expert for MoE tensors),

    in_sum2[j] = sum over calibration tokens of x_j^2   (activations)
    counts[e]  = tokens routed to expert e

It is activation-aware in a way a weight-only error DB is not: a weight
column that sees large activations, or an expert that is routed rarely,
gets a very different importance than a plain weight-MSE table suggests.

This module provides:

* :class:`Imatrix` -- read a GGUF imatrix file (Unsloth-style) or a local
  ``.npz`` capture (``scripts/capture_imatrix.py``), and resolve per-input
  column weights for HF safetensors keys.
* :func:`find_imatrix_on_hf` / :func:`download_imatrix` -- check whether a
  model repo publishes an imatrix before building one locally.
* :func:`estimate_model_bytes` / :func:`fits_in_memory` -- decide whether a
  BF16 source can be loaded for a local capture on this machine.

Usage in the pipeline: if an imatrix is available (downloaded or built),
pass it to the error-DB builder (``--imatrix``), which weights its RTN
errors per input column by ``in_sum2 / counts`` and aggregates expert
rows by routing counts instead of equal parameter counts.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

__all__ = [
    "Imatrix",
    "find_imatrix_on_hf",
    "download_imatrix",
    "estimate_model_bytes",
    "fits_in_memory",
    "hf_key_to_gguf",
    "IMATRIX_FILENAME_HINTS",
]

IMATRIX_FILENAME_HINTS = ("imatrix",)

# HF safetensors key -> GGUF imatrix tensor name.
# GGUF fuses routed experts into one tensor per (layer, projection):
#   ffn_gate_exps.weight  [in, n_experts]
#   ffn_up_exps.weight    [in, n_experts]
#   ffn_down_exps.weight  [in, n_experts]
# and adds a `.in_sum2` / `.counts` sibling per weight tensor.
_EXPERT_PATTERNS = [
    (re.compile(r"layers\.(\d+)\.mlp\.experts\.\d+\.gate_up_proj"),
     "ffn_gate_exps"),
    (re.compile(r"layers\.(\d+)\.mlp\.experts\.\d+\.gate_proj"),
     "ffn_gate_exps"),
    (re.compile(r"layers\.(\d+)\.mlp\.experts\.\d+\.up_proj"),
     "ffn_up_exps"),
    (re.compile(r"layers\.(\d+)\.mlp\.experts\.\d+\.down_proj"),
     "ffn_down_exps"),
    (re.compile(r"layers\.(\d+)\.mlp\.shared_expert\.gate_up_proj"),
     "ffn_gate_shexp"),
    (re.compile(r"layers\.(\d+)\.mlp\.shared_expert\.down_proj"),
     "ffn_down_shexp"),
]
_ATTN_PATTERNS = [
    (re.compile(r"layers\.(\d+)\.self_attn\.q_proj"), "attn_q"),
    (re.compile(r"layers\.(\d+)\.self_attn\.k_proj"), "attn_k"),
    (re.compile(r"layers\.(\d+)\.self_attn\.v_proj"), "attn_v"),
    (re.compile(r"layers\.(\d+)\.self_attn\.o_proj"), "attn_output"),
    (re.compile(r"layers\.(\d+)\.linear_attn\.in_proj_qkv"), "attn_qkv"),
    (re.compile(r"layers\.(\d+)\.linear_attn\.out_proj"), "ssm_out"),
]


def _expert_index(hf_key: str) -> Optional[int]:
    m = re.search(r"\.(?:mlp|feed_forward)\.experts\.(\d+)\.", hf_key)
    return int(m.group(1)) if m else None


# SwiGLU projection spellings differ by checkpoint family: LFM2 stores
# w1/w2/w3, OLMoE/Qwen store gate_proj/down_proj/up_proj.
_SWIGLU_RENAME = {
    "w1.weight": "gate_proj.weight",
    "w2.weight": "down_proj.weight",
    "w3.weight": "up_proj.weight",
}


def _hf_key_candidates(hf_key: str) -> Iterable[str]:
    """Yield ``hf_key`` plus its SwiGLU projection-name variants."""
    yield hf_key
    for old, new in _SWIGLU_RENAME.items():
        if old in hf_key:
            yield hf_key.replace(old, new)
        elif new in hf_key:
            yield hf_key.replace(new, old)


def hf_key_to_gguf(hf_key: str) -> Optional[Tuple[str, Optional[int]]]:
    """Map an HF safetensors key to ``(gguf_tensor, expert_index)``.

    ``expert_index`` is ``None`` for non-expert tensors. Returns ``None``
    when no mapping is known (caller should fall back to unweighted).
    """
    for pattern, stem in _EXPERT_PATTERNS:
        m = pattern.search(hf_key)
        if m:
            return f"blk.{m.group(1)}.{stem}.weight", _expert_index(hf_key)
    for pattern, stem in _ATTN_PATTERNS:
        m = pattern.search(hf_key)
        if m:
            return f"blk.{m.group(1)}.{stem}.weight", None
    return None


class Imatrix:
    """Random-access view over a GGUF or NPZ imatrix file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._gguf = None
        self._npz = None
        self._names: Optional[set] = None
        self._cache: Dict[str, np.ndarray] = {}
        if self.path.suffix == ".npz":
            self._npz = np.load(str(self.path))
            self._names = set(self._npz.files)
        elif self.path.suffix in (".gguf", ".gguf_file") or "imatrix" in self.path.name:
            from gguf import GGUFReader

            self._gguf = GGUFReader(str(self.path))
            self._names = {t.name for t in self._gguf.tensors}
        else:
            raise ValueError(
                f"Unsupported imatrix format: {self.path} "
                "(expected a GGUF imatrix file or .npz)"
            )

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def tensor_names(self) -> list:
        return sorted(self._names)

    def _get(self, name: str) -> np.ndarray:
        if self._npz is not None:
            return np.asarray(self._npz[name])
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        for tensor in self._gguf.tensors:
            if tensor.name == name:
                data = np.asarray(tensor.data, dtype=np.float64)
                self._cache[name] = data
                return data
        raise KeyError(name)

    def in_sum2(self, gguf_tensor: str) -> np.ndarray:
        """Per-(input column, expert) summed squared activations."""
        return self._get(f"{gguf_tensor}.in_sum2")

    def counts(self, gguf_tensor: str) -> np.ndarray:
        """Tokens routed per expert (shape ``[n_experts]`` or ``[1]``)."""
        try:
            return self._get(f"{gguf_tensor}.counts").reshape(-1)
        except KeyError:
            return np.array([1.0])

    def col_weights_for_hf_key(
        self, hf_key: str
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Return ``(col_weights, expert_weight)`` for an HF tensor.

        ``col_weights`` are per-input-column mean squared activations
        (``in_sum2 / counts`` for the routed expert), normalized to mean 1
        so weighted MSE stays on the same scale as unweighted MSE.
        ``expert_weight`` is the routing count for the tensor's expert
        (1.0 for non-experts) -- the correct aggregation weight for
        expert rows in the error DB. Returns ``None`` if unmapped.
        """
        # Local NPZ captures are keyed by HF tensor name directly. The capture
        # maps MLX stacked names back to HF, but SwiGLU projections are named
        # w1/w2/w3 in some checkpoints (LFM2) and gate_proj/down_proj/up_proj
        # in others (OLMoE, Qwen) -- try both spellings.
        if self._npz is not None:
            names = self._names
            assert names is not None
            for candidate in _hf_key_candidates(hf_key):
                if f"{candidate}.in_sum2" not in names:
                    continue
                col = self._get(f"{candidate}.in_sum2").reshape(-1).astype(np.float64)
                counts = (
                    self._get(f"{candidate}.counts").reshape(-1).astype(np.float64)
                    if f"{candidate}.counts" in names
                    else np.array([1.0])
                )
                return _normalise_columns(col, counts, _expert_index(hf_key))

        mapped = hf_key_to_gguf(hf_key)
        if mapped is None:
            return None
        gguf_tensor, expert = mapped
        if gguf_tensor not in self._names:
            return None
        in_sum2 = self.in_sum2(gguf_tensor).astype(np.float64)
        counts = self.counts(gguf_tensor).astype(np.float64)
        if in_sum2.ndim == 2:
            # [in, n_experts]
            e = expert if expert is not None and expert < in_sum2.shape[1] else 0
            col = in_sum2[:, e]
            routed = float(counts[e]) if e < len(counts) else 1.0
        else:
            col = in_sum2.reshape(-1)
            routed = float(counts[0]) if len(counts) else 1.0
        return _normalise_columns(col, counts, expert, routed)


def _normalise_columns(
    col: np.ndarray,
    counts: np.ndarray,
    expert: Optional[int],
    routed: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Turn raw ``in_sum2`` into mean-1 column weights + routing weight."""
    col = np.asarray(col, dtype=np.float64).reshape(-1)
    if routed is None:
        if expert is not None and expert < len(counts):
            routed = float(counts[expert])
        elif len(counts):
            routed = float(counts[0])
        else:
            routed = 1.0
    total = float(col.sum())
    if total <= 0:
        # Never routed / never activated: error here has no output effect.
        return np.zeros_like(col), float(routed)
    return col * (len(col) / total), float(routed)


# ---------------------------------------------------------------------------
# HuggingFace: check for a published imatrix before building one
# ---------------------------------------------------------------------------

def find_imatrix_on_hf(
    repo_id: str, filename: Optional[str] = None
) -> Optional[str]:
    """Return the imatrix filename in a HF repo, or ``None`` if absent.

    Only lists files -- nothing is downloaded.
    """
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(repo_id)
    except Exception:
        return None
    if filename:
        return filename if filename in files else None
    for candidate in files:
        if any(hint in candidate.lower() for hint in IMATRIX_FILENAME_HINTS):
            return candidate
    return None


def download_imatrix(repo_id: str, filename: str, local_dir: Path | str) -> Path:
    """Download a published imatrix file into ``local_dir``."""
    from huggingface_hub import hf_hub_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=str(local_dir)
    )
    return Path(path)


# ---------------------------------------------------------------------------
# Local build gate: can this BF16 source be loaded on this machine?
# ---------------------------------------------------------------------------

def estimate_model_bytes(source_dir: Path | str) -> int:
    """Total safetensors bytes in a model directory (index-aware)."""
    source_dir = Path(source_dir)
    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        total = index.get("total_size")
        if total:
            return int(total)
    return sum(p.stat().st_size for p in source_dir.glob("*.safetensors"))


def total_ram_bytes() -> int:
    """Physical memory in bytes (macOS unified memory / Linux fallback)."""
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            pass
    try:
        import subprocess

        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        )
        return int(out.strip())
    except Exception:
        return 0


def fits_in_memory(
    source_dir: Path | str, budget_fraction: float = 0.7, extra_bytes: int = 0
) -> Tuple[bool, int, int]:
    """Check whether a BF16 source fits in RAM for a capture forward pass.

    Returns ``(fits, model_bytes, total_ram_bytes)``. The budget fraction
    leaves room for activations, workspace and the OS; MLX unified memory
    means weights + activations share the same pool, so a model that
    nearly fills RAM will OOM the Metal command buffer mid-forward.
    """
    model_bytes = estimate_model_bytes(source_dir)
    ram = total_ram_bytes()
    if ram <= 0:
        return False, model_bytes, ram
    return (model_bytes + extra_bytes) <= ram * budget_fraction, model_bytes, ram
