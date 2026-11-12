#!/usr/bin/env python3
"""Offline probe: is per-group imatrix-weighted scale selection worth it?

Runs the quantize-time clip search on ONE real tensor and reports
weighted/unweighted reconstruction MSE for:

  baseline        no clipping
  per-tensor      one clip ratio for the whole tensor
  per-group (w)   per-(row, group) search, imatrix-weighted objective
  per-group (u)   per-(row, group) search, plain MSE objective

This is the cheap go/no-go check before committing to a full pipeline
run. It is a PROXY: it scores weight reconstruction, not model quality,
and a single tensor cannot rank the imatrix weighting (the weighted and
unweighted searches differ only in which groups get clipped, so their
MSEs land close together while the end-to-end gap does not). Always
confirm with a real benchmark.

Usage:
    python scripts/probe_per_group_clip.py \
        --source /path/to/bf16-snapshot \
        --imatrix /path/to/imatrix.npz \
        --key model.layers.0.mlp.experts.0.gate_proj.weight \
        --bits 2 --group-size 64
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rco_mlx.imatrix import Imatrix  # noqa: E402

_PIPE = Path(__file__).resolve().parent.parent / "examples" / "moe_pipeline.py"
_spec = importlib.util.spec_from_file_location("moe_pipeline", _PIPE)
_pipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipe)


def _quant_dequant(w: np.ndarray, bits: int, gs: int) -> np.ndarray:
    return _pipe._quantize_dequant_np(w, bits, gs, "affine")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="BF16 model directory")
    ap.add_argument("--imatrix", required=True, help="GGUF imatrix or .npz capture")
    ap.add_argument("--key", default=None, help="tensor key (default: first expert gate_proj)")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    from safetensors import safe_open

    source = Path(args.source)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    key = args.key
    if key is None:
        candidates = [
            k for k in weight_map
            if ".mlp.experts.0.gate_proj.weight" in k
            or ".mlp.experts.0.up_proj.weight" in k
        ]
        if not candidates:
            sys.exit("no expert tensor found; pass --key explicitly")
        key = sorted(candidates)[0]

    with safe_open(str(source / weight_map[key]), framework="np", device="cpu") as f:
        w = f.get_tensor(key).astype(np.float32)

    imatrix = Imatrix(Path(args.imatrix))
    col_w, routed = imatrix.col_weights_for_hf_key(key)
    if col_w is None:
        sys.exit(f"key {key} not present in the imatrix")
    col_w = np.asarray(col_w, dtype=np.float64).reshape(-1)
    if col_w.shape[0] != w.shape[-1]:
        sys.exit(
            f"imatrix width {col_w.shape[0]} != tensor width {w.shape[-1]}"
        )

    out, inn = w.shape
    gs = args.group_size
    g = inn // gs
    wg = w.reshape(out, g, gs)
    cw = col_w.reshape(g, gs)
    print(f"tensor {key} shape={w.shape} bits={args.bits} gs={gs} groups={out*g}")
    print(
        f"imatrix routed={routed:.0f} col mean={col_w.mean():.3f} "
        f"min={col_w.min():.3f} max={col_w.max():.3f}"
    )

    clip_grid = _pipe.PER_GROUP_CLIP_GRID

    def weighted_mse(clips: np.ndarray) -> float:
        clipped = _pipe._apply_clips(w, clips, gs).reshape(out, g, gs)
        dq = _quant_dequant(clipped, args.bits, gs).reshape(out, g, gs)
        return float((cw[None] * (wg - dq) ** 2).sum())

    def plain_mse(clips: np.ndarray) -> float:
        clipped = _pipe._apply_clips(w, clips, gs).reshape(out, g, gs)
        dq = _quant_dequant(clipped, args.bits, gs).reshape(out, g, gs)
        return float(((wg - dq) ** 2).sum())

    # baseline
    base_w = weighted_mse(np.ones((out, g)))

    # per-tensor: one ratio for the whole tensor, weighted objective
    errs = []
    for c in clip_grid:
        errs.append(weighted_mse(np.full((out, g), c)))
    bt = int(np.argmin(errs))
    tensor_clip = np.full((out, g), clip_grid[bt])

    # per-group searches
    pg_w = _pipe.per_group_clip_search(w, col_w, args.bits, gs, "affine", clip_grid)
    pg_u = _pipe.per_group_clip_search(w, None, args.bits, gs, "affine", clip_grid)

    print()
    print(f"{'variant':38s} {'weighted MSE':>14s} {'plain MSE':>12s}")
    print(f"{'baseline (no clip)':38s} {base_w:14.6e} {plain_mse(np.ones((out, g))):12.6e}")
    print(
        f"{'per-tensor clip ' + format(clip_grid[bt], '.2f'):38s} "
        f"{weighted_mse(tensor_clip):14.6e} {plain_mse(tensor_clip):12.6e}"
    )
    print(
        f"{'per-group (imatrix-weighted)':38s} "
        f"{weighted_mse(pg_w):14.6e} {plain_mse(pg_w):12.6e}"
    )
    print(
        f"{'per-group (unweighted)':38s} "
        f"{weighted_mse(pg_u):14.6e} {plain_mse(pg_u):12.6e}"
    )
    print()
    print(
        "Remember: this proxy scores one tensor. It cannot rank the imatrix\n"
        "weighting -- confirm with a full pipeline run + benchmark."
    )


if __name__ == "__main__":
    main()
