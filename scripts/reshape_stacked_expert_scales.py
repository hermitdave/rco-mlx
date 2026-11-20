#!/usr/bin/env python3
"""Reshape the scales/biases of pre-stacked 3-D expert tensors to match.

``stream_quantize`` flattens a stacked expert tensor ``[E, out, in]`` to
``[E*out, in]`` before ``mx.quantize``, so the scales and biases come out
2-D ``[E*out, in/gs]``.  ``mx.gather_qmm`` on a 3-D weight expects them 3-D
``[E, out, in/gs]``; feeding the flat form either fails the shape check or
dequantizes with the wrong grouping.

Every ``.scales`` / ``.biases`` whose sibling ``.weight`` is 3-D is reshaped
in place.  Shards are rewritten atomically (temp file + ``os.replace``).

Usage:
    python scripts/reshape_stacked_expert_scales.py <quantized-dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SUFFIXES = (".scales", ".biases")


def process_shard(path: Path, dry_run: bool) -> tuple[int, int]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}

    changed = 0
    for key in keys:
        if not key.endswith(SUFFIXES):
            continue
        weight_key = key.rsplit(".", 1)[0] + ".weight"
        w = tensors.get(weight_key)
        if w is None or w.ndim != 3:
            continue
        e, out, _ = w.shape
        t = tensors[key]
        if t.ndim != 2 or t.shape[0] != e * out:
            print(f"  skip {key}: weight {tuple(w.shape)} vs tensor {tuple(t.shape)}")
            continue
        tensors[key] = t.reshape(e, out, t.shape[1])
        changed += 1

    if changed and not dry_run:
        tmp = path.with_suffix(".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, path)
    return changed, len(keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shards = sorted(args.directory.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors in {args.directory}")

    total = 0
    for shard in shards:
        changed, n = process_shard(shard, args.dry_run)
        total += changed
        if changed:
            print(f"{shard.name}: reshaped {changed} tensors (of {n})")
    print(f"{'would reshape' if args.dry_run else 'reshaped'} {total} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
