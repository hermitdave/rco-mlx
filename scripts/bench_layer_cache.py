#!/usr/bin/env python3
"""Time one Qwen3.6 layer's forward under different cache strategies.

The streaming capturer is ~4x slower per layer on Qwen3.6 than the old
Nemotron script was on Bolt.  This isolates whether that is the architecture or
the engine's cache handling: same layer, same chunks, only the cache argument
varies.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SRC = Path.home() / "Projects/ml/models/Qwen3.6-35B-A3B"
N = 60

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from capture_imatrix_streaming import PROFILES  # noqa: E402

profile = PROFILES["qwen3_5_moe"]
args = profile.load_args(SRC)
wm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]

ids = mx.array([[1, 2, 3] + [0] * 2045])


def build(idx):
    raw = {}
    for k, s in wm.items():
        if k.startswith(profile.layer_key_prefix(idx)):
            with safe_open(str(SRC / s), framework="pt", device="cpu") as f:
                raw[k] = mx.array(f.get_tensor(k))
    block = profile.build_layer(args, idx)
    block.load_weights(list(profile.map_layer(raw, idx).items()), strict=False)
    mx.eval(block.parameters())
    return block


def timeit(label, idx, cache_factory):
    block = build(idx)
    h = mx.random.normal((1, 2048, args.hidden_size)).astype(mx.bfloat16)
    mx.eval(h)
    mask, _ = profile.mask_and_cache(args, idx, h, mx)
    cache = cache_factory()
    # warmup
    for _ in range(3):
        mx.eval(block(h, mask=mask, cache=cache))
    t0 = time.time()
    for _ in range(N):
        mx.eval(block(h, mask=mask, cache=cache))
    dt = (time.time() - t0) / N
    print(f"  {label:34s} layer {idx:2d}  {dt*1000:8.1f} ms/forward")
    del block
    mx.clear_cache()
    return dt


from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402

print("Qwen3.6 layer forward timing (2048 tokens, 60 iters)")
lin = timeit("linear_attn, cache=None", 0, lambda: None)
lin_c = timeit("linear_attn, ArraysCache(2)", 0, lambda: ArraysCache(size=2))
full = timeit("full_attn, cache=None", 3, lambda: None)
full_c = timeit("full_attn, KVCache()", 3, lambda: KVCache())
print()
print(f"linear:  None {lin*1000:.1f} ms vs ArraysCache {lin_c*1000:.1f} ms "
      f"-> {'cache obj faster' if lin_c < lin else 'None faster'} "
      f"({max(lin,lin_c)/min(lin,lin_c):.2f}x)")
print(f"full:    None {full*1000:.1f} ms vs KVCache {full_c*1000:.1f} ms "
      f"-> {'cache obj faster' if full_c < full else 'None faster'} "
      f"({max(full,full_c)/min(full,full_c):.2f}x)")
