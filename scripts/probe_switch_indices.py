#!/usr/bin/env python3
"""Print the shape/dtype the SwitchLinear hook actually receives.

The generic engine and the dedicated Qwen driver disagree on expert counts
(8x vs 9x the token count), so one of them is mis-reading ``indices``.  This
builds one Qwen3.6 layer, runs one chunk, and reports what arrives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path.home() / "Projects/ml/models/Qwen3.6-35B-A3B"

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_imatrix_streaming import PROFILES  # noqa: E402

profile = PROFILES["qwen3_5_moe"]
args = profile.load_args(SRC)
wm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]

raw = {}
for k, s in wm.items():
    if k.startswith(profile.layer_key_prefix(0)):
        with safe_open(str(SRC / s), framework="pt", device="cpu") as f:
            raw[k] = mx.array(f.get_tensor(k))
block = profile.build_layer(args, 0)
block.load_weights(list(profile.map_layer(raw, 0).items()), strict=False)
mx.eval(block.parameters())

seen = []
orig = {}
for path, m in block.named_modules():
    if type(m).__name__ == "SwitchLinear":
        orig[type(m)] = type(m).__call__

        def probe(self, x, indices=None, **kw):
            seen.append((path, None if indices is None else indices.shape,
                         None if indices is None else indices.dtype,
                         x.shape))
            return orig[type(self)](self, x, indices, **kw)

        type(m).__call__ = probe
        break  # one is enough

h = mx.random.normal((1, 2048, args.hidden_size)).astype(mx.bfloat16)
mask, _ = profile.mask_and_cache(args, 0, h, mx)
out = block(h, mask=mask, cache=None)
mx.eval(out)
for s in seen:
    print("SwitchLinear path=%s indices=%s dtype=%s x=%s" % s)
