#!/usr/bin/env python3
"""Is micro-batching numerically equivalent to per-chunk forwards?

Batching the calibration chunks along dim 0 is only valid if the layers treat
the batch dimension as independent sequences: attention must not mix them, and
the linear-attention (SSM) recurrence must keep per-sequence state.  This runs
the same chunks both ways and compares hidden states.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path.home() / "Projects/ml/models/Qwen3.6-35B-A3B"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_imatrix_streaming import PROFILES  # noqa: E402

profile = PROFILES["qwen3_5_moe"]
args = profile.load_args(SRC)
wm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]

calib = np.load(Path.home() / "Projects/ml/calibration_cache/"
                "Qwen3_6-35B-A3B__fineweb-edu__512x2048.npy")
ids = mx.array(calib[:4, :512].astype(np.int32))  # (4, 512) -- 4 chunks

embed = nn.Embedding(args.vocab_size, args.hidden_size)
ek = "model.language_model.embed_tokens.weight"
with safe_open(str(SRC / wm[ek]), framework="pt", device="cpu") as f:
    embed.load_weights([("weight", mx.array(f.get_tensor(ek)))], strict=False)
mx.eval(embed.parameters())


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


E = embed(ids)          # (4, 512, hidden)
E1 = [E[i:i + 1] for i in range(4)]   # four (1, 512, hidden)

for idx in (0, 3):        # 0 = linear attention, 3 = full attention
    block = build(idx)
    is_linear = (idx + 1) % args.full_attention_interval != 0
    cache_cls = ArraysCache if is_linear else KVCache
    mask_one, _ = profile.mask_and_cache(args, idx, E1[0], mx)
    mask_batch, _ = profile.mask_and_cache(args, idx, E, mx)

    def run(h, mask, cache):
        out = block(h, mask=mask, cache=cache)
        mx.eval(out)
        return out

    separate = [run(h, mask_one, cache_cls(size=2) if is_linear else cache_cls())
                for h in E1]
    batched = run(E, mask_batch, cache_cls(size=2) if is_linear else cache_cls())

    diffs = [float(np.abs(np.asarray(separate[i]) - np.asarray(batched[i:i + 1])).max())
             for i in range(4)]
    scale = float(np.abs(np.asarray(batched)).max())
    def mshape(m):
        return m if isinstance(m, str) else (m.shape if m is not None else None)

    kind = "linear_attn" if is_linear else "full_attn"
    print(f"layer {idx} ({kind}): mask one={mshape(mask_one)} "
          f"batch={mshape(mask_batch)}")
    print(f"  max|separate - batched| per chunk: {['%.3e' % d for d in diffs]}"
          f"   (|h|max {scale:.3f})")
    print(f"  -> {'EQUIVALENT' if max(diffs) < 1e-2 * max(scale, 1) else 'DIFFERENT'}")
    del block
    mx.clear_cache()
