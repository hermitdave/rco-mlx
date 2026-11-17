#!/usr/bin/env python3
"""How ambiguous is Qwen3.6's expert routing?

Two identical capture runs disagree on 255/256 expert assignments while the
router's input statistic matches exactly.  Either the router logits are
near-tied (so float noise flips argmax) or something is non-deterministic.
This prints the top-8 / 9th logit margin for one chunk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path.home() / "Projects/ml/models/Qwen3.6-35B-A3B"

import mlx.core as mx
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

import numpy as np
emb = np.load(Path.home() / "Projects/ml/calibration_cache/"
              "Qwen3_6-35B-A3B__fineweb-edu__512x2048.npy")

import mlx.nn as nn
embed = nn.Embedding(args.vocab_size, args.hidden_size)
ek = "model.language_model.embed_tokens.weight"
with safe_open(str(SRC / wm[ek]), framework="pt", device="cpu") as f:
    embed.load_weights([("weight", mx.array(f.get_tensor(ek)))], strict=False)
mx.eval(embed.parameters())
h = embed(mx.array(emb[:1].astype(np.int32)))  # (1, seq, hidden)

# run the layer's attention+norm up to the MoE input, then read gate logits
with mx.stream(mx.cpu):
    x = h
    # replicate DecoderLayer up to mlp using its submodules directly
    y = block.input_layernorm(x)
    y = block.linear_attn(y, mask=None, cache=None)
    y = x + y
    x = y
    y = block.post_attention_layernorm(x)
    logits = block.mlp.gate(y)
    mx.eval(logits)
    l = np.asarray(logits)[0]  # (seq, 256)
    s = np.sort(l, axis=-1)
    margin = s[:, -8] - s[:, -9]
    print(f"gate logits: shape {l.shape}")
    print(f"top8-vs-9th margin: mean {margin.mean():.4f}  min {margin.min():.4f}  "
          f"median {np.median(margin):.4f}")
    print(f"fraction of tokens with margin < 0.01: {(margin < 0.01).mean():.3f}")
    print(f"fraction of tokens with margin < 0.10: {(margin < 0.10).mean():.3f}")
    # determinism within one process
    idx1 = np.asarray(block.mlp.gate(y)).argmax(-1)
    idx2 = np.asarray(block.mlp.gate(y)).argmax(-1)
    print("argmax identical across two calls in-process:",
          bool((idx1 == idx2).all()))
