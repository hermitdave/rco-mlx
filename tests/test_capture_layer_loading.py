#!/usr/bin/env python3
"""Regression: standalone-layer loading must strip full module paths.

A qwen3_5 capture once ran on randomly-initialised layers because map_layer
emits full module paths while a standalone block's load_weights expects
relative keys (strict=False skips silently), and the old tail-only guard
matched the last two components anyway.
"""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from capture_imatrix_streaming import load_layer_weights


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.norm = nn.RMSNorm(4)


def test_full_paths_are_stripped_and_loaded():
    blk = Toy()
    w = mx.ones((4, 4)) * 3.0
    g = mx.ones((4,)) * 2.0
    mapped = {"language_model.model.layers.0.proj.weight": w,
              "language_model.model.layers.0.norm.weight": g}
    rel, unmatched = load_layer_weights(blk, mapped, "language_model.model.layers.0")
    mx.eval(blk.parameters())
    assert unmatched == []
    assert set(rel) == {"proj.weight", "norm.weight"}
    assert bool(mx.all(blk.proj.weight == 3.0))
    assert bool(mx.all(blk.norm.weight == 2.0))


def test_relative_keys_pass_through():
    blk = Toy()
    mapped = {"proj.weight": mx.ones((4, 4)), "norm.weight": mx.ones((4,))}
    _, unmatched = load_layer_weights(blk, mapped, "language_model.model.layers.0")
    assert unmatched == []


def test_wrong_prefix_is_caught_by_exact_guard():
    blk = Toy()
    mapped = {"some.other.prefix.proj.weight": mx.ones((4, 4)),
              "some.other.prefix.norm.weight": mx.ones((4,))}
    _, unmatched = load_layer_weights(blk, mapped, "language_model.model.layers.0")
    # tails would match ("proj.weight") -- the exact-key guard must not
    assert sorted(unmatched) == ["norm.weight", "proj.weight"]