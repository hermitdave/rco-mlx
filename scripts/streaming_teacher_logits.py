#!/usr/bin/env python3
"""Phase-1 teacher dump for KL, streaming (Bolt's BF16 is 149.8 GB).

``eval_kl_top1_split.py --dump-ref`` loads the whole teacher with mlx_lm, which
is impossible here.  This reuses the layer-at-a-time forward from
``streaming_capture_imatrix.py`` to produce exactly the same npz
(``rows`` / ``logprobs`` / ``ce``), so phase 2 is unchanged.

Usage:
  python streaming_teacher_logits.py --source <BF16 dir> \
      --dump-ref ref_logits.npz --rows 8 --seq-len 512
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "/Users/hermit.dave/Projects/ml/scripts")

from streaming_capture_imatrix import (  # noqa: E402
    build_mlx_config,
    map_layer_tensors,
)

LAYER_RE = None  # populated lazily; keeps map_layer_tensors' own regex


def load_raw(src: Path, wm: dict, keys: dict):
    """Load the given {hf_key: shard} tensors into MLX arrays."""
    from safetensors import safe_open

    by_shard: dict[str, list] = {}
    for k, s in keys.items():
        by_shard.setdefault(s, []).append(k)
    out = {}
    for shard, ks in by_shard.items():
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            for k in ks:
                out[k] = mx.array(f.get_tensor(k))
    return out


class _Tk:
    """Adapter so ``eval_kl_top1_split.wikitext_rows`` can use tokenizer.json."""

    def __init__(self, path: Path, bos):
        from tokenizers import Tokenizer

        self._t = Tokenizer.from_file(str(path))
        self.bos_token_id = bos

    def encode(self, text):
        return self._t.encode(text).ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="BF16 checkpoint directory")
    ap.add_argument("--dump-ref", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.nemotron_h import ModelArgs, NemotronHBlock, NemotronHRMSNorm

    from eval_kl_top1_split import ppl_from_logprob_mean, wikitext_rows

    src = Path(args.source)
    hf = json.loads((src / "config.json").read_text())
    llm = hf.get("llm_config", {})
    bos = llm.get("bos_token_id", hf.get("bos_token_id"))
    tk = _Tk(src / "tokenizer.json", bos)

    rows = wikitext_rows(tk, args.rows, args.seq_len)
    print(f"[rows] {rows.shape} (bos={bos})", flush=True)

    margs = ModelArgs.from_dict(build_mlx_config(src))
    pattern = margs.hybrid_override_pattern or []
    n_layers = len(pattern)
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    static = load_raw(src, wm, {
        k: s for k, s in wm.items()
        if k in ("language_model.backbone.embeddings.weight",
                 "language_model.backbone.norm_f.weight",
                 "language_model.lm_head.weight")
    })
    embeddings = nn.Embedding(margs.vocab_size, margs.hidden_size)
    embeddings.load_weights(
        [("weight", static["language_model.backbone.embeddings.weight"])], strict=False)
    norm_f = NemotronHRMSNorm(margs.hidden_size, eps=margs.layer_norm_epsilon)
    norm_f.load_weights(
        [("weight", static["language_model.backbone.norm_f.weight"])], strict=False)
    lm_head = nn.Linear(margs.hidden_size, margs.vocab_size, bias=False)
    lm_head.load_weights(
        [("weight", static["language_model.lm_head.weight"])], strict=False)
    del static
    mx.eval(embeddings.parameters(), norm_f.parameters(), lm_head.parameters())

    t0 = time.time()
    H = [embeddings(mx.array(rows[i:i + 1], mx.uint32)) for i in range(len(rows))]
    mx.eval(*H)

    for i in range(n_layers):
        bt = pattern[i]
        raw = load_raw(src, wm, {
            k: s for k, s in wm.items()
            if k.startswith(f"language_model.backbone.layers.{i}.")
        })
        mapped = map_layer_tensors(raw, i, mx)
        del raw
        block = NemotronHBlock(margs.for_layer(i), bt)
        block.load_weights(list(mapped.items()), strict=False)
        del mapped
        mx.eval(block.parameters())
        new_H = []
        for h in H:
            if bt == "*":
                mask, c = create_attention_mask(h, None), KVCache()
            elif bt == "M":
                mask, c = create_ssm_mask(h, None), ArraysCache(size=2)
            else:
                mask, c = None, None
            new_H.append(block(h, mask=mask, cache=c))
        mx.eval(*new_H)
        H = new_H
        del block
        mx.clear_cache()
        if (i + 1) % 10 == 0 or i == n_layers - 1:
            print(f"[layer {i+1}/{n_layers}] ({time.time()-t0:.0f}s)", flush=True)

    lps, ces = [], []
    for i, h in enumerate(H):
        logits = lm_head(norm_f(h))[:, :-1].astype(mx.float32)
        lp = nn.log_softmax(logits, axis=-1).astype(mx.float16)
        mx.eval(lp)
        arr = np.array(lp)[0]  # drop the batch dim, as iter_logits does
        lps.append(arr)
        ces.append(float(-arr.astype(np.float32)[np.arange(arr.shape[0]),
                                              rows[i, 1:]].mean()))
        del lp, logits
        mx.clear_cache()
        print(f"[logits] row {i+1}/{len(H)} ({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.dump_ref)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rows=rows,
             logprobs=np.stack(lps).astype(np.float16),
             ce=np.array(ces))
    print(f"[done] wrote {out} ({out.stat().st_size/1e6:.1f} MB) | teacher PPL "
          f"{ppl_from_logprob_mean(float(np.mean(ces))):.4f}", flush=True)


if __name__ == "__main__":
    main()
