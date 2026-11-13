#!/usr/bin/env python3
"""Evaluate quantized checkpoints against a BF16 reference: PPL, KL, top-1.

Runs identical token rows through the reference and each quantized model and
reports perplexity (identical protocol to macaq-mlx/eval_ppl.py when fed the
same WikiText rows), the forward KL divergence KL(ref || quant) averaged over
tokens, and top-1 argmax agreement.

Memory is bounded by holding two models plus one or two logit tensors at a
time; rows are processed one at a time because the LFM3 vocabulary makes even
a 256-token row ~131 MB of float32 logits.

Usage:
    python scripts/eval_quality.py \
        --bf16 /path/to/bf16 \
        --quant RCO=/path/to/rco \
        --quant MACaQ=/path/to/macaq \
        --calib calib_wikitext_128x2048.npy --samples 8 --max-tokens 256
"""
import argparse
import gc
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import load


def ce_loss(logits, targets):
    return nn.losses.cross_entropy(logits, targets, reduction="none")


def kl_and_top1(logits_q, logits_ref):
    """Mean forward KL(ref||q) and top-1 agreement for aligned logits."""
    lp = nn.log_softmax(logits_ref.astype(mx.float32), axis=-1)
    lq = nn.log_softmax(logits_q.astype(mx.float32), axis=-1)
    p = mx.softmax(logits_ref.astype(mx.float32), axis=-1)
    kl = mx.sum(p * (lp - lq), axis=-1)
    t1 = mx.mean(mx.argmax(logits_q, axis=-1) == mx.argmax(logits_ref, axis=-1))
    return float(mx.mean(kl)), float(t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--quant", action="append", default=[],
                    help="NAME=PATH (repeatable)")
    ap.add_argument("--calib", required=True, help="calibration .npy (N, seq)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    calib = np.load(args.calib)
    rows = calib[: args.samples, : args.max_tokens].astype(np.int32)

    print(f"Loading BF16 reference: {args.bf16}")
    model_ref, _ = load(args.bf16)
    model_ref.eval()

    results = []
    try:
        for spec in args.quant:
            name, path = spec.split("=", 1)
            print(f"\nLoading {name}: {path}")
            model_q, _ = load(path)
            model_q.eval()

            tot_ce, tot_kl, tot_t1, n_tok = 0.0, 0.0, 0.0, 0
            for i in range(rows.shape[0]):
                ids = mx.array(rows[i : i + 1])
                logits_ref = model_ref(ids).astype(mx.float32)
                logits_q = model_q(ids).astype(mx.float32)
                targets = ids[:, 1:]
                ce = ce_loss(logits_q[:, :-1, :], targets)
                kl, t1 = kl_and_top1(logits_q[:, :-1, :], logits_ref[:, :-1, :])
                n = targets.size
                tot_ce += float(mx.sum(ce))
                tot_kl += kl * n
                tot_t1 += t1 * n
                n_tok += n
                mx.eval(tot_ce)

            ppl = math.exp(tot_ce / n_tok)
            kl = tot_kl / n_tok
            t1 = tot_t1 / n_tok
            results.append((name, ppl, kl, t1))
            print(f"{name:>10s}  PPL {ppl:9.4f}  KL {kl:8.4f}  top1 {t1*100:6.2f}%")
            del model_q
            mx.clear_cache()
            gc.collect()
    finally:
        del model_ref
        mx.clear_cache()
        gc.collect()

    print("\n=== Summary (lower PPL/KL, higher top-1 is better) ===")
    print(f"{'model':>10s} {'PPL':>9s} {'KL':>8s} {'top-1':>7s}")
    for name, ppl, kl, t1 in results:
        print(f"{name:>10s} {ppl:9.4f} {kl:8.4f} {t1*100:6.2f}%")


if __name__ == "__main__":
    main()
