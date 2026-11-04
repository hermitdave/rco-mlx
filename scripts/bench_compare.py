#!/usr/bin/env python3
"""Compare BF16 vs RCO vs AD-ladder vs Unsloth on PPL, KL, top-1 agreement."""
import argparse, json, gc
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np

def load_model(path):
    model, tok = mlx_lm.load(str(path))
    return model, tok

def compute_logits(model, ids):
    x = mx.array(ids)
    logits = model(x)
    mx.eval(logits)
    return logits

def ppl(logits, ids):
    logp = nn.log_softmax(logits[:, :-1, :], axis=-1)
    targets = mx.array(ids[:, 1:])
    gathered = mx.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
    return float(mx.exp(-mx.mean(gathered)))

def top1_agree(logits_q, logits_ref):
    a = mx.argmax(logits_q, axis=-1)
    b = mx.argmax(logits_ref, axis=-1)
    return float(mx.mean(a == b))

def kl_div(logits_q, logits_ref):
    lp = nn.log_softmax(logits_q, axis=-1)
    lq = nn.log_softmax(logits_ref, axis=-1)
    p = mx.softmax(logits_ref, axis=-1)
    return float(mx.mean(mx.sum(p * (lq - lp), axis=-1)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--rco", required=True)
    ap.add_argument("--adladder", required=True)
    ap.add_argument("--calib", required=True, help="Path to .npy calibration (N, seq)")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    calib = np.load(args.calib)
    ids = calib[:4, :args.max_tokens].astype(np.int32)

    print("Loading BF16...")
    model, tok = load_model(args.bf16)
    logits_bf = compute_logits(model, ids)
    np.save("/tmp/bench_logits_bf.npy", np.array(logits_bf, dtype=np.float32))
    ppl_bf = ppl(logits_bf, ids)
    print(f"BF16 PPL: {ppl_bf:.4f}")
    del model; mx.clear_cache(); gc.collect()

    print("Loading RCO...")
    model, tok = load_model(args.rco)
    logits_rco = compute_logits(model, ids)
    ppl_rco = ppl(logits_rco, ids)
    kl_rco = kl_div(logits_rco, mx.array(np.load("/tmp/bench_logits_bf.npy")))
    t1_rco = top1_agree(logits_rco, mx.array(np.load("/tmp/bench_logits_bf.npy")))
    print(f"RCO  PPL: {ppl_rco:.4f}  KL: {kl_rco:.4f}  top1: {t1_rco:.4f}")
    del model; mx.clear_cache(); gc.collect()

    print("Loading AD ladder...")
    model, tok = load_model(args.adladder)
    logits_ad = compute_logits(model, ids)
    ppl_ad = ppl(logits_ad, ids)
    kl_ad = kl_div(logits_ad, mx.array(np.load("/tmp/bench_logits_bf.npy")))
    t1_ad = top1_agree(logits_ad, mx.array(np.load("/tmp/bench_logits_bf.npy")))
    print(f"AD   PPL: {ppl_ad:.4f}  KL: {kl_ad:.4f}  top1: {t1_ad:.4f}")
    del model; mx.clear_cache(); gc.collect()

    print("\n=== Summary ===")
    print(f"{'Model':<12} {'PPL':>8} {'KL':>8} {'top1':>8}")
    print(f"{'BF16':<12} {ppl_bf:>8.4f} {'—':>8} {'—':>8}")
    print(f"{'RCO':<12} {ppl_rco:>8.4f} {kl_rco:>8.4f} {t1_rco:>8.4f}")
    print(f"{'AD ladder':<12} {ppl_ad:>8.4f} {kl_ad:>8.4f} {t1_ad:>8.4f}")


if __name__ == "__main__":
    main()
