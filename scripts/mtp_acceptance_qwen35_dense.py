#!/usr/bin/env python3
"""1-step greedy MTP draft acceptance for Qwen3.8-27B dense checkpoints.

Binds the grafted BF16 MTP head by hand (stock mlx-lm drops mtp.* at load)
and measures the fraction of positions where the head's argmax equals the
trunk's own next-token argmax, on WikiText-2 rows (8 x 128 protocol).

Usage: python3 mtp_acceptance_qwen38.py <checkpoint-dir> [<checkpoint-dir> ...]
The graft (model-graft.safetensors, HF mtp.* names) must be present.
"""
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open
from mlx_lm import load
from mlx_lm.models.qwen3_5 import TextModelArgs, DecoderLayer

ROWS_N, SEQ = 8, 128


def shift_if_delta(w):
    return w + 1.0 if float(w.min()) < 0 else w


def main():
    src = Path.home() / "Projects/ml/models/Qwen3.8-27B-hermes-merged-BF16"
    rows = np.load(src / "work/eval_rows_8x512.npy")[:ROWS_N, :SEQ]
    ids = mx.array(rows.astype(np.int32))
    cfg = json.loads((src / "config.json").read_text())
    args = TextModelArgs.from_dict(cfg["text_config"])
    eps = args.rms_norm_eps

    # head tensors (BF16 graft, HF names)
    head = {}
    for ckpt in sys.argv[1:]:
        ckpt = Path(ckpt)
        graft = ckpt / "model-graft.safetensors"
        if not graft.exists():
            sys.exit(f"no graft in {ckpt}")
        with safe_open(str(graft), framework="pt") as f:
            head = {k: mx.array(f.get_tensor(k)) for k in f.keys() if k.startswith("mtp.")}
        if len(head) == 15:
            break
    assert len(head) == 15, f"expected 15 mtp tensors, got {len(head)}"

    # bind head modules
    n_emb = nn.RMSNorm(args.hidden_size, eps=eps)
    n_hid = nn.RMSNorm(args.hidden_size, eps=eps)
    n_out = nn.RMSNorm(args.hidden_size, eps=eps)
    n_emb.load_weights([("weight", shift_if_delta(head["mtp.pre_fc_norm_embedding.weight"]))], strict=True)
    n_hid.load_weights([("weight", shift_if_delta(head["mtp.pre_fc_norm_hidden.weight"]))], strict=True)
    n_out.load_weights([("weight", shift_if_delta(head["mtp.norm.weight"]))], strict=True)
    fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
    fc.load_weights([("weight", head["mtp.fc.weight"])], strict=True)
    blk = DecoderLayer(args, args.full_attention_interval - 1)  # full-attention block
    layer_items = []
    for k, v in head.items():
        if not k.startswith("mtp.layers.0."):
            continue
        rel = k[len("mtp.layers.0."):]
        if rel.endswith("norm.weight") and v.ndim == 1:
            v = shift_if_delta(v)
        layer_items.append((rel, v))
    blk.load_weights(layer_items, strict=True)
    mx.eval(n_emb.parameters(), n_hid.parameters(), n_out.parameters(),
            fc.parameters(), blk.parameters())

    for ckpt in sys.argv[1:]:
        model, _ = load(str(ckpt))
        trunk_norm = model.language_model.model.norm
        embed = model.language_model.model.embed_tokens
        lm_head = model.language_model.lm_head
        captured = {}
        orig = nn.RMSNorm.__call__
        tid = id(trunk_norm)

        def patched(self, x, *a, **kw):
            if id(self) == tid:
                captured["h"] = x
            return orig(self, x, *a, **kw)

        nn.RMSNorm.__call__ = patched
        logits = model(ids[:, :-1])
        mx.eval(logits)
        nn.RMSNorm.__call__ = orig
        h = captured["h"]                       # pre-final-norm residual stream
        trunk_arg = mx.argmax(logits, axis=-1)  # predicts tokens 1..SEQ-1

        e = n_emb(embed(ids[:, 1:]))
        hh = n_hid(h)
        x = fc(mx.concatenate([e, hh], axis=-1))
        x = blk(x, mask="causal", cache=None)
        draft_logits = lm_head(n_out(x))
        mx.eval(draft_logits)
        # head at t drafts token t+2; trunk argmax at t+1 predicts token t+2.
        # Drop the last head position (its target lies beyond the row).
        draft_arg = mx.argmax(draft_logits, axis=-1)[:, :-1]
        trunk_cmp = trunk_arg[:, 1:]
        acc = float((draft_arg == trunk_cmp).astype(mx.float32).mean())
        print(f"{Path(ckpt).name}: MTP 1-step greedy acceptance = {acc*100:.1f}% "
              f"({ROWS_N}x{SEQ-2} positions)", flush=True)
        del model
        mx.clear_cache()


if __name__ == "__main__":
    main()