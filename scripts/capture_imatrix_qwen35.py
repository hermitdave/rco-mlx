#!/usr/bin/env python3
"""Dedicated streaming imatrix capture for Qwen3.5 / Qwen3.6 MoE.

Why a dedicated script rather than the generic engine: the generic engine is
the right tool for *adding* an architecture, but its per-layer loop carries
overhead the Qwen path does not need, and it wrote its result only at the very
end -- so a kill at layer 19 of 40 lost everything.

This driver reuses the generic engine's ``Qwen35Profile`` for all
architecture-specific knowledge (args, key mapping, expert stacking, masks) and
only re-implements the *loop*: explicit caches, and a per-layer checkpoint so
an interrupted run resumes instead of restarting.

Usage:
    python scripts/capture_imatrix_qwen35.py \
        --source /path/to/Qwen3.6-35B-A3B \
        --calibration Qwen3_6-35B-A3B__fineweb-edu__512x2048.npy \
        --output qwen36-imatrix.npz --max-chunks 512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_imatrix_streaming import PROFILES, load_layer_weights  # noqa: E402
from rco_mlx.calibration import (  # noqa: E402
    MIN_EXPERT_COUNT,
    MIN_EXPERT_COUNT_PERCENTILE,
    expert_coverage_stats,
)

CKPT_LAYER = "__ckpt_next_layer__"
CKPT_ACTIVATED = "__ckpt_activated__"
CKPT_ZERO_LEGACY = "__ckpt_zero_experts__"


def load_raw(src: Path, wm: dict, keys: dict, mx):
    from safetensors import safe_open

    by_shard: dict = {}
    for k, s in keys.items():
        by_shard.setdefault(s, []).append(k)
    out = {}
    for shard, ks in by_shard.items():
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            for k in ks:
                out[k] = mx.array(f.get_tensor(k))
    return out


def save_checkpoint(path: Path, in_sum2: dict, counts: dict, next_layer: int,
                    activated: set | None = None):
    payload = {f"{k}.in_sum2": np.asarray(v, dtype=np.float32) for k, v in in_sum2.items()}
    payload.update({f"{k}.counts": np.array([counts[k]], dtype=np.float32) for k in counts})
    payload[CKPT_LAYER] = np.array([next_layer], dtype=np.int64)
    payload[CKPT_ACTIVATED] = np.asarray(sorted(activated), dtype=object) \
        if activated else np.asarray([], dtype=object)
    tmp = path.with_suffix(".ckpt.tmp.npz")
    np.savez_compressed(str(tmp), **payload)
    tmp.replace(path)


def load_checkpoint(path: Path):
    if not path.exists():
        return {}, {}, 0, set()
    z = np.load(path, allow_pickle=True)
    in_sum2, counts = {}, {}
    for k in z.files:
        if k in (CKPT_LAYER, CKPT_ACTIVATED, CKPT_ZERO_LEGACY):
            continue
        base, _, field = k.rpartition(".")
        if field == "in_sum2":
            in_sum2[base] = z[k].astype(np.float64)
        elif field == "counts":
            counts[base] = float(z[k][0])
    # `activated` is unknown for checkpoints written before the coverage gate:
    # report None so the caller can skip the never-activated clause honestly.
    activated = None
    if CKPT_ACTIVATED in z.files:
        activated = {str(x) for x in z[CKPT_ACTIVATED].tolist()}
    layer = int(z[CKPT_LAYER][0]) if CKPT_LAYER in z.files else 0
    return in_sum2, counts, layer, activated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chunks", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=1,
                    help="calibration chunks per forward.  Measured on "
                         "Qwen3.6: batching is numerically equivalent but ~12%% "
                         "SLOWER (the cost is per-token compute/memory, not "
                         "launch overhead), so 1 is the default")
    ap.add_argument("--max-layer", type=int, default=None,
                    help="stop after N layers (validation/timing)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--allow-foreign-calibration", action="store_true")
    ap.add_argument("--min-expert-count", type=int, default=MIN_EXPERT_COUNT,
                    help="p05 tokens/expert required (oMLX uses 16)")
    ap.add_argument("--min-expert-percentile", type=int,
                    default=MIN_EXPERT_COUNT_PERCENTILE)
    ap.add_argument("--allow-insufficient-coverage", action="store_true",
                    help="write the imatrix even if the coverage gate fails")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import ArraysCache, KVCache

    src = Path(args.source)
    profile = PROFILES["qwen3_5_moe"]
    margs = profile.load_args(src)
    n_layers = margs.num_hidden_layers
    if args.max_layer is not None:
        n_layers = min(n_layers, args.max_layer)
    print(f"[cfg] {profile.name} layers={n_layers} hidden={margs.hidden_size} "
          f"vocab={margs.vocab_size}", flush=True)

    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    static_raw = load_raw(src, wm,
                          {k: s for k, s in wm.items() if k in profile.static_keys()}, mx)
    embed_mod, norm_mod, head_mod = profile.build_static(margs, mx, nn)
    ek = next(k for k in static_raw if "embed" in k)
    nk = next(k for k in static_raw if "norm" in k and "embed" not in k)
    hk = next(k for k in static_raw if "lm_head" in k)
    embed_mod.load_weights([("weight", static_raw[ek])], strict=False)
    norm_mod.load_weights(
        [("weight", profile._sanitize("language_model.model.norm.weight",
                                      static_raw[nk]))], strict=False)
    head_mod.load_weights([("weight", static_raw[hk])], strict=False)
    del static_raw
    mx.eval(embed_mod.parameters(), norm_mod.parameters(), head_mod.parameters())

    calib = np.load(args.calibration)
    n_rows = min(args.max_chunks, calib.shape[0])
    ids = mx.array(calib[:n_rows].astype(np.int32))
    print(f"[calib] {n_rows} rows x {calib.shape[1]}", flush=True)
    if not args.allow_foreign_calibration:
        from rco_mlx.calibration import verify_calibration

        rep = verify_calibration(src, Path(args.calibration))
        print(rep.summary(), flush=True)
        if not rep.ok:
            sys.exit("calibration cache does not match this model's tokenizer.")

    out = Path(args.output)
    ckpt = out.with_suffix(".ckpt.npz")
    if args.no_resume:
        ckpt.unlink(missing_ok=True)
    in_sum2, counts, start_layer, activated = load_checkpoint(ckpt)
    if start_layer:
        print(f"[resume] layer {start_layer}, {len(in_sum2)} tensors already held",
              flush=True)

    hook_by_id: dict = {}
    patched: dict = {}

    def to_np(a):
        try:
            return np.asarray(a)
        except ValueError:
            return np.asarray(a.astype(mx.float32))

    def accumulate(key, s, n):
        if key in in_sum2 and in_sum2[key].shape == s.shape:
            in_sum2[key] += s
        elif key not in in_sum2:
            in_sum2[key] = s.copy()
        counts[key] = counts.get(key, 0.0) + n

    def linear_call(self, x, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            rows = to_np(x).reshape(-1, to_np(x).shape[-1]).astype(np.float64)
            accumulate(profile.mlx_to_hf(info[1]), (rows * rows).sum(axis=0), rows.shape[0])
        return patched[type(self)](self, x, *a, **kw)

    def switch_call(self, x, indices=None, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            x_np = to_np(x)
            idx_np = to_np(indices) if indices is not None else None
            if idx_np is not None:
                x2 = x_np.reshape(-1, x_np.shape[-1])
                k = idx_np.shape[1] if idx_np.ndim == 2 else 1
                idx_flat = idx_np.reshape(-1).astype(np.int64)
                sq = np.repeat(x2, k, axis=0).astype(np.float64)
                sq *= sq
                n_exp = info[2]
                cnt = np.bincount(idx_flat, minlength=n_exp)
                nz = np.nonzero(cnt)[0]
                sums = {}
                if nz.size:
                    starts = np.concatenate(([0], np.cumsum(cnt)))
                    order = np.argsort(idx_flat, kind="stable")
                    red = np.add.reduceat(sq[order], starts[nz], axis=0)
                    for j, e in enumerate(nz):
                        sums[int(e)] = red[j]
                pooled = sq.sum(axis=0)
                n_slots = int(sq.shape[0])
                for e in range(n_exp):
                    s = sums.get(e)
                    key = profile.mlx_to_hf(info[1], e)
                    if s is not None:
                        activated.add(key)
                    accumulate(key, s if s is not None else pooled,
                               int(cnt[e]) if s is not None else n_slots)
            else:
                rows = x_np.reshape(-1, x_np.shape[-1]).astype(np.float64)
                accumulate(profile.mlx_to_hf(info[1]), (rows * rows).sum(axis=0),
                           rows.shape[0])
        return patched[type(self)](self, x, indices, *a, **kw) \
            if indices is not None else patched[type(self)](self, x, *a, **kw)

    def register(module, prefix):
        for path, m in module.named_modules():
            kind = {"Linear": "linear", "SwitchLinear": "switch"}.get(type(m).__name__)
            if kind is None:
                continue
            full = f"{prefix}.{path}" if path else prefix
            hook_by_id[id(m)] = (kind, full) + (
                (int(m.weight.shape[0]),) if kind == "switch" else ())
            cls = type(m)
            if cls not in patched:
                patched[cls] = cls.__call__
                cls.__call__ = linear_call if kind == "linear" else switch_call

    register(head_mod, "lm_head")

    mb = max(1, args.micro_batch)
    H = [embed_mod(ids[i:i + mb]) for i in range(0, n_rows, mb)]
    mx.eval(*H)
    print(f"[batch] {len(H)} forwards/layer from {n_rows} chunks "
          f"(micro-batch {mb})", flush=True)

    t_start = time.time()
    for i in range(start_layer, n_layers):
        t0 = time.time()
        raw = load_raw(src, wm, {
            k: s for k, s in wm.items() if k.startswith(profile.layer_key_prefix(i))
        }, mx)
        mapped = profile.map_layer(raw, i)
        del raw
        block = profile.build_layer(margs, i)
        # Full module paths must be stripped to layer-relative keys, and every
        # parameter leaf verified loaded (strict=False skips silently).
        _, unmatched = load_layer_weights(block, mapped, profile.module_prefix(i))
        del mapped
        if unmatched:
            sys.exit(f"layer {i}: {len(unmatched)} parameters unmatched by "
                     f"map_layer (e.g. {unmatched[:3]})")
        mx.eval(block.parameters())
        register(block, profile.module_prefix(i))

        is_linear = (i + 1) % margs.full_attention_interval != 0
        new_H = []
        # The mask depends only on sequence length, which is identical for every
        # chunk -- build it once per layer instead of once per chunk.
        mask, _ = profile.mask_and_cache(margs, i, H[0], mx)
        for h in H:
            # One fresh cache per chunk: chunks are independent sequences.
            cache = ArraysCache(size=2) if is_linear else KVCache()
            layer_out = block(h, mask=mask, cache=cache)
            mx.eval(layer_out)  # mx.eval returns None; evaluate in place
            new_H.append(layer_out)
        H = new_H
        dt = time.time() - t0
        del block
        mx.clear_cache()
        save_checkpoint(ckpt, in_sum2, counts, i + 1, activated)
        print(f"[layer {i:3d}] {dt:6.1f}s | tensors={len(in_sum2)} "
              f"| total {time.time()-t_start:7.1f}s", flush=True)
        cov = expert_coverage_stats(
            counts, activated,
            min_required_count=args.min_expert_count,
            required_percentile=args.min_expert_percentile,
        )
        if cov.total_experts:
            print(f"    coverage: {cov.active_experts}/{cov.total_experts} experts "
                  f"active | never-activated {cov.zero_count_experts} | "
                  f"p{args.min_expert_percentile:02d} {cov.p05_count:.1f} "
                  f"(need >= {args.min_expert_count})", flush=True)

    H = [norm_mod(h) for h in H]
    mx.eval(*H)
    for h in H:
        head_mod(h)
    mx.eval()

    cov = expert_coverage_stats(
        counts, activated,
        min_required_count=args.min_expert_count,
        required_percentile=args.min_expert_percentile,
    )
    if activated is None:
        print("  (checkpoint predates the coverage gate: never-activated clause "
              "not evaluated)", flush=True)
    print(cov.summary(), flush=True)
    if not cov.ok and not args.allow_insufficient_coverage:
        sys.exit(
            "expert coverage insufficient -- the imatrix for unrouted experts is "
            "a substitute, not a measurement. Re-run with more --max-chunks, or "
            "pass --allow-insufficient-coverage to override."
        )

    payload = {}
    for k, v in in_sum2.items():
        payload[f"{k}.in_sum2"] = np.asarray(v, dtype=np.float32)
        payload[f"{k}.counts"] = np.array([counts[k]], dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **payload)
    ckpt.unlink(missing_ok=True)
    print(f"[done] {len(in_sum2)} tensors -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB) in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
