#!/usr/bin/env python3
"""Inkling-2 (``inkling2_loop``) imatrix capture.

The generic ``capture_imatrix.py`` keys experts with
``...mlp.switch_mlp.gate_proj``; Inkling 2 stores them as
``...mlp.experts.gate_up_proj`` / ``...mlp.experts.down_proj`` (one stacked
tensor per projection), so that regex never fires and every expert would be
merged into a single unweighted row. This driver matches the real paths and
emits one aggregated row per stacked expert tensor, plus one row per
attention ``Linear``.

Keys are written WITHOUT ``.weight`` so they line up with the RCO output's
tensor names (``model.language_model.layers.N.mlp.experts.gate_up_proj``).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

EXPERT_RE = re.compile(
    r"^(.*layers\.\d+\.mlp)\.experts\.(gate_up_proj|down_proj)$"
)
ATTN_RE = re.compile(r"^(.*layers\.\d+\.self_attn)\.[a-z_]+$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chunks", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm

    calib = np.load(args.calibration)
    n_rows = min(args.max_chunks, calib.shape[0])
    print(f"calibration {calib.shape}, using {n_rows} rows", flush=True)

    model, tok = mlx_lm.load(args.source, trust_remote_code=True)

    in_sum2: dict[str, np.ndarray] = {}
    counts: dict[str, float] = {}
    expert_slots: dict[str, int] = {}

    def to_np(a):
        try:
            return np.asarray(a)
        except ValueError:  # bfloat16 has no PEP 3118 format
            return np.asarray(a.astype(mx.float32))

    def accumulate(key: str, rows: np.ndarray, slots: int = 0) -> None:
        if rows.size == 0:
            return
        flat = rows.reshape(-1, rows.shape[-1]).astype(np.float64)
        s = (flat * flat).sum(axis=0)
        if key not in in_sum2:
            in_sum2[key] = np.zeros_like(s)
        in_sum2[key] += s
        counts[key] = counts.get(key, 0.0) + flat.shape[0]
        expert_slots[key] = expert_slots.get(key, 0) + int(slots)

    hook_by_id: dict[int, tuple] = {}
    patched: dict[type, object] = {}

    def install(cls, fn):
        if cls not in patched:
            patched[cls] = cls.__call__
            cls.__call__ = fn

    for path, module in model.named_modules():
        t = type(module).__name__
        if t == "Linear":
            hook_by_id[id(module)] = ("linear", path)
        elif t in ("SwitchLinear", "QuantizedSwitchLinear"):
            # The published checkpoint stores experts as QuantizedSwitchLinear
            # (NVFP4/MXFP8); the dequantised source uses SwitchLinear. Both take
            # (x, indices) and expose weight [n_experts, out, in].
            hook_by_id[id(module)] = ("switch", path, int(module.weight.shape[0]))

    def linear_call(self, x, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            accumulate(info[1], to_np(x))
        return patched[type(self)](self, x, *a, **kw)

    def switch_call(self, x, indices=None, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            x_np, idx_np = to_np(x), (to_np(indices) if indices is not None else None)
            if idx_np is None:
                accumulate(info[1], x_np)
            else:
                x2 = x_np.reshape(-1, x_np.shape[-1])
                k = idx_np.shape[1] if idx_np.ndim == 2 else 1
                idx_flat = idx_np.reshape(-1)
                per_slot = np.repeat(x2, k, axis=0)
                agg = np.zeros(x2.shape[1], dtype=np.float64)
                slots = 0
                for e in range(info[2]):
                    m = idx_flat == e
                    if m.any():
                        agg += (per_slot[m].astype(np.float64) ** 2).sum(axis=0)
                        slots += int(m.sum())
                # one aggregated row for the whole stacked expert tensor
                accumulate(info[1], per_slot, slots=0)
                expert_slots[info[1]] = expert_slots.get(info[1], 0) + slots
        orig = patched[type(self)]
        return orig(self, x, indices, *a, **kw) if indices is not None else orig(self, x, *a, **kw)

    seen = set()
    for path, module in model.named_modules():
        cls, info = type(module), hook_by_id.get(id(module))
        if info and cls not in seen:
            seen.add(cls)
            install(cls, linear_call if info[0] == "linear" else switch_call)

    print(f"hooked {len(hook_by_id)} modules", flush=True)

    rows = 0
    for i in range(n_rows):
        ids = mx.array(calib[i:i + 1].astype(np.int32))
        mx.eval(model(ids))
        rows += int(ids.shape[1])
        if (i + 1) % 4 == 0 or i + 1 == n_rows:
            print(f"  row {i+1}/{n_rows}  tokens={rows}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in in_sum2.items():
        payload[f"{k}.in_sum2"] = v.astype(np.float32)
        payload[f"{k}.counts"] = np.array([counts[k]], dtype=np.float32)
    np.savez_compressed(str(out), **payload)

    n_exp = sum(1 for k in in_sum2 if EXPERT_RE.match(k))
    n_att = sum(1 for k in in_sum2 if ATTN_RE.match(k))
    exp_counts = sum(counts[k] for k in in_sum2 if EXPERT_RE.match(k))
    print(f"saved {len(in_sum2)} tensors ({n_exp} expert, {n_att} attention, "
          f"{len(in_sum2)-n_exp-n_att} other) -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB)")
    print(f"expert routed slots {exp_counts:.0f} "
          f"(= sum of per-tensor counts; {rows} tokens x top_k x visits)")


if __name__ == "__main__":
    main()
