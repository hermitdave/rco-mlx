#!/usr/bin/env python3
"""Validate a reconstructed BF16 source built by ``dequant_fp8_to_bf16.py``.

Structural checks plus a numerical spot-check against the FP8 source: for a
sample of tensors the dequantized value is recomputed straight from the shipped
codes and block scales and compared exactly (bf16-aware).

Full-model construction is deliberately NOT attempted here -- at 255 GB it
cannot be resident. Loadability is proven separately on the synthetic fixture.

Usage::

    python scripts/validate_dequantized_bf16.py \\
        --source <fp8_dir> --output <bf16_dir> [--sample 64]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

_FP8_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}
_SIDECAR_SUFFIX = ".weight_scale_inv"


def _header(shard: Path) -> dict:
    with open(shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def fp8_e4m3_to_f32(u: np.ndarray) -> np.ndarray:
    x = u.astype(np.uint32)
    sign = (x >> 7) & 1
    exp = (x >> 3) & 0xF
    mant = x & 0x7
    normal = (1.0 + mant / 8.0) * np.exp2(exp.astype(np.float32) - 7.0)
    subnormal = (mant.astype(np.float32) / 8.0) * np.exp2(-6.0)
    val = np.where(exp == 0, subnormal, normal)
    val = np.where((exp == 0xF) & (mant == 0x7), np.nan, val)
    return np.where(sign == 1, -val, val).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample", type=int, default=64,
                    help="Number of FP8 tensors to re-verify numerically")
    args = ap.parse_args()

    src, out = Path(args.source).expanduser(), Path(args.output).expanduser()
    fails: list[str] = []

    src_shards = sorted(src.glob("*.safetensors"))
    out_shards = sorted(out.glob("*.safetensors"))
    if [s.name for s in src_shards] != [s.name for s in out_shards]:
        fails.append(f"shard sets differ ({len(src_shards)} vs {len(out_shards)})")

    src_hdr, out_hdr = {}, {}
    for s in src_shards:
        src_hdr.update({k: v for k, v in _header(s).items() if k != "__metadata__"})
    for s in out_shards:
        out_hdr.update({k: (s.name, v) for k, v in _header(s).items() if k != "__metadata__"})

    # 1. nothing quantized survives
    leftovers = [k for k in out_hdr if out_hdr[k][1].get("dtype", "").upper() in _FP8_DTYPES
                 or k.endswith(_SIDECAR_SUFFIX)]
    if leftovers:
        fails.append(f"{len(leftovers)} FP8/sidecar tensors survived: {leftovers[:4]}")

    # 2. key sets: output == source minus sidecars
    expected_keys = {k for k in src_hdr if not k.endswith(_SIDECAR_SUFFIX)}
    missing = expected_keys - set(out_hdr)
    extra = set(out_hdr) - expected_keys
    if missing:
        fails.append(f"{len(missing)} source tensors missing: {sorted(missing)[:4]}")
    if extra:
        fails.append(f"{len(extra)} unexpected tensors: {sorted(extra)[:4]}")

    # 3. shapes preserved, dtypes are float
    bad_shape = [k for k in expected_keys & set(out_hdr)
                 if tuple(src_hdr[k]["shape"]) != tuple(out_hdr[k][1]["shape"])]
    if bad_shape:
        fails.append(f"{len(bad_shape)} shape mismatches: {bad_shape[:4]}")
    bad_dtype = [k for k in out_hdr if out_hdr[k][1].get("dtype") not in ("BF16", "F16", "F32")]
    if bad_dtype:
        fails.append(f"{len(bad_dtype)} non-float dtypes: {bad_dtype[:4]}")

    # 4. index consistency
    idx_path = out / "model.safetensors.index.json"
    if not idx_path.exists():
        fails.append("no model.safetensors.index.json")
    else:
        wm = json.loads(idx_path.read_text())["weight_map"]
        orphan = sorted(set(out_hdr) - set(wm))
        absent = [k for k, sh in wm.items() if k not in out_hdr or out_hdr[k][0] != sh]
        if orphan:
            fails.append(f"{len(orphan)} on-disk tensors absent from index")
        if absent:
            fails.append(f"{len(absent)} index entries wrong/missing: {absent[:4]}")

    # 5. config is the flattened text form
    cfg = json.loads((out / "config.json").read_text())
    if cfg.get("model_type") != "bailing_hybrid":
        fails.append(f"model_type={cfg.get('model_type')!r}")
    if cfg.get("architectures") != ["BailingMoeV3ForCausalLM"]:
        fails.append(f"architectures={cfg.get('architectures')!r}")

    # 6. numerical spot-check against the source codes
    fp8_keys = [k for k in expected_keys
                if src_hdr[k].get("dtype", "").upper() in _FP8_DTYPES]
    rng = np.random.default_rng(0)
    pick = [fp8_keys[i] for i in rng.choice(len(fp8_keys), size=min(args.sample, len(fp8_keys)), replace=False)] \
        if fp8_keys else []
    # key -> source shard, resolved once (headers are small; reading them per
    # sample would dominate the runtime)
    src_shard_of = {}
    for s in src_shards:
        for k in _header(s):
            if k != "__metadata__":
                src_shard_of[k] = s
    handles: dict[str, Any] = {}

    def _handle(path_str: str):
        if path_str not in handles:
            handles[path_str] = safe_open(path_str, framework="pt", device="cpu")
        return handles[path_str]

    worst = 0.0
    checked = 0
    for k in pick:
        s_shard = src_shard_of.get(k)
        o_shard = out_hdr[k][0]
        if s_shard is None:
            fails.append(f"{k}: not found in any source shard")
            continue
        try:
            sf = _handle(str(s_shard))
            of = _handle(str(out / o_shard))
        except Exception as e:
            fails.append(f"{k}: open failed: {e}")
            continue
        t = sf.get_tensor(k)
        codes = t.numpy() if t.dtype == torch.uint8 else t.view(torch.uint8).numpy()
        scales_raw = sf.get_tensor(k + "_scale_inv").numpy()
        scales = np.exp2(scales_raw.astype(np.float32) - 127.0) if scales_raw.dtype == np.uint8 \
            else scales_raw.astype(np.float32)
        out_d, in_d = codes.shape
        r, c = scales.shape
        padded = np.zeros((r * 128, c * 128), dtype=np.uint8)
        padded[:out_d, :in_d] = codes
        want = fp8_e4m3_to_f32(padded).reshape(r, 128, c, 128) * scales[:, None, :, None]
        want = want.reshape(r * 128, c * 128)[:out_d, :in_d]
        want = torch.from_numpy(want.astype(np.float32)).to(torch.bfloat16)
        got = of.get_tensor(k)
        if got.shape != want.shape:
            fails.append(f"{k}: shape {got.shape} != {want.shape}")
            continue
        g = got.to(torch.float32).numpy()
        w = want.to(torch.float32).numpy()
        nan_ok = bool((np.isnan(g) == np.isnan(w)).all())
        d = float(np.abs(np.nan_to_num(g) - np.nan_to_num(w)).max())
        worst = max(worst, d)
        checked += 1
        if not nan_ok or d != 0.0:
            fails.append(f"{k}: max|diff|={d:.3e} nan_agree={nan_ok}")

    src_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in src_hdr.values())
    out_bytes = sum(v[1]["data_offsets"][1] - v[1]["data_offsets"][0] for v in out_hdr.values())
    print(f"tensors: source {len(src_hdr)} -> output {len(out_hdr)} | "
          f"bytes {src_bytes / 1e9:.1f} GB -> {out_bytes / 1e9:.1f} GB")
    print(f"fp8 tensors: {len(fp8_keys)} | numerically re-verified: {checked} | worst |diff| {worst:.3e}")
    if fails:
        print("\nFAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nPASS: reconstruction is exact and the layout is consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
