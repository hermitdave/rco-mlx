#!/usr/bin/env python3
"""Reconstruct a BF16 source from a block-FP8 Ling checkpoint.

RCO / MACaQ consume float weights, but Ling-3.2-air ships E4M3 block-float
experts beside BF16 attention/embeddings. Feeding those codes to the quantizer
casts them to bf16 *without the per-block scales*, silently producing garbage.
This writes a real float source first.

Per tensor, for an ``[out, in]`` weight and a ``[out/128, in/128]`` float32
scale grid::

    W[i, j] = fp8_code_to_float(code[i, j]) * scale[i // 128, j // 128]

Shapes not divisible by the block are zero-padded (code 0 == 0.0) and cropped,
so the reconstruction is exact. Non-float tensors pass through unchanged and
``weight_scale_inv`` sidecars are dropped.

Memory is bounded by one output shard, not the model: tensors are read with
``safe_open`` random access and each shard is written once. At 127 B params the
reconstruction is ~255 GB, so the source shards still dominate peak usage.

The output config is rewritten to the text-only form by default (``text_config``
hoisted, vision blocks dropped), which makes the result loadable by stock mlx-lm
for calibration and teacher work as well as by the RCO pipeline.

Usage::

    python scripts/dequant_fp8_to_bf16.py \\
        --source /Volumes/External/ling-src --output ~/Projects/ml/models/Ling-3.2-air-BF16
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_FP8_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ"}
_SIDECAR_SUFFIX = ".weight_scale_inv"


def fp8_e4m3_to_f32(u: np.ndarray) -> np.ndarray:
    """E4M3FN decode. Verified bit-exact against ``mx.from_fp8``."""
    x = u.astype(np.uint32)
    sign = (x >> 7) & 1
    exp = (x >> 3) & 0xF
    mant = x & 0x7
    normal = (1.0 + mant / 8.0) * np.exp2(exp.astype(np.float32) - 7.0)
    subnormal = (mant.astype(np.float32) / 8.0) * np.exp2(-6.0)
    val = np.where(exp == 0, subnormal, normal)
    val = np.where((exp == 0xF) & (mant == 0x7), np.nan, val)
    val = np.where(sign == 1, -val, val)
    return val.astype(np.float32)


def _read_header(shard: Path) -> dict:
    with open(shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def _scale_to_float(scale: np.ndarray) -> np.ndarray:
    """FP8 checkpoints carry either float32 scales or U8 E8M0 exponents."""
    if scale.dtype == np.uint8:
        return np.exp2(scale.astype(np.float32) - 127.0)
    return scale.astype(np.float32)


def dequant_block(
    codes_u8: np.ndarray, scales: np.ndarray, block: int
) -> np.ndarray:
    """Blockwise dequantize; pad to block multiples then crop."""
    out_dim, in_dim = codes_u8.shape
    rows, cols = scales.shape[-2], scales.shape[-1]
    pad_out, pad_in = rows * block, cols * block
    if pad_out < out_dim or pad_in < in_dim:
        raise ValueError(
            f"scale grid {scales.shape} too small for weight "
            f"{codes_u8.shape} at block {block}"
        )
    padded = codes_u8
    if pad_out != out_dim or pad_in != in_dim:
        padded = np.zeros((pad_out, pad_in), dtype=np.uint8)
        padded[:out_dim, :in_dim] = codes_u8
    vals = fp8_e4m3_to_f32(padded)
    vals = vals.reshape(rows, block, cols, block) * scales[:, None, :, None]
    return vals.reshape(pad_out, pad_in)[:out_dim, :in_dim]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--block-size", type=int, default=128,
                    help="Scale grid block (Ling uses 128x128)")
    ap.add_argument("--no-text-config", action="store_true",
                    help="Keep the source config layout instead of emitting "
                         "the flattened text-only config")
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    dst = Path(args.output).expanduser()
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors shards in {src}")
    dst.mkdir(parents=True, exist_ok=True)

    total_in = total_out = 0
    n_fp8 = n_pass = n_sidecar = 0
    for i, shard in enumerate(shards, 1):
        header = _read_header(shard)
        out_tensors = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(_SIDECAR_SUFFIX):
                    n_sidecar += 1
                    continue
                meta = header.get(key, {})
                dt = meta.get("dtype", "")
                tensor = f.get_tensor(key)
                if dt.upper() not in _FP8_DTYPES:
                    out_tensors[key] = tensor
                    n_pass += 1
                    continue
                scale_key = key + "_scale_inv"
                if scale_key not in header:
                    sys.exit(f"{key}: FP8 tensor without a scale sidecar")
                if tensor.dtype == torch.uint8:
                    codes = tensor.numpy()
                else:
                    codes = tensor.view(torch.uint8).numpy()
                scales = f.get_tensor(scale_key).numpy()
                deq = dequant_block(codes, _scale_to_float(scales), args.block_size)
                out_tensors[key] = torch.from_numpy(deq.astype(np.float32)).to(torch.bfloat16)
                n_fp8 += 1
                total_in += codes.nbytes + scales.nbytes
                total_out += deq.size * 2  # bf16 payload

        out_shard = dst / shard.name
        import safetensors.torch as st
        st.save_file(out_tensors, str(out_shard))
        size = out_shard.stat().st_size
        print(f"[{i}/{len(shards)}] {shard.name} -> {out_shard.name} "
              f"({len(out_tensors)} tensors, {size / 1e9:.2f} GB)")
        del out_tensors

    # index
    weight_map = {}
    for shard in sorted(dst.glob("*.safetensors")):
        for key in _read_header(shard):
            if key != "__metadata__":
                weight_map[key] = shard.name
    (dst / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    print(f"index: {len(weight_map)} tensors across {len(shards)} shards")

    # non-weight files: copy, rewrite config
    for f in src.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        target = dst / f.name
        if f.is_file() and not target.exists():
            if f.name != "config.json":
                shutil.copy2(f, target)
    cfg_src = src / "config.json"
    if cfg_src.exists():
        cfg = json.loads(cfg_src.read_text())
        if not args.no_text_config:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
            from rco_mlx.adapters.bailing_moe_v3 import BailingMoeV3Adapter
            cfg = BailingMoeV3Adapter().prepare_config(cfg)
            print("config: flattened to the text-only bailing_hybrid layout")
        (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"\nfp8 dequantized: {n_fp8} | passthrough: {n_pass} | sidecars dropped: {n_sidecar}")
    print(f"read (codes+scales): {total_in / 1e9:.2f} GB -> written ~{total_out / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
