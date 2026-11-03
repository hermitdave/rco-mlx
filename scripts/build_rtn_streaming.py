#!/usr/bin/env python3
"""Streaming RTN (round-to-nearest) quantizer for large models.

Builds an 8-bit (or arbitrary bit) MLX quantized checkpoint by processing one
safetensors shard at a time. Peak memory is bounded by the largest shard.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def quantize_tensor(tensor, bits, group_size):
    w = mx.array(tensor.float().numpy())
    q, scales, biases = mx.quantize(w, bits=bits, group_size=group_size)
    return (
        np.array(q).astype(np.uint32),
        np.array(scales).astype(np.float16),
        np.array(biases).astype(np.float16),
    )


def process_shard(shard_path, bits, group_size):
    results = {}
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key.endswith(".weight") and tensor.ndim == 2:
                if "embed" in key or "lm_head" in key or "linear_attn" in key or ".gate" in key or "gate_proj" in key or "shared_expert" in key or "self_attn" in key or "q_proj" in key or "k_proj" in key or "v_proj" in key or "o_proj" in key:
                    results[key] = tensor.to(torch.bfloat16) if tensor.dtype != torch.bfloat16 else tensor
                    continue
                if tensor.shape[-1] % group_size != 0:
                    # Skip tensors not divisible by group_size (keep BF16)
                    results[key] = tensor.to(torch.bfloat16) if tensor.dtype != torch.bfloat16 else tensor
                    continue
                q, s, b = quantize_tensor(tensor, bits, group_size)
                results[key] = torch.from_numpy(q)
                results[f"{key}.scales"] = torch.from_numpy(s)
                results[f"{key}.biases"] = torch.from_numpy(b)
            else:
                results[key] = tensor.to(torch.bfloat16) if tensor.dtype != torch.bfloat16 else tensor
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for f in source.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        if f.is_file():
            shutil.copy2(f, output / f.name)

    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No safetensors in {source}")

    for i, shard in enumerate(shards):
        print(f"[{i+1}/{len(shards)}] {shard.name}")
        results = process_shard(shard, args.bits, args.group_size)
        out_shard = output / f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(results, str(out_shard))
        print(f"  saved {len(results)} tensors")

    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for i in range(1, len(shards) + 1):
        shard_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        sp = output / shard_name
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for key in f.keys():
                index["weight_map"][key] = shard_name
    with open(output / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Done: {output}")


if __name__ == "__main__":
    main()