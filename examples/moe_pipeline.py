#!/usr/bin/env python3
"""
RCO mixed-precision quantization pipeline for MoE models on MLX.

MoE-only, streaming-only, GPTQ-based error measurement.

Usage:
    python examples/moe_pipeline.py \\
        --source /path/to/bf16-moe \\
        --output /path/to/output \\
        --target-bpw 3.5 \\
        --group-size 64 \\
        --model-type olmoe \\
        --calibration-dataset wikitext \\
        --quant-mode symmetric
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import shutil
import subprocess
import sys
import time
import json
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

from rco_mlx import (
    LossTable,
    MLX_SUPPORTED_BITS,
    MlxQuantConfig,
    allocation_to_config,
    attach_param_counts,
    build_layer_groups,
    effective_bits,
    validate_allocation,
    write_config,
)
from rco_mlx.search import budget_dp, values_from_losses


# ---------------------------------------------------------------------------
# Step 1: introspect
# ---------------------------------------------------------------------------

def introspect_model(source_dir: Path, group_spec: str | None = None, model_adapter=None):
    """Return (layer_names, param_counts, groups) from a BF16 safetensors dir."""
    shards = sorted(source_dir.glob("*.safetensors"))
    if not shards:
        sys.exit(f"No safetensors shards in {source_dir}")

    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        module_paths = set()
        for key in weight_map:
            if key.endswith(".weight"):
                module_paths.add(key[:-7])
            elif not key.endswith(".scales") and not key.endswith(".biases"):
                module_paths.add(key)
        layer_names = sorted(module_paths)
    else:
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            layer_names = sorted(
                k[:-7] if k.endswith(".weight") else k
                for k in f.keys()
                if not k.endswith(".scales") and not k.endswith(".biases")
            )

    param_counts = {}
    with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
        for key in f.keys():
            shape = f.get_tensor(key).shape
            param_counts[key] = int(np.prod(shape))

    group_patterns = None
    if group_spec:
        group_patterns = [
            [p.strip() for p in group.split(",") if p.strip()]
            for group in group_spec.strip().split()
            if group.strip()
        ]

    groups = build_layer_groups(layer_names, group_patterns)
    attach_param_counts(groups, param_counts)

    return layer_names, param_counts, groups


# ---------------------------------------------------------------------------
# Step 2: build error_db (self-contained, GPTQ-based)
# ---------------------------------------------------------------------------

def build_error_db(
    source_dir: Path,
    output_path: Path,
    model_adapter,
    calibration_path: Path,
    bitwidths: list[int],
    group_size: int,
    batch_size: int = 4,
    quant_mode: str = "affine",
    use_gptq: bool = True,
) -> Path:
    """Build GPTQ-based error_db using the streaming two-phase approach."""
    script = Path(__file__).parent.parent / "src" / "rco_mlx" / "scripts" / "build_moe_error_db.py"
    cmd = [
        sys.executable, str(script),
        "--source", str(source_dir),
        "--rtn-source", str(_auto_rtn_source(source_dir, bits=8, group_size=group_size)),
        "--output", str(output_path),
        "--model-type", model_adapter.model_type,
        "--calibration", str(calibration_path),
        "--bitwidths", ",".join(str(b) for b in bitwidths),
        "--group-size", str(group_size),
        "--batch-size", str(batch_size),
        "--quant-mode", quant_mode,
    ]
    if use_gptq:
        cmd.append("--use-gptq")
    else:
        cmd.append("--no-gptq")

    print(f"[2/5] building error_db: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"error_db build failed:\n{result.stderr}")
    print(result.stdout)
    return output_path


def _auto_rtn_source(source_dir: Path, bits: int = 8, group_size: int = 64) -> Path:
    """Return path to an RTN quant (streaming-built), building it if necessary."""
    rtn_dir = source_dir.parent / f"{source_dir.name}-rtn-{bits}bit"
    if rtn_dir.exists():
        return rtn_dir

    print(f"[auto] building {bits}-bit RTN quant (streaming) at {rtn_dir} ...")
    cmd = [
        sys.executable, str(Path(__file__).parent.parent / "scripts" / "build_rtn_streaming.py"),
        "--source", str(source_dir),
        "--output", str(rtn_dir),
        "--bits", str(bits),
        "--group-size", str(group_size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"RTN quant build failed:\n{result.stderr}")
    print(result.stdout)
    return rtn_dir


# ---------------------------------------------------------------------------
# Step 3: RCO allocation
# ---------------------------------------------------------------------------

def run_rco_allocation(
    losses: np.ndarray,
    fracs: np.ndarray,
    bitwidths: np.ndarray,
    target_bpw: float,
    group_size: int,
    group_names: list[str],
) -> tuple[dict[str, int], dict]:
    table = LossTable(
        losses=losses,
        fracs=fracs,
        bitwidths=bitwidths,
        group_size=group_size,
    )
    report = table.report(target_avg_bits=target_bpw)
    bits_per_group = table.assignment(target_avg_bits=target_bpw)
    assignment = {name: int(b) for name, b in zip(group_names, bits_per_group)}
    return assignment, report


def load_error_db(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    db = json.loads(path.read_text())
    if "layers" not in db:
        sys.exit(f"Not a valid error_db.json (no 'layers' key): {path}")

    bitwidths = sorted({
        int(b)
        for layer_data in db["layers"].values()
        for b in layer_data
        if b != "_meta" and isinstance(layer_data[b], dict)
    })
    param_counts = db.get("param_counts", {})

    rows = []
    for layer_path, bw_data in db["layers"].items():
        params = param_counts.get(layer_path, 0)
        mse = {str(bw): bw_data[str(bw)]["error"] for bw in bitwidths if str(bw) in bw_data}
        rows.append({"layer": layer_path, "tensor": layer_path, "params": params, "mse": mse})

    bitwidths_arr = np.array(bitwidths, dtype=np.int64)
    losses = np.array(
        [[r["mse"][str(b)] for b in bitwidths] for r in rows],
        dtype=np.float64,
    )
    params = np.array([r["params"] for r in rows], dtype=np.float64)
    fracs = params / max(params.sum(), 1.0)
    return losses, fracs, bitwidths_arr


# ---------------------------------------------------------------------------
# Step 4: emit + validate config
# ---------------------------------------------------------------------------

def emit_and_validate(
    assignment: dict[str, int],
    group_size: int,
    source_dir: Path,
    output_dir: Path,
    default_bits: int = 4,
    quant_mode: str = "affine",
    bf16_prefixes: list[str] | None = None,
) -> MlxQuantConfig:
    filtered = assignment
    if bf16_prefixes:
        filtered = _filter_bf16_prefixes(assignment, bf16_prefixes)

    quant_assignment = {k: v for k, v in filtered.items() if _is_quantizable(k)}
    cfg = allocation_to_config(
        quant_assignment,
        default_bits=default_bits,
        group_size=group_size,
        mode=quant_mode,
    )

    base_cfg_path = source_dir / "config.json"
    base_cfg = json.loads(base_cfg_path.read_text()) if base_cfg_path.exists() else {}
    out_cfg_path = output_dir / "config.json"
    merged = write_config(cfg, out_cfg_path, base_config=base_cfg)

    on_disk_keys = []
    index_path = output_dir / "model.safetensors.index.json"
    if not index_path.exists():
        index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text())
        on_disk_keys = list(idx.get("weight_map", {}).keys())

    # Pre-quantization: output has no .scales yet, so skip the scales-vs-declared
    # check (it runs post-quantization in _rewrite_config_mlx_keys instead).
    has_scales = any(k.endswith(".scales") for k in on_disk_keys)
    report = validate_allocation(
        cfg, on_disk_keys=on_disk_keys if has_scales else None
    )
    print(report.summary())
    if not report.ok:
        sys.exit("Validation failed — fix the config before loading.")
    return cfg


def _is_quantizable(path: str) -> bool:
    check = path if path.endswith(".weight") else path + ".weight"
    if "norm" in check or "embed" in check or "lm_head" in check or "mtp" in check:
        return False
    if check.endswith(".gate_proj.weight"):
        return False
    return check.endswith(".weight")


def _filter_bf16_prefixes(assignment: dict[str, int], prefixes: list[str]) -> dict[str, int]:
    filtered = {}
    for path, bits in assignment.items():
        skip = False
        for p in prefixes:
            if p in path:
                skip = True
                break
            if "." in p:
                last = p.split(".")[-1]
                if path.endswith("." + last) or ("." + last + ".") in path:
                    skip = True
                    break
        if not skip:
            filtered[path] = bits
    return filtered


# ---------------------------------------------------------------------------
# Step 5: streaming quantize (MoE-only, batched experts)
# ---------------------------------------------------------------------------

def stream_quantize(
    source_dir: Path,
    output_dir: Path,
    assignment: dict[str, int],
    group_size: int,
    model_adapter=None,
    quant_mode: str = "affine",
):
    """Stream through BF16 source, quantize per allocation, write shards."""
    import torch

    from rco_mlx.mlx_alloc import MlxQuantConfig

    cfg = allocation_to_config(assignment, group_size=group_size, mode=quant_mode)
    layer_bits = {}
    for path, bits in assignment.items():
        clean = path.removesuffix(".weight")
        layer_bits[clean] = bits
        if model_adapter is not None:
            mlx = model_adapter.map_key_to_mlx(path)
            if mlx:
                layer_bits[mlx.removesuffix(".weight")] = bits
                layer_bits[mlx] = bits

    layer_bits_with_suffix = {}
    for path, bits in assignment.items():
        if not path.endswith(".weight"):
            layer_bits_with_suffix[path + ".weight"] = bits
        layer_bits_with_suffix[path] = bits

    def map_key_to_mlx(key: str) -> str | None:
        if key.startswith("model.visual.") or key.startswith("visual."):
            return None
        mlx = key
        if mlx.startswith("model.language_model."):
            mlx = "language_model.model." + mlx[len("model.language_model."):]
        if model_adapter is not None:
            mapped = model_adapter.map_key_to_mlx(mlx)
            if mapped is not None:
                return mapped
            if mlx.endswith(".weight"):
                mapped_wo = model_adapter.map_key_to_mlx(mlx.removesuffix(".weight"))
                if mapped_wo is not None:
                    return mapped_wo
        return mlx

    def should_quantize(mlx_key: str) -> tuple[bool, int]:
        if (
            "norm" in mlx_key
            or "embed" in mlx_key
            or "lm_head" in mlx_key
            or "mtp" in mlx_key
            or (not mlx_key.endswith(".weight") and ".experts." not in mlx_key)
            or mlx_key.endswith(".bias")
            or ".conv.conv." in mlx_key
        ):
            return False, 4
        if model_adapter is not None:
            for prefix in model_adapter.non_quantized_prefixes():
                if f".{prefix}." in f".{mlx_key}." or mlx_key.startswith(f"{prefix}.") or mlx_key == prefix:
                    return False, 4
        bits = layer_bits_with_suffix.get(mlx_key)
        if bits is None:
            base = mlx_key[:-len(".weight")] if mlx_key.endswith(".weight") else mlx_key
            bits = layer_bits.get(base, None)
            if bits is None and ".experts." in base:
                parts = base.split(".")
                try:
                    experts_idx = parts.index("experts")
                    proj = parts[experts_idx + 2]
                    l = parts[3]
                    for prefix in ["feed_forward", "mlp"]:
                        stacked_key = f"model.layers.{l}.{prefix}.{proj}"
                        if stacked_key in layer_bits:
                            bits = layer_bits[stacked_key]
                            break
                        if stacked_key + ".weight" in layer_bits_with_suffix:
                            bits = layer_bits_with_suffix[stacked_key + ".weight"]
                            break
                except (ValueError, IndexError):
                    pass
        if bits is None:
            bits = 4
        return True, bits

    shards = sorted(source_dir.glob("*.safetensors"))
    output_dir.mkdir(parents=True, exist_ok=True)
    total_tensors = 0
    start = time.time()

    for i, shard_path in enumerate(shards):
        print(f"\n[quant] shard {i+1}/{len(shards)}: {shard_path.name}")
        results = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                mlx_key = map_key_to_mlx(key)
                if mlx_key is None:
                    results[key] = f.get_tensor(key).to(torch.bfloat16)
                    continue

                q_ok, bits = should_quantize(mlx_key)
                if q_ok:
                    tensor = f.get_tensor(key)
                    if tensor.ndim == 2:
                        w = mx.array(tensor.float().numpy())
                        q, scales, biases = mx.quantize(
                            w, bits=bits, group_size=group_size, mode=quant_mode
                        )
                        mx.eval(q, scales, biases)
                        base_key = mlx_key[:-len(".weight")] if mlx_key.endswith(".weight") else mlx_key
                        results[mlx_key] = torch.from_numpy(np.array(q).astype(np.uint32))
                        scales_np = np.array(scales)
                        biases_np = np.array(biases)
                        if quant_mode == "symmetric":
                            results[f"{base_key}.scales"] = torch.from_numpy(scales_np.astype(np.float16))
                        else:
                            use_fp16 = (
                                np.max(np.abs(scales_np)) <= 65504
                                and np.max(np.abs(biases_np)) <= 65504
                            )
                            if use_fp16:
                                results[f"{base_key}.scales"] = torch.from_numpy(scales_np.astype(np.float16))
                                results[f"{base_key}.biases"] = torch.from_numpy(biases_np.astype(np.float16))
                            else:
                                print(f"  {base_key}: overflow -> float32 scales/biases")
                                results[f"{base_key}.scales"] = torch.from_numpy(scales_np.astype(np.float32))
                                results[f"{base_key}.biases"] = torch.from_numpy(biases_np.astype(np.float32))
                    elif tensor.ndim == 3:
                        # Stacked MoE expert tensor: quantize ALL experts in batch
                        E, H, I = tensor.shape
                        flat = tensor.reshape(E * H, I)
                        w_flat = mx.array(flat.float().numpy())
                        q_flat, scales_flat, biases_flat = mx.quantize(
                            w_flat, bits=bits, group_size=group_size, mode=quant_mode
                        )
                        mx.eval(q_flat, scales_flat, biases_flat)
                        q_stacked = np.array(q_flat).reshape(E, H, -1).astype(np.uint32)
                        weight_key = mlx_key if mlx_key.endswith(".weight") else mlx_key + ".weight"
                        results[weight_key] = torch.from_numpy(q_stacked)

                        # Scales: stack per-expert scales
                        # For 3D tensors, MLX returns scales of shape [E, H//gs, ...]
                        # We store them stacked to match the weight layout
                        scales_np = np.array(scales_flat)
                        biases_np = np.array(biases_flat)
                        base_key = mlx_key[:-len(".weight")] if mlx_key.endswith(".weight") else mlx_key

                        if quant_mode == "symmetric":
                            results[f"{base_key}.scales"] = torch.from_numpy(scales_np.astype(np.float16))
                        else:
                            use_fp16 = (
                                np.max(np.abs(scales_np)) <= 65504
                                and np.max(np.abs(biases_np)) <= 65504
                            )
                            if use_fp16:
                                results[f"{base_key}.scales"] = torch.from_numpy(scales_np.astype(np.float16))
                                results[f"{base_key}.biases"] = torch.from_numpy(biases_np.astype(np.float16))
                            else:
                                print(f"  {base_key}: overflow -> float32 scales/biases")
                                results[f"{base_key}.scales"] = torch.from_numpy(scales_np.astype(np.float32))
                                results[f"{base_key}.biases"] = torch.from_numpy(biases_np.astype(np.float32))
                    else:
                        results[mlx_key] = tensor.to(torch.bfloat16)
                else:
                    results[key] = f.get_tensor(key).to(torch.bfloat16)

        out_shard = output_dir / f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(results, str(out_shard))
        total_tensors += len(results)
        elapsed = time.time() - start
        print(f"  saved {len(results)} tensors | {elapsed:.1f}s elapsed")

    # Post-process: rewrite config for MLX module paths
    if model_adapter is not None:
        print("\n[quant] post-processing: rewriting config for MLX keys ...")
        _rewrite_config_mlx_keys(output_dir, model_adapter, quant_mode=quant_mode)
        elapsed = time.time() - start
        print(f"  post-process done | {elapsed:.1f}s elapsed")

    # Copy non-weight files
    for f in source_dir.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        dest = output_dir / f.name
        if f.is_file() and not dest.exists():
            shutil.copy2(f, dest)

    print(f"\n[quant] done — {total_tensors} tensors, {time.time()-start:.1f}s total")


def _rewrite_config_mlx_keys(output_dir: Path, model_adapter, quant_mode: str = "affine"):
    """Rewrite config.json quantization keys from HF format to MLX format."""
    cfg_path = output_dir / "config.json"
    if not cfg_path.exists() or model_adapter is None:
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    quant = cfg.get("quantization", {})
    new_quant = {}
    for key, val in quant.items():
        if "." not in key:
            new_quant[key] = val
            continue
        mlx_key = model_adapter.map_key_to_mlx(key + ".weight")
        if mlx_key is None:
            continue
        module_path = mlx_key[:-7] if mlx_key.endswith(".weight") else mlx_key
        entry = dict(val)
        entry["mode"] = quant_mode
        new_quant[module_path] = entry

    # Keep only modules that actually have .scales on disk, so the config
    # reflects quantized artifacts (never declare an unquantized module).
    on_disk_keys = set()
    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text())
        on_disk_keys = set(idx.get("weight_map", {}).keys())
    else:
        from safetensors import safe_open as _so
        for shard in sorted(output_dir.glob("*.safetensors")):
            with _so(str(shard), framework="pt", device="cpu") as f:
                on_disk_keys.update(f.keys())
    on_disk_modules = {k[: -len(".scales")] for k in on_disk_keys if k.endswith(".scales")}
    filtered = {}
    for module_path, entry in new_quant.items():
        if "." not in module_path:
            filtered[module_path] = entry
            continue
        if module_path in on_disk_modules or (module_path + "") in on_disk_modules:
            filtered[module_path] = entry
    # Also drop top-level defaults? keep defaults (group_size/bits/mode) keys have no dot
    # Remove declared module paths lacking scales
    dropped = len(new_quant) - len(filtered)
    new_quant = filtered

    cfg["quantization"] = new_quant
    cfg["quantization_config"] = new_quant
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=False)
    print(f"  rewrote {len(quant)} config keys to MLX format ({len(new_quant)} entries, dropped {dropped} without scales)")


# ---------------------------------------------------------------------------
# MoE auto-detection
# ---------------------------------------------------------------------------

_MOE_ADAPTERS = {"olmoe", "qwen3_5_moe"}


def detect_moe_model(source_dir: Path, model_type: str | None) -> bool:
    if model_type and model_type.lower() in _MOE_ADAPTERS:
        return True
    cfg_path = source_dir / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            if cfg.get("num_experts", 0) > 1:
                return True
        except Exception:
            pass
    shards = sorted(source_dir.glob("*.safetensors"))
    if shards:
        try:
            index_path = source_dir / "model.safetensors.index.json"
            if index_path.exists():
                with open(index_path) as f:
                    weight_map = json.load(f).get("weight_map", {})
                return any(".experts." in k for k in weight_map)
            else:
                with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
                    return any(".experts." in k for k in f.keys())
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calibration_from_dataset(
    source_dir: Path,
    model_type: str,
    dataset: str,
    num_samples: int = 128,
    seq_length: int = 2048,
) -> Path:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(source_dir), trust_remote_code=True, local_files_only=True)
    print(f"[calib] loading dataset: {dataset} ({num_samples} samples x {seq_length} tokens)")

    if dataset == "wikitext":
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in ds["text"] if t.strip())
    elif dataset == "fineweb":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "default", split="train", streaming=True)
        texts = []
        for item in ds:
            texts.append(item["text"])
            if len(texts) >= num_samples:
                break
        text = "\n\n".join(texts)
    elif dataset == "c4":
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in ds:
            texts.append(item["text"])
            if len(texts) >= num_samples:
                break
        text = "\n\n".join(texts)
    else:
        sys.exit(f"Unknown calibration dataset: {dataset}. Choose: wikitext, fineweb, c4")

    all_ids = tokenizer.encode(text)
    target = num_samples * seq_length
    if len(all_ids) < target:
        all_ids = (all_ids * ((target + len(all_ids) - 1) // len(all_ids)))[:target]
    else:
        all_ids = all_ids[:target]

    ids_np = np.array(all_ids, dtype=np.int32).reshape(num_samples, seq_length)
    out = source_dir / "work" / f"calib_{dataset}_{num_samples}x{seq_length}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), ids_np)
    print(f"[calib] saved {out} ({ids_np.shape})")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="RCO MoE-only mixed-precision quantization pipeline (streaming, GPTQ-based)",
    )
    ap.add_argument("--source", required=True, help="BF16 source model dir")
    ap.add_argument("--output", required=True, help="Output quantized model dir")
    ap.add_argument("--target-bpw", type=float, required=True, help="Target bits per weight")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--default-bits", type=int, default=4)
    ap.add_argument("--model-type", default=None, help="Model adapter (olmoe, qwen3_5_moe)")
    ap.add_argument("--calibration", default=None, help="Path to calibration .npy")
    ap.add_argument("--calibration-dataset", default=None, help="Auto-generate calib: wikitext|fineweb|c4")
    ap.add_argument("--error-db", default=None, help="Pre-built error_db.json")
    ap.add_argument("--bits", default="2,3,4,6,8", help="Bitwidths to sweep")
    ap.add_argument("--quant-mode", default="affine", choices=["affine", "symmetric"],
                    help="Quantization mode (symmetric saves ~0.25 bpw on scales)")
    ap.add_argument("--no-gptq", action="store_true", help="Disable GPTQ scaling, use plain RTN")
    ap.add_argument("--ad-ladder", action="store_true",
                    help="Use AtomicChat AD ladder (rule-based) instead of RCO measurement")
    ap.add_argument("--skip-quantize", action="store_true", help="Allocate + validate only")
    ap.add_argument("--batch-size", type=int, default=4, help="Calibration batch size")

    args = ap.parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Resolve model adapter
    model_adapter = None
    if args.model_type:
        from rco_mlx.adapters import get
        adapter_cls = get(args.model_type)
        if adapter_cls is None:
            sys.exit(f"Unknown model type: {args.model_type}")
        model_adapter = adapter_cls()
        print(f"Using model adapter: {args.model_type}")

    # Validate MoE
    if not detect_moe_model(source, args.model_type):
        sys.exit("RCO pipeline is MoE-only. Use a model with num_experts > 1 or pass --model-type.")

    # --- Step 1: introspect ---
    print("[1/5] introspecting model …")
    layer_names, param_counts, groups = introspect_model(source, model_adapter=model_adapter)
    group_names = [g.layer_names[0] for g in groups]
    print(f"  {len(layer_names)} layers, {len(groups)} groups")

    # --- Step 2+3: allocation ---
    if args.ad_ladder:
        print("[2/5] AD ladder (rule-based) allocation ...")
        # Base AD ladder bits
        assignment = {}
        for g in groups:
            name = g.layer_names[0]
            low = name.lower()
            if "experts" in low or "switch_mlp" in low or "switch_glu" in low:
                if "down_proj" in low or "down" in low:
                    bits = 3
                else:
                    bits = 2
            elif any(k in low for k in ("norm", "conv1d", "a_log", "dt_bias", "embed", "lm_head")):
                continue  # keep BF16
            else:
                bits = 8
            assignment[name] = bits

        # Greedy adjustment to hit target bpw
        def weighted_bpw(assign):
            total_params = 0
            weighted = 0
            for k in assign:
                pc = param_counts.get(k, 0) or param_counts.get(k + ".weight", 0)
                total_params += pc
                weighted += assign[k] * pc
            if total_params == 0:
                return 0
            return weighted / total_params

        current = weighted_bpw(assignment)
        target = args.target_bpw
        print(f"  base AD ladder bpw: {current:.4f} (target {target})")

        # Priority for upgrades (most important first)
        def upgrade_priority(name):
            low = name.lower()
            if "self_attn" in low or "attention" in low:
                return 0
            if "shared" in low:
                return 1
            if "layers.0." in low or "layers.1." in low:
                return 2
            return 3

        # Priority for downgrades (least important first = highest number first)
        def downgrade_priority(name):
            low = name.lower()
            if "experts" in low or "switch" in low:
                # deep experts first
                idx = 0
                if "layers." in low:
                    try:
                        idx = int(low.split("layers.")[1].split(".")[0])
                    except Exception:
                        idx = 0
                return -idx  # deeper = higher priority to downgrade (more negative = later)
            return 100

        available_bits = [2, 3, 4, 5, 6, 8]

        # Group keys that must share the same bitwidth (sanitize stacking constraint).
        # Experts in the same layer+projection must all have the same bits.
        groups_map = {}
        for name in assignment:
            low = name.lower()
            if "experts" in low or "switch_mlp" in low or "switch_glu" in low:
                # group by (layer, projection)
                parts = name.split(".")
                layer_idx = None
                proj = parts[-1]
                for i, pt in enumerate(parts):
                    if pt == "layers" and i + 1 < len(parts):
                        layer_idx = parts[i + 1]
                gkey = f"layer{layer_idx}_{proj}"
            else:
                gkey = name
            groups_map.setdefault(gkey, []).append(name)

        group_keys = list(groups_map.keys())
        # priority by representative name
        group_keys_sorted_up = sorted(group_keys, key=lambda gk: upgrade_priority(groups_map[gk][0]))
        group_keys_sorted_down = sorted(group_keys, key=lambda gk: downgrade_priority(groups_map[gk][0]), reverse=True)

        if current < target:
            for gk in group_keys_sorted_up:
                if current >= target:
                    break
                names = groups_map[gk]
                cur = assignment[names[0]]
                higher = [b for b in available_bits if b > cur]
                if not higher:
                    continue
                nxt = min(higher)
                for n in names:
                    assignment[n] = nxt
                current = weighted_bpw(assignment)
        elif current > target:
            for gk in group_keys_sorted_down:
                if current <= target:
                    break
                names = groups_map[gk]
                cur = assignment[names[0]]
                lower = [b for b in available_bits if b < cur]
                if not lower:
                    continue
                nxt = max(lower)
                for n in names:
                    assignment[n] = nxt
                current = weighted_bpw(assignment)

        hist = Counter(assignment.values())
        print(f"  adjusted bpw: {current:.4f}")
        print(f"  {len(assignment)} groups allocated")
        print(f"  histogram: {dict(hist)}")
        report = {"achieved_bpw": current, "bit_histogram": dict(hist)}
        # Skip to emit+validate (step 4)
        _skip_error_db = True
    else:
        _skip_error_db = False

    if not args.ad_ladder:
        if args.error_db:
            error_db_path = Path(args.error_db).expanduser().resolve()
            if not error_db_path.exists():
                sys.exit(f"error_db not found: {error_db_path}")
            print(f"[2/5] using provided error_db: {error_db_path}")
        else:
            calib_path = None
            if args.calibration:
                calib_path = Path(args.calibration).expanduser().resolve()
                if not calib_path.exists():
                    sys.exit(f"Calibration file not found: {calib_path}")
            elif args.calibration_dataset:
                calib_path = build_calibration_from_dataset(
                    source_dir=source,
                    model_type=args.model_type or "default",
                    dataset=args.calibration_dataset,
                )
            else:
                sys.exit("Need --calibration, --calibration-dataset, or --error-db")

            error_db_path = output / "work" / "error_db.json"
            error_db_path.parent.mkdir(parents=True, exist_ok=True)
            build_error_db(
                source_dir=source,
                output_path=error_db_path,
                model_adapter=model_adapter,
                calibration_path=calib_path,
                bitwidths=[int(b) for b in args.bits.split(",")],
                group_size=args.group_size,
                quant_mode=args.quant_mode,
                use_gptq=not args.no_gptq,
            )

        # Load error_db for allocation
        losses, fracs, bitwidths = load_error_db(error_db_path)
        group_names = [g.layer_names[0] for g in groups]
        if losses.shape[0] != len(group_names):
            print(f"  warning: error_db has {losses.shape[0]} rows but {len(group_names)} groups")
            n = min(losses.shape[0], len(group_names))
            losses = losses[:n]
            fracs = fracs[:n]
            group_names = group_names[:n]

        # --- Step 3: allocation ---
        print(f"[3/5] RCO allocation  target={args.target_bpw} bpw ...")
        assignment, report = run_rco_allocation(
            losses, fracs, bitwidths, args.target_bpw, args.group_size, group_names
        )
        print(f"  achieved {report['achieved_bpw']:.4f} bpw")
        print(f"  histogram: {report['bit_histogram']}")

        # Force experts to uniform bitwidth for sanitize compatibility
        # (mixed bits per expert break OlmoeSparseMoeBlock stacking)
        # Adapt bitwidth to target bpw: lower targets -> lower expert bits
        if args.target_bpw <= 2.5:
            expert_bits = 2
        elif args.target_bpw <= 3.5:
            expert_bits = 3
        elif args.target_bpw <= 4.5:
            expert_bits = 4
        elif args.target_bpw <= 5.5:
            expert_bits = 5
        else:
            expert_bits = 6
        print(f"  expert bitwidth: {expert_bits} (target {args.target_bpw} bpw)")
        # Add all expert groups to assignment with uniform bitwidth
        for g in groups:
            name = g.layer_names[0]
            low = name.lower()
            if ("experts" in low or "switch_mlp" in low or "switch_glu" in low) and name not in assignment:
                assignment[name] = expert_bits
        for k in list(assignment.keys()):
            if "experts" in k.lower() or "switch_mlp" in k.lower() or "switch_glu" in k.lower():
                assignment[k] = expert_bits

    # --- Step 4: emit + validate ---
    print("[4/5] emitting MLX config + validating …")
    bf16_prefixes = model_adapter.non_quantized_prefixes() if model_adapter else None
    cfg = emit_and_validate(
        assignment, args.group_size, source, output, args.default_bits,
        quant_mode=args.quant_mode, bf16_prefixes=bf16_prefixes,
    )

    # --- Step 5: quantize ---
    if args.skip_quantize:
        print("[5/5] --skip-quantize set, done.")
        return

    print("[5/5] streaming quantization …")
    stream_quantize(
        source, output, assignment, args.group_size,
        model_adapter=model_adapter, quant_mode=args.quant_mode,
    )

    # Post-process: apply Qwen3.5 sanitize for quantized weights
    if args.model_type == "qwen3_5_moe":
        print("[post] applying Qwen3.5 sanitize for quantized expert weights ...")
        import json as _json
        import safetensors.torch as st
        for shard_path in sorted(output.glob("*.safetensors")):
            tensors = st.load_file(str(shard_path))
            new_tensors = dict(tensors)
            num_layers = 40
            num_experts = 256
            for l in range(num_layers):
                prefix = f"language_model.model.layers.{l}.mlp"
                for suffix in [".weight", ".scales", ".biases"]:
                    gate_up_key = f"{prefix}.experts.gate_up_proj{suffix}"
                    down_key = f"{prefix}.experts.down_proj{suffix}"
                    if gate_up_key in new_tensors:
                        gate_up = new_tensors.pop(gate_up_key)
                        if gate_up.dim() >= 3:
                            # 3D: (E, 2H, packed) -> split along dim 1
                            mid = gate_up.shape[1] // 2
                            new_tensors[f"{prefix}.switch_mlp.gate_proj{suffix}"] = gate_up[:, :mid, :].contiguous()
                            new_tensors[f"{prefix}.switch_mlp.up_proj{suffix}"] = gate_up[:, mid:, :].contiguous()
                        else:
                            # 2D: (E*2H, packed) -> split along dim 0
                            mid = gate_up.shape[0] // 2
                            E = num_experts
                            H = mid // E
                            packed = gate_up.shape[-1]
                            gate_up_3d = gate_up.reshape(E, 2 * H, packed)
                            gate = gate_up_3d[:, :H, :].reshape(E * H, packed).contiguous()
                            up = gate_up_3d[:, H:, :].reshape(E * H, packed).contiguous()
                            new_tensors[f"{prefix}.switch_mlp.gate_proj{suffix}"] = gate
                            new_tensors[f"{prefix}.switch_mlp.up_proj{suffix}"] = up
                    if down_key in new_tensors:
                        down = new_tensors.pop(down_key)
                        # down is already correct shape (2D or 3D), just rename
                        new_tensors[f"{prefix}.switch_mlp.down_proj{suffix}"] = down
            import safetensors.torch as st
            import os
            tmp_path = "/tmp/" + shard_path.name + ".tmp"
            st.save_file(new_tensors, tmp_path)
            os.replace(tmp_path, str(shard_path))
            print(f"  sanitized {shard_path.name}")
        # Update config.json quantization keys
        cfg_path = output / "config.json"
        cfg_data = json.loads(cfg_path.read_text())
        qc = cfg_data.get("quantization", {})
        if isinstance(qc, dict):
            final_qc = {}
            for k, v in qc.items():
                nk = k.replace("experts.gate_up_proj", "switch_mlp.gate_proj")
                nk = nk.replace("experts.down_proj", "switch_mlp.down_proj")
                final_qc[nk] = v
            cfg_data["quantization"] = final_qc
            cfg_path.write_text(json.dumps(cfg_data, indent=2))
            print("  config updated")
    print(f"Config: {output / 'config.json'}")
    print(f"{'='*60}")

    # Summary
    bit_hist = Counter(assignment.values())
    actual_bits = np.array([effective_bits(int(bw), args.group_size) for bw in assignment.values()])
    achieved_effective = float(actual_bits.mean())
    print(f"\nTarget: {args.target_bpw:.2f} bpw (nominal)")
    print(f"Achieved effective bpw: {achieved_effective:.4f} (includes scale overhead)")
    print(f"Quantized tensors: {len(assignment)}")
    print(f"Bitwidth distribution:")
    for bw in sorted(bit_hist.keys(), reverse=True):
        print(f"  {int(bw)}-bit: {bit_hist[bw]} tensors")


if __name__ == "__main__":
    main()
