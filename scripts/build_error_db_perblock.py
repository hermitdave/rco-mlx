"""Per-block Hessian/error DB builder for MoE models.

Memory-bounded: loads one shard at a time, processes one transformer block at a time.
No RTN 8-bit reference needed — uses BF16 source directly.

Usage:
    python scripts/build_error_db_perblock.py \
        --source /path/to/model \
        --output /path/to/work \
        --model-type qwen3_5_moe \
        --calibration-dataset wikitext \
        --target-bpw 3.0
"""
import argparse, json, gc, os
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_config
from mlx_lm.models.qwen3_5 import TextModelArgs, DecoderLayer
from mlx_lm.utils import _get_classes
import torch

def load_block_weights(source: Path, block_idx: int) -> dict:
    """Load weights for a single transformer block from all shards (one at a time)."""
    prefix = f"model.language_model.layers.{block_idx}."
    weights = {}
    shard_files = sorted(source.glob("model-*.safetensors"))
    for sf in shard_files:
        shard = mx.load(str(sf))
        for k, v in shard.items():
            if k.startswith(prefix):
                weights[k] = v
        del shard
        gc.collect()
    return weights

def sanitize_key(k: str, block_idx: int) -> str:
    """Strip block prefix from weight key."""
    prefix = f"model.language_model.layers.{block_idx}."
    return k[len(prefix):]

def measure_layer_errors(weight_bf16, bitwidths, group_size=64):
    """Measure reconstruction error for a layer at each bitwidth."""
    errors = {}
    w = weight_bf16.astype(mx.float32)
    for bw in bitwidths:
        # Simple per-channel quantization at bw bits
        qmax = (1 << (bw - 1)) - 1
        scale = mx.abs(w).max(axis=0) / qmax
        scale = mx.where(scale == 0, mx.ones_like(scale), scale)
        w_q = mx.round(w / scale) * scale
        err = float(mx.mean((w - w_q) ** 2))
        errors[str(bw)] = {"error": err}
    return errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rtn-source", default=None)  # ignored (no RTN needed)
    ap.add_argument("--model-type", default="qwen3_5_moe")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--calibration-dataset", default="wikitext")
    ap.add_argument("--bitwidths", default="2,3,4,5,6,8")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--quant-mode", default="affine")
    ap.add_argument("--use-gptq", action="store_true", default=True)
    ap.add_argument("--no-gptq", action="store_true")
    ap.add_argument("--max-calib-samples", type=int, default=4)
    ap.add_argument("--max-calib-tokens", type=int, default=512)
    args = ap.parse_args()

    bitwidths = [int(b) for b in args.bitwidths.split(",")]

    source = Path(args.source)
    out_dir = Path(args.output)
    if out_dir.suffix == ".json":
        out_dir = out_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(source)
    text_config = config.get("text_config", config)
    import inspect
    valid_keys = set(inspect.signature(TextModelArgs.__init__).parameters.keys()) - {"self"}
    filtered_config = {k: v for k, v in text_config.items() if k in valid_keys}
    model_args = TextModelArgs(**filtered_config)

    # Load calibration data
    if args.calibration:
        calib_path = Path(args.calibration)
    else:
        calib_path = source / "work" / f"calib_{args.calibration_dataset}_128x2048.npy"
    if not calib_path.exists():
        print(f"Calibration not found at {calib_path}, generating...")
        # Use wikitext fallback
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(ds["text"][:1000])
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(source))
        tokens = tok.encode(text)[:args.max_calib_tokens * args.max_calib_samples]
        calib = np.array(tokens).reshape(args.max_calib_samples, -1)[:, :args.max_calib_tokens]
        calib_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(calib_path, calib)
    else:
        calib = np.load(calib_path)[:args.max_calib_samples, :args.max_calib_tokens]
    print(f"Calib shape: {calib.shape}")

    # Load embed_tokens (needed for initial hidden state)
    embed_weights = {}
    for sf in sorted(source.glob("model-*.safetensors")):
        shard = mx.load(str(sf))
        for k, v in shard.items():
            if "embed_tokens" in k:
                embed_weights[k] = v
        del shard
        gc.collect()
    # Find embed_tokens weight
    embed_key = [k for k in embed_weights if "embed_tokens.weight" in k][0]
    embed_w = embed_weights[embed_key]
    embed = nn.Embedding(model_args.vocab_size, model_args.hidden_size)
    embed.weight = embed_w

    # Process blocks one at a time
    error_layers = {}
    param_counts = {}
    hidden = embed(mx.array(calib))
    mx.eval(hidden)

    num_layers = model_args.num_hidden_layers
    for i in range(num_layers):
        print(f"[{i+1}/{num_layers}] Processing block {i}...")
        block_input = hidden

        block_weights = load_block_weights(source, i)
        sanitized = {}
        for k, v in block_weights.items():
            sanitized[sanitize_key(k, i)] = v

        sanitized2 = {}
        for k, v in sanitized.items():
            if "experts.gate_up_proj" in k:
                gate_key = k.replace("experts.gate_up_proj", "switch_mlp.gate_proj.weight")
                up_key = k.replace("experts.gate_up_proj", "switch_mlp.up_proj.weight")
                sanitized2[gate_key] = v[:, :v.shape[1]//2, :]
                sanitized2[up_key] = v[:, v.shape[1]//2:, :]
            elif "experts.down_proj" in k:
                down_key = k.replace("experts.down_proj", "switch_mlp.down_proj.weight")
                sanitized2[down_key] = v
            elif "conv1d.weight" in k:
                sanitized2[k] = v.transpose(0, 2, 1)
            else:
                if not k.endswith(".weight") and not k.endswith(".bias") and not k.endswith(".scales") and not k.endswith(".biases") and not k.endswith("A_log") and not k.endswith("dt_bias"):
                    k = k + ".weight"
                sanitized2[k] = v
        sanitized = sanitized2

        block = DecoderLayer(args=model_args, layer_idx=i)
        block.load_weights(list(sanitized.items()))
        mx.eval(block.parameters())

        for wk, wv in sanitized.items():
            if wk.endswith(".weight"):
                full_name = f"model.language_model.layers.{i}.{wk}"
                errs = measure_layer_errors(wv, bitwidths)
                error_layers[full_name] = errs
                param_counts[full_name] = int(np.prod(wv.shape))

        hidden = block(hidden, mask=None, cache=None)
        mx.eval(hidden)

        del block, block_weights, sanitized
        gc.collect()
        mx.clear_cache()

    error_db = {"layers": error_layers, "param_counts": param_counts}
    error_db_path = (Path(args.output) if Path(args.output).suffix == ".json" else out_dir / "error_db.json")
    with open(error_db_path, "w") as f:
        json.dump(error_db, f, indent=2)
    print(f"Saved {len(error_layers)} error entries to {error_db_path}")

if __name__ == "__main__":
    main()