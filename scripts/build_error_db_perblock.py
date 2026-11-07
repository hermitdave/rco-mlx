"""Per-block error DB builder for MoE models (single-pass, resumable).

Memory-bounded and I/O-bounded: reads each safetensors shard exactly once,
measures per-tensor RTN reconstruction error at candidate bitwidths, and
checkpoints progress after every shard so a restart resumes where it stopped.

Design notes (Groot-65B / qwen4_exp):
- Routed-expert tensors are measured individually but aggregated into one
  row per (layer, projection) -- experts are forced to a uniform bitwidth
  downstream (MLX sanitize stacking), so per-expert rows are wasted I/O.
- BF16-preserved tensors (n-gram PLE table, MTP head, embeddings, lm_head)
  are skipped: they are not RCO-allocated.
- Errors use per-column scale RTN in float32 numpy (matches the previous
  mx measure semantics, no GPU watchdog risk, deterministic).
- Final ``layers`` dict is sorted by key so downstream name-alignment is
  stable. Output schema is unchanged:
  {"layers": {key: {bw: {"error": mse}}}, "param_counts": {key: int}}

Usage:
    python scripts/build_error_db_perblock.py \
        --source /path/to/model \
        --output /path/to/work/error_db.json \
        --bitwidths 2,3,4,6,8 --group-size 64
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np

SKIP_SUBSTRINGS = (
    "ngram_embedding",   # 51B PLE table -- uniform 3-bit, not RCO
    "embed_tokens",      # BF16 preserved
    "lm_head",           # BF16 preserved
)
SKIP_PREFIXES = ("mtp.",)  # MTP head -- BF16 preserved

EXPERT_RE = re.compile(
    r"^((?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.experts)\.\d+\.([a-z_]+)\.weight$"
)


def is_skipped(key: str) -> bool:
    if key.startswith(SKIP_PREFIXES):
        return True
    return any(s in key for s in SKIP_SUBSTRINGS)


def expert_row_key(key: str):
    m = EXPERT_RE.match(key)
    if m:
        return f"{m.group(1)}.{m.group(2)}"  # no .weight suffix
    return None


def rtn_errors(w: np.ndarray, bitwidths) -> dict:
    """Per-column-scale RTN MSE at each bitwidth. w is float32 numpy."""
    out = {}
    flat = w.astype(np.float32, copy=False)
    for bw in bitwidths:
        qmax = float((1 << (bw - 1)) - 1)
        scale = np.abs(flat).max(axis=0) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        w_q = np.round(flat / scale) * scale
        err = float(np.mean((flat - w_q) ** 2))
        out[str(bw)] = {"error": err}
    return out


def save_checkpoint(path: Path, done_keys, agg, rows, param_counts):
    """Persist partial state so a restart can resume."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = {
        "done_keys": sorted(done_keys),
        "agg": {k: {"err_sum": v["err_sum"], "params": v["params"],
                    "members": v["members"]} for k, v in agg.items()},
        "rows": rows,
        "param_counts": param_counts,
    }
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bitwidths", default="2,3,4,6,8")
    ap.add_argument("--group-size", type=int, default=64)  # accepted; RTN here is per-column
    ap.add_argument("--quant-mode", default="affine")
    ap.add_argument("--limit-shards", type=int, default=None, help="smoke-test helper")
    # Legacy passthrough args (ignored; kept for caller compatibility).
    ap.add_argument("--model-type", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--calibration-dataset", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--use-gptq", action="store_true", default=False)
    ap.add_argument("--no-gptq", action="store_true")
    args = ap.parse_args()

    bitwidths = [int(b) for b in args.bitwidths.split(",")]
    source = Path(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".checkpoint.json")

    from safetensors import safe_open

    # Shard discovery + key->shard map via index (fast, no header reads).
    index_path = source / "model.safetensors.index.json"
    key_to_shard = {}
    if index_path.exists():
        idx = json.loads(index_path.read_text())
        key_to_shard = dict(idx.get("weight_map", {}))
    shards = sorted(source.glob("model-*.safetensors")) or sorted(source.glob("*.safetensors"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"Shards: {len(shards)} (of full set), bitwidths: {bitwidths}", flush=True)

    # Resume state.
    done_keys = set()
    agg = {}   # row_key -> {err_sum: {bw: weighted}, params, members}
    rows = {}  # row_key -> {bw: {"error": mse}} for non-expert tensors
    param_counts = {}
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        done_keys = set(ckpt.get("done_keys", []))
        agg = ckpt.get("agg", {})
        rows = ckpt.get("rows", {})
        param_counts = ckpt.get("param_counts", {})
        print(f"Resuming: {len(done_keys)} tensors already done", flush=True)
    elif out_path.exists():
        print(f"Output already exists ({out_path}); re-running will rebuild it.", flush=True)

    total_keys = 0
    for si, sf in enumerate(shards):
        n_this = 0
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.endswith(".weight") or is_skipped(key):
                    continue
                if key in done_keys:
                    continue
                total_keys += 1
                w = f.get_tensor(key).float().numpy()
                n = int(w.size)
                errs = rtn_errors(w, bitwidths)
                del w
                row = expert_row_key(key)
                if row is not None:
                    a = agg.setdefault(row, {"err_sum": {str(b): 0.0 for b in bitwidths},
                                             "params": 0, "members": 0})
                    for b in bitwidths:
                        a["err_sum"][str(b)] += errs[str(b)]["error"] * n
                    a["params"] += n
                    a["members"] += 1
                else:
                    rows[key] = errs
                    param_counts[key] = n
                done_keys.add(key)
                n_this += 1
        print(f"[{si+1}/{len(shards)}] {sf.name}: +{n_this} tensors, "
              f"total done {len(done_keys)}", flush=True)
        save_checkpoint(ckpt_path, done_keys, agg, rows, param_counts)

    # Finalize expert aggregates -> weighted-mean rows.
    for row, a in agg.items():
        p = a["params"] or 1
        rows[row] = {str(b): {"error": a["err_sum"][str(b)] / p} for b in bitwidths}
        param_counts[row] = a["params"]

    ordered = {k: rows[k] for k in sorted(rows)}
    ordered_pc = {k: param_counts.get(k, 0) for k in ordered}
    error_db = {"layers": ordered, "param_counts": ordered_pc}
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(error_db, indent=2))
    tmp.replace(out_path)
    ckpt_path.unlink(missing_ok=True)
    print(f"Saved {len(ordered)} error entries to {out_path}", flush=True)


if __name__ == "__main__":
    main()
