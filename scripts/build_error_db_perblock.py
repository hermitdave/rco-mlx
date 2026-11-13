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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rco_mlx.hessians import hessian_key_for, load_hessian_factors  # noqa: E402

SKIP_SUBSTRINGS = (
    "ngram_embedding",   # 51B PLE table -- uniform 3-bit, not RCO
    "embed_tokens",      # BF16 preserved
    "lm_head",           # BF16 preserved
)
SKIP_PREFIXES = ("mtp.",)  # MTP head -- BF16 preserved

EXPERT_RE = re.compile(
    r"^((?:model\.)?(?:language_model\.)?layers\.\d+\.(?:mlp|feed_forward)\.experts)"
    r"\.\d+\.([a-z_]+)\.weight$"
)


def hessian_errors(w, bitwidths, group_size, lam, vec) -> dict:
    """Per-bitwidth Hessian-weighted reconstruction error.

    The output error implied by the quantized weight is
    ``tr(dW H dW^T)``; with the captured low-rank factor ``H ~ V diag(lam) V^T``
    this is ``||dW V sqrt(lam)||_F^2``, normalized per element so it stays
    comparable to the plain MSE the DB already reports.

    Unlike GPTQ this never inverts H, so it is well-defined even when the
    per-expert Hessian is rank-deficient (which it is for MoE: n < d).
    """
    rows, cols = w.shape
    if cols != vec.shape[0]:
        raise ValueError(f"hessian width {vec.shape[0]} != tensor width {cols}")
    Vs = vec.astype(np.float32) * np.sqrt(np.maximum(lam, 0.0)).astype(np.float32)[None, :]
    out = {}
    for bw in bitwidths:
        qmax = float((1 << (bw - 1)) - 1)
        grouped = w.reshape(rows, -1, group_size)
        max_abs = np.abs(grouped).max(axis=-1, keepdims=True)
        s = max_abs / qmax
        s = np.where(s == 0, 1.0, s)
        w_q = np.clip(np.round(grouped / s), -qmax, qmax) * s
        dw = (grouped - w_q).reshape(rows, cols).astype(np.float32)
        proj = dw @ Vs  # [rows, r]
        out[str(bw)] = {"error": float((proj ** 2).sum() / (rows * cols)), "clip": 1.0}
    return out


def is_skipped(key: str) -> bool:
    if key.startswith(SKIP_PREFIXES):
        return True
    return any(s in key for s in SKIP_SUBSTRINGS)


def expert_row_key(key: str):
    m = EXPERT_RE.match(key)
    if m:
        return f"{m.group(1)}.{m.group(2)}"  # no .weight suffix
    return None


def rtn_errors(w: np.ndarray, bitwidths, col_weights=None, clip_search: bool = False,
               group_size: int = 64) -> dict:
    """Per-group RTN MSE at each bitwidth, mirroring MLX affine grouping.

    Scales are computed per output row and per ``group_size`` block along
    the input dimension -- the same grouping ``mx.quantize`` uses -- not
    per column. (A per-column model systematically underestimates the
    damage of low-bit quantization, and any clip ratio tuned under it
    does not transfer: a column-max over 2048 rows is far larger than a
    64-sample group-max, so saturating 40% of that range is mild in one
    model and destructive in the other.)

    ``col_weights`` (per input column, mean-1) turn the plain MSE into an
    activation-weighted one, matching the diagonal-Hessian objective an
    imatrix encodes. ``clip_search`` also sweeps per-group scale
    multipliers below the max-abs scale and keeps the MSE-optimal one
    per bitwidth (recorded under ``"clip"``), with grid saturation
    modelled so the search transfers to the real quantizer.
    """
    out = {}
    flat = w.astype(np.float32, copy=False)
    if flat.ndim != 2:
        flat = flat.reshape(-1, flat.shape[-1])
    if col_weights is not None:
        cw = np.asarray(col_weights, dtype=np.float64).reshape(-1)
        if cw.shape[0] != flat.shape[1]:
            cw = None  # shape mismatch (e.g. fused qkv): fall back unweighted
        elif cw.sum() <= 0:
            # Never activated: quantization error here has no output effect.
            return {str(bw): {"error": 0.0, "clip": 1.0} for bw in bitwidths}
    else:
        cw = None
    # Mild clips only: per-group saturation at 0.6-0.8 crushes ~5-9% of
    # weights per group, which weight-MSE (a linear proxy) tolerates but
    # softmax/router nonlinearities may not -- measured catastrophic on
    # OLMoE at 0.6 despite a better proxy error. Cap the search at 0.9.
    clips = (1.0, 0.95, 0.9) if clip_search else (1.0,)
    gs = group_size
    if flat.shape[1] % gs != 0:
        gs = flat.shape[1]  # single group per row: fall back to whole-row scale
    grouped = flat.reshape(flat.shape[0], -1, gs)
    max_abs = np.abs(grouped).max(axis=-1)  # [out, groups]
    if cw is not None:
        cw_g = cw.reshape(-1, gs)  # [groups, gs], tiled per row below
    for bw in bitwidths:
        qmax = float((1 << (bw - 1)) - 1)
        best_err, best_clip = None, 1.0
        for clip in clips:
            scale = max_abs * clip / qmax  # [out, groups]
            scale = np.where(scale == 0, 1.0, scale)
            w_q = np.clip(
                np.round(grouped / scale[:, :, None]), -qmax, qmax
            ) * scale[:, :, None]
            sq = (grouped - w_q) ** 2  # [out, groups, gs]
            if cw is not None:
                # Weighted mean over columns, then over all elements.
                err = float((sq * cw_g[None, :, :]).sum() / cw_g.sum() / flat.shape[0])
            else:
                err = float(sq.mean())
            if best_err is None or err < best_err:
                best_err, best_clip = err, clip
        out[str(bw)] = {"error": best_err, "clip": best_clip}
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
    ap.add_argument("--imatrix", default=None,
                    help="Path to a GGUF imatrix file or local .npz capture; "
                         "weights RTN errors per input column by activation "
                         "importance and aggregates expert rows by routing counts")
    ap.add_argument("--clip-search", action="store_true",
                    help="Sweep per-column scale multipliers and keep the "
                         "MSE-optimal clip per tensor/bitwidth")
    ap.add_argument("--hessian-dir", default=None,
                    help="Directory of per-expert Hessian factors from "
                         "scripts/capture_hessians.py. When given, expert "
                         "tensors are scored by the Hessian-weighted output "
                         "error tr(dW H dW^T) instead of plain weight MSE")
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

    imatrix = None
    if args.imatrix:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from rco_mlx.imatrix import Imatrix

        imatrix = Imatrix(Path(args.imatrix))
        print(f"Imatrix: {args.imatrix} ({len(imatrix.tensor_names())} tensors)", flush=True)
    if args.clip_search:
        print("Clip search: enabled (scales 1.0/0.9/0.8/0.7/0.6 per tensor)", flush=True)

    hessian_factors = {}
    if args.hessian_dir:
        hessian_factors = load_hessian_factors(args.hessian_dir)
        print(f"Hessian factors: {len(hessian_factors)} expert tensors "
              f"from {args.hessian_dir}", flush=True)

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
        # Migrate pre-imatrix checkpoints (no weight_sum/clip_sum fields).
        for a in agg.values():
            a.setdefault("weight_sum", float(a.get("params", 0)))
            a.setdefault("clip_sum", {str(b): 0.0 for b in bitwidths})
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
                col_w, route_w = None, 1.0
                if imatrix is not None:
                    mapped = imatrix.col_weights_for_hf_key(key)
                    if mapped is not None:
                        col_w, route_w = mapped
                hk = hessian_key_for(key)
                if hessian_factors and hk in hessian_factors:
                    lam, vec = hessian_factors[hk]
                    errs = hessian_errors(w, bitwidths, args.group_size, lam, vec)
                else:
                    errs = rtn_errors(w, bitwidths, col_weights=col_w,
                                      clip_search=args.clip_search,
                                      group_size=args.group_size)
                del w
                row = expert_row_key(key)
                if row is not None:
                    a = agg.setdefault(row, {"err_sum": {str(b): 0.0 for b in bitwidths},
                                             "params": 0, "members": 0,
                                             "weight_sum": 0.0,
                                             "clip_sum": {str(b): 0.0 for b in bitwidths}})
                    agg_w = route_w if imatrix is not None else float(n)
                    for b in bitwidths:
                        a["err_sum"][str(b)] += errs[str(b)]["error"] * agg_w
                        a["clip_sum"][str(b)] += errs[str(b)]["clip"] * agg_w
                    a["weight_sum"] += agg_w
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

    # Finalize expert aggregates -> weighted-mean rows (routing-weighted
    # when an imatrix was used, param-weighted otherwise).
    clips: dict = {}
    for row, a in agg.items():
        denom = a.get("weight_sum") or a["params"] or 1
        rows[row] = {str(b): {"error": a["err_sum"][str(b)] / denom,
                              "clip": a["clip_sum"][str(b)] / denom}
                     for b in bitwidths}
        param_counts[row] = a["params"]
        clips[row] = {str(b): rows[row][str(b)]["clip"] for b in bitwidths}
    for key, errs in rows.items():
        if key in clips:
            continue
        clips[key] = {str(b): errs[str(b)].get("clip", 1.0) for b in bitwidths}

    ordered = {k: rows[k] for k in sorted(rows)}
    ordered_pc = {k: param_counts.get(k, 0) for k in ordered}
    error_db = {"layers": ordered, "param_counts": ordered_pc,
                "clips": {k: clips[k] for k in sorted(clips)}}
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(error_db, indent=2))
    tmp.replace(out_path)
    ckpt_path.unlink(missing_ok=True)
    print(f"Saved {len(ordered)} error entries to {out_path}", flush=True)


if __name__ == "__main__":
    main()
