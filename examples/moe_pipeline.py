#!/usr/bin/env python3
"""
RCO mixed-precision quantization pipeline for MLX.

Works on both dense transformers (Llama/Qwen/LFM3-style, via the generic
``default`` adapter) and MoE models (olmoe, qwen3_5_moe, qwen4_exp,
lfm2_moe, granitemoehybrid). Streaming-only, per-tensor RTN error DB.

Usage (dense):
    python examples/moe_pipeline.py \\
        --source /path/to/bf16-dense \\
        --output /path/to/output \\
        --target-bpw 4.0 \\
        --group-size 64 \\
        --calibration-dataset wikitext

Usage (MoE):
    python examples/moe_pipeline.py \\
        --source /path/to/bf16-moe \\
        --output /path/to/output \\
        --target-bpw 3.5 \\
        --group-size 64 \\
        --model-type olmoe \\
        --calibration-dataset wikitext
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
from rco_mlx.expert_alloc import (
    apply_gguf_recipe,
    enforce_group_uniformity,
    expert_group_key,
    fill_missing_expert_groups,
    mask_values_below_min_bits,
    uniform_expert_bits,
)


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
    imatrix_path: Path | None = None,
    clip_search: bool = False,
    hessian_dir: Path | None = None,
) -> Path:
    """Build GPTQ-based error_db using the streaming two-phase approach."""
    script = Path(__file__).parent.parent / "scripts" / "build_error_db_perblock.py"
    cmd = [
        sys.executable, str(script),
        "--source", str(source_dir),
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
    if imatrix_path:
        cmd += ["--imatrix", str(imatrix_path)]
    if clip_search:
        cmd.append("--clip-search")
    if hessian_dir:
        cmd += ["--hessian-dir", str(hessian_dir)]
    if getattr(model_adapter, "quantize_embeddings", False):
        cmd.append("--quantize-embeddings")

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
    expert_min_bits: int = 2,
    group_is_expert: list[bool] | None = None,
    group_is_embedding: list[bool] | None = None,
    embedding_min_bits: int = 2,
) -> tuple[dict[str, int], dict]:
    table = LossTable(
        losses=losses,
        fracs=fracs,
        bitwidths=bitwidths,
        group_size=group_size,
    )
    values = table.values()
    if expert_min_bits > int(bitwidths.min()):
        if group_is_expert is None:
            sys.exit("--expert-min-bits requires expert group flags")
        values = mask_values_below_min_bits(
            values, bitwidths, group_is_expert, expert_min_bits
        )
        print(f"  expert floor: {expert_min_bits}-bit minimum enforced in DP "
              f"({sum(group_is_expert)} expert groups)")
    if embedding_min_bits > int(bitwidths.min()):
        if group_is_embedding is None:
            sys.exit("--embedding-min-bits requires embedding group flags")
        values = mask_values_below_min_bits(
            values, bitwidths, group_is_embedding, embedding_min_bits
        )
        print(f"  embedding floor: {embedding_min_bits}-bit minimum enforced "
              f"in DP ({sum(group_is_embedding)} embedding groups)")
    idx = budget_dp(
        values, table.fracs, table.actual_bits(), target_bpw, resolution=10000
    )
    bits_per_group = table.bitwidths[idx]
    assignment = {name: int(b) for name, b in zip(group_names, bits_per_group)}
    report = {
        "target_bpw": float(target_bpw),
        "achieved_bpw": table.achieved_bpw(bits_per_group),
        "n_groups": table.n_groups,
        "bit_histogram": {
            int(b): int((bits_per_group == b).sum())
            for b in sorted(set(bits_per_group.tolist()))
        },
        "group_size": group_size,
        "transform": table.transform,
    }
    return assignment, report


def load_error_db(path: Path, group_names: list | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load error DB, optionally aligned to ``group_names`` by module name.

    When group_names is given, each group is matched to its error row by
    name (stripping the ``.weight`` suffix); expert modules fall back to
    their per-(layer, projection) aggregate row; anything unmatched gets
    the median row (it will be forced to expert/default bits downstream).
    """
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

    def _row(layer_path, bw_data):
        params = param_counts.get(layer_path, 0)
        mse = {str(bw): bw_data[str(bw)]["error"] for bw in bitwidths if str(bw) in bw_data}
        return {"layer": layer_path, "tensor": layer_path, "params": params, "mse": mse}

    by_name = {}
    for layer_path, bw_data in db["layers"].items():
        by_name[layer_path] = _row(layer_path, bw_data)
        by_name[layer_path.removesuffix(".weight")] = by_name[layer_path]

    if group_names is None:
        rows = []
        seen = set()
        for r in by_name.values():
            if r["layer"] in seen:
                continue
            seen.add(r["layer"])
            rows.append(r)
    else:
        import re as _re
        rows = []
        for name in group_names:
            row = by_name.get(name) or by_name.get(name + ".weight")
            if row is None:
                m = _re.match(r"^((?:model\.)?(?:language_model\.)?layers\.\d+\.(?:mlp|feed_forward)\.experts)\.\d+\.([a-z0-9_]+)$", name)
                if m:
                    row = by_name.get(f"{m.group(1)}.{m.group(2)}")
            if row is None:
                continue  # unmatched (e.g. BF16-preserved): filled below
            rows.append((name, row))
        if not rows:
            sys.exit("error_db has no rows matching the model's groups")
        # Median fallback for unmatched groups (params=0: excluded from the
        # budget math; their bits are forced/defaulted downstream, e.g.
        # routed experts get the uniform expert bitwidth).
        med_losses = np.median(
            np.array([[r["mse"][str(b)] for b in bitwidths] for _, r in rows], dtype=np.float64),
            axis=0,
        )
        filled = []
        row_map = dict(rows)
        for name in group_names:
            if name in row_map:
                filled.append(row_map[name])
            else:
                filled.append({"layer": name, "tensor": name, "params": 0,
                               "mse": {str(b): float(m) for b, m in zip(bitwidths, med_losses)}})
        rows = filled

    # Drop rows with any missing bitwidth (should not happen with a fresh DB).
    rows = [r for r in rows if all(str(b) in r["mse"] for b in bitwidths)]
    bitwidths_arr = np.array(bitwidths, dtype=np.int64)
    losses = np.array(
        [[r["mse"][str(b)] for b in bitwidths] for r in rows],
        dtype=np.float64,
    )
    params = np.array([r["params"] for r in rows], dtype=np.float64)
    fracs = params / max(params.sum(), 1.0)
    return losses, fracs, bitwidths_arr


def load_clip_map(
    error_db_path: Path, assignment: dict[str, int]
) -> dict[str, float]:
    """Resolve per-group clip ratios from the DB's ``clips`` table.

    Keys mirror ``load_error_db`` alignment: direct name, ``+.weight``, or
    the per-(layer, projection) expert aggregate. Groups without a clip
    entry get 1.0 (no clipping).
    """
    try:
        db = json.loads(Path(error_db_path).read_text())
    except Exception:
        return {}
    clips = db.get("clips", {})
    if not clips:
        return {}
    import re as _re

    out: dict[str, float] = {}
    for name, bits in assignment.items():
        row = clips.get(name) or clips.get(name + ".weight")
        if row is None:
            m = _re.match(
                r"^((?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.experts)"
                r"\.\d+\.([a-z_]+)$",
                name,
            )
            if m:
                row = clips.get(f"{m.group(1)}.{m.group(2)}")
        if row is not None:
            out[name] = float(row.get(str(int(bits)), 1.0))
    return out


def resolve_imatrix(args, source: Path, output: Path) -> Path | None:
    """Resolve an imatrix path: explicit > HF download > local build > none.

    Order follows the user's preference: a published imatrix on HF is
    always preferred over building one; a local build only happens when
    the BF16 source fits in memory (or with --force via the capture
    script's own gate).
    """
    from rco_mlx.imatrix import (
        download_imatrix,
        find_imatrix_on_hf,
        fits_in_memory,
    )

    if args.imatrix:
        path = Path(args.imatrix).expanduser().resolve()
        if not path.exists():
            sys.exit(f"--imatrix file not found: {path}")
        print(f"Using local imatrix: {path}")
        return path

    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)

    if args.imatrix_repo:
        filename = find_imatrix_on_hf(args.imatrix_repo, args.imatrix_filename)
        if filename:
            print(f"Found imatrix on HF: {args.imatrix_repo}/{filename} "
                  f"-- downloading...")
            return download_imatrix(args.imatrix_repo, filename, work)
        print(f"No imatrix found in HF repo {args.imatrix_repo} "
              f"({args.imatrix_filename or 'any *imatrix* file'})")

    if args.build_imatrix:
        fits, model_bytes, ram = fits_in_memory(source)
        print(f"Local imatrix build gate: model {model_bytes/1e9:.1f} GB, "
              f"RAM {ram/1e9:.1f} GB, fits={fits}")
        if not fits:
            sys.exit(
                "BF16 source does not fit in memory for a local imatrix "
                "capture (70% RAM gate). Use a published imatrix via "
                "--imatrix-repo, or run on a larger machine."
            )
        calib = None
        if args.calibration:
            calib = Path(args.calibration).expanduser().resolve()
        elif args.calibration_dataset:
            # the pipeline saves dataset calibrations under <source>/work
            cands = sorted((source / "work").glob("calib_*.npy"))
            if cands:
                calib = cands[-1]
        if calib is None or not calib.exists():
            sys.exit(
                "imatrix build needs a calibration: pass --calibration or "
                "--calibration-dataset (the pipeline saves the latter under "
                "<source>/work/calib_*.npy)"
            )
        capture = (
            Path(__file__).parent.parent / "scripts" / "capture_imatrix.py"
        )
        imatrix_path = work / "imatrix.npz"
        cmd = [
            sys.executable, str(capture),
            "--source", str(source),
            "--calibration", str(calib),
            "--output", str(imatrix_path),
        ]
        print(f"Building local imatrix: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"imatrix capture failed:\n{result.stderr}")
        print(result.stdout)
        return imatrix_path

    if args.imatrix_repo or args.build_imatrix:
        print("Proceeding without an imatrix (unweighted error DB).")
    return None


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
    group_size_map: dict[str, int] | None = None,
    quantize_embeddings: bool = False,
) -> MlxQuantConfig:
    filtered = assignment
    if bf16_prefixes:
        filtered = _filter_bf16_prefixes(assignment, bf16_prefixes)

    quant_assignment = {
        k: v for k, v in filtered.items()
        if _is_quantizable(k, quantize_embeddings=quantize_embeddings)
    }
    cfg = allocation_to_config(
        quant_assignment,
        default_bits=default_bits,
        group_size=group_size,
        mode=quant_mode,
        group_size_map=group_size_map,
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


def _is_quantizable(path: str, quantize_embeddings: bool = False) -> bool:
    check = path if path.endswith(".weight") else path + ".weight"
    if "norm" in check or "mtp" in check:
        return False
    if not quantize_embeddings and ("embed" in check or "lm_head" in check):
        return False
    # Only the MoE router gate is excluded (mlx_lm has no QuantizedLinear
    # for it). Routed-expert and shared-expert gate_proj projections ARE
    # quantizable (qwen4_exp stacks them into switch_mlp / standard MLP).
    if check.endswith(".mlp.gate.weight") or check.endswith(".switch_mlp.gate.weight"):
        return False
    return check.endswith(".weight")


def _is_embedding_name(name: str) -> bool:
    """True for token-embedding / lm_head module paths."""
    return name.split(".")[-1] in ("embed_tokens", "lm_head")


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

# Candidate per-group clip ratios for the quantize-time scale search. The
# search evaluates each candidate with the real quantizer and keeps the
# per-group MSE-optimal one, so the grid only needs to bracket the useful
# range: measured optima on OLMoE sit between 0.6 and 1.0, and aggressive
# clips (<=0.6) are where weight-MSE proxies have previously diverged from
# real model quality.
PER_GROUP_CLIP_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)


def _quantize_dequant_np(
    w_np: np.ndarray, bits: int, gs: int, mode: str
) -> np.ndarray:
    """Quantize then dequantize with MLX, returning float32 numpy."""
    import mlx.core as mx

    q, scales, biases = mx.quantize(
        mx.array(w_np), bits=bits, group_size=gs, mode=mode
    )
    mx.eval(q, scales, biases)
    dq = mx.dequantize(q, scales, biases, group_size=gs, bits=bits, mode=mode)
    mx.eval(dq)
    return np.array(dq)


def _apply_clips(w_np: np.ndarray, clips: np.ndarray, gs: int) -> np.ndarray:
    """Clamp each ``gs``-wide group at ``clips * group_max_abs``.

    ``clips`` is ``[rows, n_groups]``; a value of 1.0 leaves the group
    untouched. This is the per-group generalisation of the per-tensor clip:
    ``mx.quantize`` derives its scale from the clamped range, so a per-group
    clamp is exactly a per-group scale choice.
    """
    out, inn = w_np.shape
    g = inn // gs
    wg = w_np.reshape(out, g, gs)
    max_abs = np.abs(wg).max(axis=2, keepdims=True)
    return np.clip(
        wg, -clips[:, :, None] * max_abs, clips[:, :, None] * max_abs
    ).reshape(out, inn)


def per_group_clip_search(
    w_np: np.ndarray,
    col_w: np.ndarray | None,
    bits: int,
    gs: int,
    mode: str,
    clip_grid=PER_GROUP_CLIP_GRID,
    max_batch_bytes: int = 256 * 1024 * 1024,
    hessian: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Choose one clip ratio per (row, group) minimising the output error.

    ``col_w`` is the imatrix per-input-column importance (mean-1). The
    objective is ``sum_j col_w[j] * (w[:,j] - dequant(quant(w[:,j])))^2``,
    i.e. the diagonal-Hessian output error an imatrix encodes. Passing
    ``col_w=None`` falls back to plain MSE.

    ``hessian=(lam, vec)`` is a low-rank factor of the per-expert Hessian
    (from scripts/capture_hessians.py). When given it REPLACES the diagonal
    objective with the full one, ``tr(dW H dW^T) ~ ||dW V sqrt(lam)||^2``,
    which is what the error DB also scores. Keeping the two objectives
    aligned matters: the DB decides where bits go, the clip search decides
    how each tensor is rounded, and a mismatch shows up as a worse model
    even when each half is optimal in isolation.

    Returns ``[rows, n_groups]`` clip ratios. Candidates are evaluated in
    memory-bounded batches by stacking them into the leading axis, which
    ``mx.quantize`` treats as independent slices (verified bit-exact against
    per-slice quantization).
    """
    out, inn = w_np.shape
    if inn % gs != 0:
        raise ValueError(
            f"per_group_clip_search: last dim {inn} not divisible by group size {gs}"
        )
    g = inn // gs
    wg = w_np.reshape(out, g, gs)
    max_abs = np.abs(wg).max(axis=2)  # [out, g]
    if col_w is not None:
        cw = np.asarray(col_w, dtype=np.float64).reshape(g, gs)
    else:
        cw = np.ones((g, gs), dtype=np.float64)

    best_err = np.full((out, g), np.inf)
    best_clip = np.ones((out, g))
    clips = np.asarray(clip_grid, dtype=np.float64)

    per_clip_bytes = max(out * inn * 4, 1)
    batch = max(1, min(len(clips), max_batch_bytes // per_clip_bytes))
    for i in range(0, len(clips), batch):
        chunk = clips[i : i + batch]
        stacked = np.stack(
            [np.clip(wg, -c * max_abs[:, :, None], c * max_abs[:, :, None]) for c in chunk]
        ).reshape(len(chunk), out, inn)
        dq = _quantize_dequant_np(stacked, bits, gs, mode).reshape(
            len(chunk), out, g, gs
        )
        dw = (wg[None] - dq).reshape(len(chunk), out, inn).astype(np.float32)
        if hessian is None:
            err = (cw[None, None] * dw.reshape(len(chunk), out, g, gs) ** 2).sum(axis=-1)
        else:
            lam, vec = hessian
            if vec.shape[0] != inn:
                raise ValueError(
                    f"per_group_clip_search: hessian width {vec.shape[0]} "
                    f"!= tensor width {inn}"
                )
            VS = vec.astype(np.float32) * np.sqrt(
                np.maximum(np.asarray(lam, dtype=np.float32), 0.0)
            )[None, :]
            err = np.empty((len(chunk), out, g), dtype=np.float32)
            for gi in range(g):
                sub = dw[:, :, gi * gs : (gi + 1) * gs] @ VS[gi * gs : (gi + 1) * gs, :]
                err[:, :, gi] = (sub ** 2).sum(axis=-1)
        chunk_min = err.min(axis=0)
        chunk_arg = err.argmin(axis=0)
        mask = chunk_min < best_err
        best_err = np.where(mask, chunk_min, best_err)
        best_clip = np.where(mask, chunk[chunk_arg], best_clip)
    return best_clip


def _imatrix_col_weights(imatrix, hf_key: str, n_cols: int) -> np.ndarray | None:
    """Per-input-column importance for ``hf_key``, or ``None`` if unmapped.

    Returns a mean-1 vector of length ``n_cols`` (the imatrix layout), or
    ``None`` when the tensor is absent from the imatrix, the mapping is
    unknown, or the width does not match (e.g. a fused QKV tensor).
    """
    try:
        mapped = imatrix.col_weights_for_hf_key(hf_key)
    except Exception:
        return None
    if mapped is None:
        return None
    col_w, _ = mapped
    col_w = np.asarray(col_w, dtype=np.float64).reshape(-1)
    if col_w.shape[0] != n_cols:
        return None
    return col_w


def stream_quantize(
    source_dir: Path,
    output_dir: Path,
    assignment: dict[str, int],
    group_size: int,
    model_adapter=None,
    quant_mode: str = "affine",
    group_size_map: dict[str, int] | None = None,
    clip_map: dict[str, float] | None = None,
    imatrix_path: Path | None = None,
    per_group_clip: bool = False,
    lsq_refit: bool = False,
    clip_grid=PER_GROUP_CLIP_GRID,
    clip_max_bits: int = 8,
    hessian_dir: Path | None = None,
    scale_dtype: str = "float16",
):
    """Stream through BF16 source, quantize per allocation, write shards.

    ``scale_dtype`` selects the storage dtype for affine scales/biases. The
    default (float16) keeps the most precision, but MLX's dequant kernel
    consumes bfloat16 scales natively -- measured on Granite-4.0-H-Tiny,
    float16 scales cost ~1.9x decode throughput for a negligible quality
    difference (KL 0.0682 -> 0.0689). Use ``"bfloat16"`` when decode speed
    matters more than the extra mantissa bits.
    """
    import torch

    def _to_scale(arr: np.ndarray, dtype: str):
        if dtype == "bfloat16":
            return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
        np_dt = {"float16": np.float16, "float32": np.float32}[dtype]
        return torch.from_numpy(arr.astype(np_dt))

    def _store_scales(results, base_key, scales_np, biases_np, quant_mode):
        if quant_mode == "symmetric":
            results[f"{base_key}.scales"] = _to_scale(scales_np, scale_dtype)
            return
        dtype = scale_dtype
        if dtype == "float16" and (
            np.max(np.abs(scales_np)) > 65504 or np.max(np.abs(biases_np)) > 65504
        ):
            print(f"  {base_key}: overflow -> float32 scales/biases")
            dtype = "float32"
        results[f"{base_key}.scales"] = _to_scale(scales_np, dtype)
        results[f"{base_key}.biases"] = _to_scale(biases_np, dtype)

    from rco_mlx.mlx_alloc import MlxQuantConfig

    imatrix = None
    hessian_factors = {}
    if hessian_dir is not None:
        from rco_mlx.hessians import load_hessian_factors

        hessian_factors = load_hessian_factors(hessian_dir)
        print(
            f"[quant] full-Hessian clip objective: {len(hessian_factors)} expert tensors "
            f"from {hessian_dir}"
        )
    if (per_group_clip or lsq_refit) and imatrix_path is not None:
        from rco_mlx.imatrix import Imatrix

        imatrix = Imatrix(imatrix_path)
        print(
            f"[quant] per-group clip search: grid {tuple(clip_grid)}, "
            f"imatrix {len(imatrix.tensor_names())} tensors"
        )
    elif (per_group_clip or lsq_refit):
        print(
            "[quant] WARNING: --per-group-clip without an imatrix; skipping the "
            "search. An unweighted search is measured to be WORSE than the "
            "per-tensor clip (OLMoE @2.5 KL 0.559 vs 0.508). Pass "
            "--imatrix/--imatrix-repo/--build-imatrix."
        )

    cfg = allocation_to_config(
        assignment, group_size=group_size, mode=quant_mode,
        group_size_map=group_size_map,
    )
    layer_bits = {}
    layer_gs: dict[str, int] = {}
    layer_clip: dict[str, float] = {}
    for path, bits in assignment.items():
        clean = path.removesuffix(".weight")
        layer_bits[clean] = bits
        if group_size_map:
            gs = group_size_map.get(path) or group_size_map.get(clean)
            if gs:
                layer_gs[clean] = int(gs)
        if clip_map:
            clip = clip_map.get(path) or clip_map.get(clean)
            if clip:
                layer_clip[clean] = float(clip)
        if model_adapter is not None:
            # Match _rewrite_config_mlx_keys: the adapter's rename only fires
            # when ".weight" is present (LFM2 w1 -> gate_proj), so a bare
            # module path would leave layer_bits keyed differently from the
            # emitted config and the quantizer would fall back to default bits.
            mlx = model_adapter.map_key_to_mlx(path + ".weight")
            if mlx:
                mlx = mlx.removesuffix(".weight")
                layer_bits[mlx] = bits
                layer_bits[mlx + ".weight"] = bits
                if clean in layer_gs:
                    layer_gs[mlx] = layer_gs[clean]
                if clean in layer_clip:
                    layer_clip[mlx] = layer_clip[clean]

    def _apply_clip(w_np: np.ndarray, gs: int, clip: float) -> np.ndarray:
        """Clamp per-group magnitudes at clip * group-max before quantizing.

        mx.quantize derives its scale/bias from the clamped tensor's
        range, so clamping first is exactly the saturation behaviour the
        error-DB clip search measured.
        """
        if clip >= 1.0 or w_np.shape[-1] % gs != 0:
            return w_np
        orig = w_np.shape
        grouped = w_np.reshape(*orig[:-1], -1, gs)
        max_abs = np.abs(grouped).max(axis=-1, keepdims=True)
        grouped = np.clip(grouped, -clip * max_abs, clip * max_abs)
        return grouped.reshape(orig)

    def _lookup_clip(base: str) -> float:
        if base in layer_clip:
            return layer_clip[base]
        if ".experts." in base:
            import re as _re

            m = _re.match(r"^((?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.experts)\.\d+\.([a-z_]+)$", base)
            if m:
                agg = f"{m.group(1)}.{m.group(2)}"
                if agg in (clip_map or {}):
                    return float(clip_map[agg])
        return 1.0

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

    def should_quantize(mlx_key: str) -> tuple[bool, int, int, float]:
        gs = group_size
        quant_embed = bool(getattr(model_adapter, "quantize_embeddings", False))
        # N-gram PLE table first: its name contains "embed", which the
        # generic exclusion below would incorrectly match.
        if model_adapter is not None and hasattr(model_adapter, "is_ngram_key") and model_adapter.is_ngram_key(mlx_key):
            return True, model_adapter.ngram_quant_bits, model_adapter.ngram_group_size, 1.0
        if (
            "norm" in mlx_key
            or (not quant_embed and ("embed" in mlx_key or "lm_head" in mlx_key))
            or "mtp" in mlx_key
            or (not mlx_key.endswith(".weight") and ".experts." not in mlx_key)
            or mlx_key.endswith(".bias")
            or ".conv.conv." in mlx_key
        ):
            return False, 4, gs, 1.0
        if model_adapter is not None:
            # N-gram PLE table: uniform quantization at adapter-specified bits/group.
            if hasattr(model_adapter, "is_ngram_key") and model_adapter.is_ngram_key(mlx_key):
                return True, model_adapter.ngram_quant_bits, model_adapter.ngram_group_size, 1.0
            for prefix in model_adapter.non_quantized_prefixes():
                if f".{prefix}." in f".{mlx_key}." or mlx_key.startswith(f"{prefix}.") or mlx_key == prefix:
                    return False, 4, gs, 1.0
        bits = layer_bits_with_suffix.get(mlx_key)
        if bits is None:
            base = mlx_key[:-len(".weight")] if mlx_key.endswith(".weight") else mlx_key
            bits = layer_bits.get(base, None)
            if bits is None and ".experts." in base:
                parts = base.split(".")
                try:
                    experts_idx = parts.index("experts")
                    proj = parts[experts_idx + 2]
                    # layer index follows the "layers" component in both
                    # model.layers.N.* and model.language_model.layers.N.*
                    l = parts[parts.index("layers") + 1]
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
        base = mlx_key[:-len(".weight")] if mlx_key.endswith(".weight") else mlx_key
        return True, bits, layer_gs.get(base, gs), _lookup_clip(base)

    shards = sorted(source_dir.glob("*.safetensors"))
    output_dir.mkdir(parents=True, exist_ok=True)
    total_tensors = 0
    n_pg_search = 0
    n_lsq = 0
    n_3d_skipped = 0
    clip_hist: dict[float, int] = {}
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

                q_ok, bits, q_gs, q_clip = should_quantize(mlx_key)
                if q_ok:
                    tensor = f.get_tensor(key)
                    # mx.quantize requires the last dim to be divisible by
                    # the group size (e.g. ple.conv1d is 10240x1x4). Keep
                    # such tensors BF16 rather than crashing the whole run.
                    # (Per-tensor gs32/gs128 assignments must divide the
                    # last dim -- true for all standard MoE widths; the
                    # post-quantize config rewrite drops BF16 fallbacks
                    # from the quantization map, so this stays loadable.)
                    if tensor.shape[-1] % q_gs != 0:
                        print(f"  {mlx_key}: last dim {tensor.shape[-1]} not divisible "
                              f"by group {q_gs} -> keeping BF16")
                        results[key] = tensor.to(torch.bfloat16)
                        continue
                    if tensor.ndim == 2:
                        w_np = tensor.float().numpy()
                        col_w = (
                            _imatrix_col_weights(imatrix, key, w_np.shape[-1])
                            if imatrix is not None and (per_group_clip or lsq_refit)
                            else None
                        )
                        base_key = mlx_key[:-len(".weight")] if mlx_key.endswith(".weight") else mlx_key
                        if lsq_refit and col_w is not None:
                            # oQe-style: imatrix-weighted (scale, bias) search
                            # plus a least-squares refit.  Supersedes the
                            # weight-clamp search below.
                            from rco_mlx.weighted_affine import weighted_affine_quantize

                            q_np, scales_np, biases_np = weighted_affine_quantize(
                                w_np, col_w, bits, q_gs
                            )
                            results[mlx_key] = torch.from_numpy(q_np.astype(np.uint32))
                            _store_scales(results, base_key, scales_np, biases_np, quant_mode)
                            n_lsq += 1
                        else:
                            if per_group_clip and bits <= clip_max_bits:
                                # Per-group scale choice, weighted by the imatrix.
                                # Only run when the tensor actually has imatrix
                                # column weights: an unweighted search is
                                # measured to be WORSE than the per-tensor clip
                                # (OLMoE @2.5 KL 0.559 vs 0.508), so never
                                # silently fall back to one.
                                if col_w is not None:
                                    from rco_mlx.hessians import hessian_key_for

                                    hk = hessian_key_for(key)
                                    clips = per_group_clip_search(
                                        w_np, col_w, bits, q_gs, quant_mode, clip_grid,
                                        hessian=hessian_factors.get(hk) if hk else None,
                                    )
                                    w_np = _apply_clips(w_np, clips, q_gs)
                                    n_pg_search += 1
                                    vals, cnts = np.unique(clips, return_counts=True)
                                    for v, c in zip(vals, cnts):
                                        clip_hist[float(v)] = clip_hist.get(float(v), 0) + int(c)
                                else:
                                    w_np = _apply_clip(w_np, q_gs, q_clip)
                            else:
                                w_np = _apply_clip(w_np, q_gs, q_clip)
                            w = mx.array(w_np)
                            q, scales, biases = mx.quantize(
                                w, bits=bits, group_size=q_gs, mode=quant_mode
                            )
                            mx.eval(q, scales, biases)
                            results[mlx_key] = torch.from_numpy(np.array(q).astype(np.uint32))
                            scales_np = np.array(scales)
                            biases_np = np.array(biases)
                            _store_scales(results, base_key, scales_np, biases_np, quant_mode)
                    elif tensor.ndim == 3:
                        # Stacked MoE expert tensor: quantize ALL experts in batch
                        E, H, I = tensor.shape
                        flat = tensor.reshape(E * H, I)
                        if per_group_clip and imatrix is not None:
                            # Per-group search is implemented for 2D tensors
                            # (per-expert HF keys). Stacked 3D tensors would
                            # need per-expert column weights; fall back to the
                            # per-tensor clip rather than silently skipping.
                            n_3d_skipped += 1
                        w_flat_np = _apply_clip(flat.float().numpy(), q_gs, q_clip)
                        w_flat = mx.array(w_flat_np)
                        q_flat, scales_flat, biases_flat = mx.quantize(
                            w_flat, bits=bits, group_size=q_gs, mode=quant_mode
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

                        _store_scales(results, base_key, scales_np, biases_np, quant_mode)
                    else:
                        results[mlx_key] = tensor.to(torch.bfloat16)
                else:
                    results[key] = f.get_tensor(key).to(torch.bfloat16)

        # N-gram PLE tensors go into their own file(s) so the lookup
        # table (tens of GB) can be moved / kept on host SSD separately
        # from the main checkpoint. mlx_lm loads every *.safetensors in
        # the directory, so this is transparent at load time.
        ngram_results = {
            k: results.pop(k) for k in list(results) if "ngram_embedding" in k
        }
        out_shard = output_dir / f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(results, str(out_shard))
        if ngram_results:
            ngram_shard = output_dir / f"ngram-{i+1:05d}-of-{len(shards):05d}.safetensors"
            save_file(ngram_results, str(ngram_shard))
            print(f"  n-gram -> {ngram_shard.name} ({len(ngram_results)} tensors)")
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

    # Write the shard index last: it must reflect the final set of shards.
    write_output_index(output_dir)

    print(f"\n[quant] done — {total_tensors} tensors, {time.time()-start:.1f}s total")
    if (per_group_clip or lsq_refit) and imatrix is not None:
        print(
            f"[quant] per-group clip search applied to {n_pg_search} tensors"
            + (f", lsq-refit on {n_lsq} tensors" if n_lsq else "")
            + (f", skipped {n_3d_skipped} stacked 3D tensors" if n_3d_skipped else "")
        )
    if clip_hist:
        total = sum(clip_hist.values())
        print(f"[quant] per-group clip histogram ({total} groups):")
        for v in sorted(clip_hist, reverse=True):
            print(f"  {v:.2f}: {clip_hist[v]} ({100*clip_hist[v]/total:.1f}%)")


def write_output_index(output_dir: Path) -> None:
    """Write ``model.safetensors.index.json`` for every shard in the output.

    oMLX discovers shards from filename numbering when no index is present
    and skips layouts whose prefixes do not line up; an explicit index makes
    discovery independent of file naming (``model-*``, ``ngram-*``).
    """
    shards = sorted(output_dir.glob("*.safetensors"))
    if not shards:
        return
    weight_map = {}
    total_size = 0
    for sp in shards:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = sp.name
        total_size += sp.stat().st_size
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    tmp = output_dir / "model.safetensors.index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(output_dir / "model.safetensors.index.json")
    print(f"[quant] wrote index ({len(weight_map)} tensors across {len(shards)} shards)")


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

_MOE_ADAPTERS = {"olmoe", "qwen3_5_moe", "qwen4_exp", "lfm2_moe", "granitemoehybrid"}


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
    add_bos: bool = False,
) -> Path:
    # Some custom architectures (e.g. LFM3's hybrid rope config) crash
    # transformers' auto-config import before a tokenizer is usable. Fall back
    # to tokenizer.json, which needs no config at all.
    bos = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(source_dir), trust_remote_code=True, local_files_only=True
        )
        bos = getattr(tokenizer, "bos_token_id", None)

        def encode(text):
            return list(tokenizer.encode(text))
    except Exception as exc:
        print(f"[calib] AutoTokenizer unavailable ({type(exc).__name__}); "
              f"using tokenizer.json directly")
        from tokenizers import Tokenizer

        tk = Tokenizer.from_file(str(source_dir / "tokenizer.json"))

        def encode(text):
            return tk.encode(text).ids

        try:
            with open(source_dir / "config.json") as f:
                bos = json.load(f).get("bos_token_id")
        except Exception:
            pass

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

    all_ids = encode(text)
    target = num_samples * seq_length
    if len(all_ids) < target:
        all_ids = (all_ids * ((target + len(all_ids) - 1) // len(all_ids)))[:target]
    else:
        all_ids = all_ids[:target]

    # Some tokenizers (LFM2.5) define a bos_token_id but do not add it in
    # encode(), and the model is lost without it (wikitext PPL 210 vs ~12).
    if add_bos and bos is not None:
        per = seq_length - 1
        rows = [all_ids[i * per:(i + 1) * per] for i in range(num_samples)]
        all_ids = [t for row in rows for t in [bos, *row]]
        print(f"[calib] prepending bos {bos} to each sample")

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
        description="RCO mixed-precision quantization pipeline (dense + MoE, "
                    "streaming, per-tensor RTN error DB)",
    )
    ap.add_argument("--source", required=True, help="BF16 source model dir")
    ap.add_argument("--output", required=True, help="Output quantized model dir")
    ap.add_argument("--target-bpw", type=float, required=True, help="Target bits per weight")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--default-bits", type=int, default=4)
    ap.add_argument("--model-type", default=None,
                    help="Model adapter: olmoe, qwen3_5_moe, qwen4_exp, "
                         "lfm2_moe, granitemoehybrid, or 'default' for a dense "
                         "transformer. Omit on a dense model to auto-select "
                         "'default'.")
    ap.add_argument("--calibration", default=None, help="Path to calibration .npy")
    ap.add_argument("--calibration-dataset", default=None, help="Auto-generate calib: wikitext|fineweb|c4")
    ap.add_argument("--calibration-bos", action="store_true",
                    help="Prepend the tokenizer's bos_token_id to each calibration "
                         "sample. Needed when the tokenizer defines a bos but does "
                         "not add it in encode() and the model is lost without it "
                         "(LFM2.5: wikitext PPL 210 without vs ~12 with).")
    ap.add_argument("--error-db", default=None, help="Pre-built error_db.json")
    ap.add_argument("--bits", default="2,3,4,6,8", help="Bitwidths to sweep")
    ap.add_argument("--quant-mode", default="affine", choices=["affine", "symmetric"],
                    help="Quantization mode (symmetric saves ~0.25 bpw on scales)")
    ap.add_argument("--no-gptq", action="store_true", help="Disable GPTQ scaling, use plain RTN")
    ap.add_argument("--ad-ladder", action="store_true",
                    help="Use AtomicChat AD ladder (rule-based) instead of RCO measurement")
    ap.add_argument("--expert-allocation", default="uniform",
                    choices=["uniform", "per-projection", "gguf-recipe"],
                    help="Expert bitwidth allocation: 'uniform' forces one bitwidth "
                         "for all routed experts (legacy, sanitize-safe); "
                         "'per-projection' keeps the DP assignment but enforces "
                         "uniformity only within each (layer, projection) stacked "
                         "tensor; 'gguf-recipe' overrides experts with the measured "
                         "UD-IQ2_XS per-layer recipe (qwen4_exp, 48 layers only)")
    ap.add_argument("--expert-min-bits", type=int, default=2,
                    help="Minimum bitwidth for routed experts, enforced inside the "
                         "DP (per-projection mode only): expert groups may only "
                         "choose options >= this width and the DP compensates by "
                         "lowering non-expert widths. 2 = no floor (default). "
                         "Note: a floor can make a low target infeasible.")
    ap.add_argument("--quantize-embeddings", action="store_true",
                    help="Dense models: quantize embed_tokens/lm_head through the "
                         "DP instead of keeping them BF16. Needed to compare "
                         "against checkpoints that quantize embeddings (e.g. "
                         "MACaQ) at matched file size.")
    ap.add_argument("--embedding-min-bits", type=int, default=2,
                    help="Minimum bitwidth for embedding groups when "
                         "--quantize-embeddings is set (default 2 = no floor; "
                         "4 is safer for tied output projections).")
    ap.add_argument("--imatrix", default=None,
                    help="Path to a GGUF imatrix file or local .npz; weights the "
                         "error DB per input column by activation importance")
    ap.add_argument("--imatrix-repo", default=None,
                    help="HF repo to check for a published imatrix first "
                         "(preferred over building one locally)")
    ap.add_argument("--imatrix-filename", default=None,
                    help="Exact imatrix filename in --imatrix-repo (default: any "
                         "*imatrix* file)")
    ap.add_argument("--build-imatrix", action="store_true",
                    help="Capture a local imatrix via forward pass when the BF16 "
                         "source fits in memory (70 percent RAM gate)")
    ap.add_argument("--hessian-dir", default=None,
                    help="Directory of per-expert Hessian factors from "
                         "scripts/capture_hessians.py. Used BOTH to score expert "
                         "tensors in the error DB (tr(dW H dW^T) instead of weight "
                         "MSE) and as the clip search's objective, so the two "
                         "optimize the same quantity")
    ap.add_argument("--clip-search", action="store_true",
                    help="Sweep per-tensor scale multipliers in the error DB "
                         "and apply the recorded clip at quantize time")
    ap.add_argument("--per-group-clip", action="store_true",
                    help="At quantize time, choose a clip ratio per (row, group) "
                         "minimising imatrix-weighted reconstruction MSE, evaluated "
                         "with the real quantizer. Requires an imatrix "
                         "(--imatrix/--imatrix-repo/--build-imatrix); without one "
                         "the search is skipped, because an unweighted search is "
                         "measured to be worse than the per-tensor clip")
    ap.add_argument("--lsq-refit", action="store_true",
                    help="oQe-style imatrix-weighted (scale, bias) search plus a "
                         "least-squares refit, replacing the weight-clamp clip "
                         "search. Requires an imatrix. This is what oMLX's "
                         "enhanced=True actually does -- it is NOT GPTQ (the oMLX "
                         "package contains no Hessian compensation)")
    ap.add_argument("--clip-grid", default=None,
                    help="Comma-separated clip ratios for --per-group-clip "
                         "(default: 1.0 down to 0.6 in 0.05 steps)")
    ap.add_argument("--clip-max-bits", type=int, default=8,
                    help="Only run the --per-group-clip search at or below this "
                         "bitwidth (default 8 = all quantized tensors). Measured on "
                         "OLMoE @2.5: gating to <=3 bits is WORSE (KL 0.466 vs "
                         "0.452) because the weighted search also helps at 4/8-bit, "
                         "so the default stays 8")
    ap.add_argument("--scale-dtype", choices=["float16", "bfloat16"], default="float16",
                    help="Storage dtype for affine scales/biases. bfloat16 is "
                         "consumed natively by MLX's dequant kernel (~1.9x faster "
                         "decode, measured on Granite-4.0-H-Tiny) at a small "
                         "precision cost; float16 keeps more mantissa bits.")
    ap.add_argument("--per-tensor-group-size", action="store_true",
                    help="Assign per-tensor group sizes from measured "
                         "sensitivity (top-quartile -> gs32, bottom-half -> "
                         "gs128, else gs64)")
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
        try:
            adapter_cls = get(args.model_type)
        except KeyError:
            adapter_cls = None
        if adapter_cls is not None:
            model_adapter = adapter_cls()
            print(f"Using model adapter: {args.model_type}")
        else:
            print(f"WARNING: no adapter registered for --model-type {args.model_type!r}")

    # Detect MoE (routed experts). Dense models fall back to the generic
    # default adapter, which quantizes every nn.Linear except norms,
    # embeddings and lm_head.
    is_moe = detect_moe_model(source, args.model_type)
    if model_adapter is None:
        if is_moe:
            from rco_mlx.adapters.base import registered_names
            sys.exit(
                "MoE model detected but no adapter is registered for it. "
                f"Registered adapters: {registered_names()} "
                "(pass --model-type)."
            )
        from rco_mlx.adapters import get as _get
        model_adapter = _get("default")()
        print("Dense model detected — using the generic default adapter.")

    if args.quantize_embeddings:
        if not hasattr(model_adapter, "quantize_embeddings"):
            sys.exit(
                "--quantize-embeddings requires an adapter that supports it "
                "(the dense default adapter does; MoE adapters do not)."
            )
        model_adapter.quantize_embeddings = True
        print("Embedding/lm_head quantization: enabled")

    quantize_embeddings = bool(getattr(model_adapter, "quantize_embeddings", False))

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
                    add_bos=args.calibration_bos,
                )
            else:
                sys.exit("Need --calibration, --calibration-dataset, or --error-db")

            error_db_path = output / "work" / "error_db.json"
            error_db_path.parent.mkdir(parents=True, exist_ok=True)
            imatrix_path = resolve_imatrix(args, source, output)
            build_error_db(
                source_dir=source,
                output_path=error_db_path,
                model_adapter=model_adapter,
                calibration_path=calib_path,
                bitwidths=[int(b) for b in args.bits.split(",")],
                group_size=args.group_size,
                quant_mode=args.quant_mode,
                use_gptq=not args.no_gptq,
                imatrix_path=imatrix_path,
                clip_search=args.clip_search,
                hessian_dir=args.hessian_dir,
            )

        # Load error_db for allocation (name-aligned to groups)
        losses, fracs, bitwidths = load_error_db(error_db_path, group_names=group_names)
        group_names = [g.layer_names[0] for g in groups]
        if losses.shape[0] != len(group_names):
            print(f"  warning: error_db has {losses.shape[0]} rows but {len(group_names)} groups")
            n = min(losses.shape[0], len(group_names))
            losses = losses[:n]
            fracs = fracs[:n]
            group_names = group_names[:n]

        # --- Step 3: allocation ---
        # Collapse groups that share an identical loss row (e.g. every
        # routed expert in one (layer, projection) shares its aggregate
        # row) before the DP: budget_dp rounds each group's cost to an
        # integer, so tens of thousands of tiny groups inflate the
        # minimum feasible cost past the budget. Collapsing is exact --
        # summed fracs, same options -- and assignment expands back.
        print(f"[3/5] RCO allocation  target={args.target_bpw} bpw ...")
        collapsed = {}
        for i, name in enumerate(group_names):
            collapsed.setdefault(losses[i].tobytes(), []).append(i)
        if len(collapsed) < len(group_names):
            print(f"  collapsing {len(group_names)} groups -> {len(collapsed)} unique loss rows")
            c_losses, c_fracs, c_names, c_members = [], [], [], []
            for members in collapsed.values():
                c_losses.append(losses[members[0]])
                c_fracs.append(fracs[members].sum())
                c_names.append(group_names[members[0]])
                c_members.append(members)
            c_fracs = np.array(c_fracs, dtype=np.float64)
            c_fracs = c_fracs / max(c_fracs.sum(), 1e-300)
            c_assignment, report = run_rco_allocation(
                np.array(c_losses), c_fracs, bitwidths, args.target_bpw,
                args.group_size, c_names,
                expert_min_bits=(
                    args.expert_min_bits
                    if args.expert_allocation == "per-projection" else 2
                ),
                group_is_expert=[
                    expert_group_key(n) is not None for n in c_names
                ],
                group_is_embedding=[_is_embedding_name(n) for n in c_names],
                embedding_min_bits=args.embedding_min_bits,
            )
            assignment = {
                group_names[i]: c_assignment[c_names[j]]
                for j, members in enumerate(c_members) for i in members
            }
        else:
            assignment, report = run_rco_allocation(
                losses, fracs, bitwidths, args.target_bpw, args.group_size, group_names,
                expert_min_bits=(
                    args.expert_min_bits
                    if args.expert_allocation == "per-projection" else 2
                ),
                group_is_expert=[
                    expert_group_key(n) is not None for n in group_names
                ],
                group_is_embedding=[_is_embedding_name(n) for n in group_names],
                embedding_min_bits=args.embedding_min_bits,
            )
        print(f"  achieved {report['achieved_bpw']:.4f} bpw")
        print(f"  histogram: {report['bit_histogram']}")

        # Expert allocation (MoE only). The sanitize constraint is uniformity
        # *within* each stacked (layer, projection) tensor, not across the
        # model. 'uniform' (legacy) forces one bitwidth for every routed
        # expert; 'per-projection' keeps the DP's per-group assignment and
        # only enforces that within-group invariant; 'gguf-recipe' overrides
        # the experts with the measured UD-IQ2_XS recipe (qwen4_exp only).
        # Dense models have no routed experts, so the DP assignment stands.
        all_names = [g.layer_names[0] for g in groups]
        if not is_moe:
            pass
        elif args.expert_allocation == "uniform":
            expert_bits = uniform_expert_bits(args.target_bpw)
            print(f"  expert bitwidth: {expert_bits} (target {args.target_bpw} bpw)")
            # Add all expert groups to assignment with uniform bitwidth.
            # Substring test kept from the legacy path so any expert naming
            # the regexes don't parse still gets forced in uniform mode.
            def _is_expert(name: str) -> bool:
                low = name.lower()
                return (
                    expert_group_key(name) is not None
                    or "experts" in low
                    or "switch_mlp" in low
                    or "switch_glu" in low
                )

            for g in groups:
                name = g.layer_names[0]
                if _is_expert(name) and name not in assignment:
                    assignment[name] = expert_bits
            for k in list(assignment.keys()):
                if _is_expert(k):
                    assignment[k] = expert_bits
        elif args.expert_allocation == "per-projection":
            fallback = uniform_expert_bits(args.target_bpw)
            assignment, n_filled = fill_missing_expert_groups(
                assignment, all_names, fallback_bits=fallback
            )
            assignment, n_overridden = enforce_group_uniformity(assignment, all_names)
            proj_hist: dict[tuple[int, str], int] = {}
            for k, v in assignment.items():
                gk = expert_group_key(k)
                if gk is not None:
                    proj_hist.setdefault(gk, v)
            print(f"  per-projection experts: {len(proj_hist)} (layer, projection) "
                  f"groups, filled {n_filled}, uniformity overrides {n_overridden}")
            print(f"  expert group bit histogram: "
                  f"{dict(Counter(proj_hist.values()))}")
        else:  # gguf-recipe
            try:
                assignment, n_overridden = apply_gguf_recipe(
                    assignment, all_names, model_type=args.model_type
                )
            except ValueError as exc:
                sys.exit(str(exc))
            assignment, n_filled = fill_missing_expert_groups(
                assignment, all_names,
                fallback_bits=uniform_expert_bits(args.target_bpw),
            )
            assignment, n_conflicts = enforce_group_uniformity(assignment, all_names)
            proj_hist = {}
            for k, v in assignment.items():
                gk = expert_group_key(k)
                if gk is not None:
                    proj_hist.setdefault(gk, v)
            print(f"  gguf-recipe experts: {len(proj_hist)} groups, "
                  f"overridden {n_overridden}, filled {n_filled}, "
                  f"conflicts {n_conflicts}")
            print(f"  expert group bit histogram: "
                  f"{dict(Counter(proj_hist.values()))}")

    # --- Step 4: emit + validate ---
    print("[4/5] emitting MLX config + validating …")
    bf16_prefixes = model_adapter.non_quantized_prefixes() if model_adapter else None

    # Per-tensor group sizes from measured sensitivity at the assigned
    # bitwidth: top-quartile loss -> gs32, bottom-half -> gs128, else gs64.
    group_size_map: dict[str, int] = {}
    if args.per_tensor_group_size and not args.ad_ladder:
        try:
            sens = []
            for i, name in enumerate(group_names):
                bits = assignment.get(name)
                if bits is None:
                    continue
                opt = list(bitwidths).index(int(bits)) if int(bits) in list(bitwidths) else None
                if opt is not None:
                    sens.append((name, float(losses[i, opt])))
            if sens:
                vals = np.array([v for _, v in sens])
                hi, lo = np.quantile(vals, 0.75), np.quantile(vals, 0.5)
                for name, v in sens:
                    group_size_map[name] = 32 if v >= hi else (128 if v <= lo else 64)
                print(f"  per-tensor group sizes: {dict(Counter(group_size_map.values()))}")
        except Exception as exc:
            print(f"  per-tensor group size skipped ({exc}); using gs{args.group_size}")

    clip_map: dict[str, float] = {}
    if not args.ad_ladder and error_db_path.exists():
        clip_map = load_clip_map(error_db_path, assignment)
        if clip_map:
            print(f"  clip map: {len(clip_map)} tensors "
                  f"(range {min(clip_map.values()):.2f}-{max(clip_map.values()):.2f})")

    cfg = emit_and_validate(
        assignment, args.group_size, source, output, args.default_bits,
        quant_mode=args.quant_mode, bf16_prefixes=bf16_prefixes,
        group_size_map=group_size_map or None,
        quantize_embeddings=quantize_embeddings,
    )

    # --- Step 5: quantize ---
    if args.skip_quantize:
        print("[5/5] --skip-quantize set, done.")
        return

    print("[5/5] streaming quantization …")
    imatrix_path = None
    if (args.per_group_clip or args.lsq_refit) and not args.ad_ladder:
        imatrix_path = resolve_imatrix(args, source, output)
    clip_grid = PER_GROUP_CLIP_GRID
    if args.clip_grid:
        clip_grid = tuple(float(x) for x in args.clip_grid.split(","))
    stream_quantize(
        source, output, assignment, args.group_size,
        model_adapter=model_adapter, quant_mode=args.quant_mode,
        group_size_map=group_size_map or None, clip_map=clip_map or None,
        imatrix_path=imatrix_path, per_group_clip=args.per_group_clip,
        lsq_refit=args.lsq_refit,
        clip_grid=clip_grid, clip_max_bits=args.clip_max_bits,
        hessian_dir=args.hessian_dir, scale_dtype=args.scale_dtype,
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
                        if suffix in [".scales", ".biases"]:
                            # 2D: (E*2H, packed) -> split along dim 0, reshape to 3D
                            mid = gate_up.shape[0] // 2
                            E = num_experts
                            H = mid // E
                            packed = gate_up.shape[-1]
                            gate_up_3d = gate_up.reshape(E, 2 * H, packed)
                            gate = gate_up_3d[:, :H, :].reshape(E * H, packed).contiguous()
                            up = gate_up_3d[:, H:, :].reshape(E * H, packed).contiguous()
                            # Reshape to 3D: (E, H, packed)
                            gate = gate.reshape(E, H, packed).contiguous()
                            up = up.reshape(E, H, packed).contiguous()
                            new_tensors[f"{prefix}.switch_mlp.gate_proj{suffix}"] = gate
                            new_tensors[f"{prefix}.switch_mlp.up_proj{suffix}"] = up
                        elif gate_up.dim() >= 3:
                            # 3D: (E, 2H, packed) -> split along dim 1
                            mid = gate_up.shape[1] // 2
                            new_tensors[f"{prefix}.switch_mlp.gate_proj{suffix}"] = gate_up[:, :mid, :].contiguous()
                            new_tensors[f"{prefix}.switch_mlp.up_proj{suffix}"] = gate_up[:, mid:, :].contiguous()
                        else:
                            # 2D weight: (E*2H, packed) -> split along dim 0
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
                        if suffix in [".scales", ".biases"] and down.dim() == 2:
                            # Reshape 2D (E*H, packed) -> 3D (E, H, packed)
                            E = num_experts
                            H = down.shape[0] // E
                            packed = down.shape[-1]
                            new_tensors[f"{prefix}.switch_mlp.down_proj{suffix}"] = down.reshape(E, H, packed).contiguous()
                        else:
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
    # Post-process: LFM2/LFM2.5 expert stacking.
    # mlx_lm's lfm2_moe.sanitize stacks only `.weight` for
    # feed_forward.experts.E.*, leaving the quantized `.scales`/`.biases`
    # behind as "parameters not in model" (OLMoE's sanitize stacks all
    # three). Stack every suffix ourselves so the checkpoint loads with a
    # stock mlx_lm.
    if args.model_type == "lfm2_moe":
        print("[post] stacking LFM2 per-expert tensors into switch_mlp ...")
        import os
        import re

        import safetensors.torch as st
        import torch

        cfg_data = json.loads((source / "config.json").read_text())
        num_layers = int(cfg_data["num_hidden_layers"])
        num_experts = int(cfg_data["num_experts"])
        n_stacked = 0
        for shard_path in sorted(output.glob("*.safetensors")):
            tensors = st.load_file(str(shard_path))
            new_tensors = dict(tensors)
            for l in range(num_layers):
                prefix = f"model.layers.{l}.feed_forward"
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    for suffix in ("weight", "scales", "biases"):
                        keys = [f"{prefix}.experts.{e}.{proj}.{suffix}"
                                for e in range(num_experts)]
                        if not all(k in new_tensors for k in keys):
                            continue
                        vals = [new_tensors.pop(k) for k in keys]
                        new_tensors[f"{prefix}.switch_mlp.{proj}.{suffix}"] = \
                            torch.stack(vals, dim=0).contiguous()
                        n_stacked += 1
            tmp_path = "/tmp/" + shard_path.name + ".tmp"
            st.save_file(new_tensors, tmp_path)
            os.replace(tmp_path, str(shard_path))
            print(f"  stacked {shard_path.name} ({n_stacked} tensors)")

        # Rewrite config keys to the stacked form. Uniformity within each
        # (layer, projection) means every expert carries the same spec, so
        # collapsing the per-expert entries is lossless.
        cfg_path = output / "config.json"
        cfg_data = json.loads(cfg_path.read_text())
        qc = cfg_data.get("quantization", {})
        final_qc = {}
        for k, v in qc.items():
            nk = re.sub(r"\.experts\.\d+\.", ".switch_mlp.", k)
            if nk in final_qc and final_qc[nk] != v:
                print(f"  warning: conflicting quantization spec for {nk}")
            final_qc[nk] = v
        cfg_data["quantization"] = final_qc
        cfg_path.write_text(json.dumps(cfg_data, indent=2))
        print(f"  config rewritten ({len(qc)} -> {len(final_qc)} entries)")

    # Post-process: Granite-4.0-H hybrid MoE.
    # Two things mlx_lm's granitemoehybrid.sanitize cannot do for a quantized
    # checkpoint:
    #   1. The pipeline's 3-D path quantizes the flat [E*out, in] tensor, so
    #      scales/biases come out 2-D [E*out, in/gs]; mx.quantize on the 3-D
    #      weight produces [E, out, in/gs], which is what gather_qmm expects.
    #   2. mlx_lm's sanitize splits block_sparse_moe.input_linear into
    #      switch_mlp.gate_proj + up_proj (dim 1) and renames output_linear to
    #      down_proj. The adapter maps input_linear to the gate_proj carrier,
    #      so split it here and declare the up_proj module.
    if args.model_type == "granitemoehybrid":
        print("[post] reshaping Granite switch_mlp scales + splitting gate/up ...")
        import os

        import safetensors.torch as st
        import torch

        n_reshaped = 0
        n_split = 0
        for shard_path in sorted(output.glob("*.safetensors")):
            tensors = st.load_file(str(shard_path))
            new_tensors = dict(tensors)
            for key in list(new_tensors.keys()):
                if ".switch_mlp." not in key or not key.endswith(".scales"):
                    continue
                base = key[: -len(".scales")]
                wkey = base + ".weight"
                if wkey not in new_tensors:
                    continue
                w = new_tensors[wkey]
                if w.dim() != 3:
                    continue
                E, out, _ = w.shape
                s = new_tensors[key]
                if s.dim() == 2 and s.shape[0] == E * out:
                    new_tensors[key] = s.reshape(E, out, s.shape[-1]).contiguous()
                    bkey = base + ".biases"
                    if bkey in new_tensors and new_tensors[bkey].dim() == 2:
                        b = new_tensors[bkey]
                        if b.shape[0] == E * out:
                            new_tensors[bkey] = b.reshape(E, out, b.shape[-1]).contiguous()
                    n_reshaped += 1
                else:
                    continue
                if not base.endswith(".gate_proj"):
                    continue
                up_base = base[: -len(".gate_proj")]
                h = new_tensors[wkey].shape[1] // 2
                new_tensors[wkey] = new_tensors[wkey][:, :h, :].contiguous()
                new_tensors[up_base + ".up_proj.weight"] = w[:, h:, :].contiguous()
                for suf in ("scales", "biases"):
                    sk = base + "." + suf
                    if sk in new_tensors:
                        sv = new_tensors[sk]
                        new_tensors[sk] = sv[:, :h, :].contiguous()
                        new_tensors[up_base + ".up_proj." + suf] = sv[:, h:, :].contiguous()
                n_split += 1
            tmp_path = "/tmp/" + shard_path.name + ".tmp"
            st.save_file(new_tensors, tmp_path)
            os.replace(tmp_path, str(shard_path))
        print(f"  reshaped {n_reshaped} scale tensors, split {n_split} gate/up carriers")

        cfg_path = output / "config.json"
        cfg_data = json.loads(cfg_path.read_text())
        qc = cfg_data.get("quantization", {})
        final_qc = dict(qc)
        for key, val in qc.items():
            if key.endswith(".switch_mlp.gate_proj"):
                final_qc[key[: -len("gate_proj")] + "up_proj"] = dict(val)
        cfg_data["quantization"] = final_qc
        cfg_data["quantization_config"] = final_qc
        cfg_path.write_text(json.dumps(cfg_data, indent=2))
        print(f"  config updated ({len(qc)} -> {len(final_qc)} entries)")

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
