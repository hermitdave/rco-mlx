#!/usr/bin/env python3
"""Streaming imatrix capture for a checkpoint too large to hold in RAM.

``capture_imatrix.py`` needs the whole BF16 model resident (MLX allocates
parameters at construction, and its memory gate refuses anything over ~70% of
RAM). Nemotron 3.5 Bolt is 149.8 GB, so it cannot run there.

This variant keeps exactly ONE layer resident. Hidden states for every
calibration chunk are carried in a list (8 x 512 x 4096 x 4 B ~= 67 MB), and
each layer is streamed over all chunks in turn:

    H = [embed(chunk) for chunk in chunks]
    for layer in layers:  H = [layer(h) for h in H]   # layer resident, then freed

Peak memory is one layer (Bolt's largest is 5.31 GB) plus embeddings/lm_head
(~2.5 GB). Output is the same ``.npz`` the non-streaming capturer writes, so
``rco_mlx.imatrix.Imatrix`` reads either interchangeably.

Usage:
    python scripts/streaming_capture_imatrix.py \
        --source /path/to/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 \
        --calibration /path/to/calib.npy \
        --output /path/to/imatrix.npz --max-chunks 8
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

LAYER_RE = re.compile(r"^mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$")
SWITCH_RE = re.compile(r"^backbone\.(layers\.\d+\.mixer)\.switch_mlp\.(fc1|fc2)$")


def build_mlx_config(src: Path) -> dict:
    """Standalone MLX config from the HF checkpoint's ``llm_config``."""
    hf = json.loads((src / "config.json").read_text())
    cfg = dict(hf["llm_config"])
    cfg["model_type"] = "nemotron_h_puzzle"
    for k in (
        "auto_map", "architectures", "_megatron_bridge_hf_prefix",
        "mtp_block_configs", "mtp_layers_block_type", "blockwise_members",
    ):
        cfg.pop(k, None)
    # HF serialises this as [0.0, {"__float__": "Infinity"}], which the
    # dataclass cannot parse; None makes ModelArgs derive (min, inf).
    cfg["time_step_limit"] = None
    return cfg


def map_layer_tensors(raw: dict, layer_idx: int, mx) -> dict:
    """HF layer tensors -> MLX keys relative to ``NemotronHBlock``.

    Experts are stacked (MLX ``SwitchLinear`` wants one tensor per projection)
    and ``conv1d`` moves from ``[conv_dim, 1, kernel]`` to ``[conv_dim, kernel, 1]``.
    """
    pre = f"language_model.backbone.layers.{layer_idx}."
    experts: dict[str, dict[int, object]] = {}
    out = {}
    for k, v in raw.items():
        rel = k[len(pre):]
        m = LAYER_RE.match(rel)
        if m:
            e, proj = int(m.group(1)), m.group(2)
            experts.setdefault("fc1" if proj == "up_proj" else "fc2", {})[e] = v
            continue
        if rel.endswith("conv1d.weight"):
            v = v.moveaxis(2, 1)
        out[rel] = v
    for name, per_expert in experts.items():
        out[f"mixer.switch_mlp.{name}.weight"] = mx.stack(
            [per_expert[e] for e in sorted(per_expert)]
        )
    return out


def mlx_to_hf(path: str, expert=None) -> str:
    """MLX module path -> HF-style key the imatrix reader expects."""
    m = SWITCH_RE.match(path)
    if m:
        proj = "up_proj" if m.group(2) == "fc1" else "down_proj"
        suffix = f"experts.{expert}.{proj}" if expert is not None else f"experts.{proj}"
        return f"language_model.backbone.{m.group(1)}.{suffix}.weight"
    if path.startswith("backbone."):
        return "language_model." + path + ".weight"
    return path + ".weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--allow-foreign-calibration", action="store_true",
                    help="Skip the check that --calibration was tokenized with "
                         "this model's own tokenizer.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--max-layer", type=int, default=None,
                    help="only run the first N layers (smoke test)")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    from safetensors import safe_open
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.nemotron_h import ModelArgs, NemotronHBlock, NemotronHRMSNorm

    src = Path(args.source)
    margs = ModelArgs.from_dict(build_mlx_config(src))
    pattern = margs.hybrid_override_pattern or []
    n_layers = len(pattern)
    if args.max_layer is not None:
        n_layers = min(n_layers, args.max_layer)
    print(f"[cfg] layers={n_layers}/{len(pattern)} hidden={margs.hidden_size} "
          f"vocab={margs.vocab_size}", flush=True)

    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    def load_raw(keys: dict) -> dict:
        by_shard: dict[str, list] = {}
        for k, s in keys.items():
            by_shard.setdefault(s, []).append(k)
        out = {}
        for shard, ks in by_shard.items():
            with safe_open(str(src / shard), framework="pt", device="cpu") as f:
                for k in ks:
                    out[k] = mx.array(f.get_tensor(k))
        return out

    # ---- static parts: embeddings, final norm, lm_head -------------------
    static = load_raw({
        k: s for k, s in wm.items()
        if k in ("language_model.backbone.embeddings.weight",
                 "language_model.backbone.norm_f.weight",
                 "language_model.lm_head.weight")
    })
    embeddings = nn.Embedding(margs.vocab_size, margs.hidden_size)
    embeddings.load_weights(
        [("weight", static["language_model.backbone.embeddings.weight"])], strict=False)
    norm_f = NemotronHRMSNorm(margs.hidden_size, eps=margs.layer_norm_epsilon)
    norm_f.load_weights(
        [("weight", static["language_model.backbone.norm_f.weight"])], strict=False)
    lm_head = nn.Linear(margs.hidden_size, margs.vocab_size, bias=False)
    lm_head.load_weights(
        [("weight", static["language_model.lm_head.weight"])], strict=False)
    del static
    mx.eval(embeddings.parameters(), norm_f.parameters(), lm_head.parameters())
    print("[init] embeddings / norm_f / lm_head loaded", flush=True)

    # ---- accumulation ----------------------------------------------------
    in_sum2: dict[str, np.ndarray] = {}
    counts: dict[str, float] = {}

    def to_np(a):
        try:
            return np.asarray(a)
        except ValueError:
            return np.asarray(a.astype(mx.float32))

    def accumulate_sums(key, s, n):
        if key in in_sum2 and in_sum2[key].shape == s.shape:
            in_sum2[key] += s
        elif key not in in_sum2:
            in_sum2[key] = s.copy()
        counts[key] = counts.get(key, 0.0) + n

    def accumulate(key, rows):
        if rows.size == 0:
            return
        flat = rows.reshape(-1, rows.shape[-1]).astype(np.float64)
        accumulate_sums(key, (flat * flat).sum(axis=0), flat.shape[0])

    hook_by_id: dict[int, tuple] = {}
    patched_types: dict[type, object] = {}

    def linear_call(self, x, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            accumulate(mlx_to_hf(info[1]), to_np(x))
        return patched_types[type(self)](self, x, *a, **kw)

    def switch_call(self, x, indices=None, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            x_np = to_np(x)
            idx_np = to_np(indices) if indices is not None else None
            if idx_np is not None:
                # [tokens, top_k] slots against [tokens, D] inputs: repeat the
                # inputs per slot, then reduce per expert with a sort +
                # reduceat (a per-expert boolean mask is ~496x slower).
                x2 = x_np.reshape(-1, x_np.shape[-1])
                k = idx_np.shape[1] if idx_np.ndim == 2 else 1
                idx_flat = idx_np.reshape(-1).astype(np.int64)
                x_per_slot = np.repeat(x2, k, axis=0).astype(np.float64)
                sq = x_per_slot * x_per_slot
                n_exp = info[2]
                order = np.argsort(idx_flat, kind="stable")
                cnt = np.bincount(idx_flat, minlength=n_exp)
                nz = np.nonzero(cnt)[0]
                sums_by_e = {}
                if nz.size:
                    starts = np.concatenate(([0], np.cumsum(cnt)))
                    sums = np.add.reduceat(sq[order], starts[nz], axis=0)
                    for j, e in enumerate(nz):
                        sums_by_e[int(e)] = sums[j]
                # Experts the calibration set never activated have no observed
                # input distribution.  Every routed expert in a layer sees the
                # SAME hidden state, so the layer-pooled statistic is the
                # unbiased estimate for them -- write that instead of dropping
                # the tensor (a missing entry silently downgrades the expert to
                # an unweighted clip search, which is measured worse).
                pooled = sq.sum(axis=0)
                n_slots = int(sq.shape[0])
                n_zero = 0
                for e in range(n_exp):
                    s = sums_by_e.get(e)
                    if s is None:
                        accumulate_sums(mlx_to_hf(info[1], e), pooled, n_slots)
                        n_zero += 1
                    else:
                        accumulate_sums(mlx_to_hf(info[1], e), s, int(cnt[e]))
                if n_zero:
                    print(f"    [imatrix] {info[1]}: {n_zero}/{n_exp} experts "
                          f"never activated -> layer-pooled fallback", flush=True)
            else:
                accumulate(mlx_to_hf(info[1]), x_np)
        orig = patched_types[type(self)]
        return orig(self, x, indices, *a, **kw) if indices is not None else orig(self, x, *a, **kw)

    def register(module, prefix):
        for path, m in module.named_modules():
            tname = type(m).__name__
            kind = {"Linear": "linear", "SwitchLinear": "switch"}.get(tname)
            if kind is None:
                continue
            full = f"{prefix}.{path}" if path else prefix
            hook_by_id[id(m)] = (kind, full) + (
                (int(m.weight.shape[0]),) if kind == "switch" else ())
            cls = type(m)
            if cls not in patched_types:
                patched_types[cls] = cls.__call__
                cls.__call__ = linear_call if kind == "linear" else switch_call

    register(lm_head, "lm_head")
    print(f"[hook] static registered ({len(hook_by_id)})", flush=True)

    calib = np.load(args.calibration)
    n_rows = min(args.max_chunks, calib.shape[0])
    ids = mx.array(calib[:n_rows].astype(np.int32))
    print(f"[calib] {n_rows} rows x {calib.shape[1]}", flush=True)

    # A cache tokenized with another model's vocabulary would silently measure
    # off-distribution activations.  Refuse it unless explicitly overridden.
    if not args.allow_foreign_calibration:
        from rco_mlx.calibration import verify_calibration

        rep = verify_calibration(Path(args.source).resolve(), Path(args.calibration))
        print(rep.summary(), flush=True)
        if not rep.ok:
            sys.exit(
                "calibration cache does not match this model's tokenizer. "
                "Rebuild it (scripts/build_calibration_for_model.py) or pass "
                "--allow-foreign-calibration to override."
            )

    # hidden states for every chunk, carried through the layer stack
    H = [embeddings(ids[i:i + 1]) for i in range(n_rows)]
    mx.eval(*H)

    for i in range(n_layers):
        bt = pattern[i]
        raw = load_raw({
            k: s for k, s in wm.items()
            if k.startswith(f"language_model.backbone.layers.{i}.")
        })
        mapped = map_layer_tensors(raw, i, mx)
        del raw
        block = NemotronHBlock(margs.for_layer(i), bt)
        block.load_weights(list(mapped.items()), strict=False)
        del mapped
        mx.eval(block.parameters())
        register(block, f"backbone.layers.{i}")

        new_H = []
        for h in H:
            if bt == "*":
                mask = create_attention_mask(h, None)
                c = KVCache()
            elif bt == "M":
                mask = create_ssm_mask(h, None)
                c = ArraysCache(size=2)
            else:
                mask, c = None, None
            new_H.append(block(h, mask=mask, cache=c))
        mx.eval(*new_H)
        H = new_H
        print(f"[layer {i:3d}] {bt} done | hooked={len(hook_by_id)}", flush=True)
        del block
        mx.clear_cache()

    # final norm + lm_head
    H = [norm_f(h) for h in H]
    mx.eval(*H)
    for h in H:
        lm_head(h)
    mx.eval()

    print(f"[done] {len(in_sum2)} imatrix tensors", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in in_sum2.items():
        payload[f"{k}.in_sum2"] = v.astype(np.float32)
        payload[f"{k}.counts"] = np.array([counts[k]], dtype=np.float32)
    np.savez_compressed(str(out), **payload)
    print(f"saved -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
