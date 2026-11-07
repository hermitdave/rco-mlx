#!/usr/bin/env python3
"""Capture a local imatrix (activation importance) via an MLX forward pass.

Hooks every Linear / SwitchLinear in an mlx_lm model, runs calibration
token chunks through it, and accumulates per input column:

    in_sum2[j] = sum_tokens x_j^2
    counts     = tokens routed to the expert (per-expert for MoE)

For stacked ``SwitchLinear`` experts the hook receives flat tokens plus
top-k expert indices, so importance is accumulated per expert.

Output is an .npz keyed by HF-style tensor name:
``{tensor}.in_sum2`` / ``{tensor}.counts`` -- e.g.
``model.layers.0.mlp.experts.17.gate_proj.weight.in_sum2``.
``rco_mlx.imatrix.Imatrix`` reads this interchangeably with a GGUF
imatrix (downloaded from HF when one is published).

Gate: refuses to run unless the BF16 source fits in memory
(``rco_mlx.imatrix.fits_in_memory``, 70% of RAM by default), because
MLX unified memory means weights + activations share one pool.

Usage:
    python scripts/capture_imatrix.py \
        --source /path/to/bf16-model \
        --calibration /path/to/calib.npy \
        --output /path/to/work/imatrix.npz \
        --max-chunks 8
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _mlx_to_hf_expert_key(mlx_path: str, expert: int) -> str:
    """model.layers.N.mlp.switch_mlp.gate_proj -> ...experts.E.gate_proj.weight"""
    m = re.match(
        r"^(.*layers\.\d+\.mlp)\.switch_mlp\.(gate_proj|up_proj|down_proj)$",
        mlx_path,
    )
    if m:
        return f"{m.group(1)}.experts.{expert}.{m.group(2)}.weight"
    return f"{mlx_path}.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--calibration", required=True, help="calib .npy (samples, seq)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chunks", type=int, default=8,
                    help="number of calibration rows to run (forward-pass budget)")
    ap.add_argument("--force", action="store_true",
                    help="run even if the model does not fit the memory gate")
    args = ap.parse_args()

    from rco_mlx.imatrix import fits_in_memory

    fits, model_bytes, ram = fits_in_memory(args.source)
    print(f"Model bytes: {model_bytes/1e9:.1f} GB, RAM: {ram/1e9:.1f} GB, "
          f"fits (70% gate): {fits}")
    if not fits and not args.force:
        sys.exit(
            "Refusing to capture: BF16 source does not fit in memory. "
            "Use a published imatrix instead (--imatrix-repo in the pipeline), "
            "or pass --force on a machine with enough RAM."
        )

    import mlx.core as mx
    import mlx_lm

    calib = np.load(args.calibration)
    n_rows = min(args.max_chunks, calib.shape[0])
    chunks = [calib[i : i + 1].astype(np.int32) for i in range(n_rows)]

    print("Loading model (capture hooks installed before first forward)...", flush=True)
    model, tok = mlx_lm.load(str(args.source))

    in_sum2: dict[str, np.ndarray] = {}
    counts: dict[str, float] = {}

    def accumulate(key: str, rows: np.ndarray):
        if rows.size == 0:
            return
        flat = rows.reshape(-1, rows.shape[-1]).astype(np.float64)
        s = (flat * flat).sum(axis=0)
        if key in in_sum2 and in_sum2[key].shape == s.shape:
            in_sum2[key] += s
        elif key not in in_sum2:
            in_sum2[key] = s.copy()
        counts[key] = counts.get(key, 0.0) + flat.shape[0]

    # NOTE: Python resolves __call__ on the type, not the instance, so
    # hooks must be patched at class level and dispatch by instance id.
    hook_by_id: dict[int, object] = {}
    patched_types: dict[type, object] = {}

    def install_class_hook(cls, hooked_method):
        if cls not in patched_types:
            patched_types[cls] = cls.__call__
            cls.__call__ = hooked_method

    for path, module in model.named_modules():
        tname = type(module).__name__
        if tname == "Linear":
            hook_by_id[id(module)] = ("linear", f"{path}.weight")
        elif tname == "SwitchLinear":
            hook_by_id[id(module)] = (
                "switch", path, int(module.weight.shape[0]))

    def linear_call(self, x, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            try:
                accumulate(info[1], np.asarray(x))
            except Exception:
                pass
        return patched_types[type(self)](self, x, *a, **kw)

    def switch_call(self, x, indices=None, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            try:
                x_np = np.asarray(x)
                idx_np = np.asarray(indices) if indices is not None else None
                if idx_np is not None:
                    # SwitchGLU expands x to [tokens, 1, 1, D] and passes
                    # indices [tokens, top_k]; each token's row feeds all
                    # of its top-k slots.
                    x2 = x_np.reshape(-1, x_np.shape[-1])
                    k = idx_np.shape[1] if idx_np.ndim == 2 else 1
                    idx_flat = idx_np.reshape(-1)
                    x_per_slot = np.repeat(x2, k, axis=0)
                    for e in range(info[2]):
                        mask = idx_flat == e
                        if mask.any():
                            accumulate(
                                _mlx_to_hf_expert_key(info[1], e),
                                x_per_slot[mask],
                            )
                else:
                    accumulate(f"{info[1]}.weight", x_np)
            except Exception:
                pass
        orig = patched_types[type(self)]
        return orig(self, x, indices, *a, **kw) if indices is not None else orig(self, x, *a, **kw)

    seen_types = set()
    for path, module in model.named_modules():
        cls = type(module)
        info = hook_by_id.get(id(module))
        if info is not None and cls not in seen_types:
            seen_types.add(cls)
            install_class_hook(cls, linear_call if info[0] == "linear" else switch_call)
    patched = len(hook_by_id)
    print(f"Hooked {patched} linear modules", flush=True)
    if not patched:
        sys.exit("No linear modules could be hooked; capture unsupported for this architecture")

    for i, ids in enumerate(chunks):
        print(f"forward chunk {i+1}/{len(chunks)} ...", flush=True)
        logits = model(mx.array(ids))
        mx.eval(logits)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in in_sum2.items():
        payload[f"{key}.in_sum2"] = value.astype(np.float32)
        payload[f"{key}.counts"] = np.array([counts[key]], dtype=np.float32)
    np.savez_compressed(str(out), **payload)
    print(f"Saved {len(in_sum2)} imatrix tensors -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
