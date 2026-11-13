#!/usr/bin/env python3
"""Capture per-expert Hessians H = 2 X^T X / n for MoE models.

Same SwitchLinear hook mechanism as capture_imatrix.py, but accumulates the
FULL second-moment matrix per expert instead of only its diagonal.  Used to
build a Hessian-weighted (output-error) error DB for RCO allocation.

Memory is bounded by streaming one layer at a time: a layer's Hessians are
accumulated, normalized and written out before the next layer's forward pass.
Peak RAM is one layer's Hessians (~2.4 GB for OLMoE) plus the model.

Note the per-expert sample count is usually BELOW the input dimension for
gate/up (n ~ tokens*top_k/n_experts, d = hidden), so these matrices are
rank-deficient by construction.  That is fine for the error metric
tr(dW H dW^T) (always well-defined) but makes GPTQ's inverse unreliable --
which is exactly why this feeds the allocation DB rather than the quantizer.

Output: one .npz per layer, work/hessians/layer-NNN.npz, with arrays
    lam_{proj}_{expert}  float32 [r]      top-r eigenvalues
    vec_{proj}_{expert}  float32 [d, r]   top-r eigenvectors

Only the top-r eigenpairs are kept (r=64 by default): the metric
tr(dW H dW^T) ~ ||dW V diag(sqrt(lam))||^2 is what the error DB needs, and
storing the dense [d, d] matrix for all 48 layers would be ~116 GB.

Usage:
    python scripts/capture_hessians.py \
        --source /path/to/bf16 --calibration calib.npy \
        --output work/hessians --chunks 8 [--layers 0,1,2] [--rank 64]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def randomized_eigen(h: np.ndarray, r: int, n_iter: int = 2, seed: int = 0):
    """Top-r eigenpairs of an SPD matrix via a randomized range finder.

    Much cheaper than ``eigsh`` at MoE scale (one d x d matmul per pass
    instead of an iterative eigensolver), and the metric only needs the
    dominant subspace -- the small eigenvalues contribute little to
    tr(dW H dW^T).
    """
    rng = np.random.default_rng(seed)
    d = h.shape[0]
    r = min(r, d)
    p = min(10, d - r)
    omega = rng.standard_normal((d, r + p)).astype(np.float32)
    q, _ = np.linalg.qr(h @ omega)
    for _ in range(n_iter):
        q, _ = np.linalg.qr(h @ q)
    b = q.T @ h @ q
    s, u = np.linalg.eigh(b)
    order = np.argsort(s)[::-1][:r]
    lam = np.maximum(s[order], 0.0).astype(np.float32)
    vecs = (q @ u[:, order]).astype(np.float32)
    return lam, vecs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--layers", default=None, help="comma-separated layer indices (default: all)")
    ap.add_argument("--rank", type=int, default=64, help="top-r eigenpairs to keep (default 64)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm

    from rco_mlx.imatrix import fits_in_memory

    source = Path(args.source)
    fits, model_bytes, ram = fits_in_memory(source)
    print(f"Model {model_bytes/1e9:.1f} GB, RAM {ram/1e9:.1f} GB, fits={fits}")
    if not fits and not args.force:
        sys.exit("refusing: model does not fit the 70% RAM gate")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _ = mlx_lm.load(str(source))

    # every SwitchLinear projection, keyed by its module path
    projections: dict[str, object] = {}
    for path, module in model.named_modules():
        if type(module).__name__ == "SwitchLinear":
            projections[path] = module
    if not projections:
        sys.exit("no SwitchLinear projections found")
    print(f"{len(projections)} SwitchLinear projections")

    layer_ids = sorted({int(p.split(".")[2]) for p in projections})
    if args.layers:
        wanted = {int(x) for x in args.layers.split(",")}
        layer_ids = [l for l in layer_ids if l in wanted]

    id_of = {id(m): p for p, m in projections.items()}
    orig_call = {type(m): type(m).__call__ for m in projections.values()}

    state = {"targets": {}, "hess": {}, "counts": {}}

    def switch_call(self, x, indices=None, *a, **kw):
        path = id_of.get(id(self))
        if path is not None and path in state["targets"]:
            try:
                x_np = np.asarray(x)
                idx_np = np.asarray(indices) if indices is not None else None
                x2 = x_np.reshape(-1, x_np.shape[-1]).astype(np.float32)
                k = idx_np.shape[1] if idx_np.ndim == 2 else 1
                idx_flat = idx_np.reshape(-1)
                x_per_slot = np.repeat(x2, k, axis=0)
                e_total = int(self.weight.shape[0])
                h_dict = state["hess"].setdefault(path, {})
                n_dict = state["counts"].setdefault(path, {})
                d = x2.shape[1]
                for e in range(e_total):
                    mask = idx_flat == e
                    if mask.any():
                        xe = x_per_slot[mask]
                        h_dict[e] = h_dict.get(e, np.zeros((d, d), dtype=np.float32)) + xe.T @ xe
                        n_dict[e] = n_dict.get(e, 0) + xe.shape[0]
            except Exception as exc:
                print("  hook error:", exc)
        return orig_call[type(self)](self, x, indices, *a, **kw)

    seen: set = set()
    for m in projections.values():
        if type(m) not in seen:
            seen.add(type(m))
            type(m).__call__ = switch_call

    calib = np.load(args.calibration)
    n_chunks = min(args.chunks, calib.shape[0])
    chunks = [calib[i : i + 1].astype(np.int32) for i in range(n_chunks)]

    for layer in layer_ids:
        targets = {p: m for p, m in projections.items() if f"layers.{layer}." in p}
        state["targets"] = targets
        state["hess"] = {}
        state["counts"] = {}
        print(f"\n[layer {layer}] {len(targets)} projections: {sorted(targets)}", flush=True)
        for i, ids in enumerate(chunks):
            logits = model(mx.array(ids))
            mx.eval(logits)
            print(f"  chunk {i+1}/{n_chunks}", flush=True)

        payload = {}
        for path, h_dict in state["hess"].items():
            short = path.split(".", 2)[-1]  # layers.N.mlp.switch_mlp.PROJ
            for e, h in h_dict.items():
                n = max(state["counts"][path][e], 1)
                lam, vecs = randomized_eigen(
                    (2.0 * h / n).astype(np.float64), args.rank
                )
                payload[f"lam_{short}_{e}"] = lam
                payload[f"vec_{short}_{e}"] = vecs
                payload[f"n_{short}_{e}"] = np.int64(n)
        out_path = out_dir / f"layer-{layer:03d}.npz"
        np.savez(str(out_path), **payload)
        total = sum(v.nbytes for v in payload.values()) / 1e9
        print(f"  wrote {out_path} ({len(payload)//3} experts, r={args.rank}, {total:.2f} GB)")

    print("\ndone")


if __name__ == "__main__":
    main()
