"""Tests for the Hessian-weighted error DB (capture_hessians + per-block DB)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_builder = _load("build_error_db_perblock", _ROOT / "scripts" / "build_error_db_perblock.py")
_capture = _load("capture_hessians", _ROOT / "scripts" / "capture_hessians.py")


def test_hessian_key_for_olmoe_and_qwen35():
    assert _builder.hessian_key_for(
        "model.layers.3.mlp.experts.17.gate_proj.weight") == (3, "gate_proj", 17)
    assert _builder.hessian_key_for(
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight") == (0, "down_proj", 0)
    # non-expert tensors have no per-expert Hessian
    assert _builder.hessian_key_for("model.layers.0.self_attn.q_proj.weight") is None
    assert _builder.hessian_key_for("model.layers.0.mlp.gate.weight") is None


def test_hessian_errors_matches_dense_metric():
    """With a full-rank factor, the low-rank metric must equal tr(dW H dW^T)."""
    rng = np.random.default_rng(0)
    rows, cols, r, gs, bits = 8, 32, 32, 16, 3
    w = rng.normal(size=(rows, cols)).astype(np.float32)
    a = rng.normal(size=(cols, cols))
    h = a @ a.T  # full rank PSD
    lam, vec = np.linalg.eigh(h)[0][::-1], np.linalg.eigh(h)[1][:, ::-1]
    lam = lam[:r]
    vec = vec[:, :r]

    errs = _builder.hessian_errors(w, [bits], gs, lam.astype(np.float32), vec.astype(np.float32))
    # rebuild dW with the builder's own RTN model
    qmax = float((1 << (bits - 1)) - 1)
    grouped = w.reshape(rows, -1, gs)
    max_abs = np.abs(grouped).max(axis=-1, keepdims=True)
    s = np.where(max_abs == 0, 1.0, max_abs / qmax)
    dw = grouped - np.clip(np.round(grouped / s), -qmax, qmax) * s
    dense = float(np.trace(dw.reshape(rows, cols) @ h @ dw.reshape(rows, cols).T) / dw.size)
    assert errs[str(bits)]["error"] == pytest.approx(dense, rel=1e-4)


def test_hessian_errors_increases_with_bitwidth():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(16, 64)).astype(np.float32)
    a = rng.normal(size=(64, 64))
    h = a @ a.T
    lam, vec = np.linalg.eigh(h)[0][::-1][:32], np.linalg.eigh(h)[1][:, ::-1][:, :32]
    errs = _builder.hessian_errors(w, [2, 3, 4, 6, 8], 64, lam.astype(np.float32), vec.astype(np.float32))
    vals = [errs[str(b)]["error"] for b in (2, 3, 4, 6, 8)]
    assert all(v2 <= v1 + 1e-12 for v1, v2 in zip(vals, vals[1:]))


def test_load_hessian_factors(tmp_path):
    np.savez(
        str(tmp_path / "layer-007.npz"),
        **{
            "lam_layers.7.mlp.switch_mlp.gate_proj_3": np.array([4.0, 1.0], dtype=np.float32),
            "vec_layers.7.mlp.switch_mlp.gate_proj_3": np.ones((8, 2), dtype=np.float32),
            "n_layers.7.mlp.switch_mlp.gate_proj_3": np.int64(100),
        },
    )
    factors = _builder.load_hessian_factors(tmp_path)
    assert set(factors) == {(7, "gate_proj", 3)}
    lam, vec = factors[(7, "gate_proj", 3)]
    assert lam.shape == (2,) and vec.shape == (8, 2)


def test_randomized_eigen_recovers_dominant_subspace():
    rng = np.random.default_rng(2)
    d, r_true, r = 64, 4, 8
    vecs = np.linalg.qr(rng.normal(size=(d, r_true)))[0]
    lam = np.array([100.0, 50.0, 20.0, 5.0])
    h = (vecs * lam) @ vecs.T
    h += 1e-6 * np.eye(d)
    got_lam, got_vec = _capture.randomized_eigen(h, r)
    # the recovered top eigenvalues should be close to the true ones
    assert np.allclose(np.sort(got_lam)[::-1][:r_true], lam, rtol=0.05)
    # and the subspace should match (principal angles ~0)
    m = got_vec[:, :r_true].T @ vecs
    svals = np.linalg.svd(m, compute_uv=False)
    assert svals.min() > 0.99
