"""Tests for imatrix weighting, clip search, and per-tensor group sizes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rco_mlx.imatrix import Imatrix, fits_in_memory, hf_key_to_gguf
from rco_mlx.mlx_alloc import allocation_to_config

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_error_db_perblock.py"
_spec = importlib.util.spec_from_file_location("build_error_db_perblock", _SCRIPT)
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)


def test_hf_key_to_gguf_experts_and_attention():
    assert hf_key_to_gguf("model.layers.3.mlp.experts.17.gate_proj.weight") == (
        "blk.3.ffn_gate_exps.weight", 17)
    assert hf_key_to_gguf("model.language_model.layers.0.mlp.experts.0.down_proj.weight") == (
        "blk.0.ffn_down_exps.weight", 0)
    assert hf_key_to_gguf("model.layers.1.self_attn.q_proj.weight") == (
        "blk.1.attn_q.weight", None)
    assert hf_key_to_gguf("model.layers.1.mlp.gate.weight") is None


def test_imatrix_npz_roundtrip_weighting(tmp_path):
    path = tmp_path / "imatrix.npz"
    # Per-expert NPZ layout: one [in] vector per HF tensor.
    col = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    np.savez(
        str(path),
        **{
            "model.layers.0.mlp.experts.0.gate_proj.weight.in_sum2": col,
            "model.layers.0.mlp.experts.0.gate_proj.weight.counts":
                np.array([100.0], dtype=np.float32),
            "model.layers.0.mlp.experts.1.gate_proj.weight.in_sum2": col,
            "model.layers.0.mlp.experts.1.gate_proj.weight.counts":
                np.array([0.0], dtype=np.float32),  # never routed
        },
    )
    im = Imatrix(path)
    col, routed = im.col_weights_for_hf_key(
        "model.layers.0.mlp.experts.0.gate_proj.weight")
    # mean-1 normalized activation weights
    assert np.allclose(col, [0.4, 0.8, 1.2, 1.6])
    assert routed == 100.0
    col_cold, routed_cold = im.col_weights_for_hf_key(
        "model.layers.0.mlp.experts.1.gate_proj.weight")
    assert routed_cold == 0.0  # never routed -> zero output effect
    assert im.col_weights_for_hf_key("model.layers.0.self_attn.q_proj.weight") is None


def test_rtn_errors_weighted_and_clip_search():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(8, 16)).astype(np.float32)
    w[0, 0] = 50.0  # one outlier column entry
    unweighted = _builder.rtn_errors(w, [2])
    clipped = _builder.rtn_errors(w, [2], clip_search=True)
    # scale search can only improve (or match) the builder's RTN error
    assert clipped["2"]["error"] <= unweighted["2"]["error"] + 1e-12
    assert clipped["2"]["clip"] <= 1.0

    # Columns with zero activation weight contribute no error.
    cw = np.ones(16); cw[0] = 0.0
    weighted = _builder.rtn_errors(w, [2], col_weights=cw)
    assert weighted["2"]["error"] != pytest.approx(unweighted["2"]["error"])

    zero = _builder.rtn_errors(w, [2], col_weights=np.zeros(16))
    assert zero["2"]["error"] == 0.0


def test_allocation_to_config_per_tensor_group_size():
    cfg = allocation_to_config(
        {"a.weight": 4, "b.weight": 2},
        group_size=64,
        group_size_map={"a.weight": 32, "b.weight": 128},
    )
    block = cfg.to_dict()
    assert block["a"]["group_size"] == 32
    assert block["b"]["group_size"] == 128
    assert block["group_size"] == 64  # global default unchanged


def test_fits_in_memory_gate(tmp_path):
    # A tiny "model" (few KB of safetensors + index) fits anywhere.
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1024)
    fits, model_bytes, ram = fits_in_memory(tmp_path)
    assert model_bytes == 1024
    assert ram > 0
    assert fits is True
