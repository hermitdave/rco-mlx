#!/usr/bin/env python3
"""Adapter + imatrix-mapping tests for the Qwen3.5/3.8 dense hybrid family.

Uses the REAL Qwen3.8-27B index key list (cached fixture) so every exclusion,
drop and rename decision is tested against the checkpoint's actual keys.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from rco_mlx.adapters import get
from rco_mlx.adapters.base import registered_names
from rco_mlx.adapters.qwen3_5 import Qwen35DenseAdapter
from rco_mlx.imatrix import hf_key_to_gguf

FIXTURE = Path(__file__).parent / "fixtures" / "qwen38_27b_keys.json"


@pytest.fixture(scope="module")
def keys():
    if not FIXTURE.exists():
        pytest.skip("qwen38 key fixture not present")
    return json.loads(FIXTURE.read_text())


def test_registered_not_moe():
    assert "qwen3_5" in registered_names()
    assert get("qwen3_5") is Qwen35DenseAdapter
    import rco_pipeline
    assert "qwen3_5" not in rco_pipeline._MOE_ADAPTERS


def test_drop_decisions(keys):
    a = Qwen35DenseAdapter()
    for k in keys:
        if k.startswith("model.visual.") or k.startswith("mtp."):
            assert a.drop_key(k), k
        else:
            assert not a.drop_key(k), k


def test_bf16_prefix_decisions(keys):
    """Every norm / conv1d / A_log / dt_bias / embed / lm_head must be excluded
    by should_quantize's prefix logic; every projection must NOT be."""
    a = Qwen35DenseAdapter()
    prefixes = a.non_quantized_prefixes()

    def excluded(mlx_key):
        if "norm" in mlx_key or "embed" in mlx_key or "lm_head" in mlx_key or "mtp" in mlx_key:
            return True
        if not mlx_key.endswith(".weight"):
            return True
        for p in prefixes:
            if f".{p}." in f".{mlx_key}." or mlx_key.startswith(f"{p}."):
                return True
        return False

    n_quant = 0
    for k in keys:
        if k.startswith("model.visual.") or k.startswith("mtp."):
            continue
        mlx = a.map_key_to_mlx(k)
        is_proj = any(s in k for s in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "linear_attn.in_proj", "linear_attn.out_proj",
            ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj",
        ))
        if is_proj:
            assert not excluded(mlx), f"projection wrongly excluded: {k}"
            n_quant += 1
        elif k.endswith(".weight"):
            assert excluded(mlx), f"non-projection wrongly quantizable: {k}"
    # 64*3 MLP + 16*4 attn + 48*5 linear-attn projections
    assert n_quant == 192 + 64 + 240, n_quant


def test_map_key_idempotent(keys):
    a = Qwen35DenseAdapter()
    for k in keys:
        once = a.map_key_to_mlx(k)
        twice = a.map_key_to_mlx(once)
        assert once == twice, k
    assert a.map_key_to_mlx("model.language_model.layers.0.mlp.gate_proj.weight") == \
        "language_model.model.layers.0.mlp.gate_proj.weight"
    assert a.map_key_to_mlx("lm_head.weight") == "language_model.lm_head.weight"


def test_gguf_mapping_dense(keys):
    """Every trunk projection must map to a GGUF imatrix stem."""
    stems = {}
    for k in keys:
        if k.startswith("model.visual.") or k.startswith("mtp."):
            continue
        if not any(s in k for s in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "linear_attn.in_proj", "linear_attn.out_proj",
            ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj",
        )):
            continue
        m = hf_key_to_gguf(k)
        assert m is not None, f"no GGUF mapping for {k}"
        stems[k.rsplit(".", 2)[-2]] = m[0].split(".")[2]
    assert stems["gate_proj"] == "ffn_gate"
    assert stems["up_proj"] == "ffn_up"
    assert stems["down_proj"] == "ffn_down"
    assert stems["in_proj_qkv"] == "attn_qkv"
    assert stems["in_proj_z"] == "attn_gate"
    assert stems["in_proj_a"] == "ssm_alpha"
    assert stems["in_proj_b"] == "ssm_beta"
    assert stems["out_proj"] == "ssm_out"
    assert stems["q_proj"] == "attn_q"


def test_gguf_mapping_experts_unaffected():
    """Dense MLP patterns must not swallow MoE expert / shared-expert keys."""
    assert hf_key_to_gguf("model.layers.0.mlp.experts.3.gate_proj.weight")[0] == "blk.0.ffn_gate_exps.weight"
    assert hf_key_to_gguf("model.layers.0.mlp.shared_expert.gate_proj.weight")[0] == "blk.0.ffn_gate_shexp.weight"
    # Bare fused expert keys (no .weight, no expert index) were never mapped by
    # hf_key_to_gguf; the regression to guard is the dense pattern swallowing
    # them into ffn_gate.
    m = hf_key_to_gguf("model.layers.0.mlp.experts.gate_up_proj")
    assert m is None or m[0].endswith("ffn_gate_exps.weight")