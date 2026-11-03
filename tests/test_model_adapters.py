"""Tests for MoE model adapters."""

import json
from pathlib import Path

import numpy as np
import pytest

from rco_mlx.adapters import OLMoEAdapter, Qwen3NextMoEAdapter, get, register


def _write_fake_olmoe(root: Path):
    """Write a minimal OLMoE config + one shard."""
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "olmoe",
        "hidden_size": 2048,
        "intermediate_size": 4096,
        "num_hidden_layers": 16,
        "num_experts": 64,
        "num_experts_per_token": 8,
    }
    (root / "config.json").write_text(json.dumps(cfg))

    try:
        import torch
        from safetensors.torch import save_file
        tensors = {
            "model.embed_tokens.weight": torch.zeros((1000, 2048), dtype=torch.float32),
            "model.layers.0.mlp.gate.weight": torch.zeros((64,), dtype=torch.float32),
            "model.layers.0.mlp.experts.0.w1.weight": torch.zeros((4096, 2048), dtype=torch.float32),
            "model.layers.0.mlp.experts.0.w2.weight": torch.zeros((2048, 4096), dtype=torch.float32),
            "model.layers.0.mlp.experts.0.w3.weight": torch.zeros((4096, 2048), dtype=torch.float32),
            "model.layers.0.post_attention_layernorm.weight": torch.zeros((2048,), dtype=torch.float32),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros((2048, 2048), dtype=torch.float32),
            "model.layers.0.self_attn.k_proj.weight": torch.zeros((2048, 2048), dtype=torch.float32),
            "model.layers.0.self_attn.v_proj.weight": torch.zeros((2048, 2048), dtype=torch.float32),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros((2048, 2048), dtype=torch.float32),
            "model.layers.0.self_attn.q_norm.weight": torch.zeros((2048,), dtype=torch.float32),
            "model.layers.0.self_attn.k_norm.weight": torch.zeros((2048,), dtype=torch.float32),
            "model.norm.weight": torch.zeros((2048,), dtype=torch.float32),
            "lm_head.weight": torch.zeros((1000, 2048), dtype=torch.float32),
        }
        save_file(tensors, str(root / "model-00001-of-00001.safetensors"))
    except ImportError:
        pytest.skip("safetensors/torch not installed")


def test_olmoe_adapter_enumerate(tmp_path: Path):
    _write_fake_olmoe(tmp_path)
    adapter = OLMoEAdapter()
    layers = adapter.enumerate_layers(tmp_path)
    assert len(layers) == 16
    assert layers[0]["type"] == "moe"
    assert layers[0]["param_count"] > 0


def test_olmoe_adapter_quant_predicate():
    adapter = OLMoEAdapter()
    layer = {"index": 0, "type": "moe", "weight_keys": [], "param_count": 0}
    assert adapter.quant_predicate(layer) == {"bits": 4, "group_size": 64}


def test_olmoe_adapter_grouping():
    adapter = OLMoEAdapter()
    spec = adapter.grouping_spec()
    assert spec is not None
    assert "self_attn" in spec
    assert "experts" in spec


def test_olmoe_adapter_key_map():
    adapter = OLMoEAdapter()
    assert adapter.map_key_to_mlx("model.embed_tokens.weight") is None
    assert adapter.map_key_to_mlx("model.layers.0.mlp.experts.0.w1.weight") == "model.layers.0.mlp.experts.0.w1.weight"
    assert adapter.map_key_to_mlx("lm_head.weight") is None


def test_olmoe_adapter_registered():
    cls = get("olmoe")
    assert cls is OLMoEAdapter


def test_unknown_adapter():
    with pytest.raises(KeyError):
        get("nonexistent_model_type")
