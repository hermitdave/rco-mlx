"""Tests for the generic dense adapter and dense-model detection."""

import importlib.util
import json
from pathlib import Path

import pytest

from rco_mlx.adapters import DefaultAdapter, get


def _write_fake_dense(root: Path):
    """Write a minimal dense-transformer config + one shard."""
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }
    (root / "config.json").write_text(json.dumps(cfg))

    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    tensors = {
        "model.embed_tokens.weight": torch.zeros((32, 64), dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros((64, 64), dtype=torch.float32),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros((32, 64), dtype=torch.float32),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros((32, 64), dtype=torch.float32),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros((64, 64), dtype=torch.float32),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros((128, 64), dtype=torch.float32),
        "model.layers.0.mlp.up_proj.weight": torch.zeros((128, 64), dtype=torch.float32),
        "model.layers.0.mlp.down_proj.weight": torch.zeros((64, 128), dtype=torch.float32),
        "model.layers.0.input_layernorm.weight": torch.zeros((64,), dtype=torch.float32),
        "model.norm.weight": torch.zeros((64,), dtype=torch.float32),
        "lm_head.weight": torch.zeros((32, 64), dtype=torch.float32),
    }
    save_file(tensors, str(root / "model-00001-of-00001.safetensors"))
    return root


def test_default_adapter_registered():
    assert get("default") is DefaultAdapter


def test_default_adapter_enumerate(tmp_path: Path):
    _write_fake_dense(tmp_path)
    layers = DefaultAdapter().enumerate_layers(tmp_path)
    assert len(layers) == 2
    assert layers[0]["type"] == "dense"
    assert any("q_proj" in k for k in layers[0]["weight_keys"])
    assert layers[0]["param_count"] > 0


def test_default_adapter_quant_predicate_and_grouping():
    adapter = DefaultAdapter()
    assert adapter.quant_predicate({}) == {"bits": 4, "group_size": 64}
    assert adapter.grouping_spec() is None


def test_default_adapter_keeps_projection_keys_quantizable():
    adapter = DefaultAdapter()
    # Dense projections map through unchanged.
    for key in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.conv.in_proj.weight",
        "model.layers.0.conv.out_proj.weight",
    ):
        assert adapter.map_key_to_mlx(key) == key


def test_default_adapter_bf16_prefixes_do_not_swallow_convs():
    prefixes = DefaultAdapter().non_quantized_prefixes()
    assert "norm" in prefixes
    assert "embed_tokens" in prefixes
    # A `conv.conv`-style prefix would silently drop conv.in_proj/out_proj
    # through _filter_bf16_prefixes' last-component fallback.
    assert "conv.conv" not in prefixes


def test_default_adapter_nested_wrapper_translation():
    adapter = DefaultAdapter()
    assert (
        adapter.map_key_to_mlx("model.language_model.layers.0.self_attn.q_proj.weight")
        == "language_model.model.layers.0.self_attn.q_proj.weight"
    )


def test_detect_moe_model_false_for_dense(tmp_path: Path):
    _write_fake_dense(tmp_path)
    p = Path(__file__).resolve().parent.parent / "examples" / "moe_pipeline.py"
    spec = importlib.util.spec_from_file_location("moe_pipeline", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.detect_moe_model(tmp_path, None) is False
