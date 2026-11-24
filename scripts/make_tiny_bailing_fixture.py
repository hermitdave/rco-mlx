#!/usr/bin/env python3
"""Synthetic Ling / Bailing MoE V3 checkpoint for adapter validation.

Builds a tiny BF16 checkpoint with the *production* key layout so the whole
rco-mlx pipeline can be exercised in seconds:

- ``model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight`` (per-expert)
- ``model.layers.N.mlp.shared_experts.{gate,up,down}_proj.weight``
- dense MLP on the first ``first_k_dense_replace`` layers (no experts there)
- KDA projections (``q/k/v/f/g/o_proj``, ``b_proj``, ``dense``, ``q/k/v_conv1d``,
  ``A_log``, ``dt_bias``) and MLA projections (``q/g_proj``,
  ``kv_a_proj_with_mqa``, ``kv_b_proj``, norms) alternating on a 5:1 schedule
- router ``mlp.gate.weight`` + ``mlp.gate.expert_bias``
- an MTP head stored at ``model.layers.<num_hidden_layers>``
- vision tower + ``linear_proj.*`` connectors, which the adapter must drop

Usage::

    python make_tiny_bailing_fixture.py --out <dir> [--num-layers 6]

Run the pipeline afterwards::

    python examples/rco_pipeline.py --model-type bailing_moe_v3 \
        --source <dir> --output <dir>-quant --target-bpw 3.0 --skip-quantize

Assertions worth making afterwards (see ``validate_bailing_fixture.py``):

- every ``weight_map`` key exists in its named shard;
- no ``model.visual.*`` or ``linear_proj.*`` tensor survives;
- ``model.safetensors.index.json`` is consistent with the emitted shards;
- config entries use ``...mlp.switch_mlp.<proj>`` (not per-expert paths);
- per-group expert bitwidths are uniform.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch


def build(out: Path, num_layers: int = 6, num_experts: int = 8,
          hidden: int = 128, moe_inter: int = 64, dense_inter: int = 256,
          vocab: int = 512, heads: int = 4, layer_group_size: int = 6,
          first_k_dense_replace: int = 1, mtp_layers: int = 1,
          seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    tensors: dict[str, torch.Tensor] = {}

    def w(name: str, shape: tuple) -> None:
        # Small non-zero values keep every quantization path well-defined
        # (mx.quantize needs a nonzero range).
        arr = rng.standard_normal(shape).astype(np.float32) * 0.05
        tensors[name] = torch.from_numpy(arr)

    head_dim = hidden // heads
    qk_nope, qk_rope, v_dim = head_dim, max(head_dim // 2, 1), head_dim

    for layer in range(num_layers):
        p = f"model.layers.{layer}"
        is_linear = not (
            (layer + 1) % layer_group_size == 0
            or layer >= num_layers // layer_group_size * layer_group_size
        )
        if layer < first_k_dense_replace:
            w(f"{p}.mlp.gate_proj.weight", (dense_inter, hidden))
            w(f"{p}.mlp.up_proj.weight", (dense_inter, hidden))
            w(f"{p}.mlp.down_proj.weight", (hidden, dense_inter))
        else:
            for proj, shape in (("gate_proj", (moe_inter, hidden)),
                                ("up_proj", (moe_inter, hidden)),
                                ("down_proj", (hidden, moe_inter))):
                for e in range(num_experts):
                    w(f"{p}.mlp.experts.{e}.{proj}.weight", shape)
            for proj, shape in (("gate_proj", (moe_inter, hidden)),
                                ("up_proj", (moe_inter, hidden)),
                                ("down_proj", (hidden, moe_inter))):
                w(f"{p}.mlp.shared_experts.{proj}.weight", shape)
            w(f"{p}.mlp.gate.weight", (num_experts, hidden))
            tensors[f"{p}.mlp.gate.expert_bias"] = torch.zeros(num_experts)

        if is_linear:  # KDA -- mirrors the real checkpoint: no `dense` here
            for proj, shape in (("q_proj", (heads * head_dim, hidden)),
                                ("k_proj", (heads * head_dim, hidden)),
                                ("v_proj", (heads * head_dim, hidden)),
                                ("f_proj", (heads * head_dim, hidden)),
                                ("g_proj", (heads * head_dim, hidden)),
                                ("o_proj", (hidden, heads * head_dim)),
                                ("b_proj", (heads, hidden))):
                w(f"{p}.attention.{proj}.weight", shape)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                # HF stores [C, 1, K]; MLX wants [C, K, 1].
                w(f"{p}.attention.{name}.weight", (heads * head_dim, 1, 4))
            tensors[f"{p}.attention.A_log"] = torch.zeros(heads)
            tensors[f"{p}.attention.dt_bias"] = torch.zeros(heads * head_dim)
            tensors[f"{p}.attention.o_norm.weight"] = torch.ones(head_dim)
        else:  # gated MLA -- owns `dense`
            kv_lora = max(hidden // 4, 8)
            w(f"{p}.attention.q_proj.weight", (heads * (qk_nope + qk_rope), hidden))
            w(f"{p}.attention.g_proj.weight", (heads, hidden))
            w(f"{p}.attention.kv_a_proj_with_mqa.weight", (kv_lora + qk_rope, hidden))
            w(f"{p}.attention.kv_b_proj.weight", (heads * (qk_nope + v_dim), kv_lora))
            w(f"{p}.attention.dense.weight", (hidden, heads * v_dim))
            tensors[f"{p}.attention.kv_a_layernorm.weight"] = torch.ones(kv_lora)

        tensors[f"{p}.input_layernorm.weight"] = torch.ones(hidden)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.ones(hidden)

    # MTP head: layer index == num_hidden_layers (independent numbering).
    for m in range(mtp_layers):
        p = f"model.layers.{num_layers + m}"
        for proj, shape in (("gate_proj", (moe_inter, hidden)),
                            ("up_proj", (moe_inter, hidden)),
                            ("down_proj", (hidden, moe_inter))):
            for e in range(num_experts):
                w(f"{p}.mlp.experts.{e}.{proj}.weight", shape)
        for proj, shape in (("gate_proj", (moe_inter, hidden)),
                            ("up_proj", (moe_inter, hidden)),
                            ("down_proj", (hidden, moe_inter))):
            w(f"{p}.mlp.shared_experts.{proj}.weight", shape)
        w(f"{p}.mlp.gate.weight", (num_experts, hidden))
        tensors[f"{p}.mlp.gate.expert_bias"] = torch.zeros(num_experts)
        # The real MTP layer (42) uses the MLA attention set, not KDA.
        w(f"{p}.attention.q_proj.weight", (heads * (qk_nope + qk_rope), hidden))
        w(f"{p}.attention.g_proj.weight", (heads, hidden))
        w(f"{p}.attention.kv_a_proj_with_mqa.weight", (kv_lora + qk_rope, hidden))
        w(f"{p}.attention.kv_b_proj.weight", (heads * (qk_nope + v_dim), kv_lora))
        w(f"{p}.attention.dense.weight", (hidden, heads * v_dim))
        tensors[f"{p}.attention.kv_a_layernorm.weight"] = torch.ones(kv_lora)
        tensors[f"{p}.enorm.weight"] = torch.ones(hidden)
        tensors[f"{p}.hnorm.weight"] = torch.ones(hidden)
        tensors[f"{p}.eh_proj.weight"] = torch.zeros(hidden * 2, hidden)
        tensors[f"{p}.final_layernorm.weight"] = torch.ones(hidden)

    w("model.word_embeddings.weight", (vocab, hidden))
    w("lm_head.weight", (vocab, hidden))
    tensors["model.norm.weight"] = torch.ones(hidden)

    # Vision tower + projector: must be dropped by the adapter.
    for b in range(2):
        w(f"model.visual.blocks.{b}.attn.qkv.weight", (hidden * 3, hidden))
        w(f"model.visual.blocks.{b}.attn.proj.weight", (hidden, hidden))
        w(f"model.visual.blocks.{b}.mlp.linear_fc1.weight", (hidden * 2, hidden))
        w(f"model.visual.blocks.{b}.mlp.linear_fc2.weight", (hidden, hidden * 2))
        tensors[f"model.visual.blocks.{b}.norm1.weight"] = torch.ones(hidden)
        tensors[f"model.visual.blocks.{b}.norm2.weight"] = torch.ones(hidden)
    w("model.visual.patch_embed.proj.weight", (hidden, 3, 2, 16, 16))
    tensors["model.visual.pos_embed.weight"] = torch.zeros(4, hidden)
    for i, shape in enumerate(((hidden, 4 * hidden), (hidden, hidden))):
        w(f"linear_proj.{i}.weight", shape)
        tensors[f"linear_proj.{i}.bias"] = torch.zeros(hidden)

    out.mkdir(parents=True, exist_ok=True)
    st.save_file(tensors, str(out / "model-00001-of-00001.safetensors"))

    cfg = {
        "model_type": "bailing_moe_v3_vl",
        "architectures": ["BailingMoeV3VLForConditionalGeneration"],
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "image_token_id": 157157,
        "vision_config": {"model_type": "qwen3_moe_vit", "depth": 2},
        "text_config": {
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
            "num_experts_per_tok": min(2, num_experts),
            "hidden_size": hidden,
            "intermediate_size": dense_inter,
            "moe_intermediate_size": moe_inter,
            "moe_shared_expert_intermediate_size": moe_inter,
            "vocab_size": vocab,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "group_norm_size": 1,
            "head_dim": head_dim,
            "kv_lora_rank": max(hidden // 4, 8),
            "q_lora_rank": None,
            "qk_nope_head_dim": qk_nope,
            "qk_rope_head_dim": qk_rope,
            "v_head_dim": v_dim,
            "layer_group_size": layer_group_size,
            "first_k_dense_replace": first_k_dense_replace,
            "num_shared_experts": 1,
            "max_position_embeddings": 4096,
            "rope_theta": 6000000.0,
            "rms_norm_eps": 1e-6,
            "routed_scaling_factor": 2.5,
            "n_group": 1,
            "topk_group": 1,
            "score_function": "sigmoid",
            "norm_topk_prob": True,
            "moe_router_enable_expert_bias": True,
            "use_qk_norm": True,
            "no_kda_lora": True,
            "kda_safe_gate": True,
            "kda_lower_bound": -5.0,
            "short_conv_kernel_size": 4,
            "mtp_use_kda": False,
            "num_nextn_predict_layers": mtp_layers,
            "expert_swiglu_limit_list": [0] * (num_layers - 2) + [4, 4],
            "share_expert_swiglu_limit_list": [0] * (num_layers - 1) + [5],
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # index.json: every tensor in shard 1.
    index = {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    meta = {
        "num_layers": num_layers, "num_experts": num_experts, "hidden": hidden,
        "moe_inter": moe_inter, "vocab": vocab, "mtp_layers": mtp_layers,
        "tensors": len(tensors),
    }
    print(f"fixture: {out} — {len(tensors)} tensors, {meta}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--moe-inter", type=int, default=64)
    ap.add_argument("--dense-inter", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(Path(args.out).expanduser(), num_layers=args.num_layers,
          num_experts=args.num_experts, hidden=args.hidden,
          moe_inter=args.moe_inter, dense_inter=args.dense_inter,
          vocab=args.vocab, heads=args.heads, seed=args.seed)


if __name__ == "__main__":
    main()
