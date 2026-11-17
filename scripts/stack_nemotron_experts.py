#!/usr/bin/env python3
"""Stack Nemotron-H per-expert tensors into MLX ``switch_mlp`` tensors.

HF stores the routed experts individually
(``backbone.layers.L.mixer.experts.E.{up,down}_proj``); MLX's
``SwitchLinear`` wants one stacked tensor per projection
(``backbone.layers.L.mixer.switch_mlp.{fc1,fc2}``).

A layer's 240-496 experts are spread over SEVERAL shards, so this must be
done group-wise across the whole directory -- a per-shard pass silently
leaves most experts unstacked and the checkpoint then fails to load with
"Received N parameters not in model".

Run after ``moe_pipeline.py`` (the pipeline does this itself; this script
exists to repair an artifact built before the fix, and for verification).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import safetensors.torch as st
import torch

KEY_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.(weight|scales|biases)$"
)


def n_experts_for(layer: int, blocks: list[dict]) -> int:
    if layer < len(blocks):
        return int(blocks[layer].get("n_routed_experts") or 0)
    return 0


def stack_directory(output: Path, source: Path) -> int:
    cfg = json.loads((source / "config.json").read_text())
    llm = cfg.get("llm_config", cfg)
    blocks = llm.get("block_configs") or []

    shards = sorted(output.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors shards in {output}")

    groups: dict = {}
    for si, sp in enumerate(shards):
        with st.safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                m = KEY_RE.match(k)
                if m:
                    groups.setdefault((int(m.group(1)), m.group(3)), {}).setdefault(
                        m.group(4), []
                    ).append((si, k))
    print(f"[stack] {len(shards)} shards, {len(groups)} (layer, projection) groups")

    writes: dict = {}
    deletes: dict = {}
    n_stacked = 0
    for (layer, proj), suffixes in sorted(groups.items()):
        n_exp = n_experts_for(layer, blocks)
        carrier = "fc1" if proj == "up_proj" else "fc2"
        for suffix, items in suffixes.items():
            if not n_exp or len(items) != n_exp:
                print(
                    f"  WARNING layer {layer} {proj}.{suffix}: {len(items)} "
                    f"tensors, expected {n_exp} -- left unstacked"
                )
                continue
            items.sort(key=lambda t: int(KEY_RE.match(t[1]).group(2)))
            loaded: dict = {}
            by_shard: dict = {}
            for si, k in items:
                by_shard.setdefault(si, []).append(k)
            for si, ks in by_shard.items():
                with st.safe_open(str(shards[si]), framework="pt", device="cpu") as f:
                    for k in ks:
                        loaded[k] = f.get_tensor(k)
                deletes.setdefault(si, set()).update(ks)
            stacked = torch.stack([loaded[k] for _, k in items], dim=0).contiguous()
            writes.setdefault(items[0][0], {})[
                f"backbone.layers.{layer}.mixer.switch_mlp.{carrier}.{suffix}"
            ] = stacked
            n_stacked += 1

    for si, sp in enumerate(shards):
        if si not in writes and si not in deletes:
            continue
        with st.safe_open(str(sp), framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for k in deletes.get(si, ()):
            tensors.pop(k, None)
        tensors.update(writes.get(si, {}))
        tmp = "/tmp/" + sp.name + ".tmp"
        st.save_file(tensors, tmp)
        os.replace(tmp, str(sp))
        print(
            f"  {sp.name}: +{len(writes.get(si, {}))} / -{len(deletes.get(si, {}))}"
        )
    print(f"[stack] stacked {n_stacked} tensors")
    return n_stacked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="quantized model directory")
    ap.add_argument("--source", required=True, help="BF16 source (for block_configs)")
    args = ap.parse_args()
    stack_directory(Path(args.output), Path(args.source))


if __name__ == "__main__":
    main()
