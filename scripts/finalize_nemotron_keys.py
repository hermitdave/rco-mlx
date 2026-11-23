#!/usr/bin/env python3
"""Rename + prune an already-quantized Nemotron-H checkpoint in place.

``rco_pipeline.py`` writes quantized tensors under their MLX name but BF16
fallbacks under the *source* name, and it has no way to omit tensors whose
tower has no MLX counterpart.  The pipeline now does both correctly (see
``NemotronHPuzzleAdapter.map_key_to_mlx`` / ``drop_key``); this script applies
the same transform to an artifact built before that fix.

Pass 1  --rename   strip ``language_model.`` so every tensor uses the MLX name
Pass 2  --prune    drop vision/audio towers (no MLX module -> load_weights fails)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import safetensors.torch as st

DROP_PREFIXES = ("vision_model", "mlp1", "audio_model", "sound_model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--rename", action="store_true")
    ap.add_argument("--prune", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    shards = sorted(out.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no shards in {out}")

    for sp in shards:
        with st.safe_open(str(sp), framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        renamed = {}
        n_drop = 0
        for k, v in tensors.items():
            if args.prune and k.startswith(DROP_PREFIXES):
                n_drop += 1
                continue
            nk = k
            if args.rename and nk.startswith("language_model."):
                nk = nk[len("language_model."):]
            renamed[nk] = v
        if not args.rename and not args.prune:
            print("nothing to do (pass --rename and/or --prune)")
            return
        tmp = "/tmp/" + sp.name + ".tmp"
        st.save_file(renamed, tmp)
        os.replace(tmp, str(sp))
        print(f"  {sp.name}: {len(tensors)} -> {len(renamed)} (-{n_drop} pruned)")

    idx = out / "model.safetensors.index.json"
    if idx.exists():
        print(f"stale index: {idx} -- rebuild it with the pipeline's "
              f"write_output_index or delete it")


if __name__ == "__main__":
    main()
