"""Cross-shard expert stacking for checkpoints whose source shards split experts.

mlx-lm loads a MoE layer's routed experts from a single stacked
``switch_mlp.<proj>`` tensor, so a quantized checkpoint must present them that
way.  The obvious "stack each shard" post-process silently fails when the
*source* checkpoint sharded experts across files -- Recursive's Flint BF16
ships layer 5's 256 experts as 128 + 128 across two shards, so no single shard
ever holds a complete group and the per-shard ``all(keys)`` guard skips it.
The result loads as ``ValueError: Received N parameters not in model``.

This merges across every shard first, so group completeness is judged on the
whole checkpoint rather than one file.  It also tolerates a group that some
earlier per-shard pass already stacked: if the per-expert keys are incomplete
but a full stacked tensor exists, the stacked one wins.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import torch

# model.layers.<L>.mlp.experts.<E>.<proj>.<suffix>
_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.(\w+)$")
_STACKED_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.switch_mlp\.(\w+)\.(\w+)$")


def _load_all(path):
    import safetensors.torch as st

    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        sys.exit(f"no safetensors shards in {path}")
    tensors, origin = {}, {}
    for i, f in enumerate(shards):
        for k, v in st.load_file(f).items():
            tensors[k] = v
            origin[k] = i
    return shards, tensors, origin


def stack_flint_experts(path, num_layers, num_experts, dense_layers=()):
    """Stack per-expert routed tensors into ``switch_mlp`` across all shards.

    Returns the number of groups stacked.
    """
    import safetensors.torch as st

    shards, tensors, origin = _load_all(path)
    dense = set(dense_layers)

    per_expert, stacked = {}, {}
    for k in list(tensors):
        m = _EXPERT_RE.match(k)
        if m:
            L, e, proj, suf = (int(m.group(1)), int(m.group(2)),
                               m.group(3), m.group(4))
            per_expert.setdefault((L, proj, suf), {})[e] = k
            continue
        m = _STACKED_RE.match(k)
        if m:
            stacked[(int(m.group(1)), m.group(2), m.group(3))] = k

    n_stacked = 0
    for (L, proj, suf), members in sorted(per_expert.items()):
        if L in dense:
            continue
        key = f"model.layers.{L}.mlp.switch_mlp.{proj}.{suf}"
        if key in tensors:
            # Already stacked by an earlier partial pass; the leftover
            # per-expert keys are duplicates of experts it already holds.
            for k in members.values():
                tensors.pop(k, None)
            n_stacked += 1
            continue
        if len(members) != num_experts:
            missing = num_experts - len(members)
            sys.exit(
                f"cannot stack {key}: only {len(members)}/{num_experts} experts "
                f"present ({missing} missing) -- checkpoint is incomplete")
        vals = [tensors.pop(members[e]) for e in range(num_experts)]
        tensors[key] = torch.stack(vals, dim=0).contiguous()
        origin[key] = origin[members[0]]
        n_stacked += 1

    # Drop per-expert keys for groups that were already stacked, and any
    # dense-layer leftovers (those are plain MLPs, not routed experts).
    for (L, proj, suf), members in per_expert.items():
        if L in dense:
            for k in members.values():
                tensors.pop(k, None)

    # Re-shard: non-expert tensors keep their origin shard, stacked ones go to
    # the shard that held their first expert.  No index.json is written --
    # oMLX discovers shards from filename numbering.
    buckets = [dict() for _ in shards]
    for k, v in tensors.items():
        buckets[origin.get(k, 0)][k] = v
    for i, (f, bucket) in enumerate(zip(shards, buckets)):
        tmp = f + ".tmp"
        st.save_file(bucket, tmp)
        os.replace(tmp, f)
        print(f"  shard {i+1}/{len(shards)}: {len(bucket)} tensors")
    return n_stacked


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--num-experts", type=int, required=True)
    ap.add_argument("--dense-layers", default="")
    args = ap.parse_args()
    dense = [int(x) for x in args.dense_layers.split(",") if x.strip()]
    n = stack_flint_experts(args.path, args.num_layers, args.num_experts, dense)
    print(f"stacked {n} groups in {args.path}")


if __name__ == "__main__":
    main()
