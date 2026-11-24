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


def stack_bailing_experts(path, num_layers, num_experts, dense_layers=()):
    """Stack Ling/Bailing per-expert routed tensors into ``switch_mlp``.

    Ling stores routed experts as ``model.layers.<L>.mlp.experts.<E>.<proj>.<suffix>``
    and MLX's ``SwitchGLU`` wants one stacked tensor per (layer, projection)
    named ``model.layers.<L>.mlp.switch_mlp.<proj>.<suffix>``.  Same layout as
    Flint, so the same regexes apply; the dense layers (``first_k_dense_replace``
    of them) carry a plain MLP instead and are skipped.

    Memory is bounded by one shard: group membership is resolved from the
    shard *headers*, then each file is loaded, stacked and rewritten once.
    Unlike the Flint path this refuses to guess when a group is incomplete --
    a silently skipped group loads as ``Received N parameters not in model``.
    """
    import glob
    import json
    import os
    import struct

    import safetensors.torch as st
    import torch

    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        sys.exit(f"no safetensors shards in {path}")

    def _header_keys(shard):
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        return {k for k in header if k != "__metadata__"}

    keys_per_shard = [_header_keys(s) for s in shards]
    shard_of = {}
    for i, keys in enumerate(keys_per_shard):
        for k in keys:
            shard_of[k] = i

    per_expert, stacked = {}, set()
    for k in shard_of:
        m = _EXPERT_RE.match(k)
        if m:
            per_expert.setdefault(
                (int(m.group(1)), m.group(3), m.group(4)), {}
            )[int(m.group(2))] = k
            continue
        m = _STACKED_RE.match(k)
        if m:
            stacked.add((int(m.group(1)), m.group(2), m.group(3)))

    dense = set(dense_layers)

    # Validate before writing anything.
    for (layer, proj, suf), members in sorted(per_expert.items()):
        if layer in dense or (layer, proj, suf) in stacked:
            continue
        key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suf}"
        if len(members) != num_experts:
            sys.exit(
                f"cannot stack {key}: {len(members)}/{num_experts} experts "
                "present -- checkpoint is incomplete"
            )
        owners = {shard_of[members[e]] for e in range(num_experts)}
        if len(owners) != 1:
            # Ling shards a layer's routed experts ACROSS files (measured:
            # layer 3's down_proj weights split 374 + 138 between shards 1 and
            # 2), so no single shard ever holds a complete group. Nothing has
            # been written yet at this point, so hand the whole job to the
            # merge-everything stacker -- the same fallback Flint needs.
            print(
                f"  {key}: experts span shards {sorted(owners)}; "
                "falling back to cross-shard merge"
            )
            return stack_flint_experts(path, num_layers, num_experts, dense_layers)

    n_stacked = 0
    for i, shard in enumerate(shards):
        tensors = st.load_file(shard)
        new = dict(tensors)
        touched = 0
        for (layer, proj, suf), members in per_expert.items():
            if layer in dense or (layer, proj, suf) in stacked:
                continue
            if shard_of[members[0]] != i:
                continue
            vals = [new.pop(members[e]) for e in range(num_experts)]
            new[f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suf}"] = (
                torch.stack(vals, dim=0).contiguous()
            )
            touched += 1
        # Leftovers of groups that were already stacked, or that belong to a
        # dense layer (plain MLP, not routed experts).
        for (layer, proj, suf), members in per_expert.items():
            if layer in dense or (layer, proj, suf) in stacked:
                for k in members.values():
                    new.pop(k, None)
        if touched:
            tmp = shard + ".tmp"
            st.save_file(new, tmp)
            os.replace(tmp, shard)
            print(f"  {os.path.basename(shard)}: stacked {touched} groups")
        n_stacked += touched
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
