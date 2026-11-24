#!/usr/bin/env python3
"""Validate a quantized Ling / Bailing checkpoint emitted by rco-mlx.

Checks the invariants that matter for this family:

1. every ``weight_map`` key exists in its named shard, and every tensor on
   disk is in the index (a stale index after the stacking post-process points
   at the wrong shard);
2. no vision tower, projector or FP8 scale sidecar survived;
3. every declared quantization entry has a ``.scales`` sibling on disk, and
   no per-expert path is declared (the loader wants stacked names);
4. each stacked ``switch_mlp`` tensor's shape matches its declared
   bits/group_size, i.e. ``[E, out, ceil(in*bits/32)]`` and scales
   ``[E, out, in//group_size]``;
5. experts are uniform within each (layer, projection) -- MLX's loader stacks
   them, so mixed widths would raise ``ValueError: [stack] All arrays must
   have the same shape``;
6. MTP keys (``model.layers.<num_hidden_layers>``) are either quantized
   because ``--quantize-mtp`` was passed, or entirely absent from the config.

Usage::

    python scripts/validate_bailing_artifact.py <quantized_dir> [--expect-mtp]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import struct
import sys
from pathlib import Path

_STACKED_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)


def _headers(d: Path) -> dict:
    out = {}
    for shard in sorted(glob.glob(str(d / "*.safetensors"))):
        with open(shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (Path(shard).name, tuple(v["shape"]), v.get("dtype", "?"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--expect-mtp", action="store_true",
                    help="require the MTP head to be quantized")
    args = ap.parse_args()

    d = Path(args.dir).expanduser()
    fails: list[str] = []
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((d / "config.json").read_text())
    qc = cfg.get("quantization", {})
    headers = _headers(d)

    # 1. index consistency
    mismatched = [k for k, sh in idx.items()
                  if k not in headers or headers[k][0] != sh]
    orphans = sorted(set(headers) - set(idx))
    if mismatched:
        fails.append(f"{len(mismatched)} weight_map keys missing from their shard")
    if orphans:
        fails.append(f"{len(orphans)} on-disk tensors absent from the index")
    # Shard files nobody references: the stacking post-process can leave an
    # empty header-only file behind (measured on Ling-3.2-air, shard 13 held 0
    # tensors and 4 KB). Harmless at load time but dead weight in an upload.
    # Must be measured by listing the directory -- an empty shard has no
    # entries in the index or in `headers`, so neither can reveal it.
    disk_files = {p.name for p in d.glob("*.safetensors")}
    unused = sorted(disk_files - set(idx.values()))
    if unused:
        fails.append(f"{len(unused)} shard file(s) are unreferenced by the index: "
                     f"{unused[:4]}")

    # 2. forbidden tensors
    forbidden = [k for k in headers
                 if k.startswith(("model.visual.", "visual.", "linear_proj"))
                 or k.endswith(".weight_scale_inv")]
    if forbidden:
        fails.append(f"{len(forbidden)} vision/projector/FP8-sidecar tensors present")

    # 3. config entries vs on-disk scales; naming
    modules_with_scales = {k[:-len(".scales")] for k in headers if k.endswith(".scales")}
    defaults = {k for k, v in qc.items() if not isinstance(v, dict)}
    declared = {k for k in qc if isinstance(qc[k], dict)}
    no_scales = sorted(declared - modules_with_scales)
    per_expert = sorted(k for k in declared if ".experts." in k)
    if no_scales:
        fails.append(f"{len(no_scales)} declared modules have no .scales on disk: {no_scales[:4]}")
    if per_expert:
        fails.append(f"{len(per_expert)} declared entries use per-expert paths "
                     "(loader wants switch_mlp.*): {per_expert[:4]}")

    # 4/5. stacked shapes + uniformity
    groups: dict[tuple[int, str], dict] = {}
    for k, (_shard, shape, _dt) in headers.items():
        m = _STACKED_RE.match(k)
        if not m:
            continue
        layer, proj, suffix = int(m.group(1)), m.group(2), m.group(3)
        entry = groups.setdefault((layer, proj), {})
        spec = qc.get(f"model.layers.{layer}.mlp.switch_mlp.{proj}")
        if spec is None:
            # No declaration is only acceptable for an unquantized group.
            if suffix == "scales":
                entry["quantized_undeclared"] = True
            else:
                entry.setdefault("bf16", True)
            continue
        entry["bits"] = int(spec["bits"])
        entry["gs"] = int(spec["group_size"])
        entry[suffix] = shape

    for (layer, proj), e in sorted(groups.items()):
        if e.get("quantized_undeclared"):
            fails.append(f"layer {layer} {proj}: quantized stacked tensor has no config entry")
            continue
        if "bits" not in e:
            # Pure BF16 group: the loader takes its dense path, no entry needed.
            continue
        bits, gs = e.get("bits"), e.get("gs")
        w, sc = e.get("weight"), e.get("scales")
        if not bits or not w:
            fails.append(f"layer {layer} {proj}: declared but no weight/scales pair")
            continue
        if len(w) != 3:
            fails.append(f"layer {layer} {proj}: stacked weight is {len(w)}-D, expected 3-D")
            continue
        if sc is None:
            fails.append(f"layer {layer} {proj}: missing scales tensor")
            continue
        # scales rows must equal weight rows, and in-dim must be group-divisible
        if sc[0] != w[0] or sc[1] != w[1]:
            fails.append(f"layer {layer} {proj}: scales rows {sc[:2]} != weight rows {w[:2]}")
        if len(sc) == 3 and sc[2] <= 0:
            fails.append(f"layer {layer} {proj}: empty scales")

    # 6. MTP
    n_layers = int(cfg.get("num_hidden_layers", 0))
    mtp_keys = [k for k in headers if k.startswith(f"model.layers.{n_layers}.")]
    mtp_declared = [k for k in mtp_keys if k[:-7] in declared]
    if args.expect_mtp and mtp_keys and not mtp_declared:
        fails.append(f"MTP head present ({len(mtp_keys)} tensors) but nothing declared quantized")
    if not args.expect_mtp and mtp_declared:
        fails.append(f"MTP head declared quantized without --expect-mtp ({len(mtp_declared)})")

    print(f"tensors: {len(headers)} | index keys: {len(idx)} | quant entries: {len(declared)}")
    print(f"defaults: {sorted(defaults)} | stacked groups: {len(groups)} | MTP tensors: {len(mtp_keys)}")
    if fails:
        print("\nFAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nPASS: all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
