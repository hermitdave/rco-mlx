#!/usr/bin/env python3
"""Synthetic monotonic error DB for ADAPTER/PIPELINE VALIDATION ONLY.

Builds an ``error_db.json`` whose loss rows decrease monotonically with bitwidth
and vary by module class, so the RCO allocation can be exercised on a tiny
fixture without a calibration corpus or any model forward pass.

NOT for real quantization: the numbers are invented. A real run must build the
DB from activations (`--calibration-dataset` / `--calibration`), because the
allocation is only as good as those loss estimates.

Usage::

    python scripts/make_stub_error_db.py <source_dir> <out.json>
"""
import json, struct, glob, sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
hdr = {}
for s in sorted(glob.glob(str(src / "*.safetensors"))):
    with open(s, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k != "__metadata__":
            hdr[k] = v["shape"]


def base(m):
    if ".experts." in m:
        return 1.0
    if "shared_experts" in m:
        return 2.0
    if ".attention." in m:
        return 3.0
    if m.startswith("lm_head") or "word_embeddings" in m:
        return 5.0
    return 2.5


BITS = [2, 3, 4, 6, 8]
layers, pc = {}, {}
for k, shape in hdr.items():
    m = k[:-7] if k.endswith(".weight") else k
    ne = 1
    for d in shape:
        ne *= int(d)
    b = base(m)
    row = {str(bit): {"error": float(b * (2.0 ** (-(bit - 2))) + 0.01 * b)} for bit in BITS}
    layers[m] = row
    layers.setdefault(m + ".weight", row)
    pc[m] = ne
    pc[m + ".weight"] = ne

out.write_text(json.dumps({"layers": layers, "param_counts": pc, "_meta": {"synthetic": True}}))
print("wrote", out, "modules:", len(layers))
