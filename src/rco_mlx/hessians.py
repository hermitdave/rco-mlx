"""Per-expert Hessian factors shared by the error DB and the quantizer.

``scripts/capture_hessians.py`` writes one ``layer-NNN.npz`` per layer, each
holding a low-rank factor of the per-expert Hessian ``H ~ V diag(lam) V^T``
for every SwitchLinear projection. Both the allocation DB and the per-group
clip search need to look a factor up by HF weight name, so the parsing lives
here.

Keeping both consumers on the same factor format is what makes the
Hessian-weighted error DB and the clip search optimise the same objective.
"""

import re
from pathlib import Path

import numpy as np

# HF expert key -> (layer, projection, expert index)
_HESS_KEY_RE = re.compile(
    r"^((?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp)"
    r"\.experts\.(\d+)\.([a-z_]+)\.weight$"
)


def hessian_key_for(hf_key: str):
    """Map an HF expert weight name to ``(layer, projection, expert)``."""
    m = _HESS_KEY_RE.match(hf_key)
    if not m:
        return None
    return (int(m.group(2)), m.group(4), int(m.group(3)))


def load_hessian_factors(hessian_dir) -> dict:
    """Load ``(layer, proj, expert) -> (lam, vec)`` from capture_hessians.py."""
    factors = {}
    for f in sorted(Path(hessian_dir).glob("layer-*.npz")):
        z = np.load(f)
        for k in z.files:
            if not k.startswith("vec_"):
                continue
            stem, _, e = k[len("vec_"):].rpartition("_")
            parts = stem.split(".")
            # stem is "layers.N.mlp.switch_mlp.PROJ" (or the older
            # "N.mlp.switch_mlp.PROJ" from early captures)
            layer = int(parts[1] if parts[0] == "layers" else parts[0])
            proj = parts[-1]
            lam = np.asarray(z[f"lam_{stem}_{e}"], dtype=np.float32)
            vec = np.asarray(z[k], dtype=np.float32)
            factors[(layer, proj, int(e))] = (lam, vec)
    return factors
