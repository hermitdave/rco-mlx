#!/usr/bin/env python3
"""Model-independent streaming imatrix capture, with per-model plugins.

``capture_imatrix.py`` needs the whole BF16 model resident; that is impossible
for checkpoints over ~60 GB.  This engine keeps **one layer resident at a
time**, carries hidden states for every calibration chunk, accumulates
per-input-column ``in_sum2``/``counts`` through forward hooks, and writes the
same ``.npz`` the in-memory capturer writes -- so ``rco_mlx.imatrix.Imatrix``
reads either interchangeably.

    H = [embed(chunk) for chunk in chunks]
    for layer in layers:  H = [layer(h) for h in H]   # layer resident, then freed

Everything architecture-specific lives in a ``CaptureProfile`` plugin, which
carries exactly the shape/key knowledge that cannot be inferred: how to build
args, the static modules, one layer module, the HF<->MLX key mapping (including
expert stacking), and the mask/cache for that layer's type.  Adding a family
means adding a ~60-line plugin, not forking the engine.

Usage:
    python scripts/capture_imatrix_streaming.py \
        --source /path/to/BF16 --model-type nemotron_h_puzzle \
        --calibration calib.npy --output imatrix.npz --max-chunks 512

``--model-type`` is optional and inferred from ``config.json`` when omitted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rco_mlx.calibration import (  # noqa: E402
    MIN_EXPERT_COUNT,
    MIN_EXPERT_COUNT_PERCENTILE,
    expert_coverage_stats,
)


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------

class CaptureProfile:
    """The only place architecture knowledge lives."""

    name = ""
    model_types: tuple = ()

    def load_args(self, src: Path):
        """Build the MLX model-args object for this checkpoint."""

    def static_keys(self) -> tuple:
        """HF keys for embeddings / final norm / lm_head."""

    def build_static(self, args, mx, nn):
        """Return ``(embedding, norm, head)`` modules."""

    def layer_key_prefix(self, idx: int) -> str:
        """HF key prefix selecting one layer's tensors from the index."""

    def module_prefix(self, idx: int) -> str:
        """MLX module prefix used for imatrix keys."""

    def build_layer(self, args, idx: int):
        """One resident layer module."""

    def map_layer(self, raw: dict, idx: int) -> dict:
        """HF layer tensors ``{hf_key: array}`` -> MLX ``load_weights`` items."""

    def mlx_to_hf(self, mlx_path: str, expert: Optional[int] = None) -> str:
        """MLX module path -> HF-style imatrix key."""

    def mask_and_cache(self, args, idx: int, h, mx):
        """``(mask, cache)`` for this layer."""


class NemotronHPuzzleProfile(CaptureProfile):
    name = "nemotron_h_puzzle"
    model_types = ("nemotron_h_puzzle", "nemotron_h_omni")

    def load_args(self, src: Path):
        from mlx_lm.models.nemotron_h import ModelArgs

        hf = json.loads((src / "config.json").read_text())
        cfg = dict(hf["llm_config"])
        cfg["model_type"] = "nemotron_h_puzzle"
        for k in ("auto_map", "architectures", "_megatron_bridge_hf_prefix",
                  "mtp_block_configs", "mtp_layers_block_type", "blockwise_members"):
            cfg.pop(k, None)
        # HF writes [0.0, {"__float__": "Infinity"}], which the dataclass rejects.
        cfg["time_step_limit"] = None
        return ModelArgs.from_dict(cfg)

    def static_keys(self):
        return ("language_model.backbone.embeddings.weight",
                "language_model.backbone.norm_f.weight",
                "language_model.lm_head.weight")

    def build_static(self, args, mx, nn):
        from mlx_lm.models.nemotron_h import NemotronHRMSNorm

        return (nn.Embedding(args.vocab_size, args.hidden_size),
                NemotronHRMSNorm(args.hidden_size, eps=args.layer_norm_epsilon),
                nn.Linear(args.hidden_size, args.vocab_size, bias=False))

    def layer_key_prefix(self, idx):
        return f"language_model.backbone.layers.{idx}."

    def module_prefix(self, idx):
        return f"backbone.layers.{idx}"

    def build_layer(self, args, idx):
        from mlx_lm.models.nemotron_h import NemotronHBlock

        pattern = args.hybrid_override_pattern or []
        return NemotronHBlock(args.for_layer(idx), pattern[idx])

    def map_layer(self, raw, idx):
        import mlx.core as mx

        pre = self.layer_key_prefix(idx)
        experts: dict = {}
        out = {}
        for k, v in raw.items():
            rel = k[len(pre):]
            m = re.match(r"mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$", rel)
            if m:
                e, proj = int(m.group(1)), m.group(2)
                experts.setdefault("fc1" if proj == "up_proj" else "fc2", {})[e] = v
                continue
            if rel.endswith("conv1d.weight"):
                v = v.moveaxis(2, 1)  # [dim, 1, k] -> [dim, k, 1]
            out[rel] = v
        for name, per_expert in experts.items():
            out[f"mixer.switch_mlp.{name}.weight"] = mx.stack(
                [per_expert[e] for e in sorted(per_expert)])
        return out

    def mlx_to_hf(self, mlx_path, expert=None):
        m = re.match(r"^backbone\.(layers\.\d+\.mixer)\.switch_mlp\.(fc1|fc2)$", mlx_path)
        if m:
            proj = "up_proj" if m.group(2) == "fc1" else "down_proj"
            suffix = f"experts.{expert}.{proj}" if expert is not None else f"experts.{proj}"
            return f"language_model.backbone.{m.group(1)}.{suffix}.weight"
        if mlx_path.startswith("backbone."):
            return "language_model." + mlx_path + ".weight"
        return mlx_path + ".weight"

    def mask_and_cache(self, args, idx, h, mx):
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        from mlx_lm.models.cache import ArraysCache, KVCache

        bt = (args.hybrid_override_pattern or [])[idx]
        if bt == "*":
            return create_attention_mask(h, None), KVCache()
        if bt == "M":
            return create_ssm_mask(h, None), ArraysCache(size=2)
        return None, None


class Qwen35Profile(CaptureProfile):
    """Qwen3.5 / Qwen3.6 MoE -- hybrid GatedDeltaNet + full attention.

    HF stores routed experts as one fused stacked ``mlp.experts.gate_up_proj``
    ``(E, 2H, in)`` plus ``mlp.experts.down_proj`` ``(E, out, in)``; mlx-lm's
    ``SwitchGLU`` wants separate ``switch_mlp.{gate,up,down}_proj`` SwitchLinears
    of shape ``(E, out, in)``.  ``DecoderLayer(args, idx)`` picks linear-vs-full
    attention itself from ``(idx, full_attention_interval)``.
    """

    name = "qwen3_5_moe"
    model_types = ("qwen3_5_moe", "qwen3_5")

    # mlx-lm's sanitize() shifts these by +1.0 when the checkpoint ships MTP
    # weights (Qwen3.6 does), and moves conv1d to [dim, k, 1].
    _NORM_SUFFIXES = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        "language_model.model.norm.weight",
    )

    def load_args(self, src: Path):
        from mlx_lm.models.qwen3_5 import TextModelArgs

        cfg = json.loads((src / "config.json").read_text())
        args = TextModelArgs.from_dict(cfg.get("text_config", cfg))
        args.model_type = self.name
        return args

    def static_keys(self):
        return ("model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight",
                "lm_head.weight")

    def build_static(self, args, mx, nn):
        return (nn.Embedding(args.vocab_size, args.hidden_size),
                nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps),
                nn.Linear(args.hidden_size, args.vocab_size, bias=False))

    def layer_key_prefix(self, idx):
        return f"model.language_model.layers.{idx}."

    def module_prefix(self, idx):
        return f"language_model.model.layers.{idx}"

    def build_layer(self, args, idx):
        from mlx_lm.models.qwen3_5 import DecoderLayer

        return DecoderLayer(args, idx)

    def _sanitize(self, mlx_key: str, v):
        """Mirror mlx-lm's ``sanitize`` for this checkpoint family."""
        if "conv1d.weight" in mlx_key and v.shape[-1] != 1:
            v = v.moveaxis(2, 1)  # [dim, 1, k] -> [dim, k, 1]
        if any(mlx_key.endswith(s) for s in self._NORM_SUFFIXES) and v.ndim == 1:
            v = v + 1.0
        return v

    def map_layer(self, raw, idx):
        out = {}
        for k, v in raw.items():
            mlx = k[len("model."):] if k.startswith("model.") else k
            mlx = mlx.replace("language_model.layers", "language_model.model.layers")
            v = self._sanitize(mlx, v)
            if mlx.endswith(".mlp.experts.gate_up_proj"):
                stem = mlx[:-len(".mlp.experts.gate_up_proj")]
                half = v.shape[1] // 2
                out[stem + ".mlp.switch_mlp.gate_proj.weight"] = v[:, :half, :]
                out[stem + ".mlp.switch_mlp.up_proj.weight"] = v[:, half:, :]
                continue
            if mlx.endswith(".mlp.experts.down_proj"):
                out[mlx[:-len(".mlp.experts.down_proj")]
                    + ".mlp.switch_mlp.down_proj.weight"] = v
                continue
            out[mlx if mlx.endswith(".weight") else mlx + ".weight"] = v
        return out

    def mlx_to_hf(self, mlx_path, expert=None):
        m = re.match(
            r"^language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\."
            r"(gate_proj|up_proj|down_proj)$", mlx_path)
        if m:
            e = f"experts.{expert}." if expert is not None else "experts."
            return f"model.language_model.layers.{m.group(1)}.mlp.{e}{m.group(2)}"
        m = re.match(r"^language_model\.model\.layers\.(\d+)\.(.+)$", mlx_path)
        if m:
            return f"model.language_model.layers.{m.group(1)}.{m.group(2)}"
        return mlx_path + ".weight"

    def mask_and_cache(self, args, idx, h, mx):
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        if (idx + 1) % args.full_attention_interval != 0:
            return create_ssm_mask(h, None), None
        return create_attention_mask(h, None), None


PROFILES: Dict[str, CaptureProfile] = {}
for _cls in (NemotronHPuzzleProfile, Qwen35Profile):
    _inst = _cls()
    for _mt in _inst.model_types:
        PROFILES[_mt] = _inst


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def load_raw(src: Path, wm: dict, keys: dict, mx):
    from safetensors import safe_open

    by_shard: dict = {}
    for k, s in keys.items():
        by_shard.setdefault(s, []).append(k)
    out = {}
    for shard, ks in by_shard.items():
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            for k in ks:
                out[k] = mx.array(f.get_tensor(k))
    return out


def capture(profile: CaptureProfile, src: Path, args) -> int:
    import mlx.core as mx
    import mlx.nn as nn

    margs = profile.load_args(src)
    pattern = getattr(margs, "hybrid_override_pattern", None) or []
    n_layers = len(pattern) or getattr(margs, "num_hidden_layers", 40)
    if args.max_layer is not None:
        n_layers = min(n_layers, args.max_layer)
    print(f"[profile] {profile.name}", flush=True)
    print(f"[cfg] layers={n_layers} hidden={getattr(margs, 'hidden_size', '?')} "
          f"vocab={getattr(margs, 'vocab_size', '?')}", flush=True)

    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    # ---- static: embeddings, final norm, lm_head -------------------------
    static_raw = load_raw(src, wm,
                          {k: s for k, s in wm.items() if k in profile.static_keys()}, mx)
    embed_mod, norm_mod, head_mod = profile.build_static(margs, mx, nn)
    embed_key = next(k for k in static_raw if "embed" in k)
    norm_key = next(k for k in static_raw if "norm" in k and "embed" not in k)
    head_key = next(k for k in static_raw if "lm_head" in k)
    embed_mod.load_weights([("weight", static_raw[embed_key])], strict=False)
    norm_mod.load_weights([("weight", static_raw[norm_key])], strict=False)
    head_mod.load_weights([("weight", static_raw[head_key])], strict=False)
    del static_raw
    mx.eval(embed_mod.parameters(), norm_mod.parameters(), head_mod.parameters())
    print("[init] static modules loaded", flush=True)

    # ---- calibration + vocabulary gate ----------------------------------
    calib = np.load(args.calibration)
    n_rows = min(args.max_chunks, calib.shape[0])
    ids = mx.array(calib[:n_rows].astype(np.int32))
    print(f"[calib] {n_rows} rows x {calib.shape[1]}", flush=True)
    if not args.allow_foreign_calibration:
        from rco_mlx.calibration import verify_calibration

        rep = verify_calibration(src, Path(args.calibration))
        print(rep.summary(), flush=True)
        if not rep.ok:
            sys.exit("calibration cache does not match this model's tokenizer. "
                     "Rebuild it (scripts/build_calibration_for_model.py) or pass "
                     "--allow-foreign-calibration.")

    # ---- accumulation ----------------------------------------------------
    in_sum2: dict = {}
    counts: dict = {}
    # Experts never routed to -- tracked separately because the layer-pooled
    # fallback writes a substitute count that would hide the gap.
    zero_experts: set = set()

    def to_np(a):
        try:
            return np.asarray(a)
        except ValueError:
            return np.asarray(a.astype(mx.float32))

    def accumulate_sums(key, s, n):
        if key in in_sum2 and in_sum2[key].shape == s.shape:
            in_sum2[key] += s
        elif key not in in_sum2:
            in_sum2[key] = s.copy()
        counts[key] = counts.get(key, 0.0) + n

    hook_by_id: dict = {}
    patched: dict = {}

    def linear_call(self, x, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            x_np = to_np(x)
            rows = x_np.reshape(-1, x_np.shape[-1]).astype(np.float64)
            accumulate_sums(profile.mlx_to_hf(info[1]),
                            (rows * rows).sum(axis=0), rows.shape[0])
        return patched[type(self)](self, x, *a, **kw)

    def switch_call(self, x, indices=None, *a, **kw):
        info = hook_by_id.get(id(self))
        if info is not None:
            x_np = to_np(x)
            idx_np = to_np(indices) if indices is not None else None
            if idx_np is not None:
                x2 = x_np.reshape(-1, x_np.shape[-1])
                k = idx_np.shape[1] if idx_np.ndim == 2 else 1
                idx_flat = idx_np.reshape(-1).astype(np.int64)
                x_per_slot = np.repeat(x2, k, axis=0).astype(np.float64)
                sq = x_per_slot * x_per_slot
                n_exp = info[2]
                cnt = np.bincount(idx_flat, minlength=n_exp)
                nz = np.nonzero(cnt)[0]
                sums_by_e = {}
                if nz.size:
                    starts = np.concatenate(([0], np.cumsum(cnt)))
                    sums = np.add.reduceat(
                        sq[np.argsort(idx_flat, kind="stable")], starts[nz], axis=0)
                    for j, e in enumerate(nz):
                        sums_by_e[int(e)] = sums[j]
                # Experts the calibration never activates get the layer-pooled
                # statistic: every routed expert sees the same hidden state, so
                # that is the unbiased estimate (and beats dropping the key).
                pooled = sq.sum(axis=0)
                n_slots = int(sq.shape[0])
                n_zero = 0
                for e in range(n_exp):
                    s = sums_by_e.get(e)
                    if s is None:
                        accumulate_sums(profile.mlx_to_hf(info[1], e), pooled, n_slots)
                        zero_experts.add(profile.mlx_to_hf(info[1], e))
                        n_zero += 1
                    else:
                        accumulate_sums(profile.mlx_to_hf(info[1], e), s, int(cnt[e]))
                if n_zero:
                    print(f"    [imatrix] {info[1]}: {n_zero}/{n_exp} experts never "
                          f"activated -> layer-pooled fallback", flush=True)
            else:
                rows = x_np.reshape(-1, x_np.shape[-1]).astype(np.float64)
                accumulate_sums(profile.mlx_to_hf(info[1]),
                                (rows * rows).sum(axis=0), rows.shape[0])
        return patched[type(self)](self, x, indices, *a, **kw) \
            if indices is not None else patched[type(self)](self, x, *a, **kw)

    def register(module, prefix):
        for path, m in module.named_modules():
            kind = {"Linear": "linear", "SwitchLinear": "switch"}.get(type(m).__name__)
            if kind is None:
                continue
            full = f"{prefix}.{path}" if path else prefix
            hook_by_id[id(m)] = (kind, full) + (
                (int(m.weight.shape[0]),) if kind == "switch" else ())
            cls = type(m)
            if cls not in patched:
                patched[cls] = cls.__call__
                cls.__call__ = linear_call if kind == "linear" else switch_call

    register(head_mod, "lm_head")
    print(f"[hook] static registered ({len(hook_by_id)})", flush=True)

    H = [embed_mod(ids[i:i + 1]) for i in range(n_rows)]
    mx.eval(*H)

    # ---- one layer resident at a time ------------------------------------
    for i in range(n_layers):
        raw = load_raw(src, wm, {
            k: s for k, s in wm.items() if k.startswith(profile.layer_key_prefix(i))
        }, mx)
        mapped = profile.map_layer(raw, i)
        del raw
        block = profile.build_layer(margs, i)
        block.load_weights(list(mapped.items()), strict=False)
        del mapped
        mx.eval(block.parameters())
        register(block, profile.module_prefix(i))

        new_H = []
        for h in H:
            mask, cache = profile.mask_and_cache(margs, i, h, mx)
            new_H.append(block(h, mask=mask, cache=cache))
        mx.eval(*new_H)
        H = new_H
        print(f"[layer {i:3d}] done | hooked={len(hook_by_id)}", flush=True)
        cov = expert_coverage_stats(
            counts, zero_experts,
            min_required_count=args.min_expert_count,
            required_percentile=args.min_expert_percentile,
        )
        if cov.total_experts:
            print(f"    coverage: {cov.active_experts}/{cov.total_experts} experts "
                  f"active | never-activated {cov.zero_count_experts} | "
                  f"p{args.min_expert_percentile:02d} {cov.p05_count:.1f} "
                  f"(need >= {args.min_expert_count})", flush=True)
        del block
        mx.clear_cache()

    H = [norm_mod(h) for h in H]
    mx.eval(*H)
    for h in H:
        head_mod(h)
    mx.eval()

    print(f"[done] {len(in_sum2)} imatrix tensors", flush=True)
    cov = expert_coverage_stats(
        counts, zero_experts,
        min_required_count=args.min_expert_count,
        required_percentile=args.min_expert_percentile,
    )
    print(cov.summary(), flush=True)
    if not cov.ok and not args.allow_insufficient_coverage:
        sys.exit(
            "expert coverage insufficient -- the imatrix for unrouted experts is "
            "a substitute, not a measurement. Re-run with more --max-chunks, or "
            "pass --allow-insufficient-coverage to override."
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in in_sum2.items():
        payload[f"{k}.in_sum2"] = v.astype(np.float32)
        payload[f"{k}.counts"] = np.array([counts[k]], dtype=np.float32)
    np.savez_compressed(str(out), **payload)
    print(f"saved -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    return len(in_sum2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model-type", default=None,
                    help="profile key; inferred from config.json when omitted")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--max-layer", type=int, default=None,
                    help="only run the first N layers (smoke test)")
    ap.add_argument("--allow-foreign-calibration", action="store_true")
    ap.add_argument("--min-expert-count", type=int, default=MIN_EXPERT_COUNT,
                    help="p05 tokens/expert required (oMLX uses 16)")
    ap.add_argument("--min-expert-percentile", type=int,
                    default=MIN_EXPERT_COUNT_PERCENTILE)
    ap.add_argument("--allow-insufficient-coverage", action="store_true",
                    help="write the imatrix even if the coverage gate fails")
    args = ap.parse_args()

    src = Path(args.source)
    cfg = json.loads((src / "config.json").read_text())
    model_type = args.model_type or cfg.get("model_type")
    if model_type not in PROFILES:
        sys.exit(f"no capture profile for model_type {model_type!r}; available: "
                 f"{sorted({p.name for p in PROFILES.values()})}")
    capture(PROFILES[model_type], src, args)


if __name__ == "__main__":
    main()
