"""Calibration-cache provenance and vocabulary validation.

A calibration ``.npy`` is only meaningful for the checkpoint whose tokenizer
produced its ids.  Reusing one built for a different model silently measures
off-vocabulary activations -- the imatrix and the RCO error DB then describe a
model that does not exist.  This module makes that failure loud.

The discriminator is intrinsic to the tokenizer and needs no dataset download:
``encode(decode(ids)) == ids`` when the ids came from this tokenizer, and
essentially never otherwise (measured on Bolt: 1.000 for a cache built with
Bolt's tokenizer, 0.003 for one built with a foreign vocabulary).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

SIDECAR_SUFFIX = ".meta.json"

# Expert-coverage gate, mirroring oMLX's `_OQE_MIN_EXPERT_COUNT` /
# `_OQE_MIN_EXPERT_COUNT_PERCENTILE`: a calibration is only good enough when
# every routed expert was actually exercised, and the 5th-percentile expert
# saw at least this many tokens.
MIN_EXPERT_COUNT = 16
MIN_EXPERT_COUNT_PERCENTILE = 5

_EXPERT_KEY_RE = re.compile(
    # Captures key styles differ: Bolt/LFM2 append `.weight`, the Qwen
    # profile does not.  Without allowing it, no expert key matches and
    # the coverage gate silently sees zero expert groups.
    r"^(?P<base>.+)\.experts\.\d+\.(?P<proj>[A-Za-z0-9_]+(?:\.weight)?)$"
)


@dataclass
class CalibrationReport:
    path: Path
    model: str
    vocab_size: int
    tokenizer_sha256: str
    rows: int
    seq_len: int
    roundtrip_match: float
    frac_lt_1000: float
    mean_id: float
    sidecar_match: Optional[bool] = None
    warnings: list[str] = field(default_factory=list)

    # Measured bands for `roundtrip_match`:
    #   ~1.00  this tokenizer, plain prose
    #   ~0.87  this tokenizer, prose+code (code normalises imperfectly)
    #   ~0.01  a FOREIGN tokenizer -- the failure this gate exists to catch
    # 0.5 separates a wrong vocabulary from any real corpus.
    ROUNDTRIP_MIN = 0.5

    @property
    def ok(self) -> bool:
        return self.roundtrip_match >= self.ROUNDTRIP_MIN and not self.warnings

    def summary(self) -> str:
        verdict = "OK" if self.ok else "MISMATCH"
        lines = [
            f"[calib] {self.path.name}: {verdict}",
            f"  model {self.model} | vocab {self.vocab_size} | "
            f"tokenizer sha256 {self.tokenizer_sha256[:16]}",
            f"  rows {self.rows} x {self.seq_len} | roundtrip match "
            f"{self.roundtrip_match:.4f} | frac<1000 {self.frac_lt_1000:.4f} | "
            f"mean id {self.mean_id:.0f}",
        ]
        if self.sidecar_match is False:
            lines.append("  sidecar: DOES NOT MATCH this model")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        if not self.ok:
            lines.append(
                "  This cache was not tokenized with the model's own tokenizer. "
                "Rebuild it (scripts/build_calibration_for_model.py) -- using it "
                "measures off-vocabulary activations."
            )
        return "\n".join(lines)


def tokenizer_fingerprint(source_dir: Path) -> tuple[str, int, object]:
    """(sha256[:16], vocab_size, tokenizer) for a checkpoint's tokenizer."""
    from tokenizers import Tokenizer

    path = Path(source_dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; cannot verify a calibration cache without the "
            f"model's own tokenizer"
        )
    raw = path.read_bytes()
    tk = Tokenizer.from_file(str(path))
    return hashlib.sha256(raw).hexdigest()[:16], tk.get_vocab_size(), tk


def verify_calibration(
    source_dir: Path | str,
    npy_path: Path | str,
    sample_rows: int = 4,
) -> CalibrationReport:
    """Check that ``npy_path`` is tokenized text in this model's vocabulary."""
    source_dir = Path(source_dir)
    npy_path = Path(npy_path)
    sha, vocab, tk = tokenizer_fingerprint(source_dir)

    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{npy_path} has shape {arr.shape}, expected (rows, seq_len)")
    n_rows, seq_len = arr.shape
    ids = np.asarray(arr[: min(sample_rows, n_rows)]).astype(np.int64)

    # Round-trip: decode then re-encode.  Exact only for the producing tokenizer.
    match = []
    for row in ids:
        row = [int(x) for x in row if 0 <= int(x) < vocab]
        dec = tk.decode(row)
        re = tk.encode(dec).ids
        n = min(len(row), len(re))
        if n:
            match.append(sum(1 for a, b in zip(row[:n], re[:n]) if a == b) / n)
    roundtrip = float(np.mean(match)) if match else 0.0

    flat = ids.reshape(-1)
    frac_low = float((flat < 1000).mean())
    mean_id = float(flat.mean())

    warnings: list[str] = []
    if roundtrip < CalibrationReport.ROUNDTRIP_MIN:
        warnings.append(
            f"roundtrip match {roundtrip:.4f} < "
            f"{CalibrationReport.ROUNDTRIP_MIN} -- ids are not from this "
            f"tokenizer"
        )
    # NOTE: a low `frac_lt_1000` is NOT evidence of a foreign vocabulary.
    # It is tokenizer-dependent -- LFM2.5 legitimately produces ~48% ids
    # below 1000, Bolt ~0%.  The round-trip above is the intrinsic test.

    sidecar_match: Optional[bool] = None
    sidecar = Path(str(npy_path) + SIDECAR_SUFFIX)
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        sidecar_match = (
            meta.get("tokenizer_sha256", "")[:16] == sha
            and int(meta.get("vocab_size", -1)) == vocab
        )
        if not sidecar_match:
            warnings.append("sidecar names a different tokenizer/vocab")

    return CalibrationReport(
        path=npy_path,
        model=source_dir.name,
        vocab_size=vocab,
        tokenizer_sha256=sha,
        rows=int(n_rows),
        seq_len=int(seq_len),
        roundtrip_match=roundtrip,
        frac_lt_1000=frac_low,
        mean_id=mean_id,
        sidecar_match=sidecar_match,
        warnings=warnings,
    )


def write_sidecar(
    npy_path: Path | str,
    source_dir: Path | str,
    dataset: str,
    extra: Optional[dict] = None,
) -> Path:
    """Record how a calibration cache was produced, so reuse can be detected."""
    source_dir = Path(source_dir)
    sha, vocab, _ = tokenizer_fingerprint(source_dir)
    arr = np.load(npy_path, mmap_mode="r")
    meta = {
        "model": source_dir.name,
        "model_path": str(source_dir),
        "dataset": dataset,
        "rows": int(arr.shape[0]),
        "seq_len": int(arr.shape[1]),
        "tokenizer_sha256": sha,
        "vocab_size": vocab,
    }
    if extra:
        meta.update(extra)
    out = Path(str(npy_path) + SIDECAR_SUFFIX)
    out.write_text(json.dumps(meta, indent=2))
    return out


# ---------------------------------------------------------------------------
# Expert-coverage gate
# ---------------------------------------------------------------------------


def expert_keys(counts: dict) -> set:
    """Imatrix keys that belong to routed experts (``.experts.<id>.<proj>``)."""
    return {k for k in counts if _EXPERT_KEY_RE.match(k)}


@dataclass
class CoverageReport:
    """Did the calibration actually exercise every routed expert?

    ``counts`` is the imatrix ``name -> accumulated rows`` mapping.
    ``activated`` is the set of expert keys that routed at least one token in
    at least one chunk.  An expert is "never activated" only if it never fired
    across the *whole* capture -- tracking per-chunk misses instead would flag
    any expert absent from a single chunk, which is normal routing noise.
    """

    groups: int = 0
    total_experts: int = 0
    active_experts: int = 0
    zero_count_experts: int = 0
    min_count: float = 0.0
    p05_count: float = 0.0
    median_count: float = 0.0
    max_count: float = 0.0
    active_ratio: float = 1.0
    worst_groups: list = field(default_factory=list)
    min_required_count: int = MIN_EXPERT_COUNT
    required_percentile: int = MIN_EXPERT_COUNT_PERCENTILE

    @property
    def ok(self) -> bool:
        return (
            self.zero_count_experts == 0
            and self.p05_count >= self.min_required_count
        )

    def summary(self) -> str:
        verdict = "OK" if self.ok else "INSUFFICIENT"
        lines = [
            f"[coverage] {verdict}",
            f"  expert groups (layer x projection): {self.groups}",
            f"  experts: {self.total_experts} total, {self.active_experts} active "
            f"({self.active_ratio:.1%})",
            f"  never-activated experts: {self.zero_count_experts}",
            f"  tokens/expert: min {self.min_count:.0f} | "
            f"p{self.required_percentile:02d} {self.p05_count:.1f} | "
            f"median {self.median_count:.0f} | max {self.max_count:.0f}",
            f"  required: 0 never-activated AND "
            f"p{self.required_percentile:02d} >= {self.min_required_count}",
        ]
        for g in self.worst_groups[:5]:
            lines.append(f"  {g}")
        if not self.ok:
            lines.append(
                "  Calibration did not exercise the router fully. Re-run with "
                "more --max-chunks (coverage scales with token count); the "
                "imatrix for unrouted experts is a substitute, not a measurement."
            )
        return "\n".join(lines)


def expert_coverage_stats(
    counts: dict,
    activated: set | None = None,
    min_required_count: int = MIN_EXPERT_COUNT,
    required_percentile: int = MIN_EXPERT_COUNT_PERCENTILE,
) -> CoverageReport:
    """Summarise routed-expert coverage from imatrix counts.

    ``counts`` maps imatrix tensor name -> accumulated rows.  Expert tensors are
    recognised by ``.experts.<id>.<proj>``; the group is the SwitchLinear they
    belong to (everything up to ``.experts.`` plus the projection).

    ``activated`` lists expert keys that routed at least one token; experts
    absent from it are counted as never activated (their ``counts`` entry is the
    layer-pooled substitute, not a measurement).
    """
    keys = expert_keys(counts)
    activated = activated if activated is not None else keys
    zero_keys = keys - activated
    groups: dict = {}
    for key in keys:
        m = _EXPERT_KEY_RE.match(key)
        grp = f"{m.group('base')}.{m.group('proj')}"
        groups.setdefault(grp, []).append(0.0 if key in zero_keys else float(counts[key]))

    if not groups:
        return CoverageReport(min_required_count=min_required_count,
                              required_percentile=required_percentile)

    all_counts = np.concatenate([np.asarray(v, dtype=np.float64)
                                 for v in groups.values()])
    zero = int((all_counts <= 0).sum())
    active = int((all_counts > 0).sum())

    worst = []
    for grp, vals in groups.items():
        arr = np.asarray(vals, dtype=np.float64)
        z = int((arr <= 0).sum())
        if z:
            worst.append(f"{grp}: {z}/{arr.size} experts never activated")
    worst.sort(key=lambda s: -int(s.split(":")[1].strip().split("/")[0]))

    return CoverageReport(
        groups=len(groups),
        total_experts=int(all_counts.size),
        active_experts=active,
        zero_count_experts=zero,
        min_count=float(all_counts.min()) if all_counts.size else 0.0,
        p05_count=float(np.percentile(all_counts, required_percentile))
        if all_counts.size else 0.0,
        median_count=float(np.median(all_counts)) if all_counts.size else 0.0,
        max_count=float(all_counts.max()) if all_counts.size else 0.0,
        active_ratio=active / all_counts.size if all_counts.size else 1.0,
        worst_groups=worst,
        min_required_count=min_required_count,
        required_percentile=required_percentile,
    )
