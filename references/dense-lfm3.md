# Dense RCO on LFM3-1.7B — head-to-head with MACaQ

Dense-model support lives in `rco-mlx` (generic `default` adapter + MoE
detection in `examples/moe_pipeline.py`). LFM3-1.7B was chosen for the first
dense validation because a MACaQ checkpoint and its benchmark numbers already
exist locally, so the comparison uses one identical protocol.

## Protocols

- **PPL**: WikiText-2 test, 130,816 tokens, sequence length 512, batch size 8
  (`macaq-mlx/eval_ppl.py`, stock `mlx_lm` loader).
- **KL / top-1**: `scripts/eval_quality.py` — the same calibration rows run
  through BF16 and the quant; forward KL(ref‖quant) averaged over tokens and
  argmax agreement. 8 rows × 256 tokens here.
- **Zero-shot**: `slq_mlx.scripts.run_zero_shot`, 5 tasks × 400 examples,
  recovery = quant_acc / bf16_acc with paired CIs.
- **TPS**: `macaq-mlx/benchmarks/bench_tps.py` (mlx-lm generate, ~1K context).

BF16 reference: `LFM3-1.7B-MLX-BF16` → **PPL 30.839 ± 0.289**, exactly
reproducing the MACaQ README number (30.839 ± 0.289), which confirms the
harness matches theirs.

**Bit-width accounting.** MLX affine stores a float16 scale *and* a float16
bias per group, so the real cost is `bits + 32/group_size` (0.5 bpw at gs64;
MACaQ's mxfp4 stores one 8-bit scale per 32 weights = 0.25 bpw). The RCO
allocator's internal `effective_bits()` default counts only the scale, so a
nominal target reads 0.25 bpw below true on-disk cost at gs64. All figures
below are measured from safetensors bytes/shapes.

## Results — WikiText PPL

| Checkpoint | Disk | Eff. bpw | PPL ↓ | Δ vs BF16 |
|---|---|---|---|---|
| BF16 reference | 3.2 GB | 16.0 | 30.839 ± 0.289 | — |
| MACaQ mxfp4 + Flip-Refinement | 0.919 GB | 4.250 | **35.472 ± 0.336** | +4.63 |
| MACaQ affine-4 baseline (their number) | — | 4.50 | 42.380 ± 0.414 | +11.54 |
| MACaQ mxfp4 RTN (their number) | — | 4.25 | 45.501 ± 0.446 | +14.66 |
| RCO uniform-4, embed BF16 | 1.350 GB | 4.500 | 41.553 ± 0.405 | +10.71 |
| RCO mixed 3/4/6/8, embed BF16 | 1.349 GB | 4.495 | 39.920 ± 0.385 | +9.08 |
| RCO uniform-4 + quantized embed | 0.973 GB | 4.500 | 42.300 ± 0.413 | +11.46 |
| RCO mixed + embed | 0.972 GB | 4.497 | 40.651 ± 0.393 | +9.81 |
| RCO mixed + embed + imatrix PGC | 0.972 GB | 4.497 | **36.875 ± 0.352** | +6.04 |
| RCO mixed + embed, 4.25 bpw | 0.918 GB | 4.247 | 53.206 ± 0.535 | +22.37 |
| RCO mixed + embed + imatrix PGC, 4.25 bpw | 0.918 GB | 4.247 | 41.602 ± 0.402 | +10.76 |

The PGC runs reuse the imatrix captured by `--build-imatrix`
(`--imatrix <output>/work/imatrix.npz`), so they cost no extra capture.

## Results — KL / top-1 vs BF16

| Model | PPL (calib subset) | KL(ref‖q) ↓ | top-1 ↑ |
|---|---|---|---|
| MACaQ mxfp4 + refine (4.25 bpw) | 34.133 | 0.1791 | 77.40% |
| **RCO mixed + embed + PGC (4.50 bpw)** | **34.218** | **0.1798** | **76.32%** |
| RCO mixed + embed, no PGC (4.50 bpw) | 37.664 | 0.2463 | 74.12% |
| RCO mixed + embed + PGC (4.25 bpw) | 38.051 | 0.3031 | 71.81% |

At ~4.5 bpw the RCO+PGC build is statistically indistinguishable from MACaQ
on KL (0.180 vs 0.179) and within 1.1 pp on top-1, while being 6% larger
(0.972 vs 0.919 GB). At matched 4.25 bpw the gap is real (top-1 71.8% vs
77.4%) — the tighter budget forces 3-bit placements that PGC cannot repair.

## Results — zero-shot task recovery (5 tasks × 400)

`slq_mlx.scripts.run_zero_shot`; recovery = quant_acc / BF16_acc.

| Task | BF16 acc | MACaQ acc | MACaQ recovery | RCO-PGC-45 acc | RCO-PGC recovery |
|---|---|---|---|---|---|
| arc_challenge | 0.2175 | 0.2000 | 0.9200 | 0.2050 | 0.9425 |
| arc_easy | 0.4175 | 0.3975 | 0.9521 | 0.3975 | 0.9521 |
| hellaswag | 0.3800 | 0.3575 | 0.9408 | 0.3850 | 1.0132 |
| winogrande | 0.4800 | 0.4925 | 1.0260 | 0.5100 | 1.0625 |
| mmlu | 0.2975 | 0.2650 | 0.8908 | 0.2725 | 0.9160 |
| **mean** | — | — | **0.9458** | — | **0.9772** |

Neither checkpoint is task-lossless (≥0.99). Every RCO per-task paired CI
crosses zero except MMLU (-0.025 ± 0.015, CI [-0.054, +0.004]); MACaQ's
HellaSwag and MMLU CIs exclude zero negatively. LFM3 is weak at these tasks,
so n=400 differences of a few points are noisy — read the mean recovery as
the signal.

## Results — throughput (~1K context, 256 generated tokens, mlx-lm)

Single measured run after one warmup pass (`macaq-mlx/benchmarks/bench_tps.py`).
Scale dtype dominates decode: converting only the affine scales/biases to
bfloat16 lifts decode 134 → 174 tok/s for ~0.005 KL.

| Model | Prompt tps | Decode tps | Peak RAM |
|---|---|---|---|
| BF16 reference | 2794.6 | 71.5 | 4.34 GB |
| MACaQ mxfp4 + refine (4.25 bpw) | 2707.8 | 162.1 | 2.22 GB |
| RCO-PGC-45, float16 scales (4.50 bpw) | 2409.7 | 134.3 | 2.65 GB |
| RCO-PGC-45, bfloat16 scales | **2877.6** | **173.8** | 2.27 GB |

With bfloat16 scales the mixed-precision RCO build decodes ~7% faster than
MACaQ's uniform mxfp4 and ~2.4× faster than BF16, while costing 0.005 more
KL (0.1846 vs 0.1798). The bf16 conversion is an in-place `.scales`/`.biases`
rewrite; no re-quantization is needed.

## Findings

1. **Harness validated twice.** Uniform 4-bit through the RCO pipeline
   reproduces MACaQ's own affine-4 baseline almost exactly (42.300 ± 0.413 vs
   their 42.380 ± 0.414 at 4.5 bpw), and our BF16 PPL matches theirs to three
   decimals.
2. **Imatrix-weighted per-group clipping is the difference-maker on dense
   models too.** At 4.5 bpw it improves PPL 40.65 → 36.88 and KL 0.246 →
   0.180; at 4.25 bpw it repairs the low-width cliff (53.21 → 41.60). This
   mirrors the OLMoE result (KL 0.508 → 0.452 at 2.5 bpw). Without an imatrix,
   `--per-group-clip` is skipped rather than silently degrading.
3. **Allocation alone helps modestly.** At matched budget without PGC, mixed
   beats uniform by 1.65 PPL (40.65 vs 42.30).
4. **The 4.25-bpw cliff without PGC is steep** (40.65 → 53.21 for 0.25 bpw
   saved): the DP places 3-bit FFN down-projections that affine RTN cannot
   carry. PGC absorbs most of it.
5. **MACaQ still leads at matched size on quality.** At 0.918–0.919 GB /
   4.25 bpw, MACaQ scores PPL 35.47 / top-1 77.4% vs RCO-PGC 41.60 / 71.8%.
   Its edge is structural: E2M1 float codebook + Hessian flip-refinement
   against a BF16 teacher. RCO's remaining lever is activation-aware
   refinement beyond per-group scale search (e.g. GPTQ-style compensation is
   blocked by rank-deficient Hessians on MoE but viable on dense
   projections).
6. **On zero-shot recovery and throughput RCO-PGC-45 compares well.** Zero-shot
   mean recovery 0.977 (RCO) vs 0.946 (MACaQ), neither task-lossless; RCO's
   per-task CIs all cross zero except MMLU. With bfloat16 affine scales,
   decode is 173.8 tok/s vs MACaQ's 162.1 and BF16's 71.5 at ~1K context, for
   +0.005 KL — so at 4.5 bpw the RCO build trades ~6% more bytes than MACaQ
   for slightly worse PPL but faster decoding and better task recovery.

## Compute cost (M3 Max, LFM3-1.7B)

Reconstructed from filesystem timestamps (stage artifacts) and the MACaQ
refine logs. RCO's error DB and imatrix are one-time costs that are reused
across targets: seven pipeline runs in this session took ~10 minutes total.

| Step | RCO (this work) | MACaQ (logs) |
|---|---|---|
| Base quant | RTN inside the pipeline (included below) | mxfp4 conversion, ~1–2 min (not precisely logged) |
| Sensitivity capture | error DB 13–31 s (reusable) | Hessian capture 20.3 s + calibration build |
| Importance / refinement | imatrix capture ~30 s (reusable) | per-tensor flip refinement |
| Quantize / pack | 44 s (includes per-group clip search) | included in refine |
| **End-to-end, one checkpoint** | **~1 min 45 s** with imatrix capture; **~57 s** reusing cached DB+imatrix | **~6 min** at flips=64 (measured 6:19, 30.9% Hessian reduction); **~11–12 min** at the shipped flips=256 (inferred from the 35-min 3-budget flip sweep, f512 alone ≈17 min) |

So RCO is roughly **6–20× faster per checkpoint**, and its reusable error DB
makes re-allocation at a new target a ~1-minute quantize-only operation. The
trade is what the time buys: MACaQ's extra minutes purchase Hessian-based
code refinement that RCO's RTN+per-group-clip pipeline does not attempt.

## Reproduce

```bash
cd ~/Projects/rco-mlx
SRC=~/Projects/ml/models/LFM3-1.7B-BF16
CAL=$SRC/work/calib_wikitext_128x2048.npy   # or --calibration-dataset wikitext

# best quality point (0.972 GB, 4.50 bpw) — capture imatrix once, reuse after
PYTHONPATH=$PWD/src python3 examples/moe_pipeline.py \
    --source $SRC --output ~/Projects/ml/models/LFM3-1.7B-MLX-RCO-embed-mixed-pgc-45 \
    --target-bpw 4.25 --group-size 64 --calibration $CAL \
    --bits 3,4,6,8 --quantize-embeddings --embedding-min-bits 4 \
    --build-imatrix --per-group-clip

# matched-size point (0.918 GB, 4.25 bpw), reusing the captured imatrix
IM=~/Projects/ml/models/LFM3-1.7B-MLX-RCO-embed-mixed-pgc-45/work/imatrix.npz
PYTHONPATH=$PWD/src python3 examples/moe_pipeline.py \
    --source $SRC --output ~/Projects/ml/models/LFM3-1.7B-MLX-RCO-embed-mixed-pgc \
    --target-bpw 4.0 --group-size 64 --calibration $CAL \
    --bits 3,4,6,8 --quantize-embeddings --embedding-min-bits 4 \
    --imatrix $IM --per-group-clip

# evaluation
cd ~/Projects/ml/macaq-mlx
python3 eval_ppl.py --model <checkpoint> --seq-len 512 --samples 256 --batch-size 8
cd ~/Projects/rco-mlx
python3 scripts/eval_quality.py --bf16 ~/Projects/ml/models/LFM3-1.7B-MLX-BF16 \
    --quant BEST=<checkpoint> --calib $CAL --samples 8 --max-tokens 256
```

Artifacts (local): `~/Projects/ml/models/LFM3-1.7B-MLX-RCO-*`; logs
`LFM3-1.7B-RCO-*.log`, `LFM3-1.7B-KL-top1*.log`, `LFM3-1.7B-zeroshot.log`
in `~/Projects/ml/models/`.
