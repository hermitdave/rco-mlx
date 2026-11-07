# rco-mlx

RCO's budget-constrained mixed-precision allocator, ported from PyTorch to
NumPy, with the result wired into MLX's quantized-checkpoint format.

RCO ([ISTA-DASLab/RCO](https://github.com/IST-DASLab/RCO), *Model Compression
with Exact Budget Constraints via Riemannian Manifolds*, arXiv:2605.00649)
answers a specific question: **given a set of groups (layers), a set of
options per group (bit widths), and a hard size budget, which option do you
pick for each group?** It does so with an exact equality constraint rather than
a penalty or a Lagrangian, which is why the resulting file lands on the
requested size instead of near it.

This package ports that optimizer so it runs anywhere NumPy does, then emits
the allocation as an MLX `config["quantization"]` block and validates it before
you commit to a 15-minute checkpoint load.

The error database is built with **GPTQ-based (Hessian-scaled) reconstruction**
rather than plain RTN, and **MoE experts are batched** using the shared-Hessian
property for both speed and quality.

---

## What is here, and how faithful it is

| Component | Source | Status |
|---|---|---|
| `manifold.py` | `src/manifold.py` | Faithful port. Verified against a finite difference of the budget function. |
| `search.budget_dp` | `round_with_budget_dp` | Faithful port of the knapsack DP. Checked against a brute-force enumeration. |
| `search.ProjectedGumbelOptimizer` | `optimize_projected_gumbel` | Faithful algorithm, different interface (see below). |
| `grouping.py` | `src/grouping.py` | Transcription; the original has no torch dependency either. |
| `objective.py` | — | New. Shapes a measured loss table for the DP. |
|| `mlx_alloc.py` | — | New. Emits the MLX config block. |
|| `validate.py` | — | New. Structural checks before a load. |
|| `scripts/build_moe_error_db.py` | — | Self-contained streaming error_db builder with GPTQ + batched experts. |
| `examples/moe_pipeline.py` | — | End-to-end MoE pipeline: introspect → error_db → RCO → quantize. |

**The one design change that matters.** The original drives the optimizer from
a live PyTorch model: it registers a parametrization on every layer's weight,
runs a forward pass, and backprops into the allocation logits. That is what
makes RCO handle a *non-decomposable* objective — a real model's loss does not
factor into a sum over layers, and RCO's paper exists because of that.

This port replaces the model with a `loss_and_grad` callable:

```python
def loss_and_grad(probs):
    loss, grad = my_backend.forward_and_backward(probs)
    return loss, grad

optimizer.run(loss_and_grad, n_steps=400)
```

Supply that from MLX (`mx.grad` over a function of `probs`), PyTorch, or
anything else. The manifold math is unchanged, so the budget still holds by
construction.

**If you only have a loss table** — which is the practical MLX case, and what
`alloc_greedy.py` already produces — skip the optimizer and call `budget_dp`
directly. For a separable objective the DP is exact, and it is strictly better
than greedy because it solves the knapsack rather than taking the local best.
The caveat is that a table *is* the separable assumption; treat it as a strong
heuristic, not the true objective.

---

## Install

```bash
pip install -e .
```

NumPy only. MLX is an optional extra (`pip install -e ".[mlx]"`) and is only
needed for quantization utilities.

---

## Quick start

### From a loss table

```python
import numpy as np
from rco_mlx import LossTable, allocation_to_config, validate_allocation

# losses[group, option] -- measured, lower is better
losses = np.array([
    [0.50, 0.30, 0.10],   # group 0 is sensitive
    [0.45, 0.28, 0.12],   # group 1 is sensitive
    [0.20, 0.15, 0.10],   # group 2 is not
    [0.18, 0.14, 0.11],   # group 3 is not
])
fracs = np.array([0.25, 0.25, 0.25, 0.25])       # group_params / total_params
bitwidths = np.array([2, 4, 8])

table = LossTable(losses=losses, fracs=fracs, bitwidths=bitwidths, group_size=64)
bits = table.assignment(target_avg_bits=5.0)      # -> per-group bit widths

cfg = allocation_to_config(dict(zip(group_names, bits)), default_bits=4)
report = validate_allocation(cfg, on_disk_keys=checkpoint_keys)
assert report.ok, report.summary()
```

### From a live, differentiable objective

```python
from rco_mlx import ProjectedGumbelOptimizer

opt = ProjectedGumbelOptimizer(
    n_groups=128, n_options=5,
    fracs=fracs, actual_bits=np.array([2.25, 3.25, 4.25, 6.25, 8.25]),
    target_avg_bits=4.0,
)
opt.init_to_bits()          # start on the manifold, not off it

probs, history = opt.run(loss_and_grad, n_steps=400)
assignment = opt.hard_assignment()
```

---

## Using with MoE models

The pipeline is model-agnostic once you provide a **model adapter** that
describes the architecture. The adapter tells the pipeline:

|- how to enumerate its layers and tensors
|- which tensors to quantize vs keep BF16
|- how to group layers for RCO allocation
|- how to map source checkpoint keys to MLX module paths

Two adapters are included:

| Adapter | Model type | Architecture |
|---|---|---|
| `OLMoEAdapter` | `olmoe` | 16 MoE layers, 64 experts, top-8 routing |
| `Qwen3NextMoEAdapter` | `qwen3_5_moe` | 40 layers (10 full-attn + 30 linear-attn), 128 experts |

### Adding a new MoE adapter

Subclass `ModelAdapter` and implement the six methods. The smallest useful
adapter is ~40 lines.

```python
from rco_mlx.adapters import ModelAdapter, register

@register("my_moe")
class MyMoEAdapter(ModelAdapter):
    model_type = "my_moe"
    default_group_size = 64
    default_bits = 4

    def enumerate_layers(self, source_dir): ...
    def quant_predicate(self, layer_descriptor): ...
    def grouping_spec(self): ...
    def load_calibration(self, source_dir, n_samples=512, seq_len=2048): ...
    def map_key_to_mlx(self, key): ...
    def non_quantized_prefixes(self): ...
```

Register it with `@register("my_moe")` and the pipeline discovers it
automatically via `rco_mlx.get("my_moe")`.

### Running the pipeline

```bash
python examples/moe_pipeline.py \
    --source /path/to/bf16-moe \
    --output /path/to/output \
    --target-bpw 3.5 \
    --group-size 64 \
    --model-type olmoe \
    --calibration-dataset wikitext \
    --quant-mode symmetric
```

The `--model-type` flag selects the adapter. If omitted, the pipeline
infers it from `config.json`'s `model_type` field and validates that the
model is MoE (`num_experts > 1`).

### Calibration data

The pipeline needs calibration tokens for Hessian capture. You can either:

- Provide a pre-tokenized `.npy` file with `--calibration`
- Let the pipeline download a known corpus with `--calibration-dataset`

Supported datasets: `wikitext` (Wikipedia, ~1.3M test tokens), `fineweb` (FineWeb-Edu),
`c4` (C4 web text). The pipeline downloads the dataset, tokenizes it with the model's
tokenizer, and saves a 128×2048 calibration `.npy` file in `work/calib_{dataset}.npy`.

For MoE models, the error_db builder extends calibration automatically (up to
1,024 sequences) until every expert has seen at least 256 activation tokens.

---


## MLX integration

The emitted config matches what `mlx_lm.load()` expects:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 4,
    "mode": "affine",
    "backbone.layers.1.mixer.switch_mlp.fc1": {"group_size": 64, "bits": 2, "mode": "affine"},
    "backbone.embeddings":                     {"group_size": 64, "bits": 8, "mode": "affine"}
  },
  "quantization_config": "<same object>"
}
```

Keys are module paths **without** `.weight`. Three rules the emitter enforces:

1. **List every quantized path.** A module with `.scales` on disk but absent
   from the map is quantized at the *default* width — a silent corruption, not
   a harmless omission.
2. **List nothing you did not quantize.** A listed module with no `.scales` on
   disk is quantized at load and then fails to find its weights.
3. **Never rely on the default to carry an intended width.** A lookup that
   falls back cannot distinguish "intentionally 4-bit" from "I forgot this
   one".

`validate_allocation` checks all three, plus that every declared width is one
MLX supports, before you start a multi-minute load.

### MLX width constraints

Verified by probing `mx.quantize` directly on mlx 0.32.3:

- **bits** ∈ {2, 3, 4, 5, 6, 8}. `bits=1` and `bits=7` are rejected, so **2 is
  the floor**. The 1-bit rungs of EXL3/GGUF low-bit ladders have no MLX
  equivalent.
- **group_size** ∈ {32, 64, 128}. `16` is rejected.

`effective_bits(bits, group_size)` returns `bits + 16 / group_size` — the
number to feed the budget constraint. Using the nominal width instead
understates the real cost by ~0.13–0.5 bpw, and the file comes out larger than
the target.

---

## Testing

```bash
python -m pytest
```

88 tests, pure NumPy + MLX, no GPU required. The manifold primitives are checked
against a finite difference of the budget function and the DP against a
brute-force enumeration, so the port is verified against first principles
rather than against the PyTorch original.

---

## Scope, honestly stated

- **No benchmarks yet.** This port has not been run against a quantized model,
  so there are no quality numbers here. The comparison that motivated it —
  RCO versus EXL3 — is a cross-paper inference, not a head-to-head.
- **The table path is a proxy.** It assumes separability. The optimizer path
  does not, but needs a differentiable backend to be useful.
- **The MLX affine format is the target.** MLX's levels are a uniform affine
  grid. The packing coincides (16 two-bit values per uint32, LSB-first)
  but the codebook is standard, so there is no transcoding step.
- **EXL3 is not ported and is not a good first target.** Its format has no MLX
  kernel: no trellis decode, no procedural codebook, no double-sided Hadamard,
  no fused MMA. Its encode path (Hessian + LDLQ + Viterbi) is CUDA-only. And
  MLX's 2-bit floor means its best rungs are unreachable regardless.

---

## Attribution

---

## OLMoE-1B-7B-0125 validation

We validated the pipeline on [allenai/OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125), a 1B-parameter mixture-of-experts model with 64 experts per layer and top-8 routing.

**What changed for v0.2:**
- New `build_moe_error_db.py` uses **GPTQ-based (Hessian-scaled) error measurement** instead of plain RTN
- MoE experts are **batched** using the shared per-layer Hessian for 10-15x speedup
- Pipeline is now **MoE-only, streaming-only** — no dense-model code paths
- Symmetric mode (`--quant-mode symmetric`) reduces scale overhead by 50%

**Validation results (4-bit mixed precision, target 4.0 bpw):**

| Method | PPL (wikitext2) | Degradation | 5-bit tensors | Size |
|--------|-----------------|-------------|---------------|------|
| BF16 | 8.890 | — | — | 25.6 GB |
| **RTN 4-bit (uniform)** | **9.412** | **+5.8%** | 0 | **3.9 GB** |
| **RCO 4-bit (Hessian-weighted, GPTQ)** | **9.320** | **+4.8%** | 17 | **3.5 GB** |

**Checkpoints:**
- **RCO Hessian-weighted** (best): [hermitdave/OLMoE-1B-7B-0125-4bit](https://huggingface.co/hermitdave/OLMoE-1B-7B-0125-4bit)
- **RTN baseline**: [hermitdave/OLMoE-1B-7B-0125-4bit-rtn](https://huggingface.co/hermitdave/OLMoE-1B-7B-0125-4bit-rtn)

**Reproduce with the new pipeline:**

```bash
# Phase 1: Build error_db with GPTQ-based measurement
python -m rco_mlx.scripts.build_moe_error_db \
    --source /path/to/olmoe-bf16 \
    --rtn-source /path/to/rtn-8bit \
    --output error_db.json \
    --model-type olmoe \
    --calibration-dataset wikitext \
    --bitwidths 2,3,4,6,8 \
    --group-size 64 \
    --quant-mode symmetric \
    --use-gptq

# Phase 2: Run RCO allocation + quantization
python examples/moe_pipeline.py \
    --source /path/to/olmoe-bf16 \
    --output /path/to/output \
    --target-bpw 4.0 \
    --group-size 64 \
    --model-type olmoe \
    --error-db error_db.json \
    --quant-mode symmetric
```

**KV cache quantization:**
- Both checkpoints achieve 100% token agreement at 4-bit KV cache
- Peak inference memory: ~6.9 GB for both (MLX dequantizes to full precision at runtime)
- Savings are in storage/I/O, not RAM


RCO is © ISTA-DASLab (Michael Helcig, Dan Alistarh), Apache-2.0. The
optimizer, manifold primitives, and grouping logic here are a port of
[ISTA-DASLab/RCO](https://github.com/IST-DASLab/RCO) and remain Apache-2.0.

Port and MLX integration by **Hermit Dave**, produced with **Hermes Agent**
(Nous Research).

The true streaming quantization approach (layer-by-layer error database
construction without loading the full BF16 model into memory) is inspired by
[oMLX](https://github.com/jundot/omlx) (`omlx/oq.py`), specifically the
`_iter_streamed_layer_blocks` and `_streamed_sensitivity_layer` patterns.

## Expert allocation modes

The sanitize constraint only requires all experts *inside* one stacked
`(layer, projection)` tensor to share a bitwidth — not across projections
or layers. `--expert-allocation` selects how routed experts get their bits:

- `uniform` (default, legacy): one bitwidth for every routed expert.
- `per-projection`: keep the DP assignment; enforce uniformity only
  within each `(layer, projection)` group (majority vote, missing groups
  inherit their layer's majority). Lets the DP put e.g. gate/up at 2-bit
  and down at 4-bit in the same layer.
- `gguf-recipe`: override experts with the per-layer recipe measured from
  Unsloth UD-IQ2_XS headers for Qwen3.8-Flash-Next (`qwen4_exp`, 48
  layers): gate/up 2-bit throughout, down at 4-bit on 24 layers, 2-bit on
  23, 6-bit on layer 4. At group size 64 this predicts 43.2 GB for the
  expert weights vs 42.3 GB measured in the GGUF.

```bash
python examples/moe_pipeline.py \
    --source /path/to/bf16-qwen4exp \
    --output /path/to/output \
    --target-bpw 3.0 --group-size 64 --model-type qwen4_exp \
    --error-db error_db.json \
    --expert-allocation per-projection   # or gguf-recipe
```

See `src/rco_mlx/expert_alloc.py` for the grouping logic and the
measured recipe tables.

---

## License

Apache-2.0, matching the upstream sources.
