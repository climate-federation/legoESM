# Spectral GPU-native transform feasibility — go/no-go (2026-07-08)

Scaling-audit item 9: assess GPU-native transform options for the global
spectral dycores (`spectral_pe/sw/nh` on the Gaussian grid) — batched GEMM,
SHTns/sphericart, cuFFT/cuBLAS layout changes, or CPU-only status quo.
**Assessment + microbenchmarks only; no dycore rewrite.**

## What the transform is today

`packages/core/legoesm/grids/gaussian.py`: FFT in longitude
(`jnp.fft.rfft/irfft`) + the Legendre leg as a **dense complex GEMM**
(`_analysis/_synthesis_legendre_gemm`, einsum over the packed
`(n_lat, n_sh, nlev)` tensor), f64+complex128 hard-gated (the grid builder
raises without x64). A bf16-GEMM ablation lane already exists
(`LEGOESM_SH_GEMM_BF16`, `sh_gemm_design_2026-06-13.md`). So the "batched
GEMM on GPU" option is **not a rewrite — it is the current code running on
a CUDA backend**; the question is only whether the hardware's fp64 GEMM
rate makes it worthwhile.

## Microbenchmark (this audit)

`scripts/bench/bench_spectral_transform_micro.py` — isolated
analysis+synthesis round trip; the Legendre path is EXPLICIT per row
(`legendre_path`: the legacy gather/segment-sum default vs the opt-in
`LEGOESM_SH_GEMM=1` batched-GEMM), the timed input is band-limited first
(synthesis∘analysis is a projection, so the recorded error is the
transform's own, not truncation), and the recorded rate is an
*equivalent*-GEMM number (model FLOPs / time). Measured on the laptop
**CPU** lane (f64, nlev=30, medians of 20 — these rows are explicitly NOT
the A100/H100 answer; the backend is on every row):

| T | grid | n_sh | legacy round trip | **GEMM round trip** | GEMM share of model FLOPs | AI |
|---|------|------|------------------:|--------------------:|--------------------------:|---:|
| T42 | 64×128 | 946 | 5.34 ms | 3.04 ms | 46% | 10.0 F/B |
| T85 | 130×260 | 3,741 | 33.5 ms | 20.5 ms | 59% | 12.1 F/B |
| T170 | 256×512 | 14,706 | 145.6 ms | **43.2 ms** | 72% | 13.4 F/B |

(FLOP model: the Legendre matrices are REAL against complex fields — a
real×complex MAC is 4 real FLOPs; the first draft double-counted at 8.)

Reading: (a) the opt-in GEMM path is **3.4× faster than the legacy path
at T170 on CPU alone** (identical band-limited round-trip error,
4.2e-12) — this audit's own adversarial review caught the first draft
measuring the legacy path while labeling it GEMM; (b) the transform is
GEMM-dominated (72% of model FLOPs at T170, share growing with T) with
AI 10–13 F/B — compute-bound on fp64-weak parts, near the roofline knee
on fp64-strong parts; (c) the FFT leg shrinks with T —
**cuFFT/layout work is not the lever**.

## The consumer-vs-datacenter fp64 split (revises the prior reading)

The audit's "spectral GPU-hostile" evidence (`SCALING_SUMMARY.md`: T42
13 Mc/s fp64, CPU→GPU only 10.9×) was measured on an **RTX 5090 laptop
part — 1:64 fp64:fp32**. That measurement says consumer GPUs are bad at
fp64 GEMM, not that the transform is GPU-hostile:

- A100 fp64: 9.7 TF/s (19.5 TF/s tensor-core DGEMM). Against the measured
  T170 GEMM-path round trip (0.90 GF model / 43.2 ms ≈ 21 GF/s equivalent),
  the GEMM-roofline headroom is **roughly two orders of magnitude** — a
  ROUGH UPPER BOUND, not a prediction: the model FLOPs are the triangular
  count, the FFT leg and launch overheads are outside the GEMM roofline,
  and realized DGEMM fractions vary. The point stands at any realistic
  discount; only the measurement settles it.
- The GEMM path lowers to cuBLAS on CUDA; the measurement costs one
  command on a Derecho/Levante GPU node (``--sh-gemm both`` records the
  legacy column too, same rows schema):

```
JAX_ENABLE_X64=1 python scripts/bench/bench_spectral_transform_micro.py \
    --truncations 42,85,170,341 --nlev 60 --sh-gemm both \
    --out results/spectral_micro_a100.json
```

## Options assessed

| option | verdict | why |
|---|---|---|
| **Batched GEMM on datacenter GPU** (existing code path) | **GO — measure first** | Transform is GEMM-bound (72% at T170, AI 13); A100-class fp64 GEMM headroom is orders of magnitude; zero integration work. Run the one-command microbench above before ANY further investment. |
| **SHTns (GPU build)** | **NO-GO now** | External C dependency via FFI custom-call: breaks `jax.grad` without a hand-written custom_vjp pair, adds a build dependency to every cluster, and its GPU path targets the same GEMM/FFT arithmetic the einsum already reaches through cuBLAS. Revisit only if the GEMM microbench shows the einsum path leaving >2–3× on the table. |
| **sphericart** | **NO-GO** | Real-SH evaluation for point clouds (derivatives of Y_lm at scattered points) — not a Gauss–Legendre grid transform; wrong tool. |
| **cuFFT/cuBLAS layout surgery** | **NO-GO** | FFT share falls with T (28% at T170); the packed `(n_lat, n_sh, nlev)` GEMM layout is already the batched-GEMM-friendly one (`_maybe_chunk_trailing` handles memory). |
| **CPU-only status quo** | **Default pending measurement** | Correct, validated, and the SI solve keeps the dycore single-device anyway (below). |

## Multi-device: stays N/A (unchanged)

The audit matrix verdict stands — **no-go for multi-device spectral
investment**:

- Both sharding schemes measured ANTI-scaling
  (`spectral_level_shard_cliff.md`: T85 0.74× at 4 devices) — the
  semi-implicit solve couples all levels (all-gather per solve), and the
  transform itself is replicated under level sharding.
- The 1:64-fp64 consumer measurement does not change the collective-
  coupling math; a fast single-GPU transform makes the *single-device*
  dycore faster, not the sharded one.

## Go/no-go summary

1. **GO (cheap, first):** run the microbench on one A100/H100 node
   (command above). If the round trip lands within ~5× of the GEMM
   roofline, the existing einsum path IS the GPU-native transform and the
   spectral dycore becomes a **single-GPU** citizen on fp64-strong parts —
   no library integration, no rewrite. Wire `--truncations 341` to probe
   the production-relevant end.
2. **NO-GO:** SHTns/sphericart integration, cuFFT layout work, and any
   multi-device spectral scheme — each is dominated by either the existing
   GEMM path or the measured SI-coupling wall.
3. The dycore-side fp64 constraint is physics
   (`core/precision.py::_ATMOSPHERE_OVERRIDES`), not transform-imposed;
   the bf16-GEMM ablation lane remains the sanctioned precision
   experiment.
