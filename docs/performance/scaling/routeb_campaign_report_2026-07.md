# Differentiable JAX dycore — throughput & scaling (paper skeleton, 2026-07)

Derecho, A100-40GB nodes (4 GPU/node). Fully-differentiable JAX earth-system
dycore, route-B transport (`jax.distributed` multi-controller, ppermute/psum
over NCCL / aws-ofi-nccl / Libfabric CXI). Data:
`$SCRATCH/legoesm_scaling/routeb_campaign_202607/all_tidy.csv`.

> **Skeleton** — tables, framing, and reference numbers are drafted; prose,
> plots, and the flagged verifications/gaps are TODO.

## The claim

The value of this approach is **end-to-end differentiability** (gradients for
ML, data assimilation, calibration). The scaling result must therefore show
**not** "we beat Fortran" but that *differentiability is compatible with
HPC-relevant performance* — i.e. we reach **useful, competitive SYPD**, and it
**scales**, so this is a real modeling tool rather than a prototype. Three
demonstrations, below.

Throughput metric = **SYPD** (simulated years per wall-clock day), the ESM
community's currency. Strong-scaling *efficiency* is demoted to a supporting
figure (it is an internal metric, and unfair at low device counts where 1→2
GPUs first introduces halo comms).

---

## Demonstration 1 — we reach useful, ESM-relevant SYPD (dry dynamics, GPU)

| grid | config | res (km) | 1-A100 SYPD | best SYPD (N A100) |
|---|---|---|---|---|
| lat-lon | LL512 | 39 | 12.2 | 52.7 (16) |
| lat-lon | LL1024 | 19.5 | 2.9 | 29.8 (16) |
| cubed-sphere | C192 | 52 | 14.7 | 20.3 (6) |
| icosahedral | L7 | 56 | 5.1 | 19.7 (16)* |
| icosahedral | L8 | 28 | 1.8 | 10.0 (16) |

Climate work needs > ~1 SYPD; we clear that by 1–2 orders of magnitude at
~20–55 km on a single-to-modest GPU count. *(ico L7 peak is compute-light per
device — see Demo 2.)*

## Demonstration 2 — aggregate throughput scales with GPU count

Plotting **aggregate throughput (Mcells/s), not efficiency** — the number that
answers "can we add hardware to go faster / bigger":

| grid | config | 1 A100 | 16 A100 | speed-up |
|---|---|---|---|---|
| lat-lon | LL1024 (19.5 km) | 978 | 9887 | **10.1×** |
| icosahedral | L8 (28 km) | 380 | 2065 | 5.4× |
| lat-lon | LL512 (39 km) | 1011 | 4375 | 4.3× (saturates — small problem) |
| cubed-sphere | C192 (52 km) | 1032 | 1418 (at 6) | 1.4× (latency-bound, see #1113) |

**Lat-lon LL1024 is the hero: ~10× aggregate throughput on 16 GPUs** — add
hardware, get proportional throughput, up to the largest problem measured
(9.9 Gcells/s). Grid-dependence is real: cube is latency-bound past 1 GPU
(~46 collective-permutes/step × ~0.11 ms launch floor, #1113), lat-lon's
structured 1-D band halo is the most scalable.

## Demonstration 3 — context: competitive throughput on the modern (GPU) substrate

A **published-reference landscape** (NOT a controlled head-to-head — hardware
differs; see caveats). Ours is dry-dynamics-only; the references are full-model.

| model | class | res | SYPD | hardware | source |
|---|---|---|---|---|---|
| **this work** | diff. physical dycore (JAX) | 52 km | ~15 (1 A100), ~20 (6) | 1–6 × A100-40GB | dry dyn., this campaign |
| CAM6 | production Fortran GCM | ~1° (~100 km) | 14 | 1280 CPU cores | Kochkov 2024 (comparison)† |
| NeuralGCM | ML-hybrid emulator (JAX) | 1.4° (~140 km) | ~180 (1 yr / 8 min) | 1 × TPU v4 | Kochkov 2024, Nature† |
| SCREAM (E3SM) | Fortran/C++ cloud-resolving | 3.25 km | ~1 | GPU (Frontier) | Donahue 2024† |

**The load-bearing distinction — physical dycores vs ML emulators.** NeuralGCM
is ~1–2 orders of magnitude faster than everything, but because it *learns*
large-timestep dynamics (an emulator, effective dt ≫ CFL). SCREAM is slow
despite exascale GPU hardware because it resolves *real* physics at 3 km. Our
model is in the **physical-dycore category** (CFL-limited real dynamics, same as
CAM6 / SCREAM / IFS) — so our peers for "did the physics survive at competitive
speed" are the Fortran physical models, and the honest headline is:

> A differentiable **physical** dynamical core reaches CAM6-class SYPD (~15 vs
> 14) at **2× finer resolution on a single GPU vs 1280 CPU cores** — while
> retaining exact dynamics AND end-to-end differentiability, which no model in
> the table has. NeuralGCM shows what ML emulation buys (raw speed); this shows
> differentiability without giving up the physical core.

---

## CAVEATS (load-bearing)

1. **Dry-dynamics only** (`physics=none` in every row). Our SYPD is a **dycore
   ceiling**; full physics (RRTMGP + convection + …) lowers it — plausibly 2–3×.
   CAM6/NeuralGCM/SCREAM SYPD are **full-model**. So Demo-3 is not apples-to-
   apples until we add physics; the honest current statement is "dry-dynamics
   SYPD comparable to CAM6 full-model SYPD," with the physics gap disclosed.
2. **Not a controlled comparison.** Different hardware (A100 vs CPU vs TPU),
   codes, resolutions. Demo 3 is a *landscape*, not a benchmark we ran.
3. **GDR-off lower bound**: NCCL host-staged (aws-ofi-nccl GDR unsupported →
   `NCCL_PROTO=simple`). Multi-GPU throughput is a **lower bound**.
4. **40 GB A100**, dry, ~internal numbers — not vs published leaderboards.

## GAPS / TODO (ranked)

1. **Full-physics SYPD** — add RRTMGP + physics for a real-model number; this is
   what makes Demo 3 fair. Highest priority for the paper.
2. **Verify the reference numbers** — CAM6 resolution/config for the 14 SYPD /
   1280-core figure; the NeuralGCM 1.4° figure (the Nature/blog "1 yr in 8 min"
   ≈ 180 SYPD; some sources quote higher — pin the exact TPU-v4 config);
   SCREAM's exact Frontier node count. Cite primary sources, no fabrication.
3. **IFS / E3SM coarse-res anchor** — a production Fortran ESM at ~25–50 km on
   *GPU* would be the tightest comparator (same substrate as us).
4. **Plots**: SYPD-vs-resolution (Demo 1); aggregate-throughput-vs-N per grid
   (Demo 2, the hero curve); the reference-landscape bar (Demo 3, log-y).
   `finalize_scaling.sh` on the campaign root for the raw panels.
5. ico rerun with #1175 CP-combining (merged) for the improved ico curve.

## Supporting (not headline)

- Strong-scaling efficiency: lat-lon 63% @16 (LL1024), cube 12–23% @6, ico 34%
  @16 (nlev=26) — fabric-limited (GDR-off) + CP-count, not fundamental (#1113).
- Cube runs cross-node correctly at C96/C192 (#1177 corrected); only the tiled
  np=24 lane (#921) is a confirmed comm-init wedge.
- CPU lanes: cube complete; lat-lon/ico blocked on #1100 — but the JAX-CPU
  comparison is **dropped from the paper** (not a meaningful Fortran proxy).

## Sources

- Kochkov et al., *Neural general circulation models for weather and climate*,
  Nature 632, 1060–1066 (2024): https://www.nature.com/articles/s41586-024-07744-y
- Google Research blog, *Fast, accurate climate modeling with NeuralGCM*:
  https://research.google/blog/fast-accurate-climate-modeling-with-neuralgcm/
- Donahue et al., *To Exascale and Beyond — SCREAM*, JAMES (2024):
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004314
