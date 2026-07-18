# Differentiable JAX dycore — throughput & scaling (paper skeleton, 2026-07)

Derecho, A100-40GB nodes (4 GPU/node). Fully-differentiable JAX earth-system
dycore, route-B transport (`jax.distributed` multi-controller, ppermute/psum
over NCCL / aws-ofi-nccl / Libfabric CXI). Dry dynamics (`physics=none`). Data:
`$SCRATCH/legoesm_scaling/routeb_campaign_202607/all_tidy.csv`.

> **Skeleton** — tables and framing are drafted; prose, plots, and the flagged
> gaps are TODO.

## The claim

The contribution of this approach is **end-to-end differentiability** (gradients
for ML, data assimilation, calibration). The scaling result shows that
differentiability is **compatible with HPC-relevant performance** — we reach
useful absolute SYPD at ESM-relevant resolutions, and throughput scales with GPU
count. Self-contained characterization; **no cross-model comparison** (different
codes / hardware / physics-completeness make any head-to-head unfair — see
Scope).

Metric = **SYPD** (simulated years per wall-clock day) and aggregate throughput
(Mcells/s).

---

## Demonstration 1 — useful, ESM-relevant absolute SYPD

| grid | config | res (km) | 1-A100 SYPD | best SYPD (N A100) |
|---|---|---|---|---|
| lat-lon | LL512 | 39 | 12.2 | 52.7 (16) |
| lat-lon | LL1024 | 19.5 | 2.9 | 29.8 (16) |
| cubed-sphere | C192 | 52 | 14.7 | 20.3 (6) |
| icosahedral | L7 | 56 | 5.1 | 19.7 (16) |
| icosahedral | L8 | 28 | 1.8 | 10.0 (16) |

Climate work needs > ~1 SYPD; the dycore clears that by 1–2 orders of magnitude
at ~20–55 km on a single-to-modest GPU count.

## Demonstration 2 — aggregate throughput scales with GPU count

**Absolute throughput (Mcells/s) rising with device count** — "add hardware, go
faster / bigger" (not efficiency; see Demo 3):

| grid | config | 1 A100 | 16 A100 | speed-up |
|---|---|---|---|---|
| lat-lon | LL1024 (19.5 km) | 978 | 9887 | **10.1×** |
| icosahedral | L8 (28 km) | 380 | 2065 | 5.4× |
| lat-lon | LL512 (39 km) | 1011 | 4375 | 4.3× (saturates — small problem) |
| cubed-sphere | C192 (52 km) | 1032 | 1418 (at 6) | 1.4× (latency-bound, #1113) |

Lat-lon LL1024 is the hero: **~10× aggregate throughput on 16 GPUs, to
9.9 Gcells/s** (the largest problem measured). Grid-dependence is real — cube is
latency-bound past a few GPUs (~46 collective-permutes/step × ~0.11 ms launch
floor, #1113); the lat-lon structured 1-D band halo scales best.

## Demonstration 3 — strong-scaling efficiency, baselined to 2 GPUs

Strong-scaling efficiency **relative to N=2, not N=1** — so the one-time cost of
introducing halo communication (which does not exist at N=1) is absorbed into
the baseline, and the metric measures multi-GPU scaling quality, not the 1→2
comm onset:

| grid | config | eff @4 | eff @8 | eff @16 |
|---|---|---|---|---|
| lat-lon | LL1024 (19.5 km) | 102% | 93% | **72%** |
| lat-lon | LL512 (39 km) | 86% | 53% | 30% |
| icosahedral | L8 (28 km) | 58% | 58% | 33% |
| icosahedral | L7 (56 km) | 78% | 32% | 17% |
| | | eff @3 | eff @6 | |
| cubed-sphere | C192 (52 km) | 75% | 52% | |
| cubed-sphere | C96 (104 km) | 76% | 42% | |

Reading: at the largest problem (**LL1024, 72% at 16 GPUs**) the code strong-
scales well; smaller problems (LL512, L7) saturate at high N as fixed work runs
out per device (expected). The N=2 baseline is what makes the cube honest —
C192 holds **52% to 6 GPUs** here, versus an unfair 23% when charged the 1→2
comm onset against a single-GPU baseline. (LL1024 super-linear at N=4 = cache /
occupancy; ico L8 flat 58% at N=4–8 then drops — the #1113 n4 note.)

---

## SCOPE / CAVEATS

1. **Dry dynamics only** (`physics=none`). These are **dynamical-core**
   throughput numbers, not full-model SYPD; radiation + physics would lower
   them. Stated as a dycore characterization, not a full-model claim.
2. **No cross-model comparison.** Deliberately: differing codes, hardware
   (A100 vs CPU vs TPU), resolutions, and physics-completeness make any
   head-to-head inaccurate. This report characterizes *this* model's throughput
   and scaling only.
3. **GDR-off lower bound**: NCCL host-staged (aws-ofi-nccl GDR unsupported →
   `NCCL_PROTO=simple`). Multi-GPU throughput and Demo-3 efficiencies are a
   **lower bound**; GDRCopy/DMA-BUF would lift them.
4. **40 GB A100**, Derecho, internal numbers.

## GAPS / TODO

1. **Full-physics SYPD** — add RRTMGP + physics for a real-model throughput
   number alongside the dry-dynamics ceiling.
2. **Plots**: SYPD-vs-resolution (Demo 1); aggregate-throughput-vs-N per grid
   (Demo 2, the LL1024 hero curve); strong-scaling-eff-vs-N baselined to 2
   (Demo 3). `finalize_scaling.sh` on the campaign root for the raw panels.
3. ico rerun with #1175 CP-combining (merged) for the improved ico curve.

## Supporting (mechanism, not headline)

- Cross-node correctness: cube runs cross-node at C96/C192 (#1177 corrected);
  only the tiled np=24 lane (#921) is a confirmed comm-init wedge.
- The cube efficiency ceiling is the ppermute round-count × launch-floor law
  (#1113); CP-combining (#1175) is the merged remedy.
