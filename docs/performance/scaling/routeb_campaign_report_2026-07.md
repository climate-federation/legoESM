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
| icosahedral | L7 | 56 | 5.0 | 23.5 (**4**) |
| icosahedral | L8 | 28 | 1.8 | 9.2 (**8**) |

Climate work needs > ~1 SYPD; the dycore clears that by 1–2 orders of magnitude
at ~20–55 km on a single-to-modest GPU count.

Note the peak-N column: only the lat-lon cases peak at the top of the ladder.
Icosahedral L7 peaks at **4** GPUs and L8 at **8**, then declines — adding
devices past that point costs throughput, which is the Demo-3 story in absolute
terms. Quote "best SYPD" with its device count, never as "SYPD on 16 GPUs".

*(Table regenerated from `all_tidy.csv` 2026-07-18, post-ico-rerun. The
pre-rerun values — L7 19.7 @16, L8 10.0 @16 — were stale AND misattributed the
peak to N=16.)*

## Demonstration 2 — aggregate throughput scales with GPU count

**Absolute throughput (Mcells/s) rising with device count** — "add hardware, go
faster / bigger" (not efficiency; see Demo 3):

| grid | config | 1 A100 | max N | speed-up |
|---|---|---|---|---|
| lat-lon | LL1024 (19.5 km) | 978 | 9887 (16) | **10.1×** |
| lat-lon | LL512 (39 km) | 1011 | 4375 (16) | 4.3× (saturates — small problem) |
| icosahedral | L8 (28 km) | 380 | 1697 (16) | 4.5× |
| icosahedral | L7 (56 km) | 261 | 793 (16) | 3.0× |
| cubed-sphere | C192 (52 km) | 1032 | 1418 (6) | 1.4× (latency-bound, #1113) |
| lat-lon | LL192 (104 km ≈1°) | 950 | 1022 (16) | 1.1× (comm-starved) |
| cubed-sphere | C96 (104 km) | 759 | 533 (6) | **0.7× — ANTI-scales** |
| icosahedral | L6 (112 km ≈1°) | 479 | 315 (16) | **0.7× — ANTI-scales** |

Lat-lon LL1024 is the hero: **~10× aggregate throughput on 16 GPUs, to
9.9 Gcells/s** (the largest problem measured). The bottom three rows are the
same finding from the other end — at ~1° there is too little work per device to
hide the halo exchange, and adding GPUs is flat (LL192) or actively **negative**
(C96, L6 both end BELOW their 1-GPU throughput). Grid-dependence is real: cube
is latency-bound past a few GPUs (~46 collective-permutes/step × ~0.11 ms launch
floor, #1113); the lat-lon structured 1-D band halo scales best.

*(Regenerated from `all_tidy.csv` 2026-07-18. Pre-rerun ico L8 read 2065 / 5.4×.)*

## Demonstration 3 — strong-scaling efficiency, baselined to 2 GPUs

Strong-scaling efficiency **relative to N=2, not N=1** — so the one-time cost of
introducing halo communication (which does not exist at N=1) is absorbed into
the baseline, and the metric measures multi-GPU scaling quality, not the 1→2
comm onset:

| grid | config | eff @4 | eff @8 | eff @16 |
|---|---|---|---|---|
| lat-lon | LL1024 (19.5 km) | 102% | 93% | **72%** |
| lat-lon | LL512 (39 km) | 86% | 53% | 30% |
| icosahedral | L8 (28 km) | 59% | 60% | 27% |
| icosahedral | L7 (56 km) | 79% | 32% | 13% |
| lat-lon | LL192 (104 km ≈1°) | 53% | 24% | 13% |
| icosahedral | L6 (112 km ≈1°) | 53% | 16% | **7%** |
| | | eff @3 | eff @6 | |
| cubed-sphere | C192 (52 km) | 75% | 52% | |
| cubed-sphere | C96 (104 km) | 76% | 42% | |

Reading: at the largest problem (**LL1024, 72% at 16 GPUs**) the code strong-
scales well; smaller problems (LL512, L7) saturate at high N as fixed work runs
out per device (expected), and the ~1° cases (LL192, L6) collapse to 13% / 7%
— the resolution-dependence that is the headline of the figures. The N=2
baseline is what makes the cube honest — C192 holds **52% to 6 GPUs** here,
versus an unfair 23% when charged the 1→2 comm onset against a single-GPU
baseline. (LL1024 super-linear at N=4 = cache / occupancy; ico L8 flat ~60% at
N=4–8 then drops — the #1113 n4 note.)

*(Regenerated from `all_tidy.csv` 2026-07-18. Pre-rerun ico read L8 58/58/33,
L7 78/32/17.)*

---

## Demonstration 4 — what a gradient costs (single A100)

Demos 1–3 are **forward** evaluations. Since the contribution is
differentiability, the cost of the reverse pass is a first-class number, and
until 2026-07-18 the repo had never measured it: the benchmarks time
`model.step`, and the distributed-AD tests
(`tests/distributed/test_mpi_differentiability.py`) verify gradient
*correctness* with no timing at all.

Measured by `scripts/bench/bench_ad_overhead.py` (job:
`scripts/cluster/scaling_derecho/ad_overhead_gpu.pbs`). Same `build_amip_step`
builder as the scaling curves. **LL192, L26, fp32, `physics=none`, 8 steps,
1 × A100-40GB, 9 interleaved repeats** (run-to-run spread < 0.1%):

| mode | wall × | FLOP × | byte × | reverse tape | arithmetic intensity |
|---|---|---|---|---|---|
| forward | 1.00 | 1.00 | 1.00 | 141 MB | 0.76 FLOP/byte |
| `grad` (full BPTT) | **5.45** | 2.26 | 10.67 | **7.96 GB** | **0.16** |
| `grad_ckpt` (per-step remat) | **4.91** | 3.16 | 3.53 | **0.73 GB** | 0.68 |

**A gradient step costs 5.4× a forward step.**

**Checkpointing is FASTER here, not merely smaller** — 4.91× vs 5.45×, while
cutting the tape 11×. That inverts the usual "remat trades time for memory"
framing, and the arithmetic-intensity column is why: plain BPTT moves 32.6 GB
to do 5.3 GFLOP (intensity 0.16, badly memory-bound), whereas remat recomputes
instead of reloading and restores intensity to 0.68 — essentially the forward
pass's 0.76. On a bandwidth-bound kernel, trading bytes for FLOPs wins in BOTH
currencies. Note this is exactly the case where the FLOP ratio alone misleads:
it ranks `grad_ckpt` WORSE (3.16 vs 2.26) while the wall clock ranks it better.

Tape scales ~linearly in cells × steps (~519 B/cell/step plain,
~47 B/cell/step checkpointed), so at 8 steps on a 40 GB A100
(**extrapolated, not measured** — LL512 is the obvious confirmation run):

| config | plain BPTT | checkpointed |
|---|---|---|
| LL512 | ~57 GB — OOM | ~5 GB ✓ |
| LL1024 | ~226 GB — OOM | ~21 GB ✓ |

i.e. checkpointing is what makes gradients at the Demo-2 headline resolutions
possible on a single GPU at all — and it costs no wall-clock to use.

**Scope — do NOT over-read this number.** It is **single-device**. Reverse mode
adds tape memory, adds a transposed halo exchange per exchange, and adds
recompute under remat, all of which attack the very quantity Demo 3 measures
(work per device available to hide comms), so the forward scaling curves are an
**optimistic bound** for the gradient case, not a proxy. Distributed
reverse-mode AD has never run above 6 ranks in this repo and has never been
timed. Also `bench_ad_overhead` resolved `dt=1.0 s` for LL192 where the
campaign used `dt=60 s`; this does not affect the ratio (dt is a scalar
multiplier, identical compute graph, cancels between the modes) but the config
is therefore not byte-identical to Demo 1–3 — see GAPS.

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
2. **Plots** — generated by `scripts/plot/plot_routeb_campaign_paper.py`
   (requires `--csv all_tidy.csv`; there is deliberately NO embedded snapshot
   fallback — the old one silently diverged from the aggregated CSV). Sequential
   blue ramp coarse=light→fine=dark, marker+linestyle secondary encoding for
   grayscale. **The message is that GPU strong scaling is
   RESOLUTION-DEPENDENT**: coarse ~1° is comm-starved (too little work/device)
   and does not scale; fine ~20 km has enough compute to hide the halos and
   does. Each figure is 2-panel — (a) **Mcells/s** vs #A100 with a faint
   per-curve ideal-scaling ray, (b) strong-scaling efficiency baselined to N=2:
   - `fig1_latlon_scaling` (money): LL192 (~104 km ≈1°) → LL512 (~39 km) →
     LL1024 (~19.5 km). All three START at the same per-device throughput
     (950/1011/978 Mcells/s on 1 A100 — one GPU is saturated regardless of
     resolution) and then FAN OUT: LL192 stays flat to 1022 (eff 13% @16),
     LL512 reaches 4375 (30%), LL1024 tracks ideal to N=4 and reaches 9887
     (72%). The common origin is what makes the mechanism legible.
   - `fig2_ico_scaling` (confirmation): L6 (~112 km, eff 7% @16) → L7 (~56 km,
     13%) → L8 (~28 km, 27%) — same trend on a second grid.
   **Mcells/s, NOT SYPD**, because the grids do not share a dt at matched
   resolution (LL512 runs dt=60 s vs C192/L7 at 30 s) and the resolutions
   differ in total cells; a SYPD panel folds both back in and obscures the
   variable under test. When the full-physics run lands, add a dry-vs-full
   curve to panel (a).
3. **Cross-grid comparison is NOT clean — do not rank grids off Demo 1/2.**
   At nominally matched resolution the configs differ in more than the grid:
   LL512 (39 km) runs 13.6M cells at dt=60 s, C192 (52 km) 5.8M at dt=30 s,
   L7 (56 km) 4.3M at dt=30 s. In the dt-free metric lat-lon and cubed-sphere
   are indistinguishable on one A100 (1011 vs 1032 Mcells/s); only
   icosahedral is slower (261). So "structured beats unstructured" is at best
   latlon ≈ cube > ico, and lat-lon's SYPD lead over cube is a TIMESTEP
   advantage, not a memory-access one. A defensible ranking needs matched
   cells + matched dt + matched nlev.
4. **AD gaps** (Demo 4): (a) LL512/LL1024 gradient runs to confirm the
   extrapolated OOM boundary and whether 4.9× holds at scale; (b) fp64 and
   `physics=held_suarez` variants — both are expected to move the ratio
   (radiation showed 6.1× fp32 vs 2.5× fp64); (c) `bench_ad_overhead` resolves
   `dt=1.0 s` for LL192 where the campaign used `dt=60 s` — harmless for the
   ratio, but `_auto_dt` should be reconciled so Demo 4 is byte-identical to
   Demo 1–3; (d) NO distributed gradient timing exists at any rank count.
5. **Held-Suarez is NOT what this campaign ran.** Every row is
   `physics_level=none` (bare dry dynamics, no forcing). Held-Suarez is a
   selectable option in the same drivers (`--physics held_suarez`) that was not
   exercised. Any HS claim must either cite a separate HS run or re-run this
   ladder with the forcing on — the two must not be merged into one sentence.
6. ico rerun with #1175 CP-combining (merged) for the improved ico curve.

## Supporting (mechanism, not headline)

- Cross-node correctness: cube runs cross-node at C96/C192 (#1177 corrected);
  only the tiled np=24 lane (#921) is a confirmed comm-init wedge.
- The cube efficiency ceiling is the ppermute round-count × launch-floor law
  (#1113); CP-combining (#1175) is the merged remedy.
