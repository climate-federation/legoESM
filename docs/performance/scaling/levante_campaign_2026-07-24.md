# Levante weak/strong scaling campaign — 2026-07-24 (first hardware receipts + fixes)

One-day campaign turning the authored-but-never-run Levante job set
(`scripts/cluster/scaling_levante/`) into measured curves for every grid,
fixing what broke, and moving the worst axis (ocean strong scaling) to a
measured 2× improvement. All receipts on the post-merge tree `d3ec1ccce`+
(campaign branch `worktree-scaling-campaign`); job IDs cited throughout are
Levante SLURM jobs from 2026-07-24. Codex adversarial review: 4 rounds
(transcripts under `.physics-validator/scaling_campaign/`); every
measurement claim below carries the round-3 corrections.

Machines: Levante `gpu` partition (4× A100-80 SXM NVLink/node, IB HDR200),
`compute` (2× AMD Milan 7763). All GPU multinode = route-B
(`jax.distributed` + NCCL over IB verbs — `NET/IB mlx5` confirmed in-log;
route-A CUDA-aware mpi4jax not exercised on Levante).

## Headline results (strong scaling, f32 unless noted)

| Axis | Ladder | Result | Job(s) |
|---|---|---|---|
| Atm lat-lon LL720×1440 L26 | 4→8→16 A100 (1→4 nodes) | 7.72→5.40→3.54 ms/step, monotone; np16 = 7.6 GC/s (477 Mc/s/GPU sustained) | 26450848/26453240/26449147 |
| Atm MPAS ico L8 (28 km) L26 | 6→16 A100 | 8.66→7.08 ms/step; np16 = 2.41 GC/s — 1.6x the Derecho 16-A100 aggregate reported in `derecho_levante_sota_review_2026-07.md` SS3b (route-A, eff ~0.38 @16); CROSS-MACHINE, different stack/date - indicative, not a controlled A/B | 26453240/26449147 |
| **Atm cube C768/L60 (same-path cs-spmd)** | 6→24 A100 | 58.35→14.09 ms/step = **4.14× = eff 1.04 (at ideal)**, 15.1 GC/s (629 Mc/s/GPU) | 26453782 |
| Atm cube C384/L60 (same-path cs-spmd) | 6→24 A100 | 15.44→8.81 ms/step = 1.75× (eff 0.44), 6.0 GC/s | 26452894 |
| Atm cube C192/L60 (same-path) | 6→24 | 6.20→6.80 ms — ANTI-scales (eff 0.23): 9.2k cols/GPU is below the ~30k-column floor | 26452979 |
| Ocean lat-lon LL576×1152 L20, production implicit | 4→8→16 | 15.6→17.2→17.4 ms — anti-scales across nodes | 26452743-45 |
| Ocean same, improved (wide-halo + vmix-f32) | 4→8→16 | 12.8→11.1→8.6 ms — monotone, **2.01× at 16 GPUs** | 26452804-06, 26453279 |
| Cube tiled >6-GPU lane (its own bench) | 24 A100, C384/L60 closed loop | 9.01 ms/step, 5.9 GC/s, 18.2 SYPD — first >6-GPU production-lane receipts | 26450938/26452632 |

Single-node GPU (job 26445836): cube C192 gray_sbm strong eff 0.84–1.07
(1→3 A100); C48/C96 latency-floored. Ocean single-node solver ladder: below.

**THE cube strong-scaling result** — read 1.04 as "at ideal", NOT "better
than ideal": efficiency slightly above 1 is expected when the BASE leg is
per-device disadvantaged (np6 holds 4x the working set per GPU of np24, so
part of the 4.14x is cache/occupancy recovery rather than parallel
efficiency). The claim is that comm does not degrade this ladder, not that
parallelism is free. (same code path, harness, config, IC;
only the tile size varies — jobs 26452979/26452894/26453782): 6→24 GPU
speedup 0.91× / 1.75× / 4.14× at 9.2k / 36.9k / 147k columns per GPU. The
"poor cube strong scaling" of the earlier receipts TRACKS TILE SIZE:
holding code path, harness, config and IC fixed and varying only the tile,
efficiency goes 0.23 -> 0.44 -> 1.04, so tile size is SUFFICIENT to recover
ideal scaling at 24 A100 across 6 nodes. (That shows comm does not degrade
the ladder at production tiles; it does not prove comm costs nothing at
small tiles — that needs a per-phase profile.) Production rule confirmed:
keep >~30k columns/GPU.

Tiled-lane size sweep at fixed 24 GPUs (own bench, closed loop, CFL-scaled
dt; jobs 26450938/26452632/26453645): 1.87 / 5.89 / 14.34 GCells/s at
C192/C384/C768 = 78 / 246 / 597 Mc/s per GPU — per-device throughput still
climbing at 8.85M cells/GPU, so an A100 is not saturated even there.

## Ocean strong-scaling: bottleneck → fix (the campaign's improvement arc)

Dose-response on the implicit-CN barotropic reduction count
(LL384×768 L20 f64, solver-matched via `--force-pcg`, 4×A100 NVLink,
jobs 26447957/26449622):

| arm | reductions/step | nd1/nd2/nd4 ms | eff4 |
|---|---|---|---|
| implicit fixed-PCG standard | 123 | 29.3/21.1/15.0 | 0.49 |
| implicit PCG single_reduce | 63 | 29.6/19.8/14.0 | 0.53 |
| explicit + wide-halo | 0 (solver) | 33.4/19.1/11.5 | 0.73 |

Monotone count→efficiency mapping at every size (LL192/384/768); at LL768
all arms converge to 0.71–0.77 (tiles amortize latency). Fused-halo == plain implicit (a NULL result: pad aggregation does not
move this step, consistent with reductions being the larger cost - the
arms differ in solver internals too, so this is consistency, not proof). NCCL_PROTO
default ≡ LL, LL128 harmful (job 26449812) — protocol lever closed.
`LEGOESM_VMIX_F32_SOLVE=1` (f32 tridiagonal vmix inside the f64 step):
icn +9.1%, wide +11.7% at nd4, conservation-gated at `--cons-rtol 1e-5`
on every arm (job 26452547). Combined best config (wide + vmix-f32):
10.32 ms at LL384 nd4 = 1.45× the production config, and the multinode
2.01× above. CLAIM SCOPE (codex round-3): the count→time slope is an
*effective time per eliminated reduction-batch in these executables*
(~14–18 µs/batch), NOT a measured allreduce latency — `single_reduce`
changes solver work/fusion too; a dependency-matched microbenchmark would
be needed for a latency claim. Both winning options are existing config
selections (`barotropic_solver="explicit_substep"` + wide-halo flags,
`LEGOESM_VMIX_F32_SOLVE`); production defaults unchanged — promotion needs
the wide-halo stability gates (`SCALING_STATUS_AUDIT` item 3) and a
science sign-off on the mixed-precision vmix.

## Atm ladders added late in the campaign

**Lat-lon WEAK at a production tile** (45 rows x 1440 lon x L26 = 64.8k
columns/GPU, f32, job 26454476): efficiency 1.00 / 0.45 / 0.41 / 0.45 /
0.44 at 1/2/4/8/16 GPUs — the cost is paid ONCE on the first cross-device
step and then FLAT to 16 GPUs across two node crossings (1.09 -> 7.66
GCells/s aggregate). Weak scaling on this grid is a fixed entry toll, not
a compounding one.

**MPAS icosahedral L8 (28 km) STRONG, identical padded mesh** (jobs
26454476 + 26454618): 19.90 / 17.09 / 6.92 / 7.10 ms at np 2/4/8/16.
Taking np2 as the base (it has the BEST per-device throughput, 430
Mc/s/GPU): 2->8 = 2.88x = eff 0.72, 2->16 = 2.80x = eff 0.35 (small-tile
floor).

OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
than np2 while np8 is 2.5x faster than np4. Evidence gathered:
- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
  17.14/17.04, 6.87/6.97).
- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
  two nodes (17.12 ms), so node crossing is irrelevant.
- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
  falling, and under 3 % of owned cells at every count.
- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
  device count is a compile-time property so the HLO matches what the GPUs
  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
  bitcasts 526/582/694 — the np8 program is finer-grained.)
- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
  (geometric). Every method shows the same 2.5-3.1x jump.
- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
  command-buffers-off — every arm within 1 %, none recovers np4.
  (The multi-output-fusion arm errored on an unsupported flag name and is
  not counted.)
VERDICT: five hypotheses refuted by measurement (placement, halo volume,
collective count, partition method, codegen env knobs). The cheap levers known to this campaign are exhausted; the remaining suspect — per-device kernel efficiency
for this shape — needs a GPU op-level profile (nsys / XLA op profile of
np4 vs np8), which is a separate instrumented project, not another timing
run. Per-GPU throughput across the ladder is non-monotone in tile size
(430 / 249 / 305 / 150 Mc/s/GPU at 327688 / 163844 / 81922 / 40961
cells/device), which is itself the clue to hand the profiler.
PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
throughput is 304-340 Mc/s/GPU vs 220-248 at np4.

## Ocean strong scaling vs TILE SIZE (jobs 26456334/37 vs 26452804-06)

The same improved config (wide-halo + vmix-f32, multicontroller NCCL/IB,
f32 L20) run at two tile sizes, 4 -> 16 GPUs:

| grid | cells/GPU @16 | np4 / np8 / np16 ms | eff @8 | eff @16 | aggregate @16 |
|---|---|---|---|---|---|
| LL576x1152 (13.3M) | 0.83M | 12.81 / 11.05 / 8.63 | 0.58 | 0.37 | 1.53 GCells/s |
| LL1152x2304 (53.1M) | 3.3M | 43.47 / 27.79 / 17.24 | **0.78** | **0.63** | **3.08 GCells/s** |

Both ladders are monotone; the bigger tile is uniformly better at every
device count (jobs 26456334 / 26457693 / 26456337 vs 26452804-06).

So the ocean shows the SAME tile-size dependence the cube does: the 2.01x
multinode improvement measured at LL576 was partly a floor effect, and at
a production tile the identical code scales substantially better (0.37 ->
0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
Config is byte-identical between the two rows; only the grid changes.

REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
np4 — a LARGER per-device tile — finished in ~25 min, which looked like a
compile-time cliff at that device count. It is not: the third attempt ran
the identical configuration in **99 seconds** with a 21.6 s compile (job
26457693). The earlier hangs were transient/environmental, not
reproducible, and no compile-time defect is claimed.

## Weak scaling at production per-device size (job 26453523)

The earlier weak ladders used a 64-row base (0.17M cells/GPU — under the
latency floor). Re-run at PRODUCTION size (288 rows × 1152 lon × L20 =
6.6M cells/GPU, 1→4 A100, conservation gated): production implicit
1.00/0.72/0.70, improved wide-halo+vmix-f32 1.00/0.86/0.85.
PRECISION MATCHED (self-audit correction): BOTH ladders compared here are
**float32** — the production-tile run is f32, so it is compared against
the earlier ladder's f32 rows (eff 0.26/0.25 at nd 2/4), not its f64 rows (both from job 26445836).
An earlier revision of this file mislabelled the production-tile run f64
and cited the f64 small-base numbers; the direction and size of the effect
are unchanged, but the comparison is only valid precision-matched. So the earlier weak ladder measured a below-floor tile rather than a code
limit, and the same
config that fixes strong scaling also carries weak (+0.15 at nd4). Ideal is
flat; the improved arm holds 22.5→22.9 ms while production drifts
17.8→25.3 ms.

## "It used to be faster / did we regress?" — resolved, no regression

- Cross-machine anchor (matched bench/config/grid/physics/precision,
  job 26450081): Levante single A100-80 latlon-moist r720 f32 =
  **418.5 Mc/s** vs Derecho single-A100 ≈370. SCOPE: this establishes NO
  LARGE REGRESSION, not a precise machine ranking — the two campaigns
  differ in machine (A100-80 SXM vs A100-40), jax/tree version and date,
  so the ~13% gap is not attributable to any single factor.
- Cube "404 vs 141 Mc/s": the 404 is the single-GPU RTX-5090 Held-Suarez
  row in `SCALING_SUMMARY.md` SS1; ours is gray+SBM on A100 (job 26445836).
  That file's own tier table prices gray+SBM ~3x Held-Suarez, so ~135 is
  the expected equivalent vs 141 measured. PLAUSIBLE reconciliation from
  two published tables, NOT a matched A/B (GPU, physics and date differ).
- Ocean absolutes (A100 f64 165-201 Mc/s, f32 352, job 26445836; 5090 f64
  152 / f32 400 from `SCALING_SUMMARY.md` SS1) are of the same order -
  again a cross-machine sanity check, not a controlled comparison.
- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
  to `explicit_substep`; our explicit/wide arm reproduces that class
  (0.88 @2, LL384) — the production implicit config was never measured
  there. So the gap is explained by the solver the old bench selected;
  labelling it 'protocol, not regression' is an inference from that
  config difference, not an independent bisect.
- No merge regression: nd=1 stock-CG LL192 8.90 ms pre-merge (job 26445836)
  vs 8.94 ms post-merge (smoke on tree d3ec1ccce) - one sample each, so
  this bounds a large regression only.

## CPU-MPI (compute nodes)

Single-node ladders (job 26445986, f64): atm latlon strong eff
0.87–0.92@np2 → 0.11–0.23@np32–64; ocean implicit 0.90@2 → 0.24@32; weak
collapses ≤0.14@32. CAVEATS: np=1 ocean leg was stock-CG (solver-mismatched
— fixed via `--force-pcg` in the job scripts; np≥2 slopes valid), and
single-node ladders conflate Milan DRAM contention with comm. The first
4-node pair collided into one OUTDIR (same-second stamp) and was discarded.

4-node SPREAD ladder (job 26452578, f64, ranks round-robin, solver-matched):
atm latlon np2 eff ~1.00; spreading ranks over 4 nodes nearly doubles
efficiency at high rank counts (r128 np32: 0.38 spread vs 0.20 packed),
which is CONSISTENT with per-node memory-bandwidth contention in the
packed ladder (not isolated by a bandwidth counter), decaying to 0.06–0.16
at np64–128 — the 1-D band perimeter ceiling as designed (r256/np128 = 2
rows/rank). Ocean strong spread: np2 eff 1.32 (superlinear - typical of a base leg whose working set does
not fit cache; not instrumented here), 0.88@8,
0.35@32, wall at np64 (79 ms > np32's 74 ms). The job died in a high-rank
ocean case (one rank exit-3 → kill-on-bad-exit) before the weak tail —
np128 ocean + weak ladders and the rank-failure attribution remain open.

## Atm ICOSAHEDRAL CPU-MPI (job 26452579, 1 node, f64 moist, L26)

The last measurement gap, and the healthiest strong-scaling curve in the
campaign. Efficiency t1/(n*tn) by subdivision:

| subdiv | np2 | np4 | np8 | np16 | np32 | np64 |
|---|---|---|---|---|---|---|
| 4 | 0.90 | 0.76 | 1.00 | 0.53 | 0.26 | 0.15 |
| 5 | 1.02 | 0.88 | 1.15 | 0.56 | 0.58 | 0.30 |
| 6 | 1.02 | 0.88 | 1.07 | 0.51 | 0.54 | 0.57 |
| 7 | 1.03 | 0.92 | 1.02 | 0.51 | 0.51 | 0.52 |
| 8 | 1.04 | 0.93 | 1.18 | 0.61 | 0.49 | — |

THE TILE-SIZE LAW, THIRD INDEPENDENT COMPONENT: the np64 column collapses
on coarse grids (0.15 at subdiv4, 0.30 at subdiv5) and holds on fine ones
(0.52-0.57 at subdiv6-7). Same pattern as the cube (0.23 -> 1.04 across
tile sizes) and the ocean (0.37 -> 0.63) — now on a third component and a
different transport (CPU-MPI, not NCCL). This is the campaign's most
reproducible finding.

Shape: ~1.0 through np8, one step down, then FLAT 0.5 from np16 to np64 —
4x more ranks with no further loss, the signature of a fixed per-rank cost
rather than growing communication. (The >1 points at np8 are the same
base-leg-working-set effect noted for the cube; read as "at ideal".)

Against the 4-node SPREAD lat-lon ladder (job 26452578) at high rank
counts the contrast is large: ico holds 0.52 at np64 where lat-lon r128/r256
is at 0.12/0.16. That matches the documented expectation that a 2-D cell
partition beats a 1-D latitude band on perimeter/area. CAVEAT: ico ran
PACKED on one node and lat-lon SPREAD over four, so this compares
decomposition AND placement together, not decomposition alone.

The ico WEAK ladder from the same job is non-monotone (1.00 / 0.47 / 0.81 /
0.48 / 0.46 / 0.22 / 0.45 at np 1..64) — the per-rank problem size is not
held constant cleanly across that sweep's subdivision steps, so no weak
claim is made from it.

## DISTANCE TO THE THEORETICAL LIMIT, measured (job 26457977)

With all three ingredients measured on this machine — fabric constants
(17.82 us, 64.22 GB/s) and the per-tile single-device compute term — the
ocean LL576 f64 implicit ladder finally has a real roofline:

| nd | measured | calibrated bound | measured/bound | at % of floor |
|---|---|---|---|---|
| 2 | 42.71 ms | 34.77 ms | 1.228 | 81 % |
| 4 | 23.53 ms | 16.82 ms | 1.398 | 72 % |

Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
ratio worsens as compute shrinks.

WHAT THE GAP IS NOT — the omitted-traffic explanation is REFUTED
(`scripts/tmp/probe_ocean_halo_bytes.py`, HLO byte census on CPU virtual
devices). The bench's `comm_scope_note` correctly warns that its census is
"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
and the true volume IS much larger: **16.22 MB/step across 110
collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
completing the census moves the bound by only **0.22 ms**, because the
comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.

So with the byte census completed the unexplained residual is still 5.5 ms
(nd2) and 4.3 ms (nd4).

SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
job 26458553): the sharded formulation does NOT do measurably more work.
Timing the SHARDED step on a 1-device mesh (all the padding, band-edge and
v-row-reconstruction machinery present, ppermutes self-to-self so no real
traffic) against the UNSHARDED step at the identical tile:

| tile | unsharded | sharded on 1 device | overhead |
|---|---|---|---|
| 288x1152x20 | 33.13 ms | 32.79 ms | **-0.34 ms (-1.0 %)** |
| 144x1152x20 | 15.88 ms | 15.92 ms | **+0.04 ms (+0.2 %)** |

Zero within noise at both tiles, so the bound's compute term is the RIGHT
reference and extra sharded work is not the gap.

WHERE THAT LEAVES IT (quantified, one candidate standing): the residual
divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
versus **17.8 us** for the same collective measured in isolation — an in-
context cost 4-5x the best case. That is consistent with EXPOSED,
un-overlapped communication rather than raw wire time.

THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
If the residual were communication the scheduler is currently hiding work
behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:

| arm | nd2 | nd4 | vs default |
|---|---|---|---|
| default (LHS on) | 42.69 | 23.53 ms | — |
| `latency_hiding_scheduler=false` | 42.44 | 23.49 | **+0.6 % / +0.2 %** |
| `enable_pipelined_p2p=true` | 42.76 | 23.54 | -0.2 % / -0.1 % |
| CP combining @32 MiB | 42.55 | 23.59 | +0.3 % / -0.3 % |

Turning overlap OFF is free (marginally faster), and no scheduling flag
moves the step. The scheduler has nothing to hide the comm behind.

**MECHANISM (all three alternatives eliminated by measurement): the
residual is EXPOSED, DEPENDENCY-SERIALIZED SYNCHRONISATION.** Not bytes
(7.2x more = +0.22 ms), not sharded-formulation work (0 ms), not
hideable-by-scheduling (0 ms). It is the unavoidable cost of sync points
that sit on a dependent chain.

This retro-explains every earlier arm in the campaign, which is the check
that the mechanism is right rather than merely last-standing:
- fused-halo NULL — aggregation reduces message COUNT but not chain DEPTH;
- `single_reduce` HELPED (0.49 -> 0.53) — Chronopoulos-Gear restructures
  the recurrence into fewer DEPENDENT reduction batches;
- wide-halo HELPED MOST (-> 0.73) — it deletes the barotropic solver's sync
  points outright;
- the f64/f32 flip — more compute per sync point dilutes a fixed sync cost.

ACTIONABLE CONSEQUENCE: the lever for this lane is reducing the NUMBER OF
DEPENDENT SYNCHRONISATION POINTS, not message aggregation, byte
compression, or XLA scheduling flags — three families this campaign has
now measured to be null here.

Honest answer to "how far from the theoretical limit are we": 72-81 % of a
now-calibrated floor, with the shortfall attributable to neither bandwidth
nor byte volume.

## The mechanism's prediction, TESTED — and the lever it exposes (job 26459382)

If exposed dependent sync is the cost, PCG iteration count is the most
direct lever on it (each iteration carries dependent reduction batches).
`--pcg-fixed-iters` was wired onto the SPMD bench for this (the MPI twin
already had it) and swept at LL576 f64:

| iters | nd1 ms | nd4 ms | eff@4 | speedup@4 | residual (nd4) |
|---|---|---|---|---|---|
| 60 (default) | 63.55 | 23.52 | 0.68 | 1.00x | 2.1e-04 |
| 40 | 61.33 | 21.05 | 0.73 | 1.12x | 1.6e-03 |
| 30 | 60.24 | 19.89 | 0.76 | 1.18x | 4.6e-03 |
| 20 | 59.08 | 18.60 | 0.79 | 1.26x | 1.3e-02 |
| 10 | 57.99 | 17.31 | **0.84** | **1.36x** | 4.1e-02 |

QUANTIFICATION — and codex round-7 rates the strong form OVERSTATED, which
is recorded here rather than argued away. Regressing T(N) = intercept +
N x slope over the five iteration counts:

| | intercept | slope | R^2 |
|---|---|---|---|
| nd=1 | 56.87 +- 0.04 ms | 111.4 +- 1.0 us/iter | 0.99994 |
| nd=4 | 16.11 +- 0.09 ms | 123.8 +- 2.4 us/iter | 0.99972 |

WHAT THIS ESTABLISHES (codex objection (c), answered): the fit is
essentially exact and the intercept is tightly determined, so the cost is
genuinely PER-ITERATION, not a constant misattributed to iterations. But
the nd4 intercept EXCEEDS the measured 144-row compute term (14.63 ms) by
**1.48 ms**, so a fixed non-PCG overhead does exist and the 60 iterations
do NOT explain the entire residual.

CODEX OBJECTION (a) TESTED AND REFUTED (job 26459817). Rather than divide
the nd1 slope by 4, measure the per-iteration slope directly at each tile
on ONE device:

| tile | measured us/iter |
|---|---|
| 576 rows | 110.0 |
| 288 rows | 65.2 |
| 144 rows | **28.1** |

The linearity assumption predicted 110.0/4 = 27.5 us for the 144-row tile;
the MEASURED value is 28.1 us — 2 % apart. Per-iteration compute IS linear
in rows (the PCG iteration is a bandwidth-bound stencil+reduction, so it
scales with data even where the FULL step does not). The sync figure barely
moves: **95.7 us/iter measured** vs 96.3 assumed.

FULLY-MEASURED DECOMPOSITION OF THE nd=4 STEP — no inferences left:

| term | ms | note |
|---|---|---|
| same-tile single-device step (144 rows) | 14.65 | measured |
| + PCG dependent sync (60 x 95.7 us) | **5.74** | **65 % of the distributed overhead** |
| + other distributed overhead | 3.13 | 35 % |
| = total | 23.52 | measured 23.52 |

So of the 8.87 ms the step pays for being distributed across 4 GPUs, TWO
THIRDS is dependent PCG synchronisation and one third is everything else.
Codex objection (c) is honoured in the same table: the non-PCG part is
real, separated, and not attributed to the solver.

STRONG-SCALING CONSEQUENCE: cutting iterations raises 4-GPU efficiency from
0.68 to 0.84, because what is being removed is precisely the part that does
NOT shrink with device count.

THE CATCH, WHICH IS NOT MINE TO WAIVE: the speed is bought with solver
convergence — the zero-forcing probe residual degrades 200x from 2.1e-04 to
4.1e-02. 40 iterations (1.12x, 8x residual) and 30 (1.18x, 22x) are the
plausible operating points, but whether that residual is acceptable for the
free surface is an OCEAN-SCIENCE decision, not a performance one. Reported
as a trade curve; no default changed.

## The accuracy-free lever, and what it reveals (job 26460113)

Cutting PCG iterations trades accuracy. `single_reduce`
(Chronopoulos-Gear) attacks the SAME sync at unchanged iteration count by
halving the DEPENDENT reduction batches per iteration, so it should be
free. Measured at LL576 f64, slopes from a 60-vs-20-iteration difference:

| variant | nd | us/iter | step @60 | residual |
|---|---|---|---|---|
| standard | 1 | 111.0 | 63.54 ms | 2.079e-04 |
| standard | 4 | 117.9 | 23.40 ms | 2.079e-04 |
| single_reduce | 1 | 129.6 | 64.78 ms | 2.079e-04 |
| single_reduce | 4 | **99.7** | **22.21 ms** | **2.079e-04** |

**1.053x at 4 GPUs with a BIT-IDENTICAL residual** — a genuinely free win,
unlike the iteration cut. Note it is SLOWER at nd1 (129.6 vs 111.0 us/iter):
Chronopoulos-Gear buys fewer reductions with extra local vector ops, so it
only pays where sync dominates.

PREDICTION PARTLY WRONG, AND THE MISS IS THE INTERESTING PART. Halving the
reduction batches should have halved the 89.8 us/iter of sync; it fell only
to 66.9 (25 %). Backing out the arithmetic (and correcting for the +4.6
us/iter of extra local work at this tile) decomposes per-iteration sync:

| component | us/iter | behaviour |
|---|---|---|
| reduction sync | 45.7 | HALVED by single_reduce |
| halo sync | 44.1 | UNCHANGED — the matvec's own halo |

(check: 45.7/2 + 44.1 = 66.9, exactly the measured single_reduce value)

So per-iteration sync is almost exactly half global-reduction and half
nearest-neighbour halo. `single_reduce` can only ever address the first
half. The second half is the same quantity wide-halo removes for the
explicit solver — which is why wide-halo was the larger win in the earlier
arms, and it now has a mechanistic reason rather than just an empirical
ranking.

## Using the calibrated bound correctly (a trap worth documenting)

With the fabric constants supplied the bench flips `bound_calibrated=true`,
but `t_bound` stays null until a THIRD ingredient arrives:
`--single-dev-fused-ms`. Its contract is the nd=1 time **at the same
PER-DEVICE size** — for a strong ladder at nd=4 on LL576 that is a
144-row single-device run, not the 576-row one.

Passing the GLOBAL-size time (job 26457946) makes the compute term nd times
too large and yields `measured_over_bound` of 0.647 at nd2 and 0.358 at nd4
— i.e. the measurement beating its own lower bound, which is impossible and
is the tell that the ingredient was wrong. The bench computed exactly what
it was told; the misuse was the caller's.

Correct procedure (job 26457977): phase 1 measures nd=1 at each per-device
tile (576 / 288 / 144 rows), phase 2 feeds each ladder rung its MATCHING
compute term. Sanity rule for any future roofline: if measured/bound < 1,
the bound is wrong, not the code.

## Precision changes which ocean config wins (job 26457919)

The solver A/Bs that produced the "wide-halo wins" conclusion ran **f32**
(with vmix-f32). Re-running the LL576 ladder in **f64** narrows the gap
sharply:

| arm | nd1 | nd2 | nd4 | eff@2 | eff@4 |
|---|---|---|---|---|---|
| implicit_cn fixed-PCG | 63.56 | 42.71 | 23.61 ms | 0.74 | 0.67 |
| explicit + wide-halo | 71.54 | 43.69 | 23.31 ms | 0.82 | 0.77 |

Wide-halo still scales better (0.77 vs 0.67 at nd4) but is only 1.3 % faster
in absolute time there, versus the large margin measured in the f32 arm —
and it starts 12 % SLOWER at nd1. So the winning config is REGIME-DEPENDENT
(precision, tile size, device count), not universal. Anyone promoting the
improved config should pick the arm for the production precision, not
inherit the f32 verdict. These f64 rows are NOT comparable to the f32
multinode ladder quoted earlier; only their internal comparison is valid.

## Measured fabric constants for the roofline lines (job 26457495)

`scripts/bench/bench_ppermute_microbench.py` (new) times the SAME collective
the sharded steps use — `lax.ppermute` on a ring inside `shard_map` — over a
message-size sweep, so the SPMD benches can stop reporting
`bound_calibrated=false`:

| lane | devices | latency | bandwidth | vs line rate |
|---|---|---|---|---|
| NVLink (1 process) | 2 | 18.0 us | 53.98 GB/s | — |
| NVLink (1 process) | 4 | 17.8 us | 64.22 GB/s | — |
| NCCL over IB (8 procs, 2 nodes) | 8 | 26.3 us | 23.53 GB/s | **94 % of HDR200's 25 GB/s** |

The IB number landing at 94 % of line rate is the independent check that the
collective itself — not something else — is being timed.

HOST DISPATCH MUST BE SUBTRACTED, and this was nearly a self-inflicted
error: the FIRST version timed one jit call per exchange and reported
287-518 us "latency" (job 26457469) — two orders above what these fabrics
do, because dispatch dominates a single call. Feeding that into `t_bound`
would have produced a confidently WRONG roofline, strictly worse than the
uncalibrated generic line it was meant to replace. The tool now times 1 rep
vs n reps inside one jit and differences them; the removed dispatch
(287-529 us) is still reported alongside so the contamination stays
visible. Validated on CPU virtual devices: 275 us -> 26 us.

Quote the lane that matches the plot: intra-node NVLink and inter-node IB
differ by ~3x in bandwidth, so using the wrong one is its own confound.

## Infrastructure defects found + fixed (each with a receipt)

1. py3.14 argparse eager help validation — both ocean benches crashed at
   parser construction (bare `%`).
2. Slurm step gres non-inheritance — multinode lanes saw zero CUDA devices.
3. SLURM double-pin — CVD pin + jax's `local_device_ids=[SLURM_LOCALID]`
   compose to a nonexistent ordinal; route-B lanes now leave all node GPUs
   visible (`--gpu-bind=none`) and let jax bind LOCALID-th (Derecho/PALS
   keeps its shim + `local_device_ids=[0]`; conventions must not mix).
4. cs-spmd rank identity from the federated jax runtime (mpi4py loud-guard
   killed the NCCL cube lane that needs no MPI); launcher world size now
   global-only (PALS_LOCAL_SIZE dropped), step size preferred.
5. `make_sharded_ocean_step` replicated-geometry `device_put` asserts
   bit-equality across processes while per-process XLA autotune drifts
   ULPs (size-dependent!) — process-0 broadcast + allgathered
   tolerance-compared divergence guard (quantized-equality v1
   false-positived on a rounding boundary; v2 rtol=1e-5). Gates:
   equivalence 6/6 and selfspawn 2/2 re-run on EVERY iteration of the
   guard (last: jobs 26453906/26453981), plus live np4/8/16 multinode
   (jobs 26452743-45, 26453279).
6. Multicontroller host materialization in the tiled bench finiteness gate
   → on-device global reduce.
7. OUTDIR same-second stamp collision → job-ID suffix everywhere.
8. `setup_mpi_venv.sh` could not resolve uv-workspace members with plain
   pip → pins first + `install_federation.py --all`; mpi4jax source-built
   with the system toolchain (GLIBCXX mismatch with gcc-11-built OpenMPI
   module).
9. Diagnosis tool halo/overlap phases timed UN-JITTED eager pads
   (20.5e6 us per "exchange", bandwidth 0.0 GB/s; job 26447827) - FIXED
   this campaign (jit + dtype-correct bytes + refuse a bandwidth at
   world_size==1; overlap fractions >100% now refused), contract test
   tests/bench/test_halo_profiler_contract.py, verified 27-162 us at
   C24/L8 and 463-4791 us at C384/L60 (job 26454084). Census + scan phases
   were always sound (cube C96/L40: 0/7/14 CP/step at nd 1/2/3 + 1
   all-reduce, job 26447827; 24-dev tiled closed loop 384 CP + 1 AR,
   job 26450938).

## Closed levers (nulls with receipts — do not re-run)

NCCL_PROTO forcing (default already optimal; LL128 −8–12%), fused-halo on
the implicit arm AND on the wide arm (pad aggregation is not the residual),
`xla_gpu_collective_permute_combine_threshold_bytes` alone,
`--xla_gpu_enable_pipelined_p2p` alone, `LEGOESM_BAROCLINIC_F32` (~+0.5%, within run-to-run spread; job 26451282).
PGLE arm invalid as measured here (the 33-step window catches its
profile+recompile). The +8.5% figure is from the DERECHO lane-T campaign
(see the SOTA review), not reproduced on Levante — rerun long-window if
revisited.

## Still open (ranked) — refreshed at campaign close

1. **Wide-halo stability gates** → the blocker to promoting the improved
   ocean config (wide-halo + vmix-f32) beyond benches. It is worth 2.01x
   multinode and +0.15 weak efficiency, so this is the highest-value
   remaining item. Needs: averaging-filter stability under stale-halo
   substeps, plus a science sign-off on the f32 vmix solve.
2. **MPAS ico np4 per-device dip** — five hypotheses refuted by
   measurement (see above); needs a GPU op-level profile (nsys / XLA op
   profile of np4 vs np8). A scoped instrumentation project, not a knob.
3. **Calibrated theoretical-limit lines on every plot.** The ocean bench
   already computes a `t_bound` from a latency/bandwidth model but reports
   `bound_calibrated=false` because the latency and bandwidth inputs are
   placeholders. A dependency-matched comm microbenchmark would turn every
   ideal line from "linear" into a machine-specific roofline.
4. **Ocean wet-cell compaction + wet-balanced partitions** (~2x on
   ~40%-land grids, per `SCALING_STATUS_AUDIT.md` item 4) — untouched here.
5. **2-D lat-lon decomposition** at >=64 ranks: the 1-D band's perimeter
   ceiling is now measured (0.12-0.16 at np64 spread, vs ico's 0.52), which
   quantifies the prize.
6. **Milan np16 anomaly** in the packed CPU ladder (np16 slower than np8,
   recovering by np32) — spread ladder reduced but did not remove it.
7. Route-A CUDA-aware mpi4jax lane (`gpu_moist_scaling.slurm`) — only if a
   route-A-vs-B A/B is ever wanted; route-B beat every route-A reference
   available here.

DONE during the campaign (were open at the start): C768 same-path ladder
(eff 1.04) and tiled closed loop (14.3 GCells/s); atm lat-lon weak at a
production tile; ocean weak at a production tile; the ico CPU-MPI ladder;
the CPU spread ladder; and the diagnosis tool's halo + overlap phases,
which were found broken and fixed with a contract test.
