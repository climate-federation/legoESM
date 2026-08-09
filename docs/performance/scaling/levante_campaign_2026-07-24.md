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

RESOLVED 2026-07-26 (nsys job 26479922 + HLO dump 26480096 + sqlite
timeline): the dip is an XLA CODEGEN pathology, localized to named
kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
protocol (L8, padded-16 mesh: 21.01/17.78/7.17 ms at np2/4/8 vs campaign
19.90/17.09/6.92 — ~5% profiler overhead); (2) at np4 ONLY, giant
serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
instances are >1 ms at ~3.3 ms, the rest are the ordinary us-scale adds)
— and the sqlite timeline places all three groups' big instances at the
17.8 ms step cadence (stddev 78 us: deterministic compute, not comm
wait): 3.6 + 3.6 + 3.3 ~= 10.5-10.9 ms/step = the np4 excess; (3) in the optimized
step HLO these are mega-fusions ON THE HALO PATH: `%loop_add_fusion =
f32[491520,26]` (edge-tendency add chain, 22 operands incl. an
input_scatter_fusion) and `%loop_add_fusion.4 = f32[163844,26]` (cell
array), with the shard_map halo-pack concatenates taking the same adds +
parameter lists as operands. PLAUSIBLE (inferred from kInput fusion
semantics + operand lists, not separately timed): the emitter RECOMPUTES
the expensive scatter+add chain inside each consumer fusion, which is why
the cost multiplies. WHY np4: fusion cost-model decisions depend on the
shard shape; at np2/np8 the mega-fusion is not built. This also explains
why the earlier env-knob sweep missed it — autotune/latency-hiding flags
do not change fusion-pass decisions. Fusion-pass flag A/B at np4 ran
(job 26480162): flag route CLOSED — three of four candidate fusion flags
no longer exist in this XLA (upstream removals), the fourth is null, and
the GPU plugin does not list its flags via --help.

FIX ATTEMPTS, both measured (base 19.90 / 17.09 / 6.92 ms at np2/4/8):

| barrier placement | np2 | np4 | np8 | verdict |
|---|---|---|---|---|
| tendency INPUT side (job 26480261) | 21.40 | 16.58 | 7.01 | null at np4, -7.5% np2 — REVERTED |
| tendency OUTPUT side (job 26480310) | 22.03 | **14.09** | 7.04 | **-17.6% time np4** (1.21x), +10.7% time np2 — REVERTED |

(Single runs per arm; the campaign's np4 repeat spread (+-0.3%) supports
an informal ~+-1 pp error on these percentages, not a formal CI.)

The HLO frame table pinpointed the fusion: the 3.6 ms kernels resolve to
`pytree_ops.py:10` (`pytree_axpy.<locals>.<lambda>`) — the RK stage
combine mega-fused with the tendency graph's tail. An output-side
optimization_barrier recovers 3 ms of the ~10.9 at np4 but costs np2
10.7% (it also blocks fusion that HELPS there), so neither barrier ships
unconditionally. Parity + conservation smoke passed on both attempts.

STATUS: **FIXED, SHIPPED GATED** (codex rounds 11-12: strategy consult
BEFORE implementing, then post-review). `_FUSION_BARRIER_WORKLOADS` in
`sharded_dynamics.py` applies the tendency-output optimization_barrier
only at the measured workload signature (n_dev, edge rows, cell rows,
nlev) = (4, 1_966_080, 655_376, 26) — every operand trace-time static.
Verification ladder (job 26486288 vs same-day dead-gate 26486123):
np2 19.86 (campaign base 19.90 — at baseline), **np4 14.12 = -20.6%
same-day / -17.4% vs campaign base**, np8 6.99 (base 6.92). Parity +
conservation smoke green; 23 SPMD parity tests pass. Two instructive
misfires on the way, both caught by measurement: the first gate keyed
per-shard rows (never fired — the trace-time array is the GLOBAL view),
and edge-rows-only was over-broad (L8 edges are unpadded and divisible
several ways — codex round-12). The residual np4 gap to ideal (~14.1 vs
~9.9 from np2/2) is the un-barriered remainder of the fusion; further
recovery needs the integrator-level restructure (codex round-11 ranked
it last on blast radius) or an upstream XLA fix — both remain
follow-ups.

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

**REVISION 2026-07-27 — the subdiv-7 row was PLACEMENT-LIMITED, not
comm-limited.** Re-running it with `--distribution=block:cyclic` (the
Milan fix, discovered after this sweep) gives f64 np64 **efficiency 0.71,
up from 0.52**, and the high-rank columns move most: placement alone is
worth 2.00x at np16, 1.75x at np32, 1.38x at np64 (job 26495437 vs
26452579). The f32 ladder at matched placement (job 26495083) reaches
**0.88**. So the "np16 dip" visible across every row of this table is
substantially the same NUMA effect found later in the packed CPU atm
ladder — one fix, two symptoms. Precision itself is worth a near-constant
~1.4x here; the naive cross-job comparison would have read 2.81x at np16
and attributed placement to precision.
| 8 | 1.04 | 0.93 | 1.18 | 0.61 | 0.49 | — |

THE TILE-SIZE PATTERN, THIRD LANE (cube and ico are both atmosphere:
two components, three decomposition lanes): the np64 column collapses
on coarse grids (0.15 at subdiv4, 0.30 at subdiv5) and holds on fine ones
(0.52-0.57 at subdiv6-7). Same pattern as the cube (0.23 -> 1.04) and the ocean (0.37 -> 0.63), now
on a third lane and a different transport (CPU-MPI, not NCCL). It is the
campaign's most reproducible ASSOCIATION — but changing C-resolution, LL
size or ico subdivision also changes the global problem, so tile size is
not causally isolated.

Shape: ~1.0 through np8, one step down, then FLAT 0.5 from np16 to np64 —
4x more ranks with no further loss. CONSISTENT with a fixed per-rank cost
rather than growing communication, but flat efficiency alone does not
identify which; that needs phase-level timing. (The >1 points at np8 are the same
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

**MECHANISM (the three tested alternatives are not dominant): the
residual is best explained by EXPOSED, DEPENDENCY-SERIALIZED
SYNCHRONISATION.** These are eliminations of the TESTED implementations,
not of every possible communication explanation. Not bytes
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

DECOMPOSITION OF THE nd=4 STEP (both terms from measured slopes; the
remainder is left UNATTRIBUTED):

| term | ms | note |
|---|---|---|
| same-tile single-device step (144 rows) | 14.65 | measured |
| + PCG dependent sync (60 x 95.7 us) | **5.74** | **65 % of the distributed overhead** |
| + iteration-independent remainder | 3.13 | 35 % — NOT attributed; may include fixed PCG/setup work |
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

| component (MODEL-DERIVED, not separately timed) | us/iter | behaviour |
|---|---|---|
| reduction sync | 45.7 | assumed HALVED by single_reduce |
| halo sync | 44.1 | assumed UNCHANGED — the matvec's own halo |

This split assumes exactly two sync categories and a linear single_reduce
local overhead; the 120->4 census corroborates the COUNT reduction, not
this particular halo-time value.

(check: 45.7/2 + 44.1 = 66.9, exactly the measured single_reduce value)

So per-iteration sync is almost exactly half global-reduction and half
nearest-neighbour halo. `single_reduce` can only ever address the first
half. The second half is the same quantity wide-halo removes for the
explicit solver — which is why wide-halo was the larger win in the earlier
arms, and it now has a mechanistic reason rather than just an empirical
ranking.

## MATCHED CONFIG HEAD-TO-HEAD — the production recommendation (job 26460365)

All three arms in ONE job on ONE node, back to back, conservation-gated
(LL576x1152 L20 f64):

| arm | nd1 | nd2 | nd4 | eff@2 | eff@4 | residual |
|---|---|---|---|---|---|---|
| implicit + standard PCG (production) | 63.54 | 42.57 | 23.48 ms | 0.75 | 0.68 | 2.079e-04 |
| implicit + **single_reduce** | 64.81 | 41.63 | **22.09** | 0.78 | 0.73 | 2.079e-04 |
| explicit + wide-halo | 71.52 | 43.65 | 23.28 | 0.82 | **0.77** | n/a (different solver) |

THE TWO METRICS DISAGREE, and the distinction drives the recommendation:
`single_reduce` is FASTEST in absolute time at every count above 1 (1.063x
vs production, 1.054x vs wide-halo at nd4), while wide-halo has the BEST
EFFICIENCY — but only because it starts 10 % SLOWER at nd1, which flatters
a ratio normalised to its own single-device time. Efficiency is not speed.

THE CROSSOVER, MEASURED (jobs 26460444/45/48 + 26460501) — and my
first recommendation was right only for the corner I had measured.

Same-precision f32 ladder, both arms, LL576:

| nd | single_reduce | explicit+wide | winner |
|---|---|---|---|
| 1 | 35.01 | 36.45 ms | single_reduce (1.04x) |
| 2 | 24.09 | 22.48 | wide (1.07x) |
| 4 | 14.69 | 12.84 | wide (1.14x) |
| 8 | **15.89** | 11.04 | wide (**1.44x**) |
| 16 | 15.36 | 8.71 | wide (**1.76x**) |

Control at nd16 (job 26460877): standard PCG 17.57 ms, so the nd16 ordering
is wide 8.71 < single_reduce 15.36 < standard 17.57. The implicit variants
PLATEAU past 4 GPUs (single_reduce 14.69 -> 15.89 -> 15.36) and never
recover, while wide-halo improves monotonically (12.84 -> 11.04 -> 8.71).

Two things this shows that the f64 <=4-GPU ladder could not:
1. **single_reduce ANTI-SCALES past 4 GPUs in f32** (14.69 -> 15.89 ms from
   4 to 8) — it hits the sync wall, while wide-halo keeps improving all the
   way to 16.
2. **PRECISION FLIPS THE WINNER at nd=4**: single_reduce is 1.05x faster in
   f64, wide-halo is 1.14x faster in f32. Coherent with the mechanism —
   f64 carries more compute per iteration so sync is a smaller fraction and
   the low-extra-compute solver wins; f32 shrinks compute until sync
   dominates and wide-halo's zero-solver-sync takes over.

RECOMMENDATION (a decision table, not a winner):

| regime (all TESTED on LL576 only) | config | why |
|---|---|---|
| f64, 1 GPU | standard PCG | single_reduce is SLOWER here (64.81 vs 63.54 ms) — its extra local vector ops only pay once sync exists |
| f64, **2-4** GPUs | `pcg_variant="single_reduce"` | fastest in that range, residual bit-identical, one-line change, no scheme review |
| f32, **2-16** GPUs | explicit + wide-halo | wins from nd2 and the margin grows to 1.76x at nd16; the only arm still improving at 16 — CHANGES THE BAROTROPIC SCHEME, stability-gated |
| anything else | benchmark it | the flip depends on precision AND device count AND tile; do not extrapolate off this grid |

CORRECTION (codex round-8): an earlier revision said "f64 <=4 GPUs" and
"f32 OR >=2 GPUs", which was wrong at nd1 (standard is fastest in f64
there) and self-contradictory. "Bit-identical residual" is also a
convergence check, NOT full trajectory validation.

A single "best ocean config" claim would be wrong in one regime or the
other; the earlier campaign arms disagreed precisely because they sampled
different precisions and device counts.

## Wide-halo stability: evidence toward the gate (job 26460729)

Wide-halo is the campaign's fastest arm for f32 / scale-out but is BLOCKED
on stability review (the averaging filter under stale-halo substepping).
This does NOT clear that gate — a gate needs the filter analysis plus an
ocean-science sign-off — but it supplies the first thing a reviewer would
ask for: 600 steps at nd4, f32, conservation-gated, both arms same job.

| arm | eta drift [m] | heat_rel | salt_rel |
|---|---|---|---|
| implicit_cn (reference) | 6.01e-11 | 1.272e-05 | 4.878e-06 |
| explicit + wide-halo | 3.72e-10 | **1.248e-05** | **4.190e-06** |

REPEATED ACROSS THREE SEEDS (job 26464790) — the n=1 objection answered.
`--seed` was added to the bench for this; 3 independent IC perturbations
per arm, 600 steps each:

| metric | implicit_cn | explicit+wide | separation |
|---|---|---|---|
| heat_rel | 1.2693e-05 +-4.6e-08 | **1.2427e-05** +-4.6e-08 | **7.1 SE** (wide 2.1 % lower) |
| salt_rel | 4.8570e-06 +-3.6e-08 | **4.1483e-06** +-7.2e-08 | **15.2 SE** (wide 14.6 % lower) |
| eta drift | 7.72e-10 +-1.1e-09 m | 6.75e-10 +-6.6e-10 | 0.1 SE — INDISTINGUISHABLE |

With a variance estimate the heat and salt ordering IS established (7 and
15 standard errors — small effects, but far outside the seed spread), so
wide-halo genuinely conserves those two better in this configuration. It
also CORRECTS the single-run report above: wide-halo's apparently WORSE eta
drift was noise, and vanishes at n=3.

SCOPE, still: one grid, one base stratification (seeds vary only the IC
perturbation), UNFORCED, 600 steps, f32, nd4. This does not clear the
stability gate — that needs the filter analysis under stale halos and a
science sign-off — but "no observed failure" has become "marginally better
conservation with a measured variance estimate".

Per-step time is flat 100 -> 600 steps (12.81 -> 12.74 ms). Both arms' heat
drift grows ~linearly and is similar between them, consistent with the
shared baroclinic/tracer path dominating it.

MECHANISM CONFIRMED FROM A THIRD ANGLE. The wide-halo arm records its own
message census: **120 standard barotropic messages/step -> 4** (n_loop=30
substeps, stencil reach 3, one fixed wide exchange per chunk). A 30x cut in
barotropic exchanges is precisely why it wins where sync dominates, and it
is the SAME quantity the single_reduce analysis isolated as the half it
could not touch (44.1 us/iter of matvec halo). Three independent
measurements — the iteration sweep, the single_reduce decomposition and
this census — now agree on what the cost is.

THE GATE, RE-EXAMINED AGAINST THE CODE (2026-07-26) — the campaign's
framing was backwards. "Wide-halo needs stability gates" conflated the two
things the arm changed:

1. **wide-halo is the SAME explicit-substep operators with
   tolerance-parity coverage at every transport tier** (codex round-9
   wording), not an untested scheme variant. Coverage:
   serial `tests/ocean/unit/test_barotropic_wide_halo.py` — parity atol
   1e-12 f64 across four configs (div-damp, power-law filter, multi-chunk),
   volume-drift parity 1e-15, NaN-sentinel stencil-reach pin; re-run
   2026-07-26: 11 passed, 1 skipped (mpi4jax-gated dispatch test).
   Distributed: `tests/ocean/distributed/test_ocean_mpi_wide_halo_parity.py`
   (MPI gathered-vs-serial, 1e-10, incl. rank-cut/v-face/chunk cases) and
   `tests/parallel/test_latlon_ocean_spmd_wide_halo.py` (4-device SPMD,
   2e-4/1e-3). These are TOLERANCE parity, not bit identity — "the filter
   sees bit-identical inputs" is too strong; differences are XLA
   re-association at the serial tier and larger at the distributed tiers.
   ONE REAL BEHAVIOURAL DELTA to disclose: wide-halo requires LOCAL
   subcycle clamping, and with an active `eta_floor` the clamp/
   redistribution schedule differs from the standard path — a reviewer
   should check that config interaction, not filter stability in general.

2. **The scheme choice — explicit_substep vs implicit_cn — is a choice
   between two EXISTING schemes, not validation of a new one.**
   `barotropic_solver = "explicit_substep"` is the MODEL DEFAULT
   (`ocean/state.py:910`), the NEMO-standard split-explicit free surface
   (dynspg_ts-style filter, shared `barotropic_common.py` weights). NOTE
   the bench does NOT consume that default — `bench_ocean_latlon_spmd_
   scaling.py` explicitly defaults to `implicit_cn`, and the OMIP
   global-overturning drivers explicitly set `implicit_cn` too. The
   default supports "explicit_substep is an established scheme", nothing
   more. The promotion question remains experiment-level: "may the OMIP
   config switch that field for scale-out runs".

WHAT ACTUALLY REMAINS (scoped to that question): the OMIP override to
implicit_cn presumably encodes a preference (dt headroom / stiffness at
depth on that config). The remaining sign-off is experiment-level: run the
OMIP case with explicit_substep+wide at production dt and confirm the
3-seed conservation result (heat 7.1 SE, salt 15.2 SE lower than
implicit_cn at 600 steps unforced) holds under forcing. That is a science
review of ONE config field on ONE experiment, not a scheme-stability
program.

## Scale-out + the plateau question (2026-07-27)

The figure showed plateaus at high device counts on several grids. Codex
round-14 rejected the obvious "just run bigger ladders" plan — it MAPS a
plateau without IDENTIFYING it — and prescribed matched pairs instead:
at fixed device count vary the tile, and at fixed tile vary the device
count. Only the second contrast can show a genuine comm/N effect.

**CPU-MPI lane, first verdict (job 26495929, ico f64, block:cyclic
throughout, 4 nodes, single runs).** Aligning both meshes by CELLS PER
RANK rather than rank count:

| cells/rank | subdiv-7 | subdiv-8 |
|---|---|---|
| 10 200 | — | np64 = 960 ms |
| 5 100 | — | np128 = 526 |
| **2 600** | **np64 = 165** | **np256 = 356** |
| 1 300 | np128 = 92 | (np512 pending) |
| 600 | np256 = 56 | — |
| 300 | **np512 = 67 (REGRESSES)** | — |

Both meshes still scale at 2 600 cells/rank; subdiv-7 keeps gaining down
to 600 and only ANTI-SCALES at 300 (56 -> 67 ms). **The turnover tracks
work per rank, not rank count** — the tile-floor hypothesis, confirmed
on a lane where the two can be separated. Practical consequence: rank
counts beyond the campaign's old 64 ceiling keep paying as long as
resolution rises with them; subdiv-8 reaches 356 ms at 256 ranks, a
count the campaign never previously tested.

CAVEATS: single runs, no repeats. The np64 point here (165 ms) is FASTER
than the same configuration measured on 1 node earlier (220.6 ms, job
26495437) because this job spreads 64 ranks over 4 nodes — the
node-spreading effect the campaign already documented; ladders are
internally consistent but the two jobs are not interchangeable.
The subdiv-8 np512 arm OOM-killed at 128 ranks/node (every rank derives
the global mesh); rerun spread over 8 nodes as job 26497704.

**GPU CEILING FOUND — the cubed-sphere cannot currently exceed 54 GPUs.**
The tiled cube path is bit-identity-validated only at kt=2 (24 devices)
and kt=3 (54) (`sharded_dynamics.py:754`); kt=4 (96) falls back to
REPLICATING the global state, which is what killed the 96-GPU attempt
(job 26495955: "byte size of input/output arguments (83247045120)
exceeds the base limit"), and very likely the f64 cube retry that hit
the walltime (26495388). This is a VALIDATION limit, not a hardware one,
and it is the single biggest blocker to atmospheric scale-out: 28 idle
GPU nodes were available and unusable by that lane. Matched triangle
resubmitted inside the validated counts (job 26497294): C768@24 (147.5k
cols/GPU anchor), C768@54 (65.5k), C1152@54 (147.5k — same tile as the
anchor at 2.25x the devices).

The lat-lon band decomposition has no such ceiling; its matched pair
runs at 64 GPUs (job 26497323): LL720@16 and LL1440@64 both hold 64.8k
columns/GPU, with LL720@64 (16.2k) as the sub-floor control.

**Cube tile-floor arm, measured (job 26497294):** C768 L60 from 24 to 54
GPUs = 19.33 -> 13.93 ms, **1.39x at 2.25x devices, efficiency 0.62** —
and the tile only falls to 65.5k cols/GPU, still well ABOVE the ~30k
floor. So unlike the CPU lane, the cube's loss here is NOT explained by
the tile floor alone; there is real device-count cost to quantify.
(Note the same-job C768@24 anchor reads 19.33 ms where the campaign's
figure carries 14.09 ms for C768@24 — different lane/protocol between
those jobs, so only the within-job 24-vs-54 contrast is used.)

**THE CUBE PLATEAU, IDENTIFIED (job 26498347).** The fixed-tile contrast
finally ran inside working configs — C512@24 vs C768@54, both 65.5k
cols/GPU:

| arm | tile | devices | ms/step |
|---|---|---|---|
| C512 kt=2 | 65.5k | 24 | 14.85 |
| C768 kt=3 | 65.5k | 54 | **13.95** |

**2.25x the devices carrying 2.25x the problem costs nothing** (1.06x, in
the model's favour) — so communication does NOT grow with device count on
this lane, and the strong-scaling loss is not a comm wall. Cross-job
reproducibility is excellent: C768@54 reads 13.93 (job 26497294) and
13.95 (26498347), 0.1 % apart, which licenses combining the two jobs.

Solving the two fixed-device points for the per-device cost model:

    t(tile) = 11.27 ms FIXED + 54.6 us per 1k columns

At the production 147.5k tile the fixed term is already **58 %** of the
step, and at 65.5k it is **76 %**. This is an Amdahl ceiling, not a
network one: from a 24-GPU C768 base the step can never beat ~11.3 ms
**however many GPUs are added** — a cap of 1.72x, of which the measured
24->54 run already collected 1.39x. CONDITIONAL (codex round-15): this
comes from a TWO-POINT fit and assumes the 11.27 ms term is constant as
tiles shrink and device count rises. A perimeter-like halo term would
FALL with tile size while collective latency could RISE with rank count;
the 2.25x fixed-tile contrast supports only "little growth over the
tested range", not universality. That single number explains
the cube's efficiency 0.62, the empirical tile floor, and the plateau in
the figure.

**THE SAME STRUCTURE ON LAT-LON.** Two fixed-device (16 GPU) points —
LL720 at 64.8k cols/GPU = 4.19 ms and LL1024 at 131k = 5.73 ms — give

    t(tile) = 2.68 ms FIXED + 23.2 us per 1k columns

so the fixed term is 64 % of the step at 64.8k and 47 % at 131k: the same
fixed-cost-dominated structure as the cube, but roughly **4x smaller in
absolute terms** (2.68 vs 11.27 ms). That is consistent with lat-lon
scaling further before plateauing, and it makes the cube's 11.3 ms look
like a lane-specific overhead rather than something intrinsic to the
hardware or to SPMD. CAVEAT: these two points come from different jobs
(26497323, 26498463) — same lane, protocol and day, but not the same-job
contrast the cube pair enjoyed.

**LAT-LON FIXED-TILE CONTRAST, MEASURED (job 26502539)** — 131.1k
cols/GPU on both sides:

| arm | devices | cells | ms/step |
|---|---|---|---|
| LL1024x2048 | 16 | 4.2 M | 5.73 |
| **LL2048x4096** | **64** | **218.1 M** | **6.73** |

4x the devices carrying 4x the problem costs **+17 %** — weak-scaling
efficiency **0.85**, sustaining **32.4 GCells/s (506 Mcells/s/GPU) on
218 million cells**, the campaign's largest atmospheric run by an order
of magnitude. So communication grows only weakly with device count here
too (the cube's equivalent contrast was free at 2.25x; lat-lon pays 17 %
for 4x). Neither lane is comm-limited at these counts — both are limited
by the per-step fixed cost above.

Scale-out receipts on this lane: LL1536x3072 at 64 GPUs = 4.97 ms (job
26498266) and the LL2048 point above.

**WHAT THE FIXED TERM IS — ATTRIBUTED (nsys job 26504836): the halo
exchange, scaling with tile PERIMETER.** Profiling both tiles at the
SAME 24 GPUs isolates it by subtraction:

| tile | NCCL time / 12 steps | launches | share of GPU time |
|---|---|---|---|
| C768, 147.5k cols/GPU | 498.7 ms | 1452 | 66.5 % |
| C512, 65.5k cols/GPU | 280.7 ms | 1128 | 65.4 % |

The comm term grows **1.78x for a 2.25x larger tile AREA** — close to the
sqrt(2.25) = 1.50x a PERIMETER law predicts, and nowhere near the 2.25x
an area law would give. That is the mechanism behind the fitted "fixed"
11.27 ms: halo cost tracks the tile EDGE while compute tracks the tile
AREA, so under strong scaling compute falls off faster than communication
and the comm share rises until it dominates. It is NOT launch overhead
and NOT host synchronisation.

This also reconciles with the fixed-tile contrast (comm does not grow
with DEVICE COUNT at constant tile, 1.06x for 2.25x devices): both
statements are true because the comm cost is set by the tile geometry,
not by how many devices exist.

**LANE-MISMATCH CORRECTION (found while reading the kernel names).**
That profile omitted `--closed-loop`, so it characterised the SINGLE-SHOT
adapter lane (49.95 ms at C768), whereas the 11.27 ms fixed-cost model
was fitted to CLOSED-LOOP runs (19.33 ms). I had attributed the 2.6x gap
to nsys overhead; most of it is the lane difference. Consequences:
* the perimeter ratio (1.78x for 2.25x area) is a within-lane ratio and
  remains valid FOR THE SINGLE-SHOT LANE;
* whether it explains the CLOSED-LOOP fixed term is NOT yet established.
A matching closed-loop profile is running (job 26507936).

WHAT THE KERNEL NAMES SHOW (single-shot lane, both tiles, 24 GPUs):

| kernel | C768 | C512 | per step |
|---|---|---|---|
| `ncclDevKernel_SendRecv` (halo) | 311.8 ms / 1440 | 186.7 / 1116 | ~120 launches |
| `AllReduce_Sum_f32_RING_LL` | 186.9 ms / **12** | 94.0 / **12** | **exactly 1** |

The per-step all-reduce averages **15.6 ms at C768** and scales 1.99x
with 2.25x tile area.

**SOURCE MIS-ATTRIBUTED — corrected (codex round-15).** I cited the
`(1,)` reshape in `tiled_production_cdgrid.py:2508` as the psum behind
it. That is WRONG: that reshape lives in the closed-loop communicator
WARM-UP, and the single-shot adapter explicitly REFUSES the mass fixer
(`tiled_step_adapter.py:102`). So the origin of the per-step all-reduce
in the profiled lane is **unidentified**, and my "scalar psum, therefore
pure barrier" chain does not hold as stated.

What survives: a 15.6 ms all-reduce IS compatible with early ranks
spinning until a late rank arrives, but that is a hypothesis, not a
measurement. The discriminating test (codex round-15) is per collective
instance:

    arrival skew = latest NCCL-kernel start - earliest NCCL-kernel start

ms-scale skew with microsecond-scale service time on the last-arriving
rank confirms the barrier reading; aligned starts implicate the
collective or the scheduler instead.

**CLOSED-LOOP PROFILE (job 26510470) — the production lane, isolated
per-step with a marker kernel** (`loop_add_fusion_3`, exactly one per
timed step; naive time-windowing was still catching setup and XLA
autotune `RedzoneAllocatorKernel`, so a marker was required):

| per step (mean of 3 inter-marker intervals) | ms |
|---|---|
| wall | 17.11 |
| GPU kernel time | 15.08 |
| **of which NCCL SendRecv (halo)** | **6.65 (44 %)** |
| largest compute fusion | 0.90 |

So in the lane the cost model was fitted to, the halo exchange is **6.65
ms/step, 44 %** — substantial, but NOT the 66 % the single-shot lane
showed, and NOT the whole 11.27 ms fixed term. The remaining ~8 ms is
spread across many small compute fusions (the largest is 0.90 ms), which
is the signature of a LAUNCH-BOUND step rather than one dominated by any
single kernel.

The per-step `AllReduce_Sum_f32_RING` that dominated the single-shot
trace is ABSENT here (only tiny TREE all-reduces, 0.046 ms). It was an
artifact of the single-shot lane, which is exactly why the lane mismatch
mattered — the intervention it suggested would have targeted a
collective the production lane does not issue.

REVISED READING of the 11.27 ms fixed term: roughly half is halo
exchange (perimeter-scaling, consistent with the single-shot ratio) and
roughly half is many small compute kernels whose launch overhead does
not shrink with tile size. Both halves point at the same two fixes —
fewer/fatter halo messages, and kernel fusion to cut launch count.

CONSEQUENCE FOR REACHING THE LIMIT: the lever is halo-cost-per-tile, not
device count. Options in order of expected value: (a) larger tiles
(raise resolution — already shown to work), (b) fewer/fatter exchanges
per step (the ocean's wide-halo trick, 120 -> 4 messages, applied to the
cube), (c) overlapping halo exchange with interior compute.

**A THIRD cube limit — f64 is effectively unrunnable at 24 GPUs.** The
tiled lane (the CORRECT >6-GPU vehicle) failed to complete even C384 L60
in f64 within a 3 h wall (job 26512794), where the same arm in f32 runs
in ~9 ms/step. No output, no error — it never finished compiling. That
is consistent with #1370: if setup allocates global-sized buffers, f64
doubles them, and the compile/allocation path degrades accordingly. So
the cube's f64 GPU column stays EMPTY in the figure, and it is a
capability gap rather than a measurement I skipped.

**A SECOND cube ceiling, memory:** the fixed-tile arm C1152 L60 @54
wanted **105.7 GB per device** (rematerialization stuck at 96.6 GB)
against 80 GB of A100 HBM — job 26497294. So cube scale-out is bounded
twice over: validation caps tiles at kt=3 (54 GPUs) and HBM caps
resolution at that tile count. The fixed-tile comm contrast is
resubmitted at L30 for BOTH arms (job 26497736), which halves the
working set while holding 147.5k cols/GPU on each side.

## The resolution lever is CAPPED on both transports — and my subdiv-9 runs were invalid

The campaign's cure for every plateau is a larger tile via higher
resolution. Testing that at the largest rank counts failed on BOTH
transports, for two DIFFERENT reasons:

**CPU (atm icosahedral): subdiv-9 is not supported by the generator.**
Jobs 26512349 and 26514768 did not time out in mesh construction as I
assumed — they raised immediately:

    ValueError: subdivision_level=9 would create 2.62e+06 cells.
    Maximum supported level is 8 (655,362 cells). For higher
    resolutions, use load_mpas_mesh() with a pre-built mesh file.
    (voronoi.py:1242)

MY ERROR: I submitted two multi-node jobs at an unsupported level
without checking the generator's range, and the error message even names
the alternative. This is the THIRD time in this campaign that a library
or bench guard stated the answer before I ran the job (the etopo dt
warning and the n_lat divisibility check were the others). No pre-built
finer mesh is present in the tree, so testing beyond subdiv-8 requires
sourcing an MPAS mesh file first.

**GPU: blocked by the global-allocation defect (#1370)** — LL2304 wanted
102 GB/device, C1152 97-106 GB.

**CONSEQUENCE, and it is the campaign's sharpest practical finding:**
raising resolution is the ONLY measured cure for the tile-floor plateau,
and it is currently unavailable on both transports — capped at subdiv-8
on CPU by the mesh generator, and by per-device global allocation on GPU.
So the useful rank/device ceilings measured here are NOT hardware limits:

| lane | useful ceiling | what caps it |
|---|---|---|
| atm ico CPU | ~512-1024 ranks at subdiv-8 | mesh generator caps at subdiv-8 |
| ocean MPAS CPU | 512 ranks at subdiv-8 | same generator cap + rank-count term |
| atm/ocean GPU | 64 GPUs at LL1536-2048 | #1370 global per-device allocation |
| cube GPU | 54 GPUs at C768 | kt validation (#1360) + #1370 |

Unblocking #1370 and sourcing a subdiv-9+ mesh are therefore worth more
than any further tuning: both lanes have headroom that is currently
unreachable.

## Cube optimisation: bounded BEFORE implementing — and the bound killed the plan

Directive was to push the cube toward its limit. Codex round-16 defined
the strategy first (per the standing pre-implementation rule) and its
cheapest-bound step then REFUTED the intervention I was about to build.

**Codex corrections to my reading:**
* My "88 SendRecv / 4 rounds = 22 exchanges" was wrong. The kt=2 tile pad
  emits **16 phases per logical scalar pad** (4 edges + 4 guards + 4
  diagonals + 4 corner slivers) for serial-exact offset/corner handling
  (`cubesphere_exchange.py:1276`); the RK body pads dp/B/zeta/invT/lnps/T
  separately, the vector pad calls the scalar pad twice, x3 RK stages.
* Vertical batching is ALREADY done (4-D pads carry all L60 levels), so
  it cannot remove launches — my own payload arithmetic had hinted at
  this (strips ~41x larger than a single-field depth-1 strip).
* XLA already fuses much of it: ~384 source permutes become ~101
  optimized HLO ops and 88 traced SendRecv. So field packing's ceiling
  is the 6.65 ms NCCL time, NOT the 11.27 ms fixed term.
* A once-per-step STALE deep halo (the ocean's trick) **changes answers**
  on this dycore — it alters RK2/RK3 boundary tendencies. Exact
  communication avoidance would need a 3 x radius = 6-cell overlap with
  cube-edge interpolation, and the tiled transport supports only halo
  1/2. That is a new algorithm, not a port.

**THE BOUND (one step, 86 SendRecv, latency floor 17.8 us measured):**

| class | n | total |
|---|---|---|
| <=25 us (latency-bound) | 39 | **0.61 ms** |
| 25-100 us | 33 | 1.20 ms |
| >100 us (payload/wait) | **14** | **7.54 ms** |

Field packing removes LAUNCHES, so its absolute ceiling is the
latency-bound class: **0.61 ms of a 17.11 ms step = 3.6 %**. Not worth
the change.

**What the tail actually is.** The largest single exchange is **4.86 ms**.
At the measured 64.2 GB/s NVLink that would be ~310 MB, but the entire
per-step halo volume is ~8 MB. So that call is not moving data — it is
WAITING. Fourteen exchanges holding 7.54 ms is arrival skew absorbed at
halo sync points, the same signature codex flagged for the single-shot
all-reduce.

**CONCLUSION: the cube's fixed cost is dominated by rank-arrival skew,
not by launch count or bytes.** Packing fields, batching levels and
fattening messages all target the wrong term.

**SKEW LOCATED (4-rank profile, job 26526100).** Aligning SendRecv
launches by sequence across ranks 0/1/12/23 over two steady steps:

* 10 of 172 exchanges carry >1 ms of max duration, and **9 of the 10 are
  skew-dominated** — arrival skew ~= max duration (e.g. 13.6 ms skew vs
  12.6 ms duration with the last arriver's own service at 14 us: early
  ranks WAIT the full skew).
* The ordering is SYSTEMATIC: rank 23 arrives last 95/140 times, rank 0
  44/140; rank 12 arrives FIRST 135/140. Median idle gap before a late
  arrival is 20 us — the late rank was computing back-to-back, not
  blocked upstream.
* BUT total per-rank work is EQUAL: compute 8.2-8.5 ms/step in ~428
  kernels on every rank.

Equal totals + systematically late at fixed sequence points = **pipeline
drift, not load imbalance**: the 16-phase pad sequence has
rank-dependent participation (partial permutes let non-target ranks run
ahead), the drift accumulates within the step, and the ~10
full-participation exchanges act as resync barriers where the
accumulated drift is paid as wait. The cost is real (~7.5 ms/step in the
wait tail) but the remedy is ALGORITHMIC — reorder/merge pad phases so
drift cannot accumulate, or overlap the resync exchanges with interior
compute — a scoped dycore-scheduling follow-up, not a bench or config
change. No further profiling is needed; the mechanism chain
(launch-count -> bytes -> skew -> ordering) is now measured end to end.

## Tripole (ORCA fold) past 4 GPUs — first receipts (job 26512798)

The fold is the ocean's production topology but had only ever been
measured to 4 GPUs. LL1152x2304 L20 f32, implicit_cn + forced PCG:

| devices | tile | ms/step | GC/s |
|---|---|---|---|
| 8 | 331.8k cols/GPU | 33.97 | 1.56 |
| 16 | 165.9k | **23.79** | 2.23 |

8->16 = 1.43x, **efficiency 0.71** — the fold scales respectably at
production tiles, and the topology is not a scale-out blocker at these
counts.

SCOPE, stated because it is tempting to misread: the regular-grid
LL1152@16 number elsewhere in this report (19.83 ms) used
**explicit_substep + wide-halo**, whereas this tripole ladder used
**implicit_cn + PCG**. Those are DIFFERENT SOLVERS, so the pair licenses
NO fold-cost claim. The only licensed fold cost remains the earlier
same-job matched contrast (+1.2-3.7 %, job 26493837).

## The MPAS mesh cap — lifted (subdiv-9 unblocked for 128 GPUs)

The generator's hard subdiv-8 cap was the CPU-side resolution blocker
and made 128-GPU MPAS floor-starved by construction (subdiv-8 at np128 =
5.1k cells/GPU). Chain shipped 2026-07-30 (codex round-19 design,
commit 70f3ce636):

* **Cache-or-prewarm policy** for subdiv 9-10: a cache hit always loads;
  a miss RAISES with prewarm instructions unless the process is the
  designated single builder (opt-in env + per-key O_EXCL lockfile with
  stale takeover — the opt-in alone would be a thundering herd across an
  MPI launch). >10 stays hard-refused. Six policy tests + prewarm CLI
  (`scripts/data/prewarm_voronoi_mesh.py`) with its direct test.
* **lloyd=0 admitted as a LABELLED synthetic scaling mesh** after the
  quality gate codex specified: identical topology to the production
  SCVT, area CV 0.084 vs 0.061, 128-part imbalance 1.148 vs 1.095 (~5 %)
  at subdiv-6. Scaling receipts only, never physics. This turns the
  subdiv-9 prewarm from ~5 h (lloyd=50, measured 87 s/iteration at
  subdiv-8) into ~1 h.
* **subdiv-9 prewarmed** (job 26549180): 2,621,442 cells in 67 min,
  1.8 GB npz in project space — ranks load in seconds forever after.
* **subdiv-8 control arm at 64/128 GPUs (job 26538474)**: f32
  5.27 -> 6.47 ms and f64 8.96 -> 11.41 across np64 -> np128 —
  ANTI-scales exactly as the tile law predicts at 10.2k -> 5.1k
  cells/GPU. MPAS *runs* at 128 GPUs; subdiv-8 just has nothing left to
  parallelise there. Full f32 ladder np2->128 (np32 from job 26549646):
  19.90 / 14.12 / 6.92 / 7.10 / **8.13** / 5.27 / 6.47 — NON-MONOTONE:
  np32 is WORSE than np16 while np64 is the minimum, and f64 shows the
  same pattern (np32 14.26 vs np64 8.96). This is the np4-dip signature
  at another count — count-specific codegen/fusion behaviour layered on
  the tile floor (the fusion pathology on this lane is already proven
  shape-dependent). Recorded as observed; not chased further at subdiv-8
  since the mesh is below the floor at all these counts anyway.
* Payoff ladder submitted (job 26549775): subdiv-9 at 32/64/128 GPUs =
  81.9k/41.0k/20.5k cells/GPU — the first MPAS many-GPU ladder whose
  lower rungs sit ABOVE the ~30k floor.

OPERATIONAL NOTE: a Lustre incident mid-implementation left the module
with an undefined constant on disk for ~12 h; two queued jobs (mpas32
26534061, and possibly mpas128's first attempt) died on that NameError
window and were resubmitted post-fix. The git index inode went stale on
the login node's client (kernel-hung D-state git processes); recovery =
rebuild the index on a fresh inode and route git through a compute
node's healthy Lustre client.

## Ocean GPU scale-out to 64 devices — and a CROSS-LANE memory defect (#1370)

| arm | devices | tile | ms/step |
|---|---|---|---|
| LL1152x2304 L20 | 16 | 165.9k cols/GPU | 19.83 |
| LL1152x2304 L20 | 64 | 41.5k | **11.19** (4.75 GC/s) |
| LL2304x4608 L20 | 64 | 165.9k | **OOM — 102 GB/device** |

The strong arm reaches 64 GPUs (19.83 -> 11.19 ms, 1.77x for 4x devices,
eff 0.44 — the tile falls to 41.5k, near the ~30k floor, so this is the
floor behaving exactly as the atmosphere's does).

**The fixed-tile arm could not run, and WHY it could not is the finding.**
LL2304 at 64 GPUs asked for **102.04 GB per device**. The per-device
SHARD is 3.3 M cells — about 0.2 GB for fifteen f32 fields. But the
GLOBAL problem is 212 M cells = 12.7 GB per field-set, and ~8 such
buffers is ~102 GB: an exact match. The allocation tracks the GLOBAL
size, not the shard.

This is the SAME signature as the cube's C1152 wall (105.7 GB at L60,
97.5 GB at L30 — only 8 % for halving the levels, so level-INDEPENDENT).
Two independent lanes, one defect class: **SPMD setup materialises
global-sized buffers per device, so RESOLUTION is capped regardless of
device count.** Both lanes shard correctly one size down (ocean LL1152
@64, cube C768 @54), so the sharded step is sound — it is the
setup/allocation path.

**OCEAN FIXED-TILE CONTRAST, recovered by sizing under the wall (job
26510472).** Instead of retrying LL2304, LL1632 @32 holds the anchor's
tile (166.5k vs 165.9k cols/GPU) at HALF the global size:

| arm | devices | tile | ms/step |
|---|---|---|---|
| LL1152x2304 | 16 | 165.9k | 19.83 |
| LL1632x3264 | 32 | 166.5k | **19.55** (5.45 GC/s) |

**2x the devices carrying 2x the problem costs NOTHING** (0.99x,
weak-scaling efficiency 1.01). So the ocean GPU lane joins the
atmosphere: comm does not grow with device count at constant tile
(cube 1.06x at 2.25x, lat-lon 1.17x at 4x, ocean 0.99x at 2x). **All
three GPU lanes are tile-limited, none is device-count-limited** over
the tested ranges.

**#1370 DIAGNOSED (probe job 26523157, after two harness failures the
CPU smoke could not catch).** Matched 3.3M-cell shards at 2x global size
(LL1152@16 vs LL1632@32):

| signal | @16 | @32 | reading |
|---|---|---|---|
| compiled entry args | 0.06 GB | 0.06 GB | step is CLEAN |
| compiled temps | 0.22 | 0.21 | not remat pressure |
| state leaves (per-device) | 0.070 sharded / 0 replicated | same | sharding correct |
| **bytes_in_use** | **1.58 GB** | **3.11 GB** | **tracks GLOBAL size** |

Per-device residency is a constant **~7.4 global-field equivalents** —
and 7.4x the LL2304 field size is the observed 102 GB wall. So the
defect is NOT step-entry replication (my original hypothesis — refuted
in its specific form) and NOT remat: it is **SETUP-TIME global device
arrays that stay alive after sharding** — the globally-built initial
state, the vertex-mask cache primed FROM the global state, and the
replicated geometry stacks. The cube's level-independent wall fits: its
setup residency is mesh tables + 2-D geometry.

**FIX STAGE (i) SHIPPED AND MEASURED** (commits e1b502000/e5a541c65 +
probe 26524423): building the global model/state under
`jax.default_device(local cpu)` at nd>1 drops per-device residency
**1.58 -> 0.30 GB (@16) and 3.11 -> 0.71 GB (@32) — a 5x reduction** —
with the compiled step unchanged and parity at 1e-10. Getting there
burned four probe attempts on real multicontroller facts, each recorded:
lower/compile is COLLECTIVE (rank-0-only deadlocks the shutdown
barrier); `jax.devices()` is the GLOBAL list under jax.distributed (use
`local_devices`); `JAX_PLATFORMS=cuda` unregisters the cpu backend; and
the host-side build needs `--mem=0` or the SLURM cgroup kills it.

**STAGE (iii) SHIPPED — ACCEPTANCE MET** (commit 57494f2de, probe
26526284): sharding the band-geometry stacks P("lat") removes the
residual. Per-device residency is now **0.10 GB at BOTH probe sizes —
ratio 1.00, meeting codex's pre-registered <= 1.10 exactly**. Full arc:
1.58/3.11 GB (before) -> 0.30/0.71 (host-side build) -> **0.10/0.10**
(sharded stacks): a 16-31x reduction, residency now independent of
global size. All 13 SPMD gate suites (equivalence/tripole/wide-halo)
pass. One diagnostic casualty, harmless to production: the probe's
OUTER re-jit now refuses ("closing over a multi-process jax.Array"),
because the wrapper closes over the now-sharded stacks — the production
inner jit receives them as ARGUMENTS and is unaffected (the probe's own
step invocation ran). Remaining acceptance: the LL2304@64 wall run.

WHY THIS IS THE CAMPAIGN'S MOST IMPORTANT BLOCKER: the measured cure for
every plateau is a LARGER TILE, i.e. raising resolution as devices are
added. This defect makes that impossible — adding GPUs cannot buy
resolution — so every GPU lane is pinned at the tile floor. Filed as
**#1370** with the arithmetic; distinct from #1360 (the cube's kt
validation ceiling), and validating kt=4 alone would NOT unblock C1152.

## Ocean MPAS Voronoi scale-out — and a lane that does NOT obey the tile law

Controlled ladder, **32 ranks/node fixed** so nodes scale with ranks and
per-rank bandwidth is constant (job 26505286, f64, block:cyclic):

| ranks | subdiv-7 | cells/rank | subdiv-8 | cells/rank |
|---|---|---|---|---|
| 32 | 190.22 ms | 5 120 | 861.25 ms | 20 480 |
| 64 | 147.65 | 2 560 | 494.89 | 10 240 |
| 128 | 102.93 | 1 280 | 309.05 | 5 120 |
| 256 | **65.71** | 640 | **254.41** | 2 560 |

32->256 efficiency 0.36 (s7) and 0.42 (s8) — the lane reaches 256 ranks
but does not approach the limit.

A PROTOCOL FIX FIRST: the previous ladder (26504842) let srun fill nodes,
so np32 ran half-full (32 ranks/node) while np64-256 ran full (64) —
per-rank bandwidth changed along the ladder, and efficiency appeared to
RISE as tiles shrank, which is impossible. Fixing ranks-per-node changed
every number (s7 np32 159.67 -> 190.22 ms).

**THIS LANE BREAKS THE CELLS/RANK LAW that the atmosphere obeys.** At the
SAME 5 120 cells/rank, s7@np32 costs 190 ms but s8@np128 costs 309 ms —
1.63x more for 4x the ranks at identical per-rank work. On the atmosphere
icosahedral lane, per-doubling speedup depended on cells/rank ALONE. So
ocean-MPAS carries a genuine device-count cost the atmosphere lane does
not, and raising resolution will NOT rescue it the way it does elsewhere.

**CODEX ROUND-15 2x2 — BOTH effects are real, and separable** (jobs
26508258/26508336). Both cells hold 5 120 cells/rank; measured load
imbalance is near-identical (owned max/min within 6 %), so this is a
partition-QUALITY contrast, not a load-balance one:

| partition | s7 / 32 ranks | s8 / 128 ranks | B/A |
|---|---|---|---|
| geometric | 190.07 ms | 313.17 ms | **1.65x** |
| sfc | 232.46 | 617.66 | 2.66x |

* **Genuine device-count cost:** at the BEST partition, 4x the ranks at
  identical per-rank work still costs **1.65x**. Not partition quality,
  not load imbalance — a real rank-count term the atmosphere lane does
  not have.
* **Partition quality degrades WITH rank count:** sfc costs 1.22x at 32
  ranks but **1.97x** at 128. So the two effects compound, and the
  default `auto` is doing well to land near geometric.
* Practical: pin `--partition-method geometric` (or auto) on this lane;
  sfc is actively harmful at scale. pymetis is absent from `.venv-mpi`,
  so the low-cut METIS arm codex wanted is still unmeasured.

**512-RANK LADDER (job 26508063):** s7 63.83 ms, s8 194.80 ms.
s8 keeps gaining 256->512 (254.41 -> 194.80 = **1.31x**, at 1 280
cells/rank) while s7 goes flat (65.71 -> 63.83 = 1.03x, at 320
cells/rank — below the 300-600 floor). Same floor as the atmosphere,
reached at a different rank count because the mesh differs. So this lane
DOES scale to 512 ranks when the mesh is large enough; it just pays the
rank-count term on the way.

UNEXPLAINED, flagged not resolved: s7's per-doubling speedup RISES
(1.29 / 1.43 / 1.57) even in the controlled ladder. Efficiency improving
as the tile shrinks has no physical mechanism I can name; the likeliest
reading is that the np32 base point is anomalously slow (partition
quality at low rank counts?) rather than the high-rank points being
good. Under codex review (round 15) along with the discriminating
experiment for device-count cost vs partition degradation.

## Precision + grid-coverage verification (2026-07-27, user request)

**Mixed precision — the production storage mode (f64 state + f32
internals: `LEGOESM_VMIX_F32_SOLVE=1 LEGOESM_BAROCLINIC_F32=1`) — was the
untested corner of the decision table. Now measured** (job 26493592,
LL576 L20, same-job arms, conservation-gated 1e-5):

| arm | plain f64 nd1/nd4 | MIXED nd1/nd4 | nd4 mixed gain |
|---|---|---|---|
| implicit standard | 63.56 / 23.39 | 51.40 / 20.49 | +14.2 % |
| implicit **single_reduce** | 64.82 / 22.10 | 52.80 / **19.13** | +15.5 % |
| explicit + wide | 71.57 / 23.28 | 59.48 / 20.39 | +14.1 % |

Every arm gains ~14-15 % from mixed mode; the decision-table ordering is
OBSERVED UNCHANGED (single_reduce 6-7 % ahead at nd4, beyond the
informal ~+-1pp single-run noise — but that noise figure came from a
different lane, and the mixed arms are single runs; "wide is the
scale-out choice" is UNVERIFIED in mixed mode beyond nd4). Accuracy: NO
DIFFERENCE DETECTED IN THESE INDICATORS OVER THIS 33-STEP UNFORCED
HORIZON — heat/salt drift identical to plain (7.740e-10), zero-forcing
residual 2.067e-4 vs 2.079e-4; that is not trajectory or science
equivalence (codex round-13). **Best f64-storage nd4 config =
single_reduce + mixed at 19.13 ms (1.22x the production plain-standard
23.39).**

**MPAS np4 fix under f64 — the anomaly MOVES with dtype** (jobs 26493638
/26493734): f64 np4 is HEALTHY (eff 0.95; the f32-only gate correctly
does not fire) while f64 np8 ANTI-scaled (20.10 -> 21.42). Adding the
(8, ..., float64) signature entry — receipt: **np8 21.42 -> 18.98
(observed -11.4 %, single run, no CI; the unchanged np4 control at 20.09
supports specificity but does not quantify variance)**. The f64 ladder is now
monotone 38.34/20.09/18.98, and the shape x dtype dependence of the
fusion pathology is confirmed from a second angle (element size shifts
the pathological rung). Gate now carries dtype in the signature; both
entries have same-day receipts.

**Grid coverage — first receipts for the two unmeasured ocean grids**
(job 26493648, nd1/2/4, both precisions):
- **Tripole (synthetic ORCA fold): fold cost measured SAME-JOB (job
  26493837, codex round-13's demanded protocol): +3.7 % / +1.2 % /
  +1.2 % at nd1/2/4 vs the matched regular grid** (35.52/25.03/15.91 vs
  34.25/24.73/15.73 f32) — the fold's cost SHRINKS with device count,
  and "the fold is not a scaling bottleneck at these counts" is now
  licensed by a controlled comparison. (Cross-day numbers from
  26493648 retained above for the f64 points only.)
- **MPAS-ocean Voronoi: RETRACTION — my "ladders" (subdiv 6 AND the
  subdiv-7 discriminator, jobs 26493648/26493837) were INVALID.** The
  bench's own metadata says it: `n_ranks: 1, cells_per_rank_achieved:
  163842` — every arm ran ONE rank on the FULL mesh, because this bench
  decomposes by MPI RANK (its docstring states the SPMD multi-device
  path does not exist by design) and my CUDA_VISIBLE_DEVICES invocation
  never created ranks. The flat curves were the SAME single-device run
  repeated, not a latency floor and not a defect — codex round-13's
  "hypothesis, not verdict" was righter than it knew. What survives:
  single-device timings (s6 ~7 ms f32/f64, s7 ~21.9 ms f32).

  **THE REAL LADDER (job 26494036, CPU-MPI f64, np1-16, block:cyclic,
  ranks=N verified in metadata): MPAS-ocean SCALES.** s6 (41k cells):
  814.46 / 316.71 / 159.27 / 92.85 / 90.51 ms; s7 (164k cells): 3944.43
  / 1615.85 / 720.87 / 362.42 / 311.53 ms. The np1 base is
  cache-disadvantaged (np1->2 superlinear, same pattern as the atm
  spread ladder), so quoting np2-base efficiencies: s6 2->16 = 0.44,
  **s7 2->16 = 0.65** — the tile-size pattern reproduces on a FOURTH
  lane (bigger mesh holds efficiency deeper), and the np8->16 flattening
  sits exactly where per-rank cells fall to 2.5k (s6) vs 10k (s7).
  Single runs, no repeats; ordering claims only.
- Remaining coverage gap, flagged not measured: atmosphere SPECTRAL has
  no scaling receipts on any transport (global-transform lane, x64 by
  policy).

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

## OPERATIONAL NOTE: transient multi-node hangs (3 occurrences)

Three times this campaign a multi-node GPU job consumed its entire
walltime without emitting a timed row, then ran normally on retry with the
IDENTICAL configuration:

| job | config | hung for | retry |
|---|---|---|---|
| 26457000 / 26456335 | ocean LL1152 np8 | 90 min x2 | 99 s (job 26457693) |
| 26460447 | ocean LL576 np16 single_reduce | 50 min | 67 s (job 26460876) |

Each time the log stops during tracing/compile with no error. Twice I
suspected a real compile-time defect (a chunk-heuristic cliff, then a
16-device solver problem) and twice the retry refuted it. TREAT A SINGLE
MULTI-NODE HANG AS TRANSIENT until a second occurrence with the same
config; budget a retry rather than a diagnosis. Root cause not
established — it is not reproducible enough to bisect, and it has never
produced a WRONG number, only a missing one.

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

1. **OMIP config sign-off for explicit_substep+wide** (formerly "wide-halo
   stability gates" — reframed 2026-07-26 after codex round-9). Wide-halo
   has tolerance-parity coverage at all three transport tiers (serial
   1e-12, MPI 1e-10, SPMD 2e-4/1e-3) and explicit_substep is an
   established scheme (the model default; benches and OMIP explicitly
   choose implicit_cn). Remaining: the eta_floor x local-clamp config
   interaction, the OMIP case at production dt under forcing, and a
   science sign-off on the f32 vmix solve. Worth 2.01x multinode.
2. **MPAS ico np4 per-device dip** — five hypotheses refuted by
   measurement (see above); needs a GPU op-level profile (nsys / XLA op
   profile of np4 vs np8). A scoped instrumentation project, not a knob.
3. ~~Calibrated theoretical-limit lines~~ **DONE later in this campaign**:
   `bench_ppermute_microbench.py` measured the fabric constants (NVLink
   17.8 us / 64.2 GB/s, IB 26.3 us / 23.5 GB/s) and the roofline sections
   above use them. Kept here only so the list's numbering stays stable.
4. **Ocean wet-cell compaction + wet-balanced partitions** — SPLIT
   2026-07-26 into a cheap half and an expensive half:

   *Cheap half — wet-BALANCED bands* (no indirection, uneven band heights
   equalizing OCEAN cells per rank): already implemented on the MPI lane
   (`bench_ocean_mpi_scaling.py --wet-balance`, ETOPO continents); A/B at
   np16/np32 running (job 26480448, r128/r256 at CFL-scaled dt 100/50s).
   OPERATIONAL TRAIL kept for honesty: four earlier submissions failed —
   wrong venv (26479815), then non-finite at dt 600 and 300 (26479904/
   26480203/26480298), briefly mis-read as a lane defect until the
   bench's own WARNING surfaced: realistic coastlines are documented to
   need dt<=150 at LL96, which I had not read; the isolated-basin
   hypothesis tested along the way was refuted (`fill_isolated_basins`
   made no difference, consistent with dt being the real cause).
   MEASURED (job 26480448, r128 ETOPO
   CFL dt=100s, f64, np16/np32, single runs): wet-balancing LOSES —
   equal-rows 37.89 / 30.50 ms vs wet-balanced **47.39 / 44.20 ms**
   (25-45 % SLOWER). The mechanism is coherent with the gather
   microbench's finding: this lane computes DENSE arrays (a land cell
   costs the same as a wet one), so per-rank cost tracks TOTAL ROWS, and
   equalizing WET cells makes total rows uneven — it balances the wrong
   quantity. Wet-balancing could only pay on an implementation whose cost
   tracks wet cells (i.e. compacted), and the expensive-half measurement
   below shows compaction itself does not pay at real wet fractions.

   VERDICT on the audit's item 4 as a whole: BOTH halves measured, BOTH
   negative on this codebase — the "~2x on ~40%-land grids" projection is
   refuted twice over (gather penalty eats the compaction saving at 0.71
   wet; wet-balanced bands worsen dense-compute balance). The item is
   CLOSED as not-worth-building, with receipts. (Also moot for the SPMD
   lane: jax equal-shard sharding would need padding to the max band,
   returning exactly the imbalance removed.)

   *Expensive half — gather/scatter compaction: MEASURED, and the audit's
   "~2x" is REFUTED* (`bench_gather_vs_slice_stencil.py`, job 26479884,
   A100 f32, correctness self-checked). Per-cell gather penalty for a
   5-point Laplacian vs the dense sliced version: **1.40-1.77x**, so
   compaction wins only when wet_fraction < 0.56-0.72 (size-dependent).
   At the REAL global-ocean wet fraction (~0.71), packed-gather is a net
   LOSS on the full LL576 grid (ratio 1.16) and a wash at the nd4 tile
   (1.03). SCOPE (codex round-10): the two numbers are END-MEMBER ESTIMATES, not
   a bound — 0.86x is the pure-stencil member (measured), 1.41x the
   pure-column ideal; a real step also pays packing/scattering at the
   interface, sees real wet topology (not banded), richer stencils, and
   communication, none of which the microbench prices. What survives
   regardless: the audit's 2x assumed zero indirection cost and is
   refuted; at ~0.71 wet the measured stencil member is a net LOSS. VERDICT: do not build compaction for the global latlon ocean;
   revisit only for a configuration that is genuinely <~55% wet.
5. **2-D lat-lon decomposition** at >=64 ranks: the 1-D band's perimeter
   ceiling is now measured (0.12-0.16 at np64 spread, vs ico's 0.52), which
   quantifies the prize.
6. ~~Milan np16 anomaly~~ **RESOLVED 2026-07-26 (job 26479904): rank
   placement.** Discriminator at FIXED np16, r128 moist f64 — the script
   holds resolution/physics/precision/levels/timing fixed and varies ONLY
   the srun distribution: `block:block` 186.83 ms vs `block:cyclic`
   **87.59 ms — 2.13x from the distribution flag alone.** Leading
   interpretation (codex round-10 scoping): per-socket memory-bandwidth
   contention on the 2x Milan 7763 node, consistent with the
   spread-ladder reduction — but per-rank NUMA-binding receipts and
   bandwidth counters were NOT captured, so the mechanism is inferred
   from the placement swing, not instrumented. (Note 32 single-core ranks
   FIT in one 64-core socket, so np32 is not automatically two-socket.)
   FIX regardless of mechanism: `--distribution=block:cyclic
   --cpu-bind=cores` on packed CPU lanes.

7. **1-D bands vs 2-D pencils at np64 (job 26479904): the pencil path
   is 1.37x faster** (latlon r256 moist f64, same dt: 253.93 -> 185.03
   ms/step) — BUT this is NOT a pure decomposition A/B (codex round-10):
   `--latlon-2d` selects the wall-pole 2-D path while the band path
   keeps the atmospheric pole-fold, so boundary semantics change along
   with the decomposition. Report as a regular-vs-wall-pole path
   throughput result; attributing the 1.37x to decomposition alone would
   need a pole-matched A/B. Remaining: pole-matched ladder (np32-128);
   ocean lane pencil refusal stands (`test_2d_pencil_layout_refused` —
   wide-halo is 1-D-only by design).
7. Route-A CUDA-aware mpi4jax lane (`gpu_moist_scaling.slurm`) — only if a
   route-A-vs-B A/B is ever wanted; route-B beat every route-A reference
   available here.

DONE during the campaign (were open at the start): C768 same-path ladder
(eff 1.04) and tiled closed loop (14.3 GCells/s); atm lat-lon weak at a
production tile; ocean weak at a production tile; the ico CPU-MPI ladder;
the CPU spread ladder; and the diagnosis tool's halo + overlap phases,
which were found broken and fixed with a contract test.

## Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)

Both jobs the dropped session left behind COMPLETED; neither had been
analysed. First read-out below, CORRECTED per codex round-20
(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
in the first draft's weak-scaling claim).

### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)

Matrix at a NOMINAL MEAN target of 5,120 cells/rank (the JSONL's
`cells_per_rank_achieved` is global floor division —
`int(mesh.nCells) // n_ranks`, bench_ocean_mpas_scaling.py:751 — NOT a
balance statement; method pinned per arm, never `auto`, which flipped
meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64,
nlev 20, 32 ranks/node. Actual per-rank OWNED ranges
(`metadata.partition_metrics.cells_per_rank_min/max`): geometric
5,120–5,121 at BOTH scales; metis 5,100–5,145 @32 and 5,093–5,144 @128
(±0.5 %). WET load is looser still under metis — per-rank wet
cell-levels min/max: geometric@32 100,740–102,420; metis@32
96,800–102,720; metis@128 **78,000–102,880** (one rank 24 % under the
mean) — METIS balances owned cells (approximately), not wet cells, on
this bathymetry.

| arm | config | ms/step |
|---|---|---|
| A | s7 np32 geometric, block:cyclic | 189.82 |
| B | s8 np128 geometric, block:cyclic | 308.96 |
| C | s7 np32 metis, block:cyclic | 191.99 |
| D | s8 np128 metis, block:cyclic | 333.39 |
| E | s8 np128 metis, block:block | 537.91 |

* **This METIS configuration LOSES to geometric at s8/np128** (D/B =
  +7.9 %), despite the better offline cut (partq s8@np128: edge_cut
  1.89 % vs 2.01 %, halo mean 592 vs 629). Scale-out term (s7@32 ->
  s8@128, which crosses 1 -> 4 nodes as well as 4x ranks — NOT a pure
  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
  -> step-time inference FAILS on this lane; part of metis's loss is
  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
  out partition/mapping improvements generally (e.g. wet-cell-weighted
  METIS was NOT tested).
* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
  a byte-identical partition). NOTE the second `--distribution` field is
  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
  identically; the swing is socket-level. Mechanism (per-socket
  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
  2.13x receipt; never instrumented with bandwidth counters.
* Caveats: timing-only receipt — no parity/conservation gate ran in
  these arms, and the CPU nodes emit `UCX WARN transports
  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
  timing, but a "production config" claim would need a gated arm).

### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)

f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
reads `unknown` — same allocation, same submitted script, so the same
binary is PLAUSIBLE but that row stays non-reproduction-grade on its
own (codex r20/r21):

| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
|---|---|---|---|
| 32 | 81.9k | 12.47 | 5.47 |
| 64 | 41.0k | 9.60 | 7.10 |
| 128 | 20.5k | 11.48 | 5.94 |

* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
  its np64: 655,362 natural cells x 26 lev / 5.27 ms).
* Strong 32->64 speedup 1.299 (eff 0.65); 64->128 speedup 0.836 —
  ANTI-scales at 20.5k cells/GPU. CONSISTENT WITH the ~30k floor seen on
  the other lanes (single unreplicated point on a lane with known
  count-specific codegen variation — not by itself proof).
* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
  efficiency 0.55–0.74" pairs and the "~1.4x per 4x ranks GPU rank-count
  term".** Confounds: (a) every existing s8 receipt is the generator's
  default PRODUCTION Lloyd mesh, while s9 is lloyd=0 — different mesh
  family, not the same protocol; (b) two comparator points came from the
  np2-16 ladder (jobs 26454476/26454618), not the np32-128 extension
  rows; (c) the three ratios are 1.80/1.35/1.41 — not "consistent
  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
  no weak-scaling direction is claimed until it lands.

### Next receipts submitted 2026-08-02

1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
   after — BRACKETED, not fully counterbalanced; a penalty's attribution
   to fabric vs placement/drift needs the per-step nodelist table +
   follow-up). steps=5000 so the stepping window
   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
   wall-clock brackets logged as overlap evidence. CONFIRM bar:
   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
   >10 % = a CO-EXECUTION penalty, quantified per replica — its
   attribution (fabric contention vs placement/topology vs drift) is a
   follow-up, not a conclusion of this job.
2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
   recomputed only from these.

### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)

LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
@64 row (job 26502539, f32 6.73 ms):

| arm | ms/step | GC/s (col-levels) |
|---|---|---|
| f32 @128 (65,536 cols/GPU) | 5.5767 | **39.11** |
| f64 @128 | 9.6015 | 22.72 |

f32 strong 64->128: 1.207x for 2x devices (eff 0.60) with the tile at
**65.5k cols/GPU — comfortably ABOVE the ~30k floor** (codex round-21
caught the first draft halving this), so the loss is NOT
floor-attributable. Mechanism OPEN — candidates (uninstrumented): 1-D
band thinning to 16 rows/rank raising halo/compute ratio, and the
16 -> 32-node NCCL topology step. 39.11 GC/s (from 5.5767 ms) is the
highest measured throughput of ANY lane in the campaign. The companion
oc128 (26534067) FAILED pre-#1370-fix with the 109.5 GB resident-args
signature; retry submitted post-fix (below).

## Hundreds-of-devices push (user directive 2026-08-02)

"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
partition effectively unbounded for our rank counts. Submitted set:

| job | what | devices | why |
|---|---|---|---|
| 26628196 | s9 ensemble contention (v3; 26628021/26627810 superseded pre-start) | 128 GPU (4x32) | lever #1 receipt |
| 26628071->26636762 | oc LL2304 retry post-#1370 | 128 GPU | PREDICTION REFUTED: 26636762 failed with the byte-identical 109.5 GB args signature despite the fix being an ancestor and no fallback warning; @64 acceptance passed; see arg-linearity finding below |
| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |

s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.

### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)

r512 (512x1024 = 524k cols) L26 f64 moist, 32 rpn block:cyclic,
wall-pole 2-D pencil lane (labelled; NOT the pole fold):

| ranks | cols/rank | ms/step | speedup vs np64 | eff |
|---|---|---|---|---|
| 64 | 8,192 | 297.57 | 1.00 | 1.00 |
| 128 | 4,096 | 161.03 | 1.848 | 0.92 |
| 256 | 2,048 | 72.06 | 4.129 | 1.03 |
| 512 | 1,024 | 44.78 | 6.645 | **0.83** |

Distribution verified against the masquerade trap: result rows carry
`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
count on this mpi4jax lane, not the world size). 128->256 is
SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache
transition on Milan (mechanism PLAUSIBLE, uninstrumented). End-to-end
64->512 eff 0.83 at 1k cols/rank: TIMING-ONLY evidence that the lat-lon
CPU lane scales into the hundreds. QUALIFIER (codex r22): the run logs
an out-of-tested-range mpi4jax==0.9.0 pairing ("may fail or produce
incorrect results", parallel/reductions.py runtime check) and UCX
VM_UNMAP warnings — no parity/conservation gate ran, so this ladder is
unvalidated timing evidence until a supported-stack rerun. (Exact pencil factorisations are not recorded in the
result JSON — only `decomposition: 2d`; a follow-up could add them to
the bench metadata.)

### s8 lloyd=0 de-confound ladder landed (job 26628076): the near-matched-tile scale-out cost persists without the Lloyd confound

s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
warmup 3 — configuration-matched to the s9 ladder (26600095); NOT fully
reproduction-grade: the s9 np32 row's git_sha reads `unknown`, and this
run's `7151d12a1-dirty` has no archived dirty-file manifest (the live
diff touched only doc+plot, which supports but cannot retrospectively
prove the bench path was untouched): **6.58 / 6.43 / 7.29 ms**.

Weak pairs (~4x cells with 4x GPUs — global ratio 3.9994 after both
meshes pad +126 cells; tiles NEAR-matched to 0.015 %:
81,936/81,924, 40,968/40,962, 20,484/20,481), SAME lloyd-0 family:

| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
|---|---|---|---|---|
| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |

* The falsifiability block's CONFIRM branch fires: ratios stay well
  above 1 with the known Lloyd-family mismatch REMOVED. (This does not
  prove the old confound "only" biased the size — these are
  unreplicated single runs from separate allocations, one comparator
  without row-level provenance; the confounded draft read
  1.80/1.35/1.41 vs 1.90/1.49/1.57 here, and the production-mesh s8
  np8 was 6.92 vs lloyd-0 6.58, -4.9 %, so the mesh family does shift
  absolutes.)
* Restated: at NEAR-matched per-GPU tile, ~quadrupling devices+problem
  costs 1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
  UNATTRIBUTED (PLAUSIBLE candidates unchanged: inter-node neighbour
  fraction growth, collective latency vs count, sfc partition-quality
  decay with parts; the metis receipt argues against pure
  partition-cut explanations, on the CPU lane at least).
* The non-monotone tile dependence of the ratio (largest at the
  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.

## Distance-to-modeled-limit: atm lat-lon GPU (2026-08-02, "near theoretical limit" directive)

Closed the bench's own honest-null bound gap (audit item 4) for the
lat-lon lane, using only repo instruments:

* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
  virtual-CPU forced-host-platform lowering of the REAL
  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
  the 1-D band structure check). Exact CP payload from compiled-HLO
  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
  lowering; GPU-side collective combining could change the executed
  count (metadata.py:214) — the bound is a MODEL.
* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
  Approximation, recorded: nd=1 includes pole tiles; the bias
  DIRECTION on the compute term is PLAUSIBLE-high, not proven
  (matters most for the compute-dominated f64 row).
* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):

| row | measured | t_bound (IB) | measured/bound |
|---|---|---|---|
| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
| LL2048@128 f32 | 5.577 | 1.894 | **2.94** |
| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |

(Codex r7 corrected the @128 f32 row: the first draft fed the slab-byte
lower bound into a table labelled exact-bytes — 1.848/3.02 was the
mixed-input artefact; with the exact 18,541,632 B payload the bound is
1.894 ms. r7 also independently RERAN the census at nd=128 — 41 CP + 1
AR confirmed at the target device count, not just extrapolated from
8/16 — and measured the f64 census directly: 9,270,800 B at n_lon=1024,
four 4-byte scalar CPs staying f32, so the x2 extrapolation was 16 B
high.)

* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE this MODEL** (2.35-3.20)
  — the eff-0.60 strong leg is not close to the fabric+compute MODEL
  (a heuristic, not a proven floor). The 2-node/8-process IB
  calibration is extrapolated to a 32-node/128-process communicator. Leading
  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
  (launch + schedule + stream sync) well above the raw 26 us fabric
  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
  small-collective regime. The model itself notes the serialized-latency
  vs overlap biases pull opposite ways; treat measured/bound as a
  consistency diagnostic, not proven headroom.
* **Lever test submitted (job 26630576)**: 4-run CP-combining A/B at
  LL2048@128 (A default / B combine-8MB / C combine+pipelined-p2p /
  A2 default repeat),
  same-job control + trailing A2 drift bracket. Interpretation limit:
  without a GPU post-pass CP census per arm, a null refutes THIS
  threshold/implementation, not combinable-CP count in general. The
  ocean-lane null for these flags came from a different
  implicit-PCG/dependency mix — not predictive for the atm lane either
  way.

## Distance-to-modeled-limit: MPAS GPU (2026-08-02 late)

Codex r9 caught the first census at the WRONG topology
(`reorder_target=nd` vs the rows' `--reorder-for 128`); numbers below
are from the topology-correct recensus (job 26636684, cells/dev
40,968/40,962 exactly matching the timed rows). The CP payloads are
edge-coloured max-round-padded STATIC buffers — byte sums are payload
shapes, not measured wire bytes. Wrong-topology values (33 CP/24.0 MB
@s8, 39 CP/30.6 MB @s9) are retained in the r9 transcript only.

Same treatment as the lat-lon lane (probe twin
`scripts/tmp/_probe_mpas_halo_census.py`; virtual-CPU lowering of the
REAL `make_voronoi_sharded_step`, one compile, count-asserted):

* **The MPAS CP count is nd- AND reorder-target-DEPENDENT** — 9 / 21 /
  30 at nd=4(s7) / 8(s6) / 16(s6) with reorder_target=nd (early
  wrong-topology probes, historical), and 24 / 33 at nd=16(s8) /
  64(s9) with the rows' actual reorder-for-128 — Voronoi
  neighbour-round schedules grow with parts, unlike the 1-D band's
  fixed 41. Each MPAS bound must census ITS OWN row topology.
* **s8@nd16, reorder-for 128 (the real s8-lloyd0 np16 row topology)**:
  24 CP + 1 AR per step, 23,860,920 B static payload at 40,968
  cells/dev L26 f32.
* **Same-tile nd=1 anchor (job 26635847)**: subdiv-6 lloyd0 GLOBAL
  mesh (40,962 natural cells ~= the 41k tile) on one A100: **2.010 ms**
  — the SERIAL same-size compute PROXY used by the model (codex r9: it
  is un-reordered, lacks the sharded halo-local max_lc/max_le rows and
  pack/scatter path, and runs the bench's serial-leg dt — model-grade,
  not "the compute term").
* **Bound for s8-lloyd0@np16 (measured 6.43 ms, job 26628076)**, IB
  constants 26.3 us / 23.5 GB/s, serialized-latency comm model:
  comm = 24 x 26.3 us + 23.86 MB / 23.5 GB/s = 0.631 + 1.015 = 1.647 ms;
  t_bound = max(2.010, 1.647) + 0.026 = **2.036 ms** (compute-limited);
  **measured/bound = 3.16** — the SAME ~3x regime as the lat-lon panel
  (2.35-3.20). Both directive lanes sit ~3x above their fabric+compute
  MODEL at healthy tiles.
* **s9@nd64, reorder-for 128**: 33 CP + 1 AR per step, 29,489,064 B
  static payload at 40,962 cells/dev — payload GROWS +24 % at matched
  tile vs s8@nd16 (more neighbour rounds with parts; PLAUSIBLE
  partition-quality decay; recorded).
* **Bound for s9@np64 (measured 9.60 ms, job 26600095)**:
  comm = 33 x 26.3 us + 29.49 MB / 23.5 GB/s = 0.868 + 1.255 = 2.123;
  t_bound = max(2.010, 2.123) + 0.026 = **2.149 ms**;
  **measured/bound = 4.47**.
* **Matched-tile gap decomposition (topology-correct census)**: the
  s8@16 -> s9@64 measured gap is +3.17 ms; the MODELED comm growth
  (CP 24->33 = +0.237 ms latency, payload +5.63 MB = +0.240 ms BW; net
  bound growth +0.113 ms at the max() edge) covers <= 0.48 ms =
  **~15 %**. The
  UNDER THE FIXED-IB MODEL, direct count/payload growth accounts for
  at most ~15 % of the gap; its ACHIEVED-runtime contribution is
  unresolved (the unmeasured per-CP effective overhead could itself
  scale with count — codex r9). PLAUSIBLE residual mechanisms
  (uninstrumented): per-CP effective overhead growing with rank count
  (4 -> 16 nodes), and jitter/straggler amplification across the
  dependency-chained CP syncs per step (33 + 1 AR at nd64); "more neighbours +
  partition-quality decay" for the byte growth is likewise PLAUSIBLE.
  The 26630576 CP-combining A/B tests the LAT-LON lane only; the MPAS
  exchange is a hand-rolled edge-coloured ppermute schedule, so an
  MPAS combining test would need its own arm (and may not be
  XLA-combinable at all).

### Transport discriminator (job 26638578): NCCL runs NET/IB + GDRDMA — the gap is real

The early_init "no NCCL net plugin visible -> likely TCP sockets"
warning fired on every multi-node lane; NCCL_DEBUG=INFO on the real
bench (atm latlon @8 over 2 nodes) shows **258 NET/IB channel lines,
including NET/IB/…/GDRDMA, and zero NET/Socket** — NCCL's BUILT-IN IB
verbs transport is active (no plugin needed; the warning is a
plugin-visibility heuristic written for Derecho and is a FALSE ALARM on
Levante at this scale — softening it is a follow-up). SCOPE (codex
r11): this proves the 2-node/8-GPU case; it makes socket fallback on
the 32-node/128-GPU runs UNLIKELY (same env, same stack) but does not
directly measure them, nor rule out topology-scale transport effects
there. The mechanism hunt (per-CP effective overhead / scheduling)
continues. Bonus receipt: LL512x1024 @8 over 2 nodes =
3.60 ms (probe row, steps 12).

### s10@128 blocked by the #1100 remainder (job 26630207)

FAILED with `byte size of input/output arguments (162,319,564,800)
exceeds the base limit` — the MPAS MESH is still global per process
(state is partition-local since #1100; the mesh's SFC-partition-local
construction is the documented open remainder, quoted in the bench
source). At subdiv-10 the global geometry stacks alone exceed XLA's
arg limit at 128 GPUs (OBSERVED); 192/224 are EXPECTED to remain
blocked pending partition-local mesh construction — engineering
follow-up, not an env bug. The hundreds-of-GPUs MPAS
receipt therefore stands at subdiv-9/128 GPUs.

### CP-combining A/B: three deadlocks on one node block; pivoted to @64 with exclusion

Jobs 26630576 / 26638830 / 26640037 (LL2048@128 A/B) all hung in arm A
on the SAME 32-node block (l50000...l50187) and timed out with zero
receipts. Evidence collected in place: all 64 IB ports ACTIVE; NCCL
per-rank debug (v4) shows ALL 128 ranks reach "Connected all rings"
then every python idles at ~1 % CPU with memory preallocated — a TRUE
DEADLOCK after NCCL init (plausibly the pre-loop global barrier);
/dev/shm on l50000 clean (stale-segment hypothesis REFUTED); no stray
processes. The SAME code path ran LL2048@128 fine on 2026-07-30 (job
26534060, different nodes) and MPAS s9@128 on 2026-08-01 — so
block-correlation is PLAUSIBLE (cause unknown; candidate: switch-level
path issue invisible to port state). Excluding all 32 leaves only 24
a100_80 nodes, below a 32-node ask — so the lever test moved to
**LL2048@64 (16 nodes, block excluded), job 26641073**: the gap is
2.35x there, and a >=10 % combine win at @64 confirms the lever without
the sick block. If a later job hangs on the block again -> DKRZ ticket
with this section as the evidence.

## ENSEMBLE RECEIPT (job 26628196): lever #1 CONFIRMED — co-execution penalty <= 0.6 %

steps=5000 protocol (window ~70 s; absolute ms NOT comparable to the
steps-12 ladder rows by design). Same-job arms, disjoint 8-node sets,
per-step nodelists logged:

| arm | nodes | ms/step | GC/s |
|---|---|---|---|
| solo_pre | l50042-l50072 set | 13.86 | 4.92 |
| rep1 (concurrent) | l50042-l50072 set | 13.92 | 4.90 |
| rep2 (concurrent) | l50000-l50039 set | 13.86 | 4.92 |
| rep3 (concurrent) | l50115-l50187 set | 13.97 | 4.88 |
| solo_post | l50042-l50072 set | 13.92 | 4.89 |

* **max(replica)/mean(solo) = 13.97/13.89 = 1.006** (0.58 % max
  observed slowdown; the 1.10 bar is cleared 17x over): NO DETECTED
  co-execution penalty WITHIN THIS THREE-REPLICA EXPERIMENT. Aggregate
  ~2.994x solo for K=3. Caveats (codex r11): only rep1 shares the solo
  brackets' node set (rep2/rep3 comparisons include placement
  differences), and K=4+ linearity is untested (rep0's set failed).
* Consequence (lever #1, scoped): at K=3 the measured aggregate is
  ~14.7 GC/s from 96 GPUs vs 5.9 single-trajectory at 128 — replicas
  deliver ~3x more science throughput in this experiment. Extrapolation
  to K=4 (~19.6 GC/s) is PLAUSIBLE, not measured (the 4th set failed
  for unrelated node reasons).
* rep0 (4th replica) ABORTED at 7:05 on nodes l[50075,50078,50081,
  50100,50103,50106,50109,50112] with a Shutdown-barrier
  DEADLINE_EXCEEDED. Probe 26641282 REPRODUCED failure on rep0's
  exact 8 nodes while rep2's 8 passed — this identifies a FAILING
  8-node ALLOCATION vs a passing comparator; it does not by itself
  name a single node, nor prove the same cause as the LL2048@128
  hangs (a shutdown-barrier deadline is a different observable than
  the all-idle hang). Single-sick-node attribution = PLAUSIBLE
  hypothesis under bisection (halves queued); DKRZ ticket once
  narrowed.

### CP-combining A/B verdict (job 26641073, LL2048@64, sick block excluded): threshold null — no >=10 % benefit detected

A 6.740 / B combine-8MB 6.682 (-0.86 %) / C combine+pipelined 6.689
(-0.76 %) / A2 6.648 (-1.37 %) ms. B and C sit inside the single
A<->A2 control bracket: **no >=10 % benefit detected at @64 for these
flags**. Per the pre-registered interpretation limit this is a
threshold/implementation null (one control bracket, no GPU post-pass
CP census per arm) — not a general refutation of CP combinability. Reproduction note:
arm A matches the 26502539 receipt (6.732) at 6.740 across
days/node-sets — the lane is highly repeatable.

**Standing bottleneck picture for the ~2.4-4.5x-above-model gaps at
healthy tiles**: transport fallback UNLIKELY (NET/IB+GDRDMA proven at
2-node scale), no detected win from CP-combining at this threshold,
bytes censused, partition quality / placement / NCCL protocol
receipted null.
Remaining candidates are structural (per-CP effective launch/sync
overhead chains, jitter amplification across dependency-chained
collectives) — the arXiv:2607.16100 device-side collective API class,
NOT reachable from today's XLA/NCCL stack. Together with the ensemble
receipt (co-execution penalty 0.6 %), the campaign's operational
conclusion: single-trajectory strong scaling at small-to-mid tiles is
at the practical limit of the current stack; throughput past the floor
comes from replicas, resolution, or upstream/runtime work.


### Ocean @96 bracket (job 26642771): compiler-reported args total is NEAR-LINEAR in device count

@96 failed with args = 82,386,616,320 B; @128 showed 109,565,706,240.
Ratio 1.3299 vs nd ratio 128/96 = 1.3333 (0.3 %). The per-device-
STACKED-arguments interpretation (~858 MB per device entry at LL2304
L20) is PLAUSIBLE — the observed fact is the near-linear
compiler-reported total; the bufdump will confirm or refute the
stacking. Under that interpretation the earlier "@64 acceptance PASS" is
REINTERPRETED: 64 x ~858 MB ~= 54.9 GB simply sits UNDER the 63.8 GB
XLA arg limit — the #1370 fix reduced device RESIDENCY but did not
remove the per-device-stacked args from the program signature.
PLAUSIBLE candidates for the stacked list (uninspected): the per-band
vertex-mask cache or a per-device geometry stack. Next instrument: the
existing arg/buffer dump probe (scripts/tmp/_bufdump_compileonly.py +
oc128_bufdump.sbatch — both currently hard-coded to 128 virtual
devices, to be ADAPTED to @96) to NAME the arguments — then a targeted fix (shard-local slices as args, or
donate/rematerialise) unblocks oc at 96-224 GPUs.

## The deadlock node is NAMED: l50100 (bisect complete) — and the multicontroller ocean lane is FIXED

Bisect chain (each rung the same s9-32GPU probe config): the 32-node
block -> ensemble rep0's 8 nodes (26641282 reproduced) -> {l50081,
l50100} (26642924 passed / 26642925 failed) -> **l50100** (26645625
passed with l50081 aboard / 26645627 failed with l50100 aboard). Four
independent reproductions; all IB ports ACTIVE; NCCL rings connect;
the hang is post-init — a wedged-node class invisible to port state.
OPERATIONAL: exclude ONLY l50100 now (31 nodes return to service);
DKRZ ticket draft = this section.

### Multicontroller ocean: THREE stacked walls found and fixed (codex r14-r19)

1. **Geometry-stack broadcast** (nd x 849 MB psum program) — removed;
   per-band fingerprint gate + unit tests (PR pending).
2. **jax device_put's whole-array assert_equal** ([n_proc, field] on one
   device: 54.3 GB fits at 64 procs, 81.5 GB > A100 at 96) —
   _addressable_shard_put at 5 setup sites + exact-hash
   assert_pytree_bytes_equal preserves the bit-identity contract.
3. **jit-of-jit constant capture** ('Fetching value ... non-addressable';
   BROKEN SINCE #1370-iii sharded the stacks — the old device_put path
   fails identically in a 2-process CPU repro, so every post-Jul-30
   multicontroller ocean failure was this class behind the arg wall) —
   the stacks now thread through outer jit boundaries as ARGUMENTS:
   step.aux + timed_scan_blocks(aux=) + both run_omip JRA55 block
   builders (codex r18 P1) + all four invocation sites.
   2-proc CPU repro emits a full receipt (15.09 ms, gloo); 28 parity/
   bench tests + 44 run_omip tests green; codex r19 VERDICT: SHIP.

Falsification v3 in queue: oc LL2304 @96 (26646038) / @128 (26646039),
exclude l50081,l50100 (superset; harmless).

## OCEAN AT HUNDREDS: falsification v3 PASSES (jobs 26646038/26646039)

First-ever ocean lat-lon multicontroller receipts past 64 GPUs, on the
fixed lane (l50081+l50100 excluded; l50081 exclusion harmless-superset):

| arm | cols/GPU | ms/step | GC/s |
|---|---|---|---|
| LL2304x4608 L20 @96 | 110.6k | 18.25 | 11.63 |
| LL2304x4608 L20 @128 | 82.9k | 16.33 | **13.00** |

* 13.0 GC/s at 128 GPUs = **2.7x the previous ocean best** (4.75 GC/s,
  LL1152@64). Strong 96->128: speedup 1.117 for 1.333x devices =
  **eff 0.84** — a healthy-tile strong leg on the ocean lane.
* This closes the user directive's ocean-hundreds gap: both lat-lon
  lanes (atm + ocean) now hold receipts at 96-128 GPUs, MPAS at 128.

### Merge-port of #1362 (geometry_consistency) — shared-module fixes + one tracked follow-up

The three multicontroller fixes now live in
`legoesm.parallel.geometry_consistency` (checked_shard_put /
addressable_shard_put / assert_pytree_bytes_equal / band_fingerprint —
the ONE implementation, #1362 doctrine); the ocean lane calls them, and
#1362's entry gates gained aux coverage + numeric-scalar leaves in the
digest gate (codex r20). 167 gate/parity tests + the 2-proc repro
(15.11 ms) green post-fix.

**TRACKED FOLLOW-UP (codex r20 item 3): the ATMOSPHERE lat-lon lane
still routes its band/tile geometry stacks through `broadcast_checked`
-> `broadcast_one_to_all` (sharded_atm_latlon_step.py:492/1586) — wall
1 preserved there** (an [n_processes, stack] psum program). The atm
128-GPU receipts predate #1362, so the current main atm lane at >=96
processes is UNVERIFIED and plausibly walled exactly as ocean was.
Port = same checked_shard_put swap + aux threading; needs its own
parity run + a 2-proc repro before any atm hundreds rerun on merged
main.

## FINAL RECEIPTS (job 26657279): atm lat-lon at 144 GPUs — campaign records

The @192 ask starved 20+ h (48-node block vs a 55-healthy-node pool);
144 divides both grids and scheduled overnight:

| arm | cols/GPU | ms/step | GC/s |
|---|---|---|---|
| LL2304x4608 @144 | 73.7k | 5.78 | 47.76 |
| LL2880x5760 @144 | 115.2k | 7.40 | **58.32** |

* **58.3 GC/s is the campaign's highest throughput** (prior record
  39.1, LL2048@128).
* Strong LL2304 96 -> 144: 7.872 -> 5.78 = 1.362x for 1.5x devices =
  **eff 0.91** — the healthiest >64-GPU strong leg measured (tiles
  110.6k -> 73.7k, both far above the floor). (@96 ran the pre-merge
  worktree; the put-path fixes are SETUP-only, so steady timing is
  comparable.)
* Campaign close-out: every directive lane holds 96-144-GPU receipts
  (atm lat-lon 96/128/144, ocean lat-lon 96/128, MPAS 128) plus
  512-rank CPU lat-lon; the remaining improvement paths are the
  documented structural follow-ups (SoL-class device collectives,
  #1100 partition-local mesh, ensemble orchestration).

## AT-SCALE fabric constants (jobs 26677438/26677439) — the limit lines move

The roofline constants were measured at 8 GPUs / 2 nodes; remeasured at
the gap's own scale with the SAME chained-fori_loop microbench
(dispatch-subtracted):

| communicator | latency (us) | bandwidth (GB/s) |
|---|---|---|
| 8 GPU / 2 nodes | 26.3 | 23.5 |
| 64 GPU / 16 nodes | 28.4 | **12.1** |
| 128 GPU / 32 nodes | 29.7 | **12.1** |

* **Topology-LATENCY term: refuted for the chained-ring pattern
  measured** (+13 % at 16x the communicator; the fori_loop chain prices
  dependency-chained ring cost). SCOPE (codex r23): the microbench
  measures ppermute rings with dispatch subtracted — not production
  pair patterns, packing, synchronization skew, or all-reduce; the
  bound's one-latency AR term is a MODEL ASSUMPTION.
* **Bandwidth HALVES past 2 nodes** (ring crossing switch tiers shares
  links): 23.5 -> 12.1 GB/s. The bound's byte term doubles.
* Recomputed distances with at-scale constants:
  - LL2048@128 f32: comm = 41x29.7us + 18.54MB/12.1 = 2.750 ms;
    t_bound 2.780; **measured/bound 2.01** (was 2.94). Pricing every
    CP at the sweep's 512 KiB row (82.085 us; the census MEAN payload
    is 441.6 KiB, so this slightly over-prices): 41x82.085us + one
    29.7us AR-assumption = 3.395 ms -> **ratio 1.64** (post-overlap
    5.077 -> **1.50**). (codex r23 corrected the first draft's 3.11.)
  - LL2048@64 f32: comm 2.698 vs compute 2.837 — BALANCED regime;
    bound 2.865, ratio 2.35 (compute-edge, unchanged).
  - MPAS s9@np64: comm = 33x28.4us + 29.49MB/12.1 = 3.375; bound
    3.403; **ratio 2.82** (was 4.47).
* Honest reframe: with constants measured AT the deployment scale, the
  panels sit ~1.8-2.8x above the serialized model, and the comm term is
  now BYTE-dominated — bytes are physical (halo areas), so the
  remaining levers are OVERLAP (hide comm under compute; the closed
  ledger's null was the latency-dominated ocean nd4-16 regime, not
  this one — A/B job 26677529 submitted at @64 with the XLA
  latency-hiding scheduler + pipelined collectives) and exchange-COUNT
  packing (skew amplification).

## OVERLAP VERDICTS (jobs 26677602 / 26677668 / 26677669) — the first positive lever

XLA latency-hiding scheduler + pipelined p2p (flag names verified
against the installed stack after a guessed name aborted arm B of
26677529; that job still banked clean controls 6.667/6.724):

| lane | A | B (overlap) | A2 | effect |
|---|---|---|---|---|
| LL2048@64 | 6.615 | 6.069 | 6.635 | **-8.4 %** |
| LL2048@128 | 5.547 | **5.077** | 5.527 | **-8.3 %** |
| MPAS s9@64 | 9.710 | 9.720 | 9.680 | 0.0 % (null) |

* Twice-reproduced ~8 % on the lat-lon lane at two scales with 0.3-0.4 %
  control drift; BELOW the pre-registered 10 % bar (reported as such),
  wired strictly OPT-IN (`LEGOESM_XLA_OVERLAP=1`) — never a shared
  default: cube_tiled_step force-disables latency hiding for a known
  sensitivity, MPAS is null, and appended flags would poison future
  A/B control arms (codex r23). Parity suites green with flags on (CPU-virtual — the GPU-side
  check is the A/B rows themselves). MPAS: honest null — the
  edge-coloured hand schedule does not benefit.
* New LL2048@128 best: **5.077 ms = 43.0 GC/s**; measured/at-scale-bound
  = 5.077/2.780 = **1.83** (sweep-based bound 3.11 -> **1.63**).

## MPAS #1100 wall NAMED + FIXED: replicate_pytree

The s9 STEP program is clean (bufdump: 0.02 GB args, 4 per-shard
params) — the s10 162 GB was `replicate_pytree(mesh)`: a replicated
device_put of the 1.268 GB global-mesh pytree = 128 x 1.268 =
162.3 GB (matches the failure to 0.1 %), the SAME jax
whole-array-assert + replicated-logical wall as the ocean lane. Fixed
via the shared assert-free put + exact-hash contract gate
(PR #1457 pattern) in `parallel/mesh.py`; 2-proc multicontroller MPAS
repro green (56.97 ms, s5), voronoi parity 5 passed. NOTE (codex r23):
replicate_pytree is generic — the multi-process path can reach other
lanes (cube CLI); a structure pre-gate + direct tests added same
round. Falsification = s10@128 rerun (job 26677812, in queue): a PASS
is the receipt for MPAS at 128 GPUs; 192/224 remain EXPECTED-unlocked
pending their own runs.

## Lat-lon packing follow-up (designed, not yet built)

CP records (nd=8 census): the 41 CPs are SIX classes — 13x[1,1024,26],
6x[1,1025,26], 6x[2,1028,26], 6x[1,1026,27], 6x[1,1024,27] array
exchanges (31 one-row + 6 two-row) and 4 scalar f32[1] — codex r23
corrected the first draft's two-class count. Grouping by shape alone
does NOT prove packability: fields must be AVAILABLE at a common
program point, so the design step is a stage-local liveness analysis
across the RK stages (the `make_latlon_band_wall_multi_pad_body`
machinery exists for the ocean wall lane; the atm dycore pads
per-field). The ~6-group / ~4.0-4.4 ms projection is SPECULATIVE until
that analysis is done. Dycore surgery — staged as the next engineering
item.

## #1100 WALL DOWN (job 26677812): MPAS s10 @ 128 GPUs — new record

Post-replicate_pytree-fix falsification PASSES: subdiv-10 (10.49M
cells, lloyd-0) at 128 GPUs = **18.20 ms = 14.98 GC/s — the new
MPAS-atmosphere record** (2.1x the s9 peak of 7.10), at 81.9k
cells/GPU (above the ~30k floor). The lloyd-0 matched-tile weak series
gains a THIRD rung: s8@8 6.58 -> s9@32 12.47 -> s10@128 18.20 ms —
per-4x-scale cost 1.895 then 1.460: the scale-out term DECELERATES
with size. 192/224-GPU rungs are expected-feasible (48/56-node asks;
queue-starved historically — submit opportunistically).

## Ocean-MPAS GPU path: SCOPED (not built)

The ocean MPAS lane is CPU-MPI by design (`bench_ocean_mpas_scaling`
docstring). A GPU/SPMD twin would reuse the now-generic voronoi
machinery (cell-partition reorder, padded local meshes, edge-coloured
ppermute halo, the fixed replicate/put path) and wire the OCEAN MPAS
tendencies (`ocean_pe_mpas`) into a `make_voronoi_sharded_step`-class
factory: column-local vmix solves shard trivially; the
barotropic/baroclinic split is the design work. Estimated days-scale
feature with existing infra; staged as a follow-up, NOT attempted in
this round.

## FUSED HALO + OVERLAP: the stacked levers (jobs 26681636 / 26681858) — lat-lon at ratio 1.4

The audit-item-7 SPMD fused multi-pad already existed OPT-IN
(`LEGOESM_LATLON_SPMD_FUSED_HALO=1`, contract: "flip per deck only with
a measured GPU A/B receipt") — census 41 -> 29 CPs/step at IDENTICAL
bytes. The receipts:

| arm | @64 (ms) | @128 (ms) |
|---|---|---|
| A off | 6.590 | 5.573 |
| B fused | 6.264 (-5.4 %) | 5.325 (-4.5 %) |
| C fused+overlap | **5.278 (-20.3 %)** | **4.745 (-15.0 %)** |
| A2 off | 6.650 | 5.585 |

* The combined effect EXCEEDS the additive expectation (20.3 % vs
  13.8 % at @64; 15.0 % vs 12.9 % at @128) — CONSISTENT WITH fewer,
  larger CPs giving the latency-hiding scheduler more to hide
  (mechanism plausible, not instrumented).
* **New LL2048@128 best: 4.745 ms = 46.0 GC/s.** The bound must be
  REPRICED for the fused count (codex r26 — the lever moves the model
  too): 29 CPs x 29.7 us + 18.54 MB / 12.1 GB/s = 2.394, bound
  2.423 ms -> **measured/bound 1.96** (fit); sweep-priced at the new
  ~639 KiB/message (linear interpolation 524 KiB -> 1 MiB rows,
  ~88 us/CP): 29 x 88.2 + 29.7 us = 2.588 -> **~1.83**. @64: 5.278 vs
  the compute-dominated 2.865 bound -> 1.84 (unchanged by count).
* Parity: 23 tests green with the fused env ON; the COLLECTIVE PAYLOAD
  is byte-identical to the per-field pads (local concat/split traffic
  differs), A/A2 drift 0.2-0.9 %.
* Wired: the lat-lon hundreds launchers set BOTH envs — code-path
  validated (the exact 144-GPU points carry no dedicated A/B receipt);
  everything else stays opt-in (cube force-disables latency hiding;
  MPAS null).

## MPAS closure: at the practical stack limit

nsys (26680051): compute is a ~2.7 us MICROKERNEL storm (launch/
scheduling-bound — explains the overlap null) and NCCL SendRecv medians
~2x the clean wire estimate (skew absorbed in kernels). CUDA-graph
levers REFUTED (26680791: min-graph-size = exact no-op; command-buffer
with collectives +23 % WORSE). With protocol, partition, placement,
combining, overlap and graphs all receipted null, single-trajectory
MPAS stands at its practical XLA/NCCL stack limit (s9@64 ratio 2.82 on
the at-scale model); the scaling story there is the s10@128 record
(14.98 GC/s), the DECELERATING matched-tile cost (1.90 -> 1.46 per
4x), and ensemble parallelism (+0.6 % co-execution). Deeper wins need
XLA fusion-granularity work on unstructured ops — upstream-class.

## Codex improvement consult (2026-08-04) + first verdicts

Ranked candidate list (transcript
`.physics-validator/scaling_campaign/codex_consult_improvements_2026-08-04.md`):
(1) transfer fused+overlap to the unreceipted ocean LL2304@128 arm —
TOP PICK; (2) stage-local packing past 29 CPs; (3) fuse the 2-D pencil
wall pad; (4) MPAS s10@192; (5) MPAS profile-guided manual fusion;
(6) port the ocean geometry-consistency fix to the atm lane + retry
LL2880@192.

### #3 REFUTED — the 2-D pencil fused wall pad is SLOWER (job 26692375)

`pad_with_pole_bc_lat_multi_2d` was implemented (dtype-grouped single
sendrecv pair per cut, the band lane's pattern), verified on 2 ranks
value- and gradient-identical to the per-field path, then A/B'd at
r512 / 512 ranks:

| arm | ms/step |
|---|---|
| A off | 45.20 |
| B fused | **49.87 (+9.4 %)** |
| A2 off | 45.96 |

Outside the 1.7 % A/A2 bracket in the WRONG direction, so the change
was REVERTED (implementation + test + CI entry removed rather than
left as dead code; recoverable from this session's history). PLAUSIBLE
mechanism (not instrumented): a 2-D pencil's per-field lat slab is
`n_lon_local`-wide — much smaller than the band lane's full-row slab —
so the fused path's concatenate/slice memory traffic exceeds the
sendrecv latency it removes. NOTE this does NOT contradict the band
lane's fused win (`LEGOESM_LATLON_FUSED_HALO=1`, default on): different
slab size, different balance. Lesson for the ledger: a lever confirmed
on one decomposition is NOT transferable by analogy — every lane needs
its own A/B, and this one paid for itself by catching a regression
before it shipped.

### #1 PARTLY CONFIRMED — ocean LL2304@128: fused halo -4.9 %; overlap no benefit observed (job 26692291)

The top-pick transfer of the two atm levers to the unreceipted ocean
production arm (explicit_substep + wide halo, 18 steps):

| arm | ms/step | GC/s |
|---|---|---|
| A off | 16.620 | 12.78 |
| B fused | **15.807 (-4.9 %)** | **13.43** |
| C fused+overlap | 15.942 (-4.1 %) | 13.32 |
| A2 off | 16.610 | 12.78 |

* **Fused halo transfers: -4.9 %** (A/A2 drift 0.06 % — a very tight
  bracket), matching the atm lane's -4.5/-5.4 %. Census — an nd=8
  VIRTUAL-CPU algorithmic proxy at the same solver config, NOT a
  128-GPU collective trace: **206 -> 133 collective-permutes/step**
  (-35 % CP count; all-reduces stay 7 in both arms, so the cut is
  CP-only, and it is the largest CP-count reduction measured in this
  campaign).
* **Overlap shows NO benefit in this receipt**: C is 0.9 % slower than
  B — but that 0.135 ms difference was NOT replicated (one B/C pair),
  so the honest statement is "no benefit observed; leave the flags off
  on this lane pending a replicated B/C". HYPOTHESIS (uninstrumented):
  the barotropic subcycle is a long dependent chain with little
  independent compute to hide comm under, so there is little for the
  latency-hiding scheduler to exploit.
* Recommendation for the ocean lat-lon lane: set
  `LEGOESM_LATLON_SPMD_FUSED_HALO=1`, leave `LEGOESM_XLA_OVERLAP`
  OFF. New ocean best: **13.43 GC/s at 128 GPUs** (was 13.0).

### #6 REFUTED OFFLINE — the atm replicated-geometry broadcast is NOT walled at 192

Consult item #6 proposed porting the ocean geometry-consistency fix to
the atm lane to "remove the unverified high-process setup wall" before
retrying LL2880@192. Cheap arithmetic (measured field sizes, no GPU
hours) refutes the premise: `_build_geometry_stacks` calls
`broadcast_checked` **per field**, so the psum program is
`n_processes x ONE stacked field`, not `x the whole stack`:

| config | fields | max stacked field | broadcast program | vs 63.8 GB limit |
|---|---|---|---|---|
| LL2048@128 (ran) | 13 | 0.0336 GB | 4.3 GB | 6.7 % |
| LL2304@144 (ran) | 13 | 0.0425 GB | 6.1 GB | 9.6 % |
| LL2880@192 (target) | 13 | 0.0664 GB | **12.7 GB** | 20 % |

So the @192 attempts were QUEUE-starved, not walled — the port would
have bought ~0 % (as the consult itself predicted for steady state)
against a wall that does not exist at these sizes. THRESHOLD for the
future: the replicated broadcast reaches the limit when
`n_proc x field_bytes > 63.8 GB`, i.e. a single 2-D f32 geometry field
above ~332 MB — around LL5760x11520 at 256 processes. Revisit there,
not before. (The SHARDED-geometry mode already uses the assert-free
`checked_shard_put` from PR #1458 and is unaffected either way.)

### #2 REPRICED BY MEASUREMENT — the packing ceiling is ~6 %, not 7-16 %

The consult's #2 estimate (4.0-4.4 ms, i.e. 7-16 %) was built on the
PRE-fusion 41-CP count and a speculative grouping. The post-fusion
histogram (nd=8 virtual-CPU census, fused ON) prices what is actually
left — 29 CPs in six shape classes:

| count | shape | what it is |
|---|---|---|
| 6 | f32[1,79898] | the FUSED entry multi-pad (T,u,dp), 2/stage |
| 6 | f32[2,1028,26] | halo=2 PPM pad, 2/stage |
| 6 | f32[1,1026,27] | Bln-stack interface pad (fold family) |
| 6 | f32[1,1024,27] | v-face interface pad (BC family) |
| 4 | f32[1] | loop-invariant GEOMETRY (grid.lat, cos_lat) |
| 1 | f32[1,1024,26] | singleton |

Two findings that change the plan:

* **The four 1-D pads are pure grid geometry** (`grid.lat` /
  `cos_lat` at `_vface_cos_lat_core:87`, `curl_vertex_cgrid:1098/1218`,
  `tendencies:494`) — loop-invariant, and XLA already CSEs them across
  the three RK stages, so they cost **4 CPs / 16 bytes total**.
  Hoisting them host-side (the band grids are all available at factory
  time, so the padded metrics need NO communication) removes 4 CPs =
  4 x 29.7 us = 0.119 ms = **2.5 %** at the @128 working point.
* **The only other count lever is cross-family fusion**: the Bln
  (fold-family) and v-face (BC-family) interface pads differ ONLY in
  their pole rows — their interior exchange is identical, so an
  "exchange once, apply per-family pole fill" helper would take 12 CPs
  to 6 = a further ~3.7 %. That is delicate pole-semantics surgery on
  the dycore.

Realistic combined ceiling: **~6 %**, versus the 7-16 % the consult
projected from the stale count. Given the pole-semantics risk on the
cross-family half, #2 is recorded as MEASURED-AND-DEPRIORITIZED rather
than attempted: the cheap, safe part (geometry hoist, 2.5 %) is a
clean follow-up if wanted; the risky part is not worth ~3.7 % on a
lane already at ratio 1.83-1.96. Consistent with the campaign's
standing conclusion that the remaining distance is structural.

## Recolouring the MPAS halo schedule is capped by Vizing — measured offline (2026-08-07)

Both independent consults (codex + GLM-5.2, transcripts in
`.physics-validator/scaling_campaign/codex_consult_hundreds_2026-08-07.md`)
ranked the MPAS GPU lane as the top remaining structural lever: it is the
worst-scaling lane we have (measured/modelled-bound 3.16x at s8@16 to
4.47x at s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute rounds.
Codex priced a new partition objective that minimizes round depth at
300-800 LOC and 7-14 days, with an optimistic ceiling of 14.4% at s10@128
(12 fewer SendRecv calls/step x 218 us) and a hardware-only floor of
0.356 ms (12 x 29.7 us). That spread is why it must be measured.

**Before spending any of that, the cheap question is whether recolouring is
already exhausted** — and it is answerable offline, with no GPU, from the
colourer's own lower bound. `_build_ppermute_schedule` produces a proper
EDGE colouring of the device communication graph (one colour = one round,
properness asserted), and the `max_degree` it returns is that same graph's
maximum vertex degree. VIZING therefore bounds the chromatic index:
`Delta <= chi' <= Delta + 1`. So `coloring_gap = n_rounds - max_degree`
reads as:

* `gap == 0` -> `n_rounds == Delta`; no proper edge colouring can beat
  `Delta`. PROVABLY OPTIMAL, recolouring headroom exactly zero.
* `gap == 1` -> INCONCLUSIVE. A Class 2 graph genuinely needs `Delta + 1`,
  and Class 1 vs Class 2 is NP-complete.
* `gap >= 2` -> recolouring removes at least `gap - 1` rounds, at most `gap`.

MEASURED so far (`bench_voronoi_partition_methods.py --schedule-cost`, the
existing partition-quality bench extended to call the production
`spmd_schedule_cost` rather than its own 1-ring proxy):

| mesh | n_dev | geometric | sfc | metis | gap | strategy |
|---|---|---|---|---|---|---|
| L2/L3/L4 | 2-16 | rounds = n_dev-1 mostly | same | - | 0 everywhere | allgather (COUNTERFACTUAL) |
| L6 lloyd=0 | 8 | 7 | 7 | 6 | 0 | ppermute (5,121 cells/dev) |
| L6 lloyd=0 | 16 | 13 | 10 | 10 | 0 | ppermute (2,561 cells/dev) |
| L8 lloyd=0 | 64 | 16 | - | - | 0 | ppermute |

Two scope notes that must travel with these numbers:

* The small-mesh rows are COUNTERFACTUAL: every one auto-selects the
  ALLGATHER strategy (cells/device below the threshold), so production runs
  no ppermute schedule there. Only L6@16 upward are real. L6@8's
  `rounds == n_dev - 1` is complete-graph saturation and says nothing.
* `gap == 0` rules out a better UNDIRECTED edge colouring of THIS graph and
  nothing more. `_build_ppermute_schedule` enters a device pair into
  `comm_pairs` when EITHER direction has a halo dependency and then emits
  BOTH ppermute directions, even where one send map is empty — so a
  redesigned DIRECTED schedule exploiting one-way exchanges is not bounded
  by `Delta` at all. This is a THIRD path, not a two-way choice between
  colouring and ownership.

The neighbour fan-out that the bench's first layer already reported is NOT
a stand-in for any of this: at L6@16 it reads 8/8/7 for
geometric/sfc/metis while the real schedule reads 13/10/10.

Job 26770026 (CPU `shared`, 24 h, zero GPU hours) scores the production
working points s8/s9 @64,128 and s10@128. Arms 1-2 are a MECHANICAL
instrument check via `--expect-rounds` against the reference census in
`spmd_schedule_cost`'s docstring (s8 sfc 12/14, metis 13/19, geometric
16/21; s9 sfc 11/13, metis 14/18, geometric 14/18); arm 3's unknown s10
number is only produced if both pass, because an instrument that misses the
known answer cannot be trusted on the unknown one. The first production row
to land, s8 geometric@64 = 16 rounds, reproduces the census exactly — a
spot check, not yet the validation, which is the full six-row gate.

Eight codex adversarial rounds on this change (transcripts
`codex_review_schedule_cost{,_r2..r8}_2026-08-07.md`); round 8 SHIP. The
defects it caught are worth recording because most were in the INSTRUMENT,
not the model: a headroom figure documented backwards (`gap-1` is the
GUARANTEED reduction, not the maximum); a launcher that filed a scan with
no results as COMPLETED; an artifact guard that passed on
`{"rows": {"n_ranks": 1}}` because iterating a dict yields its keys; a
`--expect-rounds` gate bypassable by whitespace or a duplicate key; and a
first-draft test that was VACUOUS — hardcoding `coloring_gap = 0` passed
every fast fixture, because the true gap is 0 on all of them.

KNOWN GAP, reported not fixed: `tests/bench/` is not executed by the CI
test jobs (they run `tests/unit/` and selected paths), so neither these
tests nor any other test in `tests/bench/` is CI-enforced. Repo-wide and
pre-existing; Actions have been disabled repo-wide since 2026-05-27 in any
case, so a targeted wiring change here would be inert.

## The nsys collective census on the MPAS lane is UNRELIABLE (2026-08-07)

Do not build another MPAS attribution on an Nsight Systems capture until this
is resolved. The traces silently omit the very kernels being measured, with no
error and no missing-data marker.

Evidence, three independent captures:

| capture | rank / arm | total kernels | SendRecv kernels |
|---|---|---|---|
| 26680051 (single-rank) | rank 0, physics=none | — | 288 |
| 26772084 (skew, 4 ranks) | rank 0 | 187,606 | 96 |
| 26772084 | rank 2 | 188,312 | 317 |
| 26772084 | rank 1 | 187,247 | **0** |
| 26772084 | rank 3 | 187,316 | **0** |
| 26772734 (payload A/B) | dry, physics=none | 187,247 | **0** |
| 26772734 | wet, physics=kessler | 276,656 | 124 |

The zero rows are not runs without collectives: the dry arm's own receipt
records `halo_strategy_effective: ppermute` and a 9.74 ms step, i.e. the
halo exchange ran. Note also that the dry arm and skew rank 1 report the
IDENTICAL total of 187,247 kernels — the same systematic omission, not
random loss.

What survives and what does not:

* The DURATION median is robust — 216.2 / 212.4 / 213.5 us across three
  captures on different jobs. Quote it.
* Any COUNT from these traces is not. The 288-call structure that recovered
  the traced run's `--reorder-for 128` provenance happened to be
  self-consistent (8 rounds x 3 fills x 12 steps), but that consistency was
  luck, not a guarantee, and it cannot be relied on again.

CONSEQUENCE for the open payload-vs-wait question: it is NOT answerable by
more nsys jobs. Two arms were run at a fixed schedule with the packed cell
record changed from 28 to 106 values (`--physics none` vs `kessler`, mesh /
device count / partition / reorder target identical); the wet arm captured
124 collectives and the dry arm none, so there is nothing to compare. The
next instrument should be an HLO-level collective census or the XLA profiler,
both of which count what the compiled executable contains rather than what a
sampling profiler happened to record.

Also fixed along the way, and both would silently corrupt any future capture:
`ncclDevKernel_SendRecv` is recorded under `shortName`, NOT `demangledName`,
in some of these traces (querying the wrong column reads as "no collectives");
and nsys sets `QUADD_INJECTION_PROXY`, which JAX treats as distributed
coordinator configuration and hangs on — since only the PROFILED ranks get it,
they diverge from the rest and the whole job deadlocks in init.

## Cube lane: a measurement that does NOT reconcile (2026-08-07)

RETRACTED, same day it was said: I reported that a fresh 24->54 GPU
measurement "kills the cube scales at 1.04 picture". It does not. The 1.04
came from `6->24` on the cs-spmd row (job 26453782); mine is
`bench_cube_tiled_step_scaling.py` at 24->54. Different benches, different
rungs — comparing them is the confound this campaign has a standing rule
against, and I made it.

What was actually measured (job 26772775, BOTH rungs in ONE job on ONE node
set, so this part is internally controlled):

| rung | ms/step | ppermutes/step |
|---|---|---|
| C768/L60 kt=2, 24 GPUs | 67.54 | 123 |
| C768/L60 kt=3, 54 GPUs | 52.85 |  99 |

24->54 = 1.278x on 2.25x devices = **efficiency 0.568**. Not tile-floor
limited: C768 leaves 147.5k columns/GPU at 24 and 65.6k at 54, both far above
the ~30k floor.

THE UNRECONCILED NUMBER, which blocks any cube optimisation: the campaign's
own row records C768/L60 at 24 A100 as **14.09 ms/step**; this bench reports
**67.54 ms** for a nominally identical resolution, level count and device
count — 4.8x apart. Either they are different code paths (likely: face-sharded
cs-spmd vs the tiled `6*kt^2` lane) or one of them is mismeasured. Until that
is settled, no cube number here can be compared to the campaign's, and the
0.568 cannot be called a regression or a limit.

AND THE LEVER DOES NOT PAY ON THIS EVIDENCE. The exchange structure is real
and was censused from the compiled module: 396 collective-permutes on a CPU
24-device proxy fall into just 12 distinct directions (the documented schedule
— 4 edge strips + 4 guard slivers + 4 corner rounds), so the step CALLS the
exchange ~33 times, once per field, exactly the shape lat-lon fixed with a
packed multi-field pad. `packed_pad_halo_4d` already implements that idea but
requires the face-sharded `(6, n, n)` prefix, so the TILED lane has no packed
variant. Tempting — but price it first: 123 collectives x 29.7 us = 3.65 ms,
which is **5.4 % of the measured 67.54 ms step**. Even removing EVERY
collective cannot pay for the work. The cube's time is not in its halo on this
lane, and the 4.8x reconciliation is the thing to chase instead.

(GPU combines collective-permutes ~3.2x: the CPU proxy shows 396 where the GPU
executable holds 123. Use the proxy for STRUCTURE, never for the count.)

## Cube known-answer check came back OFF, on a dirty tree (2026-08-09 recovery)

The rerun designed above (`cube_bound_anchor.sbatch`, job 26804520,
2026-08-08) was left unanalysed by a session drop. Its own gate says the
result may not be quoted:

| arm | result |
|---|---|
| 0: C384/L60 kt2 @24 closed (known answer 9.01 ms, job 26452632) | **13.19 ms** — 1.46x off, GATE FAILED |
| 1: C768 kt2 @24 closed | CRASHED (coordination SetError) |
| 2: C768 kt3 @54 closed | CRASHED (coordination SetError) |

Two facts about arm 0's provenance decide nothing yet but scope the causes:
it ran on git sha `6fc6be9ea-dirty` — the dropped session's UNCOMMITTED
state — and two weeks of merges separate it from the 9.01 anchor
(clean `9051c5126`, 2026-07-24). So either (a) a real ~46% regression
merged into the tiled lane since 07-24, or (b) the dirty working tree
slowed it. Discriminator submitted: `cube_c384_anchor_recheck.sbatch`
(job 26818796) — same arm on CLEAN current main, refuses to run on a
dirty tree. ~9 ms => (b), close; ~13 ms => (a), bisect the window.

The C768 arms' 24->54 ratio therefore still has no valid closed-loop
measurement; do not quote 0.568 (that was the single-shot lane) nor any
number from 26804520 past arm 0.

## Ocean-MPAS CPU rank-count term ATTRIBUTED (2026-08-09, jobs 26819741/26819928/26820126/26820412)

The 1.65x-at-matched-tile mystery (s7@32 vs s8@128, 5120 cells/rank) is
now a measured decomposition, not a label. Current-code delta D = 42-44
ms (ratio 1.47-1.48, three replicates; the 1.65 was July code). Census
first (call-path confirmed): one ocean step = 35 halo epochs + 65
allreduces = ~100 sync points; the 10-substep barotropic scan holds 30
epochs + 60 reductions (eta-floor clamp = 3 allreduces x 2 sites x 10
substeps). The atm lane: 3 epochs + 1 reduction — no term, as measured.

| discriminator | receipt | verdict |
|---|---|---|
| substeps 10->5, D(5)/D(10) | 0.795 / 0.851 (2 passes) | scan carries only 30-41% of D; per-SYNC-POINT-uniform cost REFUTED (scan = 90/100 sync points) |
| s7@32 spread 4 nodes vs packed | 0.947 (faster) | fabric/placement REFUTED |
| conservation fixer OFF (removes all 5 non-scan reductions) | D_off 54.0 > D_on 44.5 | non-scan reductions REFUTED; fixer-off SLOWED np128 by 8 ms (PLAUSIBLE: longer fused segments = more jitter exposure; single run) |
| JAX-free spin+barrier probe, 2.5 ms segments | amp 1.033@32 -> 1.094@128 | OS/BSP jitter contributes ~15 ms (~1/3 of D) — real, not sufficient |
| s8 same-mesh ladder 32/64/128 | 400.7/232.7/132.5 ms (1.72/1.76 per doubling) | no wall; splits D into mesh-size ~10 ms + rank-growth ~32 ms |

Segment-length-proportional stalling (the GLM-5.2 BSP tail-latency
model) fits the 30-41% scan share (scan segments are short); the ~17 ms
rank-growth remainder above the pure-MPI jitter floor is PLAUSIBLY the
mpi4jax host-callback sync cost itself (unmeasured directly — the
rotted per-epoch micro was dropped; VoronoiHaloExchange.exchange_cell_field
still references the removed layout.cell_comm, reported not fixed).

PRODUCTION LEVER, mechanism-independent: every sync point carries
rank-growing cost, so cut sync points — (1) the eta-floor clamp's 60
allreduces/step (n_iter 3 x 2 sites x 10 substeps; E3SM avoids global
reductions inside barotropic substepping entirely), (2) fuse the 3
per-substep exchanges into 1 (30 -> 10 epochs). Both touch production
numerics -> physics-validator + codex chain when picked up.

### Cube known-answer RESOLVED (2026-08-09, job 26823000)

CLEAN main reproduces the anchor: C384/L60 kt2 @24 closed-loop =
**8.97 ms/step** (anchor 9.01, job 26452632). The 13.19 ms from job
26804520 was a DIRTY-TREE artifact (sha 6fc6be9ea-dirty), not a merged
regression — no bisect needed. Two operational causes burned first:
26818796 refused on the shared worktree being mid-iteration dirty (the
guard working as designed — submit from a clean tree), and 26821454
timed out with l50100 (the phase-7 sick node) in its allocation; the
exclusion is now baked into the launcher. Cube lane state: tiled-lane
anchor healthy; the halo lever remains DEAD there (5.4% ceiling); the
open cube item is only the C768 24->54 closed-loop ladder (arms crashed
in 26804520 on coordination errors — rerun when worth a slot).

## RAGGED HALO FILL — the MPAS structural lever LANDS (2026-08-09)

PR #1512 closed recolouring (Vizing gap 0 — the coloured schedule was
optimal); the remaining lever was the ROUND STRUCTURE itself. jax 0.10
exposes ragged_all_to_all (grouped P2P, one collective for all
neighbours — the MPAS-A/ICON-GPU concurrent-isend pattern; JVP+transpose
registered; XLA:CPU has NO thunk, so every gate is a GPU job).

| stage | receipt |
|---|---|
| synthetic microbench (26818265, in-job exact verify) | per-fill ratio ragged/coloured: 1.27@nd4 (loses intra-node), 0.746@nd8, **0.475@nd16** — advantage grows with scale; uniform a2a competitive small-n but O(n_dev) inflated |
| schedule row-identity | tests/parallel/test_ragged_halo_schedule.py: transfer-set equality with the ppermute schedule (shared _build_halo_send_maps) |
| GPU parity (26821453) | u/T/p_s vs serial reference OK at s6@4, halo_strategy_effective=ppermute_ragged bound in the receipt |
| production A/B/A2 (26822138+26824483) | s8@16: ppermute 7.08/7.21 ms, **ragged 4.86 ms — ratio 0.686 (-31% FULL STEP)**, drift 1.8%, pre-registered gate CONFIRMED |

Opt-in: LEGOESM_MPAS_RAGGED_HALO=1 (explicit '1'; auto-allgather tiles
unaffected; ppermute default untouched). Open: AD unexercised on this
path; scale receipts s9/s10@64-128; default-on decision after those.
Ops debris burned on the way (all fixed in-branch): census KeyError
killing multicontroller arms post-timing; ~50 min/arm s8@16 host
reorder forcing banked-arm job structure; XLA sharded-autotune cache
desync from a shared persistent cache (per-job cache now); l50100.

### Ragged halo at scale: a SCALE-BANDED lever (2026-08-09 evening)

| receipt | ppermute | ragged | ratio |
|---|---|---|---|
| production s8@16 (26822138/26824483) | 7.08/7.21 ms | 4.86 | **0.686 WIN** |
| production s9@64 (26824688, drift 1.5%) | 10.35/10.19 | 12.43 | **1.220 LOSS** |
| synthetic fixed degree-10/payload (26825475 + 26818265) | 1008 / 1484 / 1471 us @16/32/64 | 478 / 647 / **1103** | 0.475 / 0.436 / 0.750 |

The ragged collective's own cost GROWS with rank count at fixed traffic:
+625 us from nd16->nd64 ~= 49 extra zero-size slices x ~12 us — the
unpruned-no-op cost GLM-5.2 flagged as make-or-break, CONFIRMED by the
pre-registered discriminator. Production flips to a loss earlier than
the synthetic (bigger buffers + the s_max-padded gather:
int32[64,2,149795] stacked metadata at s9@64). Feature stays opt-in
OFF; band edge (32 devices) receipt = job 26825520; if it wins, the
follow-up is an auto dispatch (ragged <= band, coloured above), and the
structural fix beyond that is upstream zero-slice pruning in XLA's
ragged thunk.

### RETRACTION: the cube 13.19 was PLACEMENT, not the dirty tree (2026-08-09 evening)

Job 26825926 (clean main) reproduced **13.47 ms** on the same
known-answer arm that read 8.97 this morning — the "dirty-state
artifact" conclusion is RETRACTED. The real confound across all four
runs: launchers that pin `--nodes=6 --ntasks-per-node=4` read ~9 ms
(26452632, 26823000); launchers whose srun lines omit placement under a
14-node allocation spread 24 tasks wide and read 13.2-13.5 ms
(26804520 "dirty", 26825926 clean). Same lane, ~1.5x from node
placement alone. cube_bound_anchor now pins both 24-rank arms.
Additionally the C768 kt=3 @54 arm dies on HOST OOM — the tiled cube
bench still builds global state per rank (the #1370 residency wall,
fixed for the ocean lane, never ported here) — arm disabled with the
blocker named; kt=2 C768 @24 rerun = job 26826088.
Lesson (controlled-comparison rule, again): a discriminator job whose
placement differs from the runs it arbitrates arbitrates NOTHING.

### Cube placement split CONFIRMED; C768 ladder blocked on the bench residency wall (2026-08-09)

Pinned rerun (job 26826088): known-answer arm = **8.91 ms** — the
placement split now has four points: pinned-6-node 9.01/8.97/8.91 vs
spread-14-node 13.19/13.47. Placement is the whole story; tree state
never mattered.
C768 @24 kt2 ALSO host-OOMs when pinned (4 ranks/node x per-rank global
C768 build) — both C768 arms are blocked on the SAME wall: the tiled
cube bench builds the global model/state on every rank
(the #1370 residency issue; the ocean lane's fix — global build under
jax.default_device(cpu) + addressable shard puts, commit e1b502000 —
was never ported to bench_cube_tiled_step_scaling.py). NAMED NEXT ITEM
for the cube lane; the C768 24->54 closed-loop ratio stays unmeasured
until it lands.

### C768 closed-loop ladder MEASURED; the cube halo lever REPRICED ALIVE (2026-08-09)

Job 26826851 (both rungs one job, 2 rpn matched, dt=30 — the 26826706
failure was CFL non-finite at dt=60, not memory; the OOM fix was 2
ranks/node, closed-loop C768 = 46.5 GB host/task):

| rung | ms/step |
|---|---|
| C768/L60 kt2 @24 closed | 18.93 |
| C768/L60 kt3 @54 closed | 15.22 |

24->54 = 1.244x on 2.25x devices = **eff 0.553** at 147k->65k cols/GPU
(far above the tile floor) — the single-shot lane's 0.568 REPRODUCES on
the production assembly: a real cube scale-out deficit.
REPRICE: the halo-lever "5.4%, does not pay" verdict divided the 3.65 ms
collective cost by the single-shot 67.54 ms step; against the CLOSED
loop's 18.93 ms it is **~19% at 24 devices** — the packed multi-field
pad for the tiled lane (packed_pad_halo_4d exists face-sharded only) is
back on the table as the next cube lever.

### Cube fused wave-A halo: REFUTED on GPU — XLA already does it (2026-08-09 night)

Implemented (dtype-grouped multi-pad, 8->3 exchange calls/stage,
bit-identical on CPU + >=1.5x CPU CP cut enforced by test, codex SHIP)
and A/B'd at the banked C768@24 closed-loop protocol (job 26828313):

| arm | ms/step | HLO CPs |
|---|---|---|
| off (A/A2) | 19.26 / 19.03 | 101 |
| fused (B) | 20.28 | 96 |

ratio 1.066, drift 1.2% — **REFUTED by the pre-registered gate, with
the mechanism on the receipt**: the GPU module holds 101 CPs where the
CPU proxy holds 384 — XLA's collective-permute COMBINER had already
fused the per-field exchanges on GPU; source-level fusion saved 5 CPs
and added concat/split traffic. (Same lesson class as consult #3, the
2-D pencil fused pad: a lever confirmed on one lane/compiler path does
not transfer by analogy — and here the CPU census was the misleading
proxy; GPU counts are the only ones that price GPU levers.)
Code REVERTED same-day (delete-before-adding); the receipts + launcher
diff live in this branch's history. The cube deficit (eff 0.553)
therefore is NOT collective-count-bound at 24 devices — remaining
candidates: per-CP latency floors x 101, launch-bound compute (the
MPAS nsys story), or skew; next instrument = nsys on the closed-loop
C768 step.
