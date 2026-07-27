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
- **Tripole (synthetic ORCA fold)**: f32 35.56/24.91/15.89 ms, f64
  63.84/43.36/24.12 — within 2-8 % of the regular lat-lon ladder, BUT
  that baseline is from a DIFFERENT job/day, so "fold is not a
  bottleneck" is CONFOUNDED until the same-job matched A/B lands (job
  26493837, submitted on codex round-13's objection).
- **MPAS-ocean Voronoi (subdiv 6, L20)**: FLAT (f32 7.00/6.92/6.58; f64
  ~7.0 throughout) at ~10k cells/GPU — CONSISTENT with the tile-size
  latency floor, but a flat ladder alone cannot exclude a partition or
  parallel-path defect (codex round-13): "hypothesis, not verdict". The
  discriminating subdiv-7 ladder (~41k cells/GPU at nd4, above the
  floor) is in job 26493837 — if it scales, floor confirmed; if it stays
  flat, defect hunt.
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
