Reading additional input from stdin...
OpenAI Codex v0.145.0
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fcd9c-bd07-7981-9145-2bea3ae2df08
--------
user
CONSULT (not a review): rank NEW improvement candidates for this scaling campaign, given everything already measured. Read docs/performance/scaling/levante_campaign_2026-07-24.md (esp. the 2026-08-02..04 sections) for the full ledger.

STATE: lat-lon GPU LL2048@128 = 4.745 ms = 46 GC/s at measured/bound 1.96 (fit) after two stacked levers (fused SPMD multi-pad 41->29 CPs; XLA latency-hiding+pipelined-p2p). @64 = 5.278, ratio 1.84 (compute-dominated). Ocean latlon @128 = 16.33 ms/13.0 GC/s (no fused/overlap receipts yet on that lane!). MPAS atm s9@64 ratio 2.82, s10@128 record 14.98 GC/s; profile = ~2.7us microkernel storm, launch/scheduling-bound.
RECEIPTED NULL (do not re-propose): NCCL_PROTO, CP-combine threshold flag, partition quality (METIS), rank placement (block:block worse), wet-cell compaction, pipelining/overlap on the OCEAN implicit arm, Chebyshev, distributed V-cycle, CUDA graphs on MPAS (min-graph-size no-op, command-buffer +23% worse), MPAS overlap (null).
CONFIRMED LEVERS: fused multi-pad, XLA overlap flags, ensemble parallelism (+0.6% co-execution at K=3), tile>=30k floor, block:cyclic CPU distribution, single_reduce/wide-halo ocean solver arms.
INFRA available: shard_map/ppermute SPMD, edge-coloured MPAS halo schedule, jax 0.10, A100+IB (28-30us latency, 12.1 GB/s at scale), 224 GPUs max, existing census/bufdump/nsys/microbench probes.

Give a RANKED list of at most 6 candidate improvements NOT yet tried, each with: (a) the specific code change or experiment, (b) the number that would confirm/refute it, (c) estimated effect size with reasoning from the measured numbers above, (d) risk/effort. Prioritize things implementable in hours-to-a-day on this infra, and call out explicitly anything that is a cheap TRANSFER of an already-confirmed lever to an unreceipted lane (e.g. fused/overlap on the ocean latlon lane, or on the atm 2-D pencil path). End with 'TOP PICK: <one line>'.
codex
I’ll use the campaign ledger to distinguish genuinely untried work from transfers, then rank only short-horizon experiments with concrete go/no-go metrics.
exec
/usr/bin/bash -lc "sed -n '1,260p' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 279ms:
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

exec
/usr/bin/bash -lc "rg -n '"'^## |''^### |2026-08-0[2-4]|NULL|CONFIRMED|OPEN|STATUS'"' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 294ms:
17:## Headline results (strong scaling, f32 unless noted)
54:## Ocean strong-scaling: bottleneck → fix (the campaign's improvement arc)
67:all arms converge to 0.71–0.77 (tiles amortize latency). Fused-halo == plain implicit (a NULL result: pad aggregation does not
82:the wide-halo stability gates (`SCALING_STATUS_AUDIT` item 3) and a
85:## Atm ladders added late in the campaign
100:OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
137:kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
179:STATUS: **FIXED, SHIPPED GATED** (codex rounds 11-12: strategy consult
197:## Ocean strong scaling vs TILE SIZE (jobs 26456334/37 vs 26452804-06)
223:## Weak scaling at production per-device size (job 26453523)
240:## "It used to be faster / did we regress?" — resolved, no regression
266:## CPU-MPI (compute nodes)
287:## Atm ICOSAHEDRAL CPU-MPI (job 26452579, 1 node, f64 moist, L26)
339:## DISTANCE TO THE THEORETICAL LIMIT, measured (job 26457977)
412:- fused-halo NULL — aggregation reduces message COUNT but not chain DEPTH;
428:## The mechanism's prediction, TESTED — and the lever it exposes (job 26459382)
501:## The accuracy-free lever, and what it reveals (job 26460113)
543:## MATCHED CONFIG HEAD-TO-HEAD — the production recommendation (job 26460365)
606:## Wide-halo stability: evidence toward the gate (job 26460729)
645:MECHANISM CONFIRMED FROM A THIRD ANGLE. The wide-halo arm records its own
697:## Scale-out + the plateau question (2026-07-27)
940:## The resolution lever is CAPPED on both transports — and my subdiv-9 runs were invalid
983:## Cube optimisation: bounded BEFORE implementing — and the bound killed the plan
1056:## Tripole (ORCA fold) past 4 GPUs — first receipts (job 26512798)
1077:## The MPAS mesh cap — lifted (subdiv-9 unblocked for 128 GPUs)
1122:## Ocean GPU scale-out to 64 devices — and a CROSS-LANE memory defect (#1370)
1216:## Ocean MPAS Voronoi scale-out — and a lane that does NOT obey the tile law
1281:## Precision + grid-coverage verification (2026-07-27, user request)
1351:## Using the calibrated bound correctly (a trap worth documenting)
1370:## Precision changes which ocean config wins (job 26457919)
1389:## Measured fabric constants for the roofline lines (job 26457495)
1418:## OPERATIONAL NOTE: transient multi-node hangs (3 occurrences)
1437:## Infrastructure defects found + fixed (each with a receipt)
1474:## Closed levers (nulls with receipts — do not re-run)
1485:## Still open (ranked) — refreshed at campaign close
1587:## Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)
1591:(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
1595:### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
1641:### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
1674:### Next receipts submitted 2026-08-02
1695:### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
1708:floor-attributable. Mechanism OPEN — candidates (uninstrumented): 1-D
1715:## Hundreds-of-devices push (user directive 2026-08-02)
1732:### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)
1758:### s8 lloyd=0 de-confound ladder landed (job 26628076): the near-matched-tile scale-out cost persists without the Lloyd confound
1795:## Distance-to-modeled-limit: atm lat-lon GPU (2026-08-02, "near theoretical limit" directive)
1853:## Distance-to-modeled-limit: MPAS GPU (2026-08-02 late)
1915:### Transport discriminator (job 26638578): NCCL runs NET/IB + GDRDMA — the gap is real
1931:### s10@128 blocked by the #1100 remainder (job 26630207)
1943:### CP-combining A/B: three deadlocks on one node block; pivoted to @64 with exclusion
1962:## ENSEMBLE RECEIPT (job 26628196): lever #1 CONFIRMED — co-execution penalty <= 0.6 %
1998:### CP-combining A/B verdict (job 26641073, LL2048@64, sick block excluded): threshold null — no >=10 % benefit detected
2024:### Ocean @96 bracket (job 26642771): compiler-reported args total is NEAR-LINEAR in device count
2042:## The deadlock node is NAMED: l50100 (bisect complete) — and the multicontroller ocean lane is FIXED
2053:### Multicontroller ocean: THREE stacked walls found and fixed (codex r14-r19)
2074:## OCEAN AT HUNDREDS: falsification v3 PASSES (jobs 26646038/26646039)
2090:### Merge-port of #1362 (geometry_consistency) — shared-module fixes + one tracked follow-up
2110:## FINAL RECEIPTS (job 26657279): atm lat-lon at 144 GPUs — campaign records
2133:## AT-SCALE fabric constants (jobs 26677438/26677439) — the limit lines move
2173:## OVERLAP VERDICTS (jobs 26677602 / 26677668 / 26677669) — the first positive lever
2196:## MPAS #1100 wall NAMED + FIXED: replicate_pytree
2212:## Lat-lon packing follow-up (designed, not yet built)
2226:## #1100 WALL DOWN (job 26677812): MPAS s10 @ 128 GPUs — new record
2237:## Ocean-MPAS GPU path: SCOPED (not built)
2249:## FUSED HALO + OVERLAP: the stacked levers (jobs 26681636 / 26681858) — lat-lon at ratio 1.4
2282:## MPAS closure: at the practical stack limit

exec
/usr/bin/bash -lc "sed -n '1470,2315p' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 315ms:
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

exec
/usr/bin/bash -lc "sed -n '1790,1965p' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 317ms:
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

exec
/usr/bin/bash -lc 'rg -n "FUSED_HALO|fused.*halo|multi_pad|multi-pad|LATLON_SPMD|OVERLAP|latency.hiding|pipelined" legoesm scripts tests | head -n 240' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 4028ms:
rg: legoesm: No such file or directory (os error 2)
scripts/bench/run_scaling_diagnosis.py:128:            "--xla_gpu_enable_latency_hiding_scheduler=true",
scripts/bench/run_scaling_diagnosis.py:490:    backend-independent — GPU-only XLA collective combining / pipelined-p2p
scripts/bench/run_scaling_diagnosis.py:494:    custom-calls (invisible here AND to the XLA latency-hiding scheduler; that
scripts/bench/bench_ocean_latlon_spmd_scaling.py:128:    # Wide-halo lever (A/B): one fused wide lat-halo exchange per chunk of
scripts/bench/bench_ocean_latlon_spmd_scaling.py:308:    p.add_argument("--fused-halo", action="store_true",
scripts/bench/bench_ocean_latlon_spmd_scaling.py:310:                        "(LEGOESM_LATLON_SPMD_FUSED_HALO=1): one ppermute "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:318:                        "fused wide lat-halo exchange per chunk of substeps "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:432:    if args.fused_halo:
scripts/bench/bench_ocean_latlon_spmd_scaling.py:434:        os.environ["LEGOESM_LATLON_SPMD_FUSED_HALO"] = "1"
scripts/bench/bench_ocean_latlon_spmd_scaling.py:830:            "fused_halo": os.environ.get(
scripts/bench/bench_ocean_latlon_spmd_scaling.py:831:                "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0",
tests/parallel/test_latlon_spmd_fused_halo.py:1:"""SPMD fused multi-field halo (message aggregation, audit item 7).
tests/parallel/test_latlon_spmd_fused_halo.py:3:Gates the three contracts of ``make_latlon_band_wall_multi_pad_body`` /
tests/parallel/test_latlon_spmd_fused_halo.py:4:the ``LEGOESM_LATLON_SPMD_FUSED_HALO`` opt-in:
tests/parallel/test_latlon_spmd_fused_halo.py:25:      pytest tests/parallel/test_latlon_spmd_fused_halo.py``
tests/parallel/test_latlon_spmd_fused_halo.py:41:    make_latlon_band_wall_multi_pad_body,
tests/parallel/test_latlon_spmd_fused_halo.py:73:    fused_body = make_latlon_band_wall_multi_pad_body(
tests/parallel/test_latlon_spmd_fused_halo.py:101:def test_fused_bit_identical_wide_halo(halo):
tests/parallel/test_latlon_spmd_fused_halo.py:107:    fused_body = make_latlon_band_wall_multi_pad_body(
tests/parallel/test_latlon_spmd_fused_halo.py:126:    body = make_latlon_band_wall_multi_pad_body(mesh, n_fields=2)
tests/parallel/test_latlon_spmd_fused_halo.py:146:    fused_body = make_latlon_band_wall_multi_pad_body(
tests/parallel/test_latlon_spmd_fused_halo.py:309:    monkeypatch.setenv("LEGOESM_LATLON_SPMD_FUSED_HALO", value)
tests/parallel/test_latlon_spmd_fused_halo.py:361:            err_msg=f"fused-halo step diverged on {nm}")
tests/parallel/test_latlon_fused_halo_serial.py:1:"""Serial unit tests for the fused multi-field lat-lon halo exchange.
tests/parallel/test_latlon_fused_halo_serial.py:8:exercised by ``tests/distributed/test_fused_halo_mpi.py`` (mpirun) and
tests/parallel/test_latlon_fused_halo_serial.py:44:        fused = pad_with_pole_bc_lat_multi(fields, halo=1)
tests/parallel/test_latlon_fused_halo_serial.py:83:        monkeypatch.setenv("LEGOESM_LATLON_FUSED_HALO", "0")
tests/parallel/test_latlon_fused_halo_serial.py:85:        fused = pad_with_pole_bc_lat_multi(fields, halo=1)
scripts/bench/roofline_probe.py:247:    pipelined loop measures *throughput*, not per-call *latency*):
scripts/bench/roofline_probe.py:396:        # streams 3*array_bytes — pipelined/CSE'd repeats of an identical
scripts/bench/roofline_probe.py:820:    # blocked round-trip (codex BLOCKER — a pipelined loop would hide the
scripts/bench/run_levante_gpu_scaling.py:142:            "--xla_gpu_enable_latency_hiding_scheduler=true",
scripts/bench/run_levante_gpu_scaling.py:1911:    # and enabling XLA's latency-hiding scheduler to pipeline
scripts/bench/metadata.py:145:    count, because XLA collective-permute combining / pipelined-p2p
scripts/bench/metadata.py:209:    pipelined collectives are reflected as executed, not as emitted) and runs
scripts/bench/metadata.py:216:    combining / pipelined-p2p (``--xla_gpu_collective_permute_combine_*``, lane
scripts/bench/metadata.py:772:    (overlapping/pipelined messages beat it), while a partial halo census
scripts/bench/slurm_scaling_diagnosis.sh:107:export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
tests/distributed/test_fused_halo_mpi.py:1:"""MPI (mpirun -np 2) tests for the fused lat-lon halo exchange (O4 lever).
tests/distributed/test_fused_halo_mpi.py:3:Run:  mpirun -np 2 python -m pytest tests/distributed/test_fused_halo_mpi.py
tests/distributed/test_fused_halo_mpi.py:79:def test_fused_halo2_matches_singles():
tests/distributed/test_fused_halo_mpi.py:87:    fused = pad_with_pole_bc_lat_multi_mpi((f2d, f3d), layout, halo=2)
scripts/bench/bench_ocean_mpi_scaling.py:394:        # LEGOESM_BARO_WIDE_HALO=1 -> ONE fused wide lat-halo exchange per
scripts/bench/bench_ocean_mpi_scaling.py:970:#     critically under MPI — communication/computation OVERLAP is lost:
scripts/bench/bench_ocean_mpi_scaling.py:1071:    ``_time_fused_fn`` exactly like the full step).  ``halos`` /
scripts/bench/bench_ocean_mpi_scaling.py:1921:        "  comm/compute OVERLAP lost => the residual below absorbs "
scripts/cluster/scaling_levante/README.md:108:`fused` (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`, bit-identical packing,
scripts/cluster/scaling_levante/README.md:111:`--xla_gpu_enable_pipelined_p2p=true`), `pgle`
scripts/cluster/scaling_levante/_env.sh:86:# Latency-hiding scheduler + pipelined p2p: -8.4% at LL2048@64 (job
scripts/cluster/scaling_levante/_env.sh:91:# (cube_tiled_step.sbatch force-disables latency hiding for a known
scripts/cluster/scaling_levante/_env.sh:93:# (codex r23): set LEGOESM_XLA_OVERLAP=1 in validated lat-lon
scripts/cluster/scaling_levante/_env.sh:96:if [ "${LEGOESM_XLA_OVERLAP:-0}" = 1 ]; then
scripts/cluster/scaling_levante/_env.sh:97:  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:28:export LEGOESM_LATLON_SPMD_FUSED_HALO=1
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:29:export LEGOESM_XLA_OVERLAP=1
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:24:#   Lane T (RUN_TUNE=1, default 0) comm-tuning A/B ladder (fused-halo /
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:25:#                         XLA CP-combining + pipelined p2p / PGLE); outputs
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:227:# fused (LEGOESM_LATLON_SPMD_FUSED_HALO=1 — bit-identical multi-pad packing,
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:229:# pipelined p2p), pgle (profile-guided latency estimation; recompiles after
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:236:    _XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:241:              fused) export LEGOESM_LATLON_SPMD_FUSED_HALO=1 ;;
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:259:            [ "$ARM" = fused ] && export LEGOESM_LATLON_SPMD_FUSED_HALO=1
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:79:export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_latency_hiding_scheduler=false"
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:28:export LEGOESM_LATLON_SPMD_FUSED_HALO=1
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:29:export LEGOESM_XLA_OVERLAP=1
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:25:#   C +combine +pipelined-p2p      : scheduling interaction
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:56:run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:59:for T in A_default B_combine C_combine_pipelined A2_default; do
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:25:#   C +combine +pipelined-p2p      : scheduling interaction
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:56:run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:59:for T in A_default B_combine C_combine_pipelined A2_default; do
scripts/run/run_omip.py:600:                        "fused wide lat-halo exchange per chunk of substeps "
scripts/validate/fv3_native/compare_extvec_stages.py:295:                print(f"{st:6s} {nm:7s} t{t}   BAD-OVERLAP {got} != "
scripts/validate/fv3_native/compare_extvec_stages.py:311:                print(f"{st:6s} {nm:7s} t{t}   EMPTY-OVERLAP")
scripts/cluster/scaling_derecho/README.md:526:already at floor; atm latlon 41 → 29 behind the fused-halo flag — see
scripts/cluster/scaling_derecho/README.md:531:1. `fused` — `LEGOESM_LATLON_SPMD_FUSED_HALO=1`.  **MEASURED 2026-07-09
scripts/cluster/scaling_derecho/README.md:535:2. `xla` — CP-combine 32 MiB + pipelined p2p.  **MEASURED: −10 % — not
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:18:# XLA latency-hiding applies) and the CXI ticket stops blocking GPU scaling.
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:40:#   RUN_TUNE=1 (default 0)   — lane T comm-tuning A/B ladder (fused-halo /
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:41:#     XLA collective-permute combining + pipelined p2p / PGLE) on the latlon
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:89:# XLA collective-permute combining + pipelined p2p. #1113 found the route-B MPAS
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:97:_XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:220:        # CP-combining + pipelined p2p ON by default (#1113): the ppermute
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:287:#   fused : LEGOESM_LATLON_SPMD_FUSED_HALO=1 (bit-identical multi-pad packing;
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:289:#   xla   : collective-permute combining + pipelined p2p — the cube/latlon
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:308:              fused) export LEGOESM_LATLON_SPMD_FUSED_HALO=1 ;;
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:348:            [ "$ARM" = fused ] && export LEGOESM_LATLON_SPMD_FUSED_HALO=1
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:81:# XLA GPU defaults (latency-hiding scheduler ON, async collectives ON,
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:90:# (bisect: re-enable latency_hiding_scheduler alone, keep
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:92:export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_latency_hiding_scheduler=false"
tests/unit/test_diff_tools.py:749:    documents an ``_OVERLAP_EPS`` guard "NaN-free" at ``cf = 1``.  The forward
tests/unit/test_scale_halo.py:2152:        refused `pad_halo_vector(duogrid=<grid>)` under MPI, but scalar
scripts/data/build_ec_gapfree_driver.py:65:_MIN_OVERLAP = 100
scripts/data/build_ec_gapfree_driver.py:238:                   if n_overlap >= _MIN_OVERLAP else None)
scripts/data/build_ec_gapfree_driver.py:254:            if int(new_obs.sum()) >= _MIN_OVERLAP:
scripts/data/build_ec_gapfree_driver.py:262:                print(f"  NOTE {dv}: {fv} has < {_MIN_OVERLAP} observations; keeping "
scripts/data/build_ec_gapfree_driver.py:266:            if n_overlap < _MIN_OVERLAP:
scripts/data/build_ec_gapfree_driver.py:269:                    f"(< {_MIN_OVERLAP}); cannot verify units/alignment — refusing.")

exec
/usr/bin/bash -lc "ls && rg --files | rg '(sharded_atm_latlon_step|latlon.*spmd|sharded.*latlon|voronoi.*shard|bench_ocean_latlon)'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 349ms:
CHANGELOG.md
CITATION.cff
CLAUDE.md
CMIP.md
COMMERCIAL-LICENSE.md
CONTRIBUTING.md
FEDERATION.md
LICENSE
README.md
anchor_1gpu.26450081.log
atm128.26534060.log
atm_ladders.26454476.log
atm_ll_144.26657279.log
atm_ll_192.26628072.log
bufdump.26549933.log
bufdump2.26555987.log
bufdump96.26644327.log
bufdump96.26644374.log
bufdump96.26644472.log
comm_micro.26457469.log
comm_micro.26457495.log
config
cpu1024.26509869.log
cpu1024m.26512349.log
cpu_f32.26534068.log
cpu_ll2d.26628073.log
cube_f64t.26512794.log
cube_face6.26452602.log
cube_face6.26452633.log
cube_fair.26452894.log
cube_fair.26452979.log
cube_fair.26453782.log
cube_ft.26498347.log
cube_ft30.26497736.log
cube_kt1.26452553.log
cube_kt3.26497294.log
cube_matched.26495955.log
cube_nsys.26504836.log
cube_nsys_cl.26507936.log
cube_nsys_cl2.26510470.log
cube_skew.26526100.log
cube_tiled_26446699.log
cube_tiled_26450318.log
cube_tiled_26450938.log
cube_tiled_26452632.log
cube_tiled_26453524.log
cube_tiled_26453645.log
data
docs
evaluations
fig_cubef64.26495027.log
fig_cubef64b.26495388.log
fig_f64.26494902.log
fig_icof32.26495083.log
fig_icof64.26495437.log
gap_inst.26677283.log
gate_selfspawn.26452426.log
gate_selfspawn.26452895.log
gate_selfspawn.26453280.log
gate_selfspawn.26453906.log
gate_selfspawn.26453981.log
gather_ab.26479833.log
gather_ab.26479884.log
legoesm_cpu_scaling.26445986.log
legoesm_cpu_scaling.26447093.log
legoesm_cpu_scaling.26447094.log
legoesm_cpu_scaling.26448158.log
legoesm_cpu_scaling.26452578.log
legoesm_cpu_scaling.26452579.log
legoesm_diag.26445834.log
legoesm_diag.26447827.log
legoesm_diag.26454084.log
legoesm_gpu_multinode.26445837.log
legoesm_gpu_multinode.26447612.log
legoesm_gpu_multinode.26449146.log
legoesm_gpu_multinode.26449147.log
legoesm_gpu_multinode.26450847.log
legoesm_gpu_multinode.26450848.log
legoesm_gpu_multinode.26452743.log
legoesm_gpu_multinode.26452744.log
legoesm_gpu_multinode.26452745.log
legoesm_gpu_multinode.26452804.log
legoesm_gpu_multinode.26452805.log
legoesm_gpu_multinode.26452806.log
legoesm_gpu_multinode.26453240.log
legoesm_gpu_multinode.26453279.log
legoesm_gpu_multinode.26456334.log
legoesm_gpu_multinode.26456335.log
legoesm_gpu_multinode.26456337.log
legoesm_gpu_multinode.26457000.log
legoesm_gpu_multinode.26457693.log
legoesm_gpu_multinode.26460444.log
legoesm_gpu_multinode.26460445.log
legoesm_gpu_multinode.26460447.log
legoesm_gpu_multinode.26460448.log
legoesm_gpu_multinode.26460876.log
legoesm_gpu_multinode.26460877.log
legoesm_gpu_scaling.26445836.log
ll128_comb.26630576.log
ll128_comb.26638830.log
ll128_comb.26640037.log
ll2304.26524165.log
ll2304b.26525329.log
ll2304c.26525484.log
ll64_comb.26641073.log
ll_2048.26502539.log
ll_bound.26630370.log
ll_fused.26681636.log
ll_fused128.26681858.log
ll_ovl.26677529.log
ll_ovl.26677602.log
ll_ovl128.26677668.log
ll_scaleout.26497323.log
ll_scaleout.26498266.log
ll_scaleout2.26498463.log
memprobe.26520891.log
memprobe2.26522816.log
memprobe3.26523157.log
memprobe4.26524044.log
memprobe5.26524163.log
memprobe6.26524423.log
memprobe7.26526284.log
mpas128.26538474.log
mpas32.26534061.log
mpas32.26549646.log
mpas_bar.26480261.log
mpas_bar.26480310.log
mpas_bar.26486123.log
mpas_bar.26486163.log
mpas_bar.26486288.log
mpas_bound.26635847.log
mpas_census.26635865.log
mpas_census.26636032.log
mpas_census.26636684.log
mpas_cg.26680791.log
mpas_codegen.26455948.log
mpas_f64.26493638.log
mpas_f64r.26493734.log
mpas_fuse.26480162.log
mpas_hlo.26480096.log
mpas_nsys.26479737.log
mpas_nsys.26479922.log
mpas_ovl.26677669.log
mpas_part_ab.26455829.log
mpas_recheck.26454618.log
mpas_s10_128.26630207.log
mpas_s10_128.26677812.log
mpas_s8_l0.26628076.log
mpas_s9.26549775.log
mpas_s9.26600095.log
mpas_s9_ens.26628196.log
mpasoc_2x2.26508258.log
mpasoc_2x2b.26508336.log
mpasoc_32.26494908.log
mpasoc_512.26508063.log
mpasoc_cpu.26494036.log
mpasoc_metis.26600076.log
mpasoc_metis.26600094.log
mpasoc_out.26504842.log
mpasoc_rpn.26505286.log
nccl_probe.26638578.log
oc128.26534067.log
oc128.26628071.log
oc128.26636762.log
oc128.26642771.log
oc128.26644681.log
oc128.26644682.log
oc128.26644853.log
oc128.26644854.log
oc128.26646038.log
oc128.26646039.log
oc128ab.26549699.log
oc_64.26509600.log
oc_ft32.26510472.log
oc_scaleout.26504841.log
ocean_ab_nccl.26449812.log
ocean_ab_pcg.26449622.log
ocean_ab_solver.26447957.log
ocean_ab_vmix.26452547.log
ocean_ab_wf.26449827.log
ocean_ab_xla.26451282.log
ocean_bound.26457919.log
ocean_cov.26493642.log
ocean_cov.26493648.log
ocean_cov2.26493837.log
ocean_h2h.26460365.log
ocean_itchk.26459817.log
ocean_iters.26459382.log
ocean_mixed.26493592.log
ocean_ovl.26458930.log
ocean_rl2.26457977.log
ocean_roofline.26457946.log
ocean_sr.26460113.log
ocean_stab.26460729.log
ocean_stab3.26464790.log
ocean_weak.26453523.log
ocean_x32.26460501.log
packages
plotfig.26646185.log
plotfig.26646224.log
plotfig.26669923.log
plotfig.26678532.log
plotfig.26678591.log
plotfig.26685105.log
ppmu_scale.26677438.log
ppmu_scale.26677439.log
prewarm_s10.26628074.log
prewarm_s9.26549180.log
project_status.md
pyproject.toml
requirements_mpi.txt
results
s8_512.26497704.log
s9_ins.26514768.log
s9_nsys.26680051.log
scaleout_cpu.26495929.log
scripts
shard_over.26458553.log
sicknode.26641282.log
sicknode.26641376.log
sicknode.26641377.log
sicknode.26642924.log
sicknode.26642925.log
sicknode.26645625.log
sicknode.26645627.log
src
templates
tests
tripole16.26512798.log
uv.lock
wetbal2.26480203.log
wetbal2.26480298.log
wetbal2.26480388.log
wetbal2.26480448.log
wetbal_numa.26479815.log
wetbal_numa.26479904.log
tests/bench/test_bench_ocean_latlon_spmd_gates.py
scripts/bench/bench_atm_latlon_spmd_scaling.py
scripts/bench/bench_ocean_latlon_spmd_pcg.py
scripts/bench/bench_ocean_latlon_spmd_scaling.py
tests/unit/test_run_omip_latlon_spmd.py
tests/unit/test_plot_atm_latlon_spmd_scaling.py
tests/unit/test_bench_ocean_latlon_spmd_scaling.py
tests/unit/test_latlon_spmd_driver_dispatch.py
tests/unit/test_latlon_spmd_config.py
tests/unit/test_bench_atm_latlon_spmd_scaling.py
tests/parallel/test_latlon_ocean_spmd_step.py
tests/parallel/test_latlon_spmd_fused_halo.py
tests/parallel/test_latlon_spmd_northfold.py
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py
tests/parallel/test_latlon_spmd_shard_leaf.py
tests/parallel/test_latlon_ocean_spmd_multicontroller.py
tests/parallel/test_voronoi_sharded_equivalence.py
tests/parallel/test_atm_latlon_spmd_driver.py
tests/parallel/test_latlon_spmd_pcg.py
tests/parallel/test_atm_latlon_spmd_multicontroller.py
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py
tests/parallel/test_latlon_ocean_spmd_tripole.py
tests/parallel/test_latlon_ocean_spmd_multiprocess.py
tests/parallel/test_atm_latlon_operator_split_spmd.py
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py
tests/parallel/test_latlon_ocean_spmd_wide_halo.py
tests/parallel/test_atm_latlon_spmd_step.py
tests/parallel/test_latlon_spmd_halo.py
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py
docs/scaling/atm_latlon_spmd_scaling.md
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
scripts/plot/plot_atm_latlon_spmd_scaling.py
packages/core/legoesm/parallel/latlon_spmd.py

exec
/usr/bin/bash -lc "sed -n '110,145p;290,325p;415,445p' scripts/bench/bench_ocean_latlon_spmd_scaling.py && rg -n \"make_latlon_band_wall_multi_pad_body|LEGOESM_LATLON_SPMD_FUSED_HALO\" packages scripts/bench/bench_ocean_latlon_spmd_scaling.py packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py packages/core/legoesm/parallel/latlon_spmd.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1770ms:
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    if tripole:
        # Synthetic tripole (ORCA fold): the sharded step's fold support
        # is gated by tests/parallel/test_latlon_ocean_spmd_tripole.py;
        # the wide-halo lever refuses folds at model construction.
        from legoesm.grids.tripole import create_synthetic_tripole

        grid = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)
    else:
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    # Wide-halo lever (A/B): one fused wide lat-halo exchange per chunk of
    # barotropic substeps instead of ~4 ppermute pads per substep.  The wide
    # path's per-substep clamp is local by contract, so pin local clamping
    # in BOTH arms for a controlled comparison.
    # Production-matching solver (scaling audit, bottleneck 4): OMIP runs
    # implicit_cn (run_omip.py full preset); the config-dataclass default
    # is explicit_substep, so it MUST be set explicitly here or the bench
    # measures a non-production step.
    flat = {"barotropic_solver": baro_solver}
    if force_pcg:
        # Solver-matched strong ladders (codex 2026-07-24 finding 2): the
        # implicit-CN dispatch runs adaptive stock CG on a SINGLE device but
        # the fixed-iteration distributed PCG under SPMD/MPI — an nd=1
        # reference leg without this flag times a DIFFERENT solver than the
        # nd>1 legs. Forces the fixed-M PCG everywhere.
        if baro_solver != "implicit_cn":
            raise SystemExit(
                "--force-pcg only affects the implicit_cn barotropic solve; "
             "against the single-device trajectory at the sharded "
             "split-explicit re-association-floor tolerances (smoke windows "
             "only; the floor grows with steps).")
    p.add_argument(
        "--check-conservation", action="store_true",
        help="Gate global area/eta/heat/salt drift over the run "
             "(pre-shard global state vs gathered final state; exits "
             "nonzero on breach).")
    p.add_argument(
        "--cons-rtol", type=float, default=None,
        help="Conservation tolerance (default: 1e-9 f64 / 1e-4 f32; the "
             "raw scheme drifts ~1e-8/step — calibrate to the window).")
    p.add_argument("--tripole", action="store_true",
                   help="Synthetic tripole (ORCA-fold) lane: the sharded "
                        "step folds the north band data-dependently "
                        "(SPMD equivalence gated at 4 devices). Rows are "
                        "tagged grid=tripole. Incompatible with "
                        "--wide-halo (fold refused at construction).")
    p.add_argument("--fused-halo", action="store_true",
                   help="Opt-in SPMD halo message aggregation "
                        "(LEGOESM_LATLON_SPMD_FUSED_HALO=1): one ppermute "
                        "pair per direction per dtype group at every "
                        "pad_multi site instead of one per field — "
                        "measured 25%% fewer static collective-permutes on "
                        "this step, bit-identical results. A/B against "
                        "the default run.")
    p.add_argument("--wide-halo", action="store_true",
                   help="Opt-in wide-halo split-explicit barotropic: one "
                        "fused wide lat-halo exchange per chunk of substeps "
                        "instead of ~4 ppermute pads per substep (implies "
                        "local per-substep clamping in this arm; A/B against "
                        "the default run).")
    p.add_argument("--wide-halo-chunk", type=int, default=0,
                   help="Substeps per wide exchange (0 = auto from the band "
                        "height).")
    p.add_argument("--multicontroller", action="store_true",
    if args.multicontroller and nd != avail:
        # A mesh over a strict subset would leave some processes' devices out
        # of the program (non-addressable participation hazard). Route-B uses
        # ALL global devices: one band per device across every process.
        raise SystemExit(
            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
            f"device count ({avail} across {jax.process_count()} processes).")
    n_lat = args.n_lat if args.mode == "strong" else args.nlat_per_dev * nd
    if n_lat % nd != 0:
        raise SystemExit(f"n_lat {n_lat} not divisible by n_devices {nd}")

    if args.parity_gate and args.steps > SPMD_PARITY_MAX_STEPS:
        raise SystemExit(
            f"--parity-gate is a smoke gate (re-association floor grows "
            f"with steps); --steps {args.steps} > {SPMD_PARITY_MAX_STEPS} "
            f"cap.")

    if args.fused_halo:
        # Trace-time switch — set BEFORE the sharded step is built/jitted.
        os.environ["LEGOESM_LATLON_SPMD_FUSED_HALO"] = "1"

    # #1370 fix stage (i), codex round-18: build the GLOBAL model/state on
    # the HOST cpu backend, not the accelerator. rest-state init runs jnp
    # ops at GLOBAL shape; on the default (GPU) device that materialises
    # ~7.4 global-field-equivalents of setup residency per device AND
    # compiles a global-sized init program — the 102 GB wall that killed
    # LL2304@64 (probe 26523157: the compiled STEP is clean; the residency
    # is setup-time). Host-side globals are RAM, and only the per-band
    # shards reach the accelerator via shard_state_latlon's device_put.
    # Identical values on every process (deterministic init + the step
    # factory's existing process-0 broadcast + content-hash guard).
scripts/bench/bench_ocean_latlon_spmd_scaling.py:310:                        "(LEGOESM_LATLON_SPMD_FUSED_HALO=1): one ppermute "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:434:        os.environ["LEGOESM_LATLON_SPMD_FUSED_HALO"] = "1"
scripts/bench/bench_ocean_latlon_spmd_scaling.py:831:                "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0",
packages/core/legoesm/parallel/latlon_spmd.py:164:    zeros.  Mirrors :func:`make_latlon_band_wall_multi_pad_body`'s "per dtype
packages/core/legoesm/parallel/latlon_spmd.py:640:def make_latlon_band_wall_multi_pad_body(mesh, halo: int = 1,
packages/core/legoesm/parallel/latlon_spmd.py:669:            f"make_latlon_band_wall_multi_pad_body: needs a mesh carrying "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:814:            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:907:        # dispatch — flipping LEGOESM_LATLON_SPMD_FUSED_HALO on a reused
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:913:            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
packages/core/legoesm/parallel/latlon_spmd.py:164:    zeros.  Mirrors :func:`make_latlon_band_wall_multi_pad_body`'s "per dtype
packages/core/legoesm/parallel/latlon_spmd.py:640:def make_latlon_band_wall_multi_pad_body(mesh, halo: int = 1,
packages/core/legoesm/parallel/latlon_spmd.py:669:            f"make_latlon_band_wall_multi_pad_body: needs a mesh carrying "
packages/core/legoesm/grids/halo_latlon.py:775:        "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
packages/core/legoesm/grids/halo_latlon.py:780:                make_latlon_band_wall_multi_pad_body,
packages/core/legoesm/grids/halo_latlon.py:782:            body = make_latlon_band_wall_multi_pad_body(

exec
/usr/bin/bash -lc 'rg -n "latlon-2d|2d|pencil|FUSED_HALO|OVERLAP|LATLON_SPMD" scripts/bench/bench_atm* scripts/cluster/scaling_levante packages/atmosphere/legoesm/atmosphere/dynamics/gcm | head -n 260' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 310ms:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:519:    _dealias_2d = _dealias[:, None]            # (n_sh, 1) for (n_sh, nlev[+1])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:520:    dvor_hat = dvor_hat * _dealias_2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:521:    ddiv_hat = ddiv_hat * _dealias_2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:522:    dw_hat = dw_hat * _dealias_2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:523:    dtheta_p_hat = dtheta_p_hat * _dealias_2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:524:    drho_p_hat = drho_p_hat * _dealias_2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1092:    lat_2d = grid.lat[:, None] * jnp.ones(grid.n_lon)[None, :]  # (n_lat, n_lon)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1093:    lon_2d = grid.lon2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1099:    dlat = lat_2d - lat0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1100:    dlon = lon_2d - lon0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1103:        + jnp.cos(lat_2d) * jnp.cos(lat0) * jnp.sin(dlon / 2) ** 2
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1157:    dims_2d = ("spectral",)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1170:        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1193:    dims_2d = ("spectral",)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1214:        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1265:    lat_2d = small_grid.lat[:, None] * jnp.ones(n_lon)[None, :]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1266:    lon_2d = small_grid.lon2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1269:        dlon = jnp.mod(lon_2d - p["chain_lon"] + jnp.pi, 2 * jnp.pi) - jnp.pi
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1270:        x_dist = dlon * small_grid.radius * jnp.cos(lat_2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1271:        y_dist = (lat_2d - p["gap_lat"]) * small_grid.radius
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1277:        dlat = lat_2d - p["mountain_lat"]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1278:        dlon = lon_2d - p["mountain_lon"]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1280:                 + jnp.cos(lat_2d) * jnp.cos(p["mountain_lat"])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1316:    dims_2d = ("spectral",)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1325:        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1405:    lat_2d = small_grid.lat[:, None] * jnp.ones(n_lon)[None, :]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1406:    lon_2d = small_grid.lon2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1410:        dlat = lat_2d - lat_c
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1411:        dlon = jnp.mod(lon_2d - p["bubble_lon"] + jnp.pi, 2 * jnp.pi) - jnp.pi
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1412:        x_dist = dlon * small_grid.radius * jnp.cos(lat_2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1431:    dims_2d = ("spectral",)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1440:        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:523:    lat2d = grid.lat2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:524:    sin_lat_2d = jnp.sin(lat2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:527:    h = (gh_0 - (R * Omega * u_0 + u_0**2 / 2.0) * sin_lat_2d**2) / g
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:533:    vor = 2.0 * u_0 / R * sin_lat_2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:566:    lat2d = grid.lat2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:567:    lon2d = grid.lon2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:568:    cos_lat_2d = jnp.cos(lat2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:569:    sin_lat_2d = jnp.sin(lat2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:572:    vor = 2.0 * u_0 / R * sin_lat_2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:582:        jnp.sin(lat_c) * sin_lat_2d +
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:583:        jnp.cos(lat_c) * cos_lat_2d * jnp.cos(lon2d - lon_c),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:591:    h_free = (gh_0 - (R * Omega * u_0 + u_0**2 / 2.0) * sin_lat_2d**2) / g
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:647:    cos_lat_2d = jnp.clip(grid.cos_lat[:, None], 0.01, None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:648:    u = u_cos / cos_lat_2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:649:    v = v_cos / cos_lat_2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_latlon.py:43:# Signature: (time, lon2d, lat2d, sigma_coord) -> (u_east, v_north, sigma_dot)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_latlon.py:44:#   u_east:      shape matching lon2d/lat2d + (nlev,)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_latlon.py:45:#   v_north:     shape matching lon2d/lat2d + (nlev,)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_latlon.py:46:#   sigma_dot:   shape matching lon2d/lat2d + (nlev+1,)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_latlon.py:121:        ``(t, lon2d, lat2d, sigma_coord) → (u, v, sigma_dot)``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_latlon.py:146:    _, _, sigma_dot = wind_fn(t, grid.lon2d, grid.lat2d, sigma_coord)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_latlon.py:195:        ``(t, lon2d, lat2d, sigma_coord) → (u, v, sigma_dot)``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:5:(:func:`legoesm.parallel.tiled_production_cdgrid.make_tiled_fv3_hydrostatic_step_stage_2d`),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:168:        make_tiled_fv3_hydrostatic_step_stage_2d,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:180:    tiled = make_tiled_fv3_hydrostatic_step_stage_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:259:    make_tiled_fv3_hydrostatic_step_blocked_2d`):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:299:        make_tiled_fv3_hydrostatic_step_blocked_2d,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:323:    tiled = make_tiled_fv3_hydrostatic_step_blocked_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:599:    cos_lat_2d = jnp.clip(grid.cos_lat[:, None], _COS_LAT_MIN, None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:600:    dlnps_dx = _dfdlon_lnps / (a * cos_lat_2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:602:    dlnps_dy = -dfdtheta_cos / (a * cos_lat_2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:1065:    sf_2d = spectral_filter           # (n_sh,) for 2D fields
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:1071:        lnps_hat=state.lnps_hat.replace(data=state.lnps_hat.data * sf_2d),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2223:    dims_2d = ("spectral",)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2266:        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2267:        phis_hat=Field(data=phis_hat_data, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:686:    lat = grid.lat2d   # (n_lat, n_lon)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:724:    lat = grid.lat2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:725:    lon = grid.lon2d
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_cdgrid.py:1264:    dims_2d = ("face", "x", "y")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_cdgrid.py:1270:        dp_s_dt=Field(data=dp_s_dt_data, name="dp_s_dt", dims=dims_2d, units="Pa/s"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_cdgrid.py:1272:            data=jnp.zeros_like(phis), name="dphis_dt", dims=dims_2d, units="m^2/s^3"
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler.py:460:                                          # plane pencil decomposition (zero comm,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:302:    dims_2d = ("lat", "lon")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:315:        p_s=Field(data=state.p_s, name="p_s", dims=dims_2d, units="Pa"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:316:        phis=Field(data=state.phis, name="phis", dims=dims_2d, units="m^2/s^2"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_cdgrid.py:1059:    dims_2d = ("face", "x", "y")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_cdgrid.py:1074:            name="dphis_dt", dims=dims_2d, units="m^2/s^3",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:211:    pert2d = jax.random.normal(key, (n_lat, n_lon), dtype=_dtype) * jnp.asarray(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:231:        return block.at[:, :, -1].add(pert2d[idx[0], idx[1]])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:243:    sh_2d = (n_lat, n_lon)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:248:        p_s=_make(sh_2d, _ps_cb),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:249:        phis=_make(sh_2d, _zeros_cb(sh_2d)),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:415:    ``(n_lat, n_lon)`` 2-D fields ``lat2d, lon2d, f, dx, area``).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:638:    collective -- ``n_steps`` validation, ``_check_2d_mesh``, the ``mesh is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:657:      ``n_axes=2, p_lat=2, p_lon=3``. The gate passed, then ``_check_2d_mesh``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1416:# sequential lat-then-lon exchanges inside ``make_latlon_2d_pad_body``; the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1421:# path (choose_latlon_2d_topology returns (N, 1) whenever the band is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1434:def shard_state_atm_latlon_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1455:    _agree_mesh_entry(mesh, state, where="shard_state_atm_latlon_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1471:def gather_state_atm_latlon_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1474:    """Inverse of :func:`shard_state_atm_latlon_2d`: replicate every leaf,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1481:    _agree_mesh_entry(mesh, state, where="gather_state_atm_latlon_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1504:def build_tile_grids_atm_2d(grid, p_lat: int, p_lon: int):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1518:        make_latlon_2d_layout, slice_latlon_grid_to_block_2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1537:            slice_latlon_grid_to_block_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1539:                make_latlon_2d_layout(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1548:def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1564:    tile_grids = build_tile_grids_atm_2d(grid, p_lat, p_lon)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1599:                         context="make_sharded_atm_latlon_step_2d",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1607:                context="make_sharded_atm_latlon_step_2d",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1616:                    context="make_sharded_atm_latlon_step_2d")),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1624:def _make_tile_step_body_2d(model, template, array_field_names,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1673:def _check_2d_mesh(mesh) -> tuple[int, int]:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1680:            f"axis_names=('lat', 'lon')) — choose_latlon_2d_topology picks "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1685:def _refuse_unsupported_spmd_config_2d(model, p_lon: int) -> None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1704:            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1705:            "lat-pencil transpose; the SPMD ppermute equivalent is a "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1709:def make_sharded_atm_latlon_step_2d(model, mesh, physics_fn=None, *,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1716:    :func:`shard_state_atm_latlon_2d` (``v`` as ``v_lower``, ``u`` as
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1721:    routes ``pad_halo_latlon*`` through ``make_latlon_2d_pad_body`` (lat
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1730:    ``tests/parallel/test_atm_latlon_2d_tiling.py``).  The 1-D band factory
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1750:                      where="make_sharded_atm_latlon_step_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1760:    p_lat, p_lon = _check_2d_mesh(mesh)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1761:    _refuse_unsupported_spmd_config_2d(model, p_lon)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1764:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1766:    tile_step = _make_tile_step_body_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1778:                         where="make_sharded_atm_latlon_step_2d.step")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1783:                "make_sharded_atm_latlon_step_2d: a stateful PhysicsState "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1803:def make_sharded_atm_latlon_segment_2d(model, mesh, n_steps: int,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1819:                      where="make_sharded_atm_latlon_segment_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1833:    p_lat, p_lon = _check_2d_mesh(mesh)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1834:    _refuse_unsupported_spmd_config_2d(model, p_lon)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1841:                "make_sharded_atm_latlon_segment_2d: a stateful PhysicsState "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1846:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1848:    tile_step = _make_tile_step_body_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1859:        # trace, caught by test_2d_segment_matches_sequential_and_serial).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1871:                         where="make_sharded_atm_latlon_segment_2d.segment")
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:28:export LEGOESM_LATLON_SPMD_FUSED_HALO=1
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:29:export LEGOESM_XLA_OVERLAP=1
scripts/cluster/scaling_levante/README.md:108:`fused` (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`, bit-identical packing,
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:227:# fused (LEGOESM_LATLON_SPMD_FUSED_HALO=1 — bit-identical multi-pad packing,
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:241:              fused) export LEGOESM_LATLON_SPMD_FUSED_HALO=1 ;;
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:259:            [ "$ARM" = fused ] && export LEGOESM_LATLON_SPMD_FUSED_HALO=1
scripts/cluster/scaling_levante/_env.sh:93:# (codex r23): set LEGOESM_XLA_OVERLAP=1 in validated lat-lon
scripts/cluster/scaling_levante/_env.sh:96:if [ "${LEGOESM_XLA_OVERLAP:-0}" = 1 ]; then
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:28:export LEGOESM_LATLON_SPMD_FUSED_HALO=1
scripts/cluster/scaling_levante/atm_latlon_144.sbatch:29:export LEGOESM_XLA_OVERLAP=1
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:2:#SBATCH --job-name=cpu_ll2d
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:8:#SBATCH --output=cpu_ll2d.%j.log
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:10:# 2-D pencil path (wall poles, labelled throughput lane — NOT the pole
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:21:OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cpu_ll2d_j${SLURM_JOB_ID}}"
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:26:  echo "--- atm latlon 2-D pencil r512 f64 np=$NP ---"
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:30:      --grid latlon --latlon-2d --mode single --physics moist \

exec
/usr/bin/bash -lc "sed -n '1390,1435p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py && sed -n '1650,1712p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py && sed -n '1,100p' scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch && rg -n \"make_latlon_2d_pad_body|pad.*multi|multi.*pad\" packages/core/legoesm/parallel/latlon_spmd.py packages/core/legoesm/grids/halo_latlon.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 301ms:
            on_segment(hs_out, done)
    return gather_atm_latlon_to_hydrostatic(c_state, model.grid, mesh), status


# ==============================================================================
# M3a — native 2-D ("lat", "lon") tiling for the atmosphere SPMD step
# ==============================================================================
# The 1-D lat-band decomposition's halo perimeter is the CONSTANT n_lon per
# cut (independent of the device count) — the term that caps band scaling.
# The 2-D tiling shards latitude AND longitude: lat stays the pole-terminated
# line (ppermute at cuts, the serial 180-deg fold at the pole tiles), lon
# becomes a periodic ring (cyclic ppermute — the wrap IS the roll
# permutation).  Staggered ownership mirrors the 1-D v convention:
#
#   * v (n_lat+1 rows)   -> v_lower = v[:n_lat]; each tile's north boundary
#     face is the lat-neighbour's v_lower[0] (reconstruct_vface_lower).
#   * u (n_lon+1 columns) -> u_left = u[:, :n_lon]; each tile's east seam
#     face is the lon-neighbour's u_left[:, 0] (reconstruct_uface_left) —
#     the +1 seam column is OWNED by the tile whose slice starts there and
#     reconstructed on the periodic wrap (u[:, n_lon] == u[:, 0] identity).
#
# The step body is the SAME un-jitted ``model._step_cgrid_impl`` the band
# path runs: all lon-direction neighbour access inside the operators already
# routes through the backend-dispatched ``pad_lon_cgrid`` / ``pad_halo_latlon*``
# (which this lane arms with the 2-D mesh), so no operator numerics are
# duplicated here.  Corner (diagonal) dependencies compose through the
# sequential lat-then-lon exchanges inside ``make_latlon_2d_pad_body``; the
# C-grid chain has no explicit-diagonal stencil (vertex circulations combine
# lat-padded u with lon-padded v).
#
# The 1-D band lane above is UNTOUCHED and remains the default production
# path (choose_latlon_2d_topology returns (N, 1) whenever the band is
# FEASIBLE — the 2-D pad's pole-fold lon-all_gathers outweigh its perimeter
# advantage until the partner-ppermute fold lands; the 2-D lane is for the
# beyond-band regime n_devices > n_lat/min_tile or indivisible n_lat.  The
# (N, 1) 2-D mesh degenerates bit-identically anyway).


def tile_spec(arr) -> P:
    """``P("lat", "lon", None, ...)`` for an array tiled on its two leading
    (lat, lon) axes — the 2-D twin of :func:`lat_spec`."""
    return P("lat", "lon", *((None,) * (arr.ndim - 2)))


def shard_state_atm_latlon_2d(
    state: CGridLatLonHydrostaticState, mesh,
        pmask = (stacks_local["__polar_mask"][gi, gj]
                 if "__polar_mask" in stacks_local else None)
        pmaskv = (stacks_local["__polar_mask_v"][gi, gj]
                  if "__polar_mask_v" in stacks_local else None)
        # Reconstruct the tile's nl+1 v-faces (shared interface row via the
        # lat ppermute) and w+1 u-faces (periodic seam column via the lon
        # ring), run the un-jitted step, convert both staggers back.
        v_full = reconstruct_vface_lower(state_local.v, "lat", perm_north)
        u_full = reconstruct_uface_left(state_local.u, "lon", p_lon)
        state_tile = state_local._replace(u=u_full, v=v_full)
        out, ps_out = model._step_cgrid_impl(
            state_tile, dt,
            physics_fn=physics_fn, phys_state=ps_local,
            grid=tile_geom, sigma_coord=model.sigma_coord,
            polar_mask=pmask, polar_mask_v=pmaskv,
            pole_v_bc_masks=spmd_pole_end_masks(),
        )
        return (out._replace(u=to_uface_left(out.u),
                             v=to_vface_lower(out.v)), ps_out)

    return tile_step


def _check_2d_mesh(mesh) -> tuple[int, int]:
    """Validate the 2-D tile mesh axes and return ``(p_lat, p_lon)``."""
    names = tuple(mesh.axis_names)
    if names != ("lat", "lon"):
        raise ValueError(
            f"atm 2-D SPMD tiling: mesh axes must be ('lat', 'lon'); got "
            f"{names}.  Build it as Mesh(devices.reshape(p_lat, p_lon), "
            f"axis_names=('lat', 'lon')) — choose_latlon_2d_topology picks "
            f"(p_lat, p_lon).")
    return int(mesh.shape["lat"]), int(mesh.shape["lon"])


def _refuse_unsupported_spmd_config_2d(model, p_lon: int) -> None:
    """2-D-specific dispatch-hardening on top of the shared band refusals.

    Like the 1-D twin this performs NO agreement itself; it relies on
    :func:`_agree_spmd_entry` having already agreed ``use_polar_filter`` (and
    the mesh's axis ORDER, which decides ``p_lon``) at the entry point.  That
    dependency is the sharpest instance of codex-2026-07-29 blocker 1:
    `use_polar_filter` ALSO controls whether the
    `__polar_mask`/`__polar_mask_v` entries exist in the geometry field list,
    so a per-process difference would BOTH skew the field schema AND make one
    rank raise here while another blocked in the schema collective.
    """
    _refuse_unsupported_spmd_config(model)
    polar = bool(getattr(model.config, "use_polar_filter", False))
    if p_lon > 1 and polar:
        raise NotImplementedError(
            "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 is "
            "not wired — the polar filter FFTs the full longitude circle "
            "and needs a lon-gather FFT.  (The route-A MPI path "
            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
            "lat-pencil transpose; the SPMD ppermute equivalent is a "
            "follow-up.)  Use p_lon == 1 or disable the filter.")


def make_sharded_atm_latlon_step_2d(model, mesh, physics_fn=None, *,
                                    shard_geometry: bool = True):
    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic
    atm step 2-D-tile-SPMD over a ``("lat", "lon")`` mesh — the M3a native
#!/bin/bash -l
#SBATCH --job-name=cpu_ll2d
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=cpu_ll2d.%j.log
# HUNDREDS-OF-CPUS lat-lon atmosphere (user directive 2026-08-02): the
# 2-D pencil path (wall poles, labelled throughput lane — NOT the pole
# fold; pole-matched A/B remains open) at r512 (512x1024 = 524k cols) so
# np512 still holds ~1k cols/rank.  Self-contained strong ladder
# np64->512, f64 moist, 32 rpn, block:cyclic per the placement receipt.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cpu_ll2d_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
for NP in 64 128 256 512; do
  NODES=$(( NP / 32 )); [ "$NODES" -lt 1 ] && NODES=1
  echo "--- atm latlon 2-D pencil r512 f64 np=$NP ---"
  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
      --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
    "$PY" scripts/bench/run_cpu_mpi_scaling.py \
      --grid latlon --latlon-2d --mode single --physics moist \
      --precision float64 --n-levels 26 --resolution 512 \
      --n-warmup 2 --n-timing 6 \
      --output-dir "$OUTDIR/np${NP}" < /dev/null || { echo "np$NP FAILED"; rc=1; }
done
echo "=== RESULTS ==="
for NP in 64 128 256 512; do
  F=$(find "$OUTDIR/np${NP}" -name "*.json" 2>/dev/null | head -1)
  "$PY" -c "
import json,math,sys
try:
    d=json.load(open('$F' or '/nonexistent'))
    r=d['results'][-1] if isinstance(d.get('results'),list) else d
    ms=r.get('ms_per_step') or r.get('time_per_step_ms')
    assert ms is not None and math.isfinite(ms) and ms > 0
except Exception as e:
    print('np$NP: MISSING/INVALID ->', e); sys.exit(1)
print(f'np$NP: {ms:9.2f} ms')" || { echo "np$NP receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
packages/core/legoesm/parallel/latlon_spmd.py:164:    zeros.  Mirrors :func:`make_latlon_band_wall_multi_pad_body`'s "per dtype
packages/core/legoesm/parallel/latlon_spmd.py:349:    :func:`make_latlon_2d_pad_body`.
packages/core/legoesm/parallel/latlon_spmd.py:409:    :func:`make_latlon_band_pad_body` (1-D) or :func:`make_latlon_2d_pad_body`
packages/core/legoesm/parallel/latlon_spmd.py:480:def make_latlon_2d_pad_body(mesh, halo: int = 1, negate: bool = False):
packages/core/legoesm/parallel/latlon_spmd.py:640:def make_latlon_band_wall_multi_pad_body(mesh, halo: int = 1,
packages/core/legoesm/parallel/latlon_spmd.py:644:    """FUSED multi-field twin of :func:`make_latlon_band_wall_pad_body`.
packages/core/legoesm/parallel/latlon_spmd.py:653:    already fuses via ``pad_with_pole_bc_lat_multi_mpi``, the SPMD leg
packages/core/legoesm/parallel/latlon_spmd.py:669:            f"make_latlon_band_wall_multi_pad_body: needs a mesh carrying "
packages/core/legoesm/parallel/latlon_spmd.py:889:        parity (``make_latlon_2d_pad_body._fold_rows``, uniform program —
packages/core/legoesm/grids/halo_latlon.py:152:    :func:`legoesm.parallel.latlon_spmd.make_latlon_2d_pad_body` — lat
packages/core/legoesm/grids/halo_latlon.py:164:        from legoesm.parallel.latlon_spmd import make_latlon_2d_pad_body
packages/core/legoesm/grids/halo_latlon.py:165:        return make_latlon_2d_pad_body(mesh, halo=halo, negate=negate)(data)
packages/core/legoesm/grids/halo_latlon.py:707:def pad_with_pole_bc_lat_multi(
packages/core/legoesm/grids/halo_latlon.py:760:            "pad_with_pole_bc_lat_multi: south_values/north_values must "
packages/core/legoesm/grids/halo_latlon.py:780:                make_latlon_band_wall_multi_pad_body,
packages/core/legoesm/grids/halo_latlon.py:782:            body = make_latlon_band_wall_multi_pad_body(
packages/core/legoesm/grids/halo_latlon.py:792:            pad_with_pole_bc_lat_multi_mpi,
packages/core/legoesm/grids/halo_latlon.py:796:            return pad_with_pole_bc_lat_multi_mpi(
packages/core/legoesm/grids/halo_latlon.py:883:    (:func:`pad_with_pole_bc_lat_multi`); wall-zero at physical poles,
packages/core/legoesm/grids/halo_latlon.py:888:    padded = pad_with_pole_bc_lat_multi(fields, halo=halo)
packages/core/legoesm/grids/halo_latlon.py:921:    padded = pad_with_pole_bc_lat_multi(interiors, halo=halo + 1)

exec
/usr/bin/bash -lc 'rg -n "edge.colou|edge_color|edge.*colour|ppermute|collective_permute|halo.*schedule" packages/atmosphere/legoesm/atmosphere packages/core/legoesm/parallel scripts/bench/bench_atm_mpas* scripts/bench | head -n 360' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 394ms:
rg: scripts/bench/bench_atm_mpas*: No such file or directory (os error 2)
scripts/bench/bench_mpas_spmd_scaling.py:3:ppermute halo).
scripts/bench/bench_mpas_spmd_scaling.py:10:only the partition-boundary halo per stage via ``jax.lax.ppermute``.
scripts/bench/bench_mpas_spmd_scaling.py:17:nlev caveat (#1113): the multi-node ceiling at THIN nlev is the ppermute ROUND
scripts/bench/bench_mpas_spmd_scaling.py:18:count (``hlo_collective_permutes``, recorded per row) x the ~0.11 ms launch
scripts/bench/bench_mpas_spmd_scaling.py:34:``jax.devices()``, and the existing ``make_voronoi_sharded_step`` ppermute
scripts/bench/bench_mpas_spmd_scaling.py:41:corrupt the halo schedule).
scripts/bench/bench_mpas_spmd_scaling.py:80:# of the sharded step (ppermute halo + mass-fix psum reduction-order change),
scripts/bench/bench_mpas_spmd_scaling.py:206:                   choices=["auto", "ppermute", "allgather"],
scripts/bench/bench_mpas_spmd_scaling.py:210:                        "cell threshold — force 'ppermute' to exercise "
scripts/bench/bench_mpas_spmd_scaling.py:320:        # partitions agree before any collective uses the halo schedule.
scripts/bench/bench_mpas_spmd_scaling.py:322:        # connectivity the ppermute schedule + TRiSK stencils read; a
scripts/bench/bench_mpas_spmd_scaling.py:419:    # the sharded step — the ppermute ROUND count that decomposes multi-node
scripts/bench/bench_mpas_spmd_scaling.py:426:    # no ppermute halo -> 0.
scripts/bench/bench_mpas_spmd_scaling.py:432:    # wall) is the collective_permute member, so no second compile for it.
scripts/bench/bench_mpas_spmd_scaling.py:434:    hlo_cp = hlo_census["collective_permute"] if hlo_census else None
scripts/bench/bench_mpas_spmd_scaling.py:505:        # saying "auto" would not reveal whether ppermute or allgather
scripts/bench/bench_mpas_spmd_scaling.py:519:        # ppermute round count/step (static compile property; #1113) — the
scripts/bench/bench_mpas_spmd_scaling.py:522:        hlo_collective_permutes=hlo_cp,
scripts/bench/run_cpu_mpi_scaling.py:575:    ``make_sharded_step`` + multiface-ppermute halo runs unchanged, and
scripts/bench/run_cpu_mpi_scaling.py:658:        # exchange (which rides the SAME multiface-ppermute halo as T under the
scripts/bench/run_cpu_mpi_scaling.py:1536:             "ppermute; the A1 path).  Replaces the replicated-dynamics "
scripts/bench/bench_cube_shardmap_halo.py:7:(``jax.lax.ppermute`` inside ``shard_map``) scales, unlike the ``mpi4jax``
scripts/bench/bench_cube_shardmap_halo.py:20:* **AD** — a nonuniform-cotangent VJP through the SPMD ``ppermute`` halo matches
scripts/bench/bench_cube_shardmap_halo.py:22:  contain a ``collective_permute`` — the ppermute kernel's signature (the
scripts/bench/bench_cube_shardmap_halo.py:23:  all_gather diagnostic kernel must NOT satisfy this) — proving the ppermute path
scripts/bench/bench_cube_shardmap_halo.py:24:  ran, not a silent local fallback. This is the ``collective_permute`` VJP
scripts/bench/bench_cube_shardmap_halo.py:90:    ppermute/all_gather kernel-selection flag through the public API (importing
scripts/bench/bench_cube_shardmap_halo.py:91:    the private ``_use_ppermute`` across modules is disallowed by repo policy), so
scripts/bench/bench_cube_shardmap_halo.py:92:    reactivation resets it to the default ppermute selection and warns.
scripts/bench/bench_cube_shardmap_halo.py:103:            "restoring a prior SPMD halo backend: the private ppermute/all_gather "
scripts/bench/bench_cube_shardmap_halo.py:104:            "selection is reset to the default ppermute (not snapshottable via the "
scripts/bench/bench_cube_shardmap_halo.py:133:def _hlo_has_ppermute(fn, *args) -> bool | None:
scripts/bench/bench_cube_shardmap_halo.py:135:    ``collective_permute`` — the signature of the *ppermute* cube-halo kernel?
scripts/bench/bench_cube_shardmap_halo.py:139:    bandwidth-optimal ppermute path; the all_gather diagnostic kernel replicates
scripts/bench/bench_cube_shardmap_halo.py:154:                       "NOT the ppermute collective this experiment measures.")
scripts/bench/bench_cube_shardmap_halo.py:240:    SPMD (``ppermute``/``shard_map``) backend over a face-sharded mesh. For SPMD
scripts/bench/bench_cube_shardmap_halo.py:303:        hlo_collective = _hlo_has_ppermute(lambda st: model.step(st, dt), s0)
scripts/bench/bench_cube_shardmap_halo.py:346:# AD gate: VJP through the SPMD ppermute halo vs single-device reference
scripts/bench/bench_cube_shardmap_halo.py:354:    The lowered SPMD function is also asserted to contain a ``collective_permute``
scripts/bench/bench_cube_shardmap_halo.py:355:    (the ppermute kernel's signature), proving the ppermute path ran rather than a
scripts/bench/bench_cube_shardmap_halo.py:383:        # --- SPMD VJP (ppermute) ---
scripts/bench/bench_cube_shardmap_halo.py:391:            has_collective = _hlo_has_ppermute(_f, x_sh)
scripts/bench/bench_cube_shardmap_halo.py:474:            # Require the ppermute collective to be PROVEN present (True). None
scripts/bench/bench_cube_shardmap_halo.py:476:            # have exercised the cross-device ppermute VJP, not a local fallback.
scripts/bench/bench_cube_shardmap_halo.py:492:    # --- Timed-path ppermute gate -----------------------------------------
scripts/bench/bench_cube_shardmap_halo.py:493:    # The AD gate proves ``explicit_pad_halo`` lowered ppermute, but the
scripts/bench/bench_cube_shardmap_halo.py:495:    # at every >1-device count actually used the ppermute collective (else we may
scripts/bench/bench_cube_shardmap_halo.py:498:    timed_ppermute_pass = (
scripts/bench/bench_cube_shardmap_halo.py:513:             "efficiency": efficiency_pass, "timed_ppermute": timed_ppermute_pass}
scripts/bench/bench_cube_shardmap_halo.py:518:        enforced += ["efficiency", "timed_ppermute"]
scripts/bench/bench_cube_shardmap_halo.py:632:    logger.info("POINT n_devices=%d (procs=%d) ms/step=%.3f hlo_ppermute=%s",
scripts/bench/aggregate_cube_shardmap_scaling.py:13:* **ppermute proof** — every ``n_devices > 1`` point lowered a
scripts/bench/aggregate_cube_shardmap_scaling.py:14:  ``collective_permute`` (``spmd_hlo_collective is True``), so the measured
scripts/bench/aggregate_cube_shardmap_scaling.py:15:  executable is the bandwidth-optimal ppermute path, not a replicated all_gather
scripts/bench/aggregate_cube_shardmap_scaling.py:157:    # --- ppermute proof: every >1-device point lowered collective_permute ---
scripts/bench/aggregate_cube_shardmap_scaling.py:159:    ppermute_pass = all(r["spmd_hlo_collective"] is True for r in multi) if multi else True
scripts/bench/aggregate_cube_shardmap_scaling.py:197:             "ppermute": ppermute_pass, "efficiency": efficiency_pass}
scripts/bench/aggregate_cube_shardmap_scaling.py:202:        enforced.append("ppermute")
scripts/bench/aggregate_cube_shardmap_scaling.py:248:             "| n_devices | ms/step | speedup | efficiency | ppermute | procs |",
scripts/bench/bench_ocean_mpi_scaling.py:54:MPI-halo custom_vjp).  Every rank runs the identical collective schedule
scripts/bench/bench_ocean_mpi_scaling.py:1139:        # the op/halo schedule, not the numerics, is what we time).
scripts/bench/bench_ocean_mpi_scaling.py:1586:        # current transport (correction is then ~0, but the op/halo schedule
scripts/bench/bench_ocean_latlon_spmd_scaling.py:25:and the existing ``make_sharded_ocean_step`` band-ppermute halo + psum
scripts/bench/bench_ocean_latlon_spmd_scaling.py:82:# of the sharded split-explicit barotropic (ppermute/psum reduction-order
scripts/bench/bench_ocean_latlon_spmd_scaling.py:129:    # barotropic substeps instead of ~4 ppermute pads per substep.  The wide
scripts/bench/bench_ocean_latlon_spmd_scaling.py:310:                        "(LEGOESM_LATLON_SPMD_FUSED_HALO=1): one ppermute "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:319:                        "instead of ~4 ppermute pads per substep (implies "
scripts/bench/bench_gather_vs_slice_stencil.py:65:    # same trick as bench_ppermute_microbench.
scripts/bench/bench_ppermute_microbench.py:1:"""Measure this machine's ppermute latency + bandwidth for roofline lines.
scripts/bench/bench_ppermute_microbench.py:10:collective the sharded steps use (``jax.lax.ppermute`` on a ring, inside a
scripts/bench/bench_ppermute_microbench.py:27:    python scripts/bench/bench_ppermute_microbench.py --n-devices 4
scripts/bench/bench_ppermute_microbench.py:31:        python scripts/bench/bench_ppermute_microbench.py \
scripts/bench/bench_ppermute_microbench.py:56:    """jit'd program doing n_reps back-to-back ring ppermutes on device."""
scripts/bench/bench_ppermute_microbench.py:63:                return jax.lax.ppermute(v, axis_name=AXIS, perm=perm)
scripts/bench/bench_ppermute_microbench.py:94:    """Per-ppermute time with HOST DISPATCH SUBTRACTED.
scripts/bench/bench_ppermute_microbench.py:138:                   help="Back-to-back ppermutes inside ONE jit call; the "
scripts/bench/bench_ppermute_microbench.py:153:            f"ppermute needs >=2 devices; got {n_dev}. A latency/bandwidth "
scripts/bench/bench_ppermute_microbench.py:189:        "collective": "ppermute_ring",
scripts/bench/bench_atm_latlon_spmd_scaling.py:43:``make_sharded_atm_latlon_step`` + band-ppermute halo runs unchanged — the
scripts/bench/bench_atm_latlon_spmd_scaling.py:44:ppermute/psum collectives cross processes via the distributed runtime (NCCL on
scripts/bench/bench_ocean_latlon_spmd_pcg.py:10:``lax.ppermute`` + pole fold; CG dots via ``jax.lax.psum``).  Pure jax, NO
scripts/bench/metadata.py:103:# Match the OP-CALL form ``collective-permute(`` / ``collective_permute(`` /
scripts/bench/metadata.py:106:# ``xla_gpu_collective_permute_combine_threshold_bytes=`` (set by #1175's
scripts/bench/metadata.py:125:def count_collective_permutes(hlo_text: str) -> int:
scripts/bench/metadata.py:126:    """Count ``collective_permute`` OPS in a lowered/compiled HLO text dump.
scripts/bench/metadata.py:128:    The ppermute halo kernel's signature and a STATIC compile property (the
scripts/bench/metadata.py:133:    so config-header flag names that merely CONTAIN "collective_permute" never
scripts/bench/metadata.py:135:    ``hlo_collective_permutes`` (cube tiled, MPAS ico) — no re-implementation."""
scripts/bench/metadata.py:139:def hlo_collective_permutes(fn, *args) -> int | None:
scripts/bench/metadata.py:155:        return count_collective_permutes(text) if text else None
scripts/bench/metadata.py:174:#: bit-identical to :func:`count_collective_permutes`); ``all-reduce`` is the
scripts/bench/metadata.py:179:    "collective_permute": _COLLECTIVE_PERMUTE_RE,
scripts/bench/metadata.py:190:    Superset of :func:`count_collective_permutes`: adds ``all_reduce`` (the
scripts/bench/metadata.py:208:    Superset of :func:`hlo_collective_permutes` — compiles (so combined /
scripts/bench/metadata.py:216:    combining / pipelined-p2p (``--xla_gpu_collective_permute_combine_*``, lane
scripts/bench/run_scaling_diagnosis.py:751:                      f"{c['collective_permute']} collective-permute, "
packages/core/legoesm/parallel/sharded_dynamics.py:14:   ``jax.lax.ppermute``-like collective inside shard_map.
packages/core/legoesm/parallel/sharded_dynamics.py:701:        ``activate_spmd_halo_backend``.  The old ppermute-vs-all_gather
packages/core/legoesm/parallel/sharded_dynamics.py:702:        volume auto-selection is RETIRED: ppermute is always selected;
packages/core/legoesm/parallel/sharded_dynamics.py:717:    then routes through shard_map ppermute kernels (multiface; one-face
packages/core/legoesm/parallel/sharded_dynamics.py:740:    # ppermute-multiface refit then made ppermute the DEFAULT exchange
packages/core/legoesm/parallel/sharded_dynamics.py:748:    # 6*kt^2 sub-face tiling: the tiled ppermute EXCHANGE is serial-
packages/core/legoesm/parallel/sharded_dynamics.py:1269:        Per-device partition descriptors (for ppermute schedule building).
packages/core/legoesm/parallel/sharded_dynamics.py:1500:def _greedy_edge_coloring_ordered(comm_pairs, order):
packages/core/legoesm/parallel/sharded_dynamics.py:1508:    edge_colors: dict[tuple[int, int], int] = {}
packages/core/legoesm/parallel/sharded_dynamics.py:1514:        edge_colors[(u, v)] = color
packages/core/legoesm/parallel/sharded_dynamics.py:1517:    return edge_colors
packages/core/legoesm/parallel/sharded_dynamics.py:1520:def _greedy_edge_coloring(comm_pairs):
packages/core/legoesm/parallel/sharded_dynamics.py:1522:    baseline for :func:`_multi_ordering_edge_coloring`). Worst case
packages/core/legoesm/parallel/sharded_dynamics.py:1523:    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
packages/core/legoesm/parallel/sharded_dynamics.py:1528:    return _greedy_edge_coloring_ordered(comm_pairs, edges)
packages/core/legoesm/parallel/sharded_dynamics.py:1531:def _check_proper_edge_coloring(edge_colors, comm_pairs):
packages/core/legoesm/parallel/sharded_dynamics.py:1535:    if set(edge_colors) != {tuple(sorted(p)) for p in comm_pairs}:
packages/core/legoesm/parallel/sharded_dynamics.py:1538:    for (u, v), c in edge_colors.items():
packages/core/legoesm/parallel/sharded_dynamics.py:1548:# schedule (the coloring must agree across ranks or the ppermute pattern
packages/core/legoesm/parallel/sharded_dynamics.py:1554:def _multi_ordering_edge_coloring(comm_pairs):
packages/core/legoesm/parallel/sharded_dynamics.py:1556:    using the FEWEST colors (= ppermute rounds) across several deterministic
packages/core/legoesm/parallel/sharded_dynamics.py:1568:    shuffles). Returns ``(edge_colors, max_degree)``.
packages/core/legoesm/parallel/sharded_dynamics.py:1593:        ec = _greedy_edge_coloring_ordered(comm_pairs, order)
packages/core/legoesm/parallel/sharded_dynamics.py:1602:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py:1604:    """Build a ppermute-based halo exchange schedule.
packages/core/legoesm/parallel/sharded_dynamics.py:1607:    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
packages/core/legoesm/parallel/sharded_dynamics.py:1609:    so that each round of ppermute moves data between non-conflicting
packages/core/legoesm/parallel/sharded_dynamics.py:1624:        ppermute_perms, send_cell_idx, recv_cell_pos,
packages/core/legoesm/parallel/sharded_dynamics.py:1670:            'ppermute_perms': [],
packages/core/legoesm/parallel/sharded_dynamics.py:1680:    # 3. Edge-color the graph: each color = one bidirectional ppermute
packages/core/legoesm/parallel/sharded_dynamics.py:1689:    greedy_colors = _greedy_edge_coloring(comm_pairs)
packages/core/legoesm/parallel/sharded_dynamics.py:1691:    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
packages/core/legoesm/parallel/sharded_dynamics.py:1699:        edge_colors, n_rounds, coloring_method = (
packages/core/legoesm/parallel/sharded_dynamics.py:1702:        edge_colors, n_rounds, coloring_method = (
packages/core/legoesm/parallel/sharded_dynamics.py:1704:    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
packages/core/legoesm/parallel/sharded_dynamics.py:1705:        "improper ppermute edge coloring — two same-round exchanges "
packages/core/legoesm/parallel/sharded_dynamics.py:1708:    for (u, v), color in edge_colors.items():
packages/core/legoesm/parallel/sharded_dynamics.py:1740:    # 5. Assemble per-round ppermute patterns and index arrays
packages/core/legoesm/parallel/sharded_dynamics.py:1742:    ppermute_perms_out: list[list[tuple[int, int]]] = []
packages/core/legoesm/parallel/sharded_dynamics.py:1763:        # Bidirectional ppermute pattern
packages/core/legoesm/parallel/sharded_dynamics.py:1771:        ppermute_perms_out.append(perm)
packages/core/legoesm/parallel/sharded_dynamics.py:1811:        'ppermute_perms': ppermute_perms_out,
packages/core/legoesm/parallel/sharded_dynamics.py:1906:def _ppermute_halo_fill(cell_pack, u_shard, halo_sl, ppermute_perms,
packages/core/legoesm/parallel/sharded_dynamics.py:1908:    """Fill (owned + halo) local buffers from owned shards via ppermute.
packages/core/legoesm/parallel/sharded_dynamics.py:1914:    posts ONE flat ppermute carrying BOTH entity classes for ALL packed
packages/core/legoesm/parallel/sharded_dynamics.py:1923:    schedule.  ``ppermute_perms`` is the static per-round permutation.
packages/core/legoesm/parallel/sharded_dynamics.py:1941:        recv_packed = jax.lax.ppermute(
packages/core/legoesm/parallel/sharded_dynamics.py:1942:            send_packed, "device", perm=ppermute_perms[r])
packages/core/legoesm/parallel/sharded_dynamics.py:1957:    ppermute_cells_per_device_threshold: int = 2_000,
packages/core/legoesm/parallel/sharded_dynamics.py:1973:       flat ppermute payload per neighbor round.
packages/core/legoesm/parallel/sharded_dynamics.py:1981:    leading device axis), the ppermute schedule index arrays, and the
packages/core/legoesm/parallel/sharded_dynamics.py:2002:        ``"auto"`` (default) selects ``"ppermute"`` for large grids and
packages/core/legoesm/parallel/sharded_dynamics.py:2004:        *ppermute_cells_per_device_threshold*.
packages/core/legoesm/parallel/sharded_dynamics.py:2005:        ``"ppermute"`` forces neighbor-only exchange via
packages/core/legoesm/parallel/sharded_dynamics.py:2006:        ``jax.lax.ppermute`` — O(halo) communication.
packages/core/legoesm/parallel/sharded_dynamics.py:2009:    ppermute_cells_per_device_threshold : int
packages/core/legoesm/parallel/sharded_dynamics.py:2010:        When ``halo_strategy="auto"``, use ppermute only if each device
packages/core/legoesm/parallel/sharded_dynamics.py:2012:        per-round packing/scatter overhead of ppermute exceeds the
packages/core/legoesm/parallel/sharded_dynamics.py:2103:        if cells_per < ppermute_cells_per_device_threshold:
packages/core/legoesm/parallel/sharded_dynamics.py:2107:                "threshold=%d — ppermute packing overhead would dominate.",
packages/core/legoesm/parallel/sharded_dynamics.py:2108:                cells_per, ppermute_cells_per_device_threshold,
packages/core/legoesm/parallel/sharded_dynamics.py:2111:            halo_strategy = "ppermute"
packages/core/legoesm/parallel/sharded_dynamics.py:2113:                "Auto-selected ppermute strategy: cells_per_device=%d >= "
packages/core/legoesm/parallel/sharded_dynamics.py:2115:                cells_per, ppermute_cells_per_device_threshold,
packages/core/legoesm/parallel/sharded_dynamics.py:2135:        partitions_out,   # list[VoronoiPartition] (for ppermute schedule)
packages/core/legoesm/parallel/sharded_dynamics.py:2162:    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
packages/core/legoesm/parallel/sharded_dynamics.py:2165:    use_ppermute = halo_strategy == "ppermute"
packages/core/legoesm/parallel/sharded_dynamics.py:2167:    if use_ppermute:
packages/core/legoesm/parallel/sharded_dynamics.py:2168:        # Build ppermute schedule: neighbor-only halo exchange
packages/core/legoesm/parallel/sharded_dynamics.py:2170:        pp_sched = _build_ppermute_schedule(
packages/core/legoesm/parallel/sharded_dynamics.py:2175:        ppermute_perms = pp_sched['ppermute_perms']
packages/core/legoesm/parallel/sharded_dynamics.py:2204:            "  ppermute schedule: %d rounds, max halo cells/edges per round: %s / %s",
packages/core/legoesm/parallel/sharded_dynamics.py:2210:            "  comm volume per stage: ppermute ~%.1f KB vs allgather ~%.1f KB (%.1fx reduction)",
packages/core/legoesm/parallel/sharded_dynamics.py:2215:        logger.info("  ppermute schedule built in %.3fs", time.time() - t1)
packages/core/legoesm/parallel/sharded_dynamics.py:2245:            stacked local meshes and the halo schedule (leading axis 1).
packages/core/legoesm/parallel/sharded_dynamics.py:2253:            if use_ppermute:
packages/core/legoesm/parallel/sharded_dynamics.py:2254:                cell_local, u_local = _ppermute_halo_fill(
packages/core/legoesm/parallel/sharded_dynamics.py:2255:                    cell_pack, u_shard, halo_sl, ppermute_perms,
packages/core/legoesm/parallel/sharded_dynamics.py:2379:        doctrine); the local-mesh / halo-schedule / area constants are
packages/core/legoesm/parallel/latlon_spmd.py:10:  * latitude interior band cuts: ``jax.lax.ppermute`` of the ``halo`` edge rows
packages/core/legoesm/parallel/latlon_spmd.py:20:foundation for the ocean lat-lon multi-GPU SPMD step (pure-jax ppermute over the
packages/core/legoesm/parallel/latlon_spmd.py:86:    columns, moved by ``jax.lax.ppermute`` over the ``"lon"`` mesh axis (the
packages/core/legoesm/parallel/latlon_spmd.py:97:    (trailing axes ride through).  AD-safe: ``ppermute`` is
packages/core/legoesm/parallel/latlon_spmd.py:110:    east_ghost = jax.lax.ppermute(f[:, :halo], "lon", perm_to_west)
packages/core/legoesm/parallel/latlon_spmd.py:111:    west_ghost = jax.lax.ppermute(f[:, -halo:], "lon", perm_to_east)
packages/core/legoesm/parallel/latlon_spmd.py:124:    ``ppermute(..., perm_north)``; the north-most band has no neighbour there and
packages/core/legoesm/parallel/latlon_spmd.py:125:    receives the pole-wall zero (the ppermute non-target). Pure array core (no
packages/core/legoesm/parallel/latlon_spmd.py:128:    ``_reconstruct_v`` closure). AD-safe: ``ppermute`` is self-transposing.
packages/core/legoesm/parallel/latlon_spmd.py:140:    boundary = jax.lax.ppermute(v_lower[0:1], axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py:156:    aggregation, scaling-M4): ONE ``ppermute`` per DTYPE GROUP for the whole
packages/core/legoesm/parallel/latlon_spmd.py:163:    split are layout ops), including the north band's ppermute non-target
packages/core/legoesm/parallel/latlon_spmd.py:201:        # ONE ppermute for the whole dtype group (north band receives 0).
packages/core/legoesm/parallel/latlon_spmd.py:202:        recv = jax.lax.ppermute(buf, axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py:231:    periodic so EVERY tile is a ppermute target — the wrap pair
packages/core/legoesm/parallel/latlon_spmd.py:252:        boundary = jax.lax.ppermute(u_left[:, 0:1], axis, perm_to_west)
packages/core/legoesm/parallel/latlon_spmd.py:347:    """Tile's padded 180-deg pole-fold window via ONE antipodal ``ppermute``
packages/core/legoesm/parallel/latlon_spmd.py:372:    AD-safe: ``ppermute`` is self-transposing; the window ``take`` is a
packages/core/legoesm/parallel/latlon_spmd.py:387:    # 1. E/W-extend the edge rows by 2h (one lon ring ppermute pair).
packages/core/legoesm/parallel/latlon_spmd.py:389:    # 2. ONE antipodal ppermute over the lon ring (shift by p_lon/2 is a
packages/core/legoesm/parallel/latlon_spmd.py:392:    recv = jax.lax.ppermute(ext, "lon", perm_anti)
packages/core/legoesm/parallel/latlon_spmd.py:445:    ppermute (interior) + pole fold (ends).  ``negate=True`` folds with a sign
packages/core/legoesm/parallel/latlon_spmd.py:462:        # 2. latitude band ppermute of the edge rows (axis 0 = south->north,
packages/core/legoesm/parallel/latlon_spmd.py:466:        north_recv = jax.lax.ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
packages/core/legoesm/parallel/latlon_spmd.py:467:        south_recv = jax.lax.ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge
packages/core/legoesm/parallel/latlon_spmd.py:469:        # 3. pole fold at the end bands (ppermute non-targets receive zeros).
packages/core/legoesm/parallel/latlon_spmd.py:491:    * latitude interior cuts: ``ppermute`` of the ``halo`` edge rows over
packages/core/legoesm/parallel/latlon_spmd.py:494:      ring ``ppermute`` over ``"lon"`` (``p_lon == 1``: the LOCAL wrap, a
packages/core/legoesm/parallel/latlon_spmd.py:506:      EVEN ``p_lon`` (with ``w >= 2h``): ONE antipodal ``ppermute`` of the
packages/core/legoesm/parallel/latlon_spmd.py:522:    AD-safe: ``ppermute`` is self-transposing, ``all_gather`` has a defined
packages/core/legoesm/parallel/latlon_spmd.py:533:        antipodal ppermute / all_gather — static branch on ``p_lon``)."""
packages/core/legoesm/parallel/latlon_spmd.py:541:            # 180-deg partner-tile ppermute (the documented follow-up,
packages/core/legoesm/parallel/latlon_spmd.py:554:        # 1. latitude ppermute of the (pre-lon-pad) edge rows over "lat".
packages/core/legoesm/parallel/latlon_spmd.py:556:        # no lat neighbour exists, so skip the (empty-perm) ppermute
packages/core/legoesm/parallel/latlon_spmd.py:562:            north_recv = jax.lax.ppermute(tile[:halo], "lat", perm_north)
packages/core/legoesm/parallel/latlon_spmd.py:563:            south_recv = jax.lax.ppermute(tile[-halo:], "lat", perm_south)
packages/core/legoesm/parallel/latlon_spmd.py:570:        # 3. pole fold at the physical pole tiles (ppermute non-targets
packages/core/legoesm/parallel/latlon_spmd.py:593:    ppermute of the edge rows at INTERIOR cuts (so the cut ghost row is the
packages/core/legoesm/parallel/latlon_spmd.py:607:    axis — the SAME ppermute, keyed on ``mesh.shape["lat"]`` (identical to
packages/core/legoesm/parallel/latlon_spmd.py:619:        # lat-band ppermute of the edge rows (axis 0 = south->north; row 0 is the
packages/core/legoesm/parallel/latlon_spmd.py:624:        north_recv = jax.lax.ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
packages/core/legoesm/parallel/latlon_spmd.py:625:        south_recv = jax.lax.ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge
packages/core/legoesm/parallel/latlon_spmd.py:627:        # CONSTANT wall pad at the physical pole end bands (ppermute non-targets
packages/core/legoesm/parallel/latlon_spmd.py:646:    One ``ppermute`` pair per DIRECTION for the whole field GROUP instead of
packages/core/legoesm/parallel/latlon_spmd.py:659:    (and one ppermute pair) per dtype group, matching the MPI fused path's
packages/core/legoesm/parallel/latlon_spmd.py:708:            # ONE ppermute pair for the whole dtype group.
packages/core/legoesm/parallel/latlon_spmd.py:709:            north_recv = jax.lax.ppermute(south_buf, axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py:710:            south_recv = jax.lax.ppermute(north_buf, axis, perm_south)
packages/core/legoesm/parallel/latlon_spmd.py:816:      face from the neighbour band's edge row (``ppermute`` via the armed spmd
packages/core/legoesm/parallel/latlon_spmd.py:870:    COMMUNICATION VOLUME of the actual pad program (ppermute perimeter PLUS
packages/core/legoesm/parallel/latlon_spmd.py:879:        exchange, moves edge blocks of that depth in ONE ppermute hop).
packages/core/legoesm/parallel/latlon_spmd.py:885:      * lat cut (``p_lat > 1``): the N/S ppermute pair moves ``2h*w`` cells
packages/core/legoesm/parallel/latlon_spmd.py:887:      * lon cut (``p_lon > 1``): the E/W ring ppermute pair moves ``~2h*nl``
packages/core/legoesm/parallel/latlon_spmd.py:939:        # ppermute pair (w) + E/W ring pair (nl) + the pole-fold term —
packages/core/legoesm/parallel/latlon_spmd.py:940:        # even p_lon (w >= 2*_FOLD_REF_HALO): partner-ppermute fold
scripts/bench/roofline_probe.py:9:(``legoesm.parallel`` ppermute / sendrecv / batch_allreduce) and times
scripts/bench/roofline_probe.py:22:   (``measure_ppermute_collective`` + ``measure_sendrecv_collective``):
scripts/bench/roofline_probe.py:24:     (a) GPU/SPMD ``jax.lax.ppermute`` PRIMITIVE inside a ``shard_map``
scripts/bench/roofline_probe.py:26:         cubed-sphere halo (``cubesphere_exchange._make_exchange_ppermute``
scripts/bench/roofline_probe.py:109:GPU (1 or 2 devices; ppermute needs >=2, gracefully skipped on 1)::
scripts/bench/roofline_probe.py:260:    have data-dependent outputs (triad, ppermute, sendrecv), so XLA
scripts/bench/roofline_probe.py:455:    transport: str           # "ppermute_spmd" | "mpi4jax_sendrecv".
scripts/bench/roofline_probe.py:467:def measure_ppermute_collective(
scripts/bench/roofline_probe.py:475:    """Time the ``jax.lax.ppermute`` PRIMITIVE inside a ``shard_map`` over
scripts/bench/roofline_probe.py:477:    devices are visible (single-GPU / single CPU process) — ppermute
scripts/bench/roofline_probe.py:480:    Scope (codex MAJOR — do NOT over-claim): this measures the *ppermute
scripts/bench/roofline_probe.py:483:    (``cubesphere_exchange._make_exchange_ppermute`` uses ``lax.ppermute``
scripts/bench/roofline_probe.py:497:    DEPENDS on the ppermute result, so XLA cannot DCE the collective.
scripts/bench/roofline_probe.py:510:        transport="ppermute_spmd",
scripts/bench/roofline_probe.py:529:                got = jax.lax.ppermute(local, "face", perm)
scripts/bench/roofline_probe.py:537:        # whose value DEPENDS on the ppermute, so the collective cannot be
scripts/bench/roofline_probe.py:560:    """Tiny adapter so ``measure_ppermute_collective`` reads cleanly:
scripts/bench/roofline_probe.py:562:    ``check_vma=False`` (matching the model's ppermute kernel, which also
scripts/bench/roofline_probe.py:875:    hlo_collective_permute: int
scripts/bench/roofline_probe.py:876:    hlo_collective_permute_start: int
scripts/bench/roofline_probe.py:877:    hlo_collective_permute_done: int
scripts/bench/roofline_probe.py:905:    hlo_collective_permute: int = -1,
scripts/bench/roofline_probe.py:906:    hlo_collective_permute_start: int = -1,
scripts/bench/roofline_probe.py:907:    hlo_collective_permute_done: int = -1,
scripts/bench/roofline_probe.py:920:    ``run_levante_gpu_scaling.TimingResult`` (``hlo_collective_permute*``)
scripts/bench/roofline_probe.py:960:    if collective_latency_ms is not None and hlo_collective_permute >= 0:
scripts/bench/roofline_probe.py:961:        n_rounds = max(hlo_collective_permute, 0) + max(
scripts/bench/roofline_probe.py:962:            hlo_collective_permute_start, 0)
scripts/bench/roofline_probe.py:966:            f"(sync={max(hlo_collective_permute,0)} + "
scripts/bench/roofline_probe.py:967:            f"async-start={max(hlo_collective_permute_start,0)}) × "
scripts/bench/roofline_probe.py:1013:        hlo_collective_permute=hlo_collective_permute,
scripts/bench/roofline_probe.py:1014:        hlo_collective_permute_start=hlo_collective_permute_start,
scripts/bench/roofline_probe.py:1015:        hlo_collective_permute_done=hlo_collective_permute_done,
scripts/bench/roofline_probe.py:1285:        ppm = measure_ppermute_collective(
scripts/bench/roofline_probe.py:1291:                print(f"[2a] ppermute SPMD: latency floor "
packages/core/legoesm/parallel/tiled_d2a2c.py:9:adjacent strips are applied IN-STAGE: four 1-cell ``lax.ppermute`` halos on
packages/core/legoesm/parallel/tiled_d2a2c.py:98:        ut_hi = jax.lax.ppermute(ut_t[:, :, 0], "tile_j", perm_hi)
packages/core/legoesm/parallel/tiled_d2a2c.py:99:        ut_lo = jax.lax.ppermute(ut_t[:, :, nl - 1], "tile_j", perm_lo)
packages/core/legoesm/parallel/tiled_d2a2c.py:100:        vt_hi = jax.lax.ppermute(vt_t[:, 0, :], "tile_i", perm_hi)
packages/core/legoesm/parallel/tiled_d2a2c.py:101:        vt_lo = jax.lax.ppermute(vt_t[:, nl - 1, :], "tile_i", perm_lo)
scripts/bench/bench_cube_tiled_step_scaling.py:61:    annotate_incomplete, count_collective_permutes, count_collectives,
scripts/bench/bench_cube_tiled_step_scaling.py:73:# Shared canonical CP census (metadata.count_collective_permutes); kept as a
scripts/bench/bench_cube_tiled_step_scaling.py:75:_count_collective_permutes = count_collective_permutes
scripts/bench/bench_cube_tiled_step_scaling.py:222:    # (the tiled stage manages its own ppermutes through the mesh).
scripts/bench/bench_cube_tiled_step_scaling.py:250:    n_ppermute = _count_collective_permutes(hlo)
scripts/bench/bench_cube_tiled_step_scaling.py:256:    if n_ppermute == 0:
scripts/bench/bench_cube_tiled_step_scaling.py:371:        hlo_collective_permutes=n_ppermute,
scripts/bench/bench_cube_tiled_step_scaling.py:409:            "hlo_collective_permutes": n_ppermute,
scripts/bench/bench_cube_tiled_step_scaling.py:431:              f"ppermutes={n_ppermute}")
scripts/bench/run_levante_gpu_scaling.py:332:    hlo_collective_permute: int = -1
scripts/bench/run_levante_gpu_scaling.py:333:    hlo_collective_permute_start: int = -1
scripts/bench/run_levante_gpu_scaling.py:334:    hlo_collective_permute_done: int = -1
scripts/bench/run_levante_gpu_scaling.py:615:def _count_collective_permute_ops(hlo_text: str) -> dict[str, int]:
scripts/bench/run_levante_gpu_scaling.py:645:        "hlo_collective_permute": hlo_counts["collective-permute"],
scripts/bench/run_levante_gpu_scaling.py:646:        "hlo_collective_permute_start": hlo_counts["collective-permute-start"],
scripts/bench/run_levante_gpu_scaling.py:647:        "hlo_collective_permute_done": hlo_counts["collective-permute-done"],
scripts/bench/run_levante_gpu_scaling.py:804:            hlo_counts = _count_collective_permute_ops(_hlo_text)
scripts/bench/run_levante_gpu_scaling.py:1686:            # Tripwire (codex ppermute-multiface review): the row below is
scripts/bench/run_levante_gpu_scaling.py:2387:                "hlo_collective_permute": r.hlo_collective_permute,
packages/core/legoesm/parallel/halo_exchange.py:62:  - For pure multi-GPU (no MPI), ``jax.lax.ppermute`` is the preferred
packages/core/legoesm/parallel/halo_exchange_voronoi.py:23:    sched = build_batched_halo_schedule(partition.cell_comm,
packages/core/legoesm/parallel/halo_exchange_voronoi.py:52:    build_batched_halo_schedule,  # noqa: F401  (re-export: schedule + exchange live together)
packages/core/legoesm/parallel/halo_exchange_voronoi.py:538:        From :func:`build_batched_halo_schedule` (layout-build constant).
packages/core/legoesm/parallel/tiled_transport.py:5:body with NO in-stage halo ppermute):
packages/core/legoesm/parallel/tiled_transport.py:18:the halo — NO in-stage halo ppermute, the same insight as the d2a2c stage).
packages/core/legoesm/parallel/voronoi_mpi.py:56:    build_batched_halo_schedule,
packages/core/legoesm/parallel/voronoi_mpi.py:141:    PURE (reads ownership masks + the batched-halo schedule constants); issues
packages/core/legoesm/parallel/voronoi_mpi.py:259:        batched_comm=build_batched_halo_schedule(
packages/core/legoesm/parallel/voronoi_mpi.py:405:        sched = build_batched_halo_schedule(
packages/core/legoesm/parallel/voronoi_mpi.py:478:        sched = build_batched_halo_schedule(
packages/core/legoesm/parallel/voronoi_mpi.py:879:            _batched_sched = build_batched_halo_schedule(
packages/core/legoesm/parallel/async_halo.py:90:       (ppermute multiface exchange; all_gather only as the explicit
packages/core/legoesm/parallel/async_halo.py:95:    ``jax.lax.ppermute`` for device-to-device communication instead of MPI.
packages/core/legoesm/parallel/async_halo.py:106:        the experimental ``ppermute`` path for device-to-device halos;
packages/core/legoesm/parallel/async_halo.py:116:    The ppermute implementation is experimental and:
packages/core/legoesm/parallel/async_halo.py:119:    2. Applies ``jax.lax.ppermute`` calls for each edge/halo strip.
packages/core/legoesm/parallel/async_halo.py:128:        # ppermute path only works for face-only sharding (6 faces, no tiles).
packages/core/legoesm/parallel/async_halo.py:133:                "ppermute halo exchange does not support sub-face tiling "
packages/core/legoesm/parallel/async_halo.py:141:            "jax_native_halo_exchange: the ppermute-based code path is "
packages/core/legoesm/parallel/async_halo.py:147:        return _ppermute_halo_exchange(data, grid, mesh)
packages/core/legoesm/parallel/async_halo.py:153:def _ppermute_halo_exchange(data, grid, mesh):
packages/core/legoesm/parallel/async_halo.py:154:    """Implement halo exchange via jax.lax.ppermute (experimental).
packages/core/legoesm/parallel/async_halo.py:214:        strips_permuted = jax.lax.ppermute(
packages/core/legoesm/parallel/tiled_production_cdgrid.py:9:bit-exactly.  NOT Ginsburg-benchable (np>6 = CPU shard_map / cross-node ppermute
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2452:    the halo collective-permutes (``jax.lax.ppermute`` over the
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2470:    1. the halo ``ppermute`` cliques — every table the step's pad body issues
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2511:    # Every ppermute the blocked step's halo pad body issues, in the SAME table
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2530:            acc = acc + jax.lax.ppermute(v, AXES, perm)
packages/core/legoesm/parallel/tiled_production_cdgrid.py:3005:# ``make_tiled_pad_vector_body`` vector) — the SAME ppermute-over-(face,tile_i,
packages/core/legoesm/parallel/tiled_production_cdgrid.py:3184:# -> the per-tile PPM reconstruction is LOCAL (NO in-stage ppermute).  The cc
packages/core/legoesm/parallel/distributed.py:63:    reductions run through pure-JAX ``ppermute``/``psum`` inside ``shard_map``,
packages/core/legoesm/parallel/distributed.py:97:      pure-JAX ppermute/psum); a genuinely single-process run (no MPI/PMI/SLURM
packages/core/legoesm/parallel/cubesphere_exchange.py:7:* **ppermute multiface** (DEFAULT for every face-sharded device count
packages/core/legoesm/parallel/cubesphere_exchange.py:12:  device-pair schedule of ``jax.lax.ppermute`` rounds, one
packages/core/legoesm/parallel/cubesphere_exchange.py:16:* **ppermute one-face** (halo=1, exactly 6 devices): the original
packages/core/legoesm/parallel/cubesphere_exchange.py:98:# ppermute round tables
packages/core/legoesm/parallel/cubesphere_exchange.py:101:# into 4 perfect matchings so that each round is one ppermute call
packages/core/legoesm/parallel/cubesphere_exchange.py:104:def _build_ppermute_tables():
packages/core/legoesm/parallel/cubesphere_exchange.py:105:    """Compute the static ppermute schedule from CONNECTIVITY.
packages/core/legoesm/parallel/cubesphere_exchange.py:152:    # call ``_PPERMUTE_* = _build_ppermute_tables()`` does NOT
packages/core/legoesm/parallel/cubesphere_exchange.py:157:    # the ``_make_exchange_ppermute._exchange`` closure.
packages/core/legoesm/parallel/cubesphere_exchange.py:167:    _build_ppermute_tables()
packages/core/legoesm/parallel/cubesphere_exchange.py:261:# Reachable only with the module ppermute flag off (force_allgather /
packages/core/legoesm/parallel/cubesphere_exchange.py:265:# 1.00 at 2 devices) — production routing uses the ppermute kernels.
packages/core/legoesm/parallel/cubesphere_exchange.py:291:    # detailed rationale in `_make_exchange_ppermute` below.
packages/core/legoesm/parallel/cubesphere_exchange.py:318:        # any divisor of 6.  The ppermute multi-face refit has since
packages/core/legoesm/parallel/cubesphere_exchange.py:385:# ppermute kernel)
packages/core/legoesm/parallel/cubesphere_exchange.py:414:    # shard_map body. See _make_exchange_ppermute rationale.
packages/core/legoesm/parallel/cubesphere_exchange.py:510:# Backend B: ppermute  (bandwidth-optimal for high resolution)
packages/core/legoesm/parallel/cubesphere_exchange.py:513:def _make_exchange_ppermute(mesh, ndim, with_offsets=False):
packages/core/legoesm/parallel/cubesphere_exchange.py:514:    """Build a shard_map exchange using 4 rounds of ppermute.
packages/core/legoesm/parallel/cubesphere_exchange.py:541:    # `_make_exchange_ppermute` is called from
packages/core/legoesm/parallel/cubesphere_exchange.py:544:    ppermute_send_j = jnp.asarray(_PPERMUTE_SEND)
packages/core/legoesm/parallel/cubesphere_exchange.py:545:    ppermute_recv_j = jnp.asarray(_PPERMUTE_RECV)
packages/core/legoesm/parallel/cubesphere_exchange.py:546:    ppermute_rev_j = jnp.asarray(_PPERMUTE_REV)
packages/core/legoesm/parallel/cubesphere_exchange.py:579:            send_edge = ppermute_send_j[r, my_idx]   # traced int
packages/core/legoesm/parallel/cubesphere_exchange.py:581:            received = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py:584:            recv_edge = ppermute_recv_j[r, my_idx]
packages/core/legoesm/parallel/cubesphere_exchange.py:585:            rev = ppermute_rev_j[r, my_idx]
packages/core/legoesm/parallel/cubesphere_exchange.py:616:# Backend C: multi-face ppermute (k = 6 / n_devices faces per shard)
packages/core/legoesm/parallel/cubesphere_exchange.py:619:# Generalizes the one-face ppermute kernel to n_devices ∈ {1, 2, 3, 6}
packages/core/legoesm/parallel/cubesphere_exchange.py:625:#   * cross-shard face edges → static schedule of ppermute rounds.
packages/core/legoesm/parallel/cubesphere_exchange.py:646:    """Static per-layout tables for the multi-face ppermute exchange.
packages/core/legoesm/parallel/cubesphere_exchange.py:654:        ``perms[r]`` is the (src_dev, dst_dev) pair list for ppermute
packages/core/legoesm/parallel/cubesphere_exchange.py:663:        as a placeholder; they are provably overwritten by a ppermute
packages/core/legoesm/parallel/cubesphere_exchange.py:667:        — applied receiver-side to BOTH local and ppermute strips
packages/core/legoesm/parallel/cubesphere_exchange.py:696:    round repeats a src or a dst (each round is a valid ppermute
packages/core/legoesm/parallel/cubesphere_exchange.py:732:    The tables are halo-depth independent (the schedule moves whole
packages/core/legoesm/parallel/cubesphere_exchange.py:787:                f"multiface ppermute schedule coloring failed for "
packages/core/legoesm/parallel/cubesphere_exchange.py:864:# the ppermute kernel consumes it.  Edge conventions are EXACTLY the

exec
/usr/bin/bash -lc "sed -n '1490,1610p' packages/core/legoesm/parallel/sharded_dynamics.py && sed -n '1600,1820p' packages/core/legoesm/parallel/sharded_dynamics.py && sed -n '1880,1960p' packages/core/legoesm/parallel/sharded_dynamics.py && sed -n '1,115p' scripts/bench/bench_mpas_spmd_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 310ms:
        jnp.array(gather_edges),
        n_owned_cells,
        n_owned_edges,
        max_lc,
        max_le,
        partitions,
        cell_owner,
    )


def _greedy_edge_coloring_ordered(comm_pairs, order):
    """First-fit edge coloring visiting ``order`` (a list of normalized
    ``(min,max)`` pairs). Always a PROPER coloring; the color count depends
    on the visitation order.
    """
    from collections import defaultdict

    vertex_colors: dict[int, set[int]] = defaultdict(set)
    edge_colors: dict[tuple[int, int], int] = {}
    for u, v in order:
        used = vertex_colors[u] | vertex_colors[v]
        color = 0
        while color in used:
            color += 1
        edge_colors[(u, v)] = color
        vertex_colors[u].add(color)
        vertex_colors[v].add(color)
    return edge_colors


def _greedy_edge_coloring(comm_pairs):
    """Legacy first-fit coloring on sorted pairs (the reference/never-regress
    baseline for :func:`_multi_ordering_edge_coloring`). Worst case
    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
    pure wall-clock.
    """
    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
    return _greedy_edge_coloring_ordered(comm_pairs, edges)


def _check_proper_edge_coloring(edge_colors, comm_pairs):
    """Every pair colored, and no vertex sees a color twice."""
    from collections import defaultdict

    if set(edge_colors) != {tuple(sorted(p)) for p in comm_pairs}:
        return False
    seen: dict[int, set[int]] = defaultdict(set)
    for (u, v), c in edge_colors.items():
        if c in seen[u] or c in seen[v]:
            return False
        seen[u].add(c)
        seen[v].add(c)
    return True


# Fixed shuffle seeds for the multi-start greedy edge coloring below —
# a constant so every MPI rank / process builds the byte-identical
# schedule (the coloring must agree across ranks or the ppermute pattern
# desynchronises). NOT Math.random / device randomness: this is host-side
# schedule construction, deterministic by seed.
_COLORING_SHUFFLE_SEEDS = tuple(range(16))


def _multi_ordering_edge_coloring(comm_pairs):
    """Proper edge coloring via multi-start first-fit; returns the coloring
    using the FEWEST colors (= ppermute rounds) across several deterministic
    visitation orders.

    First-fit greedy is order-sensitive: on the reordered MPAS comm graphs
    the sorted order can overshoot the chromatic index by up to 3 rounds at
    16 devices, while a degree-descending or shuffled order reaches the
    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
    {4,8,16} devices, auto/sfc partitions). Every candidate is a proper
    coloring by construction, so taking the min can NEVER produce an
    invalid schedule and can never regress below the legacy sorted greedy.

    Deterministic across ranks (sorted + degree orders + fixed-seed
    shuffles). Returns ``(edge_colors, max_degree)``.
    """
    import random
    from collections import defaultdict

    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
    deg: dict[int, int] = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    max_degree = max(deg.values(), default=0)

    orders = [
        edges,                                                   # sorted
        sorted(edges, key=lambda e: -(deg[e[0]] + deg[e[1]])),   # sum-deg desc
        sorted(edges, key=lambda e: -max(deg[e[0]], deg[e[1]])),  # max-deg desc
    ]
    for seed in _COLORING_SHUFFLE_SEEDS:
        shuffled = edges[:]
        random.Random(seed).shuffle(shuffled)
        orders.append(shuffled)

    best_colors: dict[tuple[int, int], int] | None = None
    best_rounds = None
    for order in orders:
        ec = _greedy_edge_coloring_ordered(comm_pairs, order)
        rounds = max(ec.values(), default=-1) + 1
        if best_rounds is None or rounds < best_rounds:
            best_rounds, best_colors = rounds, ec
            if best_rounds <= max_degree:
                break            # hit the chromatic-index floor — optimal
    return best_colors, max_degree


def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
                             edges_per, max_lc, max_le):
    """Build a ppermute-based halo exchange schedule.

    Instead of all-gathering the full state (O(N) communication),
    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
    between neighboring devices.  The communication graph is edge-colored
    so that each round of ppermute moves data between non-conflicting
    pairs simultaneously.


def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
                             edges_per, max_lc, max_le):
    """Build a ppermute-based halo exchange schedule.

    Instead of all-gathering the full state (O(N) communication),
    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
    between neighboring devices.  The communication graph is edge-colored
    so that each round of ppermute moves data between non-conflicting
    pairs simultaneously.

    Parameters
    ----------
    partitions : list[VoronoiPartition]
    cell_owner : np.ndarray, (nCells,)
    n_dev, cells_per, edges_per : int
    max_lc, max_le : int
        Maximum local cell/edge counts (owned + halo) across devices.

    Returns
    -------
    dict with keys:
        n_rounds, n_rounds_greedy, max_degree, coloring_method,
        ppermute_perms, send_cell_idx, recv_cell_pos,
        send_edge_idx, recv_edge_pos, halo_cells_per_round,
        halo_edges_per_round.
    """
    from collections import defaultdict

    import numpy as np

    # ------------------------------------------------------------------
    # 1. For each device pair, find which cells/edges cross the boundary
    # ------------------------------------------------------------------
    # halo_cells_from[d][d'] = global indices of d's halo cells owned by d'
    halo_cells_from: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    halo_edges_from: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list))

    for d, part in enumerate(partitions):
        for h_idx in range(part.n_owned_cells, part.n_local_cells):
            g = int(part.local_cells[h_idx])
            owner = int(cell_owner[g])
            halo_cells_from[d][owner].append(g)

        for h_idx in range(part.n_owned_edges, part.n_local_edges):
            g = int(part.local_edges[h_idx])
            owner = min(g // edges_per, n_dev - 1)
            halo_edges_from[d][owner].append(g)

    # ------------------------------------------------------------------
    # 2. Build undirected communication graph
    # ------------------------------------------------------------------
    comm_pairs: set[tuple[int, int]] = set()
    for d in range(n_dev):
        for d_prime in halo_cells_from[d]:
            if d != d_prime:
                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
        for d_prime in halo_edges_from[d]:
            if d != d_prime:
                comm_pairs.add((min(d, d_prime), max(d, d_prime)))

    if not comm_pairs:
        return {
            'n_rounds': 0,
            'n_rounds_greedy': 0,
            'max_degree': 0,
            'coloring_method': 'none',
            'ppermute_perms': [],
            'send_cell_idx': [],
            'recv_cell_pos': [],
            'send_edge_idx': [],
            'recv_edge_pos': [],
            'halo_cells_per_round': [],
            'halo_edges_per_round': [],
        }

    # ------------------------------------------------------------------
    # 3. Edge-color the graph: each color = one bidirectional ppermute
    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
    #    fewer colors = directly less wall-clock. First-fit greedy is
    #    order-sensitive; the multi-start coloring reaches the
    #    chromatic-index floor (= max_degree) on every probed MPAS config
    #    where the legacy sorted greedy overshoots (up to 3 rounds at 16
    #    devices). It can never regress: the legacy sorted order is one of
    #    its candidates and it takes the min. Both are verified proper.
    # ------------------------------------------------------------------
    greedy_colors = _greedy_edge_coloring(comm_pairs)
    n_rounds_greedy = max(greedy_colors.values()) + 1
    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
    n_rounds_multi = max(multi_colors.values()) + 1
    # Adopt the multi-start coloring ONLY when it STRICTLY reduces rounds;
    # on a tie keep the exact legacy sorted-greedy coloring so the produced
    # schedule is byte-identical to before wherever there is no round win
    # (the win only appears at high device counts — >=16 on the probed
    # MPAS meshes). Both colorings are proper.
    if n_rounds_multi < n_rounds_greedy:
        edge_colors, n_rounds, coloring_method = (
            multi_colors, n_rounds_multi, "multi_greedy")
    else:
        edge_colors, n_rounds, coloring_method = (
            greedy_colors, n_rounds_greedy, "greedy")
    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
        "improper ppermute edge coloring — two same-round exchanges "
        "would collide at a device")
    rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (u, v), color in edge_colors.items():
        rounds[color].append((u, v))

    # ------------------------------------------------------------------
    # 4. Build directed send/recv maps for each device pair
    # ------------------------------------------------------------------
    # cell_send_map[(src, dst)] = list of owned-local indices in src to send
    # cell_recv_map[(dst, src)] = list of local positions in dst to place data
    cell_send_map: dict[tuple[int, int], list[int]] = {}
    cell_recv_map: dict[tuple[int, int], list[int]] = {}
    edge_send_map: dict[tuple[int, int], list[int]] = {}
    edge_recv_map: dict[tuple[int, int], list[int]] = {}

    for d in range(n_dev):
        for d_prime, cells_g in halo_cells_from[d].items():
            if d_prime == d:
                continue
            # d_prime sends its owned cells that d needs as halo
            cell_send_map[(d_prime, d)] = [
                g - d_prime * cells_per for g in cells_g]
            cell_recv_map[(d, d_prime)] = [
                int(partitions[d].cell_g2l[g]) for g in cells_g]

        for d_prime, edges_g in halo_edges_from[d].items():
            if d_prime == d:
                continue
            edge_send_map[(d_prime, d)] = [
                g - d_prime * edges_per for g in edges_g]
            edge_recv_map[(d, d_prime)] = [
                int(partitions[d].edge_g2l[g]) for g in edges_g]

    # ------------------------------------------------------------------
    # 5. Assemble per-round ppermute patterns and index arrays
    # ------------------------------------------------------------------
    ppermute_perms_out: list[list[tuple[int, int]]] = []
    send_cell_idx_out: list[jnp.ndarray] = []
    recv_cell_pos_out: list[jnp.ndarray] = []
    send_edge_idx_out: list[jnp.ndarray] = []
    recv_edge_pos_out: list[jnp.ndarray] = []
    halo_cells_per_round: list[int] = []
    halo_edges_per_round: list[int] = []

    for r in range(n_rounds):
        # Max halo size across all pairs in this round
        max_c = 0
        max_e = 0
        for u, v in rounds[r]:
            for src, dst in [(u, v), (v, u)]:
                max_c = max(max_c, len(cell_send_map.get((src, dst), [])))
                max_e = max(max_e, len(edge_send_map.get((src, dst), [])))
        max_c = max(max_c, 1)  # at least 1 for array shape
        max_e = max(max_e, 1)
        halo_cells_per_round.append(max_c)
        halo_edges_per_round.append(max_e)

        # Bidirectional ppermute pattern
        perm: list[tuple[int, int]] = []
        partner: dict[int, int] = {}
        for u, v in rounds[r]:
            perm.append((u, v))
            perm.append((v, u))
            partner[u] = v
            partner[v] = u
        ppermute_perms_out.append(perm)

        # Per-device index arrays (padded with safe defaults)
        sc = np.zeros((n_dev, max_c), dtype=np.int64)
        # Garbage slot: writes go to max_lc (trimmed off later)
        rc = np.full((n_dev, max_c), max_lc, dtype=np.int64)
        se = np.zeros((n_dev, max_e), dtype=np.int64)
        re = np.full((n_dev, max_e), max_le, dtype=np.int64)

        for d in range(n_dev):
            if d not in partner:
                continue
            dp = partner[d]

            cs = cell_send_map.get((d, dp), [])
            for j, idx in enumerate(cs):
                sc[d, j] = idx

            cr = cell_recv_map.get((d, dp), [])
            for j, pos in enumerate(cr):
                rc[d, j] = pos

            es = edge_send_map.get((d, dp), [])
            for j, idx in enumerate(es):
                se[d, j] = idx

            er = edge_recv_map.get((d, dp), [])
            for j, pos in enumerate(er):
                re[d, j] = pos

        send_cell_idx_out.append(jnp.array(sc))
        recv_cell_pos_out.append(jnp.array(rc))
        send_edge_idx_out.append(jnp.array(se))
        recv_edge_pos_out.append(jnp.array(re))

    return {
        'n_rounds': n_rounds,
        'n_rounds_greedy': n_rounds_greedy,
        'max_degree': max_degree,
        'coloring_method': coloring_method,
        'ppermute_perms': ppermute_perms_out,
        'send_cell_idx': send_cell_idx_out,
        'recv_cell_pos': recv_cell_pos_out,
        'send_edge_idx': send_edge_idx_out,
        'recv_edge_pos': recv_edge_pos_out,
        'halo_cells_per_round': halo_cells_per_round,
        'halo_edges_per_round': halo_edges_per_round,
    }


            "make_voronoi_sharded_step: state.v must be None on MPAS "
            "(the wind is edge-normal u); a non-None v would be silently "
            "dropped by the packed halo exchange."
        )
    return tuple(sorted(state.tracers)) if state.tracers is not None else ()


def _pack_cell_state(T, p_s, phis, q_flat):
    """Production cell-pack WIRE layout: ``T | p_s | phis | tracers``.

    ``q_flat`` is the tracer block ``(n, nlev * n_q)`` concatenated in
    the canonical SORTED-key order (width 0 for a dry run).  The single
    source of truth for the packed exchange layout — the shard_map
    kernel and the sentinel routing test both go through here, so an
    omitted field or a swapped slot cannot hide in a hand-rolled copy.
    """
    return jnp.concatenate(
        [T, p_s[:, jnp.newaxis], phis[:, jnp.newaxis], q_flat], axis=-1)


def _unpack_cell_state(cell_buf, nlev):
    """Inverse of :func:`_pack_cell_state`: ``(T, p_s, phis, q_flat)``."""
    return (cell_buf[:, :nlev], cell_buf[:, nlev],
            cell_buf[:, nlev + 1], cell_buf[:, nlev + 2:])


def _ppermute_halo_fill(cell_pack, u_shard, halo_sl, ppermute_perms,
                        max_lc, max_le):
    """Fill (owned + halo) local buffers from owned shards via ppermute.

    Runs INSIDE ``shard_map``.  ``cell_pack`` ``(cells_per, W)`` is the
    packed owned-cell buffer — ALL cell-centred prognostics (T | p_s |
    phis | tracers) concatenated on the trailing axis; ``u_shard``
    ``(edges_per, nlev)`` the owned-edge buffer.  Each edge-colored round
    posts ONE flat ppermute carrying BOTH entity classes for ALL packed
    fields — the SPMD mirror of route-A's batched union-neighbor exchange
    (one message per neighbor per dtype group; the compute-precision cast
    upstream guarantees a single dtype group here).

    ``halo_sl`` is a tuple of per-round ``(send_cell_idx, recv_cell_pos,
    send_edge_idx, recv_edge_pos)`` tuples whose arrays are ALREADY
    device-local ``(1, n_round)`` shard_map arguments (``P("device")``
    specs) — per-rank LOCAL metadata; no device materializes the global
    schedule.  ``ppermute_perms`` is the static per-round permutation.

    Returns ``(cell_local, u_local)`` of shapes ``(max_lc, W)`` /
    ``(max_le, nlev)``; ghost tail rows stay zero.
    """
    cells_per = cell_pack.shape[0]
    edges_per = u_shard.shape[0]
    # +1 garbage slot for padded scatter targets (trimmed at the end):
    # schedule rows are padded to the round's max halo count, and padding
    # entries target position max_lc / max_le.
    cell_local = jnp.pad(cell_pack, ((0, max_lc + 1 - cells_per), (0, 0)))
    u_local = jnp.pad(u_shard, ((0, max_le + 1 - edges_per), (0, 0)))

    for r, (sc, rc, se, re) in enumerate(halo_sl):
        send_c = cell_pack[sc[0]]             # (hc_r, W)
        send_e = u_shard[se[0]]               # (he_r, nlev)
        send_c_flat = send_c.ravel()
        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
        recv_packed = jax.lax.ppermute(
            send_packed, "device", perm=ppermute_perms[r])
        split_at = send_c_flat.shape[0]       # static
        recv_c = recv_packed[:split_at].reshape(send_c.shape)
        recv_e = recv_packed[split_at:].reshape(send_e.shape)
        cell_local = cell_local.at[rc[0]].set(recv_c)
        u_local = u_local.at[re[0]].set(recv_e)

    return cell_local[:max_lc], u_local[:max_le]


def make_voronoi_sharded_step(
    model,
    dev_config: DeviceConfig,
    *,
    halo_strategy: str = "auto",
    ppermute_cells_per_device_threshold: int = 2_000,
    return_phys_state: bool = False,
):
    """Create a halo-partitioned multi-GPU step for Voronoi (MPAS/TRiSK) grids.
"""Strong scaling bench for the device-sharded icosahedral/MPAS (TRiSK)
hydrostatic atm step (``make_voronoi_sharded_step`` — cell-partition reorder +
ppermute halo).

The Voronoi twin of ``bench_atm_latlon_spmd_scaling.py`` (mirrored
flag-for-flag where the grids allow): the global mesh is REORDERED with
``reorder_voronoi_for_sharding`` (METIS/RCB/Hilbert-SFC cell partition, ghost-
padded to an even device split) so each device's contiguous ``P("device")``
shard is a spatially compact cell cluster, then the SSP-RK3 step exchanges
only the partition-boundary halo per stage via ``jax.lax.ppermute``.

  strong: fixed subdivision level, vary n_devices -> speedup = t(1)/t(n).
  (Weak scaling rides the subdivision ladder: one level = 4x the cells, so
  L at 4*n_dev matches L-1 at n_dev per-device load — there is no per-device
  row knob like the lat-lon benches' --nlat-per-dev.)

nlev caveat (#1113): the multi-node ceiling at THIN nlev is the ppermute ROUND
count (``hlo_collective_permutes``, recorded per row) x the ~0.11 ms launch
floor, NOT bandwidth — a fixed per-step overhead. At the ``--nlev 8`` default it
dominates (~1.34 Gcells/s wall from N=4), so the default UNDERSTATES production
scalability: at ``--nlev 26`` the per-cell compute grows ~3.25x, the flat wall
dissolves (~2.07+ Gcells/s, ~1.9x higher at 16 GPUs), and a size-dependent term
enters. Report the production curve at production thickness; nlev=8 is the
overhead-mechanism receipt, not the campaign number.

Device count is fixed at process start, so each n_devices runs as a SEPARATE
process; this script benches ONE n_devices and appends a JSON line.
JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count gives virtual
CPU devices (communication-overhead characterization, NOT a real speedup).

Multi-controller (route-B, ``--multicontroller``): identical contract to the
lat-lon benches — every process calls ``jax.distributed.initialize`` BEFORE
any other JAX use, the ("device",) mesh is built over the GLOBAL
``jax.devices()``, and the existing ``make_voronoi_sharded_step`` ppermute
halo + the mass-fix psum run unchanged across processes (NCCL on GPU / gloo
on CPU). NO mpi4jax is armed in this mode (the documented mixed-stack
deadlock hazard). Every process computes the SAME reorder host-side; under
``--multicontroller`` the partition checksum is asserted equal across
processes (a rank-divergent partition — e.g. one rank resolving
``--partition-method auto`` to METIS and another to RCB — would silently
corrupt the halo schedule).

Launch (cluster, one process per GPU):
  srun -n 6 python bench_mpas_spmd_scaling.py --multicontroller \
      --n-devices 6 ...            # SLURM: coordinator auto-detected
  mpiexec -n 6 python ... --multicontroller --coordinator host0:9876
CPU smoke (single process, virtual devices):
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  JAX_ENABLE_X64=1 python scripts/bench/bench_mpas_spmd_scaling.py \
      --subdivision 3 --nlev 4 --n-devices 2 --steps 4 --parity-gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import jax
import numpy as np

# Repo root on the path for tests.test_cases.baroclinic_wave (the same
# baroclinic-wave IC the icosahedral lanes of run_levante_gpu_scaling use).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Bench dir for the shared metadata module (sibling-script import pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
# imports JAX lazily, so this is safe before jax.distributed.initialize.
from metadata import (  # noqa: E402
    annotate_incomplete, hlo_collective_census, scaling_metadata,
    tidy_throughput_fields)

# SPMD full-step parity tolerances — the FLOATING-POINT RE-ASSOCIATION floor
# of the sharded step (ppermute halo + mass-fix psum reduction-order change),
# NOT a bug margin; a real halo/partition regression shows up orders of
# magnitude above these.  Values extend the 1-step envelope of
# tests/parallel/test_voronoi_sharded_equivalence.py (u/T atol 1e-6, p_s
# atol 1e-1) to the smoke window; the floor grows with steps, hence the cap.
MPAS_PARITY_TOLS = {  # precision -> field -> (rtol, atol)
    "float64": {"u": (1.0e-5, 1.0e-5), "T": (1.0e-6, 1.0e-5),
                "p_s": (1.0e-5, 1.0)},
    "float32": {"u": (1.0e-3, 1.0e-3), "T": (1.0e-4, 1.0e-3),
                "p_s": (1.0e-3, 50.0)},
}
MPAS_PARITY_MAX_STEPS = 8

# Conservation gate default: with fix_mass=True the step restores the global
# dry mass to the pre-step value each step, so the drift over a smoke window
# is the allreduce rounding floor, not scheme drift.
MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}


def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
                          moist=False, lloyd_iterations=50):
    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.

    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
    device count of THIS run's mesh/model (the two differ for the
    single-device reference leg of a ladder, via ``--reorder-for``).
    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
    wave) so the sharded step's packed tracer halo exchange + RK tracer
    advection sit on the timed/gated path.
    """
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate

exec
/usr/bin/bash -lc 'rg -n "ppermute schedule|rounds|round|color|collective" mpas_s9.26600095.log mpas_s10_128.26677812.log mpas_nsys.26680051.log | head -n 240' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 415ms:
rg: mpas_nsys.26680051.log: No such file or directory (os error 2)
mpas_s10_128.26677812.log:514:{"component": "mpas_atm", "subdivision": 10, "n_devices": 128, "n_cells": 10485888, "n_edges": 31457280, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7877.2, "steady_median_ms": 18.2, "steady_min_ms": 17.9, "per_step_ms": [7877.2, 21.5, 18.7, 18.2, 18.9, 17.9, 18.3, 17.9, 18.2, 17.9, 18.3, 18.2], "cells": 272633088, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 10, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 18.203451007138938, "total_cells": 272633088, "sypd": 4.5120852953515, "mcells_per_s": 14977.000124486292, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-04T07:50:57.964734+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L10", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 2129946, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50009.lvt.dkrz.de", "slurm_job_id": "26677812", "git_sha": "c478a951d", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2129946}}}
mpas_s9.26600095.log:67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
mpas_s9.26600095.log:294:{"component": "mpas_atm", "subdivision": 9, "n_devices": 64, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 6474.6, "steady_median_ms": 9.6, "steady_min_ms": 9.55, "per_step_ms": [6474.6, 13.0, 10.1, 10.0, 9.6, 9.8, 9.6, 9.6, 9.6, 9.6, 9.5, 9.6], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 9.601722995284945, "total_cells": 68160768, "sypd": 8.55424840461413, "mcells_per_s": 7098.805915716509, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:31:30.728837+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 1065012, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 1065012}}}
mpas_s9.26600095.log:929:{"component": "mpas_atm", "subdivision": 9, "n_devices": 128, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7532.3, "steady_median_ms": 11.48, "steady_min_ms": 11.37, "per_step_ms": [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 11.482901012641378, "total_cells": 68160768, "sypd": 7.152854798934616, "mcells_per_s": 5935.8491312398055, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:45:45.637722+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}

exec
/usr/bin/bash -lc 'rg -n "overlap|pipelined|latency.hiding|fused-halo|fused halo" docs/performance/scaling/levante_campaign_2026-07-24.md | head -n 120 && rg -n "overlap|pipelined|latency|fused" ocean_ovl.26458930.log ocean_ab_wf.26449827.log ocean_ab_xla.26451282.log oc128.26646039.log | head -n 260' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 545ms:
121:  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
156:why the earlier env-knob sweep missed it — autotune/latency-hiding flags
386:un-overlapped communication rather than raw wire time.
390:behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:
395:| `latency_hiding_scheduler=false` | 42.44 | 23.49 | **+0.6 % / +0.2 %** |
396:| `enable_pipelined_p2p=true` | 42.76 | 23.54 | -0.2 % / -0.1 % |
399:Turning overlap OFF is free (marginally faster), and no scheduling flag
412:- fused-halo NULL — aggregation reduces message COUNT but not chain DEPTH;
921:cube), (c) overlapping halo exchange with interior compute.
1003:  communication avoidance would need a 3 x radius = 6-cell overlap with
1051:drift cannot accumulate, or overlap the resync exchanges with interior
1464:9. Diagnosis tool halo/overlap phases timed UN-JITTED eager pads
1467:   world_size==1; overlap fractions >100% now refused), contract test
1476:NCCL_PROTO forcing (default already optimal; LL128 −8–12%), fused-halo on
1479:`--xla_gpu_enable_pipelined_p2p` alone, `LEGOESM_BAROCLINIC_F32` (~+0.5%, within run-to-run spread; job 26451282).
1584:the CPU spread ladder; and the diagnosis tool's halo + overlap phases,
1683:   wall-clock brackets logged as overlap evidence. CONFIRM bar:
1841:  vs overlap biases pull opposite ways; treat measured/bound as a
1844:  LL2048@128 (A default / B combine-8MB / C combine+pipelined-p2p /
2000:A 6.740 / B combine-8MB 6.682 (-0.86 %) / C combine+pipelined 6.689
2158:    29.7us AR-assumption = 3.395 ms -> **ratio 1.64** (post-overlap
2170:  latency-hiding scheduler + pipelined collectives) and exchange-COUNT
2175:XLA latency-hiding scheduler + pipelined p2p (flag names verified
2179:| lane | A | B (overlap) | A2 | effect |
2188:  default: cube_tiled_step force-disables latency hiding for a known
2260:| C fused+overlap | **5.278 (-20.3 %)** | **4.745 (-15.0 %)** |
2265:  larger CPs giving the latency-hiding scheduler more to hide
2279:  everything else stays opt-in (cube force-disables latency hiding;
2285:scheduling-bound — explains the overlap null) and NCCL SendRecv medians
2289:combining, overlap and graphs all receipted null, single-trajectory
oc128.26646039.log:515:{"component": "ocean", "mode": "strong", "n_devices": 128, "n_lat": 2304, "n_lon": 4608, "nlev": 20, "steps": 18, "platform": "gpu", "n_processes": 128, "multicontroller": true, "steady_median_ms": 16.3326, "cells": 212336640, "compile_ms": 16628.9, "scan_compile_ms": 16944.1, "step_latency_ms": 20.765, "block_ms": [293.15, 293.95], "parallel_block_ms": [293.61, 294.36], "fused_step_ms": 16.3326, "rank_imbalance": 1.0006, "rank_imbalance_per_block": [1.0007, 1.0006], "block_steps": 18, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 2304, "n_levels": 20, "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 150.0, "time_per_step_ms": 16.3326, "total_cells": 212336640, "sypd": 25.14465658069231, "mcells_per_s": 13000.786157745859, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 188743680, "wet_cell_levels_per_device": 1474560.0, "wet_fraction": 0.8888888888888888, "wet_equals_total": false, "wet_cell_levels_per_device_min": 0, "wet_cell_levels_per_device_max": 1658880, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0006}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-03T06:27:25.701171+00:00", "grid": "latlon", "component": "ocean", "resolution": "2304x4608", "n_levels": 20, "precision": "float32", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 1658880, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26646039", "git_sha": "b8ec472cc-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 18, "warmup": 3, "multicontroller": true, "fused_halo": false, "nccl": {"nccl_net_plugin": null, "nccl_net": null, "nccl_ib_disable": "0", "nccl_ib_hca": "mlx5", "nccl_socket_ifname": "ib0", "nccl_debug": null, "n_nodes_declared": 32, "missing_net_plugin_multi_node": true}, "parity_gate": false, "check_conservation": false, "cells_per_device": 1658880, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
oc128.26646039.log:516:[ocean nd=128 strong 2304x4608x20] compile=16628.9ms fused=16.333ms/step latency=20.765ms/step imbalance=1.0006 blocks=[293.15, 293.95]
oc128.26646039.log:518:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ovl.26458930.log:3:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 576, "n_lon": 1152, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 42.693, "cells": 13271040, "compile_ms": 22136.0, "scan_compile_ms": 22364.0, "step_latency_ms": 46.583, "block_ms": [1404.36, 1413.38], "parallel_block_ms": [1404.36, 1413.38], "fused_step_ms": 42.693, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 576, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 42.693, "total_cells": 13271040, "sypd": 38.47727899841334, "mcells_per_s": 310.8481484084042, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 0.00020792623894866767, "zero_forcing_probe_measured": true, "wet_cell_levels": 11796480, "wet_cell_levels_per_device": 5898240.0, "wet_fraction": 0.8888888888888888, "wet_equals_total": false, "wet_cell_levels_per_device_min": 5898240, "wet_cell_levels_per_device_max": 5898240, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 18432, "halo_bytes_per_step": 2248704, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": true, "bound_incomplete_reason": ["single_device_fused_step_ms"], "bound_ingredients": {"compute_ms": null, "comm_ms": 2.209056, "reduction_ms": 2.19186, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 17.82, "bandwidth_GBs": 64.22, "halo_messages_per_step": 122, "halo_bytes_per_step": 2248704, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-25T02:06:34.953488+00:00", "grid": "latlon", "component": "ocean", "resolution": "576x1152", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 13271040, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50060.lvt.dkrz.de", "slurm_job_id": "26458930", "git_sha": "7bfbf998a", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 6635520, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ovl.26458930.log:4:[ocean nd=2 strong 576x1152x20] compile=22136.0ms fused=42.693ms/step latency=46.583ms/step imbalance=1.0 blocks=[1404.36, 1413.38]
ocean_ovl.26458930.log:6:[bound] t_bound_ms=None measured_over_bound=None calibrated=True incomplete=['single_device_fused_step_ms']
ocean_ovl.26458930.log:8:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 576, "n_lon": 1152, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 23.5256, "cells": 13271040, "compile_ms": 11736.5, "scan_compile_ms": 12379.3, "step_latency_ms": 27.25, "block_ms": [779.18, 773.51], "parallel_block_ms": [779.18, 773.51], "fused_step_ms": 23.5256, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 576, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 23.5256, "total_cells": 13271040, "sypd": 69.82650696599708, "mcells_per_s": 564.1105859149181, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 0.00020792623895153295, "zero_forcing_probe_measured": true, "wet_cell_levels": 11796480, "wet_cell_levels_per_device": 2949120.0, "wet_fraction": 0.8888888888888888, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2580480, "wet_cell_levels_per_device_max": 3317760, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 18432, "halo_bytes_per_step": 2248704, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": true, "bound_incomplete_reason": ["single_device_fused_step_ms"], "bound_ingredients": {"compute_ms": null, "comm_ms": 2.209056, "reduction_ms": 2.19186, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 17.82, "bandwidth_GBs": 64.22, "halo_messages_per_step": 122, "halo_bytes_per_step": 2248704, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-25T02:07:19.108366+00:00", "grid": "latlon", "component": "ocean", "resolution": "576x1152", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 13271040, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50060.lvt.dkrz.de", "slurm_job_id": "26458930", "git_sha": "7bfbf998a", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 3317760, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ovl.26458930.log:9:[ocean nd=4 strong 576x1152x20] compile=11736.5ms fused=23.526ms/step latency=27.25ms/step imbalance=1.0 blocks=[779.18, 773.51]
ocean_ovl.26458930.log:11:[bound] t_bound_ms=None measured_over_bound=None calibrated=True incomplete=['single_device_fused_step_ms']
ocean_ovl.26458930.log:13:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 576, "n_lon": 1152, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 42.4368, "cells": 13271040, "compile_ms": 21191.3, "scan_compile_ms": 22496.6, "step_latency_ms": 46.27, "block_ms": [1396.47, 1404.36], "parallel_block_ms": [1396.47, 1404.36], "fused_step_ms": 42.4368, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 576, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 42.4368, "total_cells": 13271040, "sypd": 38.70957452680836, "mcells_per_s": 312.7248048863251, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 0.00020792623894866767, "zero_forcing_probe_measured": true, "wet_cell_levels": 11796480, "wet_cell_levels_per_device": 5898240.0, "wet_fraction": 0.8888888888888888, "wet_equals_total": false, "wet_cell_levels_per_device_min": 5898240, "wet_cell_levels_per_device_max": 5898240, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 18432, "halo_bytes_per_step": 2248704, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": true, "bound_incomplete_reason": ["single_device_fused_step_ms"], "bound_ingredients": {"compute_ms": null, "comm_ms": 2.209056, "reduction_ms": 2.19186, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 17.82, "bandwidth_GBs": 64.22, "halo_messages_per_step": 122, "halo_bytes_per_step": 2248704, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-25T02:08:22.517259+00:00", "grid": "latlon", "component": "ocean", "resolution": "576x1152", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 13271040, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50060.lvt.dkrz.de", "slurm_job_id": "26458930", "git_sha": "7bfbf998a", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 6635520, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ovl.26458930.log:14:[ocean nd=2 strong 576x1152x20] compile=21191.3ms fused=42.437ms/step latency=46.27ms/step imbalance=1.0 blocks=[1396.47, 1404.36]
ocean_ovl.26458930.log:16:[bound] t_bound_ms=None measured_over_bound=None calibrated=True incomplete=['single_device_fused_step_ms']
ocean_ovl.26458930.log:18:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 576, "n_lon": 1152, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 23.4901, "cells": 13271040, "compile_ms": 11661.2, "scan_compile_ms": 12375.5, "step_latency_ms": 26.949, "block_ms": [779.56, 770.78], "parallel_block_ms": [779.56, 770.78], "fused_step_ms": 23.4901, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 576, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 23.4901, "total_cells": 13271040, "sypd": 69.93203401770366, "mcells_per_s": 564.9631121195737, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 0.00020792623895043844, "zero_forcing_probe_measured": true, "wet_cell_levels": 11796480, "wet_cell_levels_per_device": 2949120.0, "wet_fraction": 0.8888888888888888, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2580480, "wet_cell_levels_per_device_max": 3317760, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 18432, "halo_bytes_per_step": 2248704, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": true, "bound_incomplete_reason": ["single_device_fused_step_ms"], "bound_ingredients": {"compute_ms": null, "comm_ms": 2.209056, "reduction_ms": 2.19186, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 17.82, "bandwidth_GBs": 64.22, "halo_messages_per_step": 122, "halo_bytes_per_step": 2248704, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-25T02:09:05.963432+00:00", "grid": "latlon", "component": "ocean", "resolution": "576x1152", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 13271040, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50060.lvt.dkrz.de", "slurm_job_id": "26458930", "git_sha": "7bfbf998a", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 3317760, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ovl.26458930.log:19:[ocean nd=4 strong 576x1152x20] compile=11661.2ms fused=23.490ms/step latency=26.949ms/step imbalance=1.0 blocks=[779.56, 770.78]
ocean_ovl.26458930.log:21:[bound] t_bound_ms=None measured_over_bound=None calibrated=True incomplete=['single_device_fused_step_ms']
ocean_ovl.26458930.log:23:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 576, "n_lon": 1152, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 42.7631, "cells": 13271040, "compile_ms": 21180.1, "scan_compile_ms": 22551.5, "step_latency_ms": 46.621, "block_ms": [1407.71, 1414.65], "parallel_block_ms": [1407.71, 1414.65], "fused_step_ms": 42.7631, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 576, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 42.7631, "total_cells": 13271040, "sypd": 38.41420458945353, "mcells_per_s": 310.33858630454756, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 0.00020792623894866767, "zero_forcing_probe_measured": true, "wet_cell_levels": 11796480, "wet_cell_levels_per_device": 5898240.0, "wet_fraction": 0.8888888888888888, "wet_equals_total": false, "wet_cell_levels_per_device_min": 5898240, "wet_cell_levels_per_device_max": 5898240, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 18432, "halo_bytes_per_step": 2248704, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": true, "bound_incomplete_reason": ["single_device_fused_step_ms"], "bound_ingredients": {"compute_ms": null, "comm_ms": 2.209056, "reduction_ms": 2.19186, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 17.82, "bandwidth_GBs": 64.22, "halo_messages_per_step": 122, "halo_bytes_per_step": 2248704, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-25T02:10:09.536893+00:00", "grid": "latlon", "component": "ocean", "resolution": "576x1152", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 13271040, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50060.lvt.dkrz.de", "slurm_job_id": "26458930", "git_sha": "7bfbf998a", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 6635520, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ovl.26458930.log:24:[ocean nd=2 strong 576x1152x20] compile=21180.1ms fused=42.763ms/step latency=46.621ms/step imbalance=1.0 blocks=[1407.71, 1414.65]
ocean_ovl.26458930.log:26:[bound] t_bound_ms=None measured_over_bound=None calibrated=True incomplete=['single_device_fused_step_ms']
ocean_ovl.26458930.log:28:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 576, "n_lon": 1152, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 23.5444, "cells": 13271040, "compile_ms": 11729.6, "scan_compile_ms": 12417.2, "step_latency_ms": 27.078, "block_ms": [776.6, 777.34], "parallel_block_ms": [776.6, 777.34], "fused_step_ms": 23.5444, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 576, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 23.5444, "total_cells": 13271040, "sypd": 69.77075110341572, "mcells_per_s": 563.6601484854148, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 0.00020792623895043844, "zero_forcing_probe_measured": true, "wet_cell_levels": 11796480, "wet_cell_levels_per_device": 2949120.0, "wet_fraction": 0.8888888888888888, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2580480, "wet_cell_levels_per_device_max": 3317760, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 18432, "halo_bytes_per_step": 2248704, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": true, "bound_incomplete_reason": ["single_device_fused_step_ms"], "bound_ingredients": {"compute_ms": null, "comm_ms": 2.209056, "reduction_ms": 2.19186, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 17.82, "bandwidth_GBs": 64.22, "halo_messages_per_step": 122, "halo_bytes_per_step": 2248704, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-25T02:10:53.491429+00:00", "grid": "latlon", "component": "ocean", "resolution": "576x1152", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 13271040, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50060.lvt.dkrz.de", "slurm_job_id": "26458930", "git_sha": "7bfbf998a", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 3317760, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ovl.26458930.log:29:[ocean nd=4 strong 576x1152x20] compile=11729.6ms fused=23.544ms/step latency=27.078ms/step imbalance=1.0 blocks=[776.6, 777.34]
ocean_ovl.26458930.log:31:[bound] t_bound_ms=None measured_over_bound=None calibrated=True incomplete=['single_device_fused_step_ms']
ocean_ovl.26458930.log:33:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 576, "n_lon": 1152, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 42.5536, "cells": 13271040, "compile_ms": 21165.1, "scan_compile_ms": 22471.4, "step_latency_ms": 46.365, "block_ms": [1400.22, 1408.32], "parallel_block_ms": [1400.22, 1408.32], "fused_step_ms": 42.5536, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 576, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 42.5536, "total_cells": 13271040, "sypd": 38.60332550663776, "mcells_per_s": 311.86644608211753, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 0.00020792623895138585, "zero_forcing_probe_measured": true, "wet_cell_levels": 11796480, "wet_cell_levels_per_device": 5898240.0, "wet_fraction": 0.8888888888888888, "wet_equals_total": false, "wet_cell_levels_per_device_min": 5898240, "wet_cell_levels_per_device_max": 5898240, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 18432, "halo_bytes_per_step": 2248704, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": true, "bound_incomplete_reason": ["single_device_fused_step_ms"], "bound_ingredients": {"compute_ms": null, "comm_ms": 2.209056, "reduction_ms": 2.19186, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 17.82, "bandwidth_GBs": 64.22, "halo_messages_per_step": 122, "halo_bytes_per_step": 2248704, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-25T02:11:56.740441+00:00", "grid": "latlon", "component": "ocean", "resolution": "576x1152", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 13271040, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50060.lvt.dkrz.de", "slurm_job_id": "26458930", "git_sha": "7bfbf998a", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 6635520, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ovl.26458930.log:34:[ocean nd=2 strong 576x1152x20] compile=21165.1ms fused=42.554ms/step latency=46.365ms/step imbalance=1.0 blocks=[1400.22, 1408.32]
ocean_ovl.26458930.log:36:[bound] t_bound_ms=None measured_over_bound=None calibrated=True incomplete=['single_device_fused_step_ms']
ocean_ovl.26458930.log:38:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 576, "n_lon": 1152, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 23.587, "cells": 13271040, "compile_ms": 11723.8, "scan_compile_ms": 12513.3, "step_latency_ms": 27.156, "block_ms": [779.04, 777.7], "parallel_block_ms": [779.04, 777.7], "fused_step_ms": 23.587, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 576, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 23.587, "total_cells": 13271040, "sypd": 69.64473957176669, "mcells_per_s": 562.642133378556, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 0.00020792623895043844, "zero_forcing_probe_measured": true, "wet_cell_levels": 11796480, "wet_cell_levels_per_device": 2949120.0, "wet_fraction": 0.8888888888888888, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2580480, "wet_cell_levels_per_device_max": 3317760, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 18432, "halo_bytes_per_step": 2248704, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": true, "bound_incomplete_reason": ["single_device_fused_step_ms"], "bound_ingredients": {"compute_ms": null, "comm_ms": 2.209056, "reduction_ms": 2.19186, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 17.82, "bandwidth_GBs": 64.22, "halo_messages_per_step": 122, "halo_bytes_per_step": 2248704, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-25T02:12:40.515085+00:00", "grid": "latlon", "component": "ocean", "resolution": "576x1152", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 13271040, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50060.lvt.dkrz.de", "slurm_job_id": "26458930", "git_sha": "7bfbf998a", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 3317760, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ovl.26458930.log:39:[ocean nd=4 strong 576x1152x20] compile=11723.8ms fused=23.587ms/step latency=27.156ms/step imbalance=1.0 blocks=[779.04, 777.7]
ocean_ovl.26458930.log:41:[bound] t_bound_ms=None measured_over_bound=None calibrated=True incomplete=['single_device_fused_step_ms']
ocean_ovl.26458930.log:66:*                    worktrees/scaling-campaign/scripts/tmp/ocean_overlap_test.sbatch
ocean_ab_xla.26451282.log:4:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 21.0854, "cells": 5898240, "compile_ms": 12467.7, "scan_compile_ms": 12005.7, "step_latency_ms": 23.98, "block_ms": [693.38, 698.26], "parallel_block_ms": [693.38, 698.26], "fused_step_ms": 21.0854, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 21.0854, "total_cells": 5898240, "sypd": 77.90748443374376, "mcells_per_s": 279.7309987005226, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.7874249856471415e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:29:40.614372+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:5:[ocean nd=2 strong 384x768x20] compile=12467.7ms fused=21.085ms/step latency=23.98ms/step imbalance=1.0 blocks=[693.38, 698.26]
ocean_ab_xla.26451282.log:7:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:9:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 19.0496, "cells": 5898240, "compile_ms": 11994.7, "scan_compile_ms": 12434.5, "step_latency_ms": 21.337, "block_ms": [628.84, 628.43], "parallel_block_ms": [628.84, 628.43], "fused_step_ms": 19.0496, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 19.0496, "total_cells": 5898240, "sypd": 86.23333152818225, "mcells_per_s": 309.62539895850824, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:30:17.721562+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:10:[ocean nd=2 strong 384x768x20] compile=11994.7ms fused=19.050ms/step latency=21.337ms/step imbalance=1.0 blocks=[628.84, 628.43]
ocean_ab_xla.26451282.log:12:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:14:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 14.9098, "cells": 5898240, "compile_ms": 12680.6, "scan_compile_ms": 12441.6, "step_latency_ms": 17.691, "block_ms": [490.38, 493.66], "parallel_block_ms": [490.38, 493.66], "fused_step_ms": 14.9098, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 14.9098, "total_cells": 5898240, "sypd": 110.1765598652739, "mcells_per_s": 395.5948436598747, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.787424985652218e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:31:00.804504+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:15:[ocean nd=4 strong 384x768x20] compile=12680.6ms fused=14.910ms/step latency=17.691ms/step imbalance=1.0 blocks=[490.38, 493.66]
ocean_ab_xla.26451282.log:17:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:19:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 11.4798, "cells": 5898240, "compile_ms": 12567.3, "scan_compile_ms": 12664.8, "step_latency_ms": 13.535, "block_ms": [378.66, 379.0], "parallel_block_ms": [378.66, 379.0], "fused_step_ms": 11.4798, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 11.4798, "total_cells": 5898240, "sypd": 143.09573967135844, "mcells_per_s": 513.7929232216588, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:31:38.011801+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:20:[ocean nd=4 strong 384x768x20] compile=12567.3ms fused=11.480ms/step latency=13.535ms/step imbalance=1.0 blocks=[378.66, 379.0]
ocean_ab_xla.26451282.log:22:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:24:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 20.9907, "cells": 5898240, "compile_ms": 11769.8, "scan_compile_ms": 12019.8, "step_latency_ms": 23.529, "block_ms": [690.02, 695.37], "parallel_block_ms": [690.02, 695.37], "fused_step_ms": 20.9907, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 20.9907, "total_cells": 5898240, "sypd": 78.25896574574745, "mcells_per_s": 280.99301119067013, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.7874249856471015e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:32:19.224780+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:25:[ocean nd=2 strong 384x768x20] compile=11769.8ms fused=20.991ms/step latency=23.529ms/step imbalance=1.0 blocks=[690.02, 695.37]
ocean_ab_xla.26451282.log:27:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:29:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 19.0344, "cells": 5898240, "compile_ms": 11822.7, "scan_compile_ms": 12444.4, "step_latency_ms": 21.456, "block_ms": [628.63, 627.64], "parallel_block_ms": [628.63, 627.64], "fused_step_ms": 19.0344, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 19.0344, "total_cells": 5898240, "sypd": 86.30219351696194, "mcells_per_s": 309.87265162022436, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:32:54.865045+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:30:[ocean nd=2 strong 384x768x20] compile=11822.7ms fused=19.034ms/step latency=21.456ms/step imbalance=1.0 blocks=[628.63, 627.64]
ocean_ab_xla.26451282.log:32:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:34:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 14.9817, "cells": 5898240, "compile_ms": 12580.6, "scan_compile_ms": 12512.3, "step_latency_ms": 17.818, "block_ms": [492.37, 496.42], "parallel_block_ms": [492.37, 496.42], "fused_step_ms": 14.9817, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 14.9817, "total_cells": 5898240, "sypd": 109.6478018034843, "mcells_per_s": 393.69630949758704, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.787424985652218e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:33:38.045712+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:35:[ocean nd=4 strong 384x768x20] compile=12580.6ms fused=14.982ms/step latency=17.818ms/step imbalance=1.0 blocks=[492.37, 496.42]
ocean_ab_xla.26451282.log:37:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:39:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 11.4625, "cells": 5898240, "compile_ms": 12526.2, "scan_compile_ms": 12734.3, "step_latency_ms": 13.624, "block_ms": [378.04, 378.49], "parallel_block_ms": [378.04, 378.49], "fused_step_ms": 11.4625, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 11.4625, "total_cells": 5898240, "sypd": 143.3117096863041, "mcells_per_s": 514.568375136314, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:34:15.249743+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:40:[ocean nd=4 strong 384x768x20] compile=12526.2ms fused=11.463ms/step latency=13.624ms/step imbalance=1.0 blocks=[378.04, 378.49]
ocean_ab_xla.26451282.log:42:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:44:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 21.0984, "cells": 5898240, "compile_ms": 11909.7, "scan_compile_ms": 12077.6, "step_latency_ms": 23.998, "block_ms": [694.12, 698.38], "parallel_block_ms": [694.12, 698.38], "fused_step_ms": 21.0984, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 21.0984, "total_cells": 5898240, "sypd": 77.85948092174101, "mcells_per_s": 279.5586395176885, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.7874249856471015e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:34:56.812213+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:45:[ocean nd=2 strong 384x768x20] compile=11909.7ms fused=21.098ms/step latency=23.998ms/step imbalance=1.0 blocks=[694.12, 698.38]
ocean_ab_xla.26451282.log:47:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:49:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 19.06, "cells": 5898240, "compile_ms": 11921.1, "scan_compile_ms": 12463.3, "step_latency_ms": 21.293, "block_ms": [629.4, 628.56], "parallel_block_ms": [629.4, 628.56], "fused_step_ms": 19.06, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 19.06, "total_cells": 5898240, "sypd": 86.18627871349742, "mcells_per_s": 309.4564533053515, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:35:33.643493+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:50:[ocean nd=2 strong 384x768x20] compile=11921.1ms fused=19.060ms/step latency=21.293ms/step imbalance=1.0 blocks=[629.4, 628.56]
ocean_ab_xla.26451282.log:52:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:54:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 14.9182, "cells": 5898240, "compile_ms": 12713.4, "scan_compile_ms": 12505.8, "step_latency_ms": 17.866, "block_ms": [491.93, 492.68], "parallel_block_ms": [491.93, 492.68], "fused_step_ms": 14.9182, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 14.9182, "total_cells": 5898240, "sypd": 110.1145226823116, "mcells_per_s": 395.3720958292555, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.787424985652218e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:36:16.677822+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:55:[ocean nd=4 strong 384x768x20] compile=12713.4ms fused=14.918ms/step latency=17.866ms/step imbalance=1.0 blocks=[491.93, 492.68]
ocean_ab_xla.26451282.log:57:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:59:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 11.5016, "cells": 5898240, "compile_ms": 12421.1, "scan_compile_ms": 12516.8, "step_latency_ms": 13.568, "block_ms": [379.28, 379.82], "parallel_block_ms": [379.28, 379.82], "fused_step_ms": 11.5016, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 11.5016, "total_cells": 5898240, "sypd": 142.82451765660957, "mcells_per_s": 512.819086040203, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:36:53.464335+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:60:[ocean nd=4 strong 384x768x20] compile=12421.1ms fused=11.502ms/step latency=13.568ms/step imbalance=1.0 blocks=[379.28, 379.82]
ocean_ab_xla.26451282.log:62:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:64:W0724 18:37:30.254974  170250 profile_guided_latency_estimator.cc:192] Found 339 instructions from the profile.
ocean_ab_xla.26451282.log:65:W0724 18:37:30.255124  170250 profile_guided_latency_estimator.cc:194] Missing 7 instructions from the profile.
ocean_ab_xla.26451282.log:66:W0724 18:37:30.255137  170250 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.3
ocean_ab_xla.26451282.log:67:W0724 18:37:30.255144  170250 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.2
ocean_ab_xla.26451282.log:68:W0724 18:37:30.255150  170250 profile_guided_latency_estimator.cc:197]   loop_compare_fusion
ocean_ab_xla.26451282.log:69:W0724 18:37:30.255155  170250 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.1
ocean_ab_xla.26451282.log:70:W0724 18:37:30.255161  170250 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.4
ocean_ab_xla.26451282.log:71:W0724 18:37:30.255166  170250 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.5
ocean_ab_xla.26451282.log:72:W0724 18:37:30.255171  170250 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.6
ocean_ab_xla.26451282.log:73:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 99.2897, "cells": 5898240, "compile_ms": 12134.4, "scan_compile_ms": 14940.0, "step_latency_ms": 396.562, "block_ms": [3273.97, 3279.15], "parallel_block_ms": [3273.97, 3279.15], "fused_step_ms": 99.2897, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 99.2897, "total_cells": 5898240, "sypd": 16.544621166941393, "mcells_per_s": 59.404349091597624, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.7874249856471015e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:39:05.028705+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:74:[ocean nd=2 strong 384x768x20] compile=12134.4ms fused=99.290ms/step latency=396.562ms/step imbalance=1.0 blocks=[3273.97, 3279.15]
ocean_ab_xla.26451282.log:76:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:78:W0724 18:39:41.845006  172841 profile_guided_latency_estimator.cc:192] Found 347 instructions from the profile.
ocean_ab_xla.26451282.log:79:W0724 18:39:41.845153  172841 profile_guided_latency_estimator.cc:194] Missing 7 instructions from the profile.
ocean_ab_xla.26451282.log:80:W0724 18:39:41.845166  172841 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.5
ocean_ab_xla.26451282.log:81:W0724 18:39:41.845172  172841 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.2
ocean_ab_xla.26451282.log:82:W0724 18:39:41.845178  172841 profile_guided_latency_estimator.cc:197]   loop_compare_fusion
ocean_ab_xla.26451282.log:83:W0724 18:39:41.845183  172841 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.1
ocean_ab_xla.26451282.log:84:W0724 18:39:41.845188  172841 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.4
ocean_ab_xla.26451282.log:85:W0724 18:39:41.845194  172841 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.6
ocean_ab_xla.26451282.log:86:W0724 18:39:41.845199  172841 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.3
ocean_ab_xla.26451282.log:87:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 68.3662, "cells": 5898240, "compile_ms": 12225.3, "scan_compile_ms": 14349.6, "step_latency_ms": 362.396, "block_ms": [2249.19, 2262.98], "parallel_block_ms": [2249.19, 2262.98], "fused_step_ms": 68.3662, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 68.3662, "total_cells": 5898240, "sypd": 24.028108513845446, "mcells_per_s": 86.27421152557842, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:40:11.814730+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:88:[ocean nd=2 strong 384x768x20] compile=12225.3ms fused=68.366ms/step latency=362.396ms/step imbalance=1.0 blocks=[2249.19, 2262.98]
ocean_ab_xla.26451282.log:90:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:92:W0724 18:40:50.581166  174702 profile_guided_latency_estimator.cc:192] Found 341 instructions from the profile.
ocean_ab_xla.26451282.log:93:W0724 18:40:50.581324  174702 profile_guided_latency_estimator.cc:194] Missing 7 instructions from the profile.
ocean_ab_xla.26451282.log:94:W0724 18:40:50.581337  174702 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.6
ocean_ab_xla.26451282.log:95:W0724 18:40:50.581344  174702 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.1
ocean_ab_xla.26451282.log:96:W0724 18:40:50.581349  174702 profile_guided_latency_estimator.cc:197]   loop_compare_fusion
ocean_ab_xla.26451282.log:97:W0724 18:40:50.581355  174702 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.2
ocean_ab_xla.26451282.log:98:W0724 18:40:50.581360  174702 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.4
ocean_ab_xla.26451282.log:99:W0724 18:40:50.581365  174702 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.5
ocean_ab_xla.26451282.log:100:W0724 18:40:50.581370  174702 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.3
ocean_ab_xla.26451282.log:101:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 153.4636, "cells": 5898240, "compile_ms": 12973.8, "scan_compile_ms": 17592.9, "step_latency_ms": 476.956, "block_ms": [5065.85, 5062.74], "parallel_block_ms": [5065.85, 5062.74], "fused_step_ms": 153.4636, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 153.4636, "total_cells": 5898240, "sypd": 10.704235221116022, "mcells_per_s": 38.434130308424926, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.787424985652218e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:42:34.913066+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:102:[ocean nd=4 strong 384x768x20] compile=12973.8ms fused=153.464ms/step latency=476.956ms/step imbalance=1.0 blocks=[5065.85, 5062.74]
ocean_ab_xla.26451282.log:104:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:106:W0724 18:43:13.769082  177320 profile_guided_latency_estimator.cc:192] Found 351 instructions from the profile.
ocean_ab_xla.26451282.log:107:W0724 18:43:13.769235  177320 profile_guided_latency_estimator.cc:194] Missing 7 instructions from the profile.
ocean_ab_xla.26451282.log:108:W0724 18:43:13.769247  177320 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.4
ocean_ab_xla.26451282.log:109:W0724 18:43:13.769253  177320 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.1
ocean_ab_xla.26451282.log:110:W0724 18:43:13.769259  177320 profile_guided_latency_estimator.cc:197]   loop_compare_fusion
ocean_ab_xla.26451282.log:111:W0724 18:43:13.769264  177320 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.2
ocean_ab_xla.26451282.log:112:W0724 18:43:13.769269  177320 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.5
ocean_ab_xla.26451282.log:113:W0724 18:43:13.769274  177320 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.6
ocean_ab_xla.26451282.log:114:W0724 18:43:13.769279  177320 profile_guided_latency_estimator.cc:197]   loop_compare_fusion.3
ocean_ab_xla.26451282.log:115:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 91.7699, "cells": 5898240, "compile_ms": 12935.7, "scan_compile_ms": 15600.8, "step_latency_ms": 392.628, "block_ms": [3024.93, 3031.88], "parallel_block_ms": [3024.93, 3031.88], "fused_step_ms": 91.7699, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 91.7699, "total_cells": 5898240, "sypd": 17.900318865763833, "mcells_per_s": 64.27205434461625, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:43:46.767940+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:116:[ocean nd=4 strong 384x768x20] compile=12935.7ms fused=91.770ms/step latency=392.628ms/step imbalance=1.0 blocks=[3024.93, 3031.88]
ocean_ab_xla.26451282.log:118:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:120:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 18.7317, "cells": 5898240, "compile_ms": 11979.0, "scan_compile_ms": 11973.0, "step_latency_ms": 21.548, "block_ms": [616.34, 619.95], "parallel_block_ms": [616.34, 619.95], "fused_step_ms": 18.7317, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 18.7317, "total_cells": 5898240, "sypd": 87.69681728189437, "mcells_per_s": 314.88012300004806, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.7874183359213272e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:44:27.937402+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "1", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:121:[ocean nd=2 strong 384x768x20] compile=11979.0ms fused=18.732ms/step latency=21.548ms/step imbalance=1.0 blocks=[616.34, 619.95]
ocean_ab_xla.26451282.log:123:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:125:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 16.7396, "cells": 5898240, "compile_ms": 12324.3, "scan_compile_ms": 12649.9, "step_latency_ms": 18.944, "block_ms": [552.27, 552.54], "parallel_block_ms": [552.27, 552.54], "fused_step_ms": 16.7396, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 16.7396, "total_cells": 5898240, "sypd": 98.13319746465035, "mcells_per_s": 352.3525054362111, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:45:04.082274+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "1", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:126:[ocean nd=2 strong 384x768x20] compile=12324.3ms fused=16.740ms/step latency=18.944ms/step imbalance=1.0 blocks=[552.27, 552.54]
ocean_ab_xla.26451282.log:128:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:130:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 13.713, "cells": 5898240, "compile_ms": 12691.0, "scan_compile_ms": 12417.4, "step_latency_ms": 16.418, "block_ms": [451.4, 453.65], "parallel_block_ms": [451.4, 453.65], "fused_step_ms": 13.713, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 13.713, "total_cells": 5898240, "sypd": 119.7922024560097, "mcells_per_s": 430.12032378035445, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.7874184976180632e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:45:46.873743+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "1", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:131:[ocean nd=4 strong 384x768x20] compile=12691.0ms fused=13.713ms/step latency=16.418ms/step imbalance=1.0 blocks=[451.4, 453.65]
ocean_ab_xla.26451282.log:133:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:135:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 10.3343, "cells": 5898240, "compile_ms": 12897.7, "scan_compile_ms": 12886.4, "step_latency_ms": 12.372, "block_ms": [340.69, 341.38], "parallel_block_ms": [340.69, 341.38], "fused_step_ms": 10.3343, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 10.3343, "total_cells": 5898240, "sypd": 158.9571110069633, "mcells_per_s": 570.7440271716516, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:46:24.368252+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "1", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:136:[ocean nd=4 strong 384x768x20] compile=12897.7ms fused=10.334ms/step latency=12.372ms/step imbalance=1.0 blocks=[340.69, 341.38]
ocean_ab_xla.26451282.log:138:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:140:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 21.0847, "cells": 5898240, "compile_ms": 11766.3, "scan_compile_ms": 11990.7, "step_latency_ms": 23.888, "block_ms": [694.16, 697.42], "parallel_block_ms": [694.16, 697.42], "fused_step_ms": 21.0847, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 21.0847, "total_cells": 5898240, "sypd": 77.91007091773943, "mcells_per_s": 279.74028560994464, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.787099541624968e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:47:05.653889+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "1", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:141:[ocean nd=2 strong 384x768x20] compile=11766.3ms fused=21.085ms/step latency=23.888ms/step imbalance=1.0 blocks=[694.16, 697.42]
ocean_ab_xla.26451282.log:143:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:145:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 19.0263, "cells": 5898240, "compile_ms": 12011.7, "scan_compile_ms": 12408.7, "step_latency_ms": 21.352, "block_ms": [628.45, 627.29], "parallel_block_ms": [628.45, 627.29], "fused_step_ms": 19.0263, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 19.0263, "total_cells": 5898240, "sypd": 86.33893464726513, "mcells_per_s": 310.00457261790257, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:47:41.398405+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "1", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:146:[ocean nd=2 strong 384x768x20] compile=12011.7ms fused=19.026ms/step latency=21.352ms/step imbalance=1.0 blocks=[628.45, 627.29]
ocean_ab_xla.26451282.log:148:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_xla.26451282.log:150:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 14.8389, "cells": 5898240, "compile_ms": 12780.7, "scan_compile_ms": 12632.9, "step_latency_ms": 17.687, "block_ms": [488.57, 490.8], "parallel_block_ms": [488.57, 490.8], "fused_step_ms": 14.8389, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 14.8389, "total_cells": 5898240, "sypd": 110.70298150666562, "mcells_per_s": 397.48498877949174, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "solver_residual": null, "residual_reason": "the timed step's in-loop residual is not captured (fixed-iteration PCG exposes no residual in the hot path); a standalone zero-slow-forcing solve at the final state is reported as zero_forcing_probe_residual \u2014 solver-health evidence, not the benchmarked solve's residual", "zero_forcing_probe_residual": 1.787099541625265e-05, "zero_forcing_probe_measured": true, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": true, "halo_messages_per_step": 122, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 1499136, "comm_scope_note": "analytic, barotropic implicit-CN PCG scope only; 2-D eta row slabs (nlev=1), one row per direction (rows_per_message=2); baroclinic 3-D pads NOT counted \u2014 bytes are a lower census", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "latency_us(placeholder with nonzero comm)", "bandwidth_GBs(placeholder with nonzero comm)"], "bound_ingredients": {"compute_ms": null, "comm_ms": 3.199914, "reduction_ms": 3.075, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 122, "halo_bytes_per_step": 1499136, "n_reductions_per_step": 123, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:48:24.786805+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "implicit_cn", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "1", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": null, "solver_iters": 60, "solver_iters_mode": "fixed_pcg[standard,jacobi]", "zero_forcing_probe_measured": true}}}
ocean_ab_xla.26451282.log:151:[ocean nd=4 strong 384x768x20] compile=12780.7ms fused=14.839ms/step latency=17.687ms/step imbalance=1.0 blocks=[488.57, 490.8]
ocean_ab_xla.26451282.log:153:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'latency_us(placeholder with nonzero comm)', 'bandwidth_GBs(placeholder with nonzero comm)']
ocean_ab_xla.26451282.log:155:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 11.3889, "cells": 5898240, "compile_ms": 12691.4, "scan_compile_ms": 12636.0, "step_latency_ms": 13.52, "block_ms": [375.55, 376.12], "parallel_block_ms": [375.55, 376.12], "fused_step_ms": 11.3889, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 11.3889, "total_cells": 5898240, "sypd": 144.2378519680795, "mcells_per_s": 517.8937386402549, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T16:49:01.868660+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "1", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50142.lvt.dkrz.de", "slurm_job_id": "26451282", "git_sha": "3b75507a8-dirty", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_xla.26451282.log:156:[ocean nd=4 strong 384x768x20] compile=12691.4ms fused=11.389ms/step latency=13.52ms/step imbalance=1.0 blocks=[375.55, 376.12]
ocean_ab_xla.26451282.log:158:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_wf.26449827.log:4:{"component": "ocean", "mode": "strong", "n_devices": 1, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 33.4332, "cells": 5898240, "compile_ms": 11695.8, "scan_compile_ms": 12169.3, "step_latency_ms": 36.321, "block_ms": [1102.96, 1103.63], "parallel_block_ms": [1102.96, 1103.63], "fused_step_ms": 33.4332, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 33.4332, "total_cells": 5898240, "sypd": 49.13410837967232, "mcells_per_s": 176.41864972542263, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 5253120.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 5253120, "wet_cell_levels_per_device_max": 5253120, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": false, "halo_messages_per_step": 0, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 0, "comm_scope_note": "single device: no inter-device halo/reduction traffic", "t_bound_ms": 33.4332, "measured_over_bound": 1.0, "bound_calibrated": false, "bound_incomplete_reason": null, "bound_ingredients": {"compute_ms": 33.4332, "comm_ms": 0.0, "reduction_ms": 0.0, "imbalance_ms": 0.0, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 0, "halo_bytes_per_step": 0, "n_reductions_per_step": 0, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T14:52:50.135557+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "none", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 1, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50190.lvt.dkrz.de", "slurm_job_id": "26449827", "git_sha": "18251b0cb", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 5898240, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_wf.26449827.log:5:[ocean nd=1 strong 384x768x20] compile=11695.8ms fused=33.433ms/step latency=36.321ms/step imbalance=1.0 blocks=[1102.96, 1103.63]
ocean_ab_wf.26449827.log:8:--- widefused nd=1 LL384 ---
ocean_ab_wf.26449827.log:9:{"component": "ocean", "mode": "strong", "n_devices": 1, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 33.3974, "cells": 5898240, "compile_ms": 11629.6, "scan_compile_ms": 12239.1, "step_latency_ms": 35.847, "block_ms": [1101.74, 1102.49], "parallel_block_ms": [1101.74, 1102.49], "fused_step_ms": 33.3974, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 33.3974, "total_cells": 5898240, "sypd": 49.186777182632795, "mcells_per_s": 176.6077598855001, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 5253120.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 5253120, "wet_cell_levels_per_device_max": 5253120, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": false, "halo_messages_per_step": 0, "halo_bytes_per_message": 12288, "halo_bytes_per_step": 0, "comm_scope_note": "single device: no inter-device halo/reduction traffic", "t_bound_ms": 33.3974, "measured_over_bound": 1.0, "bound_calibrated": false, "bound_incomplete_reason": null, "bound_ingredients": {"compute_ms": 33.3974, "comm_ms": 0.0, "reduction_ms": 0.0, "imbalance_ms": 0.0, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": 0, "halo_bytes_per_step": 0, "n_reductions_per_step": 0, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T14:53:25.577556+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "none", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 1, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50190.lvt.dkrz.de", "slurm_job_id": "26449827", "git_sha": "18251b0cb", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": true, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 5898240, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_wf.26449827.log:10:[ocean nd=1 strong 384x768x20] compile=11629.6ms fused=33.397ms/step latency=35.847ms/step imbalance=1.0 blocks=[1101.74, 1102.49]
ocean_ab_wf.26449827.log:14:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 19.0275, "cells": 5898240, "compile_ms": 12484.7, "scan_compile_ms": 12537.0, "step_latency_ms": 21.219, "block_ms": [628.0, 627.81], "parallel_block_ms": [628.0, 627.81], "fused_step_ms": 19.0275, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 19.0275, "total_cells": 5898240, "sypd": 86.33348954299097, "mcells_per_s": 309.9850216791486, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T14:54:01.945623+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50190.lvt.dkrz.de", "slurm_job_id": "26449827", "git_sha": "18251b0cb", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_wf.26449827.log:15:[ocean nd=2 strong 384x768x20] compile=12484.7ms fused=19.027ms/step latency=21.219ms/step imbalance=1.0 blocks=[628.0, 627.81]
ocean_ab_wf.26449827.log:17:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_wf.26449827.log:18:--- widefused nd=2 LL384 ---
ocean_ab_wf.26449827.log:19:{"component": "ocean", "mode": "strong", "n_devices": 2, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 18.9135, "cells": 5898240, "compile_ms": 12691.3, "scan_compile_ms": 13846.2, "step_latency_ms": 20.856, "block_ms": [623.74, 624.55], "parallel_block_ms": [623.74, 624.55], "fused_step_ms": 18.9135, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 18.9135, "total_cells": 5898240, "sypd": 86.85385953309861, "mcells_per_s": 311.85343802046157, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 2626560.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 2626560, "wet_cell_levels_per_device_max": 2626560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T14:54:39.974450+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 2, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50190.lvt.dkrz.de", "slurm_job_id": "26449827", "git_sha": "18251b0cb", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": true, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 2949120, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_wf.26449827.log:20:[ocean nd=2 strong 384x768x20] compile=12691.3ms fused=18.913ms/step latency=20.856ms/step imbalance=1.0 blocks=[623.74, 624.55]
ocean_ab_wf.26449827.log:22:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_wf.26449827.log:24:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 11.5197, "cells": 5898240, "compile_ms": 12528.4, "scan_compile_ms": 12651.0, "step_latency_ms": 13.704, "block_ms": [380.0, 380.3], "parallel_block_ms": [380.0, 380.3], "fused_step_ms": 11.5197, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 11.5197, "total_cells": 5898240, "sypd": 142.60010870762784, "mcells_per_s": 512.0133336805645, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T14:55:17.007794+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50190.lvt.dkrz.de", "slurm_job_id": "26449827", "git_sha": "18251b0cb", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": false, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_wf.26449827.log:25:[ocean nd=4 strong 384x768x20] compile=12528.4ms fused=11.520ms/step latency=13.704ms/step imbalance=1.0 blocks=[380.0, 380.3]
ocean_ab_wf.26449827.log:27:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_wf.26449827.log:28:--- widefused nd=4 LL384 ---
ocean_ab_wf.26449827.log:29:{"component": "ocean", "mode": "strong", "n_devices": 4, "n_lat": 384, "n_lon": 768, "nlev": 20, "steps": 33, "platform": "gpu", "n_processes": 1, "multicontroller": false, "steady_median_ms": 11.4961, "cells": 5898240, "compile_ms": 13296.6, "scan_compile_ms": 13697.0, "step_latency_ms": 13.614, "block_ms": [379.19, 379.55], "parallel_block_ms": [379.19, 379.55], "fused_step_ms": 11.4961, "rank_imbalance": 1.0, "rank_imbalance_per_block": [1.0, 1.0], "block_steps": 33, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 384, "n_levels": 20, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 600.0, "time_per_step_ms": 11.4961, "total_cells": 5898240, "sypd": 142.89284820758874, "mcells_per_s": 513.0644305460112, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "solver_residual": null, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "wet_cell_levels": 5253120, "wet_cell_levels_per_device": 1313280.0, "wet_fraction": 0.890625, "wet_equals_total": false, "wet_cell_levels_per_device_min": 1152000, "wet_cell_levels_per_device_max": 1474560, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "wide-halo arm: per-chunk exchange count depends on the auto chunk size; census in extra.barotropic_halo_messages \u2014 bytes not derived (not fabricated)", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-24T14:55:55.943698+00:00", "grid": "latlon", "component": "ocean", "resolution": "384x768", "n_levels": 20, "precision": "float64", "decomposition": "band", "solver_variant": "explicit_substep+wide_halo", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 1, "n_gpus": 4, "device_count": 4, "process_count": 1, "devices_per_rank": 4, "cells_per_rank": 5898240, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "xla-local", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50190.lvt.dkrz.de", "slurm_job_id": "26449827", "git_sha": "18251b0cb", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"steps": 33, "warmup": 3, "multicontroller": false, "fused_halo": true, "nccl": null, "parity_gate": false, "check_conservation": false, "cells_per_device": 1474560, "barotropic_halo_messages": {"standard_messages": 120, "wide_messages_fixed": 4, "wide_messages_per_chunk": 2, "n_loop": 30, "stencil_reach": 3, "chunk_config": 0}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_measured": false}}}
ocean_ab_wf.26449827.log:30:[ocean nd=4 strong 384x768x20] compile=13296.6ms fused=11.496ms/step latency=13.614ms/step imbalance=1.0 blocks=[379.19, 379.55]
ocean_ab_wf.26449827.log:32:[bound] t_bound_ms=None measured_over_bound=None calibrated=False incomplete=['single_device_fused_step_ms', 'halo_messages_per_step', 'halo_bytes_per_step', 'n_reductions_per_step']
ocean_ab_wf.26449827.log:57:*                    worktrees/scaling-campaign/scripts/tmp/ocean_ab_widefused.sbatch

exec
/usr/bin/bash -lc "sed -n '330,425p' docs/performance/scaling/levante_campaign_2026-07-24.md && sed -n '900,1070p' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 355ms:
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

exec
/usr/bin/bash -lc 'rg -n "make_latlon_2d_mpi_step|latlon_2d|pad_with_pole_bc_lat_multi|pad_halo.*multi|pencil" packages/atmosphere packages/core scripts/bench/run_cpu_mpi_scaling.py | head -n 260 && rg -n "latlon-2d|make_latlon_2d" scripts/bench/run_cpu_mpi_scaling.py packages/atmosphere/legoesm/atmosphere/dynamics/gcm/*.py | head -n 220' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 362ms:
scripts/bench/run_cpu_mpi_scaling.py:233:    # (proc_lat x proc_lon pencil).  Lets the collector/plotter separate the
scripts/bench/run_cpu_mpi_scaling.py:234:    # 2-D-pencil curve from the 1-D band laggard.  N/A for other grids ("band"
scripts/bench/run_cpu_mpi_scaling.py:280:    # (mass-weighted flux form, serial + band/2-D-pencil MPI), and
scripts/bench/run_cpu_mpi_scaling.py:461:    latlon_2d: bool = False,
scripts/bench/run_cpu_mpi_scaling.py:494:                            physics_level, _cast, latlon_2d=latlon_2d)
scripts/bench/run_cpu_mpi_scaling.py:809:    """Factor ``n_ranks`` into ``(proc_lat, proc_lon)`` for the 2-D pencil,
scripts/bench/run_cpu_mpi_scaling.py:845:                   physics_level, cast_fn, latlon_2d=False):
scripts/bench/run_cpu_mpi_scaling.py:894:    if n_ranks > 1 and latlon_2d:
scripts/bench/run_cpu_mpi_scaling.py:895:        # 2-D pencil (proc_lat x proc_lon) decomposition — the SOTA fix for
scripts/bench/run_cpu_mpi_scaling.py:900:        # make_latlon_2d_mpi_step (which arms the MPI halo backend + sets the
scripts/bench/run_cpu_mpi_scaling.py:903:            make_latlon_2d_layout,
scripts/bench/run_cpu_mpi_scaling.py:904:            make_latlon_2d_mpi_step,
scripts/bench/run_cpu_mpi_scaling.py:905:            scatter_state_latlon_2d,
scripts/bench/run_cpu_mpi_scaling.py:910:        layout2d = make_latlon_2d_layout(
scripts/bench/run_cpu_mpi_scaling.py:917:        state = scatter_state_latlon_2d(cgrid_global, layout2d)
scripts/bench/run_cpu_mpi_scaling.py:918:        step_fn = make_latlon_2d_mpi_step(
scripts/bench/run_cpu_mpi_scaling.py:1150:    latlon_2d: bool = False,
scripts/bench/run_cpu_mpi_scaling.py:1169:        latlon_2d=latlon_2d,
scripts/bench/run_cpu_mpi_scaling.py:1171:    # "2d" only for a genuine multi-rank lat-lon pencil; everything else
scripts/bench/run_cpu_mpi_scaling.py:1174:        latlon_2d and grid_type == "latlon" and n_ranks > 1
scripts/bench/run_cpu_mpi_scaling.py:1394:    # Tag a non-default (2-D) decomposition into the filename so a 2-D-pencil
scripts/bench/run_cpu_mpi_scaling.py:1545:        help="Lat-lon C-grid 2-D pencil decomposition (proc_lat x proc_lon "
scripts/bench/run_cpu_mpi_scaling.py:1551:             "tests/distributed/test_latlon_2d_mpi_step.py (mass<1e-12 + "
scripts/bench/run_cpu_mpi_scaling.py:1581:    if args.sweep and args.latlon_2d:
scripts/bench/run_cpu_mpi_scaling.py:1772:    if args.latlon_2d and grid_type != "latlon":
scripts/bench/run_cpu_mpi_scaling.py:1778:    # lat).  The 2-D pencil splits lat over proc_lat (< n_ranks), so its own
scripts/bench/run_cpu_mpi_scaling.py:1781:    if (grid_type == "latlon" and not args.latlon_2d
scripts/bench/run_cpu_mpi_scaling.py:1818:        latlon_2d=bool(args.latlon_2d),
packages/core/legoesm/grids/operators_latlon_cgrid.py:65:    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat_multi`).
packages/core/legoesm/grids/operators_latlon_cgrid.py:69:    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
packages/core/legoesm/grids/operators_latlon_cgrid.py:70:    return pad_with_pole_bc_lat_multi(fields, halo=1)
packages/core/legoesm/grids/operators_latlon_cgrid.py:279:    * 2-D pencil (``LatLon2DLayout``) — longitude is split, so the wrap
packages/core/legoesm/grids/operators_latlon_cgrid.py:404:    # Band AND 2-D pencil expose the same pole-terminated lat-LINE
packages/core/legoesm/grids/operators_latlon_cgrid.py:408:    # ``interp_cell_to_vface_halo`` correct on a ``proc_lat>1`` pencil rank
packages/core/legoesm/grids/polar_filter.py:30:# (``make_latlon_2d_mpi_step``) injects an AD-safe gather/scatter pair here
packages/core/legoesm/parallel/latlon_mpi.py:31:                              pencil keeps ``n_lon_local+1`` faces, so W/E
packages/core/legoesm/parallel/latlon_mpi.py:34:                              ``scatter_state_latlon_2d`` / the ``is_u_face``
packages/core/legoesm/parallel/latlon_mpi.py:322:    """2-D pencil (lat × lon) decomposition layout for MPI.
packages/core/legoesm/parallel/latlon_mpi.py:325:    (``docs/performance/scaling/latlon_2d_decomposition_design.md``).  Generalises
packages/core/legoesm/parallel/latlon_mpi.py:368:def make_latlon_2d_layout(
packages/core/legoesm/parallel/latlon_mpi.py:376:    """Build a 2-D pencil decomposition layout for ``rank``.
packages/core/legoesm/parallel/latlon_mpi.py:428:def scatter_field_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py:440:def gather_field_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py:450:    Staggering (mirror of :func:`scatter_state_latlon_2d`): a cell-centred
packages/core/legoesm/parallel/latlon_mpi.py:468:            "gather_field_latlon_2d: a field is u-face OR v-face, not both."
packages/core/legoesm/parallel/latlon_mpi.py:484:            f"gather_field_latlon_2d: layout n_ranks={layout.n_ranks} "
packages/core/legoesm/parallel/latlon_mpi.py:555:    """Adjoint of the lat-pencil gather.
packages/core/legoesm/parallel/latlon_mpi.py:581:    """In-trace lat-pencil transpose: assemble the FULL longitude axis
packages/core/legoesm/parallel/latlon_mpi.py:979:    Increment 1 of the lat-lon 2-D pencil decomposition
packages/core/legoesm/parallel/latlon_mpi.py:980:    (``docs/performance/scaling/latlon_2d_decomposition_design.md``).  Longitude is
packages/core/legoesm/parallel/latlon_mpi.py:1106:    """Lat-axis (N/S) wall pad for a 2-D pencil row — the shared core of
packages/core/legoesm/parallel/latlon_mpi.py:1107:    :func:`pad_halo_latlon_2d` and :func:`pad_with_pole_bc_lat_2d`.
packages/core/legoesm/parallel/latlon_mpi.py:1121:    # would truncate/hang (same guard rationale as pad_halo_latlon_2d's
packages/core/legoesm/parallel/latlon_mpi.py:1179:    """Lat-axis-ONLY wall pad for a 2-D pencil layout (NO longitude halo).
packages/core/legoesm/parallel/latlon_mpi.py:1185:    wall-BC pad is lat-only; the 2-D pencil keeps that contract and the
packages/core/legoesm/parallel/latlon_mpi.py:1197:def pad_halo_latlon_2d(
packages/core/legoesm/parallel/latlon_mpi.py:1205:    """Full 2-D halo pad (lat + lon) for ANY pencil row — the increment-3
packages/core/legoesm/parallel/latlon_mpi.py:1225:        lat-pencil transpose (:func:`lon_gather_full`).  NOT YET wired
packages/core/legoesm/parallel/latlon_mpi.py:1233:            f"pad_halo_latlon_2d: pole_bc must be 'wall', 'fold', or "
packages/core/legoesm/parallel/latlon_mpi.py:1237:            "pad_halo_latlon_2d: pole_bc='fold'/'tripole' needs the "
packages/core/legoesm/parallel/latlon_mpi.py:1238:            "lat-pencil transpose (lon_gather_full) to do the pole-fold's "
packages/core/legoesm/parallel/latlon_mpi.py:1257:            f"pad_halo_latlon_2d: halo={halo} exceeds the smallest local "
packages/core/legoesm/parallel/latlon_mpi.py:1367:                "pad_halo_latlon_mpi: multi-rank lat halo requires "
packages/core/legoesm/parallel/latlon_mpi.py:1410:                "pad_halo_latlon_mpi: multi-rank lat halo requires "
packages/core/legoesm/parallel/latlon_mpi.py:1710:def pad_with_pole_bc_lat_multi_mpi(
packages/core/legoesm/parallel/latlon_mpi.py:1751:            "pad_with_pole_bc_lat_multi_mpi: south_values/north_values "
packages/core/legoesm/parallel/latlon_mpi.py:1762:                "pad_with_pole_bc_lat_multi_mpi: all fields must share "
packages/core/legoesm/parallel/latlon_mpi.py:1768:            f"pad_with_pole_bc_lat_multi_mpi: halo={halo} exceeds "
packages/core/legoesm/parallel/latlon_mpi.py:1900:def scatter_state_latlon_2d(state, layout: LatLon2DLayout):
packages/core/legoesm/parallel/latlon_mpi.py:1904:    band ``[lat_start, lat_end)`` and the longitude pencil
packages/core/legoesm/parallel/latlon_mpi.py:2469:    (the shared boundary row, matching :func:`scatter_state_latlon_2d`).
packages/core/legoesm/parallel/latlon_mpi.py:2897:def make_latlon_2d_mpi_step(
packages/core/legoesm/parallel/latlon_mpi.py:2903:    """Build an MPI step for the lat-lon C-grid dycore on a 2-D pencil.
packages/core/legoesm/parallel/latlon_mpi.py:2911:    the 2-D dispatch (``pad_halo_latlon`` → :func:`pad_halo_latlon_2d`,
packages/core/legoesm/parallel/latlon_mpi.py:2916:    atmosphere's 180° pole fold (that needs a lat-pencil transpose; see
packages/core/legoesm/parallel/latlon_mpi.py:2917:    ``docs/performance/scaling/latlon_2d_build_plan.md``).
packages/core/legoesm/parallel/latlon_mpi.py:2934:    longitude split needs the lat-pencil transpose (fail loud, not a silent
packages/core/legoesm/parallel/latlon_mpi.py:2936:    :func:`pad_halo_latlon_2d`).
packages/core/legoesm/parallel/latlon_mpi.py:2956:            "make_latlon_2d_mpi_step: proc_lon>1 on a TRIPOLAR grid is not "
packages/core/legoesm/parallel/latlon_mpi.py:2959:            "needs the lat-pencil transpose.  Use proc_lon=1 for tripolar, or "
packages/core/legoesm/parallel/latlon_mpi.py:2965:    # WIRED via the AD-safe lat-pencil transpose (``lon_gather_full`` /
packages/core/legoesm/parallel/latlon_mpi.py:2981:            "make_latlon_2d_mpi_step: use_polar_filter=True under a longitude "
packages/core/legoesm/parallel/latlon_mpi.py:3020:            "make_latlon_2d_mpi_step does not thread the PhysicsState "
packages/core/legoesm/grids/halo_latlon.py:152:    :func:`legoesm.parallel.latlon_spmd.make_latlon_2d_pad_body` — lat
packages/core/legoesm/grids/halo_latlon.py:164:        from legoesm.parallel.latlon_spmd import make_latlon_2d_pad_body
packages/core/legoesm/grids/halo_latlon.py:165:        return make_latlon_2d_pad_body(mesh, halo=halo, negate=negate)(data)
packages/core/legoesm/grids/halo_latlon.py:184:def _dispatch_latlon_2d_fold(data, topology, halo, *, is_vector_v):
packages/core/legoesm/grids/halo_latlon.py:197:    lat-pencil transpose (not yet wired) — fall back to the labeled
packages/core/legoesm/grids/halo_latlon.py:198:    wall-pole benchmark pad (``pad_halo_latlon_2d(pole_bc="wall")``).
packages/core/legoesm/grids/halo_latlon.py:199:    :func:`legoesm.parallel.latlon_mpi.make_latlon_2d_mpi_step` now WIRES
packages/core/legoesm/grids/halo_latlon.py:207:        pad_halo_latlon_2d,
packages/core/legoesm/grids/halo_latlon.py:218:    return pad_halo_latlon_2d(data, topology, halo=halo, pole_bc="wall")
packages/core/legoesm/grids/halo_latlon.py:267:            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py:321:            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py:397:            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py:430:            return _dispatch_latlon_2d_fold(
packages/core/legoesm/grids/halo_latlon.py:529:        # Band AND 2-D pencil share the pole-touch test: a rank zeros its
packages/core/legoesm/grids/halo_latlon.py:532:        # pencil this is the proc_row-0 / proc_row-last test (the riskiest
packages/core/legoesm/grids/halo_latlon.py:673:        # 2-D pencil: lat-axis-ONLY wall pad (interior lat cut sendrecv +
packages/core/legoesm/grids/halo_latlon.py:678:        # seam needs the lat-pencil transpose (the wall-pole 2-D benchmark
packages/core/legoesm/grids/halo_latlon.py:682:                "pad_with_pole_bc_lat 2-D pencil: tripolar north_fold / "
packages/core/legoesm/grids/halo_latlon.py:683:                "is_vector_u need the lat-pencil transpose (wall-pole 2-D "
packages/core/legoesm/grids/halo_latlon.py:707:def pad_with_pole_bc_lat_multi(
packages/core/legoesm/grids/halo_latlon.py:760:            "pad_with_pole_bc_lat_multi: south_values/north_values must "
packages/core/legoesm/grids/halo_latlon.py:792:            pad_with_pole_bc_lat_multi_mpi,
packages/core/legoesm/grids/halo_latlon.py:796:            return pad_with_pole_bc_lat_multi_mpi(
packages/core/legoesm/grids/halo_latlon.py:800:    # Local backend / non-latlon topology / 2-D pencil / fused-off:
packages/core/legoesm/grids/halo_latlon.py:802:    # legacy one-sendrecv-pair-per-field schedule).  The 2-D pencil routes
packages/core/legoesm/grids/halo_latlon.py:831:    2-D pencil: pole ownership of the proc-row (lat) axis.
packages/core/legoesm/grids/halo_latlon.py:883:    (:func:`pad_with_pole_bc_lat_multi`); wall-zero at physical poles,
packages/core/legoesm/grids/halo_latlon.py:888:    padded = pad_with_pole_bc_lat_multi(fields, halo=halo)
packages/core/legoesm/grids/halo_latlon.py:921:    padded = pad_with_pole_bc_lat_multi(interiors, halo=halo + 1)
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane_halo.py:53:  ``make_plane_pencil_layout(..., halo=3)`` is bit-identical to the
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane_halo.py:59:  build with ``make_plane_pencil_layout(..., halo=2)`` is bit-identical
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane_halo.py:570:            f"make_plane_pencil_layout(..., halo={_required_halo}) "
packages/core/legoesm/parallel/plane_mpi.py:1:"""MPI 2D periodic pencil decomposition for the doubly-periodic plane.
packages/core/legoesm/parallel/plane_mpi.py:10:* 2D pencil decomposition (both ``ny`` and ``nx`` split) instead of
packages/core/legoesm/parallel/plane_mpi.py:29:``tests/unit/test_plane_mpi_pencil.py``.
packages/core/legoesm/parallel/plane_mpi.py:38:the exchange. Validated by ``tests/distributed/test_plane_pencil_mpi.py``
packages/core/legoesm/parallel/plane_mpi.py:43:* :func:`make_plane_pencil_layout` — stable.
packages/core/legoesm/parallel/plane_mpi.py:48:* :func:`make_plane_pencil_grid` — stable; recomputes ``f_y``,
packages/core/legoesm/parallel/plane_mpi.py:67:    """2D periodic pencil decomposition metadata for the plane grid.
packages/core/legoesm/parallel/plane_mpi.py:118:def make_plane_pencil_layout(
packages/core/legoesm/parallel/plane_mpi.py:127:    """Construct a 2D periodic pencil layout.
packages/core/legoesm/parallel/plane_mpi.py:308:    # uses >999 distinct tags. Two-axis pencil layout only emits 2
packages/core/legoesm/parallel/plane_mpi.py:539:def make_plane_pencil_grid(
packages/core/legoesm/parallel/latlon_spmd.py:39:# choose_latlon_2d_topology).
packages/core/legoesm/parallel/latlon_spmd.py:84:    and of the MPI 2-D pencil's ``exchange_halo_lon``: each tile owns a lon
packages/core/legoesm/parallel/latlon_spmd.py:349:    :func:`make_latlon_2d_pad_body`.
packages/core/legoesm/parallel/latlon_spmd.py:368:    neighbour tile); ``min_tile`` in :func:`choose_latlon_2d_topology`
packages/core/legoesm/parallel/latlon_spmd.py:409:    :func:`make_latlon_band_pad_body` (1-D) or :func:`make_latlon_2d_pad_body`
packages/core/legoesm/parallel/latlon_spmd.py:480:def make_latlon_2d_pad_body(mesh, halo: int = 1, negate: bool = False):
packages/core/legoesm/parallel/latlon_spmd.py:653:    already fuses via ``pad_with_pole_bc_lat_multi_mpi``, the SPMD leg
packages/core/legoesm/parallel/latlon_spmd.py:860:def choose_latlon_2d_topology(
packages/core/legoesm/parallel/latlon_spmd.py:889:        parity (``make_latlon_2d_pad_body._fold_rows``, uniform program —
packages/core/legoesm/parallel/latlon_spmd.py:961:            f"choose_latlon_2d_topology: no feasible (p_lat, p_lon) for "
packages/core/legoesm/parallel/distributed_fft.py:20:2-D FFT: exactly one transpose, one ``alltoall``. A 2-D pencil decomposition
packages/core/legoesm/parallel/distributed_fft.py:23:for spectral runs; the compressible CRM/LES keeps its 2-D pencil ``step_halo``.
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_mpi.py:139:    with the pencil-layout decomposition (typically all-ones on
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_mpi.py:140:    interior + halo zeroed if the pencil includes ghost rows).
packages/atmosphere/legoesm/atmosphere/dynamics/shared/tracer_transport.py:96:        ``pad_halo_4d`` exchange instead of one per tracer (multi-GPU MPI
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1416:# sequential lat-then-lon exchanges inside ``make_latlon_2d_pad_body``; the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1421:# path (choose_latlon_2d_topology returns (N, 1) whenever the band is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1434:def shard_state_atm_latlon_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1455:    _agree_mesh_entry(mesh, state, where="shard_state_atm_latlon_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1471:def gather_state_atm_latlon_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1474:    """Inverse of :func:`shard_state_atm_latlon_2d`: replicate every leaf,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1481:    _agree_mesh_entry(mesh, state, where="gather_state_atm_latlon_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1518:        make_latlon_2d_layout, slice_latlon_grid_to_block_2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1539:                make_latlon_2d_layout(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1680:            f"axis_names=('lat', 'lon')) — choose_latlon_2d_topology picks "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1704:            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1705:            "lat-pencil transpose; the SPMD ppermute equivalent is a "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1716:    :func:`shard_state_atm_latlon_2d` (``v`` as ``v_lower``, ``u`` as
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1721:    routes ``pad_halo_latlon*`` through ``make_latlon_2d_pad_body`` (lat
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1730:    ``tests/parallel/test_atm_latlon_2d_tiling.py``).  The 1-D band factory
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:447:    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:456:         _ps_lat_pad) = pad_with_pole_bc_lat_multi(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:468:        _T_lat_pad, _u_lat_pad, _dp_lat_pad = pad_with_pole_bc_lat_multi(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler.py:460:                                          # plane pencil decomposition (zero comm,
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py:31:    ``LatLon2DLayout`` — a lat×lon pencil).  Returns ``override`` UNCHANGED when no
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1416:# sequential lat-then-lon exchanges inside ``make_latlon_2d_pad_body``; the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1518:        make_latlon_2d_layout, slice_latlon_grid_to_block_2d)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1539:                make_latlon_2d_layout(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1704:            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1721:    routes ``pad_halo_latlon*`` through ``make_latlon_2d_pad_body`` (lat
scripts/bench/run_cpu_mpi_scaling.py:900:        # make_latlon_2d_mpi_step (which arms the MPI halo backend + sets the
scripts/bench/run_cpu_mpi_scaling.py:903:            make_latlon_2d_layout,
scripts/bench/run_cpu_mpi_scaling.py:904:            make_latlon_2d_mpi_step,
scripts/bench/run_cpu_mpi_scaling.py:910:        layout2d = make_latlon_2d_layout(
scripts/bench/run_cpu_mpi_scaling.py:918:        step_fn = make_latlon_2d_mpi_step(
scripts/bench/run_cpu_mpi_scaling.py:1544:        "--latlon-2d", action="store_true",
scripts/bench/run_cpu_mpi_scaling.py:1576:    # --latlon-2d does not propagate through the sweep/--case round-trip yet
scripts/bench/run_cpu_mpi_scaling.py:1577:    # (CaseSpec carries no decomposition field), so a ``--sweep --latlon-2d``
scripts/bench/run_cpu_mpi_scaling.py:1579:    # Fail loud — use a direct ``--resolution N --latlon-2d`` invocation (the
scripts/bench/run_cpu_mpi_scaling.py:1583:            "ERROR: --latlon-2d is not threaded through --sweep yet (CaseSpec "
scripts/bench/run_cpu_mpi_scaling.py:1586:            "--latlon-2d' run (one per rank count) instead.",
scripts/bench/run_cpu_mpi_scaling.py:1774:            print("ERROR: --latlon-2d applies only to --grid latlon.",
scripts/bench/run_cpu_mpi_scaling.py:1780:    # guard for --latlon-2d.

exec
/usr/bin/bash -lc "sed -n '650,820p' packages/core/legoesm/grids/halo_latlon.py && sed -n '1700,1825p' packages/core/legoesm/parallel/latlon_mpi.py && sed -n '1170,1355p' packages/core/legoesm/parallel/latlon_mpi.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 752ms:
            south_value=south_value, north_value=north_value)(interior)

    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    if get_halo_backend() != "mpi":
        # Symmetric constants → single Pad HLO via ``constant_values``
        # tuple (matches the operators' historical formulation).
        return jnp.pad(
            interior, pad_widths,
            constant_values=((south_value, north_value),)
            + ((0, 0),) * (interior.ndim - 1),
        )

    topology = get_mpi_topology()
    # Lat-lon topology is the only kind that maps onto this helper;
    # if the active backend is MPI but for a different grid, fall
    # back to the local serial pad.
    from legoesm.parallel.latlon_mpi import (
        LatLon2DLayout,
        LatLonBandLayout,
        pad_with_pole_bc_lat_2d,
        pad_with_pole_bc_lat_mpi,
    )
    if isinstance(topology, LatLon2DLayout):
        # 2-D pencil: lat-axis-ONLY wall pad (interior lat cut sendrecv +
        # pole wall constant).  Longitude is left untouched — the band path
        # never split lon, so its wall-BC pad is lat-only; the 2-D path
        # keeps that contract and the operator adds lon ghosts through its
        # own dispatched lon halo.  The tripolar north fold / vector-u sign
        # seam needs the lat-pencil transpose (the wall-pole 2-D benchmark
        # excludes it) — fail loud rather than silently wall a fold seam.
        if north_fold or is_vector_u:
            raise NotImplementedError(
                "pad_with_pole_bc_lat 2-D pencil: tripolar north_fold / "
                "is_vector_u need the lat-pencil transpose (wall-pole 2-D "
                f"only). north_fold={north_fold!r} is_vector_u={is_vector_u!r}"
            )
        return pad_with_pole_bc_lat_2d(
            interior, topology, halo=halo,
            south_value=south_value, north_value=north_value,
        )
    if not isinstance(topology, LatLonBandLayout):
        return jnp.pad(
            interior, pad_widths,
            constant_values=((south_value, north_value),)
            + ((0, 0),) * (interior.ndim - 1),
        )
    return pad_with_pole_bc_lat_mpi(
        interior, topology,
        halo=halo,
        south_value=south_value,
        north_value=north_value,
        is_vector_v=is_vector_v,
        is_vector_u=is_vector_u,
        north_fold=north_fold,
    )


def pad_with_pole_bc_lat_multi(
    fields,
    halo: int = 1,
    south_values=None,
    north_values=None,
) -> tuple:
    """Batched :func:`pad_with_pole_bc_lat` for independent wall-BC scalars.

    Pads every field in ``fields`` along the lat axis with constant
    boundary values, backend-dispatched.  Value-identical to calling
    :func:`pad_with_pole_bc_lat` once per field — but under the MPI
    lat-lon band backend the interior partition cuts are exchanged in
    ONE fused sendrecv pair per cut per dtype group instead of one pair
    per field.  mpi4jax sendrecvs are token-serialized (no overlap), so
    each fused cluster of N pads saves ``(N-1) x 2`` sendrecv latencies
    per step — the measured rank-growing term of the ocean baroclinic
    phase (scaling campaign audit lever O4).

    Scalar wall-BC fields only: no ``is_vector_*`` / ``north_fold``
    support (those callers keep the single-field path; their boundary
    handling is field-specific, while the interior-cut exchange this
    helper fuses is flag-independent).

    Set ``LEGOESM_LATLON_FUSED_HALO=0`` to force the per-field
    single-exchange fallback (A/B lever; trace-time Python switch, same
    pattern as the other feature gates).

    Parameters
    ----------
    fields : sequence of jax.Array
        Fields to pad along axis 0.  Must share ``n_lat`` (axis 0);
        trailing shapes / dtypes may differ.
    halo : int
    south_values, north_values : sequence of float, optional
        Per-field boundary constants (default all-zero, i.e.
        ``pad_ns_zero`` semantics).

    Returns
    -------
    tuple of jax.Array, in input order.
    """
    fields = tuple(fields)
    n = len(fields)
    if n == 0:
        return ()
    if south_values is None:
        south_values = (0.0,) * n
    if north_values is None:
        north_values = (0.0,) * n
    south_values = tuple(south_values)
    north_values = tuple(north_values)
    if len(south_values) != n or len(north_values) != n:
        raise ValueError(
            "pad_with_pole_bc_lat_multi: south_values/north_values must "
            f"match len(fields)={n}; got {len(south_values)}/"
            f"{len(north_values)}."
        )

    import os

    from legoesm.grids.halo import get_halo_backend, get_mpi_topology

    # SPMD leg of the message-aggregation lever (audit item 7): ONE
    # ppermute pair per direction per dtype group instead of one per
    # field.  OPT-IN (default off — flip per deck only with a measured
    # GPU A/B receipt, per the audit item's contract).  Value-identical
    # to the per-field pads (the exchange is a bit-copy).
    _spmd_fused = os.environ.get(
        "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
    if _spmd_fused:
        mesh = _spmd_lat_mesh()
        if mesh is not None:
            from legoesm.parallel.latlon_spmd import (
                make_latlon_band_wall_multi_pad_body,
            )
            body = make_latlon_band_wall_multi_pad_body(
                mesh, halo=halo,
                south_values=south_values, north_values=north_values,
                n_fields=n)
            return body(*fields)

    fused = os.environ.get("LEGOESM_LATLON_FUSED_HALO", "1") != "0"
    if get_halo_backend() == "mpi" and fused:
        from legoesm.parallel.latlon_mpi import (
            LatLonBandLayout,
            pad_with_pole_bc_lat_multi_mpi,
        )
        topology = get_mpi_topology()
        if isinstance(topology, LatLonBandLayout):
            return pad_with_pole_bc_lat_multi_mpi(
                fields, topology, halo=halo,
                south_values=south_values, north_values=north_values,
            )
    # Local backend / non-latlon topology / 2-D pencil / fused-off:
    # per-field pads (bit-identical semantics; under band MPI this is the
    # legacy one-sendrecv-pair-per-field schedule).  The 2-D pencil routes
    # HERE on purpose — the fused path above is keyed to LatLonBandLayout,
    # so each field re-enters ``pad_with_pole_bc_lat`` and takes its
    # lat-only ``pad_with_pole_bc_lat_2d`` branch (fusing the 2-D lat
    # sendrecv is a later perf increment, not a correctness gap).
    return tuple(
        pad_with_pole_bc_lat(
            f, halo=halo,
            south_value=south_values[i], north_value=north_values[i],
        )
        for i, f in enumerate(fields)
    )


# ============================================================================
# Wide-halo band widening (opt-in wide-halo split-explicit barotropic)
# ============================================================================

def band_pole_flags():
# identical to per-field sendrecvs; no arithmetic).
#
# Scope: scalar wall-BC fields ONLY (``pad_ns_zero`` /
# ``pad_with_pole_bc_lat`` with constant boundary values, no
# ``north_fold``, no ``is_vector_*``) — boundary slabs are constant fills,
# so per-field flags reduce to per-field constants and the interior cut
# exchange is flag-independent.  Pole-fold / tripolar-fold callers keep
# the single-field path.


def pad_with_pole_bc_lat_multi_mpi(
    fields,
    layout: LatLonBandLayout,
    halo: int = 1,
    south_values=None,
    north_values=None,
):
    """Fused MPI variant of N independent ``pad_with_pole_bc_lat`` calls.

    Pads every field in ``fields`` along the lat axis (axis 0) with
    ``halo`` rows per side: constant ``south_values[i]`` /
    ``north_values[i]`` at pole-touching boundaries, MPI-sendrecv'd
    neighbour rows at interior partition cuts.  All fields must share
    ``n_lat_local`` (axis 0); trailing shapes and dtypes may differ
    (fields are flattened and concatenated per dtype group — ONE
    sendrecv pair per cut per dtype group instead of one per field).

    Value-identical to ``tuple(pad_with_pole_bc_lat_mpi(f, layout,
    halo, sv, nv) for ...)`` for wall-BC scalars: the single-field path
    pole-folds at boundary ranks and then overwrites the boundary slabs
    with the constants, so skipping the fold and filling constants
    directly produces the same result with less local compute.

    AD-safe: the fused buffer goes through the same
    :func:`get_sendrecv_vjp` custom-vjp as the single-field path;
    ``concatenate``/``slice`` carry native JAX VJPs.

    Returns a tuple of padded arrays, in input order.
    """
    fields = tuple(fields)
    n = len(fields)
    if n == 0:
        return ()
    if south_values is None:
        south_values = (0.0,) * n
    if north_values is None:
        north_values = (0.0,) * n
    south_values = tuple(south_values)
    north_values = tuple(north_values)
    if len(south_values) != n or len(north_values) != n:
        raise ValueError(
            "pad_with_pole_bc_lat_multi_mpi: south_values/north_values "
            f"must match len(fields)={n}; got {len(south_values)}/"
            f"{len(north_values)}."
        )
    if halo <= 0:
        return fields

    n_lat_local = fields[0].shape[0]
    for i, f in enumerate(fields):
        if f.shape[0] != n_lat_local:
            raise ValueError(
                "pad_with_pole_bc_lat_multi_mpi: all fields must share "
                f"n_lat_local (axis 0); field 0 has {n_lat_local}, field "
                f"{i} has {f.shape[0]}."
            )
    if halo > n_lat_local:
        raise ValueError(
            f"pad_with_pole_bc_lat_multi_mpi: halo={halo} exceeds "
            f"n_lat_local={n_lat_local} on rank {layout.rank}."
        )

    south_slabs: list = [None] * n
    north_slabs: list = [None] * n

    # Pole-touching boundaries: constant wall-BC fill, no comm.
    if layout.south_rank is None:
        for i, f in enumerate(fields):
            south_slabs[i] = jnp.full(
                (halo,) + f.shape[1:],
                jnp.asarray(south_values[i], dtype=f.dtype),
            )
    if layout.north_rank is None:
        for i, f in enumerate(fields):
            north_slabs[i] = jnp.full(
                (halo,) + f.shape[1:],
                jnp.asarray(north_values[i], dtype=f.dtype),
            )

    # Interior partition cuts: ONE fused sendrecv per cut per dtype group.
    if layout.south_rank is not None or layout.north_rank is not None:
        try:
            import mpi4jax
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError(
                "Lat-lon fused MPI halo exchange (n_ranks>1) requires "
                "mpi4jax and mpi4py."
            ) from exc
        comm = MPI.COMM_WORLD
        sendrecv = get_sendrecv_vjp(mpi4jax)

        # Group field indices by dtype in first-appearance order — the
        # order is trace-deterministic, so every rank issues the same
        # fused-message schedule (sendrecv pairing relies on it).
        groups: dict = {}
        for i, f in enumerate(fields):
            groups.setdefault(jnp.dtype(f.dtype), []).append(i)

        for idxs in groups.values():
            sizes = [
                halo * int(np.prod(fields[i].shape[1:], dtype=np.int64))
                for i in idxs
            ]
            offsets = np.concatenate([[0], np.cumsum(sizes)])

            if layout.south_rank is not None:
                send_bot = jnp.concatenate(
                    [fields[i][:halo].reshape(-1) for i in idxs]
                )
                recv_south = sendrecv(
                    send_bot, jnp.zeros_like(send_bot),
                    layout.south_rank,      # source
                    layout.south_rank,      # dest
                    layout.rank,            # sendtag = sender's rank
                    layout.south_rank,      # recvtag = source's rank


def pad_with_pole_bc_lat_2d(
    interior: jax.Array,
    layout: LatLon2DLayout,
    halo: int = 1,
    south_value: float = 0.0,
    north_value: float = 0.0,
) -> jax.Array:
    """Lat-axis-ONLY wall pad for a 2-D pencil layout (NO longitude halo).

    The 2-D analogue of the band
    :func:`legoesm.grids.halo_latlon.pad_with_pole_bc_lat` MPI path: pad
    only the lat axis (interior cut sendrecv + pole wall constant), leaving
    longitude untouched.  The band path never split longitude, so its
    wall-BC pad is lat-only; the 2-D pencil keeps that contract and the
    operator adds lon ghosts through its own dispatched lon halo.  Routed
    here from ``halo_latlon.pad_with_pole_bc_lat`` when the active topology
    is a :class:`LatLon2DLayout` (wall poles only; the tripolar north fold /
    vector-u seam is excluded upstream).  AD-safe via the shared sendrecv
    VJP.
    """
    if halo <= 0:
        return interior
    return _pad_lat_wall_2d(interior, layout, halo, south_value, north_value)


def pad_halo_latlon_2d(
    field: jax.Array,
    layout: LatLon2DLayout,
    halo: int = 1,
    pole_bc: str = "wall",
    south_value: float = 0.0,
    north_value: float = 0.0,
) -> jax.Array:
    """Full 2-D halo pad (lat + lon) for ANY pencil row — the increment-3
    integration of the 2-D building blocks.

    N/S then E/W (corners ride the lat-padded edge columns into the E/W
    exchange).  N/S: interior cuts MPI-sendrecv with the lat neighbour;
    pole-touching rows fill the pole side per ``pole_bc``.  Every rank
    participates in its interior-facing sendrecv (the line terminates at
    the pole rows' local fill) so there is NO collective-line deadlock —
    the failure of the earlier "guard-and-skip" interior-only attempt.
    E/W: the periodic-ring :func:`exchange_halo_lon`.

    ``pole_bc``:
      * ``"wall"`` (default) — pole ghost rows = constant wall BC
        (``south_value``/``north_value``).  This is the REGULAR lat-lon
        case (and ocean, whose poles are closed walls).  Fully LOCAL at
        the poles ⇒ no longitude transpose, and the whole pad is
        DEADLOCK-FREE and AD-SAFE (sendrecv-VJP on both axes).
      * ``"fold"`` / ``"tripole"`` — the atmospheric 180° pole-fold and
        the ocean tripole north-fold need the FULL longitude axis at the
        pole row (180° shift / permutation), so they require the
        lat-pencil transpose (:func:`lon_gather_full`).  NOT YET wired
        (next increment); raises so a fold deck can't silently get a
        wall.

    Returns ``(n_lat_local + 2*halo, n_lon_local + 2*halo[, nlev])``.
    """
    if pole_bc not in ("wall", "fold", "tripole"):
        raise ValueError(
            f"pad_halo_latlon_2d: pole_bc must be 'wall', 'fold', or "
            f"'tripole', got {pole_bc!r}")
    if pole_bc in ("fold", "tripole"):
        raise NotImplementedError(
            "pad_halo_latlon_2d: pole_bc='fold'/'tripole' needs the "
            "lat-pencil transpose (lon_gather_full) to do the pole-fold's "
            "global-longitude shift/permutation on a lon-split row — next "
            "increment.  Use pole_bc='wall' for regular lat-lon / "
            "closed-pole ocean."
        )
    if halo <= 0:
        return field

    # halo must fit the SMALLEST local block on BOTH axes — with an
    # uneven split a neighbour can own fewer than `halo` rows/cols, so
    # its send/recv would be shorter than this rank expects and the MPI
    # exchange truncates/aborts/hangs before any reshape (codex MAJOR
    # 2026-06-13). Smallest block on an even-ish split = floor(n/proc).
    # _even_split gives the first (n % parts) blocks one extra row/col, so
    # the SMALLEST block is exactly floor(n_global / parts).
    min_lat_block = layout.n_lat_global // layout.proc_lat
    min_lon_block = layout.n_lon_global // layout.proc_lon
    if halo > min_lat_block or halo > min_lon_block:
        raise ValueError(
            f"pad_halo_latlon_2d: halo={halo} exceeds the smallest local "
            f"block (lat {min_lat_block}=n_lat_global "
            f"{layout.n_lat_global}//proc_lat {layout.proc_lat}, lon "
            f"{min_lon_block}=n_lon_global {layout.n_lon_global}//proc_lon "
            f"{layout.proc_lon}); a neighbour would send/recv a mismatched "
            f"halo and the MPI exchange would abort/hang.")

    # N/S — lat-axis wall pad (interior cut sendrecv at the AD-safe
    # rank-as-tag LINE pattern, pole side wall); shared verbatim with
    # pad_with_pole_bc_lat_2d so the two lat exchanges stay bit-identical.
    ns = _pad_lat_wall_2d(field, layout, halo, south_value, north_value)
    # E/W ring on the lat-padded block → fills lon ghosts + corners.
    return exchange_halo_lon(
        ns, layout.west_rank, layout.east_rank, layout.rank, halo=halo)


# ============================================================================
# Backend-dispatched pad_halo_latlon implementation
# ============================================================================
#
# Used by :mod:`legoesm.grids.halo_latlon` when the global halo
# backend is ``"mpi"`` and the active topology is a
# :class:`LatLonBandLayout`.  Composes the existing
# :func:`exchange_halo_latlon` (lat MPI sendrecv + boundary pole-fold)
# with a periodic-lon wrap that every rank performs locally.  Output
# shape matches the serial :func:`legoesm.grids.halo_latlon.pad_halo_latlon`
# family — operators stay backend-oblivious.
#
# This is the architectural reuse point the user asked for: no
# parallel registry, no operator-side branching, no duplicate
# halo-machinery.  The cubed-sphere precedent
# (``set_halo_backend`` → ``_halo_backend`` global → ``pad_halo``
# dispatch) is mirrored exactly.


def pad_halo_latlon_mpi(
    data, layout: LatLonBandLayout, halo: int = 1,
    is_vector_v: bool = False, is_vector_u: bool = False,
):
    """MPI variant of :func:`legoesm.grids.halo_latlon.pad_halo_latlon`.

    Step 1 — periodic lon wrap (same on every rank; every rank owns
    the full longitude axis).  This matches the serial code's
    lon-pad order so the operator's downstream stencil sees the same
    layout under both backends.

    Step 2 — lat halo via :func:`exchange_halo_latlon`:
      - Boundary ranks (``layout.<side>_rank is None``) apply the
        pole-fold convention used by the serial
        :func:`legoesm.grids.halo_latlon.pad_halo_latlon` (mirror +
        180° lon shift + sign flip for vectors).
      - Interior partition cuts MPI-sendrecv with the neighbour rank.

    The result has the same shape and semantics as the serial
    helper.  Under MPI the lat axis is the rank's band plus the halo
    rows; the lon axis is the full global lon plus its periodic
    halo, identical to serial.
    """
    if data.ndim not in (2, 3):
        raise ValueError(
            f"pad_halo_latlon_mpi: data.ndim must be 2 or 3, got {data.ndim}"
        )

    # Step 1: periodic lon wrap (local on every rank — every rank owns
    # the full lon axis).  This produces ``lon_padded`` shape
    # ``(n_lat_local, n_lon + 2*halo[, nlev])``.
    if data.ndim == 2:
        lon_padded = jnp.pad(data, ((0, 0), (halo, halo)), mode="wrap")
    else:
        lon_padded = jnp.pad(
            data, ((0, 0), (halo, halo), (0, 0)), mode="wrap",
        )

    # Step 2: lat halo — pole-fold the LON-PADDED data at boundary
    # ranks (matches serial ``pad_halo_latlon``'s lon-pad-then-fold
    # order, so the pole-fold lon-shift is ``(n_lon + 2*halo) // 2``
    # like serial does).  Interior partition cuts MPI sendrecv with
    # the neighbour rank.
    #
    # We do NOT call ``exchange_halo_latlon`` here because that
    # function operates on an *unpadded* field and is designed to
    # return serial-stripped (lon-halo-stripped) halos.  Routing
    # through it would either re-lon-pad or produce stripped halos
    # — neither matches the full-pad result serial expects.
    if halo > data.shape[0]:
        raise ValueError(
            f"pad_halo_latlon_mpi: halo={halo} exceeds n_lat_local="
            f"{data.shape[0]} on rank {layout.rank}."
        )

    # --- South halo ---
    if layout.south_rank is None:
        # Pole-fold the lon-padded first ``halo`` rows.  Uses
        # ``fold_pole_rows*`` whose lon-shift is ``data.shape[1] // 2``
        # — for lon-padded input that's exactly ``(n_lon + 2*halo) // 2``
        # which is the serial convention.
        if data.ndim == 2:
            south_halo, _ = fold_pole_rows(lon_padded, halo, negate=is_vector_v)
        else:

exec
/usr/bin/bash -lc "nl -ba docs/performance/scaling/levante_campaign_2026-07-24.md | sed -n '2110,2295p'; nl -ba packages/core/legoesm/grids/halo_latlon.py | sed -n '760,815p'; nl -ba packages/core/legoesm/parallel/latlon_mpi.py | sed -n '1700,1718p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 307ms:
  2110	## FINAL RECEIPTS (job 26657279): atm lat-lon at 144 GPUs — campaign records
  2111	
  2112	The @192 ask starved 20+ h (48-node block vs a 55-healthy-node pool);
  2113	144 divides both grids and scheduled overnight:
  2114	
  2115	| arm | cols/GPU | ms/step | GC/s |
  2116	|---|---|---|---|
  2117	| LL2304x4608 @144 | 73.7k | 5.78 | 47.76 |
  2118	| LL2880x5760 @144 | 115.2k | 7.40 | **58.32** |
  2119	
  2120	* **58.3 GC/s is the campaign's highest throughput** (prior record
  2121	  39.1, LL2048@128).
  2122	* Strong LL2304 96 -> 144: 7.872 -> 5.78 = 1.362x for 1.5x devices =
  2123	  **eff 0.91** — the healthiest >64-GPU strong leg measured (tiles
  2124	  110.6k -> 73.7k, both far above the floor). (@96 ran the pre-merge
  2125	  worktree; the put-path fixes are SETUP-only, so steady timing is
  2126	  comparable.)
  2127	* Campaign close-out: every directive lane holds 96-144-GPU receipts
  2128	  (atm lat-lon 96/128/144, ocean lat-lon 96/128, MPAS 128) plus
  2129	  512-rank CPU lat-lon; the remaining improvement paths are the
  2130	  documented structural follow-ups (SoL-class device collectives,
  2131	  #1100 partition-local mesh, ensemble orchestration).
  2132	
  2133	## AT-SCALE fabric constants (jobs 26677438/26677439) — the limit lines move
  2134	
  2135	The roofline constants were measured at 8 GPUs / 2 nodes; remeasured at
  2136	the gap's own scale with the SAME chained-fori_loop microbench
  2137	(dispatch-subtracted):
  2138	
  2139	| communicator | latency (us) | bandwidth (GB/s) |
  2140	|---|---|---|
  2141	| 8 GPU / 2 nodes | 26.3 | 23.5 |
  2142	| 64 GPU / 16 nodes | 28.4 | **12.1** |
  2143	| 128 GPU / 32 nodes | 29.7 | **12.1** |
  2144	
  2145	* **Topology-LATENCY term: refuted for the chained-ring pattern
  2146	  measured** (+13 % at 16x the communicator; the fori_loop chain prices
  2147	  dependency-chained ring cost). SCOPE (codex r23): the microbench
  2148	  measures ppermute rings with dispatch subtracted — not production
  2149	  pair patterns, packing, synchronization skew, or all-reduce; the
  2150	  bound's one-latency AR term is a MODEL ASSUMPTION.
  2151	* **Bandwidth HALVES past 2 nodes** (ring crossing switch tiers shares
  2152	  links): 23.5 -> 12.1 GB/s. The bound's byte term doubles.
  2153	* Recomputed distances with at-scale constants:
  2154	  - LL2048@128 f32: comm = 41x29.7us + 18.54MB/12.1 = 2.750 ms;
  2155	    t_bound 2.780; **measured/bound 2.01** (was 2.94). Pricing every
  2156	    CP at the sweep's 512 KiB row (82.085 us; the census MEAN payload
  2157	    is 441.6 KiB, so this slightly over-prices): 41x82.085us + one
  2158	    29.7us AR-assumption = 3.395 ms -> **ratio 1.64** (post-overlap
  2159	    5.077 -> **1.50**). (codex r23 corrected the first draft's 3.11.)
  2160	  - LL2048@64 f32: comm 2.698 vs compute 2.837 — BALANCED regime;
  2161	    bound 2.865, ratio 2.35 (compute-edge, unchanged).
  2162	  - MPAS s9@np64: comm = 33x28.4us + 29.49MB/12.1 = 3.375; bound
  2163	    3.403; **ratio 2.82** (was 4.47).
  2164	* Honest reframe: with constants measured AT the deployment scale, the
  2165	  panels sit ~1.8-2.8x above the serialized model, and the comm term is
  2166	  now BYTE-dominated — bytes are physical (halo areas), so the
  2167	  remaining levers are OVERLAP (hide comm under compute; the closed
  2168	  ledger's null was the latency-dominated ocean nd4-16 regime, not
  2169	  this one — A/B job 26677529 submitted at @64 with the XLA
  2170	  latency-hiding scheduler + pipelined collectives) and exchange-COUNT
  2171	  packing (skew amplification).
  2172	
  2173	## OVERLAP VERDICTS (jobs 26677602 / 26677668 / 26677669) — the first positive lever
  2174	
  2175	XLA latency-hiding scheduler + pipelined p2p (flag names verified
  2176	against the installed stack after a guessed name aborted arm B of
  2177	26677529; that job still banked clean controls 6.667/6.724):
  2178	
  2179	| lane | A | B (overlap) | A2 | effect |
  2180	|---|---|---|---|---|
  2181	| LL2048@64 | 6.615 | 6.069 | 6.635 | **-8.4 %** |
  2182	| LL2048@128 | 5.547 | **5.077** | 5.527 | **-8.3 %** |
  2183	| MPAS s9@64 | 9.710 | 9.720 | 9.680 | 0.0 % (null) |
  2184	
  2185	* Twice-reproduced ~8 % on the lat-lon lane at two scales with 0.3-0.4 %
  2186	  control drift; BELOW the pre-registered 10 % bar (reported as such),
  2187	  wired strictly OPT-IN (`LEGOESM_XLA_OVERLAP=1`) — never a shared
  2188	  default: cube_tiled_step force-disables latency hiding for a known
  2189	  sensitivity, MPAS is null, and appended flags would poison future
  2190	  A/B control arms (codex r23). Parity suites green with flags on (CPU-virtual — the GPU-side
  2191	  check is the A/B rows themselves). MPAS: honest null — the
  2192	  edge-coloured hand schedule does not benefit.
  2193	* New LL2048@128 best: **5.077 ms = 43.0 GC/s**; measured/at-scale-bound
  2194	  = 5.077/2.780 = **1.83** (sweep-based bound 3.11 -> **1.63**).
  2195	
  2196	## MPAS #1100 wall NAMED + FIXED: replicate_pytree
  2197	
  2198	The s9 STEP program is clean (bufdump: 0.02 GB args, 4 per-shard
  2199	params) — the s10 162 GB was `replicate_pytree(mesh)`: a replicated
  2200	device_put of the 1.268 GB global-mesh pytree = 128 x 1.268 =
  2201	162.3 GB (matches the failure to 0.1 %), the SAME jax
  2202	whole-array-assert + replicated-logical wall as the ocean lane. Fixed
  2203	via the shared assert-free put + exact-hash contract gate
  2204	(PR #1457 pattern) in `parallel/mesh.py`; 2-proc multicontroller MPAS
  2205	repro green (56.97 ms, s5), voronoi parity 5 passed. NOTE (codex r23):
  2206	replicate_pytree is generic — the multi-process path can reach other
  2207	lanes (cube CLI); a structure pre-gate + direct tests added same
  2208	round. Falsification = s10@128 rerun (job 26677812, in queue): a PASS
  2209	is the receipt for MPAS at 128 GPUs; 192/224 remain EXPECTED-unlocked
  2210	pending their own runs.
  2211	
  2212	## Lat-lon packing follow-up (designed, not yet built)
  2213	
  2214	CP records (nd=8 census): the 41 CPs are SIX classes — 13x[1,1024,26],
  2215	6x[1,1025,26], 6x[2,1028,26], 6x[1,1026,27], 6x[1,1024,27] array
  2216	exchanges (31 one-row + 6 two-row) and 4 scalar f32[1] — codex r23
  2217	corrected the first draft's two-class count. Grouping by shape alone
  2218	does NOT prove packability: fields must be AVAILABLE at a common
  2219	program point, so the design step is a stage-local liveness analysis
  2220	across the RK stages (the `make_latlon_band_wall_multi_pad_body`
  2221	machinery exists for the ocean wall lane; the atm dycore pads
  2222	per-field). The ~6-group / ~4.0-4.4 ms projection is SPECULATIVE until
  2223	that analysis is done. Dycore surgery — staged as the next engineering
  2224	item.
  2225	
  2226	## #1100 WALL DOWN (job 26677812): MPAS s10 @ 128 GPUs — new record
  2227	
  2228	Post-replicate_pytree-fix falsification PASSES: subdiv-10 (10.49M
  2229	cells, lloyd-0) at 128 GPUs = **18.20 ms = 14.98 GC/s — the new
  2230	MPAS-atmosphere record** (2.1x the s9 peak of 7.10), at 81.9k
  2231	cells/GPU (above the ~30k floor). The lloyd-0 matched-tile weak series
  2232	gains a THIRD rung: s8@8 6.58 -> s9@32 12.47 -> s10@128 18.20 ms —
  2233	per-4x-scale cost 1.895 then 1.460: the scale-out term DECELERATES
  2234	with size. 192/224-GPU rungs are expected-feasible (48/56-node asks;
  2235	queue-starved historically — submit opportunistically).
  2236	
  2237	## Ocean-MPAS GPU path: SCOPED (not built)
  2238	
  2239	The ocean MPAS lane is CPU-MPI by design (`bench_ocean_mpas_scaling`
  2240	docstring). A GPU/SPMD twin would reuse the now-generic voronoi
  2241	machinery (cell-partition reorder, padded local meshes, edge-coloured
  2242	ppermute halo, the fixed replicate/put path) and wire the OCEAN MPAS
  2243	tendencies (`ocean_pe_mpas`) into a `make_voronoi_sharded_step`-class
  2244	factory: column-local vmix solves shard trivially; the
  2245	barotropic/baroclinic split is the design work. Estimated days-scale
  2246	feature with existing infra; staged as a follow-up, NOT attempted in
  2247	this round.
  2248	
  2249	## FUSED HALO + OVERLAP: the stacked levers (jobs 26681636 / 26681858) — lat-lon at ratio 1.4
  2250	
  2251	The audit-item-7 SPMD fused multi-pad already existed OPT-IN
  2252	(`LEGOESM_LATLON_SPMD_FUSED_HALO=1`, contract: "flip per deck only with
  2253	a measured GPU A/B receipt") — census 41 -> 29 CPs/step at IDENTICAL
  2254	bytes. The receipts:
  2255	
  2256	| arm | @64 (ms) | @128 (ms) |
  2257	|---|---|---|
  2258	| A off | 6.590 | 5.573 |
  2259	| B fused | 6.264 (-5.4 %) | 5.325 (-4.5 %) |
  2260	| C fused+overlap | **5.278 (-20.3 %)** | **4.745 (-15.0 %)** |
  2261	| A2 off | 6.650 | 5.585 |
  2262	
  2263	* The combined effect EXCEEDS the additive expectation (20.3 % vs
  2264	  13.8 % at @64; 15.0 % vs 12.9 % at @128) — CONSISTENT WITH fewer,
  2265	  larger CPs giving the latency-hiding scheduler more to hide
  2266	  (mechanism plausible, not instrumented).
  2267	* **New LL2048@128 best: 4.745 ms = 46.0 GC/s.** The bound must be
  2268	  REPRICED for the fused count (codex r26 — the lever moves the model
  2269	  too): 29 CPs x 29.7 us + 18.54 MB / 12.1 GB/s = 2.394, bound
  2270	  2.423 ms -> **measured/bound 1.96** (fit); sweep-priced at the new
  2271	  ~639 KiB/message (linear interpolation 524 KiB -> 1 MiB rows,
  2272	  ~88 us/CP): 29 x 88.2 + 29.7 us = 2.588 -> **~1.83**. @64: 5.278 vs
  2273	  the compute-dominated 2.865 bound -> 1.84 (unchanged by count).
  2274	* Parity: 23 tests green with the fused env ON; the COLLECTIVE PAYLOAD
  2275	  is byte-identical to the per-field pads (local concat/split traffic
  2276	  differs), A/A2 drift 0.2-0.9 %.
  2277	* Wired: the lat-lon hundreds launchers set BOTH envs — code-path
  2278	  validated (the exact 144-GPU points carry no dedicated A/B receipt);
  2279	  everything else stays opt-in (cube force-disables latency hiding;
  2280	  MPAS null).
  2281	
  2282	## MPAS closure: at the practical stack limit
  2283	
  2284	nsys (26680051): compute is a ~2.7 us MICROKERNEL storm (launch/
  2285	scheduling-bound — explains the overlap null) and NCCL SendRecv medians
  2286	~2x the clean wire estimate (skew absorbed in kernels). CUDA-graph
  2287	levers REFUTED (26680791: min-graph-size = exact no-op; command-buffer
  2288	with collectives +23 % WORSE). With protocol, partition, placement,
  2289	combining, overlap and graphs all receipted null, single-trajectory
  2290	MPAS stands at its practical XLA/NCCL stack limit (s9@64 ratio 2.82 on
  2291	the at-scale model); the scaling story there is the s10@128 record
  2292	(14.98 GC/s), the DECELERATING matched-tile cost (1.90 -> 1.46 per
  2293	4x), and ensemble parallelism (+0.6 % co-execution). Deeper wins need
  2294	XLA fusion-granularity work on unstructured ops — upstream-class.
   760	            "pad_with_pole_bc_lat_multi: south_values/north_values must "
   761	            f"match len(fields)={n}; got {len(south_values)}/"
   762	            f"{len(north_values)}."
   763	        )
   764	
   765	    import os
   766	
   767	    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
   768	
   769	    # SPMD leg of the message-aggregation lever (audit item 7): ONE
   770	    # ppermute pair per direction per dtype group instead of one per
   771	    # field.  OPT-IN (default off — flip per deck only with a measured
   772	    # GPU A/B receipt, per the audit item's contract).  Value-identical
   773	    # to the per-field pads (the exchange is a bit-copy).
   774	    _spmd_fused = os.environ.get(
   775	        "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
   776	    if _spmd_fused:
   777	        mesh = _spmd_lat_mesh()
   778	        if mesh is not None:
   779	            from legoesm.parallel.latlon_spmd import (
   780	                make_latlon_band_wall_multi_pad_body,
   781	            )
   782	            body = make_latlon_band_wall_multi_pad_body(
   783	                mesh, halo=halo,
   784	                south_values=south_values, north_values=north_values,
   785	                n_fields=n)
   786	            return body(*fields)
   787	
   788	    fused = os.environ.get("LEGOESM_LATLON_FUSED_HALO", "1") != "0"
   789	    if get_halo_backend() == "mpi" and fused:
   790	        from legoesm.parallel.latlon_mpi import (
   791	            LatLonBandLayout,
   792	            pad_with_pole_bc_lat_multi_mpi,
   793	        )
   794	        topology = get_mpi_topology()
   795	        if isinstance(topology, LatLonBandLayout):
   796	            return pad_with_pole_bc_lat_multi_mpi(
   797	                fields, topology, halo=halo,
   798	                south_values=south_values, north_values=north_values,
   799	            )
   800	    # Local backend / non-latlon topology / 2-D pencil / fused-off:
   801	    # per-field pads (bit-identical semantics; under band MPI this is the
   802	    # legacy one-sendrecv-pair-per-field schedule).  The 2-D pencil routes
   803	    # HERE on purpose — the fused path above is keyed to LatLonBandLayout,
   804	    # so each field re-enters ``pad_with_pole_bc_lat`` and takes its
   805	    # lat-only ``pad_with_pole_bc_lat_2d`` branch (fusing the 2-D lat
   806	    # sendrecv is a later perf increment, not a correctness gap).
   807	    return tuple(
   808	        pad_with_pole_bc_lat(
   809	            f, halo=halo,
   810	            south_value=south_values[i], north_value=north_values[i],
   811	        )
   812	        for i, f in enumerate(fields)
   813	    )
   814	
   815	
  1700	# identical to per-field sendrecvs; no arithmetic).
  1701	#
  1702	# Scope: scalar wall-BC fields ONLY (``pad_ns_zero`` /
  1703	# ``pad_with_pole_bc_lat`` with constant boundary values, no
  1704	# ``north_fold``, no ``is_vector_*``) — boundary slabs are constant fills,
  1705	# so per-field flags reduce to per-field constants and the interior cut
  1706	# exchange is flag-independent.  Pole-fold / tripolar-fold callers keep
  1707	# the single-field path.
  1708	
  1709	
  1710	def pad_with_pole_bc_lat_multi_mpi(
  1711	    fields,
  1712	    layout: LatLonBandLayout,
  1713	    halo: int = 1,
  1714	    south_values=None,
  1715	    north_values=None,
  1716	):
  1717	    """Fused MPI variant of N independent ``pad_with_pole_bc_lat`` calls.
  1718	

exec
/usr/bin/bash -lc "nl -ba docs/performance/scaling/levante_campaign_2026-07-24.md | sed -n '2070,2112p;1030,1070p' && nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1900,1952p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 318ms:
  1030	**SKEW LOCATED (4-rank profile, job 26526100).** Aligning SendRecv
  1031	launches by sequence across ranks 0/1/12/23 over two steady steps:
  1032	
  1033	* 10 of 172 exchanges carry >1 ms of max duration, and **9 of the 10 are
  1034	  skew-dominated** — arrival skew ~= max duration (e.g. 13.6 ms skew vs
  1035	  12.6 ms duration with the last arriver's own service at 14 us: early
  1036	  ranks WAIT the full skew).
  1037	* The ordering is SYSTEMATIC: rank 23 arrives last 95/140 times, rank 0
  1038	  44/140; rank 12 arrives FIRST 135/140. Median idle gap before a late
  1039	  arrival is 20 us — the late rank was computing back-to-back, not
  1040	  blocked upstream.
  1041	* BUT total per-rank work is EQUAL: compute 8.2-8.5 ms/step in ~428
  1042	  kernels on every rank.
  1043	
  1044	Equal totals + systematically late at fixed sequence points = **pipeline
  1045	drift, not load imbalance**: the 16-phase pad sequence has
  1046	rank-dependent participation (partial permutes let non-target ranks run
  1047	ahead), the drift accumulates within the step, and the ~10
  1048	full-participation exchanges act as resync barriers where the
  1049	accumulated drift is paid as wait. The cost is real (~7.5 ms/step in the
  1050	wait tail) but the remedy is ALGORITHMIC — reorder/merge pad phases so
  1051	drift cannot accumulate, or overlap the resync exchanges with interior
  1052	compute — a scoped dycore-scheduling follow-up, not a bench or config
  1053	change. No further profiling is needed; the mechanism chain
  1054	(launch-count -> bytes -> skew -> ordering) is now measured end to end.
  1055	
  1056	## Tripole (ORCA fold) past 4 GPUs — first receipts (job 26512798)
  1057	
  1058	The fold is the ocean's production topology but had only ever been
  1059	measured to 4 GPUs. LL1152x2304 L20 f32, implicit_cn + forced PCG:
  1060	
  1061	| devices | tile | ms/step | GC/s |
  1062	|---|---|---|---|
  1063	| 8 | 331.8k cols/GPU | 33.97 | 1.56 |
  1064	| 16 | 165.9k | **23.79** | 2.23 |
  1065	
  1066	8->16 = 1.43x, **efficiency 0.71** — the fold scales respectably at
  1067	production tiles, and the topology is not a scale-out blocker at these
  1068	counts.
  1069	
  1070	SCOPE, stated because it is tempting to misread: the regular-grid
  2070	
  2071	Falsification v3 in queue: oc LL2304 @96 (26646038) / @128 (26646039),
  2072	exclude l50081,l50100 (superset; harmless).
  2073	
  2074	## OCEAN AT HUNDREDS: falsification v3 PASSES (jobs 26646038/26646039)
  2075	
  2076	First-ever ocean lat-lon multicontroller receipts past 64 GPUs, on the
  2077	fixed lane (l50081+l50100 excluded; l50081 exclusion harmless-superset):
  2078	
  2079	| arm | cols/GPU | ms/step | GC/s |
  2080	|---|---|---|---|
  2081	| LL2304x4608 L20 @96 | 110.6k | 18.25 | 11.63 |
  2082	| LL2304x4608 L20 @128 | 82.9k | 16.33 | **13.00** |
  2083	
  2084	* 13.0 GC/s at 128 GPUs = **2.7x the previous ocean best** (4.75 GC/s,
  2085	  LL1152@64). Strong 96->128: speedup 1.117 for 1.333x devices =
  2086	  **eff 0.84** — a healthy-tile strong leg on the ocean lane.
  2087	* This closes the user directive's ocean-hundreds gap: both lat-lon
  2088	  lanes (atm + ocean) now hold receipts at 96-128 GPUs, MPAS at 128.
  2089	
  2090	### Merge-port of #1362 (geometry_consistency) — shared-module fixes + one tracked follow-up
  2091	
  2092	The three multicontroller fixes now live in
  2093	`legoesm.parallel.geometry_consistency` (checked_shard_put /
  2094	addressable_shard_put / assert_pytree_bytes_equal / band_fingerprint —
  2095	the ONE implementation, #1362 doctrine); the ocean lane calls them, and
  2096	#1362's entry gates gained aux coverage + numeric-scalar leaves in the
  2097	digest gate (codex r20). 167 gate/parity tests + the 2-proc repro
  2098	(15.11 ms) green post-fix.
  2099	
  2100	**TRACKED FOLLOW-UP (codex r20 item 3): the ATMOSPHERE lat-lon lane
  2101	still routes its band/tile geometry stacks through `broadcast_checked`
  2102	-> `broadcast_one_to_all` (sharded_atm_latlon_step.py:492/1586) — wall
  2103	1 preserved there** (an [n_processes, stack] psum program). The atm
  2104	128-GPU receipts predate #1362, so the current main atm lane at >=96
  2105	processes is UNVERIFIED and plausibly walled exactly as ocean was.
  2106	Port = same checked_shard_put swap + aux threading; needs its own
  2107	parity run + a 2-proc repro before any atm hundreds rerun on merged
  2108	main.
  2109	
  2110	## FINAL RECEIPTS (job 26657279): atm lat-lon at 144 GPUs — campaign records
  2111	
  2112	The @192 ask starved 20+ h (48-node block vs a 55-healthy-node pool);
  1900	def _unpack_cell_state(cell_buf, nlev):
  1901	    """Inverse of :func:`_pack_cell_state`: ``(T, p_s, phis, q_flat)``."""
  1902	    return (cell_buf[:, :nlev], cell_buf[:, nlev],
  1903	            cell_buf[:, nlev + 1], cell_buf[:, nlev + 2:])
  1904	
  1905	
  1906	def _ppermute_halo_fill(cell_pack, u_shard, halo_sl, ppermute_perms,
  1907	                        max_lc, max_le):
  1908	    """Fill (owned + halo) local buffers from owned shards via ppermute.
  1909	
  1910	    Runs INSIDE ``shard_map``.  ``cell_pack`` ``(cells_per, W)`` is the
  1911	    packed owned-cell buffer — ALL cell-centred prognostics (T | p_s |
  1912	    phis | tracers) concatenated on the trailing axis; ``u_shard``
  1913	    ``(edges_per, nlev)`` the owned-edge buffer.  Each edge-colored round
  1914	    posts ONE flat ppermute carrying BOTH entity classes for ALL packed
  1915	    fields — the SPMD mirror of route-A's batched union-neighbor exchange
  1916	    (one message per neighbor per dtype group; the compute-precision cast
  1917	    upstream guarantees a single dtype group here).
  1918	
  1919	    ``halo_sl`` is a tuple of per-round ``(send_cell_idx, recv_cell_pos,
  1920	    send_edge_idx, recv_edge_pos)`` tuples whose arrays are ALREADY
  1921	    device-local ``(1, n_round)`` shard_map arguments (``P("device")``
  1922	    specs) — per-rank LOCAL metadata; no device materializes the global
  1923	    schedule.  ``ppermute_perms`` is the static per-round permutation.
  1924	
  1925	    Returns ``(cell_local, u_local)`` of shapes ``(max_lc, W)`` /
  1926	    ``(max_le, nlev)``; ghost tail rows stay zero.
  1927	    """
  1928	    cells_per = cell_pack.shape[0]
  1929	    edges_per = u_shard.shape[0]
  1930	    # +1 garbage slot for padded scatter targets (trimmed at the end):
  1931	    # schedule rows are padded to the round's max halo count, and padding
  1932	    # entries target position max_lc / max_le.
  1933	    cell_local = jnp.pad(cell_pack, ((0, max_lc + 1 - cells_per), (0, 0)))
  1934	    u_local = jnp.pad(u_shard, ((0, max_le + 1 - edges_per), (0, 0)))
  1935	
  1936	    for r, (sc, rc, se, re) in enumerate(halo_sl):
  1937	        send_c = cell_pack[sc[0]]             # (hc_r, W)
  1938	        send_e = u_shard[se[0]]               # (he_r, nlev)
  1939	        send_c_flat = send_c.ravel()
  1940	        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
  1941	        recv_packed = jax.lax.ppermute(
  1942	            send_packed, "device", perm=ppermute_perms[r])
  1943	        split_at = send_c_flat.shape[0]       # static
  1944	        recv_c = recv_packed[:split_at].reshape(send_c.shape)
  1945	        recv_e = recv_packed[split_at:].reshape(send_e.shape)
  1946	        cell_local = cell_local.at[rc[0]].set(recv_c)
  1947	        u_local = u_local.at[re[0]].set(recv_e)
  1948	
  1949	    return cell_local[:max_lc], u_local[:max_le]
  1950	
  1951	
  1952	def make_voronoi_sharded_step(

codex
Ranked by likely payoff per day of engineering/queue time:

1. **Ocean LL2304@128 fused-halo + XLA-overlap 2×2 — CHEAP TRANSFER.**  
(a) Run A / fused / fused+overlap / A2 on the current explicit-substep+wide arm, with `LEGOESM_LATLON_SPMD_FUSED_HALO=1` and `LEGOESM_XLA_OVERLAP=1`; capture executed CP census. (b) Confirm if fused+overlap is ≤15.5 ms versus 16.33 ms with tight A/A2; refute if <2% after a stable bracket. (c) Expect 5–12%: roughly 13.7–14.8 GC/s. The old ocean null was small-scale implicit/wide; this is the unreceipted, bandwidth-limited 128-GPU production arm where atmosphere saw 15% stacked benefit. (d) Low effort/risk; parity and conservation gate required.

2. **Atmosphere LL2048 stage-local halo packing, beyond 41→29 — new code.**  
(a) Do the RK-stage liveness pass, then replace co-live per-field pad sites with existing multi-pad calls; this is application-level dependency reduction, not the closed XLA combine-threshold flag. (b) Confirm with GPU HLO CP count materially below 29 and @128 ≤4.4 ms; refute if fewer than ~5 CPs disappear or gain is <3%. (c) Ledger’s stated, still-speculative projection is 4.0–4.4 ms: 50–55 GC/s, a further 7–16%. (d) Medium effort, medium correctness risk: stage ordering and numerical parity must remain exact.

3. **Fuse scalar wall-pad exchange on the measured atmospheric 2-D CPU pencil — CHEAP TRANSFER.**  
(a) Add a `LatLon2DLayout` twin of `pad_with_pole_bc_lat_multi_mpi` and dispatch to it from `pad_with_pole_bc_lat_multi`; today that path deliberately falls back to one lat sendrecv pair per field. (b) Confirm by reducing sendrecv calls from fields to dtype groups and achieving ≤42.5 ms at r512/512 ranks from 44.78 ms; refute below 2% with A/A2 controls. (c) Estimate 2–8% (about 0.9–3.6 ms): smaller than the GPU result, but directly attacks an explicitly unoptimized path. (d) Low-to-medium effort; wall-pole-only semantics and MPI custom-VJP tests constrain it cleanly.

4. **MPAS s10 at 192 GPUs, then 224 only if capacity permits — scale-out experiment.**  
(a) Submit the existing post-`replicate_pytree` s10 deck at 192 first; its 54.6k cells/GPU remains well above the 30k floor. (b) Confirm with 128→192 strong efficiency ≥0.70, i.e. ≤17.33 ms; refute if it fails to improve on 18.20 ms. (c) Expect 15.7–18.9 GC/s, +5–26% aggregate, based on the healthy tile and the decelerating matched-tile scale-out term. (d) Near-zero engineering risk; high queue/allocation risk for 48+ nodes.

5. **MPAS profile-guided manual fusion pilot — not graphs, not overlap.**  
(a) Scope the top repeated non-NCCL ~2.7 µs kernel family with named frames, then replace one RK cell/edge update chain with a single fusion-friendly/Pallas kernel. Do not touch already-packed halo payloads. (b) Confirm only if that frame’s kernel launches fall ≥25% and s10@128 reaches ≤17.0 ms with parity/conservation intact; otherwise stop. (c) Plausible 5–15% (0.9–2.7 ms), but unbounded uncertainty because the profiler establishes the storm, not its reducible share. (d) High effort/risk; this is the only credible local response to the MPAS launch/scheduling wall.

6. **Port the ocean geometry-consistency setup fix to atmosphere, then retry LL2880@192.**  
(a) Replace the atmospheric `broadcast_checked → broadcast_one_to_all` geometry route with `checked_shard_put` plus aux threading, then run the fused+overlap LL2880 high-tile deck. (b) Confirm via the two-process parity/hash repro, absence of the `[n_processes, stack]` geometry program, and @192 ≤6.9 ms. (c) The port itself should change steady-state time by ~0%; it removes the unverified high-process setup wall and makes a 62–70 GC/s @192 receipt plausible from the 58.3 GC/s @144 result. (d) Medium effort, low numerical risk; scheduler starvation remains the practical constraint.

TOP PICK: Run the ocean @128 fused+overlap 2×2 immediately; it is the cheapest unreceipted transfer with a credible double-digit payoff.
tokens used
141,165
Ranked by likely payoff per day of engineering/queue time:

1. **Ocean LL2304@128 fused-halo + XLA-overlap 2×2 — CHEAP TRANSFER.**  
(a) Run A / fused / fused+overlap / A2 on the current explicit-substep+wide arm, with `LEGOESM_LATLON_SPMD_FUSED_HALO=1` and `LEGOESM_XLA_OVERLAP=1`; capture executed CP census. (b) Confirm if fused+overlap is ≤15.5 ms versus 16.33 ms with tight A/A2; refute if <2% after a stable bracket. (c) Expect 5–12%: roughly 13.7–14.8 GC/s. The old ocean null was small-scale implicit/wide; this is the unreceipted, bandwidth-limited 128-GPU production arm where atmosphere saw 15% stacked benefit. (d) Low effort/risk; parity and conservation gate required.

2. **Atmosphere LL2048 stage-local halo packing, beyond 41→29 — new code.**  
(a) Do the RK-stage liveness pass, then replace co-live per-field pad sites with existing multi-pad calls; this is application-level dependency reduction, not the closed XLA combine-threshold flag. (b) Confirm with GPU HLO CP count materially below 29 and @128 ≤4.4 ms; refute if fewer than ~5 CPs disappear or gain is <3%. (c) Ledger’s stated, still-speculative projection is 4.0–4.4 ms: 50–55 GC/s, a further 7–16%. (d) Medium effort, medium correctness risk: stage ordering and numerical parity must remain exact.

3. **Fuse scalar wall-pad exchange on the measured atmospheric 2-D CPU pencil — CHEAP TRANSFER.**  
(a) Add a `LatLon2DLayout` twin of `pad_with_pole_bc_lat_multi_mpi` and dispatch to it from `pad_with_pole_bc_lat_multi`; today that path deliberately falls back to one lat sendrecv pair per field. (b) Confirm by reducing sendrecv calls from fields to dtype groups and achieving ≤42.5 ms at r512/512 ranks from 44.78 ms; refute below 2% with A/A2 controls. (c) Estimate 2–8% (about 0.9–3.6 ms): smaller than the GPU result, but directly attacks an explicitly unoptimized path. (d) Low-to-medium effort; wall-pole-only semantics and MPI custom-VJP tests constrain it cleanly.

4. **MPAS s10 at 192 GPUs, then 224 only if capacity permits — scale-out experiment.**  
(a) Submit the existing post-`replicate_pytree` s10 deck at 192 first; its 54.6k cells/GPU remains well above the 30k floor. (b) Confirm with 128→192 strong efficiency ≥0.70, i.e. ≤17.33 ms; refute if it fails to improve on 18.20 ms. (c) Expect 15.7–18.9 GC/s, +5–26% aggregate, based on the healthy tile and the decelerating matched-tile scale-out term. (d) Near-zero engineering risk; high queue/allocation risk for 48+ nodes.

5. **MPAS profile-guided manual fusion pilot — not graphs, not overlap.**  
(a) Scope the top repeated non-NCCL ~2.7 µs kernel family with named frames, then replace one RK cell/edge update chain with a single fusion-friendly/Pallas kernel. Do not touch already-packed halo payloads. (b) Confirm only if that frame’s kernel launches fall ≥25% and s10@128 reaches ≤17.0 ms with parity/conservation intact; otherwise stop. (c) Plausible 5–15% (0.9–2.7 ms), but unbounded uncertainty because the profiler establishes the storm, not its reducible share. (d) High effort/risk; this is the only credible local response to the MPAS launch/scheduling wall.

6. **Port the ocean geometry-consistency setup fix to atmosphere, then retry LL2880@192.**  
(a) Replace the atmospheric `broadcast_checked → broadcast_one_to_all` geometry route with `checked_shard_put` plus aux threading, then run the fused+overlap LL2880 high-tile deck. (b) Confirm via the two-process parity/hash repro, absence of the `[n_processes, stack]` geometry program, and @192 ≤6.9 ms. (c) The port itself should change steady-state time by ~0%; it removes the unverified high-process setup wall and makes a 62–70 GC/s @192 receipt plausible from the 58.3 GC/s @144 result. (d) Medium effort, low numerical risk; scheduler starvation remains the practical constraint.

TOP PICK: Run the ocean @128 fused+overlap 2×2 immediately; it is the cheapest unreceipted transfer with a credible double-digit payoff.
