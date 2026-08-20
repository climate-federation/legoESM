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
session id: 019fce8a-2756-7800-9455-46acb4ac7102
--------
user
Round-29 review of the ocean-transfer verdict section just added at the tail of docs/performance/scaling/levante_campaign_2026-07-24.md. Facts (job 26692291, LL2304x4608 L20 explicit_substep+wide, 18 steps, 128 GPUs): A 16.620/12.78, B fused 15.807/13.43, C fused+overlap 15.942/13.32, A2 16.610/12.78. Census (virtual-CPU nd=8, same solver config): 206 CPs off vs 133 fused, AR 7 both. Claims: fused -4.9% (vs mean(A,A2)=16.615), overlap adds nothing (C 0.9% slower than B), 35% count cut, per-lane recommendation fused-on/overlap-off, new ocean best 13.43 GC/s. Attack: percentage arithmetic, whether one A/B justifies the 'does not transfer' claim for overlap, the barotropic-chain mechanism labelling, census-vs-GPU-scale caveat (nd=8 CPU census quoted for a 128-GPU result), GC/s consistency. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll audit the newly added verdict against the supplied measurements, check the surrounding wording and arithmetic, then give a ship/fix-first call.
exec
/usr/bin/bash -lc 'tail -n 260 docs/performance/scaling/levante_campaign_2026-07-24.md' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1155ms:
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

### #1 CONFIRMED (partly) — ocean LL2304@128: fused halo -4.9 %, overlap ADDS NOTHING (job 26692291)

The top-pick transfer of the two atm levers to the unreceipted ocean
production arm (explicit_substep + wide halo, 18 steps):

| arm | ms/step | GC/s |
|---|---|---|
| A off | 16.620 | 12.78 |
| B fused | **15.807 (-4.9 %)** | **13.43** |
| C fused+overlap | 15.942 (-4.1 %) | 13.32 |
| A2 off | 16.610 | 12.78 |

* **Fused halo transfers: -4.9 %** (A/A2 drift 0.06 % — a very tight
  bracket), matching the atm lane's -4.5/-5.4 %. Census (virtual-CPU,
  nd=8, same solver config): **206 -> 133 CPs/step**, a 35 % count cut,
  the largest of any lane so far.
* **Overlap does NOT transfer to the ocean**: C is 0.9 % SLOWER than B,
  i.e. the flags are neutral-to-mildly-negative here versus the
  -8 % they buy on the atm lane. PLAUSIBLE (uninstrumented): the ocean
  step's barotropic subcycle is a long dependent chain with little
  independent compute to hide comm under, so the latency-hiding
  scheduler has nothing to overlap and only perturbs the schedule.
* Recommendation for the ocean lat-lon lane: set
  `LEGOESM_LATLON_SPMD_FUSED_HALO=1`, leave `LEGOESM_XLA_OVERLAP`
  OFF. New ocean best: **13.43 GC/s at 128 GPUs** (was 13.0).

exec
/usr/bin/bash -lc 'git diff --check && git diff -- docs/performance/scaling/levante_campaign_2026-07-24.md && rg -n "LL2304|GC/s|SPMD_FUSED_HALO|XLA_OVERLAP|26692291|barotropic" docs/performance/scaling/levante_campaign_2026-07-24.md' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 593ms:
diff --git a/docs/performance/scaling/levante_campaign_2026-07-24.md b/docs/performance/scaling/levante_campaign_2026-07-24.md
index da33c3dfc..eae1c1950 100644
--- a/docs/performance/scaling/levante_campaign_2026-07-24.md
+++ b/docs/performance/scaling/levante_campaign_2026-07-24.md
@@ -2328,3 +2328,29 @@ slab size, different balance. Lesson for the ledger: a lever confirmed
 on one decomposition is NOT transferable by analogy — every lane needs
 its own A/B, and this one paid for itself by catching a regression
 before it shipped.
+
+### #1 CONFIRMED (partly) — ocean LL2304@128: fused halo -4.9 %, overlap ADDS NOTHING (job 26692291)
+
+The top-pick transfer of the two atm levers to the unreceipted ocean
+production arm (explicit_substep + wide halo, 18 steps):
+
+| arm | ms/step | GC/s |
+|---|---|---|
+| A off | 16.620 | 12.78 |
+| B fused | **15.807 (-4.9 %)** | **13.43** |
+| C fused+overlap | 15.942 (-4.1 %) | 13.32 |
+| A2 off | 16.610 | 12.78 |
+
+* **Fused halo transfers: -4.9 %** (A/A2 drift 0.06 % — a very tight
+  bracket), matching the atm lane's -4.5/-5.4 %. Census (virtual-CPU,
+  nd=8, same solver config): **206 -> 133 CPs/step**, a 35 % count cut,
+  the largest of any lane so far.
+* **Overlap does NOT transfer to the ocean**: C is 0.9 % SLOWER than B,
+  i.e. the flags are neutral-to-mildly-negative here versus the
+  -8 % they buy on the atm lane. PLAUSIBLE (uninstrumented): the ocean
+  step's barotropic subcycle is a long dependent chain with little
+  independent compute to hide comm under, so the latency-hiding
+  scheduler has nothing to overlap and only perturbs the schedule.
+* Recommendation for the ocean lat-lon lane: set
+  `LEGOESM_LATLON_SPMD_FUSED_HALO=1`, leave `LEGOESM_XLA_OVERLAP`
+  OFF. New ocean best: **13.43 GC/s at 128 GPUs** (was 13.0).
21:| Atm lat-lon LL720×1440 L26 | 4→8→16 A100 (1→4 nodes) | 7.72→5.40→3.54 ms/step, monotone; np16 = 7.6 GC/s (477 Mc/s/GPU sustained) | 26450848/26453240/26449147 |
22:| Atm MPAS ico L8 (28 km) L26 | 6→16 A100 | 8.66→7.08 ms/step; np16 = 2.41 GC/s — 1.6x the Derecho 16-A100 aggregate reported in `derecho_levante_sota_review_2026-07.md` SS3b (route-A, eff ~0.38 @16); CROSS-MACHINE, different stack/date - indicative, not a controlled A/B | 26453240/26449147 |
23:| **Atm cube C768/L60 (same-path cs-spmd)** | 6→24 A100 | 58.35→14.09 ms/step = **4.14× = eff 1.04 (at ideal)**, 15.1 GC/s (629 Mc/s/GPU) | 26453782 |
24:| Atm cube C384/L60 (same-path cs-spmd) | 6→24 A100 | 15.44→8.81 ms/step = 1.75× (eff 0.44), 6.0 GC/s | 26452894 |
28:| Cube tiled >6-GPU lane (its own bench) | 24 A100, C384/L60 closed loop | 9.01 ms/step, 5.9 GC/s, 18.2 SYPD — first >6-GPU production-lane receipts | 26450938/26452632 |
56:Dose-response on the implicit-CN barotropic reduction count
80:selections (`barotropic_solver="explicit_substep"` + wide-halo flags,
358:"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
415:- wide-halo HELPED MOST (-> 0.73) — it deletes the barotropic solver's sync
646:message census: **120 standard barotropic messages/step -> 4** (n_loop=30
648:barotropic exchanges is precisely why it wins where sync dominates, and it
661:   serial `tests/ocean/unit/test_barotropic_wide_halo.py` — parity atol
678:   `barotropic_solver = "explicit_substep"` is the MODEL DEFAULT
680:   (dynspg_ts-style filter, shared `barotropic_common.py` weights). NOTE
963:**GPU: blocked by the global-allocation defect (#1370)** — LL2304 wanted
1061:| devices | tile | ms/step | GC/s |
1127:| LL1152x2304 L20 | 64 | 41.5k | **11.19** (4.75 GC/s) |
1128:| LL2304x4608 L20 | 64 | 165.9k | **OOM — 102 GB/device** |
1135:LL2304 at 64 GPUs asked for **102.04 GB per device**. The per-device
1150:26510472).** Instead of retrying LL2304, LL1632 @32 holds the anchor's
1156:| LL1632x3264 | 32 | 166.5k | **19.55** (5.45 GC/s) |
1177:and 7.4x the LL2304 field size is the observed 102 GB wall. So the
1207:step invocation ran). Remaining acceptance: the LL2304@64 wall run.
1651:| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
1657:* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
1658:  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
1685:   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
1700:| arm | ms/step | GC/s (col-levels) |
1710:16 -> 32-node NCCL topology step. 39.11 GC/s (from 5.5767 ms) is the
1724:| 26628071->26636762 | oc LL2304 retry post-#1370 | 128 GPU | PREDICTION REFUTED: 26636762 failed with the byte-identical 109.5 GB args signature despite the fix being an ancestor and no fallback warning; @64 acceptance passed; see arg-linearity finding below |
1725:| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
1968:| arm | nodes | ms/step | GC/s |
1983:  ~14.7 GC/s from 96 GPUs vs 5.9 single-trajectory at 128 — replicas
1985:  to K=4 (~19.6 GC/s) is PLAUSIBLE, not measured (the 4th set failed
2028:STACKED-arguments interpretation (~858 MB per device entry at LL2304
2071:Falsification v3 in queue: oc LL2304 @96 (26646038) / @128 (26646039),
2079:| arm | cols/GPU | ms/step | GC/s |
2081:| LL2304x4608 L20 @96 | 110.6k | 18.25 | 11.63 |
2082:| LL2304x4608 L20 @128 | 82.9k | 16.33 | **13.00** |
2084:* 13.0 GC/s at 128 GPUs = **2.7x the previous ocean best** (4.75 GC/s,
2115:| arm | cols/GPU | ms/step | GC/s |
2117:| LL2304x4608 @144 | 73.7k | 5.78 | 47.76 |
2120:* **58.3 GC/s is the campaign's highest throughput** (prior record
2122:* Strong LL2304 96 -> 144: 7.872 -> 5.78 = 1.362x for 1.5x devices =
2187:  wired strictly OPT-IN (`LEGOESM_XLA_OVERLAP=1`) — never a shared
2193:* New LL2048@128 best: **5.077 ms = 43.0 GC/s**; measured/at-scale-bound
2229:cells, lloyd-0) at 128 GPUs = **18.20 ms = 14.98 GC/s — the new
2245:barotropic/baroclinic split is the design work. Estimated days-scale
2252:(`LEGOESM_LATLON_SPMD_FUSED_HALO=1`, contract: "flip per deck only with
2267:* **New LL2048@128 best: 4.745 ms = 46.0 GC/s.** The bound must be
2292:(14.98 GC/s), the DECELERATING matched-tile cost (1.90 -> 1.46 per
2300:(1) transfer fused+overlap to the unreceipted ocean LL2304@128 arm —
2332:### #1 CONFIRMED (partly) — ocean LL2304@128: fused halo -4.9 %, overlap ADDS NOTHING (job 26692291)
2337:| arm | ms/step | GC/s |
2351:  step's barotropic subcycle is a long dependent chain with little
2355:  `LEGOESM_LATLON_SPMD_FUSED_HALO=1`, leave `LEGOESM_XLA_OVERLAP`
2356:  OFF. New ocean best: **13.43 GC/s at 128 GPUs** (was 13.0).

exec
/usr/bin/bash -lc "sed -n '620,710p;340,370p;2040,2100p;1680,1720p' docs/performance/scaling/levante_campaign_2026-07-24.md && rg -n \"[0-9]+ *(->|→) *[0-9]+ (CP|CPs|collective|messages|message)|CPs?/step|CP records|CP count\" docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 391ms:

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
920:per step (the ocean's wide-halo trick, 120 -> 4 messages, applied to the
1470:   were always sound (cube C96/L40: 0/7/14 CP/step at nd 1/2/3 + 1
1848:  threshold/implementation, not combinable-CP count in general. The
1867:* **The MPAS CP count is nd- AND reorder-target-DEPENDENT** — 9 / 21 /
2214:CP records (nd=8 census): the 41 CPs are SIX classes — 13x[1,1024,26],
2253:a measured GPU A/B receipt") — census 41 -> 29 CPs/step at IDENTICAL
2346:  nd=8, same solver config): **206 -> 133 CPs/step**, a 35 % count cut,

exec
/usr/bin/bash -lc "node - <<'NODE'
const cells = 2304 * 4608 * 20;
const a = 16.620, a2 = 16.610, b = 15.807, c = 15.942;
const base = (a+a2)/2;
for (const [name, ms] of Object.entries({A:a,A2:a2,B:b,C:c})) console.log(name, 'GC/s=',cells/(ms/1000)/1e9, 'vs_mean_reduction=',(base-ms)/base*100);
console.log('baseline',base);
console.log('drift % of mean', (a-a2)/base*100);
console.log('C slower B', (c-b)/b*100);
console.log('C vs B absolute ms', c-b);
console.log('CP reduction',(206-133)/206*100);
NODE" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 232ms:
/usr/bin/bash: cannot create temp file for here-document: Read-only file system

exec
/usr/bin/bash -lc "node -e 'const cells=2304*4608*20,a=16.620,a2=16.610,b=15.807,c=15.942,base=(a+a2)/2; for (const [name,ms] of Object.entries({A:a,A2:a2,B:b,C:c})) console.log(name, \"GC/s=\", cells/(ms/1000)/1e9, \"vs_mean_reduction=\", (base-ms)/base*100); console.log(\"baseline\",base); console.log(\"drift % of mean\",(a-a2)/base*100); console.log(\"C slower B\",(c-b)/b*100); console.log(\"C vs B absolute ms\",c-b); console.log(\"CP reduction\",(206-133)/206*100);'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 330ms:
A GC/s= 12.775971119133576 vs_mean_reduction= -0.03009328919650319
A2 GC/s= 12.78366285370259 vs_mean_reduction= 0.03009328919652457
B GC/s= 13.433076485101536 vs_mean_reduction= 4.863075534155892
C GC/s= 13.319322544222807 vs_mean_reduction= 4.050556725850146
baseline 16.615000000000002
drift % of mean 0.06018657839302776
C slower B 0.8540520022774707
C vs B absolute ms 0.1349999999999998
CP reduction 35.43689320388349

codex
Arithmetic and throughput are correct: B is 4.863% faster than mean(A,A2), C is 4.051% faster, C is 0.854% slower than B, and 206→133 is a 35.437% cut. L20 cell-level throughput reproduces 12.78/13.43/13.32 GC/s.

Issues:

- “Overlap does NOT transfer” is too categorical for one B/C comparison. A/A2 establishes a stable control, but does not replicate the 0.135 ms B–C difference. Say “no benefit observed in this receipt; leave off pending a replicated B/C comparison.”
- Keep the barotropic-chain explanation explicitly a hypothesis. “Nothing to overlap” and “only perturbs the schedule” claim causal evidence that was not collected.
- Qualify the census as an `nd=8` virtual-CPU, same-config algorithmic proxy—not a 128-GPU collective trace. Also state that AR remains 7 in both arms; the measured count reduction is CP-only.
- “Largest of any lane so far” is overly broad/ambiguous given the document’s other message-count reductions; scope it to this CP census if retained.
- Label the throughput column “GC/s (cell-levels)” for auditability.

VERDICT: FIX-FIRST
tokens used
23,706
Arithmetic and throughput are correct: B is 4.863% faster than mean(A,A2), C is 4.051% faster, C is 0.854% slower than B, and 206→133 is a 35.437% cut. L20 cell-level throughput reproduces 12.78/13.43/13.32 GC/s.

Issues:

- “Overlap does NOT transfer” is too categorical for one B/C comparison. A/A2 establishes a stable control, but does not replicate the 0.135 ms B–C difference. Say “no benefit observed in this receipt; leave off pending a replicated B/C comparison.”
- Keep the barotropic-chain explanation explicitly a hypothesis. “Nothing to overlap” and “only perturbs the schedule” claim causal evidence that was not collected.
- Qualify the census as an `nd=8` virtual-CPU, same-config algorithmic proxy—not a 128-GPU collective trace. Also state that AR remains 7 in both arms; the measured count reduction is CP-only.
- “Largest of any lane so far” is overly broad/ambiguous given the document’s other message-count reductions; scope it to this CP census if retained.
- Label the throughput column “GC/s (cell-levels)” for auditability.

VERDICT: FIX-FIRST
