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
session id: 019fc467-12aa-7283-8a5d-1999b3bf8aff
--------
user
Round-10 confirm on the finalized MPAS bound section (tail of docs/performance/scaling/levante_campaign_2026-07-24.md). Your r9 items: (1) topology — recensused at reorder_for=128, cells/dev 40,968/40,962 exactly matching rows; new counts 24 CP/23,860,920 B (s8@nd16) and 33 CP/29,489,064 B (s9@nd64); wrong-topology values noted as retained-in-transcript. (2) bytes renamed static payload shapes. (3) recomputed: s8 comm 0.631+1.015=1.647, t_bound max(2.010,1.647)+0.026=2.036 compute-limited, ratio 3.16; s9 comm 0.868+1.255=2.123, t_bound 2.149, ratio 4.47; gap decomposition: latency +0.237, payload +0.240, net bound +0.113, <=0.48 of 3.17 = ~15% under fixed-IB model, achieved contribution unresolved. (4) anchor reworded serial proxy. (5) causal wording softened; lat-lon-only A/B scoping noted. Verify arithmetic and that no stale number remains. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll inspect the finalized MPAS-bound tail, trace each r9 value and wording, then check the arithmetic and flag any stale numbers.
exec
/usr/bin/bash -lc 'tail -n 260 docs/performance/scaling/levante_campaign_2026-07-24.md' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 635ms:
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
| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
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

* **The MPAS CP count is nd-DEPENDENT** — 9 / 21 / 30 / 33 per step at
  nd=4(s7) / 8(s6) / 16(s6) / 16(s8) — Voronoi neighbour-offset classes
  grow with parts, unlike the 1-D band's fixed 41. So each MPAS bound
  must census ITS OWN row topology (no cross-count reuse).
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
  at most ~14 % of the gap; its ACHIEVED-runtime contribution is
  unresolved (the unmeasured per-CP effective overhead could itself
  scale with count — codex r9). PLAUSIBLE residual mechanisms
  (uninstrumented): per-CP effective overhead growing with rank count
  (4 -> 16 nodes), and jitter/straggler amplification across 39
  dependency-chained syncs per step (33 at nd64); "more neighbours +
  partition-quality decay" for the byte growth is likewise PLAUSIBLE.
  The 26630576 CP-combining A/B tests the LAT-LON lane only; the MPAS
  exchange is a hand-rolled edge-coloured ppermute schedule, so an
  MPAS combining test would need its own arm (and may not be
  XLA-combinable at all).

exec
/usr/bin/bash -lc "rg -n -i -C 2 'wrong.topology|33 CP/24|39 CP/30|24 CP|23,860,920|29,489,064|2\\.036|4\\.47|0\\.48|14 %|15 %|serial.*proxy|reorder.?target|reorder.for 128' docs/performance/scaling/levante_campaign_2026-07-24.md scripts/tmp/_probe_mpas_halo_census.py scripts/cluster/scaling_levante 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 304ms:
scripts/tmp/_probe_mpas_halo_census.py-10-(REORDER_FOR defaults to ND; the timed ladder rows ran --reorder-for
scripts/tmp/_probe_mpas_halo_census.py-11-128, and the CP schedule is topology-dependent — codex r9: censusing
scripts/tmp/_probe_mpas_halo_census.py:12:reorder_target=nd is NOT the row's topology.)
scripts/tmp/_probe_mpas_halo_census.py-13-"""
scripts/tmp/_probe_mpas_halo_census.py-14-import os
--
scripts/tmp/_probe_mpas_halo_census.py-40-
scripts/tmp/_probe_mpas_halo_census.py-41-mesh, model, state, dev_config = bench.build_model_and_state(
scripts/tmp/_probe_mpas_halo_census.py:42:    sub, nlev, reorder_target=reorder_for, run_nd=nd, method="sfc",
scripts/tmp/_probe_mpas_halo_census.py-43-    lloyd_iterations=lloyd)
scripts/tmp/_probe_mpas_halo_census.py-44-step = make_voronoi_sharded_step(model, dev_config)
--
docs/performance/scaling/levante_campaign_2026-07-24.md-333-
docs/performance/scaling/levante_campaign_2026-07-24.md-334-The ico WEAK ladder from the same job is non-monotone (1.00 / 0.47 / 0.81 /
docs/performance/scaling/levante_campaign_2026-07-24.md:335:0.48 / 0.46 / 0.22 / 0.45 at np 1..64) — the per-rank problem size is not
docs/performance/scaling/levante_campaign_2026-07-24.md-336-held constant cleanly across that sweep's subdivision steps, so no weak
docs/performance/scaling/levante_campaign_2026-07-24.md-337-claim is made from it.
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1292-| explicit + wide | 71.57 / 23.28 | 59.48 / 20.39 | +14.1 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-1293-
docs/performance/scaling/levante_campaign_2026-07-24.md:1294:Every arm gains ~14-15 % from mixed mode; the decision-table ordering is
docs/performance/scaling/levante_campaign_2026-07-24.md-1295-OBSERVED UNCHANGED (single_reduce 6-7 % ahead at nd4, beyond the
docs/performance/scaling/levante_campaign_2026-07-24.md-1296-informal ~+-1pp single-run noise — but that noise figure came from a
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
docs/performance/scaling/levante_campaign_2026-07-24.md-1691-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
docs/performance/scaling/levante_campaign_2026-07-24.md:1692:   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
docs/performance/scaling/levante_campaign_2026-07-24.md-1693-   recomputed only from these.
docs/performance/scaling/levante_campaign_2026-07-24.md-1694-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1758-### s8 lloyd=0 de-confound ladder landed (job 26628076): the near-matched-tile scale-out cost persists without the Lloyd confound
docs/performance/scaling/levante_campaign_2026-07-24.md-1759-
docs/performance/scaling/levante_campaign_2026-07-24.md:1760:s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
docs/performance/scaling/levante_campaign_2026-07-24.md-1761-warmup 3 — configuration-matched to the s9 ladder (26600095); NOT fully
docs/performance/scaling/levante_campaign_2026-07-24.md-1762-reproduction-grade: the s9 np32 row's git_sha reads `unknown`, and this
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1766-
docs/performance/scaling/levante_campaign_2026-07-24.md-1767-Weak pairs (~4x cells with 4x GPUs — global ratio 3.9994 after both
docs/performance/scaling/levante_campaign_2026-07-24.md:1768:meshes pad +126 cells; tiles NEAR-matched to 0.015 %:
docs/performance/scaling/levante_campaign_2026-07-24.md-1769-81,936/81,924, 40,968/40,962, 20,484/20,481), SAME lloyd-0 family:
docs/performance/scaling/levante_campaign_2026-07-24.md-1770-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1853-## Distance-to-modeled-limit: MPAS GPU (2026-08-02 late)
docs/performance/scaling/levante_campaign_2026-07-24.md-1854-
docs/performance/scaling/levante_campaign_2026-07-24.md:1855:Codex r9 caught the first census at the WRONG topology
docs/performance/scaling/levante_campaign_2026-07-24.md:1856:(`reorder_target=nd` vs the rows' `--reorder-for 128`); numbers below
docs/performance/scaling/levante_campaign_2026-07-24.md-1857-are from the topology-correct recensus (job 26636684, cells/dev
docs/performance/scaling/levante_campaign_2026-07-24.md-1858-40,968/40,962 exactly matching the timed rows). The CP payloads are
docs/performance/scaling/levante_campaign_2026-07-24.md-1859-edge-coloured max-round-padded STATIC buffers — byte sums are payload
docs/performance/scaling/levante_campaign_2026-07-24.md:1860:shapes, not measured wire bytes. Wrong-topology values (33 CP/24.0 MB
docs/performance/scaling/levante_campaign_2026-07-24.md:1861:@s8, 39 CP/30.6 MB @s9) are retained in the r9 transcript only.
docs/performance/scaling/levante_campaign_2026-07-24.md-1862-
docs/performance/scaling/levante_campaign_2026-07-24.md-1863-Same treatment as the lat-lon lane (probe twin
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1869-  grow with parts, unlike the 1-D band's fixed 41. So each MPAS bound
docs/performance/scaling/levante_campaign_2026-07-24.md-1870-  must census ITS OWN row topology (no cross-count reuse).
docs/performance/scaling/levante_campaign_2026-07-24.md:1871:* **s8@nd16, reorder-for 128 (the real s8-lloyd0 np16 row topology)**:
docs/performance/scaling/levante_campaign_2026-07-24.md:1872:  24 CP + 1 AR per step, 23,860,920 B static payload at 40,968
docs/performance/scaling/levante_campaign_2026-07-24.md-1873-  cells/dev L26 f32.
docs/performance/scaling/levante_campaign_2026-07-24.md-1874-* **Same-tile nd=1 anchor (job 26635847)**: subdiv-6 lloyd0 GLOBAL
docs/performance/scaling/levante_campaign_2026-07-24.md-1875-  mesh (40,962 natural cells ~= the 41k tile) on one A100: **2.010 ms**
docs/performance/scaling/levante_campaign_2026-07-24.md:1876:  — the SERIAL same-size compute PROXY used by the model (codex r9: it
docs/performance/scaling/levante_campaign_2026-07-24.md-1877-  is un-reordered, lacks the sharded halo-local max_lc/max_le rows and
docs/performance/scaling/levante_campaign_2026-07-24.md-1878-  pack/scatter path, and runs the bench's serial-leg dt — model-grade,
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1881-  constants 26.3 us / 23.5 GB/s, serialized-latency comm model:
docs/performance/scaling/levante_campaign_2026-07-24.md-1882-  comm = 24 x 26.3 us + 23.86 MB / 23.5 GB/s = 0.631 + 1.015 = 1.647 ms;
docs/performance/scaling/levante_campaign_2026-07-24.md:1883:  t_bound = max(2.010, 1.647) + 0.026 = **2.036 ms** (compute-limited);
docs/performance/scaling/levante_campaign_2026-07-24.md-1884-  **measured/bound = 3.16** — the SAME ~3x regime as the lat-lon panel
docs/performance/scaling/levante_campaign_2026-07-24.md-1885-  (2.35-3.20). Both directive lanes sit ~3x above their fabric+compute
docs/performance/scaling/levante_campaign_2026-07-24.md-1886-  MODEL at healthy tiles.
docs/performance/scaling/levante_campaign_2026-07-24.md:1887:* **s9@nd64, reorder-for 128**: 33 CP + 1 AR per step, 29,489,064 B
docs/performance/scaling/levante_campaign_2026-07-24.md-1888-  static payload at 40,962 cells/dev — payload GROWS +24 % at matched
docs/performance/scaling/levante_campaign_2026-07-24.md-1889-  tile vs s8@nd16 (more neighbour rounds with parts; PLAUSIBLE
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1892-  comm = 33 x 26.3 us + 29.49 MB / 23.5 GB/s = 0.868 + 1.255 = 2.123;
docs/performance/scaling/levante_campaign_2026-07-24.md-1893-  t_bound = max(2.010, 2.123) + 0.026 = **2.149 ms**;
docs/performance/scaling/levante_campaign_2026-07-24.md:1894:  **measured/bound = 4.47**.
docs/performance/scaling/levante_campaign_2026-07-24.md-1895-* **Matched-tile gap decomposition (topology-correct census)**: the
docs/performance/scaling/levante_campaign_2026-07-24.md-1896-  s8@16 -> s9@64 measured gap is +3.17 ms; the MODELED comm growth
docs/performance/scaling/levante_campaign_2026-07-24.md-1897-  (CP 24->33 = +0.237 ms latency, payload +5.63 MB = +0.240 ms BW; net
docs/performance/scaling/levante_campaign_2026-07-24.md:1898:  bound growth +0.113 ms at the max() edge) covers <= 0.48 ms =
docs/performance/scaling/levante_campaign_2026-07-24.md:1899:  **~15 %**. The
docs/performance/scaling/levante_campaign_2026-07-24.md-1900-  UNDER THE FIXED-IB MODEL, direct count/payload growth accounts for
docs/performance/scaling/levante_campaign_2026-07-24.md:1901:  at most ~14 % of the gap; its ACHIEVED-runtime contribution is
docs/performance/scaling/levante_campaign_2026-07-24.md-1902-  unresolved (the unmeasured per-CP effective overhead could itself
docs/performance/scaling/levante_campaign_2026-07-24.md-1903-  scale with count — codex r9). PLAUSIBLE residual mechanisms
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-70-      --multicontroller --n-devices 32 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-71-      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:72:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-73-      --out "$OUTDIR/$1.jsonl"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-74-  s=$?
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-18-#   SATURATING (cost flattens past 32 devices):          ~12.5 ms
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-19-# Either outcome discriminates; protocol byte-matched to the s9 ladder
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:20:# (f32, sfc, --reorder-for 128, steps 12 / warmup 3).
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-21-set -uo pipefail
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-22-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
--
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-35-    --multicontroller --n-devices 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-36-    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:37:    --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-38-    --out "$OUTDIR/np128.jsonl" || { echo "np128 FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s10_128.sbatch-39-"$PY" -c "
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-14-# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-18-# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-48-      --multicontroller --n-devices "$NP" \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-49-      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done

exec
/usr/bin/bash -lc "rg -n -C 4 'calibrated_bound|all.reduce|all_reduce|latency|bandwidth|payload' scripts/tmp/_probe_mpas_halo_census.py src tests scripts | head -n 260" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 309ms:
scripts/tmp/_probe_mpas_halo_census.py-1-"""Virtual-device HLO census of the MPAS SPMD step (bound treatment for
scripts/tmp/_probe_mpas_halo_census.py-2-the MPAS panel, twin of _probe_latlon_halo_census.py).
scripts/tmp/_probe_mpas_halo_census.py-3-
scripts/tmp/_probe_mpas_halo_census.py:4:ONE compile; census counts and payload parse read the SAME compiled HLO
scripts/tmp/_probe_mpas_halo_census.py-5-text; result-shape parsing valid for synchronous CP; regex matches bare
scripts/tmp/_probe_mpas_halo_census.py-6-CP and -start, never -done; count-assert fails loudly on layout change.
scripts/tmp/_probe_mpas_halo_census.py-7-CAVEAT: CPU lowering — counts are census=virtual-cpu, bound is a MODEL.
scripts/tmp/_probe_mpas_halo_census.py-8-
--
scripts/tmp/_probe_mpas_halo_census.py-76-    "cells_per_dev": int(mesh.nCells) // nd,
scripts/tmp/_probe_mpas_halo_census.py-77-    "census_backend": "virtual-cpu (forced host platform)",
scripts/tmp/_probe_mpas_halo_census.py-78-    "census": census,
scripts/tmp/_probe_mpas_halo_census.py-79-    "cp_ops_with_shape": cp_ops,
scripts/tmp/_probe_mpas_halo_census.py:80:    "cp_static_payload_bytes_sum": cp_bytes,  # max-round-padded static buffers, NOT measured wire bytes
scripts/tmp/_probe_mpas_halo_census.py-81-    "cp_records_summary": {
scripts/tmp/_probe_mpas_halo_census.py-82-        "n": len(cp_records),
scripts/tmp/_probe_mpas_halo_census.py-83-        "max_bytes": max((r["bytes"] for r in cp_records), default=0),
scripts/tmp/_probe_mpas_halo_census.py-84-    },
--
scripts/bench/run_cpu_mpi_scaling.py-1017-        if mesh is None:
scripts/bench/run_cpu_mpi_scaling.py-1018-            mesh = create_voronoi_mesh(subdivision_level=resolution)  # load from cache
scripts/bench/run_cpu_mpi_scaling.py-1019-    else:
scripts/bench/run_cpu_mpi_scaling.py-1020-        mesh = create_voronoi_mesh(subdivision_level=resolution)
scripts/bench/run_cpu_mpi_scaling.py:1021:    # Mass fixer adds one global allreduce per step (267.8 us latency floor on
scripts/bench/run_cpu_mpi_scaling.py-1022-    # Ginsburg/Gloo). LEGOESM_NO_MASS_FIX=1 disables it for a scaling ABLATION
scripts/bench/run_cpu_mpi_scaling.py-1023-    # that isolates the dynamics+halo cost from the conservation allreduce
scripts/bench/run_cpu_mpi_scaling.py-1024-    # (codex MPI-improve #3). Production keeps it ON (conservation).
scripts/bench/run_cpu_mpi_scaling.py-1025-    _fix_mass = os.environ.get("LEGOESM_NO_MASS_FIX") != "1"
--
scripts/bench/run_cpu_mpi_scaling.py-1391-    SPMD) while the device count lives in ``n_gpus`` / ``device_count``.
scripts/bench/run_cpu_mpi_scaling.py-1392-    """
scripts/bench/run_cpu_mpi_scaling.py-1393-    output_dir.mkdir(parents=True, exist_ok=True)
scripts/bench/run_cpu_mpi_scaling.py-1394-    # Tag a non-default (2-D) decomposition into the filename so a 2-D-pencil
scripts/bench/run_cpu_mpi_scaling.py:1395:    # run never overwrites the band run at the same grid/res/np (the payload
scripts/bench/run_cpu_mpi_scaling.py-1396-    # also carries ``decomposition`` for the collector).
scripts/bench/run_cpu_mpi_scaling.py-1397-    _dtag = "" if result.decomposition == "band" else f"_{result.decomposition}"
scripts/bench/run_cpu_mpi_scaling.py-1398-    fname = (
scripts/bench/run_cpu_mpi_scaling.py-1399-        f"{result.grid_type}{_dtag}_{result.physics_level}_{result.mode}_"
scripts/bench/run_cpu_mpi_scaling.py-1400-        f"r{result.resolution}_n{result.n_ranks}_{result.precision}.json"
scripts/bench/run_cpu_mpi_scaling.py-1401-    )
scripts/bench/run_cpu_mpi_scaling.py-1402-    path = output_dir / fname
scripts/bench/run_cpu_mpi_scaling.py:1403:    payload = asdict(result)
scripts/bench/run_cpu_mpi_scaling.py-1404-    # Move partition metrics into the metadata block (not a top-level dup).
scripts/bench/run_cpu_mpi_scaling.py:1405:    _part_metrics = payload.pop("partition_metrics", None)
scripts/bench/run_cpu_mpi_scaling.py-1406-    # Record the actual JAX backend so downstream aggregation does not have to
scripts/bench/run_cpu_mpi_scaling.py-1407-    # infer CPU-vs-GPU from the output-dir name (codex review): cpu/gpu/tpu.
scripts/bench/run_cpu_mpi_scaling.py-1408-    try:
scripts/bench/run_cpu_mpi_scaling.py-1409-        import jax
scripts/bench/run_cpu_mpi_scaling.py:1410:        payload["backend"] = jax.default_backend()
scripts/bench/run_cpu_mpi_scaling.py-1411-    except Exception:
scripts/bench/run_cpu_mpi_scaling.py:1412:        payload["backend"] = ""
scripts/bench/run_cpu_mpi_scaling.py-1413-
scripts/bench/run_cpu_mpi_scaling.py-1414-    def _live_process_count() -> int:
scripts/bench/run_cpu_mpi_scaling.py-1415-        try:
scripts/bench/run_cpu_mpi_scaling.py-1416-            import jax
--
scripts/bench/run_cpu_mpi_scaling.py-1428-    # wrong for the route-B cube lane on PBS).
scripts/bench/run_cpu_mpi_scaling.py-1429-    _cpt = int(os.environ.get("SLURM_CPUS_PER_TASK")
scripts/bench/run_cpu_mpi_scaling.py-1430-               or os.environ.get("OMP_NUM_THREADS")
scripts/bench/run_cpu_mpi_scaling.py-1431-               or "1")
scripts/bench/run_cpu_mpi_scaling.py:1432:    payload["cpus_per_task"] = _cpt
scripts/bench/run_cpu_mpi_scaling.py:1433:    payload["n_cores"] = result.n_ranks * _cpt
scripts/bench/run_cpu_mpi_scaling.py-1434-    # Record conservation mode so a LEGOESM_NO_MASS_FIX ablation never dedups
scripts/bench/run_cpu_mpi_scaling.py-1435-    # with / is mislabeled as a production (mass-conserving) run (codex audit).
scripts/bench/run_cpu_mpi_scaling.py:1436:    payload["fix_mass"] = os.environ.get("LEGOESM_NO_MASS_FIX") != "1"
scripts/bench/run_cpu_mpi_scaling.py-1437-    # Self-describing metadata block (roadmap item 9): backend / precision knobs
scripts/bench/run_cpu_mpi_scaling.py-1438-    # / GPU-direct mode / decomposition / cells-per-rank so this row is
scripts/bench/run_cpu_mpi_scaling.py-1439-    # comparable and a host-staged or f32 run is falsifiable from the record.
scripts/bench/run_cpu_mpi_scaling.py-1440-    # cs-spmd: result.n_ranks is the DEVICE count (rewritten upstream); leave
scripts/bench/run_cpu_mpi_scaling.py-1441-    # metadata n_ranks to auto process-count.  Non-cs-spmd (mpi4jax): jax is
scripts/bench/run_cpu_mpi_scaling.py-1442-    # unaware of the MPI world, so the real MPI rank count must be passed.
scripts/bench/run_cpu_mpi_scaling.py-1443-    _md_n_ranks = None if cs_spmd else result.n_ranks
scripts/bench/run_cpu_mpi_scaling.py:1444:    payload["metadata"] = annotate_incomplete(scaling_metadata(
scripts/bench/run_cpu_mpi_scaling.py-1445-        grid=result.grid_type,
scripts/bench/run_cpu_mpi_scaling.py-1446-        component="atmosphere",
scripts/bench/run_cpu_mpi_scaling.py-1447-        resolution=result.resolution,
scripts/bench/run_cpu_mpi_scaling.py-1448-        n_levels=result.n_levels,
--
scripts/bench/run_cpu_mpi_scaling.py-1454-        # cs-spmd (route-B) keeps auto-resolution.
scripts/bench/run_cpu_mpi_scaling.py-1455-        transport=("mpi4jax" if (not cs_spmd and result.n_ranks > 1)
scripts/bench/run_cpu_mpi_scaling.py-1456-                   else None),
scripts/bench/run_cpu_mpi_scaling.py-1457-        n_gpus=(result.n_ranks
scripts/bench/run_cpu_mpi_scaling.py:1458:                if payload["backend"] in ("gpu", "cuda", "rocm") else 0),
scripts/bench/run_cpu_mpi_scaling.py-1459-        decomposition=result.decomposition,
scripts/bench/run_cpu_mpi_scaling.py-1460-        # cells_per_rank is per PROCESS (n_ranks semantics).  cs-spmd:
scripts/bench/run_cpu_mpi_scaling.py-1461-        # result.cells_per_rank is per global DEVICE (total // n_global),
scripts/bench/run_cpu_mpi_scaling.py-1462-        # while metadata n_ranks defaults to jax.process_count() — divide
--
scripts/bench/run_cpu_mpi_scaling.py-1468-        partition_metrics=_part_metrics,
scripts/bench/run_cpu_mpi_scaling.py-1469-        extra={
scripts/bench/run_cpu_mpi_scaling.py-1470-            "physics_level": result.physics_level,
scripts/bench/run_cpu_mpi_scaling.py-1471-            "mode": result.mode,
scripts/bench/run_cpu_mpi_scaling.py:1472:            "cpus_per_task": payload["cpus_per_task"],
scripts/bench/run_cpu_mpi_scaling.py:1473:            "n_cores": payload["n_cores"],
scripts/bench/run_cpu_mpi_scaling.py:1474:            "fix_mass": payload["fix_mass"],
scripts/bench/run_cpu_mpi_scaling.py-1475-            "cells_per_device": result.cells_per_rank if cs_spmd else None,
scripts/bench/run_cpu_mpi_scaling.py-1476-        },
scripts/bench/run_cpu_mpi_scaling.py-1477-    ))
scripts/bench/run_cpu_mpi_scaling.py-1478-    with open(path, "w", encoding="utf-8") as f:
scripts/bench/run_cpu_mpi_scaling.py:1479:        json.dump(payload, f, indent=2)
scripts/bench/run_cpu_mpi_scaling.py-1480-    print(f"  Result: {path}")
scripts/bench/run_cpu_mpi_scaling.py-1481-
scripts/bench/run_cpu_mpi_scaling.py-1482-
scripts/bench/run_cpu_mpi_scaling.py-1483-def print_summary(result: TimingResult) -> None:
--
scripts/bench/metadata.py-18-Consumers: ``run_levante_gpu_scaling.py``, ``run_cpu_mpi_scaling.py``,
scripts/bench/metadata.py-19-``bench_atm_latlon_spmd_scaling.py``, ``bench_ocean_latlon_spmd_scaling.py``,
scripts/bench/metadata.py-20-``bench_mpas_spmd_scaling.py``, ``bench_ocean_mpi_scaling.py``,
scripts/bench/metadata.py-21-``bench_ocean_gpu_scaling.py`` (and any future bench driver) merge
scripts/bench/metadata.py:22:``scaling_metadata(...)`` under the ``"metadata"`` key of their JSON payload.
scripts/bench/metadata.py:23:Aggregators read ``payload["metadata"]``.
scripts/bench/metadata.py-24-"""
scripts/bench/metadata.py-25-from __future__ import annotations
scripts/bench/metadata.py-26-
scripts/bench/metadata.py-27-import os
--
scripts/bench/metadata.py-159-
scripts/bench/metadata.py-160-def _op_call_re(op_name: str) -> "re.Pattern[str]":
scripts/bench/metadata.py-161-    """Op-call-form matcher for a single HLO collective ``op_name``.
scripts/bench/metadata.py-162-
scripts/bench/metadata.py:163:    ``op_name`` is the hyphen spelling (``"all-reduce"``).  Matches BOTH the
scripts/bench/metadata.py-164-    optimized-XLA hyphen and StableHLO underscore spellings, an optional async
scripts/bench/metadata.py-165-    ``-start``/``_start`` suffix, and requires the ``(`` op-call form so a
scripts/bench/metadata.py-166-    config-header ``XLA_FLAGS`` echo that merely CONTAINS the op name can never
scripts/bench/metadata.py-167-    inflate the count (same guard as :data:`_COLLECTIVE_PERMUTE_RE`)."""
--
scripts/bench/metadata.py-170-
scripts/bench/metadata.py-171-
scripts/bench/metadata.py-172-#: Every collective OP family a scaling row can run.  ``collective-permute`` is
scripts/bench/metadata.py-173-#: the band/face halo (reuse the canonical permute regex so its count stays
scripts/bench/metadata.py:174:#: bit-identical to :func:`count_collective_permutes`); ``all-reduce`` is the
scripts/bench/metadata.py-175-#: conservation fixer AND the ocean implicit-CN PCG reduction wall (~120/step —
scripts/bench/metadata.py-176-#: the #1 ocean strong-scaling bottleneck, invisible to a permute-only census);
scripts/bench/metadata.py-177-#: the rest surface any SPMD resharding an operator introduces.
scripts/bench/metadata.py-178-_COLLECTIVE_OP_RES: dict[str, "re.Pattern[str]"] = {
scripts/bench/metadata.py-179-    "collective_permute": _COLLECTIVE_PERMUTE_RE,
scripts/bench/metadata.py:180:    "all_reduce": _op_call_re("all-reduce"),
scripts/bench/metadata.py-181-    "all_gather": _op_call_re("all-gather"),
scripts/bench/metadata.py-182-    "all_to_all": _op_call_re("all-to-all"),
scripts/bench/metadata.py-183-    "reduce_scatter": _op_call_re("reduce-scatter"),
scripts/bench/metadata.py-184-}
--
scripts/bench/metadata.py-186-
scripts/bench/metadata.py-187-def count_collectives(hlo_text: str) -> dict[str, int]:
scripts/bench/metadata.py-188-    """Full per-family collective census of a lowered/compiled HLO text dump.
scripts/bench/metadata.py-189-
scripts/bench/metadata.py:190:    Superset of :func:`count_collective_permutes`: adds ``all_reduce`` (the
scripts/bench/metadata.py-191-    reduction wall that dominates ocean implicit-CN strong scaling and hides
scripts/bench/metadata.py-192-    from a permute-only count), ``all_gather``, ``all_to_all``,
scripts/bench/metadata.py-193-    ``reduce_scatter``.  Same STATIC, op-call-form discipline (see
scripts/bench/metadata.py-194-    :func:`_count_op_calls`: config-header flag names never inflate; an async
--
scripts/bench/metadata.py-622-        "mcells_per_s": mcells_per_s,
scripts/bench/metadata.py-623-    }
scripts/bench/metadata.py-624-
scripts/bench/metadata.py-625-
scripts/bench/metadata.py:626:#: Placeholder comm-fabric numbers for :func:`calibrated_bound` when the
scripts/bench/metadata.py-627-#: caller passes no measured values.  Ballpark single-node GPU-interconnect
scripts/bench/metadata.py-628-#: figures (order NVLink/PCIe), NOT measurements of THIS machine —
scripts/bench/metadata.py-629-#: MACHINE-CALIBRATED-REQUIRED: any bound built on them is emitted with
scripts/bench/metadata.py-630-#: ``bound_calibrated=False`` and must never be quoted as a hardware
scripts/bench/metadata.py-631-#: roofline.  Calibrate with a ping-pong / allreduce microbenchmark on the
scripts/bench/metadata.py:632:#: actual fabric and pass ``latency_us`` / ``bandwidth_GBs`` explicitly.
scripts/bench/metadata.py-633-DEFAULT_COMM_LATENCY_US = 25.0
scripts/bench/metadata.py-634-DEFAULT_COMM_BANDWIDTH_GBS = 10.0
scripts/bench/metadata.py-635-
scripts/bench/metadata.py-636-
--
scripts/bench/metadata.py-648-) -> dict[str, Any]:
scripts/bench/metadata.py-649-    """Per-step communication VOLUME fields for a scaling record (audit item 4).
scripts/bench/metadata.py-650-
scripts/bench/metadata.py-651-    ``halo_bytes_per_step = messages x bytes_per_message`` where the
scripts/bench/metadata.py:652:    per-message payload is either given explicitly (``bytes_per_message``)
scripts/bench/metadata.py-653-    or sized as a lat-row slab ``n_lon * nlev * dtype_bytes *
scripts/bench/metadata.py-654-    rows_per_message`` (the lat-band halo exchanges whole latitude rows;
scripts/bench/metadata.py-655-    2-D barotropic slabs pass ``nlev=1``, 3-D baroclinic slabs the real
scripts/bench/metadata.py-656-    level count).  ``halo_messages_per_step=None`` means the census is
--
scripts/bench/metadata.py-682-            if n_lon is None:
scripts/bench/metadata.py-683-                raise ValueError(
scripts/bench/metadata.py-684-                    "comm_accounting: pass bytes_per_message OR the slab "
scripts/bench/metadata.py-685-                    "dimensions (n_lon [+ nlev/dtype_bytes/rows_per_message]) "
scripts/bench/metadata.py:686:                    "— a message count without a payload size cannot yield "
scripts/bench/metadata.py-687-                    "bytes."
scripts/bench/metadata.py-688-                )
scripts/bench/metadata.py-689-            bytes_per_message = (
scripts/bench/metadata.py-690-                int(n_lon) * int(nlev) * int(dtype_bytes)
--
scripts/bench/metadata.py-750-        out["wet_cell_levels_per_device_max"] = max(per)
scripts/bench/metadata.py-751-    return out
scripts/bench/metadata.py-752-
scripts/bench/metadata.py-753-
scripts/bench/metadata.py:754:def calibrated_bound(
scripts/bench/metadata.py-755-    *,
scripts/bench/metadata.py-756-    measured_fused_step_ms: float | None = None,
scripts/bench/metadata.py-757-    single_device_fused_step_ms: float | None = None,
scripts/bench/metadata.py-758-    halo_messages_per_step: int | None = None,
scripts/bench/metadata.py-759-    halo_bytes_per_step: int | None = None,
scripts/bench/metadata.py-760-    n_reductions_per_step: int | None = None,
scripts/bench/metadata.py-761-    rank_imbalance: float | None = None,
scripts/bench/metadata.py:762:    latency_us: float | None = None,
scripts/bench/metadata.py:763:    bandwidth_GBs: float | None = None,
scripts/bench/metadata.py-764-    launch_host_ms: float = 0.0,
scripts/bench/metadata.py-765-) -> dict[str, Any]:
scripts/bench/metadata.py-766-    """Calibrated per-fused-step time MODEL (audit item 8).
scripts/bench/metadata.py-767-
scripts/bench/metadata.py-768-        T_bound = max(compute, comm) + reduction + launch_host
scripts/bench/metadata.py-769-
scripts/bench/metadata.py-770-    A heuristic roofline-STYLE model, NOT a guaranteed lower bound (codex
scripts/bench/metadata.py:771:    batch4): the comm term serializes the per-message latency sum
scripts/bench/metadata.py-772-    (overlapping/pipelined messages beat it), while a partial halo census
scripts/bench/metadata.py-773-    (e.g. barotropic-only) UNDERcounts bytes — the two biases pull in
scripts/bench/metadata.py-774-    opposite directions.  Use ``measured_over_bound`` as a consistency
scripts/bench/metadata.py-775-    diagnostic, not as proven headroom.
--
scripts/bench/metadata.py-778-
scripts/bench/metadata.py-779-    - ``compute``   = single-device ``fused_step_ms`` at the SAME
scripts/bench/metadata.py-780-      per-device size (the caller passes its nd=1 row; ``None`` -> the
scripts/bench/metadata.py-781-      bound is emitted null and flagged incomplete).
scripts/bench/metadata.py:782:    - ``comm``      = ``messages x latency + bytes / bandwidth`` — halo
scripts/bench/metadata.py-783-      traffic, modeled as overlappable with compute, hence the ``max``.
scripts/bench/metadata.py:784:    - ``reduction`` = ``n_reductions x latency`` — sequentially DEPENDENT
scripts/bench/metadata.py:785:      allreduce-type collectives (CG dot products); latency-bound at
scripts/bench/metadata.py-786-      bench scales, so bytes are neglected (small-message model).
scripts/bench/metadata.py-787-    - ``launch_host`` — per-step dispatch overhead; ~0 inside a fused
scripts/bench/metadata.py-788-      ``lax.scan`` block (amortized), so benches pass the default 0.0;
scripts/bench/metadata.py-789-      drivers stepping one-at-a-time should pass their measured
scripts/bench/metadata.py:790:      ``step_latency_ms - fused_step_ms``.
scripts/bench/metadata.py-791-
scripts/bench/metadata.py-792-    ``rank_imbalance`` is reported as a DIAGNOSTIC ingredient
scripts/bench/metadata.py-793-    (``imbalance_ms = (rank_imbalance - 1) x compute``) and deliberately
scripts/bench/metadata.py-794-    NOT added to ``T_bound``: the measured max/median ratio already
scripts/bench/metadata.py-795-    contains communication/reduction jitter, so adding it would
scripts/bench/metadata.py-796-    double-count terms already modeled (codex batch4).
scripts/bench/metadata.py-797-
scripts/bench/metadata.py:798:    ``latency_us`` / ``bandwidth_GBs`` default to the
scripts/bench/metadata.py-799-    MACHINE-CALIBRATED-REQUIRED placeholders
scripts/bench/metadata.py-800-    (:data:`DEFAULT_COMM_LATENCY_US` / :data:`DEFAULT_COMM_BANDWIDTH_GBS`);
scripts/bench/metadata.py-801-    whenever either default is used the result carries
scripts/bench/metadata.py-802-    ``bound_calibrated=False``.  Placeholders may only ever MULTIPLY ZERO
--
scripts/bench/metadata.py-806-    ``bound_incomplete_reason``).  Any missing ingredient ->
scripts/bench/metadata.py-807-    ``t_bound_ms=None`` + ``bound_incomplete_reason`` naming it — an
scripts/bench/metadata.py-808-    incomplete bound is reported as incomplete, never invented.
scripts/bench/metadata.py-809-    """
scripts/bench/metadata.py:810:    calibrated = latency_us is not None and bandwidth_GBs is not None
scripts/bench/metadata.py:811:    lat_us = (DEFAULT_COMM_LATENCY_US if latency_us is None
scripts/bench/metadata.py:812:              else float(latency_us))
scripts/bench/metadata.py:813:    bw_gbs = (DEFAULT_COMM_BANDWIDTH_GBS if bandwidth_GBs is None
scripts/bench/metadata.py:814:              else float(bandwidth_GBs))
scripts/bench/metadata.py-815-    if lat_us < 0.0:
scripts/bench/metadata.py:816:        raise ValueError(f"calibrated_bound: latency_us must be >= 0, "
scripts/bench/metadata.py-817-                         f"got {lat_us}")
scripts/bench/metadata.py-818-    if bw_gbs <= 0.0:
scripts/bench/metadata.py:819:        raise ValueError(f"calibrated_bound: bandwidth_GBs must be > 0, "
scripts/bench/metadata.py-820-                         f"got {bw_gbs}")
scripts/bench/metadata.py-821-
scripts/bench/metadata.py-822-    missing = [name for name, v in (
scripts/bench/metadata.py-823-        ("single_device_fused_step_ms", single_device_fused_step_ms),
--

exec
/usr/bin/bash -lc "sed -n '810,895p' scripts/bench/metadata.py && sed -n '1,130p' scripts/tmp/_probe_mpas_halo_census.py && git diff --check && git status --short" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 684ms:
    calibrated = latency_us is not None and bandwidth_GBs is not None
    lat_us = (DEFAULT_COMM_LATENCY_US if latency_us is None
              else float(latency_us))
    bw_gbs = (DEFAULT_COMM_BANDWIDTH_GBS if bandwidth_GBs is None
              else float(bandwidth_GBs))
    if lat_us < 0.0:
        raise ValueError(f"calibrated_bound: latency_us must be >= 0, "
                         f"got {lat_us}")
    if bw_gbs <= 0.0:
        raise ValueError(f"calibrated_bound: bandwidth_GBs must be > 0, "
                         f"got {bw_gbs}")

    missing = [name for name, v in (
        ("single_device_fused_step_ms", single_device_fused_step_ms),
        ("halo_messages_per_step", halo_messages_per_step),
        ("halo_bytes_per_step", halo_bytes_per_step),
        ("n_reductions_per_step", n_reductions_per_step),
    ) if v is None]
    # Placeholder fabric numbers + nonzero comm/reduction work would put a
    # fabricated latency/bandwidth INTO the bound: honest null instead.
    # (Zero work is fabric-independent — nd=1 rows keep their trivial
    # compute-only bound.)
    has_comm_work = any(
        v is not None and float(v) > 0.0
        for v in (halo_messages_per_step, halo_bytes_per_step,
                  n_reductions_per_step))
    if has_comm_work and not calibrated:
        if latency_us is None:
            missing.append("latency_us(placeholder with nonzero comm)")
        if bandwidth_GBs is None:
            missing.append("bandwidth_GBs(placeholder with nonzero comm)")

    compute_ms = (None if single_device_fused_step_ms is None
                  else float(single_device_fused_step_ms))
    comm_ms = None
    if halo_messages_per_step is not None and halo_bytes_per_step is not None:
        comm_ms = (float(halo_messages_per_step) * lat_us * 1e-3
                   + float(halo_bytes_per_step) / (bw_gbs * 1e9) * 1e3)
    reduction_ms = (None if n_reductions_per_step is None
                    else float(n_reductions_per_step) * lat_us * 1e-3)
    # Diagnostic ONLY — not a T_bound term (see the docstring).
    imbalance_ms = None
    if rank_imbalance is not None and compute_ms is not None:
        imbalance_ms = max(float(rank_imbalance) - 1.0, 0.0) * compute_ms

    if missing:
        t_bound_ms = None
        measured_over_bound = None
    else:
        t_bound_ms = (max(compute_ms, comm_ms) + reduction_ms
                      + float(launch_host_ms))
        measured_over_bound = (
            float(measured_fused_step_ms) / t_bound_ms
            if measured_fused_step_ms is not None and t_bound_ms > 0.0
            else None)

    return {
        "t_bound_ms": (None if t_bound_ms is None else round(t_bound_ms, 4)),
        "measured_over_bound": (None if measured_over_bound is None
                                else round(measured_over_bound, 4)),
        "bound_calibrated": bool(calibrated),
        "bound_incomplete_reason": (missing or None),
        "bound_ingredients": {
            "compute_ms": compute_ms,
            "comm_ms": (None if comm_ms is None else round(comm_ms, 6)),
            "reduction_ms": (None if reduction_ms is None
                             else round(reduction_ms, 6)),
            "imbalance_ms": (None if imbalance_ms is None
                             else round(imbalance_ms, 6)),
            "launch_host_ms": float(launch_host_ms),
            "latency_us": lat_us,
            "bandwidth_GBs": bw_gbs,
            "halo_messages_per_step": halo_messages_per_step,
            "halo_bytes_per_step": halo_bytes_per_step,
            "n_reductions_per_step": n_reductions_per_step,
            "rank_imbalance": rank_imbalance,
        },
    }


def timed_scan_blocks(
    advance,
    state,
    *,
    block_steps: int,
    n_blocks: int = 2,
"""Virtual-device HLO census of the MPAS SPMD step (bound treatment for
the MPAS panel, twin of _probe_latlon_halo_census.py).

ONE compile; census counts and payload parse read the SAME compiled HLO
text; result-shape parsing valid for synchronous CP; regex matches bare
CP and -start, never -done; count-assert fails loudly on layout change.
CAVEAT: CPU lowering — counts are census=virtual-cpu, bound is a MODEL.

Usage: python _probe_mpas_halo_census.py ND SUBDIVISION [NLEV] [LLOYD] [REORDER_FOR]
(REORDER_FOR defaults to ND; the timed ladder rows ran --reorder-for
128, and the CP schedule is topology-dependent — codex r9: censusing
reorder_target=nd is NOT the row's topology.)
"""
import os
import sys

nd = int(sys.argv[1]) if len(sys.argv) > 1 else 8
sub = int(sys.argv[2]) if len(sys.argv) > 2 else 6
nlev = int(sys.argv[3]) if len(sys.argv) > 3 else 26
lloyd = int(sys.argv[4]) if len(sys.argv) > 4 else 0
reorder_for = int(sys.argv[5]) if len(sys.argv) > 5 else nd

os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + f" --xla_force_host_platform_device_count={nd}")
os.environ["JAX_PLATFORMS"] = "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bench"))

import importlib
import json
import re

import jax

bench = importlib.import_module("bench_mpas_spmd_scaling")
from metadata import count_collectives  # noqa: E402

from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step  # noqa: E402

mesh, model, state, dev_config = bench.build_model_and_state(
    sub, nlev, reorder_target=reorder_for, run_nd=nd, method="sfc",
    lloyd_iterations=lloyd)
step = make_voronoi_sharded_step(model, dev_config)

lowered = jax.jit(lambda s: step(s, 30.0)).lower(state).compile()
txt = lowered.as_text()
census = count_collectives(txt)

DT = {"f32": 4, "f64": 8, "bf16": 2, "f16": 2, "s32": 4, "u32": 4,
      "pred": 1, "s8": 1, "u8": 1, "c64": 8, "c128": 16}
cp_bytes = 0
cp_ops = 0
cp_records = []
for m in re.finditer(r"collective-permute(?:-start)?\(", txt):
    line = txt[txt.rfind("\n", 0, m.start()) + 1:m.start()]
    sm = re.search(r"(\w+)\[([0-9,]*)\]", line)
    if not sm:
        continue
    dt, dims = sm.group(1), sm.group(2)
    if dt not in DT:
        continue
    n = 1
    for d in dims.split(","):
        if d:
            n *= int(d)
    cp_bytes += n * DT[dt]
    cp_ops += 1
    cp_records.append({"dtype": dt, "dims": dims, "bytes": n * DT[dt]})
if census is not None:
    assert cp_ops == census["collective_permute"], (cp_ops, census)
out = {
    "nd": nd, "subdivision": sub, "nlev": nlev, "lloyd": lloyd,
    "reorder_for": reorder_for,
    "n_cells_padded": int(mesh.nCells),
    "cells_per_dev": int(mesh.nCells) // nd,
    "census_backend": "virtual-cpu (forced host platform)",
    "census": census,
    "cp_ops_with_shape": cp_ops,
    "cp_static_payload_bytes_sum": cp_bytes,  # max-round-padded static buffers, NOT measured wire bytes
    "cp_records_summary": {
        "n": len(cp_records),
        "max_bytes": max((r["bytes"] for r in cp_records), default=0),
    },
}
print(json.dumps(out))
 M docs/performance/scaling/levante_campaign_2026-07-24.md
?? .physics-validator/scaling_campaign/codex_mpas_bound_r10_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_mpas_bound_r9_2026-08-02.md
?? scripts/cluster/scaling_levante/mpas_bound_base.sbatch

exec
/usr/bin/bash -lc "sed -n '1,240p' .physics-validator/scaling_campaign/codex_mpas_bound_r10_2026-08-02.md && sed -n '1,240p' .physics-validator/scaling_campaign/codex_mpas_bound_r9_2026-08-02.md && git diff -- docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 359ms:
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
session id: 019fc467-12aa-7283-8a5d-1999b3bf8aff
--------
user
Round-10 confirm on the finalized MPAS bound section (tail of docs/performance/scaling/levante_campaign_2026-07-24.md). Your r9 items: (1) topology — recensused at reorder_for=128, cells/dev 40,968/40,962 exactly matching rows; new counts 24 CP/23,860,920 B (s8@nd16) and 33 CP/29,489,064 B (s9@nd64); wrong-topology values noted as retained-in-transcript. (2) bytes renamed static payload shapes. (3) recomputed: s8 comm 0.631+1.015=1.647, t_bound max(2.010,1.647)+0.026=2.036 compute-limited, ratio 3.16; s9 comm 0.868+1.255=2.123, t_bound 2.149, ratio 4.47; gap decomposition: latency +0.237, payload +0.240, net bound +0.113, <=0.48 of 3.17 = ~15% under fixed-IB model, achieved contribution unresolved. (4) anchor reworded serial proxy. (5) causal wording softened; lat-lon-only A/B scoping noted. Verify arithmetic and that no stale number remains. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll inspect the finalized MPAS-bound tail, trace each r9 value and wording, then check the arithmetic and flag any stale numbers.
exec
/usr/bin/bash -lc 'tail -n 260 docs/performance/scaling/levante_campaign_2026-07-24.md' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 635ms:
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
| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
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

* **The MPAS CP count is nd-DEPENDENT** — 9 / 21 / 30 / 33 per step at
  nd=4(s7) / 8(s6) / 16(s6) / 16(s8) — Voronoi neighbour-offset classes
  grow with parts, unlike the 1-D band's fixed 41. So each MPAS bound
  must census ITS OWN row topology (no cross-count reuse).
* **s8@nd16, reorder-for 128 (the real s8-lloyd0 np16 row topology)**:
  24 CP + 1 AR per step, 23,860,920 B static payload at 40,968
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
session id: 019fc446-304e-76c2-845d-f60e8804ac9a
--------
user
Round-9 review: the MPAS distance-to-modeled-limit section (tail of docs/performance/scaling/levante_campaign_2026-07-24.md, from '## Distance-to-modeled-limit: MPAS GPU') + probe scripts/tmp/_probe_mpas_halo_census.py. Verify:
1. Census receipts: s6@nd8 21CP/4,576,872B (5121 cells/dev); s6@nd16 30CP; s7@nd4 9CP/5,473,392B (40,961 cells/dev); s8@nd16 33CP/24,028,536B (40,961); s9@nd64 39CP/30,561,048B (40,961). nd-dependence claim. Note s7@nd4 vs s8@nd16 byte discrepancy at the SAME cells/dev (5.47 vs 24.03 MB!) — is my bound wrong to use per-topology bytes, or does this expose something about the probe (e.g. nd4 lowers fewer offset classes)? Attack this hard.
2. Bound arithmetic: s8@np16: comm=33x26.3us+24.03MB/23.5GB/s=1.890; t_bound=max(2.010,1.890)+0.026=2.036; ratio 6.43/2.036=3.16. s9@np64: comm=1.026+1.300=2.326; t_bound=2.352; ratio 9.60/2.352=4.08. Gap decomposition: modeled growth <=0.44 of 3.17ms = ~14%, count/byte growth REFUTED as dominant; residual mechanisms labelled PLAUSIBLE.
3. Compute anchor validity: s6 global (40,962 cells) 1-GPU 2.010ms as the 41k-tile compute term for s8/s9 rows (mesh-family lloyd0 matched, padding differs, model-grade label).
4. Wording: any CONFIRMED that should be PLAUSIBLE; the 'REFUTED as dominant' claim's strength given the model is heuristic.
End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll audit the cited MPAS section and probe against the raw receipts and recompute the model, with particular focus on whether the same-cells-per-device byte mismatch invalidates the stated bound.
exec
/usr/bin/bash -lc "pwd && rg -n -C 10 \"Distance-to-modeled-limit: MPAS GPU|s6@nd8|s7@nd4|s8@nd16|s9@nd64|REFUTED as dominant|CONFIRMED|PLAUSIBLE\" docs/performance/scaling/levante_campaign_2026-07-24.md && sed -n '1,260p' scripts/tmp/_probe_mpas_halo_census.py && git status --short" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 408ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
127-for this shape — needs a GPU op-level profile (nsys / XLA op profile of
128-np4 vs np8), which is a separate instrumented project, not another timing
129-run. Per-GPU throughput across the ladder is non-monotone in tile size
130-(430 / 249 / 305 / 150 Mc/s/GPU at 327688 / 163844 / 81922 / 40961
131-cells/device), which is itself the clue to hand the profiler.
132-PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
133-throughput is 304-340 Mc/s/GPU vs 220-248 at np4.
134-
135-RESOLVED 2026-07-26 (nsys job 26479922 + HLO dump 26480096 + sqlite
136-timeline): the dip is an XLA CODEGEN pathology, localized to named
137:kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
138-protocol (L8, padded-16 mesh: 21.01/17.78/7.17 ms at np2/4/8 vs campaign
139-19.90/17.09/6.92 — ~5% profiler overhead); (2) at np4 ONLY, giant
140-serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
141-per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
142-once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
143-instances are >1 ms at ~3.3 ms, the rest are the ordinary us-scale adds)
144-— and the sqlite timeline places all three groups' big instances at the
145-17.8 ms step cadence (stddev 78 us: deterministic compute, not comm
146-wait): 3.6 + 3.6 + 3.3 ~= 10.5-10.9 ms/step = the np4 excess; (3) in the optimized
147-step HLO these are mega-fusions ON THE HALO PATH: `%loop_add_fusion =
148-f32[491520,26]` (edge-tendency add chain, 22 operands incl. an
149-input_scatter_fusion) and `%loop_add_fusion.4 = f32[163844,26]` (cell
150-array), with the shard_map halo-pack concatenates taking the same adds +
151:parameter lists as operands. PLAUSIBLE (inferred from kInput fusion
152-semantics + operand lists, not separately timed): the emitter RECOMPUTES
153-the expensive scatter+add chain inside each consumer fusion, which is why
154-the cost multiplies. WHY np4: fusion cost-model decisions depend on the
155-shard shape; at np2/np8 the mega-fusion is not built. This also explains
156-why the earlier env-knob sweep missed it — autotune/latency-hiding flags
157-do not change fusion-pass decisions. Fusion-pass flag A/B at np4 ran
158-(job 26480162): flag route CLOSED — three of four candidate fusion flags
159-no longer exist in this XLA (upstream removals), the fourth is null, and
160-the GPU plugin does not list its flags via --help.
161-
--
241-
242-- Cross-machine anchor (matched bench/config/grid/physics/precision,
243-  job 26450081): Levante single A100-80 latlon-moist r720 f32 =
244-  **418.5 Mc/s** vs Derecho single-A100 ≈370. SCOPE: this establishes NO
245-  LARGE REGRESSION, not a precise machine ranking — the two campaigns
246-  differ in machine (A100-80 SXM vs A100-40), jax/tree version and date,
247-  so the ~13% gap is not attributable to any single factor.
248-- Cube "404 vs 141 Mc/s": the 404 is the single-GPU RTX-5090 Held-Suarez
249-  row in `SCALING_SUMMARY.md` SS1; ours is gray+SBM on A100 (job 26445836).
250-  That file's own tier table prices gray+SBM ~3x Held-Suarez, so ~135 is
251:  the expected equivalent vs 141 measured. PLAUSIBLE reconciliation from
252-  two published tables, NOT a matched A/B (GPU, physics and date differ).
253-- Ocean absolutes (A100 f64 165-201 Mc/s, f32 352, job 26445836; 5090 f64
254-  152 / f32 400 from `SCALING_SUMMARY.md` SS1) are of the same order -
255-  again a cross-machine sanity check, not a controlled comparison.
256-- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
257-  to `explicit_substep`; our explicit/wide arm reproduces that class
258-  (0.88 @2, LL384) — the production implicit config was never measured
259-  there. So the gap is explained by the solver the old bench selected;
260-  labelling it 'protocol, not regression' is an inference from that
261-  config difference, not an independent bisect.
--
635-SCOPE, still: one grid, one base stratification (seeds vary only the IC
636-perturbation), UNFORCED, 600 steps, f32, nd4. This does not clear the
637-stability gate — that needs the filter analysis under stale halos and a
638-science sign-off — but "no observed failure" has become "marginally better
639-conservation with a measured variance estimate".
640-
641-Per-step time is flat 100 -> 600 steps (12.81 -> 12.74 ms). Both arms' heat
642-drift grows ~linearly and is similar between them, consistent with the
643-shared baroclinic/tracer path dominating it.
644-
645:MECHANISM CONFIRMED FROM A THIRD ANGLE. The wide-halo arm records its own
646-message census: **120 standard barotropic messages/step -> 4** (n_loop=30
647-substeps, stencil reach 3, one fixed wide exchange per chunk). A 30x cut in
648-barotropic exchanges is precisely why it wins where sync dominates, and it
649-is the SAME quantity the single_reduce analysis isolated as the half it
650-could not touch (44.1 us/iter of matvec halo). Three independent
651-measurements — the iteration sweep, the single_reduce decomposition and
652-this census — now agree on what the cost is.
653-
654-THE GATE, RE-EXAMINED AGAINST THE CODE (2026-07-26) — the campaign's
655-framing was backwards. "Wide-halo needs stability gates" conflated the two
--
1623-  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
1624-  -> step-time inference FAILS on this lane; part of metis's loss is
1625-  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
1626-  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
1627-  out partition/mapping improvements generally (e.g. wet-cell-weighted
1628-  METIS was NOT tested).
1629-* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
1630-  a byte-identical partition). NOTE the second `--distribution` field is
1631-  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
1632-  identically; the swing is socket-level. Mechanism (per-socket
1633:  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
1634-  2.13x receipt; never instrumented with bandwidth counters.
1635-* Caveats: timing-only receipt — no parity/conservation gate ran in
1636-  these arms, and the CPU nodes emit `UCX WARN transports
1637-  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
1638-  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
1639-  timing, but a "production config" claim would need a gated arm).
1640-
1641-### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
1642-
1643-f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
1644-scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
1645-L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
1646-np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
1647-reads `unknown` — same allocation, same submitted script, so the same
1648:binary is PLAUSIBLE but that row stays non-reproduction-grade on its
1649-own (codex r20/r21):
1650-
1651-| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
1652-|---|---|---|---|
1653-| 32 | 81.9k | 12.47 | 5.47 |
1654-| 64 | 41.0k | 9.60 | 7.10 |
1655-| 128 | 20.5k | 11.48 | 5.94 |
1656-
1657-* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
1658-  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
--
1738-|---|---|---|---|---|
1739-| 64 | 8,192 | 297.57 | 1.00 | 1.00 |
1740-| 128 | 4,096 | 161.03 | 1.848 | 0.92 |
1741-| 256 | 2,048 | 72.06 | 4.129 | 1.03 |
1742-| 512 | 1,024 | 44.78 | 6.645 | **0.83** |
1743-
1744-Distribution verified against the masquerade trap: result rows carry
1745-`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
1746-count on this mpi4jax lane, not the world size). 128->256 is
1747-SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache
1748:transition on Milan (mechanism PLAUSIBLE, uninstrumented). End-to-end
1749-64->512 eff 0.83 at 1k cols/rank: TIMING-ONLY evidence that the lat-lon
1750-CPU lane scales into the hundreds. QUALIFIER (codex r22): the run logs
1751-an out-of-tested-range mpi4jax==0.9.0 pairing ("may fail or produce
1752-incorrect results", parallel/reductions.py runtime check) and UCX
1753-VM_UNMAP warnings — no parity/conservation gate ran, so this ladder is
1754-unvalidated timing evidence until a supported-stack rerun. (Exact pencil factorisations are not recorded in the
1755-result JSON — only `decomposition: 2d`; a follow-up could add them to
1756-the bench metadata.)
1757-
1758-### s8 lloyd=0 de-confound ladder landed (job 26628076): the near-matched-tile scale-out cost persists without the Lloyd confound
--
1778-  above 1 with the known Lloyd-family mismatch REMOVED. (This does not
1779-  prove the old confound "only" biased the size — these are
1780-  unreplicated single runs from separate allocations, one comparator
1781-  without row-level provenance; the confounded draft read
1782-  1.80/1.35/1.41 vs 1.90/1.49/1.57 here, and the production-mesh s8
1783-  np8 was 6.92 vs lloyd-0 6.58, -4.9 %, so the mesh family does shift
1784-  absolutes.)
1785-* Restated: at NEAR-matched per-GPU tile, ~quadrupling devices+problem
1786-  costs 1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
1787-  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
1788:  UNATTRIBUTED (PLAUSIBLE candidates unchanged: inter-node neighbour
1789-  fraction growth, collective latency vs count, sfc partition-quality
1790-  decay with parts; the metis receipt argues against pure
1791-  partition-cut explanations, on the CPU lane at least).
1792-* The non-monotone tile dependence of the ratio (largest at the
1793-  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
1794-
1795-## Distance-to-modeled-limit: atm lat-lon GPU (2026-08-02, "near theoretical limit" directive)
1796-
1797-Closed the bench's own honest-null bound gap (audit item 4) for the
1798-lat-lon lane, using only repo instruments:
--
1803-  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
1804-  the 1-D band structure check). Exact CP payload from compiled-HLO
1805-  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
1806-  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
1807-  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
1808-  lowering; GPU-side collective combining could change the executed
1809-  count (metadata.py:214) — the bound is a MODEL.
1810-* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
1811-  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
1812-  Approximation, recorded: nd=1 includes pole tiles; the bias
1813:  DIRECTION on the compute term is PLAUSIBLE-high, not proven
1814-  (matters most for the compute-dominated f64 row).
1815-* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
1816-  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
1817-
1818-| row | measured | t_bound (IB) | measured/bound |
1819-|---|---|---|---|
1820-| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
1821-| LL2048@128 f32 | 5.577 | 1.894 | **2.94** |
1822-| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
1823-
--
1827-1.894 ms. r7 also independently RERAN the census at nd=128 — 41 CP + 1
1828-AR confirmed at the target device count, not just extrapolated from
1829-8/16 — and measured the f64 census directly: 9,270,800 B at n_lon=1024,
1830-four 4-byte scalar CPs staying f32, so the x2 extrapolation was 16 B
1831-high.)
1832-
1833-* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE this MODEL** (2.35-3.20)
1834-  — the eff-0.60 strong leg is not close to the fabric+compute MODEL
1835-  (a heuristic, not a proven floor). The 2-node/8-process IB
1836-  calibration is extrapolated to a 32-node/128-process communicator. Leading
1837:  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
1838-  (launch + schedule + stream sync) well above the raw 26 us fabric
1839-  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
1840-  small-collective regime. The model itself notes the serialized-latency
1841-  vs overlap biases pull opposite ways; treat measured/bound as a
1842-  consistency diagnostic, not proven headroom.
1843-* **Lever test submitted (job 26630576)**: 4-run CP-combining A/B at
1844-  LL2048@128 (A default / B combine-8MB / C combine+pipelined-p2p /
1845-  A2 default repeat),
1846-  same-job control + trailing A2 drift bracket. Interpretation limit:
1847-  without a GPU post-pass CP census per arm, a null refutes THIS
1848-  threshold/implementation, not combinable-CP count in general. The
1849-  ocean-lane null for these flags came from a different
1850-  implicit-PCG/dependency mix — not predictive for the atm lane either
1851-  way.
1852-
1853:## Distance-to-modeled-limit: MPAS GPU (2026-08-02 late)
1854-
1855-Same treatment as the lat-lon lane (probe twin
1856-`scripts/tmp/_probe_mpas_halo_census.py`; virtual-CPU lowering of the
1857-REAL `make_voronoi_sharded_step`, one compile, count-asserted):
1858-
diff --git a/docs/performance/scaling/levante_campaign_2026-07-24.md b/docs/performance/scaling/levante_campaign_2026-07-24.md
index 760be4d84..85e0fcd53 100644
--- a/docs/performance/scaling/levante_campaign_2026-07-24.md
+++ b/docs/performance/scaling/levante_campaign_2026-07-24.md
@@ -1849,3 +1849,63 @@ high.)
   ocean-lane null for these flags came from a different
   implicit-PCG/dependency mix — not predictive for the atm lane either
   way.
+
+## Distance-to-modeled-limit: MPAS GPU (2026-08-02 late)
+
+Codex r9 caught the first census at the WRONG topology
+(`reorder_target=nd` vs the rows' `--reorder-for 128`); numbers below
+are from the topology-correct recensus (job 26636684, cells/dev
+40,968/40,962 exactly matching the timed rows). The CP payloads are
+edge-coloured max-round-padded STATIC buffers — byte sums are payload
+shapes, not measured wire bytes. Wrong-topology values (33 CP/24.0 MB
+@s8, 39 CP/30.6 MB @s9) are retained in the r9 transcript only.
+
+Same treatment as the lat-lon lane (probe twin
+`scripts/tmp/_probe_mpas_halo_census.py`; virtual-CPU lowering of the
+REAL `make_voronoi_sharded_step`, one compile, count-asserted):
+
+* **The MPAS CP count is nd-DEPENDENT** — 9 / 21 / 30 / 33 per step at
+  nd=4(s7) / 8(s6) / 16(s6) / 16(s8) — Voronoi neighbour-offset classes
+  grow with parts, unlike the 1-D band's fixed 41. So each MPAS bound
+  must census ITS OWN row topology (no cross-count reuse).
+* **s8@nd16, reorder-for 128 (the real s8-lloyd0 np16 row topology)**:
+  24 CP + 1 AR per step, 23,860,920 B static payload at 40,968
+  cells/dev L26 f32.
+* **Same-tile nd=1 anchor (job 26635847)**: subdiv-6 lloyd0 GLOBAL
+  mesh (40,962 natural cells ~= the 41k tile) on one A100: **2.010 ms**
+  — the SERIAL same-size compute PROXY used by the model (codex r9: it
+  is un-reordered, lacks the sharded halo-local max_lc/max_le rows and
+  pack/scatter path, and runs the bench's serial-leg dt — model-grade,
+  not "the compute term").
+* **Bound for s8-lloyd0@np16 (measured 6.43 ms, job 26628076)**, IB
+  constants 26.3 us / 23.5 GB/s, serialized-latency comm model:
+  comm = 24 x 26.3 us + 23.86 MB / 23.5 GB/s = 0.631 + 1.015 = 1.647 ms;
+  t_bound = max(2.010, 1.647) + 0.026 = **2.036 ms** (compute-limited);
+  **measured/bound = 3.16** — the SAME ~3x regime as the lat-lon panel
+  (2.35-3.20). Both directive lanes sit ~3x above their fabric+compute
+  MODEL at healthy tiles.
+* **s9@nd64, reorder-for 128**: 33 CP + 1 AR per step, 29,489,064 B
+  static payload at 40,962 cells/dev — payload GROWS +24 % at matched
+  tile vs s8@nd16 (more neighbour rounds with parts; PLAUSIBLE
+  partition-quality decay; recorded).
+* **Bound for s9@np64 (measured 9.60 ms, job 26600095)**:
+  comm = 33 x 26.3 us + 29.49 MB / 23.5 GB/s = 0.868 + 1.255 = 2.123;
+  t_bound = max(2.010, 2.123) + 0.026 = **2.149 ms**;
+  **measured/bound = 4.47**.
+* **Matched-tile gap decomposition (topology-correct census)**: the
+  s8@16 -> s9@64 measured gap is +3.17 ms; the MODELED comm growth
+  (CP 24->33 = +0.237 ms latency, payload +5.63 MB = +0.240 ms BW; net
+  bound growth +0.113 ms at the max() edge) covers <= 0.48 ms =
+  **~15 %**. The
+  UNDER THE FIXED-IB MODEL, direct count/payload growth accounts for
+  at most ~14 % of the gap; its ACHIEVED-runtime contribution is
+  unresolved (the unmeasured per-CP effective overhead could itself
+  scale with count — codex r9). PLAUSIBLE residual mechanisms
+  (uninstrumented): per-CP effective overhead growing with rank count
+  (4 -> 16 nodes), and jitter/straggler amplification across 39
+  dependency-chained syncs per step (33 at nd64); "more neighbours +
+  partition-quality decay" for the byte growth is likewise PLAUSIBLE.
+  The 26630576 CP-combining A/B tests the LAT-LON lane only; the MPAS
+  exchange is a hand-rolled edge-coloured ppermute schedule, so an
+  MPAS combining test would need its own arm (and may not be
+  XLA-combinable at all).

exec
/usr/bin/bash -lc "rg -n -i -C 2 '26636684|23860920|29489064|24028536|30561048|\"collective_permute\"|\"all_reduce\"' . --glob '"'!docs/performance/scaling/levante_campaign_2026-07-24.md'"' --glob '"'!*.pyc'"' 2>/dev/null | head -n 360" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 5123ms:
./tests/bench/test_scaling_metadata.py-427-    ])
./tests/bench/test_scaling_metadata.py-428-    c = md.count_collectives(hlo)
./tests/bench/test_scaling_metadata.py:429:    assert c["collective_permute"] == 2
./tests/bench/test_scaling_metadata.py:430:    assert c["all_reduce"] == 2          # hyphen op + async-start (done dropped)
./tests/bench/test_scaling_metadata.py-431-    assert c["all_gather"] == 1
./tests/bench/test_scaling_metadata.py-432-    assert c["all_to_all"] == 1
--
./tests/bench/test_scaling_metadata.py-434-    assert c["total"] == 7
./tests/bench/test_scaling_metadata.py-435-    # permute family stays bit-identical to the canonical scalar helper
./tests/bench/test_scaling_metadata.py:436:    assert c["collective_permute"] == md.count_collective_permutes(hlo)
./tests/bench/test_scaling_metadata.py-437-
./tests/bench/test_scaling_metadata.py-438-    empty = md.count_collectives("no collectives here")
./tests/bench/test_scaling_metadata.py-439-    assert empty["total"] == 0
./tests/bench/test_scaling_metadata.py:440:    assert set(empty) == {"collective_permute", "all_reduce", "all_gather",
./tests/bench/test_scaling_metadata.py-441-                          "all_to_all", "reduce_scatter", "total"}
./tests/bench/test_scaling_metadata.py-442-
--
./tests/bench/test_scaling_metadata.py-454-    ])
./tests/bench/test_scaling_metadata.py-455-    c = md.count_collectives(hlo)
./tests/bench/test_scaling_metadata.py:456:    assert c["all_reduce"] == 1        # NOT dropped despite "done" in metadata
./tests/bench/test_scaling_metadata.py:457:    assert c["collective_permute"] == 1  # NOT dropped despite %done_ SSA name
./tests/bench/test_scaling_metadata.py-458-    assert c["all_gather"] == 1
./tests/bench/test_scaling_metadata.py-459-    # canonical permute helper is fixed by the same shared counter
--
./scripts/bench/bench_mpas_spmd_scaling.py-432-    # wall) is the collective_permute member, so no second compile for it.
./scripts/bench/bench_mpas_spmd_scaling.py-433-    hlo_census = hlo_collective_census(_census_fn, s)
./scripts/bench/bench_mpas_spmd_scaling.py:434:    hlo_cp = hlo_census["collective_permute"] if hlo_census else None
./scripts/bench/bench_mpas_spmd_scaling.py-435-
./scripts/bench/bench_mpas_spmd_scaling.py-436-    # --- Correctness gates (before any timing is reported) -----------------
--
./scripts/bench/metadata.py-131-    Matches the op-call form only (StableHLO underscore + optimized-XLA hyphen,
./scripts/bench/metadata.py-132-    async ``-start`` counted once, ``-done`` companion excluded by the regex),
./scripts/bench/metadata.py:133:    so config-header flag names that merely CONTAIN "collective_permute" never
./scripts/bench/metadata.py-134-    inflate the count.  Canonical for every bench that reports
./scripts/bench/metadata.py-135-    ``hlo_collective_permutes`` (cube tiled, MPAS ico) — no re-implementation."""
--
./scripts/bench/metadata.py-177-#: the rest surface any SPMD resharding an operator introduces.
./scripts/bench/metadata.py-178-_COLLECTIVE_OP_RES: dict[str, "re.Pattern[str]"] = {
./scripts/bench/metadata.py:179:    "collective_permute": _COLLECTIVE_PERMUTE_RE,
./scripts/bench/metadata.py:180:    "all_reduce": _op_call_re("all-reduce"),
./scripts/bench/metadata.py-181-    "all_gather": _op_call_re("all-gather"),
./scripts/bench/metadata.py-182-    "all_to_all": _op_call_re("all-to-all"),

exec
/usr/bin/bash -lc "awk 'BEGIN {lat=26.3; bw=23.5e9; b8=23860920; b9=29489064; c8=24*lat/1000+b8/bw*1000; c9=33*lat/1000+b9/bw*1000; t8=(2.010>c8?2.010:c8)+lat/1000; t9=(2.010>c9?2.010:c9)+lat/1000; printf(\"s8: latency %.6f, payload %.6f, comm %.6f, t %.6f, ratio %.6f\\n\",24*lat/1000,b8/bw*1000,c8,t8,6.43/t8); printf(\"s9: latency %.6f, payload %.6f, comm %.6f, t %.6f, ratio %.6f\\n\",33*lat/1000,b9/bw*1000,c9,t9,9.60/t9); printf(\"deltas: latency %.6f, payload %.6f, comm %.6f, bound %.6f, share %.6f%%; exact payload delta %d B\\n\",9*lat/1000,(b9-b8)/bw*1000,c9-c8,t9-t8,(c9-c8)/3.17*100,b9-b8) }' && sed -n '1853,1912p' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 277ms:
s8: latency 0.631200, payload 1.015358, comm 1.646558, t 2.036300, ratio 3.157688
s9: latency 0.867900, payload 1.254854, comm 2.122754, t 2.149054, ratio 4.467082
deltas: latency 0.236700, payload 0.239495, comm 0.476195, bound 0.112754, share 15.021940%; exact payload delta 5628144 B
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

* **The MPAS CP count is nd-DEPENDENT** — 9 / 21 / 30 / 33 per step at
  nd=4(s7) / 8(s6) / 16(s6) / 16(s8) — Voronoi neighbour-offset classes
  grow with parts, unlike the 1-D band's fixed 41. So each MPAS bound
  must census ITS OWN row topology (no cross-count reuse).
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
  at most ~14 % of the gap; its ACHIEVED-runtime contribution is
  unresolved (the unmeasured per-CP effective overhead could itself
  scale with count — codex r9). PLAUSIBLE residual mechanisms
  (uninstrumented): per-CP effective overhead growing with rank count
  (4 -> 16 nodes), and jitter/straggler amplification across 39
  dependency-chained syncs per step (33 at nd64); "more neighbours +
  partition-quality decay" for the byte growth is likewise PLAUSIBLE.
  The 26630576 CP-combining A/B tests the LAT-LON lane only; the MPAS
  exchange is a hand-rolled edge-coloured ppermute schedule, so an
  MPAS combining test would need its own arm (and may not be
  XLA-combinable at all).

codex
Arithmetic checks out: both bounds/ratios and the exact 15.02% modeled-comm share are correct. Static-payload wording, serial-proxy caveat, softened causality, and lat-lon-only A/B scope are also correct.

Issues:

1. [The nd-dependence bullet](docs/performance/scaling/levante_campaign_2026-07-24.md:1867) still states **33 CP at s8@nd16**, a wrong-topology stale value that contradicts the corrected **24 CP** immediately below. Remove or explicitly label old-topology historical values.

2. [“39 dependency-chained syncs”](docs/performance/scaling/levante_campaign_2026-07-24.md:1904) is stale. The corrected s9 census is 33 CP (+1 AR); use 33 CP exchanges (or 34 collectives including AR), unless separately evidenced.

3. [“~14%”](docs/performance/scaling/levante_campaign_2026-07-24.md:1901) contradicts the preceding correct `<=0.48 / 3.17 = ~15%`. Exact values: `0.476195 / 3.17 = 15.02%`; net bound change is `0.112754 ms` (3.56% of the measured gap).

VERDICT: FIX-FIRST
tokens used
45,331
Arithmetic checks out: both bounds/ratios and the exact 15.02% modeled-comm share are correct. Static-payload wording, serial-proxy caveat, softened causality, and lat-lon-only A/B scope are also correct.

Issues:

1. [The nd-dependence bullet](docs/performance/scaling/levante_campaign_2026-07-24.md:1867) still states **33 CP at s8@nd16**, a wrong-topology stale value that contradicts the corrected **24 CP** immediately below. Remove or explicitly label old-topology historical values.

2. [“39 dependency-chained syncs”](docs/performance/scaling/levante_campaign_2026-07-24.md:1904) is stale. The corrected s9 census is 33 CP (+1 AR); use 33 CP exchanges (or 34 collectives including AR), unless separately evidenced.

3. [“~14%”](docs/performance/scaling/levante_campaign_2026-07-24.md:1901) contradicts the preceding correct `<=0.48 / 3.17 = ~15%`. Exact values: `0.476195 / 3.17 = 15.02%`; net bound change is `0.112754 ms` (3.56% of the measured gap).

VERDICT: FIX-FIRST
