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
session id: 019fc236-6b9b-76b1-b8af-1eeede817a22
--------
user
Round-4 final check. Your round-3 had 2 items: (1) doc ensemble wording contradicted v3 script (counterbalanced claim, stale job id 26628021, refute-bar attribution); (2) CPU r512 ladder needed unsupported-stack qualification. Both now patched in docs/performance/scaling/levante_campaign_2026-07-24.md: 'BRACKETED, not fully counterbalanced' + follow-up attribution language at the ensemble bullet AND its REFUTE line, job table row now 26628196 (v3, supersessions noted), CPU receipt labelled TIMING-ONLY/unvalidated with the mpi4jax 0.9.0 out-of-tested-range + UCX VM_UNMAP quotes. Read the '### Next receipts' + CPU-receipt sections and confirm consistency with scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch. Anything still inconsistent? End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll cross-check the two documentation sections against the v3 ensemble submission script, focusing only on the prior discrepancies and any remaining contradictory claims.
exec
/usr/bin/bash -lc 'pwd && rg -n -C 10 "Next receipts|CPU|ensemble|26628196|26628021|TIMING-ONLY|mpi4jax|VM_UNMAP|counterbalanced|BRACKETED|REFUTE" docs/performance/scaling/levante_campaign_2026-07-24.md scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 335ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-24-#   solo_post : solo again AFTER phase B (brackets ordering/thermal
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-25-#               drift; contrast uses mean of the two solos)
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-26-#
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-27-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-28-#   numbers : 2 solo + 4 replica steady_median_ms
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-29-#   CONFIRM : max(replica) <= 1.10 x mean(solo) -> guaranteed aggregate
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-30-#             >= 4/1.10 = 3.64x the 32-GPU solo rate (~19.9 GC/s if solo
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-31-#             reproduces 5.47) = ~3.3x the observed 128-GPU
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-32-#             single-trajectory rate.  NOT "4x": 1.10 is the bar, the
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-33-#             margin below it is the measured contention.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:34:#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-35-#             per replica.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-36-# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-37-# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-38-# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-39-# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-40-# table sacct prints at the end.  Absolute ms/step is therefore only
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-41-# compared WITHIN this job (solo vs replicas), never against the
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-42-# steps-12 ladder rows.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-43-# INTERPRETATION LIMIT (codex r21 item 4): this design SCREENS for a
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-44-# co-execution penalty vs the solo brackets; if a penalty appears, its
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-45-# attribution (shared IB fabric vs node/topology placement vs drift)
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-46-# needs the per-step nodelist table + follow-up, and the solo/replica
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:47:# ordering is bracketed (pre+post), not fully counterbalanced.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-48-set -uo pipefail
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-49-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-50-export JAX_PLATFORMS=cuda,cpu
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-51-export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-52-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-53-export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-54-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-55-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-56-OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s9_ens_j${SLURM_JOB_ID}}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-57-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
--
docs/performance/scaling/levante_campaign_2026-07-24.md-5-fixing what broke, and moving the worst axis (ocean strong scaling) to a
docs/performance/scaling/levante_campaign_2026-07-24.md-6-measured 2× improvement. All receipts on the post-merge tree `d3ec1ccce`+
docs/performance/scaling/levante_campaign_2026-07-24.md-7-(campaign branch `worktree-scaling-campaign`); job IDs cited throughout are
docs/performance/scaling/levante_campaign_2026-07-24.md-8-Levante SLURM jobs from 2026-07-24. Codex adversarial review: 4 rounds
docs/performance/scaling/levante_campaign_2026-07-24.md-9-(transcripts under `.physics-validator/scaling_campaign/`); every
docs/performance/scaling/levante_campaign_2026-07-24.md-10-measurement claim below carries the round-3 corrections.
docs/performance/scaling/levante_campaign_2026-07-24.md-11-
docs/performance/scaling/levante_campaign_2026-07-24.md-12-Machines: Levante `gpu` partition (4× A100-80 SXM NVLink/node, IB HDR200),
docs/performance/scaling/levante_campaign_2026-07-24.md-13-`compute` (2× AMD Milan 7763). All GPU multinode = route-B
docs/performance/scaling/levante_campaign_2026-07-24.md-14-(`jax.distributed` + NCCL over IB verbs — `NET/IB mlx5` confirmed in-log;
docs/performance/scaling/levante_campaign_2026-07-24.md:15:route-A CUDA-aware mpi4jax not exercised on Levante).
docs/performance/scaling/levante_campaign_2026-07-24.md-16-
docs/performance/scaling/levante_campaign_2026-07-24.md-17-## Headline results (strong scaling, f32 unless noted)
docs/performance/scaling/levante_campaign_2026-07-24.md-18-
docs/performance/scaling/levante_campaign_2026-07-24.md-19-| Axis | Ladder | Result | Job(s) |
docs/performance/scaling/levante_campaign_2026-07-24.md-20-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-21-| Atm lat-lon LL720×1440 L26 | 4→8→16 A100 (1→4 nodes) | 7.72→5.40→3.54 ms/step, monotone; np16 = 7.6 GC/s (477 Mc/s/GPU sustained) | 26450848/26453240/26449147 |
docs/performance/scaling/levante_campaign_2026-07-24.md-22-| Atm MPAS ico L8 (28 km) L26 | 6→16 A100 | 8.66→7.08 ms/step; np16 = 2.41 GC/s — 1.6x the Derecho 16-A100 aggregate reported in `derecho_levante_sota_review_2026-07.md` SS3b (route-A, eff ~0.38 @16); CROSS-MACHINE, different stack/date - indicative, not a controlled A/B | 26453240/26449147 |
docs/performance/scaling/levante_campaign_2026-07-24.md-23-| **Atm cube C768/L60 (same-path cs-spmd)** | 6→24 A100 | 58.35→14.09 ms/step = **4.14× = eff 1.04 (at ideal)**, 15.1 GC/s (629 Mc/s/GPU) | 26453782 |
docs/performance/scaling/levante_campaign_2026-07-24.md-24-| Atm cube C384/L60 (same-path cs-spmd) | 6→24 A100 | 15.44→8.81 ms/step = 1.75× (eff 0.44), 6.0 GC/s | 26452894 |
docs/performance/scaling/levante_campaign_2026-07-24.md-25-| Atm cube C192/L60 (same-path) | 6→24 | 6.20→6.80 ms — ANTI-scales (eff 0.23): 9.2k cols/GPU is below the ~30k-column floor | 26452979 |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-95-26454476 + 26454618): 19.90 / 17.09 / 6.92 / 7.10 ms at np 2/4/8/16.
docs/performance/scaling/levante_campaign_2026-07-24.md-96-Taking np2 as the base (it has the BEST per-device throughput, 430
docs/performance/scaling/levante_campaign_2026-07-24.md-97-Mc/s/GPU): 2->8 = 2.88x = eff 0.72, 2->16 = 2.80x = eff 0.35 (small-tile
docs/performance/scaling/levante_campaign_2026-07-24.md-98-floor).
docs/performance/scaling/levante_campaign_2026-07-24.md-99-
docs/performance/scaling/levante_campaign_2026-07-24.md-100-OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
docs/performance/scaling/levante_campaign_2026-07-24.md-101-np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
docs/performance/scaling/levante_campaign_2026-07-24.md-102-than np2 while np8 is 2.5x faster than np4. Evidence gathered:
docs/performance/scaling/levante_campaign_2026-07-24.md-103-- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
docs/performance/scaling/levante_campaign_2026-07-24.md-104-  17.14/17.04, 6.87/6.97).
docs/performance/scaling/levante_campaign_2026-07-24.md:105:- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
docs/performance/scaling/levante_campaign_2026-07-24.md-106-  two nodes (17.12 ms), so node crossing is irrelevant.
docs/performance/scaling/levante_campaign_2026-07-24.md:107:- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
docs/performance/scaling/levante_campaign_2026-07-24.md-108-  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
docs/performance/scaling/levante_campaign_2026-07-24.md-109-  falling, and under 3 % of owned cells at every count.
docs/performance/scaling/levante_campaign_2026-07-24.md:110:- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
docs/performance/scaling/levante_campaign_2026-07-24.md-111-  device count is a compile-time property so the HLO matches what the GPUs
docs/performance/scaling/levante_campaign_2026-07-24.md-112-  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
docs/performance/scaling/levante_campaign_2026-07-24.md-113-  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
docs/performance/scaling/levante_campaign_2026-07-24.md-114-  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
docs/performance/scaling/levante_campaign_2026-07-24.md-115-rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
docs/performance/scaling/levante_campaign_2026-07-24.md-116-  bitcasts 526/582/694 — the np8 program is finer-grained.)
docs/performance/scaling/levante_campaign_2026-07-24.md:117:- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
docs/performance/scaling/levante_campaign_2026-07-24.md-118-  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
docs/performance/scaling/levante_campaign_2026-07-24.md-119-  (geometric). Every method shows the same 2.5-3.1x jump.
docs/performance/scaling/levante_campaign_2026-07-24.md:120:- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
docs/performance/scaling/levante_campaign_2026-07-24.md-121-  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
docs/performance/scaling/levante_campaign_2026-07-24.md-122-  command-buffers-off — every arm within 1 %, none recovers np4.
docs/performance/scaling/levante_campaign_2026-07-24.md-123-  (The multi-output-fusion arm errored on an unsupported flag name and is
docs/performance/scaling/levante_campaign_2026-07-24.md-124-  not counted.)
docs/performance/scaling/levante_campaign_2026-07-24.md-125-VERDICT: five hypotheses refuted by measurement (placement, halo volume,
docs/performance/scaling/levante_campaign_2026-07-24.md-126-collective count, partition method, codegen env knobs). The cheap levers known to this campaign are exhausted; the remaining suspect — per-device kernel efficiency
docs/performance/scaling/levante_campaign_2026-07-24.md-127-for this shape — needs a GPU op-level profile (nsys / XLA op profile of
docs/performance/scaling/levante_campaign_2026-07-24.md-128-np4 vs np8), which is a separate instrumented project, not another timing
docs/performance/scaling/levante_campaign_2026-07-24.md-129-run. Per-GPU throughput across the ladder is non-monotone in tile size
docs/performance/scaling/levante_campaign_2026-07-24.md-130-(430 / 249 / 305 / 150 Mc/s/GPU at 327688 / 163844 / 81922 / 40961
--
docs/performance/scaling/levante_campaign_2026-07-24.md-206-
docs/performance/scaling/levante_campaign_2026-07-24.md-207-Both ladders are monotone; the bigger tile is uniformly better at every
docs/performance/scaling/levante_campaign_2026-07-24.md-208-device count (jobs 26456334 / 26457693 / 26456337 vs 26452804-06).
docs/performance/scaling/levante_campaign_2026-07-24.md-209-
docs/performance/scaling/levante_campaign_2026-07-24.md-210-So the ocean shows the SAME tile-size dependence the cube does: the 2.01x
docs/performance/scaling/levante_campaign_2026-07-24.md-211-multinode improvement measured at LL576 was partly a floor effect, and at
docs/performance/scaling/levante_campaign_2026-07-24.md-212-a production tile the identical code scales substantially better (0.37 ->
docs/performance/scaling/levante_campaign_2026-07-24.md-213-0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
docs/performance/scaling/levante_campaign_2026-07-24.md-214-Config is byte-identical between the two rows; only the grid changes.
docs/performance/scaling/levante_campaign_2026-07-24.md-215-
docs/performance/scaling/levante_campaign_2026-07-24.md:216:REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
docs/performance/scaling/levante_campaign_2026-07-24.md-217-np4 — a LARGER per-device tile — finished in ~25 min, which looked like a
docs/performance/scaling/levante_campaign_2026-07-24.md-218-compile-time cliff at that device count. It is not: the third attempt ran
docs/performance/scaling/levante_campaign_2026-07-24.md-219-the identical configuration in **99 seconds** with a 21.6 s compile (job
docs/performance/scaling/levante_campaign_2026-07-24.md-220-26457693). The earlier hangs were transient/environmental, not
docs/performance/scaling/levante_campaign_2026-07-24.md-221-reproducible, and no compile-time defect is claimed.
docs/performance/scaling/levante_campaign_2026-07-24.md-222-
docs/performance/scaling/levante_campaign_2026-07-24.md-223-## Weak scaling at production per-device size (job 26453523)
docs/performance/scaling/levante_campaign_2026-07-24.md-224-
docs/performance/scaling/levante_campaign_2026-07-24.md-225-The earlier weak ladders used a 64-row base (0.17M cells/GPU — under the
docs/performance/scaling/levante_campaign_2026-07-24.md-226-latency floor). Re-run at PRODUCTION size (288 rows × 1152 lon × L20 =
--
docs/performance/scaling/levante_campaign_2026-07-24.md-256-- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
docs/performance/scaling/levante_campaign_2026-07-24.md-257-  to `explicit_substep`; our explicit/wide arm reproduces that class
docs/performance/scaling/levante_campaign_2026-07-24.md-258-  (0.88 @2, LL384) — the production implicit config was never measured
docs/performance/scaling/levante_campaign_2026-07-24.md-259-  there. So the gap is explained by the solver the old bench selected;
docs/performance/scaling/levante_campaign_2026-07-24.md-260-  labelling it 'protocol, not regression' is an inference from that
docs/performance/scaling/levante_campaign_2026-07-24.md-261-  config difference, not an independent bisect.
docs/performance/scaling/levante_campaign_2026-07-24.md-262-- No merge regression: nd=1 stock-CG LL192 8.90 ms pre-merge (job 26445836)
docs/performance/scaling/levante_campaign_2026-07-24.md-263-  vs 8.94 ms post-merge (smoke on tree d3ec1ccce) - one sample each, so
docs/performance/scaling/levante_campaign_2026-07-24.md-264-  this bounds a large regression only.
docs/performance/scaling/levante_campaign_2026-07-24.md-265-
docs/performance/scaling/levante_campaign_2026-07-24.md:266:## CPU-MPI (compute nodes)
docs/performance/scaling/levante_campaign_2026-07-24.md-267-
docs/performance/scaling/levante_campaign_2026-07-24.md-268-Single-node ladders (job 26445986, f64): atm latlon strong eff
docs/performance/scaling/levante_campaign_2026-07-24.md-269-0.87–0.92@np2 → 0.11–0.23@np32–64; ocean implicit 0.90@2 → 0.24@32; weak
docs/performance/scaling/levante_campaign_2026-07-24.md-270-collapses ≤0.14@32. CAVEATS: np=1 ocean leg was stock-CG (solver-mismatched
docs/performance/scaling/levante_campaign_2026-07-24.md-271-— fixed via `--force-pcg` in the job scripts; np≥2 slopes valid), and
docs/performance/scaling/levante_campaign_2026-07-24.md-272-single-node ladders conflate Milan DRAM contention with comm. The first
docs/performance/scaling/levante_campaign_2026-07-24.md-273-4-node pair collided into one OUTDIR (same-second stamp) and was discarded.
docs/performance/scaling/levante_campaign_2026-07-24.md-274-
docs/performance/scaling/levante_campaign_2026-07-24.md-275-4-node SPREAD ladder (job 26452578, f64, ranks round-robin, solver-matched):
docs/performance/scaling/levante_campaign_2026-07-24.md-276-atm latlon np2 eff ~1.00; spreading ranks over 4 nodes nearly doubles
docs/performance/scaling/levante_campaign_2026-07-24.md-277-efficiency at high rank counts (r128 np32: 0.38 spread vs 0.20 packed),
docs/performance/scaling/levante_campaign_2026-07-24.md-278-which is CONSISTENT with per-node memory-bandwidth contention in the
docs/performance/scaling/levante_campaign_2026-07-24.md-279-packed ladder (not isolated by a bandwidth counter), decaying to 0.06–0.16
docs/performance/scaling/levante_campaign_2026-07-24.md-280-at np64–128 — the 1-D band perimeter ceiling as designed (r256/np128 = 2
docs/performance/scaling/levante_campaign_2026-07-24.md-281-rows/rank). Ocean strong spread: np2 eff 1.32 (superlinear - typical of a base leg whose working set does
docs/performance/scaling/levante_campaign_2026-07-24.md-282-not fit cache; not instrumented here), 0.88@8,
docs/performance/scaling/levante_campaign_2026-07-24.md-283-0.35@32, wall at np64 (79 ms > np32's 74 ms). The job died in a high-rank
docs/performance/scaling/levante_campaign_2026-07-24.md-284-ocean case (one rank exit-3 → kill-on-bad-exit) before the weak tail —
docs/performance/scaling/levante_campaign_2026-07-24.md-285-np128 ocean + weak ladders and the rank-failure attribution remain open.
docs/performance/scaling/levante_campaign_2026-07-24.md-286-
docs/performance/scaling/levante_campaign_2026-07-24.md:287:## Atm ICOSAHEDRAL CPU-MPI (job 26452579, 1 node, f64 moist, L26)
docs/performance/scaling/levante_campaign_2026-07-24.md-288-
docs/performance/scaling/levante_campaign_2026-07-24.md-289-The last measurement gap, and the healthiest strong-scaling curve in the
docs/performance/scaling/levante_campaign_2026-07-24.md-290-campaign. Efficiency t1/(n*tn) by subdivision:
docs/performance/scaling/levante_campaign_2026-07-24.md-291-
docs/performance/scaling/levante_campaign_2026-07-24.md-292-| subdiv | np2 | np4 | np8 | np16 | np32 | np64 |
docs/performance/scaling/levante_campaign_2026-07-24.md-293-|---|---|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-294-| 4 | 0.90 | 0.76 | 1.00 | 0.53 | 0.26 | 0.15 |
docs/performance/scaling/levante_campaign_2026-07-24.md-295-| 5 | 1.02 | 0.88 | 1.15 | 0.56 | 0.58 | 0.30 |
docs/performance/scaling/levante_campaign_2026-07-24.md-296-| 6 | 1.02 | 0.88 | 1.07 | 0.51 | 0.54 | 0.57 |
docs/performance/scaling/levante_campaign_2026-07-24.md-297-| 7 | 1.03 | 0.92 | 1.02 | 0.51 | 0.51 | 0.52 |
docs/performance/scaling/levante_campaign_2026-07-24.md-298-
docs/performance/scaling/levante_campaign_2026-07-24.md-299-**REVISION 2026-07-27 — the subdiv-7 row was PLACEMENT-LIMITED, not
docs/performance/scaling/levante_campaign_2026-07-24.md-300-comm-limited.** Re-running it with `--distribution=block:cyclic` (the
docs/performance/scaling/levante_campaign_2026-07-24.md-301-Milan fix, discovered after this sweep) gives f64 np64 **efficiency 0.71,
docs/performance/scaling/levante_campaign_2026-07-24.md-302-up from 0.52**, and the high-rank columns move most: placement alone is
docs/performance/scaling/levante_campaign_2026-07-24.md-303-worth 2.00x at np16, 1.75x at np32, 1.38x at np64 (job 26495437 vs
docs/performance/scaling/levante_campaign_2026-07-24.md-304-26452579). The f32 ladder at matched placement (job 26495083) reaches
docs/performance/scaling/levante_campaign_2026-07-24.md-305-**0.88**. So the "np16 dip" visible across every row of this table is
docs/performance/scaling/levante_campaign_2026-07-24.md:306:substantially the same NUMA effect found later in the packed CPU atm
docs/performance/scaling/levante_campaign_2026-07-24.md-307-ladder — one fix, two symptoms. Precision itself is worth a near-constant
docs/performance/scaling/levante_campaign_2026-07-24.md-308-~1.4x here; the naive cross-job comparison would have read 2.81x at np16
docs/performance/scaling/levante_campaign_2026-07-24.md-309-and attributed placement to precision.
docs/performance/scaling/levante_campaign_2026-07-24.md-310-| 8 | 1.04 | 0.93 | 1.18 | 0.61 | 0.49 | — |
docs/performance/scaling/levante_campaign_2026-07-24.md-311-
docs/performance/scaling/levante_campaign_2026-07-24.md-312-THE TILE-SIZE PATTERN, THIRD LANE (cube and ico are both atmosphere:
docs/performance/scaling/levante_campaign_2026-07-24.md-313-two components, three decomposition lanes): the np64 column collapses
docs/performance/scaling/levante_campaign_2026-07-24.md-314-on coarse grids (0.15 at subdiv4, 0.30 at subdiv5) and holds on fine ones
docs/performance/scaling/levante_campaign_2026-07-24.md-315-(0.52-0.57 at subdiv6-7). Same pattern as the cube (0.23 -> 1.04) and the ocean (0.37 -> 0.63), now
docs/performance/scaling/levante_campaign_2026-07-24.md:316:on a third lane and a different transport (CPU-MPI, not NCCL). It is the
docs/performance/scaling/levante_campaign_2026-07-24.md-317-campaign's most reproducible ASSOCIATION — but changing C-resolution, LL
docs/performance/scaling/levante_campaign_2026-07-24.md-318-size or ico subdivision also changes the global problem, so tile size is
docs/performance/scaling/levante_campaign_2026-07-24.md-319-not causally isolated.
docs/performance/scaling/levante_campaign_2026-07-24.md-320-
docs/performance/scaling/levante_campaign_2026-07-24.md-321-Shape: ~1.0 through np8, one step down, then FLAT 0.5 from np16 to np64 —
docs/performance/scaling/levante_campaign_2026-07-24.md-322-4x more ranks with no further loss. CONSISTENT with a fixed per-rank cost
docs/performance/scaling/levante_campaign_2026-07-24.md-323-rather than growing communication, but flat efficiency alone does not
docs/performance/scaling/levante_campaign_2026-07-24.md-324-identify which; that needs phase-level timing. (The >1 points at np8 are the same
docs/performance/scaling/levante_campaign_2026-07-24.md-325-base-leg-working-set effect noted for the cube; read as "at ideal".)
docs/performance/scaling/levante_campaign_2026-07-24.md-326-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-345-| nd | measured | calibrated bound | measured/bound | at % of floor |
docs/performance/scaling/levante_campaign_2026-07-24.md-346-|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-347-| 2 | 42.71 ms | 34.77 ms | 1.228 | 81 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-348-| 4 | 23.53 ms | 16.82 ms | 1.398 | 72 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-349-
docs/performance/scaling/levante_campaign_2026-07-24.md-350-Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
docs/performance/scaling/levante_campaign_2026-07-24.md-351-modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
docs/performance/scaling/levante_campaign_2026-07-24.md-352-FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
docs/performance/scaling/levante_campaign_2026-07-24.md-353-ratio worsens as compute shrinks.
docs/performance/scaling/levante_campaign_2026-07-24.md-354-
docs/performance/scaling/levante_campaign_2026-07-24.md:355:WHAT THE GAP IS NOT — the omitted-traffic explanation is REFUTED
docs/performance/scaling/levante_campaign_2026-07-24.md:356:(`scripts/tmp/probe_ocean_halo_bytes.py`, HLO byte census on CPU virtual
docs/performance/scaling/levante_campaign_2026-07-24.md-357-devices). The bench's `comm_scope_note` correctly warns that its census is
docs/performance/scaling/levante_campaign_2026-07-24.md-358-"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
docs/performance/scaling/levante_campaign_2026-07-24.md-359-and the true volume IS much larger: **16.22 MB/step across 110
docs/performance/scaling/levante_campaign_2026-07-24.md-360-collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
docs/performance/scaling/levante_campaign_2026-07-24.md-361-completing the census moves the bound by only **0.22 ms**, because the
docs/performance/scaling/levante_campaign_2026-07-24.md-362-comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
docs/performance/scaling/levante_campaign_2026-07-24.md-363-is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.
docs/performance/scaling/levante_campaign_2026-07-24.md-364-
docs/performance/scaling/levante_campaign_2026-07-24.md-365-So with the byte census completed the unexplained residual is still 5.5 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-366-(nd2) and 4.3 ms (nd4).
docs/performance/scaling/levante_campaign_2026-07-24.md-367-
docs/performance/scaling/levante_campaign_2026-07-24.md:368:SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
docs/performance/scaling/levante_campaign_2026-07-24.md-369-job 26458553): the sharded formulation does NOT do measurably more work.
docs/performance/scaling/levante_campaign_2026-07-24.md-370-Timing the SHARDED step on a 1-device mesh (all the padding, band-edge and
docs/performance/scaling/levante_campaign_2026-07-24.md-371-v-row-reconstruction machinery present, ppermutes self-to-self so no real
docs/performance/scaling/levante_campaign_2026-07-24.md-372-traffic) against the UNSHARDED step at the identical tile:
docs/performance/scaling/levante_campaign_2026-07-24.md-373-
docs/performance/scaling/levante_campaign_2026-07-24.md-374-| tile | unsharded | sharded on 1 device | overhead |
docs/performance/scaling/levante_campaign_2026-07-24.md-375-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-376-| 288x1152x20 | 33.13 ms | 32.79 ms | **-0.34 ms (-1.0 %)** |
docs/performance/scaling/levante_campaign_2026-07-24.md-377-| 144x1152x20 | 15.88 ms | 15.92 ms | **+0.04 ms (+0.2 %)** |
docs/performance/scaling/levante_campaign_2026-07-24.md-378-
docs/performance/scaling/levante_campaign_2026-07-24.md-379-Zero within noise at both tiles, so the bound's compute term is the RIGHT
docs/performance/scaling/levante_campaign_2026-07-24.md-380-reference and extra sharded work is not the gap.
docs/performance/scaling/levante_campaign_2026-07-24.md-381-
docs/performance/scaling/levante_campaign_2026-07-24.md-382-WHERE THAT LEAVES IT (quantified, one candidate standing): the residual
docs/performance/scaling/levante_campaign_2026-07-24.md-383-divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
docs/performance/scaling/levante_campaign_2026-07-24.md-384-versus **17.8 us** for the same collective measured in isolation — an in-
docs/performance/scaling/levante_campaign_2026-07-24.md-385-context cost 4-5x the best case. That is consistent with EXPOSED,
docs/performance/scaling/levante_campaign_2026-07-24.md-386-un-overlapped communication rather than raw wire time.
docs/performance/scaling/levante_campaign_2026-07-24.md-387-
docs/performance/scaling/levante_campaign_2026-07-24.md:388:THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
docs/performance/scaling/levante_campaign_2026-07-24.md-389-If the residual were communication the scheduler is currently hiding work
docs/performance/scaling/levante_campaign_2026-07-24.md-390-behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:
docs/performance/scaling/levante_campaign_2026-07-24.md-391-
docs/performance/scaling/levante_campaign_2026-07-24.md-392-| arm | nd2 | nd4 | vs default |
docs/performance/scaling/levante_campaign_2026-07-24.md-393-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-394-| default (LHS on) | 42.69 | 23.53 ms | — |
docs/performance/scaling/levante_campaign_2026-07-24.md-395-| `latency_hiding_scheduler=false` | 42.44 | 23.49 | **+0.6 % / +0.2 %** |
docs/performance/scaling/levante_campaign_2026-07-24.md-396-| `enable_pipelined_p2p=true` | 42.76 | 23.54 | -0.2 % / -0.1 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-397-| CP combining @32 MiB | 42.55 | 23.59 | +0.3 % / -0.3 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-398-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-449-| nd=1 | 56.87 +- 0.04 ms | 111.4 +- 1.0 us/iter | 0.99994 |
docs/performance/scaling/levante_campaign_2026-07-24.md-450-| nd=4 | 16.11 +- 0.09 ms | 123.8 +- 2.4 us/iter | 0.99972 |
docs/performance/scaling/levante_campaign_2026-07-24.md-451-
docs/performance/scaling/levante_campaign_2026-07-24.md-452-WHAT THIS ESTABLISHES (codex objection (c), answered): the fit is
docs/performance/scaling/levante_campaign_2026-07-24.md-453-essentially exact and the intercept is tightly determined, so the cost is
docs/performance/scaling/levante_campaign_2026-07-24.md-454-genuinely PER-ITERATION, not a constant misattributed to iterations. But
docs/performance/scaling/levante_campaign_2026-07-24.md-455-the nd4 intercept EXCEEDS the measured 144-row compute term (14.63 ms) by
docs/performance/scaling/levante_campaign_2026-07-24.md-456-**1.48 ms**, so a fixed non-PCG overhead does exist and the 60 iterations
docs/performance/scaling/levante_campaign_2026-07-24.md-457-do NOT explain the entire residual.
docs/performance/scaling/levante_campaign_2026-07-24.md-458-
docs/performance/scaling/levante_campaign_2026-07-24.md:459:CODEX OBJECTION (a) TESTED AND REFUTED (job 26459817). Rather than divide
docs/performance/scaling/levante_campaign_2026-07-24.md-460-the nd1 slope by 4, measure the per-iteration slope directly at each tile
docs/performance/scaling/levante_campaign_2026-07-24.md-461-on ONE device:
docs/performance/scaling/levante_campaign_2026-07-24.md-462-
docs/performance/scaling/levante_campaign_2026-07-24.md-463-| tile | measured us/iter |
docs/performance/scaling/levante_campaign_2026-07-24.md-464-|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-465-| 576 rows | 110.0 |
docs/performance/scaling/levante_campaign_2026-07-24.md-466-| 288 rows | 65.2 |
docs/performance/scaling/levante_campaign_2026-07-24.md-467-| 144 rows | **28.1** |
docs/performance/scaling/levante_campaign_2026-07-24.md-468-
docs/performance/scaling/levante_campaign_2026-07-24.md-469-The linearity assumption predicted 110.0/4 = 27.5 us for the 144-row tile;
--
docs/performance/scaling/levante_campaign_2026-07-24.md-654-THE GATE, RE-EXAMINED AGAINST THE CODE (2026-07-26) — the campaign's
docs/performance/scaling/levante_campaign_2026-07-24.md-655-framing was backwards. "Wide-halo needs stability gates" conflated the two
docs/performance/scaling/levante_campaign_2026-07-24.md-656-things the arm changed:
docs/performance/scaling/levante_campaign_2026-07-24.md-657-
docs/performance/scaling/levante_campaign_2026-07-24.md-658-1. **wide-halo is the SAME explicit-substep operators with
docs/performance/scaling/levante_campaign_2026-07-24.md-659-   tolerance-parity coverage at every transport tier** (codex round-9
docs/performance/scaling/levante_campaign_2026-07-24.md-660-   wording), not an untested scheme variant. Coverage:
docs/performance/scaling/levante_campaign_2026-07-24.md-661-   serial `tests/ocean/unit/test_barotropic_wide_halo.py` — parity atol
docs/performance/scaling/levante_campaign_2026-07-24.md-662-   1e-12 f64 across four configs (div-damp, power-law filter, multi-chunk),
docs/performance/scaling/levante_campaign_2026-07-24.md-663-   volume-drift parity 1e-15, NaN-sentinel stencil-reach pin; re-run
docs/performance/scaling/levante_campaign_2026-07-24.md:664:   2026-07-26: 11 passed, 1 skipped (mpi4jax-gated dispatch test).
docs/performance/scaling/levante_campaign_2026-07-24.md-665-   Distributed: `tests/ocean/distributed/test_ocean_mpi_wide_halo_parity.py`
docs/performance/scaling/levante_campaign_2026-07-24.md-666-   (MPI gathered-vs-serial, 1e-10, incl. rank-cut/v-face/chunk cases) and
docs/performance/scaling/levante_campaign_2026-07-24.md-667-   `tests/parallel/test_latlon_ocean_spmd_wide_halo.py` (4-device SPMD,
docs/performance/scaling/levante_campaign_2026-07-24.md-668-   2e-4/1e-3). These are TOLERANCE parity, not bit identity — "the filter
docs/performance/scaling/levante_campaign_2026-07-24.md-669-   sees bit-identical inputs" is too strong; differences are XLA
docs/performance/scaling/levante_campaign_2026-07-24.md-670-   re-association at the serial tier and larger at the distributed tiers.
docs/performance/scaling/levante_campaign_2026-07-24.md-671-   ONE REAL BEHAVIOURAL DELTA to disclose: wide-halo requires LOCAL
docs/performance/scaling/levante_campaign_2026-07-24.md-672-   subcycle clamping, and with an active `eta_floor` the clamp/
docs/performance/scaling/levante_campaign_2026-07-24.md-673-   redistribution schedule differs from the standard path — a reviewer
docs/performance/scaling/levante_campaign_2026-07-24.md-674-   should check that config interaction, not filter stability in general.
--
docs/performance/scaling/levante_campaign_2026-07-24.md-695-program.
docs/performance/scaling/levante_campaign_2026-07-24.md-696-
docs/performance/scaling/levante_campaign_2026-07-24.md-697-## Scale-out + the plateau question (2026-07-27)
docs/performance/scaling/levante_campaign_2026-07-24.md-698-
docs/performance/scaling/levante_campaign_2026-07-24.md-699-The figure showed plateaus at high device counts on several grids. Codex
docs/performance/scaling/levante_campaign_2026-07-24.md-700-round-14 rejected the obvious "just run bigger ladders" plan — it MAPS a
docs/performance/scaling/levante_campaign_2026-07-24.md-701-plateau without IDENTIFYING it — and prescribed matched pairs instead:
docs/performance/scaling/levante_campaign_2026-07-24.md-702-at fixed device count vary the tile, and at fixed tile vary the device
docs/performance/scaling/levante_campaign_2026-07-24.md-703-count. Only the second contrast can show a genuine comm/N effect.
docs/performance/scaling/levante_campaign_2026-07-24.md-704-
docs/performance/scaling/levante_campaign_2026-07-24.md:705:**CPU-MPI lane, first verdict (job 26495929, ico f64, block:cyclic
docs/performance/scaling/levante_campaign_2026-07-24.md-706-throughout, 4 nodes, single runs).** Aligning both meshes by CELLS PER
docs/performance/scaling/levante_campaign_2026-07-24.md-707-RANK rather than rank count:
docs/performance/scaling/levante_campaign_2026-07-24.md-708-
docs/performance/scaling/levante_campaign_2026-07-24.md-709-| cells/rank | subdiv-7 | subdiv-8 |
docs/performance/scaling/levante_campaign_2026-07-24.md-710-|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-711-| 10 200 | — | np64 = 960 ms |
docs/performance/scaling/levante_campaign_2026-07-24.md-712-| 5 100 | — | np128 = 526 |
docs/performance/scaling/levante_campaign_2026-07-24.md-713-| **2 600** | **np64 = 165** | **np256 = 356** |
docs/performance/scaling/levante_campaign_2026-07-24.md-714-| 1 300 | np128 = 92 | (np512 pending) |
docs/performance/scaling/levante_campaign_2026-07-24.md-715-| 600 | np256 = 56 | — |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-744-cols/GPU anchor), C768@54 (65.5k), C1152@54 (147.5k — same tile as the
docs/performance/scaling/levante_campaign_2026-07-24.md-745-anchor at 2.25x the devices).
docs/performance/scaling/levante_campaign_2026-07-24.md-746-
docs/performance/scaling/levante_campaign_2026-07-24.md-747-The lat-lon band decomposition has no such ceiling; its matched pair
docs/performance/scaling/levante_campaign_2026-07-24.md-748-runs at 64 GPUs (job 26497323): LL720@16 and LL1440@64 both hold 64.8k
docs/performance/scaling/levante_campaign_2026-07-24.md-749-columns/GPU, with LL720@64 (16.2k) as the sub-floor control.
docs/performance/scaling/levante_campaign_2026-07-24.md-750-
docs/performance/scaling/levante_campaign_2026-07-24.md-751-**Cube tile-floor arm, measured (job 26497294):** C768 L60 from 24 to 54
docs/performance/scaling/levante_campaign_2026-07-24.md-752-GPUs = 19.33 -> 13.93 ms, **1.39x at 2.25x devices, efficiency 0.62** —
docs/performance/scaling/levante_campaign_2026-07-24.md-753-and the tile only falls to 65.5k cols/GPU, still well ABOVE the ~30k
docs/performance/scaling/levante_campaign_2026-07-24.md:754:floor. So unlike the CPU lane, the cube's loss here is NOT explained by
docs/performance/scaling/levante_campaign_2026-07-24.md-755-the tile floor alone; there is real device-count cost to quantify.
docs/performance/scaling/levante_campaign_2026-07-24.md-756-(Note the same-job C768@24 anchor reads 19.33 ms where the campaign's
docs/performance/scaling/levante_campaign_2026-07-24.md-757-figure carries 14.09 ms for C768@24 — different lane/protocol between
docs/performance/scaling/levante_campaign_2026-07-24.md-758-those jobs, so only the within-job 24-vs-54 contrast is used.)
docs/performance/scaling/levante_campaign_2026-07-24.md-759-
docs/performance/scaling/levante_campaign_2026-07-24.md-760-**THE CUBE PLATEAU, IDENTIFIED (job 26498347).** The fixed-tile contrast
docs/performance/scaling/levante_campaign_2026-07-24.md-761-finally ran inside working configs — C512@24 vs C768@54, both 65.5k
docs/performance/scaling/levante_campaign_2026-07-24.md-762-cols/GPU:
docs/performance/scaling/levante_campaign_2026-07-24.md-763-
docs/performance/scaling/levante_campaign_2026-07-24.md-764-| arm | tile | devices | ms/step |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-936-resolution at that tile count. The fixed-tile comm contrast is
docs/performance/scaling/levante_campaign_2026-07-24.md-937-resubmitted at L30 for BOTH arms (job 26497736), which halves the
docs/performance/scaling/levante_campaign_2026-07-24.md-938-working set while holding 147.5k cols/GPU on each side.
docs/performance/scaling/levante_campaign_2026-07-24.md-939-
docs/performance/scaling/levante_campaign_2026-07-24.md-940-## The resolution lever is CAPPED on both transports — and my subdiv-9 runs were invalid
docs/performance/scaling/levante_campaign_2026-07-24.md-941-
docs/performance/scaling/levante_campaign_2026-07-24.md-942-The campaign's cure for every plateau is a larger tile via higher
docs/performance/scaling/levante_campaign_2026-07-24.md-943-resolution. Testing that at the largest rank counts failed on BOTH
docs/performance/scaling/levante_campaign_2026-07-24.md-944-transports, for two DIFFERENT reasons:
docs/performance/scaling/levante_campaign_2026-07-24.md-945-
docs/performance/scaling/levante_campaign_2026-07-24.md:946:**CPU (atm icosahedral): subdiv-9 is not supported by the generator.**
docs/performance/scaling/levante_campaign_2026-07-24.md-947-Jobs 26512349 and 26514768 did not time out in mesh construction as I
docs/performance/scaling/levante_campaign_2026-07-24.md-948-assumed — they raised immediately:
docs/performance/scaling/levante_campaign_2026-07-24.md-949-
docs/performance/scaling/levante_campaign_2026-07-24.md-950-    ValueError: subdivision_level=9 would create 2.62e+06 cells.
docs/performance/scaling/levante_campaign_2026-07-24.md-951-    Maximum supported level is 8 (655,362 cells). For higher
docs/performance/scaling/levante_campaign_2026-07-24.md-952-    resolutions, use load_mpas_mesh() with a pre-built mesh file.
docs/performance/scaling/levante_campaign_2026-07-24.md-953-    (voronoi.py:1242)
docs/performance/scaling/levante_campaign_2026-07-24.md-954-
docs/performance/scaling/levante_campaign_2026-07-24.md-955-MY ERROR: I submitted two multi-node jobs at an unsupported level
docs/performance/scaling/levante_campaign_2026-07-24.md-956-without checking the generator's range, and the error message even names
--
docs/performance/scaling/levante_campaign_2026-07-24.md-959-warning and the n_lat divisibility check were the others). No pre-built
docs/performance/scaling/levante_campaign_2026-07-24.md-960-finer mesh is present in the tree, so testing beyond subdiv-8 requires
docs/performance/scaling/levante_campaign_2026-07-24.md-961-sourcing an MPAS mesh file first.
docs/performance/scaling/levante_campaign_2026-07-24.md-962-
docs/performance/scaling/levante_campaign_2026-07-24.md-963-**GPU: blocked by the global-allocation defect (#1370)** — LL2304 wanted
docs/performance/scaling/levante_campaign_2026-07-24.md-964-102 GB/device, C1152 97-106 GB.
docs/performance/scaling/levante_campaign_2026-07-24.md-965-
docs/performance/scaling/levante_campaign_2026-07-24.md-966-**CONSEQUENCE, and it is the campaign's sharpest practical finding:**
docs/performance/scaling/levante_campaign_2026-07-24.md-967-raising resolution is the ONLY measured cure for the tile-floor plateau,
docs/performance/scaling/levante_campaign_2026-07-24.md-968-and it is currently unavailable on both transports — capped at subdiv-8
docs/performance/scaling/levante_campaign_2026-07-24.md:969:on CPU by the mesh generator, and by per-device global allocation on GPU.
docs/performance/scaling/levante_campaign_2026-07-24.md-970-So the useful rank/device ceilings measured here are NOT hardware limits:
docs/performance/scaling/levante_campaign_2026-07-24.md-971-
docs/performance/scaling/levante_campaign_2026-07-24.md-972-| lane | useful ceiling | what caps it |
docs/performance/scaling/levante_campaign_2026-07-24.md-973-|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md:974:| atm ico CPU | ~512-1024 ranks at subdiv-8 | mesh generator caps at subdiv-8 |
docs/performance/scaling/levante_campaign_2026-07-24.md:975:| ocean MPAS CPU | 512 ranks at subdiv-8 | same generator cap + rank-count term |
docs/performance/scaling/levante_campaign_2026-07-24.md-976-| atm/ocean GPU | 64 GPUs at LL1536-2048 | #1370 global per-device allocation |
docs/performance/scaling/levante_campaign_2026-07-24.md-977-| cube GPU | 54 GPUs at C768 | kt validation (#1360) + #1370 |
docs/performance/scaling/levante_campaign_2026-07-24.md-978-
docs/performance/scaling/levante_campaign_2026-07-24.md-979-Unblocking #1370 and sourcing a subdiv-9+ mesh are therefore worth more
docs/performance/scaling/levante_campaign_2026-07-24.md-980-than any further tuning: both lanes have headroom that is currently
docs/performance/scaling/levante_campaign_2026-07-24.md-981-unreachable.
docs/performance/scaling/levante_campaign_2026-07-24.md-982-
docs/performance/scaling/levante_campaign_2026-07-24.md-983-## Cube optimisation: bounded BEFORE implementing — and the bound killed the plan
docs/performance/scaling/levante_campaign_2026-07-24.md-984-
docs/performance/scaling/levante_campaign_2026-07-24.md-985-Directive was to push the cube toward its limit. Codex round-16 defined
docs/performance/scaling/levante_campaign_2026-07-24.md-986-the strategy first (per the standing pre-implementation rule) and its
docs/performance/scaling/levante_campaign_2026-07-24.md:987:cheapest-bound step then REFUTED the intervention I was about to build.
docs/performance/scaling/levante_campaign_2026-07-24.md-988-
docs/performance/scaling/levante_campaign_2026-07-24.md-989-**Codex corrections to my reading:**
docs/performance/scaling/levante_campaign_2026-07-24.md-990-* My "88 SendRecv / 4 rounds = 22 exchanges" was wrong. The kt=2 tile pad
docs/performance/scaling/levante_campaign_2026-07-24.md-991-  emits **16 phases per logical scalar pad** (4 edges + 4 guards + 4
docs/performance/scaling/levante_campaign_2026-07-24.md-992-  diagonals + 4 corner slivers) for serial-exact offset/corner handling
docs/performance/scaling/levante_campaign_2026-07-24.md-993-  (`cubesphere_exchange.py:1276`); the RK body pads dp/B/zeta/invT/lnps/T
docs/performance/scaling/levante_campaign_2026-07-24.md-994-  separately, the vector pad calls the scalar pad twice, x3 RK stages.
docs/performance/scaling/levante_campaign_2026-07-24.md-995-* Vertical batching is ALREADY done (4-D pads carry all L60 levels), so
docs/performance/scaling/levante_campaign_2026-07-24.md-996-  it cannot remove launches — my own payload arithmetic had hinted at
docs/performance/scaling/levante_campaign_2026-07-24.md-997-  this (strips ~41x larger than a single-field depth-1 strip).
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1069-
docs/performance/scaling/levante_campaign_2026-07-24.md-1070-SCOPE, stated because it is tempting to misread: the regular-grid
docs/performance/scaling/levante_campaign_2026-07-24.md-1071-LL1152@16 number elsewhere in this report (19.83 ms) used
docs/performance/scaling/levante_campaign_2026-07-24.md-1072-**explicit_substep + wide-halo**, whereas this tripole ladder used
docs/performance/scaling/levante_campaign_2026-07-24.md-1073-**implicit_cn + PCG**. Those are DIFFERENT SOLVERS, so the pair licenses
docs/performance/scaling/levante_campaign_2026-07-24.md-1074-NO fold-cost claim. The only licensed fold cost remains the earlier
docs/performance/scaling/levante_campaign_2026-07-24.md-1075-same-job matched contrast (+1.2-3.7 %, job 26493837).
docs/performance/scaling/levante_campaign_2026-07-24.md-1076-
docs/performance/scaling/levante_campaign_2026-07-24.md-1077-## The MPAS mesh cap — lifted (subdiv-9 unblocked for 128 GPUs)
docs/performance/scaling/levante_campaign_2026-07-24.md-1078-
docs/performance/scaling/levante_campaign_2026-07-24.md:1079:The generator's hard subdiv-8 cap was the CPU-side resolution blocker
docs/performance/scaling/levante_campaign_2026-07-24.md-1080-and made 128-GPU MPAS floor-starved by construction (subdiv-8 at np128 =
docs/performance/scaling/levante_campaign_2026-07-24.md-1081-5.1k cells/GPU). Chain shipped 2026-07-30 (codex round-19 design,
docs/performance/scaling/levante_campaign_2026-07-24.md-1082-commit 70f3ce636):
docs/performance/scaling/levante_campaign_2026-07-24.md-1083-
docs/performance/scaling/levante_campaign_2026-07-24.md-1084-* **Cache-or-prewarm policy** for subdiv 9-10: a cache hit always loads;
docs/performance/scaling/levante_campaign_2026-07-24.md-1085-  a miss RAISES with prewarm instructions unless the process is the
docs/performance/scaling/levante_campaign_2026-07-24.md-1086-  designated single builder (opt-in env + per-key O_EXCL lockfile with
docs/performance/scaling/levante_campaign_2026-07-24.md-1087-  stale takeover — the opt-in alone would be a thundering herd across an
docs/performance/scaling/levante_campaign_2026-07-24.md-1088-  MPI launch). >10 stays hard-refused. Six policy tests + prewarm CLI
docs/performance/scaling/levante_campaign_2026-07-24.md-1089-  (`scripts/data/prewarm_voronoi_mesh.py`) with its direct test.
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1156-| LL1632x3264 | 32 | 166.5k | **19.55** (5.45 GC/s) |
docs/performance/scaling/levante_campaign_2026-07-24.md-1157-
docs/performance/scaling/levante_campaign_2026-07-24.md-1158-**2x the devices carrying 2x the problem costs NOTHING** (0.99x,
docs/performance/scaling/levante_campaign_2026-07-24.md-1159-weak-scaling efficiency 1.01). So the ocean GPU lane joins the
docs/performance/scaling/levante_campaign_2026-07-24.md-1160-atmosphere: comm does not grow with device count at constant tile
docs/performance/scaling/levante_campaign_2026-07-24.md-1161-(cube 1.06x at 2.25x, lat-lon 1.17x at 4x, ocean 0.99x at 2x). **All
docs/performance/scaling/levante_campaign_2026-07-24.md-1162-three GPU lanes are tile-limited, none is device-count-limited** over
docs/performance/scaling/levante_campaign_2026-07-24.md-1163-the tested ranges.
docs/performance/scaling/levante_campaign_2026-07-24.md-1164-
docs/performance/scaling/levante_campaign_2026-07-24.md-1165-**#1370 DIAGNOSED (probe job 26523157, after two harness failures the
docs/performance/scaling/levante_campaign_2026-07-24.md:1166:CPU smoke could not catch).** Matched 3.3M-cell shards at 2x global size
docs/performance/scaling/levante_campaign_2026-07-24.md-1167-(LL1152@16 vs LL1632@32):
docs/performance/scaling/levante_campaign_2026-07-24.md-1168-
docs/performance/scaling/levante_campaign_2026-07-24.md-1169-| signal | @16 | @32 | reading |
docs/performance/scaling/levante_campaign_2026-07-24.md-1170-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1171-| compiled entry args | 0.06 GB | 0.06 GB | step is CLEAN |
docs/performance/scaling/levante_campaign_2026-07-24.md-1172-| compiled temps | 0.22 | 0.21 | not remat pressure |
docs/performance/scaling/levante_campaign_2026-07-24.md-1173-| state leaves (per-device) | 0.070 sharded / 0 replicated | same | sharding correct |
docs/performance/scaling/levante_campaign_2026-07-24.md-1174-| **bytes_in_use** | **1.58 GB** | **3.11 GB** | **tracks GLOBAL size** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1175-
docs/performance/scaling/levante_campaign_2026-07-24.md-1176-Per-device residency is a constant **~7.4 global-field equivalents** —
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1327-  subdiv-7 discriminator, jobs 26493648/26493837) were INVALID.** The
docs/performance/scaling/levante_campaign_2026-07-24.md-1328-  bench's own metadata says it: `n_ranks: 1, cells_per_rank_achieved:
docs/performance/scaling/levante_campaign_2026-07-24.md-1329-  163842` — every arm ran ONE rank on the FULL mesh, because this bench
docs/performance/scaling/levante_campaign_2026-07-24.md-1330-  decomposes by MPI RANK (its docstring states the SPMD multi-device
docs/performance/scaling/levante_campaign_2026-07-24.md-1331-  path does not exist by design) and my CUDA_VISIBLE_DEVICES invocation
docs/performance/scaling/levante_campaign_2026-07-24.md-1332-  never created ranks. The flat curves were the SAME single-device run
docs/performance/scaling/levante_campaign_2026-07-24.md-1333-  repeated, not a latency floor and not a defect — codex round-13's
docs/performance/scaling/levante_campaign_2026-07-24.md-1334-  "hypothesis, not verdict" was righter than it knew. What survives:
docs/performance/scaling/levante_campaign_2026-07-24.md-1335-  single-device timings (s6 ~7 ms f32/f64, s7 ~21.9 ms f32).
docs/performance/scaling/levante_campaign_2026-07-24.md-1336-
docs/performance/scaling/levante_campaign_2026-07-24.md:1337:  **THE REAL LADDER (job 26494036, CPU-MPI f64, np1-16, block:cyclic,
docs/performance/scaling/levante_campaign_2026-07-24.md-1338-  ranks=N verified in metadata): MPAS-ocean SCALES.** s6 (41k cells):
docs/performance/scaling/levante_campaign_2026-07-24.md-1339-  814.46 / 316.71 / 159.27 / 92.85 / 90.51 ms; s7 (164k cells): 3944.43
docs/performance/scaling/levante_campaign_2026-07-24.md-1340-  / 1615.85 / 720.87 / 362.42 / 311.53 ms. The np1 base is
docs/performance/scaling/levante_campaign_2026-07-24.md-1341-  cache-disadvantaged (np1->2 superlinear, same pattern as the atm
docs/performance/scaling/levante_campaign_2026-07-24.md-1342-  spread ladder), so quoting np2-base efficiencies: s6 2->16 = 0.44,
docs/performance/scaling/levante_campaign_2026-07-24.md-1343-  **s7 2->16 = 0.65** — the tile-size pattern reproduces on a FOURTH
docs/performance/scaling/levante_campaign_2026-07-24.md-1344-  lane (bigger mesh holds efficiency deeper), and the np8->16 flattening
docs/performance/scaling/levante_campaign_2026-07-24.md-1345-  sits exactly where per-rank cells fall to 2.5k (s6) vs 10k (s7).
docs/performance/scaling/levante_campaign_2026-07-24.md-1346-  Single runs, no repeats; ordering claims only.
docs/performance/scaling/levante_campaign_2026-07-24.md-1347-- Remaining coverage gap, flagged not measured: atmosphere SPECTRAL has
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1403-collective itself — not something else — is being timed.
docs/performance/scaling/levante_campaign_2026-07-24.md-1404-
docs/performance/scaling/levante_campaign_2026-07-24.md-1405-HOST DISPATCH MUST BE SUBTRACTED, and this was nearly a self-inflicted
docs/performance/scaling/levante_campaign_2026-07-24.md-1406-error: the FIRST version timed one jit call per exchange and reported
docs/performance/scaling/levante_campaign_2026-07-24.md-1407-287-518 us "latency" (job 26457469) — two orders above what these fabrics
docs/performance/scaling/levante_campaign_2026-07-24.md-1408-do, because dispatch dominates a single call. Feeding that into `t_bound`
docs/performance/scaling/levante_campaign_2026-07-24.md-1409-would have produced a confidently WRONG roofline, strictly worse than the
docs/performance/scaling/levante_campaign_2026-07-24.md-1410-uncalibrated generic line it was meant to replace. The tool now times 1 rep
docs/performance/scaling/levante_campaign_2026-07-24.md-1411-vs n reps inside one jit and differences them; the removed dispatch
docs/performance/scaling/levante_campaign_2026-07-24.md-1412-(287-529 us) is still reported alongside so the contamination stays
docs/performance/scaling/levante_campaign_2026-07-24.md:1413:visible. Validated on CPU virtual devices: 275 us -> 26 us.
docs/performance/scaling/levante_campaign_2026-07-24.md-1414-
docs/performance/scaling/levante_campaign_2026-07-24.md-1415-Quote the lane that matches the plot: intra-node NVLink and inter-node IB
docs/performance/scaling/levante_campaign_2026-07-24.md-1416-differ by ~3x in bandwidth, so using the wrong one is its own confound.
docs/performance/scaling/levante_campaign_2026-07-24.md-1417-
docs/performance/scaling/levante_campaign_2026-07-24.md-1418-## OPERATIONAL NOTE: transient multi-node hangs (3 occurrences)
docs/performance/scaling/levante_campaign_2026-07-24.md-1419-
docs/performance/scaling/levante_campaign_2026-07-24.md-1420-Three times this campaign a multi-node GPU job consumed its entire
docs/performance/scaling/levante_campaign_2026-07-24.md-1421-walltime without emitting a timed row, then ran normally on retry with the
docs/performance/scaling/levante_campaign_2026-07-24.md-1422-IDENTICAL configuration:
docs/performance/scaling/levante_campaign_2026-07-24.md-1423-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1451-   ULPs (size-dependent!) — process-0 broadcast + allgathered
docs/performance/scaling/levante_campaign_2026-07-24.md-1452-   tolerance-compared divergence guard (quantized-equality v1
docs/performance/scaling/levante_campaign_2026-07-24.md-1453-   false-positived on a rounding boundary; v2 rtol=1e-5). Gates:
docs/performance/scaling/levante_campaign_2026-07-24.md-1454-   equivalence 6/6 and selfspawn 2/2 re-run on EVERY iteration of the
docs/performance/scaling/levante_campaign_2026-07-24.md-1455-   guard (last: jobs 26453906/26453981), plus live np4/8/16 multinode
docs/performance/scaling/levante_campaign_2026-07-24.md-1456-   (jobs 26452743-45, 26453279).
docs/performance/scaling/levante_campaign_2026-07-24.md-1457-6. Multicontroller host materialization in the tiled bench finiteness gate
docs/performance/scaling/levante_campaign_2026-07-24.md-1458-   → on-device global reduce.
docs/performance/scaling/levante_campaign_2026-07-24.md-1459-7. OUTDIR same-second stamp collision → job-ID suffix everywhere.
docs/performance/scaling/levante_campaign_2026-07-24.md-1460-8. `setup_mpi_venv.sh` could not resolve uv-workspace members with plain
docs/performance/scaling/levante_campaign_2026-07-24.md:1461:   pip → pins first + `install_federation.py --all`; mpi4jax source-built
docs/performance/scaling/levante_campaign_2026-07-24.md-1462-   with the system toolchain (GLIBCXX mismatch with gcc-11-built OpenMPI
docs/performance/scaling/levante_campaign_2026-07-24.md-1463-   module).
docs/performance/scaling/levante_campaign_2026-07-24.md-1464-9. Diagnosis tool halo/overlap phases timed UN-JITTED eager pads
docs/performance/scaling/levante_campaign_2026-07-24.md-1465-   (20.5e6 us per "exchange", bandwidth 0.0 GB/s; job 26447827) - FIXED
docs/performance/scaling/levante_campaign_2026-07-24.md-1466-   this campaign (jit + dtype-correct bytes + refuse a bandwidth at
docs/performance/scaling/levante_campaign_2026-07-24.md-1467-   world_size==1; overlap fractions >100% now refused), contract test
docs/performance/scaling/levante_campaign_2026-07-24.md-1468-   tests/bench/test_halo_profiler_contract.py, verified 27-162 us at
docs/performance/scaling/levante_campaign_2026-07-24.md-1469-   C24/L8 and 463-4791 us at C384/L60 (job 26454084). Census + scan phases
docs/performance/scaling/levante_campaign_2026-07-24.md-1470-   were always sound (cube C96/L40: 0/7/14 CP/step at nd 1/2/3 + 1
docs/performance/scaling/levante_campaign_2026-07-24.md-1471-   all-reduce, job 26447827; 24-dev tiled closed loop 384 CP + 1 AR,
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1526-
docs/performance/scaling/levante_campaign_2026-07-24.md-1527-   VERDICT on the audit's item 4 as a whole: BOTH halves measured, BOTH
docs/performance/scaling/levante_campaign_2026-07-24.md-1528-   negative on this codebase — the "~2x on ~40%-land grids" projection is
docs/performance/scaling/levante_campaign_2026-07-24.md-1529-   refuted twice over (gather penalty eats the compaction saving at 0.71
docs/performance/scaling/levante_campaign_2026-07-24.md-1530-   wet; wet-balanced bands worsen dense-compute balance). The item is
docs/performance/scaling/levante_campaign_2026-07-24.md-1531-   CLOSED as not-worth-building, with receipts. (Also moot for the SPMD
docs/performance/scaling/levante_campaign_2026-07-24.md-1532-   lane: jax equal-shard sharding would need padding to the max band,
docs/performance/scaling/levante_campaign_2026-07-24.md-1533-   returning exactly the imbalance removed.)
docs/performance/scaling/levante_campaign_2026-07-24.md-1534-
docs/performance/scaling/levante_campaign_2026-07-24.md-1535-   *Expensive half — gather/scatter compaction: MEASURED, and the audit's
docs/performance/scaling/levante_campaign_2026-07-24.md:1536:   "~2x" is REFUTED* (`bench_gather_vs_slice_stencil.py`, job 26479884,
docs/performance/scaling/levante_campaign_2026-07-24.md-1537-   A100 f32, correctness self-checked). Per-cell gather penalty for a
docs/performance/scaling/levante_campaign_2026-07-24.md-1538-   5-point Laplacian vs the dense sliced version: **1.40-1.77x**, so
docs/performance/scaling/levante_campaign_2026-07-24.md-1539-   compaction wins only when wet_fraction < 0.56-0.72 (size-dependent).
docs/performance/scaling/levante_campaign_2026-07-24.md-1540-   At the REAL global-ocean wet fraction (~0.71), packed-gather is a net
docs/performance/scaling/levante_campaign_2026-07-24.md-1541-   LOSS on the full LL576 grid (ratio 1.16) and a wash at the nd4 tile
docs/performance/scaling/levante_campaign_2026-07-24.md-1542-   (1.03). SCOPE (codex round-10): the two numbers are END-MEMBER ESTIMATES, not
docs/performance/scaling/levante_campaign_2026-07-24.md-1543-   a bound — 0.86x is the pure-stencil member (measured), 1.41x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1544-   pure-column ideal; a real step also pays packing/scattering at the
docs/performance/scaling/levante_campaign_2026-07-24.md-1545-   interface, sees real wet topology (not banded), richer stencils, and
docs/performance/scaling/levante_campaign_2026-07-24.md-1546-   communication, none of which the microbench prices. What survives
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1555-   holds resolution/physics/precision/levels/timing fixed and varies ONLY
docs/performance/scaling/levante_campaign_2026-07-24.md-1556-   the srun distribution: `block:block` 186.83 ms vs `block:cyclic`
docs/performance/scaling/levante_campaign_2026-07-24.md-1557-   **87.59 ms — 2.13x from the distribution flag alone.** Leading
docs/performance/scaling/levante_campaign_2026-07-24.md-1558-   interpretation (codex round-10 scoping): per-socket memory-bandwidth
docs/performance/scaling/levante_campaign_2026-07-24.md-1559-   contention on the 2x Milan 7763 node, consistent with the
docs/performance/scaling/levante_campaign_2026-07-24.md-1560-   spread-ladder reduction — but per-rank NUMA-binding receipts and
docs/performance/scaling/levante_campaign_2026-07-24.md-1561-   bandwidth counters were NOT captured, so the mechanism is inferred
docs/performance/scaling/levante_campaign_2026-07-24.md-1562-   from the placement swing, not instrumented. (Note 32 single-core ranks
docs/performance/scaling/levante_campaign_2026-07-24.md-1563-   FIT in one 64-core socket, so np32 is not automatically two-socket.)
docs/performance/scaling/levante_campaign_2026-07-24.md-1564-   FIX regardless of mechanism: `--distribution=block:cyclic
docs/performance/scaling/levante_campaign_2026-07-24.md:1565:   --cpu-bind=cores` on packed CPU lanes.
docs/performance/scaling/levante_campaign_2026-07-24.md-1566-
docs/performance/scaling/levante_campaign_2026-07-24.md-1567-7. **1-D bands vs 2-D pencils at np64 (job 26479904): the pencil path
docs/performance/scaling/levante_campaign_2026-07-24.md-1568-   is 1.37x faster** (latlon r256 moist f64, same dt: 253.93 -> 185.03
docs/performance/scaling/levante_campaign_2026-07-24.md-1569-   ms/step) — BUT this is NOT a pure decomposition A/B (codex round-10):
docs/performance/scaling/levante_campaign_2026-07-24.md-1570-   `--latlon-2d` selects the wall-pole 2-D path while the band path
docs/performance/scaling/levante_campaign_2026-07-24.md-1571-   keeps the atmospheric pole-fold, so boundary semantics change along
docs/performance/scaling/levante_campaign_2026-07-24.md-1572-   with the decomposition. Report as a regular-vs-wall-pole path
docs/performance/scaling/levante_campaign_2026-07-24.md-1573-   throughput result; attributing the 1.37x to decomposition alone would
docs/performance/scaling/levante_campaign_2026-07-24.md-1574-   need a pole-matched A/B. Remaining: pole-matched ladder (np32-128);
docs/performance/scaling/levante_campaign_2026-07-24.md-1575-   ocean lane pencil refusal stands (`test_2d_pencil_layout_refused` —
docs/performance/scaling/levante_campaign_2026-07-24.md-1576-   wide-halo is 1-D-only by design).
docs/performance/scaling/levante_campaign_2026-07-24.md:1577:7. Route-A CUDA-aware mpi4jax lane (`gpu_moist_scaling.slurm`) — only if a
docs/performance/scaling/levante_campaign_2026-07-24.md-1578-   route-A-vs-B A/B is ever wanted; route-B beat every route-A reference
docs/performance/scaling/levante_campaign_2026-07-24.md-1579-   available here.
docs/performance/scaling/levante_campaign_2026-07-24.md-1580-
docs/performance/scaling/levante_campaign_2026-07-24.md-1581-DONE during the campaign (were open at the start): C768 same-path ladder
docs/performance/scaling/levante_campaign_2026-07-24.md-1582-(eff 1.04) and tiled closed loop (14.3 GCells/s); atm lat-lon weak at a
docs/performance/scaling/levante_campaign_2026-07-24.md:1583:production tile; ocean weak at a production tile; the ico CPU-MPI ladder;
docs/performance/scaling/levante_campaign_2026-07-24.md:1584:the CPU spread ladder; and the diagnosis tool's halo + overlap phases,
docs/performance/scaling/levante_campaign_2026-07-24.md-1585-which were found broken and fixed with a contract test.
docs/performance/scaling/levante_campaign_2026-07-24.md-1586-
docs/performance/scaling/levante_campaign_2026-07-24.md-1587-## Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)
docs/performance/scaling/levante_campaign_2026-07-24.md-1588-
docs/performance/scaling/levante_campaign_2026-07-24.md-1589-Both jobs the dropped session left behind COMPLETED; neither had been
docs/performance/scaling/levante_campaign_2026-07-24.md-1590-analysed. First read-out below, CORRECTED per codex round-20
docs/performance/scaling/levante_campaign_2026-07-24.md-1591-(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
docs/performance/scaling/levante_campaign_2026-07-24.md-1592-VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
docs/performance/scaling/levante_campaign_2026-07-24.md-1593-in the first draft's weak-scaling claim).
docs/performance/scaling/levante_campaign_2026-07-24.md-1594-
docs/performance/scaling/levante_campaign_2026-07-24.md:1595:### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
docs/performance/scaling/levante_campaign_2026-07-24.md-1596-
docs/performance/scaling/levante_campaign_2026-07-24.md-1597-Matrix at a NOMINAL MEAN target of 5,120 cells/rank (the JSONL's
docs/performance/scaling/levante_campaign_2026-07-24.md-1598-`cells_per_rank_achieved` is global floor division —
docs/performance/scaling/levante_campaign_2026-07-24.md-1599-`int(mesh.nCells) // n_ranks`, bench_ocean_mpas_scaling.py:751 — NOT a
docs/performance/scaling/levante_campaign_2026-07-24.md-1600-balance statement; method pinned per arm, never `auto`, which flipped
docs/performance/scaling/levante_campaign_2026-07-24.md-1601-meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64,
docs/performance/scaling/levante_campaign_2026-07-24.md-1602-nlev 20, 32 ranks/node. Actual per-rank OWNED ranges
docs/performance/scaling/levante_campaign_2026-07-24.md-1603-(`metadata.partition_metrics.cells_per_rank_min/max`): geometric
docs/performance/scaling/levante_campaign_2026-07-24.md-1604-5,120–5,121 at BOTH scales; metis 5,100–5,145 @32 and 5,093–5,144 @128
docs/performance/scaling/levante_campaign_2026-07-24.md-1605-(±0.5 %). WET load is looser still under metis — per-rank wet
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1616-| D | s8 np128 metis, block:cyclic | 333.39 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1617-| E | s8 np128 metis, block:block | 537.91 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1618-
docs/performance/scaling/levante_campaign_2026-07-24.md-1619-* **This METIS configuration LOSES to geometric at s8/np128** (D/B =
docs/performance/scaling/levante_campaign_2026-07-24.md-1620-  +7.9 %), despite the better offline cut (partq s8@np128: edge_cut
docs/performance/scaling/levante_campaign_2026-07-24.md-1621-  1.89 % vs 2.01 %, halo mean 592 vs 629). Scale-out term (s7@32 ->
docs/performance/scaling/levante_campaign_2026-07-24.md-1622-  s8@128, which crosses 1 -> 4 nodes as well as 4x ranks — NOT a pure
docs/performance/scaling/levante_campaign_2026-07-24.md-1623-  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
docs/performance/scaling/levante_campaign_2026-07-24.md-1624-  -> step-time inference FAILS on this lane; part of metis's loss is
docs/performance/scaling/levante_campaign_2026-07-24.md-1625-  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
docs/performance/scaling/levante_campaign_2026-07-24.md:1626:  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
docs/performance/scaling/levante_campaign_2026-07-24.md-1627-  out partition/mapping improvements generally (e.g. wet-cell-weighted
docs/performance/scaling/levante_campaign_2026-07-24.md-1628-  METIS was NOT tested).
docs/performance/scaling/levante_campaign_2026-07-24.md:1629:* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
docs/performance/scaling/levante_campaign_2026-07-24.md-1630-  a byte-identical partition). NOTE the second `--distribution` field is
docs/performance/scaling/levante_campaign_2026-07-24.md-1631-  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
docs/performance/scaling/levante_campaign_2026-07-24.md-1632-  identically; the swing is socket-level. Mechanism (per-socket
docs/performance/scaling/levante_campaign_2026-07-24.md-1633-  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
docs/performance/scaling/levante_campaign_2026-07-24.md-1634-  2.13x receipt; never instrumented with bandwidth counters.
docs/performance/scaling/levante_campaign_2026-07-24.md-1635-* Caveats: timing-only receipt — no parity/conservation gate ran in
docs/performance/scaling/levante_campaign_2026-07-24.md:1636:  these arms, and the CPU nodes emit `UCX WARN transports
docs/performance/scaling/levante_campaign_2026-07-24.md-1637-  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
docs/performance/scaling/levante_campaign_2026-07-24.md:1638:  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
docs/performance/scaling/levante_campaign_2026-07-24.md-1639-  timing, but a "production config" claim would need a gated arm).
docs/performance/scaling/levante_campaign_2026-07-24.md-1640-
docs/performance/scaling/levante_campaign_2026-07-24.md-1641-### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
docs/performance/scaling/levante_campaign_2026-07-24.md-1642-
docs/performance/scaling/levante_campaign_2026-07-24.md-1643-f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
docs/performance/scaling/levante_campaign_2026-07-24.md-1644-scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
docs/performance/scaling/levante_campaign_2026-07-24.md-1645-L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
docs/performance/scaling/levante_campaign_2026-07-24.md-1646-np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
docs/performance/scaling/levante_campaign_2026-07-24.md-1647-reads `unknown` — same allocation, same submitted script, so the same
docs/performance/scaling/levante_campaign_2026-07-24.md-1648-binary is PLAUSIBLE but that row stays non-reproduction-grade on its
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1664-* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
docs/performance/scaling/levante_campaign_2026-07-24.md-1665-  efficiency 0.55–0.74" pairs and the "~1.4x per 4x ranks GPU rank-count
docs/performance/scaling/levante_campaign_2026-07-24.md-1666-  term".** Confounds: (a) every existing s8 receipt is the generator's
docs/performance/scaling/levante_campaign_2026-07-24.md-1667-  default PRODUCTION Lloyd mesh, while s9 is lloyd=0 — different mesh
docs/performance/scaling/levante_campaign_2026-07-24.md-1668-  family, not the same protocol; (b) two comparator points came from the
docs/performance/scaling/levante_campaign_2026-07-24.md-1669-  np2-16 ladder (jobs 26454476/26454618), not the np32-128 extension
docs/performance/scaling/levante_campaign_2026-07-24.md-1670-  rows; (c) the three ratios are 1.80/1.35/1.41 — not "consistent
docs/performance/scaling/levante_campaign_2026-07-24.md-1671-  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
docs/performance/scaling/levante_campaign_2026-07-24.md-1672-  no weak-scaling direction is claimed until it lands.
docs/performance/scaling/levante_campaign_2026-07-24.md-1673-
docs/performance/scaling/levante_campaign_2026-07-24.md:1674:### Next receipts submitted 2026-08-02
docs/performance/scaling/levante_campaign_2026-07-24.md-1675-
docs/performance/scaling/levante_campaign_2026-07-24.md:1676:1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
docs/performance/scaling/levante_campaign_2026-07-24.md-1677-   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
docs/performance/scaling/levante_campaign_2026-07-24.md-1678-   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
docs/performance/scaling/levante_campaign_2026-07-24.md:1679:   after — BRACKETED, not fully counterbalanced; a penalty's attribution
docs/performance/scaling/levante_campaign_2026-07-24.md-1680-   to fabric vs placement/drift needs the per-step nodelist table +
docs/performance/scaling/levante_campaign_2026-07-24.md-1681-   follow-up). steps=5000 so the stepping window
docs/performance/scaling/levante_campaign_2026-07-24.md-1682-   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
docs/performance/scaling/levante_campaign_2026-07-24.md-1683-   wall-clock brackets logged as overlap evidence. CONFIRM bar:
docs/performance/scaling/levante_campaign_2026-07-24.md-1684-   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
docs/performance/scaling/levante_campaign_2026-07-24.md-1685-   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
docs/performance/scaling/levante_campaign_2026-07-24.md:1686:   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
docs/performance/scaling/levante_campaign_2026-07-24.md-1687-   >10 % = a CO-EXECUTION penalty, quantified per replica — its
docs/performance/scaling/levante_campaign_2026-07-24.md-1688-   attribution (fabric contention vs placement/topology vs drift) is a
docs/performance/scaling/levante_campaign_2026-07-24.md-1689-   follow-up, not a conclusion of this job.
docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
docs/performance/scaling/levante_campaign_2026-07-24.md-1691-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
docs/performance/scaling/levante_campaign_2026-07-24.md-1692-   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
docs/performance/scaling/levante_campaign_2026-07-24.md-1693-   recomputed only from these.
docs/performance/scaling/levante_campaign_2026-07-24.md-1694-
docs/performance/scaling/levante_campaign_2026-07-24.md-1695-### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
docs/performance/scaling/levante_campaign_2026-07-24.md-1696-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-1707-caught the first draft halving this), so the loss is NOT
docs/performance/scaling/levante_campaign_2026-07-24.md-1708-floor-attributable. Mechanism OPEN — candidates (uninstrumented): 1-D
docs/performance/scaling/levante_campaign_2026-07-24.md-1709-band thinning to 16 rows/rank raising halo/compute ratio, and the
docs/performance/scaling/levante_campaign_2026-07-24.md-1710-16 -> 32-node NCCL topology step. 39.11 GC/s (from 5.5767 ms) is the
docs/performance/scaling/levante_campaign_2026-07-24.md-1711-highest measured throughput of ANY lane in the campaign. The companion
docs/performance/scaling/levante_campaign_2026-07-24.md-1712-oc128 (26534067) FAILED pre-#1370-fix with the 109.5 GB resident-args
docs/performance/scaling/levante_campaign_2026-07-24.md-1713-signature; retry submitted post-fix (below).
docs/performance/scaling/levante_campaign_2026-07-24.md-1714-
docs/performance/scaling/levante_campaign_2026-07-24.md-1715-## Hundreds-of-devices push (user directive 2026-08-02)
docs/performance/scaling/levante_campaign_2026-07-24.md-1716-
docs/performance/scaling/levante_campaign_2026-07-24.md:1717:"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
docs/performance/scaling/levante_campaign_2026-07-24.md-1718-GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
docs/performance/scaling/levante_campaign_2026-07-24.md-1719-partition effectively unbounded for our rank counts. Submitted set:
docs/performance/scaling/levante_campaign_2026-07-24.md-1720-
docs/performance/scaling/levante_campaign_2026-07-24.md-1721-| job | what | devices | why |
docs/performance/scaling/levante_campaign_2026-07-24.md-1722-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md:1723:| 26628196 | s9 ensemble contention (v3; 26628021/26627810 superseded pre-start) | 128 GPU (4x32) | lever #1 receipt |
docs/performance/scaling/levante_campaign_2026-07-24.md-1724-| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
docs/performance/scaling/levante_campaign_2026-07-24.md-1725-| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
docs/performance/scaling/levante_campaign_2026-07-24.md:1726:| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
docs/performance/scaling/levante_campaign_2026-07-24.md:1727:| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
docs/performance/scaling/levante_campaign_2026-07-24.md-1728-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
docs/performance/scaling/levante_campaign_2026-07-24.md-1729-
docs/performance/scaling/levante_campaign_2026-07-24.md-1730-s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
docs/performance/scaling/levante_campaign_2026-07-24.md-1731-
docs/performance/scaling/levante_campaign_2026-07-24.md:1732:### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)
docs/performance/scaling/levante_campaign_2026-07-24.md-1733-
docs/performance/scaling/levante_campaign_2026-07-24.md-1734-r512 (512x1024 = 524k cols) L26 f64 moist, 32 rpn block:cyclic,
docs/performance/scaling/levante_campaign_2026-07-24.md-1735-wall-pole 2-D pencil lane (labelled; NOT the pole fold):
docs/performance/scaling/levante_campaign_2026-07-24.md-1736-
docs/performance/scaling/levante_campaign_2026-07-24.md-1737-| ranks | cols/rank | ms/step | speedup vs np64 | eff |
docs/performance/scaling/levante_campaign_2026-07-24.md-1738-|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-1739-| 64 | 8,192 | 297.57 | 1.00 | 1.00 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1740-| 128 | 4,096 | 161.03 | 1.848 | 0.92 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1741-| 256 | 2,048 | 72.06 | 4.129 | 1.03 |
docs/performance/scaling/levante_campaign_2026-07-24.md-1742-| 512 | 1,024 | 44.78 | 6.645 | **0.83** |
docs/performance/scaling/levante_campaign_2026-07-24.md-1743-
docs/performance/scaling/levante_campaign_2026-07-24.md-1744-Distribution verified against the masquerade trap: result rows carry
docs/performance/scaling/levante_campaign_2026-07-24.md-1745-`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
docs/performance/scaling/levante_campaign_2026-07-24.md:1746:count on this mpi4jax lane, not the world size). 128->256 is
docs/performance/scaling/levante_campaign_2026-07-24.md-1747-SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache
docs/performance/scaling/levante_campaign_2026-07-24.md-1748-transition on Milan (mechanism PLAUSIBLE, uninstrumented). End-to-end
docs/performance/scaling/levante_campaign_2026-07-24.md:1749:64->512 eff 0.83 at 1k cols/rank: TIMING-ONLY evidence that the lat-lon
docs/performance/scaling/levante_campaign_2026-07-24.md:1750:CPU lane scales into the hundreds. QUALIFIER (codex r22): the run logs
docs/performance/scaling/levante_campaign_2026-07-24.md:1751:an out-of-tested-range mpi4jax==0.9.0 pairing ("may fail or produce
docs/performance/scaling/levante_campaign_2026-07-24.md-1752-incorrect results", parallel/reductions.py runtime check) and UCX
docs/performance/scaling/levante_campaign_2026-07-24.md:1753:VM_UNMAP warnings — no parity/conservation gate ran, so this ladder is
docs/performance/scaling/levante_campaign_2026-07-24.md-1754-unvalidated timing evidence until a supported-stack rerun. (Exact pencil factorisations are not recorded in the
docs/performance/scaling/levante_campaign_2026-07-24.md-1755-result JSON — only `decomposition: 2d`; a follow-up could add them to
docs/performance/scaling/levante_campaign_2026-07-24.md-1756-the bench metadata.)

exec
/usr/bin/bash -lc "sed -n '1,260p' scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch && sed -n '1674,1693p;1721,1758p' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 302ms:
#!/bin/bash -l
#SBATCH --job-name=mpas_s9_ens
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=mpas_s9_ens.%j.log
# ENSEMBLE-PARALLELISM RECEIPT (codex 2026-07-31 consult lever #1; design
# fixed per codex round-20 items 9-12).  The tile floor caps ONE
# trajectory's strong scaling; past it the honest use of 128 GPUs is K
# independent replicas at the per-trajectory sweet spot.  This job
# measures the only thing that can refute that: FABRIC CONTENTION between
# replicas sharing the IB tree.
#
#   solo_pre  : ONE 32-GPU s9 run, other 24 nodes idle
#   phase B   : FOUR concurrent 32-GPU s9 runs on disjoint 8-node sets
#               (SLURM_STEP_NODELIST is per-step -> per-replica
#               coordinator autodetect; jobid-derived port shared but
#               hosts differ)
#   solo_post : solo again AFTER phase B (brackets ordering/thermal
#               drift; contrast uses mean of the two solos)
#
# Falsifiability, written BEFORE submit:
#   numbers : 2 solo + 4 replica steady_median_ms
#   CONFIRM : max(replica) <= 1.10 x mean(solo) -> guaranteed aggregate
#             >= 4/1.10 = 3.64x the 32-GPU solo rate (~19.9 GC/s if solo
#             reproduces 5.47) = ~3.3x the observed 128-GPU
#             single-trajectory rate.  NOT "4x": 1.10 is the bar, the
#             margin below it is the measured contention.
#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
#             per replica.
# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
# table sacct prints at the end.  Absolute ms/step is therefore only
# compared WITHIN this job (solo vs replicas), never against the
# steps-12 ladder rows.
# INTERPRETATION LIMIT (codex r21 item 4): this design SCREENS for a
# co-execution penalty vs the solo brackets; if a penalty appears, its
# attribution (shared IB fabric vs node/topology placement vs drift)
# needs the per-step nodelist table + follow-up, and the solo/replica
# ordering is bracketed (pre+post), not fully counterbalanced.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s9_ens_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0

run_arm () { # tag  (one 32-GPU replica on 8 disjoint nodes)
  echo "[$1] launch epoch=$(date +%s.%N)"
  # rank 0 prints its step's nodelist BEFORE exec'ing the bench — the
  # per-replica disjointness + coordinator-host evidence codex asked for
  # ("$0" inside bash -c is $PY, passed as the first post-script arg).
  JAX_ENABLE_X64=0 srun --nodes=8 --ntasks=32 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --exact --kill-on-bad-exit=1 \
      --job-name="arm_$1" \
    bash -c '[ "${SLURM_PROCID:-1}" = 0 ] && echo "[step $ARM_TAG] nodelist=$SLURM_STEP_NODELIST"; exec "$0" "$@"' \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 32 \
      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
      --partition-method sfc --reorder-for 128 \
      --out "$OUTDIR/$1.jsonl"
  s=$?
  echo "[$1] exit=$s epoch=$(date +%s.%N)"
  return $s
}

echo "=== solo_pre: 1x32 GPUs, 24 nodes idle ==="
ARM_TAG=solo_pre; export ARM_TAG
run_arm solo_pre || { echo "solo_pre FAILED"; rc=1; }

echo "=== phase B: 4 concurrent 32-GPU replicas ==="
pids=(); tags=(rep0 rep1 rep2 rep3)
for T in "${tags[@]}"; do ARM_TAG=$T run_arm "$T" & pids+=($!); done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "${tags[$i]} FAILED"; rc=1; }
done

echo "=== solo_post: 1x32 GPUs, 24 nodes idle ==="
ARM_TAG=solo_post; export ARM_TAG
run_arm solo_post || { echo "solo_post FAILED"; rc=1; }

echo "=== RESULTS ==="
for T in solo_pre rep0 rep1 rep2 rep3 solo_post; do
  F="$OUTDIR/$T.jsonl"
  "$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$F').readline())
    ms=d['steady_median_ms']
    assert math.isfinite(ms) and ms > 0, f'non-finite {ms}'
except Exception as e:
    print('$T: MISSING/INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.2f} ms  {d.get(\"mcells_per_s\",0)/1000:.2f} GC/s')" \
    || { echo "$T receipt invalid"; rc=1; }
done

echo "=== overlap evidence: per-step nodelist + wall window ==="
sacct -j "$SLURM_JOB_ID" \
  --format=JobID%18,JobName%12,NodeList%45,Start,End,State -P \
  || { echo "sacct overlap table UNAVAILABLE"; rc=1; }
echo "DONE rc=$rc"; exit $rc
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

exec
/usr/bin/bash -lc "rg -n \"26628021|26628196|fully counterbalanced|BRACKETED|REFUTE:|TIMING-ONLY|out-of-tested-range|VM_UNMAP|unvalidated timing|mpi4jax==0\\.9\\.0\" docs/performance/scaling/levante_campaign_2026-07-24.md && git diff --check -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch && git status --short -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 404ms:
1679:   after — BRACKETED, not fully counterbalanced; a penalty's attribution
1686:   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
1723:| 26628196 | s9 ensemble contention (v3; 26628021/26627810 superseded pre-start) | 128 GPU (4x32) | lever #1 receipt |
1749:64->512 eff 0.83 at 1k cols/rank: TIMING-ONLY evidence that the lat-lon
1751:an out-of-tested-range mpi4jax==0.9.0 pairing ("may fail or produce
1753:VM_UNMAP warnings — no parity/conservation gate ran, so this ladder is
1754:unvalidated timing evidence until a supported-stack rerun. (Exact pencil factorisations are not recorded in the
 M docs/performance/scaling/levante_campaign_2026-07-24.md
?? scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch

codex
No remaining inconsistency found.

The ensemble wording, REFUTE attribution limit, and v3 job ID align with the script. The CPU r512 result is clearly limited to unvalidated timing evidence pending a supported-stack, gated rerun.

VERDICT: SHIP — issues: none.
tokens used
25,473
No remaining inconsistency found.

The ensemble wording, REFUTE attribution limit, and v3 job ID align with the script. The CPU r512 result is clearly limited to unvalidated timing evidence pending a supported-stack, gated rerun.

VERDICT: SHIP — issues: none.
