# Weak/strong scaling bottleneck diagnosis — atm + ocean, MPI + GPU (2026-07-20)

Scope: consolidate the measured weak/strong scaling bottlenecks for atmosphere
and ocean on both transports (route-A mpi4jax, route-B NCCL/SPMD), pin the
LATENCY-bound signal with fresh numbers, and close the two instrument gaps that
kept that signal off the actual Derecho/Levante runs. Read the campaign context
first: `derecho_levante_sota_review_2026-07.md` (measured baselines + SOTA),
`spmd_message_census_2026-07-08.md` (the message-count analysis this extends),
`SCALING_STATUS_AUDIT.md` (support matrix).

Every claim below is labelled **CONFIRMED** (measured / read from code) or
**PLAUSIBLE** (inferred, needs a machine receipt).

---

## 1. The bottleneck table (what limits each axis, and why)

| Component × axis | Limiter | Evidence | Tier |
|---|---|---|---|
| Atm cube strong (≤6 GPU) | **collective-permute COUNT** grows with shard count while per-msg NCCL p2p latency (~30–80 µs) doesn't amortise on small tiles → anti-scales | census: 12 CP @2dev, 41 @6dev (C24/L8, below); f64≡f32 curves ⇒ latency-bound | CONFIRMED |
| Atm latlon strong | 4→8 GPU node-crossing plateau on route-A (mpi4jax sendrecv OPAQUE to XLA latency-hiding scheduler ⇒ no comm/compute overlap) | Derecho 78 km throughput 600→480→recover@16; route-B multicontroller SHIPPED to remove it | CONFIRMED |
| Atm ico strong | best-scaling grid (low perimeter/area cell partition); multihost SPMD still open | Derecho 28 km ico eff ~0.38 @16 A100, still rising | CONFIRMED |
| Atm/all coarse | per-device saturation floor (<~30k cols/GPU flat/anti) — NOT a defect | 111 km latlon/ico FLAT 1→16 A100 | CONFIRMED |
| Ocean strong (CPU-MPI) | **implicit-CN PCG reduction wall**: default `pcg_variant="standard"` = 2 *sequentially-dependent* all-reduces/iter × `fixed_iters=60` ⇒ ~120 latency-serialized all-reduces/step | `barotropic_common.py` `_fixed_iteration_pcg` (p·Ap @:619 → dependent r·z @:626); np16→32 eff 0.55 | CONFIRMED |
| Ocean strong (mitigation) | `single_reduce` (Chronopoulos–Gear) → 1 all-reduce/iter; split-explicit + `barotropic_local_subcycle_clamp` → 3 all-reduce/step (reduction-free subcycle) | both exist, both NON-default; beats standard at ≥2 nodes 1.15–1.65× | CONFIRMED |
| Ocean weak | ~1.0 eff (as SOTA); the strong ceiling is naive land imbalance, not comm | 0.97 weak @590k cells/rank | CONFIRMED |
| Spectral | single-device by design (both multi-device schemes measured anti-scaling) | prior campaign | CONFIRMED |

Two structural hot-loop items found in the code map (both PLAUSIBLE as
scaling costs at high step counts, neither a hot-loop collective):
- **Per-step host dispatch** in the operator-split SPMD driver
  (`model_driver.py:7783` Python `for` over `seg_steps`) — a compiled
  `lax.scan`-per-segment path exists (M3b) but the operator-split loop dispatches
  per step; host-dispatch overhead scales with step count.
- **Blocking all-gather + full replicated allocation once per sim-day**
  (`model_driver.py:7798`) for the NaN-guard/callback — not per-step, but a
  cross-process barrier that materialises the global state each segment.

## 2. The LATENCY-bound signal, pinned (fresh, this session)

The 2026-07-08 census established: f64 and f32 GPU strong-scaling curves
coincide ⇒ the multi-GPU leg is **latency-bound, not bandwidth-bound**, so the
lever is the per-step *message count*, not the byte volume. The count is a
STATIC compile property — a CPU virtual-device compile yields the same count the
GPU executes.

Fresh census (sharded cube full PE step, `run_scaling_diagnosis.py --mode
census`, C24/L8, CPU virtual devices):

| devices | collective-permute / step | all-reduce / step |
|---|---|---|
| 2 | 12 | 1 |
| 6 | 41 | 1 |

Readings (CONFIRMED): the single all-reduce is the conservation fixer (fine).
The permute count scales ~linearly with the shard count (edge-coloring: each
face has 4 neighbours ⇒ 4 ppermute rounds × the exchange points) — this IS the
cube anti-scaling. (The absolute counts are lower than the 15/46 in the
2026-07-08 note — halo packing improved since — which is exactly why the count
now belongs on every row as a REGRESSION-tracked metric, not a scratch probe.)

## 3. Instrument gaps closed this session

The diagnosis was well-characterised in the docs, but two gaps kept the signal
off the real machine runs:

1. **The message census counted only collective-permutes.** The ocean PCG
   all-reduce wall — the #1 ocean strong-scaling limiter — was invisible to a
   permute-only census. Fixed: canonical `metadata.count_collectives()` /
   `hlo_collective_census()` count **all** families (permute + all-reduce +
   all-gather + all-to-all + reduce-scatter) with the same op-call-form
   discipline (config-header flag echoes never inflate; async `-start` counted
   once). The permute family stays bit-identical to `count_collective_permutes`.
2. **The bottleneck tool ran on NO Derecho/Levante job.** The cluster campaign
   invoked only the throughput benches (SYPD + permute census); no per-phase
   halo/reduction/roofline/overlap breakdown was ever captured on the target
   machines. Fixed:
   - `run_scaling_diagnosis.py` gained a `census` mode + a census phase in
     `full`/`quick` (recorded in `summary.json`) — the count is now first-class.
   - `scripts/cluster/scaling_derecho/diagnosis.pbs` +
     `scripts/cluster/scaling_levante/diagnosis.sbatch` run the tool across the
     cube face-shard `1 2 3` ladder (must divide 6; >6 uses the tiled bench), so
     the message-count-vs-shard curve AND the full per-phase diagnosis land on
     the real hardware.
   - The SPMD throughput benches the multinode jobs already run
     (`bench_cube_tiled_step_scaling`, `bench_mpas_spmd_scaling`) now record the
     full `hlo_collectives` dict alongside the legacy scalar.

## 4. What to run on Derecho / Levante now

1. **Diagnosis lane** (new): `qsub scripts/cluster/scaling_derecho/diagnosis.pbs`
   / `sbatch scripts/cluster/scaling_levante/diagnosis.sbatch` at production
   size (C96/L40, ≥30k cols/GPU). Yields the census ladder + per-phase halo BW,
   reduction latency, roofline, overlap — the WHERE, not just the SYPD.
2. **Ocean reduction-wall A/B** (the highest-value code lever, already
   selectable): re-run `bench_ocean_mpi_scaling.py` / the SPMD ocean bench with
   `--pcg-variant single_reduce` and with split-explicit +
   `barotropic_local_subcycle_clamp` at ≥16 ranks — the calculus flips toward
   the reduction-free path where per-message latency dominates.
3. **Lane-T GPU-runtime A/B** (already wired, `RUN_TUNE=1`): PGLE was the
   measured winner (+8.5 %); the XLA collective-permute-combine + pipelined-p2p
   arms still need SPLITTING to attribute (the combined arm hurt −10 %).
4. Per §1, cube >6 GPU needs the sub-face tiled production step (separate
   project); the count floor is otherwise reached in code.

## 5. Verification

- `count_collectives` / `hlo_collective_census`:
  `tests/bench/test_scaling_metadata.py` (all-family synthetic HLO + error-safe
  probe).
- `run_scaling_diagnosis.py --mode census` verified locally (2/6 virtual
  devices, table §2); the census phase records into `summary.json`.
- cube/mpas bench census gates green (`test_bench_cube_tiled_step_scaling`,
  `test_bench_mpas_spmd_gates`).
- `diagnosis.pbs` / `.sbatch`: `bash -n` clean; no hardware receipt yet
  (submit-ready, PLAUSIBLE until a real run lands).
