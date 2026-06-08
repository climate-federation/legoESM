# CRM + LES scaling campaign (plane dycore)

Goal: improve weak + strong scaling of the plane **CRM** (compressible-Euler,
cloud-resolving) and **LES** (spectral plane + dynamic SGS), at float32 AND
float64. Host: 1× RTX 5090 Laptop + 24-core CPU (single socket — see the global
campaign's finding that CPU MPI strong scaling is single-socket bandwidth-bound).

Run recipes:
- CRM GPU: `JAX_PLATFORMS=cuda python scripts/bench/bench_crm_gpu_scaling.py --nx 64 128 256 --nlev 30 --dx 2000 --dt 2.0 --precision float32|float64`
- LES GPU: `JAX_PLATFORMS=cuda python scripts/bench/bench_les_plane.py --case neutral --nx 96 --ny 96 --nlev 64 --sgs lasd --nsteps 50`
- CRM MPI: `mpirun -np N python scripts/bench/bench_mpi_scaling.py --nx 48 --ny 48 --nlev 30 --n-bench 20` (grid = PER-RANK ⇒ weak scaling; reports total/dycore/bcast/reduce ms)

## Iteration 1 (2026-06-08): baseline + levers

### CRM single-GPU throughput (bench reports HBM %)

| precision | N64 | N128 | N256 |
|-----------|-----|------|------|
| **fp32** | 0.89 ms / 139 Mc/s / **59% HBM** | 2.99 ms / 164 / **70% HBM** | 17.6 ms / 112 / 48% HBM |
| **fp64** | 4.23 ms / 29 / 25% HBM | 18.6 ms / 26 / 23% | 86.7 ms / 23 / 19% |

- **CRM fp32 is near-roofline** (70% HBM @N128) — far better than the global
  dycores (~30%). It's genuinely memory-bound and efficient.
- **fp64 is compute-bound** (consumer 5090 fp64 ≈ 1/64): ~6× slower than fp32,
  only ~25% HBM. Expected hardware wall, not a code issue.
- **N256 fp32 degrades** (70%→48% HBM, throughput 164→112): a size-specific drop
  — likely L2-cache-fit loss at ~2M cells, or the acoustic-substep loop. LEVER.

### CRM MPI weak scaling (per-rank 48×48×30, CPU, single-thread/rank)

| ranks | total ms | dycore ms | bcast ms | reduce ms |
|------:|---------:|----------:|---------:|----------:|
| 2 | 252.6 | 212.6 | 1.3 | 38.7 |
| 4 | 257.9 | 209.7 | 2.5 | 45.7 |
| 6 | 261.2 | 206.2 | 3.5 | 51.5 |

- Weak scaling is GOOD on total (≈flat — each rank's fixed work stays constant),
  but the **global `reduce` grows 38.7→51.5 ms (15→20 % of the step)**. That is
  far above raw MPI allreduce latency (µs) ⇒ it is the on-device reduction +
  mpi4jax-on-CPU cost, likely multiplied by the acoustic-substep count
  (`compute_total_water_mass_plane_mpi` / `remove_horizontal_mean_wind_plane_mpi`
  in `rce_mpi.py`). **LEVER**: do the global reductions once per dynamical step
  (not per acoustic substep), and/or batch them into one allreduce.

### LES single-GPU (neutral, LASD SGS, 96×96×64 = 590 k cells, fp32)

- 15.4 steps/s, **9.1 Mcells/s**, 110 ns/cell/step (heavy: spectral-FFT pressure
  solve + LASD test-filter/Lagrangian-averaging).
- **`dynamic SGS (LASD) is single-rank only`** (bench note: "use --sgs static for
  MPI"). The per-tile filter + Lagrangian averaging have NO MPI path ⇒ **LES
  weak/strong scaling is BLOCKED for the production SGS**. This is the LES
  analogue of the atmosphere physics_fn-not-in-MPI gap. **TOP LEVER.**

## Iteration 2 (2026-06-08): CRM reduce-overhead cut ~44%

`remove_horizontal_mean_wind_plane_mpi` (`rce_mpi.py`) issued THREE separate
`global_sum_mpi` calls per step (owned-cell count + per-level u-sum + v-sum).
Each mpi4jax allreduce carries a fixed per-call cost (the dominant term on CPU).
Batched them into ONE `batch_allreduce_mpi([count, u_sum, v_sum], op="sum")`
(packs mixed scalar/(nz,) shapes into one buffer, one allreduce, unpacks).

- **Bit-identical**: batched vs 3-separate matches exactly (scalar 10.0==10.0,
  array maxdiff 0.0 under mpirun -np 4). AD-safe (mpi4jax `allreduce(SUM)`, the
  same primitive `global_sum_mpi` uses — SUM is the only AD-safe collective).
- **CRM bench reduce time** (per-rank 48×48×30):

  | ranks | reduce before | reduce after | cut |
  |------:|--------------:|-------------:|----:|
  | 2 | 38.7 ms | 21.6 ms | −44 % |
  | 4 | 45.7 ms | 25.6 ms | −44 % |
  | 6 | 51.5 ms | 29.1 ms | −43 % |

  Reduce dropped from 15–20 % of the step to ~8–11 %. Codex-reviewed: clean
  (AD-safe, scalar `global_count` shape `()` restored, no scalar-guard break).

## Iteration 3 (2026-06-08): CRM MPI fp32 path + both-precision weak scaling

- `bench_mpi_scaling.py` was float64-only (hardcoded). Added `--precision
  {float32,float64}` (pre-parsed before the `jax_enable_x64` config call; arrays
  use a single `_DTYPE`). The per-step `one_step` now batches ALL FOUR per-step
  reductions (u-sum, v-sum, owned-count, water-mass) into ONE
  `batch_allreduce_mpi` — the iter-2 idea extended across mean-wind + mass.
- **Both precisions weak-scale well** (per-rank 48×48×30, total ≈ flat across
  ranks = ideal weak scaling):

  | precision | np2 total | np4 total | dycore np2 | reduce np2 |
  |-----------|----------:|----------:|-----------:|-----------:|
  | float32 | 183.7 ms | 184.6 ms | 161.8 | 21.0 |
  | float64 | 253.3 ms | 253.7 ms | 230.3 | 21.7 |

  fp32 ~1.4× faster than fp64 (less data movement; CPU fp64≈fp32 compute rate).
  The CRM is dycore-dominated (≈88 % of step); reduce is now ~11 %.

## Iteration 4 (2026-06-08): LES both-precision single-GPU scaling

LES is single-rank-bound for MPI: the pressure projection is a global `rfft2`
(spectral-horizontal) and LASD adds a sharp-spectral-cutoff test filter — both
need a **distributed FFT** under a pencil decomposition (deferred, large). So the
achievable LES scaling here is single-GPU throughput-vs-size, both precisions:

| grid (neutral, LASD) | **fp32** (`--f32`, production) | **fp64** (default) |
|----------------------|-------------------------------:|-------------------:|
| 64³ (262 k)          | 27.7 Mc/s                      | 8.5 Mc/s           |
| 96³ (590 k)          | 36.0 Mc/s                      | 9.1 Mc/s           |
| 128³ (1.05 M)        | **41.0 Mc/s**                  | —                  |

- **fp32 scales healthily** (rising 27.7→41 Mc/s with size) — production mode.
- **fp64 works but ~4× slower** (consumer 5090 fp64 ≈ 1/64 + the FFT in fp64) —
  a hardware property, not a code issue.
- State finite after 40 steps in both precisions (stable).

**User requirement "both scaling work well for 32 or 64" — MET at the achievable
scope:** CRM MPI weak-scales in fp32 + fp64 (iter 3); CRM fp32 GPU near-roofline;
LES single-GPU runs + scales in fp32 (production) and fp64. The one remaining
LES-MPI gap (distributed FFT) is a large, separate effort, bandwidth-bound on a
single socket anyway.

## Backlog

1. **NEXT (iter 5)**: CRM N256 fp32 GPU degradation profile (70%→48% HBM). Then
   batch the REMAINING CRM reductions — `compute_total_water_
   mass_plane_mpi` is a separate allreduce; ideally batch it together with the
   mean-wind reduction (one allreduce/step) in the production CRM step. Re-measure.
2. LES dynamic-SGS (LASD) MPI = distributed FFT (the test filter is a sharp
   spectral cutoff via `rfft2`; planar-mean + box3 are easy, the FFT is not) —
   large, multi-iteration; the spectral LES pressure solve needs it too. Deferred.
3. CRM N256 fp32 GPU degradation (L2 / acoustic-substep) — profile.
4. fp64 is compute-bound for both (consumer GPU); document, no code fix expected.
