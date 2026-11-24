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

## Iteration 5 (2026-06-08): CRM N256 GPU degradation = L2-fit, not thermal

Thermal-controlled A/B (N128 measured before AND after the hot N256 run):

| nx  | cells   | step ms | Mc/s | HBM % |
|-----|--------:|--------:|-----:|------:|
| 128 | 491 520 | 3.65 | 134.6 | 58 |
| 192 | 1 105 920 | 8.64 | 127.9 | 55 |
| 256 | 1 966 080 | 20.91 | 94.0 | 40 |
| 128 (again) | 491 520 | 3.59 | **136.9** | 59 |

N128 fully RECOVERS after the hot N256 run (134.6→136.9) ⇒ the N256 drop is **not
thermal throttling** but a **real size effect**: the smooth 58→55→40 % HBM decline
is the L2-cache-fit-loss curve — the working set spills the GPU L2 as the domain
grows. Fixing needs horizontal **tiling/blocking** of the dycore kernel (deep
work); impact is modest (still 40 % HBM at ~2 M cells). LEVER, low priority.

## Campaign status (CRM + LES, this host)

Both precisions scale well at the achievable scope (the user requirement):
- **CRM**: MPI weak-scales fp32 (184 ms) + fp64 (254 ms) — flat; reduce overhead
  cut ~44 % then folded to one allreduce/step; GPU fp32 near-roofline (58–70 %
  HBM); fp64 GPU is the consumer 1/64 compute wall.
- **LES**: single-GPU fp32 (27.7→41 Mc/s, production) + fp64 (≈9 Mc/s); stable.

Remaining gaps are large or hardware-bound (same as the global campaign):
- **LES MPI = distributed FFT** (spectral pressure projection + sharp-spectral
  LASD test filter) — large, separate effort; bandwidth-bound on a single socket.
- **CRM N256 L2 spill** — needs kernel tiling; modest gain.
- **fp64 on consumer GPU** — 1/64 hardware wall; not a code issue.
- **CPU MPI strong scaling** — single-socket memory-bandwidth-bound (global
  campaign's finding); needs multiple sockets/nodes.

## Iteration 7 (2026-06-08): REAL CRM strong scaling — step_halo runs EAGER (100× slow)

**Correction**: iters 1–5 used `bench_mpi_scaling.py`, which computes dynamics on
**rank 0 only** then `_bcast`s — a reduction/bcast *overhead probe*, NOT real
domain decomposition. The genuine DD bench is `bench_plane_crm_dd_scaling.py`
(`step_halo` path, local slabs + halo exchange, strong/weak modes).

Real CRM **strong** scaling (fixed global 48×48×20, x64, CPU):

| np | ms/step | vs np=1 |
|---:|--------:|--------:|
| 1 | 68.6 | 1.0× |
| 2 | **7135** | **104× SLOWER** |

`PlaneCompressibleEulerModel.step_halo` (line ~2794) runs **eager-mode on
multi-rank by design** — docstring: "mpi4jax sendrecv branch not jit-safe on
macOS shared-mem." So every op + every halo `sendrecv` dispatches eagerly with a
host sync ⇒ the 100× blowup. **That caveat does NOT apply on this Linux+MPICH
host**: mpi4jax `sendrecv` is JIT-safe inside `@jax.jit` here — `voronoi_mpi`'s
`make_voronoi_mpi_step` already runs mpi4jax collectives inside a jit at scale.

⇒ **The real CRM strong-scaling fix: JIT the multi-rank `step_halo`** (mpi4jax
halo exchange inside the jit, as `make_voronoi_mpi_step` does). Single-rank
already routes to the jit'd `step()`. Expected: the 100× eager overhead collapses.
This is the genuine lever the campaign had been missing (wrong bench).

## Iteration 8 (2026-06-08): JIT the multi-rank step_halo — 123× CRM strong scaling

Implemented. Three changes:
1. `plane_mpi.py`: the ConcretizationTypeError under tracing came from
   `int(jnp.prod(jnp.asarray(static_shape)))` in `exchange_halo_plane_yxz` →
   replaced with `math.prod` (numerically identical, jit-safe).
2. `compressible_euler_plane_halo.py`: removed the preemptive "cannot be
   JIT-compiled on multi-rank" guard (a macOS-era caveat; mpi4jax `sendrecv` is
   jit-safe on Linux/MPICH, as `voronoi_mpi` proves).
3. `step_halo`: the multi-rank branch builds+caches a jit'd split-explicit core
   (mass fixer stays eager — it mutates `self._target_mass`). Cache key includes
   the full layout decomposition (codex-hardening: a stale-layout reuse would
   compute wrong halos).

**Result (DD bench, fixed global 48×48×20, `.venv-mpi` tested stack):**

| np | before (eager) | after (jit) | speedup |
|---:|---------------:|------------:|--------:|
| 1 | 67.7 ms | 67.7 ms | (already jit via `step`) |
| 2 | **7135 ms** | **57.9 ms** | **123×** |
| 4 | (worse) | 51.8 ms | — |

The catastrophic 100×-slower multi-rank path is gone, and CRM DD **now actually
strong-scales** (67.7→57.9→51.8 ms; 58 % eff @2, 33 % @4 — single-socket
bandwidth-limited, but functional). jit ≡ eager numerically (jit doesn't change
op semantics); the 123× is pure eager-dispatch/host-sync removal.

⚠️ **Pre-existing correctness caveat (NOT introduced by this change):** the
distributed test `test_plane_slow_tend_halo_mpi::...matches_single_process` FAILS
on clean `main` too (confirmed via `git stash`) — the multi-rank halo
slow-tendency mismatches the single-process reference (du_dt ~0.5 abs) on the
tested `.venv-mpi` stack (JAX 0.9.2 + mpi4jax 0.8.1). This JIT change makes the
existing path **faster, not more/less correct**; the distributed-numerics bug is
a SEPARATE pre-existing issue needing its own investigation before the DD path is
trusted for science. Flagged honestly — do not claim DD correctness on this stack.

Obsolete `test_halo_raises_on_jit_multirank` (asserted the removed guard) →
`test_halo_multirank_jit_traceable` (make_jaxpr trace-check; skips without MPI).
Codex-reviewed (cache-key layout-hardening applied; Q1 math.prod≡int(jnp.prod) OK;
Q3 jit≡eager OK).

## Iteration 9 (2026-06-08): DD halo CORRECTNESS bug fixed (the iter-8 caveat)

The iter-8 caveat (multi-rank DD halo mismatches single-process, du_dt ~0.5) was a
real **silent halo-corruption bug** in `exchange_halo_plane_yxz`, now fixed.

Diagnosis (narrowed step by step):
- Halo fn at n_ranks=1 (`jnp.roll`) == single-process EXACTLY (diff 0) → the
  operators are correct; the bug is in the multi-rank EXCHANGE.
- Mismatch persists with hyperdiffusion OFF → not the biharmonic stencil width.
- Isolated the exchange: `exchange_halo_plane_yxz` at np=2 gave max|halo−expected|
  = **5.0** (should be 0).
- Root cause: the exchange used `sendrecv(send=my_west_edge, source=west_rank,
  dest=west_rank, tag=rank/west_rank)` — **send-to-X & recv-from-X** per face. At
  any axis with exactly **2 ranks** (`west_rank == east_rank`), the two
  sendrecvs to the same neighbour shared a tag ⇒ MPI matched the WRONG message ⇒
  east/west (or north/south) halos SWAPPED. (66.7 % of cells wrong = the 4 of 6
  boundary columns affected at NX=12/np=2.)

Fix: standard **directional ring-shift** — each call sends to ONE neighbour and
receives from the OTHER (`send east-edge→east_rank, recv←west_rank`), one tag per
shift direction. Unambiguous and deadlock-free at np=2; correct for np≥3 too.

Validated (`.venv-mpi`, JAX 0.9.2 + mpi4jax 0.8.1):
- exchange isolation np=2/3/4 → **0.0** (was 5.0 @np2).
- distributed tendency test `...matches_single_process` → **PASSES** (was failing).
- single-rank halo suite → 48 passed (no regression; single-rank uses `jnp.roll`).
- AD: `jax.grad` through the exchange is finite AND correct (halo-adjoint
  accumulation, grad range [2,8] = expected for x=rank+1).
- DD strong np=2 still 54 ms/step (the iter-8 123× speedup intact, no deadlock).

**The CRM DD path is now FAST (iter 8, 123×) AND CORRECT (iter 9).** The bug
affected every 2-rank-per-axis decomposition — i.e. most small runs — so this is a
significant pre-existing correctness fix, not just a scaling one.

## Iteration 10 (2026-06-08): DD CRM scaling, both precisions (correct path)

Added `--precision {float32,float64}` to `bench_plane_crm_dd_scaling.py` (was
fp64-only). Real DD CRM scaling on the now-correct + fast path:

| mode   | prec | np1   | np2   | np4   | efficiency |
|--------|------|------:|------:|------:|------------|
| strong | fp64 | 66.6 | 59.5 | 55.0 ms | 56 % @2, 30 % @4 |
| strong | fp32 | 50.7 | 51.6 | 47.4 ms | bandwidth-bound (≈flat) |
| weak   | fp64 | 34.4 | 58.7 | 71.0 ms | ≈48 % (total grows ~2× to np4) |

- **Both precisions work**; fp32 ~1.3× faster than fp64.
- Scaling is **single-socket memory-bandwidth-limited** — the SAME ceiling the
  global campaign hit: at N ranks all share one DRAM bus, so neither strong
  (per-rank work shrinks but bandwidth fixed) nor weak (more ranks contend) scales
  near-ideal. Functional + correct, hardware-limited; real scaling needs multiple
  sockets/nodes.

The CRM DD path is now FAST (iter 8), CORRECT (iter 9), and scaling-characterized
in both precisions (iter 10) — the user requirement met at the achievable scope.

## Iteration 11 (2026-06-08): CRM DD full-step validation + bug scope

- **Tag bug is plane-specific.** Audited the other MPI halos: lat-lon is safe (at
  2 ranks the poles make `south_rank=None`, so each rank does ONE sendrecv to its
  single neighbour — no same-neighbour-twice ambiguity); voronoi/icosahedral is
  safe (tag = `rank*1000 + nbr_rank`, unique per pair; distinct RCB neighbours).
  Only the doubly-periodic plane has the topology that triggered it.
- **Full `step_halo` validated** (not just slow-tendency): 3 steps, acoustic
  substeps + Smagorinsky, multi-rank gathered vs single-rank = **1e-15
  (bit-identical)** for u/v/w/θ′/ρ′. Locked as a regression test
  (`test_full_step_halo_matches_single_process`) guarding the iter-8 JIT + iter-9
  tag fix.
- **NEW separate finding (flagged, not from this work):** with NON-zero moisture
  tracers the multi-rank full step diverges **~6e-4 over 3 steps** vs single-rank
  (the tracer-derived `b_moist` buoyancy in the acoustic substep). The dynamics
  halo is correct (1e-15 with zero tracers); the moist-coupling path has a
  separate multi-rank discrepancy needing its own investigation before moist CRM
  DD runs are trusted. (The slow-tendency `dtracers_dt` matches, so it's in the
  acoustic-substep moist coupling, not tracer advection.)

## Iteration 12 (2026-06-08): moist divergence = deliberate scalability tradeoff

Root-caused the iter-11 moist-coupling discrepancy. It is the **rank-local
horizontal mean** in `_acoustic_moist_buoyancy_w` (`compressible_euler_plane.py`
~L1712, explicitly documented): the SAM moist buoyancy subtracts the horizontal
mean of qv / qcond / θ′ via `hmean_fn = jnp.mean(f, axis=(0,1))`. At n_ranks=1
that is the true domain mean (serial parity holds); under a horizontal
decomposition each rank uses its **slab-local** mean ⇒ the ~6e-4 divergence vs
single-rank.

**This is NOT a bug — it is a deliberate scalability tradeoff** (same convention
the dynamic-Smagorinsky plane average uses). `_moisture_buoyancy_w_half` calls
`hmean_fn` 3× per b_moist (qv, qcond, θ′), and b_moist is evaluated once per RK
stage, so making the mean GLOBAL would cost ~3–9 mpi4jax allreduces/step
(~5–16 % overhead) — which would **degrade the strong scaling this campaign just
fixed**. The rank-local mean is the zero-communication choice and *aligns* with
the scaling goal; the small serial-vs-DD divergence is its price.

Recommendation (design decision for the user, not forced): keep the rank-local
mean as the scalable default; if exact serial parity is needed for oracle /
validation runs, add an OPT-IN `acoustic_moist_global_mean` config flag that
threads `layout` to `_acoustic_moist_buoyancy_w` and uses one BATCHED allreduce
(qv+qcond+θ′ sums packed) — exact, ~3 allreduces/step, off by default. The
dynamics DD path is already exact (1e-15, iter 11); only the moist-mean closure
trades exactness for scalability.

## Iteration 13 (2026-06-08): opt-in exact moist serial parity (implemented)

Added `CompressibleEulerConfig.acoustic_moist_global_mean` (default False). When
True AND multi-rank, `_acoustic_moist_buoyancy_w` uses a GLOBAL horizontal mean
(`global_sum_mpi(local_sum) / (ny_global*nx_global)`, AD-safe) instead of the
rank-local `jnp.mean`. Threaded `layout` through the 3 acoustic-substep fns +
`step_halo`. Validated (np=2, 3 steps, physical moisture):

| flag | u vs single-rank |
|------|-----------------:|
| OFF (default) | 3.85e-5 (rank-local, scalable, unchanged) |
| **ON** | **1.11e-15 (bit-identical serial parity)** |

No regression: plane MPI suite 2 passed, single-rank 17 passed, new
`test_moist_global_mean_exact_serial_parity` passes. Default off → zero impact on
production scaling; opt-in for oracle/validation runs needing exact parity.

**CRM domain-decomposition: fully complete** — fast (123×), correct dynamics
(1e-15), both precisions, moist tradeoff characterized AND given an exact opt-in.

## Iteration 14 (2026-06-08): compressible-plane LES IS MPI-scalable

There are TWO LES paths in legoESM:
1. **Spectral incompressible LES** (`spectral_les_plane`, FFT pressure + LASD
   dynamic SGS) — single-rank (needs a distributed FFT; deferred, multi-node).
2. **Compressible-plane LES** (`compressible_euler_plane` + Smagorinsky SGS) —
   the SAME dycore as the CRM with `smagorinsky_cs > 0`. The iter-8/9 `step_halo`
   fix makes it **MPI-scalable NOW**, and it was already validated bit-identical
   (1e-15, iter 11 used `smagorinsky_cs=0.2`).

DD bench in LES mode (`--smag-cs 0.2`, fixed global 48×48×20):

| precision | np1 | np2 | np4 | np1→np2 |
|-----------|----:|----:|----:|--------:|
| fp32 | 111.4 | 50.6 | 50.0 ms | **2.2× (super-linear)** |
| fp64 | 143.9 | 56.6 | 56.4 ms | **2.5× (super-linear)** |

The SGS compute raises the compute-to-bandwidth ratio, so strong scaling is
SUPER-LINEAR at np=2 (vs 56 % for the bandwidth-bound dry dynamics) before
plateauing at np=4 (bandwidth). Both precisions; correct (iter 11).

⇒ **"LES scaling works for both precisions" is met via the compressible-plane LES
on MPI** (and the spectral LES on single-GPU, iter 4). Only the spectral-LES *MPI*
path remains gated on a distributed FFT.

## Backlog (deferred / large)

1. LES distributed FFT for MPI (the LASD/spectral-pressure blocker).
2. CRM dycore kernel tiling for L2-fit at large domains.
3. (done) batch CRM per-step reductions — `compute_total_water_
   mass_plane_mpi` is a separate allreduce; ideally batch it together with the
   mean-wind reduction (one allreduce/step) in the production CRM step. Re-measure.
2. LES dynamic-SGS (LASD) MPI = distributed FFT (the test filter is a sharp
   spectral cutoff via `rfft2`; planar-mean + box3 are easy, the FFT is not) —
   large, multi-iteration; the spectral LES pressure solve needs it too. Deferred.
3. CRM N256 fp32 GPU degradation (L2 / acoustic-substep) — profile.
4. fp64 is compute-bound for both (consumer GPU); document, no code fix expected.

## Iteration 16 (2026-06-09): spectral-LES MPI foundation — distributed 2-D FFT

Started closing the one genuine remaining gap (spectral-LES MPI = distributed
FFT). The spectral LES funnels the pressure solve, the sharp spectral filter and
the LASD test filter through a global `jnp.fft.rfft2` — the sole single-rank
blocker. Built the distributed 2-D real FFT it needs:
`packages/core/legoesm/parallel/distributed_fft.py`.

- **Slab decomposition along y** (`n_ranks_x=1`): rank owns `(ny_local, nx, nz)`.
  `distributed_rfft2` = local rfft-x → all-to-all transpose y-slab→kx-slab →
  local fft-y (result distributed along kx); `distributed_irfft2` reverses.
  Slab (1-D) is minimal for a 2-D FFT — one transpose / one all-to-all.
- **AD-safe all-to-all** (`ad_alltoall`, `custom_vjp`, comm `nondiff_argnums`):
  an all-to-all is an orthogonal permutation ⇒ adjoint = same all-to-all of the
  cotangent. Keeps the spectral LES end-to-end differentiable (a bare
  `mpi4jax.alltoall` is diagnostic-only here). Complex = two real all-to-alls.
- **kx padding**: the reduced axis `nx//2+1` is zero-padded to `P·ceil(nkx/P)`
  before the transpose, dropped on inverse — `nx//2+1` need not divide by `P`.

Validated (mpirun np=1/2/4, `.venv-mpi`): round-trip identity 1e-12; end-to-end
∂/∂x via distributed FFT == serial `rfft2` derivative 1e-10; **grad == serial
grad 1e-9** (custom-VJP all-to-all correct). np=4 exercises the kx-padding path.
Codex blocked (sandbox bwrap); self-reviewed against the np=2/4 evidence.
Residual risk: not yet under `jax.jit` (eager only) — to cover when wired in.

**Remaining (next increment):** wire `distributed_fft` into `spectral_les_plane`
(pressure Poisson `k²` solve + `_apply_filter` + LASD test filter) under a y-slab
layout, JIT the step, validate a few-step gathered trajectory vs single-rank,
then bench. The hard FFT primitive — the actual blocker — is now done + AD-safe.

## Iteration 17 (2026-06-09): spectral-LES pressure projection runs on MPI

Wired the iter-16 distributed 2-D FFT into the spectral LES and distributed the
**pressure projection** `project()` — the core incompressibility solve — across a
y-slab (`SpectralLESLayout`, `n_ranks_x=1`).

- `make_grid(layout=…)` slices `kx/ky/k2`/masks to each rank's kx-column slab
  (padded to `P·ceil(nkx/P)`); `_fft`/`_ifft` dispatch serial↔distributed on the
  grid; `project()` derives the tridiagonal shape from the spectral arrays
  (`ny_global × nkx_local`) not the physical slab. Pointwise in wavenumber, no
  planar means ⇒ exact.
- Validated (mpirun np=1/2/4): distributed `project()` == serial **1e-9** (u,v,w);
  projected velocity divergence-free **1e-9**; **grad == serial 1e-8**;
  `jax.jit(project)` distributed finite (retires the iter-16 eager-only risk);
  serial spectral-LES unit suite **12/12 unchanged**. 3/2-rule de-aliasing under
  MPI raises `NotImplementedError` (needs a distributed padded FFT — follow-up).

**Remaining for the full spectral-LES step on MPI:** LASD dynamic-SGS test filter
(`lasd_core.py`, sharp spectral cutoff) onto the distributed FFT; wall-model /
`ustar` planar means → `global_sum_mpi`; the 3/2-rule padded FFT; then a gathered
few-step `step()` trajectory vs single-rank + a bench. The hard primitive (the
FFT) and the hardest operator (the projection) are now done + AD-safe.

## Iteration 18 (2026-06-09): full spectral-LES step runs distributed on MPI

Completed the minimal distributed spectral-LES `step()` (FFT iter16 + projection
iter17 + this): a full step runs across a y-slab and matches single-rank.

- `_planar_mean(f, g)`: the three horizontal means (MOST wall stress ⟨|u₁|⟩,
  buoyancy ⟨θ⟩, `u_*` diagnostic) become a GLOBAL `global_sum_mpi` reduction
  under the y-slab (AD-safe) — rank-local means would give a wrong wall stress.
- Guards: LASD dynamic Smagorinsky + 3/2-rule de-aliasing raise under MPI (not
  yet distributed); static smagorinsky/vreman + the 2/3 mask run.
- Validated (np=1/2/4, dealias=False + static Smag + neutral, ab2, 3 steps):
  gathered (u,v,w) == serial **1e-9**; `u_*` == serial scalar **1e-10**; grad ==
  serial (interior ~1e-9, surface k=0 ~1e-5 = wall-mean reduction reorder, proven
  localized); serial unit suite **12/12 unchanged**.
- Bench (global 64²×32, fp64): 30.3 → 30.2 → 28.6 ms/step (np 1/2/4) — functional
  + correct (was single-rank-only); strong scaling flat on one socket
  (bandwidth + all-to-all transpose comm) ⇒ needs multi-node aggregate bandwidth.

**Spectral-LES MPI: from single-rank-only → runs distributed and correct.**
Remaining for the full oracle LES on MPI: distribute the LASD test filter + the
3/2-rule padded FFT (both guarded). The FFT, the projection, the wall model and
the full static-SGS step are done + AD-safe.

## Iteration 19 (2026-06-09): LASD dynamic SGS runs distributed on MPI

Distributed the Bou-Zeid scale-dependent dynamic Smagorinsky (`lasd_cs2`) — the
oracle-faithful closure — onto the y-slab, removing the last guarded operator on
the distributed spectral-LES step (only static SGS ran before).

- `spectral_test_filter`: 2Δ/4Δ sharp test filters on the distributed FFT, kx
  cutoff on the local kx slab via global column indices.
- `imfilter_box3`: ∂x roll local; ∂y roll via an AD-safe 1-row ring halo
  (`_get_sendrecv_vjp`, tags 700/701, n=2-safe).
- `pm` planar mean → `global_sum_mpi` (AD-safe).

**Codex review caught a HIGH bug** (the standing always-codex rule paid off): the
test-filter cutoffs `cut1y/cut2y` were taken from `uc.shape[0]` = ny_LOCAL under
the slab, shrinking them per rank (NY=12,np=4 → `cut2y=0`, zeroing the level-2 y
filter). My relaxed step test masked it. Fixed → cutoffs from `layout.ny_global`;
added a DIRECT `lasd_cs2` serial-vs-distributed regression on a well-conditioned
smooth field (matches ~1e-12). Codex round-2: CLEAN.

Validated np=1/2/4: test_filter 1e-10, box3 1e-13 (grad 1e-10), `lasd_cs2` 1e-9,
full dynamic-SGS step 1e-4; serial 12/12 unchanged.

**Spectral-LES MPI now runs the FULL oracle closure (dynamic LASD), AD-safe.**
Only the 3/2-rule de-aliasing remains serial-under-MPI (guarded; needs a
distributed padded FFT). Methodology note: random unphysical strain is a BAD test
for `lasd_cs2` (quintic β-root hypersensitivity masks cutoff bugs behind the
non-bit-identical FFT) — use smooth well-conditioned fields for the direct check.

## Iteration 20 (2026-06-09): distributed spectral-LES scaling characterized (both precisions)

With the full distributed spectral LES working (dynamic LASD, iter16-19), measured
its weak + strong scaling, both precisions, via the new
`scripts/bench/bench_spectral_les_dd_scaling.py` (dynamic-SGS, y-slab, .venv-mpi).

**Strong** (global 64²×32, dynamic LASD):

| precision | np1 | np2 | np4 | speedup @4 |
|-----------|----:|----:|----:|-----------|
| fp64 | 88.9 | 77.9 | 74.6 ms | 1.19× (~30% eff) |
| fp32 | 60.6 | 51.9 | 41.0 ms | 1.48× (~37% eff) |

**Weak** (per-rank 32×64×32):

| precision | np1 | np2 | np4 |
|-----------|----:|----:|----:|
| fp64 | 40.3 | 81.2 | 136.3 ms |
| fp32 | 29.5 | 51.9 | 98.7 ms |

- **Both precisions run** the full oracle closure distributed; fp32 ~1.5× faster
  and strong-scales better (the LASD test-filter FFTs add compute that partly
  hides the bandwidth wall).
- **Weak scaling is poor — and inherently so.** A pseudo-spectral solver has a
  GLOBAL coupling every step (the pressure-Poisson FFT spans the whole domain),
  so growing the domain with the rank count grows both the FFT size (O(N log N))
  and the all-to-all TRANSPOSE volume. This is the well-known spectral-method
  communication wall, fundamentally different from the CRM's local-stencil halo
  (which weak-scales flat). Production spectral codes hit the same wall and lean
  on specialized FFT libraries + fat interconnects; on a single socket the
  all-to-all is bandwidth-bound.
- **Strong scaling is modest** (bandwidth + all-to-all comm), best in fp32.

⇒ The distributed spectral LES is FUNCTIONAL + CORRECT + AD-safe in both
precisions; its scaling ceiling is the spectral FFT all-to-all (algorithmic), not
a code defect. The CRM (local stencil) remains the better-scaling path; the
spectral LES is the faithful-physics path now usable across ranks.

## Iteration 22 (2026-06-09): 3/2-rule de-aliasing distributed — spectral-LES MPI COMPLETE

Distributed the last serial-under-MPI operator: the oracle-faithful 3/2-rule
de-aliasing. The distributed spectral LES now uses the SAME de-aliasing as the
serial/oracle path (not just the 2/3 mask). User-requested accuracy refinement.

- `distributed_fft.py`: `distributed_pad_to_fine` / `distributed_truncate_from_fine`.
  The 2-D spectral zero-pad is SEPARABLE → x done LOCALLY (undecomposed), y via one
  all-to-all transpose (`ad_alltoall`); two 1-D inverse-FFT norms compose to the
  serial 2-D `irfft2` norm. Requires ny_local even (guarded).
- `_pad_to_fine`/`_truncate_from_fine` dispatch serial↔distributed on the grid.

**Codex caught a 2nd HIGH bug** (always-codex earns its keep again): `scalar_rhs`
(θ advection) also de-aliases and still omitted `g` → padded only the local slab
under MPI+dealias+θ. My no-θ test missed it. Fixed + added a θ+buoyancy dealias
test. Codex round-2: CLEAN.

Validated np=1/2/4 (NY=16): pad/truncate 1e-10, grad 1e-8, dealias=True step 1e-9
(momentum + θ); serial 12/12 unchanged.

**SPECTRAL-LES MPI NOW COMPLETE** — the full oracle closure runs distributed and
AD-safe: distributed 2-D FFT, pressure projection, global-mean wall model, dynamic
LASD SGS, AND 3/2-rule de-aliasing. No operator remains serial-under-MPI. Scaling
remains FFT-all-to-all-bound (algorithmic, iter20) — the ceiling is the spectral
method's global coupling, not any missing capability.

## Iteration 23 (2026-06-09): full certification gate — 21/21 np=2 AND np=4

Ran the complete distributed plane-MPI suite (CRM halo + spectral-LES FFT,
projection, step, LASD, 3/2-dealias) together at np=2 and np=4.

- Surfaced a CRM-DD test-config gap (NOT a regression — the spectral/FFT commits
  never touch CRM halo code): the halo test hard-coded a 1×n slab, giving
  nx_local=3 at np=4 (< the del4 stencil floor of 4). Fixed to a balanced 2×2
  pencil at np=4 → certifies the directional ring-shift halo on BOTH axes at once
  (stronger than the np=2 1×2). CRM DD now 1e-15 at np=4 too.
- **Final state: 21/21 pass at np=2 AND np=4.** The whole distributed plane stack
  — CRM domain decomposition (bit-identical) and the spectral LES full oracle
  closure (FFT + projection + wall model + dynamic LASD + 3/2-rule de-aliasing,
  all AD-safe) — is certified across rank counts.

**Campaign complete.** Both CRM and LES weak+strong scale at the achievable scope
in fp32 and fp64; the spectral-LES MPI gap is closed end-to-end (was the single
biggest software gap). All genuinely-remaining levers are hardware-bound
(multi-node/multi-GPU, consumer-fp64 1/64, single-socket bandwidth) or
deep-low-ROI (CRM N256 GPU L2 tiling). No distributed operator remains
unimplemented or unvalidated.

## Derecho CPU, 1–4 nodes (2026-11-20 → 11-23): CRM weak + strong, fp32 / mixed / fp64

Plane CRM `step_halo`, RCEMIP1 production deck (`scripts/run/run_rcemip1_crm.sh`
numerics, nlev 64, semi-implicit, substepped horizontal acoustics, Morrison),
one rank per core, single-threaded XLA, `scripts/cluster/crm_les_scaling/crm_les_ladder.pbs`
(`PPN_MAX` spreads ranks over nodes). Weak: 64×64×64 per rank. Strong:
512×512×64 global. ms/step = median of 3 windows × 5 steps, dycore only
(column physics timed separately per rank; tile-only). "Packed" = the
no-communication ceiling: independent 1-rank copies at the same ranks per
node (`packed_node_probe.pbs`; the halo code path gives the same numbers,
748 vs 746 ms fp32 at 32/node). Mixed = fp32 state, fp64 acoustic substeps,
microphysics inputs, radiation and fixer sums.

Code: tracer tendency fused per tracer under `lax.map` (bd09bfe38e);
rows marked † were measured after the fused fixer allreduce and the
crm-mpi-gaps merge (26dc651af2; that change alone measured 0.6–1.6 %
faster at 32–64/node; its effect at 128/node is unmeasured). Every
unmarked cell in both tables predates it, so any comparison pairing a †
cell with an unmarked one mixes the two code versions.

### Weak scaling (ms/step; efficiency vs the packed ceiling at the same ranks/node)

| Nodes × ranks/node | ranks | fp32 | mixed | fp64 | packed fp32 / mixed / fp64 | eff fp32 / mixed / fp64 |
|---|---|---|---|---|---|---|
| 1 × 1 | 1 | 673 | 1227 | 1385 | — | — |
| 1 × 16 | 16 | 830 | 1317 | 1588 | — | — |
| 1 × 32 | 32 | 966 | 1605 | 1961 | 746 / 1337 / 1607 | 77 / 83 / 82 % |
| 4 × 32 | 128 | 994 | 1654 | 2005 | 746 / 1337 / 1607 | 75 / 81 / 80 % |
| 1 × 64 | 64 | 1389 | 2432 | 3202 | 1077 / 1806 / 2394 | 78 / 74 / 75 % |
| 2 × 64 | 128 | 1410 | 2435 | 3268 | 1077 / 1806 / 2394 | 76 / 74 / 73 % |
| 4 × 64 | 256 | 1413 | 2449 | 3306 | 1077 / 1806 / 2394 | 76 / 74 / 72 % |
| 1 × 128 | 128 | 2776 | 4758† | 6212 | 2541 / 4154 / 5539 | 92 / 87 / 89 % |
| 2 × 128 | 256 | 2834 | 4860† | 6313† | 2541 / 4154 / 5539 | 90 / 85 / 88 % |
| 4 × 128 | 512 | 2817 | 4860† | 6319† | 2541 / 4154 / 5539 | 90 / 85 / 88 % |

Against the 1-rank time the 128/node efficiency is 24 % (fp32): 88.8 % of
the excess step time already occurs in the packed control (1 → 128 copies:
673 → 2541 ms), consistent with node memory-bandwidth contention; the rest
is the halo-vs-packed gap (2776 vs 2541 ms). Peak RSS 1.6–3.0 GB/rank.

### Strong scaling, 512×512×64 (ms/step)

| Nodes × ranks/node | ranks | fp32 | mixed | fp64 |
|---|---|---|---|---|
| 1 × 1 | 1 | 35643 | 51458 | 54660 |
| 1 × 16 | 16 | 5888 | 5503 | 6572 |
| 1 × 32 | 32 | 2840 | 3775 | 4508 |
| 1 × 64 | 64 | 1392 | 2428 | 3200 |
| 1 × 128 | 128 | 1180 | 1963 | 2662 |
| 2 × 64 | 128 | 594 | 889 | 1241 |
| 4 × 32 | 128 | 425 | 567 | 740 |
| 2 × 128 | 256 | 713 | 834 | 1191 |
| 4 × 64 | 256 | 607 | 392 | 530 |
| 4 × 128 | 512 | 363 | 461† | 544† |

fp32 speed-up 1 → 512 ranks: 98× (19 % of ideal); 16 → 512: 16× (51 %).
Single unrepeated runs with orderings that do not fit the rest of the
table, not used in any conclusion: the whole 4 × 64 row (fp32 607 is slower
than mixed 392 and than 4 × 32 fp32 425) and the 1 × 16 mixed cell (5503,
faster than fp32 5888 although mixed does more work; the fp32 cell that
feeds the 16 → 512 figure scales as expected from 1 × 32).
Spreading the same rank count over more nodes gave the largest gain
measured here (128 ranks, fp32: 1180 → 425 ms, 2.78×, from 1 to 4 nodes),
consistent with the packed control's finding that per-rank memory bandwidth
dominates on a full node.

### Decomposition of the halo-vs-packed gap at 32/node (fp32, 128 ranks on 4 nodes, perfetto diff MPI rank 0 vs a packed copy)

| term | ms/step |
|---|---|
| blocking mpi4jax sendrecv (312 calls, incl. neighbour waits) | +92 |
| compute fusions split/slower around the custom calls | ≈ +90 |
| halo copies / concatenates (net) | ≈ +25 |
| dry-mass fixer allreduces (now one allreduce) | ≈ +6 |
| not attributed to an op (host dispatch, gaps between ops) | ≈ +30 |

Op terms are CPU time summed per op class over 2 traced steps (992 ms/step
on that run vs 748 packed, gap 244 ms).

Levers measured in same-job A/Bs (128 ranks, ms/step, base → lever):
- rejected, in-place halo assembly: 1 node fp32 2769 → 2900.
- rejected, 42 slow-tendency exchanges/step grouped into 3 packed rounds:
  32/node fp32 991 → 1217, fp64 1999 → 2496; 64/node fp32 1409 → 1883.
- rejected, communication-avoiding acoustics (one wide exchange per stage):
  32/node fp64 2005 → 1850 but fp32 992 → 1008, 64/node fp32 1409 → 1750,
  1 node fp32 2759 → 3814; state not bit-identical.
- kept, fused per-tracer tendency: 1 node fp32 4168 → 2769, fp64 8684 →
  6176; bit-identical.
- kept, fused fixer allreduce: 32/node fp32 994 → 985, fp64 2004 → 1992;
  64/node fp32 1409 → 1387; bit-identical.
The remaining 32–64/node gap is the observed overhead of the tested
blocking halo implementation; further work was stopped by vote, with an
XLA-native collective path recorded as the open alternative.

Weak fp64/mixed at 128/node across nodes could not time column physics: the
steady state (~1.9 GB/rank) leaves no room for the radiation compile within
230 GB/node; those rows are dycore-only (`TIME_PHYSICS=0`).

## Derecho CPU campaign 2026-11 (spectral LES, final — optimisation stopped)

Spectral plane LES, production LES-suite config (dry CBL, dynamic LASD SGS,
3/2 de-aliasing, RK3), Derecho CPU nodes (128 cores, one single-threaded rank
per core), branch `perf/les-plane-fft` at `e55ef4b4` (`8248140b` only changes
the bench's layout selection). Weak: 64x64x128 per rank. Strong: 512x512x128
total. Efficiency = T1 / T(N) (weak) or T1 / (N T(N)) (strong) against the
1-rank run of the same code in the planes layout (the fastest 1-rank path;
pencil at 1 rank runs the serial core), for both columns: weak fp32 1197 ms, fp64
1998 ms; strong fp32 81.9 s, fp64 142.3 s per step. "Ceiling" = packed-node
control (independent 1-rank copies, no communication) at the same ranks per
node: it is the memory-bandwidth limit of the 1-rank workload.

Layouts: *pencil* (two all-to-alls per transform on row / column
sub-communicators) and *planes* (one all-to-all over all ranks to whole
horizontal planes of a z-chunk). The bench default `--decomp auto` picks
planes when every rank shares one node, else pencil (the faster one at every
point below).

Weak efficiency, flat time / judged against N log N FFT growth
(`x log2(4096 N)/12`, generous: only ~half the step is FFTs):

| ranks / per node / nodes | pencil fp32 | pencil fp64 | planes fp32 | planes fp64 | ceiling fp32 / fp64 |
|---|---|---|---|---|---|
| 32 / 32 / 1 | 42 / 60% | 37 / 52% | **54 / 77%** | **47 / 67%** | 95 / 88% |
| 64 / 32 / 2 | **35 / 53%** | **31 / 46%** | 30 / 45% | 27 / 40% | 95 / 88% |
| 128 / 32 / 4 | **29 / 45%** | **25 / 40%** | 22 / 34% | 20 / 31% | 95 / 88% |
| 64 / 64 / 1 | 24 / 36% | 19 / 29% | **32 / 47%** | **26 / 39%** | 81 / 59% |
| 128 / 64 / 2 | **21 / 33%** | **16 / 26%** | 14 / 23% | 13 / 20% | 81 / 59% |
| 256 / 64 / 4 | **17 / 29%** | **13 / 22%** | 6 / 9% | 4 / 7% | 81 / 59% |

Strong efficiency:

| ranks / per node / nodes | pencil fp32 | pencil fp64 | planes fp32 | planes fp64 |
|---|---|---|---|---|
| 32 / 32 / 1 | 35% | 34% | **45%** | **45%** |
| 64 / 32 / 2 | **38%** | **34%** | 32% | 30% |
| 128 / 32 / 4 | **34%** | **32%** | 22% | 23% |
| 64 / 64 / 1 | 26% | 22% | **34%** | **29%** |
| 128 / 64 / 2 | **25%** | **22%** | 14% | 15% |
| 256 / 64 / 4 | **21%** | **19%** | 4% | 4% |

Where the time goes (all-rank traces, weak fp32, 64 ranks per node, ms/step;
slowest and fastest rank within ~0.4 s, so not load imbalance). "step" is the
untraced wall time; the components are summed per-op trace times, which run
~4-9% longer under the profiler, so they add up to slightly more:

| run | step | FFT | local copies | all-to-all | rest |
|---|---|---|---|---|---|
| 1 rank, planes | 1061 | 589 | ~260 | 53 (self copy) | ~200 |
| 1 node, planes | 3713 | 1152 | 911 | 1439 | 550 |
| 1 node, pencil | 4677 | 1032 | 1186 | 1996 | 671 |
| 2 nodes, pencil | 5455 | 1044 | 1144 | 2968 | 640 |
| 2 nodes, planes | 8195 | 1374 | 999 | 5589 | 690 |

Bandwidth-limit estimate (PLAUSIBLE: node memory 230 GB/s measured, NIC
25 GB/s per node nominal, all-to-all send ~1.0 GB/rank/step planes, ~1.9 GB
pencil): compute at the packed ceiling plus the all-to-all volume at those
limits gives a best reachable weak fp32 efficiency of ~59% (1 node, planes),
~47% (1 node, pencil) and ~34% (2 nodes, pencil) at 64 ranks per node. On
this estimate (same PLAUSIBLE label) the all-to-all volume per step is the
binding term, and 90% would need fewer all-to-all bytes per step (an
algorithm change), which was not pursued. A byte count
from the compiled HLO shapes was rejected as an instrument (it implied
24 GB/s per core with 64 ranks sharing a node).

What moved the numbers (same exclusive node, before -> after):
- level-major LASD body + spectral divergence (c180b86): 1-rank fp64 step
  4811 -> 2612 ms, fp32 3733 -> 1943 ms (pinned weak bench).
- planes FFT mode + level-major packing (6d9e51d, aa638c1): 1-node weak fp32
  at 32 ranks ~4230 ms (pencil, 2e94c07) -> 2665 ms; the packing alone took a
  4-rank planes step 6038 -> 2079 ms.
- shared forward FFT for the 2-delta / 4-delta LASD filters (d9d3904):
  1-rank -6% fp32, -5% fp64.
- horizontal velocity gradients computed once per rhs (e80dfe7): 1-rank fp32
  1262 -> 1059 ms, fp64 1788 -> 1560 ms.
- level-major pencil transforms (e55ef4b): 64 ranks on 1 node pencil
  5780 -> 4652 ms.

Known limits: planes mode at 128 ranks per node fp64 runs out of memory
(cgroup 230 GB/node; full-spectrum wavenumber constants per rank). LASD
serial / distributed parity is round-off (1e-12 x64; the earlier 1e-4 was
only the tolerance). Every result row is stamped with git SHA and clean-tree
flag in `/glade/derecho/scratch/pg2328/crm_les_scaling/les_cmp_*/les.jsonl`.
