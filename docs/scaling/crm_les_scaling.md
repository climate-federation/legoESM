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
