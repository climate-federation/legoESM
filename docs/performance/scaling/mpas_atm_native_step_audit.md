# MPAS-atmosphere NATIVE (ppermute SPMD) step — production-support audit (2026-07-13)

Scaling-M3c increment-1 deliverable. User directive audited: "Native
MPAS-atmosphere ppermute code exists but production uses blocking mpi4jax
and the native path lacks full production state/physics support."

**Verdict: correct on both counts (pre-increment).** The native path is
`make_voronoi_sharded_step` (`packages/core/legoesm/parallel/sharded_dynamics.py`,
shard_map + `jax.lax.ppermute`, single- and multi-controller); production
(ModelDriver, `packages/coupler/legoesm/driver/model_driver.py:5217`) drives
route-A `make_voronoi_mpi_step` (`packages/core/legoesm/parallel/voronoi_mpi.py:618`,
mpi4jax sendrecv — latency-bound, no comm/compute overlap). Route-A had full
production support; the native path had a dry-dynamics subset.

## Gap table (pre-increment state, line numbers at main 04265c79e)

Native = `make_voronoi_sharded_step` (sharded_dynamics.py:1721);
production reference = serial `MPASPrimitiveEquationModel._step_jit`
(`packages/atmosphere/legoesm/atmosphere/dynamics/primitive_eq_mpas.py:725`)
and route-A `make_voronoi_mpi_step` (voronoi_mpi.py:618).

| # | Item | Native (pre) | Production reference | Increment-1 |
|---|------|--------------|----------------------|-------------|
| 1 | Tracers in halo exchange | absent — cell pack = T/p_s/phis only (sharded_dynamics.py:1922-1927) | route-A batched union-neighbor exchange incl. all tracers (voronoi_mpi.py:778-832) | **CLOSED** — tracers ride the packed cell buffer in canonical sorted wire order |
| 2 | Tracer advection in RK | dropped (1984-2006; RK advances u/T/p_s only, 2099-2115) | pytree RK carries tracer advection tendencies (primitive_eq_mpas.py:764-787; voronoi_mpi.py:889-913) | **CLOSED** — tendency-shaped state pytree incl. tracers through `dispatch_integrator` |
| 3 | Integrator | hard-coded SSP-RK3 (2099-2115); config default `ssp_rk54_scan` silently overridden | `dispatch_integrator(cfg.time_integrator)` (primitive_eq_mpas.py:785; voronoi_mpi.py:935) | **CLOSED** — same dispatch as serial |
| 4 | Forcing | not threaded; physics called with `forcing=None` (2133) | traced jit arg, SegmentForcing doctrine (primitive_eq_mpas.py:731,750-751; voronoi_mpi.py:916-927) | **CLOSED** — traced jit arg; steady-state no-retrace gated |
| 5 | phys_state carry | refused loudly (2134-2146, 2192) | threaded + returned, `return_phys_state=True` (voronoi_mpi.py:945-959,1012-1027) | **CLOSED** — `return_phys_state=True` factory kwarg mirrors route-A; refusals kept loud |
| 6 | Physics tracer tendencies | not applied (2147-2149) | applied (voronoi_mpi.py:973-982) | **CLOSED** |
| 7 | Tracer non-negativity floor | missing (2152-2153 T only) | present (voronoi_mpi.py:990-994) | **CLOSED** |
| 8 | compute/storage precision cast | missing | both ends (voronoi_mpi.py:934,1004; serial 760,848) | **CLOSED** — `cast_pytree` mirror; f64 mass-budget accumulator added |
| 9 | Local-only metadata | ALL n_dev local meshes + ppermute index arrays REPLICATED per device (1858-1891); global `areaCell` replicated closure (2075) | route-A rank holds only `layout.local_mesh` (voronoi_mpi.py:721) | **CLOSED for dynamics** — stacked local meshes, halo schedule, and areaCell are `P("device")`-sharded jit ARGUMENTS (multi-controller-safe where sharded closures raise); each device holds only its slice. REMAINDER: the operator-split physics term (below) |
| 10 | Schema tripwire | none — a new HydrostaticState field silently unexchanged | ocean twin `_expected` set (voronoi_mpi.py:379-390) | **CLOSED** — `check_voronoi_spmd_state_schema` + synthetic-violation self-test |

## What landed (increment 1)

* `make_voronoi_sharded_step` rewritten (same public factory; signature
  gains `return_phys_state: bool = False`):
  full state through the packed ppermute/allgather exchange (one flat
  payload per neighbor round carrying u + T + p_s + phis + all tracers —
  the SPMD mirror of M3d's `exchange_state_mpas_ocean` batched pattern),
  `dispatch_integrator` dynamics with tracer advection, operator-split
  physics with traced `forcing` + prognostic `phys_state` carry, tracer
  floors, compute/storage casts, fp64 fused mass fix, local-only
  dynamics metadata via sharded jit args.
* `_ppermute_halo_fill` factored module-level so the exchange is
  independently testable (five-field sentinel routing gate).
* Bench `bench_mpas_spmd_scaling.py`: `--physics kessler` (moist BCW
  tracers on the timed/gated path; tracer parity gate).
* Gates: `tests/parallel/test_mpas_atm_native_step.py` (parity vs serial
  for moist kessler × {ppermute, allgather}, traced-forcing no-retrace,
  stateful-carry threading, integrator dispatch, sentinel routing,
  10-step scan dtype fixed point, refusal contracts, route-A np=1
  cross-check) + a second (moist) np=2 multicontroller selfspawn test.

## Honest remainders (NOT closed in increment 1)

1. **Physics-term metadata is not local-only.** The operator-split
   physics runs OUTSIDE shard_map on the GSPMD-sharded global state with
   the replicated global-mesh closure (`sharded_dynamics.py`, physics
   block in `_build_step`). Column-local AMIP physics reads only 1-D
   cell fields (latCell/lonCell/areaCell — O(nCells) scalars, not the
   2-D connectivity), so the memory cost is small, but a strict
   per-process-local physics would need the physics evaluated inside
   shard_map on the owned shard with per-device mesh columns. Deferred.
2. **Single-device `return_phys_state=True` is refused** (ValueError):
   `model.step` stashes the carry eagerly on the model and cannot honor
   the `(state, carry)` return contract under an outer trace (gh-417).
   Serial carry threading stays the model/driver's own contract.
3. **`anchor_mass_to_initial` is not implemented** on the native path
   (mass fix is the per-step telescoping old-vs-new correction) — same
   as route-A `make_voronoi_mpi_step`. With the fixer applied every
   step the trajectories agree to the reduction-order floor; the anchor
   only suppresses the fp random walk of the pinned value.
4. **ModelDriver still selects route-A** for MPAS MPI production runs.
   Swapping the driver to the native path (and the NCCL 8→16 GPU
   re-measurement from the audit) is the next increment.

## Parity envelope (measured, x64, subdiv 3, nlev 4, 2 devices)

Documented tolerance vs serial (floating-point re-association floor of
the sharded step — halo-cell operator coverage + mass-fix allreduce
ordering; same envelope as `test_voronoi_sharded_equivalence.py`):
u/T atol 1e-6, p_s atol 1e-1, tracers atol 1e-9. The np=2
multicontroller moist bench gate passes with the same
`MPAS_PARITY_TOLS` (T-tier tolerances for tracers, atol scaled 1e-3).
