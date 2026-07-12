# MPAS-ocean distributed step — stage-correctness audit (2026-07-11)

Scaling-M3d increment-1 deliverable. Reconciles two claims:

* `SCALING_STATUS_AUDIT.md` (ocean matrix, MPAS/Voronoi row, 2026-07-07):
  "`voronoi_mpi` step exists (`make_voronoi_mpi_step`); MPI conservation
  tested; NO scaling bench lane drives it".
* User directive: "a complete stage-correct distributed MPAS-ocean step does
  not yet exist".

**Verdict: the user directive is correct; the status row was stale/conflated
on three counts.**

1. `make_voronoi_mpi_step` (`packages/core/legoesm/parallel/voronoi_mpi.py:618`)
   is the **atmosphere** MPAS step (`MPASHydrostaticState`; per-RK-stage
   packed halo refreshes + owned-cell mass fixer). It cannot step
   `MPASOceanState`. The OCEAN has no equivalent wrapper.
2. A bench lane DOES exist since audit item 6 landed:
   `scripts/bench/bench_ocean_mpas_scaling.py` steps `MPASOceanModel` on
   `layout.local_mesh` multi-rank with parity + conservation gates and
   partition metrics — but the step it drives is not stage-correct (below),
   and the lane lacked the M1 measurement contract (fused-scan
   `timed_scan_blocks`, solver-residual reporting, wet-cell metrics) until
   this increment.
3. "MPI conservation tested" for the OCEAN reduces to: the np=2
   scatter/gather roundtrip (`tests/ocean/distributed/test_mpas_ocean_scatter.py`),
   the distributed-PCG solver parity
   (`tests/ocean/distributed/test_barotropic_pcg_mpas_mpi.py` — solver-level,
   its own docstring: "the full ocean-state scatter is a separate follow-on"),
   and the bench lane's own `--check-conservation` gate.
   `tests/ocean/distributed/test_ocean_mpi_conservation.py` is CUBED-SPHERE,
   not MPAS.

## Spec audited against

"Packed stage-level cell/edge/vertex halo refreshes and owned-only
reductions." For every stage of `MPASOceanModel._step_impl`
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:350`) under an
armed `VoronoiPartitionLayout` (rank-local mesh, `halo_depth=2`):
which halo exchanges happen, are they packed or per-field, and are global
reductions owned-only?

Key mechanics that bound correctness:

* `build_local_mesh` (`packages/core/legoesm/parallel/voronoi_partition.py:781`)
  remaps out-of-partition connectivity to `-1`; TRiSK operators mask those
  entries. So a local stencil op produces MASKED-WRONG (deterministic, not
  garbage) values on the outermost halo ring; each additional stencil hop
  propagates the wrongness one ring inward. With `halo_depth=2`, owned cells
  stay exactly correct through **2 stencil hops after the last halo
  refresh** — beyond that, owned cells adjacent to the partition boundary
  silently diverge from serial.
* The only packed full-state exchange machinery in the tree is the
  atmosphere's `_exchange_mpas_state`
  (`voronoi_mpi.py:778` batched / `voronoi_mpi.py:840` legacy per-entity).
  Until this increment the ocean had NO state exchange helper at all; the
  only ocean halo traffic was the per-field cell exchange inside the
  distributed PCG matvec. This increment adds the packed ocean twin
  `exchange_state_mpas_ocean` (`voronoi_mpi.py`, next to
  `scatter_state_mpas_ocean`) — one batched union-neighbor message per
  neighbor per dtype group for u (edge) + T, S, eta, w (cell).
* Vertex quantities (PV, curl) are recomputed locally from edge `u` each
  evaluation (`curl_vertex_3d`); no vertex halo exchange exists anywhere
  (atmosphere included) and none is needed PROVIDED edge halos are fresh and
  the vertex consumer sits within the hop budget — vertex values on the
  outer ring are masked-wrong and count as one hop.

## Stage table — `MPASOceanModel._step_impl` under a Voronoi partition

Hops = horizontal stencil depth consumed by the stage (rings of halo
correctness eaten since the last refresh). Verdicts assume fresh halos at
step ENTRY (which nothing guaranteed before this increment: the lane now
refreshes per step; within-step staleness remains).

| # | stage | file:line | stencil hops | halo refresh in stage | global reductions | verdict |
|---|-------|-----------|--------------|----------------------|-------------------|---------|
| 0 | scatter / initial fill (`scatter_state_mpas_ocean`) | `voronoi_mpi.py:300` | — | initial (slice = fresh) | — | CORRECT; np=2 roundtrip-tested |
| 1 | baroclinic tendencies (`mpas_ocean_baroclinic_tendencies`) | `ocean_pe_mpas.py:92` (called `ocean_model_mpas.py:401`) | up to 4 (B_h del4 `ocean_pe_mpas.py:585`; K_bih grad→div→grad→div `:846-854`; PV/curl/tangential 1–2; fills 1) | NONE | none on default path; `normalize_freshwater` rank-local mean is REFUSED multi-rank at the source (`ocean_pe_mpas.py:947-968`) | NOT stage-correct: consumes >2 hops with no refresh; owned boundary cells diverge |
| 2 | tracer forward-Euler + land fill | `ocean_model_mpas.py:406-415` | 1 (`fill_land_cells_mpas`) | NONE | — | eats 1 more hop of stage-1 output |
| 2a | implicit vertical tracer diffusion + KPP/TKE/conv K-profiles | `ocean_model_mpas.py:432-521` | 0 (column-local; EOS cell-local) | not needed | — | stage-correct given fresh inputs |
| 2b | GM/Redi + MLE bolus tendencies | `ocean_model_mpas.py:529-553` | 1–2 | NONE | — | further hop consumption |
| 3 | momentum Euler + implicit vertical viscosity | `ocean_model_mpas.py:560-641` | 1 (`min_cell_to_edge`) | NONE | — | |
| 3b | forward-backward Coriolis on u' | `ocean_model_mpas.py:59-136` (called `:647`) | 2 (`tangential_velocity_3d` ×2, `edgesOnEdge`) | NONE | — | |
| 4 | barotropic EXPLICIT substeps (`barotropic_substeps_mpas`) | `barotropic_mpas.py:48-382` | 2–6 PER SUBSTEP (div 1 + grad 1 + tangential 1–2 + optional div-damp/visc 2 each) × `n_barotropic_substeps` (default 30) | NONE inside the `lax.scan` | eta-floor clamp owned-masked + forced-global (`barotropic_mpas.py:122-126,272,357` via `eta_floor.py:53`) | reductions owned-only CORRECT; stencils NOT stage-correct multi-rank (≫2 hops); **no multi-rank refusal on this branch** (unlike implicit_cn) |
| 4' | barotropic IMPLICIT CN (`barotropic_implicit_mpas`) | `barotropic_implicit_mpas.py:376-794` | predictor ~3 before the solve; PCG matvec halo-composed per iteration | per-field CELL exchange inside every `A_op` (`:609-610`); solution halo refreshed post-solve (`:634`) | entry refusal for layout-less multi-rank (`:437-444`); owned-masked area-weighted dots (`:615-616`); owned-masked mass projection, ONE batched allreduce (`:665-684`); owned-masked clamp (`:695-699`); owned-masked residual (`:709-719`) | solver INTERNALLY stage-correct (np=2 parity-tested); its predictor/RHS consume stale input halos like every other stage |
| 5–6 | layer thickness + `reconcile_3d_velocity` | `ocean_model_mpas.py:753-786`, `barotropic_mpas.py:385-415` | 1 | NONE | — | |
| 7 | transport-consistency correction `delta_u` | `ocean_model_mpas.py:799-807` | 0 (edge-local) | NONE | — | |
| 8 | w diagnosis (`divergence_cell_3d`) | `ocean_model_mpas.py:819` | 1 | NONE | — | |
| 9 | flux-form tracer advection (upwind/TVD) | `ocean_model_mpas.py:838-890` | 1 (upwind) / 2 (TVD upup) | NONE | — | |
| 10 | conservation fixer | `ocean_model_mpas.py:914-972` + `conservation_mpas.py:59-97` | 0 | — | owned mask pulled from the matching layout (`ocean_model_mpas.py:923-941`); expected-forcing sums `global_sum_mpi` (`:960-966`); fixer core via `global_sum_if_distributed` | owned-only CORRECT |
| 11 | freeze floor | `ocean_model_mpas.py:974-1002` | 0 | — | — | correct |
| — | `check_barotropic_cfl` / `_assert_runtime_invariants` | `ocean_model_mpas.py:297-332,1048-1107` | — | — | rank-LOCAL host reductions | diagnostics-only (warnings/checks may differ per rank); acceptable, documented |

## Findings

1. **Owned-only reductions: PASS.** Every collective on the step path is
   owned-cell-masked (eta-floor clamp, implicit-PCG dots/projection/residual,
   conservation fixer) or refused multi-rank at the source
   (`normalize_freshwater`). No double-count bug found — nothing to fix here.
2. **Stage-level halo refreshes: FAIL (structural).** No stage of the ocean
   step refreshes state halos; the only in-step exchange is the per-field
   cell exchange inside the implicit-PCG matvec. The cumulative hop count of
   one step (≥10 even without the explicit substep loop; hundreds with it)
   vastly exceeds `halo_depth=2`, so a multi-rank step silently diverges from
   serial near partition boundaries at a rate bounded by field evolution per
   step (slow flows ⇒ small parity error — why the smoke-window parity gate
   measured ~9e-9 at L2×2 steps; that gate bounds staleness, it does not
   prove stage correctness).
3. **Explicit-substep branch has no multi-rank guard** while `implicit_cn`
   refuses a layout-less multi-rank launch
   (`ocean_model_mpas.py:727-739`, `barotropic_implicit_mpas.py:437-444`).
   The bench lane is the intended multi-rank driver of that branch and now
   labels every row with `halo_refresh` + `stage_halo_correct=false` so a row
   cannot masquerade as a stage-correct scaling claim.

## Fixed in this increment (small, gated)

* Packed full-state ocean halo refresh helper `exchange_state_mpas_ocean`
  (`voronoi_mpi.py`, batched union-neighbor exchange; schema tripwire matching
  the scatter/gather pair; identity at np=1). Unit-tested single-process;
  exercised np=2 by the bench lane.
* Bench lane (`scripts/bench/bench_ocean_mpas_scaling.py`) extended to the M1
  measurement contract: fused-scan `timed_scan_blocks` timing (cross-rank
  MAX-reduced), `wet_cell_metrics`, solver-iteration + zero-forcing
  Helmholtz residual probe (`barotropic_implicit_mpas(..., return_residual=True)`
  outside the timed loop), `--barotropic-solver` selection, and a per-step
  packed halo refresh (`--halo-refresh auto|per_step|none`, default auto ⇒
  per_step at np>1) so multi-rank rows pay representative exchange cost and
  the step INPUT is fresh each step. Rows are self-describing:
  `stage_halo_correct=false` until the per-stage refreshes land.

## Deferred (structural, precisely scoped)

Threading per-stage refreshes into `_step_impl` is the real distributed-step
milestone, NOT a small fix:

* stage 1 entry exchange (u,T,S,eta packed — now available as
  `exchange_state_mpas_ocean`), plus `halo_depth ≥ 4` or operator-splitting
  of the del4/K_bih stencils;
* per-substep (or wide-halo, `halo_depth ≥ 2·n_substeps`-style) exchange of
  `(eta, u_bar)` inside `barotropic_substeps_mpas`'s scan — the same
  trade-off the lat-lon wide-halo split-explicit lane measures
  (`SCALING_STATUS_AUDIT.md` improvement candidate 3);
* post-barotropic and pre-tracer-advection refreshes of `(u_3d, eta)`;
* thread `owned_mask` into the freshwater normalization means to retire the
  multi-rank refusal (`ocean_pe_mpas.py:947`; helper support already exists —
  `freshwater.py:181-261`).

Until then, multi-rank MPAS-ocean full-step rows are throughput/parity-bounded
evidence (tag (c) of `SCALING_STATUS_AUDIT.md`), not stage-correct scaling
claims — enforced by the row metadata.
