# FESOM + lock-exchange 4-dycore comparison — session state (2026-08-10)

Branch: aimip-wb2-mass-anchor-and-latlon-fix (working tree; lock-exchange work
uncommitted, entangled with unrelated AIMIP edits — stage explicitly).

## Root cause found (2026-08-10)
The latlon/mpas negative sorted-RPE + front over/undershoot was NEVER the
tracer limiter: tvd->fct2 (latlon) and tvd->upwind (mpas) changed the
violation at the 7th digit only. It is the EXPLICIT SPLIT BAROTROPIC's
eta/advecting-flux inconsistency breaking flux-form constancy preservation.
Measured with the uniform-T=15 + eta-step probe
(`scripts/validate/lockex_rpe_trace.py --uniform-t`, 1 day, dt=300):
  explicit_substep: latlon T -> [14.954, 15.046]; mpas [14.963, 15.039]
  implicit_cn:      latlon 15 +/- 1e-10; mpas +/- 2e-11; tripole +/- 9e-9
Fix: lock-exchange arms use `barotropic_solver="implicit_cn"` (tc.case-gated
in `_create_ocean_setup`). All four arms then: bounded front (~1e-9) + POSITIVE
RPE_mov at 1 day: fesom +3.0e12, tripole +2.1e12, latlon +6.1e12, mpas +1.0e13.

## Metric decision (RETRACTION of the fixed-volume rationale)
Matrix `_compute_sorted_rpe` = RPE_mov: densest-at-bottom, MOVING volumes
area*h(eta). Fixed dz_ref volumes drift NEGATIVE (-1.0e14/-1.3e14 per day)
even on the two clean arms (FESOM FCT, tripole) — for z-star arms fixed
parcels are not what the model conserves; for FESOM (linfs, fixed internal
thickness) the fixed-volume drift is the linfs concentration/dilution term,
and eta+H volumes (the physical column) remove it. Documented at the vol=
line in `_compute_sorted_rpe`. Packing kernel `_pack_sorted_rpe` shared with
the trace script. Discrimination tests:
`tests/ocean/unit/test_sorted_rpe_metric.py` (3 pass; catches sign-inverted
packing, fixed-volume regression; rearrangement invariance).

## Matrix gates added (run_lock_exchange)
- T_min_front/T_max_front: WHOLE-RUN ocean-masked extremes (per-sample via
  `_rpe_extract`, so land T=0 no longer prints as undershoot) within
  [T_cold_C - 1e-6, T_warm_C + 1e-6]; `_init_lock_exchange` now reads
  LockExchangeConfig for T_cold/T_warm (single source with gates + FESOM IC).
- RPE_rel_mixing_sign >= -1e-12 (deadband for metric noise floor).
- `_compute_sorted_rpe` flat-bottom tripwire: reject wet H_bathy spread > 5%.

## Matrix status (5-day arms, job 26837495)
fesom PASS +1.2e-8 | mpas PASS +6.2e-8 | latlon PASS +2.9e-8 |
tripole FAIL: T_max reaches 30.000209 (+2.1e-4 K) by day 5, growing smoothly
(job 26837318 trace). RETRACTED: NOT the PCG iteration count — single-rank
implicit_cn uses the direct/tol solve path (`barotropic_implicit_latlon_cgrid
.py:1561`), fixed_iters is never consumed, and setting it 60->300 was
bit-identical.

Tripole excursion CHARACTERISED (all CONFIRMED, jobs 26837567/96/658/659):
- 5-day uniform-T probe: 15 +/- 9e-9 → NOT eta/flux inconsistency; needs
  the front.
- Location columns: excursion pinned at 76.8N, 2.9E (warm-side front x
  northern land wall, land_lat_threshold=80) from day 2.5; before that at
  the southern wall (-77S).
- dt=150 arm: T_max 30.000206 vs dt300 30.000209 → dt-INDEPENDENT (spatial).
- upwind arm: bounded to 1.6e-9 over 5 days, RPE_mov +1.23e13 → the defect
  is the dim-split Van Leer TVD's multi-D overshoot at wall x front corners
  on distorted eORCA1 cells (first-order upwind clean; fct2 WORSE, 0.12 K —
  the latlon-C-grid FCT low-order bounds are also metric-blind there).

## SOLVED 2026-08-10 (second session): tripole defect closed
Two components, both fixed:
1. FCT certification bug (the real dycore defect): fct_tracer_advection
   certified the Zalesak box against h_OLD while the z-star update divides
   by h_new = h - dt*div(mf) -> actual update outside the box by exactly
   T*dt*div/h under divergent flow (repro: violation == T*max|dt*div/h| to
   4 digits; 0.083-0.12 K/day at the front, GRID-INDEPENDENT — reproduced
   on uniform latlon+implicit_cn). FIX in ocean/advection.py: h_new derived
   from the same advecting fluxes, q_td=(h*base - dt*div_low)/h_new,
   budgets normalised by h_new (NEMO zwi/e3t(Kaa) faithful).
   fixed_thickness=True for key_linssh (thickness pinned). Regression test
   test_bounded_under_divergent_flow_zstar FAILS on pre-fix code.
   Leapfrog certification still approximate (BEFORE thickness not
   threaded) — documented, not fixed. DINO probes divide by h_old
   (diagnostic-only, acceptable).
2. Dim-split TVD multi-D corner overshoot (scheme-class, not a bug):
   lock-exchange C-grid arms (latlon, tripole) now run fct2
   (LOCKEX_CGRID_TRACER_ADV) — matches FESOM's internal Zalesak FCT, so
   3 of 4 arms share the advection family; MPAS keeps tvd (no FCT in the
   port; clean to 3e-12/day at ico3).
5-day matrix: ALL FOUR ARMS PASS (tripole RPE_rel +1.357e-8, latlon
+1.711e-8, mpas +6.201e-8, fesom +1.153e-8; T within 1e-6 whole-run).
Codex adversarial review: 2 rounds, all findings fixed
(linssh, AB2/leapfrog wording, test numbers re-measured on 12x16).

## Open
2. Modular `scripts/matrix/ocean_test_matrix/experiments.py` has a DIVERGENT
   lock-exchange runner (no fesom/tripole, old fixed-volume metric, no
   gates) — pre-existing parallel refactor, not touched; needs a dedup PR.
3. `legoesm.ocean.rpe.compute_rpe` (committed Phase-C) packs
   densest-at-SURFACE while its docstring says bottom; trace logs it
   side-by-side as RPE_committed. Needs its own fix + test.
4. Overflow runner still uses unmasked T_min/T_max (same land-0 artifact).
5. No heat-conservation gate in the matrix (GLM suggestion); trace's
   heat_mov/vol_mov/Tbar columns cover it manually (all arms conserve to
   machine eps except FESOM heat_mov -9e-8/day = linfs approximation).

## Protocol invariants (do not break)
Same LockExchangeConfig (H_max=20, nlev=20, land_lat_threshold=80, T from
T_cold_C/T_warm_C), DEFAULT_DT=300, MATCHED_A_V=1e-4, MATCHED_K_V=1e-5,
MATCHED_TRACER_ADV=tvd (mpas), LOCKEX_CGRID_TRACER_ADV=fct2 (latlon,
tripole; fesom internal FCT), lock-exchange arms
barotropic_solver=implicit_cn, FESOM zbar = z_coord.z_half_ref.
sbatch: scripts/cluster/lock_exchange_dycore_comparison.sbatch (array 0-3,
5 days); traces: scripts/cluster/lockex_rpe_trace.sbatch (ARMS/EXTRA env).
