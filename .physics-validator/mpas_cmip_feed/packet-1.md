# Adversarial review request: MPAS monthly-CMOR accumulator feed

You are an independent adversarial physics/software reviewer for LegoESM, a
JAX-native differentiable Earth System Model. Find EVERY bug, sign error, unit
inconsistency, shape mismatch, broken-gradient pattern, conservation violation,
retrace/host-sync hazard, and test-case discrepancy in the change below. Cite
line numbers (relative to the snippets). If you believe a candidate concern is
NOT a bug, say so and explain why.

## Problem being fixed

On the MPAS (Voronoi/unstructured) AMIP execution path (`ModelDriver._run_mpas`),
the `DiagnosticCollector` constructs the CMOR accumulators
(`SpatialMonthlyAccumulator` / `SpatialDailyAccumulator` / zonal
`MonthlyAccumulator`) and the restart **sidecar SAVE** runs each checkpoint
(`cmor_accum_day_*.npz` appear), but the per-step **FEED never happened** on this
path. Verified on a real pilot run: every accumulator manifest showed
`call_counts: []` / `max_count_ever: 0`. The cube/lat-lon path feeds via
`DiagnosticCollector.collect(...)`, which regrids state fields to the CMOR
lat-lon grid and calls `_spatial_monthly.add_2d/add_3d`. The lean MPAS loop
never calls `collect()` — it only writes lightweight scalar timeseries.

## Design

- The collector's Voronoi regrid helpers already exist and are tested
  (`_regrid_to_latlon_2d/_3d` dispatch to `apply_voronoi_to_latlon[_3d]` via
  IDW k-nearest weights built from `mesh.latCell/lonCell`; `_interp_to_plev19`
  is column-wise and works on `(nCells, nlev)` with `p_s (nCells,)`).
- NEW collector method `feed_cmip_accumulators_native(day, *, T, p_s, ...)`
  feeds the three accumulators from NATIVE cell arrays, reusing the SAME
  regrid + plev19 helpers `collect()` uses (no re-derived numerics).
- NEW driver helper `_feed_mpas_cmip_accumulators(day)` reconstructs geographic
  cell winds from the edge-normal `state.u` via `reconstruct_cell_velocity`
  (Perot), pulls the surface-precip export `self.model._sfc_diag[2]`, and calls
  the collector method. Called at diag cadence inside `_run_mpas`, gated by a
  precomputed `self._mpas_cmip_feed_on` flag (spatial or zonal accumulators
  enabled AND serial — `self._voronoi_layout is None`).
- MPI Voronoi is intentionally OUT of scope (documented): the collector holds
  LOCAL regrid weights and each rank holds only local cells, so a per-rank feed
  would bin one rank's cells into the global lat-lon boxes. The lightweight
  timeseries is already a true global via `_mpas_global_diag`.

## Key facts about the surrounding code (verified by reading)

- MPAS state layout: `state.T.data (nCells, nlev)`, `state.p_s.data (nCells,)`,
  `state.phis.data (nCells,)`, `state.u.data (nEdges, nlev)` (edge-normal),
  `state.tracers["q_v"].data (nCells, nlev)`. `sigma_full` ascending, index
  `-1` = lowest/surface level.
- `reconstruct_cell_velocity(u_edge, mesh) -> (u_east, v_north)` each
  `(nCells, nlev)`; exact for uniform flow (Perot area-weighted).
- Cube path uses ABSOLUTE `day = START_DAY + elapsed` and
  `doy,_ = day_to_calendar(day); year = int(day // 365.0)`. `day_to_calendar`
  returns `doy = day % 365 + 1` (noleap, 1-based).
- CMOR conventions in `collect()`: spatial `pr` in kg/m2/s (native), zonal
  `precip` in mm/day (`*86400`), `hus`/`q_v` profile in g/kg (`*1000`), `psl`
  hypsometric `p_s*exp(phis/(R_d*max(T_low,T_min)))`, daily 850 hPa index is
  `argmin(|sort(PLEV19)-85000|)`.
- Accumulator APIs: `SpatialMonthlyAccumulator.add_2d/3d(doy, year, fields)`
  requires exact `(nlat,nlon)` / `(nlat,nlon,nlev)` shapes and raises otherwise;
  zonal `MonthlyAccumulator.add_2d/3d(doy, year, fields, lat_deg)` bins by
  `np.digitize(lat_deg.ravel(), lat_edges)`. Both bump `_call_counts` /
  `_max_count_ever`.

## THE CHANGES

<INSERT my_changes.md HERE>

## Static analysis (my self-review)

- **Units**: tas [K] (lowest-level T proxy — documented; 2 m MOST refinement
  needs sst/sic not threaded here, marked TODO); ps/psl [Pa]; pr [kg/m2/s] to
  spatial, [mm/day] to zonal; prw [kg/m2] via `column_water_vapor`; ta/ua/va on
  plev19 [K]/[m/s]; hus [kg/kg] to spatial, [g/kg] to zonal profile. All mirror
  `collect()`.
- **Signs**: no flux/tendency signs (diagnostic-only, host-side, cannot perturb
  prognostic state). `u_east` is geographic eastward (correct CMOR `ua`);
  `psl` uses `+phis/(R_d*T)` (standard hypsometric, phis>0 raises psl above ps).
- **Differentiability**: N/A — pure host-side numpy diagnostic path, NOT in any
  `jax.grad`/`jit` trace. `reconstruct_cell_velocity` runs eagerly at diag
  cadence only; its output is immediately `np.asarray`-materialized.
- **Conservation**: none claimed; this only accumulates diagnostics. IDW regrid
  is a convex combination (partition of unity) — verified a uniform field maps
  to the same uniform value (ps/pr exact in tests).
- **Retrace/host-sync**: feed runs only at DIAG cadence (daily), after the diag
  block already synced state to host. Placed AFTER the finiteness/bounds guard
  so a blown-up state never enters the accumulator. `except Exception` makes a
  feed failure LOUD-but-nonfatal (matches the sidecar's defensive pattern), so
  a 100-yr run cannot abort on a diagnostic bug.
- **Limiters**: `np.maximum(T_low, T_min_atmosphere)` guards the psl exponent
  (same as `collect()`); `_interp_to_plev19` clips extrapolation via `alpha`.

## Test results (all pass, srun CPU, JAX_ENABLE_X64=1)

10 new tests in `tests/unit/test_mpas_cmip_accumulator_feed.py`:
- `test_feed_populates_all_three_accumulators` — pre-condition empty, post non-empty.
- `test_feed_binned_values_are_plausible` — ps/pr/psl EXACT-uniform (1e5, 2e-5),
  tas in (250,320), ua recovers ~7 m/s uniform flow, va ~0.
- `test_feed_zonal_and_daily_finalize` — zonal profile_T finalizes; daily 2 days + tas extremes.
- `test_feed_dry_minimal_inputs` — no q_v/precip/wind still feeds tas/ps + zonal.
- `test_feed_noop_when_accumulators_disabled` — returns False, no crash.
- `test_feed_manifest_roundtrip_nonempty` — save_cmor_accumulators sidecar manifest
  non-empty (the exact `call_counts:[]` symptom, fixed).
- `test_regrid_uniform_field_is_uniform` / `test_regrid_hemisphere_split_keeps_sign`.
- `test_driver_helper_feeds_via_reconstruct` / `test_driver_helper_dry_run_no_precip`.
Adjacent existing suites still green (voronoi regrid, diagnostic collector).
Constants + physics-contract ratchets green (3489 passed).

## The test file

<INSERT test file HERE>

## Your task

Cite line numbers. Focus on: (1) shape/units bugs in `feed_cmip_accumulators_native`
and the regrid/plev calls; (2) the `year = int(day // 365.0)` + doy calendar
consistency with the cube path (esp. restart chains and the mm/day vs kg/m2/s
split); (3) whether the driver helper's `state.u.data`/`_sfc_diag[2]` handles are
correct and whether `reconstruct_cell_velocity` at diag cadence introduces a
retrace/host-sync/perf hazard; (4) the MPI-serial gate — is skipping the MPI
feed the right call, or is there a silent-wrong-result risk; (5) test rigor — do
the assertions actually catch the failure mode, or are any vacuous. If there are
no substantive bugs, say so explicitly and justify why each candidate concern is
not one.
