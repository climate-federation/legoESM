# Simulation Full-Check Branch — Progress Log

**Branch**: `simulation_full_check`
**Goal**: end-to-end validation of every supported simulation tier
(shallow water → hydrostatic Held-Suarez / RCE / AMIP → ocean test
cases / OMIP) on every grid (lat-lon FV, cubed sphere, icosahedral,
spectral), with shared-colorbar / shared-projection cross-grid
comparison plots, physical-consistency checks vs. reference papers,
and GPU/MPI efficiency.

The full task is too large for one iteration.  This document tracks
what each iteration delivered and what remains.

---

## Iteration 1 — atmosphere cross-grid comparison plotting (DONE)

**Problem**: `scripts/run_atmosphere_test_matrix.py` runs every test
case on every grid but does NOT produce any cross-grid comparison
artifacts.  Comparing grids requires manually opening 4 separate
folders.  The ocean test matrix (`run_ocean_test_matrix.py:4591-5547`)
already had this functionality (cross-grid summary, time series, and
shared-colorbar snapshot panels).  The atmosphere matrix did not.

**Deliverable** (`scripts/run_atmosphere_test_matrix.py`):

1. `ATMOSPHERE_COMPARISON_FIELDS` — per-case field metadata
   (`vmin`, `vmax`, `cmap`, `units`).  Currently covers `williamson2`,
   `williamson5`, `cosine_bell`, `held_suarez`, `baroclinic`, `amip`.
2. `_collect_grid_results_atmosphere(test_case_dir)` — walks
   `<grid>/<resolution>/[<vertical>/]` to load each grid's
   `mean_timeseries.csv`, `snapshots_latlon.npz`, `results.txt`.
   Robust to the optional `<vertical_coord>` level used by hydrostatic
   runs and to missing artifacts (skips the grid quietly).
3. `_create_atmosphere_comparison_snapshots(test_case_dir, grid_results, fields)`
   — 4-panel layout, **shared cartopy PlateCarrée projection** and
   **shared colorbar** per field.  Cartopy is preferred but the plot
   gracefully falls back to plain `imshow` if cartopy is unavailable.
4. `_create_atmosphere_comparison_timeseries(test_case_dir, grid_results)`
   — overlay plot of every column shared by all grids' CSVs.
5. `_create_atmosphere_comparison_summary(test_case_dir, grid_results)`
   — plain-text per-grid metric table.
6. `_walk_atmosphere_test_cases(output_base)` — directory walker for
   batch generation.
7. CLI: `--no-cross-grid-plots` (skip post-run plotting) and
   `--cross-grid-plots-only` (regenerate plots from an existing tree
   without re-running the matrix).
8. Wired into `main()` so every full or partial matrix run emits
   comparison artifacts after the test summary.

**Verification**:
- SW quick mode on all 4 grids (12 cases): all PASS, comparison plots
  rendered for height / u / wind_speed across `williamson2`,
  `williamson5`, `cosine_bell`.
- Visual inspection of `comparison_snapshots_height.png` confirms the
  4-panel layout, shared coastlines, and identical colorbar — the
  Williamson-5 lee wave appears in the same lat/lon location on all
  four grids with the same intensity.
- `--cross-grid-plots-only` regenerates artifacts from existing output
  without running tests (useful for iteration on plot styling).

**Findings exposed by the new comparison plots**:
- `mean_height` time series for Williamson 2 differs by ~300 m
  between cubed_sphere/icosahedral (~2360 m) and latlon/spectral
  (~2050 m).  This appears to be a metric-area-weighting inconsistency
  in how `mean_height` is computed per grid.  Tracked as iter-2 task.

---

## Iteration 2 — TODO

Pick ONE of the following high-value next steps for iter-2:

### Option A — physical-consistency assertions
Tighten cross-grid agreement bounds for SW cases.  e.g.:
- Williamson 2: pin `max(|height_gridA − height_gridB|) < 0.1 m`
  (steady-state, regridded to common 181×360 mesh).
- Cosine bell at full revolution: pin error norms within 2× of the
  best grid.
- Williamson 5 lee-wave amplitude consistency.

### Option B — extend matrix coverage
- Wire the same comparison plots into `run_held_suarez_rrtmgp_allgrids.py`.
- Add `rce` to the atmosphere matrix (currently a separate runner
  hint in `CATEGORY_RUNNER_HINTS`).
- Add ocean test cases / OMIP cross-link (the ocean side is already
  done; just consolidate the user-facing entry point).

### Option C — fix the `mean_height` metric inconsistency
Resolve why latlon/spectral mean_height ≈ 2050 m while
cubed_sphere/icosahedral mean_height ≈ 2360 m for the same Williamson
2 initial state.  Likely a missing area-weighting in one of the
mean-statistics paths.

### Option D — adversarial review
Run `/codex:adversarial-review` on the new cross-grid plotting code.

**Recommendation**: Option C is the most scientifically critical
because it is a real bug exposed by iter-1, not just plumbing.  The
~15% disagreement in domain-mean h between grids is unphysical for a
Williamson-2 reference run.

---

## Iteration backlog

- [ ] Hydrostatic test matrix cross-grid comparison (vertical-coord
      handling — should already work via `_walk_atmosphere_test_cases`,
      verify on `held_suarez` quick mode).
- [ ] Add tests for the new helpers under
      `tests/scripts/test_atmosphere_cross_grid_plots.py`.
- [ ] Cross-grid physical-consistency *assertions* (not just plots) —
      e.g. `max(|h_grid - h_ref|) < tol` for steady-state cases.
- [ ] AMIP physical realism: realistic GHG/aerosol/ozone forcing,
      compare against ERA5 zonal-mean climatology.
- [ ] OMIP integration with ocean test matrix.
- [ ] GPU / MPI efficiency benchmarking pass with cross-grid timing
      table in the summary.
- [ ] `/codex:adversarial-review` round once iter-2 lands.
