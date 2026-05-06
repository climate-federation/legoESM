# Simulation Full-Check Branch — Progress Log

**Branch**: `simulation_full_check`
**Goal**: end-to-end validation of every supported simulation tier
(shallow water → hydrostatic Held-Suarez / RCE / AMIP → ocean test
cases / OMIP) on every grid (lat-lon FV, cubed sphere, icosahedral,
spectral), with shared-colorbar / shared-projection cross-grid
comparison plots, physical-consistency checks vs. reference papers,
and GPU/MPI efficiency.

The full task is too large for one iteration.  This document tracks
what each iteration delivered.

---

## Iteration 1 — atmosphere cross-grid comparison plotting

Add `_collect_grid_results_atmosphere`, `_create_atmosphere_comparison_*`,
and CLI flags `--no-cross-grid-plots`, `--cross-grid-plots-only` to
`scripts/run_atmosphere_test_matrix.py`.  Cross-grid plots produced for
SW (W2, W5, cosine_bell) on all 4 grids: shared cartopy PlateCarrée
projection, shared colorbar per field, 4-panel layout.

Bug exposed: domain-mean Williamson-2 height differed by ~300 m
between cube/ico (~2360 m) and lat-lon/spectral (~2050 m) — cross-grid
diagnostic inconsistency, not a real solution difference.

## Iteration 2 — area-weighted mean for SW

Replace bare `jnp.mean(s.h)` with `_area_weighted_mean(s.h, grid.area)`
at all 8 SW `scalar_fn` sites (run_shallow_water + run_cosine_bell ×
4 grids).  Cross-grid spread of `mean_height` for Williamson 2:
313 m → 0.16 m (13 % → 5e-5).

## Iteration 3 — area-weighted mean for hydrostatic / AMIP / NH

Extend the helper to handle 3-D fields (collapse extra trailing axes
via uniform mean, then horizontal area-weight) and apply to 17 more
sites: held_suarez (4 grids), baroclinic, dcmip_transport, amip
(4 grids × 3 fields), nonhydrostatic (3 grids × 2 fields).  Add
`_grid_cell_area(grid_or_mesh)` helper.  Fix pre-existing missing
`fix_ps_mass` import in `primitive_eq_cdgrid.py` (was raising
NameError on cubed-sphere HS).

Held-Suarez `mean_T_t0` after fix: 299.9997–300.0011 K across 4
grids (1.4 mK / 4.7 ppm spread on the isothermal 300 K initial
state).

## Iteration 4 — cross-grid plots per (case × vertical_coord)

Iter-1 silently picked the alphabetically-first vertical_coord per
case.  Refactor `_collect_grid_results_atmosphere` to take an
optional `vertical_coord` argument; add `_vertical_coords_for_case`;
output goes to `<case>/<vertical_coord>/`.  HS sigma vs hybrid now
produce separate comparison sets.  Replace the `T` field
(2-D, not saved) with `T_3d` (4-D, saved) for HS/baroclinic.

## Iteration 5 — adversarial review fixes (REJECT → WARN)

Codex review (gpt-5.5, xhigh) flagged 1 HIGH + 4 MEDIUM + 2 LOW.
* H1 (HIGH): spectral `mass` diagnostic was Pa (mean) while
  cube/latlon/ico used Pa·m² (integral).  My iter-3 fix had
  introduced this units mismatch.  Replaced with
  `float(jnp.sum(p_s * area))` everywhere.  Spectral HS verified:
  mass_t0 = 5.10e19 Pa·m² (matches Earth area × 1 atm).
* M1: `mass` missing from TS candidate list.  Added.
* M2: AMIP fields `T_sfc`/`precip` not actually saved — replaced
  with `T_3d`/`p_s`/`wind_speed`.
* M3: DCMIP transport had no comparison metadata — added entries
  for `dcmip_transport_11/12/13`.
* M4: silent first-resolution-dir selection — added warning.
* L1/L2: mixed-layout warning + `--cross-grid-plots-only` filter
  honoring.

## Iteration 6 — codex iter-5 review LOW residuals

Codex re-review verdict: WARN, 0 HIGH/MED, 3 LOW.
* L1: centralize multi-resolution selection in
  `_select_resolution_dir(grid_dir)` helper; warn ONCE per
  grid_dir via module-level `_RES_DIR_WARNED` set.
* L2: replace `Pa*sr` with `Pa*m^2` in 3 `scalar_units` sites.
* L3: `--cross-grid-plots-only --grid` now restricts loaded
  grids end-to-end.

## Iteration 7 — codex iter-6 review final cleanup

Codex re-re-review verdict: WARN, 2 LOW side effects.
* `_RES_DIR_WARNED.clear()` at `main()` entry — invocation-local
  warning behaviour.
* `_vertical_coords_for_case` now takes `allowed_grids` keyword,
  matching the collector's signature.

The iterate-with-codex protocol has converged: REJECT (1 HIGH
+ 4 MED + 2 LOW) → WARN (3 LOW) → WARN (2 LOW) → present (clean).

## Iteration 8 — zonal-mean (lat, σ) cross-grid comparison

Add the canonical Held-Suarez Fig. 3 (H&S 1994) layout: 4-panel
plot of zonal-mean `T(φ, σ)` cross-section per grid with shared
colormap.  Helpers: `_zonal_mean_at_final_time`,
`_create_atmosphere_comparison_zonal_mean`,
`ATMOSPHERE_ZONAL_MEAN_FIELDS`.  Wired into
`_create_cross_grid_comparisons_atmosphere`.

## Iteration 9 — quantitative cross-grid RMS metric

Add `_interp_zonal_mean_to_target` (linear interp to 72-point common
lat axis) and `_compute_cross_grid_rms_agreement` (pair-wise RMS,
per-grid ensemble RMS, ensemble-averaged RMS).  Wired into
`_create_atmosphere_comparison_summary`.

iter-3 30-day HS quick-mode result:
| metric | value |
|--------|-------|
| best pair (icosahedral vs spectral) | 0.93 K |
| worst pair (cubed_sphere vs latlon) | 5.05 K |
| most divergent grid | cubed_sphere (3.26 K from ensemble) |
| ensemble-averaged RMS | 2.19 K |

## Iteration 10 — climatology mean (not snapshot) for cross-grid

User's iter-9 directive: "Held-Suarez cases show different
latitude-sigma structure.  Assess why and solve the issue so they
are all consistent."

**Investigation**: The forcing parameters and formulas are
SHARED constants in `held_suarez.py`; every grid uses identical
K_A, K_S, K_F, SIGMA_B, DELTA_T_Y, DELTA_THETA_Z, T_MIN, P_0 and
calls the same `held_suarez_equilibrium_temperature`.  Init
functions all use isothermal 300 K with seed=42 random
perturbation.

**Sources of the ~2 K cross-grid spread at 30 days** are
genuine numerical differences:
* dt: cube/ico=200 s, latlon=10 s (CFL near pole), spectral=600 s
* hyperdiffusion form: biharmonic on cube/ico/spectral, Laplacian
  on latlon
* random-perturbation realization: same seed, different array
  shape, so different eddy patterns

**Physical explanation**: at finite spin-up (HS canonical is 200
days), the climatology hasn't equilibrated.  iter-9's RMS used a
single t_final snapshot, contaminated with eddy variance from
the still-spinning-up baroclinic-instability field.  Wrong
diagnostic.

**Fix**: switch zonal-mean cross-section + RMS metric from a
single-snapshot diagnostic to a CLIMATOLOGY mean over the
trailing 50 % of snapshots (`CLIMATOLOGY_AVG_FRACTION = 0.5`).
Same metric, applied to the iter-3 30-day data:

| metric | snapshot | climatology |
|--------|----------|-------------|
| best pair | 0.93 K | 0.65 K |
| worst pair | 5.05 K | 3.74 K |
| most divergent | 3.26 K | 2.39 K |
| ensemble RMS | 2.19 K | 1.63 K (-26 %) |

Also added `--days <N>` CLI flag for spin-up convergence studies.

## Iteration 11 (in progress) — 60-day HS run

A 60-day run is in flight (background).  Cubed_sphere
mean_T trajectory shows continued cooling at t=60d (251 K and
falling at -0.29 K/d, vs. -1.08 K/d at t=0d).  Decay rate has
slowed by ~3.7×, indicating approach to equilibrium but not yet
there.

Iter-12 will analyse the 60-day data once latlon (CFL-limited
dt=10 s, ~30 min wall time per coord) completes.

---

## Iteration backlog

- [ ] iter-12: 60-day HS run analysis — does climatology RMS drop
      as expected with longer spin-up?  Goal: ≤ 1.0 K.
- [ ] iter-13+: 200-day HS run for canonical climatology
      verification.
- [ ] AMIP physical realism with realistic GHG/aerosol/ozone
      forcing.  Currently AMIP uses Held-Suarez forcing as a
      placeholder.
- [ ] OMIP integration with ocean test matrix (already has
      cross-grid plots from iter-1's predecessor).
- [ ] GPU / MPI efficiency benchmarking pass with cross-grid
      timing table in the summary.
- [ ] Direct unit test for the new helpers under
      `tests/scripts/test_atmosphere_cross_grid_plots.py`.
- [ ] Long-running / GPU-aware codex review on the post-iter-10
      branch.
