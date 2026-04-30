# Partial cells: results summary

Status as of 2026-04-30 on branch `ocean-partial-cells`.

## What works

### Phase 6 — model integration on stepped bathymetry
- 24-hour rest-state integration on a step (deep/shallow) bathymetry
  is stable: `|u| = 7 mm/s`, `|v| = 24 mm/s`, `|eta| = 1 cm`, T/S
  unchanged. Previously NaN'd at step 4 (xfail).
- Flat-bottom forward + AD bit-exact regression preserved (every
  state field after stepping under `OceanZStarCoordinate` and
  `OceanPartialCellCoordinate` is bit-exact identical).
- 28 Phase 3-6 partial-cell unit tests pass; broader ocean suite
  (189 unit tests) regression-clean.

### Phase 3a — Beckmann-Haidvogel seamount
- Partial cells SURVIVE the unsmoothed Gaussian seamount with
  `r_max = 0.72` (Beckmann-Haidvogel "stable" threshold is 0.2).
  Legacy `z*` does not NaN here (the implicit-CN solver is robust)
  but produces 83 mm/s spurious flow.
- Rest-state PGF residual is **15-28× lower** than legacy `z*` at
  every smoothing level (Adcroft-Campin face PGF correction working).

| smoothing | r_max | partial PGF |du/dt|max | z* PGF |du/dt|max | ratio |
|:---------:|:-----:|:----------------------:|:----------------:|:-----:|
| 0 (none)  | 0.72  | 6.9e-7 m/s²            | 1.9e-5 m/s²      | 28×   |
| 2         | 0.56  | 6.9e-7 m/s²            | 1.4e-5 m/s²      | 20×   |
| 5         | 0.54  | 6.8e-7 m/s²            | 1.4e-5 m/s²      | 20×   |

The partial-cell PGF residual is essentially independent of bathymetry
steepness — the Adcroft correction does its job. Legacy `z*` benefits
from smoothing as expected.

## What does NOT yet pass

### Beckmann-Haidvogel 5 mm/s threshold
30-day final `|u|max` on the seamount, with implicit-CN, no surface
forcing, with bottom drag `r = 1e-3` m/s (partial-cell-aware: applied
at each column's `bottom_level`):

| smoothing | r_max | z* (mm/s) | partial (mm/s) |
|:---------:|:-----:|:---------:|:--------------:|
| 0         | 0.72  | 83        | 375            |
| 5         | 0.54  | 32        | 114            |
| 20        | 0.54  | 32        | 114            |

Neither coord meets the 5 mm/s threshold. Two observations:

1. Rest-state PGF residual on partial is much lower, but the
   equilibrated spurious flow is **higher**. Diagnostic shows the
   flow is bottom-trapped at `lev 17-19` (depth 3000-3800 m) over
   the seamount slopes. The PGF residual is small but persistent;
   without a Coriolis-balancing partner at the seafloor, even tiny
   PGF accelerates flow until friction balances it.

2. Smoothing beyond ~5 passes converges to the same r_max = 0.54
   (the seamount itself has structure that smoothing cannot reduce
   below this) and the residual saturates accordingly.

## Outstanding work

Partial-cell-related issues to address before a merge:

1. **Higher-order PGF**: the leading-order Adcroft-Campin face
   correction kills the J=1, eta=0 baroclinic PGF residual to
   second order in the centroid mismatch. Going to a full
   density-Jacobian PGF (Shchepetkin & McWilliams 2003) would push
   this further. Tracked but explicitly skipped in the partial-cells
   plan as out-of-scope for the first cut.

2. **Bottom drag form**: the BH 1993 paper applies linear drag with
   `r = 1e-3 m/s`; we now apply this at the actual `bottom_level`
   per column for partial cells. But quadratic Mellor-Yamada-style
   drag may be more appropriate near steep topography.

3. **T initialization sensitivity**: centroid-aware T (used in
   tests) makes the rest-state machine-zero, but the BH integration
   still drifts. Worth exploring whether the seamount-induced
   adjustment is a real spurious-flow signal or a transient that
   eventually decays.

4. **Real ETOPO 30-day run**: the headline experiment. Previously
   blocked on the long-integration NaN — that is now fixed. Next
   to run.

## Files

### Code (Phases 0-7)
- `src/legoesm/ocean/vertical.py` — `OceanPartialCellCoordinate`,
  `create_partial_cell_coordinate`, `compute_centroid_depth`,
  dispatch in `compute_layer_thickness` / `compute_ocean_jacobian`.
- `src/legoesm/ocean/eos.py` — `h_actual` kwarg through to
  `compute_hydrostatic_pressure`.
- `src/legoesm/ocean/dynamics/ocean_tendency_common.py` — h_actual
  in `iterate_eos_and_pressure_anomaly`.
- `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py` — Adcroft
  face PGF correction (`partial_cell_pgf_correction_x/y`),
  `compute_face_masks_3d`.
- `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` — partial
  coord dispatch, 3D face masks, partial-aware `h_u_old/h_v_old`,
  partial-aware bottom drag.
- `src/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py`
  — min face thickness, 3D face activity mask in u/v reconstruction.
- `src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` — 3D
  face mask + 3D activity gate for tracer transport.
- `src/legoesm/ocean/init_latlon_cgrid.py` — `H_bathy_override`.

### Tests
- `tests/ocean/unit/test_partial_cells_phase{0..6}.py` — 28 tests.

### Scripts
- `scripts/realistic_geometry_validation/run_phase3a_seamount.py` —
  BH seamount stress test, supports `--coord {zstar,partial}` and
  `--bottom-drag-r`.
