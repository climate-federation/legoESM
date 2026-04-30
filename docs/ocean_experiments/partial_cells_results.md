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

Neither coord meets the 5 mm/s threshold.

### Update after P0 face-thickness consistency fix

Re-running BH at smoothing=5 with all P0 + P1 fixes (face-thickness
unified to ``min``, conservation tests, distributed BBL drag option,
sigma fix in ``diagnose_w_from_flux_div``):

| smoothing | r_max | z* (mm/s) | partial (mm/s) |
|:---------:|:-----:|:---------:|:--------------:|
| 0         | 0.72  | 97        | 101            |
| 5         | 0.54  | 35        | 99             |

Face-thickness fix delivered the predicted 3.7x improvement at
smoothing=0 (375 → 101 mm/s) but only modest improvement at smoothing=5
(114 → 99 mm/s).  Distributed BBL drag (BBL=50, 100, 200 m) is a wash
at this test (the residual concentrates at thick deep cells where
BBL fits inside the bottom cell).

### Root-cause diagnostic (Phase 7 follow-up)

Direct probing of the partial-cells |u|max profile at smoothing=5,
day 30, at the column where |u|max occurs (lat=−2.5°, lon=205°,
H_bathy=3805 m, bottom_level=19, surface):

```
  lev  0..10 (0..1100 m):  smooth profile, |u| = 2-22 mm/s
  lev 11..14 (1300..2100 m): mid amplitude, |u| ~ 2-62 mm/s
  lev 15: u =   2 mm/s
  lev 16: u = -14 mm/s
  lev 17: u =   2 mm/s
  lev 18: u = -99 mm/s   ← |u|max
  lev 19: u =   0 mm/s   (drag sink, partial bottom cell)
```

This is **NOT** a smooth bottom-trapped boundary current.  It is a
**vertical 2Δz computational mode** with alternating sign at adjacent
levels, growing in amplitude toward the partial seafloor.  Z* on the
same setup shows a smooth surface-trapped profile (|u|max=35 mm/s at
lev 0), no oscillation.

**Consequences for the BH gap fix path**:

- Increasing vertical viscosity ``A_v`` 100× (from 1e-3 to 1e-1) only
  drops |u|max from 99 to 81 mm/s.  The mode is being actively forced
  faster than ``A_v`` can damp it.
- Disabling the Adcroft correction makes |u|max blow up to 3500 mm/s
  with a smooth profile.  So Adcroft IS doing essential magnitude
  work — but introduces the 2Δz mode as a side effect.

The 2Δz mode is consistent with the Adcroft per-face pressure shift
having a discontinuous z-structure: the correction at each face level
depends on the centroid mismatch at THAT level, which jumps when the
adjacent columns have different ``bottom_level``.  The vertical
profile of the correction therefore has step-function behaviour that
the discrete vertical viscosity cannot smooth.

### Advection scheme is NOT the source

Empirical sweep across momentum (vector_invariant, weno5) × tracer
(upwind, tvd) advection schemes after all P0+P1 fixes:

| momentum   | tracer | |u|max (mm/s) | mode signature lev 14-19    |
|------------|--------|--------------|-----------------------------|
| vector_inv | upwind | 102          | 58, 8, −7, −6, **−102**, 0  |
| vector_inv | tvd    | 99           | 62, 2, −14, 2, **−99**, 0   |
| weno5      | upwind | 101          | 56, 11, −8, −3, **−101**, 0 |
| weno5      | tvd    | 98           | 60, 3, −16, 7, **−98**, 0   |

The 2Δz mode amplitude and signature are essentially identical
across schemes (<4% variation).  The mode is a **forced equilibrium
response to the PGF z-structure**, not a transport artefact.

Side note: WENO5 tracer + partial cells crashes (NaN) — partial-cell
incompatibility in the WENO stencil at closed faces.  Does not
affect this diagnostic; will be picked up by the future test matrix
that exercises all advection schemes against the partial-cell
invariants.

### Why z-smoothing of the Adcroft correction does NOT help

Tested as a quick experiment: apply 1-2-1/4 vertical smoother to the
``partial_cell_pgf_correction_x/y`` arrays (N passes, then add to
``dp_dx/dy``).  Result:

| smooth_passes | |u|max (mm/s) |
|:-------------:|:-------------:|
| 0             | 99            |
| 1             | 2306          |
| 2             | 6770          |
| 4             | 22606         |
| 8             | −inf (NaN)    |

Smoothing makes things *dramatically* worse.  The Adcroft per-face
correction is a **single-level spike** in PGF at the partial-bottom
level (zero correction at adjacent levels) because that is where the
geometric pressure mismatch *physically lives*.  Smoothing redistri-
butes the correction to neighbouring levels where the actual mismatch
is zero, leaving the rest-state PGF un-canceled at multiple levels
and amplifying spurious flow by 25-200x.

### Why the 2Δz mode appears

The Adcroft correction at partial-cell faces is a single-level PGF
spike.  ``u`` responds locally to that spike, while vertical
viscosity and barotropic continuity (which averages U across all
levels) compete to redistribute the response.  The equilibrium of
this competition is a **2Δz vertical oscillation** in u(z): the
spike level itself is largely damped, but adjacent levels hold the
out-of-phase response.

### Implication for next steps

Density-Jacobian PGF (Shchepetkin & McWilliams 2003, originally task
#12) is still the right next lever — but the reason is more specific
than the audits stated.  It is not just "smaller per-face residual".
It is "**continuous-in-z PGF**": the per-column ρ(z) is reconstructed
as a polynomial spline, integrated analytically to give p(z) at any
depth, and horizontal Jacobian taken between columns at a common
depth.  The result is a **smoothly-distributed** PGF correction in z
(not a single-level spike), which does not force a 2Δz mode at the
partial bottom.

This rebrands #12 as the targeted fix, deferred to a follow-up PR:
the implementation is multi-day work (cubic spline ρ(z)
reconstruction per column, analytical integration, horizontal
Jacobian).  The current branch delivers significant standalone value
at this point: long-integration NaN fixed, conservation invariants
tested, distributed BBL drag infrastructure landed, and the BH gap
is now precisely characterised as a forced 2Δz mode rather than a
mystery residual — concrete enough to specify the next PR.

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
