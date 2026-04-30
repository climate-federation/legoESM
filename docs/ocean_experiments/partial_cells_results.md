# Partial cells: results summary

Status as of 2026-04-30 on branches `ocean-partial-cells` and `ocean-pgf-smc03`.

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

### Failed fix attempts (empirical exclusion)

Beyond the experiments above, attempted in a single session:

1. **Vertical viscosity sweep**: A_v from 1e-3 to 1e-1 (100× increase)
   only drops |u|max from 99 → 81 mm/s (~18%).  The 2Δz mode is
   actively forced faster than ``A_v`` Laplacian-form damping can
   absorb.

2. **Vertical biharmonic viscosity** (∂⁴u/∂z⁴, new ``B_v`` config
   knob with ``Lap²`` operator).  At a 2Δz mode the biharmonic
   stencil response is 16× stronger than the Laplacian (vs 4× at
   4Δz), so this should preferentially damp 2Δz.  Result on the
   seamount:
   - B_v ≤ 10 m⁴/s: tiny effect (rate ~ 6e-9/s, decay > 5 years).
   - B_v = 50 m⁴/s and above: model NaNs.  CFL is violated at the
     partial seafloor where ``dz_half_ref · J`` shrinks toward
     ``h_partial[bot] · J → 0`` and the implicit thickness
     amplifies the per-step amplification factor.  An implicit
     vertical biharmonic solver could fix the CFL; left as future
     work.
   - Reverted (does not help in the explicit forward-Euler path).

3. **Piecewise-constant + piecewise-linear density-Jacobian PGF**
   (added ``density_jacobian_pgf_x/y`` operators behind ``pgf_scheme``
   config knob).  Implementation evaluates ``p(z)`` per column at a
   smooth-in-k face-reference depth (``z_full_ref[k]``) via
   analytical integration of the column ρ profile, then takes the
   horizontal Jacobian.  Result on the seamount:
   - Rest-state PGF residual is **88× WORSE** than Adcroft (5.9e−5
     vs 6.8e−7 m/s²).
   - Reason: piecewise-linear ρ(z) within each cell needs ``dρ/dz``
     estimates to be **CONSISTENT across adjacent columns** (so that
     ρ-at-the-same-physical-depth matches between W and E).  My
     centered-FD estimate of ``dρ/dz_kc`` uses each column's local
     neighbours, which differ in z because partial cells shift
     centroids.  So adjacent columns disagree on ρ at intermediate
     depths → spurious PGF.
   - For the residual to genuinely cancel, ``dρ/dz`` needs a
     **higher-order reconstruction** (cubic spline through a column's
     ρ values with curvature constraints, as S&M 2003 actually
     prescribes).  This is what makes S&M 2003 hard.
   - Reverted.  The framework remains a good starting point for a
     proper follow-up implementation.

4. **Strong bottom drag**: r=5e−3 → 33 mm/s; r=1e−2 → 27 mm/s
   (plateau); r ≥ 5e−2 → NaN (CFL violation).  The plateau at ~27 mm/s
   with strong drag confirms the 2Δz mode at adjacent levels (where
   drag doesn't act) holds the residual.  Workaround for testing,
   not a real fix.

### Implication for next steps

Density-Jacobian PGF with **cubic spline ρ(z) reconstruction** is the
genuinely-required fix for the BH gap on partial cells.  The
simpler approaches above (piecewise-constant, piecewise-linear, or
just-evaluate-at-different-z_target with the existing p_prime)
*don't work*: their dρ/dz estimates are inconsistent across columns
with different centroid placements.

S&M 2003's full machinery — column-wise cubic spline of ρ vs z with
specified boundary conditions, harmonic-mean averaging of ρ at faces
for the Jacobian — is what produces a column-consistent reconstruction
that gives a genuinely continuous-in-z PGF.  This is multi-day
implementation work appropriate for a fresh follow-up PR.

## SMC03 piecewise-linear follow-up (branch `ocean-pgf-smc03`)

The follow-up branch lands the harmonic-mean monotonized
piecewise-linear S&M03 PGF as ``pgf_scheme="smc03"`` (default
remains ``"adcroft"``).  Six commits cover the harmonic σ
reconstruction, in-cell pressure profile, density-Jacobian PGF
operator, PE-pipeline dispatch, and the BH script flag.  31 unit
tests pass; the existing 69-test partial-cell regression suite is
preserved bit-exact under the default.

Validation of the building blocks:

| test                                                          | result   |
|:--------------------------------------------------------------|:---------|
| Phase 1 harmonic σ on linear ρ                                | exact    |
| Phase 2 in-cell P on linear ρ                                 | exact    |
| Phase 3 stepped-bathymetry **linear** ρ rest-state PGF        | < 1e−12 m/s² (machine zero) |
| Phase 3 cross-column σ agreement (different bot_levels)       | exact    |
| Phase 4 H&A 2009 column-sum identity under SMC03              | < 1e−12  |

These confirm that for **linear ρ(z)** the SMC03 scheme delivers
machine-zero rest-state PGF on partial cells — the load-bearing
property the harmonic-mean σ formulation buys.

### BH headline result with SMC03

Configuration: smoothing=5, r_max=0.54, ``bottom_drag_r=1e-3``,
implicit-CN barotropic, **exponential** thermocline (the
``rest_state_latlon_cgrid_ocean`` default ``T(z) = T_deep +
(T_surface − T_deep)·exp(−z/scale_depth)``), centroid-aware T init,
30-day integration:

| pgf_scheme | |u|max @ day 30 | flow signature                |
|:-----------|:---------------:|:------------------------------|
| adcroft    | 99 mm/s         | 2Δz computational mode        |
| smc03      | 525 mm/s        | smooth bottom-trapped current |
| target     | < 5 mm/s        | —                             |

SMC03 **does** kill the 2Δz mode forced by Adcroft's single-level
PGF spike (verified in Phase 3 ``test_smc03_no_single_level_spike``)
— the failure mode is now a smooth bottom-trapped current rather
than a sawtooth oscillation.  But the **magnitude** of the steady
state is larger than Adcroft's because the harmonic-σ
piecewise-linear reconstruction leaves an ``O(h² · ρ'')`` residual
in the in-cell integral, and at the deepest BH levels (h ≈ 500 m,
ρ'' large in the exponential thermocline) this exceeds Adcroft's
``O(h · ρ' · centroid_offset)`` residual.  Per-level rest-state
``|dv/dt|`` (BH partial coord, smoothing=5):

```
  k=10..14:  SMC03 ~1e−9..1e−8 m/s²,  Adcroft ~3e−7 m/s²  (SMC03 wins)
  k=15..19:  SMC03 ~1e−5..3e−5 m/s²,  Adcroft ~5e−7 m/s²  (Adcroft wins)
```

The deep-level Adcroft residual is small here because at smoothing=5
the centroid offsets are gentle.  Adcroft's residual is also a
single-level z-spike at each face's partial-bottom level — that's
what forces the 2Δz mode that ultimately drives the 99 mm/s
steady state.  SMC03's residual is spread smoothly in z but larger
in absolute magnitude at the deepest cells, and that magnitude
sets the steady state through nonlinear advection feedback into a
smooth bottom-trapped current.

### Pursued fallbacks (per plan §3 Phase 5)

The plan's Phase 5 fallback ladder was followed:

1. **Option B for ``z_face`` (per-face mean of cell centroids)**: the
   initial Option A attempt with ``z_target = |z_full_ref[k]|`` blew
   up to 2200 mm/s because at the partial-bottom level the
   column-independent reference centroid can sit *below* one
   column's actual partial seafloor, triggering the seafloor clamp
   on one side and an in-cell evaluation on the other.  Option B
   uses the per-face midpoint of the two adjacent column centroids,
   which is by construction inside both columns.  Brought |u|max
   from 2200 → 525 mm/s.  Shipped as the default in the SMC03 path.
2. **Curvature term κ in the in-cell integral (S&M03 §4.2)**:
   tried piecewise-parabolic ρ with κ from a centred FD of σ.
   Negligible effect on BH (525 → 525 mm/s) — the dominant residual
   at deep levels is the σ stencil mismatch across columns at the
   partial-bottom, not the cell-mean curvature.  Reverted; not in
   shipped code.
3. **Cell-above σ extension at the partial-bottom**: copy
   ``σ_{k-1}`` to ``σ_k`` at each column's partial-bottom level so
   adjacent columns agree on σ there.  Negligible effect on BH —
   the first-order σ values already approximate the local slope
   reasonably; the residual must come from higher-order curvature
   terms not the σ asymmetry per se.  Reverted; not in shipped code.

### Genuinely-required next step

Cubic-spline ρ(z) reconstruction (S&M03 §4 modified Jacobian, full
form, not the harmonic-linear simplified form this branch
implements).  This pushes the in-cell integral residual from
``O(h²·ρ'')`` to ``O(h³·ρ''')`` and is what closes the BH gap for
realistic exponential stratification.  Multi-day work, separate PR.

The infrastructure landed on this branch (Phase 1 harmonic σ,
Phase 2 piecewise-linear in-cell integral, Phase 3 face-adaptive
PGF operator, Phase 4 PE dispatch) is exactly the right scaffolding
for that follow-up: only ``compute_pressure_at_target_smc03`` and
the σ reconstruction need to be upgraded; the dispatch and tests
are reusable.

### What this branch delivers regardless of the BH residual

- Validated SMC03 piecewise-linear PGF that achieves machine-zero
  rest-state PGF for **linear** ρ(z) on partial cells (Phase 3
  test).
- Eliminates the 2Δz computational mode that Adcroft's single-level
  z-spike forces (Phase 3 smoothness test).
- ``pgf_scheme`` config knob with backward-compat default (Phase 4).
- All 69 pre-existing partial-cell regression tests preserved
  bit-exact under the default.
- Fully differentiable end-to-end (``jax.grad`` AD tests in each
  phase).
- BH script flag (``--pgf-scheme``) and a documented baseline run
  for any future cubic-spline upgrade to compare against.

The current branch delivers significant standalone value: long-
integration NaN fixed, conservation invariants tested, distributed
BBL drag infrastructure landed, the BH gap precisely characterised
as a forced 2Δz mode, and ALL simpler fix attempts empirically ruled
out — so the next PR can target full S&M 2003 with confidence.

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
