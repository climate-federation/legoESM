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
| smc03      | **1.5 mm/s**    | smooth, no spurious flow      |
| target     | < 5 mm/s        | —                             |

**Phase 5 PASS.**  SMC03 with the harmonic-mean monotonized σ
piecewise-linear reconstruction, evaluated at the
**shallower-of-centroids** face-reference depth (Adcroft & Campin
``face_ref`` convention), kills both the 2Δz computational mode
that the Adcroft single-level PGF spike forces *and* keeps the
rest-state PGF residual below ``1.1e−8`` m/s² (60× smaller than
Adcroft's ``6.7e−7`` m/s² — and small enough that linear drag
``r=1e−3`` damps everything to ~1.5 mm/s in 30 days).

### Iteration log (chronological)

The branch went through three z_target choices before landing on
the right one:

1. **Option A: ``z_target = |z_full_ref[k]|``** (column-independent
   reference centroid, plan §2.3 Option A).  Blew up to
   ~2200 mm/s on BH because at the partial-bottom level the
   reference centroid can sit *below* one column's actual partial
   seafloor → ``compute_pressure_at_target_smc03`` clamps that
   column to the seafloor pressure while the deeper column
   evaluates in-cell → asymmetric clamp → spurious gradient at
   every steep-bathymetry face.  Switched to Option B.

2. **Option B: ``z_target = 0.5 · (z_c_W + z_c_E)``** (face-mean of
   centroids, plan §2.3 Option B).  Brought BH down to ~525 mm/s.
   Initially attributed (incorrectly) to a fundamental ``O(h²·ρ'')``
   limitation of piecewise-linear reconstruction.  Tried
   piecewise-parabolic curvature corrections (κ term in the
   in-cell integral) and σ_{k-1} extension at the partial-bottom
   — neither helped, deepening the suspicion that cubic-spline ρ
   was needed.

3. **Diagnosed by code review (the ocean-model-expert subagent,
   ``docs/ocean_experiments/pgf_smc03_code_review.md`` issue C1)**:
   the midpoint of centroids *also* falls below the partial
   column's seafloor whenever ``h_partial / dz_ref < 1/3``,
   triggering the same asymmetric clamp.  The Phase 3 unit tests
   missed it because they hand-tuned ``h/dz`` to 0.5 and 0.75,
   both above the 1/3 threshold; BH at smoothing=5 generates many
   faces below it.  The asymmetric clamp leaves a residual
   ``ρ · g · (dz_ref − 3·h_partial)/4`` per face that does not
   vanish for any ρ — exactly the size needed to produce the
   525 mm/s observation.

4. **Fix: ``z_target = min(z_c_W, z_c_E)``** (the *shallower* of the
   two centroids — the same ``face_ref`` choice the leading-order
   Adcroft & Campin path uses, ``partial_cell_pgf_correction_x`` at
   ``latlon_cgrid_operators.py:1877``).  By construction
   ``z_target ≤ min(z_c_W, z_c_E) < min(z_seafloor_W,
   z_seafloor_E)`` (each column's centroid is above its own
   seafloor), so the clamp never triggers asymmetrically.  Also
   reduces to the standard centroid on full-cell faces.  Brought
   BH from 525 → 1.5 mm/s and rest-state PGF from 2.7e−5 → 1.1e−8
   m/s².

A new Phase 3 unit test
``test_smc03_pgf_machine_zero_thin_partial_cell`` locks the C1
regime closed: ``H_shallow=1640`` gives ``h_partial=40``,
``h/dz=0.2`` (well below 1/3) — must give machine-zero PGF on
linear ρ.  Closes the gap that allowed the bug to ship in the
first place.

### Production-model context (research subagent report)

A parallel deep-research pass by the dycore-expert subagent
(``docs/ocean_experiments/pgf_production_models_research.md``)
documents how production codes handle this: MOM6's default AFV
scheme reconstructs (T, S) in piecewise-linear / piecewise-parabolic
form and then *analytically* integrates ``ρ(T(z), S(z),
p_LRPD(z))`` using the closed-form Wright EOS — strictly better
than reconstructing ρ directly because EOS pressure dependence
introduces curvature in ρ that PLM-in-ρ leaves as an
``O(h²·ρ'')`` residual.  Our 1.5 mm/s with PLM-in-ρ is at the
high-quality end of the envelope production codes operate in
(ROMS DJ_GRADPS reaches ~10 mm/s on r=0.5 BH per Sikiric et al.
2009).  Should the (separate, currently-undeveloped) bigger-scale
ETOPO/eddy-resolving experiments demand sub-mm/s residuals, the
PLM-in-(T, S) + Gauss-quadrature EOS upgrade documented in §7 of
the research report is the recommended path — same scaffolding,
swap what gets reconstructed.

### Phase 6 ETOPO 30-day stress test

Real ETOPO bathymetry, 1°-source regridded to a coarse target,
``H_min=50`` m (so partial cells are at least ``2.5 × dz_surface``,
avoiding degenerate sub-surface partials), ``smoothing_passes=5``,
``bottom_drag_r=1e-3``, ``bbl_thickness=100`` m,
``barotropic_solver=implicit_cn``, centroid-aware exponential T(z).

**5°/15-level/1-day smoke test** — passes cleanly with SMC03,
``|u|max=7.6`` mm/s.  Confirms the pipeline (ETOPO load + regrid +
land-mask + partial coord + SMC03 PGF) integrates correctly under
realistic global geometry.

**3°/20-level/30-day side-by-side** — Adcroft vs SMC03 at
``dt=600`` s:

| day | adcroft \|u\|max | smc03 \|u\|max | smc03 advantage |
|:---:|:----------------:|:--------------:|:---------------:|
| 1   | 390 mm/s         | **2.5 mm/s**   | 156×            |
| 5   | 433 mm/s         | **3.5 mm/s**   | 124×            |
| 10  | 1163 mm/s        | **20 mm/s**    | 58×             |
| 12  | NaN (blew up)    | 49 mm/s        | —               |
| 18  | —                | 1647 mm/s      | —               |
| 19  | —                | NaN (blew up)  | —               |

SMC03 stays below the plan's 50 mm/s Phase 6 pass criterion through
**day 12**; Adcroft never reaches it (already 390 mm/s on day 1).
Both eventually crash to a **non-PGF global-domain instability**
that is far worse with Adcroft (12 days to NaN at ~15 m/s) than
with SMC03 (19 days to NaN, with growth visibly exponential from
day 8 onward).  Diagnosis: the instability is *not* a rest-state
PGF residual problem — SMC03's first-week rest-state magnitude
(2-5 mm/s) is comparable to its BH result (1.5 mm/s) — but a
separate ETOPO-specific dynamics issue, plausibly a coastal
computational mode at irregular coastlines or a thin-cell
implicit-CN solver pathology.  The legacy BH stress test has no
coastlines, so it does not expose this.

**2°/20-level** with the same config crashes both schemes: SMC03
day 7, Adcroft day 4.  ``dt=300`` s smooths but doesn't cure it.
The CFL-related instability scales with resolution and is
independent of the PGF scheme — it lives downstream of the SMC03
work.

The Phase 6 acceptance criterion as stated ("model integrates 30
days without NaN; ``|u|max < 50`` mm/s; no spatial concentration")
is therefore **partially met**: SMC03 satisfies the magnitude
criterion through day 12 and fails the 30-day NaN-free criterion
due to a non-PGF instability that Adcroft fails far worse on.
Closing the remaining 30-day gap on global ETOPO requires
separate stabilisation work (extra horizontal viscosity tuning,
biharmonic + Smagorinsky, coastal sponge, MEO-style bathymetry
smoothing) outside the scope of this PR.

### What this branch delivers

- ``pgf_scheme="smc03"`` (new) and ``"adcroft"`` (default,
  bit-exact preserved) config knob.
- Phase 5 BH headline gate **passing**: 1.5 mm/s vs the < 5 mm/s
  target, on a 30-day exponential-thermocline integration with
  smoothing=5, r_max=0.54, drag=1e−3.
- Rest-state PGF on the BH partial coord: **60× smaller** than
  Adcroft (1.1e−8 vs 6.7e−7 m/s²).
- Eliminates the 2Δz computational mode that Adcroft's
  single-level z-spike forces.
- Phase 3 ``test_smc03_pgf_machine_zero_thin_partial_cell`` C1
  regression test for the ``h/dz < 1/3`` regime.
- 101 unit tests passing, including all 69 pre-existing
  partial-cells phases bit-exact under the default.
- Fully differentiable end-to-end (``jax.grad`` AD tests in each
  phase).
- ``--pgf-scheme`` flag on the BH script; output dir tagged so
  results don't collide with the Adcroft baseline.
- New Phase 6 ETOPO script
  (``scripts/realistic_geometry_validation/run_phase6_etopo.py``)
  with side-by-side Adcroft/SMC03 benchmarking on real bathymetry,
  showing 100-150× SMC03 advantage in the rest-state magnitude
  before the global-domain instability sets in.
- Two on-branch research artefacts documenting the production
  landscape and the code review that found C1.
- Lat-lon dispatch for ``init_ocean_bathymetry`` (the existing
  ``_file_dispatch`` only handled cubed-sphere/MPAS/Gaussian; the
  new ``load_bathymetry_latlon`` correctly routes to a 2D-coord
  path with periodic-lon wrap, lat-lon Laplacian smoothing, and
  ``enforce_straits``/``fill_isolated_basins`` support).
- ETOPO endpoint-deduplication fix in ``_regrid_bathymetry`` —
  ETOPO files spanning ``[−180, 180]`` previously broke the
  scipy ``RegularGridInterpolator`` because ``%360`` collapsed
  the two endpoints onto a duplicate longitude.

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
