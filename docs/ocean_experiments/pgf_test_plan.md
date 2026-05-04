# PGF Testing Plan — Cross-Grid Structured Validation

**Branch**: `test/PGF_latlon`
**Date**: 2026-05-04
**Motivation**: Lat-lon ocean goes unstable under JRA realistic winds with
land + topography.  Steady winds produce realistic circulation, suggesting
the PGF discretization (not the overall architecture) is the weak link
under strong forcing.  MPAS also showing PGF-related issues.  Need a
systematic, grid-agnostic PGF test ladder.

---

## Part A: Known bugs from code audit (2026-05-04)

These were found by auditing the lat-lon C-grid ocean code on the
`test/PGF_latlon` branch.  The `fix-partial-cell-pgf` branch (now merged
to main) addressed the PGF *scheme* itself (Adcroft → SMC03) and the
Coriolis stencil (AL81), but these face-thickness consistency bugs appear
to have survived.

### Bug #1 (HIGH) — F_slow face-thickness uses arithmetic mean, not min-rule

**Location**: `src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:687-691`

```python
h_u_pre = 0.5 * (jnp.roll(h_k_pre, 1, axis=1) + h_k_pre)   # line 687
h_v_pre_int = 0.5 * (h_k_pre[:-1] + h_k_pre[1:])            # line 691
```

The PE tendency function (`ocean_pe_latlon_cgrid.py:~812`) uses
`min_cell_to_uface(h_k)` (the MOM6/MITgcm `hFacW` convention).  The
mismatch means `F_slow_u = depth_avg(du_dt, h_u_pre)` has wrong weights
wherever bathymetry varies — creating an unphysical residual between
barotropic and baroclinic forcing at every topographic step.

**Why it matters for JRA**: JRA winds create stronger baroclinic
tendencies than steady winds, amplifying the F_slow inconsistency.

**Fix**: Use `min_cell_to_uface` / `min_cell_to_vface` for `h_u_pre` /
`h_v_pre` in the F_slow computation.

### Bug #2 (MEDIUM) — Explicit barotropic H_u uses mean, implicit uses min

**Location**: `src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:270`

```python
H_u = 0.5 * (jnp.roll(H_total_c, 1, axis=1) + H_total_c)
```

Compare with the implicit solver
(`barotropic_implicit_latlon_cgrid.py:143,146`):

```python
H_u_inner = jnp.minimum(jnp.roll(H_total, 1, axis=1), H_total)
H_v_int = jnp.minimum(H_total[:-1], H_total[1:])
```

Mean overestimates face depth at topographic steps → transport mismatch.

**Fix**: Use `jnp.minimum` in the explicit barotropic solver, matching
the implicit solver.

### Concern #3 (MEDIUM) — Pressure cumsum in float32 for lat-lon

**Location**: `ocean_pe_latlon_cgrid.py:791-797` calls
`iterate_eos_and_pressure_anomaly` without passing
`hi_precision_pressure` (defaults to `False`).
The cubed-sphere path (`ocean_pe_cdgrid.py:141-143`) explicitly passes
`True`.

**Fix**: Pass `hi_precision_pressure=True` for realistic-bathymetry runs.

---

## Part B: What `fix-partial-cell-pgf` already solved

The branch (now merged to main) delivered substantial PGF infrastructure.
Key results to build on, not re-derive:

### PGF scheme work
- `pgf_scheme="smc03"` (density-Jacobian, Shchepetkin & McWilliams 2003)
  with harmonic-mean monotonized piecewise-linear reconstruction
- Beckmann-Haidvogel seamount: **1.5 mm/s** (vs 99 mm/s Adcroft),
  rest-state PGF 60× smaller (1.1e-8 vs 6.7e-7 m/s²)
- Eliminates the 2Δz computational mode that Adcroft's single-level
  z-spike forces
- C1 bug fix: `z_target = min(z_c_W, z_c_E)` (shallower centroid)
  prevents asymmetric clamping when `h/dz < 1/3`

### Coriolis stencil
- AL81 PV-flux (Arakawa-Lamb 1981, 12-point triad) replaced the
  2-point Sadourny form that was unstable on partial-cell step vertices
- `h_vtx` switched from 4-cell mean to min-over-active (MITgcm/AHM97)

### ETOPO stress tests
- Phase 6: 30-day rest-state on ETOPO at 3°/20-level
  - SMC03: 2.5 mm/s day 1 → NaN day 19 (non-PGF instability)
  - Adcroft: 390 mm/s day 1 → NaN day 12
  - SMC03 advantage: 100-150× in rest-state magnitude
- Root cause decomposition (bisection ladder):
  1. **C-grid topographic computational mode** (dominant seed)
  2. **Bottom-cell ρ-curvature in SMC03 PLM** (6× amplifier)
  3. **Live-T positive feedback** (converts seed → NaN)
- AL81 + drag bump (2.5e-3) stabilised 30-day ETOPO: saturated at
  ~1.15 m/s (physical geostrophic adjustment, not numerical artifact)

### Phase 4 (Wolfe-Cessi spinup on ETOPO)
- 20-year run: buoyancy thermodynamics works, but velocity field
  dominated by partial-cell grid-scale noise preventing coherent gyres
- "Stable but not production-quality for dynamics"

### Existing test infrastructure
- SMC03 unit tests: 4 phases (31 tests) covering harmonic slopes,
  pressure interpolation, full PGF operators, pipeline integration
- Partial-cell unit tests: phases 0-7 (28+ tests)
- BH seamount script: `scripts/realistic_geometry_validation/run_phase3a_seamount.py`
- ETOPO script: `scripts/realistic_geometry_validation/run_phase6_etopo.py`
- Rest-state experiment: `src/legoesm/ocean/experiments/rest_state.py`

### What was NOT tested on that branch
- JRA or ERA5 realistic wind forcing
- Equatorial dynamics specifically (EUC, equatorial waves)
- Face-thickness consistency (Bug #1, #2 above)
- Cross-grid PGF comparison (cubed-sphere, MPAS)
- Multi-year forced integrations with topography
- Seasonal forcing cycles

---

## Part C: Tiered PGF Test Ladder

The goal: under realistic forcing and bathymetry, PGF discretization
doesn't generate spurious circulations large enough to corrupt the
solution at climatological scales.  "Spurious flows small compared to
physical flows, no secular growth."

The equator is the worst case: f → 0 removes geostrophic restoring, and
steep bathymetry (East Pacific Rise, Atlantic mid-ocean ridge) sits right
on top of physically important features (EUC, equatorial currents).

### Tier 1: τ = 0, idealized bathymetry, uniform stratification

**What**: Motionless ocean, smooth/step bathymetry (BH seamount or
simple ridge), uniform T(z), S.  No forcing.

**Purpose**: Isolates the PGF seed mechanism cleanly.  Already largely
covered by the existing BH seamount tests — but needs to be formalised
as a cross-grid test.

**Pass criterion**: max|u| < 1e-3 m/s for 90 days; KE doesn't grow
secularly.

**Diagnostics**: max|u|(t), KE(t), spatial map of |du/dt_PGF| at
surface and bottom.

**Grids to test**: lat-lon, cubed-sphere, MPAS.
**PGF schemes**: Adcroft, SMC03 (lat-lon); whatever each grid uses.

**Status**: Partially covered by existing BH tests (lat-lon only, 30
days, exponential T not uniform).

### Tier 2: τ = 0, idealized bathymetry, realistic stratification

**What**: Same bathymetry as Tier 1, but initialise from horizontally-
uniform T(z), S(z) profiles from equatorial Pacific WOA climatology.

**Purpose**: This is where ρ_ref(z) subtraction earns its keep.  Under
uniform stratification, ρ' = ρ − ρ_ref is exactly zero at t=0, so
you can't distinguish PGF scheme quality.  Under realistic
stratification, ρ' is small but nonzero and varies with depth, so
the partial-cell PGF residual reappears at reduced level.

**Pass criterion**: Same as Tier 1 (max|u| < 1e-3 m/s, 90 days, no
secular KE growth).

**Diagnostics**: Same as Tier 1, plus ρ'(z) profile to verify the
reference-density subtraction is working.

**New infrastructure needed**: WOA T(z), S(z) profile loader (or
hardcoded representative profiles).

### Tier 3: τ = 0, realistic bathymetry (ETOPO), realistic stratification

**What**: Real ETOPO bathymetry, WOA T(z)/S(z), no forcing.  90 days
to 1 year.

**Purpose**: Tests PGF on the actual ∇h_partial / A_eff distribution
(East Pacific Rise, trenches, shelf breaks).  The real-world stress
test for the rest-state PGF.

**Pass criterion**:
- max|u| in the abyss stays below physical abyssal flow scales
  (~1 cm/s)
- No coherent modes growing along ridge crests or trenches

**Diagnostics**:
- KE(t) total and binned by ∇h_partial / A_eff
- Map of time-averaged |u| at depth (surface, 1000m, bottom)
- Spatial decomposition: step faces vs interior

**Status**: Partially covered by Phase 6 ETOPO script, but that used
exponential T(z), not WOA profiles.  Need to extend.

### Tier 4: Wind-driven, idealized bathymetry

**What**: Steady zonal wind (or ERA5 climatological annual mean) over
rest ocean, flat or gently sloping bottom.

**Purpose**: Verify that wanted dynamics (Ekman, Sverdrup, equatorial
undercurrent) emerge correctly and aren't contaminated by PGF noise.
Compare to analytical solutions where they exist (Stommel/Munk gyre
transport, Ekman layer depth).

**Pass criterion**: Sverdrup transport within 20% of analytical;
no grid-scale noise in velocity field.

**Diagnostics**: Streamfunction, Ekman pumping, equatorial cross-
section of u(y, z).

### Tier 5: Wind-driven, realistic bathymetry, 1-year smoke test

**What**: Force with ERA5 or JRA55-do annual climatology on real ETOPO.
Integrate 1 year.

**Purpose**: "Realistic but short" — the test that currently fails.
This is where Bugs #1/#2 are most likely to manifest.

**Pass criterion**:
- EUC shows up at the right depth (~100-200m) and magnitude (~0.5-1 m/s)
- Abyssal flows are physical (< few cm/s)
- No noisy bottom-trapped mode along ridges
- Model doesn't blow up

**Diagnostics**:
- Equatorial cross-section u(y, z) at 140°W
- Abyssal |u| map
- KE budget decomposed by tendency (PGF, advection, wind, drag)
- Compare against published OMIP results or MOM6 OM4

**Status**: This is what triggered this investigation — currently
unstable under JRA.

### Tier 6: Full OMIP-style multi-decade run

**What**: Multi-decade JRA55-do or CORE-II forced integration.

**Purpose**: The destination, not a test.  By the time we get here, PGF
should be a non-issue.

**Not part of the immediate test suite.**

---

## Part D: Equatorial-Specific Diagnostics

Three diagnostics to instrument now (log at every output step):

### D1. Equatorial KE budget binned by depth

Spurious PGF residuals deposit KE at specific vertical scales.  Plot
KE(z) on the equator vs latitude.  Healthy: surface-intensified KE +
localized EUC peak at ~100-200m.  Pathological: KE growing in abyss or
bottom-trapped peak.

### D2. KE injection decomposed by tendency

At each output step, log the contribution of each tendency term to
equatorial KE:
- PGF contribution: u · (−∇p/ρ₀)
- Advection contribution: u · (advective tendency)
- Wind contribution: u · (wind stress / H_Ekman)
- Drag contribution: u · (−r·u)

Verify PGF share is small compared to wind and advection.

### D3. Partial-cell PGF residual as standalone diagnostic

At every partial-cell edge, compute the difference between the actual
PGF and the reference (zero for rest-state; or a high-resolution
reference for forced runs).  Map this spatially — useful for both
validation ("error concentrated at known steep features") and triage
("is a problem at location X a partial-cell issue?").

---

## Part E: Equatorial Stress Test (targeted single test)

**Setup**: Equatorial Pacific band, ±15° latitude, full longitude,
realistic bathymetry (ETOPO).  Initialise from WOA climatology.
Force with annual-mean ERA5 wind stress, zero buoyancy flux.
Integrate 5 years.

**Diagnostics**:
- EUC core depth, core speed, transport
- Compare against TAO mooring climatology
- Compare against MOM6 OM4 at comparable resolution
- Abyssal noise level

**Pass criterion**: EUC matches MOM6 within 10-20%.  Abyssal noise
< 1 cm/s.

**Decision value**: If this passes, PGF is ready for OMIP.  If it
fails in a clearly PGF-driven way, escalate to higher-order PGF
(AHH08 analytic-integration scheme, or PLM-in-(T,S) + Gauss-quadrature
EOS as documented in the `pgf_production_models_research.md` §7).

---

## Part F: Decision Logic

| Test result | Implication |
|---|---|
| Tier 1 + 2 pass | PGF seed elimination confirmed |
| Tier 3 passes (90d τ=0, ETOPO, quiet) | PGF geometrically robust for OMIP |
| Tier 5 shows abyssal noise along ridges or EUC distortion | Current PGF is borderline — check spatial pattern; if concentrated at steep features and grows slowly, try targeted bottom drag; if pervasive or grows fast, escalate PGF scheme |
| Equatorial stress test fails on EUC structure | Clearest signal to upgrade to AHH08 / PLM-in-(T,S) |

**Important caveat**: OMIP involves strong seasonal cycles (JRA55-do).
Tests with annual-mean forcing miss a real source of variability that
can excite spurious modes.  Before committing to a PGF scheme decision,
run at least one seasonal-cycle case at Tier 5 to verify seasonally-
varying η and stratification don't reactivate the partial-cell pathology.

---

## Part G: Implementation Order

### Phase 1: Fix the known bugs (immediate)

1. Fix Bug #1 (F_slow face-thickness) — `ocean_model_latlon_cgrid.py`
2. Fix Bug #2 (explicit barotropic H_u) — `barotropic_latlon_cgrid.py`
3. Enable `hi_precision_pressure=True` for lat-lon
4. Unit tests verifying the fixes

### Phase 2: Build the test infrastructure

1. Parameterised rest-state test fixture (grid × PGF scheme × bathymetry
   × stratification) covering Tiers 1-3
2. WOA T(z)/S(z) profile loader (or representative hardcoded profiles)
3. Equatorial diagnostics (D1, D2, D3) as reusable functions
4. Extend the ocean test matrix to run PGF tiers

### Phase 3: Run the tier ladder (lat-lon first)

1. Tier 1: uniform stratification, idealized bathymetry, 90 days
2. Tier 2: WOA stratification, idealized bathymetry, 90 days
3. Tier 3: WOA stratification, ETOPO, 90 days to 1 year
4. Tier 4: wind-driven, idealized bathymetry
5. Tier 5: JRA winds, ETOPO, 1 year

### Phase 4: Cross-grid extension

1. Run Tiers 1-3 on cubed-sphere and MPAS
2. Compare PGF residuals across grids
3. Identify grid-specific vs scheme-specific issues

### Phase 5: Equatorial stress test

1. Build the ±15° equatorial band setup
2. 5-year integration with ERA5 annual-mean winds
3. Compare against TAO/MOM6

---

## Part H: Files to create/modify

### New files
- `tests/ocean/pgf/test_pgf_tiers.py` — parameterised Tier 1-3 tests
- `tests/ocean/pgf/test_pgf_equatorial.py` — equatorial diagnostics
- `src/legoesm/ocean/diagnostics/pgf_diagnostics.py` — D1, D2, D3
- `scripts/pgf_validation/run_pgf_tier_ladder.py` — full tier runner

### Files to modify
- `src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` — Bug #1 fix
- `src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py` — Bug #2 fix
- `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` — Concern #3
- `scripts/run_ocean_test_matrix.py` — add PGF tier entries
