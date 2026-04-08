# Ocean Model Development Log

Running log of ocean dynamics work — what we tried, what worked, what didn't, and why.

---

## 2026-04-07: Lat-lon barotropic instability diagnosis

**Problem**: Lat-lon barotropic wave test blows up to max|eta| = 75 m by day 10 (initial perturbation 1 m). Instability originates at land-ocean boundaries near +/-80 deg latitude.

**Root cause**: The lat-lon barotropic solver uses a collocated (A-grid) discretization where both the FV divergence and gradient operators use 2-cell centered differences. This creates a checkerboard null space — the 2dx mode is invisible to both operators. Boundary interactions (wave reflection at the land mask wall) excite this computational mode faster than the Laplacian diffusion (`barotropic_diffusion_alpha = 0.01`) can damp it.

**Evidence**:
- Snapshots show grid-scale oscillations growing at the coast, then spreading equatorward
- Volume conservation violated by 756% over 10 days (mass being created)
- MPAS C-grid (TRiSK) solver is stable on the same test case — confirms A-grid is the issue

**Resolution**: C-grid lat-lon model added in #87, then 5 stability bugs fixed (see 2026-04-08 entry below).

---

## 2026-04-07: Regional spherical MPAS mesh

**Goal**: Run MPAS ocean tests on a regional domain without wasting computation on global land cells.

**Approach chosen**: Spherical regional mesh (not planar). Rationale: if the goal is to validate ocean dynamics en route to global simulations, we want the same geometry and discretization. A planar mesh would test Euclidean operators that don't transfer to global.

**Implementation** (`create_regional_voronoi_mesh` in `voronoi.py`):
- Seed quasi-uniform hex generators in a lat/lon region on the sphere + buffer ring
- Delaunay triangulation via stereographic projection (ConvexHull doesn't work for partial sphere)
- Feed triangles to existing `_build_mesh_from_generators` (only change: accept optional `triangles` parameter)
- Buffer cells become land via existing lat/lon-based masking in init functions
- All TRiSK operators and ocean model work unchanged

### What didn't work

**Variable dlon per row (cos(lat) spacing)**: Initial approach used `dlon = resolution / (R * cos(lat))` at each latitude, creating rows with different cell counts. The Delaunay triangulation at row boundaries produced elongated triangles with degenerate Voronoi edges (`dvEdge/dcEdge` as low as 0.016). The barotropic solver blew up within 9 substeps.

**Fix**: Uniform angular dlon spacing based on center latitude. Every row has the same cell count, giving a topologically regular hex grid. `dvEdge/dcEdge` improved to ~0.086.

**Lloyd relaxation with SphericalVoronoi**: Attempted to smooth the mesh with the existing `_lloyd_relaxation` function. The SphericalVoronoi of regional points extends Voronoi cells to the rest of the globe, pulling boundary generators far from their initial positions. Made `dvEdge/dcEdge` worse (0.0001). Disabled by default.

### Stereographic projection pole sign bug

Commit c37c3d4 introduced `_stereo_project(xyz, pole)` with `denom = 1 + dot(xyz, pole)` and set `pole = -centroid`. This effectively projects from `+centroid` (because `_stereo_project` projects from `-pole`), sending the regional data (near `+centroid`) to near-infinity. The Delaunay triangulation in this numerically-degenerate coordinate system produced garbage. Ocean `dvEdge/dcEdge` dropped to 0.007, causing barotropic blowup.

**Fix**: `pole = +centroid` so the projection is from `-centroid`, giving well-behaved coordinates.

### Validation

- 10-day barotropic wave: stable, mean eta conserved to 6 digits
- 30-day double gyre: stable, SSH +/-0.012 m, surface speed 0.18 m/s
- 959 cells total (725 ocean) vs 642 for global ico3

---

## 2026-04-07: Regional plot fixes

**Problem**: MPAS snapshot plots showed the full globe (-180 to 180, -90 to 90) with the regional data in a small corner surrounded by gray.

**Cause 1**: Axis extent was hardcoded to global bounds for MPAS grids.
**Cause 2**: KNN regridding extrapolated to the entire 181x360 global lat-lon grid.

**Fix**: Auto-detect regional meshes (lat span < 80% of globe), apply a distance cutoff in the KNN weights to prevent extrapolation, and crop the plot to the bounding box of non-NaN data.

---

## 2026-04-08: C-grid lat-lon stability fixes

**Problem**: The C-grid lat-lon ocean model (added in #87) blew up within 1-5 days on a barotropic wave test.

**5 bugs found and fixed**:

1. **Zonal face interpolation off-by-one**: `roll(-1)` averages cell j and j+1, but face j is between j-1 and j. Fix: `roll(+1)` in all cell-to-u-face interpolation.
2. **Pressure splitting error**: EOS uses time-varying Jacobian, creating spurious baroclinic PG for uniform T/S. Fix: use reference J=1 and eta=0 in hydrostatic pressure.
3. **Baroclinic-barotropic mode splitting**: Momentum equation acts on full velocity, double-counting barotropic solver. Fix: operate on perturbation velocity u'=u-U_bar.
4. **Forward Euler Coriolis**: Amplifies inertial oscillations by sqrt(1+(f*dt)^2)/step. Fix: forward-backward (Matsuno) stepping.
5. **Advective-form tracer drift**: -div(T*u) creates spurious T gradient with divergent velocity. Fix: flux-form correction dT/dt += T*div(h*u)/h.

**After fixes**: C-grid latlon barotropic wave runs 10 days stable with max|eta| = 0.027 m (was blowup at day 1.4).

---

## 2026-04-08: Tracer conservation investigation

**Problem**: Stratified rest-state test showed T drift of 2.84e-5 degC/day.

### Diagnosis (what we eliminated)

1. **Nonlinear EOS**: Tested linear vs Wright EOS — identical drift. Not the cause.
2. **Vertical diffusion operator**: With K_v=0, drift is exactly zero. With K_v>0, drift appears. Initially suspected the diffusion discretization was non-conservative.
3. **Per-column tendency conservation**: Checked `sum(dT/dt * dz)` for a single column — machine precision (-3.5e-15). The vertical diffusion operator IS exactly conservative.

### Root causes found

**Float32 precision**: The default `PrecisionPolicy` is float32 even with `JAX_ENABLE_X64=1`. Diffusion tendencies (~1e-8 degC/s) are at or below float32's relative precision when added to T values of ~2-20 degC. Rounding errors accumulate linearly. In float64, the stratified rest state gives T drift = 1.7e-15 (machine precision).

**Unweighted diagnostic**: The test matrix computed `nanmean(T)` over all cells and levels, treating 52m and 1048m layers equally. Vertical diffusion redistributes heat from thin warm surface layers to thick cold deep layers, changing the unweighted mean even when volume-weighted heat is perfectly conserved.

**Split-explicit h mismatch**: The tracer update `T_new = T + dt * dT_dt` uses old layer thickness, but the barotropic solver changes eta (and thus h_k). Fixed with `T_corrected = T_new * h_old / h_new`. Single-step conservation improved from 9.4e-10 to 3.7e-16.

**Diagnostic using reference dz_ref**: Even with the h_old/h_new correction, the volume-weighted diagnostic used `dz_ref` instead of actual `h_k`. When eta is nonzero, these differ by O(eta/H), creating a spurious drift in the diagnostic. Fixed by computing `h_k = compute_layer_thickness(eta, H_bathy, z_coord)` inside the diagnostic.

### Flux-form attempt (tried but deprioritized)

Implemented `h_new * T_new = h_old * T_old - dt * div(h*u*T) + dt * h_old * source` with `flux_form_tracers=True` config flag. Gives exact h*T conservation to machine precision at every step. BUT: unstable after ~3 days in geostrophic adjustment because the tracer flux uses the instantaneous baroclinic velocity, which is inconsistent with the barotropic solver's continuity equation.

The fix (barotropic-averaged transport from substep time-averaging) is standard in MPAS-Ocean and MOM6 but requires significant changes to the barotropic solver interface. Implementation plan documented in issue #94.

**Decision**: Deprioritized. The default advective-form path already achieves machine-precision conservation (6e-15 relative over 10 days). The drift rate (~6e-16/day) would take ~10,000 years to reach 1e-9 relative — negligible for any practical simulation. The flux-form infrastructure was left out of the consolidated PR to keep things clean.

### Final conservation results (MPAS ico3, float64, 10 days)

| Test | Before fixes | After fixes |
|------|-------------|-------------|
| Rest state (stratified) | 2.84e-5 degC/day | 1.7e-15 |
| Geostrophic adjustment | 1.52e-7 relative | 6.0e-15 |
| Rest state (uniform) | 0 | 0 (unchanged) |

---

## 2026-04-08: Consolidation into PR #98

The exploration branch (`dhruv/exploration`, PR #89) had accumulated a lot of experimental debris — regional mesh iterations, flux-form infrastructure, WIP commits, diagnostic plots. The useful fixes were scattered across three open PRs (#89, #92, #95).

**Action taken**: Created a clean branch (`dhruv/ocean-model-fixes`) by cherry-picking 8 commits and manually applying the h_k diagnostic fix. Closed PRs #89, #92, #95 and opened PR #98 as the single consolidated PR.

**Also**: Removed `barotropic_gyre` from the test matrix (redundant with `barotropic_double_gyre`). Test matrix now has 34 cases across 5 grids.

### Current test matrix

| Experiment | cubed_sphere | latlon | mpas | mpas_regional | spectral |
|---|---|---|---|---|---|
| rest_state | C24, 1d | 36x72, 1d | ico3, 1d | — | — |
| rest_state_no_land | C24, 1d | 36x72, 1d | ico3, 1d | — | T21, 1d |
| barotropic_wave | C24, 2d | 48x72, 2d | ico4, 2d | — | T21, 2d |
| barotropic_double_gyre | C24, 30d | 36x72, 30d | ico3, 30d | 300km, 30d | — |
| geostrophic_adjustment | C24, 10d | 36x72, 10d | ico3, 10d | — | T21, 10d |
| phillips_two_layer | C24, 10d | 36x72, 10d | ico3, 10d | — | T21, 10d |
| inertia_gravity_wave | C24, 2d | 36x72, 2d | ico3, 2d | — | T21, 2d |
| lock_exchange | C24, 1d | 36x72, 1d | — | — | — |
| overflow | C24, 0.5d | 36x72, 0.5d | — | — | — |
| stommel_gyre_tracer | C24, 60d | 36x72, 60d | ico3, 60d | — | — |

---

## 2026-04-08: Latlon A-grid → C-grid migration in test matrix

**Problem**: The latlon ocean model in the test matrix was using the A-grid (`LatLonOceanModel`), which has a fundamental checkerboard instability at land boundaries. The geostrophic adjustment test blew up around day 4-5 with exponential growth of grid-scale noise (max|eta| going from 0.12 m to 3.3 m over days 5-10).

**Root cause**: A-grid (collocated) discretization creates a 2dx null space invisible to both divergence and gradient operators. Land boundary reflections excite this computational mode faster than diffusion can damp it.

**Fix**: Switched all latlon paths in the test matrix to the C-grid model (`LatLonCGridOceanModel` + `LatLonCGridOceanConfig`), which uses compact stencils that resolve the checkerboard mode. Also created `wind_driven_gyre_latlon_cgrid` init function for the gyre experiments.

**C-grid shape handling**: The C-grid has staggered velocity arrays — u at (n_lat, n_lon+1) and v at (n_lat+1, n_lon). Can't compute `sqrt(u² + v²)` directly. Diagnostic functions now use `max(|u|, |v|)` as the speed proxy.

**Result**: Geostrophic adjustment now runs the full 10 days stable on latlon C-grid (T drift = 8.4e-4, vs blowup on A-grid).

---

## 2026-04-08: MPAS wind forcing diagnostic bug

**Problem**: MPAS ocean showed `max_speed = 0.0` in all wind-driven experiments (double gyre, global wind), even though the wind forcing was correctly applied and velocities were developing.

**Root cause**: Diagnostic key mismatch. The MPAS scalar function reported velocity as `max_abs_u`, but the gyre/wind runners looked for `max_speed`. The key was simply missing from the MPAS diagnostic dict.

**Fix**: Added `max_speed` key to MPAS scalar function output. Also added the `global_wind` profile to `mpas_physics.py` (was only in the generic `prescribed.py` used by cubed_sphere/latlon).

**Verification**: MPAS global wind now shows 0.276 m/s after 5 days (physically reasonable for early spin-up).

---

## 2026-04-08: Spectral ocean removed from test matrix

Opened issue #99. The spectral grid has fundamental limitations for ocean dynamics:
- Land boundaries create Gibbs ringing (spectral transform + sharp transitions)
- Rest state T drift: 2.6e-5 (vs machine precision on all FV grids)
- Geostrophic adjustment T drift: 1.2e-2 (orders of magnitude worse)
- Barotropic wave: had to be skipped entirely

Spectral methods work well for the atmosphere but not for ocean dynamics with closed basins and no-flux walls.

---

## 2026-04-08: Volume-weighted diagnostic fix for cubed_sphere/latlon

**Problem**: The conservation diagnostic for cubed_sphere and latlon used reference layer thickness `dz_ref` instead of actual `h_k = compute_layer_thickness(eta, H_bathy)`. This was the same artifact we fixed for MPAS earlier — when eta ≠ 0, `dz_ref` diverges from the true conserved quantity `sum(T * h_k * area)`.

**Fix**: Switched the cubed_sphere/latlon scalar function to compute actual h_k, matching the MPAS fix. For rest states (eta ≈ 0) this makes no practical difference, but for dynamic cases like geostrophic adjustment it correctly tracks the conserved quantity.

**Note on diagnostic precision floor**: Stratified rest states show T drift of 1e-15 to 1e-14 while uniform T/S cases show exactly zero. This is floating-point rounding noise in the global summation, not real conservation error. Vertical diffusion changes per-level T values, and the O(n_cells × n_levels) sum picks up different rounding patterns at each diagnostic step. Float64 precision limits conservation measurement to ~1e-14 relative without compensated summation.

---

## 2026-04-08: Test matrix restructuring

**Changes**:
- Removed `barotropic_gyre` (redundant with `barotropic_double_gyre`)
- Renamed rest-state cases for clarity:
  - `rest_state` → `rest_state_stratified_with_land`
  - `rest_state_uniform_ts` → `rest_state_uniform_with_land`
  - `rest_state_no_land` → `rest_state_stratified_no_land`
  - `rest_state_uniform_ts_no_land` → `rest_state_uniform_no_land`
- All rest states grouped under single `rest_state/` output folder
- Added `global_barotropic_wind` test case (simplified continent, 3-belt wind, 60 days)
- Regional grids (mpas_regional, latlon_regional) for double gyre
- Cross-grid time evolution comparison plots (rows=grids, cols=time)
- Cross-variant rest-state comparison (4 variants × 3 grids)
- Shared colorbar ranges across all multi-panel plots
- Spectral removed from all ocean tests

### Current test matrix (36 cases)

| Experiment | cubed_sphere | latlon (C-grid) | mpas | mpas_regional | latlon_regional |
|---|---|---|---|---|---|
| rest_state_stratified_with_land | 1d | 1d | 1d | — | — |
| rest_state_uniform_with_land | 1d | 1d | 1d | — | — |
| rest_state_stratified_no_land | 1d | 1d | 1d | — | — |
| rest_state_uniform_no_land | 1d | 1d | 1d | — | — |
| barotropic_wave | 2d | 2d | 2d | — | — |
| barotropic_double_gyre | — | — | — | 30d | 30d |
| global_barotropic_wind | 60d | 60d | 60d | — | — |
| geostrophic_adjustment | 10d | 10d | 10d | — | — |
| phillips_two_layer | 10d | 10d | 10d | — | — |
| inertia_gravity_wave | 2d | 2d | 2d | — | — |
| lock_exchange | 1d | 1d | — | — | — |
| overflow | 0.5d | 0.5d | — | — | — |
| stommel_gyre_tracer | 60d | 60d | 60d | — | — |

### Latest results (cases 1-27)

| Test | cubed_sphere | latlon (C-grid) | mpas |
|------|-------------|-----------------|------|
| rest_state (all 4 variants) | 12/12 PASS | machine precision | |
| barotropic_wave | PASS | PASS | PASS |
| geostrophic_adjustment | FAIL (T:9e-5) | PASS (T:8.4e-4) | PASS (T:6e-15) |

**Known issues**:
- Cubed_sphere geostrophic adjustment T drift (9e-5) — needs h_old/h_new thickness correction (same fix as MPAS)
- Wind-driven cases at low resolution develop unrealistic speeds over 60 days (viscosity tuning, not a code bug)

---

## 2026-04-08: Cubed-sphere face-boundary instability (issue #100)

**Problem**: The cubed-sphere ocean has a hard exponential instability at face boundaries. The geostrophic adjustment test blows up by day 5-7 (e-folding time ~0.8 days), while MPAS and latlon are perfectly stable. Rest-state tests show a face-imprint pattern at ~1e-15 level in all cubed-sphere cases, with the stratified+land case 5 orders of magnitude worse (1.27e-12 eta drift).

**Root causes identified**:

1. **Bug: Binary land mask interpolated at face boundaries.** `_fill_land_cells` (barotropic.py:82) passes the {0,1} mask through Lagrange interpolation in `pad_halo`, producing fractional values (~0.3-0.7) at face boundaries. This corrupts the neighbor-count logic for coastal filling.

2. **Design weakness: Inconsistent barotropic/baroclinic gradient operators.** The barotropic solver uses A-grid centered differences while the baroclinic dynamics use C-D grid Arakawa-Lamb gradients. The different error patterns at face boundaries don't cancel, creating a spurious tendency every timestep.

3. **Ocean-specific amplifier: density-pressure feedback loop.** Halo interpolation error → spurious PG → velocity error → advects T/S → density perturbation (amplified by Wright EOS) → larger PG → feedback. The atmosphere uses the same C-D grid operators but is stable because its temperature doesn't feed back through density into the pressure gradient in the same tight loop.

**Not yet fixed.** Proposed fix priority: (P0) don't interpolate mask — 1 line; (P1) float64 halo offsets — 1 line; (P2) normalize Lagrange weights — 3 lines; (P3) targeted face-boundary diffusion; (P4) reference pressure subtraction; (P5 long-term) C-grid barotropic solver.

---

## 2026-04-08: Conservation strategy overhaul (issue #101)

**Problem**: The additive conservation fixers (uniform correction to eta, T, S after each timestep) are harmful for forced/coupled simulations. They can't distinguish numerical error from real external forcing — a net surface heat flux gets "corrected" away, smeared uniformly through the full water column.

**Key finding**: The MPAS model's `h_old/h_new` thickness correction (ocean_model_mpas.py:157-192) already achieves machine-precision conservation (4.3e-16 relative heat drift over 1 day) *without* the additive fixer. The fixer actually makes conservation slightly worse (2.1e-15) by adding global-sum rounding noise.

**The h_old/h_new correction**: After the barotropic solver changes eta (and thus layer thickness from h_old to h_new), rescale tracers: `T_corrected = T_new * h_old / h_new`. This preserves thickness-weighted content `h*T` through the split-explicit step. It's local (no global reductions), differentiable (pointwise multiply), and doesn't interfere with external forcing.

**Missing from cubed-sphere and latlon C-grid**: Only MPAS has this correction. The cubed-sphere geostrophic adjustment T drift (9e-5) and the latlon double-gyre SSH drift (-0.012 m over 30 days) are both caused by this missing correction.

**Changes made**:
- `use_conservation_fixer` default changed to `False` in all 5 ocean config classes
- Fixer remains available via explicit opt-in for debugging

**Phased plan**:
- Phase 0: Port h_old/h_new correction to cubed-sphere and latlon C-grid (~20 lines each)
- Phase 1: Add conservation budget diagnostic (reports actual vs expected heat/salt change)
- Phase 2: Flux-form tracer transport with barotropic-averaged transport (issue #94, long-term)

### Phase 0 implementation: latlon C-grid h_old/h_new correction

Added the h_old/h_new thickness correction to `LatLonCGridOceanModel.step()` (ocean_model_latlon_cgrid.py), porting the pattern from MPAS (ocean_model_mpas.py:157-192).

**Results after the correction:**

| Test | Before | After | Notes |
|------|--------|-------|-------|
| rest_state_stratified_with_land | T=9.85e-15 | T=9.85e-15 | No change (no velocity → no h mismatch) |
| geostrophic_adjustment | T=8.39e-4 | T=8.70e-4 | No improvement |
| barotropic_double_gyre | eta=-1.16e-2 | eta=-1.18e-2 | No improvement (eta drift is volume, not tracer) |

**Why the latlon correction doesn't help like MPAS:**

The h_old/h_new correction fixes the split-explicit h mismatch — the error from tracers being updated with old h while the barotropic solver changes h. On MPAS, this was the **dominant** conservation error (6e-15 with correction vs 2.8e-5 without). On latlon C-grid, the dominant error is elsewhere.

The latlon C-grid tracer transport (ocean_pe_latlon_cgrid.py:299-316) uses an approximate flux-form approach:
```
dT/dt = -div(T*u)/area + T * div(h*u) / h
```

This combines `scalar_advection_cgrid` (advective form: `-div(Tu)`) with a correction term (`T * div(hu)/h`). Analytically these cancel to give `-u*grad(T)`, but the two **discrete** operators use different stencils that don't cancel exactly. The residual is O(8e-4) over 10 days — far larger than the h mismatch error the correction fixes.

On MPAS, TRiSK uses the **same** edge operator for both scalar transport and mass flux divergence, so the discrete cancellation is exact. That's why MPAS achieves 6e-15 and latlon achieves 8e-4.

**The eta drift (-0.012 m over 30 days)** is a separate volume conservation issue from the non-conservative barotropic Laplacian diffusion (`nu_dt = alpha * area`, area-dependent coefficient). The h_old/h_new correction only fixes tracer conservation, not volume.

### Why not switch to proper flux form?

A naive flux-form implementation (`h_new * T_new = h_old * T_old - dt * div(h*u*T)`) was previously tried on MPAS and went unstable after ~3 days (documented in the "Flux-form attempt" section above). The instability occurs because the tracer flux uses the instantaneous baroclinic velocity, which is inconsistent with the barotropic solver's time-averaged continuity equation. The barotropic solver's `div(h*u)` is accumulated over 30 substeps, but the tracer sees only the single baroclinic-step velocity.

The fix (Hallberg 1997, Higdon 2005, as used in MOM6 and MPAS-Ocean) requires the barotropic solver to accumulate and return the time-averaged thickness flux `<h*u>_baro`, and the tracer equation to use that averaged transport. This is issue #94 (Phase 2). Switching the latlon transport to proper flux form without this barotropic averaging would likely hit the same instability.

**Current conservation status across grids:**

| Grid | Transport form | T conservation (geoadj 10d) | Limiting factor |
|------|---------------|----------------------------|-----------------|
| MPAS | TRiSK (exact flux form) | 6e-15 | Machine precision |
| latlon C-grid | Approximate flux-form correction | 8.7e-4 | Discrete stencil mismatch in transport |
| cubed_sphere | FCT (flux-corrected transport) | 9e-5 | Missing h_old/h_new + face-boundary instability |

**The h_old/h_new correction is still worth keeping on latlon** — it's architecturally correct, costs nothing, and will matter when the transport operator is improved in Phase 2.

---

## 2026-04-08: Conservative barotropic diffusion (latlon C-grid)

**Problem**: The barotropic Laplacian diffusion on the latlon C-grid used `nu_cell * div(grad(eta))` where `nu_cell = alpha * grid.area` varies as `cos(lat)`. This places the spatially varying coefficient *outside* the divergence, breaking volume conservation. Over 30 days in a barotropic double gyre, the leak accumulated to -0.012 m mean SSH drift (~0.14 m/year).

**Root cause**: `sum(nu_cell * laplacian(eta) * area) != 0` when `nu_cell` varies spatially. The divergence theorem only guarantees `sum(div(F) * area) = 0` — the coefficient must be *inside* the divergence.

**Fix**: Reformulated as `div(nu_face * grad(eta))` (flux-form diffusion):
1. Compute gradient: `grad_x, grad_y = gradient(eta)`
2. Multiply by face-centered coefficient: `flux_x = nu_face_u * grad_x`
3. Take divergence: `delta_eta = div(flux_x, flux_y)`

Where `nu_face_u = baro_alpha * 0.5 * (area_west + area_east)` — average of adjacent cell areas at each face. Face masks zero the flux at land boundaries. The face coefficients are precomputed once outside the substep loop (grid geometry only).

This is the standard approach in MOM6 and NEMO. Conservative by construction: `sum(div(F) * area) = 0` for any flux F with no-flux BCs.

**Results** (barotropic double gyre, latlon_regional 24x48, 30 days):

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| eta drift | -1.16e-2 | -2.87e-3 | 4x |
| max speed | 0.093 m/s | 0.093 m/s | unchanged |
| rest state (eta) | 0.00e+00 | 0.00e+00 | unchanged |
| rest state (T) | 9.85e-15 | 9.85e-15 | unchanged |

The diffusion fix eliminated ~75% of the volume leak. The remaining drift (~2.9e-3 over 30 days, ~0.035 m/year) comes from a secondary source — not the eta floor clamping (floor is at -5499.5 m, eta stays within ±0.02 m), likely the continuity divergence operator or mask interactions. This is a 4x improvement and adequate for multi-decade runs; the secondary source can be investigated as a follow-up.

**Same issue exists on cubed-sphere and MPAS**: Both use `nu_cell * laplacian(eta)` with area-dependent coefficients. The MPAS case leaks less because Voronoi cell areas are more uniform (~1.2x vs ~3.7x variation). The same flux-form fix should be applied to both solvers for consistency.

---

## 2026-04-08: Test matrix comparison plot fixes

**Problem**: Cross-grid comparison plots for regional grids (MPAS regional, latlon regional) showed incorrect extents and indistinguishable lines.

1. **MPAS regional on global axes**: The MPAS regional regridding outputs to a global 181x360 lat-lon grid with NaN outside the domain. The comparison snapshot and evolution plots used the full coordinate extent instead of cropping to non-NaN data. Fixed by adding bounding-box crop (same logic already used in per-grid snapshots).

2. **Black-on-black timeseries**: The color dict only had entries for global grids (cubed_sphere, latlon, mpas, spectral). Regional grids fell through to default black. Fixed by adding color entries for regional grids (tab:red/tab:green with dashed linestyle).

---

## Issues and PRs

### Open issues
- #100 — Cubed-sphere ocean face-boundary instability (exponential blowup at face boundaries in dynamic simulations)
- #101 — Conservation strategy: replace additive fixers with h_old/h_new thickness correction
- #94 — Split-explicit tracer conservation: flux-form path needs barotropic-averaged transport (long-term, Phase 2 of #101)
- #99 — Remove spectral grid from ocean (land boundary issues, not worth investing)
- #87 — Latlon A-grid instability (C-grid fixes in #98; A-grid removed from test matrix)
- #88 — Regional MPAS mesh (pole fix + test matrix in #98)
- #81 — Rest-state stability (diagnostic artifact fix in #98)

### PRs
- #98 — Consolidated ocean model fixes (open, replaces #89, #92, #95)
- #89 — Ocean exploration (closed, superseded by #98)
- #92 — Regional mesh pole fix (closed, superseded by #98)
- #95 — C-grid latlon stability (closed, superseded by #98)
