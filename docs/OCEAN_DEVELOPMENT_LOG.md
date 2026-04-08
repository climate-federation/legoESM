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

## Issues and PRs

### Open issues
- #94 — Split-explicit tracer conservation: flux-form path needs barotropic-averaged transport (deprioritized — default path is machine-precision)
- #87 — Latlon A-grid instability (C-grid fixes in #98; A-grid removed from test matrix)
- #88 — Regional MPAS mesh (pole fix + test matrix in #98)
- #81 — Rest-state stability (diagnostic artifact fix in #98)
- #99 — Remove spectral grid from ocean (land boundary issues, not worth investing)

### PRs
- #98 — Consolidated ocean model fixes (open, replaces #89, #92, #95)
- #89 — Ocean exploration (closed, superseded by #98)
- #92 — Regional mesh pole fix (closed, superseded by #98)
- #95 — C-grid latlon stability (closed, superseded by #98)
