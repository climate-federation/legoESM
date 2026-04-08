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

**Not attempted**: Increasing Laplacian diffusion coefficient. The growth rate is too fast (e-folding ~0.3 days) for reasonable diffusion to control.

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

**Hex packing row spacing (dlat = d_rad * sqrt(3)/2)**: Introduced later by Pierre in commit c37c3d4. Mixes physical (sqrt(3)/2 factor) and angular (dlon based on center lat) spacing, creating degenerate cells at the domain center. Fix: use `dlat = d_rad` (uniform angular).

### Stereographic projection pole sign bug (Pierre's refactoring)

Pierre's commit c37c3d4 introduced `_stereo_project(xyz, pole)` with `denom = 1 + dot(xyz, pole)` and set `pole = -centroid`. This effectively projects from `+centroid` (because `_stereo_project` projects from `-pole`), sending the regional data (near `+centroid`) to near-infinity. The Delaunay triangulation in this numerically-degenerate coordinate system produced garbage. Ocean `dvEdge/dcEdge` dropped to 0.007, causing barotropic blowup.

**Fix**: `pole = +centroid` so the projection is from `-centroid`, giving well-behaved coordinates. Filed as PR #92.

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

## 2026-04-07: Rename baroclinic -> geostrophic_adjustment

The "baroclinic" test case initializes a meridional temperature front and lets it adjust under gravity and rotation — no wind forcing, no gyre forms. Renamed to `geostrophic_adjustment` to accurately describe the physics.

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

### What didn't work

**Flux-form tracer update** (`flux_form_tracers=True`): Implemented `h_new * T_new = h_old * T_old - dt * div(h*u*T) + dt * h_old * source`. Gives exact h*T conservation to machine precision at every step. BUT: unstable after ~3 days in geostrophic adjustment because the tracer flux uses the instantaneous baroclinic velocity, which is inconsistent with the barotropic solver's continuity equation. The fix (barotropic-averaged transport from substep time-averaging) is standard in MPAS-Ocean and MOM6 but requires significant changes to the barotropic solver interface. Left as configurable flag (default off) and documented in issue #94.

### Final results

| Test | Before | After |
|------|--------|-------|
| Rest state (stratified) | 2.84e-5 degC/day | 1.7e-15 |
| Geostrophic adjustment (10 days) | 1.52e-7 relative | 6.0e-15 |
| Rest state (uniform) | 0 | 0 (unchanged) |

---

## Issues opened

- #88 — Regional MPAS mesh for ocean dynamics testing
- #92 — Fix stereographic projection pole sign (Pierre's bug in c37c3d4)
- #94 — Split-explicit tracer conservation (h*T drift from advective-form tendency)

## PRs

- #89 — Ocean dynamics exploration (all work on `dhruv/exploration` branch)
- #92 — Clean fix for projection pole sign (from `fix/regional-mesh-stereo-pole` branch)
