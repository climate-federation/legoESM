# Ocean Model Development Log

Running log of ocean dynamics work — what we tried, what worked, what didn't, and why.

---

## 2026-04-18: ACC Channel Experiment with Gaussian Ridge

### Motivation

The ocean test matrix had no experiment with spatially varying bathymetry
interacting with dynamics. The "with land" experiments use flat bottoms;
the overflow test has variable bathymetry but only on cubed_sphere/latlon
(not channels). We needed a test case that exercises the z-star coordinate's
Jacobian with a realistic topographic feature.

### What we built

An ACC-like channel experiment inspired by Zhang et al. (2024, JPO),
featuring a meridional Gaussian ridge on the sphere. Design decisions
were made through an interview process (see spec in
`docs/ocean_experiments/zhang2024_acc_channel_spec.md`).

**Key parameters:**
- Spherical channel centered at 40S (configurable), zonally periodic
- H_max = 3000 m, Gaussian ridge h0 = 1000 m, sigma = 150 km
- Abernathey et al. (2011) exponential stratification (delta_T = 8 degC)
- Linear EOS (alpha_T = 2e-4, salinity passive)
- Half-sine zonal wind stress (tau0 = 0.1 N/m^2)
- Northern-only sponge (200 km, 7-day restoring to initial Tstar(z))
- Linear bottom drag r = 1.1e-3

**Resolution tiers:** quick ~100 km, default ~50 km, research 10-25 km (configurable).

### New code

- `src/legoesm/ocean/experiments/acc_channel.py` — ACCChannelConfig, IC
  (stratification + ridge bathymetry + perturbation), forcings, sponge
- `src/legoesm/ocean/physics/surface_forcing/wind_profiles.py` — added
  `"channel_sine"` wind profile (half-sine, zero at walls, peak at center)
- `docs/ocean_experiments/zhang2024_acc_channel_spec.md` — full MITgcm
  reference spec for the Zhang et al. setup

### Test matrix wiring

- Runner: `run_acc_channel()` in `experiments.py`
- Test cases: latlon_channel (20x36) and mpas_channel (50km) at 30/2 day durations
- First experiment to actually pass `sponge=` to `model.step()` (Eady computed sponge gamma but never wired it through)

### Results (quick mode, 2 days)

| Grid | Status | max_speed | eta_drift | T_drift | Wall |
|------|--------|-----------|-----------|---------|------|
| latlon_channel 20x36 | PASS | 0.137 m/s | 1e-19 m | 4e-6 degC | 6.5s |
| mpas_channel 50km | PASS | 0.162 m/s | 5e-19 m | 3e-6 degC | 191s |

Volume conservation at machine precision. Small T drift from sponge restoring (expected).

### Issue opened

- #202: Linear bottom drag units are wrong — should divide by dz_bottom (m/s units) like quadratic drag, not use Rayleigh damping (1/s units). All existing experiments need r values audited after fix.

### Follow-up items

- Surface heat flux (sinusoidal Q_net) — deferred
- Fix bottom drag units (#202)
- Topographic form stress diagnostic
- Longer runs (weeks-months) to validate wind-TFS equilibration
- Homogeneous (barotropic-only) variant

---

## 2026-04-18: Issue #198 — Bottom Drag and Sponge Layers (Closed)

### Bottom drag fixes

1. **Baroclinic tendency: u_prime → u (lat-lon)**: Bottom drag was applying to perturbation velocity `u_prime`, but the ocean floor sees the total flow. Changed to `u` (full velocity), consistent with MPAS physics pipeline and MOM6.

2. **Barotropic substep drag (both grids)**: Added `-r * U_bar * dz_bot / H_total` inside the barotropic substep loop. This continuously damps the barotropic mode during substeps, rather than relying on a single drag application per baroclinic step. Uses explicit treatment (MOM6 uses implicit `Cg_u`, acceptable for deep-ocean experiments).

3. **MPAS config**: Added `bottom_drag_r` to `MPASOceanConfig` for consistency with lat-lon. Applied as direct tendency on full velocity in `ocean_pe_mpas.py`.

### Sponge layer implementation

Created `src/legoesm/ocean/sponge.py` with:
- `SpongeForcing` NamedTuple: `gamma`, `T_ref`, `S_ref`, optional `u_ref`/`v_ref`
- `compute_sponge_gamma_latlon()`: quadratic ramp from 0 to 1/tau near walls
- `compute_sponge_gamma_mpas()`: same for Voronoi meshes

Sponge applied as a tendency in the baroclinic step (no barotropic sponge — matches MOM6 ALE_sponge):
```
dT/dt += gamma * (T_ref - T)
du/dt += gamma_face * (u_ref - u)
```

Threaded `sponge` parameter through `step()` → `tendencies()` → PE functions on both grids.

### Files changed

- `ocean_pe_latlon_cgrid.py` — u_prime → u for drag; sponge application
- `ocean_pe_mpas.py` — bottom_drag_r + sponge wired
- `barotropic_latlon_cgrid.py` — bottom drag in substep loop
- `barotropic_mpas.py` — bottom drag in substep loop
- `ocean_model_latlon_cgrid.py` — sponge threaded through step/tendencies
- `ocean_model_mpas.py` — sponge threaded through step/tendencies
- `mpas_config.py` — bottom_drag_r field added
- `src/legoesm/ocean/sponge.py` — new module
- `tests/ocean/unit/test_bottom_drag_sponge.py` — 9 CI tests

---

## 2026-04-18: Issue #189 — Biharmonic Smagorinsky Viscosity (Closed)

### Summary

Completed and tested flow-dependent biharmonic Smagorinsky viscosity on both lat-lon C-grid and MPAS Voronoi grids. This is the single highest-impact dissipation improvement — it's the default scheme in MOM6 and the standard recommendation for eddy-resolving ocean models.

### Lat-lon C-grid fixes

1. **Wrap-column periodicity bug fixed**: `strain_rate_cgrid` used `.at[:,-1].set(u[:,0])` (JAX scatter op, complicates gradients). Replaced with structural periodicity: `jnp.concatenate([u[:, :n_lon], u[:, 0:1]])` so the wrap column always references column 0. Also added post-step enforcement `u[:,-1] = u[:,0]` in `ocean_model_latlon_cgrid.py` as belt-and-suspenders.

2. **Biharmonic scaling factor wired**: `biharmonic_scaling_factor(grid)` was implemented but never used. Now applied to constant-coefficient biharmonic (`B_h`) in `ocean_pe_latlon_cgrid.py`. Scales as `(cos(lat)/cos_max)^4` to prevent CFL violation near poles (MOM6 convention). Not needed for Smagorinsky path since `A_smag` already includes area-dependent Delta.

3. **AD-safe sqrt**: Added `1e-30` epsilon to `sqrt(D_T² + D_S²)` in both `smagorinsky_viscosity_cgrid` and `smagorinsky_viscosity_q_cgrid` to prevent NaN gradients at masked points where the deformation rate is zero.

### MPAS Voronoi implementation

Added `smagorinsky_biharmonic_3d()` to `core/operators_voronoi.py` using the MPAS-Ocean/ICON-O approach (NOT the stress-tensor two-pass used on lat-lon). The approach:

```
1. Strain rate: div(u) at cells + curl(u) at vertices
2. Average to edges: D_T_edge = avg(div), D_S_edge = avg(curl)
3. Deformation: |D| = sqrt(D_T² + D_S²) at edges
4. Coefficient: A_smag = (C_smag × sqrt(dcEdge × dvEdge))² × |D|
5. Two-pass: -del2(A_smag × del2(u))
```

Key design choice per ocean-expert recommendation: the stress-tensor approach doesn't fit TRiSK's mimetic framework. The `del2(A × del2(u))` approach composes naturally with existing TRiSK operators and is what operational Voronoi-mesh ocean models use.

Also added constant-coefficient biharmonic (`B_h × del4(u)`) to MPAS using existing `vector_laplacian_del4_3d`.

### Config changes

- `MPASOceanConfig`: added `B_h: float = 0.0` and `C_smag: float = 0.0`
- Both default to zero (harmonic-only, backward compatible)

### CI Test Suite (22 tests, ~16 seconds)

New `tests/ocean/unit/test_smagorinsky.py` covering both grids:

**Lat-lon (12 tests):** energy dissipation, monotone dissipation, uniform flow invariance, 3D=2D consistency, masked regions zero, antisymmetry under velocity reversal, wrap-column periodicity, JAX autodiff, JIT compilation, checkerboard scale-selectivity, strain wrap-column consistency, coefficient magnitude

**MPAS (10 tests):** monotone dissipation, uniform flow invariance, solid body rotation invariance, antisymmetry, masked edge handling, JAX autodiff, JIT compilation, multi-level consistency, checkerboard scale-selectivity, strain rate shape verification

### Files changed

- `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py` — wrap-column stencil fix, AD-safe sqrt
- `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` — biharmonic scaling factor wired
- `src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` — post-step periodicity enforcement
- `src/legoesm/core/operators_voronoi.py` — new `smagorinsky_biharmonic_3d()`
- `src/legoesm/ocean/mpas_config.py` — `B_h` and `C_smag` fields added
- `src/legoesm/ocean/dynamics/ocean_pe_mpas.py` — biharmonic/Smagorinsky wired into momentum tendency
- `tests/ocean/unit/test_smagorinsky.py` — new CI test suite (22 tests)

---

## 2026-04-18: Classical Eady — 500 Days Stable

### Breakthrough

Achieved 500-day stable Eady baroclinic instability with correct exponential growth of the most-unstable mode. The key was switching to **depth-uniform dT/dy** (classical Eady with short-wave cutoff) combined with proper boundary treatment.

### Configuration that works

- N=1.2e-3, U_surface=0.2 (later 0.8 for faster growth), H=5500m, 25°N
- L_d=107km, λ_max=428km, k=2 perturbation
- **Depth-uniform dT/dy** (jet_depth_scale=H_max) → zero interior PV → Eady short-wave cutoff at 280km → no grid-scale Charney modes
- A_h=100, B_h=1e12, bottom drag r=1e-4, sponge layers 2° at walls
- Zonal wavenumber perturbation (not broadband noise)

### Why surface-intensified dT/dy failed

The earlier setup (jet_depth_scale=2000m) created a surface PV gradient that supports **Charney-type instability** with no short-wave cutoff. Grid-scale modes grew faster than any viscosity could damp them. On a β-plane, β also provides interior PV gradient, but this is much weaker than the surface-concentrated gradient.

### Wall instability diagnosis

Spatial tracking revealed blowup always originated at the **north wall** (33°N). Cause: Eliassen-Palm flux from an unstable eastward jet on β-plane is poleward — eddy energy accumulates at the north wall. Free-slip BC allows unbounded tangential velocity. Fix: **sponge layers** (2° width, 1-day relaxation) near N/S walls, standard in MITgcm/MOM6 channel experiments.

### Vertical diffusion sign bug found

Audit of all dissipation schemes revealed A_v and K_v were **anti-diffusive** on the lat-lon C-grid: `jnp.diff(f, axis=-1)` gives `f[k+1]-f[k]` instead of `f[k]-f[k+1]`. Fixed to `f[..., :-1] - f[..., 1:]`. MPAS and physics pipeline versions were correct.

### Ocean checkpoint utility

Added `scripts/ocean_test_matrix/checkpoint.py` with `save_ocean_checkpoint` / `load_ocean_checkpoint` for saving/restoring ocean state to NPZ. Avoids re-running spinup during parameter tuning.

---

## 2026-04-17: Smagorinsky Biharmonic — Resolved

### Problem (original)

Biharmonic Smagorinsky viscosity destabilized simulations — more C_smag = earlier blowup (opposite of expected). Tested with 7 formulations: post-multiplier, sandwich, stress-tensor variants.

### Resolution

The correct formulation is the **MOM6-style two-pass stress-tensor**:

```
Pass 1: u* = stress_divergence(strain(u), 1, 1, normalize=False)  # unnormalized
Pass 2: tend = stress_divergence(strain(u*), A_smag_h, A_smag_q, normalize=True)
```

Key: first pass UNNORMALIZED (returns m/s), second pass uses A_smag (m²/s, NOT B_smag m⁴/s).

### Bugs found during investigation

1. **Post-multiplier `B(x) × ∇⁴(u)` is anti-dissipative** for spatially varying B — cross-terms from grad(B) inject energy. More C_smag → earlier blowup.
2. **B_smag in second pass violates Laplacian CFL** — B × dt / dx² >> 0.5.
3. **Normalized first pass gives wrong units** — 1/(ms²) instead of m/s².
4. The "day 69 blowup" that appeared identical across formulations was actually from the surface-intensified dT/dy (Charney modes at grid scale), NOT the Smagorinsky operator.

### Validation

Monotonicity test passes: all C_smag values are dissipative, KE dissipation scales as C_smag². Currently testing C_smag=0.1 in the classical Eady experiment (depth-uniform dT/dy).

### Remaining

- MPAS implementation
- Full test suite integration
- JAX autodiff verification

---

## 2026-04-16: Eady Baroclinic Instability — From Blowup to Eddies

### Goal

Develop a classical Eady baroclinic instability test case that produces physically correct mesoscale eddies. This validates the ocean model's ability to simulate baroclinic instability, the primary mechanism for mesoscale eddy generation in the real ocean.

### Domain and Parameters

- 1000×2000 km zonally periodic channel at 25°N
- Surface-intensified dT/dy (jet_depth_scale=2000m), warm south / cold north
- Linear EOS, starting from rest
- L_d ≈ 97 km, most-unstable wavelength λ_max ≈ 390 km, e-folding ≈ 36 days
- Tested at 20 km (100×50) and 10 km (200×100) resolution, 20 vertical levels

### Thermal Wind Sign Error

The initial implementation had `dTdy = +f₀Λ/(gα_T)` which gives warm north / cold north → westward surface jet. The correct thermal wind relation `f ∂u/∂z = -g α_T ∂T/∂y` requires `dTdy = -f₀Λ/(gα_T)` for an eastward surface-intensified jet. Fixed by adding the minus sign.

### Why Depth-Uniform dT/dy Creates Wrong Jet Direction

Starting from rest with η≈0, the horizontal pressure gradient at depth z is:

    ∂p/∂y(z) = ρ₀ g ∂η/∂y + ρ₀ g α_T (∂T/∂y) z

With η≈0, the surface (z=0) has no pressure gradient while the bottom (z=-H) has the full hydrostatic integral. This creates strong bottom flow and weak surface flow — the opposite of the expected thermal wind jet. The fix: make dT/dy surface-intensified using `exp(z/D)` depth weighting with D=2000m.

### The Great Blowup Mystery: Constant Viscosity Fails

Every run blew up at day 73-87 during the nonlinear cascade, regardless of viscosity:

| Config | Blowup day |
|--------|-----------|
| B_h=1e10, A_h=0 | ~26 |
| B_h=1e10, A_h=100 | ~35 |
| B_h=1e11, A_h=0 | ~73 |
| B_h=1e11, A_h=100 | ~81 |
| B_h=5e11, A_h=100 | ~82 |
| B_h=1e12, A_h=100 | ~87 |

The blowup pattern was always the same: baroclinic instability grows at the correct Eady rate from day 40-75, then SSH suddenly explodes from ~1m to ~6m in 3 days (much faster than the Eady growth rate), followed by NaN.

**Key observation**: Increasing B_h by 100× (1e10→1e12) only delayed blowup by ~17 days. The viscosity wasn't addressing the root cause.

### Root Cause: Missing Barotropic Energy Sink

Analysis with ocean-expert agent revealed:

1. **Biharmonic viscosity only acts on the baroclinic perturbation velocity** (u_prime in the tendency code). It does NOT remove energy from the barotropic mode.

2. **The barotropic solver has NO momentum dissipation** — only Laplacian diffusion on η (the free surface), which damps SSH signals but not barotropic velocity.

3. **During nonlinear saturation, baroclinic instability cascades energy to the barotropic mode** via Reynolds stress rectification (eddy-eddy interactions driving a mean barotropic flow).

4. **Without bottom drag, barotropic KE accumulates without bound** → SSH explodes.

This is consistent with standard practice: MOM6 and MITgcm **always** include linear bottom drag (r=1e-4) for Eady tests. It is not optional.

### Smagorinsky Biharmonic: Implemented but Not the Solution

We implemented flow-dependent Smagorinsky biharmonic viscosity:
- `strain_rate_cgrid()` — D_T at cell centers, D_S at vertices
- `smagorinsky_viscosity_cgrid()` — A_smag = (C_s×Δ)² × |D|
- `smagorinsky_biharmonic_tendency_cgrid()` — ∇²(B_smag × ∇²(u,v)) sandwich form

Two formulations were tested:
1. **Post-multiplier**: B_smag(x,y) × ∇⁴(u,v) — dimensionally correct but creates instabilities at coefficient boundaries
2. **Sandwich form**: ∇²(B_smag × ∇²(u,v)) — requires B_smag = A_smag × Δ² for correct units

Both failed at the same day as constant viscosity — because the root cause was the missing barotropic energy sink, not insufficient horizontal viscosity. Smagorinsky is still valuable for flow-dependent grid-scale control but is not a substitute for bottom drag.

**Dimensional pitfall**: The naive ∇²(A_smag × ∇²(u,v)) has units 1/(m·s²), not m/s². Must use B_smag = A_smag × Δ² (m⁴/s) to get the correct acceleration units.

### The Fix: Linear Bottom Drag

Adding linear bottom drag r=1e-4 m/s (applied at the bottom vertical level) immediately stabilized the simulation:

    du/dt[..., -1] += -r × u[..., -1]

**Results with B_h=1e11 + bottom drag r=1e-4:**
- Day 0-30: geostrophic adjustment (max_speed ~0.06 m/s)
- Day 30-60: baroclinic instability growing (speed 0.1→0.65 m/s)
- Day 60-85: nonlinear saturation (SSH peaks at 1.4m, then decreases)
- Day 85-120: equilibration (speed ~0.15-0.19 m/s, SSH ~0.06m)
- Day 120-200: stable equilibrium (speed 0.2-0.3 m/s)

The instability saturates because bottom drag removes barotropic KE at the rate it's generated by eddy rectification.

### Missing Zonal Variability: Need Perturbation Seeding

The initial stable runs showed only meridional (zonally banded) structures — no eddies. Cause: the initial condition was perfectly zonally symmetric (perturbation had been set to zero). Baroclinic instability requires k>0 modes which must grow from numerical round-off (~1e-16). At the 36-day Eady growth rate, this takes ~1160 days to reach visible amplitude.

**Fix**: Add broadband white-noise temperature perturbation (0.01 K, standard in MOM6/MITgcm) at the surface level, localized by the jet envelope. This seeds all unstable modes and lets the most-unstable wavelength emerge naturally.

### XLA Compilation Time Issue

Adding new fields to `LatLonCGridOceanConfig` (a NamedTuple) forces complete recompilation of `model.step` (decorated with `@jax.jit(static_argnums=(0,))`). At 200×100×20, this takes 30+ minutes — too slow for iterative development.

**Workaround**: Apply bottom drag as a post-step velocity correction outside the JIT boundary, avoiding NamedTuple changes. The Smagorinsky C_smag field was removed from the config for the same reason; the operator code remains in `latlon_cgrid_operators.py` but is not wired into the tendency.

**Future fix**: Use JAX persistent compilation cache (`jax_compilation_cache_dir`) or refactor the config to avoid NamedTuple type changes triggering retrace.

### Dissipation Scheme Survey (GitHub #194)

A comprehensive survey of dissipation in MOM6, MITgcm, NEMO, POP, MPAS-Ocean, HYCOM, and FESOM2 led to a prioritized implementation plan:

1. **P1**: Biharmonic Smagorinsky (#189) — implemented, needs wiring
2. **P2**: Higher-order tracer advection (#190) — first-order upwind is the dominant spurious diffusion source
3. **P3**: Leith viscosity (#191)
4. **P4**: Adaptive GM (Visbeck) (#192)
5. **P5**: Energy backscatter (#193)

### Lessons Learned

1. **Bottom drag is not optional for nonlinear ocean simulations.** Without a barotropic energy sink, any baroclinic instability experiment will blow up during nonlinear saturation — no amount of horizontal viscosity can fix this.

2. **Constant biharmonic viscosity cannot handle the enstrophy cascade.** The coefficient that controls grid noise kills the physical instability, and vice versa. Flow-dependent (Smagorinsky) viscosity is needed for eddy-resolving simulations.

3. **Think about energy pathways, not just dissipation coefficients.** The root cause was not "too little dissipation" but "dissipation in the wrong mode." The baroclinic mode had plenty of viscosity; the barotropic mode had none.

4. **Zonally symmetric initial conditions cannot produce eddies.** Always seed with broadband noise. The amplitude (0.01 K) is physically negligible but computationally essential.

5. **NamedTuple config changes are expensive in JAX.** Adding a field forces full recompilation. Plan config structure changes carefully or use persistent compilation caches.

### Files Changed

- `src/legoesm/ocean/experiments/eady_uniform.py` — new experiment (IC, forcing, validation)
- `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py` — Smagorinsky operators (strain_rate, viscosity, biharmonic)
- `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` — Smagorinsky tendency (import only, not wired)
- `scripts/ocean_test_matrix/experiments.py` — runner with eu_config parameter, velocity cross-sections
- `scripts/ocean_test_matrix/diagnostic_io.py` — symmetric colorbars, field-tagged cross-section filenames, cmap parameter
- `scripts/ocean_test_matrix/timeloop.py` — max_speed monitoring and early termination (#187)
- `scripts/ocean_test_matrix/setup.py` — domain bounds and C_smag/bottom_drag_r passthrough
- `scripts/ocean_test_matrix/testcase.py` — eady_uniform grid resolution

### Related Issues

- #189 — Biharmonic Smagorinsky (implemented, not wired)
- #190 — Higher-order tracer advection (planned)
- #191 — Leith viscosity (planned)
- #194 — Dissipation roadmap (tracking)
- #187 — Runtime CFL monitoring (closed)

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

The reported -2.87e-3 "remaining drift" turned out to be a **diagnostic artifact**: the `mean_eta` diagnostic used `nanmean` (unweighted), which is biased on grids with non-uniform cell areas. On this 15-75N domain with 4:1 area ratio, the double gyre's asymmetric SSH pattern produced an apparent drift even though total volume `sum(eta * area)` was conserved to machine precision (1.83e-17). Fixed by switching to area-weighted mean.

**Same fix applied to MPAS** (commit 589702b): reformulated `nu_dt_cell * _del2_cell(eta)` to `div(nu_dt_edge * grad(eta))` in `barotropic_mpas.py`. MPAS double gyre eta drift improved from -9.81e-04 to 1.86e-17.

**Final volume conservation (both grids at machine precision):**

| Grid | eta drift (double gyre, 30d) |
|------|------|
| latlon_regional | 1.83e-17 |
| mpas_regional | 1.86e-17 |

**Same issue exists on cubed-sphere** (`barotropic.py` line 231) but is deferred until the face-boundary instability (#100) is resolved.

---

## 2026-04-08: Conservation budget diagnostic (issue #101, Phase 1)

Added `conservation_budget.py` — a grid-agnostic diagnostic that computes the actual change in volume, heat, and salt between two states, and optionally compares against expected forcing input.

```python
from legoesm.ocean.conservation_budget import conservation_budget, print_budget

budget = conservation_budget(state_new, state_old, area, z_coord,
    heat_forcing=expected_heat_input)  # optional
print_budget(budget, "after 1 day")
```

Reports `ConservationBudget` NamedTuple with: old/new integrals, change, expected forcing, and residual (change - forcing) for volume, heat, and salt. Uses precision upcasting for accurate global sums.

For **unforced runs**, the residual equals the change and should be ~0 (machine precision). For **forced/coupled runs**, pass the expected forcing integrals and the residual isolates the numerical error from the physical signal.

Verified on MPAS rest state: volume residual = 0, heat residual = 4.3e-16 relative, salt residual = 0.

---

## 2026-04-08: Test matrix comparison plot fixes

**Problem**: Cross-grid comparison plots for regional grids (MPAS regional, latlon regional) showed incorrect extents and indistinguishable lines.

1. **MPAS regional on global axes**: The MPAS regional regridding outputs to a global 181x360 lat-lon grid with NaN outside the domain. The comparison snapshot and evolution plots used the full coordinate extent instead of cropping to non-NaN data. Fixed by adding bounding-box crop (same logic already used in per-grid snapshots).

2. **Black-on-black timeseries**: The color dict only had entries for global grids (cubed_sphere, latlon, mpas, spectral). Regional grids fell through to default black. Fixed by adding color entries for regional grids (tab:red/tab:green with dashed linestyle).

---

## 2026-04-08: Global barotropic wind-driven experiment

**Goal**: Stand up a global wind-driven barotropic gyre experiment with simplified continent geometry on latlon C-grid and MPAS, and get the two grids producing comparable results.

### Setup

- **Domain**: Single meridional continent (20–60°E) from north polar cap (80°N) to 55°S, open Drake Passage south of 55°S, polar caps at ±80°.
- **Forcing**: 3-belt zonal wind stress: τ_x = −τ₀ cos(2φ) cos²(φ), τ₀ = 0.1 Pa. Gives easterly trades near equator, westerlies at mid-latitudes, polar easterlies tapered to zero at poles.
- **Physics**: Linear bottom drag (r = 1e-4 s⁻¹), lateral viscosity A_h = 5×10⁵ m²/s. No vertical mixing, convection, or heat/freshwater forcing.
- **Initial condition**: Uniform T = 10°C, S = 35 PSU. Flat bottom H = 5500 m.
- **Grids**: latlon C-grid (36×72), MPAS ico3. Cubed-sphere excluded due to face-boundary instability (issue #100).

### Bug 1: Latlon C-grid blowup (stale face masks)

**Problem**: Latlon C-grid blew up at step 300 (~1 day), while geostrophic adjustment was stable for 10 days on the same grid.

**Root cause**: When the simplified continent land mask was applied, only the cell-center `land_mask` was updated. The C-grid's `u_mask` and `v_mask` (at velocity faces) were not recomputed. Stale face masks allowed flow through continent boundaries, creating unbounded pressure gradients and a positive feedback → exponential blowup.

**Fix**: Call `compute_face_masks(land_mask)` after replacing the land mask to regenerate consistent `u_mask` and `v_mask`.

### Bug 2: Latlon C-grid zero velocity (physics never called)

**Problem**: After fix 1, latlon ran 60 days stable but with zero velocity — the wind forcing was never applied.

**Root cause**: `latlon_cgrid_ocean_baroclinic_tendencies()` accepted `physics_fn` as a parameter but never called it. The physics pipeline (wind + bottom drag) was silently dropped.

**Fix**: Added physics call in `ocean_pe_latlon_cgrid.py` before land masking (section 10b). Because the physics pipeline produces cell-center tendencies while C-grid momentum lives at face points, a cell-center proxy state is created for the physics call and the resulting du_dt/dv_dt are interpolated to faces via `_interp_to_u_points` / `_interp_to_v_points`. T/S tendencies are used directly (already at cell centers).

### Investigation: 10-level latlon vs MPAS speed discrepancy

With 10 vertical levels, latlon gave max speed 0.47 m/s while MPAS gave 9.9 m/s — a 20× difference. Step-by-step diagnostics showed:

1. **Initial forcing is identical**: max|du_dt| = 1.85e-6 (latlon) vs 1.86e-6 (MPAS). First ~100 steps track perfectly.
2. **Divergence is gradual**: solutions start splitting around day 1 and reach 2× by day 3.5.
3. **Vertical structure dominates**: Wind forces only the top layer (52 m thick), creating a surface-trapped jet. Bottom drag acts on the bottom layer (1048 m thick) and is negligible. The barotropic-baroclinic splitting communicates momentum to depth differently on the two grids, leading to very different equilibria.

**Conclusion**: The 10-level experiment is not a well-posed barotropic test. Wind and bottom drag act on different levels with no effective vertical coupling, making the result highly sensitive to the split-explicit formulation details.

### Fix: Single-level experiment (truly barotropic)

Added `global_barotropic_wind_1lev` test case with 1 vertical level. With a single layer, wind forcing and bottom drag act on the same layer. The analytical steady-state balance is u = τ_x/(ρ₀·H·r) = 1.77×10⁻⁴ m/s.

**Results (1 level, 60 days)**:

| Grid | Max Speed (m/s) | Analytical | Ratio |
|------|----------------|------------|-------|
| latlon C-grid | 2.14e-4 | 1.77e-4 | 1.21 |
| MPAS ico3 | 1.45e-4 | 1.77e-4 | 0.82 |

Both grids equilibrate properly. The remaining ~20% differences come from Coriolis deflection, lateral viscosity, and boundary effects.

### MPAS edge-normal projection: cos²(angleEdge) ≈ 0.5

**Finding**: On the isotropic ico3 mesh, edges are uniformly oriented, giving ⟨cos²(angleEdge)⟩ = 0.498 ≈ 0.5. A purely zonal wind stress projected onto edge normals (τ_n = τ_x·cos(angleEdge)) delivers ~50% of the energy input compared to a latlon grid where all u-faces are perfectly zonal.

**This is physically correct** — you can't apply a zonal force to a north-south edge. The TRiSK Perot reconstruction (`reconstruct_cell_velocity` in `init_mpas.py`) recovers the correct cell-center velocity from edge-normal components. Verified: for a uniform zonal flow, reconstruction gives mean u_east = 0.997 (expected 1.0).

The apparent 2× speed difference was partly a diagnostic artifact: the test matrix reported `max|u_edge|` (edge-normal speed) for MPAS vs `max(|u|, |v|)` (component speed) for latlon. Fixed by using Perot-reconstructed cell-center speed for MPAS diagnostics.

### MPAS barotropic velocity diffusion

**Finding**: The MPAS barotropic solver applied Laplacian diffusion to both eta AND velocity, while the latlon solver only diffused eta. The velocity diffusion over-damped MPAS barotropic flow.

**Fix**: Removed velocity diffusion from `barotropic_mpas.py`, matching the latlon solver. MPAS equilibrium speed improved from 1.13e-4 to 1.45e-4 m/s (closer to analytical 1.77e-4).

### Test matrix and diagnostic improvements

1. **New test case**: `global_barotropic_wind_1lev` — 1-level barotropic wind on latlon + MPAS, 60 days.
2. **Uniform T/S**: Both `global_barotropic_wind` variants use T = 10°C, S = 35 PSU (no stratification).
3. **Continent width**: 40° (20–60°E), minimum for gap-free coverage on ico3 MPAS mesh.
4. **MPAS speed diagnostic**: Uses Perot-reconstructed cell-center velocity instead of raw edge-normal speed.
5. **Volume-weighted KE**: Added to scalar diagnostics on all grids.
6. **Comparison timeseries**: 4 panels — Mean η, Max |η|, Max Speed, Mean KE. Fixed missing labels.
7. **Longitude convention**: All plots now use 0–360° longitude (was -180–180 for MPAS, 0–360 for latlon).
8. **Forcing profile plot**: `forcing_profile.png` saved in test case directory showing τ_x(lat) and wind stress curl.
9. **`--levels` CLI flag**: Fixed to actually take effect (was baked in at function-definition time).

### Remaining issues

- **10-level vertical coupling**: The multi-level barotropic wind experiment has latlon (0.47 m/s) vs MPAS (9.9 m/s) due to surface-trapped flow and different barotropic-baroclinic splitting. Not a bug — the experiment is not well-posed as a barotropic test with 10 levels.
- **Latlon mean eta drift**: The 1-level latlon case shows a steady mean eta drift of ~5.5e-5/day (MPAS is stable at ~4e-6). Likely related to the non-conservative barotropic diffusion documented earlier.
- **Speed gap at 1 level**: Latlon is 21% above analytical, MPAS is 18% below. Differences from Coriolis, viscosity, and boundary treatment on the two grids.

---

## 2026-04-08: Conservation strategy wrap-up (issue #101 closed)

Issue #101 is resolved for latlon C-grid and MPAS. Summary of what was achieved:

| Grid | Volume (eta) | Heat (T, 1-level) | Heat (T, multi-level) | Salt (S) |
|------|-------------|-------------------|----------------------|----------|
| MPAS | 1.86e-17 | 6e-15 | 6e-15 | 0 |
| latlon C-grid | 1.83e-17 | 1.78e-15 | 2.59e-7 (vertical w) | ~same |
| cubed-sphere | deferred (#100) | deferred (#100) | deferred (#100) | deferred (#100) |

Volume conservation and horizontal tracer conservation are at machine precision on both production grids. The multi-level latlon T gap (2.59e-7) is from the vertical-horizontal transport inconsistency (Phase 2b of #102).

---

## 2026-04-08: Consistent tracer transport operator (latlon C-grid, issue #102)

**Problem investigated**: The latlon tracer transport used `scalar_advection_cgrid` (upwind, velocity-only flux `-div(Tu)`) with a correction `+T*div(hu)/h`. Two bugs:
1. Wrong equation: `-div(Tu)` != `-div(huT)/h` when h varies
2. Different stencils: upwind (tracer) vs centered (mass flux) don't cancel exactly

**Fix applied**: Replaced with the MPAS/TRiSK pattern — use `divergence_cgrid` for BOTH `div(h*u*T)` and `div(h*u)`, with centered face reconstruction for tracers (same interpolation as h). This ensures exact cancellation for uniform T.

**Result**: Per-step conservation is now at machine precision (1.4e-16 after 1 step). But the 10-day T drift is **unchanged** at 8.6e-4. The fix is correct but addresses the secondary error source, not the dominant one.

**Dominant error identified**: The barotropic-baroclinic time-splitting inconsistency. The baroclinic step uses instantaneous velocity for tracer transport, but the barotropic solver determines h_new via 30 substeps with evolving velocities. The effective transport that changes h is NOT the same as the instantaneous transport used for dT/dt.

This is why MPAS achieves 6e-15 without barotropic averaging — on MPAS, the TRiSK operators + consistent edge connectivity mean the per-level transport sums match the barotropic solver's transport. On latlon, the barotropic solver uses different face interpolations (`H_total` at faces) than the baroclinic code (per-layer `h_k` at faces), creating a structural mismatch even at the first substep.

**Conclusion**: The full Phase 2a plan from issue #102 (barotropic transport accumulation) IS needed. The consistent operator fix is a correct prerequisite but not sufficient alone. The fix is kept because it:
- Corrects a real mathematical error
- Ensures machine-precision per-step conservation
- Improves differentiability (removes upwind `jnp.where` kink)
- Is required for Phase 2a to work correctly

---

## 2026-04-08: Flux-form tracer transport with barotropic-averaged transport (issue #102, Phase 2a)

**Goal**: Make horizontal tracer advection exactly consistent with the barotropic continuity equation by using time-averaged barotropic transport instead of instantaneous baroclinic velocity.

### Implementation

Three files changed:

1. **barotropic_latlon_cgrid.py**: Carry state enlarged from `(eta, U_bar, V_bar)` to `(eta, U_bar, V_bar, Hu_sum, Hv_sum)`. Mass fluxes `H_u * U_bar * u_mask` accumulated each substep. After the loop, time-averaged transports `Hu_avg = Hu_sum / n_substeps` returned as a separate tuple alongside the state.

2. **ocean_pe_latlon_cgrid.py**: Horizontal tracer advection removed from the baroclinic tendency function. The tendency now contains only vertical advection + diffusion + physics. Horizontal transport is handled in the step function using the barotropic-averaged transport.

3. **ocean_model_latlon_cgrid.py**: Flux-form horizontal tracer update using barotropic-averaged transport distributed to layers:
   ```
   h_new * T_new = h_old * T_mid - dt * div(Hu_avg_k * T_face)
   ```
   where `Hu_avg_k = Hu_avg * h_old_k / H_old` distributes the 2D barotropic transport proportional to layer thickness.

### Key insight: pure flux form, not skew-symmetric

The initial attempt used the skew-symmetric form `hT = h*T - dt*(div(huT) - T*div(hu))`. This is NOT globally conservative for non-uniform T because `sum(T * div(hu) * area) != 0` when T varies spatially. The pure flux form `hT_new = h_old*T_mid - dt*div(mf*T_face)` IS exactly conservative by the divergence theorem: `sum(div(F)*area) = 0` for any flux F with no-flux boundaries.

### Results

| Test | Heat conservation | Notes |
|------|------------------|-------|
| **1-level barotropic** | **1.78e-15** | Machine precision! Proves horizontal transport is exactly conservative. |
| 10-level stratified | 2.59e-7 | Limited by vertical w inconsistency (Phase 2b) |
| Rest state | 7.98e-15 | Unchanged, still perfect |

The 1-level result proves the horizontal flux-form + barotropic-averaged transport machinery works correctly. The 10-level residual comes from the vertical velocity `w` being diagnosed from the instantaneous baroclinic divergence, not from the barotropic-averaged transport. This vertical-horizontal inconsistency is Phase 2b.

### What improved vs where we started

| Metric | Before all fixes | After Phase 2a |
|--------|-----------------|----------------|
| Volume (eta) conservation | -1.16e-2 (30d) | 1.83e-17 (machine precision) |
| Horizontal T conservation (1-level) | ~8.7e-4 | 1.78e-15 (machine precision) |
| Multi-level T conservation | ~8.7e-4 | 2.59e-7 (vertical limited) |
| Additive fixer | On (harmful for forcing) | Off (unnecessary) |
| Diagnostic accuracy | Unweighted mean (biased) | Area-weighted (correct) |

### Phase 2b (done)

Diagnosed w from barotropic-averaged per-layer divergence. Vertical tracer advection moved to step function. 10-level heat: 2.59e-7 to 8.69e-14. Geostrophic adj T drift: 8.7e-4 to 2.01e-11. Latlon matches MPAS quality.

---

## Issues and PRs

### Open issues
- #100 — Cubed-sphere ocean face-boundary instability (exponential blowup at face boundaries in dynamic simulations)
- #106 — Ocean surface forcing cannot combine wind stress with temperature restoring (architecture limitation)

- #99 — Remove spectral grid from ocean (land boundary issues, not worth investing)
- #87 — Latlon A-grid instability (C-grid fixes in #98; A-grid removed from test matrix)
- #88 — Regional MPAS mesh (pole fix + test matrix in #98)
- #81 — Rest-state stability (diagnostic artifact fix in #98)

### Closed issues
- #189 — Biharmonic Smagorinsky viscosity (resolved: lat-lon stress-tensor + MPAS TRiSK two-pass, 22 CI tests)
- #105 — Vector Laplacian and barotropic diffusion fixes for latlon C-grid (resolved: proper grad(div)-curl×grad(curl) + disabled excessive SSH diffusion)
- #103 — MPAS Coriolis double-counting in split-explicit stepping (resolved: perturbation velocity + semi-implicit Coriolis)
- #101 — Conservation strategy (resolved: conservative diffusion, h_old/h_new, fixer disabled, budget diagnostic)
- #102 — Flux-form tracer transport (resolved: Phase 2a horizontal + Phase 2b vertical)
- #94 — Split-explicit tracer conservation (superseded by #102)

### PRs
- #104 — Cubed-sphere rest-state fix (merged)
- #98 — Consolidated ocean model fixes (merged, resolved #89, #92, #95)
- #89 — Ocean exploration (closed, superseded by #98)
- #92 — Regional mesh pole fix (closed, superseded by #98)
- #95 — C-grid latlon stability (closed, superseded by #98)

---

## Recent Development Activities (2026-04-09)

### Ocean Test Matrix Analysis and Current Status

**Comprehensive ocean journey review** to understand where ocean development stands and what bugs need solving.

**Current Status Summary**:
- **Working grids**: latlon C-grid, MPAS (2 of 3 target grids production-ready)
- **Broken grid**: cubed_sphere (face-boundary instability #100)  
- **Removed grid**: spectral (fundamental limitations with land boundaries)
- **Test matrix**: 37 test cases spanning rest states → complex multi-layer dynamics

**Key Recent Fixes Validated**:
- ✅ Volume conservation: Machine precision (1.83e-17) on MPAS and latlon C-grid
- ✅ Tracer conservation: Flux-form transport with barotropic-averaged transport  
- ✅ MPAS Coriolis fix (#103): Perturbation velocity approach prevents double-counting
- ✅ Latlon C-grid improvements (#105): Vector Laplacian + disabled excessive diffusion

### Regional Baroclinic Gyre Experiment Development

**Motivation**: Create a test case that validates recent ocean fixes (especially #103 Coriolis double-counting) with realistic multi-level baroclinic dynamics.

**Implementation**:
- **New experiment module**: `src/legoesm/ocean/experiments/baroclinic_gyre.py`  
- **Domain**: Same as successful barotropic_double_gyre (0-120°E, 15-75°N)
- **Physics**: Double-gyre wind pattern + realistic 20°C→2°C exponential stratification
- **Target grids**: `mpas_regional`, `latlon_regional` 
- **Duration**: 60 days (5 days quick mode)

**Scientific Value**:
- Tests **multi-level momentum transfer** (validates Coriolis fix #103)
- Tests **thermal wind dynamics** with realistic stratification
- Tests **long-term conservation** over extended baroclinic integrations  
- Enables **cross-grid validation** (latlon C-grid vs MPAS performance)

**Integration**: Added to ocean test matrix as cases 18-19 (now 39 total test cases)

**Test Results**: ✅ **PASS** - `latlon_regional` quick test successful
- Speed: 0.0703 m/s (realistic baroclinic gyre velocities)
- Volume conservation: 5.33×10⁻¹⁸ (machine precision)
- Heat conservation: 0.000°C (perfect conservation)
- Performance: 3.9s for 5-day integration

### Surface Forcing Architecture Limitation Discovery

**Problem Identified**: Ocean surface forcing system cannot combine wind stress with temperature restoring, limiting realistic baroclinic experiments.

**Root Cause**: `SurfaceForcingConfig.scheme` only allows one forcing type:
- `"prescribed"`: Wind OR heat flux (not both)
- `"restoring"`: Thermal only (no wind)  
- `"bulk_formulas"`: Realistic fluxes (different use case)

**Scope**: Universal limitation affecting all ocean grids (latlon, MPAS, etc.) - not grid-specific

**Impact**: 
- Blocks realistic baroclinic experiments requiring thermal-mechanical coupling
- Had to disable temperature restoring in baroclinic_gyre experiment
- Limits air-sea interaction studies

**Documentation**: Created **Issue #106** with detailed analysis and proposed solutions

**Workaround**: baroclinic_gyre uses wind-driven stratified flow (still scientifically valuable for testing multi-level dynamics)

### Next Priority Actions

1. **Validate MPAS regional baroclinic gyre** - test Coriolis fix (#103) cross-grid consistency
2. **Fix cubed-sphere face-boundary instability (#100)** - last critical blocker for 3-grid support
3. **Extend surface forcing architecture (#106)** - enable combined wind+thermal forcing
4. **Run full ocean test matrix** - validate all recent changes end-to-end

**Ocean Model Readiness**: 2 of 3 target grids production-ready, with comprehensive test validation framework in place.

---

## 2026-04-11: MPAS visualization artifacts fixed (issue #135)

### Problem

MPAS regional baroclinic gyre SST plots showed severe artifacts: cold temperature bands (9.8–15°C) at land-ocean boundaries, 396 spurious cold spots in the regridded data. The native MPAS data was correct (ocean cells 19.521±0.0001°C, land cells 0.000°C) — the problem was entirely in the visualization pipeline.

### Root cause

`_bin_to_latlon()` built a KDTree from **all** MPAS cell centers (ocean + land), then used 6-nearest-neighbor IDW interpolation. Near coastlines, target grid points blended ocean cells (~19.5°C) with land cells (0°C), producing unphysical intermediate values. The post-hoc land mask only removed points classified as land — ocean points near the coast that were contaminated by IDW averaging remained corrupted.

### Research: standard MPAS visualization approaches

Investigated the MPAS community practices (MPAS-Tools, UXarray, Project Pythia cookbook):

1. **Native PolyCollection plotting** — Draw each Voronoi cell as its actual polygon. No interpolation, zero artifacts. Standard for small-to-moderate meshes in `mpas_tools.viz`.
2. **Delaunay triangulation with `tripcolor`** — Decompose cells into triangles via `mesh_to_triangles`. Good for smooth contouring.
3. **Mask-aware IDW** — Exclude land cells from the KDTree before interpolation. Recommended by xESMF for mask-aware regridding.
4. **UXarray/Datashader** — Overkill for ~1000 cells, designed for million-cell meshes.

### Fix: two complementary approaches

**1. Mask-aware IDW interpolation** (for regridded data, cross-sections, NPZ output)

Added `ocean_mask` parameter to `_build_latlon_weights()`. When provided, land cells are filtered out before building the KDTree, so they can never contribute to interpolation weights. The returned indices are mapped back to the full array so `_apply_weights()` works unchanged. Propagated through `_bin_to_latlon`, `_regrid_2d`, `_regrid_3d_level`, `_bin_cross_section`, and all callers.

**Result**: Cold artifacts in regridded NPZ data: **396 → 0**. SST range now 19.52–19.53°C (all physically realistic).

**2. Native Voronoi polygon plotting** (for MPAS snapshot plots)

Added `_build_voronoi_polygons()` and `_plot_voronoi_field()` using matplotlib `PolyCollection`. Each Voronoi cell is drawn as its actual polygon with `edgecolor='face'` to eliminate gaps. Land cells rendered in gray. Ocean cells colored by field value. No interpolation at all.

Threaded `mesh` parameter through `_save_case_diagnostics` → `_save_snapshot_plots`. When `mesh` is provided and `coord_kind` is MPAS, the native polygon path is used instead of regrid+imshow. Updated all 14 runner call sites with `mesh=grid if coord_kind == "mpas" else None`.

**Also removed**: The SST `> 1.0°C` color range hack that was working around the artifacts.

---

## 2026-04-11: Per-panel colorbars for snapshot evolution plots

### Problem

Snapshot evolution plots (`snapshots_SST.png`) used a single shared colorbar across all 8 time panels. The temporal cooling trend (19.53→19.46°C over 30 days, range ~0.07°C) dominated the color range, making the ~0.004°C spatial patterns within each panel invisible. The plots looked spatially uniform even though spatial structure was developing.

### Fix

Changed both the Voronoi and regrid+imshow plotting paths to compute per-panel vmin/vmax and add a colorbar per panel. Each panel's colorbar is fitted to that snapshot's data range. Spatial patterns (meridional SST gradients from Ekman convergence, boundary cooling) are now clearly visible at every timestep.

Same fix applied to `_create_comparison_evolution()` — the cross-grid evolution comparison now uses per-panel colorbars so each panel reveals its spatial structure, and the different magnitudes between grids are honestly shown on separate colorbars rather than hidden by a shared one.

For the vertical section evolution plot, the colorbar is shared per row (per grid) and pinned to the initial condition range (t=0), so any departure from the initial stratification would show as a visible color change.

---

## 2026-04-11: Vertical section evolution comparison plot

Added `_create_comparison_vertical_evolution()` — a new cross-grid comparison showing meridional temperature cross-sections (latitude × depth at 60°E) evolving over time. Layout: rows = grids, columns = time steps. One shared colorbar per row pinned to the initial condition range. Wired into `_create_cross_grid_comparisons` for `baroclinic_gyre` cases.

At 30 days the stratification remains very close to the IC (~2–19.5°C) — the wind-driven isopycnal tilting is too small relative to the background to be visible. Longer integrations or surface buoyancy forcing (issue #106) would produce visible thermocline tilting.

---

## 2026-04-11: --replot mode for ocean test matrix

### Problem

Every visualization change required rerunning the full simulation, even though the data was already saved in NPZ files. For a 30-day run this meant ~3 minutes of unnecessary computation; for longer runs (multi-year), it would be prohibitive.

### Implementation

Added `--replot` flag to `run_ocean_test_matrix.py`. When set:

1. Discovers existing result directories by searching for `snapshots_latlon.npz` files
2. Applies `--only` and `--grid` filters to select which cases to replot
3. Loads the saved NPZ data and reconstructs the `snapshots` dict expected by plotting functions
4. Regenerates per-case snapshot plots, cross-sections, and vertical profiles
5. Regenerates all cross-grid comparison plots

**Performance**: Replotting the baroclinic gyre (2 grids) takes **9.5 seconds** vs **172 seconds** for simulation+plot (18× faster).

**Limitation**: MPAS snapshot plots in replot mode use regridded latlon data (since the VoronoiMesh isn't available without running the model). Native polygon rendering requires the simulation path.

Usage:
```bash
python scripts/run_ocean_test_matrix.py --only baroclinic_gyre --replot
```

---

## 2026-04-11: MPAS performance benchmarking (issue #137)

### Finding

Benchmarked per-step cost for MPAS vs latlon at comparable cell counts:

| Grid | Cells | ms/step | JIT compile |
|------|-------|---------|-------------|
| latlon 24×48 | 1,200 | **1.86** | 1.18s |
| MPAS 300km | 1,044 | **16.16** | 0.60s |

**8.7× slower per step** despite having 13% fewer cells.

### Root cause

Not grid size — it's the computational pattern. Structured grids use regular array stencils (simple slicing) that JAX/XLA fuses into efficient vectorized kernels. MPAS operators use indirect addressing through connectivity arrays (`cellsOnEdge`, `edgesOnCell`, etc.) — gather/scatter patterns that XLA cannot optimize as well.

This is a known tradeoff for unstructured grids: geometric flexibility at the cost of computational efficiency per DOF. Filed as issue #137 with optimization approaches (fused scan, sparse matrix operators, GPU acceleration).

**Practical impact**: A 5-year baroclinic gyre takes ~16 min on latlon vs ~2.4 hours on MPAS.

The same `VoronoiMesh` and TRiSK operators are used by the atmosphere MPAS dycores (shallow water, primitive equation, compressible Euler), so any optimization would benefit both components.

---

## 2026-04-11: Cross-grid physics consistency fixes (issue #114)

Resolved all 6 subproblems from the automated code audit, using ocean-expert, dycore-expert, differentiability-expert, and slopbuster agents for consensus-driven fixes.

### Items fixed

**1. Physics Field dims hardcoded to cubed-sphere**: `combined.py` was creating `OceanTendencies` with `("face", "x", "y", "level")` dims regardless of grid type. Since `Field.dims` is part of pytree `aux_data` (static metadata), this could cause pytree structure mismatches under `jax.grad`. Fixed by inferring from `state.T.dims` / `state.eta.dims`, matching the pattern already used by `zero_ocean_tendencies` in the same file.

**2. EOS iteration mismatch**: `compute_ocean_rho_and_pressure` (used by plume convection) did only 1 EOS-pressure iteration while `compute_ocean_rho` (used by all other physics) did 2. At 4000m depth, this caused ~0.018 kg/m³ density error — significant for convective dynamics where buoyancy differences are O(0.001–0.01 kg/m³). Refactored `compute_ocean_rho_and_pressure` to delegate to `compute_ocean_rho`, eliminating duplication and ensuring consistency.

**3. Stale OceanConfig fields**: Removed `edge_blend_strength` and `edge_blend_depth` from `OceanConfig` — defined and validated but never used in any computation (separate from actively-used `BathymetryConfig` fields). Removed dead test `test_ocean_model_fv_tracer_transport` that referenced non-existent `use_fv_tracer_transport` field. Fixed broken coupler test with same stale field.

**4. Restoring forcing grid compatibility**: Already fixed in current code — uses `grid.grid_lat`. Cleaned stale `CubedSphereGrid` type hint.

**5. Shortwave penetration z inconsistency**: Expert analysis showed error is O(eta/H) ~ O(1e-4), negligible vs Jerlov parameter uncertainty. Standard practice in MOM6/NEMO/POP. Added documentation comment; no code change needed.

**6. Cross-module private imports**: Already resolved in current code. No changes needed.

### Slopbuster review

Ran slopbuster on all changes. All core changes **CLEAN**. Fixed redundant `cKDTree` import and EOS code duplication during review.

---

## 2026-04-11: 5-year baroclinic gyre — latlon blowup diagnosis (issue #138)

### The 5-year run

Launched a 5-year (1825-day) baroclinic gyre simulation on both grids:

| Grid | Duration | max_speed | T drift | Outcome |
|------|----------|-----------|---------|---------|
| MPAS 300km | 1825 days (2.3 hrs) | 0.052 m/s | 0.000°C | **PASS** |
| latlon 24×48 | 456 days (4.8 min) | 0.094 m/s | 0.008°C | **BLOWUP** |

MPAS ran the full 5 years perfectly stable. Latlon blew up at day 456 (~1.25 years).

### Root cause analysis

The ocean expert diagnosed a **positive feedback loop** driven by two numerical deficiencies:

**Primary: Non-conservative vertical advection** — The tracer update used tendency-form vertical advection (`-w * dT/dz`) inside a flux-form horizontal framework. The tendency form does not telescope when summed over levels, creating a **systematic heat source** that grew with the circulation. Evidence: latlon drifted +0.008°C in 456 days with zero surface forcing, while MPAS drifted 3.2×10⁻⁷°C.

**Secondary: Unlimited centered horizontal tracer interpolation** — Centered (arithmetic mean) face interpolation permits new extrema near sharp gradients, producing a checkerboard pattern in w concentrated at the northern boundary.

**Not a CFL violation**: Advective CFL was only 0.0004 at blowup.

**Why MPAS survived**: So coarse (300km) that baroclinic structure barely develops — T spatial variation 50-600× smaller than latlon. The positive feedback never triggered.

### Fix 1: Flux-form vertical tracer advection

Added `flux_form_vertical_tracer_advection()` in `vertical.py`:
- Computes `F_top[k] - F_bot[k]` where `F = w * T_upwind` at interfaces
- Uses first-order upwind: at interface k, `w > 0` (upward) → `T_face = T[k]` (from below)
- Returns flux divergence NOT divided by h (units: `[tracer]*[m/s]`)
- Telescopes exactly: `sum(flux_div) = F[surface] - F[bottom] = 0`

Tracer update changed from:
```
hT_new = h_old * (T + dt * vert_tendency) - dt * div_h(flux)
```
to:
```
hT_new = h_old * T - dt * vert_flux_div - dt * div_h(flux)
```

**Hidden bug found**: `diagnose_w_from_flux_div` was multiplying flux divergence by `dz_ref` when the C-grid input was already thickness-weighted (`div(h*u)`, units m/s). This made w ~100× too large. The old tendency-form code had a compensating `1/dz` in the gradient calculation, so the errors canceled. The new flux form exposed it. Added `thickness_weighted=True` parameter for the C-grid caller.

**Result**: T drift went from 0.001°C in 5 days to **0.000°C in 30 days**. The spurious heat source is eliminated.

### Fix 2: First-order upwind horizontal tracer advection

Added `_upwind_to_u_points()` and `_upwind_to_v_points()` in `ocean_pe_latlon_cgrid.py`:
- u-face j: if `mass_flux_u > 0` (eastward), upwind is cell (j-1); else cell j
- v-face i: if `mass_flux_v > 0` (northward), upwind is cell (i-1); else cell i
- Uses `jnp.where` for data-dependent selection (JAX-compatible)
- Periodic longitude handled via `jnp.roll` + concatenation
- Solid wall boundaries at latitude edges (zero tracer, zero flux)

The dycore expert independently verified the staggering conventions and confirmed the sign conventions match. Both agents agreed on the formulas.

Replaced centered interpolation in the tracer flux computation:
```python
# Before (permits new extrema):
tr_u = _interp_to_u_points(tr)
tr_v = _interp_to_v_points(tr)

# After (monotonicity-preserving):
tr_u = _upwind_to_u_points(tr, mass_flux_u)
tr_v = _upwind_to_v_points(tr, mass_flux_v)
```

The centered interpolation functions are preserved for non-tracer quantities (thickness, velocity) where monotonicity is not a concern.

### Verification

All 90 ocean unit tests pass. Baroclinic gyre quick test: PASS with max_speed=0.0657 m/s, T_drift=0.000°C.

### 5-year validation results

Launched with both fixes to test long-term stability. Output in `results/ocean_5yr_v2/`.

---

## 2026-04-11: Issue #113 analysis — MPAS physics pipeline

Sent ocean-expert, dycore-expert, and differentiability-expert agents to analyze. Key consensus:

### Sub-problem 1: Duplicate _fill_land_cells_mpas

Already mostly resolved. A shared mpas_fill.py exists and is used by 2 of 3 call sites. Only ocean_model_mpas.py still inlines the logic (~18 lines). Fix: replace with 4 lines calling the shared function.

### Sub-problem 2: MPAS silently ignores advanced physics

All three agents agree: do NOT unify the physics pipelines. The state types, tendency types, and operator requirements are fundamentally incompatible. Forcing unification creates AD risks (pytree structure mismatch).

Recommended layered approach:
- Share raw physics kernels (already grid-agnostic)
- Keep separate integration bridges per grid type
- Extend make_mpas_ocean_physics with MPAS-specific wrappers for 6 trivial modules (constant mixing, enhanced_diffusion, plume, shortwave, restoring, quadratic drag)
- Defer lateral mixing (harmonic, biharmonic, GM-Redi) — needs MPAS-specific operators, significant effort

---

## 2026-04-11: Barotropic time-averaging for split-explicit stability (commit 06f4de9)

Added time-averaging of eta, U_bar, and V_bar over barotropic substeps for use in the baroclinic coupling. The barotropic solver now accumulates and averages these fields across all substeps, reducing mode-splitting noise that previously required higher barotropic diffusion. The 3D velocity correction after the barotropic solve uses the time-averaged `U_bar_avg` instead of the final-substep value, preserving baroclinic structure while using a smoother barotropic component.

---

## 2026-04-12: Baroclinic gyre has no vertical transport — flat interior isotherms (issue #140)

### The problem

After the #138 fixes (flux-form vertical advection, upwind horizontal tracer), the 5-year baroclinic gyre simulation ran stably but the **ocean interior remained completely flat**. SST showed a double-gyre pattern, but:
- Vertical cross-sections showed perfectly flat isotherms at all depths — identical to initial condition
- Vertical velocity at 133m was O(1e-16) m/s — **machine zero**
- N-S surface temperature contrast was only 0.003°C after 5 years
- The wind-driven circulation was purely barotropic with no baroclinic coupling to the interior

This is deeply unphysical. In a wind-driven double gyre, Ekman pumping/suction should create O(1e-5 to 1e-6) m/s vertical velocities, tilt isotherms, and develop thermocline structure.

### Root cause: barotropic-only tracer transport gives w ≡ 0

The tracer mass fluxes in `ocean_model_latlon_cgrid.py` distributed the 2D barotropic transport uniformly to all layers:

```python
frac_u = h_u_old / H_u_old  # = dz_ref_k / H_max in z-star (spatially constant!)
mass_flux_u = Hu_avg * frac_u  # every layer gets the same velocity U_bar
```

In z-star coordinates, the fraction `frac_u_k = h_u_k / H_u = dz_ref_k / H_max` is **spatially constant** (the Jacobian cancels between numerator and denominator). This means every layer has velocity `u_k = Hu_avg / H_u = U_bar` for all k — the baroclinic velocity structure is completely discarded.

When w is diagnosed from these uniform mass fluxes, the per-layer divergence is `flux_div_k = (dz_ref_k / H_max) * div(Hu_avg)`. After bottom-up cumulative summation and the z-star sigma correction:

```
w_euler[k] = -(div(Hu_avg)/H_max) * S_k       where S_k = sum_{j>=k} dz_ref_j
sigma_k    = S_k / H_max                        (z-star coordinate definition)
w[k]       = w_euler[k] - sigma_k * w_euler[0] = 0    ∀k
```

**w is identically zero at all interior levels.** This was independently verified by both the ocean expert and dycore expert agents, with the dycore expert providing the full mathematical proof.

### Secondary bug: thickness double-counting in momentum w

In `ocean_pe_latlon_cgrid.py` (line 321), `flux_div_k = div(h*u)` already includes layer thickness, but `diagnose_w_from_flux_div` was called without `thickness_weighted=True`. This multiplied by `dz_ref` again, making the vertical advection of momentum wrong by a factor of layer thickness (50m at surface, ~1000m at depth). This was partially masked because the pressure gradient and wind forcing dominate the momentum budget, but it made the vertical structure of velocity less physical.

### Fix (commit 54982ce)

**Primary fix**: Replaced barotropic-only tracer mass fluxes with the full 3D velocity (which preserves baroclinic shear from pressure gradients and wind stress), with a uniform barotropic correction so the depth-integrated transport matches `Hu_avg` for consistency with the barotropic continuity equation:

```python
u_3d = state_new.u.data                          # full 3D velocity with baroclinic shear
Hu_3d = jnp.sum(u_3d * h_u_old, axis=-1)         # current depth-integrated transport
delta_U = (Hu_avg - Hu_3d) / jnp.maximum(H_u_old, 1e-10)  # barotropic correction
u_corrected = u_3d + delta_U[..., jnp.newaxis]    # shear preserved, depth-integral = Hu_avg
mass_flux_u = h_u_old * u_corrected * u_mask_3d
```

This follows the Hallberg & Adcroft (2009) / Shchepetkin & McWilliams (2005) approach: the baroclinic shear is preserved unchanged (`u_corrected[k] - u_corrected[k'] = u_3d[k] - u_3d[k']`), while the depth-integrated transport matches `Hu_avg` exactly (`sum_k(h_k * u_corrected_k) = Hu_avg`).

**Secondary fix**: Added `thickness_weighted=True` to the momentum w diagnosis call.

### Results

1-year baroclinic gyre comparison (latlon_regional 24×48):

| Metric | Before fix (5 yr) | After fix (1 yr) |
|--------|-------------------|------------------|
| max\|w\| at 133m | 1e-16 m/s (machine zero) | 3.5e-5 m/s |
| dT_NS_sfc | -0.003°C | 1.118°C |
| dT_NS_thermo | -0.001°C | 0.313°C |
| max_speed | 0.066 m/s | 0.130 m/s |
| T_drift | 0.000°C | 0.000°C |
| eta_drift | 1.2e-16 | 7.5e-16 |

Diagnostic plots in `results/ocean_1yr_fix/baroclinic_gyre/latlon_regional/24x48/` show:
- **SST**: Strong N-S contrast with western boundary current signatures
- **Vertical cross-sections**: Isotherms tilting, thermocline deepening toward basin center
- **w at 133m**: Real Ekman pumping/suction pattern (downwelling in subtropical gyre, upwelling at boundaries)
- **Vertical profiles**: Surface cooling and thermocline evolution visible in upper 500m
- **Conservation**: Exact to machine precision — no degradation from the fix

All 154 ocean unit tests pass.

---

## 2026-04-13: MPAS vertical tracer advection fix (issue #145)

### The problem

The MPAS ocean model had **no vertical tracer advection** — the same root cause as the latlon fix in #140, but never ported to MPAS. Tracers only moved horizontally, vertical velocity was never diagnosed, and isotherms remained flat in baroclinic experiments.

### Three-file fix mirroring the latlon approach

1. **`barotropic_mpas.py`**: Added `Hu_sum` accumulator to the barotropic scan. At each substep, depth-integrated edge transport `H_e * u_bar * edge_mask` is accumulated. Returns `Hu_avg = Hu_sum / n_substeps` as a third value (matching the latlon solver pattern).

2. **`ocean_pe_mpas.py`**: Removed flux-form horizontal tracer advection from the PE tendencies. Only horizontal diffusion (`K_h * lap(T)`) and physics remain. Advection is now handled entirely in `step()` for consistency with barotropic-averaged transport. After merge with main, uses batched 3D operators (`divergence_cell_3d`, `gradient_edge_3d`).

3. **`ocean_model_mpas.py`**: Full flux-form tracer transport in `step()`:
   - Barotropic correction: uniform velocity shift so `sum_k(h_e * u_k) = Hu_avg`
   - Per-layer mass fluxes preserving baroclinic shear → non-zero w
   - Vertical velocity diagnosed from `divergence_cell_3d(mass_flux)` via continuity
   - Horizontal: first-order upwind interpolation to edges
   - Vertical: `flux_form_vertical_tracer_advection(tr, w)` — upwind at interfaces
   - Conservative update: `h_new * T_new = h_old * T_mid - dt * vert - dt * horiz`

### Results

5-year baroclinic gyre (sin² wind):

| Metric | latlon (5 yr) | MPAS (60 d, pre-fix would be ~0) |
|--------|---------------|----------------------------------|
| dT_NS_sfc | -0.911°C | -0.143°C |
| dT_NS_thermo | -1.281°C | -0.042°C |
| max_speed | 0.073 m/s | 0.064 m/s |
| T_drift | 0.000°C | 0.000°C |
| eta_drift | ~1e-15 | ~1e-17 |

All 252 ocean tests pass. Differentiability tests pass.

### Merge with main

Merged 7 commits from main including batched 3D TRiSK operators (#137 fix). Resolved conflict in `ocean_pe_mpas.py` — took main's batched operators, applied our advection removal. Also switched `step()` from vmapped `divergence_cell` to native `divergence_cell_3d`.

---

## 2026-04-13: Wind-driven gyre test matrix variants

### New wind profile: `double_gyre_tapered`
Cosine double-gyre wind with smooth sin² taper to zero at basin walls. Implemented in `prescribed.py` and `mpas_physics.py`. Properties:
- Basin-integrated wind ≈ 0 (unlike sin² which has net eastward stress)
- Avoids spurious coastal Ekman transport at walls
- Curl pattern identical to standard cosine → same gyre structure

### Test matrix additions
Added both cosine and sin² wind variants for barotropic and baroclinic gyre experiments:

| Case | Wind | Grid |
|------|------|------|
| `barotropic_double_gyre` | cosine (existing) | mpas_regional, latlon_regional |
| `barotropic_double_gyre_sin2` | sin² (new) | mpas_regional, latlon_regional |
| `baroclinic_gyre` | sin² (existing) | mpas_regional, latlon_regional |
| `baroclinic_gyre_cos` | cosine (new) | mpas_regional, latlon_regional |

Hierarchical output structure: `barotropic_double_gyre/{variant}/`, `baroclinic_gyre/{variant}/` (matching rest_state pattern). Generalized `_VARIANT_GROUPS` replaces hardcoded `_REST_STATE_GROUP`.

Added `wind_profile` and `wind_buffer_deg` fields to `BaroclinicGyreConfig` for parameterization.

---

## 2026-04-13: Plotting fixes

1. **Vertical cross-sections**: Fixed `invert_yaxis()` toggle bug — calling inside loop with `sharey=True` toggles inversion, leaving even-panel counts non-inverted. Now called once after loop. Surface is correctly at top.
2. **w snapshots**: Force symmetric colorscale centered at 0 for `w_*` fields.
3. **u_sfc/v_sfc velocity maps**: Added to baroclinic gyre diagnostics and replot mode.
4. **Symmetric colorscale**: Also applied to `u_sfc`, `v_sfc` fields (diverging quantities).

---

## 2026-04-13: Code quality cleanup (slopbuster + modularity audit)

Ran slopbuster and modularity audits on all recently changed files. Fixed 10 of 12 flagged issues:

**Modularity fixes:**
- **Shared wind profiles**: Extracted all 7 wind stress profile computations from `prescribed.py` (100+ lines) and `mpas_physics.py` (60+ lines) into shared `surface_forcing/wind_profiles.py`. Both callers now use `compute_wind_stress(lat, cfg)`. Net 82 lines deleted.
- **Inline land fill**: Replaced 15-line inline Neumann fill in `ocean_model_mpas.py` with existing `fill_land_cells_mpas()` helper.
- **Global wind helper**: `_make_global_wind_physics()` now delegates to `_make_gyre_physics("global_wind")`.
- **Hardcoded depths**: `_add_stratification()` in `baroclinic_gyre.py` now uses `z_coord.z_full_ref` instead of a hardcoded 10-level depth array.

**Slopbuster fixes:**
- Removed duplicate `FIELD_RANGES["baroclinic_gyre"]` dict key (silent overwrite)
- Removed dead `_REST_STATE_GROUP` backward-compat alias
- Removed unused imports (`g`, `Path`) and dead variable (`T_uniform`) in `baroclinic_gyre.py`
- Fixed `_add_stratification` signature to accept `z_coord` parameter
- Fixed misnumbered step comment (#7 → #10) in `ocean_model_mpas.py`
- Fixed misleading `CubedSphereGrid` type annotation in `prescribed.py` (works with any grid)

**Remaining architectural items** (deferred to #148):
- Barotropic solver return type inconsistency across grids
- Flux-form tracer transport duplication between latlon and MPAS step()
- Cubed-sphere barotropic solver lacks Hu_avg for transport-consistent advection

---

## Issues and PRs (updated 2026-04-13)

### Open issues — numerics & physics
- #155 — Add CFL monitoring and adaptive barotropic substep warning
- #154 — Latlon: hardcoded S_ref=35.0 in virtual salt flux
- #153 — Latlon: missing relative vorticity flux in momentum equation
- #152 — MPAS: missing vertical advection of momentum (w * du/dz)
- #151 — Latlon: missing freshwater in barotropic solver (+ MPAS double-counting)
- #150 — Latlon: vertical tracer diffusion skipped when physics pipeline active
- #149 — MPAS barotropic returns instantaneous eta instead of time-averaged
- #113 — MPAS physics pipeline limited (only prescribed wind + linear drag)
- #109 — A-grid/cubed-sphere baroclinic pressure double-counts free-surface
- #106 — Surface forcing cannot combine wind + thermal restoring
- #100 — Cubed-sphere ocean face-boundary instability

### Open issues — infrastructure
- #148 — Refactor ocean test matrix: use experiment modules, split monolith, adopt xarray/xgcm
- #146 — Generic state checkpoint/restart system for all model components
- #137 — MPAS/Voronoi unstructured grid 9× slower than latlon (partially addressed)
- #112 — Vector Laplacian Python for-loop hurts JIT performance

### Recently closed
- #145 — MPAS vertical tracer advection (resolved: full flux-form transport in step())
- #140 — Baroclinic gyre has no vertical transport (resolved: full 3D velocity for tracer mass fluxes)
- #138 — Latlon C-grid long-term instability (resolved: flux-form vertical advection + upwind horizontal tracer)
- #135 — MPAS regional regridding visualization artifacts (resolved: mask-aware IDW + native PolyCollection)
- #134 — MPAS land cell temperature masking (resolved)
- #130 — Vertical advection not producing spatial T patterns (resolved)
- #114 — Cross-grid physics consistency (resolved: 6/6 items fixed)
- #111 — C-grid ocean missing step_checked, conservation fixer, freshwater (resolved)
- #108 — Dead/stale code cleanup in ocean dynamics (resolved)
- #105 — Vector Laplacian and barotropic diffusion fixes
- #103 — MPAS Coriolis double-counting
- #102 — Flux-form tracer transport
- #101 — Conservation strategy
