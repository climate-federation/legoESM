# DCMIP Dynamical Core Test Suite — Developer Reference

**Curated by increasing complexity for Atmospherax (JAX)**

Based on: DCMIP-2012 (Ullrich et al., 2012) and DCMIP-2016 (Ullrich et al., 2016)

---

## Source Code Repositories

| Repository | Contents | URL |
|---|---|---|
| **DCMIP2012 Fortran Init** | `dcmip_initial_conditions_test_1_2_3_v5.f90`, `dcmip_initial_conditions_test_4_v3.f90`, `dcmip_initial_conditions_test_5_v1.f90`, `simple_physics_v6.f90` | <https://public.websites.umich.edu/~cjablono/dycore_test_suite.html> |
| **DCMIP2016 GitHub** | All 2016 init routines + Kessler microphysics + simple physics + Terminator chemistry + plotting scripts | <https://github.com/ClimateGlobalChange/DCMIP2016> |
| **DCMIP2016 Zenodo (v1.0)** | Archived release with DOI | <https://doi.org/10.5281/zenodo.1298671> |
| **DCMIP2012 Test Case PDF** | Full mathematical specification (v1.7) | <https://public.websites.umich.edu/~cjablono/DCMIP-2012_TestCaseDocument_v1.7.pdf> |
| **DCMIP2016 Test Case PDF** | Full mathematical specification | `DCMIP2016-TestCaseDocument_v1.pdf` in the GitHub repo |
| **Google Drive folder** | Collected idealized test cases for GCMs | <https://drive.google.com/drive/folders/1bqN07JJ10DbgCoU8O1uiULdRP_2Sxenh> |

### Key Fortran Source Files (DCMIP2012)

These are **standalone** routines — each takes a single grid point `(lon, lat, p, z)` and returns all initial fields:

```
dcmip_initial_conditions_test_1_2_3_v5.f90   → Tests 1-1, 1-2, 1-3, 2-0, 2-1, 2-2, 3-1
dcmip_initial_conditions_test_4_v3.f90        → Tests 4-1-x, 4-2, 4-3
dcmip_initial_conditions_test_5_v1.f90        → Tests 5-1, 5-2
simple_physics_v6.f90                         → Simple physics package (BL, sfc fluxes, condensation)
```

### Key Fortran Source Files (DCMIP2016)

Located in `DCMIP2016/interface/` on GitHub:

```
dcmip2016_baroclinic_wave.f90                 → D16-1 (moist baroclinic wave)
dcmip2016_tropical_cyclone.f90                → D16-2 (tropical cyclone)
dcmip2016_supercell.f90                       → D16-3 (splitting supercell)
dcmip2016_kessler.f90                         → Kessler warm-rain microphysics
dcmip2016_simple_physics.f90                  → Updated simple physics
dcmip2016_terminator.f90                      → Terminator "toy chemistry" tracer
```

---

## Level 1 — Pure Advection (Prescribed Winds)

> **What's tested:** Tracer transport only. Wind fields are analytically prescribed (not computed). No momentum equations are integrated.

### Test 1-1: 3D Deformational Flow

| | |
|---|---|
| **Fortran init** | `dcmip_initial_conditions_test_1_2_3_v5.f90` → `subroutine test1_advection_deformation(lon, lat, p, z, zcoords, u, v, w, t, phis, ps, rho, q, q1, q2, q3, q4)` |
| **Test case doc** | DCMIP-2012 §1.1 |
| **Duration** | 12 days (flow reverses at day 6) |
| **Resolution** | 1° × 1°, 60 vertical levels |
| **Vertical extent** | 0–12 km |
| **Output fields** | `q1, q2, q3, q4` (passive tracers), `u, v, w` |
| **Validation** | L2/Linf norms at t=12 days against initial condition (exact reversal) |

**Init signature (Fortran → JAX translation target):**

```fortran
subroutine test1_advection_deformation(  &
    lon,     &  ! longitude (radians)
    lat,     &  ! latitude (radians)
    p,       &  ! pressure (Pa)        — input if zcoords=0
    z,       &  ! height (m)           — input if zcoords=1
    zcoords, &  ! 0=pressure coords, 1=height coords
    u, v, w, &  ! wind components (m/s) — OUTPUT
    t,       &  ! temperature (K)       — OUTPUT
    phis,    &  ! surface geopotential   — OUTPUT
    ps,      &  ! surface pressure (Pa)  — OUTPUT
    rho,     &  ! density (kg/m³)        — OUTPUT
    q,       &  ! specific humidity      — OUTPUT (=0 for this test)
    q1, q2, q3, q4  &  ! passive tracers — OUTPUT
)
```

**What to check:**
- Positivity preservation of `q1` (cosine bell)
- Nonlinear tracer correlation between `q1` and `q2`
- Shape preservation after deformation + reversal
- Convergence rates with resolution

---

### Test 1-2: 3D Hadley-like Meridional Circulation

| | |
|---|---|
| **Fortran init** | `dcmip_initial_conditions_test_1_2_3_v5.f90` → `subroutine test1_advection_hadley(lon, lat, p, z, zcoords, u, v, w, t, phis, ps, rho, q, q1)` |
| **Test case doc** | DCMIP-2012 §1.2 |
| **Duration** | 12 days |
| **Resolution** | 1° × 1°, 60 vertical levels |
| **Key parameters** | `w0 = 0.15 m/s`, `z1 = 2000 m`, `z2 = 5000 m` (v5 update) |

**What to check:**
- Vertical–horizontal transport coupling accuracy
- Tracer shape after one complete overturning cycle
- Sensitivity to vertical resolution (try L30, L60, L120)

---

### Test 1-3: Solid-Body Rotation with Orography

| | |
|---|---|
| **Fortran init** | `dcmip_initial_conditions_test_1_2_3_v5.f90` → `subroutine test1_advection_orography(lon, lat, p, z, zcoords, hybrid_eta, hyam, hybm, u, v, w, t, phis, ps, rho, q, q1)` |
| **Test case doc** | DCMIP-2012 §1.3 |
| **Duration** | 12 days (one full rotation) |
| **Orography** | Schär-type mountain: `phis(lon,lat)` defined analytically |

**What to check:**
- Spurious vertical mixing over the mountain
- Tracer integrity in terrain-following coordinates
- **Extra input:** `hybrid_eta, hyam, hybm` needed for pressure-based models with hybrid-σ coordinates

---

## Level 2 — Orographic Flows (Hydrostatic & Non-Hydrostatic)

> **What's tested:** Pressure-gradient accuracy, mountain wave generation/propagation. First actual dynamical integration.

### Test 2-0: Steady-State + Mountain (Pressure Gradient Accuracy)

| | |
|---|---|
| **Fortran init** | `dcmip_initial_conditions_test_1_2_3_v5.f90` → `subroutine test2_steady_state_mountain(lon, lat, p, z, zcoords, hybrid_eta, hyam, hybm, u, v, w, t, phis, ps, rho, q)` |
| **Test case doc** | DCMIP-2012 §2.0 |
| **Duration** | Static (check at t=0, t=1 day, t=10 days) |
| **True solution** | `u = v = w = 0` everywhere (resting atmosphere) |
| **Small planet** | X=1 (Earth-size, hydrostatic) |

**What to check:**
- Any non-zero `u, v, w` is a *direct measure of pressure-gradient error*
- Max |u|, max |w| as a function of time
- Must pass this before attempting tests 2-1, 2-2

---

### Test 2-1: Schär Mountain Waves (Small Planet, No Shear)

| | |
|---|---|
| **Fortran init** | `dcmip_initial_conditions_test_1_2_3_v5.f90` → `subroutine test2_schaer_mountain(lon, lat, p, z, zcoords, hybrid_eta, hyam, hybm, shear, u, v, w, t, phis, ps, rho, q)` with `shear=0` |
| **Test case doc** | DCMIP-2012 §2.1 |
| **Duration** | ~100 time steps (short integration) |
| **Small planet** | **X = 125** → effective Δx ≈ 500 m |
| **Background flow** | Uniform zonal: `u0 = 20 m/s`, no shear |
| **Model top** | ~30 km with absorbing/sponge layer |

**Small-planet adjustments (critical!):**

```python
# JAX pseudo-code for small-planet scaling
X = 125
a = a_ref / X                    # scaled radius
Omega = Omega_ref * X            # scaled rotation (=0 for non-rotating)
dt = dt_ref / X                  # scaled time step
K_diffusion = K_ref / X**(2*k-1) # scaled diffusion coefficient (order k)
```

**What to check:**
- Vertically propagating gravity waves above the mountain
- Comparison with linear analytic solution (Schär et al. 2002)
- Wave reflection at model top (quality of sponge layer)

---

### Test 2-2: Schär Mountain Waves with Wind Shear

| | |
|---|---|
| **Fortran init** | Same as 2-1 but with `shear=1` |
| **Test case doc** | DCMIP-2012 §2.2 |
| **Background flow** | Zonal with linear shear profile |

**What to check:**
- Tilted wave pattern due to shear
- Critical-level absorption behavior

---

## Level 3 — Non-Orographic Gravity Waves (Small Planet)

> **What's tested:** Acoustic-gravity wave propagation without orography or Coriolis. Clean test of the NH wave solver.

### Test 3-1: Gravity Waves on a Non-Rotating Small Planet

| | |
|---|---|
| **Fortran init** | `dcmip_initial_conditions_test_1_2_3_v5.f90` → `subroutine test3_gravity_wave(lon, lat, p, z, zcoords, u, v, w, t, phis, ps, rho, q)` |
| **Test case doc** | DCMIP-2012 §3.1 |
| **Duration** | 3600 s |
| **Small planet** | **X = 125** |
| **Rotation** | Ω = 0 (non-rotating) |
| **Perturbation** | Thermal anomaly centered at equator |

**ERRATUM (v3):** The density must be initialized with the **unperturbed** background temperature, not the perturbed T. See equation (101) in the test case document.

```python
# WRONG: rho = p / (Rd * T_perturbed)
# CORRECT:
rho = p / (Rd * T_background)  # Use unperturbed T for density init
```

**What to check:**
- Gravity wave dispersion relation
- Wave propagation speed along equator
- Amplitude accuracy after one traverse

---

## Level 4 — Baroclinic Instability (Dry and Moist)

> **What's tested:** Full 3D rotating dynamics — Rossby waves, baroclinic energy conversion, frontal structure. The most important benchmark tier for a global dycore.

### Test 4-1-0: Dry Baroclinic Wave (Earth-Size, X=1)

| | |
|---|---|
| **Fortran init** | `dcmip_initial_conditions_test_4_v3.f90` → `subroutine test4_baroclinic_wave(moist, X, lon, lat, p, z, zcoords, u, v, t, thetav, phis, ps, rho, q, q1, q2)` with `moist=0, X=1` |
| **Test case doc** | DCMIP-2012 §4.1; Jablonowski & Williamson (2006) |
| **Duration** | 10–30 days |
| **Resolution** | 1° × 1° (medium), L30 |
| **Perturbation** | Localized zonal wind perturbation triggers instability |
| **Dynamic tracers** | `q1` = potential temperature Θ, `q2` = |EPV| (Ertel's Potential Vorticity) |

**Init signature:**

```fortran
subroutine test4_baroclinic_wave(  &
    moist,   &  ! 0=dry, 1=moist (test 4-2)
    X,       &  ! planet reduction factor (1, 10, 100, 1000)
    lon, lat, p, z, zcoords, &
    u, v,    &  ! horizontal winds (m/s)     — OUTPUT
    t,       &  ! temperature (K)             — OUTPUT
    thetav,  &  ! virtual potential temp (K)  — OUTPUT
    phis,    &  ! surface geopotential         — OUTPUT
    ps,      &  ! surface pressure (Pa)        — OUTPUT
    rho,     &  ! density (kg/m³)              — OUTPUT
    q,       &  ! specific humidity            — OUTPUT (=0 if moist=0)
    q1,      &  ! potential temperature tracer — OUTPUT
    q2       &  ! |EPV| tracer                 — OUTPUT
)
```

**ERRATUM (v3):** Extra `if` construct prevents division by zero in the EPV calculation at the perturbation center, its antipode, and the poles.

**What to check:**
- Surface pressure minimum deepens to ~960–970 hPa by day 9–10
- Cold/warm front structure visible in 850 hPa temperature
- EPV conservation quality
- Day-8 surface pressure pattern should match published results

---

### Test 4-1-1 / 4-1-2 / 4-1-3: Small-Planet Variants (X=10, 100, 1000)

| | |
|---|---|
| **Fortran init** | Same as 4-1-0 with `X=10`, `X=100`, `X=1000` |
| **Effective Δx** | X=10: ~11 km, X=100: ~1.1 km, X=1000: ~110 m |
| **Duration** | Scale by 1/X (X=10 → 1–3 days, X=1000 → ~2500 s) |

**Critical small-planet settings:**

```python
def small_planet_params(X):
    return {
        'a': 6.37122e6 / X,           # scaled radius
        'Omega': 7.292e-5 * X,        # scaled rotation rate
        'dt': dt_base / X,            # scaled time step
        # If using explicit diffusion K * del^(2k):
        'K_2k': K_base / X**(2*k - 1),
        # If using Rayleigh sponge with coefficient r:
        'r_sponge': r_base * X,
    }
```

**What to check:**
- X=10: Hydrostatic and NH results should be nearly identical (transitional regime)
- X=100: NH effects become important; hydrostatic solution diverges
- X=1000: Deeply non-hydrostatic; stress-tests vertical implicit solver and IMEX ARK

---

### Test 4-2: Moist Baroclinic Wave with Large-Scale Condensation

| | |
|---|---|
| **Fortran init** | `dcmip_initial_conditions_test_4_v3.f90` → same subroutine with `moist=1, X=1` |
| **Physics** | Large-scale condensation only (from `simple_physics_v6.f90`, condensation module) |
| **Test case doc** | DCMIP-2012 §4.2 |

**Condensation scheme (pseudo-code for JAX):**

```python
def large_scale_condensation(T, q, p, dt):
    """Large-scale condensation: remove supersaturation, release latent heat."""
    q_sat = 0.622 * e_sat(T) / (p - e_sat(T))  # saturation specific humidity
    if q > q_sat:
        dq = (q - q_sat) / (1 + L_v**2 * q_sat / (cp * Rv * T**2))
        q_new = q - dq
        T_new = T + L_v * dq / cp
        precip_rate = dq * dp / (g * rho_water * dt)
    else:
        q_new, T_new = q, T
        precip_rate = 0.0
    return T_new, q_new, precip_rate
```

**What to check:**
- Precipitation concentrated along warm/cold fronts
- Latent heating accelerates cyclone deepening vs. dry case

---

### Test 4-3: Moist Baroclinic Wave with Simple Physics (Optional)

| | |
|---|---|
| **Fortran init** | Same as 4-2, plus `simple_physics_v6.f90` |
| **Physics** | Large-scale condensation → Surface fluxes → BL mixing (time-split, in this order) |
| **SST** | Latitude-dependent: set `test=1` flag in `simple_physics` |

**Simple physics call order (MUST be time-split in this sequence):**

```python
# 1. Large-scale condensation
T, q, precip = large_scale_condensation(T, q, p, dt_phys)

# 2. Surface fluxes (bulk aerodynamic)
T, q, u, v = surface_fluxes(T, q, u, v, p_surf, T_sst(lat), Cd, dt_phys)

# 3. Boundary-layer mixing (vertical diffusion)
T, q, u, v = boundary_layer_mixing(T, q, u, v, p, dt_phys)
```

---

## Level 5 — Tropical Cyclones (Moist + Physics)

> **What's tested:** Full integration — NH dynamics + moisture transport + surface fluxes + BL physics. A self-intensifying vortex.

### Test 5-1: Idealized Tropical Cyclone with Simple Physics

| | |
|---|---|
| **Fortran init** | `dcmip_initial_conditions_test_5_v1.f90` → `subroutine test5_tropical_cyclone(lon, lat, p, z, zcoords, u, v, t, thetav, phis, ps, rho, q)` |
| **Physics** | `simple_physics_v6.f90` with `test=0` (constant SST) |
| **Test case doc** | DCMIP-2012 §5.1; Reed & Jablonowski (2011, 2012) |
| **Duration** | 10 days |
| **Resolution** | ≥ 0.5° (~50 km), L30 |
| **SST** | Constant 29°C (302.15 K) — set `test=0` in simple_physics |
| **Initial vortex** | Weak balanced tropical depression, ~20 m/s max wind |

**Init signature:**

```fortran
subroutine test5_tropical_cyclone(  &
    lon, lat, p, z, zcoords, &
    u, v,    &  ! wind components (m/s)       — OUTPUT (includes vortex)
    t,       &  ! temperature (K)              — OUTPUT
    thetav,  &  ! virtual potential temp (K)   — OUTPUT
    phis,    &  ! surface geopotential          — OUTPUT
    ps,      &  ! surface pressure (Pa)         — OUTPUT
    rho,     &  ! density (kg/m³)               — OUTPUT
    q        &  ! specific humidity (kg/kg)     — OUTPUT (tropical moisture profile)
)
```

**What to check:**
- Minimum surface pressure (MSP) drops from ~1015 to ~960–980 hPa
- Maximum wind speed (MWS) reaches hurricane strength (33+ m/s)
- Warm-core structure visible in temperature anomaly cross-section
- Wind–pressure relationship follows empirical curves
- Axisymmetric radial profiles of tangential wind

---

### Test 5-2: Tropical Cyclone with Full Physics (Optional)

| | |
|---|---|
| **Fortran init** | Same as 5-1, but use model's own physics package |
| **Test case doc** | DCMIP-2012 §5.2 |

Out of scope for initial Atmospherax development. Future target once ML parameterizations are integrated.

---

## Level 6 — DCMIP-2016 Extensions (Advanced Non-Hydrostatic)

> **What's tested:** Advanced NH dynamics with Kessler microphysics and physics-dynamics coupling at convective scales. All source code in the DCMIP2016 GitHub repo.

### Test D16-1: Moist Baroclinic Wave + Terminator Tracer

| | |
|---|---|
| **Fortran init** | `DCMIP2016/interface/dcmip2016_baroclinic_wave.f90` |
| **Physics** | `dcmip2016_kessler.f90` (Kessler warm-rain) + `dcmip2016_simple_physics.f90` |
| **Chemistry** | `dcmip2016_terminator.f90` (Cl/Cl₂ toy chemistry) |
| **Test case doc** | DCMIP2016 Test Case Document §1 |
| **Duration** | 10 days |
| **Resolution** | ~1° (~110 km), L30 |

**Init subroutine:**

```fortran
subroutine dcmip2016_baroclinic_wave(  &
    lon, lat, p, z, zcoords, &
    moist,   &  ! 0=dry, 1=moist
    pertt,   &  ! 0=exponential, 1=streamfunction perturbation
    u, v, w, t, thetav, phis, ps, rho, q, q1, q2  &
)
```

**Terminator tracer init (Cl, Cl₂):**

```fortran
subroutine dcmip2016_terminator_initial_condition(lon, lat, cl, cl2)
    ! Returns initial Cl and Cl₂ mixing ratios
    ! Cl + Cl₂ = const (conservation constraint: cly = cl + 2*cl2)
```

**Kessler microphysics interface:**

```fortran
subroutine dcmip2016_kessler(  &
    t, qv, qc, qr,   &  ! temperature, vapor, cloud, rain — INOUT
    rho, pk, dt, z, nz &  ! density, Exner pressure, timestep, heights, levels
)
```

**What to check:**
- Terminator tracer total `Cly = Cl + 2*Cl₂` should be conserved
- Precipitation patterns along fronts with Kessler microphysics
- Compare dry vs. moist baroclinic wave development

---

### Test D16-2: Idealized Tropical Cyclone (DCMIP-2016 version)

| | |
|---|---|
| **Fortran init** | `DCMIP2016/interface/dcmip2016_tropical_cyclone.f90` |
| **Physics** | `dcmip2016_kessler.f90` + `dcmip2016_simple_physics.f90` |
| **Test case doc** | DCMIP2016 Test Case Document §2; Willson et al. (2024), GMD 17:2493–2507 |
| **Duration** | 10 days |
| **Resolution** | ~50 km default; also run at ~25 km |

**Init subroutine:**

```fortran
subroutine dcmip2016_tropical_cyclone(  &
    lon, lat, p, z, zcoords, &
    u, v, w, t, thetav, phis, ps, rho, q  &
)
```

**What to check:**
- Compare MWS/MSP evolution against 12-model DCMIP2016 ensemble (Willson et al. 2024)
- Variable-resolution capability: refine around vortex center

---

### Test D16-3: Splitting Supercell (Small Planet)

| | |
|---|---|
| **Fortran init** | `DCMIP2016/interface/dcmip2016_supercell.f90` |
| **Physics** | `dcmip2016_kessler.f90` (no surface fluxes, no BL mixing) |
| **Test case doc** | DCMIP2016 Test Case Document §3; Zarzycki et al. (2019), GMD 12:879–892 |
| **Duration** | 7200 s (120 min) |
| **Small planet** | **X = 120** → effective circumference ~335 km |
| **Resolutions** | Δx = 4, 2, 1, 0.5 km (run all for convergence study) |
| **Model top** | 20 km, Δz = 500 m (40 levels) |
| **Boundary** | Free-slip lower boundary (no surface drag) |

**Init subroutine:**

```fortran
subroutine dcmip2016_supercell(  &
    lon, lat, p, z, zcoords, &
    u, v, w, t, thetav, phis, ps, rho, q  &
)
! Includes:
!   - Horizontally uniform environment with convective instability
!   - Small thermal perturbation centered at equator to trigger updraft
!   - Initial qv from moist sounding; qc = qr = 0
```

**Critical implementation notes:**
- **Thermal perturbation must be applied AFTER initial hydrostatic balance iteration.** Without this, the model generates spurious large vertical velocities as it adjusts.
- **Diffusion is critical.** DCMIP2016 models used varying diffusion strategies. Results are very sensitive to explicit/implicit numerical diffusion at these scales.
- Output every 15 min minimum: full 3D prognostic fields + qv, qc, qr

**What to check:**
- Single updraft → storm split into two counter-rotating cells
- Cells propagate poleward (away from equator) symmetrically
- Convergence study: solutions should converge as Δx → 0.5 km
- Some models required Δx = 0.25 km for convergence

---

## JAX Translation Notes

### Common Patterns for Porting Fortran Init → JAX

All DCMIP init routines follow the same pattern: **pointwise functions** that map `(lon, lat, z_or_p)` → field values. This maps naturally to JAX's `vmap`:

```python
import jax
import jax.numpy as jnp

def dcmip_test_1_1_init(lon, lat, z):
    """Pointwise init for test 1-1. Port from Fortran."""
    # ... analytic expressions from the test case document ...
    return u, v, w, T, rho, q1, q2, q3, q4

# Vectorize over all grid points
init_vectorized = jax.vmap(dcmip_test_1_1_init)

# Apply to grid arrays
lon_grid, lat_grid, z_grid = make_grid(...)  # shape (N,)
u, v, w, T, rho, q1, q2, q3, q4 = init_vectorized(lon_grid, lat_grid, z_grid)
```

### Physical Constants (use these exact values for reproducibility)

```python
# DCMIP physical constants — DO NOT CHANGE for intercomparison
a_ref   = 6.37122e6    # Earth radius (m)
Omega_ref = 7.292e-5   # Earth rotation rate (s⁻¹)
g       = 9.80616      # gravity (m/s²)
p0      = 1e5          # reference pressure (Pa)
cp      = 1004.5       # specific heat dry air at const pressure (J/kg/K)
cv      = 717.5        # specific heat dry air at const volume (J/kg/K)
Rd      = 287.0        # gas constant dry air (J/kg/K)
Rv      = 461.5        # gas constant water vapor (J/kg/K)
kappa   = Rd / cp      # ≈ 2/7
epsilon = Rd / Rv      # ≈ 0.622
rho_w   = 1000.0       # density of water (kg/m³)
Lv      = 2.5e6        # latent heat of vaporization (J/kg)
```

### Coordinate Systems

The init routines support both height-based and pressure-based coordinates via the `zcoords` flag:

```python
# zcoords = 1: height-based (input z, compute p internally)
# zcoords = 0: pressure-based (input p, compute z internally)
# For Atmospherax with height-based vertical: always use zcoords=1
```

---

## Output Conventions

### NetCDF-CF Naming Convention

```
{model}.{test_id}.{resolution}.L{levels}.{grid}.{equation}.{description}.nc
```

Example: `atmospherax.41-0.medium.L30.icosahedral.nonhydro.nc`

### Required Output Variables per Test

| Test | Required Fields |
|---|---|
| 1-1, 1-2, 1-3 | `q1, q2, q3, q4` at t=0, 6, 12 days |
| 2-0 | `u, v, w, ps` at t=0, 1, 10 days |
| 2-1, 2-2 | `u, v, w, theta` at steady state |
| 3-1 | `u, v, w, theta, rho` at t=0, 900, 1800, 3600 s |
| 4-1-x | `ps, T, u, v, q1(Θ), q2(EPV)` every 12–24 h for 30 days |
| 4-2, 4-3 | Same as 4-1 + `q, precip` |
| 5-1, 5-2 | `ps, T, u, v, w, q, precip` every 6 h for 10 days |
| D16-1 | Same as 4-2 + `Cl, Cl2` |
| D16-2 | `ps, T, u, v, w, q, qc, qr, precip` every 6 h |
| D16-3 | Full 3D fields + `qv, qc, qr` every 15 min for 2 h |

---

## Quick Reference: Implementation Order

```
Phase 1 (Transport):     1-1 → 1-2 → 1-3
Phase 2 (NH Solver):     2-0 → 2-1 → 2-2 → 3-1
Phase 3 (Rotating):      4-1-0 → 4-1-1 → 4-1-2 → 4-1-3
Phase 4 (Moist):         4-2 → 4-3 → 5-1
Phase 5 (Advanced):      D16-1 → D16-2 → D16-3
```
