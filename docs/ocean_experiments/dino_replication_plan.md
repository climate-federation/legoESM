# Plan: Replicate Kamm et al. (2025) DINO Experiment in legoESM

## Reference Paper

**Kamm, D., Deshayes, J., & Madec, G. (2025).** "DINO: a diabatic model of pole-to-pole ocean dynamics to assess subgrid parameterizations across horizontal scales." *Geosci. Model Dev.*, 18, 8091–8107. https://doi.org/10.5194/gmd-18-8091-2025

Local PDF: `docs/references/Kamm_etal2025.pdf` — read with PyMuPDF:
```python
import fitz
doc = fitz.open("docs/references/Kamm_etal2025.pdf")
for page in doc: print(page.get_text())
```

Source code and input files for the original NEMO implementation: https://doi.org/10.5281/zenodo.15016824

## Goal

Replicate the DINO 1° (R1) configuration from Kamm et al. (2025) in legoESM on both lat-lon (Mercator) and MPAS grids. All domain geometry, forcing profiles, initial conditions, and numerical parameters are taken directly from the paper (equations, tables, and appendices). Where our model infrastructure differs from NEMO (EOS, vertical mixing closure), we document the approximation and its expected impact.

This provides:
- A pole-to-pole overturning circulation benchmark with diabatic processes
- A platform for testing/training ML eddy parameterizations in legoESM
- Cross-grid comparison (lat-lon vs MPAS) on a scientifically meaningful configuration

Key design decisions (from user interview):
- Wright EOS (not simplified Roquet) — more accurate, already implemented. Acknowledged: density field will differ quantitatively from DINO; ACC/MOC metrics won't match paper exactly.
- KPP + enhanced diffusion convection (not TKE closure). Use DINO's background values: A_v=1.2e-4 m²/s, K_v=1.2e-5 m²/s.
- Grid-dependent lateral mixing coefficients
- GM/Redi for isopycnal tracer diffusion + geopotential Laplacian for momentum. NOTE: paper uses isopycnal viscosity for momentum too — documented as known approximation (see expert review issue #7).
- Mercator grid for lat-lon (new)
- Exact DINO 36-level vertical grid
- Annual-mean forcing first (seasonal cycle deferred to Phase 2)
- TVD for MPAS tracers (FCT deferred)
- MPAS: global mesh + land mask (re-entrant channel handled naturally on global mesh — see Phase 3 notes)
- 1° target resolution
- Add to test matrix (short run) + standalone script (long runs)

### Expert Review Corrections (post-plan audit)

Nine critical issues were identified by the ocean expert and are incorporated below:

1. **Non-solar/solar heat flux split**: Q_sr must be subtracted from restoring in surface layer (eq. 8), then distributed via Jerlov penetration (eq. 10). Without this, solar heating is double-counted.
2. **Restoring uses heat-flux coefficients, not timescales**: A_Θ=40 W/m²/K → tau_T = ρ₀·cp·Δz₀/A_Θ ≈ 11.85 days for 10m layer. A_S=3.858e-3 kg/m²/s → tau_S = ρ₀·Δz₀/A_S ≈ 30.8 days.
3. **Salinity restoring Gaussian equatorial dip**: eq. B2 has −1.25·exp(−φ²/7.5²) reducing S at equator.
4. **IC meridional gradient goes to surface values at poles** (isothermal columns), NOT bottom values.
5. **Wind must be piecewise cubic** (PCHIP), not linear interpolation.
6. **Tréguier(1997) ≈ Visbeck(1997)** for practical purposes at 1° — use Visbeck as approximation.
7. **Isopycnal momentum viscosity**: paper uses isoneutral viscosity; we use geopotential. Known approximation — affects ACC where slopes are steep.
8. **MPAS re-entrant channel**: on global mesh, Drake Passage is naturally open. Domain carved by land mask.
9. **Missing parameters**: dt=45min, bottom drag C_d~1e-3, background A_v/K_v values.

---

## Pre-Implementation Audit Findings (2026-05-14)

After reading the full Kamm et al. (2025) paper (incl. Appendices A–E) and auditing the lat-lon C-grid ocean codebase, here is the concrete state of play.

### Already in repo (no work needed beyond configuration)

| Capability | Module | Notes |
|---|---|---|
| Wright EOS, differentiable | `ocean/eos.py:72–129` (`make_eos_fn("wright")`) | Valid T∈[-2,40]°C, S∈[0,42] PSU; ρ₀=1026 kg/m³ |
| KPP with configurable backgrounds | `ocean/physics/vertical_mixing/kpp.py`; `config.py:43–44` | Defaults A_bg=1e-4, K_bg=1e-5 → set 1.2e-4, 1.2e-5 |
| Enhanced-diffusion convection | `ocean/physics/convection/enhanced_diffusion.py`; `config.py:10` | K_conv default 1.0 m²/s → set 100 |
| GM/Redi (lat-lon C-grid) + Visbeck adaptive κ_GM | `ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py`; `_gm_redi_common.py` | α=0.015, κ_min/κ_max configurable via `GMRediConfig`; tracer diffusion = Griffies-1998 triads |
| Jerlov type I shortwave penetration | `ocean/physics/shortwave_penetration.py:37–80` | `water_type="I"` → R=0.58, ζ₀=0.35m, ζ₁=23m (exact DINO) |
| Quadratic bottom drag | `ocean/physics/bottom_drag/quadratic.py`; `config.py:19–21` | Default C_d=2.5e-3 → set 1e-3 |
| Free-slip lateral BC | `ocean/dynamics/ocean_pe_latlon_cgrid.py:14–16` | Implicit via v=0 at poles; matches NEMO |
| Vector-invariant momentum + EEN-like Coriolis | `ocean/dynamics/ocean_pe_latlon_cgrid.py:1139–1167` | Arakawa-Lamb 1981 12-point PV-flux (Stewart-Dellar partial-cell); energy + enstrophy conserving |
| Tracer FCT (NEMO-equivalent) | `ocean/state.py:569`; `ocean/advection.py` | `tracer_advection="ppm_fct"` (TVD is current default; FCT is the NEMO match) |
| SMC03 density-Jacobian PGF (PR #250) | `ocean/dynamics/pgf_smc03.py`; `state.py:612–621` | `pgf_scheme="smc03"` — 100–150× more accurate than `"adcroft"` for stratified flow on bathymetry |
| Nonlinear z* free surface | `ocean/vertical.py:25–100` | Dynamic Jacobian J=(η+H)/H rescales layer thicknesses each step |
| Implicit-CN barotropic solver | `ocean/state.py:596–611` | `barotropic_solver="implicit_cn"` — eliminates checkerboard noise (recommended for DINO; see issue `docs/issues/barotropic_mode_noise.md`) |
| MPAS physics wiring | `ocean/init_mpas.py`, `gm_redi_mpas.py`, etc. | KPP / GM/Redi / restoring / SW / drag all present or grid-agnostic |
| Experiment dispatch | `ocean/experiments/__init__.py:64–100` | Pattern: `create_initial_conditions(grid_type, grid, z_coord, config)` + `create_forcings(...)` |

### Real implementation gaps (must build)

1. **Mercator grid generator** — `src/legoesm/grids/latlon.py` only has `create_latlon_grid()` (equirectangular). `LatLonGrid` NamedTuple already carries per-cell `dx(n_lat, n_lon)` so grid-dependent mixing will work once Mercator generator lands.
2. **Grid-dependent mixing coefficient infrastructure** — `HarmonicConfig` and `BiharmonicConfig` (`ocean/physics/lateral_mixing/config.py:8–17`) only support constant A_h/K_h/B_h. Need to add `U_viscosity`, `U_diffusivity`, `scaling: "constant"|"linear"|"cubic"` fields and thread Δx through `harmonic.py` / `biharmonic.py`.
3. **PCHIP wind interpolation** — `PrescribedForcingConfig.wind_profile` (`ocean/physics/surface_forcing/config.py:10–29`) supports `"constant"`, `"cosine_latitude"`, `"single_gyre"`, `"double_gyre"`, ERA5 — but **not piecewise-cubic Hermite interpolation of arbitrary tau-vs-lat knots**. Cleanest fix: implement PCHIP entirely inside the DINO module (precompute coefficients with NumPy at config time, evaluate analytically on the grid), do not modify the general infrastructure.
4. **Q_sr / non-solar split for restoring** — existing `restoring.py` uses **timescale (seconds)** not flux coefficient (W/m²/K) and does NOT subtract Q_sr from the restoring tendency before applying. DINO requires both. Cleanest fix: implement a custom restoring-with-Q_sr-split tendency entirely inside the DINO module rather than refactoring the general API. The conversion at construction time is `tau_T = ρ₀·c_p·Δz₀ / A_Θ ≈ 11.85 days` and `tau_S = ρ₀·Δz₀ / A_S ≈ 30.8 days` for a 10 m surface layer.

### Known limitations to call out (cannot fix in Phase 1–4)

- **No isoneutral momentum viscosity in lat-lon C-grid** — `gm_redi_latlon_cgrid.py` only diffuses tracers along isopycnals; momentum uses geopotential Laplacian via `harmonic.py`. Paper uses isoneutral for both. Affects ACC magnitude where slopes are steep. Documented as approximation; fix is Phase 5.
- **No sea-ice component in DINO (paper limitation)** — Paper Sect 2.3 explicitly notes this limits AABW formation realism. Worth documenting prominently because it bounds what the validation can plausibly demand.
- **Tracer scheme is FCT-via-PPM, not the exact NEMO FCT** — `ppm_fct` is the closest available; bit-exact reproduction of paper figures should not be expected.
- **GM coefficient is Visbeck (1997) approximation of Tréguier (1997)** — both growth-rate-dependent; paper notes ACC transport is sensitive to GM tuning.

### Recommended scheme selections for DINO

```python
# To pass to ExperimentConfig / state construction
pgf_scheme = "smc03"                 # density-Jacobian PGF (PR #250)
barotropic_solver = "implicit_cn"     # avoid cosine-filter checkerboard noise
barotropic_implicit_theta_eta = 0.55  # CN default
tracer_advection = "ppm_fct"          # matches NEMO FCT
free_surface = "z_star_nonlinear"     # OceanZStarCoordinate
hi_precision_pressure = True          # avoid float32 PGF error (issue #2 in pgf_test_plan.md)
```

### Open issues from cross-checking the paper

- **Sill Gaussian width `s` in eq A5 is not stated in the paper text.** Need to extract from Zenodo source code (https://doi.org/10.5281/zenodo.15016824) — bathymetry generator should be a near-direct port to lock down all sill parameters.
- **Reference density profile ρ_ref(z) for σ_2 diagnostics** — paper uses `ρ_ref(0)=1026`, `ρ_ref(2000)=1035` to compute σ_2. Need to compute potential density referenced to 2000 m for MOC-in-density-space (Fig 5) and stratification (Fig 6).
- **Annual-mean Q_sr is NOT `230·cos(φ)`** — the time average of `max(230·cos(π/180·[φ - 23.5·cos(...)]), 0)` over 360 days is what should be used. Compute by NumPy quadrature once at config time.
- **Bathymetry H_max / H_min naming convention** — paper appendix uses `H_max=2000m` to mean *shallowest seafloor* (largest z) and `H_min=4000m` to mean *deepest seafloor* (smallest z). In our `DINOConfig` we use `H_shallow=2000m`, `H_deep=4000m` for clarity. Worked example for eq A3 with these conventions:
  - In interior (`g_φ=g_λ=1`): `b = 1·(2000 - 4000) + 4000 = 2000` ❌ wrong — gives 2000m at deep interior
  - Therefore implement as `b = g_φ·g_λ·(H_deep - H_shallow) + H_shallow` so that `g=1` (interior) → 4000m, `g=0` (boundary) → 2000m, matching Fig 1.
- **Sill is anchored at the western wall and extends EAST into the channel** (the smooth-step `S(λ, λ_m, λ_m+s)` with `λ_m=-50°E` is 0 west of `-50°E` and 1 east of `-50°E+s`). Earlier wording "restricted to western side" was misleading.

---

## Phase 1: Infrastructure (Mercator grid + grid-dependent coefficients)

### 1A. Mercator Grid Generator

**File**: `src/legoesm/grids/latlon.py`

Add `create_mercator_grid(n_lon, lat_max_deg, radius, omega)`:
- Latitude placement per DINO eq. C2: `φ(j) = (180/π) · arcsin(tanh(Δλ·π/180 · j))`
- Δλ = 360/n_lon (zonal spacing in degrees, e.g., 1° for n_lon=360)
- Automatically compute J (number of lat points per hemisphere) from lat_max_deg:
  - `j_max = (180/π) · arctanh(sin(lat_max · π/180)) / (Δλ · π/180)`
  - For 1° and ±70°: j_max ≈ 99, so ~198 total latitude points
- Grid metrics: `dx(i,j) = R·cos(φ)·Δλ·π/180`, `dy(i,j) = R·cos(φ)·Δλ·π/180` (isotropic by construction)
- `area(i,j) = dx·dy` (but note: dx/dy in LatLonGrid are 2-cell-spanning distances; single-cell dx₁ = dx/2)
- Coriolis: `f(j) = 2Ω·sin(φ)`
- Return a `LatLonGrid` NamedTuple (same structure as existing grids)
- Subset to DINO domain: 50° longitude span, ±70° latitude

**Reuse**: `create_regional_latlon_grid()` pattern for domain subsetting.

**Verification**: At equator dx≈111 km, at 70° dx≈38 km (matching Table 2).

### 1B. Grid-Dependent Lateral Mixing Coefficients

**Files to modify**:
- `src/legoesm/ocean/physics/lateral_mixing/config.py` — add scaling parameters
- `src/legoesm/ocean/physics/lateral_mixing/harmonic.py` — compute Δx-scaled A_h, K_h
- `src/legoesm/ocean/physics/lateral_mixing/biharmonic.py` — compute Δx³-scaled B_h

**Approach**:
- Add fields to `HarmonicConfig`: `U_viscosity: float = 0.0`, `U_diffusivity: float = 0.0`, `scaling: str = "constant"` (or `"linear"`)
- When `scaling == "linear"`: `A_h(i,j) = 0.5 * U_viscosity * Δx(i,j)`, `K_h(i,j) = 0.5 * U_diffusivity * Δx(i,j)`
- Add fields to `BiharmonicConfig`: `U_viscosity: float = 0.0`, `U_diffusivity: float = 0.0`, `scaling: str = "constant"` (or `"cubic"`)
- When `scaling == "cubic"`: `B_h(i,j) = (1/12) * U_viscosity * Δx(i,j)³`
- Pass grid metrics (Δx array) through physics pipeline
- For MPAS: `Δx = sqrt(areaCell)`; for lat-lon: use single-cell dx (= grid.dx / 2)

---

## Phase 2: DINO Experiment Module

**File**: `src/legoesm/ocean/experiments/dino.py` (~500-600 LOC)

### 2A. DINOConfig dataclass

Fields from Table 1 + Table 2 of the paper:
```python
# Domain
H_deep = 4000.0            # m (deep basin interior)
H_shallow = 2000.0         # m (minimum depth at boundaries/walls)
H_sill = 2500.0            # m (Scotia Ridge sill depth)
lon_west = -50.0            # deg E
lon_east = 0.0              # deg E
lat_south = -70.0           # deg N (approx, set by Mercator grid)
lat_north = 70.0            # deg N (approx)
channel_lat_south = -65.0   # deg N
channel_lat_north = -45.0   # deg N

# Physical constants (from Table 1)
rho_0 = 1026.0             # kg/m³
cp = 3991.86               # J/kg/K

# Surface forcing (eqs. 7-10)
A_theta = 40.0             # W/m²/K (temperature restoring coefficient)
A_S = 3.858e-3             # kg/m²/s (salinity restoring coefficient)
tau_values = [0, 0.2, -0.1, -0.02, -0.1, 0.1, 0]  # N/m²
tau_latitudes = [-70, -45, -15, 0, 15, 45, 70]      # degrees

# Restoring profiles (Appendix B, annual mean)
T_star_eq = 27.0           # °C
T_star_n_mean = 5.0        # °C (from eq. B3: 5 + 3*cos(...), mean=5)
T_star_s_mean = -0.5       # °C (from eq. B4: -0.5 - 0.5*cos(...), mean=-0.5)
S_star_n = 35.0            # g/kg
S_star_s = 35.1            # g/kg
S_star_eq = 37.25          # g/kg
S_star_eq_gaussian_amp = 1.25   # g/kg (Gaussian dip at equator, eq. B2)
S_star_eq_gaussian_sigma = 7.5  # degrees (Gaussian width, eq. B2)
L_phi = 140.0              # degrees (meridional domain extent for cosine profiles)

# Solar radiation
Q_sr_max = 230.0           # W/m² (peak annual-mean insolation at equator)
# Shortwave penetration: Jerlov type I (zeta_0=0.35m, zeta_1=23m)

# Vertical grid (Appendix C, eq. C3)
n_levels = 36
dz_min = 10.0              # m
k_th = 35                  # inflection level index
a_cr = 10.5                # stretching parameter

# Lateral mixing (R1 values, Table 2)
U_T = 0.027                # m/s (tracer diffusivity velocity scale)
U_M = 0.27                 # m/s (momentum viscosity velocity scale)

# Vertical mixing (KPP + convection)
A_v_bg = 1.2e-4            # m²/s (background vertical viscosity)
K_v_bg = 1.2e-5            # m²/s (background vertical diffusivity)
K_conv = 100.0             # m²/s (convective adjustment diffusivity)

# GM/Redi
use_gm = True
visbeck_alpha = 0.015      # Visbeck dimensionless coefficient (≈ Tréguier 1997)
kappa_gm_min = 200.0       # m²/s
kappa_gm_max = 2000.0      # m²/s

# Bottom drag
C_d_bottom = 1.0e-3        # quadratic drag coefficient

# Time stepping
dt = 2700.0                # s (45 minutes, from Table 2)
n_barotropic_substeps = 30 # default

# Bathymetry slope parameters (Appendix A)
s_lambda = 1.0/3.0         # 1/degrees (zonal slope)
# s_phi = cos(pi*phi_max/180) * s_lambda  (Mercator-corrected meridional slope)
channel_width_deg = 20.0   # Δφ_c in eq A4 (channel from -65 to -45°N)

# Sill (eq. A5) — semicircular ridge anchored at western wall
sill_lon_m = -50.0         # λ_m (Drake-passage anchor longitude)
sill_lat_m = -55.0         # φ_m (Drake-passage anchor latitude)
sill_gaussian_width_s = None  # TBD — extract from Zenodo source (https://doi.org/10.5281/zenodo.15016824); paper text omits this value

# Reference density profile (for σ_2 diagnostics, MOC-in-density plots, paper Figs 5-6)
rho_ref_z0 = 1026.0        # kg/m³ at surface
rho_ref_z2000 = 1035.0     # kg/m³ at 2000 m depth
sigma_2_ref_depth = 2000.0 # m (depth used for potential-density anomaly)
```

### 2B. Analytical Bathymetry (Appendix A, eqs. A1-A5)

**Strongly recommended**: port from the Zenodo reference implementation (https://doi.org/10.5281/zenodo.15016824) rather than translate equations by eye, especially for the sill (eq A5) where one parameter (`s`, Gaussian width) is not stated in the paper text.

Implement carefully:
- `_smooth_step(x, a, b)` — eq. A2 (6th-degree polynomial: `6t⁵ - 15t⁴ + 10t³`)
- `_tapered_exponential(x, x1, x2, s, d, delta_lambda)` — eq. A1 (3-branch piecewise with tapering)
- `_dino_bathymetry(lon_deg, lat_deg, config)`:
  - Compute `g_phi` (meridional shape) and `g_lambda` (zonal shape) per eq. A3
  - **Sign convention check** (paper Appendix A's `H_max=2000m`/`H_min=4000m` is opposite of intuitive labeling — see Pre-Implementation Audit, "Open issues from cross-checking the paper"). Implement as `b = g_φ·g_λ·(H_deep - H_shallow) + H_shallow` so that interior (g=1) → 4000m and boundary (g=0) → 2000m, matching Fig 1.
  - **Slope parameter correction**: `s_phi = cos(π·φ_max/180) · s_lambda` for Mercator grid
  - Apply channel modification (eq. A4): remove zonal walls within channel latitudes (Δφ_c=20°)
  - Add Scotia Ridge sill (eq. A5): Gaussian ring centered at (λ_m, φ_m)=(-50°E, -55°N), applied only where current depth ≤ H_sill, **anchored at the western wall and extending east** into the channel via the smooth step `S(λ, λ_m, λ_m+s)`. The Gaussian width `s` is **not in the paper text — extract from Zenodo source**.
- Return (H_bathy, land_mask) — H_bathy in meters (positive down), land_mask (1=ocean, 0=land)

### 2C. Surface Forcing Profiles (Annual Mean)

**Wind** (eq. 7):
- Piecewise cubic (PCHIP) interpolation of `tau_values` at `tau_latitudes`
- Pre-compute spline coefficients using NumPy at init time, evaluate on grid
- Applied as: `F_u = tau_u / (rho_0 * dz_0)` (source term to topmost layer)
- Purely zonal (tau_v = 0), zonally uniform

**Temperature restoring** (eq. 8, annual mean):
- `T_star(φ) = T_star_ns + (T_star_eq - T_star_ns) · cos(π·φ/L_φ)` where T_star_ns is T_star_n or T_star_s depending on hemisphere
- Convert A_Theta to timescale: `tau_T = rho_0 * cp * dz_0 / A_theta` (layer-thickness dependent)
- **CRITICAL**: Subtract Q_sr from non-solar component: `F_Theta_ns = [A_Theta·(T*-T) - Q_sr] / (cp·rho_0·dz_0)`
- Then add penetrating shortwave separately through the column (eq. 10)

**Salinity restoring** (eq. 9, B2):
- `S_star(φ) = S_ns + (S_eq - S_ns)·(1+cos(2π·φ/L_φ))/2 - 1.25·exp(-φ²/7.5²)`
- **Include the Gaussian equatorial dip** (−1.25 g/kg at equator)
- Convert A_S to timescale: `tau_S = rho_0 * dz_0 / A_S`

**Solar radiation** (eq. B5, annual mean):
- Full seasonal expression (eq B5): `Q_sr(t,φ) = max(230·cos(π/180·[φ - 23.5·cos(π·(d-171)/180)]), 0)` (the 23.5° term is solar declination)
- **For Phase 1 (annual mean)**: numerically integrate eq B5 over 360 days at each grid latitude using NumPy quadrature, once at config time. Do NOT use the lazy `230·cos(π·φ/180)` approximation — the polar-night clipping makes the analytical annual mean fall off faster than `cos(φ)` and the right answer is cheap to compute.
- Shortwave penetration: Jerlov type I (ζ₀=0.35m, ζ₁=23m) — `ShortwavePenetrationConfig.water_type="I"` (already exact match in `shortwave_penetration.py:37–80`)

**Implementation note on non-solar/solar split**:
The DINO experiment module will compute Q_sr(φ) internally and pass it to the forcing. The restoring tendency for T must subtract Q_sr before dividing by dz_0. If the existing restoring module doesn't support this, implement it directly in the DINO experiment's forcing function rather than modifying the general restoring infrastructure.

### 2D. Initial Conditions (Appendix D)

- **T(z)**: Complex multi-branch tanh profile (eq. D2) — two terms weighted by complementary smoothed step functions around z=500m
- **S(z)**: Similar multi-branch tanh profile (eq. D3)
- **Meridional gradient** (eqs. D4-D5): `T̃(φ,z) = [T(z) - T(z=0)] · (φ₁ - |φ|)/φ₁ + T(z=0)`
  - **CORRECTED**: At poles (|φ|=φ₁), T̃ = T(z=0) for ALL z → isothermal columns at the surface temperature value
  - This is NOT "transition to bottom values" — it's transition to surface values (isothermal columns)
- Initialize from rest: u=v=0, η=0

### 2E. DINO Vertical Grid (Appendix C, eq. C3)

```
z(k) = a2 + a1·k + a0·acr·ln(cosh((k - kth)/acr))
```
with:
```
a0 = (dz_min - H/(K-1)) / (tanh((1-kth)/acr) - acr·(ln(cosh((K-kth)/acr)) - ln(cosh((1-kth)/acr)))/(K-1))
a1 = dz_min - a0·tanh((1-kth)/acr)
a2 = -a1 - a0·acr·ln(cosh((1-kth)/acr))
```
where K=36, H=4000m, dz_min=10m, kth=35, acr=10.5.

Create helper `create_dino_z_star(config)` → `OceanZStarCoordinate`:
- Compute z at half-levels (interfaces) using eq. C3 for k ∈ [1, K+1]
- dz_ref = diff(z_half)
- z_full_ref = midpoints

### 2F. create_initial_conditions / create_forcings

Follow pattern of `global_overturning.py`:
- `create_initial_conditions(grid_type, grid, z_coord, config)` — T, S, u, v, η, bathymetry, land mask
- `create_forcings(grid_type, grid, z_coord, config)` — returns physics config with:
  - **Wind**: custom PCHIP profile in DINO module (general infrastructure does not support arbitrary tau-vs-lat knots — see audit gap #3). Pre-compute coefficients with NumPy at config time, evaluate analytically on the grid, apply via prescribed forcing as F_u = τ_u/(ρ₀·dz_0).
  - **Restoring + Q_sr split**: custom restoring tendency in DINO module (not a refactor of `restoring.py` — see audit gap #4). Convert A_Θ=40 W/m²/K → τ_T = ρ₀·c_p·dz_0/A_Θ ≈ 11.85 days; A_S=3.858e-3 → τ_S ≈ 30.8 days; subtract Q_sr from T restoring before applying.
  - **KPP**: `KPPConfig(A_bg=1.2e-4, K_bg=1.2e-5)`
  - **Convection**: `EnhancedDiffusionConfig(K_conv=100.0)`
  - **GM/Redi**: `GMRediConfig(visbeck=VisbeckConfig(enabled=True, alpha=0.015), kappa_min=200, kappa_max=2000, slope_scheme="triads")`
  - **Harmonic viscosity (momentum only, geopotential)**: `HarmonicConfig(U_viscosity=0.27, scaling="linear", U_diffusivity=0.0)` — geopotential is a documented approximation vs paper's isoneutral
  - **Bottom drag**: `QuadraticDragConfig(C_d=1.0e-3)`
  - **Shortwave**: `ShortwavePenetrationConfig(water_type="I")`
  - **Numerical-scheme selections** (locked in based on audit; see "Recommended scheme selections for DINO" above):
    - `pgf_scheme="smc03"` (NOT default `"adcroft"`) — eliminates known PGF accuracy bugs on stratified bathymetry
    - `barotropic_solver="implicit_cn"` (NOT default `"explicit_substep"`) — eliminates checkerboard barotropic noise documented in `docs/issues/barotropic_mode_noise.md`
    - `barotropic_implicit_theta_eta=0.55` (CN default)
    - `tracer_advection="ppm_fct"` (the NEMO-FCT match)
    - `hi_precision_pressure=True` (avoid float32 PGF cumsum error)
    - Free-slip lateral BC: already the default; nothing to set
    - Vector-invariant momentum + AL81 EEN PV-flux: already the default; nothing to set
- `validate_results(final_state, diagnostics, config)` — see Verification section for the numeric targets

Support both `grid_type="latlon"` and `grid_type="mpas"`.

---

## Phase 3: MPAS Mesh for DINO

**Approach**: Use **regional spherical Voronoi mesh with sub-360° periodic-x**, via existing `create_regional_voronoi_mesh()` in `src/legoesm/grids/voronoi.py:1465`.

Call signature:
```python
mesh = create_regional_voronoi_mesh(
    lon_range=(-50.0, 0.0),     # match lat-lon Mercator basin
    lat_range=(-70.0, 70.0),
    resolution_km=110.0,         # ~1° at equator
    periodic_x=True,             # east/west meridians identified through channel
)
```

The `periodic_x=True` path uses the unroll-and-ghost Delaunay scheme so TRiSK operators see correct through-the-seam distances, midpoints, and areas. **This gives a true 50°-wide re-entrant channel matching the lat-lon Mercator domain** — no full-globe channel-width mismatch.

Land mask: apply same DINO bathymetry outside the channel latitudes (-65 to -45°N), so the meridians are walled north and south of the channel and open through it. Inside the channel, the periodic-x identification provides the re-entrant flow.

**Resolution**: 110 km ≈ 1° at equator (matches paper R1). Use 220 km for the test-matrix smoke test.

---

## Phase 4: Test Matrix Integration + Standalone Script

### 4A. Test Matrix Entry

**File**: `scripts/run_ocean_test_matrix.py`

Add DINO as a test case:
- Short run: 60 days
- Grids: latlon (Mercator), mpas (global+mask)
- Validation: conservation check, SSH within [-2, 2] m, SST within [-2, 30] °C, no NaN
- Resolution: use 2° Mercator for test matrix speed (fewer lat points)
- Default dt: 2700s (45 min)

### 4B. Standalone Production Script

**File**: `scripts/run_dino.py`

- Full 1° DINO configuration
- Configurable duration. Default: short (50 yr) for shake-down; `--mode spin-up` for long runs (paper does 3000 yr).
- Diagnostics output: barotropic stream function, MOC in σ_2 space (referenced to 2000 m), meridional heat transport (mean + eddy + GM decomposition), zonal-mean potential density σ_2, KE time series
- Support both lat-lon (Mercator) and MPAS via `--grid {latlon,mpas}` flag
- Snapshot output every N years (configurable)
- dt = 2700 s (45 min) for R1; will need shorter dt at higher resolution (R4: 900 s, R16: 180 s per Table 2)

---

## Phase 5 (Deferred): Enhancements

- **Seasonal cycle forcing**:
  - T_star with 1-month lag (eqs B3-B4): `Θ_n*(d) = 5 + 3·cos(π·(d-201)/180)`, `Θ_s*(d) = -0.5 - 0.5·cos(π·(d-201)/180)`
  - Seasonal Q_sr (eq B5 full form, NOT annual mean): `Q_sr(t,φ) = max(230·cos(π/180·[φ - 23.5·cos(π·(d-171)/180)]), 0)`. The 23.5° solar declination shift is the dominant seasonal driver.
- FCT tracer advection for MPAS (currently TVD)
- Simplified Roquet EOS option (eq. 6 with cabbeling + thermobaric) for closer paper match
- Higher resolution variants (R4=1/4°, R16=1/16°) with biharmonic (Δx³ scaling for tracers) + Smagorinsky (C_smag=3.5) for momentum
- TKE vertical mixing closure (Blanke & Delecluse 1993)
- **Coarse-graining and subgrid flux diagnostics** (Sect 2.4 of paper, eqs 11-14) — required for the "ML eddy parameterization training" use case stated as goal #2 of this plan; currently NOT in scope of Phases 1-4
- Isopycnal (isoneutral) momentum viscosity — currently using geopotential Laplacian. Affects ACC where slopes are steep.
- ~~Tréguier (1997) GM coefficient~~ — **decided: stick with Visbeck (1997)**, will not implement Tréguier

---

## Files to Create

| File | Description | Est. LOC |
|------|-------------|----------|
| `src/legoesm/ocean/experiments/dino.py` | DINO experiment module (config, bathymetry port from Zenodo, PCHIP wind, custom restoring + Q_sr split, ICs) | ~700 |
| `tests/ocean/experiments/test_dino.py` | Unit + smoke tests (rest-state PGF, bathymetry shape, IC values, forcing profiles match paper figures) | ~250 |
| `scripts/run_dino.py` | Standalone production script | ~150 |

## Files to Modify

| File | Change | Est. LOC |
|------|--------|----------|
| `src/legoesm/grids/latlon.py` | Add `create_mercator_grid()` (audit gap #1) | ~80 |
| `src/legoesm/ocean/physics/lateral_mixing/config.py` | Add `U_viscosity`, `U_diffusivity`, `scaling` to Harmonic/BiharmonicConfig (audit gap #2) | ~25 |
| `src/legoesm/ocean/physics/lateral_mixing/harmonic.py` | Grid-dependent coefficient when `scaling="linear"` | ~40 |
| `src/legoesm/ocean/physics/lateral_mixing/biharmonic.py` | Grid-dependent coefficient when `scaling="cubic"` | ~40 |
| `src/legoesm/ocean/experiments/__init__.py` | Register DINO experiment in `AVAILABLE_EXPERIMENTS` | ~5 |
| `scripts/run_ocean_test_matrix.py` | Add 60-day DINO smoke test | ~25 |
| `tests/ocean/unit/test_lateral_mixing_scaling.py` | Direct test for new grid-dependent scaling fields (per CLAUDE.md "every new dispatch branch must have a test") | ~80 |
| `tests/grids/test_mercator.py` | Direct test for `create_mercator_grid` (latitude placement, dx isotropy, area integration) | ~60 |

**Note on what is NOT modified**: `restoring.py` and `prescribed.py` (wind) infrastructure are NOT changed. The Q_sr-split restoring and PCHIP wind are implemented in the DINO experiment module so that the general-purpose modules retain their simpler API. If a future experiment needs the same patterns, factor them out then.

## Existing Code to Reuse

| Module | What to reuse |
|--------|---------------|
| `ocean/eos.py` → `make_eos_fn("wright")` | Wright EOS — no new EOS needed |
| `ocean/physics/vertical_mixing/kpp.py` | KPP with background A_v=1.2e-4, K_v=1.2e-5 |
| `ocean/physics/convection/enhanced_diffusion.py` | K_conv=100 m²/s |
| `ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py` | Redi isopycnal diffusion + GM bolus |
| `ocean/physics/lateral_mixing/_gm_redi_common.py` | Visbeck adaptive GM coefficient |
| `ocean/physics/surface_forcing/restoring.py` | Haney SST/SSS restoring (with unit conversion) |
| `ocean/physics/surface_forcing/prescribed.py` | Wind stress application |
| `ocean/physics/shortwave_penetration.py` | Jerlov type I penetrating SW |
| `ocean/physics/bottom_drag/quadratic.py` | Quadratic bottom friction C_d=1e-3 |
| `ocean/vertical.py` → `OceanZStarCoordinate` | z-star coordinate infrastructure |
| `ocean/init_latlon_cgrid.py` | Lat-lon C-grid state initialization |
| `ocean/init_mpas.py` | MPAS state initialization |
| `ocean/experiments/global_overturning.py` | Pattern for experiment structure |
| `ocean/experiments/acc_channel.py` | Pattern for analytical bathymetry + channel |
| `legoesm.constants` | All physical constants |

## Known Approximations vs Paper

| Aspect | DINO (paper) | legoESM (this implementation) | Impact |
|--------|-------------|-------------------------------|--------|
| EOS | Simplified Roquet (linear + cabbeling + thermobaric) | Wright (full nonlinear) | Quantitative density differences; ACC and MOC magnitudes won't match paper exactly |
| Vertical mixing | TKE (Blanke & Delecluse 1993) | KPP (LMD94) + enhanced diffusion | Different BL depth diagnosis; bulk behavior similar |
| Momentum viscosity | Isoneutral Laplacian | Geopotential Laplacian | Affects ACC where slopes are steep; documented audit gap |
| GM coefficient | Tréguier (1997) | Visbeck (1997) — **deliberate choice, not to be revisited** | Both growth-rate-dependent; Visbeck uses depth-averaged Eady. Typical ~10-20% ACC transport difference at 1°, absorbed by `α` tuning. |
| Solar forcing | Seasonal cycle | Annual mean by quadrature of eq B5 (Phase 1) | Loses seasonal MLD cycle; mean state similar |
| Restoring | Native heat-flux coefficient (W/m²/K) | Coefficient → timescale conversion done in DINO module | Mathematically equivalent in surface layer; conversion uses Δz_0 |
| Tracer advection | NEMO FCT | `ppm_fct` (closest available) | Bit-exact match not expected; numerical diffusion comparable |
| MPAS channel | 50° wide periodic | 50° wide periodic via `create_regional_voronoi_mesh(periodic_x=True)` | Match. Cross-grid comparison can be quantitative. |
| Sea ice | Absent (paper limitation) | Absent (matches paper) | Limits AABW formation realism in both |
| Spin-up duration | 3000 yr R1 + 400 yr production | TBD (likely 50-200 yr at first) | Mean state will not be fully equilibrated; deep cells especially slow to adjust |

## Verification

### Stage gates (pass before moving on)

1. **Rest-state test** (lat-lon AND MPAS): Initialize DINO with no forcing, run 30 days → max|u|, max|v| < 1e-4 m/s. With SMC03 PGF this should be at the float64 noise floor; with the default `"adcroft"` PGF it would be 100–150× larger (see `docs/ocean_experiments/pgf_test_plan.md`).
2. **Wind spin-up**: Apply wind only (no restoring, no solar) → after 30 days, subtropical and subpolar gyres should be visible in the barotropic stream function with sensible western intensification. ACC should be ramping up in the channel.
3. **Short full-forcing run** (60 days, test-matrix entry): wind + restoring + solar. Checks:
   - SST distribution tracks restoring profile (RMS error < 5 K after 60 d at 2°)
   - No NaN, no |η| > 2 m, no SST outside [-2, 32] °C
   - Volume drift < 0.01%; heat content drift consistent with surface flux integral (within 1%)
   - σ_2 stratification monotonically increases with depth on average (no static instability surviving past convective adjustment)
4. **Cross-grid comparison** (lat-lon Mercator vs MPAS, both at ~1° equatorial): qualitatively similar gyre structure and channel flow direction. **Quantitative ACC transport will differ** because the MPAS channel is full-globe (360°) vs lat-lon's 50° — see Phase 3 caveat.

### Numeric targets vs Kamm et al. 2025 (after multi-century spin-up)

The paper's R1 production uses **3000 yr spin-up + 400 yr R1**, with diagnostics averaged over the last 50 yr. Our short runs cannot reproduce these magnitudes. Treat as targets for *long-run* production-script validation, not Phase 4 test-matrix gates.

| Metric | Paper R1 | Paper R4 | Notes |
|---|---|---|---|
| ACC transport (Sv) | **206.0** | 149.7 | Sensitive to GM tuning (Visbeck≠Tréguier) and momentum-viscosity formulation (geopotential vs isoneutral). Acceptable band for our R1: 150–250 Sv. |
| Total domain KE (J) | n/a in paper | ≈0.8 × 10¹⁸ | Paper Fig 8 |
| Total domain KE (J) — R16 | n/a | ≈2.1 × 10¹⁸ | Paper Fig 8; not a Phase-1 target |
| MHT peak northward (PW) | ≈0.4 | ≈0.25 | Paper Fig 7 |
| MOC topology | 3 cells (tropical, subtropical, deep) | same | In σ_2 space referenced to 2000 m; paper Fig 5 |
| σ_2 surface | < 35 kg/m³ | same | Paper E1 water-mass partition |
| σ_2 NADW | 35–36 kg/m³ | same | |
| σ_2 AABW | > 36 kg/m³ | same | Limited by no sea-ice — see Known Approximations |

### Required diagnostics for paper-style figures

- Barotropic stream function (lat-lon: integrate U·dz then take meridional cumsum; MPAS: same on dual grid)
- Zonal-mean MOC in σ_2 space referenced to 2000 m (using `ρ_ref(z=0)=1026`, `ρ_ref(z=2000)=1035`)
- Zonal-mean potential density σ_2 (Fig 6)
- Meridional heat transport: total + eddy decomposition (mean-flow MHT vs eddy MHT vs GM-bolus MHT)
- KE time series (domain-integrated)
- (Eddy-permitting only, deferred Phase 5) KE spectra and coarse-grained subgrid flux fields
