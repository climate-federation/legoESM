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
```

### 2B. Analytical Bathymetry (Appendix A, eqs. A1-A5)

Implement carefully:
- `_smooth_step(x, a, b)` — eq. A2 (6th-degree polynomial: `6t⁵ - 15t⁴ + 10t³`)
- `_tapered_exponential(x, x1, x2, s, d, delta_lambda)` — eq. A1 (3-branch piecewise with tapering)
- `_dino_bathymetry(lon_deg, lat_deg, config)`:
  - Compute `g_phi` (meridional shape) and `g_lambda` (zonal shape) per eq. A3
  - **Slope parameter correction**: `s_phi = cos(π·φ_max/180) · s_lambda` for Mercator grid
  - Apply channel modification (eq. A4): remove zonal walls within channel latitudes
  - Add Scotia Ridge sill (eq. A5): Gaussian ring at (-50°E, -55°N), only where depth < H_sill, restricted to western side via smooth step
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
- `Q_sr(φ) = max(230·cos(φ), 0)` (zeroth-order annual mean, overestimates at high lat)
- Shortwave penetration: Jerlov type I (ζ₀=0.35m, ζ₁=23m) — use existing module

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
  - Wind: DINO piecewise cubic profile (custom, computed in experiment module)
  - Restoring: T and S with DINO-specific profiles and coefficients
  - Non-solar/solar split: Q_sr subtracted from T restoring in surface layer
  - KPP: with A_v_bg=1.2e-4, K_v_bg=1.2e-5
  - Convection: enhanced diffusion with K_conv=100
  - GM/Redi: Visbeck adaptive, kappa_min=200, kappa_max=2000
  - Harmonic viscosity: grid-dependent, U_M=0.27 m/s (momentum only, geopotential)
  - Bottom drag: quadratic, C_d=1e-3
  - Shortwave: Jerlov type I
- `validate_results(final_state, diagnostics, config)` — basic checks

Support both `grid_type="latlon"` and `grid_type="mpas"`.

---

## Phase 3: MPAS Mesh for DINO

**Approach**: Use a global icosahedral MPAS mesh with land masking.

On a global MPAS mesh, the re-entrant channel is naturally open — there are no zonal walls at the channel latitudes because the mesh covers the entire globe. The DINO basin is carved by masking out all cells outside the 50°-wide sector as land, EXCEPT within the channel latitudes (45-65°S) where ocean extends around the full globe (or at least wraps continuously). The key insight: the channel doesn't need periodic BCs on a global mesh because the flow can literally go around the world.

However, this means the MPAS DINO domain will have a larger channel than the lat-lon version (360° vs 50° wide). This is an acceptable difference for a first implementation — the channel dynamics are set by the wind and bathymetry, not the domain width.

**Alternative** (if exact match needed): Use `mpas_channel` mode for the channel latitudes + closed basin for the rest. This is more complex and deferred.

**Resolution**: ico5 (~240 km) or ico6 (~120 km). ico6 is closer to 1° (111 km at equator).

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
- Configurable duration (default: 400 years for R1)
- Diagnostics output: barotropic stream function, MOC in density space, meridional heat transport, KE time series
- Support both lat-lon and MPAS via command-line flag
- Snapshot output every N years
- dt = 2700s (45 min)

---

## Phase 5 (Deferred): Enhancements

- Seasonal cycle forcing (time-dependent T_star, Q_sr with 1-month lag)
- FCT tracer advection for MPAS
- Simplified Roquet EOS option (eq. 6 with cabbeling + thermobaric)
- Higher resolution variants (R4=1/4°, R16=1/16°) with biharmonic + Smagorinsky
- TKE vertical mixing closure (Blanke & Delecluse 1993)
- Coarse-graining and subgrid flux diagnostics (Sect. 2.4 of paper)
- Isopycnal momentum viscosity (currently using geopotential approximation)
- Tréguier (1997) GM coefficient (currently using Visbeck approximation)

---

## Files to Create

| File | Description | Est. LOC |
|------|-------------|----------|
| `src/legoesm/ocean/experiments/dino.py` | DINO experiment module | ~600 |
| `scripts/run_dino.py` | Standalone production script | ~150 |

## Files to Modify

| File | Change | Est. LOC |
|------|--------|----------|
| `src/legoesm/grids/latlon.py` | Add `create_mercator_grid()` | ~80 |
| `src/legoesm/ocean/physics/lateral_mixing/config.py` | Add scaling params to Harmonic/BiharmonicConfig | ~20 |
| `src/legoesm/ocean/physics/lateral_mixing/harmonic.py` | Grid-dependent coefficient computation | ~30 |
| `src/legoesm/ocean/physics/lateral_mixing/biharmonic.py` | Grid-dependent coefficient computation | ~30 |
| `src/legoesm/ocean/experiments/__init__.py` | Register DINO experiment | ~5 |
| `scripts/run_ocean_test_matrix.py` | Add DINO test case | ~20 |

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
| EOS | Simplified Roquet (cabbeling+thermobaric) | Wright (full nonlinear) | Quantitative density differences; metrics won't match paper exactly |
| Vertical mixing | TKE (Blanke & Delecluse 1993) | KPP (LMD94) + enhanced diffusion | Different BL depth diagnosis; similar bulk behavior |
| Momentum viscosity | Isopycnal Laplacian | Geopotential Laplacian | Affects ACC where isopycnal slopes are steep |
| GM coefficient | Tréguier (1997) | Visbeck (1997) | Both growth-rate-dependent; Visbeck uses Eady approximation |
| Solar forcing | Seasonal cycle | Annual mean (Phase 1) | Loses seasonal variability; mean state similar |
| MPAS channel | 50° wide periodic | Full-globe (360°) channel | Different channel width; dynamics set by local wind+bathymetry |

## Verification

1. **Rest-state test**: Initialize DINO with no forcing → velocities should stay < 1e-6 m/s
2. **Wind spin-up**: Apply wind only (no restoring) → gyres should develop within 30 days
3. **Full forcing**: Wind + restoring + solar → check:
   - Subtropical/subpolar gyres form
   - ACC develops in channel
   - SST distribution matches restoring profile qualitatively
   - Conservation: volume, heat, salt drift < 0.1% over 10 years
4. **Cross-grid comparison**: Lat-lon vs MPAS should produce qualitatively similar circulations
5. **Benchmark against paper**: After 400-year spin-up, compare ACC transport (~150-200 Sv), MOC structure, meridional heat transport against Figs 4-7 of Kamm et al.
