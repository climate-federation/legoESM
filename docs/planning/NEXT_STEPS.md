# legoESM: Next Steps Implementation Guide

## Context

legoESM is a differentiable Earth System Model in JAX with 24+ dynamical cores, 25+ physics schemes, ocean/land/ice/lake components, and year-long AMIP runs completed. The model needs targeted work to become a credible climate model. This document defines the implementation tasks in priority order.

**Current baseline**: 365-day AMIP at C16/L20 (cubed-sphere) and T21/L20 (spectral), both with gray radiation + SBM convection + saturation adjustment. LW_TOA ≈ 236–255 W/m², precip ≈ 4–4.6 mm/day.

**Target**: 10-year AMIP at C48/L40 with RRTMGP, diurnal cycle, clouds, ozone, real topography. Global energy balance < 1 W/m², zonal-mean T within 5 K of ERA5.

---

## Task 1: Diurnal Cycle in Radiation ✅ COMPLETE

**Priority**: Highest — without this, surface energy partitioning is fundamentally wrong.

**Current state**: `solar.py` exists with zenith angle computation. AMIP runs use globally-averaged S₀/4 insolation (no day/night).

**Files to modify**:
- `src/legoesm/atmosphere/physics/radiation/integration.py` — the `make_radiation_physics()` factory
- `src/legoesm/atmosphere/physics/radiation/solar.py` — zenith angle utilities
- `src/legoesm/atmosphere/physics/radiation/gray.py` — gray radiation (for testing)
- `src/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py` — RRTMGP wrapper
- `scripts/run/run_amip.py` and `scripts/run_amip_spectral.py` — pass time-of-day

**Implementation**:
1. In `solar.py`, ensure there is a function `compute_cos_zenith(lat, lon, day_of_year, seconds_of_day) → cos_sza` that returns the cosine of the solar zenith angle per grid cell. If it already exists, verify it handles the full orbital parameters (declination, hour angle).
2. In the radiation integration, replace the constant `S₀/4` with `S₀ · max(cos_sza, 0)` per column. The factor of 1/4 goes away because the zenith angle already accounts for the geometric projection.
3. The `cos_sza` must be passed to RRTMGP's shortwave solver. Check the RRTMGP bundle at `atmosphere/physics/radiation/rrtmgp/rte/` — the SW solver should accept `mu0` (cosine zenith) as input.
4. For gray radiation, modify the SW Beer-Lambert law: `F_sw_down(k) = S₀ · cos_sza · exp(-D · τ_sw · σ_k)` (currently uses `S₀/4`).
5. In `run_amip.py` / `run_amip_spectral.py`, track elapsed seconds and pass `day_of_year` and `seconds_of_day` to the physics.
6. Add a `RadiationConfig` option `diurnal_cycle: bool = True` (default True for RRTMGP, False for gray to preserve backward compatibility).

**Validation**: Run 30-day C16/L20 with gray + diurnal vs. gray + S₀/4. The diurnal run should show:
- Day/night temperature contrast of ~10–20 K at the surface
- Same time-mean global energy balance (within 5 W/m²)
- No numerical instability from the sudden day/night transition

**Estimated effort**: 3–5 days.

---

## Task 2: Prescribed Ozone for RRTMGP ✅ COMPLETE

**Priority**: Highest — stratospheric temperature structure is completely wrong without ozone absorption.

**Current state**: `forcing/external.py` has `OzoneConfig` as a placeholder (`source="constant"` with no actual ozone profile). RRTMGP gas optics accept ozone as an input.

**Files to modify**:
- `src/legoesm/forcing/external.py` — implement ozone profile loading
- `src/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py` — pass ozone to gas optics
- `src/legoesm/atmosphere/physics/radiation/config.py` — add ozone config fields

**Implementation**:
1. Create a simple analytical ozone profile as a function of pressure (latitude-independent first pass):
   ```
   O₃(p) = O₃_max · exp(-((log(p) - log(p_peak))² / (2σ²)))
   ```
   where `p_peak ≈ 30 hPa`, `O₃_max ≈ 8 ppmv`, `σ ≈ 1.5` (log-pressure units). This gives a reasonable stratospheric ozone layer.
2. For a better approximation, add latitude dependence: ozone mixing ratio is ~2x higher at poles than at the equator in the lower stratosphere. A simple `1 + 0.5·sin²(lat)` scaling factor works.
3. In `external.py`, implement `source="analytical"` mode that returns `o3_mmr(pressure, lat)` as a JAX array of shape `(ncol, nlev)`.
4. For `source="file"`, plan to load CMIP6 ozone forcing (netCDF). This can be deferred — analytical is sufficient for Phase 1.
5. In the RRTMGP wrapper, locate where gas concentrations are assembled (likely in `rrtmgp/optics/` gas optics). Currently CO₂/CH₄/N₂O are set from `GHGConfig`. Add O₃ from the profile. Ozone is typically passed as volume mixing ratio per level per column, not a single scalar like the well-mixed gases.
6. Ensure ozone is on the correct vertical grid (RRTMGP may need values at layer midpoints or interfaces — check the RRTMGP bundle internals).

**Validation**: Run 30-day C16/L20 with RRTMGP + ozone vs. RRTMGP without ozone.
- Stratospheric temperature should warm by ~30–50 K (from ~180 K to ~220–240 K at 10 hPa)
- Stratopause near 1 hPa should appear
- Tropospheric climate should be minimally affected

**Estimated effort**: 3–5 days.

---

## Task 3: Cloud Fraction and Cloud–Radiation Coupling ✅ COMPLETE

**Priority**: Highest — the dominant source of uncertainty in climate models, and currently completely absent.

**Current state**: Prognostic `q_cloud` and `q_ice` exist in `AtmosphereState`. Six microphysics schemes exist. RRTMGP has cloud optics (`rrtmgp/optics/`). But **nothing connects them**: no cloud fraction is diagnosed, and radiation sees a clear-sky atmosphere.

**Files to create**:
- `src/legoesm/atmosphere/physics/clouds/cloud_fraction.py` — diagnostic cloud fraction

**Files to modify**:
- `src/legoesm/atmosphere/physics/radiation/integration.py` — wire cloud properties to RRTMGP
- `src/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py` — pass cloud optical properties
- `src/legoesm/atmosphere/physics/combined.py` — ensure cloud fraction is computed before radiation

**Implementation**:
1. **Diagnostic cloud fraction** — implement a Sundqvist-style or Xu-Randall scheme:
   ```
   Xu-Randall (1996):
   cf = (RH)^p · [1 - exp(-α · q_c / ((1 - RH) · q_s))]
   ```
   where `RH` is relative humidity, `q_c = q_cloud + q_ice`, `q_s` is saturation mixing ratio, and `p ≈ 0.25`, `α ≈ 100`. This gives cloud fraction that is 0 in dry subsaturated air and approaches 1 near saturation with condensate present.

   Alternatively, a simpler scheme: `cf = max(0, min(1, (RH - RH_crit) / (1 - RH_crit)))` with `RH_crit = 0.7`. This is the Sundqvist (1988) approach and is easier to tune.

2. **Cloud optical properties** for RRTMGP:
   - Liquid water path per layer: `LWP = q_cloud · dp/g · cf` (in-cloud value = `q_cloud / cf`)
   - Ice water path per layer: `IWP = q_ice · dp/g · cf`
   - Effective radius: assume `r_eff_liq = 10 μm`, `r_eff_ice = 30 μm` (constants first, later from droplet number)
   - The RRTMGP cloud optics module should take (LWP, IWP, r_eff_liq, r_eff_ice) per layer and return optical depth, single scatter albedo, and asymmetry parameter.

3. **Wiring**: In `combined.py`, the physics calling order should be:
   ```
   microphysics → cloud_fraction → radiation → convection → turbulence → GWD
   ```
   Cloud fraction must be computed *after* microphysics updates `q_cloud`/`q_ice` and *before* radiation.

4. **Max-random overlap**: RRTMGP likely expects cloud optical properties with an overlap assumption. Start with maximum-random overlap (the standard). If the RRTMGP bundle has a McICA sampler, use it. If not, pass total cloud optical depth per layer and let RRTMGP treat each layer independently (equivalent to random overlap).

5. Add `CloudConfig(scheme="xu_randall", rh_crit=0.7, r_eff_liq=10e-6, r_eff_ice=30e-6)` to the physics config.

**Validation**: Run 30-day C16/L20 with RRTMGP + clouds vs. clear-sky RRTMGP.
- Cloud radiative effect (CRE): LW CRE ≈ +25–30 W/m² (warming), SW CRE ≈ −45 to −50 W/m² (cooling), net CRE ≈ −20 W/m²
- Global cloud fraction ~60–70%
- TOA energy balance should be closer to observed (~240 W/m² OLR)

**Estimated effort**: 2 weeks.

---

## Task 4: Hybrid Sigma-Pressure Vertical Coordinate ✅ COMPLETE

**Priority**: High — required for real topography without catastrophic PGF errors.

**Current state**: `grids/vertical.py` has `SigmaCoordinate` (pure σ = p/p_s) and `HeightCoordinate` (z*). No hybrid coordinate.

**Files to modify**:
- `src/legoesm/grids/vertical.py` — add `HybridSigmaPressureCoordinate`
- `src/legoesm/atmosphere/dynamics/primitive_eq.py` — update pressure computation
- `src/legoesm/atmosphere/dynamics/primitive_eq_fv.py` — same
- `src/legoesm/atmosphere/dynamics/spectral_pe.py` — update Simmons-Burridge geopotential
- All PE-using physics modules — pressure now = `A(k)·p₀ + B(k)·p_s` instead of `σ·p_s`

**Implementation**:
1. Hybrid coordinate: `p(k) = A(k)·p₀ + B(k)·p_s` where:
   - Near top: A → 1, B → 0 (pure pressure levels)
   - Near surface: A → 0, B → 1 (pure sigma)
   - Smooth transition in mid-troposphere
2. Standard Simmons-Burridge vertical discretization still applies, but:
   - `Δp_k = (A_{k+½} - A_{k-½})·p₀ + (B_{k+½} - B_{k-½})·p_s`
   - `∂p_s/∂t` equation changes: `∂p_s/∂t = -∫ div(v·Δp_k) dk / Σ(B_{k+½} - B_{k-½})`
   - Sigma-dot replacement: generalized η-dot with A/B coefficients
3. Provide standard A/B coefficient sets:
   - L40 set matching ECMWF L40 (p_top ≈ 2 hPa)
   - L60 set matching ECMWF L60 (p_top ≈ 0.1 hPa)
   - Allow custom A/B arrays
4. The `HybridSigmaPressureCoordinate` NamedTuple should store:
   ```python
   A_half: jax.Array   # (nlev+1,) — interface A coefficients
   B_half: jax.Array   # (nlev+1,) — interface B coefficients
   A_full: jax.Array   # (nlev,) — midlevel A
   B_full: jax.Array   # (nlev,) — midlevel B
   p_ref: float        # reference pressure (typically 1e5 Pa)
   ```
5. Update the Simmons-Burridge geopotential computation: the `ln(p_{k+½}/p_{k-½})` and `α_k` terms now use the full hybrid pressure, not just σ ratios.
6. The spectral PE's semi-implicit scheme (`semi_implicit.py`) will need the Gamma matrix updated for hybrid coordinates — the vertical structure matrix depends on the coordinate definition.

**Validation**: Run Held-Suarez with hybrid σ-p at L40. Compare zonal-mean zonal wind and temperature with the pure-σ L20 run. Results should be nearly identical in the troposphere but better resolved in the stratosphere. Then add real topography (Task 6) and verify no blowup at the Himalayas/Andes.

**Estimated effort**: 2 weeks.

---

## Task 5: Increase Vertical Resolution to L40–L60 ✅ COMPLETE

**Priority**: High — 20 levels is insufficient for resolving the tropopause, boundary layer, and stratosphere.

**Current state**: All AMIP runs use 20 sigma levels.

**Implementation**:
1. Design a 40-level hybrid σ-p layout:
   - 5 levels in lowest 1 km (BL resolution)
   - ~15 levels from 1 km to tropopause (free troposphere)
   - ~10 levels from tropopause to 10 hPa (UTLS)
   - ~10 levels from 10 hPa to 2 hPa (stratosphere)
2. Implement as a function `make_hybrid_levels(nlev, p_top)` that returns A/B arrays.
3. Update `AMIPExperimentConfig` default to L40.
4. Ensure all physics schemes handle the variable layer thickness gracefully (some may assume roughly uniform Δσ).
5. Check the Thomas algorithm in `tridiagonal.py` for stability with very thin layers near the surface.

**Depends on**: Task 4 (hybrid coordinate).

**Estimated effort**: 3–5 days.

---

## Task 6: Real Topography and Land-Sea Mask ✅ COMPLETE

**Priority**: High — idealized topography means the model cannot represent orographic precipitation, monsoons, or realistic stationary waves.

**Files to modify**:
- `src/legoesm/grids/topography.py` — add real topography loading
- `src/legoesm/coupler/tile_fractions.py` — initialize from real land-sea mask
- `scripts/run/run_amip.py` — add topography configuration

**Files to create**:
- `scripts/prepare_topography.py` — offline preprocessing script that converts raw DEM to Zarr
- `data/topography/` — directory for preprocessed Zarr stores

**Data format**: All static boundary condition data (topography, land-sea mask, soil type, bathymetry) are stored as **Zarr stores**, consistent with legoESM's primary I/O format. No NetCDF dependency for runtime data loading.

**Implementation**:

1. **Offline preprocessing** (`scripts/prepare_topography.py`): A one-time script that:
   - Downloads ETOPO 2022 (preferred, 15 arc-second) or GMTED2010 from source
   - For each target grid resolution (C16, C48, C96, etc.), computes:
     - Area-averaged elevation per grid cell (conservative regridding from high-res to model grid)
     - Land fraction per grid cell: fraction of sub-grid DEM pixels above sea level (continuous 0–1, not binary)
     - Raw and filtered orographic surface geopotential `phis = g · z_surface`
   - Applies spectral truncation or Gaussian smoothing to remove scales the model can't resolve (2Δx filter). For the spectral grid, truncate at T_max. For cubed-sphere, use iterative Laplacian smoothing (3–5 passes).
   - Saves to Zarr store: `data/topography/etopo2022_C48.zarr` with variables:
     ```
     phis: (6, n, n) or (n_lat, n_lon)     — surface geopotential [m²/s²]
     z_surface: (6, n, n) or (n_lat, n_lon) — surface elevation [m]
     land_fraction: (6, n, n)               — fractional land cover [0–1]
     land_mask: (6, n, n)                   — binary land mask (land_fraction > 0.5)
     ```
   - Script is idempotent: if the Zarr store already exists at the target resolution, skip.

2. **Runtime loading** in `topography.py`: Add `load_real_topography(grid, zarr_path) → phis, land_fraction` that:
   - Opens the Zarr store with `xarray.open_zarr(zarr_path)` (lazy loading)
   - Extracts arrays as JAX arrays: `jnp.array(ds["phis"].values)`
   - Validates shape matches the grid
   - Returns `phis` as a `Field` with proper metadata

3. **Tile fraction initialization** from `land_fraction`:
   - `f_land = land_fraction` (continuous, from Zarr)
   - `f_ocean = 1 - f_land` (adjusted later by ice forcing)
   - `f_ice` initialized from AMIP sea-ice forcing data
   - `f_lake` = 0 initially (can add lake mask later from HydroLAKES or similar)
   - Store `TileFractions` initialization as part of the experiment config, not hardcoded.

4. **Cubed-sphere face-boundary smoothing**: Apply `edge_blending.py` to smooth topography at face edges. This prevents sharp gradients at cube corners from triggering numerical instabilities, especially with the PGF.

5. **Spectral grid variant**: For Gaussian grid runs, topography should be spectrally truncated at T_max (set spectral coefficients with n > n_max to zero after SH analysis of the raw field). This is the standard approach for spectral models and avoids Gibbs ringing.

6. **Config integration**: Add to `AMIPExperimentConfig`:
   ```python
   topography_source: str = "idealized"  # "idealized", "zarr"
   topography_zarr_path: str = ""        # path to Zarr store
   ```

**Depends on**: Task 4 (hybrid coordinate, for PGF accuracy over topography).

**Validation**: Run Held-Suarez with real topography. Check for:
- No PGF noise over steep topography (Himalayas, Andes)
- Stationary wave pattern (Rossby wave train from orography)
- Mountain-induced precipitation enhancement (crude, since Held-Suarez has no moisture)

**Estimated effort**: 1–2 weeks.

---

## Task 7: Activate Microphysics in AMIP Runs ✅ COMPLETE

**Priority**: High — saturation adjustment is not sufficient for a climate model.

**Current state**: Kessler, Sundqvist, Seifert-Beheng, Morrison, Thompson, and ML emulator all exist as code. AMIP runs use saturation adjustment only.

**Files to modify**:
- `scripts/run/run_amip.py` — add `--microphysics` flag
- `src/legoesm/atmosphere/physics/combined.py` — wire microphysics into the physics sequence
- `src/legoesm/atmosphere/physics/microphysics/integration.py` — ensure `make_microphysics_physics()` works with the AMIP state

**Implementation**:
1. Start with Kessler warm-rain microphysics — it is the simplest and easiest to debug.
2. In `combined.py`, add microphysics after convection:
   ```
   radiation → convection → microphysics → turbulence → GWD
   ```
3. Ensure that microphysics tendencies for `q_vapor`, `q_cloud`, `q_rain` are applied, and that `q_cloud` is advected by the dynamics (check that the tracer transport in the dycore handles it).
4. Add precipitation from microphysics to the surface forcing (for land/ocean coupling).
5. Once Kessler is working, switch to Sundqvist for a more realistic large-scale condensation.
6. Sub-cloud rain evaporation: as rain falls through unsaturated layers, evaporate a fraction proportional to `(1 - RH) · sqrt(rain_rate)`. Add this to Kessler/Sundqvist if not already present.

**Depends on**: Task 3 (cloud fraction, which needs q_cloud from microphysics).

**Validation**: 30-day AMIP with Kessler vs. saturation adjustment. The Kessler run should show:
- Cloud water mixing ratios of ~0.1–0.5 g/kg in the tropics
- Precipitation more concentrated in convergence zones (less uniform drizzle)
- Non-zero `q_cloud` and `q_rain` in the state

**Estimated effort**: 1 week.

---

## Task 8: Multi-Layer Soil with Richards Equation, Runoff, and Retention Curves ✅ COMPLETE

**Priority**: Medium-high — the slab land model has no memory beyond the current timestep, no vertical water transport, and no runoff generation.

**Files to create**:
- `src/legoesm/land/soil_grid.py` — flexible vertical soil grid
- `src/legoesm/land/soil_hydraulics.py` — retention curves and hydraulic conductivity
- `src/legoesm/land/richards.py` — Richards equation solver (mixed-form, Celia et al. 1990)
- `src/legoesm/land/soil_thermal.py` — multi-layer soil heat diffusion
- `src/legoesm/land/runoff.py` — surface and subsurface runoff generation
- `src/legoesm/land/infiltration.py` — infiltration with ponding

**Files to modify**:
- `src/legoesm/land/slab_land.py` → rename to `land_model.py`, upgrade to multi-layer
- `src/legoesm/land/state.py` — extend `LandState` with per-layer arrays
- `src/legoesm/land/config.py` — add `SoilGridConfig`, `SoilHydraulicsConfig`, `RunoffConfig`
- `src/legoesm/coupler/coupler.py` — pass runoff to ocean coastal cells
- `src/legoesm/ocean/dynamics/ocean_pe.py` — receive freshwater runoff forcing
- `src/legoesm/coupler/coupling_fields.py` — add `runoff` field to `SurfaceToAtm` / `SurfaceToOcean`

### 8A. Flexible Vertical Soil Grid (`soil_grid.py`)

The soil grid must be configurable, with a sensible default that increases layer thickness with depth (power-of-2 geometric progression).

**Default grid** — each layer is 2× the thickness of the one above:

| Layer | Thickness (m) | Depth to bottom (m) | Purpose |
|-------|--------------|---------------------|---------|
| 1 | 0.025 | 0.025 | Skin / diurnal cycle |
| 2 | 0.05 | 0.075 | Upper root zone |
| 3 | 0.10 | 0.175 | Root zone |
| 4 | 0.20 | 0.375 | Root zone |
| 5 | 0.40 | 0.775 | Deep root zone |
| 6 | 0.80 | 1.575 | Subsoil |
| 7 | 1.60 | 3.175 | Deep subsoil |
| 8 | 3.20 | 6.375 | Bedrock interface |

The grid is defined by a `SoilGridConfig` NamedTuple:

```python
@dataclass(frozen=True)
class SoilGridConfig:
    """Flexible soil vertical grid. Default: geometric (power-of-2) spacing."""
    n_layers: int = 8
    dz_top: float = 0.025          # thickness of first layer [m]
    growth_factor: float = 2.0     # each layer is growth_factor × previous
    custom_dz: tuple | None = None # if set, overrides geometric grid
    total_depth: float | None = None  # if set with n_layers, auto-compute dz_top
```

Usage:
- Default: `SoilGridConfig()` → 8 layers, 0.025 m top, factor 2, total depth ~6.4 m
- Custom: `SoilGridConfig(custom_dz=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0))` → 6 layers, user-specified
- Auto: `SoilGridConfig(n_layers=10, total_depth=8.0, growth_factor=1.8)` → solver computes `dz_top` to fill target depth
- CLM-like: `SoilGridConfig(custom_dz=(0.018, 0.028, 0.046, 0.075, 0.124, 0.204, 0.336, 0.554, 0.913, 1.506))` — 10 layers matching CLM5 node spacing

Build function: `make_soil_grid(config) → SoilGrid` returning a NamedTuple with:
```python
SoilGrid(
    dz: jax.Array,           # (n_layers,) layer thicknesses [m]
    z_node: jax.Array,       # (n_layers,) depth of layer midpoints [m]
    z_interface: jax.Array,  # (n_layers+1,) depth of layer interfaces [m]
    dz_interface: jax.Array, # (n_layers-1,) distance between adjacent midpoints [m]
    n_layers: int,
)
```

### 8B. Soil Water Retention Curves (`soil_hydraulics.py`)

Implement a **pluggable retention curve system** with four options. All functions must be JAX-traceable (no Python branching on runtime values).

**Option 1: Clapp-Hornberger (1978)** — power-law, used by many GCMs (GFDL, early CLM):
```
ψ(θ) = ψ_sat · (θ/θ_sat)^(-b)
K(θ) = K_sat · (θ/θ_sat)^(2b+3)
C(θ) = dθ/dψ = -θ_sat / (b · ψ_sat) · (θ/θ_sat)^(b+1)
```
Parameters per soil type: `θ_sat, ψ_sat, b, K_sat` from Cosby et al. (1984) 11-class lookup table.

**Option 2: van Genuchten-Mualem (1980)** — smooth sigmoid, state of the art, used by CLM5/ELM/HYDRUS:
```
Se(ψ) = [1 + |α·ψ|^n]^(-m)     where m = 1 - 1/n
θ(ψ) = θ_r + (θ_sat - θ_r) · Se
K(Se) = K_sat · Se^(1/2) · [1 - (1 - Se^(1/m))^m]^2
C(ψ) = α·m·n·|α·ψ|^(n-1) · (θ_sat - θ_r) · [1 + |α·ψ|^n]^(-m-1)
```
Parameters: `θ_r, θ_sat, α, n, K_sat`. Pedotransfer from Schaap et al. (2001) Rosetta, or per-texture-class defaults.

**Option 3: Brooks-Corey (1964)** — piecewise power-law with explicit air entry:
```
Se(ψ) = |ψ_b/ψ|^λ    for ψ < ψ_b (unsaturated)
Se(ψ) = 1             for ψ ≥ ψ_b (saturated)
K(Se) = K_sat · Se^(3 + 2/λ)
```
Parameters: `θ_r, θ_sat, ψ_b, λ, K_sat`. Sharp air-entry makes this simpler but less smooth — use `sigmoid_switch` from `core/smooth.py` at the air-entry transition for differentiability.

**Option 4: Campbell (1974)** — simplified power-law (Clapp-Hornberger is the discretized version):
```
ψ(Se) = ψ_e · Se^(-b)
K(Se) = K_sat · Se^(2b+3)
```
Parameters: `ψ_e, b, θ_sat, K_sat`.

**Config**:
```python
@dataclass(frozen=True)
class SoilHydraulicsConfig:
    retention_curve: str = "van_genuchten"  # "clapp_hornberger", "van_genuchten", "brooks_corey", "campbell"
    # Van Genuchten defaults (loam):
    theta_r: float = 0.078
    theta_sat: float = 0.43
    alpha: float = 3.6      # [1/m]
    n_vg: float = 1.56
    K_sat: float = 2.89e-6  # [m/s] (~0.25 m/day)
    # Clapp-Hornberger defaults (loam):
    psi_sat: float = -0.478  # [m]
    b_ch: float = 5.39
    # Allow per-grid-cell parameters (for heterogeneous soil maps):
    spatially_varying: bool = False
    soil_type_map: str = ""  # path to Zarr with per-cell soil type index
```

**Interblock hydraulic conductivity**: Use geometric mean between adjacent layers `K_{k+1/2} = sqrt(K_k · K_{k+1})` for unsaturated flow, arithmetic mean for saturated. This avoids smearing of wetting fronts (Haverkamp & Vauclin 1979).

### 8C. Richards Equation Solver (`richards.py`)

Use the **Celia et al. (1990) mixed-form modified Picard iteration** — this is the gold standard for mass-conservative unsaturated flow and handles both dry and wet extremes robustly.

**Governing equation** (1D vertical, mixed form):
```
∂θ/∂t = ∂/∂z [K(ψ) · (∂ψ/∂z + 1)] - S(z)
```
where the `+1` is the gravitational term (z positive downward), and S is the sink term (root water uptake, evaporation).

**Why mixed form**: The head-based form `C(ψ)·∂ψ/∂t = ...` has poor mass conservation because `C(ψ)` is evaluated at the old iterate. The θ-based form `∂θ/∂t = ∇·[D(θ)∇θ]` fails near saturation (D → ∞) and at layer interfaces with different soil types (θ is discontinuous but ψ is continuous). The mixed form time-steps in θ (guaranteeing mass conservation) but parameterizes fluxes in ψ (handling saturation and heterogeneity).

**Discretization** — modified Picard iteration (Celia et al. 1990):

For each Picard iteration `m` within timestep `n → n+1`:

1. Compute the residual δψ from the linearized system:
   ```
   [C^m/Δt + A(K^m)] · δψ = -(θ^m - θ^n)/Δt + A(K^m)·ψ^m + g_flux
   ```
   where `A` is the tridiagonal diffusion operator built from interblock `K` values, and `g_flux` includes gravitational drainage and boundary fluxes.

2. Update: `ψ^{m+1} = ψ^m + δψ`, then `θ^{m+1} = θ(ψ^{m+1})` from the retention curve.

3. Convergence check: `max|θ^{m+1} - θ^m| < ε_θ` where `ε_θ = 1e-6 m³/m³`. Checking convergence in θ (not ψ) ensures mass conservation.

4. The tridiagonal system is solved with the Thomas algorithm from `tridiagonal.py`.

**JAX implementation**: Use `jax.lax.while_loop` for the Picard iteration (variable number of iterations, max 10). The convergence criterion uses `jnp.max(jnp.abs(theta_new - theta_old)) < eps`. For JIT compatibility, always run to `max_iter` but mask updates after convergence using `jnp.where`.

**Boundary conditions**:

- **Top**: Flux boundary (Neumann) = `precip - evap - runoff_surface`. When infiltration demand exceeds `K_sat` of the top layer, excess becomes surface runoff (ponding/infiltration-excess). Implement as:
  ```
  flux_demand = precip - evap
  flux_max = K_sat_top · (1 + |ψ_top|/dz_top)  # infiltration capacity
  flux_infiltration = smooth_min(flux_demand, flux_max)  # differentiable
  runoff_surface = flux_demand - flux_infiltration
  ```
  When `ψ_top ≥ 0` (saturated), the surface is ponded: switch to Dirichlet `ψ_top = max(0, ponding_depth)`. Use `sigmoid_switch` for the smooth transition between flux and ponded BCs.

- **Bottom**: Free drainage (gravitational flux only): `q_bottom = -K(ψ_N)` (Neumann). This produces subsurface runoff. Alternative: zero-flux (no drainage), or prescribed water table depth — selectable via config.

**Elastic storage term for saturation**: Near and at saturation (ψ ≥ 0), add a small specific storage `S_s ≈ 1e-4 m⁻¹` to the capacity: `C_total = C(ψ) + S_s · θ_sat`. This regularizes the system when `C → 0` at saturation and naturally handles the saturated→unsaturated transition without branching. This is the standard approach from Miller et al. (1998).

### 8D. Soil Thermal Diffusion (`soil_thermal.py`)

Multi-layer heat equation, coupled to moisture through thermal properties:

```
C_eff(z) · ∂T/∂t = ∂/∂z [k_eff(z) · ∂T/∂z]
```

**Thermal conductivity** — Johansen (1975) method:
```
k_eff = k_dry + (k_sat - k_dry) · K_e(Sr)
```
where `Sr = (θ - θ_r)/(θ_sat - θ_r)` is the saturation ratio, `K_e` is the Kersten number:
- Unfrozen coarse soil: `K_e = 0.7 · log10(Sr) + 1.0` for `Sr > 0.05`
- Unfrozen fine soil: `K_e = log10(Sr) + 1.0` for `Sr > 0.1`
- `k_dry` from de Vries (1963): `k_dry = (0.135·ρ_b + 64.7) / (2700 - 0.947·ρ_b)` [W/m/K]
- `k_sat = k_solid^(1-θ_sat) · k_water^θ_sat` (geometric mean)

**Heat capacity**:
```
C_eff = (1 - θ_sat)·C_soil + θ·C_water + (θ_sat - θ)·C_air
```
where `C_soil ≈ 2.0e6 J/m³/K` (mineral), `C_water = 4.18e6`, `C_air = 1.25e3`.

Discretized with backward Euler + Thomas algorithm (already in `tridiagonal.py`). Top BC: surface energy balance flux `G = R_net - H - LE`. Bottom BC: zero flux.

### 8E. Runoff Generation (`runoff.py`)

Runoff has two components generated at every land grid cell, every timestep:

1. **Surface (Hortonian/infiltration-excess) runoff**: `R_surface = max(0, P - E - f_infiltration)` where `f_infiltration` is the infiltration capacity from the Richards solver top BC (see 8C above). When the top soil layer is saturated (ponding), all additional precipitation becomes surface runoff.

2. **Subsurface (baseflow/drainage) runoff**: `R_subsurface = K(ψ_bottom)` — the gravitational drainage flux at the bottom of the soil column. This represents slow groundwater-fed baseflow.

3. **Saturation-excess (Dunne) runoff**: When the water table rises to the surface (all layers saturated), additional precipitation becomes runoff regardless of `K_sat`. This emerges naturally from the Richards solver with elastic storage — no special treatment needed.

**Total runoff per grid cell**: `R_total = R_surface + R_subsurface` [kg/m²/s].

**Runoff routing to ocean** — implement in the coupler:
- Each land grid cell accumulates `R_total · A_cell · dt` over the coupling interval.
- Simple nearest-coast routing: for each land cell, find the nearest ocean cell. Route the accumulated runoff to that ocean cell as a freshwater flux.
- Add `runoff_freshwater` to `SurfaceToOcean` coupling fields [kg/s per ocean cell].
- The ocean model receives this as a surface freshwater flux that dilutes surface salinity:
  ```
  dS/dt|_runoff = -S · (R_runoff / (ρ_0 · h_surface))
  ```
  where `h_surface` is the thickness of the top ocean layer.
- This must work for **all ocean model variants** (FV cubed-sphere, spectral, FC-Gram, SFNO). The coupler is grid-agnostic — it passes freshwater flux through `coupling_fields.py`, so each ocean model just needs to apply it in its tendency computation.

**Config**:
```python
@dataclass(frozen=True)
class RunoffConfig:
    bottom_bc: str = "free_drainage"   # "free_drainage", "zero_flux", "water_table"
    water_table_depth: float = 10.0    # [m], only if bottom_bc="water_table"
    route_to_ocean: bool = True
    routing_method: str = "nearest_coast"  # "nearest_coast" (future: "river_network")
```

### 8F. Updated State and Integration

**Extended `LandState`**:
```python
@dataclass(frozen=True)
class LandState:
    T_soil: Field           # (ncol, n_layers) soil temperature [K]
    psi_soil: Field         # (ncol, n_layers) soil matric potential [m]
    theta_soil: Field       # (ncol, n_layers) volumetric water content [m³/m³]
    snow_depth: Field       # (ncol,) snow water equivalent [kg/m²]
    snow_cover: Field       # (ncol,) fractional snow cover [-]
    ponding_depth: Field    # (ncol,) surface water depth [m]
    runoff_surface: Field   # (ncol,) accumulated surface runoff [kg/m²/s]
    runoff_subsurface: Field # (ncol,) accumulated subsurface runoff [kg/m²/s]
```

Surface temperature is `T_soil[:, 0]` (top layer). A thin skin layer (layer 0 at 2.5 cm) resolves the diurnal cycle.

**Physics step ordering** in the land model:
1. Receive atmospheric forcing: precipitation, radiation, wind, temperature, humidity
2. Surface energy balance → top BC for soil heat (G) and soil moisture (infiltration)
3. Richards equation solve → updated θ, ψ, runoff
4. Soil thermal diffusion → updated T_soil
5. Snow accumulation/melt
6. Return: surface temperature, sensible/latent heat flux, albedo, runoff

**Validation**: Run 1-year AMIP. The multi-layer land should show:
- Realistic diurnal temperature range (~10–15 K over continents)
- Soil moisture memory of weeks to months (e-folding time increases with depth)
- Infiltration-excess runoff during intense precipitation events
- Baseflow runoff proportional to soil moisture
- No surface temperature blowup during polar night
- Mass conservation: total water in soil column + cumulative runoff + cumulative ET = initial water + cumulative precip (residual < 1e-8 kg/m²)
- Richards solver converges in ≤ 5 Picard iterations for typical Δt = 600 s
- Validate against CLM5 single-column results for a standard soil type

**Estimated effort**: 3–4 weeks.

---

## Task 9: PBL Height Diagnosis ✅ COMPLETE

**Priority**: Medium — needed for proper convection triggering and BL-free troposphere decoupling.

**Files to create**:
- `src/legoesm/atmosphere/physics/turbulence/pbl_height.py`

**Implementation**:
1. Bulk Richardson number method:
   ```
   Ri_b(z) = g·z·(θ_v(z) - θ_v(sfc)) / (θ_v(sfc) · (u(z)² + v(z)²))
   ```
   PBL height = lowest z where `Ri_b > Ri_crit` (typically 0.25).
2. Use linear interpolation between model levels to get a smooth PBL height.
3. Store as a diagnostic field in the state or pass to physics as a computed quantity.
4. Use PBL height in:
   - Convection: SBM trigger (CAPE computed from PBL top instead of surface)
   - Turbulence: Louis/TKE schemes can use PBL height to set mixing length caps
   - Holtslag-Boville and YSU already expect a PBL height input — wire it in

**Estimated effort**: 3–5 days.

---

## Task 10: Surface Albedo Improvements ✅ COMPLETE

**Priority**: Medium — constant albedo gives wrong energy balance at high latitudes.

**Files to modify**:
- `src/legoesm/land/slab_land.py` — snow albedo feedback
- `src/legoesm/coupler/coupling_fields.py` — pass surface albedo to radiation
- `src/legoesm/ice/sea_ice.py` — ice albedo depends on temperature and melt ponds

**Implementation**:
1. Land albedo: `α_land = α_veg · (1 - f_snow) + α_snow · f_snow` where:
   - `α_veg` depends on latitude (proxy for vegetation type): 0.15 in tropics, 0.20 at midlat, 0.25 at high lat
   - `α_snow = 0.8` for fresh snow, decaying with time: `α_snow(t) = α_snow_min + (α_snow_max - α_snow_min) · exp(-t/τ_snow)` with `τ_snow ≈ 5 days`
   - `f_snow` from snow depth: `f_snow = min(1, snow_depth / snow_depth_crit)` with `snow_depth_crit = 50 kg/m²`
2. Ice albedo: `α_ice = 0.65` (cold), `0.45` (warm/melting), interpolated by `T_ice - T_freeze`.
3. Ocean albedo: `α_ocean = 0.06` (constant) or zenith-angle dependent.
4. Pass tile-blended albedo to radiation via `SurfaceToAtm`.

**Estimated effort**: 3–5 days.

---

## Task 11: Energy Budget Closure Validation ✅ COMPLETE

**Priority**: Medium — must validate before claiming climate model status.

**Files to modify/create**:
- `src/legoesm/diagnostics/energy_budget.py` — online energy diagnostics

**Implementation**:
1. Track at every timestep (or every N steps):
   - TOA net radiation: `R_TOA = SW_down_TOA - SW_up_TOA - LW_up_TOA`
   - Surface net radiation: `R_sfc = SW_net_sfc + LW_net_sfc`
   - Surface sensible + latent heat flux
   - Column energy tendency: `d/dt ∫(c_p·T + L_v·q + g·z + ½v²) dp/g`
2. The residual `R_TOA - ∂E/∂t` should be < 0.1 W/m² for a climate model.
3. If the residual is large, identify the leak: is it in the physics (radiation-convection-microphysics energy non-conservation), the dynamics (advection not conserving energy), or the conservation fixer (over/under-correcting)?
4. Output as a timeseries diagnostic.

**Estimated effort**: 1 week.

---

## Task 12: 10-Year AMIP Run at C48/L40 ✅ COMPLETE

**Priority**: The capstone validation of Phase 1.

**Depends on**: Tasks 1–11.

**Configuration**:
```yaml
grid: cubed_sphere, C48
vertical: hybrid_sigma_pressure, L40, p_top=2hPa
dt: 600s (may need 450s at C48 for CFL)
radiation: rrtmgp, diurnal_cycle=True, ozone=analytical
clouds: xu_randall cloud fraction → RRTMGP cloud optics
microphysics: sundqvist (or kessler)
convection: sbm
turbulence: louis (or tke)
gwd: rayleigh
land: multi_layer_soil, 6 layers
forcing: COBE-SST2 or HadISST (real SST + sea ice)
topography: ETOPO1 filtered to C48
duration: 10 years
output: monthly means + daily snapshots
```

**Validation targets**:
- Global mean T_2m: 287–289 K (observed ~288 K)
- Global mean precipitation: 2.5–3.0 mm/day (observed ~2.7 mm/day)
- Net TOA imbalance: < 1 W/m² (ideally < 0.5 W/m²)
- OLR: 235–245 W/m² (observed ~240 W/m²)
- Zonal-mean zonal wind: subtropical jet at ~30° lat, ~30–40 m/s
- Zonal-mean temperature: within 5 K of ERA5 at all levels
- Hadley cell extent: ITCZ within 5° of observed position
- Seasonal cycle amplitude: comparable to ERA5
- WeatherBench2 RMSE evaluation at Day 1, 3, 5, 10

**Estimated effort**: 2 weeks (setup, tuning, running, analysis).

---

## Implementation Order Summary

| Order | Task | Effort | Depends On | Status |
|-------|------|--------|------------|--------|
| 1 | Diurnal cycle | 3–5 days | — | ✅ Complete |
| 2 | Prescribed ozone | 3–5 days | — | ✅ Complete |
| 3 | Cloud fraction + cloud-radiation coupling | 2 weeks | — | ✅ Complete |
| 4 | Hybrid σ-p coordinate | 2 weeks | — | ✅ Complete |
| 5 | Vertical resolution L40 | 3–5 days | Task 4 | ✅ Complete |
| 6 | Real topography + land-sea mask | 1–2 weeks | Task 4 | ✅ Complete |
| 7 | Activate microphysics | 1 week | — | ✅ Complete |
| 8 | Multi-layer soil + Richards + runoff | 3–4 weeks | — | ✅ Complete |
| 9 | PBL height diagnosis | 3–5 days | — | ✅ Complete |
| 10 | Surface albedo | 3–5 days | — | ✅ Complete |
| 11 | Energy budget validation | 1 week | Tasks 1–3 | ✅ Complete |
| 12 | 10-year C48/L40 AMIP run | 2 weeks | Tasks 1–11 | ✅ Complete |

Tasks 1, 2, 3, 4, 7, 8, 9, 10 can proceed in parallel across two implementation threads:
- **Thread A** (radiation/clouds): Tasks 1 → 2 → 3 → 11
- **Thread B** (infrastructure/surface): Tasks 4 → 5 → 6 → 8 → 9 → 10
- Task 7 (microphysics) fits in either thread
- Task 12 is the integration milestone

**Total estimated wall time**: 10–13 weeks of focused implementation (assumes single developer with Claude Code).

---

## Key Principles for Implementation

1. **Always run the existing test suite** after modifying any core file. `pytest tests/unit/ -x` should pass before committing.
2. **Preserve backward compatibility**: new features should be off-by-default or behind config flags. Existing AMIP scripts must still work.
3. **Keep everything differentiable**: no Python `if` on runtime values, no `np.where` — use `jnp.where`, `jax.lax.cond`, `sigmoid_switch` from `core/smooth.py`.
4. **Follow the factory pattern**: new physics modules get `config.py`, `output.py`, the scheme implementation, and `integration.py` with `make_*_physics()`.
5. **Test at C8 or C16 first**: never debug at C48. Use C8/L10 for correctness, C16/L20 for short validation, C48/L40 only for production runs.
6. **Conservation first**: after adding any new physics, verify that the conservation fixer still closes the mass/energy/moisture budget. Print residuals.
7. **One task at a time**: implement, test, validate, commit. Do not batch multiple tasks into one large change.
