# Plan: OMIP-Style Forced Global Ocean Run at 1°

## Recent Status (2026-05-05)

### What works
- 10-year stable run completed (88 min, 2× V100): `results/jra55_10yr_production/`
- Stability stack: basin-removal flood-fill + narrow-passage fill (min_width=2)
  + 3D maxvel clip (3 m/s) + slope-foot viscosity (alpha=3) + Smag (C=0.2)
  + SSS restoring + freeze cap + open Southern Ocean (south-cap-lat=−90).
- Production config: `A_h=2e5, K_h=1e3, B_h=5e9, B_h_barotropic=1e14`,
  `bottom_drag_r=2.5e-3, DRAG_BG_VEL=0.1`, GM/Redi (Visbeck adaptive
  kappa_GM=kappa_Redi=800, range [200, 2000]).

### Limitations of the production run
- Circulation looks like zonal jets, no closed gyres / ACC.
- Surface eq currents 0.15 m/s — over-damped (real EUC ~1 m/s at depth,
  surface SEC ~0.3–0.5 m/s).
- Global SST too cold (16.2°C vs ~17–18°C observed).
- Hypothesis: A_h=2e5 is way too high; baseline's "good-looking" equator
  is an accident of over-damping.

### Failed sweep — A_h reduction with K_h=0 (2026-05-05)

Tested two runs in parallel (one GPU each via `CUDA_VISIBLE_DEVICES`):

| Run | A_h | K_h | Result at day 524/1825 |
|-----|-----|-----|------------------------|
| A | 5e4 | 0 | Eq SST 5.9°C, surface |u| 1.15 m/s, w≈10 µm/s |
| B | 2e4 | 0 | Eq SST 6.5°C, surface |u| 1.08 m/s, w≈17 µm/s |

Both blew up dynamically at the equator. Top 5 levels homogenized
to ~6°C — runaway upwelling cold pool.

**Diagnosis** — NOT a diffusion problem:
- Wind stress is the same (JRA55-do).
- With low A_h, equatorial response is unconstrained because f→0 leaves
  no rotational stiffness; only viscosity can damp eq currents.
- Strong currents → strong vertical pumping → cold deep water reaches
  the surface mixed layer.
- `cos²(lat)` scaling makes effective A_h *largest* at the equator
  (=user value), so reducing A_h hits the eq band hardest. Wrong sign
  of latitude dependence for this problem — the eq band needs *more*
  viscosity at coarse resolution, not less.
- K_h=0 was a red herring; even with K_h=1e3 the dynamic blowup would
  have happened.

Runs killed at day 524 to free GPUs for redesigned sweep.

### Next sweep design (TODO)

Need latitude-dependent A_h that *boosts* viscosity within ±5° and lets
midlatitudes use lower A_h for realistic gyres. Options:

1. **Eq-band A_h boost**: keep `A_h=1e5` baseline, multiply by 5× within
   ±5° (linearly tapered to 1× by ±15°). Real OGCM practice (MOM6 OM4,
   NEMO ORCA1 use latitude-dependent A_h profiles).
2. **Reverse cos²(lat) scaling near eq**: change `laplacian_scaling_factor`
   to add an eq enhancement term, e.g. `cos²(lat) + α·exp(−(lat/5°)²)`.
3. **Heavier Smag at eq**: increase `C_smag` from 0.2 to 0.5 — Smag
   responds to deformation, but eq currents are largely zonal so
   shear is small. May not be enough.

Probably option 1 is cleanest. Implementation: add a meridional A_h
profile to `LatLonCGridOceanConfig` and apply it inside
`laplacian_scaling_factor`. Or precompute a `A_h_field(lat)` and pass
through.

## Goal

Run a forced global ocean simulation at 1° lat-lon resolution driven by
atmospheric reanalysis (ERA5 or JRA55-do), following the OMIP-2 protocol.
This validates the ocean model's physical fidelity for climate-scale
integrations and exercises the full physics stack: KPP, GM/Redi, sea ice,
bulk air-sea fluxes, realistic bathymetry, and conservation.

Target: 10-year pilot run first, then 60-year OMIP cycle if pilot succeeds.

## Current Readiness

### Physics stack — READY

| Component | Module | Status |
|-----------|--------|--------|
| Primitive equations | `ocean_pe_latlon_cgrid.py` | Production |
| Split-explicit barotropic | `barotropic_latlon_cgrid.py` | Production |
| Wright nonlinear EOS | `ocean/eos.py` | Production |
| z-star vertical coordinate | `ocean/vertical.py` | Production |
| KPP boundary layer | `ocean/physics/vertical_mixing/kpp.py` | Implemented, needs global validation |
| Richardson interior mixing | `ocean/physics/vertical_mixing/richardson.py` | Implemented |
| Convective adjustment | `ocean/physics/combined.py` → `convection` | Enhanced diffusion scheme |
| GM/Redi (Visbeck adaptive) | `ocean/physics/lateral_mixing/gm_redi.py` | Implemented + tested |
| Bulk air-sea fluxes | `coupler/bulk_flux.py`, `ocean/physics/surface_forcing/bulk_formulas.py` | COARE 3.0 + LY04 |
| Sea ice (thermo + EVP) | `ice/sea_ice.py`, `ice/dynamics.py`, `ice/rheology.py` | Functional |
| Bottom drag (quadratic) | `ocean/physics/bottom_drag/quadratic.py` | Production |
| Shortwave penetration | `ocean/physics/combined.py` → `shortwave_penetration` | Implemented |
| Horizontal viscosity | Laplacian + biharmonic + Smagorinsky + Leith | Production |
| Tracer advection | TVD / PPM / DST-3 / WENO5/7 | Production |
| Conservation fixers | `ocean_model_latlon_cgrid.py` | Production |
| Freshwater (virtual salt flux) | `ocean_model_latlon_cgrid.py` | Production |
| Restart I/O | `io/restart.py`, `io/checkpoint.py` | Functional |
| Bathymetry loading | `grids/topography.py` | ETOPO/GEBCO support |

### Infrastructure gaps — MUST FIX

| Gap | Severity | Effort | Notes |
|-----|----------|--------|-------|
| **Ocean forcing data loader** | Blocking | Medium | Need to read ERA5/JRA55 atmospheric fields (u10, v10, t2m, q2m, SW, LW, precip) and feed them to bulk formulae via `OceanSurfaceForcing`. The `forcing/external.py` framework can load Zarr/NetCDF but there's no ocean-specific adapter that maps ERA5 variables → `OceanSurfaceForcing` fields. |
| **Ocean-atmosphere coupler wiring** | Blocking | Medium | Bulk flux code exists (`bulk_formulas.py`) but needs to be called each ocean timestep with atmospheric state + SST to produce tau, q_net, freshwater. Need a `forced_ocean_driver` that reads forcing, computes fluxes, passes to `model.step()`. |
| **Calendar (noleap only)** | Blocking for real dates | Small | `forcing/time_utils.py` hardcodes 365-day. For OMIP with real dates (1958-2018), need Gregorian calendar with leap years. For a pilot with cycling forcing, noleap is acceptable if forcing is pre-processed to noleap. |
| **Diagnostic output** | Important | Medium | Monthly means accumulator exists (`diagnostics/monthly_means.py`) but not wired for ocean-specific variables (SST, SSS, MOC, MHT, barotropic streamfunction). Need ocean diagnostic output routine. |

### Nice to have — CAN DEFER

| Gap | Notes |
|-----|-------|
| River runoff routing | Runoff fields exist in `LandState` but routing to river mouths not implemented. Can apply as uniform coastal freshwater or skip for pilot. |
| Sea ice snow | Snow depth not tracked. Affects ice albedo seasonality. Acceptable for pilot. |
| Tidal mixing | No internal tide parameterization. Affects deep stratification. Can compensate with enhanced K_v at depth. |
| Overflow parameterization | No Nordic/Weddell overflow. Affects AMOC strength. Not needed for pilot. |
| Parallel NetCDF I/O | Single-rank file writes. Fine for 1° (small state). |

## Implementation Plan

### Phase 1: Forced Ocean Driver (~200 LOC)

Create `scripts/run_omip_ocean.py` — a standalone script that:

1. Creates a global 1° lat-lon grid with realistic bathymetry
2. Initializes from WOA/Levitus climatology (T/S) or rest state
3. Reads atmospheric forcing from pre-staged ERA5 Zarr/NetCDF
4. Computes bulk air-sea fluxes each timestep
5. Steps the ocean model with KPP + GM/Redi + sea ice
6. Saves monthly-mean diagnostics and periodic restarts

**Key design decision**: Rather than building a full coupler, create a
`ForcedOceanDriver` class that wraps `LatLonCGridOceanModel` and handles:
- Loading and interpolating atmospheric forcing to model timesteps
- Computing bulk fluxes from atmospheric state + ocean SST
- Passing `OceanSurfaceForcing` to `model.step()`
- Accumulating diagnostics

This bypasses the full atmosphere model and coupler complexity.

**Template**: Follow the pattern of `scripts/run_amip.py` (atmosphere
forced by prescribed SST) but in reverse (ocean forced by prescribed
atmosphere).

### Phase 2: Forcing Data Preparation (~100 LOC)

Create `scripts/prepare_omip_forcing.py` that:

1. Downloads ERA5 surface fields (or points to existing Zarr store)
2. Regrids to 1° lat-lon (if needed)
3. Saves as a single Zarr store with variables:
   - `u10`, `v10` — 10m wind components [m/s]
   - `t2m` — 2m air temperature [K]
   - `d2m` or `q2m` — 2m dewpoint or specific humidity
   - `ssrd` — surface downwelling shortwave [W/m²]
   - `strd` — surface downwelling longwave [W/m²]
   - `tp` — total precipitation [m/s or kg/m²/s]
   - `msl` — mean sea level pressure [Pa]
4. Time coordinates: 6-hourly, noleap calendar

Alternatively, use existing ERA5 infrastructure in `training/era5_to_state.py`
which already handles ERA5 Zarr loading with local caching.

### Phase 3: Initialization (~100 LOC)

Create `src/legoesm/ocean/init_omip.py` with:

1. `create_omip_state()` — global 1° state with:
   - Realistic bathymetry from ETOPO (via `grids/topography.py`)
   - T/S from WOA18 annual-mean climatology (load from NetCDF)
   - Zero velocity, zero eta (cold start, or warm start from restart)
   - Land mask derived from bathymetry (H > 10m = ocean)
   - Proper treatment of enclosed seas, straits

2. `create_omip_physics()` — returns `OceanPhysicsConfig` with:
   ```python
   OceanPhysicsConfig(
       vertical_mixing=VerticalMixingConfig(scheme="kpp"),
       lateral_mixing=LateralMixingConfig(
           scheme="gm_redi",
           gm_redi=GMRediConfig(kappa_GM=1e3, visbeck=VisbeckConfig(enabled=True)),
       ),
       surface_forcing=SurfaceForcingConfig(scheme="bulk_formulas"),
       bottom_drag=BottomDragConfig(scheme="quadratic", C_D=1.5e-3),
       convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
       shortwave_penetration=ShortwavePenetrationConfig(),
   )
   ```

3. `create_omip_config()` — returns `LatLonCGridOceanConfig` with:
   ```python
   LatLonCGridOceanConfig(
       A_h=0.0,                    # GM/Redi handles lateral mixing
       B_h=2e10,                   # Biharmonic for grid noise
       K_h=0.0,                    # GM/Redi handles tracer diffusion
       A_v=1e-4,                   # Background (KPP overrides in BL)
       K_v=1e-5,                   # Background (KPP overrides in BL)
       bottom_drag_r=0.0,          # Quadratic via physics pipeline
       n_barotropic_substeps=60,   # CFL: sqrt(gH)/dx ~ 200/110000
       tracer_advection="ppm",     # 4th-order, monotone
       eos="wright",               # Nonlinear
       physics=create_omip_physics(),
   )
   ```

### Phase 4: Ocean Diagnostics (~150 LOC)

Add ocean-specific diagnostics to the driver:

1. **Monthly means**: SST, SSS, SSH, barotropic streamfunction, MOC
2. **Global integrals**: Total heat content, salt content, volume
3. **Transects**: Drake Passage transport, AMOC at 26.5N
4. **Output**: NetCDF via xarray, one file per year

### Phase 5: Validation

Compare 10-year pilot against:

1. **WOA18 climatology**: T/S drift from initial conditions
2. **AVISO SSH**: Mean dynamic topography pattern
3. **Drake Passage transport**: ~130-170 Sv (observed)
4. **AMOC at 26.5N**: ~15-20 Sv (RAPID array)
5. **Global mean SST**: should track ERA5 forcing SST
6. **Sea ice extent**: Arctic September minimum, Antarctic max

## Grid and Timestep

| Parameter | Value |
|-----------|-------|
| Grid | 1° lat-lon, ~180x360 |
| Vertical | 50 levels, surface-refined (10m at top, 200m at bottom) |
| H_max | 5500 m |
| dt (baroclinic) | 3600 s (1 hour) |
| dt_baro | 120 s (n_sub=30) |
| Barotropic CFL | sqrt(9.81*5500)*120/110000 = 0.25 ✓ |

## Computational Cost Estimate

At 1° (180x360x50), per timestep:
- State arrays: ~130 MB (float64)
- KPP column physics: O(n_lat * n_lon * nlev) — cheap
- GM/Redi: O(n_lat * n_lon * nlev) — moderate (slope calculation)
- Barotropic substeps: 30 × O(n_lat * n_lon) — cheap
- Total per step: ~0.5s on GPU (estimate from scaling existing benchmarks)

10-year run: 10 * 365 * 24 = 87,600 steps × 0.5s ≈ 12 hours on GPU.
60-year OMIP: ~3 days on GPU.

## Forcing Data Requirements

For ERA5 at 0.25° regridded to 1°, 6-hourly, 10 years:
- 8 variables × 4 times/day × 365 days × 10 years × 180×360 × 4 bytes
- ≈ 27 GB (manageable, fits in memory with lazy loading)

## Priority Order

1. **Phase 1** (forced ocean driver) — the critical path
2. **Phase 3** (initialization with real bathymetry + WOA T/S)
3. **Phase 2** (forcing data prep — can use existing ERA5 infra)
4. **Phase 4** (diagnostics — can start with minimal and add later)
5. **Phase 5** (validation — after first successful run)

## Dependencies on Other Work

- **Calendar fix**: Not strictly needed if forcing is pre-processed to
  noleap and we don't need real dates. For a cycling pilot (repeat
  year 2000 forcing), noleap is fine.
- **Issue #160**: Not needed for 1° (eddies unresolved, vorticity flux
  is small). The perturbation-velocity formulation is acceptable at
  non-eddying resolution.
- **WENO advection**: Not useful at 1° (flow is smooth, see discussion
  in WENO plan). Use TVD or PPM for tracers, vector_invariant for
  momentum.

## Key Files

- `scripts/run_amip.py` — template for a forced model driver
- `src/legoesm/forcing/external.py` — Zarr/NetCDF forcing loader
- `src/legoesm/training/era5_to_state.py` — ERA5 data loading patterns
- `src/legoesm/ocean/physics/combined.py` — OceanPhysicsConfig + make_ocean_physics
- `src/legoesm/coupler/bulk_flux.py` — bulk air-sea flux computation
- `src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py` — COARE/LY04
- `src/legoesm/ocean/physics/vertical_mixing/kpp.py` — KPP
- `src/legoesm/ocean/physics/lateral_mixing/gm_redi.py` — GM/Redi
- `src/legoesm/grids/topography.py` — bathymetry loading
- `src/legoesm/ice/sea_ice.py` — sea ice model
- `src/legoesm/io/restart.py` — checkpoint I/O
