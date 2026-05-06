# Plan: OMIP-Style Forced Global Ocean Run at 1°

## Recent Status (2026-05-06)

### Current state: searching for the right A_h

The central problem is finding horizontal viscosity (A_h) that is:
- **Low enough** at midlatitudes for realistic gyres, WBCs, and ACC
- **High enough** everywhere to prevent grid-scale noise blowup at 1°
- **Boosted at the equator** where f→0 removes rotational stiffness

We have the equatorial boost mechanism implemented and working. What
remains is finding the right A_h baseline value.

### Stable baseline — A_h=2e5 (over-damped)

10-year stable run completed (88 min, 2× V100): `results/jra55_10yr_production/`

**Stability stack**: basin-removal flood-fill + narrow-passage fill (min_width=2)
+ 3D maxvel clip (3 m/s) + slope-foot viscosity (alpha=3) + Smag (C=0.2)
+ SSS restoring + freeze cap + open Southern Ocean (south-cap-lat=−90).

**Production config**: `A_h=2e5, K_h=1e3, B_h=5e9, B_h_barotropic=1e14`,
`bottom_drag_r=2.5e-3, DRAG_BG_VEL=0.1`, GM/Redi (Visbeck adaptive
kappa_GM=kappa_Redi=800, range [200, 2000]).

**Results**:
- SST: 19.8 → 15.9 (yr1) → 13.9 (yr2) → 9.5°C (yr10). Too cold
  (observed global mean ~17–18°C).
- SSS: 35.0 → 33.6 (yr1) → 33.2 (yr10). Modest freshening.
- Velocity clip: **never triggered** (0/3651 days).
- Circulation: zonal jets, no closed gyres or ACC. Over-damped.
- Equatorial currents: 0.15 m/s (real SEC ~0.3–0.5 m/s, EUC ~1 m/s).

**Diagnosis**: A_h=2e5 is 10–100× higher than production OGCMs
(MOM6 OM4 uses ~600 m²/s background + Smagorinsky). The "good-looking"
equator is an accident of blanket over-damping.

### Failed sweep 1 — uniform A_h reduction (Runs A/B, 2026-05-05)

Tested with K_h=0 (GM/Redi handles tracer mixing), no equatorial boost:

| Run | A_h | K_h | Result |
|-----|-----|-----|--------|
| A | 5e4 | 0 | Killed day 524: eq SST 5.9°C, runaway cold pool |
| B | 2e4 | 0 | Killed day 524: eq SST 6.5°C, runaway cold pool |

**Diagnosis** — dynamic blowup, not a diffusion problem:
- With low A_h, the equatorial band (f→0) has no damping mechanism.
- Wind-driven Ekman divergence → unconstrained upwelling → cold deep
  water homogenizes the top 5 levels to ~6°C.
- `cos²(lat)` scaling makes effective A_h *largest* at the equator
  (= user-set value), so reducing A_h hits the eq band hardest — the
  wrong sign of latitude dependence.
- K_h=0 was a red herring; the failure is purely momentum dynamics.

### Failed sweep 2 — equatorial boost (Runs C/D, 2026-05-05)

Implemented `equatorial_boost_factor()`: Gaussian enhancement
`1 + (boost-1) * exp(-(lat/sigma)²)` applied multiplicatively on top
of cos²(lat) CFL scaling. Config: `A_h_eq_boost`, `A_h_eq_sigma_deg`.

| Run | A_h | eq_boost | sigma | Eff. eq A_h | Eff. midlat (45°) |
|-----|-----|----------|-------|-------------|-------------------|
| C | 5e4 | 5× | 5° | 2.5e5 | 2.5e4 |
| D | 3e4 | 7× | 5° | 2.1e5 | 1.5e4 |

Both ran 5 years to completion. Results: `results/sweep_C_eqboost5/`,
`results/sweep_D_eqboost7/`.

**Results**:
- SST: 19.8 → 10.4 (yr1) → 7.5 (yr2) → **3.8°C** (yr5). Catastrophic.
- SSS: 35.0 → 30.0 (yr1) → **24.7** (yr5). Massive freshening.
- Velocity clip: **triggered every single day** (1825/1826 days).
  Baseline never triggered it (0/3651 days).
- Both runs nearly identical despite different eq boost → problem is
  NOT the equator (boost is working), it's the midlatitudes.
- Max speed location: lat ≈ −78.5° (Southern Ocean near Antarctica).

**Diagnosis**:
- Midlatitude A_h = 2.5–5e4 is below the stability threshold for 1°.
- Grid-scale noise blows up immediately at high latitudes where:
  (a) dx shrinks as cos(lat), increasing effective grid Reynolds number;
  (b) Southern Ocean topography forces strong shelf-break flows;
  (c) cos²(lat) further reduces already-low A_h near poles.
- The 3D velocity clip (3 m/s) catches the blowup but destroys KE
  artificially, driving anomalous vertical mixing and global cooling.
- The equatorial boost itself works — no localized eq cold pool like
  Runs A/B. The failure is entirely at midlatitudes/poles.

### Plots from sweeps

- `results/sweep_snapshots.png` — SST, SSS, SSH, speed maps (Baseline vs C vs D)
- `results/sweep_timeseries.png` — Time series comparison
- `results/sweep_equatorial_sections.png` — Lat-depth T sections

### What to try next

The stable range is A_h ∈ [~1e5, 2e5]. The baseline (2e5) is stable
but over-damped. Runs C/D (5e4, 3e4) blow up. The sweet spot is
probably 1.0–1.5e5 with a modest equatorial boost.

**Recommended next sweep**:

| Config | A_h | eq_boost | Eff. eq | Eff. midlat (45°) | vs baseline |
|--------|-----|----------|---------|-------------------|-------------|
| E | 1.2e5 | 3× | 3.6e5 | 6.0e4 | 0.6× midlat |
| F | 1.5e5 | 2× | 3.0e5 | 7.5e4 | 0.75× midlat |

Alternative/complementary approaches:
1. **Increase Smagorinsky** (C_smag 0.2 → 0.5): adaptive damping where
   shear is strong (Southern Ocean, WBCs) without blanket over-damping.
2. **Increase biharmonic B_h**: targets grid-scale noise selectively
   without damping large-scale flow. Currently B_h=5e9; could try 1e10.
3. **Anisotropic viscosity**: high meridional, low zonal near equator
   (POP-style). More complex to implement.

**Key insight**: the 3D maxvel clip is a canary — if it triggers at all,
the run is probably in trouble. A good configuration should never need it.

### Implementation details

The equatorial boost is fully implemented and committed:
- `equatorial_boost_factor()` in `latlon_cgrid_operators.py`
- Config fields `A_h_eq_boost`, `A_h_eq_sigma_deg` in `LatLonCGridOceanConfig`
- Wired into both Laplacian application sites in `ocean_pe_latlon_cgrid.py`
- CLI args `--A-h-eq-boost`, `--A-h-eq-sigma` in `scripts/run_omip.py`

### How to launch a sweep

Pin each run to a separate GPU on the 2× V100S machine:

```bash
# Run E — conservative reduction
CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_omip.py \
  --days 1825 --A-h 1.2e5 --K-h 0 --B-h 5e9 \
  --A-h-eq-boost 3.0 --A-h-eq-sigma 5.0 \
  --C-smag 0.2 --slope-foot-alpha 3.0 \
  --min-passage-width 2 --remove-enclosed-basins \
  --south-cap-lat -90 --maxvel-3d 3.0 \
  --result-dir results/sweep_E_Ah1.2e5_boost3 \
  > results/sweep_E.log 2>&1 &

# Run F — moderate reduction
CUDA_VISIBLE_DEVICES=1 nohup python scripts/run_omip.py \
  --days 1825 --A-h 1.5e5 --K-h 0 --B-h 5e9 \
  --A-h-eq-boost 2.0 --A-h-eq-sigma 5.0 \
  --C-smag 0.2 --slope-foot-alpha 3.0 \
  --min-passage-width 2 --remove-enclosed-basins \
  --south-cap-lat -90 --maxvel-3d 3.0 \
  --result-dir results/sweep_F_Ah1.5e5_boost2 \
  > results/sweep_F.log 2>&1 &
```

Each run takes ~3.5 hours for 5 years on a single V100S.

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
