# legoESM — Implementation Summary

*Last updated: 2026-03-23*

## Overview

legoESM is a fully differentiable Earth System Model implemented in JAX, comprising atmosphere, ocean, land surface, cryosphere, and coupler components. The model supports 24 atmospheric dynamical cores across 8 discretizations, 25+ physics parameterizations, multi-layer land/ocean, sea ice with EVP dynamics, and full CMIP infrastructure.

---

## 1. Atmosphere Dynamics

### Dynamical Cores (24 total)

| Equations | Discretization | Grid | File |
|---|---|---|---|
| Shallow Water | Centered | Cubed-sphere | `shallow_water.py` |
| Shallow Water | FV/PPM | Cubed-sphere | `shallow_water_fv.py` |
| Shallow Water | FV/PPM | Lat-lon | `shallow_water_fv_latlon.py` |
| Shallow Water | FC-Gram | Cubed-sphere | `shallow_water_fc.py` |
| Shallow Water | FC-Gram C-grid | Cubed-sphere | `shallow_water_fc_cgrid.py` |
| Shallow Water | C-grid | Lat-lon | `shallow_water_cgrid_latlon.py` |
| Shallow Water | Spectral | Gaussian | `spectral_sw.py` |
| Shallow Water | SFNO (learned) | Cubed-sphere | `sfno_sw.py` |
| Hydrostatic PE | Centered | Cubed-sphere | `primitive_eq.py` |
| Hydrostatic PE | FV/PPM | Cubed-sphere | `primitive_eq_fv.py` |
| Hydrostatic PE | Centered | Lat-lon | `primitive_eq_latlon.py` |
| Hydrostatic PE | FV/PPM | Lat-lon | `primitive_eq_fv_latlon.py` |
| Hydrostatic PE | FC-Gram | Cubed-sphere | `primitive_eq_fc.py` |
| Hydrostatic PE | FC-Gram C-grid | Cubed-sphere | `primitive_eq_fc_cgrid.py` |
| Hydrostatic PE | Spectral | Gaussian | `spectral_pe.py` |
| Hydrostatic PE | SFNO (learned) | Cubed-sphere | `sfno_pe.py` |
| Non-hydrostatic CE | Centered | Cubed-sphere | `compressible_euler.py` |
| Non-hydrostatic CE | FV/PPM | Cubed-sphere | `compressible_euler_fv.py` |
| Non-hydrostatic CE | FV/PPM | Lat-lon | `compressible_euler_fv_latlon.py` |
| Non-hydrostatic CE | FC-Gram | Cubed-sphere | `compressible_euler_fc.py` |
| Non-hydrostatic CE | FC-Gram C-grid | Cubed-sphere | `compressible_euler_fc_cgrid.py` |
| Non-hydrostatic CE | C-grid | Cubed-sphere | `compressible_euler_cgrid.py` |
| Non-hydrostatic CE | Spectral | Gaussian | `spectral_nh.py` |
| MPAS SW | Voronoi | Unstructured | (via MPAS state) |

### Vertical Coordinates
- **Sigma coordinate** (`SigmaCoordinate`): Standard terrain-following
- **Hybrid sigma-pressure** (`HybridSigmaPressureCoordinate`): Factory functions `make_hybrid_levels()`, `standard_hybrid_levels()`, `hybrid_from_sigma()`
- **Sinh stretching**: Boundary-layer concentration via `stretching` parameter
- **Resolution**: L20 (p_top=1000Pa), L40 (p_top=200Pa, default), L60 (p_top=10Pa)

### Grid Infrastructure
- **Cubed-sphere**: 6-face gnomonic projection, C16-C64 tested
- **Lat-lon**: Regular grid with polar filter
- **Gaussian**: Gauss-Legendre quadrature + spherical harmonic transforms
- **Voronoi/MPAS**: Unstructured dual mesh (triangles + hexagons)
- **Halo exchange**: Cubed-sphere face-to-face, lat-lon periodic, Voronoi entity-based
- **Topography**: Real NetCDF loading, bilinear regridding, Laplacian smoothing, land fraction derivation

---

## 2. Atmosphere Physics (25+ schemes)

### Radiation (2 backends)
- **Gray two-stream** (`gray.py`): Held-Suarez-like OLR
- **RRTMGP** (`rrtmgp_radiation.py`): Full correlated-k gas + cloud optics
- **Diurnal cycle**: Optional cos(SZA) insolation with hour-angle and declination
- **Ozone**: Standard (US Std Atm 1976), analytical (Gaussian in log-p), or none
- **Cloud-radiation coupling**: Cloud fraction → LWP/IWP → cloud optics → RRTMGP

### Cloud Fraction (2 schemes)
- **Sundqvist**: RH-only threshold scheme
- **Xu-Randall**: RH + explicit condensate, pressure-dependent exponent
- Diagnostic condensate partitioned by temperature when no microphysics active

### Convection (5 backends)
- SBM (Simplified Betts-Miller), DCA, Kuo, mass flux, EDMF

### Microphysics (6 backends)
- Kessler, Sundqvist, Seifert-Beheng, Morrison, Thompson, ML emulator
- Operator-split integration in AMIP: called after convection, before BL exchange
- q_c/q_r managed as parallel arrays; saturation adjustment disabled when active

### Turbulence (8 backends)
- Smagorinsky, Louis, TKE, CLUBB-lite, Holtslag-Boville, YSU, EDMF, ML emulator
- **PBL height diagnosis** (`pbl_height.py`): Bulk Richardson number, sigmoid-weighted or linear interpolation to Ri_crit crossing
- All backends return `h_pbl` in `TurbulenceOutput`

### Gravity Wave Drag (6 backends)
- Rayleigh, Lindzen, McFarlane, Hines, prognostic spectral, ML emulator

### Combined Physics
- `combined.py`: Sums tendencies from all active physics modules
- Factory pattern: `make_*_physics()` returns `physics_fn(state, grid, coords)`
- Time-dependent: `set_time(day_of_year, seconds_of_day)` method on closures

---

## 3. Ocean

### Dynamics
- **3D split-explicit** (`ocean_model.py`): Baroclinic PE + barotropic substeps
- **Spectral ocean** (`spectral_ocean_pe.py`): Spherical harmonic ocean
- **SFNO ocean** (`sfno_ocean.py`): Learned ocean dynamics
- **FC-Gram ocean** (`ocean_pe_fc_cgrid.py`): High-order ocean PE
- **Simple ocean**: Slab mixed-layer (`simple_ocean.py`) and two-layer

### Physics
- Vertical mixing: constant, Richardson, KPP
- Lateral mixing: harmonic, biharmonic, GM-Redi
- Surface forcing: prescribed, restoring, bulk formulas
- Bottom drag: linear, quadratic
- Convection: enhanced diffusion, plume

### Biogeochemistry
- **Abiotic** (`carbon_cycle.py`): DIC + ALK, carbonate equilibria, air-sea CO2 flux (Wanninkhof 2014)
- **NPZD** (`npzd.py`): Nutrient-Phytoplankton-Zooplankton-Detritus ecosystem + Redfield coupling
- EOS: Wright (1997), z-star vertical coordinate

---

## 4. Land Surface

### Slab Land (`slab_land.py`)
- Energy balance: R_net = SH + LH + G
- Bucket hydrology: precipitation, evaporation, runoff
- Stomatal conductance: Farquhar photosynthesis + Ball-Berry/Medlyn/Jarvis

### Multi-Layer Land (`multilayer_land.py`)
- **Richards equation** (`richards.py`): Mixed-form Picard iteration (Celia 1990), Thomas algorithm, surface/subsurface runoff
- **6 retention curves** (`soil_hydraulics.py`): van Genuchten, Clapp-Hornberger, Brooks-Corey, Campbell, Peters-Durner-Iden (2015), Lu (2016)
- **Soil thermal** (`soil_thermal.py`): Johansen (1975) conductivity, backward Euler diffusion
- **Soil grid** (`soil_grid.py`): Geometric growth factor, configurable n_layers/dz_top/total_depth

### Carbon Cycle (`carbon/`)
- **DALEC-990**: 6-pool model (labile/foliage/root/wood/litter/SOM), LUE-based GPP, Q10 decomposition, DALEC phenology
- **Seasonal**: Sinusoidal NEE with latitude-dependent amplitude/phase
- Integrated into both slab and multi-layer land (returns 3-tuple: state, response, carbon_state)

### Snow & Albedo (`surface_albedo.py`)
- Snow accumulation from precipitation, melt above T_melt
- Age-dependent snow albedo decay
- Latitude-varying vegetation albedo (tropics/midlat/highlat)
- All feedback optional and disabled by default

---

## 5. Cryosphere

### Sea Ice (`sea_ice.py`)
- **Thermodynamic slab**: Growth/melt, conductive flux
- **Free drift**: Diagnostic ice velocity
- **EVP rheology**: Hunke & Dukowicz (1997), subcycled momentum solver
- **Multi-category**: CICE framework, Lipscomb (2001) linear remapping
- **Temperature-dependent albedo**: Cold/warm transition via `IceAlbedoConfig`

---

## 6. Coupler

### Surface Coupling (`coupler.py`)
- Tile-based: ocean, ice, land, lake with area-weighted blending
- Flux accumulator with window-mean and residual carry
- q_surface recomputed from updated T in all tiles

### Bulk Flux (`bulk_flux.py`)
- COARE 3.0, Large & Yeager 2004, fixed-z0
- Monin-Obukhov surface layer theory

### Lake (`lake/`)
- Two-layer model: epilimnion + hypolimnion with wind-enhanced mixing

---

## 7. Diagnostics

### Energy Budget (`diagnostics/energy_budget.py`)
- `column_moist_static_energy()`: Integrates c_p·T + L_v·q + Phi + 0.5·v^2 via `jax.lax.scan`
- `toa_net_radiation()`, `surface_net_radiation()`, `surface_energy_flux()`
- `EnergyBudgetTracker`: Timeseries accumulation, dE/dt computation, residual R_TOA - dE/dt

### Monthly Means (`diagnostics/monthly_means.py`)
- `MonthlyAccumulator`: Bins 2D fields by latitude (zonal means), accumulates 3D profiles, tracks global-mean scalars
- 365-day calendar month mapping, save/load to npz

---

## 8. External Forcing

- **GHG**: `GHGConfig` with CO2/CH4/N2O, constant or time-varying from NetCDF
- **Ozone**: Monthly zonal-mean climatology + analytical profiles
- **Aerosol**: AOD climatology from NetCDF
- **Solar**: TSI constant or time-varying
- **CMIP6 experiments**: piControl, historical, SSP2-4.5, SSP5-8.5, AMIP, 1pctCO2 templates with built-in GHG time series
- **Topography**: Real NetCDF loading, bilinear regrid, Laplacian smoothing, edge blending

---

## 9. Parallelism

### Structured Grid Sharding (`parallel/mesh.py`)
- Cubed-sphere face sharding (1-6 devices)
- Sub-face tiling (>6 devices, multiples of 6)
- Lat-lon domain decomposition by latitude
- Level-parallel spectral transforms

### Voronoi Mesh Decomposition (`parallel/voronoi_partition.py`)
- Capability-aware `method="auto"` default: METIS graph partitioner when `pymetis` (the `[mesh]` extra) is present, else Recursive Coordinate Bisection (RCB)
- Hilbert space-filling-curve partitioner (`method="sfc"`) + within-shard Hilbert ordering in `reorder_voronoi_for_sharding()`
- 2-layer halo depth, entity ownership rules
- `VoronoiPartition`, `HaloCommSchedule` data structures
- `build_local_mesh()`, `scatter_to_local()` utilities

### Voronoi Halo Exchange (`parallel/halo_exchange_voronoi.py`)
- `VoronoiHaloExchange` class with MPI and simulated backends
- mpi4jax sendrecv with entity-type tag offsets
- Global-to-local index remapping, connectivity masking

### Ensemble Parallelism (`parallel/ensemble.py`)
- `stack_states()` / `unstack_states()`: Batch/unbatch pytree states
- `perturb_initial_conditions()` / `perturb_parameters()`: Stochastic ensemble generation
- `make_ensemble_step()`: vmap wrapper for single-device vectorization
- `ensemble_integrate()`: scan-outside-vmap-inside architecture (one compiled loop for all members)
- `ensemble_integrate_with_forcing()`: Time-varying external forcing support
- `create_ensemble_mesh()` / `shard_ensemble()` / `gather_ensemble()`: Multi-device NamedSharding
- Statistics: `ensemble_mean()`, `ensemble_std()`, `ensemble_percentile()`, `ensemble_spread()`
- Gradient checkpointing via `jax.checkpoint` for O(sqrt(n)) memory in reverse-mode AD

### MPI Distributed (`parallel/distributed.py`, `halo_exchange.py`, `reductions.py`, `layout.py`)
- `initialize_distributed(global_n=N)` → CommTopology + DeviceConfig + DistributedLayout
- `scatter_to_local()` / `gather_to_global()` for rank-local data ownership
- **Native 4D halo exchange**: `pad_halo_4d()` / `pad_halo_vector_4d()` — one MPI message per neighbor for all vertical levels
- Face-only mode uses batched neighbor `sendrecv` (not allgather)
- Sub-face tiling for >6 ranks (6 × k² decomposition)
- `_sendrecv_vjp` (`@jax.custom_vjp`): reverse-mode AD through MPI halo exchange
- `global_sum_mpi` (allreduce SUM): fully differentiable; `global_max/min_mpi`: diagnostics only
- `batch_allreduce_mpi()`: pack multiple reductions into one MPI call
- Per-rank distributed checkpoint (`io/distributed_checkpoint.py`)
- mpi4jax 0.8.x version compatibility guardrails

### Apple Silicon (`parallel/metal.py`)
- Metal/CPU hybrid routing for mixed-precision workloads

---

## 10. CMIP Infrastructure

- **CF-compliant output** (`io/cmor_output.py`): CF-1.8, CMIP6 DRS naming, 27 variables (Amon + Lmon)
- **Restart** (`io/restart.py`): SHA-256 state digests, config hashes, platform metadata
- **Tuning** (`tuning.py`): 16-parameter registry with resolution-appropriate defaults

---

## 11. AMIP Simulations

### Scripts
- `run_amip.py`: Full-featured AMIP (cubed-sphere), supports gray/RRTMGP radiation, all physics, checkpointing, energy budget tracking, monthly means
- `run_amip_spectral.py`: Spectral PE AMIP (Gaussian grid)
- `run_amip_production.sh`: Launcher for C48/L40 10-year production
- `validate_amip.py`: Post-processing validation against observational targets

### CLI Flags (run_amip.py)
`--resolution`, `--nlev`, `--days`, `--dt`, `--radiation`, `--diurnal-cycle`, `--ozone-source`, `--clouds`, `--microphysics`, `--topography`, `--vertical-coord`, `--p-top`, `--stretching`, `--dynamic-albedo`, `--monthly-means`, `--carbon-cycle`

### Validated Configurations
- 365-day AMIP at C16/L40 (cubed-sphere) and T21/L40 (spectral)
- Production target: 10-year AMIP at C48/L40

---

## 12. Test Suite

**123 test files, 2500+ individual tests** across 10 test directories. All tests pass (0 failures; 9 skipped for optional dependencies).

### Test Categories

| Category | Files | Tests | Description |
|---|---|---|---|
| Core unit tests | ~15 | ~200 | Field, grid, operators, halo, conservation, timestepping |
| Atmosphere dynamics | ~25 | ~400 | All 24 dynamical cores, FV integration, spectral |
| Atmosphere physics | ~10 | ~250 | Radiation (diurnal, ozone, clouds), convection, microphysics, turbulence (PBL), GWD |
| Ocean | ~8 | ~150 | Dynamics, biogeochem, MPI conservation, FC-Gram, FV |
| Land surface | ~10 | ~220 | Slab + multilayer soil × SimpleSEB + TwoLeafCanopy surface schemes (4 combos), carbon cycle, canopy biophysics, Newton A-gs coupling, stability |
| Sea ice / albedo | ~3 | ~80 | EVP dynamics, multi-category, surface albedo |
| Coupler / forcing | ~5 | ~100 | Bulk flux, tile coupling, AMIP config, external forcing |
| Parallelism | ~5 | ~80 | Device mesh, Voronoi partition/halo (34 tests), ensemble (32 tests) |
| Diagnostics | ~3 | ~50 | Energy budget, monthly means |
| Validation | ~8 | ~100 | Differentiability, eigenvalues, stability, held-suarez |
| CMIP infrastructure | ~3 | ~50 | CMOR output, experiments, restart |
| DCMIP test cases | ~6 | ~30 | Williamson, DCMIP-2025 cases 1-3 |

### Key Test Files (recent additions)

| File | Tests | Runtime | Description |
|---|---|---|---|
| `test_radiation.py` | 26+ | ~30s | Diurnal cycle, ozone profiles, cloud fraction, cloud-radiation |
| `test_hybrid_vertical.py` | 34 | ~10s | Hybrid coord construction, pressure, geopotential, semi-implicit |
| `test_topography.py` | 27 | ~15s | Regridding, land fraction, smoothing, NetCDF, edge blending |
| `test_turbulence.py` | 10+ | ~20s | PBL height (Ri, shape, bounds, interp, all backends) |
| `test_surface_albedo.py` | 25 | ~10s | Vegetation, snow, ice, ocean albedo |
| `test_energy_budget.py` | 25 | ~15s | Column energy, TOA/surface flux, tracker |
| `test_monthly_means.py` | 21 | ~10s | Day-to-month, 2D/3D/scalar accumulation, save/load |
| `test_multilayer_land.py` | 49 | ~40s | Grid, 6 retention curves, Richards, thermal, integrated |
| `test_carbon_cycle.py` | 43 | ~30s | DALEC-990, GPP, seasonal, coupler integration |
| `test_microphysics.py` | 8 | ~15s | Checkpoint with hydrometeors, config, multi-step stability |
| `test_voronoi_halo.py` | 34 | ~36s | Partitioning (13), local mesh (4), operators (8), halo exchange (9) |
| `test_ensemble.py` | 32 | ~10s | Batching (3), perturbation (8), step (4), integrate (5), statistics (5), sharding (4) |
| `test_land_stability.py` | 32 | ~720s | Slab (6), multilayer (7), energy/water budget, snow, carbon, seasonal |

### Running Tests

```bash
# All tests (requires float64)
JAX_ENABLE_X64=1 pytest tests/

# Specific component
JAX_ENABLE_X64=1 pytest tests/land/
JAX_ENABLE_X64=1 pytest tests/unit/test_ensemble.py

# MPI distributed tests
mpirun -np 6 python -m pytest tests/distributed/test_halo_mpi.py
```

---

## 13. Implementation Roadmap (Completed)

All 12 tasks from `NEXT_STEPS.md` are complete:

1. **Diurnal Cycle** — cos(SZA) insolation, hour-angle, `set_time()` mechanism
2. **Prescribed Ozone** — Standard/analytical profiles, RRTMGP integration
3. **Cloud Fraction + Cloud-Radiation** — Sundqvist/Xu-Randall, diagnostic condensate, RRTMGP cloud optics
4. **Hybrid Sigma-Pressure Coordinate** — Full vertical coordinate with semi-implicit support
5. **Vertical Resolution L40-L60** — Sinh stretching, `standard_hybrid_levels()`
6. **Real Topography** — NetCDF loading, bilinear regrid, Laplacian smoothing
7. **Microphysics in AMIP** — Operator-split, q_c/q_r parallel arrays, checkpoint extension
8. **Multi-Layer Soil** — Richards equation (6 retention curves), Johansen thermal
9. **PBL Height Diagnosis** — Bulk Richardson, all 8 backends updated
10. **Surface Albedo** — Snow age, vegetation latitude, ice temperature, ocean zenith
11. **Energy Budget Closure** — Column MSE, TOA/surface flux, budget tracker
12. **10-Year AMIP at C48/L40** — Monthly means, validation script, production launcher

Additional completed features beyond the roadmap:
- **Land Carbon Cycle** — DALEC-990 6-pool + seasonal, integrated into slab/multilayer
- **Voronoi Mesh Decomposition** — RCB + METIS partitioning, halo exchange
- **Ensemble Parallelism** — vmap + sharding + scan architecture
- **Land Stability Validation** — 32 comprehensive tests (energy/water budget, seasonal behavior)
