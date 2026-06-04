# MPAS Ocean with JRA55-do Forcing — Experiment Plan

## Goal

Test whether the MPAS Voronoi ocean model (proven stable for 20-year
idealized runs in PR #261) can run under realistic JRA55-do atmospheric
forcing with ETOPO bathymetry for 100 years.  This is the first step
toward an MPAS-based OMIP (Ocean Model Intercomparison Project) configuration.

## Motivation

PR #261 demonstrated that the MPAS ocean (ico5, ~1° / 120 km) is stable
for 20 years with idealized forcing (prescribed zonal wind + SST/SSS
Haney restoring).  The natural next question: does it survive under
realistic, spatially and temporally varying atmospheric forcing?  The
JRA55-do repeat-year forcing (RYF9091, Stewart et al. 2020) is the
OMIP-2 standard — if the model is stable under it, we have a viable
path to OMIP compliance.

## Branch

`feature/mpas-omip-jra55`, based on `feature/mpas_topo` (PR #261).

## What Was Built

### 1. Lat-lon → MPAS Regridding (`src/legoesm/grids/regridding.py`)

The JRA55 cache is on a regular 1° lat-lon grid.  MPAS uses unstructured
Voronoi cells.  Added `compute_latlon_to_voronoi_weights()` — KD-tree in
Cartesian coordinates + inverse-distance weighting (k=4 neighbors).
Pre-computed once at setup (~0.5s), applied per forcing field.

Tested: max error 3.8e-5 on a cos(lat) field at 1° resolution.

### 2. JRA55 Slice Regridding (`src/legoesm/forcing/jra55_do.py`)

`regrid_jra55_slice()` regrids all 10 JRA55 fields from the lat-lon
cache to MPAS cell centres using precomputed weights.  No vector
rotation needed — uas/vas are east-north components on the sphere.

### 3. External Forcing in MPAS Physics (`src/legoesm/ocean/physics/mpas_physics.py`)

Added an external tau/q_net application block inside the MPAS physics
pipeline.  When `surface_forcing` carries `tau_x`, `tau_y`, `q_net`
fields (from the JRA55 bulk-flux solver), projects cell-centred wind
stress onto edge normals and applies heat flux to the surface layer.
Uses the same edge-projection pattern as the existing prescribed-wind
block.

### 4. MPAS Production Config in `run_omip.py`

Replaced the minimal MPAS config with the proven PR #261 setup:

| Parameter | Value | Source |
|-----------|-------|--------|
| A_h (Laplacian viscosity) | 1e4 m²/s | PR #261 |
| C_smag_lap (Smagorinsky) | 0.33 | PR #261 |
| K_zeta_bih (vorticity damping) | 1e14 m⁴/s | PR #261 |
| Barotropic solver | implicit-CN (PCG, tol=1e-10) | PR #261 |
| PGF scheme | Adcroft | PR #261 |
| Tracer advection | TVD (Van Leer) | PR #261 |
| Vertical mixing | KPP (K_conv=1.0) | PR #261 |
| Convection | Enhanced diffusion (K_conv=1.0) | PR #261 |
| GM/Redi | κ_GM=600, κ_Redi=600, S_max=0.005 | PR #261 |
| Bottom drag | r=1e-3, BBL=100m, u_bg=0.1 | PR #261 |
| Implicit vertical mixing | True | PR #261 |
| dt | 1200 s | PR #261 |

### 5. lax.scan Block Path for MPAS

Added `_step_impl()` (non-JIT inner method) to `MPASOceanModel` so the
model step can be called inside `lax.scan` without nested JIT.  The
GPU-interp scan-block path pre-loads ~9 native 3-hourly JRA55 records
per block, regrids them to MPAS cells on the host, then runs N timesteps
in one JIT-compiled `lax.scan` with time interpolation + solar zenith +
bulk fluxes + ocean step all fused into one XLA graph.

**Speedup: 16x** (0.8s/day vs 38s/day single-step).

### 6. ETOPO Bathymetry for MPAS

Full PR #261 recipe wired into `run_omip.py`:
`load_bathymetry_mpas` → optional north cap → 30% partial cell snapping
→ `create_partial_cell_coordinate` → model rebuild.

### 7. Auto-Snapshot Generation

Tripcolor + cartopy Robinson projection snapshots (`scripts/plot/plot_mpas_omip_snapshot.py`)
are auto-generated alongside each 30-day restart checkpoint.

Layout (2×3): Surface speed | SSH | SST | Zonal-mean T(lat,z) | SSS | Deep T

## Bugs Found and Fixed

### 1. Conservative Regrid Longitude Wrap (critical)

**Symptom**: Cold north-south streak at the Greenwich Meridian in SST.

**Root cause**: The JRA55 TL319 source grid (640 lon points) has its
last centre at 359.4375°.  `_grid_edges_from_centers` inferred edges
up to 359.72° — a 0.28° gap before 360°.  The conservative regridder
has no periodic wrap, so the last target column (359°–360°) was missing
28% of its source coverage, producing systematically low values in all
10 forcing fields.

**Fix**: Append a ghost column (first source column at +360°) to close
the gap before computing regrid weights.  File: `jra55_do.py`,
function `build_jra55_cache`.

### 2. North Pole Land Cap

**Symptom**: No ocean cells above 80°N despite ETOPO showing deep
Arctic basin.

**Root cause**: Three independent 80° caps stacked:
1. `rest_state_mpas_ocean` default `land_lat_threshold=80`
2. MPAS ETOPO post-processing block hardcoded `north_cap_lat=80`
3. CLI `--north-cap-lat` default was 80

**Fix**: Set all three to 90° (no cap) when ETOPO bathymetry is active.
The actual coastlines come from the bathymetry data.

### 3. Return Value Mismatch in `_run_omip_loop`

The JRA55 single-step and scan-block paths returned 4 values but the
caller expected 5 (with `blowup_info`).  Fixed by adding `blowup_info`
to all return paths.

### 4. `_create_setup` Missing `forcing_mode` Parameter

The MPAS config branch needed to know whether JRA55 or restoring mode
was selected (different physics config), but `_create_setup` didn't
receive `args`.  Added `forcing_mode` parameter.

## Current Run (v4)

| Field | Value |
|-------|-------|
| PID | 2449703 |
| GPU | 1 (V100S 32GB) |
| Grid | ico5 (10242 cells, 7376 ocean) |
| Levels | 20 (dz_sfc=20m, dz_deep=500m, H_max=5500m) |
| Forcing | JRA55-do RYF9091, --jra55-cycle |
| Bathymetry | ETOPO 1°, no north cap, 30% snap, 2 smoothing passes |
| Initial condition | Rest state (exponential T profile, uniform S=35) |
| Closures | No sponge, no SSS restoring, no freeze cap |
| JRA55 cache | `data/jra55_ryf_cache/jra55_do_v14_omip2_1deg_noleap_fixed.zarr` |
| Output | `results/mpas_jra55_etopo_100yr_v4/mpas/ico5/` |
| Log | `results/mpas_jra55_etopo_100yr_v4.log` |
| Speed | ~0.7 s/day (scan-block path) |
| ETA | ~8 hours from launch (~5 PM 2026-05-15) |

### Year 1 Diagnostics (from validated 1-year run)

| Day | SST (°C) | SSS (PSU) | SSH (m) | max |u| (m/s) |
|-----|----------|-----------|---------|----------------|
| 1   | 19.62    | 35.00     | -0.0003 | 0.39           |
| 45  | 19.47    | 34.98     | -0.015  | 0.85           |
| 91  | 19.67    | 34.97     | -0.036  | 0.80           |
| 140 | 19.68    | 34.96     | -0.059  | 1.05           |
| 189 | 19.51    | 34.96     | -0.085  | 1.16           |
| 236 | 19.47    | 34.95     | -0.106  | 0.96           |
| 365 | 19.35    | 34.96     | -0.179  | 0.78           |

Clear seasonal SST cycle driven by the JRA55 repeat-year forcing.
SSS drift is 0.04 PSU/year (no SSS restoring active).

## What's Not Included

- **Sponge layers**: Disabled (no polar damping).  May be needed if
  high-latitude instabilities develop over decades.
- **SSS restoring**: Disabled.  Standard OMIP practice is weak global
  SSS restoring (piston velocity ~5e-7 m/s) to prevent freshwater
  drift.  Should be enabled for production runs.
- **Freeze cap / sea ice**: See critical note below.
- **WOA initialization**: Starting from rest state with idealized T(z)
  profile, not WOA18 climatology.  WOA init available via `--woa-init`.
- **Equatorial bathymetry smoothing**: The lat-lon-specific equatorial
  smoothing in `run_omip.py` doesn't apply to MPAS.  The 20-year PR #261
  ETOPO run was stable without it, but 100-year stability is TBD.

## Critical: Sea Ice and the Freeze Cap

**OMIP is defined as ocean/sea-ice** (Griffies et al. 2016, Section 2).
Sea ice is mandatory for protocol compliance.  Running without it
produces a forced-ocean validation experiment, not an OMIP submission.

### Why it matters physically

Without sea ice, polar regions are catastrophically affected:

1. **Enormous heat loss**: Bulk-flux solver sees SST ~ -1.8°C against
   air at -30 to -50°C → 500-1000 W/m² heat loss.  In reality, sea
   ice insulates the ocean (heat flux through ice is limited by its
   thermal resistance).

2. **Sub-freezing SST**: Without frazil ice formation to absorb latent
   heat, SST drops below -1.8°C → unphysical densities in the Wright
   EOS (never calibrated below freezing).

3. **Catastrophic deep convection**: Unrealistically cold, dense surface
   water triggers deep convection everywhere in the polar ocean
   simultaneously (in reality, deep water formation is localized and
   driven by brine rejection from ice formation).

4. **Broken freshwater budget**: Sea ice is the polar freshwater pump
   (brine rejection + melt cycle).  Without it, the Arctic halocline
   disappears and AMOC source water formation is wrong.

### The freeze cap: minimal band-aid

`_apply_freeze_cap` clamps SST ≥ -1.8°C (seawater freezing point).

**What it does**: Prevents sub-freezing SST, avoids unphysical
densities and model blowup.

**What it does NOT do**: No brine rejection, no freshwater from melt,
no momentum damping (ice reduces wind stress on ocean), no albedo
change, and the heat removed by clamping is NOT conserved (it
vanishes silently).

### Decision for current runs

All current runs have `--jra55-no-freeze-cap` (freeze cap DISABLED).
This is a known limitation:

- **Tropical/mid-latitude results are meaningful** — the bulk-flux
  forcing, SSS restoring, and WOA nudging all work correctly there.
- **Polar results are unphysical** — SST can go below freezing,
  deep convection will be too vigorous, AMOC source water is wrong.
- **Global-mean diagnostics** (mean SST, total KE) are contaminated
  by polar artifacts.

**Recommendation for production**: Enable the freeze cap at minimum
(`remove --jra55-no-freeze-cap`).  It is not physically correct but
prevents the worst artifacts.  For OMIP compliance, the full sea-ice
model (EVP dynamics + thermodynamics, already in `src/legoesm/ice/`)
must be coupled — this is Phase B.

## Next Steps

### Immediate: WOA18 Initialization (next run)

The current v4 run starts from an idealized rest state (exponential T
profile, uniform S=35 PSU).  The next iteration will use WOA18
climatological T/S as the initial condition, which is the standard
OMIP spin-up approach.

**What exists already:**
- `init_ocean_from_woa(grid, z_coord, T_path, S_path)` in
  `src/legoesm/ocean/init_woa.py` already supports VoronoiMesh
  (line 245-248: dispatches on `hasattr(grid, 'latCell')`).
- Performs nearest-neighbor horizontal interpolation from WOA18 1°
  to MPAS cell centres, then vertical interpolation to model levels.
- NaN fill with 1.5°C / 34.5 PSU for cells outside WOA coverage.
- WOA18 data available at `data/woa18/woa18_decav_t00_01.nc` and
  `data/woa18/woa18_decav_s00_01.nc`.
- `run_omip.py` already calls `init_ocean_from_woa` at line ~2514
  and applies the WOA T/S when `--woa-init` is passed (line ~2909).
- The WOA masking uses `state.land_mask.data` which is correct
  because the MPAS ETOPO block sets the real land mask *before*
  the WOA init block runs.

**What needs to change:** Nothing in the code — it's already wired.
Just add the CLI flags to the launch command:

```bash
CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 nohup .venv/bin/python scripts/run_omip.py \
  --grid mpas --resolution ico5 --nlev 20 --days 36500 --dt 1200 \
  --H-max 5500 --H-min 10 \
  --bathymetry data/bathymetry/etopo_1deg.nc --smoothing-passes 2 \
  --forcing-mode jra55_do_tropical \
  --jra55-cache data/jra55_ryf_cache/jra55_do_v14_omip2_1deg_noleap_fixed.zarr \
  --jra55-cycle \
  --woa-init \
  --woa-t data/woa18/woa18_decav_t00_01.nc \
  --woa-s data/woa18/woa18_decav_s00_01.nc \
  --jra55-no-sponge --jra55-no-sss-restoring --jra55-no-freeze-cap \
  --output results/mpas_jra55_etopo_100yr_woa \
  > results/mpas_jra55_etopo_100yr_woa.log 2>&1 &
```

**Potential concerns:**
- WOA T/S creates strong horizontal pressure gradients (warm tropics,
  cold poles, realistic halocline).  Geostrophic adjustment from rest
  (u=0, eta=0) will produce transient currents of O(1 m/s) in the
  first few days.  The 1200s timestep should handle this (CFL ~0.2
  with max baro speed ~200 m/s), but monitor day-1 max_speed.
- Sub-seafloor WOA values are NaN-filled to 1.5°C / 34.5 PSU, then
  masked by `land_mask * partial_cell.is_active`.  Verify no stale
  sub-seafloor values leak into active cells.
- WOA S has a different range (~33-37 PSU) than uniform 35 — the
  virtual salt flux and freshwater forcing interact differently.

**Validation:**
- Compare day-1 max_speed against the rest-state run (expect ~2-5x
  higher initial transient, then relaxation over ~30 days).
- Compare year-1 SST seasonal cycle — should be similar to v4 but
  with faster adjustment since the initial state is already close
  to the forced equilibrium.
- Check that the zonal-mean T section at day 0 matches WOA18
  (thermocline depth, polar cold water, etc.).

### Later Steps

1. **Monitor 100-year run** — check for drift, instabilities, blowup
2. **Enable SSS restoring** — weak piston velocity (~5e-7 m/s) to
   prevent long-term freshwater drift
3. **Enable freeze cap** — prevent unphysical sub-freezing SST at poles
4. **Enable sponge** — polar damping at ±60° if needed for stability
5. **Production OMIP config** — 5-cycle (300-year) run per OMIP-2 protocol
6. **Diagnostics** — OMIP Table fields (~150 variables) for comparison
   with other models

## Run Status (as of 2026-05-20)

### Completed Runs

**ico5 v5 — 100 years, PASSED** (the reference run)
- Output: `results/mpas_jra55_etopo_100yr_woa_v5/mpas/ico5/`
- IC: 9-year WOA nudge (tau=365d), then free run from day 3300
- Config: A_h=1e5, correct wind stress, SW penetration, freshwater
  normalization, freeze cap, SSS restoring (5e-7), basin removal
- Wall time: 12.6 hours on 1× V100S
- Key results:
  - AMOC at 26.5°N: 34 Sv (z-space), 33 Sv (sigma-space) — too
    strong (observed ~17 Sv), common in no-ice models
  - ACC Drake transport: ~163 Sv mean (observed 130-170 Sv) — good
  - Pacific MOC: 0.7 Sv z-space (correct), 12.8 Sv sigma-space
  - AABW: -3.4 Sv (observed -4 to -6 Sv) — reasonable
  - Warm pool SST stuck at ~38-39°C (legacy from 32 years of
    reversed wind forcing before the sign fix)
  - Equatorial upwelling correct after wind fix

**ico5 10-year WOA nudge v2** (IC source for ico5 v5)
- Output: `results/mpas_jra55_etopo_10yr_nudge_v2/mpas/ico5/`
- SSH mean: -0.003m (volume-neutral nudging working)
- Deep T converged to 0.08°C RMS of WOA

**ico6 10-year WOA nudge** (IC source for ico6 free run)
- Output: `results/mpas_jra55_etopo_10yr_nudge_ico6/mpas/ico6/`
- Final restart: `restarts/restart_day003650.npz`

### In Progress (stopped, can restart)

**ico6 100-year free run — stopped at day 14490 (year 29.7 of free run)**
- Output: `results/mpas_jra55_etopo_100yr_ico6/mpas/ico6/`
- Latest restart: `restarts/restart_day014490.npz`
- Config: same as ico5 v5 but at ico6 (~112 km, 40962 cells)
- Performance: 1.7s compute + 3.8s io = 5.5s/day (io-bound from
  per-block JRA55 regridding at 40k cells)
- Known issue: per-block regridding is the bottleneck — should be
  done once at preload time, not per block

### How to Restart the ico6 Run

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 nohup .venv/bin/python scripts/run_omip.py \
  --grid mpas --resolution ico6 --nlev 20 --days 40150 --dt 1200 \
  --H-max 5500 --H-min 10 \
  --bathymetry data/bathymetry/etopo_1deg.nc --smoothing-passes 2 \
  --forcing-mode jra55_do_tropical \
  --jra55-cache data/jra55_ryf_cache/jra55_do_v14_omip2_1deg_noleap_fixed.zarr \
  --jra55-cycle \
  --woa-t data/woa18/woa18_decav_t00_01.nc \
  --woa-s data/woa18/woa18_decav_s00_01.nc \
  --sss-piston-velocity 5e-7 \
  --jra55-no-sponge \
  --restart results/mpas_jra55_etopo_100yr_ico6/mpas/ico6/restarts/restart_day014490.npz \
  --output results/mpas_jra55_etopo_100yr_ico6 \
  > results/mpas_jra55_etopo_100yr_ico6.log 2>&1 &
```

### How to Generate Snapshots

```bash
bash scripts/gen_yearly_snapshots.sh results/mpas_jra55_etopo_100yr_ico6/mpas/ico6 6
```

### Performance Optimization Needed for ico6

The io=3.8s/day bottleneck is from regridding the JRA55 preloaded
cache (lat-lon → 40k MPAS cells) at every block.  The fix: regrid
the entire cache once during preload, not per block.  Currently in
`_preload_jra55_raw_records` the regridding happens per block; for
`_preload_jra55_full_cache` + `_slice_preloaded_records` the data
is already regridded at preload time but the slicing still triggers
a JAX computation.  Needs investigation.

### Bugs Found and Fixed During This Campaign

1. **Wind stress sign** — bulk-flux tau was in atmosphere convention,
   applied without negation to ocean. Fixed in mpas_physics.py.
2. **SW penetration missing** — all heat into 21m surface cell.
   Added Jerlov penetration in mpas_physics.py.
3. **Freshwater normalization** — P-E+R imbalance without ice caused
   -0.34 m/yr SSH drift. Fixed with global-mean subtraction.
4. **Volume-neutral WOA nudging** — nudge-induced density changes
   caused +5m SSH bias. Fixed with eta correction post-nudge.
5. **Enclosed basin removal** — flood-fill algorithm for all grids
   (Baltic, Black Sea, Caspian, White Sea at ico5; 10 basins at ico6).
6. **Salinity clamp** — virtual salt flux can overshoot to S<0 in
   shallow cells with heavy runoff. Clamped at 0.
7. **Greenwich meridian cache bug** — conservative regrid longitude
   wrap missing ghost column.
8. **North cap default** — CLI default was 80°N, should be 90°.
9. **SSS restoring land mask** — missing in GPU-interp scan path.
10. **Matplotlib thread safety** — background snapshot threads crash
    without lock serialization.
11. **Tripcolor long triangles** — Delaunay creates cross-land
    connections, filtered by max edge length.
