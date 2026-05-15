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

Tripcolor + cartopy Robinson projection snapshots (`scripts/plot_mpas_omip_snapshot.py`)
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
- **Freeze cap / sea ice**: Disabled.  Polar SST can go below freezing.
  Phase B of OMIP requires a coupled sea-ice model.
- **WOA initialization**: Starting from rest state with idealized T(z)
  profile, not WOA18 climatology.  WOA init available via `--woa-init`.
- **Equatorial bathymetry smoothing**: The lat-lon-specific equatorial
  smoothing in `run_omip.py` doesn't apply to MPAS.  The 20-year PR #261
  ETOPO run was stable without it, but 100-year stability is TBD.

## Next Steps

1. **Monitor 100-year run** — check for drift, instabilities, blowup
2. **Enable SSS restoring** — add `--jra55-no-sss-restoring` removal
   once stability is confirmed
3. **Enable freeze cap** — prevent unphysical sub-freezing SST at poles
4. **WOA initialization** — test with realistic T/S initial condition
5. **Enable sponge** — polar damping at ±60° if needed for stability
6. **Production OMIP config** — 5-cycle (300-year) run per OMIP-2 protocol
7. **Diagnostics** — OMIP Table fields (~150 variables) for comparison
   with other models
