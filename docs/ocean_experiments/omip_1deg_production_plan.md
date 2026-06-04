# Plan: 1° Global Ocean Production Configuration

## Status (2026-05-07)

We have a working 1° lat-lon C-grid ocean model configuration that is stable
for multi-decadal integrations on both idealized (flat-bottom) and realistic
(ETOPO) bathymetry. 100-year flat-bottom and 50-year ETOPO runs completed
successfully. This document records the configuration, the reasoning behind
each choice, and next steps.

## Critical Choices That Made 1° Work

We arrived at the working configuration through systematic decomposition —
changing one thing at a time and diagnosing failures through momentum balance
analysis and blowup location tracking. Each choice below solved a specific
failure mode.

### 1. KPP vertical mixing (not constant A_v)

**Problem solved**: Unbounded equatorial jet.

With constant A_v=1e-3, the equatorial zonal jet grows at 0.025 m/s/day
and blows up at day 98. The wind drives the surface westward but nothing
arrests it — f=0 at the equator means no Coriolis restoring.

**Why KPP works**: KPP provides depth-dependent mixing — strong in the
wind-driven boundary layer (0.01-0.1 m²/s), weak in the thermocline
(1e-4 m²/s). This communicates wind stress into a deeper layer while
allowing the thermocline to tilt, creating a zonal pressure gradient
that physically arrests the jet. This is the actual equatorial momentum
balance: Wind ≈ ZPG + vertical stress divergence.

**Key detail**: A_v background must be 1e-4 (not 1e-3). The 1e-3
background was 10× too high — it smeared the thermocline and prevented
the ZPG from developing.

### 2. cos¹(lat) viscosity scaling (not cos²)

**Problem solved**: Polar grid-scale blowup.

cos²(lat) keeps the viscous CFL constant but makes the grid Reynolds
number blow up as 1/cos(lat) at the poles. At 80° with A_h=5e4:
Re_grid = U·dx / (A_h·cos²) ≈ 28 — unstable.

**Why cos¹ works**: cos¹(lat) keeps Re_grid = U·R·dlon/A_h constant
at all latitudes. The viscous CFL grows as 1/cos(lat) toward the poles
but stays well below the stability limit (CFL < 0.03 at 85° for
A_h=1e5, dt=300s).

**Note**: The original CLAUDE.md comment claiming cos² is the
"standard MOM6/NEMO convention" was incorrect. MOM6 uses
resolution-dependent u₂·Δ scaling which is effectively cos¹. NEMO
ORCA1 uses ~cos¹ for Laplacian.

### 3. Equatorial A_h boost (factor 2, σ=5°)

**Problem solved**: Equatorial jet shear instability.

Even with KPP arresting the large-scale jet, the equatorial jet flanks
(lat ±1.5°) develop grid-scale shear instability at low A_h. At A_h=1e4
+ KPP, the jet reaches 2 m/s but the flanks go nonlinear and blow up
at day 62.

**Why boost works**: The boost provides A_h_eff=6e4 at the equator
(2× the base 3e4), keeping Re low enough at the jet flanks during
the ~100-day ZPG spinup period.

**Why not Smagorinsky for this**: Smagorinsky A_smag = (C·Δ)²·|D|
provides zero viscosity at the jet CENTER because the strain rate
|D| = 0 there (∂u/∂y = 0 at the maximum). Smagorinsky helps at jet
flanks and coastal features but cannot arrest the jet itself.

**Fundamental cause**: With 20 vertical levels, the thermocline is
poorly resolved and the ZPG takes ~100 days to develop. MOM6 at 1° with
75 levels doesn't need an equatorial boost because the ZPG develops
faster with better vertical resolution.

### 4. Laplacian Smagorinsky (C_smag_lap=0.15)

**Problem solved**: Coastal grid-scale blowup with realistic coastlines.

With A_h=1e4 + KPP (no Smagorinsky), the flat-bottom + real ETOPO
coastlines case blows up at day 182. Complex coastlines (capes, narrow
passages) create locally intense shear that constant A_h cannot damp.

**Why Smagorinsky works**: A_smag = (C·Δ)²·|D| automatically provides
strong viscosity where the flow is deforming (coastal jets, WBC shear)
and zero where it's quiescent (interior). Unlike the equatorial jet
center, coastal features have strong deformation.

**Convention note**: MOM6 uses C·Δ²·|D| (not (C·Δ)²·|D|), so their
C=0.15 is equivalent to our C=sqrt(0.15)≈0.39. Our effective
Smagorinsky is 6.7× weaker than MOM6's.

### 5. ETOPO bathymetry processing pipeline

**Problem solved**: Instant blowup (day 4) with realistic topography.

Even with all the physics above, ETOPO topography causes blowup within
4 days — including at A_h=2e5 with brute force. The issue is NOT the
physics parameters; it's the bathymetry itself creating PGF errors that
no reasonable viscosity can suppress.

**Three critical processing steps** (all required):

**(a) Equatorial smoothing (30 Laplacian passes within ±15°)**:
At the equator f≈0, so Coriolis cannot damp PGF errors from steep
bathymetry (Java Trench, deep Atlantic). The smoother reduces slopes
where the model has no physical damping mechanism.

Bug found: the original code had a radians/degrees mismatch that made
the smoothing accidentally global. Fixed to use proper radians.

**(b) Partial-cell snap (30% cutoff)**:
Thin partial cells (e.g., 5m adjacent to 200m) create enormous
potential vorticity q = ζ/h_vtx at step vertices, driving extreme
momentum tendencies. The snap removes ~12,000 thin cells by rounding
H_bathy to the nearest interface.

**(c) B_h_barotropic = 1e14 m⁴/s**:
Damps barotropic standing modes that develop over steep topographic
steps. Without this, the barotropic mode resonates at shelf breaks.

**Why z-star needs this but MOM6 doesn't**: MOM6 uses hybrid ALE
(isopycnal + z*) coordinates. Isopycnal layers in the interior
eliminate PGF errors over topography entirely. Our pure z-star is
fundamentally more sensitive. The processing pipeline compensates
for what MOM6 gets from its coordinate system.

### 6. A_h = 3e4 m²/s (the compromise)

**Problem**: Lower A_h gives better physics but worse stability.

| A_h | Physics quality | Stability |
|-----|----------------|-----------|
| 2e5 | Wrong: Wind ≈ A_h viscosity, no gyres | Stable everywhere |
| 5e4 | Better: Wind/Cor = 13×, still overdamped | Stable with KPP |
| 3e4 | Good: PGF/Cor ≈ 1 (geostrophy), approaching Ekman | Stable with full stack |
| 1e4 | Best midlat balance but Re=11 for U=1 m/s | WBC blowup at coasts |

A_h=3e4 gives Re=3.7 for U=1 m/s, which Smagorinsky supplements at
hotspots. Geostrophic balance holds at subtropics through high latitudes.
The equatorial balance (Wind/Cor=3.3×) is much better than the original
160× at A_h=2e5.

## Verified Stable Configurations

| Test | Duration | Max |u| | Notes |
|------|----------|---------|-------|
| Simple continent, flat, 1° | 100 yr (running) | ~1.1 m/s | Gyres developing |
| Real coastlines, flat, 1° | 5 yr | 1.9 m/s | Stable throughout |
| ETOPO realistic topo, 1° | 50 yr | 2.7 m/s | WBC at 27.5°N |
| Simple continent, flat, 5° | 50 yr | 1.2 m/s | Reference case |
| ETOPO realistic topo, 5° | 50 yr | 1.2 m/s | Reference case |

## Known Limitations

### Float32 precision

The model runs entirely in float32 (precision policy default). All
production ocean models (MOM6, NEMO, POP, MITgcm) use float64 for
dynamics. Consequences:

- CG solver chases tol=1e-10 but float32 precision is ~1.2e-7,
  so it likely runs all 200 iterations every step (wasting compute)
- Accumulation errors over millions of timesteps may cause drift in
  conservation (mass, heat, salt)
- Need to switch to float64 or mixed precision for production runs

### 20 vertical levels

Production models use 60-75 levels. With 20 levels:
- Equatorial thermocline poorly resolved (4 levels in 50-200m)
- KPP boundary layer coarsely represented
- Equatorial A_h boost needed to compensate for slow ZPG development
- PGF errors larger at partial-cell steps (fewer, thicker cells)

### Spinup timescales

At year 5, the circulation is barotropic-dominated (depth-independent
zonal jets). Baroclinic adjustment needs 10-20 years (first baroclinic
Rossby wave crossing) to develop surface-intensified gyres and WBCs.
Full equilibrium takes 50-100+ years.

### Smagorinsky convention

Our formula: A = (C·Δ)²·|D| = C²·Δ²·|D|
MOM6 formula: A = C·Δ²·|D|
Our C=0.15 gives 6.7× less viscosity than MOM6's C=0.15.

## Bug Fixes Applied

1. Radians bug in `slope_foot_enhancement_3d` (grid.lat already radians)
2. Face-thickness inconsistency (arithmetic mean → min-rule)
3. Barotropic fori_loop dtype mismatch (cast to eta.dtype)
4. Implicit CG solver dtype mismatch (same pattern)
5. `cast_pytree(allow_downcast=True)` at step boundary
6. H_bathy=0 on land causing div-by-zero
7. Equatorial smoothing radians/degrees bug in run_omip.py
8. cos²→cos¹ Laplacian scaling

## Performance: GPU-side Forcing Interpolation

The JRA55-do cache stores 3-hourly records (8 per day, 2920 per year).
At dt=300s, each simulated day requires 288 timesteps.  The original
path pre-interpolated all 288 steps in Python on the host before passing
them to the JIT-compiled `lax.scan` block.  This spent ~1.9s/day on
host-side I/O (29% of wall time).

**Optimization (--gpu-interp, now the default):** Load only the ~9
native 3-hourly records that bracket each block, pass them as stacked
arrays to the GPU, and do the linear interpolation + solar zenith
computation inside the `lax.scan` body.  This reduces host-side I/O
from 288 Python calls to ~9 Zarr reads per block.

| | CPU interp (old) | GPU interp (new) |
|---|-----------------|------------------|
| I/O per day | 1.9 s | 0.2 s |
| Compute per day | 4.6 s | 4.5 s |
| Total per day | 6.5 s | 4.7 s |
| 50-year wall time | ~33 h | ~24 h |

The interpolation inside the scan body is straightforward:
- `pos = day * 8` gives the fractional position in the record array
- `i_lo = floor(pos)`, `alpha = pos - i_lo` for linear weights
- `cos_zenith` computed from `doy = day % 365 + 1`, `hour = (day % 1) * 24`

No data is lost — the same 3-hourly forcing is used, just interpolated
on GPU instead of CPU.  Physics results are bit-identical.

## Momentum Budget Diagnostics

**Critical note:** The `MomentumTendencyDiagnostics` from the PE
tendency function does NOT include the planetary Coriolis term (f×u).
Coriolis is applied in the forward-backward step function and is never
captured in any diagnostic field.  The `vortcor_u/v` fields are
**relative vorticity advection** (ζ×F/h), not Coriolis.

To check geostrophic balance, compute `f×v` independently from the
restart velocity field and compare with `KE_PGF`.  The script
`scripts/tmp/diagnose_omip_momentum.py` does this automatically.

At year 1 of the uniform-profile spinup, PGF/f×v ≈ 1 at 600-800m
depth (geostrophy developing), with PGF exceeding Coriolis toward
the bottom (z-star bathymetry-step PGF errors).

## Next Steps

### Immediate

1. Switch to float64 (or mixed precision) — benchmark performance impact
2. Reduce CG maxiter 200→50 (at float32 it can't converge anyway)
3. Increase vertical resolution to 40-50 levels
4. Wire ETOPO processing pipeline into BathymetryConfig

### Production OMIP path

5. Switch to Wright nonlinear EOS
6. Switch to JRA55-do forcing via bulk formulas
7. Initialize from WOA18 climatology
8. Run OMIP-2 protocol

## Key Files

- `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` — PE tendencies + Smagorinsky
- `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py` — cos¹ scaling, operators
- `src/legoesm/ocean/state.py` — config (A_h_floor, C_smag_lap)
- `src/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py` — CG solver
- `src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` — model step
- `scripts/run/run_omip.py` — OMIP pipeline with bathymetry processing
- `scripts/global_overturning/diagnose_momentum_balance.py` — budget analysis
