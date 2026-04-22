# Advection Scheme Comparison: Eady Baroclinic Instability

## Motivation

Issue #210 implements several tracer advection schemes of increasing sophistication.
We need to compare their implicit numerical diffusion, stability, and conservation
properties in a physically meaningful test case before selecting a default for OMIP runs.

Hill et al. (2012, Ocean Modelling 45-46) showed that the advection scheme is the
dominant source of spurious diapycnal mixing in z-coordinate ocean models, with
effective diffusivities spanning 3 orders of magnitude between schemes.

## Experiment Design

### Test case: Classical Eady baroclinic instability

A zonally periodic channel with uniform N^2, linear vertical shear, and linear EOS.
The instability grows from a small temperature perturbation, develops eddies, and
(with sufficient dissipation) saturates at finite amplitude. This tests:

- **Implicit numerical diffusion**: sharper fronts = less numerical mixing
- **Conservation**: total heat should be preserved
- **Stability**: scheme must not blow up
- **Front structure**: slumping (physical) vs spreading (numerical diffusion)

### Domain

| Parameter | Value |
|-----------|-------|
| Longitude | 0-10 deg (periodic) |
| Latitude | 16-34 deg (solid walls, no sponge) |
| Depth | 5500 m |
| Resolution | 100x50 (~20 km), 20 vertical levels |
| Timestep | 300 s |

### Physics (same for all schemes)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| KPP | On (defaults) | Physical BL mixing; used by Hill et al. |
| A_h | 0 | No Laplacian viscosity |
| B_h | 2.3e11 m^4/s | Biharmonic viscosity, scaled from Hill et al. (9e8 at 5km, ~ dx^4) |
| C_smag | 0 | No Smagorinsky (single clear viscosity) |
| K_h | 0 | No explicit tracer diffusion |
| K_bih | 0 | No biharmonic tracer diffusion |
| A_v | 1e-5 m^2/s | Background vertical viscosity |
| K_v | 5e-6 m^2/s | Background vertical diffusivity |
| bottom_drag | 0.001 m/s | Linear bottom drag |
| Sponge | Off | Clean comparison, no artificial relaxation |

### Stratification and flow

| Parameter | Value |
|-----------|-------|
| N | 1.2e-3 s^-1 |
| T_ref | 10 degC |
| alpha_T | 2e-4 K^-1 (linear EOS) |
| U_surface | 0.8 m/s (thermal wind shear) |
| L_d | ~107 km (deformation radius) |
| tau_Eady | ~5 days (e-folding time) |
| Perturbation | k=3, 0.1 K |

### Schemes under test

| Scheme | Order | Monotone | Expected implicit K_eff |
|--------|-------|----------|------------------------|
| `upwind` | 1st | Yes | ~|u|*dx/2 ~ 5000 m^2/s |
| `tvd` (Van Leer) | 2nd | Yes | Moderate |
| `ppm_fct` (PPM+Zalesak) | 4th | Yes | Low |
| `som` (Prather 1986) | 2nd moments | No (smooth) | Near zero |

### Commands

```bash
# Common flags
FLAGS="--only eady_uniform --grid latlon_channel --resolution 100x50 \
       --days 200 --no-sponge --B-h 2.3e11 --C-smag 0"

# Run each scheme
JAX_ENABLE_X64=1 python3 scripts/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection upwind --tag upwind_nosponge

JAX_ENABLE_X64=1 python3 scripts/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection tvd --tag tvd_nosponge

JAX_ENABLE_X64=1 python3 scripts/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection ppm_fct --tag ppm_fct_nosponge

JAX_ENABLE_X64=1 python3 scripts/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection som --tag som_nosponge
```

### Output location

```
results/ocean/eady_uniform/latlon_channel/100x50/<tag>/
```

Each directory contains: SST snapshots, T cross-sections, velocity profiles,
conservation timeseries, mean timeseries, and restart data.

### Key diagnostics to compare

1. **T_drift**: |mean_T(final) - mean_T(initial)| — conservation quality
2. **max_speed**: velocity evolution — stability indicator
3. **T_latitude_vertical_cross_sections**: front sharpness — slumping vs diffusion
4. **snapshots_SST**: eddy structure and grid-scale noise
5. **conservation_timeseries**: heat/salt/volume drift over time

## Results

### Run 1: 2026-04-22 — Baseline comparison (no sponge)

**Parameters**: 100x50, 200 days, dt=300s, B_h=2.3e11, C_smag=0, KPP on, no sponge.

| Scheme | Status | Days | T_drift | max_speed | Wall | Notes |
|--------|--------|------|---------|-----------|------|-------|
| upwind | **PASS** | 200 | 3.79e-04 | 8.28 | 16 min | Eddies peak ~day 120, then decay |
| tvd | **PASS** | 200 | 6.06e-04 | 5.04 | 17 min | Similar pattern, lower peak velocity |
| ppm_fct | FAIL | ~9 | 1.26e-03 | 1.54 | 1 min | Blowup during initial growth |
| som | FAIL | ~111 | 1.25e-04 | 30.4 | 22 min | Stable through day 95, then velocity explosion |

**Velocity evolution (max_speed in m/s)**:

| Day | upwind | tvd | som |
|-----|--------|-----|-----|
| 10 | 0.41 | 0.41 | 0.41 |
| 20 | 0.52 | 0.50 | 0.51 |
| 40 | 1.34 | 1.03 | 1.02 |
| 60 | 1.63 | 1.17 | 1.24 |
| 80 | 2.84 | 2.70 | 2.33 |
| 100 | 5.94 | 6.01 | 8.17 |
| 120 | 10.4 | 6.87 | — (NaN) |
| 150 | 9.65 | 6.65 | — |
| 200 | 8.28 | 5.04 | — |

**Key findings**:

1. **SOM has the best conservation**: T_drift=1.25e-04 (3-5x better than upwind/tvd).
   The improved conservation comes from near-zero implicit numerical diffusion.

2. **SOM tracks tvd closely through day 95**: velocity evolution is nearly identical,
   showing the instability growth rate is physical, not numerical.

3. **SOM blows up at day ~111**: velocity jumps from 4.5 to 30 m/s in 15 days.
   This coincides with eddies reaching the channel walls (no sponge to absorb them).
   Without a sponge, wall-reflected eddies create grid-scale noise that less-diffusive
   schemes cannot damp.

4. **PPM-FCT fails early (day 9)**: likely incompatible with the large B_h=2.3e11.
   May need separate tuning or the original B_h=1e10 + Smagorinsky.

5. **Upwind/tvd survive** because their implicit diffusion damps wall reflections.

**Next steps**:
- Re-run with sponge enabled to test SOM stability at longer times
- Re-run PPM-FCT with original B_h=1e10 + C_smag=0.2
- Compare T cross-sections visually for front sharpness
- Try higher resolution (200x100) where wall effects are delayed

## References

- Hill, Ferreira, Campin, Marshall, Abernathey, Barrier (2012).
  "Controlling spurious diapycnal mixing in eddy-resolving height-coordinate
  ocean models." Ocean Modelling, 45-46, 14-26.
  PDF: `docs/references/Hill_etal_11.pdf`

- Prather, M. (1986). "Numerical advection by conservation of second-order
  moments." J. Geophys. Res., 91, 6671-6681.
