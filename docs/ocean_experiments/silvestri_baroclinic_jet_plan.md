# Plan: Silvestri Baroclinic Jet Experiment Script

## Status: IMPLEMENTED (2026-04-24)

Script written and smoke-tested on CPU. Ready for GPU production runs.

**Branch:** `dhruv/weno-core`
**Script:** `scripts/run_silvestri_baroclinic_jet.py` (~480 LOC)

### Smoke test results (20x20, 10 days, CPU)

| Scheme | Final KE | Final EKE | Max Speed | Status |
|--------|----------|-----------|-----------|--------|
| centered (Smag) | 6.43e+11 | 8.16e+06 | 0.075 m/s | OK |
| weno5 (ILES) | 6.49e+11 | 1.13e+07 | 0.075 m/s | OK |

WENO5 retains ~38% more eddy KE than centered+Smagorinsky — expected ILES behavior.

### GPU commands to run

```bash
# Eddy-permitting (1/4 deg, ~20 min on GPU)
JAX_ENABLE_X64=1 python scripts/run_silvestri_baroclinic_jet.py \
    --resolution 80x80 --days 200 --dt 300 \
    --schemes centered,leith,weno5,weno5_leith \
    --output-dir results/ocean/silvestri_jet_quarter --plot

# Paper resolution (1/8 deg, ~hours on GPU)
JAX_ENABLE_X64=1 python scripts/run_silvestri_baroclinic_jet.py \
    --resolution 160x160 --days 1000 --dt 300 \
    --schemes centered,leith,weno5,weno5_leith \
    --output-dir results/ocean/silvestri_jet_eighth --plot
```

### What to look for in results

1. **EKE growth**: all schemes should show exponential eddy KE growth starting ~day 30-50
2. **EKE saturation**: WENO5 (ILES) should saturate at higher EKE than centered+Smagorinsky (less dissipation)
3. **Surface vorticity**: WENO5 snapshots should show sharper, finer-scale eddies
4. **Stability**: WENO5 with no closure should remain stable (the implicit WENO dissipation replaces explicit viscosity)
5. If WENO5 blows up, try `weno5_leith` (adds mild Leith closure as backup)

### Next steps after GPU runs

- Wire energy/enstrophy spectra diagnostics (Phase 1d exists) into the script
- Compare spectra slopes: WENO5 should preserve the k^-3 enstrophy cascade better
- If results are promising, proceed to Phase 4 (fix #160, D term, C+D splitting)

---

## Context

With WENO momentum advection (Phase 2b) and tracer advection (Phase 2a) implemented, we need a physically realistic test case to compare scheme behavior. The Silvestri et al. (2024) paper uses a baroclinic jet on a spherical sector (Section 5) as its primary 3D validation. We already have Eady instability and ACC channel experiments as templates. This script sets up the Silvestri-specific configuration so the user can run comparisons on GPU.

## Physical Setup (matching Silvestri Section 5)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Domain | 60S-40S, 0-20E | Spherical sector, periodic in x |
| Depth | 1000 m | Flat bottom |
| Vertical | 50 levels, uniform 20 m | `create_ocean_z_star(50, 1000, 20, 20)` |
| N^2 | 4e-6 s^-2 | Uniform stratification |
| Front | tanh at 50S, width 2deg, Db=5e-3 m/s^2 | DT ~ 2.55 K |
| EOS | Linear, alpha_T=2e-4 | `eos="linear"` + `LinearEOSConfig` |
| A_v | 1e-4 m^2/s | Vertical viscosity |
| K_v | 1e-5 m^2/s | Vertical diffusivity |
| Sponge | 3deg width, 50-day tau, both N/S walls | Restores T,S,u,v to initial |
| Bottom drag | r=1.1e-3 m/s | Linear |
| IC velocity | Thermal wind balance, depth-mean removed | eta=0 (purely baroclinic) |
| Perturbation | White noise in T, amplitude 1e-3 K | Seeds instability |

### Buoyancy-to-temperature mapping

```
T(lat, z) = T_ref + (N^2 / g*alpha_T)*z + (Db / 2*g*alpha_T) * tanh((lat - lat0) / w) * taper(lat)
```

With alpha_T=2e-4: dT/dz = 0.00204 K/m, DT_front = 2.55 K.

## Scheme Configurations

| Scheme | momentum_advection | tracer_advection | Closure |
|--------|-------------------|------------------|---------|
| centered | vector_invariant | tvd | C_smag=0.25 |
| leith | vector_invariant | tvd | C_leith=1.5 |
| weno5 | weno5 | weno7 | None (ILES) |
| weno5_leith | weno5 | weno7 | C_leith=1.0 |

## Resolution Presets

| Preset | n_lat x n_lon | dx at 50S | Use case |
|--------|--------------|-----------|----------|
| 1deg | 20 x 20 | ~70 km | Quick smoke test |
| 0.5deg | 40 x 40 | ~35 km | Development testing |
| 1/4deg | 80 x 80 | ~18 km | Eddy-permitting |
| 1/8deg | 160 x 160 | ~9 km | Paper resolution (GPU) |

## Key Functions

### `create_initial_state(grid, z_coord, wall_mask, cfg) -> LatLonCGridOceanState`
1. Call `rest_state_latlon_cgrid_ocean(grid, z_coord, ..., land_mask_override=wall_mask)`
2. Set `T(lat, z)` from analytical formula (uniform N^2 + tanh front + wall taper)
3. Compute thermal wind velocity bottom-up: `f0 * du/dz = -g * alpha_T * dT/dy` using analytical dT/dy = sech^2 derivative. Follow pattern from `eady_instability.py:274-352`.
4. Remove depth-mean u for purely baroclinic IC.
5. Add white noise to T.

### `create_sponge(grid, z_coord, state, cfg) -> SpongeForcing`
Use `compute_sponge_gamma_latlon()` with both N/S boundaries active. T_ref/S_ref = initial profiles (zonally uniform). u_ref = initial thermal wind, v_ref = 0.

### `run_one_scheme(scheme, grid, z_coord, state0, sponge, cfg, dt, days, output_dir)`
Create model with scheme-specific config, run time loop, collect diagnostics (total KE, eddy KE, max speed, mean T) at regular intervals, save snapshots and time series as `.npz`.

### CLI
```
python scripts/run_silvestri_baroclinic_jet.py \
    --schemes centered,weno5 \
    --resolution 20x20 \
    --days 200 --dt 600 \
    --output-dir results/ocean/silvestri_jet \
    --plot
```

## Reused Infrastructure

- `create_regional_latlon_grid(periodic_x=True)` from `grids/latlon.py:187`
- `create_ocean_z_star(50, 1000, 20, 20)` from `ocean/vertical.py:54`
- `rest_state_latlon_cgrid_ocean()` from `ocean/init_latlon_cgrid.py:53`
- `SpongeForcing` + `compute_sponge_gamma_latlon()` from `ocean/sponge.py`
- `LinearEOSConfig` from `ocean/eos.py:206`
- `LatLonCGridOceanModel` from `ocean/dynamics/ocean_model_latlon_cgrid.py`
- `curl_vertex_cgrid` from `ocean/dynamics/latlon_cgrid_operators.py`
- Thermal wind pattern from `ocean/experiments/eady_instability.py:274-352`
- Wall taper pattern from `ocean/experiments/eady_instability.py:129`

## Verification

1. **Smoke test (CPU):** `--resolution 20x20 --days 10 --schemes centered` — should complete in <2 min, produce finite output
2. **Instability check:** `--resolution 40x40 --days 200 --schemes centered` — should show exponential KE growth starting ~day 30-50
3. **Scheme comparison:** `--resolution 80x80 --days 200 --schemes centered,weno5` — WENO5 should show sharper eddies and different energy spectra
4. **GPU production run (user):** `--resolution 160x160 --days 1000` on GPU

## Not Included

- Energy/enstrophy spectra computation (Phase 1d diagnostics exist but not wired into this script — can be added later)
- AB2 time stepping (paper uses AB2, we use forward Euler + Matsuno Coriolis — fine for scheme comparison)
- D (divergence flux) and C+D vertical splitting (Phase 4b remainder — orthogonal to this experiment)
