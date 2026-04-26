# Global Overturning Experiment Plan (Wolfe & Cessi 2010 inspired)

*Created: 2026-04-25*

## Context

**Goal**: Create a global baroclinic ocean experiment with overturning circulation,
stratification, and a reentrant Southern Ocean channel as a stepping stone toward
OMIP. Inspired by Wolfe & Cessi (2010, JPO) but adapted:

- Spherical geometry instead of Cartesian beta-plane
- 1-0.5 deg resolution (non-eddy-resolving; OMIP-class)
- Side-by-side lat-lon / MPAS testing

**Starting point**: The existing `global_barotropic_wind` experiment already has
the right geometry (simplified continent 20-60 deg E, Drake Passage open south
of 55 deg S, polar caps at 80 deg) and runs on both lat-lon and MPAS grids.
We extend it from barotropic (uniform T) to baroclinic (stratified + surface
restoring).

**Paper requirements vs what we have**: ~90% of physics infrastructure exists.
The two gaps are: (1) MPAS physics pipeline lacks surface T restoring, and
(2) MPAS has no convective adjustment. Both are addressable.

**Reference**: Wolfe, C.L. and P. Cessi, 2010: What Sets the Strength of the
Middepth Stratification and Overturning in Eddying Ocean Models?
*J. Phys. Oceanogr.*, **40**, 1520-1538. (`docs/references/Wolfe&Cessi2010.pdf`)

---

## Phase 1: Create `global_overturning.py` experiment (lat-lon first)

**New file**: `src/legoesm/ocean/experiments/global_overturning.py`

Fork the structure of `global_barotropic_wind.py` (same continent mask, grid
support pattern, validation pattern). Key additions:

### Config: `GlobalOverturningConfig`

| Parameter | Value | Notes |
|-----------|-------|-------|
| T_surface | 20.0 degC | Surface temperature |
| T_deep | 2.0 degC | Abyssal temperature |
| T_scale_depth | 1000.0 m | Stratification e-folding depth |
| S_uniform | 35.0 PSU | Uniform salinity (T-only EOS) |
| H_max | 4000.0 m | Ocean depth (W&C: 2400; 4000 more realistic) |
| n_levels | 20 | Vertical levels |
| dz_surface | 10.0 m | Top layer thickness |
| dz_deep | 500.0 m | Bottom layer thickness |
| tau_T_days | 30.0 days | SST restoring timescale |
| T_star_eq | 25.0 degC | Equatorial target SST |
| T_star_pole | 0.0 degC | Polar target SST |
| tau_max | 0.1 Pa | Max wind stress (global_wind profile) |
| A_h | 5e4 m2/s | Laplacian viscosity (tunable for resolution) |
| K_v_bg | 1e-5 m2/s | Background vertical diffusivity |
| A_v | 1e-3 m2/s | Vertical viscosity |
| bottom_drag_coeff | 1.1e-3 m/s | Linear bottom drag |
| continent geometry | Same as global_barotropic_wind | 20-60 deg E, Drake at -55 deg S |

### Initial conditions

- Reuse `rest_state_latlon_cgrid_ocean` / `rest_state_mpas_ocean` with continent mask
- Add exponential T stratification: `T(z) = T_deep + (T_surface - T_deep) * exp(z / scale_depth)`
- Pattern: same as `baroclinic_gyre._add_stratification()` but on global domain

### Forcing (lat-lon path)

- `SurfaceForcingConfig(scheme="combined")` for wind + T restoring
  - Prescribed: `wind_profile="global_wind"`, `tau_max=0.1`
  - Restoring: `T_profile="cosine"`, `T_star_eq=25`, `T_star_pole=0`, `tau_T=30d`
- `VerticalMixingConfig(scheme="constant", A_v=1e-3, K_v=1e-5)`
- `OceanConvectionConfig(scheme="enhanced_diffusion")`
- `BottomDragConfig(scheme="linear", r=1.1e-3)`
- `LateralMixingConfig(scheme="none")` (A_h handled in dynamics config)

### Diagnostics

- SSH, SST, surface speed (inherited from global_barotropic_wind)
- Add: mean vertical T profile, T at 1000m depth (middepth stratification)

### Validation criteria

- Fields finite (no NaN)
- Circulation develops (max_speed > threshold)
- Stratification maintained (surface T > deep T)
- Volume conservation (eta drift < threshold)

### Reuse (no duplication)

- `_create_simplified_continent_mask()` from `global_barotropic_wind.py` — factor into shared helper
- `_add_stratification()` from `baroclinic_gyre.py` — reuse or factor out
- `restoring_surface_forcing()` via `scheme="combined"`
- `compute_wind_stress()` with `"global_wind"` profile
- `create_ocean_z_star()` for vertical coordinate

---

## Phase 2: Extend MPAS physics for surface restoring

**File**: `src/legoesm/ocean/physics/mpas_physics.py`

Add surface restoring to `make_mpas_ocean_physics()`:

```python
if has_restoring:
    lat_cell = mesh.grid_lat  # VoronoiMesh implements GridProtocol
    T_star = cfg_r.T_star_eq - (cfg_r.T_star_eq - cfg_r.T_star_pole) * jnp.sin(lat_cell)**2
    dT_dt = dT_dt.at[:, 0].add(-(T_3d[:, 0] - T_star) / cfg_r.tau_T * mask)
```

Support `scheme="combined"` and `scheme="restoring"` (in addition to `"prescribed"`).

### Testing

- Unit test in `tests/ocean/unit/test_mpas_ocean.py` for MPAS surface restoring
- Verify dT_dt shape and sign

---

## Phase 3: Test matrix integration + lat-lon validation

**File**: `scripts/ocean_test_matrix/experiments.py`

Add `run_global_overturning()` runner:

- Default duration: 90 days
- Quick duration: 5 days
- Default resolution: 36x72 (5 deg, fast)
- Grid support: latlon, mpas (cubed_sphere excluded per issue #100)
- Vertical: 20 levels, 10m surface, 500m deep

### Validation (lat-lon 1 deg, 60-90 days)

```bash
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py \
  --grid latlon --resolution 180x360 --levels 20 --only global_overturning
```

Check: gyres, ACC, stratification, WBC, stability.

---

## Phase 4: MPAS cross-grid validation

```bash
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py \
  --grid mpas --resolution ico6 --levels 20 --only global_overturning
```

Compare lat-lon vs MPAS: SSH, SST, circulation patterns, stratification profiles.

---

## Phase 5: Resolution scaling + diagnostics (future)

- 0.5 deg lat-lon (720x360) — needs timestep adjustment
- MPAS ico7 (~0.9 deg equivalent)
- MOC streamfunction computation (new diagnostic)
- Meridional heat transport diagnostic
- Multi-year runs for deep-ocean adjustment
- GM/Redi eddy parameterization (not yet in codebase)

---

## Key risks and mitigations

1. **Stability at 1 deg with 90-day run**: CFL is comfortable with dt=3600s.
   Convective adjustment prevents static instability. Fallback: reduce dt or
   increase A_h.

2. **MPAS surface restoring**: Straightforward — restoring has zero momentum
   tendency, only dT_dt. VoronoiMesh has `grid_lat` via GridProtocol.

3. **No eddy parameterization at 1 deg**: W&C is eddy-resolving; at 1 deg
   stratification and MOC will be diffusively maintained. Acceptable as
   stepping stone — OMIP models also run at 1 deg (typically with GM/Redi).

4. **Vertical coordinate**: z-star with 20 levels and stretching.
   `create_ocean_z_star(n_levels=20, H_max=4000, dz_surface=10, dz_deep=500)`.

---

## Implementation order

1. Factor shared continent mask helper (avoid code duplication)
2. Create `global_overturning.py` with lat-lon support
3. Quick test at 5-deg resolution, 5 days — verify it runs
4. Extend MPAS physics with surface restoring
5. Add MPAS support to `global_overturning.py`
6. Register in test matrix
7. Run at 1-deg lat-lon, 60 days — visual inspection
8. Run at MPAS ico5/ico6, 60 days — cross-grid comparison
