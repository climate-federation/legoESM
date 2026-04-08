# Lat-Lon Ocean C-Grid Migration Plan

## Motivation

The latlon ocean model uses collocated A-grid staggering (u, v, eta, T, S all at cell centers). Both the gradient and divergence operators use 2-cell centered stencils that share a checkerboard null space: nonlinear terms project energy into the 2-delta-x mode, and no dynamical restoring mechanism exists to remove it. This causes grid-scale instability at moderate resolution (~1-4 degrees) after 3-5 days of integration, as documented in issues #58 and #87.

The MPAS ocean model in the same codebase uses C-grid (TRiSK) and is perfectly stable at all tested resolutions. Migrating the latlon ocean to Arakawa C-grid staggering eliminates the computational mode structurally, without relying on tunable diffusion.

## Overview

### What changes
- Velocity staggering: u moves to east cell faces, v moves to north cell faces
- All operators touching velocity (gradient, divergence, vorticity, advection, diffusion, Coriolis) are rewritten with compact 1-cell stencils
- The barotropic solver operates on face-staggered velocity internally

### What stays the same
- All scalar fields (eta, T, S, density, pressure) remain at cell centers
- EOS/density computation: unchanged
- Conservation fixer: unchanged (only touches cell-center scalars)
- Tracer PPM reconstruction: unchanged (actually simplifies -- face velocity is directly available)
- Vertical coordinate and w diagnosis formula: unchanged
- AD/differentiability: all C-grid operators are smooth linear averages + differences

### Scope
12 source files, ~3000-4000 lines of new/modified code. The biggest changes are in the operators, barotropic solver, and baroclinic tendency computation.

## State Representation

### Array shapes

On the Arakawa C-grid for a latlon domain with `n_lat` cells in latitude and `n_lon` cells in longitude (periodic):

| Field | Location | Shape (2D) | Shape (3D) | Notes |
|-------|----------|------------|------------|-------|
| eta, T, S, rho, p | Cell center | (n_lat, n_lon) | (n_lat, n_lon, nlev) | Unchanged |
| u | East face | (n_lat, n_lon) | (n_lat, n_lon, nlev) | Periodic lon: n_lon faces for n_lon cells |
| v | North face | (n_lat+1, n_lon) | (n_lat+1, n_lon, nlev) | n_lat+1 interfaces including pole walls |
| vorticity (zeta) | Vortex/corner point | (n_lat+1, n_lon) | (n_lat+1, n_lon, nlev) | At intersection of u-lines and v-lines |
| land_mask | Cell center | (n_lat, n_lon) | — | Unchanged |
| u_mask | East face | (n_lat, n_lon) | — | Derived: `mask * roll(mask, -1, axis=1)` |
| v_mask | North face | (n_lat+1, n_lon) | — | Derived: `mask[i-1] * mask[i]`, zero at poles |

### Why these shapes

**u at (n_lat, n_lon)**: With periodic longitude, there are exactly n_lon east faces. Face j sits between cell j and cell (j+1) mod n_lon. `jnp.roll(-1, axis=1)` handles the periodicity. u lives at the same latitude as the cell center (the east face shares the cell's latitude).

**v at (n_lat+1, n_lon)**: There are n_lat+1 meridional interfaces (including the two pole boundaries). Face i sits between cell i-1 and cell i. v lives at the interface latitude `lat_half[i] = 0.5*(lat[i-1] + lat[i])`. The pole faces (i=0 and i=n_lat) are always zero (no-normal-flow wall BC).

Storing the pole zeros explicitly is essential for JAX compatibility (static shapes) and clean indexing: the north-face flux of cell i is `v[i+1, j]`, the south-face flux is `v[i, j]`.

This convention matches MOM6 and NEMO.

### State type

Introduce a new `LatLonCGridOceanState` rather than modifying the existing `LatLonOceanState`. This keeps the A-grid path functional during the transition and for regression testing.

### Velocity masks

Compute u_mask and v_mask on-the-fly from the cell-center land_mask rather than storing them in state:

```
u_mask[i, j] = mask[i, j] * mask[i, (j+1) % n_lon]     # both cells flanking the face must be ocean
v_mask[i, j] = mask[i-1, j] * mask[i, j]                # for i in 1..n_lat-1
v_mask[0, :] = v_mask[n_lat, :] = 0                     # wall BC at poles
```

Note: u is tangent to the pole wall and is NOT zero at the poles. Only v (the wall-normal component) is zero. The boundary conditions for u and v are fundamentally different.

## Grid Infrastructure

### New fields needed in LatLonGrid

```
lat_v: jax.Array           # (n_lat+1,) interface latitudes (v-face locations)
cos_lat_v: jax.Array       # (n_lat+1,) cos(lat) at v-faces
sin_lat_v: jax.Array       # (n_lat+1,) sin(lat) at v-faces
dx_cell: jax.Array         # (n_lat,) single-cell zonal width = R * dlon * cos(lat)
dy_cell: float             # single-cell meridional width = R * dlat
dx_u: jax.Array            # (n_lat,) u-face edge length = R * dlon * cos(lat_cell)
dy_v: float                # v-face edge length = R * dlat (constant)
dx_v: jax.Array            # (n_lat+1,) north-face length = R * dlon * cos(lat_v)
f_u: jax.Array             # (n_lat, n_lon) Coriolis at u-points (same lat as cell center)
f_v: jax.Array             # (n_lat+1, n_lon) Coriolis at v-face latitudes
area_q: jax.Array          # (n_lat+1, n_lon) dual-cell area at vortex points
```

The existing fields (lat, lon, cos_lat, area, dx, dy, f, etc.) are kept unchanged for backward compatibility with the atmosphere and A-grid ocean.

## Operator Changes

### Classification of existing operators

| Operator | Current | C-Grid Status |
|----------|---------|---------------|
| `_ppm_edge_values_periodic` | 1D reconstruction | (a) No change |
| `_ppm_edge_values_bounded` | 1D reconstruction | (a) No change |
| `_ppm_limit` | Limiter | (a) No change |
| `fv_divergence_latlon` | A-grid, averages velocity to faces | (c) Replace with compact C-grid divergence |
| `fv_scalar_advection_latlon` | A-grid, averages velocity to faces | (b) Simplify: use face velocity directly |
| `gradient_x_latlon` | 2-cell centered | (c) Replace with 1-cell compact at u-face |
| `gradient_y_latlon` | 2-cell centered | (c) Replace with 1-cell compact at v-face |
| `vorticity_latlon` | At cell centers | (c) Replace with C-grid at vortex points |
| `laplacian_latlon` | Scalar Laplacian at cell centers | (b) Keep for tracer diffusion; add vector Laplacian for velocity |
| `_neumann_fill_latlon` | Cell-center fill | (a) Keep; add separate fill routines for u-face and v-face fields |

### New C-grid operators needed

**Compact gradient (replaces centered gradient):**
```
gradient_x_to_uface(f, grid):  # (f[j+1] - f[j]) / (R * dlon * cos_lat)  at east face
gradient_y_to_vface(f, grid):  # (f[i+1] - f[i]) / (R * dlat)  at north face
```
These are 1-cell differences with no checkerboard null space.

**Compact divergence:**
```
divergence_cgrid(u, v, grid):
    # div = (1/A) * (u[j]*dy_u - u[j-1]*dy_u + v[i+1]*dx_v[i+1] - v[i]*dx_v[i])
```
No velocity averaging needed -- face velocities are used directly.

**C-grid vorticity at vortex points:**
```
vorticity_cgrid(u, v, grid):
    # zeta[i,j] = (1/A_q) * (v[i,j]*dy_v - v[i,j-1]*dy_v - u[i,j]*dx_u[i] + u[i-1,j]*dx_u[i-1])
```

**Interpolation operators:**
```
u_to_cell(u):       # 0.5 * (u[j] + u[j-1])                   -> (n_lat, n_lon)
v_to_cell(v):       # 0.5 * (v[i] + v[i+1])                   -> (n_lat, n_lon)
cell_to_uface(f):   # 0.5 * (f[j] + f[j+1])                   -> (n_lat, n_lon)
cell_to_vface(f):   # 0.5 * (f[i] + f[i+1])                   -> (n_lat+1, n_lon)
```

**Kinetic energy at cell centers (Sadourny 1975):**
```
KE[i,j] = 0.5 * (u_to_cell(u)^2 + v_to_cell(v)^2)
```

**Vector Laplacian via grad-div minus curl-curl:**
```
del2_u = gradient_x_to_uface(div) - gradient along u-face of curl
del2_v = gradient_y_to_vface(div) + gradient along v-face of curl
```
This gives the correct vector Laplacian on a sphere, including metric correction terms near the poles. Should be implemented from the start rather than using a scalar Laplacian on each component separately.

**Neumann fill for face fields:**
Separate from the existing cell-center fill. Needed for u-face and v-face fields near land boundaries.

## Momentum Formulation

### Sadourny (1975) Energy-Conserving Scheme

The vector-invariant momentum equation on the C-grid:

```
du/dt = +q_at_u * Fv_at_u - grad_x(KE + p'/rho_0)    at u-points
dv/dt = -q_at_v * Fu_at_v - grad_y(KE + p'/rho_0)    at v-points
```

where:
- `q = (zeta + f) / h` is potential vorticity at vortex points
- `Fv = h_v * v` is the meridional thickness flux at v-faces
- `Fu = h_u * u` is the zonal thickness flux at u-faces

**Critical detail**: The Coriolis term acts on **thickness-weighted velocity** (transport), not bare velocity. This is what makes the scheme energy-conserving.

### PV and thickness flux averaging

**PV at u-points** (2-point average from adjacent vortex points):
```
q_at_u[i, j] = 0.5 * (q[i, j] + q[i+1, j])
```

**Thickness flux Fv at u-points** (4-point average from surrounding v-faces):
```
Fv_at_u[i, j] = 0.25 * (
    hv[i,j]*v[i,j] + hv[i+1,j]*v[i+1,j]
  + hv[i,(j+1)%n]*v[i,(j+1)%n] + hv[i+1,(j+1)%n]*v[i+1,(j+1)%n]
)
```

The 4-point average for the thickness flux (not 2-point) is required for the discrete energy conservation identity to hold.

### Why Sadourny first, not Arakawa-Lamb

- Sadourny energy-conserving is the simplest C-grid Coriolis scheme
- Conserves kinetic + potential energy exactly for non-divergent flow
- Does NOT conserve enstrophy (potential for spectral energy pile-up at grid scale)
- At the target resolutions (1-10 degrees), lateral viscosity controls grid-scale noise, so enstrophy non-conservation is acceptable
- If noise problems appear at 1-2 degrees, skip directly to Arakawa-Lamb (1981) rather than trying enstrophy-conserving Sadourny

## Barotropic Solver

### Structure

The barotropic solver operates on:
- eta at cell centers (n_lat, n_lon)
- U_bar at u-faces (n_lat, n_lon)
- V_bar at v-faces (n_lat+1, n_lon)

Forward-backward substep:
```
# Forward: continuity (compact divergence)
eta_new = eta - dt_s * divergence_cgrid(H_u * U_bar, H_v * V_bar)

# Backward: momentum (compact gradient)
U_bar_new = U_bar - dt_s * g * gradient_x_to_uface(eta_new) + dt_s * coriolis_u
V_bar_new = V_bar - dt_s * g * gradient_y_to_vface(eta_new) + dt_s * coriolis_v
```

### Coriolis in the barotropic solver

Semi-implicit via predictor-corrector (following MPAS pattern):
1. Predict U* using old V_bar for Coriolis
2. Update V_bar using U* for Coriolis
3. Correct U_bar using updated V_bar

### CFL

The maximum group velocity on C-grid is `c = sqrt(gH)`, identical to A-grid. The current `n_barotropic_substeps = 30` should be sufficient. C-grid does not need more substeps; it may need fewer because the computational mode no longer exists.

### Thickness at faces

`H_u` and `H_v` (total water column at u-faces and v-faces) must be computed by averaging cell-center values:
```
H_u[i,j] = 0.5 * (H_total[i,j] + H_total[i, (j+1)%n_lon])
H_v[i,j] = 0.5 * (H_total[i-1,j] + H_total[i,j])
```

This averaging must be identical between the barotropic solver and the baroclinic tendency computation to avoid spurious barotropic modes from thickness flux inconsistency.

### Barotropic-baroclinic reconciliation

The existing formula generalizes directly:
```
u_new = (u - U_bar_old) + U_bar_new    at u-faces
v_new = (v - V_bar_old) + V_bar_new    at v-faces
```

The depth-averaging that produces U_bar from 3D u requires face-averaged layer thickness `h_u_k`, which must be consistent with the barotropic solver's `H_u`.

## Tracer Advection

C-grid simplifies the PPM advection because face-normal velocities are directly available:
- Zonal PPM: u[i, j] at east face is the advecting velocity (no averaging needed)
- Meridional PPM: v[i+1, j] at north face is the advecting velocity

The PPM reconstruction of the tracer at cell interfaces stays the same. The upwind selection uses the face velocity directly instead of averaging cell-center velocity to faces. This eliminates interpolation error in the advecting velocity and ensures consistency with the continuity equation's divergence.

## Vertical Velocity

The w diagnosis formula (bottom-up integration of horizontal flux divergence) is unchanged. However, the horizontal flux divergence is now computed from face velocities via the compact C-grid divergence, which is more accurate than the A-grid's averaged velocity divergence. This eliminates the checkerboard mode in w.

Note: `h_u_k` and `h_v_k` (layer thickness at faces) must be computed for the thickness-weighted flux divergence, and must be consistent with the barotropic solver (see above).

## Boundary Conditions

### Poles
- v = 0 at pole faces (no-normal-flow wall BC)
- u is NOT zero at poles (tangent to the wall). Pole-row u needs appropriate lateral BC: either free-slip (du/dy = 0) or no-slip (u = 0), configurable
- Vortex points at poles are degenerate (all longitudes converge). Clamp vortex-point area `A_q` to a minimum value to avoid division by zero in PV computation

### Land masking
- Cell-center mask: unchanged
- u-mask: face is active only if both flanking cells are ocean
- v-mask: face is active only if both flanking cells are ocean, plus zero at poles
- Neumann fill for face fields: separate routines needed (cannot reuse cell-center fill directly)

### Wetting/drying
For face masks to handle wetting/drying, the u-mask and v-mask must be recomputed each barotropic substep based on whether adjacent cells have positive water column. Not needed for closed-basin experiments but required for realistic bathymetry:
```
u_wet[i,j] = (H_total[i,j] > min_wc) AND (H_total[i,(j+1)%n] > min_wc)
```

## Differentiability

All C-grid operators (linear interpolation, 1-cell differences, averaging) are smooth and differentiable. No AD blockers.

Considerations for the training path:
- The 4-point Coriolis averaging introduces more intermediate arrays, increasing the AD tape size
- The vortex-point PV adds a new intermediate with shape (n_lat+1, n_lon)
- Semi-implicit Coriolis via predictor-corrector requires two tangential velocity reconstructions per substep
- Memory-profile the AD path after the barotropic solver is working, before proceeding to the baroclinic step

## Float32 Precision

The additional interpolation operations (4-point averages, PV computation) can lose precision in float32:
- PV = (zeta + f) / h: clamp h to a minimum value (e.g., 1e-10)
- Vortex-point area involves cos(lat_v) which goes to zero at poles: clamp to minimum
- 4-point Coriolis average at high latitudes where f is large: monitor for noise

## Phasing

### Phase 1: Grid infrastructure + operators (no dynamics changes)
1. Add C-grid metric fields to `LatLonGrid` (lat_v, cos_lat_v, dx_u, dy_v, dx_v, f_v, area_q)
2. Create new C-grid operators (gradient, divergence, vorticity, interpolation, vector Laplacian)
3. Introduce `LatLonCGridOceanState` with the new v-shape
4. Unit tests: verify div(grad) = Laplacian, curl(grad) = 0, div(curl) = 0

### Phase 2: C-grid barotropic solver (the critical deliverable)
1. Implement `barotropic_cgrid_latlon()` with compact operators
2. Validate on the inertia-gravity wave test (pure barotropic, no tracers, no bathymetry)
3. Validate on barotropic wave test: confirm checkerboard is eliminated
4. Compare propagation speed and dispersion with analytical shallow water solution

### Phase 3: C-grid baroclinic dynamics
1. Implement C-grid baroclinic tendencies (pressure gradient, Coriolis/PV flux, divergence/w, viscosity)
2. Validate on rest state: tendencies should be small and finite
3. Validate on barotropic wave with full model: should match Phase 2
4. Validate on baroclinic adjustment test

### Phase 4: Full integration
1. Create `LatLonCGridOceanModel` (or config switch on existing model)
2. Update initialization functions for staggered v
3. Physics adapter for surface forcing and bottom drag (project to face locations)
4. Update test matrix diagnostics for staggered velocity (interpolate v to cell centers for plotting)
5. Run full ocean test matrix

### Phase 5: Validation and cleanup
1. Compare A-grid vs C-grid on all test cases, document checkerboard elimination
2. Run differentiability tests, memory-profile AD path
3. Update CLAUDE.md
4. Keep A-grid path as selectable option (`staggering: Literal["A", "C"]` in config)

## Files Requiring Changes

| File | Change | Effort |
|------|--------|--------|
| `grids/latlon.py` | Add C-grid metric fields | Medium |
| `ocean/state.py` | New `LatLonCGridOceanState`, `LatLonCGridOceanTendencies` | Small |
| `ocean/dynamics/latlon_operators.py` | ~400 lines of new C-grid operators | Large |
| `ocean/dynamics/barotropic_latlon.py` | New C-grid barotropic solver | Large |
| `ocean/dynamics/ocean_pe_latlon.py` | New C-grid baroclinic tendencies | Large |
| `ocean/dynamics/ocean_model_latlon.py` | Wire up C-grid model | Small |
| `ocean/init_latlon.py` | v-shape updates | Small |
| `ocean/conservation_latlon.py` | **No change** | None |
| `ocean/physics/combined.py` | Adapter for C-grid face shapes | Small |
| `ocean/experiments/*.py` (7 files) | v-shape updates | Small each |
| `scripts/run_ocean_test_matrix.py` | Diagnostics for staggered v | Medium |
| Tests | New C-grid operator tests, shape updates | Medium |

## References

- Arakawa, A. and V.R. Lamb, 1977: Computational design of the basic dynamical processes of the UCLA general circulation model. *Methods in Computational Physics*, 17, 173-265.
- Arakawa, A. and V.R. Lamb, 1981: A potential enstrophy and energy conserving scheme for the shallow water equations. *Monthly Weather Review*, 109, 18-36.
- Sadourny, R., 1975: The dynamics of finite-difference models of the shallow-water equations. *Journal of the Atmospheric Sciences*, 32, 680-689.
- Lin, S.-J., 1997: A finite-volume integration method for computing pressure gradient force in general vertical coordinates. *Quarterly Journal of the Royal Meteorological Society*, 123, 1749-1762.
- Ringler, T.D., J. Thuburn, J.B. Klemp, and W.C. Skamarock, 2010: A unified approach to energy conservation and potential vorticity dynamics for arbitrarily-structured C-grids. *Journal of Computational Physics*, 229, 3065-3090.

## Related Issues

- #58: Ocean grid staggering: document current status and barotropic A-grid trade-off
- #87: Latlon ocean A-grid computational mode instability at moderate resolution
