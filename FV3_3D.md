# FV3 3D Cubed-Sphere Edge-Artifact Investigation

Goal: solve cube-edge artifacts in the 3D atmospheric cubed-sphere paths
(`primitive_eq_cdgrid.py` hydrostatic + `compressible_euler_cdgrid.py`
non-hydrostatic) with **perfect conservation and no edge effects**, by
being faithful to the GFDL FV3 Fortran reference at
`../FV3/atmos_cubed_sphere-symmetryclean/model/`.

The shallow-water FV3 path is "decent"; the 3D atmospheric paths produce
visible cube imprint (concentric blobs at face centres bordered by
red/blue rings at panel boundaries) in u/v wind snapshots from
Held-Suarez and baroclinic test cases.

## Reference oracle (read-only, never modify)

- `sw_core.F90` (3917 LOC):
  - `c_sw` (line 79): C-grid half of forward-backward scheme
  - `d_sw1` (line 500): D-grid half — handles uc→ut at edges, 2x2 corner solves
  - `d_sw5` (line 1474): canonical D-grid update with divergence damping
  - `divergence_corner` (line 2124): edge-aware divergence at corners with sin_sg
    metric and explicit corner-removal terms (`if (sw_corner) delpc(1,1) =
    delpc(1,1) - vort(1,0)`)
  - `fill2_4corners`, `fill_4corners` (line 3794, 3856): scalar halo fill at the
    8 cube vertices
  - `d2a2c_vect`: D→A→C vector conversion with edge stencils
- `dyn_core.F90`: time integration, sponges
- `fv_dynamics.F90`: top-level dynamics
- `a2b_edge.F90`: A→B grid 4th-order interpolation (used for vorticity at corners)
- `tools/fv_mp_mod.F90:fill_corners_2d_r8` (line 1032): generic corner fill,
  diagonal mirror at the 8 cube vertices

## Iteration 1 (2026-05-07): Diagnose then port

### Survey of current 3D path

`fv3_hydrostatic_tendencies` in `src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py`:
- Uses `dgrid_to_cgrid` from `core/operators_cdgrid.py` — simpler than
  FV3's `d2a2c_vect`, no edge stencils, no corner 2x2 solves
- Uses `_arakawa_lamb_gradient` for B-function gradient at D-grid corners
  (Cartesian 2x2 matrix)
- Uses `dgrid_vorticity` for ζ at cell centres (circulation form, OK)
- Uses `cgrid_divergence` at cell centres (no edge handling, no corner removal)
- Uses `_interp_center_to_corner` (4-point average, no FV3 corner mirror)
- **Does NOT call `_extrapolate_boundary_corners`** (which SW + ocean both call)

`cdgrid_compressible_euler_slow_tendencies` in
`src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py`: same picture.

### Failed attempt 1.A: post-tendency vertex extrapolation

Mirrored SW + ocean by calling `_extrapolate_boundary_corners(du_d_dt,
dv_d_dt, n)` after the momentum tendency. Result on Held-Suarez C36 hybrid
30-day quick mode: cube imprint **got worse in the middle of each face**.

**Why it failed**: bilinear extrapolation `tend(0,0) = tend(1,0) + tend(0,1)
- tend(1,1)` is fundamentally an *interpolation device*, not an FV3 mechanism.
With 3D dynamics at the corner, both `tend(1,0)` and `tend(0,1)` can have the
same sign while `tend(1,1)` is small, so the extrapolation **amplifies**
rather than dampens the corner. Reverted (cleanly — `git diff` is empty).

User direction: "do not improvise — be faithful to the fortran code". This
extrapolation is a Python-specific stabilizer (existing comments call it
out as such); it is NOT in `sw_core.F90` and should not be the answer.

### Direction for next iteration

Plan to port the actual FV3 mechanisms in priority order:

1. **`fill_4corners` for scalars before transport** (`sw_core.F90:3856`).
   Fill the cube-vertex halos of `delp`, `pt`, `w`, `ln(p_s)`, `T`, `q` from
   face-local interior values BEFORE PPM/upwind transport reads halo
   data. Today our `pad_halo` exchanges with the neighbour face; at the
   8 cube vertices this leaves an ambiguous "3-face-meet" cell whose
   value is averaged across panels and creates the imprint.

2. **`divergence_corner` edge handling** (`sw_core.F90:2124`). When
   computing `cgrid_divergence` at the j==0 / j==n-1 / i==0 / i==n-1 rows,
   use the sin_sg metric instead of the full va·cos_sg correction (the
   simpler edge formula matches FV3 lines 2187-2197, 2209-2213, 2216-2219).

3. **`_d2a2c_vect`-style edge stencils in `dgrid_to_cgrid`**. Today
   `dgrid_to_cgrid` uses a single bilinear average + non-orth correction
   everywhere; FV3 uses `c1/c2/c3` one-sided cubic stencils at i=1,n-1
   and `edge_interpolate4` at the face boundary i=0,n. The faithful
   version already exists in `src/legoesm/core/fv3_sw_core.py:_d2a2c_vect`
   but is only wired into the SW FV3-Edge model.

4. **`_arakawa_lamb_gradient` corner removal**. Mirror the FV3
   `if (sw_corner) delpc(1,1) = delpc(1,1) - vort(1,0)` semantics by
   subtracting the spurious 4th-stencil contribution at the 8 cube
   vertices when computing `dB/dx` and `dB/dy_perp` at D-grid corners.

These are operator-level fixes during computation, not post-tendency
fix-ups. Each step is verified against the FV3 Fortran source line by
line and validated visually on the HS C36 hybrid v-wind snapshot.
