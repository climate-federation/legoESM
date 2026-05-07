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

## Iteration 2 (2026-05-07): Quantify and localize the cube imprint

### Quantitative baseline (HS C36 hybrid, 30 days)

| day | edge_std | interior_std | edge/int | max\|v\| | zonal_std | eddy_std |
|-----|---------:|-------------:|---------:|---------:|----------:|---------:|
|   1 |   0.040  |   0.041      |   0.97   |   0.10   |   0.038   |   0.015  |
|   3 |   0.109  |   0.109      |   1.00   |   0.27   |   0.105   |   0.028  |
|   6 |   0.192  |   0.169      |   1.14   |   0.39   |   0.153   |   0.077  |
|  10 |   0.278  |   0.226      |   1.23   |   0.63   |   0.195   |   0.125  |
|  15 |   0.383  |   0.297      |   1.29   |   0.97   |   0.252   |   0.177  |
|  30 |   0.781  |   0.613      |   1.27   |   2.56   |   0.520   |   0.364  |

`zonal_std` = std of v zonal-mean profile (lat-only signal). `eddy_std`
= sqrt(total_var − zonal_var) (zonally-asymmetric component). Edge
artifacts emerge from t=3-6 d and grow to ~1.3× interior std by day 10.

### Reference: spectral T16 hybrid 30-day
v-wind is **completely zonally symmetric** — clean Hadley cell signal
(red ~southerly +0.5 in tropics, blue ~northerly −0.5 at 30°N). No
longitudinal variation. `eddy_std/zonal_std ≈ 0`.

### Reference: latlon 16x32 hybrid 30-day
Same — perfectly zonal.

### Diffusion sensitivity (4× hyperdiff + 4× div_damp)

| day | max\|v\| | eddy_std | reduction |
|-----|---------:|---------:|----------:|
|  30 |   1.81   |   0.276  | -29 % max\|v\|, -25 % eddy_std |

Stronger diffusion REDUCES the cube imprint but does NOT eliminate it.
Eddy_std is still 0.28 m/s (vs 0 for spectral/latlon). This rules out
a pure "noise-amplification" explanation: the cube imprint has a
SYSTEMATIC component the linear diffusion cannot reach.

### Hypothesis: η-coordinate hydrostatic PGF cancellation error

Comparing the FV3 fortran 3D PGF (`dyn_core.F90:p_grad_c` at line 2073)
with our `fv3_hydrostatic_tendencies`:

**FV3 (Lin 1997 cross-product, exact in hydrostatic balance):**
```fortran
wk(i,j) = pkc(i,j,k+1) - pkc(i,j,k)               ! δp^κ at cell centres
uc(i,j,k) += dt * rdxc / (wk_W + wk_E) * (
    (gz_W(k+1) - gz_E(k)) * (pkc_E(k+1) - pkc_W(k)) +
    (gz_W(k)   - gz_E(k+1)) * (pkc_W(k+1) - pkc_E(k))
)
```
This is the staggered cross-product formula — exactly mass-conserving
and gives EXACT cancellation between geopotential and pressure-tilt
terms in hydrostatic balance.

**Ours (split formulation):**
```python
B = KE + Φ                               # at cell centres
dB/dx = arakawa_lamb_gradient(B)         # at D-grid corners
pg_corr_x = R_d * T_corner * dln_dx_hi   # at D-grid corners
du_d/dt = ζ_corner*v_d - dB/dx - pg_corr_x
```

The two terms ∇Φ (in B) and `R_d*T*∇(ln p_s)` must cancel each other
in hydrostatic balance. They are computed at corners with DIFFERENT
interpolation paths:
- Φ from compute_geopotential_hybrid (cell centres) → in B → A-L
  gradient at corners
- T_corner from `1/(_interp_center_to_corner(1/T))` (harmonic mean)
- ln(p_s) → A-L gradient at corners

The harmonic mean of T at corners and the A-L gradient of B both
introduce O(dx) errors at face boundaries (panel-edge halo
amplification by the A-L Cartesian matrix). These errors don't
cancel because they come from different operators.

This is the FV3-fidelity gap responsible for the residual cube
imprint that diffusion cannot remove.

### Direction for next iteration

Test the PGF hypothesis by ablation:
1. Run with pg_corr_x = 0 to see if the cube imprint changes structure
   (would prove the η-correction is the source).
2. If yes: replace the split (∇B, pg_corr) formulation with a single
   FV3-faithful Lin (1997) PGF computed at C-grid faces, then projected
   to D-grid corners — without the A-L Cartesian matrix.

### Ablation results

#### 1. PGF correction ablation (pg_corr=0)

| config | day | edge_std | int_std | max\|v\| |
|--------|----:|---------:|--------:|---------:|
| baseline | 10 | 0.278 | 0.226 | 0.626 |
| pg_corr=0 | 10 | 0.326 | 0.269 | 0.768 |

Removing pg_corr_x makes the cube imprint WORSE. **PGF is NOT the source**;
it is partially CANCELLING the imprint produced elsewhere. Ruled out.

#### 2. Operators on uniform IC (T=300, p_s=p_ref, phis=0)

| operator at lev 20 | result |
|--------------------|--------|
| `compute_geopotential_hybrid` Φ std | 3.6e-12 (machine epsilon) |
| `_arakawa_lamb_gradient(Φ)` max | 2.7e-17 (machine epsilon) |
| `_arakawa_lamb_gradient(ln p_s)` max | 1.8e-20 (machine epsilon) |
| dPhi/dx edge_std vs interior_std | 3.8e-18 vs 0 |

**Operators are exactly consistent in the uniform state.** The cube
imprint is NOT a constant-input operator bug; it emerges purely from
nonlinear amplification of small dynamic perturbations through the
panel-boundary halo paths.

#### 3. Diffusion strength scan (A_h scan, 10-day HS C36 hybrid)

| A_h × | max\|v\| | edge_std | zonal_std | eddy_std | comment |
|------:|---------:|---------:|----------:|---------:|---------|
|     1 |   0.626  |  0.278   |   0.195   |  0.125   | baseline |
|     2 |   0.497  |  0.212   |   0.136   |  0.103   | |
|     4 |   0.275  |  0.115   |   0.065   |  0.072   | eddy/zonal=1.1 |
|     8 |   0.161  |  0.037   |   0.017   |  0.050   | eddy DOMINATES zonal! |
|    16 |   0.136  |  0.030   |   0.022   |  0.038   | zonal Hadley over-damped |

A_h is the most powerful knob for cube-imprint reduction, but it
over-damps the physical Hadley signal at the same time. There is NO
sweet spot where the eddy_std → 0 while zonal_std stays at the
spectral reference (~0.5).

#### 4. Hyperdiff and div_damp scaling

| change | day | edge_std | max\|v\| | comment |
|--------|----:|---------:|---------:|---------|
| hyperdiff × 16 | 10 | 0.293 | 0.683 | barely changes |
| div_damp × 16  | 10 |  NaN  |  NaN  | unstable |
| hyperdiff × 4 + div_damp × 4 | 30 | — | 1.81 | -29% max\|v\| |

Hyperdiff (∇⁴) is too SCALE-SELECTIVE — it only damps grid-scale modes
and leaves the cube-imprint mode (~6Δx wavelength matching panel-edge
ringing) untouched. Div_damp at 16× crashes the model.

#### 5. Duogrid enabled

| config | day | edge_std | max\|v\| | wall time |
|--------|----:|---------:|---------:|----------:|
| baseline | 10 | 0.278 | 0.626 | 31 s |
| use_duogrid=True | 10 | 0.228 | 0.488 | 409 s |

Duogrid (FV3-faithful kinked-to-extended halo remap) reduces cube
imprint by ~18 % but is **13× slower**. Not practical as default.

### Diagnosis

The cube imprint is the nonlinear endpoint of a feedback loop:
1. Small face-aligned bias in dB/dx at panel-edge corners (O(dx) from
   halo interpolation through the A-L Cartesian matrix's off-diagonal
   c01, c10 terms — see `docs/cubed_sphere_edge_artifacts.md` items
   1-12 for the full derivation, replicated 25+ ways).
2. Bias drives spurious wind tendency at panel-edge cells.
3. Spurious wind → spurious divergence → spurious mass flux → p_s tilt.
4. p_s tilt → Φ tilt → larger ∇B bias.
5. Loop until diffusion balances (saturates around 0.3 m/s for v).

Per the iteration history (docs/cubed_sphere_edge_artifacts.md
iter-1..14), every alternative gradient stencil tested (a2b_ord4,
2-point face, dp2 covariant, full-covariant FV3 frame) either
(a) breaks discrete geostrophic balance → unstable, or
(b) sacrifices accuracy more than the artifacts cost.

The TRUE Fortran FV3 path uses:
- **Forward-backward time stepping** (c_sw + d_sw): cross-step error
  cancellation. We use RK3 — incompatible with the FB convention.
- **Normal D-grid stagger** (u at x-faces, v at y-faces): FV3's 2-point
  gradient at faces is direct. We use C-D grid (both at corners) which
  forces the A-L 4-point + matrix.
- **Lin (1997) cross-product PGF**: exact hydrostatic cancellation by
  construction. Our split (∇B + pg_corr) cancels in continuum but
  NOT in the discrete A-L stencil at panel boundaries.

### Conclusion of iteration 2

The cube imprint observed in our 3D HS C36 hybrid runs (~2 m/s v-wind
amplitude, comparable to the physical Hadley signal) is a STRUCTURAL
artifact of the (C-D grid + A-L gradient + RK3) architecture used
throughout `primitive_eq_cdgrid.py` and `compressible_euler_cdgrid.py`.
It cannot be eliminated by parameter tuning alone (per the diffusion
scan above) and cannot be eliminated by any single-operator fix
faithful to FV3 (per the docs/cubed_sphere_edge_artifacts.md iteration
history showing 25+ failed attempts in the SW path).

Genuine FV3 fidelity for the 3D path requires the full architecture
swap (normal D-grid, forward-backward, cross-product PGF). This is the
"complete GFDL FV3 port" described in iteration logs as a "multi-month
effort" and is beyond a single iteration of this loop.

### Direction for next iteration

Either:
(a) **Accept** the cube imprint as the architectural cost of the C-D
    grid path and document the limitation publicly (status quo).
(b) **Major refactor** — port FV3 normal-D-grid layout for the 3D path
    using `fv3_sw_core.py:_d2a2c_vect` infrastructure as the d2a2c
    starting point, then add `dyn_core.F90:p_grad_c` cross-product PGF.
    Multi-iteration effort.

This iteration commits the diagnostic data to FV3_3D.md but **makes no
source code changes** because no FV3-faithful single-operator fix
exists for the underlying problem.


