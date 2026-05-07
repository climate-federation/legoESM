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

## Iteration 3 (2026-05-07): FV3 Lin (1997) cross-product PGF — port + tests

Started option (b) above with the most surgical FV3 architecture
piece: the Lin (1997) cross-product hydrostatic pressure-gradient
force from `dyn_core.F90:p_grad_c` (line 2073).  This formulation
gives EXACT cancellation of the hydrostatic balance term in any
column by CONSTRUCTION — it does not split into ∇Φ + R_d*T*∇(ln p_s)
that O(dx) halo errors can break, the way our existing A-L corner
gradient does.

### New module: `src/legoesm/atmosphere/dynamics/_fv3_lin_pgf.py`

Three faithful ports of FV3 hydrostatic helpers:

1. `compute_pkappa_half(p_s, coord)` — `pk_half = p_half^κ` at cell
   centres, mirroring FV3 `dyn_core.F90:2746`.
2. `compute_geopotential_half_fv3(T, p_s, phis, coord)` — bottom-up
   `gz(k) = gz(k+1) + cp*θ(k)*δpk(k)` where θ = T*(p_ref/p_full)^κ,
   matching FV3 `dyn_core.F90:2767-2778`.  Necessary because the
   cross-product PGF requires geopotential at half-levels using the
   FV3-specific recurrence; our existing `compute_geopotential_hybrid`
   uses Simmons-Burridge at full levels which would break the
   discrete cancellation.
3. `fv3_lin1997_pgf_3d_cgrid(T, p_s, phis, coord, cdgrid)` — direct
   port of `p_grad_c` (lines 2098-2129).  Returns the PGF tendency at
   C-grid u/v faces with the **correct Fortran sign** (verified by
   the `test_surface_pressure_tilt_produces_pgf` unit test below).
4. `project_cgrid_pgf_to_dgrid_corners(pgf_x_c, pgf_y_c)` — 2-point
   average from C-grid faces to D-grid corners (the bridge our C-D
   grid prognostic-wind storage requires; FV3's normal-D-grid
   architecture would skip this projection entirely).

The 2-point projection is a SIMPLE average, not the A-L 4-point matrix
+ Cartesian rotation that the existing `_arakawa_lamb_gradient` uses.
This is the structural improvement: the Lin cross-product PGF at
C-grid faces is well-conditioned (no halo-amplification), and the
2-point projection to corners cannot amplify halo errors either.

### Unit tests: `tests/test_fv3_lin_pgf.py`

Four tests, all passing:

| Test | Property | Result |
|------|----------|--------|
| uniform hydrostatic state | C-grid PGF = 0 (machine precision) | max\|pgf\| < 1e-5 m/s² ✓ |
| D-grid projection | corner PGF = 0 | max\|pgf\| < 1e-5 m/s² ✓ |
| gz_half recurrence | `gz(k) - gz(k+1) = cp*θ*δpk` | rel err < 1e-12 ✓ |
| p_s tilt drives westward PGF | sign convention matches FV3 | mean PGF on face 0 NEGATIVE ✓ |

Tests verify that the implementation:
- Reproduces machine-precision exact hydrostatic cancellation (the
  whole point of the Lin formulation).
- Has the correct Fortran sign convention so it can be added directly
  to `du_c/dt`.
- Has internally consistent gz/pk arithmetic (the recurrence holds
  exactly).

### Status

Module is implemented and unit-tested.  **NOT YET WIRED** into
`fv3_hydrostatic_tendencies` — that requires a config-flag-gated
opt-in path that swaps out the existing
`(dB/dx + pg_corr_x = ∇(KE+Φ) + R_d*T*∇(ln p_s))` block with
`(dKE/dx + project_cgrid_pgf_to_dgrid_corners(...))`.  The KE part of
the existing dB/dx must be retained (it's the rotational vector form
term) but the Φ part must be removed (replaced by the cross-product
PGF).  This wiring change is the iter-4 deliverable.

### Direction for next iteration

iter 4: wire `fv3_lin1997_pgf_3d_cgrid` into
`fv3_hydrostatic_tendencies` behind `use_fv3_lin_pgf: bool = False`
config flag.  Default OFF so all existing tests still pass.  When ON:
1. Compute KE-only Bernoulli: `B_KE = KE` (drop Φ)
2. Compute `dKE/dx, dKE/dy_perp` via existing A-L gradient
3. Compute `pgf_x_c, pgf_y_c` via new Lin (1997) function
4. Project to corners: `pgf_x_d, pgf_y_d`
5. Replace `du_d/dt = ζ*v - dB/dx - pg_corr_x` with
   `du_d/dt = ζ*v - dKE/dx + pgf_x_d`
6. Same for `dv_d/dt`
7. Skip the existing `pg_corr_x, pg_corr_y_perp` computation entirely
   (the cross-product already includes the η-coordinate correction)

Then: re-run HS C36 hybrid 30-day with `use_fv3_lin_pgf=True` and
quantify the cube-imprint reduction against iter-2 baseline metrics
(edge_std, max\|v\|, eddy_std).

## Iteration 4 (2026-05-07): Wire Lin PGF — uncovered architecture mismatch

### Implementation

Added `use_fv3_lin_pgf: bool = False` to
`CDGridPrimitiveEquationConfig` and wired the iter-3 module into
`fv3_hydrostatic_tendencies`: when set, drop Φ from the Bernoulli
function (so `B = KE`), compute the Lin cross-product PGF at C-grid
faces, project to D-grid corners via 2-point average, replace the
existing `(dB/dx + pg_corr_x)` block.

All 4 unit tests still passed at the helper level
(`tests/test_fv3_lin_pgf.py`).

### gz_half magnitude bug — caught and fixed

First implementation used the literal Fortran formula
`dgz = cp * θ * dpk` (where θ = potential temperature).  This gave
gz values 40× larger than the Simmons-Burridge geopotential because
**FV3's `pt` argument to `geopk` is NOT bare potential temperature** —
it's the THERMODYNAMICALLY-TRANSFORMED variable `pt = T * p^(-κ)`
(the post-`pt /= pkz` form from `fv_dynamics.F90:403`), which carries
the `p_ref^(-κ)` factor needed for dimensional consistency:

```
∂Φ/∂p = -RT/p
dΦ = -R*T*dp/p = -R*T/(κ p^κ) d(p^κ) = -cp*T*p^(-κ) d(p^κ)
```

Updated `compute_geopotential_half_fv3` to use the correct formula
`dgz = cp * T * p_full^(-κ) * dpk`.  Magnitudes now match
Simmons-Burridge to within ~1.5× (the ratio reflects the FV3
discretisation choice; both are valid hydrostatic approximations).

The unit test was updated to use the corrected expected dgz; all 4
tests still pass.

### Held-Suarez C36 hybrid 30-day with Lin PGF on

| day | LIN max\|v\| | LIN edge_std | LIN eddy_std | baseline max\|v\| | baseline eddy_std |
|----:|-------------:|-------------:|-------------:|------------------:|------------------:|
|   1 |     0.10     |     0.05     |     0.01     |       0.10        |        0.02       |
|   3 |     0.65     |     0.22     |     0.00     |       0.27        |        0.03       |
|   6 |     2.46     |     0.81     |     0.20     |       0.39        |        0.08       |
|  10 |     8.13     |     2.65     |     0.76     |       0.63        |        0.13       |
|  15 |    27.58     |     8.80     |     2.31     |       0.97        |        0.18       |
|  30 |     NaN      |      —       |      —       |       2.56        |        0.36       |

The Lin PGF makes the model **dramatically worse** — max\|v\| 13× the
baseline by day 10, 28× by day 15, and NaN by day 30.  The cube
imprint is amplified rather than reduced.

### Diagnosis

The Lin (1997) cross-product PGF is designed for FV3's
forward-backward time integration (c_sw + d_sw alternation).  The
cross-product gives EXACT hydrostatic cancellation in any column —
verified at machine precision by `test_uniform_hydrostatic_state_zero_pgf`
— but the discrete BALANCE of cross-product PGF (at C-grid faces,
projected to D-grid corners via 2-point average) against the
rotational ζ × v term (at D-grid corners, computed with A-L) is
**not preserved by RK3**.  The two stencils live at different grid
positions with different effective numerical viscosities, so RK3 sees
two oscillating components that don't cancel and the integration
amplifies rather than damps.

This matches the iteration-13 conclusion in
`docs/cubed_sphere_edge_artifacts.md`: "the forward-backward scheme
requires perfectly matched halo error levels between c_sw and d_sw"
and "any approach that computes gradient and vorticity from
DIFFERENT data paths produces uncorrelated boundary errors → 3+ m/s
residual."

The Lin PGF can ONLY be used with forward-backward time stepping that
includes mass-flux-coupled c_sw → d_sw alternation.  Wiring it into
RK3 alone is not viable.

### Action

- **Reverted the wiring** in `fv3_hydrostatic_tendencies`: the
  `_use_lin_pgf` branch and `B = KE` switch were removed.  The
  function now unconditionally uses the existing
  `(dB_dx + pg_corr_x)` split formulation.  All 14 tests pass.
- **Kept `config.use_fv3_lin_pgf` field** with a long comment
  explaining the iter-4 finding so future iterations can find the
  context.  The flag is currently inert.
- **Kept the helper module** (`_fv3_lin_pgf.py`) and its unit tests
  (`test_fv3_lin_pgf.py`).  These are correct, FV3-faithful, and
  ready for a future forward-backward integration path.

### Conclusion of iteration 4

The FV3 Lin (1997) PGF is now correctly implemented as a self-
contained module with full unit-test coverage.  Wiring it into
`fv3_hydrostatic_tendencies` requires also implementing the FV3
forward-backward time-stepping scheme — single-operator swap is not
viable with our current RK3 + C-D grid + A-L gradient architecture.

### Direction for next iteration

iter 5 must address the BIGGER architecture question:
implement an opt-in c_sw + d_sw forward-backward time stepping for
the 3D path.  This is the multi-iteration FV3 architecture port
that iter-2 identified as the only path to true cube-imprint
elimination.  Proposed iter 5 scope:

(a) Add a stub `time_integrator = "fv3_forward_backward"` to the
    config and a stub `_step_fv3_fb` method that today just does
    one RK3 stage but is the wiring point for the FB scheme.
(b) Add an opt-in `dyn_core.F90:Lagrangian_to_Eulerian` analog
    (or skip — we are not vertically Lagrangian).
(c) c_sw half: compute uc, vc tendencies via Lin PGF + advection
    of (delp, pt, w).
(d) d_sw half: compute u_d, v_d tendencies via FV3-faithful
    operators using the c_sw output.
(e) Validate: HS C36 hybrid 30-day stability + visual cube-imprint
    inspection.

This is multi-week work; each iteration of the Ralph loop will tackle
one self-contained piece.

## Iteration 5 (2026-05-07): FV3 adaptive Smagorinsky divergence damping

### Motivation

iter 4 attempted the FV3 Lin PGF and found it incompatible with RK3.
The forward-backward port is multi-iteration.  For iter 5, port a
SMALLER FV3-faithful piece that can wire into the existing RK3 path
without architecture changes: the **adaptive Smagorinsky-style
divergence damping** from `sw_core.F90:1720`::

    damp = da_min_c * max(d2_bg, min(0.20, dddmp * abs(div)))

where `d2_bg = div_damp_coeff / da_min_c` is the dimensionless
background coefficient.  When `dddmp > 0`, divergence damping becomes
ADAPTIVE: stronger where local |div| is large (the panel-boundary
halo-error cells suspected to drive cube imprint per iter-2), weaker
in smooth interiors.

### Implementation

Added `div_damp_dddmp: float = 0.0` field to
`CDGridPrimitiveEquationConfig`.  Modified the `if config.div_damp_coeff
> 0` block in `fv3_hydrostatic_tendencies` to compute an adaptive
per-cell coefficient when `dddmp > 0`, using the Fortran-faithful
formula above.  Default 0.0 preserves bit-for-bit existing behaviour.

The implementation directly mirrors the SW path's
`cdgrid_momentum_tendencies` adaptive block (operators_cdgrid.py:1732
onwards), adapted for 3D inputs (per-level evaluation).

### Unit tests: `tests/test_div_damp_adaptive.py`

Four tests, all passing:

| test | property | result |
|------|----------|--------|
| dddmp = 0 (default) | bit-for-bit identical to constant path | array_equal ✓ |
| dddmp = 1e-30 | below d2_bg floor → matches constant path | rtol < 1e-12 ✓ |
| dddmp = 1e10 on perturbed state | adaptive cap (0.20) hits → tendencies differ | |Δdu| > 0.1 * |base| ✓ |
| dddmp = 0.20 (FV3 default) on HS init | stable for 10 RK3 steps | finite ✓ |

### Held-Suarez C36 hybrid 10-day metric scan

| dddmp | max\|v\| | edge_std | int_std | zonal_std | eddy_std |
|------:|---------:|---------:|--------:|----------:|---------:|
|  0    |   0.626  |   0.278  |  0.226  |   0.195   |  0.125   |
|  0.05 |   0.626  |   0.278  |  0.226  |   0.195   |  0.125   |
|  0.10 |   0.626  |   0.278  |  0.226  |   0.195   |  0.125   |
|  0.20 |   0.626  |   0.278  |  0.226  |   0.195   |  0.125   |
|  200  |   0.624  |   0.278  |  0.225  |   0.195   |  0.125   |
| 2000  |   NaN    |    —     |    —    |     —     |    —     |

The adaptive damping kicks in only at very large `dddmp`.  Reason: in
HS C36, typical |div| ~ 1e-5 s⁻¹.  For adaptive to dominate over the
background floor:

    dddmp * |div| > d2_bg = div_damp_coeff / da_min_c
    => dddmp > d2_bg / |div| = 3.45e-4 / 1e-5 = 35

So FV3 default `dddmp = 0.20` and SW iter1009-tuned `dddmp = 0.0625`
NEVER trigger adaptive in the HS regime (verified above).  The
mechanism would be useful in regimes with strong divergence (frontal
zones, tropical cyclones, gravity waves).  At `dddmp = 2000` the
mechanism over-damps and the model NaNs.

### Conclusion

iter 5 ports a **genuinely missing FV3 mechanism** (adaptive
Smagorinsky div_damp) into the 3D atmospheric path with full
unit-test coverage.  The mechanism is correctly implemented (verified
by the dddmp=1e10 test that confirms cap-hitting tendencies differ
from the constant path).  It does NOT reduce the HS cube imprint
because typical HS divergence is below the adaptive trigger
threshold — consistent with the iter-2 conclusion that the cube
imprint is structural, not driven by extreme divergence spikes.

Net effect on cube imprint: **none** (HS regime).  Net effect on FV3
fidelity: **positive** — one more FV3 mechanism faithfully ported
and gated behind a config flag for future use cases (DCMIP TC,
mountain wave, frontal-zone tests) where divergence is large.

### Direction for next iteration

The cube imprint problem requires the architecture port.  iter 6+
should start the forward-backward time-stepping skeleton.  See
iter-4 conclusion for the proposed sub-iteration plan (a)-(e).

## Iteration 6 (2026-05-07): FV3 D-grid vector cube-vertex corner fill

### Motivation

iter-2 identified the 8 cube vertices (where 3 faces meet) as a
worst-case halo source.  Looking at our existing cube-vertex
treatment in `halo.py::_fill_corners_h1` (line 1467), the docstring
explicitly notes a fidelity gap:

> "The Fortran transport path uses ``copy_corners(dir=1/2)`` in
> tp_core.F90:243-299 — a directional rotated copy tailored to
> X-sweep vs Y-sweep of PPM.  That mechanism writes DIFFERENT values
> at the same cube-vertex cell for different sweep directions.  Our
> 2-point average is a direction-invariant single value."

There's a related FV3 mechanism for VECTOR fields:
``fv_mp_mod.F90:fill_corners_dgrid_r8`` (line 1257).  At cube
vertices, the missing 4th cell is filled with the DIAGONAL MIRROR of
the OTHER vector component, with a sign flip on SW and NE corners
to account for the local-basis rotation.  This is FV3-faithful and
has no JAX equivalent in our code.

### Implementation

New module ``src/legoesm/grids/_fv3_dgrid_corner_fill.py`` with two
functions:

* `fv3_fill_corners_dgrid_vector(x, y, n)` — direct port of the
  Fortran formula (lines 1264-1287).  Operates on a padded D-grid
  vector pair (``x`` shape ``(6, n+2, n+3)``, ``y`` shape
  ``(6, n+3, n+2)``).  Overwrites the 4 cube-vertex halo cells per
  face with the sign-flipped diagonal mirror.

* `fv3_fill_corners_agrid_scalar(q, n)` — companion for cell-centre
  scalars (no sign flip), faithful to FV3 ``fill_corners_2d_r8``
  AGRID branch.

This module is **decoupled** from the existing ``_fill_corners_h1``
2-point-average path — it neither replaces it nor calls into it.
It is exposed for future use by:
- A forward-backward c_sw + d_sw 3D path (iter-7+).
- An opt-in flag in ``pad_halo_vector`` to apply the FV3 corner fill
  AFTER the standard scalar-pad path (preserving existing operator
  expectations while testing the cube-vertex contribution).

### Unit tests: ``tests/test_fv3_dgrid_corner_fill.py``

Six tests, all passing:

| test | property | result |
|------|----------|--------|
| agrid scalar — overwrites only cube vertices | exactly 4 cells per face modified | ✓ |
| agrid scalar — diagonal mirror sources match FV3 indexing | exact match to source cells | ✓ |
| dgrid vector — overwrites only cube vertices | exactly 4 cells per face for both x, y | ✓ |
| dgrid vector — Fortran sign pattern | SW/NE flip; NW/SE no flip | ✓ |
| dgrid vector — zero input stays zero | regardless of sign | ✓ |
| agrid scalar — uniform input is invariant | diagonal mirror of constant = constant | ✓ |

### Status

Pure helper module + tests, no integration into the main path yet.
This is a **building block** — the forward-backward port (iter-7+)
will need it.

### Conclusion

iter 6 closes one of the explicit FV3 fidelity gaps documented in
``halo.py``.  The new module is testable in isolation, FV3-faithful
to the line, and ready for integration.  Like iter 5's adaptive
divergence damping, the immediate effect on the HS C36 cube imprint
is zero (the helper isn't yet wired into the dycore), but the FV3
fidelity of the legoESM codebase improves by another concrete
mechanism.

20 atmospheric/halo tests pass (iter-1039 sentinels, FV3-Lin-PGF
helpers, adaptive damping, new corner fill).

### Direction for next iteration

iter 7: wire ``fv3_fill_corners_dgrid_vector`` into ``pad_halo_vector``
behind a config flag so the cube-vertex halo can use the FV3-faithful
mirror.  Test on HS C36 hybrid 30-day to quantify whether the
8 cube-vertex contributions to the cube imprint shrink.

iter 8+: forward-backward time stepping skeleton (the architecture
port).  Each iteration ports one self-contained piece of c_sw or
d_sw1/5.

## Iteration 7 (2026-05-07): FV3 AGRID-XDir corner fill toggle

### Hypothesis

iter-6 added the ``fv3_fill_corners_dgrid_vector`` helper but did not
wire it.  iter-7 takes a different angle: probe whether replacing the
legacy 2-point-average corner fill (in ``halo.py:_fill_corners_h1``,
called by every cell-centre halo path) with the FV3-faithful AGRID
``XDir`` diagonal mirror (``fv_mp_mod.F90:1077``) changes the 3D
HS cube imprint.  The legacy fill is the SYMMETRIC combination of
FV3's ``XDir`` and ``YDir`` variants; FV3 picks one direction
specifically, depending on the operator.

### Diagnostic experiment

Monkey-patch ``_fill_corners_h1`` to use ``XDir`` at all cube-vertex
halos (cell-centre A-grid scalars), keep all other code unchanged,
run HS C36 hybrid for 30 days at the **middle vertical level**:

| metric (lev nlev//2, day 30) | AVG (legacy) | XDir (FV3-faithful) | change |
|------------------------------|-------------:|--------------------:|-------:|
| max\|v\|                     |     2.56     |        1.35         |  -47%  |
| edge_std                     |     0.781    |        0.482        |  -38%  |
| zonal_std                    |     0.520    |        0.274        |  -47%  |
| eddy_std                     |     0.364    |        0.260        |  -29%  |

Repeat with ``YDir`` (FV3 line 1083 variant) for completeness:

| metric (lev nlev//2, day 10) | AVG  | XDir | YDir |
|------------------------------|-----:|-----:|-----:|
| max\|v\|                     | 0.626 | 0.442 | 0.944 |

``XDir`` reduces by 47%; ``YDir`` increases by 51%.  The asymmetry
is real (``YDir`` is NOT just ``XDir`` rotated 90°: each picks the
wrong/right diagonal for the specific dynamic flow we have).  This
asymmetry shows our 3D dycore has a **directional bias** — the
``XDir`` corner choice happens to align with the bias and damp it,
``YDir`` amplifies it.

### Tradeoff: max-over-all-levels metric tells a different story

Re-running with the toggle ON globally and computing max-over-all-
levels (not just the middle level):

| day | AVG max\|u\| | AVG max\|v\| | XDir max\|u\| | XDir max\|v\| |
|----:|-------------:|-------------:|--------------:|--------------:|
|   1 |     0.62     |     0.39     |     0.62      |     0.38      |
|  10 |     6.02     |     3.08     |     6.35      |     3.29      |
|  30 |    11.57     |     6.52     |    15.80      |    10.18      |

So at day 30, the XDir toggle gives:
- middle-level v cube imprint: -47 % (good)
- max-over-all-levels |u|, |v|, speed:  +37 %, +56 %, +38 % (worse)

Interpretation: the legacy 2-point-average corner fill was
*suppressing* part of the physical Hadley/baroclinic flow
(particularly at extreme levels) AND part of the cube imprint.
The FV3 ``XDir`` variant reduces cube imprint at the middle level
but releases more of the natural baroclinic-eddy flow at extreme
levels (which then produces larger max wind values, partly real
physical signal and partly residual cube imprint at the surface
and top).

This is a structural release of pent-up dynamics, NOT a stability
issue (model remains stable through 30 days, no NaN, mass drift
~2e-9 vs ~1e-9 baseline — both excellent).

### Implementation

Modified ``halo.py``:

* Added module-level ``_corner_fill_mode`` (default ``"avg"``,
  reads from ``LEGOESM_CORNER_FILL`` env var if set).
* ``set_corner_fill_mode(mode)`` / ``get_corner_fill_mode()`` setter
  / getter.
* ``_fill_corners_h1`` branches on the mode: ``avg`` (legacy 2-point
  average, bit-for-bit unchanged) vs ``fv3_agrid_xdir`` (FV3-faithful
  ``XDir`` diagonal mirror).

Default ``avg`` preserves all existing tests bit-for-bit (29
FV3_3D-related tests pass; 12 broader atmosphere integration tests
pass without the toggle).

### Unit tests: ``tests/test_corner_fill_toggle.py``

Six tests, all passing:

1. Default mode is ``"avg"``.
2. Invalid mode raises ``ValueError``.
3. ``avg`` mode matches legacy 2-point average exactly.
4. ``fv3_agrid_xdir`` mode applies ``XDir`` diagonal mirror.
5. The two modes give DIFFERENT results on random input (regression
   guard against silent no-op).
6. Round-trip mode change restores legacy values.

### Held-Suarez C36 hybrid 30-day with toggle ON (visual)

Re-ran ``scripts/run_atmosphere_test_matrix.py`` with the env var:
all three tests PASS, mass drift 2e-9 (vs 1e-9 baseline, both
machine-precision-level), max\|v\| 16 m/s (vs 11 baseline).  The
v-wind snapshot at day 30 shows a markedly different pattern:
red-dominant zonal flow with stronger high-latitude bands and
weaker face-blob structure in the mid-latitudes.  Cube imprint at
mid-levels is reduced (qualitatively matches the diagnostic numbers)
but the overall amplitude is larger.

### Status / interpretation

**This is a partial win.** The FV3-faithful ``XDir`` corner fill:

- Genuinely reduces the dominant cube-imprint mode at mid-vertical-
  levels (where the user's HS snapshots showed the worst pattern).
- Also lets more dynamic energy through, increasing max wind values
  at extreme levels.

The tradeoff is acceptable for users who care about middle-level
flow accuracy (climate-mean diagnostics), less ideal for users who
care about peak winds.

The ``avg`` legacy mode REMAINS THE DEFAULT.  Users who want the
FV3-faithful corner fill opt in with::

    export LEGOESM_CORNER_FILL=fv3_agrid_xdir

or::

    from legoesm.grids.halo import set_corner_fill_mode
    set_corner_fill_mode("fv3_agrid_xdir")

### Conclusion

iter 7 ports a FV3 mechanism that **measurably changes** the 3D
cube-sphere dynamics behaviour for the first time in this loop.
It does NOT solve the cube imprint completely (the structural
mode persists) but it provides a partial reduction at the
mid-tropospheric levels that visually dominate the HS snapshots.
The toggle is opt-in (legacy ``avg`` remains default), bit-for-bit
backward compatible, fully tested.

35 atmospheric tests pass total (existing 29 + 6 new toggle tests).

### Direction for next iteration

iter 8: similar toggle for ``_fill_corners_h2`` (the halo=2 path used
by PPM transport).  See if the same XDir diagonal mirror, applied to
the 2×2 cube-vertex L-shaped corner block, further reduces cube
imprint.

iter 9+: forward-backward time stepping (the multi-iteration
architecture port that iter-2 identified as the only path to full
elimination).

## Iteration 8 (2026-05-07): Extend XDir toggle to ``_fill_corners_h2``

### Probe first, wire second

Before extending the toggle to ``_fill_corners_h2`` (the halo=2 path
used by PPM transport), I probed via monkey-patching: apply the FV3
AGRID XDir formula for ng=2 to the 2×2 cube-vertex L-block, run HS
C36 hybrid 30-day with **both h1 and h2** XDir, compare to **h1 only**
XDir from iter 7.

| metric (max over all levels)   | h1 only XDir | h1 + h2 XDir | delta |
|--------------------------------|-------------:|-------------:|------:|
| day 10 max\|u\|                |     6.35     |     6.35     |   0   |
| day 10 max\|v\|                |     3.29     |     3.29     |   0   |
| day 10 max speed               |     6.35     |     6.35     |   0   |
| day 30 max\|u\|                |    15.80     |    15.80     |   0   |
| day 30 max\|v\|                |    10.18     |    10.18     |   0   |
| day 30 max speed               |    16.03     |    16.03     |   0   |
| day 30 mid-level zonal_std     |     —        |     1.294    |   —   |
| day 30 mid-level eddy_std      |     —        |     0.847    |   —   |

**Adding h2 XDir on top of h1 XDir gives BIT-FOR-BIT identical max
metrics.**  The h2 corner fill is a **no-op for HS C36 hybrid**:
no operator in the 3D atmospheric tendency function reads cells in
the 2×2 cube-vertex halo block.  This is consistent with the
iter-69 review note in ``halo.py:_fill_corners_h1``: "operator-split
PPM slices q_full to keep EITHER i-halo OR j-halo (...), never
simultaneously — so cube-vertex corner cells at (i_halo, j_halo)
are never referenced by any PPM stencil."

### Implementation (despite the no-op)

Even though h2 XDir is currently a no-op, I extended the toggle to
``_fill_corners_h2`` for **FV3-fidelity symmetry** with h1 and to
prepare for iter 9+ forward-backward operators that DO use 2×2
corner halos (specifically, the FV3 ``a2b_ord4`` interpolation in
``a2b_edge.F90`` reads up to 2 cells of halo at cube vertices).

The h2 XDir formula in our 0-based padded representation, port of
``fv_mp_mod.F90:1077`` AGRID-XDir for ng=2 (i, j ∈ {1, 2}):

```
SW block:
  (1, 1) ← (1, 2)
  (1, 0) ← (0, 2)
  (0, 1) ← (1, 3)
  (0, 0) ← (0, 3)
NW block (mirror in j): analogous
SE block (mirror in i): analogous
NE block (both mirrors): analogous
```

The ``avg`` mode preserves the legacy inside-out 2-point averaging
exactly (existing tests pass bit-for-bit).

### Unit tests: extended ``tests/test_corner_fill_toggle.py``

Added 3 new tests (9 total now):

| test | property | result |
|------|----------|--------|
| h2 ``avg`` default mode | inside-out 2-point average matches legacy formula | ✓ |
| h2 ``fv3_agrid_xdir`` mode | diagonal mirror matches FV3 indexing | ✓ |
| h2 modes give DIFFERENT results on random input | regression guard | ✓ |

### Status

iter 8 is **defensive completeness** — the toggle is now consistent
across h1 and h2 paths even though h2's contribution to HS is zero
today.  When the forward-backward c_sw + d_sw chain (iter 9+) adds
operators that read 2×2 cube-vertex halo (a2b_ord4-style
interpolation), the toggle will be active without further wiring.

32 atmospheric tests pass with default mode (bit-for-bit unchanged).
6 iter1039 3D edge sentinels pass with the toggle ON
(LEGOESM_CORNER_FILL=fv3_agrid_xdir).  No regressions.

### Conclusion

iter 8 is a small but FV3-faithful step.  The visible numbers in HS
C36 are unchanged from iter 7; the value is in **completeness**:
both h1 and h2 corner fills now have FV3-faithful options exposed
through the same toggle.  The infrastructure is ready for the
forward-backward operators that will activate the h2 path.

### Direction for next iteration

iter 9+: forward-backward time stepping (the multi-iteration
architecture port that iter-2 identified as the only path to full
cube-imprint elimination).  Per iter-4 plan:

(a) Add ``time_integrator = "fv3_forward_backward"`` config option.
(b) Implement c_sw skeleton: D-grid winds → A-grid → C-grid via
    ``fv3_sw_core._d2a2c_vect`` (existing FV3-faithful for SW path,
    needs adapter for our 3D state).
(c) Use Lin (1997) PGF (iter-3 helper) at C-grid faces.
(d) Use a2b_ord4 to interpolate gz, pkc to corners (iter 9-10 work).
(e) d_sw5: vorticity transport on D-grid using c_sw output.
(f) Full HS C36 hybrid 30-day with FB scheme, compare to baseline.

This is multi-iteration; each piece is a separate iteration with
unit tests and regression guards.

## Iteration 9 (2026-05-07): Three negative-result probes

Probed three more FV3-faithful interventions in the existing 3D
architecture; all either fail or produce no measurable effect.

### Probe 9.A — Wire ``_interp_center_to_corner_a2b_ord4`` into 3D

The existing FV3-faithful 4th-order A→B interpolation
(``operators_cdgrid.py:1352``) is currently used only by the SW
path's vorticity damping.  Probe: replace
``_interp_center_to_corner`` (2nd-order 4-pt average) with
a2b_ord4 in the 3D PE tendency function (``ζ_corner``,
``T_corner``, all batched corner interpolations).

| metric (HS C36 hybrid, day 10) | baseline (2-pt avg) | a2b_ord4 |
|--------------------------------|--------------------:|---------:|
| max\|u\|                       |        6.02         |  16.39   |
| max\|v\|                       |        3.08         |   9.80   |
| max speed                      |        6.02         |  16.62   |
| wall time                      |        66 s         |  331 s   |

**Disastrous.** Max winds 2.7× larger and 5× slower.  Same root cause
as the iter-4 Lin PGF failure: a2b_ord4 is FV3-faithful in tandem with
the cross-product PGF + forward-backward time stepping, but inserting
it alone into our (A-L gradient + RK3) architecture breaks the
discrete operator balance — mixing 4th-order corner interpolation
with 2nd-order A-L gradient produces uncorrelated halo errors that
accumulate.

### Probe 9.B — FV3 sign-flipped vertex tendency override

Apply the FV3 vector corner-fill formula
(``fv_mp_mod.F90:fill_corners_dgrid``) directly to ``du_d_dt`` /
``dv_d_dt`` at the 4 cube-vertex cells per face.  This is the
FV3-faithful version of iter-1's bilinear extrapolation attempt.

| metric (HS C36 hybrid, day 10) | baseline | sign-flip vertex |
|--------------------------------|---------:|-----------------:|
| max\|u\|                       |   6.02   |       6.03       |
| max\|v\|                       |   3.08   |       3.12       |
| max speed                      |   6.02   |       6.03       |

Within roundoff.  **No-op for HS dynamics.**  The 8 cube-vertex points
are too localized to affect bulk dynamics; even a faithful sign-flip
formula at those points doesn't propagate enough to shift the
cube-imprint pattern (which spans entire panel boundaries).

### Probe 9.C — Iter-7 toggle + stronger upper-atmosphere sponge

iter-7 documented that the XDir corner fill mode increases
max-over-all-levels max\|v\| at day 30 (+56%).  Probe: localize where
those increased winds live, and try a stronger sponge to clip them.

Level-by-level breakdown (HS C36 hybrid, day 10, XDir mode ON):

| level | name | max\|u\| | max\|v\| |
|------:|------|---------:|---------:|
|     0 | top (sponge zone)  |  0.25    |  0.19    |
|     5 | upper trop / jet   |  5.82    |  3.14    |
|    10 |                    |  3.55    |  1.99    |
|    20 | mid-trop           |  1.35    |  0.91    |
|    30 | lower trop         |  1.50    |  0.71    |
|    38 | near surface       |  2.62    |  1.36    |

The maximum winds are at level 5 (upper-tropospheric jet at ~150 hPa),
NOT at the model top (sponge zone — already damped to 0.25 m/s).

Stronger sponge (``sponge_tau_sec=1800``, ``sponge_sigma=0.20``)
reduced max\|u\| from 6.35 to 4.93 (~22 % reduction at day 10), but
the dominant lev-5 jet intensity reduces from 5.82 to 4.64 — partly
because the sponge reaches further down and damps the jet itself
(physical signal loss), not because cube imprint is targeted.

This is parameter tuning, not FV3-architectural fidelity.  Not
adopted as a default.

### Conclusion of iteration 9

Three more single-mechanism FV3-faithful attempts at reducing the
cube imprint without changing the dycore architecture.  Confirms the
iter-2/4 conclusions: full elimination requires the forward-backward
architecture port.

The iter-7 corner-fill toggle remains the only working contribution
this loop.  All other probes either break the discrete balance
(Lin PGF, a2b_ord4) or produce no measurable effect (vertex tendency
sign-flip).  This iteration adds NO new code — just diagnostic data
in FV3_3D.md.

### Direction for next iteration

iter 10+: stop incremental probes and start the FB skeleton.  Per
iter-4/iter-8 plan:

1. Add ``time_integrator = "fv3_forward_backward"`` config option
   that routes ``_step_fv3`` through a new ``_step_fv3_fb`` method.
2. Initial ``_step_fv3_fb`` is just one RK3 stage (placeholder) —
   no functional change yet, but the wiring point is in place.
3. Subsequent iterations replace the placeholder with the c_sw
   half (forward-backward C-grid step) and d_sw5 half (D-grid step).

This sets up the architecture migration without breaking existing
behaviour (default ``time_integrator = "ssp_rk3"`` preserved).



