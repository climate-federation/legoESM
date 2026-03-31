# Cubed-Sphere Edge Artifacts: Investigation Summary

**Date**: 2026-03-30 to 2026-03-31
**Investigators**: Pierre Gentine, Claude Opus 4.6
**Branch**: `main` (metrics merged), `duo-grid-edge-fix`, `fv3-forward-backward`, `fv3-dsw-port`, `fv3-clean-room` (experiments)

## Problem Statement

The cubed-sphere shallow water solver shows v-wind artifacts of ~0.6 m/s at C36 along all 12 cube-panel edges after 1 day of integration (Williamson Test Case 2). These artifacts are visible in v-wind snapshots as vertical streaks at panel boundaries.

## Root Cause

The Arakawa-Lamb (A-L) gradient at D-grid corners uses a 4-point cell-centre stencil with a precomputed 2x2 transformation matrix. At face boundaries, two of the four cell centres come from the neighbouring panel via scalar halo exchange. The halo exchange introduces O(dx^2) interpolation error, and the transformation matrix amplifies this to O(dx) gradient error. This O(dx) error does not cancel with the vorticity term (which uses a different code path), producing persistent edge artifacts.

## Current Baseline Performance

| Test Case | Resolution | Duration | L2 Error | Mass Drift | Notes |
|-----------|-----------|----------|----------|------------|-------|
| Williamson 2 | C36 | 1 day | 3.51e-03 | ~8e-5 | Edge artifacts visible in v-wind |
| Williamson 5 | C36 | 1 day | — | 2.81e-05 | Mountain flow, edge artifacts masked by physical signal |

Configuration: A_h=1e4, hyperdiff_coeff=dx^4/(86400*10), SSP-RK3 time stepping.

The v-wind artifacts manifest as ~0.6 m/s streaks along panel boundaries. The `_extrapolate_boundary_corners` function replaces the 4 vertex corner tendencies with interior values, bounding but not eliminating the artifacts. Wind speed and height fields show no visible cube imprint.

## Approaches Tested

### 1. Duo-Grid Sub-Grid Metrics (sin_sg / cos_sg)

**Idea**: Compute the non-orthogonality angle at 9 sub-grid positions per cell (Mouallem, Harris & Chen 2023). Use upwind sin_sg/cos_sg at face boundaries to eliminate the metric averaging that loses accuracy.

**Result**: Metrics computed correctly (sin^2+cos^2=1, boundary discontinuity confirmed). But applying them to the D-to-C projection made things worse:
- Upwind cos_sg everywhere: +10% L2 (interior metric noise from 0.0005 stencil-width diff)
- Upwind cos_sg at boundaries only: not tested in isolation

**Lesson**: The sin_sg/cos_sg metrics are correct but the edge artifacts come from the gradient halo exchange, not from the D-to-C projection metrics.

### 2. Upgraded Halo Interpolation (Linear to Quadratic)

**Idea**: The halo=1 exchange uses linear interpolation (O(dx^2) error). Upgrade to 3-point Lagrange (O(dx^3) error) to reduce gradient error to O(dx^2).

**Result**: Much worse (L2 from 0.031 to 0.056 at C16). The Lagrange interpolation overshoots near cube corners where the strip values have steep gradients.

**Lesson**: Higher-order halo interpolation requires monotonicity constraints; unconstrained Lagrange is unstable near cube vertices.

### 3. Face-Local Extrapolation of Halo Values

**Idea**: Replace cross-face halo values in the gradient stencil with face-local linear extrapolations from interior cells.

**Result**: Worse (L2 from 0.031 to 0.035). The extrapolated values lose the actual cross-face field information, which is more accurate than the extrapolation despite the position error.

**Lesson**: The cross-face halo exchange provides correct field values at slightly wrong positions. This is better than wrong values at correct positions.

### 4. Extended Boundary Tendency Fix

**Idea**: Copy boundary tendencies from interior (not just 4 vertex corners, but entire boundary rows/columns).

**Result**: Full row copy: L2 = 0.125 (much worse, creates stiff boundary layer). 50% blend: L2 = 0.035 (still worse).

**Lesson**: The `_extrapolate_boundary_corners` vertex-only fix is locally optimal. Extending it to full rows creates numerical artifacts.

### 5. 4th-Order Cell-Centre-to-C-Grid Interpolation

**Idea**: Upgrade fv3_cc2c from 2nd-order to 4th-order Lagrange with FV3-style c1/c2/c3 boundary stencils.

**Result**: 10x more divergence in C-grid velocities (mass tendency 0.009 vs 0.0009 baseline). The one-sided boundary stencils amplify halo noise in the v-component at j=0/j=n faces.

**Lesson**: Higher-order interpolation near face boundaries amplifies halo errors. The simple 2nd-order average is more robust.

### 6. 2-Point C-Grid Bernoulli Gradient (Replacing A-L)

**Idea**: Compute the Bernoulli gradient at C-grid face positions as a simple 2-point divided difference of haloed cell-centre values (no transformation matrix). Project to D-grid edge midpoints via 4-point average.

**Result**: 60x worse du (0.006 vs 0.0001). The 4-point stagger projection from C-grid faces to D-grid edge midpoints breaks the geostrophic cancellation between gradient and vorticity.

**Key Insight**: Gradient and vorticity MUST be at the same stagger position. The Arakawa-Lamb gradient at corners + corner vorticity = consistent. C-grid gradient + corner vorticity = inconsistent.

### 7. C-Grid Gradient + C-Grid Vorticity (Projected Sum)

**Idea**: Compute BOTH gradient and vorticity at C-grid face positions (consistent stagger), then project the TOTAL tendency (which is near zero for balanced flow) to D-grid edge midpoints.

**Result**: Stable, but 2x worse L2 (0.058 vs 0.031) and worse v-wind artifacts (30 vs 27 m/s boundary, 12 vs 5 m/s interior). Projecting the sum is better than projecting separately, but the 4-point C-to-D average still introduces noise exceeding the edge artifact.

### 8. dp2 Gradient Without Matrix (Covariant Convention)

**Idea**: Compute the gradient at D-grid edge midpoints as a 2-point difference of corner-interpolated B values (dp2), WITHOUT the A-L transformation matrix. Use the same 4-cell stencil but skip the matrix rotation.

**Result**: 80x worse dv (0.0014 vs 0.000017). The gradient is in the grid-aligned direction, not the physical direction. On the non-orthogonal cubed sphere, these differ by up to 25% near face corners. The covariant gradient doesn't cancel with the physical-frame vorticity.

**Key Insight**: The A-L matrix is NOT a bug — it correctly rotates the gradient to the physical direction. Removing it creates a 25% directional error that's much worse than the O(dx) halo amplification.

### 9. Full Covariant Switch (Gradient + Vorticity)

**Idea**: Convert EVERYTHING to the covariant frame — covariant vorticity (no non-orth correction in circulation), dp2 gradient (no matrix), covariant Coriolis cross-product.

**Result**: 15x worse du (0.0015 vs 0.0001), visual v-wind max 71 m/s (vs 26 baseline). The covariant balance has a larger discrete residual than the physical-frame balance on the gnomonic cubed sphere.

**Lesson**: The A-L gradient + physical vorticity provides better discrete geostrophic cancellation than the covariant gradient + covariant vorticity on the non-orthogonal cubed sphere.

### 10. FV3 Forward-Backward Time Stepping

**Idea**: Replace RK3 with FV3's native forward-backward scheme. c_sw handles nonlinear terms (KE + vorticity) at C-grid. d_sw handles linear terms (Coriolis + pressure gradient) at D-grid.

**Variants tested**:
- d_sw with A-L gradient of h_star + full dt: NaN by step 200 (A-L error accumulates via Euler instability)
- d_sw with A-L gradient of original h + full dt: NaN by step 100 (forward Euler unconditionally unstable without implicit coupling)
- d_sw with pressure only (no Coriolis): NaN by step 200 (unbalanced geostrophic gradient)
- d_sw with A-L Coriolis + pressure + dt: NaN by step 200 (A-L error mismatch between c_sw and d_sw)
- d_sw with covariant Coriolis + dp2 + dt/2: NaN by step 200 (covariant imbalance)

**Key Insight**: The forward-backward scheme requires perfectly matched halo error levels between c_sw and d_sw. Our hybrid (FV3 c_sw at C-grid + A-L d_sw at D-grid) doesn't achieve this matching. Forward Euler in d_sw amplifies the O(dx) residual exponentially.

### 11. Velocity Convention Mismatch Discovery

**Finding**: FV3 uses covariant C-grid velocities (uc = interpolated covariant utmp, no sin/cos correction). Our code uses physical face-normal C-grid velocities (uc = utmp*sin(a) - vtmp*cos(a)). These are fundamentally different quantities. The existing PPM mass transport expects physical face-normal velocities; FV3's mass transport uses sin_sg flux scaling with covariant velocities.

Mixing conventions (FV3 d2a2c covariant uc with our PPM) produces 140x worse mass divergence. Converting covariant to physical before PPM didn't help (same divergence).

## Why the Baseline is Locally Optimal

The current Arakawa-Lamb approach is the best for this architecture because:

1. **Frame consistency**: The transformation matrix rotates the gradient to the physical direction, matching the vorticity computation. Both are in the same frame → geostrophic cancellation works.

2. **Stagger consistency**: Both gradient and vorticity are at D-grid corners, averaged to edge midpoints by the same 2-point average. No cross-stagger projection.

3. **Halo error structure**: The O(dx) halo error in the gradient is bounded by `_extrapolate_boundary_corners` at vertex corners. The error doesn't grow because RK3 provides stability.

4. **Proven at scale**: Works for 3D atmosphere, ocean, and all equation sets.

## What Would Actually Fix It

1. **Higher resolution**: Edge artifacts are O(dx). At C72 they halve; at C144 they're negligible.

2. **Complete GFDL FV3 port**: Replace the ENTIRE operator chain (D-grid storage convention, halo exchange, mass transport, gradient, vorticity, time stepping) as a monolithic unit. This requires months of effort and would break all existing 3D atmosphere and ocean code.

3. **Spectral transform**: The spectral PE solver has no edge artifacts (no cubed-sphere panels). Already available in the codebase.

## Infrastructure Built (Available on main)

| Component | Location | Status |
|-----------|----------|--------|
| sin_sg/cos_sg (6,n,n,9) | `CubedSphereCDGrid` | Merged, 6 unit tests |
| dxc/dyc/rdxc/rdyc/rarea_c | `CubedSphereCDGrid` | Merged |
| d2a2c_vect | `fv3_sw_core.py` | Working, tested |
| c_sw operators (KE, vorticity, gradient, flux) | `fv3_sw_core.py` | Working at C-grid |
| edge_interpolate4 | `fv3_sw_core.py` | Working |
| Forward-backward step | `fv3_sw_core.py` | Available but unstable with current architecture |
| Duo-grid unit tests | `test_duogrid_metrics.py` | 6 tests passing |

## References

- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Mouallem, Harris & Chen (2023): Implementation of the Novel Duo-Grid in GFDL's FV3
- GFDL sw_core.F90: c_sw (lines 79-488), d2a2c_vect (lines 3006-3345)
