# FV3 Cubed-Sphere Fidelity Gap Analysis

Current repo status vs faithful FV3 (GFDL SHiELD) implementation.

## Metrics / Grid Consistency

| Gap | Current Status | Faithful FV3 Requires | Status |
|-----|---------------|----------------------|--------|
| `cosa/sina` computation | Consistent: `sina = sqrt(1-cosa^2)` everywhere. Float32 precision (~1e-7). | Same convention — no gap. | **Completed** |
| `sin_sg/cos_sg` (9-point Duo-Grid) | Implemented via 2x-refined supergrid. Identity holds to float32 eps. | Faithful to Mouallem et al. (2023). | **Completed** |
| `dxc/dyc` center-to-center distances | Computed from halo-exchanged Cartesian positions, great-circle distance. | FV3 uses supergrid-based distances. Results are equivalent at float64. | **Completed** |
| `area_corner/rarea_c` | From halo-exchanged cell areas (4-cell average). | FV3 uses supergrid quadrilateral areas. Halo-aware path is operationally equivalent. | **Completed** |
| `_compute_supergrid_metrics()` | Dead code (documented). Halo-aware path used instead. | Could serve as validation reference. | **Left for later** |
| Grad-c transformation matrix | Precomputed 2x2 matrix from 3D Cartesian geometry with halo-interpolated positions. | FV3 does not precompute this; uses inline non-orthogonality correction. Both are mathematically equivalent. | **Completed** |

## Tile-Edge / Halo Treatment

| Gap | Current Status | Faithful FV3 Requires | Status |
|-----|---------------|----------------------|--------|
| Halo interpolation | 3-point Lagrange quadratic via `_interp_strip` (O(dx^3) accuracy). Corrects gnomonic mismatch. | FV3 uses exact tile-edge coupling with no interpolation: neighbor cells are at exact grid points. | **Left for later** |
| Corner fill | Vectorized L-shaped fill in `_fill_corners_h1`. | FV3 has exact 3-face corner stitching. Current approach is a surrogate. | **Left for later** |
| Vector halo exchange | `pad_halo_vector` rotates across faces. | Same semantics. | **Completed** |
| Edge-midpoint stagger | Production path uses edge-midpoint D-grid (winds half a cell from boundaries). Eliminates boundary sync but is not FV3-faithful. | FV3 uses true D-grid at cell corners with explicit tile-edge coupling. | **Left for later** |

## Staggering / State Placement

| Gap | Current Status | Faithful FV3 Requires | Status |
|-----|---------------|----------------------|--------|
| Shallow water winds | Edge-midpoint D-grid: `u_d:(6,n,n+1)`, `v_d:(6,n+1,n)` (production). Corner D-grid: `(6,n+1,n+1)` also available. | True D-grid at corners with tile-edge sync. | **Partially completed** (corner path exists but edge-midpoint is production) |
| Hydrostatic 3D winds | D-grid corners `(6,n+1,n+1,nlev)` — prognostic. | Same stagger. | **Completed** |
| Non-hydrostatic winds | Cell-centre `(6,n,n,nlev)` stored, converted to D-grid internally. | D-grid throughout; physics coupler must consume D-grid winds. | **Left for later** |
| Physics coupling | D-grid → cell-centre interpolation at coupling boundary. | FV3 uses D-grid throughout with a2b_ord4 for physics interface. | **Left for later** |

## Shallow-Water Time Integration

| Gap | Current Status | Faithful FV3 Requires | Status |
|-----|---------------|----------------------|--------|
| Production integrator | SSP-RK3 with `fv3_sw_tendencies` (Arakawa-Lamb gradient). Stable, tested on cosine bell, W2, W5. | FV3 uses forward-backward splitting (c_sw + d_sw). | **Left for later** |
| Forward-backward step | `fv3_forward_backward_step` in `fv3_sw_core.py` — **EXPERIMENTAL**, known unstable (85 m/s spurious wind, 3% mass error by day 1). | Stable forward-backward with exact dissipation control at each phase. | **Blocked** |
| `use_experimental_csw` flag | Renamed from `use_fv3_fb`. Wraps `fv3_csw_tendencies` (C-grid half only) in RK3. Known unstable (NaN by step ~50). | Should call the actual forward-backward step. | **Partially completed** (renamed for honesty) |
| Blocker for FB step | Forward-backward coupling unstable without FV3's exact dissipation at c_sw/d_sw interface (del2/del4 at specific phases). | Implement FV3's `dddmp`, `d4_bg`, `d_con` dissipation at correct phases. | **Blocked** |

## Hydrostatic 3D Requirements

| Gap | Current Status | Faithful FV3 Requires | Status |
|-----|---------------|----------------------|--------|
| Vertical coordinate | Sigma/hybrid sigma-pressure (Eulerian). | FV3 uses vertically Lagrangian with periodic remapping. | **Left for later** |
| Vertical remapping | Not implemented. | PPM remapping after each dynamical step. | **Left for later** |
| PGF formulation | Simmons-Burridge with harmonic mean T at corners. | FV3 uses Lin (2004) pressure gradient with Lagrangian surfaces. | **Left for later** |
| Tracer transport | C-grid PPM with FCT. | Same approach. | **Completed** |

## Non-Hydrostatic 3D Requirements

| Gap | Current Status | Faithful FV3 Requires | Status |
|-----|---------------|----------------------|--------|
| State storage | Cell-centre (physics compatibility). D-grid conversion internal. | D-grid throughout with dedicated physics coupler. | **Left for later** |
| Acoustic substeps | Height-based vertical differencing. | FV3 uses Lagrangian + remapping for acoustic modes. | **Left for later** |
| `w` equation | Explicit height-coordinate formulation. | FV3 solves for `w` in terrain-following Lagrangian coordinates. | **Left for later** |

## Validation / Reference-Equivalence Gaps

| Test | Current Status | Status |
|------|---------------|--------|
| Metric identities (cosa^2+sina^2=1, reciprocals) | 29 tests in `test_fv3_audit_harness.py`, all passing at float32 precision. | **Completed** |
| Solid-body divergence | Tested at C8/C16, convergence verified. | **Completed** |
| TC2 balanced residual | Tested at C8/C16, residual < 5e-4 * Omega. | **Completed** |
| One-step mass conservation | With fixer: ~1e-7 (float32). Without: ~1e-5. | **Completed** |
| Cross-face mismatch | Halo roundtrip, grad-c matrix, edge positions tested. | **Completed** |
| Cosine bell transport | 100-step stability + mass conservation tested. Full 12-day revolution not yet in audit harness. | **Partially completed** |
| Williamson 2 (5-day) | C8 passes (L2 < 0.1). C16 passes (L2 ~ 0.23, higher than corner model). | **Completed** |
| Williamson 5 (5-day) | C8 passes (stable, h > 0, h < 10000). Mass conserved. | **Completed** |
| Ocean inertial gravity wave | 9 tests in `tests/ocean/unit/test_inertia_gravity_wave.py`: IC validation, 10-step stability, volume conservation, analytical dispersion. All passing. | **Completed** |
| 3D atmosphere | PE/CE cdgrid paths exist and have unit tests (9 tests). No FV3-specific 3D validation. | **Left for later** |
| Visual edge artifact check | W2 1-day snapshots at C16 inspected: wind speed, v-wind, height error. **No face-boundary edge artifacts found.** Some grid-scale checkerboard noise on polar faces from edge-midpoint computational mode. | **Completed** |
| `_interp_strip` docs vs implementation | Docs and code match: 3-point quadratic Lagrange, linear fallback for n<3, precision-aware weight casting. No mismatch found. | **Completed** |

## Performance / Scaling Gaps

| Gap | Current Status | Faithful FV3 Requires | Status |
|-----|---------------|----------------------|--------|
| MPI halo exchange | 4D halo with `sendrecv` VJP wrapper (AD-safe). | FV3 uses MPI tile boundaries with exact indexing. | **Partially completed** |
| Buffer donation | `.raw` variant for AD compatibility. | Same approach. | **Completed** |
| GPU scaling | Tested on Levante (see `REAL_HARDWARE_SCALING.md`). | Production FV3 targets GPU clusters. | **Left for later** |

## Summary

The cubed-sphere CD-grid path is a **stabilized research implementation** inspired by FV3, not a faithful port. The main blockers for faithful FV3 fidelity are:

1. **Forward-backward time integration**: The c_sw + d_sw scheme is unstable without FV3's exact dissipation control. The production path uses RK3 instead.
2. **Tile-edge coupling**: Uses interpolating halo exchange instead of FV3's exact tile-edge stitching.
3. **Vertically Lagrangian coordinate**: Not implemented; uses Eulerian sigma/hybrid.
4. **Edge-midpoint stagger**: Production path avoids boundary sync by offsetting winds, which is not FV3-faithful.

The stabilized path is validated on Williamson TC2, TC5, cosine bell transport, and ocean inertial gravity waves. Visual inspection confirms no edge artifacts on wind components. It is suitable for differentiable ESM research but should not be labeled "FV3 exact" or "production FV3."
