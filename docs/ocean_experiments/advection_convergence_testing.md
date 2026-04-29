# Ocean Advection Scheme Convergence Testing

## Motivation

Testing advection schemes in turbulent Eady simulations (see
`advection_scheme_comparison.md`) revealed that the implicit barotropic
solver (PR #218) changes the dynamics significantly — 8-65x higher
max_speed, schemes that were stable now blow up. We cannot distinguish
between (a) a bug in a scheme, (b) parameter tuning, or (c) correct
physics revealed by the cleaner solver.

Before re-tuning dissipation, we need **clean, isolated tests** with
known analytical solutions to verify each scheme works as expected.

## Test hierarchy

### Level 1: Pure 1D advection of known shapes
Advect a Gaussian/cosine-bell zonally with uniform velocity on the
latlon C-grid. Periodic BCs. After one revolution, compare to initial
condition. Verify convergence rate matches expected formal order.

### Level 2: 2D prescribed-flow advection
(A) Solid body rotation: uniform zonal flow, cosine bell wraps around.
(B) Deformational flow: time-reversing deformation, tracer returns to IC.

### Level 3: Prescribed 3D flow with vertical advection
Uniform zonal flow + prescribed overturning cell. Tests the full 3D
tracer update (horizontal + vertical + layer thickness) in isolation
from pressure/barotropic/physics.

## Schemes under test

| Scheme | Type | Expected order | Key property |
|---|---|---|---|
| upwind | Tracer | 1st | Most diffusive, always stable |
| tvd (Van Leer) | Tracer | 2nd | Monotone, safe default |
| dst3 (Sweby) | Tracer | 3rd | Less diffusive than TVD |
| ppm_fct (Zalesak) | Tracer | 4th | Monotone, known limiter issue |
| som (Prather) | Tracer | 2nd moments | Near-zero implicit diffusion |
| weno5 | Tracer | 5th | High-order, ENO oscillation suppression |
| weno7 | Tracer | 7th | Highest-order available |

## Tools

- **Test file**: `tests/ocean/test_advection_convergence.py` (pytest)
- **Runner script**: `scripts/run_advection_convergence.py` (standalone, produces plots)
- **Output**: `results/advection_convergence/`

---

## Log

### 2026-04-29: Initial setup

Created branch `dhruv/advection-convergence-tests`. Starting with Level 1.

#### Level 1 results (1D zonal advection, CFL~0.5, one revolution)

| Scheme | 32 pts L2 | 64 pts L2 | 128 pts L2 | 256 pts L2 | Rate | Expected | Status |
|---|---|---|---|---|---|---|---|
| upwind | 4.54e-01 | 3.44e-01 | 2.75e-01 | 2.47e-01 | 0.3 | 1 | SLOW |
| tvd | 4.03e-01 | 4.55e-01 | 3.10e-01 | 2.55e-01 | 0.2 | 2 | SLOW |
| dst3 | 2.55e-01 | 2.46e-01 | 2.46e-01 | 2.47e-01 | 0.0 | 3 | NO CONVERGENCE |
| weno5 | 9.51e+00 | 6.28e+00 | 5.28e+04 | 2.54e+14 | -14.9 | 5 | UNSTABLE |
| weno7 | 8.92e+01 | 4.86e+03 | 1.04e+12 | 5.97e+30 | -31.9 | 7 | UNSTABLE |

Conservation: upwind, tvd, dst3 conserve to machine precision. WENO5/7
lose conservation as they blow up (mass_drift=0.003 at 256 pts for weno5).

**Key findings:**

1. **WENO5 and WENO7 are unstable in a pure 1D advection test.** Errors grow
   exponentially with resolution. This is a bug — WENO is purely spatial
   reconstruction, should be stable at CFL ≤ 0.5 with flux-form update.
   The instability may explain some Eady blowups.

2. **DST-3 shows zero convergence.** Error is flat at ~0.25 regardless of
   resolution. The Sweby limiter may be clamping the 3rd-order correction.

3. **Upwind and TVD converge but at sub-theoretical rates** (0.3 and 0.2
   instead of 1 and 2). Likely a test setup issue — Gaussian spans only
   ~10 grid points at n_lon=64. Need wider feature or finer resolutions.

**Next**: Investigate WENO instability. Check if the bug is in the
reconstruction (`weno5_to_u_points`) or in how it's called in the
flux-form update.

**Background**: The Eady comparison on `dhruv/eady-advection-comparison` found
that the implicit barotropic solver produces dramatically different dynamics
from the explicit solver at the same physics settings:

| Scheme | Solver | Status (weak 600d) | max_spd | Var(T)% |
|---|---|---|---|---|
| upwind | explicit | PASS | 0.11 | -12.4% |
| upwind | implicit | PASS | 0.85 | -36.7% |
| tvd | explicit | PASS | 0.19 | -7.3% |
| tvd | implicit | FAIL d~116 | 1.33 | -- |
| som+Csmag | explicit | PASS | 0.12 | -2.5% |
| som+Csmag | implicit | "PASS" | 8.02 | -16.7% |

Conclusion: the explicit solver's cosine time filter noise was acting as
accidental damping. Need to verify schemes are correct before re-tuning.
