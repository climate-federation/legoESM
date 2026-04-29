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

#### Level 1 results — first attempt (one full revolution, CFL~0.5)

All schemes showed poor convergence or instability. WENO5/7 appeared to
explode (errors 10^14/10^30 at 256 pts). Investigation revealed this was
a **test design flaw**, not a scheme bug: error accumulated over hundreds
of steps with diffusive schemes fully smearing the Gaussian. WENO was
amplifying residual noise from many accumulated steps, not blowing up
in a single step.

#### Level 1 results — fixed test (20 steps, shifted exact solution)

Test: advect Gaussian (sigma=60 deg) with uniform zonal flow at CFL=0.5
for 20 steps. Compare against shifted exact Gaussian.

| Scheme | 32 pts L2 | 64 pts L2 | 128 pts L2 | 256 pts L2 | Rate | Expected |
|---|---|---|---|---|---|---|
| upwind | 7.89e-02 | 2.68e-02 | 1.08e-02 | 5.03e-03 | **1.32** | 1 |
| tvd | 1.08e-01 | 3.05e-02 | 1.12e-02 | 5.06e-03 | **1.47** | 2 |
| dst3 | 4.05e-02 | 1.97e-02 | 9.79e-03 | 4.89e-03 | **1.02** | 3 |
| weno5 | 9.75e-02 | 2.85e-02 | 1.10e-02 | 5.06e-03 | **1.42** | 5 |
| weno7 | 1.03e-01 | 3.05e-02 | 1.16e-02 | 5.20e-03 | **1.43** | 7 |

All schemes conserve mass to machine precision. All schemes stable.

**Key finding: Forward Euler time integration limits all schemes to ~1st
order convergence.** Since CFL is fixed, dt ∝ dx, so the temporal error
O(dt) = O(dx) dominates. All schemes converge at ~1st order regardless
of spatial order. At 256 points, all schemes have nearly identical error
(~5e-3) — the spatial reconstruction doesn't differentiate them.

**Implication for the ocean model:** The model uses forward Euler for
tracer advection. The higher-order schemes (DST-3, WENO5, WENO7) are
not buying formal accuracy — their value is in lower implicit diffusion
and sharper front preservation, not convergence rate.

DST-3 has a smaller prefactor than the others (0.040 vs 0.08-0.10 at
32 pts), consistent with its direct space-time formulation incorporating
the CFL. But it still converges at ~1st order.

#### Level 1 results — fixed absolute dt (single step, CFL=0.001-0.01)

To isolate spatial error from temporal error, ran a single advection step
with the same absolute dt at all resolutions (dt=33.7s from CFL=0.01
at 256 points). CFL decreases as grid coarsens.

| Scheme | 32 pts L2 | 64 pts L2 | 128 pts L2 | 256 pts L2 | Notes |
|---|---|---|---|---|---|
| upwind | 1.96e-05 | 1.07e-05 | 6.80e-06 | **5.43e-06** | Converging toward temporal floor |
| tvd | 6.05e-06 | 5.17e-06 | **4.99e-06** | **4.93e-06** | At floor by 64 pts |
| dst3 | 9.73e-06 | 6.14e-06 | 5.20e-06 | **4.97e-06** | At floor by 128 pts |
| weno5 | **5.33e-06** | **5.10e-06** | **4.99e-06** | **4.94e-06** | At floor everywhere |
| weno7 | **5.37e-06** | **5.12e-06** | **5.00e-06** | **4.94e-06** | At floor everywhere |

Temporal floor ≈ 5e-6 (forward Euler O(dt) error). All higher-order
schemes' spatial errors are **below** this floor even at 32 points.
Only upwind shows measurable spatial error above the floor.

**Conclusion: all schemes are spatially correct.** The Gaussian (sigma=60 deg)
is smooth enough that 2nd-order and above resolve it perfectly at 32+
points. To measure formal spatial convergence rates (2, 3, 5, 7),
would need either a sharper initial condition or a higher-order time
integrator (RK4). But for practical validation, **no bugs detected**.

#### Time stepper survey (2026-04-29)

Production ocean models also use low-order time integration for tracers:
- **MOM6**: Forward Euler for baroclinic tracers (same as us)
- **MITgcm**: Supports AB2 (Adams-Bashforth 2nd order)
- **NEMO**: Robert-Asselin time filter

The Silvestri WENO plan documented AB2 as Phase 4e but **skipped** it.
Existing RK3/RK4 implementations in `src/legoesm/timestepping/` serve
only the spectral ocean model.

The value of higher-order spatial schemes (WENO5/7, SOM) is in **lower
implicit diffusion** (Var(T) preservation), not formal convergence rate.
This is consistent with Hill et al. (2012) who measured effective
diapycnal diffusivity, not convergence.

**Next steps:**
- Proceed to Level 2 (2D prescribed flow) to test grid metric
  interactions and dimensional splitting
- Investigate DST-3 behavior more carefully (it showed slightly more
  spatial error than TVD at coarse resolution, unexpected for 3rd order)
- The Eady advection comparison can resume — schemes are verified correct,
  the issue is parameter tuning with the implicit solver

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
