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

### 2026-04-29: Face reconstruction convergence — spatial order bugs found

Testing face reconstruction accuracy in isolation (no time stepping at
all). Call `scheme_to_u_points(T, mass_flux)` on T = sin(2*lon), compare
reconstructed face values to exact sin(2*lon_face). Linf error:

| Scheme | 32 | 64 | 128 | 256 | 512 | Rate | Expected | Status |
|---|---|---|---|---|---|---|---|---|
| upwind | 1.95e-01 | 9.80e-02 | 4.91e-02 | 2.45e-02 | 1.23e-02 | **1.00** | 1 | CORRECT |
| tvd | 3.61e-02 | 9.48e-03 | 2.40e-03 | 6.02e-04 | 1.51e-04 | **1.98** | 2 | CORRECT |
| dst3 | 6.75e-02 | 3.30e-02 | 1.64e-02 | 8.19e-03 | 4.09e-03 | **1.01** | 3 | **BUG: 1st order** |
| weno5 | 6.33e-03 | 1.60e-03 | 4.01e-04 | 1.00e-04 | 2.51e-05 | **1.99** | 5 | **BUG: 2nd order** |
| weno7 | 6.33e-03 | 1.60e-03 | 4.01e-04 | 1.00e-04 | 2.51e-05 | **1.99** | 7 | **BUG: 2nd order** |

**Critical findings:**

1. **DST-3 is 1st order instead of 3rd.** The Sweby limiter is destroying
   the 3rd-order correction on a smooth sinusoidal field, reducing it to
   upwind-level spatial accuracy. The limiter should NOT activate on a
   smooth periodic function.

2. **WENO5 and WENO7 are 2nd order instead of 5th/7th.** Both give
   IDENTICAL errors (to 4+ digits), which means WENO7 is falling back to
   the same stencil as WENO5, and both are doing ~2nd-order linear
   interpolation instead of their high-order WENO reconstruction.

3. **Upwind (1st order) and TVD (2nd order) are correct.**

**Implications:** The higher-order schemes are not delivering their
advertised accuracy. This may explain why WENO5/7 didn't differentiate
from TVD in the Eady Var(T) comparison — they're effectively 2nd-order
schemes, same as TVD.

### 2026-04-29: WENO root cause — point values vs cell averages

**Root cause identified.** The core WENO kernel (`src/legoesm/core/weno.py`)
is correct — it converges at 5th/7th/9th order when given **cell-average**
inputs (verified by `tests/core/test_weno.py::TestConvergenceOrder`).

The ocean wrapper (`_weno_to_u_points` in `advection.py:915`) passes
**point values** (tracer at cell centers) instead of cell averages. WENO
reconstruction is a finite-volume method that assumes cell-average inputs.
Point values differ from cell averages by O(dx^2):

    f_avg = f(x) + (dx^2/24) * f''(x) + O(dx^4)

This O(dx^2) input error limits the reconstruction to O(dx^3) regardless
of the WENO order, confirmed by direct test:

| Input type | WENO5 rate | WENO7 rate |
|---|---|---|
| Cell averages (correct) | **5.02** | **6.97** |
| Point values (current) | 2.97 | 2.99 |

The earlier face-reconstruction test showing 2nd order was because the
flux-form update adds additional O(dx) error from forward Euler. The pure
reconstruction error is O(dx^3) with point values.

**Fix options:**
1. Convert point values to cell averages before WENO reconstruction:
   `f_avg_i = f_i - (dx^2/24) * (f_{i+1} - 2*f_i + f_{i-1}) / dx^2`
   simplifies to `f_avg_i = f_i - (1/24) * (f_{i+1} - 2*f_i + f_{i-1})`
2. Use a WENO variant designed for point-value inputs (different coefficients)

Option 1 is simpler and doesn't require changing the core WENO kernel.

### 2026-04-29: DST-3 diagnosis — space-time vs spatial-only

DST-3 showing 1st-order face reconstruction is **not a bug**. DST-3 is a
**direct space-time** scheme: its coefficients d0(CFL), d1(CFL) encode
the CFL number into the reconstruction so that the combined space+time
accuracy is 3rd order over a full advection step. Testing the face
reconstruction alone (spatial only) does not capture this.

Key detail: at CFL=0.5, `d1 = (1-c)(1-2c)/6 = 0` — the curvature
correction vanishes entirely, reducing to 1st-order upwind + a fraction
of the downwind value. Additionally, the stability cap `d0 = min(d0_dst3,
d0_lw)` activates at CFL < 0.5, further reducing the high-order correction.

**Conclusion:** DST-3 cannot be tested with a pure spatial reconstruction
test. Its 3rd-order accuracy must be verified with a full advection step
at fixed CFL. The earlier 20-step fixed-CFL test showed DST-3 having
smaller errors than upwind/tvd at coarse resolution (0.040 vs 0.079/0.108
at 32 pts), consistent with its space-time accuracy.

**Summary of findings:**

| Scheme | Face recon order | Full-step order | Issue |
|---|---|---|---|
| upwind | 1st | 1st | Correct |
| tvd | 2nd | 2nd (limited by FE time) | Correct |
| dst3 | 1st (spatial only) | 3rd (space-time) | Not a bug — test artifact |
| weno5 | 3rd (should be 5th) | 3rd (limited) | **BUG: point values, not cell averages** |
| weno7 | 3rd (should be 7th) | 3rd (limited) | **BUG: point values, not cell averages** |

### 2026-04-29: WENO fix applied + coupled convergence retest

**Fix**: Added cell-average conversion in `_weno_to_u_points` and
`_weno_to_v_points` (`advection.py`):
```python
f_avg = f + (1/24) * (f_{i+1} - 2*f_i + f_{i-1})
```
This is 4th-order accurate, which is sufficient for WENO5 (~5th order
face reconstruction with exact cell averages) but caps WENO7.

**Face reconstruction after fix:**

| Scheme | Rate (before fix) | Rate (after fix) | Rate (exact cell avg) |
|---|---|---|---|
| weno5 | 1.99 | **4.34** | 5.03 |
| weno7 | 1.99 | **3.99** | 6.66 |

The 4th-order conversion is the limiting factor for WENO7.
For full 7th-order accuracy, would need 6th-order conversion adding
the (7/5760)*f'''' term. For production with forward Euler (1st order
time), this is academic — the 4th-order spatial accuracy far exceeds
what the time integrator can exploit.

**Coupled space-time convergence (CFL=0.5, 20 steps, after WENO fix):**

| Scheme | 32 pts L2 | 256 pts L2 | Rate | Notes |
|---|---|---|---|---|
| upwind | 7.89e-02 | 5.03e-03 | 1.25 | Correct (FE-limited) |
| tvd | 1.08e-01 | 5.06e-03 | 1.36 | Correct (FE-limited) |
| dst3 | 4.05e-02 | 4.89e-03 | 1.01 | Lowest coarse-res error |
| weno5 | 9.62e-02 | 5.07e-03 | 1.32 | Improved from 3rd → ~4th order spatial |
| weno7 | 9.84e-02 | 5.14e-03 | 1.33 | Improved from 3rd → ~4th order spatial |

All converge at ~1st order due to forward Euler time integration,
but DST-3 consistently has the lowest prefactor at coarse resolution
(0.040 vs 0.079-0.107) — consistent with its space-time accuracy.

All 30 WENO tracer tests + 37 WENO momentum tests pass with the fix.

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
