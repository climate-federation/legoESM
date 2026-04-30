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

#### Time stepper survey (2026-04-29) — CORRECTED

**Previous claim that "MOM6 uses forward Euler" was wrong.** Deep study
of Hill et al. (2012) and the Silvestri plan reveals:

- **MITgcm**: Uses **Adams-Bashforth** (AB2/AB3) time stepping — explicitly
  stated in Hill et al. (2012): "An Adams–Bashforth time stepping scheme
  is used with a stabilization factor of 0.1"
- **Oceananigans (Silvestri)**: Uses **AB2** for 3D baroclinic jet, **RK3**
  for 2D turbulence — documented in silvestri2024_weno_iles_plan.md
- **legoESM**: Uses **forward Euler** — the only production-relevant ocean
  model using 1st-order time integration for tracers

This explains why WENO is stable in MITgcm/Oceananigans but unstable
in our Level 2 tests: AB2/RK3 don't have the leading-order anti-diffusive
truncation error that forward Euler produces, which WENO's non-monotone
reconstruction amplifies.

**Action item**: Implement AB2 time stepping for the ocean tracer
advection. This is ~80 LOC (Silvestri plan Phase 4e estimate) and
requires storing one previous tendency in the scan carry.

#### Hill et al. (2012) key findings

Hill et al. tested 8 advection schemes in MITgcm on an ACC-like
eddying channel (5 km resolution, AB time stepping, KPP, biharmonic
viscosity). Effective diapycnal diffusivity from virtual tracer release:

| Scheme | κ_eff (300 lvl) | κ_eff (30 lvl) | Our equivalent |
|---|---|---|---|
| SOM no limiter | **0.005** | 0.93 ± 0.2 | `som` |
| SOM with limiter | 0.52 | 3.8 ± 2 | not implemented |
| 2nd order Superbee | 0.017 | 11 ± 5 | ~ `tvd` |
| DST-3 (no limiter) | 18 | 32 ± 4 | `dst3` |
| DST-3 + Sweby | 4.9 | 24 ± 4 | `dst3` (uses Van Leer) |
| 7th order monotone | 2.6 | 12 ± 3 | ~ `weno7` |
| Centered 2nd + diffusion | 16.9 | 19 ± 5 | not tested |

(All values ×10⁻⁵ m²/s. Observed ocean interior: ~1 ×10⁻⁵.)

**Key takeaways:**
1. SOM is 3-4 orders of magnitude less diffusive than DST-3
2. DST-3 performed **worse** than centered 2nd order + explicit diffusion
3. The 7th-order monotone scheme (closest to our WENO7) is ~500× more
   diffusive than SOM but still reasonable (2.6 at 300 levels)
4. All tests used AB time stepping — not forward Euler

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

### 2026-04-29: Full-order WENO fix — correct conversion coefficients

Initial fix used wrong coefficients (7/5760 for δ⁴ term — this is from the
*inverse* operator, not the direct conversion). Rederived correct stencil
weights from scratch by matching Taylor-series moments.

**Correct 5-point stencil (6th-order):**
```
f_avg = (863/960)*f_0 + (77/1440)*(f_{-1} + f_{+1}) - (17/5760)*(f_{-2} + f_{+2})
```

Verified: p=0, p=2, p=4 moments all match exactly. Conversion accuracy:
- 3-point (order=4): rate = 3.99
- 5-point (order=6): rate = **5.85**

**Face reconstruction after full fix:**

| Scheme | Rate (before any fix) | Rate (4th-order conv) | Rate (6th-order conv) |
|---|---|---|---|
| weno5 | 1.99 | 4.34 | **5.03** |
| weno7 | 1.99 | 3.99 | **6.45** |

WENO5 now achieves its designed 5th order. WENO7 achieves 6.45 (slightly
limited by the 6th-order conversion; 8th-order conversion would need a
7-point stencil). For practical purposes with forward Euler time
integration this is academic, but the spatial reconstruction is now correct.

Also applied the cell-average conversion to **momentum WENO**:
- `_weno_cell_to_uface` (D-term, zonal periodic)
- `_weno_cell_to_vface` (D-term, meridional bounded)
- `_weno_zeta_at_u` (Z-term vorticity, meridional)
- `_weno_zeta_at_v` (Z-term vorticity, zonal periodic)

All 112 WENO tests pass (30 tracer + 37 momentum + 45 core).

### Open: DST-3 convergence order unverified

DST-3 is a space-time scheme — its reconstruction coefficients d0(CFL),
d1(CFL) encode the Courant number so the combined reconstruction + Euler
update is 3rd-order accurate. The face reconstruction alone is NOT
3rd order (at CFL=0.5, d1=0 and it reduces to 1st order by design).

Cannot verify 3rd-order space-time accuracy with our current test
infrastructure because:
1. Face-only test: invalid (spatial reconstruction is not separable)
2. Fixed-CFL multi-step test: forward Euler O(dt)=O(dx) dominates,
   masking the space-time coupling
3. Full-revolution test: error saturates from accumulated diffusion

A proper test would need to compare the **prefactor** of the O(dx) error
(not the rate) against a known analytical result for DST-3, or use a
manufactured solution approach. Left as future work — not blocking
because DST-3 shows lower errors than upwind/TVD at coarse resolution
(0.040 vs 0.079/0.108 at 32 pts), consistent with its expected behavior.

### 2026-04-29: SOM included in Level 1 — behaves correctly

SOM added to the coupled convergence test. Like all schemes, limited to
~1st order by forward Euler. Key observations:

- SOM and DST-3 tied for lowest errors at coarse resolution (3.9e-02 at 32 pts)
- SOM conserves mass to machine precision (0.00e+00 drift)
- SOM is ~5x slower than other schemes (5.3s vs ~1s at 256 pts) due to
  9-moment directional sweeps
- At fine resolution (256 pts), all schemes converge to the same temporal
  floor (~5e-3 L2 error)

**Level 1 summary (all schemes, after WENO fix):**

| Scheme | L2 @ 32 | L2 @ 256 | Conservation | Face recon order |
|---|---|---|---|---|
| upwind | 7.89e-02 | 5.03e-03 | machine-eps | 1st (correct) |
| tvd | 1.08e-01 | 5.06e-03 | machine-eps | 2nd (correct) |
| dst3 | 4.05e-02 | 4.89e-03 | machine-eps | space-time (untested) |
| weno5 | 9.61e-02 | 5.07e-03 | machine-eps | **5th (fixed)** |
| weno7 | 9.81e-02 | 5.13e-03 | machine-eps | **6.5th (fixed)** |
| som | 3.91e-02 | 4.89e-03 | machine-eps | N/A (moment method) |

All schemes verified stable and conservative. WENO5/7 spatial order
fixed from 2nd to 5th/6.5th. Ready for Level 2 (2D prescribed flow).

### 2026-04-29 (later): Level 1 redone with AB2 and RK3, after audit fixes

Branch `time_stepper` added AB2 and RK3 tracer integrators to the model.
Re-ran Level 1 across all 5 schemes × 3 integrators. Audit (dycore expert)
found two test-design bugs that were also fixed:

1. **Fixed n_steps was wrong**: with CFL=0.5 fixed and n_steps=20 fixed,
   total advection distance shrank linearly with dx (112° at n=32, only
   14° at n=256). This created a fixed-distance ceiling that mimicked
   "first-order convergence" for all schemes regardless of their actual
   spatial order. Fix: hold **total advection distance fixed at 30°**;
   n_steps grows with refinement.

2. **Constant velocity in m/s was wrong**: the field was uniform in
   latitude, but constant linear velocity gives lat-varying angular
   speed (`dlon/dt = u/(R*cos(lat))`). This created a phase error baked
   into the comparison. Fix: use `u = u0 * cos(lat)` so all latitudes
   rotate at the same angular speed and the analytical-shifted Gaussian
   is exact at every latitude.

**Results (L2 error at n=256, fixed total time, smooth Gaussian):**

| Scheme | Euler | AB2 | RK3 |
|---|---|---|---|
| upwind | 2.6e-3 | 5.6e-3 | 5.1e-3 |
| tvd | 2.8e-3 | **4.1e-2** ✗ | 2.4e-4 |
| dst3 | **1.6e-4** ★ | **1.6e-2** ✗ | 2.6e-3 |
| weno5 | 4.2e-3 | 4.5e-4 | **8.0e-5** ★★ |
| weno7 | 1.7e-2 | 4.9e-4 | **8.0e-5** ★★ |

**Findings:**

- **WENO5+RK3 and WENO7+RK3 are the most accurate** combinations
  (8e-5 at n=256, ~50× better than WENO+Euler).
- **TVD+RK3** is excellent for monotone schemes (2.4e-4), 12× better
  than TVD+Euler.
- **AB2 + nonlinear limiters (TVD, DST3) loses monotonicity** — Linf
  blows up to 0.84 at fine resolution for TVD+AB2. The model docstring
  in `ocean_model_latlon_cgrid.py` (lines 84-88) documents this: the
  AB2 linear combination of two limited fluxes is not itself TVD.
  This is a known limitation, not a bug.
- **DST3+Euler is surprisingly competitive** at fine resolution
  (1.6e-4) — its space-time coefficients are tuned for Euler, exactly
  as designed.
- **WENO+Euler shows Linf >> L2 at fine resolution** (weno7 Euler L2=
  1.7e-2 with rate ≈ 0), foreshadowing the 2D instability we found in
  Level 2 — Euler can't damp WENO's small overshoots over many steps.
- All schemes conserve mass to machine precision (drift < 3e-15).

**Implementation caveats** (from audit):
- Test's RK3 uses **Shu-Osher SSP form**; model uses **Butcher form**.
  Equivalent for linear schemes + constant h, but model's TVD+RK3 may
  show slightly different overshoot behavior than the test reports.
- Test's AB2 first step matches the model's eager `step()` path. The
  `integrate_scan` (training) path pre-initializes prev flux to zero,
  giving a different first-step behavior (1.6× Euler).

**Implication**: Higher-order time integration (RK3) lets WENO and TVD
deliver their actual accuracy. With Euler, all schemes look similar
because the temporal floor dominates. AB2 helps WENO but breaks TVD/DST3
monotonicity.

### 2026-04-29: Sine sanity-check + dycore audit reveals O(dx²) FV ceiling

Added `--initial-condition sine` mode (sin(2*lon), infinitely smooth,
periodic) to expose true convergence rates. Expected: WENO5+RK3 should
give rate 3 (min of 5th spatial, 3rd temporal at fixed CFL). Observed:
**rate exactly 2.00** for both WENO5+RK3 and WENO7+RK3.

Dycore-expert audit traced the cause:

The discrete FV divergence `(F_E - F_W)/dx`, even with **exact** face
values, is only **2nd-order accurate** as an approximation of `dF/dx`
at the cell center. For sin(2x) with exact face values:
```
(F_E - F_W)/dx = 2*cos(2x_c) * sinc(dx)
                = 2*cos(2x_c) * (1 - dx²/6 + ...)
```
The `sinc(dx)` factor introduces a per-unit-time phase error of
`u*(1-sinc(dx)) ≈ u*dx²/6`, giving global error O(dx²).

This is NOT the WENO reconstruction's fault. WENO5 face values ARE
5th-order accurate (verified). The ceiling is **inherent to the discrete
FV divergence operator** when the prognostic variable is interpreted
as a point value (as in the model's WENO wrapper which converts
point→cellavg before reconstruction).

**The model has a built-in O(dx²) accuracy ceiling for tracer advection.**
Higher-order spatial schemes buy lower prefactors at coarse resolution
but identical asymptotic rate. To break this ceiling would require
either:
- Treating T as a true cell-average throughout (skip WENO's internal
  point→cellavg conversion, which would require model-level changes)
- Or using a high-order divergence operator (e.g., 4th-order
  staggered-grid finite differences instead of `(F_E - F_W)/dx`)

This was validated numerically: at n=32 with sin(2*lon), `dx²/6 ≈ 6.4e-3`,
matches observed WENO5+RK3 L2 = 6.7e-3. At n=256, `dx²/6 ≈ 1.0e-4`,
matches observed L2 = 1.05e-4.

**Practical implication**: WENO+RK3's biggest win in our ocean model
isn't formal accuracy — it's **stability** (the Level 2 deformational
flow blowups we saw with WENO+Euler should disappear with RK3) and
**lower error magnitudes at production resolutions**. The asymptotic
2nd-order rate from the divergence operator is something the model
just lives with.

**DST3+RK3 puzzle (rate 0.99)**: DST-3's CFL-dependent coefficients
encode the time-truncation correction (Lax-Wendroff style). RK3
re-evaluates F at each substage, double-counting this correction with
the wrong CFL. Result: DST3+RK3 degrades to ~1st-order, behaving like
upwind. **Don't pair DST-3 with RK3.**

**AB2+ε is 1st-order in time (not 2nd)**: The eps=0.1 stabilization
adds an `eps*dt*g_t` term to the LTE, breaking standard AB2's 2nd-order
accuracy. This is by design (for stability) but means AB2 is closer to
"forward Euler with one previous tendency for damping" than a true
2nd-order scheme. Verified: WENO+AB2 mixes the rate-2 FV ceiling with
rate-1 temporal error, giving the apparent rate ≈ 0.6 we observed.

### 2026-04-29: Level 2 redone with AB2 and RK3 — main hypothesis confirmed

Test: Nair & Lauritzen (2010) swirling deformation flow on lat-lon
channel. Cosine bell at (180°E, 45°N) deforms into a filament during
0 ≤ t ≤ T/2, then flow reverses, bell should return at t = T.
Previously: WENO5 unstable at 128×256, WENO7 catastrophic blowup
(L2 = 6.18, max = 35).

**Results at finest grid (128×256):**

| Scheme | Euler L2 | AB2 L2 | RK3 L2 |
|---|---|---|---|
| upwind | 4.83e-2 | 5.25e-2 | 5.18e-2 |
| tvd | 2.45e-2 | 2.90e-2 | 2.82e-2 |
| dst3 | 3.28e-2 | 3.73e-2 | 3.65e-2 |
| weno5 | **5.51e-2** (Linf=0.30) | 1.78e-3 | **7.80e-4** |
| weno7 | **6.18e+0** (Linf=35) | 1.72e-3 | **4.24e-4** |

**Headline:** WENO7+Euler→RK3 gives **15,000× lower L2** at 128×256.
WENO5 and WENO7 are now **stable at all tested resolutions** with
either AB2 or RK3.

**Convergence rates (L2, n=32→256):**

| Scheme + Integrator | Rate | L2(coarse) | L2(fine) |
|---|---|---|---|
| weno5 + euler | -0.26 | 3.82e-2 | **5.51e-2** (worsens — instability) |
| weno7 + euler | -3.67 | 3.81e-2 | **6.18e+0** (catastrophic blowup) |
| weno5 + ab2 | 1.22 | 9.67e-3 | 1.78e-3 (clean) |
| weno7 + ab2 | 1.09 | 7.77e-3 | 1.72e-3 (clean) |
| **weno5 + rk3** | **1.69** | 8.12e-3 | 7.80e-4 (best) |
| **weno7 + rk3** | **1.84** | 5.45e-3 | 4.24e-4 (best) |

Rates of 1.7-1.8 are consistent with the O(dx²) FV-divergence ceiling
identified in Level 1 (Lvl 1 cosine-bell with sharp filaments hits the
asymptotic limit faster).

**Visual confirmation:** WENO7+Euler at t/T=1 shows visible streaks of
overshoots east of the bell; WENO7+RK3 shows the bell identical to the
initial condition. No grid-scale artifacts.

**For the ocean model:** WENO5/7 + RK3 is the recommended pairing for
stability under sharp-gradient flows. AB2 is a cheaper alternative
(~3× faster than RK3) at modest accuracy cost (~2× higher error). All
non-WENO schemes (upwind, tvd, dst3) are essentially insensitive to the
time integrator at this resolution — choose based on cost.

**Mass conservation:** All combinations conserve to machine precision
across the integration (drift ≤ 1.6e-14 in float64).

### 2026-04-29: Level 2(A) — solid body rotation (Williamson Test 1)

Added the second Level 2 test: classical Williamson (1992) Test 1 — pure
solid-body rotation around an axis tilted by α from the geographic pole.
Bell traces a great circle and returns to IC after one full revolution
(12 days). Unlike the deformational flow, errors **accumulate
monotonically** (no reversal cancellation) over a much longer integration.

**Setup adjustments after wall-impact bug found:**
- Bell at (180°, 25°N) with α=10° (originally α=30° at 45°N → bell hit
  northern wall and gave nonsense results)
- Bell trajectory: latitude oscillates between 25°N and 45°N. With bell
  radius 30°, total bell extent stays within [-5°, 75°N], comfortably
  inside the [-10°, 80°N] channel.

**Results at finest grid (128×256, 12-day full revolution):**

| Scheme | Euler L2 | AB2 L2 | RK3 L2 |
|---|---|---|---|
| upwind | 0.48 | 0.48 | 0.48 |
| tvd | 0.11 | 0.066 | 0.060 |
| dst3 | 0.23 | 0.28 | 0.27 |
| weno5 | 0.28 | 0.022 | 0.016 |
| weno7 | **4.28 BLOWUP** | 0.023 | 0.018 |

**Convergence rates (L2, n=32→256):**

| Scheme | Euler | AB2 | RK3 |
|---|---|---|---|
| upwind | 0.34 | 0.32 | 0.32 |
| tvd | 0.65 | 1.26 | 1.30 |
| dst3 | 0.65 | 0.56 | 0.58 |
| weno5 | 0.45 | 1.44 | **1.65** |
| weno7 | −1.33 (blowup) | 1.48 | **1.75** |

**Findings consistent with deformational test:**

1. **WENO7+Euler blows up at 128×256** (L2 = 4.28). Same root cause as
   deformational test: forward Euler's anti-diffusive truncation amplifies
   WENO's non-monotone overshoots over thousands of time steps.

2. **WENO5/7 + AB2 or RK3 are clean and converge** at rate 1.65-1.75 —
   consistent with the O(dx²) FV-divergence ceiling. AB2 gives errors
   within ~30% of RK3 at 3× lower cost (one tendency eval per step
   vs three).

3. **All non-WENO schemes are stable** with all integrators. TVD shows
   the best convergence among monotone schemes (rate 1.3 with RK3).
   Upwind doesn't converge well (rate 0.3) — its strong diffusion
   dominates, making the bell heavily smeared after a full revolution.

4. **Mass conservation excellent**: Euler/AB2 drift ≤ 1.9e-16, RK3
   drift ≤ 2e-13 (still effectively zero in float64; RK3 has more
   floating-point operations per step).

**Correction to earlier claim:** An initial run with α=30° starting at
45°N showed AB2 catastrophically blowing up for ALL schemes (errors
10^15–10^85). That was caused by the bell smashing into the northern
wall and AB2's two-step memory amplifying the wall-impact artifacts.
**With the corrected, properly-contained trajectory, AB2 is fully
stable on solid body rotation** — the earlier "AB2 instability" was a
test-setup bug, not a real scheme issue.

**Practical takeaways:**
- WENO5+AB2 is the recommended pairing for accuracy + cost: errors
  within ~30% of RK3 at 3× lower compute.
- WENO5/7+Euler should be avoided at production resolutions.
- Upwind+anything is too diffusive for long-distance transport.
- DST-3 with any integrator gives mediocre convergence (rate 0.6) —
  the design tuning for forward Euler doesn't combine well with multi-
  stage methods, and even with Euler it's worse than expected here.

### 2026-04-30: Level 3 — vertical advection (overturning cell)

Designed a 2D zonal-vertical (x-z) test using a stream function:
  ψ(x, z, t) = ψ₀ sin(2πx/Lx) sin(πz/H) cos(πt/T)
with ψ₀ = 3.5e4 giving u_max ≈ 20 m/s, w_max ≈ 5.5e-3 m/s (= 475 m/day).
Bell starts at (lon=0, z=H/2) where w is at maximum magnitude.
Time-reversing flow returns the bell to IC at t = T = 5 days.

Test exercises both **horizontal** (`*_to_u_points`) and **vertical**
(`flux_form_vertical_tracer_advection_*`) advection routines together.
Domain: 4 × n_lon × n_lev with uniform 100m layers, total depth 5500m.

**Results at finest grid (128×64):**

| Scheme | Euler L2 | AB2 L2 | RK3 L2 |
|---|---|---|---|
| upwind | 0.043 | 0.060 | 0.057 |
| tvd | 0.018 | 0.017 | 0.013 |
| dst3 | 0.017 | 0.027 | 0.024 |
| weno5 | 0.027 | 0.0050 | **0.00027** |
| weno7 | 0.027 | 0.0050 | **0.00027** |

**Convergence rates (L2, n_lon=32→128):**

| Scheme | Euler | AB2 | RK3 |
|---|---|---|---|
| upwind | 0.87 | 0.81 | 0.82 |
| tvd | 0.92 | 1.11 | 1.19 |
| dst3 | 1.05 | 0.99 | 1.03 |
| weno5 | 0.99 | 0.91 | **2.02** |
| weno7 | 1.01 | 0.87 | **1.91** |

**Findings:**

1. **No blowups for any combination.** WENO+Euler is stable here — the
   vertical advection's smaller CFL (w_max << u_max) doesn't amplify
   the WENO oscillations the way solid body rotation did. So Euler
   instability is **flow-regime-dependent**, not a universal property
   of WENO+Euler.

2. **WENO+RK3 dominates by 50×** at fine resolution (0.00027 vs 0.013
   for TVD+RK3, the second-best). This is the largest scheme-pairing
   advantage we've seen across all three Level tests.

3. **WENO+RK3 hits rate ~2.0** — same FV-divergence ceiling documented
   in Level 1 sine test. WENO5 and WENO7 give identical errors,
   consistent with both being limited by the 6th-order cell-average
   conversion (our docs note WENO7 effectively converges as 6th order
   given that conversion).

4. **AB2 doesn't help WENO here** — only rate 0.91 vs RK3's 2.02. The
   AB2 ε-stabilization adds an O(dx) error that dominates in this 3D
   regime where vertical and horizontal temporal errors compound.

5. **Mass conservation**: Euler/AB2 to machine precision (~1.9e-16),
   RK3 to ~6e-15 (still effectively zero in fp64; RK3 has more
   floating-point operations per step accumulated over many steps).

6. **WENO5 and WENO7 are indistinguishable** at this resolution —
   spatial accuracy is fully shadowed by the FV-divergence ceiling
   and (for AB2) by temporal error.

**Combined practical takeaways (across all three levels):**

- **Best pairing: WENO5 + RK3** — confirmed across 1D, 2D deformational,
  2D solid body rotation, and now 3D overturning cell. Always stable,
  always lowest error, mass conserves to machine precision.
- **AB2 is a 3× cheaper alternative** when raw accuracy isn't critical:
  errors within ~30% of RK3 in Levels 2(A)/2(B), but degrades to ~20×
  worse in Level 3 vertical.
- **WENO+Euler is unstable in horizontal sharp-gradient flows**
  (deformational, solid body) but stable in pure vertical (Level 3).
  The instability mode is anti-diffusive Euler truncation amplifying
  WENO's non-monotone overshoots — only happens when sharp gradients
  are sustained over many steps with negligible damping.
- **TVD + any integrator** is the best monotone scheme. Decent accuracy,
  always stable, never produces overshoots. Cheap.
- **DST-3** consistently underperforms — its CFL-encoded coefficients
  don't survive multi-stage time integration.
- **Mass conservation is excellent across all scheme/integrator
  combinations** — flux-form FV update preserves mass to machine
  precision regardless of scheme order or integrator stage count.

**Status of testing plan: COMPLETE**
- Level 1 (1D zonal): 5 schemes × 3 integrators × {sine, gaussian} ✓
- Level 2(A) (solid body rotation): 5 schemes × 3 integrators ✓
- Level 2(B) (deformational): 5 schemes × 3 integrators ✓
- Level 3 (3D overturning): 5 schemes × 3 integrators ✓

**Recommended pairings (for the model):**
- Production tracer advection: **TVD+Euler** (cheap, robust) or
  **DST3+Euler** (best Euler accuracy, space-time tuned)
- Accuracy-critical (e.g., adiabatic interior tracking): **WENO5+RK3**
  (3× cost, 50× lower error)
- AVOID: TVD+AB2, DST3+AB2 (monotonicity loss)

### 2026-04-29: Level 2 — 2D deformational flow (swirling reversal)

Test: Nair & Lauritzen (2010) swirling deformation on lat-lon channel.
Cosine bell at (180°E, 45°N), non-divergent flow deforms and reverses
over 5 days. Error at t=T measures combined spatial, temporal, and
dimensional-splitting accuracy.

Script: `scripts/run_advection_convergence_2d.py`

| Scheme | 32x64 L2 | 64x128 L2 | 128x256 L2 | Rate | Status |
|---|---|---|---|---|---|
| upwind | 1.34e-01 | 7.76e-02 | 4.31e-02 | 0.82 | OK |
| tvd | 7.97e-02 | 4.20e-02 | 2.21e-02 | 0.93 | OK |
| dst3 | 9.89e-02 | 5.46e-02 | 2.95e-02 | 0.87 | OK |
| weno5 | 3.55e-02 | 2.07e-02 | **1.26e-02** | 0.75 | **Best L2** |
| weno7 | 3.74e-02 | 2.33e-02 | **8.90e-02** | -0.62 | **UNSTABLE at 128x256** |

All schemes conserve mass to machine precision.

**Key findings:**

1. **Error ordering at 64x128**: weno5 (0.021) < weno7 (0.023) < tvd (0.042)
   < dst3 (0.055) < upwind (0.078). WENO5 is the best scheme in 2D.

2. **WENO7 is unstable at 128x256**: L2 jumps from 0.023 to 0.089, Linf
   reaches 0.487. Visible oscillatory noise around the cosine bell.
   Likely related to the wider stencil (4 cells each side) interacting
   with the bounded meridional direction — the 8th-order cell-average
   conversion along the wall-bounded axis may introduce artifacts within
   the Neumann ghost-cell region.

3. **Convergence rates are ~0.8-0.9** for all stable schemes, consistent
   with forward Euler limiting as in Level 1.

4. **DST-3 shows higher error than TVD in 2D** (0.055 vs 0.042 at 64x128),
   whereas in 1D it was lower. The multi-directional flow may be
   less favorable for DST-3's operator-split space-time design.

### 2026-04-29: Audit found divergent flow bug + retest

Dycore expert audit found **critical bug**: the swirl velocity field was
divergent (cos²(lat) instead of cos(lat) in v, plus sign inconsistency).
Divergent flow creates spurious T*div(u) source/sink bias ~ O(0.07).

**Fixed** to non-divergent flow from stream function ψ = A*sin²(λ)*cos²(φ):
  u = +A*sin²(lon)*sin(2*lat)*cos(πt/T)
  v = +A*sin(2*lon)*cos(lat)*cos(πt/T)

**Retest results (corrected non-divergent flow):**

| Scheme | 32x64 L2 | 64x128 L2 | 128x256 L2 | Rate | Status |
|---|---|---|---|---|---|
| upwind | 1.48e-01 | 8.68e-02 | 4.83e-02 | 0.81 | OK |
| tvd | 8.91e-02 | 4.68e-02 | 2.45e-02 | 0.93 | OK |
| dst3 | 1.10e-01 | 6.07e-02 | 3.28e-02 | 0.87 | OK |
| weno5 | 3.82e-02 | 1.96e-02 | **5.51e-02** | -0.26 | **UNSTABLE at 128x256** |
| weno7 | 3.81e-02 | 2.58e-02 | **6.18e+00** | -3.67 | **BLOWUP at 128x256** |

**WENO instability is worse with corrected flow.** Now WENO5 also shows
instability at 128x256 (previously only WENO7). The divergent flow was
acting as accidental stabilization.

**Root cause (confirmed by audit):** WENO is essentially non-oscillatory
but NOT monotone. Small oscillations near sharp gradients grow over many
forward-Euler steps. At higher resolution, more steps are needed (fixed
CFL → dt ∝ dx → n_steps ∝ 1/dx), giving oscillations more time to
accumulate. Monotone schemes (upwind, TVD, DST-3) don't have this issue.

**Implications for production:**
1. WENO5/7 tracer advection with forward Euler is marginally stable in
   2D at fine resolution. In the full ocean model, other damping
   mechanisms (viscosity, diffusion, barotropic coupling) may prevent
   the instability, but this is a fragile situation.
2. A monotone WENO limiter (e.g., Zalesak FCT post-processing) or a
   higher-order time integrator (RK3) would fix this properly.

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
