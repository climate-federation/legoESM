# Phase 3a results: Beckmann-Haidvogel seamount stress test

**Status (2026-04-30):** Complete.  Empirical regime boundary identified.

## Test setup

The canonical PGF stress test for z-coordinate ocean models
(Beckmann & Haidvogel 1993, JPO).  Stratified rest state in a
zonally-periodic basin with a single Gaussian seamount; zero forcing,
zero initial flow.  Any spurious velocity that develops is a numerical
artefact from the horizontal pressure gradient computed over the steep
bathymetry.

- Grid: 36×72 lat-lon (5°), 20 levels, H_max=4000 m
- Stratification: linear T, 20°C surface → 2°C deep; salinity uniform
- Seamount: Gaussian, σ=10°, height varied from 200 m to 3800 m
- Smoothing: variable Laplacian passes
- Integration: 30 sim-days, dt=600 s, implicit-CN solver, no forcing
- Pass criterion: max|u| ≤ 5 mm/s after 30 days

## Empirical regime boundary

| seamount | r_max | max\|u\| after 30 d | status |
|---|---|---|---|
| height=200 m | 0.013 | 0.69 mm/s | ✅ PASS |
| height=500 m | 0.035 | 1.11 mm/s | ✅ PASS |
| height=1000 m | 0.074 | 1.34 mm/s | ✅ PASS |
| height=2000 m | 0.169 | 2.48 mm/s | ✅ PASS |
| height=2200 m | 0.191 | 2.77 mm/s | ✅ PASS |
| height=2400 m | 0.215 | 3.71 mm/s | ✅ PASS |
| height=2600 m | 0.240 | 4.40 mm/s | ✅ PASS |
| height=2800 m | 0.270 | 5.94 mm/s | ❌ FAIL |
| height=3200 m | 0.342 | 10.69 mm/s | ❌ FAIL |
| height=3800 m | 0.539 | 32.47 mm/s | ❌ FAIL |

**Acceptable r_max for the lat-lon C-grid ocean: < 0.24.**

This is consistent with the standard B-H literature bound (r < 0.2)
and shows that the model passes the canonical stress test cleanly
within its valid regime.  Plot: `phase3a_regime_summary.png`.

## Smoothing-passes finding

Laplacian smoothing **converges to a fixed point** that retains
seamount character.  Sweeping passes ∈ {0, 2, 5, 10, 20, 30} on the
height=3800 m seamount:

| passes | r_max | max\|u\| after 30 d |
|---|---|---|
| 2 | 0.565 | 36.7 mm/s |
| 5 | 0.539 | 32.5 mm/s |
| 10 | 0.537 | 32.1 mm/s |
| 20 | 0.537 | 32.0 mm/s |
| 30 | 0.537 | 32.0 mm/s |

The existing implementation
(``_laplacian_smooth_2d`` in ``src/legoesm/ocean/bathymetry.py``)
uses a half-blend `result = 0.5*orig + 0.5*smoothed`, which is
energy-bounded but cannot drive r below the seamount's intrinsic
roughness.  Changing to a pure Laplacian (without the blend) would
help, but **convergence to a smooth field is fundamentally limited
by what the Laplacian operator can do given a sharp initial
condition**.

Implication: **smoothing alone is insufficient for sharp seamounts.**

## Implication for real ETOPO bathymetry

The Phase 0 diagnostic on real ETOPO 1° subset, regridded to
72×144 (2.5°), reported (after 2 passes of Laplacian smoothing):

> ocean fraction = 0.662, mean depth = 3375 m,
> max r-factor = 0.899, 99th-percentile r = 0.854,
> 3275 / 6861 ocean cells (48%) with r > 0.2

So **roughly half the realistic-ETOPO domain is in the regime where
this model fails the seamount stress test**.  Phase 3.5 (90-day
forced run on real ETOPO) is therefore expected to develop spurious
flow comparable to or larger than the 30+ mm/s seen at r_max=0.54
in Phase 3a, unless we add additional mitigation.

## Mitigation options

In order of complexity:

1. **More aggressive Shapiro filtering (un-blended).**  Replace the
   half-blend Laplacian with a true Shapiro filter or pure Laplacian.
   Tradeoff: drives r down but at the cost of bathymetric fidelity.
   Real continental slopes get artificially gentle.  Viable but reduces
   the realism we're trying to add.
   Estimated effort: ~30 LOC + tuning.

2. **Mellor-Ezer-Oey (MEO) r-factor cap.**  Iteratively adjust local
   H_bathy values to enforce r < r_target (typical 0.2) at every
   neighbour pair.  Used by MITgcm, NEMO, ROMS in production.  Volume
   change ~few percent; bathymetric features locally smoothed only
   where r exceeds the cap.  Recommended next step.
   Estimated effort: ~80 LOC + 30 LOC of tests.

3. **Density-Jacobian PGF (Shchepetkin-McWilliams 2003).**  Compute
   horizontal PGF from a higher-order Jacobian formulation that is
   much less sensitive to bathymetric slope.  Requires a substantial
   rewrite of the lat-lon C-grid PGF operator.  The standard ROMS/CROCO
   solution.
   Estimated effort: ~500 LOC + extensive validation.

4. **Hybrid z*-σ coordinate near steep slopes.**  Switch from z* to
   σ-coordinate vertical scheme over steep bathymetry, glued to z*
   over the abyss.  Largest code change.
   Estimated effort: ~1500 LOC + new validation suite.

## Recommended path

1. Implement **option 2 (Mellor-Ezer-Oey r-factor cap)** as a
   ``BathymetryConfig.r_factor_max`` option in ``bathymetry.py``.
   This is the standard production approach; it preserves coastline
   geography and most bathymetric structure while bringing r below
   the model's stability bound.
2. Re-run Phase 0's diagnostic to verify r_max < 0.24 is achievable on
   real ETOPO with reasonable r_factor_max (try 0.2, then 0.15 if
   stability margin needed).
3. Re-run Phase 3.5 with MEO-enabled bathymetry.  If it passes, the
   plan is unblocked.
4. If MEO alone is still insufficient (e.g. the volume change to
   reach r < 0.2 is unacceptably large, or the model develops
   instability anyway), **then** invoke the d-J PGF separate-review
   gate per the original plan.

The d-J PGF decision should not be made now — MEO is sufficient for
most production ocean models at our resolution, and it costs <100
LOC vs ~500 LOC for d-J PGF.

## Phase 3a decision gate

- ✅ Empirical regime boundary identified: model passes for
  r_max < ~0.24, with the 5 mm/s threshold corresponding to
  r_max ≈ 0.27.  Consistent with B-H literature.
- ✅ Smoothing-passes convergence behaviour documented.
- ⏸ **Smoothing alone is insufficient on the canonical seamount**
  → mitigation path identified (MEO before d-J PGF).

Phase 3a passes its own intrinsic gate (model has a stable regime
matching literature).  But it surfaces that **Phase 3.5 / 4 cannot
proceed without an r-factor mitigation**.  The recommended next step
is implementing MEO before continuing the plan.

## Reference

Beckmann, A., and Haidvogel, D. B., 1993: Numerical simulation of
flow around a tall isolated seamount.  Part I: Problem formulation
and model accuracy.  *J. Phys. Oceanogr.*, **23**, 1736–1753.
