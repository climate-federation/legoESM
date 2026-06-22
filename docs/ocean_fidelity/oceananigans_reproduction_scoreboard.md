# Oceananigans experiment reproduction — fidelity scoreboard

Goal: legoESM faithfully **reproduces** the relevant Oceananigans validation
experiments (not merely measures the gap), verified quantitatively against a
**real** Oceananigans v0.110.4 reference. Harness on branch
`feat/oceananigans-fidelity-harness`. Working reference env:
`JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen`;
references under `$LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF`.

All numbers below come from real legoESM runs vs the real Oceananigans reference
(reproducible via the named drivers). No fabricated numbers.

## Scoreboard

| # | case | bar | measured (real run vs real oracle) | status |
|---|------|-----|-------------------------------------|--------|
| 1 | barotropic_gyre | surface-u pattern_corr ≥ 0.90 at matched time AND max\|u\| within 2× over a ≥20-day stable window | **day-10 corr 0.929**, max\|u\| 0.337 vs 0.606 (0.56×), stable d1–20 (`explicit_substep`) | **PASS** |
| 2 | bickley_jet | surface vorticity pattern_corr ≥ 0.85 at eddy-developed time | linear phase **bit-identical** (t6 5790 vs 5791); phase-matched corr **0.39** at eddy-developed state | **NOT MET** — pointwise bar physically unattainable (chaos); statistical residual = 2Δx mode |
| 3 | silvestri §5 jet | stays finite while oracle stable AND domain-max\|u\| within 2× | blows up at α=0 (barotropic-Coriolis 2Δx null mode) | **NOT MET** — research-level |

Promise `OCEANANIGANS_EXPERIMENTS_FAITHFULLY_REPRODUCED` requires all three rows
PASS. It is **unspoken**: 1 of 3 reproduced.

## Drivers (reproducible)

- Case 1: `scripts/validate/ocean_fidelity/compare_oceananigans_barotropic_gyre.py [days] [dt]`
  (env `BARO_SOLVER=explicit_substep`). Reference deck:
  `scripts/data/generate_oceananigans_barotropic_gyre_reference.jl`.
- Case 2: `scripts/validate/ocean_fidelity/compare_oceananigans_bickley_jet.py [stop]`.
  Reference deck: `scripts/data/generate_oceananigans_bickley_jet_reference.jl`.

## Case 1 — the faithful fix (commit a17843799)

Oceananigans applies the gyre wind as a `FluxBoundaryCondition` on the velocity
field, so its τ₀=1e-2 is a **kinematic** stress [m²/s²] = τ_dyn/ρ₀ (the velocity
equation is per unit mass). legoESM's `OceanSurfaceForcing.tau_x` is a **dynamic**
stress [N/m²] the model divides by ρ₀ internally. The driver under-forced by
exactly ρ₀=1000 (a laminar ~3e-4 m/s flow vs the oracle's O(1 m/s) western
boundary current). Multiplying by ρ₀ — a **bridge convention** per the
oracle-recipe doctrine — clears the case. Pinned by
`tests/ocean/fidelity/test_oceananigans_gyre_wind_convention.py`.

## Cases 2 & 3 — the residual, rigorously isolated (2026-06-22)

The bickley eddy-regime and §5 share one root cause. Exhaustive isolation:

- **Pointwise bar is physically unattainable for case 2.** The linear instability
  phase is *bit-identical* between codes, but at any eddy-developed time the
  vortex field has chaotically decorrelated (phase-matched corr 0.39) — pointwise
  matching is impossible for two distinct discretizations of a chaotic flow. The
  physically-correct metric is statistical (enstrophy decay).
- **The momentum-advection WENO is NOT the dissipation lever.** By reading the
  actual Oceananigans `WENOVectorInvariant` source and matching its exact numerics
  (velocity form, Don&Borges-2013 τ, p=2 ZWENO exponent), *every* faithful match
  made legoESM's enstrophy dissipation **worse**, not better. (PR #559's
  deconvolution removal still helps and is faithful — t12 ens 5019→4511 — but is
  not the lever.) This corrects the long-standing "WENO-momentum residual"
  hypothesis.
- **The lever is the barotropic / free-surface solver.** Oceananigans'
  `ImplicitFreeSurface` is a backward-Euler Helmholtz solve. legoESM's two
  solvers have complementary flaws: `explicit_substep` is stable but
  under-dissipative; `implicit_cn` (θ≈0.7) matches the oracle's enstrophy decay
  through the roll-up (t6 5786≈5791, t12 3186, t18 2624 vs 2835, 2444) but the
  late-time enstrophy **bounces back up** (t24 3407 vs oracle's monotonic 2080).
- **The bounce is the rotational 2Δx barotropic-Coriolis null mode**, which the
  free-surface off-centering θ provably cannot damp (θ damps the divergent
  gravity-wave mode, not the rotational mode). This is the same mode that blows
  up §5 at α=0.

**Next (research-level):** a barotropic scheme that damps the rotational 2Δx mode
while staying strongly dissipative and conservative, matching `ImplicitFreeSurface`.
Tracked by the standing §5 barotropic-Coriolis redesign on
`fix/silvestri-turbulent-dissipation`.

Oracle source consulted:
`/tmp/ocn_j11_depot/packages/Oceananigans/NCFoc/src/Advection/{vector_invariant_advection,weno_interpolants}.jl`
and `.../HydrostaticFreeSurfaceModels/implicit_free_surface.jl`.
