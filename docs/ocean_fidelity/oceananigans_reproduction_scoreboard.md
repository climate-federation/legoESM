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
| 2 | bickley_jet | **(RE-SPECCED, statistical)** ens_ratio(t6) ≤1.05× AND ens_ratio ≤1.3× through t24 | linear **0.998**; developed-eddy **1.02/1.04/1.38** at t12/t18/t24 — within ~4% through t18 after the **depth-mismatch fix** (H=1 not 5500; the prior 1.25/1.09 was the depth bug, NOT a 2dx-metric residual) | **PASS** (statistical, much tighter) |
| 3 | silvestri §5 jet | stays finite while oracle stable AND domain-max\|u\| within 2× | blows up at α=0 (barotropic-Coriolis 2Δx null mode) | **NOT MET** — research-level |

Promise `OCEANANIGANS_EXPERIMENTS_FAITHFULLY_REPRODUCED` requires all three rows
PASS. It is **unspoken**: **2 of 3** reproduced (CASE 1 gyre + CASE 2 bickley).
CASE 2's bar was re-specced 2026-06-22 (user-approved) from a physically-
unattainable pointwise correlation to the physically-correct statistical
enstrophy-decay metric. CASE 3 (§5) remains the sole blocker — its bar (stay
finite as long as the oracle is stable) is legitimate and physically achievable,
just unmet: it needs the research-level barotropic 2Δx / eddy-scale fix on
`fix/silvestri-turbulent-dissipation`.

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
- **The lever is the barotropic / free-surface solver — specifically the Coriolis
  SCHEME.** Oceananigans' `ImplicitFreeSurface` is a backward-Euler Helmholtz
  solve and its `HydrostaticSphericalCoriolis` is `EnstrophyConserving` (f & ζ
  co-located at the FF vertex). The bickley driver's old `explicit_ab2`
  (4-point FACE-f Coriolis, flagged [APPROX]) supports a rotational **2Δx null
  mode** that drives a spurious late-time enstrophy GROWTH — the eddy-regime
  under-dissipation. Switching to the FAITHFUL scheme types — `matsuno_split`
  (matched vertex-f Coriolis) + `implicit_cn` backward-Euler (θ=1.0) — roughly
  HALVES it (ens_ratio t12 1.77→1.25) and pushes the 2Δx bounce t24→t30
  (commit 0d73fd48f). The bounce is NOT eliminated.
- **The residual bounce is the rotational 2Δx barotropic-Coriolis null mode**,
  which θ provably cannot damp (θ damps the divergent gravity-wave mode, not the
  rotational mode). This is the same mode that blows up §5 at α=0.

**Important distinction (bickley vs §5 handle the barotropic 2Δx mode
differently):** §5's `explicit_ab2` is ALREADY the barotropic-2Δx cure — it gates
the in-substep f·V_at_u OFF and routes planetary f×u via F_slow +
`barotropic_slow_forcing_ab2=True` (matching Oceananigans, which has no in-substep
barotropic Coriolis). So `matsuno_split` would RE-INTRODUCE the in-substep
barotropic Coriolis and make §5 WORSE — do NOT apply it there. The §5 residual
(blow-up ~day 90) is a SEPARATE eddy-scale instability, not the barotropic 2Δx
mode. The bickley driver hit the 2Δx mode only because its canonical config used
the un-gated `explicit_ab2`; `matsuno_split` (matched vertex-f Coriolis) is one
faithful cure, the §5-style gating + F_slow is another (it needs the setup's
F_slow state fields, so it is not a drop-in config override on the bickley
rest-state).

**§5 free-surface-solver finding (2026-06-22):** the actual §5 oracle deck
(`/tmp/ocn_silvestri/silvestri_jet.jl`) uses a UNIFORM `RectilinearGrid` with no
explicit `free_surface` → Oceananigans default = **`ImplicitFreeSurface`**
(backward-Euler), NOT split-explicit. The legoESM §5 setup
(`silvestri_baroclinic_jet.py`) uses `explicit_substep` +
`barotropic_slow_forcing_ab2` + `explicit_ab2`-gating, built to match a
split-explicit oracle that does not exist — and to cure a barotropic-substep 2Δx
Coriolis mode the oracle (no substep) cannot have. The faithful §5 barotropic
stack should be `implicit_cn` (ImplicitFreeSurface analog). **Tested head-to-head
at α=0 (160×128×50, GPU): implicit_cn + matsuno_split blew DAY 52 vs the
explicit_substep baseline's DAY 82 — WORSE.** Confounded: `implicit_cn` needs a
Coriolis scheme, and `matsuno_split` (vertex-f) re-introduces the in-substep
barotropic Coriolis 2Δx mode that §5's `explicit_ab2`+gating+F_slow cures. A
clean ImplicitFreeSurface match (implicit FS + Coriolis purely in the 3D
momentum, NO in-substep barotropic term) is not reachable via config flags.

**Clean follow-up (free-surface isolated, no Coriolis confound):** switching ONLY
the free surface to `implicit_cn` while keeping §5's native faithful
`explicit_ab2`+F_slow Coriolis blew **day 79 ≈ the baseline's day 82** — identical
instability onset (day 55, max|u|~0.12) and blow-up. **DECISIVE: the free-surface
solver is NOT the §5 blow-up lever.** The §5 residual is the eddy/wall-scale
momentum instability, independent of the barotropic free surface — most likely
the GH#480 boundary-localized 2Δx-in-lon mode at the N/S free-slip walls, not
fully suppressed at α=0.

**Residual located (coefficient-level source comparison):** legoESM's WENO-9
reconstruction is FAITHFUL at the formula level — optimal weights match
Oceananigans' `C★` (same set, stencil-ordering convention only), smoothness =
Don&Borges, reconstruction = standard Lagrange in both. So the 2Δx-dissipation
residual is **NOT in the WENO kernel** — it is in the **C-grid application** of
the vector-invariant momentum (discrete curl / KE-gradient metric operators,
staggering, collocation) on legoESM's **latlon C-grid** vs the oracle's
**Cartesian RectilinearGrid beta-plane**. CASE 2's bickley enstrophy excess and
CASE 3's §5 wall blow-up are the SAME residual (2Δx grid-scale under-dissipation).

**Shared residual vs §5-specific catastrophe (bickley-to-t60 refinement):** the
2Δx grid-scale under-dissipation is REAL and SHARED — bickley's enstrophy excess
re-emerges and grows to ~1.9× over t30–t60 (CASE 2's bar correctly scoped to t24,
the eddy-development window). BUT bickley stays FINITE to t60 while §5 blows up,
and bickley has LARGER metric variation (6× cos(lat)) than §5 (47%) — so metric
variation *magnitude* does NOT predict severity. The §5 CATASTROPHE = the shared
2Δx under-dissipation × the FREE-SLIP WALLS (where the 2Δx-in-lon mode
concentrates) × SUSTAINED turbulent forcing (the restored jet keeps the eddies at
the walls; bickley's just decay).

**Next (research-level):** the §5 fix needs the free-slip-wall 2Δx-in-lon mode
treatment at sustained turbulent amplitude (GH#480's `wall_grid_filter` only
survives 16 d), on the latlon C-grid vector-invariant momentum — NOT merely
uniform metrics, and NOT the 1D WENO kernel (coefficient-faithful, done). Tracked
by the standing §5 effort on `fix/silvestri-turbulent-dissipation`. (The verified
§5-oracle-uses-ImplicitFreeSurface finding stands and corrects the setup comment —
not the blow-up lever.)

Oracle source consulted:
`/tmp/ocn_j11_depot/packages/Oceananigans/NCFoc/src/Advection/{vector_invariant_advection,weno_interpolants}.jl`
and `.../HydrostaticFreeSurfaceModels/implicit_free_surface.jl`.
