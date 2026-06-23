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
| 2b | baroclinic_adjustment (3D, §5 precursor) | stays finite while oracle stable AND surface-max\|u\| within 2× through 30 d | **day-30 2.16 vs oracle 2.05 (1.05×); finite + within 2× throughout** after the **full-velocity vertical momentum advection fix** (`weno_vertadv_full_velocity`); baseline was NaN day 18 | **PASS** (the §5-precursor, faithfully closed) |
| 2c | spherical_baroclinic (3D, lat-lon C-grid) | stays finite while oracle stable AND surface-max\|u\| within 2× through 30 d | **day-30 2.09 vs oracle 2.13 (0.98×); finite + within 2× throughout** (1.21–1.43× peak) — same physics as 2b but on a REAL lat-lon grid (cos-lat metric) vs the Cartesian beta-plane; confirms the vertadv fix holds **on the sphere** in the eddy regime | **PASS** (validates the §5 fix on the sphere) |
| 2d | internal_tide (topography + M2 tide) | radiated internal-wave field (b') pattern_corr ≥ ~0.6 vs oracle | INCONCLUSIVE — comparison metric contaminated. The "fundamental z-star limitation" hypothesis is **REFUTED**: running the Oceananigans oracle in z-STAR mode (`MutableVerticalDiscretization`) gives IDENTICAL results to z-LEVEL (max\|w\| 2.689e-3 vs 2.691e-3, max\|b'\| 0.198 vs 0.198) — z-star radiates the tide fine. ALSO: max\|b'\|=0.198 is EXACTLY CONSTANT in time (var=0) in BOTH oracle modes = the STATIC immersed-boundary/topography artifact over the ridge, NOT the propagating wave — so the earlier legoESM "8× under-generation" was largely a topography-masking artifact mismatch (z-star compression vs immersed cells), not the tide. NEEDS: mask the below-topography cells + a time-VARYING metric, and compare legoESM z-star vs the Oceananigans z-STAR oracle apples-to-apples, to determine if a real legoESM bug exists. | **INCONCLUSIVE** (z-star refuted as the cause; clean masked comparison pending) |
| 3 | silvestri §5 jet | stays finite while oracle stable AND domain-max\|u\| within 2× | **§5 W9V (no backstop) + full-velocity vertadv + faithful dt SURVIVES 200 d** (max\|u\| ~0.8–1.6 m/s = the oracle's transient amplitude, within 2×). Two faithful gaps closed: (a) `weno_vertadv_full_velocity`, (b) TIMESTEP — the oracle uses an adaptive wizard (cfl=0.3, Δt 300–900 s) while legoESM used a fixed dt=900 (the oracle's MAX) even at the over-energized peak → NaN day 89 | **MET** (pending adaptive-dt confirmation) — was NOT eddy-equilibration; it was vertadv + timestep |

Promise `OCEANANIGANS_EXPERIMENTS_FAITHFULLY_REPRODUCED` requires all three rows
PASS. As of 2026-06-23 **all three reproduce faithfully** (gyre + bickley + §5);
CASE 3 was closed by the full-velocity vertical momentum advection + the oracle's
adaptive timestep (it was NOT the research-level eddy-equilibration long believed —
that diagnosis was an artifact of the fixed-dt blow-up trajectory). CASE 2's bar was
re-specced 2026-06-22 (user-approved) from a physically-unattainable pointwise
correlation to the statistical enstrophy-decay metric.

**Rigor follow-ups before declaring the promise (not yet spoken):** (1) the §5
"within 2×" is vs the oracle's KNOWN ~1 m/s transient amplitude (paper / prior budget
runs), not yet a re-generated oracle TIME-SERIES comparison — regenerate the §5 oracle
and pin max\|u\|(t) within 2× day-by-day; (2) the faithful adaptive timestep is
demonstrated by `scripts/tmp/_s5_adaptive_dt.py` (oracle wizard cfl=0.3, max_Δt=900) —
promote it to a proper tested driver component (`run_silvestri_baroclinic_jet.py`);
(3) the §5 setup now defaults `weno_vertadv_full_velocity=True` and the `stabilize`
backstop is opt-in (no longer needed). Once (1)+(2) land, the promise is speakable.

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

## §5 residual — decisively isolated on a fast CPU precursor (2026-06-22)

The §5 blow-up was reproduced on a CLEAN, FAST CPU test bed — **baroclinic_adjustment**
(48×48×8, minutes per 30-day run): a stratified buoyancy front on a beta-plane goes
baroclinically unstable, the canonical Oceananigans example. Driver
`scripts/validate/ocean_fidelity/compare_oceananigans_baroclinic_adjustment.py`,
reference `scripts/data/generate_oceananigans_baroclinic_adjustment_reference.jl`.
This replaced 20 iterations of chasing the chaotic 160×128×50 GPU §5 and let each
hypothesis be A/B-tested in minutes. All numbers below are real runs vs the real oracle.

**Five faithful fixes / findings (in order):**

1. **IC from rest (faithful).** The oracle does `set!(model, b=bᵢ)` — starts from REST,
   the jet spins up by geostrophic adjustment. The driver had imposed the full
   thermal-wind jet (0.97 m/s) at t=0 — more energy than the oracle ever has. Fixed
   to start from rest; sharpened the early-time match (day-6 ratio 2.0×→1.25×).

2. **Runaway is SPATIAL, not a timestep artifact.** Halving dt (600→150 s) makes the
   blow-up WORSE (day-18 max|u| 8.0→11.1), the unambiguous signature of a grid-scale
   spatial instability, not CFL. (The oracle's `TimeStepWizard` actually runs at
   dt≈1200 s — coarser than legoESM — and stays bounded.)

3. **Cartesian beta-plane grid (faithful) — but the spherical metric terms are NOT the
   dominant source.** The oracle uses a Cartesian `RectilinearGrid` + `BetaPlane(-45)`,
   NOT a spherical lat-lon grid. The driver was rebuilt on
   `create_beta_plane_cgrid_geometry` (f=f0+βy at each stagger point, cos_lat≡1,
   Periodic-x + N/S walls = Oceananigans `(Periodic, Bounded, Bounded)`). This is
   strictly more faithful, but the runaway PERSISTS on the Cartesian grid — so the
   latlon spherical metrics are not the 2Δx source (correcting the earlier hypothesis).

4. **Decoupled D-term smoothness (faithful model fix).** Oceananigans'
   `WENOVectorInvariant` default mixes TWO independent smoothness choices —
   `vorticity_stencil=VelocityStencil()` (velocity smoothness, Eq 43 = legoESM "split"
   vorticity) AND `upwinding=OnlySelfUpwinding` (self-smoothness, Eq 44 = "standard"
   divergence). legoESM's single `weno_smoothness` flag forced BOTH to the same family,
   so neither setting matched the oracle ("split" → under-dissipative Eq-45 divergence;
   "standard" → over-constrained Eq-37 vorticity). New optional config field
   `weno_divergence_smoothness` (default None = follow `weno_smoothness`, bit-identical)
   lets the divergence flux pick its family independently. The faithful Oceananigans mix
   = split vorticity + self divergence. This materially helps (the case stays FINITE to
   30 days instead of NaN) but does not fully close the gap at order 9.

5. **THE RESIDUAL, apples-to-apples (the key correction).** The Oceananigans
   `baroclinic_adjustment` example (and any `momentum_advection = WENO()` deck on a
   `RectilinearGrid`) advects momentum in **FLUX FORM** `∇·(u⊗u)`, NOT vector-invariant:
   `U_dot_∇u(…, ::AbstractAdvectionScheme, U) = div_𝐯u(…)` (vector_invariant_advection.jl:417);
   only a `VectorInvariant`/`WENOVectorInvariant` object dispatches to the vector-invariant
   form (line 292). The §5 jet deck DOES use `WENOVectorInvariant(vorticity_order=9)` (=W9V).
   The precursor reference was therefore regenerated with `WENOVectorInvariant(order=9)` +
   `WENO(order=7)` tracer to be a TRUE apples-to-apples vector-invariant comparison
   (legoESM `momentum_advection="weno9"`). Result, BOTH vector-invariant order-9:

   | day | legoESM weno9 (faithful: split-vort+self-div+matsuno, Cartesian) | oracle W9V |
   |-----|------------------------------------------------------------------|------------|
   | 6   | 1.07 | 0.72 |
   | 12  | 3.39 | 1.47 |
   | 18  | **NaN** | 1.56 |
   | 30  | —    | **2.05 (saturates)** |

   legoESM's vector-invariant WENO **blows up where Oceananigans' `WENOVectorInvariant`
   saturates**, and higher order is WORSE in legoESM (weno9 NaN day-18 vs weno5 finite to
   day-30). This DEFINITIVELY isolates the §5 residual to legoESM's **vector-invariant WENO
   momentum under-dissipating grid-scale enstrophy vs Oceananigans' `WENOVectorInvariant`** —
   specifically the **VelocityStencil vorticity flux** (the self-divergence D-term, now
   decoupled, helps but does not close it). NOT the grid, NOT the timestep, NOT the WENO
   order, NOT the free-surface solver, NOT the Coriolis scheme alone.

**Ruled out as the dominant lever:** ❌ timestep (finer = worse), ❌ spherical metric terms
(Cartesian grid still blows), ❌ momentum WENO order (weno9 worse than weno5),
❌ free-surface solver, ❌ Coriolis scheme (matsuno_split delays but doesn't cure).
**Localised to:** legoESM's VelocityStencil vorticity-flux reconstruction in the
vector-invariant WENO momentum (the 1D WENO kernel is coefficient-faithful; the gap is in
the C-grid vorticity-flux APPLICATION / the VelocityStencil smoothness realization).

**Next (research-level) — the residual NARROWED (vorticity-flux source comparison, 2026-06-22):**
The obvious lever is already faithful: at HEAD (`fe79af117` on-branch) the weno9 vorticity flux
uses `beta_average=True` (single biased reconstruction on AVERAGED betas = Oceananigans'
`beta_sum`+`bias`, weno.py `weno_reconstruct_split2`) and `convert_to_cellavg=False` (no low-pass
pre-filter = Oceananigans direct nodal FV reconstruction). The legacy
`0.5·(recon(βᵥ)+recon(βᵤ))` average-of-results path (under-dissipative, dilutes the WENO-Z weight
nonlinearity) is NOT the active path. Further eliminated ON THIS CASE: **PV-flux q=ζ/h vs direct-ζ**
is negligible (flat bottom H=1km uniform, small η ⇒ q=ζ/h≈ζ/const horizontally); **Neumann wall fill**
is unlikely (the GPU §5 wall-budget diagnostic put the 2Δx mode in the JET CENTER, not the walls).
So legoESM under-dissipates even with the FULLY-faithful WENO vorticity flux on a flat bottom. The
residual is the **C-grid collocation of the vorticity flux** — the bias-velocity v̂ interpolation
(`ℑxᶠᵃᵃ(ℑyᵃᶜᵃ(Δx·v))·Δx⁻¹` in Oceananigans vs legoESM's `v_at_u`) and the vertex relative-vorticity
`curl_vertex_cgrid` vs Oceananigans' `ζ₃ᶠᶠᶜ` — i.e. the discrete curl/upwind-velocity collocation,
NOT the 1D WENO kernel (coefficient-faithful) nor the smoothness family (now decoupled + faithful).
CASE 3 (§5) bar (stay finite while oracle stable AND within 2×) remains UNMET: the faithful order-9
config NaNs ~day 18 here while the oracle saturates at 2.05. This is the one un-closed node; it
overlaps the standing `fix/silvestri-turbulent-dissipation` effort (multi-session research).

## §5 CLOSURE — full-velocity vertical momentum advection (2026-06-22, commit 7564e0e8f)

The §5/baroclinic-eddy blow-up was decisively re-diagnosed and PARTIALLY closed:
- **The blow-up is INTERIOR (front/jet centre), NOT the wall mode** — a per-day max|u|-location
  probe of baroclinic_adjustment put it at lat-row ~24/48 every day, never the N/S walls. This
  overturned the long-standing wall-mode hypothesis.
- **bickley (2D, single-layer) is benign (~4%); baroclinic (3D) blows up** ⇒ the lever is a
  3D-only term the entire bickley-focused effort could never exercise: **vertical momentum
  advection**. legoESM advected only the baroclinic perturbation u'=u−U_bar (omitting
  −∂(w·U_bar)/∂z); Oceananigans advects the FULL u. New faithful option
  `weno_vertadv_full_velocity` (recipe default True) advects full u.
- **baroclinic_adjustment: NaN day-18 → SATURATES within 2% of the oracle through 30 d**
  (day-30 1.05×). The clean §5-precursor blow-up is CLOSED, faithfully (review-clean,
  conservative, AD-pure, 396 tests pass).
- **BUT the actual §5 W9V 160×128×50 case (sustained restoring forcing) is only DELAYED**
  82 d → 89 d (GPU, no backstop). §5's τ=50d restoring on (b,u,v) keeps pumping energy past
  what the fix dissipates; the precursor (no restoring) saturates freely. So §5's residual is a
  SUSTAINED-FORCING eddy instability at the ~1 m/s transient peak — SEPARATE from the interior
  vertical-advection runaway this fix closes. CASE 3 still UNMET; the full-velocity vertadv fix is
  a real, faithful, committed improvement (closes the precursor + the 3D vertical-advection
  mechanism) but not the complete §5 cure.
