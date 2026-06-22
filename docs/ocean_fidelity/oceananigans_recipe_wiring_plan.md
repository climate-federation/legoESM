# Plan: an Oceananigans recipe in legoESM via wiring-diagram matching

Status: DRAFT (2026-06-22). Owner: ocean-fidelity. Branch: `feat/oceananigans-fidelity-harness`
→ feeding `fix/silvestri-turbulent-dissipation` for the dynamics work.

## 0. The reframe

We stop chasing individual cases and instead make legoESM's lat-lon C-grid ocean
dynamics a **faithful Oceananigans recipe**: a single `oceananigans` recipe in the
catalog whose dynamics graph (components + order + operators) MATCHES
Oceananigans' `HydrostaticFreeSurfaceModel`. Once the wiring matches, the relevant
Oceananigans experiments reproduce as a consequence, instead of one-off patches.

This is the oracle-recipe doctrine taken to its conclusion: the recipe selects
shared canonical blocks; the blocks are made to match Oceananigans' wiring as
**selectable options** (never a bespoke `veros_*`/`oceananigans_*` solver).

## 1. Why wiring-diagram matching (what the case grind taught us)

The case-by-case work (gyre PASS, bickley PASS, §5 NOT MET) showed the residuals
are NOT individual knobs but **structural** choices in the computational graph:
- The WENO reconstruction KERNEL is coefficient-faithful (optimal weights, smoothness,
  reconstruction coeffs all match) — matching it harder makes things *worse*. So the
  residual is NOT the 1D kernel; it is in how terms are **assembled and ordered**.
- The §5 day-82 blow-up is the **interior eddy 2dx grid-scale under-dissipation**
  (same residual as bickley's enstrophy excess), in the **C-grid vector-invariant
  momentum metric application** — a structural/operator difference, not a parameter.
- Multiple independent levers (Coriolis scheme, free-surface solver, timestep, wall
  treatment) each moved the answer the WRONG way or not at all, because each is one
  node of a graph that differs from Oceananigans' in several places at once.

Conclusion: diff the WHOLE graph, classify every node, and match node-by-node.

## 2. Methodology

1. **Extract both wiring diagrams** as ordered node lists (stage → term → operator),
   with file:line. (Oceananigans extraction in flight via a source-reading agent.)
2. **Diff** node-by-node: classify each as MATCH / DIFFERENT-OPERATOR /
   DIFFERENT-ORDER / MISSING / EXTRA, and tag the expected fidelity impact.
3. **Targeted coding**: for each mismatch, add the Oceananigans form as a SELECTABLE
   canonical option (config-selected), default OFF (bit-identical), wired through the
   `oceananigans` recipe. One mismatch → one option → one tendency-match test.
4. **Validate** per node with a per-term TENDENCY MATCH against an Oceananigans
   reference (single-step, identical IC), then per CASE with the fidelity bars.

## 3. The two wiring diagrams

### 3a. legoESM lat-lon C-grid ocean step (extracted; `ocean_pe_latlon_cgrid.py`)

Tendency assembly order (one step), then the outer integrator + barotropic solver:

1. geometry / density / EOS / pressure anomaly (`_bc_geometry_and_density`)
2. vertical + depth-mean velocity; w from flux divergence (`_bc_vertical_and_depthmean_velocity`)
3. KE gradient + hydrostatic + free-surface pressure gradients (`_bc_ke_and_pressure_gradients`)
4. **Stage 7b** horizontal momentum advection — vector-invariant PV flux `q=ζ/h`
   (WENO, `_bc_pv_flux`) OR flux-form
5. **Stage 7b′** PLANETARY Coriolis as a SEPARATE explicit tendency (Veros-faithful;
   `coriolis_cgrid` face-f, routed to the barotropic mode under `explicit_ab2`)
6. **Stage 7c** WENO divergence "D-term" momentum dissipation (`_bc_dterm`)
7. **Stage 8** flux-form vertical momentum advection (`_bc_vertical_momentum_advection`)
8. **Stage 8b** GH#480 N/S wall grid-mode filter (default OFF)
9. **Stage 9** tracer diffusion tendencies
10. **Stage 10b/10b′/10b″** physics pipeline / surface forcing / surface restoring
11. **Stage 10c** sponge relaxation
12. **Stage 12** free-surface (η) tendency
13. outer time integrator (**AB2**), then the **barotropic solver**
    (`implicit_cn` / `explicit_substep`)

### 3b. Oceananigans `HydrostaticFreeSurfaceModel` step (extracted from source)

Default (regular grid): **QuasiAdamsBashforth2** stepper, **ImplicitFreeSurface**,
**VectorInvariant** (vorticity=EnstrophyConserving, vertical=EnergyConserving),
**HydrostaticSphericalCoriolis EnstrophyConserving** (vertex-f, Sadourny).

One step `time_step!` (`quasi_adams_bashforth_2.jl:90`):
- `χ=0.1` (Euler χ=−0.5 on first step / Δt-change); **tendencies Gⁿ were computed at
  the END of the PREVIOUS step inside `update_state!`** (`update_..._state.jl:80-81`).
- `ab2_step!` (predictor–corrector, `hydrostatic_free_surface_ab2_step.jl:88`):
  A copy velocities→transport_velocities; B momentum-flux BCs;
  **C predictor velocity step** `u* = uⁿ + Δt[(3/2+χ)Gⁿ−(1/2+χ)G⁻]` + implicit vertical
  viscosity (NO barotropic η-gradient in the predictor); D open-bdy;
  **E free-surface solve** (ImplicitFreeSurface backward-Euler Helmholtz for ηⁿ⁺¹ from
  the predictor barotropic flux 𝐐⋆); **F barotropic correction** `u −= gΔt ∂ₓηⁿ⁺¹`;
  G mask immersed; H fill u,v halos; I compute transport velocities + w̃;
  **J tracer tendencies** (using the barotropic-corrected transport velocities);
  K z-star σ; **L tracer AB2 step** + implicit vertical diffusion.
- `cache_previous_tendencies!` (Gⁿ→G⁻); `tick!`; closure prognostics;
  **`update_state!`** → mask/halos, buoyancy grads, **w from continuity**
  (`compute_w_from_continuity.jl`), **hydrostatic pressure pHY′**, closure fields, then
  **compute Gⁿ for the NEXT step**.

Momentum tendency `Gu` term order (`kernel_functions.jl:44-51`, all negated but forcing):
1 `U_dot_∇u` = vorticity flux (relative ζ, EnstrophyConserving or WENO/VelocityStencil)
+ vertical advection + Bernoulli/KE-gradient head; 2 curvilinear metric advection;
3 barotropic η-gradient (**=0 for ImplicitFreeSurface** — applied in step F instead);
**4 Coriolis SEPARATE** `f×U` (vertex-f `fᶠᶠᵃ=2Ω sinφ`, area/transport-weighted Sadourny);
5 hydrostatic pressure-anomaly gradient `∂pHY′`; 6 closure; 7 immersed; 8 forcing.

### 3c. Node-by-node diff table (CORRECTED with the source extraction)

| node | legoESM | Oceananigans | class | impact |
|------|---------|--------------|-------|--------|
| time-stepper | AB2 outer | **QuasiAB2** (χ=0.1, Euler-start); Gⁿ computed in prior step's update_state | DIFFERENT(-detail) | medium |
| **free-surface COUPLING structure** | free-surface PG inside the tendency; barotropic solver | **predictor–corrector**: barotropic η-gradient OUTSIDE the tendency, applied as `u−=gΔt∂η` AFTER the η solve; tracers use barotropic-corrected transport velocity | **DIFFERENT-ORDER** | **high** |
| planetary Coriolis PLACEMENT | SEPARATE explicit term (Stage 7b′) | SEPARATE explicit term (`f×U`) | **MATCH** (both separate) | — |
| Coriolis DISCRETIZATION | face-f `explicit_ab2` (routed to barotropic via F_slow) or vertex-f matsuno | **vertex-f EnstrophyConserving (Sadourny), area/transport-weighted, in-tendency** | DIFFERENT-OPERATOR | medium–high |
| vorticity flux form | PV `q=ζ/h` × mass flux `h·v` | **relative ζ** reconstructed × transport velocity (f NOT in it) | DIFFERENT-OPERATOR | medium |
| WENO reconstruction kernel | WENO-Z (coeffs/weights) | WENO-Z (coeffs/weights) | **MATCH** | — |
| KE-gradient | Bernoulli head, WENO-5 | Bernoulli head, WENO-5 | MATCH | — |
| hydrostatic PGF | in tendency | in tendency (`∂pHY′`, explicit) | MATCH | — |
| free-surface solver | implicit_cn / explicit_substep | ImplicitFreeSurface backward-Euler Helmholtz | DIFFERENT | medium |
| tracer advection velocity | (check) | **barotropic-corrected transport velocity** | check | medium |
| w from continuity | Stage 4 (in tendency) | in `update_state!` end-of-step | DIFFERENT-ORDER | low |
| free-slip wall BC | masked-land + neumann-fill of q | Bounded + mirror halo + near-wall order reduction | DIFFERENT | low (wall mode already solved) |
| horizontal grid/metric | lat-lon (cos(lat) varies ~47% across §5 channel) | Cartesian RectilinearGrid (uniform) | **DIFFERENT** | **high (interior 2dx under-dissipation)** |

## 4. Structurally most important choices (the high-impact nodes) — CORRECTED

⚠️ My earlier "Coriolis folded into the vorticity flux (f+ζ)" hypothesis is **WRONG**:
Oceananigans keeps Coriolis SEPARATE (relative-ζ flux + a separate `f×U` term), exactly
like legoESM. So the high-impact nodes are:

1. **Free-surface coupling STRUCTURE (predictor–corrector)** — Oceananigans keeps the
   barotropic η-gradient OUT of the momentum tendency and applies it as `u −= gΔt ∂η`
   AFTER a backward-Euler η-solve of the PREDICTOR transport, then advects tracers with
   the barotropic-CORRECTED velocity. legoESM's coupling differs in where/when the
   free-surface pressure gradient enters → affects the divergent/eddy energy balance.
2. **Grid/metric application** — Cartesian uniform vs lat-lon cos(lat)-varying metrics
   in the discrete curl / Bernoulli-head / Coriolis area-weights. The 2dx-in-lon
   dissipation is metric-dependent → the interior eddy 2dx under-dissipation residual.
3. **Coriolis discretization** — vertex-f EnstrophyConserving (Sadourny, area/transport
   weighted), applied in-tendency, vs legoESM's face-f explicit_ab2 (routed to the
   barotropic mode). Match the Sadourny vertex-f form as an in-tendency option.
4. **Vorticity-flux form** — relative-ζ × transport velocity vs legoESM PV `q=ζ/h`.
5. **Time-stepper** — QuasiAB2 (χ=0.1, Gⁿ from prior step) vs legoESM AB2.

### 3d. Phase-1 tendency-match RESULTS (2026-06-22) — momentum tendency mostly MATCHES

Harness: `scripts/data/generate_oceananigans_tendency_reference.jl` (dumps Oceananigans
one-step Gu/Gv from the bickley IC) + `scripts/validate/ocean_fidelity/compare_oceananigans_tendency.py`
(legoESM du_dt from the same IC; IC alignment validated at corr 1.0000). At the IC
η=0 + buoyancy=nothing → PGF=0, so this isolates ADVECTION + CORIOLIS.

Findings:
- **Gv matches at 0.9998** (nrmse 0.004) once legoESM's planetary Coriolis is added
  (legoESM applies f×u OUTSIDE du_dt — it has NO coriolis diagnostic term; the
  face-f `coriolis_cgrid` already reproduces Oceananigans' Gv for this flow).
- **Gu matches at 0.9962** (nrmse 0.009) **once the additive Silvestri D-term is
  dropped**. WITH the D-term, Gu is corr 0.975 / +40% magnitude. Decomposition:
  vortcor 0.196, KE-grad 0.109, **Dterm 0.079** (≈ the whole Gu excess).
- ⇒ **the legoESM momentum-tendency wiring already MATCHES Oceananigans** for
  advection+Coriolis, with ONE extra node: the **additive D-term** (Oceananigans
  folds divergence into the OnlySelfUpwinding — no separate additive term).

Implications:
- The instantaneous momentum dynamics are faithful → the §5 interior-2dx residual
  is NOT the advection/Coriolis tendency. It is in the parts NOT yet tendency-matched:
  the **free-surface predictor-corrector coupling** (needs an η≠0 reference) and the
  **time-stepper** — consistent with the wiring diff's high-impact node #1.
- The D-term ADDS dissipation yet §5 still blows up → removing it (to match
  Oceananigans) makes §5 LESS dissipative, so the §5 fix is NOT the D-term; the
  D-term is just a faithfulness mismatch to expose as a recipe option.

### 3e. Phase-2 free-surface COUPLING result (2026-06-22) — the §5-residual node, localized

Harness: `scripts/data/generate_oceananigans_geostrophic_adjustment_reference.jl`
(unbalanced η Gaussian bump, u=v=0, ImplicitFreeSurface, deterministic adjustment)
+ `scripts/validate/ocean_fidelity/compare_oceananigans_geostrophic_adjustment.py`.

Findings:
- IC η aligns (corr 0.997). Oceananigans adjusts: the bump drives a geostrophic
  flow max|u| 0.011→0.034 over 40 steps.
- **legoESM (implicit_cn) drives ~NO flow** (max|u| ~2e-5, decaying) — ~100×
  too weak; η pattern decorrelates.
- ROOT: at the η-bump IC legoESM's `KE_PGF_u = 0`, `du_dt = 0`, `deta_dt = 0`
  → **the free-surface η pressure gradient is ABSENT from the 3D momentum
  tendency**; the η→flow coupling is handled ENTIRELY in the barotropic solver,
  which (implicit_cn) does not drive the geostrophic flow from an η bump.
- `explicit_substep` DOES drive flow (max|u| 3e-3 at step 10) but is gravity-wave
  CFL-unstable at dt=0.01 (NaN by step 20).

⇒ **The free-surface COUPLING is the concrete legoESM↔Oceananigans difference.**
ROOT CAUSE (η-evolution diagnostic, dt=0.01): legoESM's `implicit_cn` **collapses
the η bump 88% in the first 10 steps** (0.0997→0.0123) then plateaus; the oracle
decays gradually (0.099→0.052 over 40 steps). So the implicit_cn free-surface
SOLVE is **catastrophically over-damping the η field**, killing the pressure
gradient before the geostrophic flow can develop. θ-INDEPENDENT (even θ=0.5 gives
max|u| 1.8e-4 vs oracle 3.4e-2 — ~200× weak). The Crank-Nicolson PG itself is
correct (net `−g·dt·[(1−θ)∂η_old+θ∂η_new]`, not cancelled). **BOTH legoESM
barotropic solvers over-damp η** — explicit_substep (dt=0.002, stable) also
collapses η 0.099→0.037 by t=0.1 (vs oracle 0.095) and drives only u~4e-4. So it
is a GENERAL legoESM free-surface over-damping of the η field, NOT implicit_cn-
specific. The barotropic solver DOES use `state.eta` (u develops ~1e-4/step), but
the η bump radiates/damps ~10-20× too fast vs the oracle (which preserves ~95% of
the bump at t=0.1, physically correct for the slow gravity-wave radiation of a
0.5-rad bump at c=1).

This is high-impact node #1 AND the suspected §5/bickley eddy-residual home,
reproduced in a clean deterministic test.

**ROOT-CAUSE PINNED (1-step debug, BARO_SOLVER=implicit_cn):** the implicit
Helmholtz SOLVE collapses η in ONE step: RHS = eta_old (0.0997, confirming the
predictor/RHS algebra cancels to eta_old) but `solve_helmholtz_freesurface`
returns η_new = 0.0357. NOT under-convergence (400 PCG iters, resid 2.4e-3, gives
the SAME 0.0357) → 0.0357 IS the converged solution of `A·η=eta_old`, so the
operator `A = I − coeff·∇·(H∇)` **amplifies the smooth bump ~2.8×** (equivalently
the inverse over-SMOOTHS η). SCALE-SELECTIVE: σ=30°(~5cell)→×0.36,
60°(~11cell)→×0.63, 120°(~21cell)→×0.85 — a diffusion that hammers high
wavenumbers. The effective `coeff·∇²` for the 5-cell mode is ~1.8, but
`coeff=g·dt²=1e-4` and CFL²~0.01 predict ~0.01 → **the operator is ~50–100× too
strong**, a METRIC SCALING BUG in `_helmholtz_apply` / `_helmholtz_coupling_pieces`
(the dy_u/dx/area combination on the lat-lon C-grid; the ~100× ≈ a missing/extra
metric factor, possibly deg-vs-rad or an area mis-weight). Bickley masks it because
there η develops FROM the flow each step (flow→η); the geostrophic adjustment is
the η→flow case the over-damping kills.

NEXT: trace the exact metric factor in `_helmholtz_apply`/`_helmholtz_coupling_pieces`
(barotropic_implicit_latlon_cgrid.py:216,243) against the geostrophic-adjustment
1-step test (eta_new should ≈ eta_old for a smooth bump); FIX it; re-check
gyre/bickley/§5. This is now a SPECIFIC, CPU-testable operator bug — not a chaotic
research problem.

## 5. Phased plan

- **Phase 0 — wiring diagrams + diff** (in progress): finish 3b, complete 3c, rank
  the mismatches by expected impact. Deliverable: this doc, finalized.
- **Phase 1 — single-step tendency match harness**: identical-IC one-step Gu/Gv/Gtracer
  comparison legoESM vs an Oceananigans reference (extend the fidelity harness). This
  is the per-node validation oracle (tier-3 tendency match).
- **Phase 2 — match the high-impact nodes as selectable options**, in impact order:
  (1) absolute-vorticity Coriolis-in-the-vorticity-flux, (2) the metric application,
  (3) free-surface coupling default, (4) time-stepper. Each: option + tendency-match
  test + case re-run. Default OFF; the `oceananigans` recipe turns the matched set ON.
- **Phase 3 — the `oceananigans` recipe + broadened case suite** (see §6).
- **Phase 4 — close §5** as the capstone (the interior 2dx residual should fall out
  of the metric + Coriolis matching).

## 6. Broadened Oceananigans case suite (validate the recipe, not one case)

Add CLEAN, mostly-deterministic hydrostatic cases (decisive, non-chaotic wins) plus
the canonical instabilities. Each gets a real Oceananigans reference + a fidelity bar.

Clean / deterministic (do FIRST — they isolate single nodes):
- `solid_body_rotation` — Coriolis + free surface, balanced/steady.
- `stommel_gyre` — linear steady Stommel gyre (cleaner than barotropic_gyre).
- `coriolis` / `implicit_free_surface` — operator/solver validation.
- `internal_tide` — internal-wave dynamics (deterministic).
- `advection` / `periodic_advection` / `convergence_tests` — analytic tracer advection + order.

Canonical instabilities (statistical bars, chaotic — same caveat as bickley):
- `baroclinic_adjustment`, `spherical_baroclinic_instability`, `mesoscale_turbulence`.
- `barotropic_gyre` (done, PASS), `bickley_jet` (done, PASS), `silvestri §5` (capstone).

Out of scope: nonhydrostatic / LES / Lagrangian-particle / lid-driven-cavity cases.

## 7. Acceptance

- Per node: single-step tendency match within tolerance vs the Oceananigans reference.
- Per case: the fidelity bar (pattern for deterministic; statistical enstrophy/spectra
  for chaotic). Truth tiers (conservation/equivariance/analytic) outrank tendency-match.
- The `oceananigans` recipe selects the matched canonical options; production defaults
  unchanged (each option default OFF / bit-identical).
- §5 at α=0 survives as long as the oracle, max|u| within 2× (the capstone).
