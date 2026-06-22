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
