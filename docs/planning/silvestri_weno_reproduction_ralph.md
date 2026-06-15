# Ralph-loop task: reproduce Silvestri et al. 2024 (WENO vector-invariant momentum advection)

## OBJECTIVE
Reproduce **Silvestri, Wagner, Campin, Constantinou, Hill, Souza, Ferrari (2024)**, *"A New
WENO-Based Momentum Advection Scheme for Simulations of Ocean Mesoscale Turbulence,"* JAMES
16(7), e2023MS004130 (PDF: `docs/references/Silvestri_etal_2024_WENO_vector_invariant_Oceananigans.pdf`)
inside legoESM, to the **FULL paper matrix** (user directive 2026-06-15):

1. **Build experiment setups that EXACTLY match theirs**, with the **same metrics** — both test
   cases: (A) 2D decaying homogeneous turbulence, (B) baroclinic jet in a periodic channel.
2. **Build a model recipe that matches theirs** — the W9V scheme (9th-order WENO vector-invariant
   with `{ζ;u}` decoupled smoothness + divergence flux).
3. **Systematically compare the available + appropriate legoESM recipes** against each other and
   against the paper's results, the way the paper does (Tables 1–3, Figs 3–10).
4. Reference code: clone/fetch from GitHub **as needed**
   (`github.com/simone-silvestri/BaroclinicAdjustment.jl`, `github.com/CliMA/Oceananigans.jl`).

This is a large multi-session effort. Compute is NOT a constraint (user accepts re-launching
GPU runs). Work the PHASES below in order; **commit each phase**; append to the PROGRESS LOG
every iteration. Use the **iterate-with-codex** review loop after each major code change
(CLAUDE.md mandate).

---

## MAJOR FINDING (scoped 2026-06-15) — the W9V recipe is ~80% already built
The lat-lon C-grid ocean dycore ALREADY implements the paper's core scheme (likely Pierre's
NEMO-track work). Verify before rebuilding; EXTEND, don't duplicate.
- `momentum_advection="weno5"`/`"weno7"` = **WENO-Z vector-invariant** with the paper's
  **decoupled `{ζ;u}` smoothness** (Eq. 43): `_weno_zeta_at_u/v` in
  `packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:586-721`, calling
  `weno_reconstruct_split(phi,psi,order)` in `packages/core/legoesm/core/weno.py:419-476`.
- **Divergence-flux D-term** (Eqs. 31–32, Appendix-C asymmetric WENO/centered split):
  `_bc_dterm`, `ocean_pe_latlon_cgrid.py:1620-1658`, gated by `config.weno_d_term` (default True).
- **WENO9 kernels exist** (`weno.py:389` `weno9_z`) but are NOT wired into momentum →
  the flagship **W9V is a wiring job**, not a from-scratch build.
- Viscosities present: `A_h`, `B_h`, `C_smag` (biharmonic), `C_smag_lap` (Laplacian), `C_leith`.
  **MISSING: QG-Leith, OM4p25-exact Smagorinsky preset, UP3 flux-form upwind momentum.**
- Tracer WENO5/7 present (`advection.py`); 7th-order = paper's tracer choice. WENO9 tracer not exposed.
- KE-gradient: `centered` / `hollingsworth` (NOT WENO). Paper W9V uses WENO5 K (states order of
  K has "minimal impact") → Hollingsworth is an acceptable approximation; WENO-K is a refinement.

### Scheme → legoESM config mapping (the 7 repo cases; paper's main 5 in **bold**)
| Paper | Repo case | momentum_advection | smoothness | closure | legoESM build |
|---|---|---|---|---|---|
| **W9V** | weno9pV | WENO order9 (vort+div+vert) | `{ζ;u}` (CrossAndSelf) | none | wire weno9 + `weno_smoothness="split"` |
| **W9D** | weno9pAllD | WENO order9 | `{ζ;ζ}` (OnlySelf) | none | wire weno9 + `weno_smoothness="standard"` |
| **UP3** | upwind | UpwindBiased order3 flux-form | — | none | NEW UP3 flux-form momentum option |
| **SM2** | omp25 | VectorInvariant EnergyConserving | — | OM4p25 Smag | NEW preset: max(static,smag) Lap+bihar, F=1/(1+0.25 Rh⁴), C₂=.15 Cu₂=.01 C₄=.06 Cu₄=.01 |
| **QG2** | qgleith | VectorInvariant EnergyConserving | — | QG-Leith C=2 | NEW QG-Leith operator: ν=(CΔ/π)³√(∂Q²+∂δ²) |
| extra | bileith | EnergyConserving | — | Biharmonic Leith C=2 | use existing `C_leith` |
| extra | ebs | EnergyConserving | — | energy backscatter | use existing backscatter if present, else skip |

"EnergyConserving vorticity" base = our `momentum_advection="vector_invariant"` (AL81 energy-
enstrophy-conserving PV flux) — paper says "2nd-order energy-conserving rotational form". OK as base.

---

## TEST CASE A — 2D decaying homogeneous turbulence (paper §4)
**Physics:** 2D incompressible Navier–Stokes, NON-rotating (f=0), non-dimensional,
`Re = 3.3e4`, doubly-periodic box `2π × 2π`. RK3 + pressure projection (FFT).
Eq (46): ∂ₜu + ζ k×u = −∇(p+K) + (1/Re)∇²u.
**IC** (Ishiko 2009 narrow-band spectrum, Eqs 48–50): `E(k)=½ a_s k_p⁻¹ (k/k_p)⁷ exp[−7/2 (k/k_p)²]`,
`a_s=16/3`, `k_p=12`. Vorticity in Fourier space `ζ̂=[k/π·E(k)]^½ e^{iφ}` (random phases, real field);
`û=i k_y/k² ζ̂`, `v̂=−i k_x/k² ζ̂`.
**Run:** `t=6` nondim (~18 eddy turnovers; `T_e=(∫k²E dk)^½≈0.33`). Snapshots at `t=3.6` (11 turnovers).
**Grids:** DNS benchmark `4096²` (2nd-order energy-conserving, Eq 17). Coarse: `64², 128², 256², 1024²`.
**Schemes (Table 1):** DNS, Leith1 (C=1), Leith2 (C=2), W5D `{ζ;ζ}`, W9D `{ζ;ζ}`, W5V `{ζ;u}`, W9V `{ζ;u}`.
**Metrics (Figs 3–5):** vorticity field at t=3.6; integrated KE(t) & enstrophy(t); KE spectrum &
enstrophy spectrum at t=3.6 (isotropic). Win = WENO converges to DNS at coarser res; `{ζ;u}`>`{ζ;ζ}`;
W9V captures DNS energy spectrum down to 64².

**legoESM realization (avoid a parallel system — use canonical blocks):** run the ocean C-grid
model in a **single-layer, non-rotating (f=0), flat-bottom, doubly-periodic** config so the
momentum operator reduces to 2D NS, with explicit Laplacian `A_h=1/Re` (non-dim) for DNS/Leith
and the model's WENO momentum for W*. The free surface at small dt enforces ≈non-divergence
(projection analog). If free-surface/barotropic coupling contaminates the pure-2D comparison,
isolate the canonical **vorticity-flux kernel** (`_weno_zeta_at_u/v`) via a thin 2D test harness
(lives in the fidelity/test harness, NOT the model — CLAUDE.md oracle-fidelity rule). Decide
empirically; document the choice.

---

## TEST CASE B — baroclinic jet (paper §5; BaroclinicAdjustment.jl confirms)
**Domain:** periodic channel, spherical sector **60°S–40°S** (φ₀=−50°, Δφ=20°), **20° wide in lon**
(−10→10), **1 km deep**, **Nz=50** (dz=20 m), halo (7,7,7).
**Stratification/front:** `N²=4e-6 s⁻²`; `b=N²z + Δb·B(x)`, `Δb=5e-3 m/s²` (≈2.5°C);
`B = 0 if γ<0; [γ−sinγcosγ]/π if 0≤γ≤π; 1 if γ>π`, `γ=(π/2)−2π(φ−φ₀)/Δφ` (Eqs 52–53).
Thermal-wind-balanced U (vanishes at z=−H), + weak white noise.
**BCs:** no-flux, free-slip walls. Background `ν=1e-4 m²/s`, `κ=1e-5 m²/s` (vertical).
**Restoring:** linearly restore **zonal-mean buoyancy AND velocity** to initial profiles,
**τ=50 days** (sets the transport, doesn't suppress eddies). (Soufflet 2016 style.)
**Time stepping:** AB2 (QuasiAdamsBashforth2, Δt=5 min) [or RK3 Δt=15 min];
SplitExplicitFreeSurface (cfl=0.7). **stop_time = 1000 days**; statistically steady ~day 250.
**Resolutions:** 1/8°, 1/16°, 1/32° → max meridional Δ ≈ 14, 7, 3.5 km (`Ny=20/res`). Tracer = WENO7 (all cases).
**L_d:** initial 5.5 km → equilibrium ~6.75 km (`L_d=(1/(π|f|))∫₀^{−H}(∂_z b)^½ dz`, Eq 54).
**Schemes:** the 5 (UP3, W9V, W9D, SM2, QG2) × 3 resolutions = **15 runs of 1000 days**. (+ bileith/ebs optional.)
**Metrics (Figs 7–10):** surface vorticity (day ~220–230); 10-day-running TKE / EKE / eddy-APE(t);
zonal **energy / enstrophy / w′b′ cospectra** (top 200 m, days 250–1000, vs L_d line); zonal-mean
buoyancy (days 250–1000) = "effective resolution". Win = W9V most energetic (lowest implicit
dissipation), converges ~half the resolution of others (W9V@7km ≈ others@3.5km), no grid-scale ringing.

---

## BUILDING BLOCKS (code) — checklist
- [ ] **B1. Wire WENO9 momentum**: `_weno_zeta_at_u/v` add `order=9` (hw={5:3,7:4,9:5}); dispatch
  `_weno_order={"weno5":5,"weno7":7,"weno9":9}`; config `momentum_advection="weno9"`; D-term + vertical
  inherit order. Unit test the order-9 reconstruction.
- [ ] **B2. `weno_smoothness` config** = `"split"` (`{ζ;u}`, =W*V, default) vs `"standard"` (`{ζ;ζ}`, =W*D).
  Thread into `_weno_zeta_at_u/v`. Unit test both branches differ.
- [ ] **B3. UP3 flux-form momentum**: 3rd-order upwind-biased flux-form momentum advection option
  (`momentum_advection="upwind3"`). Reuse DST3/upwind kernels if present; else add. Unit test.
- [ ] **B4. OM4p25 Smagorinsky preset (SM2)**: assemble from A_h/B_h/C_smag/C_smag_lap the exact
  `ν₂=max(C₂Δ²|D|,Cu₂Δ)·F`, `ν₄=max(C₄Δ⁴|D|,Cu₄Δ³)`, `F=1/(1+0.25 Rh⁴)` (Rh=L_d/Δ),
  `|D|=√(Ds²+Dt²)`, Ds=∂ₓu−∂ᵧv, Dt=∂ₓv+∂ᵧu, C₂=.15 Cu₂=.01 C₄=.06 Cu₄=.01. As a `*Config`/preset,
  NOT magic numbers. Unit test. (Ref: `BaroclinicAdjustment.jl/src/Parameterizations/omp25_lateral_friction.jl`.)
- [ ] **B5. QG-Leith viscosity (QG2)**: NEW operator `ν=(CΔ/π)³√(∂Q²+∂δ²)`, ∂Q² bounded by the
  three q-gradient terms (Eq A2–A3), reverts to 2D Leith where QG breaks; C=2. As a scheme + `*Config`.
  Unit test + physics contract. (Ref: `.../qg_leith_viscosity.jl`.)
- [ ] **B6. WENO-K KE gradient (refinement, optional)**: `ke_gradient_scheme="weno"` for full W9V
  faithfulness. Low priority (paper: minimal impact).
- [ ] **B7. WENO9 tracer (optional)**: expose `weno9_to_u/v_points` + vertical. Paper uses WENO7
  tracer for ALL cases, so weno7 (exists) suffices; weno9 tracer not required.

## DIAGNOSTICS — checklist (reuse existing where noted)
- [ ] **D1. Relative vorticity** field diagnostic (C-grid curl `curl_vertex_cgrid` exists; wrap).
- [ ] **D2. KE(t) & enstrophy(t)** integrals (2D + 3D).
- [ ] **D3. Eddy decomposition** util `x' = x − ⟨x⟩_zonal`; **EKE, TKE, eddy-APE** time series
  (eddy-APE = ½ g² ⟨b'²⟩/(N²ρ₀²) or ½⟨b'²⟩/N² form — pick + document).
- [ ] **D4. w′b′** vertical eddy buoyancy flux + its zonal cospectrum.
- [ ] **D5. Zonal energy / enstrophy spectra** (reuse `run_eady_rebuilt.py::_zonal_spectrum` pattern +
  `ocean/diagnostics.py::isotropic_energy_spectrum/isotropic_enstrophy_spectrum`).
- [ ] **D6. Zonal-mean buoyancy** (days 250–1000 average).
- [ ] **D7. Deformation radius** L_d — `ocean/diagnostics.py::first_baroclinic_deformation_radius` EXISTS.

## RECIPES / EXPERIMENTS — checklist
- [ ] **R1. 2D turbulence experiment** `ocean/experiments/decaying_turbulence_2d.py`: config + Ishiko
  spectral IC + doubly-periodic non-rotating single-layer grid. `*Recipe` NamedTuple + `build_*` fn. Test.
- [ ] **R2. Silvestri baroclinic-jet experiment** — adapt `eady_uniform.py` OR new
  `ocean/experiments/baroclinic_jet_silvestri.py` with the EXACT §5 config (front Eqs 52–53,
  τ=50d zonal-mean restoring of b AND u/v, 1000 d, 1/8–1/32°). `*Recipe` + `build_*` selecting the
  5 scheme presets. Test.
- [ ] **R3. Driver A** `scripts/run/run_silvestri_2d_turbulence.py`: runs a scheme×res, emits
  parseable VERDICT + metrics (KE/enstrophy/spectra), saves arrays for plots.
- [ ] **R4. Driver B** `scripts/run/run_silvestri_baroclinic_jet.py`: runs a scheme×res 1000 d,
  emits VERDICT (EKE_sat, L_d, gridscale_frac, ringing flag) + saves time series + spectra + zonal-mean b.
- [ ] **R5. Comparison/plot** `scripts/plot/plot_silvestri_comparison.py`: reproduces Figs 4–5 (2D)
  and 8–10 (jet) analogues from saved arrays.

---

## PHASES (loop roadmap — do in order, commit each)
- **Phase 1 — building blocks B1–B5** (+ B6/B7 if cheap): the scheme code. Codex review. Unit tests green.
- **Phase 2 — diagnostics D1–D7**: with unit tests.
- **Phase 3 — recipes/experiments R1–R2 + drivers R3–R4**: forward steps finite, scan-carry stable, tests.
- **Phase 4 — 2D turbulence matrix**: run DNS(4096²) + {64,128,256,1024}² × {Leith1,Leith2,W5D,W9D,W5V,W9V}.
  Collect KE(t)/enstrophy(t)/spectra. Reproduce Figs 4–5. Verdict: does W9V converge fastest, `{ζ;u}`>`{ζ;ζ}`?
- **Phase 5 — baroclinic-jet matrix**: run {1/8,1/16,1/32°} × {UP3,W9V,W9D,SM2,QG2} × 1000 d.
  Collect TKE/EKE/eddy-APE + spectra + zonal-mean b + L_d. Reproduce Figs 7–10. Verdict: W9V most
  energetic / highest effective resolution?
- **Phase 6 — synthesis**: comparison report `docs/ocean_experiments/silvestri_weno_reproduction.md`
  (scoreboard vs paper, where legoESM matches/diverges, effective-resolution table). Final commit.

## SUCCESS CRITERIA
1. The 5 paper schemes are SELECTABLE legoESM recipes (config knobs / presets), each unit-tested.
2. Both experiments reproduce the paper's QUALITATIVE result: (A) WENO noise-free vs Leith grid-noise;
   W9V converges to DNS at coarsest res; `{ζ;u}`>`{ζ;ζ}`. (B) W9V most energetic, ≈half-resolution
   convergence, no ringing; dispersive (SM2/QG2) show ringing at 3.5/7 km.
3. Metrics MATCH the paper's metrics (same definitions: integrated KE/enstrophy, isotropic spectra,
   EKE/TKE/eddy-APE, w′b′ cospectrum, zonal-mean b, L_d).
4. Each numerical-code change passes the iterate-with-codex adversarial review (CLAUDE.md mandate)
   + the guardrail harness (contracts/ratchets) where touched.
5. Report documents every match AND divergence honestly (truth tiers outrank oracle-matching).

## REFERENCE CODE (fetch from GitHub as needed)
- `github.com/simone-silvestri/BaroclinicAdjustment.jl`:
  `src/baroclinic_adjustment.jl` (setup), `run_and_visualize/run_and_postprocess.jl` (the 7 TestCases),
  `src/Parameterizations/{omp25_lateral_friction,qg_leith_viscosity,biharmonic_leith_viscosity,
  leith_laplacian_viscosity,energy_backscatter}.jl`, `src/Diagnostics/{spectra,integrated_diagnostics,
  diagnostic_fields}.jl`, `run_and_visualize/{deformation_radius,energy_plots,spectra_plot,buoyancy_contour}.jl`.
- `github.com/CliMA/Oceananigans.jl` (≥v0.84.0): `src/Advection/vector_invariant_*.jl`,
  `src/Advection/weno_*.jl` (WENOVectorInvariant, CrossAndSelfUpwinding vs OnlySelfUpwinding = V vs D).
- Use `api.github.com/repos/<repo>/git/trees/main?recursive=1` to list, `raw.githubusercontent.com/...` to fetch.

## ENV CAVEATS
- Shell is **csh**; venv python `.venv/bin/python`; science runs need `JAX_ENABLE_X64=1`.
- 2× V100S; **pin GPUs** with `CUDA_VISIBLE_DEVICES`; GPU runs intermittently SIGTERM/SIGURG-killed
  (exit 143/144) — re-run killed configs, prefer harness-tracked background tasks, poll log files.
- Branch: `feat/silvestri-weno-reproduction` (off the Eady rebuild). Stage explicit pathspecs, never `git add -A`.
- File layout: experiments→`ocean/experiments/`, drivers→`scripts/run/`, plots→`scripts/plot/`,
  diagnostics→`ocean/diagnostics.py`, tests mirror under `tests/ocean/`. No files at repo root.

---

## PROGRESS LOG (append every iteration — newest on top)

### Iteration 4 — B3 DONE (UP3 flux-form momentum) — 2026-06-15
- **B3 COMPLETE + adversarially reviewed + committed** (`feat(ocean): UP3 3rd-order
  upwind-biased flux-form momentum advection`). `momentum_flux_scheme="upwind3"` (paper UP3 =
  Oceananigans `UpwindBiased(order=3)`, NEMO/ROMS κ=1/3). New module-level `_up3_reconstruct`
  (4-pt upwind-biased face value), wired into all 4 flux terms of the flux-form path:
  zonal-u/zonal-v (periodic-lon rolls) + meridional-u/meridional-v (edge-padded Neumann,
  damped by Qy→0 at poles). Flux telescoping preserved → momentum conserved.
- **Adversarial review (skeptical subagent): no blockers.** Numerically verified the highest-
  risk part — the edge-padded meridional stencils — element-by-element for small n_lat (no
  off-by-one), UP3 cubic-convergence ratio-8 (3rd order), conservation residual 1.9e-18 (holds
  even with a corrupted wrap column), AD/JIT static + finite grads. Applied the one NIT: made
  zonal-u `far_neg` wrap-robust (build from the distinct core via roll, not the wrap-inclusive
  u[:,1:]) to match the routine's existing wrap-robust style.
- Tests: 12 flux-form (UP3 conservation u+v, uniform-flow-zero, UP3≠upwind; `_up3_reconstruct`
  constant/linear-exact + upwind-bias) + 61 dispatch/eady green.
- **Recipe-scheme coverage so far:** W9V (B1+B2 split), W9D (B2 standard), UP3 (B3). Remaining
  comparison schemes: SM2 (B4 = OM4p25 Smagorinsky preset), QG2 (B5 = QG-Leith).
- **NEXT: B4 = OM4p25 Smagorinsky lateral-friction preset (SM2).** The pieces likely exist
  (A_h/B_h/C_smag/C_smag_lap operators) — assemble the exact OM4p25 combo: ν₂=max(C₂Δ²|D|,Cu₂Δ)·F,
  ν₄=max(C₄Δ⁴|D|,Cu₄Δ³), F=1/(1+0.25·Rh⁴) with Rh=L_d/Δ, |D|=√(Ds²+Dt²), Ds=∂ₓu−∂ᵧv,
  Dt=∂ₓv+∂ᵧu, C₂=.15 Cu₂=.01 C₄=.06 Cu₄=.01. As a config preset/NamedTuple, NOT magic numbers.
  Read the existing smagorinsky_laplacian/biharmonic operators in latlon_cgrid_operators.py +
  the `lateral_viscosity_operator` dispatch FIRST. Ref:
  BaroclinicAdjustment.jl/src/Parameterizations/omp25_lateral_friction.jl.

### Iteration 3 — B2 DONE (weno_smoothness split/standard) — 2026-06-15
- **B2 COMPLETE + adversarially reviewed + committed** (`feat(ocean): weno_smoothness split
  (W*V) vs standard (W*D)`). `config.weno_smoothness ∈ {"split","standard"}` now selects the
  V vs D scheme family for BOTH the vorticity flux AND the divergence flux:
  - split (W*V, default): vorticity `{ζ;u}` (Eq 43) + divergence `{δU;D}` (Eq 45, full-divergence
    smoothness); standard (W*D): `{ζ;ζ}` (Eq 37) + `{δU;δU}` (Eq 44).
  - `_weno_zeta_at_u/v` standard branch = `weno_reconstruct_split(phi, phi)` (review confirmed
    bit-exact to the plain weno{5,7,9}_z kernel). `_bc_dterm` split passes full divergence
    D=δU+δV as the matching-direction smoothness psi. Validated at construction.
- **DEFAULT BEHAVIOR CHANGE (intended, faithful):** the prior default was a HYBRID — `{ζ;u}`
  vorticity (V) but `{δU;δU}` divergence (D). New default "split" = consistent W*V
  (`{ζ;u}`+`{δU;D}`), the paper's recommended scheme ("the divergence choice has a large impact").
  Review confirmed: NO committed recipe/golden/fidelity test selects WENO momentum (the only
  fidelity recipe uses flux_form), so no validated result is silently altered.
- **Adversarial review (skeptical subagent — codex still unavailable): CLEAN, no blockers.**
  Verified: standard==plain-WENO (Δ=0 all orders), split divergence = Eq 45, cross-direction
  stays centered (Appendix C holds), threading complete (K + vertical C correctly left as
  self-smoothness per Eqs 33/41), validation raises on unknown, AD/JIT static. One NIT applied
  (commented that u_smooth/v_smooth are inert in the standard branch).
- Tests: 7 new smoothness tests (split≠standard on sharp ζ for Z and D; constant-exact both;
  config accept/reject; full-step W9V≠W9D with injected sharp feature — and verified bit-identical
  on the SMOOTH Eady IC, which is correct: WENO reconstructs linear fields identically regardless
  of smoothness). 106 weno+dispatch + 25 eady/no-dup green.
- **NEXT: B3 = UP3 flux-form 3rd-order upwind momentum.** Scoping: `momentum_advection="flux_form"`
  exists (`_bc_horizontal_momentum_advection_flux_form`) with `momentum_flux_scheme` ∈
  {"upwind","centered"} — but "upwind" is 1st-order. Need a 3rd-order upwind-biased flux-form
  option (NEMO/ROMS UP3, Madec 2022 / Shchepetkin-McWilliams). Check whether the existing
  flux-form path's reconstruction order is configurable or if a `momentum_flux_scheme="upwind3"`
  (DST3-style) reconstruction must be added. Read `_bc_horizontal_momentum_advection_flux_form`
  first. (Paper UP3 = `UpwindBiased(order=3)`.)

### Iteration 2 — B1 DONE (WENO9 momentum wired) — 2026-06-15
- **B1 COMPLETE + committed** (`feat(ocean): wire WENO9 vector-invariant momentum advection`).
  `momentum_advection="weno9"` now selectable. Verified the existing weno5/7 path FIRST
  (it already does `{ζ;u}` decoupled smoothness + D-term, as scoped). Extended:
  - Z (vorticity) + D (divergence) → order 9; C (vertical) capped at 5 (paper Table 2,
    "minimal impact" + vertical kernel only has 5/7); K unchanged.
  - **NEW order-8 (7-point) `point_to_cellavg` conversion**, coeffs solved from cell-avg
    Taylor moments + verified exact to degree 7. This was the real find: the code already
    *intended* conv_order=8 for weno7 but `point_to_cellavg` silently capped it at 6 (the
    `else` branch). Implementing it (a) fulfills that intent and (b) unblocks the **W9V-vs-W5V
    effective-resolution distinction the paper's headline result depends on** — surfaced by a
    failing accuracy-ordering test (WENO9 was no better than WENO5 with order-6 conversion).
  - Tests: 123 WENO (core+momentum+tracer) + 61 dispatch/validate/recipe green; no weno5/7
    regression. Added: order-8 conversion exactness, order-8>order-6 on smooth field, WENO9
    vorticity shapes/constant/cell-to-face/resolved-accuracy, full ocean step weno9 finite.
- **ADVERSARIAL REVIEW of B1 DONE** (codex CLI not installed in this env → independent
  skeptical-subagent review instead; note for future iterations: `which codex` = not found,
  so the literal `/codex:adversarial-review` + codex-driven physics-validator can't run here;
  use a skeptical-reviewer subagent). Verdict: **no BLOCKERs**. Independently re-derived the
  order-8 coeffs (sympy-exact; convergence rate 7.94→7.98), confirmed order threading
  (Z=9,D=9,C=5 cap correct), dispatch coverage complete (no missed `("weno5","weno7")` site),
  AD/JIT staticness OK, ghost-cell ranges in-bounds on small grids. **One clarification:**
  weno7's D-term changes **5→7** with B1 (was hardcoded 5 for both weno5/weno7; now follows the
  momentum order). This is CORRECT + paper-faithful (W7V pairs D=7 with Z=7) and FIXES a prior
  Z=7/D=5 inconsistency — the B1 commit message wrongly called it "unchanged (D=5/7)". No weno7
  momentum golden/regression baseline exists (only tracer-weno7 + kernel tests, all green), so
  the change is safe. Recorded here since the commit msg is misleading.
- **NEXT: B2 = `weno_smoothness` config `"split"` (W*V) vs `"standard"` (W*D).** IMPORTANT
  finding while reading the code — the smoothness choice affects BOTH the vorticity AND the
  divergence flux, and the current default is a MIX (not a pure paper scheme):
  - **Vorticity Z:** split → `{ζ;u}` (Eq 43, velocity smoothness — CURRENT default, correct for V);
    standard → `{ζ;ζ}` (Eq 37, self-smoothness). `_weno_zeta_at_u/v` currently always passes
    `u_smooth`/`v_smooth` (split). For W9D, reconstruct with `psi=phi` (self). NOTE: the
    `u_smooth=None` fallback is `{ζ;v}` (still velocity!), NOT `{ζ;ζ}` — standard needs psi=phi.
  - **Divergence D:** the paper distinguishes `{D}` (Eq 44, W9D: `{δ_iU; δ_iU}` self) from
    `{D;D}` (Eq 45, W9V: `{δ_iU; D}` smoothness = the FULL divergence `D=δ_iU+δ_jV`). Paper:
    "This difference has a large impact on the solution." The current `_bc_dterm` passes
    `_weno_cell_to_uface(dU_di_filled, dU_di_filled, ...)` = `{δU; δU}` = the **W9D `{D}`**.
    So the CURRENT divergence is the D-variant even though the CURRENT vorticity is the V-variant
    → the shipped default is a hybrid {ζ;u}+{D}, neither pure W9V nor pure W9D.
  - **B2 work:** add `config.weno_smoothness ∈ {"split","standard"}` (default "split"); thread into
    `_weno_zeta_at_u/v` (split→velocity psi, standard→phi psi) AND `_bc_dterm` (split→pass full
    divergence `D=dU_di+dV_dj` as psi for the matching reconstruction `{δU; D}`; standard→`{δU; δU}`).
    `{u}_k` (vertical) and `{δu²}` (K) are self-smoothness in both per Eqs 37/41 — leave as-is.
    Unit-test that split≠standard for both Z and D, and that W9V (split) vs W9D (standard) build.
  - In Oceananigans: `CrossAndSelfUpwinding` (V) vs `OnlySelfUpwinding` (D).

### Iteration 1 (setup) — 2026-06-15
- Read the full paper (24 pp); extracted both experiment specs + all metrics + appendices
  (Leith, QG-Leith, OM4p25 closures; energy-conservation derivation).
- Scoped the repo (2 Explore agents): **W9V ~80% built** — WENO5/7 vector-invariant with `{ζ;u}`
  smoothness + D-term ALREADY exist; weno9 kernels exist unwired; QG-Leith/UP3/OM4p25-preset missing;
  2D-turbulence experiment + several diagnostics missing. Mapping table above.
- Fetched reference specs from BaroclinicAdjustment.jl (the 7 TestCases, exact §5 config, OM4p25 +
  QG-Leith formulas). PDF saved to `docs/references/`.
- Wrote this spec. Branch `feat/silvestri-weno-reproduction` created off the Eady rebuild.
- **NEXT: Phase 1 / B1 — wire WENO9 momentum** (`_weno_zeta_at_u/v` order=9 + dispatch + config),
  with a unit test, then B2 (`weno_smoothness` split vs standard) which W9D needs. Verify the
  existing weno5/7 path first (read `ocean_pe_latlon_cgrid.py:586-721,1599-1658`) before extending.
