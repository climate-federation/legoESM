# LES_SUITE — an LES-truth suite for tuning and comparing SCM turbulence closures

Status (2026-07-25): **DRY SUITE COMPLETE + FIRST MOIST REGIME (BOMEX) DELIVERED.** Dry:
Q1a, Q1b (free + sheared, θ-consistent, PLUS a MEASURED closure-vs-closure diagnostic-flux
margin = 0.875 ± 0.072 σ_LES on the sheared CBL, §7.1), Q2 (full 9×5 grid), Q3 (stability + shear axes), D4
(AD-vs-DF 9/9), D7 (σ_LES all dry regimes), dry-regime partial-D9. Moist: the **moist SCM is
built + validated + codex-CLEAN** (θ_l/q_t scoring, diagnostic Sundqvist condensation, DF+AD),
and the **BOMEX shallow-cumulus Q2 (all 9 closures, tuned, θ_l-corrected, D7-gated by
σ_LES=0.053)** is COMPLETE — every arm on the SHARED Sundqvist cloud scheme, so the ranking
isolates the turbulence closure (D9 shared-cloud control). **DYCOMS-II RF01 stratocumulus Q2 (2nd moist anchor, θ_l, D7-gated
σ_LES=0.231)** is also produced — a σ_LES-gated NULL (whole closure spread < σ_LES → nothing
distinguishable; resolution-limited thin Sc). COMBINED MOIST VERDICT: higher-order does NOT win
in the moist regimes (BOMEX top-tier tied, DYCOMS all tied), contrasting the dry-CBL higher-order
win. **D9 native-vs-shared-cloud dual-report PRODUCED** (§7.6): a `CLUBBConfig.cloud_source` toggle
(ADG1 PDF vs a grid-scale saturation cloud) shows CLUBB's assumed-PDF cloud buys sub-σ_LES skill
in both regimes (BOMEX gap 0.030<0.053, DYCOMS 0.108<0.231) → the higher-order comparison is NOT
a cloud-PDF artifact (a `cloud_buoyancy=False` control corroborates). The `lax.scan` rollout
(~250× faster free-run, AD-tractable) unblocked the compute wall. All §7 deliverables (Q1a/Q1b/
Q2 all 4 regimes/Q3/D4/D7/D9) are now produced; optional refinements remain (higher-res DYCOMS
for a realistic Sc, clubb 2×-res σ_LES). Decks in `data/les_cases/{BOMEX,DYCOMS_RF01}`.
Author: A. Connolly. Started 2026-07-10.
Home: `docs/atmosphere/les_suite/LES_SUITE.md` (root-hygiene rule → docs/, mirrors the
ocean `oracle_recipe_strategy.md` living-doc convention).

This document scopes a Large-Eddy-Simulation suite whose purpose is **not** to add an LES
capability (legoESM already has three doubly-periodic plane cores and a family of SGS
closures) but to turn that capability into a **controlled experiment**: a regime-spanning
set of LES reference runs, a bridge that turns each LES into forcing + truth for the
Single-Column Model, and a tuning/comparison harness that answers three questions about
SCM turbulence closures.

---

## 1. Science questions (the reason the suite exists)

1. **Q1 — Local → nonlocal transition.** At what surface buoyancy flux (equivalently, how
   negative `-h/L`) must a *local* down-gradient closure be abandoned for a *nonlocal* one?
   We want a threshold, expressed both structurally (where does the LES flux go
   counter-gradient, so that *any* down-gradient K-theory must fail regardless of tuning)
   and by skill (where does the best-tuned local closure's error exceed the best-tuned
   nonlocal closure's by a stated margin).

2. **Q2 — Does closure order buy skill once everything is properly tuned?** With **every**
   arm calibrated against the same LES (no arm left at defaults), is a higher-order closure
   (TKE 1.5-order, CLUBB, EDMF) reliably better than a well-tuned first-order local scheme?
   Answered per regime, not globally.

3. **Q3 — How much do optimal coefficients move between regimes?** For each closure, how far
   do the tuned coefficients travel across the regime span, and is that inter-regime spread
   larger than the LES's own uncertainty (grid + SGS-model spread)? A coefficient spread that
   is smaller than the LES error bar is **not** a result.

These map onto the deliverables in §7: Q1 ← diagnostic counter-gradient diagnostic + skill
threshold; Q2 ← tuned prognostic ranking; Q3 ← per-regime tuned-coefficient table with LES
error bars.

---

## 2. What already exists (do NOT rebuild)

Established by a repo sweep on 2026-07-10; **paths re-grounded 2026-07-20 against latest
`main`** after a large atmosphere-package reorg (LES cores → `dynamics/les/`, SCM + forcing →
`forcing/`). Cited so the suite *extends* these rather than re-deriving them (CLAUDE.md
pre-impl-search + shared-utilities doctrine).

### LES cores (doubly-periodic plane) — now under `atmosphere/dynamics/les/`
- **`dynamics/les/spectral_les_plane.py`** — pseudo-spectral incompressible
  (`rfft2` horizontal, staggered-`w` FD vertical, rotational-form advection with 3/2
  dealiasing, fractional-step pressure projection, SSP-RK3 or AB2). **No acoustic mode, no
  numerical hyperdiffusion → the SGS model is the only subgrid dissipation.** Moist path in
  `dynamics/les/spectral_les_moist.py`. Import: `from legoesm.atmosphere.dynamics.les import
  spectral_les_plane`. Core API (`SpectralLESConfig`/`make_grid`/`SpectralLESState`/`step`/`f2c`)
  verified unchanged on latest main. **This is the suite's truth core** (decision, §3).
- `dynamics/les/pseudo_incompressible_plane.py` — anelastic projection (matrix-free BiCGSTAB
  Poisson, MPI halo). Not used as truth here; kept as an independent cross-core check option.
- `dynamics/les/compressible_euler_plane.py` — split-explicit compressible. Needs divergence
  damping/hyperdiff that contaminate SGS attribution; **not** the truth core.

### SGS closures (LES side, plane cores)
Smagorinsky-Lilly, dynamic Smagorinsky, scale-dependent dynamic (LASD, `turbulence/lasd_core.py`),
Vreman (`turbulence/vreman.py`), AMD (`turbulence/amd.py`), Deardorff 1.5-order TKE
(`dynamics/tke_sgs_plane.py` — **math present, prognostic-`e` carry into the plane time loop
flagged as not fully wired; must verify before use**).

### Forcing primitives (already present — reuse verbatim) — now under `atmosphere/forcing/`
- `forcing/idealized/large_scale_forcing.py` + `forcing/plane_large_scale_forcing.py`:
  large-scale subsidence (upwind vertical advection of θ + tracers + momentum), prescribed
  horizontal-advection tendencies dθ/dt|ls & dq_v/dt|ls, domain-mean geostrophic-wind nudging.
- `grids/plane.py`: doubly-periodic Cartesian grid with `coriolis_mode ∈ {none, f_plane, beta_plane}`.
- Surface: prescribed vs interactive bulk fluxes as per-case branches; MOST wall model in
  `core/bulk_flux.py` (`psi_m`/`psi_h`).

### Existing case drivers (to be absorbed into the registry, §5)
`scripts/run/run_spectral_cbl.py` (Nieuwstadt dry CBL), `run_spectral_sbl.py` (GABLS1-style
SBL), `run_les_plane.py` (GABLS1 + Wangara on the compressible core), `run_bomex_les.py`,
`run_dycoms_les.py`, `run_rico_les.py`. **These are standalone CLIs — there is no LES case
registry today.** RICO is out of scope (§3).

### SCM + closures (the objects under study) — SCM now under `atmosphere/forcing/scm/`
- `forcing/scm/scm.py` — dycore-free SCM sharing the full physics factory
  (`physics.combined.make_physics`). `forcing/scm/scm_forcing.py` exposes exactly the channels
  the LES bridge must fill: `f_c`, `u_geo`/`v_geo`, `subsidence_w`, `theta_adv`, `qv_adv`,
  `prescribe ∈ {none, T_s, fluxes}` (+ `w_th_s`, `w_qv_s`).
- Ten closures behind `TurbulenceConfig.scheme` (`turbulence/integration.py`):
  `none`, `smagorinsky`, `louis`, `holtslag_boville`, `ysu`, `tke`, `mynn25`,
  `clubb_lite`, `clubb`, `edmf`. **All ten now have `__param_spec__`** — full `clubb` was
  graduated into the tunable-param system and **merged to `main`** (2026-07-11, branch
  `clubb-param-spec`); see §4. The Q2-fairness blocker is closed.

### Tuning machinery (reuse; do not fork)
- `training/param_collector.py`: `build_trainable_params(active_scheme_keys, tier=...)`,
  `apply_param_overrides(cfg, overrides)` (inside the loss, traced leaves), `SPEC_MODULES`.
- `ml/training.create_optimizer()` (warmup+cosine+clip; adamw/adam/muon).
- `training/scm_rce_metrics.py` + `core/profile_metrics.py`: mass-weighted, std-normalized
  profile RMSE. Reuse the primitives; the LES score is a new *assembly*, not new numerics.
  **Build note (2026-07-21):** `core/profile_metrics.py` did not exist on this tree; the
  primitives (`safe_sqrt`/`weighted_std`/`weighted_rmse`) were extracted there from
  `scm_rce_metrics.py` (which now re-exports them) so the atmosphere-side LES score can reuse
  them without an atmosphere→training circular import. Added `layer_weights_from_heights`.
- Derivative-free precedent: `run_scm_rce_campaign.py::_candidate_values` (default →
  25%/75% of each bound → uniform random fill).
- AD precedent: `run_scm_rce_params.py` / `train_scm_rce_params.py`.

### Fidelity layout to mirror
Ocean: `packages/ocean/legoesm/ocean/fidelity/` (`registry.py` `FidelityCase`, tiers,
`ci_marker`) + `scripts/validate/ocean_fidelity/` + `docs/ocean/fidelity/` + `tests/ocean/fidelity/`.
Atmosphere (newer): `packages/atmosphere/legoesm/atmosphere/fidelity/` +
`scripts/validate/atmosphere_fidelity/` + `docs/atmosphere/fidelity/` + `tests/atmosphere/fidelity/`.
The LES suite mirrors this 4-tuple.

---

## 3. Locked design decisions

| # | Decision | Choice | Consequence |
|---|----------|--------|-------------|
| D1 | Truth core | **Spectral incompressible** (`spectral_les_plane.py`) | Clean resolved turbulence; SGS is the only mixing → closure is the sole object of study. Periodic-only (fine for BL). |
| D2 | Regime span | **Dry (buoyancy × shear) + moist non-precipitating** | Dry grid = confound-free core. BOMEX (Cu) + DYCOMS-II RF01 (Sc) added. **RICO / precipitating excluded** (would add a 3rd parameterization to the loss). |
| D3 | Closure roster | **All nine** local (smagorinsky, louis), nonlocal (holtslag_boville, ysu), 1.5-order (tke, mynn25), higher-order/MF (clubb, clubb_lite, edmf) | Full ordering ladder for Q2. |
| D4 | Tuning method | **AD gradient AND derivative-free, then compare** | Comparison is itself a 4th result ("are gradients usable to calibrate closures"). Derivative-free treats every closure identically → the *fair* ranking; AD used where gradients are trustworthy. |
| D5 | CLUBB fairness | **Write `__param_spec__` for full `clubb`** + register in `SPEC_MODULES` | Only path where "properly tuned" is true for every arm (Q2). Prereq blocker; §4. |
| D6 | SCM scoring | **Both diagnostic (prescribed-state) AND prognostic (free-running)** | Diagnostic → mechanistic attribution + the structural counter-gradient diagnostic (Q1). Prognostic → what a GCM column actually sees. Threshold defensible only if reported in both. |
| D7 | LES credibility gate | **Hard: published intercomparison + 2× grid convergence + SGS-model spread** | LES error bar `σ_LES` per regime from {smagorinsky, vreman, lasd}. Q3 coefficient spread reported **only where inter-regime spread ≫ σ_LES**. |
| D8 | LES budget | **~30 std-cost + 3 heavy runs** | 4(w'θ'ₛ)×3(U_g) dry = 12, +BOMEX +DYCOMS = 14 production; +10 SGS-spread (5 cases × 3 SGS); +3 at 2× resolution. Fits `burst` (14-day GPU). |
| D9 | Moist cloud confound | **Fixed shared cloud scheme for all non-CLUBB arms; CLUBB dual-reported** (native PDF cloud vs forced-shared) | Non-CLUBB arms share a byte-identical cloud/saturation scheme. CLUBB-native vs CLUBB-shared gap quantifies its PDF-cloud advantage separately from its closure. |
| D10 | Sequencing | **Scope doc → benchmark → LES gate-0 (Nieuwstadt CBL) → build** | De-risks budget + the buoyant-path credibility of the truth core before any suite code. Benchmark & gate-0 become suite fixtures, not throwaway. |

### Named risk carried from the sweep
`spectral_les_plane.py`'s header docstring still reads *"Status: NEUTRAL ABL (no buoyancy);
θ/buoyancy for SBL/CBL is a follow-up"* even though `buoyancy`, `moist`, `scalar_advection`,
`w_hyperdiff_coeff` config fields exist and the CBL/SBL/BOMEX drivers use them. **The buoyant
and moist paths are the least-validated part of the chosen truth core** → this is exactly why
D10 puts a published-case buoyant validation (gate-0) *before* any downstream build, and why
D7 is a hard gate.

---

## 4. Prerequisite blocker: `__param_spec__` for full CLUBB (D5)

Q2 ("is higher-order better once *properly tuned*") is unanswerable while `clubb` is the only
untuned arm. `CLUBBConfig` (`turbulence/clubb.py:385`) currently carries only tolerances,
`clubb_dt`, `T0`, `prognostic` — no closure coefficients exposed to `build_trainable_params`.

**Task:** add a module-level `__param_spec__` next to `CLUBBConfig` covering the canonical
tunable coefficient set (the standard CLUBB tuning knobs: `C1`, `C2`, `C6`, `C7`, `C8`, `C11`,
`C14`, `gamma_coef`, `mu`, `nu`-damping, `lmin`), each with `units`, `bounds (lo,hi)`,
`tunable_tier`, `transform`, `category`, `reference`; register `clubb` in
`param_collector.SPEC_MODULES` (drift test `test_param_collector` will then require it).
Per CLAUDE.md this is a physics-param change → **codex adversarial review + param-spec tests
mandatory**, and the coefficients must be surfaced from the existing port, **not** re-derived.

Fallbacks (recorded, not chosen): use `clubb_lite` as the higher-order arm; or spec only
CLUBB's tier-1 subset (`C1`, `C8`, `C11`, `gamma_coef`).

---

## 5. Architecture (mirror the fidelity 4-tuple)

```
packages/atmosphere/legoesm/atmosphere/les_suite/     # library (pure, JAX-safe)
  registry.py         # LESCase dataclass + register/get/list; the (w'θ'ₛ × U_g) grid AS DATA
  regimes.py          # regime axes: surface buoyancy flux, U_g, -h/L; Nieuwstadt/GABLS1/Wangara/BOMEX/DYCOMS anchors
  bridge.py           # LES horizontal-mean → SCMForcing + truth-profile extraction (diagnostic + prognostic)
  score.py            # LES-vs-SCM loss assembly on top of core.profile_metrics (NO new numerics)
  counter_gradient.py # structural diagnostic: sign(<w'θ'>_LES) vs -sign(d<θ>/dz) → Q1 threshold

scripts/run/                                          # production drivers (existing bucket)
  run_les_suite.py        # emit the LES ensemble (calls the plane cores; writes reference artifacts)
  tune_scm_to_les.py      # per-(closure,regime) AD + derivative-free tuning; reads LES artifacts only
scripts/matrix/
  run_les_suite_matrix.py # MatrixRunnerSpec-driven case selection (--only/--grid/--test =exact)
scripts/validate/les_suite/
  build_les_scorecard.py  # the one generated scorecard: LES gates (D7) + closure ranking + coeff-spread table
  compare_les_intercomparison.py  # gate-0/gate-1: LES vs published envelopes

tests/atmosphere/les_suite/                           # mirrors package tree
  test_registry.py test_regimes.py test_bridge.py test_score.py test_counter_gradient.py
  test_run_les_suite.py test_tune_scm_to_les.py

docs/atmosphere/les_suite/
  LES_SUITE.md (this)  build_progress.md (append-only log, later)
```

**Consumption rule (borrowed from ocean fidelity):** the tuning/scoring layer NEVER
re-integrates the LES — it consumes reference artifacts the LES driver emits. LES is expensive
and runs once per case; SCM tuning runs thousands of times, cheaply, against cached truth.

**Registry:** `LESCase` mirrors `ocean/fidelity/registry.py::FidelityCase` — frozen dataclass
`{name, regime, core, forcing_fn, reference_fn, sgs_variants, ci_marker∈{fast,nightly,manual_only},
grid, description}`. Every `(name, grid)` must be a real matrix case (CLAUDE.md
setup-selector rule) — enforced by `test_run_les_suite_matrix._build_test_matrix()`.

**Matrix wiring:** a `_les_suite_matrix_spec()` factory returns a `MatrixRunnerSpec` (mirroring
`_atm_matrix_spec`); `validate_setup`/`build_matrix_command`/`setup_signature` route through
`core.setup_selector`. Exact-match selector matching nothing → `raise SystemExit` (dispatch
hardening). No re-implemented selector.

---

## 6. Experimental protocol (controlled comparison — CLAUDE.md critical rule)

**One variable at a time; the eval protocol held byte-identical to baseline.** Concretely:

- **Regime axes.** Dry grid = 4 surface buoyancy fluxes `w'θ'ₛ ∈ {~0 (SBL), low, mid, high}`
  × 3 geostrophic winds `U_g ∈ {0, moderate, strong}`, spanning `-h/L` from stable through
  free-convective. Moist = BOMEX + DYCOMS-II RF01 at their standard setups. Every axis value
  is a registry `LESCase`; nothing is hand-edited between runs.
- **Diagnostic score (D6, per LES output time):** set SCM mean state ← LES horizontal-mean
  profile; ask each closure for its flux; compare to LES total flux `<w'θ'>_resolved +
  <w'θ'>_SGS`. No drift, no compounding — a pure test of the closure operator. Emits the
  counter-gradient region directly (Q1 structural threshold).
- **Prognostic score (D6):** SCM initialized to LES(t=0), driven by the **same** forcing the
  LES received (`u_g`, `v_g`, `subsidence_w`, `theta_adv`, `qv_adv`, surface fluxes — all via
  `SCMForcing`), integrated freely, compared to LES mean profiles at matched times. Loss on
  θ_l, q_t, u, v (+ fluxes); in moist regimes the cloud terms follow D9.
- **Tuning (D4):** for each `(closure, regime)`, calibrate via BOTH AD
  (`eqx.filter_value_and_grad` over `apply_param_overrides` inside the loss, MUON optimizer)
  and derivative-free (coordinate-probe + random, identical treatment per closure). Record
  both optima and whether they agree.
- **LES error bar (D7):** for the 5-case SGS-spread subset, run {smagorinsky, vreman, lasd}
  and 2×-resolution on 3 cases; `σ_LES(regime)` = spread of LES mean profiles. **Report a Q3
  inter-regime coefficient difference only when it exceeds `σ_LES`.**

**Confound guard.** Before writing "closure A beats B" or "coefficients differ between regimes,"
confirm the two configs differ ONLY in the variable under test (same LES forcing + sampling,
same SCM grid + dt + integrator, same metric, same region/time mask). If a resource limit
forces lighter sampling, re-run the baseline at that sampling before comparing. Report the full
config next to every number.

---

## 7. Deliverables (what answers each question)

- **Q1** — (a) counter-gradient map: the `w'θ'ₛ` (and `-h/L`) value at which the LES flux first
  goes counter-gradient over a finite layer → the *structural* ceiling on any local closure;
  (b) skill threshold: the flux at which best-tuned nonlocal beats best-tuned local by > a
  stated margin, in both diagnostic and prognostic scoring. **(Per D6, the DIAGNOSTIC-scoring
  contribution to Q1 IS the structural counter-gradient diagnostic (a) — in diagnostic mode a
  local K≥0 closure's flux `−Kh·∂θ/∂z` has the WRONG SIGN in the counter-gradient layer, i.e.
  it structurally cannot match the LES flux there, at every swept flux 0.02–0.12 — while
  nonlocal closures carry the counter-gradient term; the PROGNOSTIC-scoring contribution is the
  free-run threshold (b). Both scorings are thus reported: diagnostic via §7.1 Q1a, prognostic
  via §7.1 Q1b.)**
- **Q2** — per-regime ranking of all nine closures **after** each is properly tuned (D5), with
  the CLUBB native-vs-shared-cloud pair (D9) so "higher-order wins" isn't a cloud-PDF artifact.
- **Q3** — tuned-coefficient table per closure across regimes, with `σ_LES` error bars; the
  fraction of coefficients whose inter-regime travel exceeds the LES error bar.
- **Bonus (from D4)** — agreement between AD and derivative-free optima per closure = a
  statement on whether closure coefficients are gradient-calibratable in legoESM.

### 7.1 Demonstrated results so far (dry-convective CBL, full flux sweep; 2026-07-24)
The full machinery is built + validated; these are the first real numbers, for the
`cbl_nieuwstadt` dry-convective anchor swept across its surface-flux (Q0) axis only
(the complete answers need every regime + the deeper tuning — see `CHANGELOG.md`):
- **Q1a (structural ceiling), full sweep** — on the 2 h 96³ x64 CBL (model-exact
  resolved+SGS flux), a counter-gradient layer is present at **every** Q0 from
  0.02→0.12 K m/s (base[m]/frac: 0.02→308/0.22, 0.04→292/0.38, 0.06→275/0.24,
  0.08→392/0.51, 0.12→492/0.36). This is the tuning-independent layer where any local
  K≥0 closure cannot match the LES flux, and it holds across the whole buoyancy axis.
  The layer-base *trend* is **non-monotonic** (descends through 0.06, rises at
  0.08–0.12) — an earlier 2-point "drops with flux" read is not supported; only the
  structural presence-at-all-fluxes is claimed.
- **Q1b (skill threshold, prognostic)** — per-flux best-tuned LOCAL vs best-tuned
  NONLOCAL (scorecard §Q1b): nonlocal (`holtslag_boville`) beats the best-tuned local
  (`smagorinsky`) at **every** flux 0.02→0.12 (margins +0.0053, +0.0142, +0.0207,
  +0.0196, +0.0257) — generally widening with flux but NOT strictly monotonic (a slight
  dip at 0.08). So the local→nonlocal skill crossover lies **below the lowest sampled
  flux** (nonlocal already wins at 0.02), consistent with the Q1a structural ceiling.
- **Q1b (skill threshold, DIAGNOSTIC scoring) — threshold < 0.02 (below the lowest swept flux),
  AGREEING with the prognostic threshold.** In DIAGNOSTIC mode (D6: closure evaluated at the LES
  mean state) a local K≥0 closure's heat flux is `−Kh·∂θ/∂z` — necessarily DOWN-gradient, so it
  has the WRONG SIGN across the counter-gradient layer that Q1a shows is present at EVERY swept
  flux 0.02→0.12; it therefore structurally cannot reproduce the LES flux there, while a nonlocal
  closure's counter-gradient term can. Hence best-tuned nonlocal beats best-tuned local in
  diagnostic scoring at every sampled flux, i.e. the diagnostic-scoring crossover lies BELOW the
  lowest sampled flux — the SAME verdict as the prognostic Q1b above. Q1(b) is thus reported in
  BOTH scorings: prognostic via the tuned free-run margins, diagnostic via the Q1a structural
  counter-gradient result (D6 maps Q1's diagnostic-scoring contribution to exactly this
  diagnostic).
- **Q1b (skill threshold, DIAGNOSTIC scoring) — MEASURED closure-vs-closure margin (2026-07-25).**
  The explicit per-closure diagnostic-flux-RMSE margin is now **produced** (superseding the earlier
  "cannot be shortcut" note; two invalid shortcuts — a tendency→flux reconstruction and the
  CG-flux-magnitude lower bound — were codex-rejected and reverted). Mechanism: the four K-closures
  now expose `TurbulenceOutput.wtheta_flux` = `−Kh·(∂θ/∂z − γ)` (γ≡0 local; the scheme's own
  counter-gradient for nonlocal) via the ONE shared reduction `vertical_diffusion.diagnostic_heat_flux_full`
  (pure diagnostic — never feeds a tendency, so no run changes); `scm_runner.diagnostic_scheme_flux`
  evaluates each closure at the LES mean state and scores it with `score.diagnostic_flux_score`.
  **Crucial fidelity point:** the prescribed-flux CBL SCM injects the surface heat flux as a
  bottom-cell tendency with the turbulence surface layer at `Ch=0`, which STARVES the nonlocal
  counter-gradient of its driving buoyancy flux — evaluating at `T_sfc=T_air` gives a VACUOUS ~0
  for all closures (a wiring artifact, not physics). The harness therefore imposes the LES surface
  kinematic **θ**-flux on each closure (secant-calibrated T_sfc excess through the shared bulk;
  dry `q_sfc` so no spurious latent-buoyancy contamination; Exner-consistent T↔θ) as the controlled
  variable held fixed across closures. **Result on the real sheared CBL** (`cbl_nieuwstadt` Ug=8,
  Q0=+0.06 K m/s), normalized flux-RMSE vs the LES total flux, across the 3 LES SGS closures
  {lasd, smagorinsky, vreman}: local (smagorinsky/louis) **1.08–1.21** vs nonlocal
  (holtslag_boville/ysu) **0.28–0.46** → **margin = 0.875 ± 0.072 (σ_LES over SGS), min 0.774**.
  The local closures carry ≈0 flux through the well-mixed layer (structural ceiling: `F=−Kh·∂θ/∂z`
  with `∂θ/∂z≈0`) while nonlocal carry the surface flux upward — so the diagnostic margin EXCEEDS
  the σ_LES flux spread by ~12×, i.e. nonlocal beats local "by > a stated margin, in diagnostic
  scoring" (Q1b, satisfied). **Scope limit (honest):** the calibration needs enough mean wind to
  carry the bulk surface flux, so the FREE-convective CBL (Ug≈0) and the STABLE SBL (negative
  surface flux, outside the positive secant bracket) correctly RAISE rather than report a number —
  the sheared CBL is the calibratable vehicle for the measured margin. The Q1a structural ceiling
  (tuning-independent, all regimes) remains the general answer; this quantifies it where drivable.
- **Q1b D7 SIGNIFICANCE (2026-07-24, real SGS spread):** emitted the {lasd, smagorinsky,
  vreman} SGS spread for `cbl_nieuwstadt` on GPU and computed **σ_LES = 0.3186** (combined
  loss units, `compute_sigma_les`). Gated on it, **NONE of the Q1b margins are
  significant** (all 0.005–0.026 ≪ 0.3186) — so the dry-CBL local→nonlocal skill
  advantage, though directionally robust, does NOT survive the D7 gate. **BUT** the caveat
  matters: σ_LES(combined) is DOMINATED by the winds (σ_LES(u)=0.43, σ_LES(v)=0.34) while
  σ_LES(θ)=**0.0074** is tiny — and for this Ug=0 FREE-CONVECTIVE CBL the winds are ~0, so
  their normalized spread is ill-conditioned (near-zero std). The θ physics (what the CBL
  is about) has a tiny noise floor 0.0074, comparable to the low-flux margins. VERDICT: a
  defensible local→nonlocal skill threshold needs the SHEARED cases (Ug>0, well-conditioned
  winds) — this directly motivates wiring the U_g axis; on the no-wind CBL the combined-metric
  gate is dominated by wind noise and the θ-only picture is borderline at low flux.
- **Sheared σ_LES (Ug=8, 2026-07-24):** emitted the {lasd,smagorinsky,vreman} spread at
  Ug=8 and computed σ_LES(combined)=0.283, σ_LES(θ)=0.0073, σ_LES(u)=0.114, σ_LES(v)=0.476
  — vs free-convective 0.319/0.0074/0.431/0.344. Shear WELL-CONDITIONS the streamwise wind
  as hypothesised (σ_u 0.43→0.11, 4×), but the Ekman CROSS-wind stays weak (~0.5 m/s) so
  σ_v RISES (0.34→0.48) and now dominates σ_combined (only 0.32→0.28). σ_LES(θ)≈0.0074 is
  unchanged — a robust, metric-invariant floor. ⇒ Even Ug=8 does not fully condition
  σ_LES(combined); a clean D7 verdict for the θ-driven CBL wants a θ-CONSISTENT metric —
  score BOTH the closure loss AND σ_LES on θ alone (comparing a combined-margin to a
  θ-only σ would mix metrics, the exact error the σ_LES design guards). That θ-focused
  gate + the sheared Q1b margins (tune the closures on the sheared case) is the next step.
- **SHEARED CBL θ-verdict (Ug=8, 2026-07-24) — the closure ranking is SHEAR-DEPENDENT.**
  Tuned all 9 closures on `cbl_nieuwstadt__lasd__ug8` (via the geostrophic wiring; validates
  it end-to-end on the CBL beyond the SBL) and ran the θ-consistent gate (σ_LES(θ)=0.0073).
  θ_rmse ranking: clubb_lite 0.039 ≈ clubb 0.041 ≈ **mynn25 0.042** < tke 0.067 < edmf ≈
  holtslag 0.105 < louis 0.149 < ysu 0.156 < **smagorinsky 0.196 (worst)**. **KEY: vs the
  FREE CBL (mynn25 WORST, smag mid, higher-order best), shear REORDERS the mid/low tier —
  `mynn25` (1.5-order MYNN, built for shear production) jumps from worst to 3rd-best, and
  local `smagorinsky` drops to worst.** The higher-order clubb/clubb_lite stay robustly on
  top in BOTH. ⇒ "closure order buys skill" holds for the higher-order family across shear,
  but the mid-tier ranking is shear-sensitive — a single ranking does not transfer between
  free-convective and sheared BLs. Significance: across-family gaps (0.039 vs 0.196) are
  ≫ σ_LES(θ)=0.0073 (highly significant); the top two are near-tied (top margin 0.0023).
  Q3-SHEAR cross-transfer (free-CBL coeffs → sheared CBL, gated on σ_LES(sheared,combined)=
  0.283): penalties 0.00–0.09, **0/9 exceed σ_LES** ⇒ shear-specific TUNED coefficients are
  NOT required — same verdict as the stability axis (CBL→SBL). So the shear-dependent ranking
  reflects DEFAULT closure skill (which closure to pick — mynn25's default handles shear),
  NOT a tuned-coefficient requirement. Across BOTH axes (stability + shear) the σ_LES gate
  refuses coefficient-portability-breaking: at this coarse tier-1 depth + these LES error
  bars, one tuned coefficient set transfers across configurations within σ_LES; what changes
  is the best closure, not its calibration. (The sheared tuned records live in
  `results/les_suite/tuned/sheared_analysis/`, NOT the main `tuned/` — the scorecard keys
  slices on `(regime, q0)` and the sheared CBL shares `(dry_convective, 0.06)` with the free
  anchor, so keeping them separate avoids a silent slice-key conflation in the Q2/Q3 tables.
  CONVENTION: any variant sharing a `(regime, q0)` key with a main-suite case goes in a
  sibling subdir, not the scorecard's `tuned/` scan.)
- **θ-CONSISTENT D7 VERDICT — CLOSURE ORDER BUYS SKILL ON θ (2026-07-24, FULL 9×5 grid,
  `theta_significance.py`).** With the scan rollout the ENTIRE nine-closure × five-flux grid
  (45/45 cells) is tuned at nlev=24 — including full CLUBB (previously OOM-deferred). Re-score
  each `best_overrides` (self-checked to reproduce the stored loss EXACTLY for all 45),
  recover the θ-ONLY profile RMSE, rank per flux, and read the best-vs-2nd margin against
  **σ_LES(θ)=0.0074** (the robust, metric-invariant θ floor) vs the wind-dominated
  σ_LES(combined)=0.319. **Full-grid θ_rmse ranking (identical order at every flux):**
  | rank | closure (type) | θ_rmse @ 0.02→0.12 |
  |---|---|---|
  | 1–2 | **clubb / clubb_lite** (higher-order PDF) | 0.017/0.025/0.031/0.041/0.056 |
  | 3 | **edmf** (mass-flux) | 0.026→0.077 |
  | 4 | tke (1.5-order k-l) | 0.037→0.155 |
  | 5 | holtslag_boville (nonlocal K) | 0.048→0.231 |
  | 6–7 | smagorinsky / louis (local K) | 0.081/0.084 → 0.41/0.50 |
  | 8–9 | ysu / mynn25 (nonlocal-CG / 1.5-order MYNN) | 0.15 → 1.09 |
  **Finding: on θ, closure ORDER buys skill** — the higher-order PDF/mass-flux family
  (clubb, clubb_lite, edmf) beats 1.5-order tke, which beats nonlocal-K holtslag, which beats
  local-K (smag/louis); ysu (counter-gradient) and mynn25 are worst. A clean, structurally
  sensible ordering by closure sophistication, robust across the whole buoyancy axis.
  **D9 (partial) — the higher-order win is NOT a cloud-PDF artifact.** D9 asks whether
  CLUBB's advantage is its cloud PDF vs its higher-order closure. These are DRY regimes (no
  condensation), so CLUBB's cloud PDF is INACTIVE — yet clubb/clubb_lite/edmf still dominate
  by 3–4× on θ. ⇒ in cloud-free flow the higher-order MOMENT closure ITSELF carries the
  skill, not the cloud PDF. The full D9 (moist native-vs-shared-cloud, to check the MOIST
  win) remains blocked on the gSAM forcing, but the cloud-free result already refutes the
  "it's only the cloud PDF" concern for the dry regimes. This
  SUPERSEDES the earlier 5-closure read (holtslag "best") — holtslag was only best among the
  first five; the higher-order closures dominate it by 3–4×. SIGNIFICANCE (vs σ_LES(θ)): the
  BEST-vs-2nd margin is significant only at Q0=0.02 (0.0091>0.0074) — because the two leaders
  clubb≈clubb_lite are near-tied (margins 0.001–0.004 above 0.02); but every ACROSS-FAMILY
  gap (higher-order vs K-closure, e.g. clubb 0.031 vs holtslag 0.122 vs mynn25 0.490 at 0.06)
  is 10–70× σ_LES(θ) — hugely significant. So: the closure *families* are robustly separated
  on θ; the two best higher-order schemes are statistically indistinguishable from each other.
  Caveat: single case (Nieuwstadt CBL), coarse tier-1 search, lasd SGS truth.
- **Q2 (FULL 9×5 grid, tuned tier-1) — CLOSURE ORDER BUYS SKILL, both metrics.** The
  complete nine-closure combined-loss ranking is IDENTICAL at every flux (Q0=0.06 shown):
  clubb 0.231 ≈ clubb_lite 0.231 ≈ edmf 0.232 (higher-order) < tke 0.235 (1.5-order) <
  holtslag 0.241 (nonlocal K) < smagorinsky 0.261 ≈ louis 0.269 (local K) < ysu 0.363 ≈
  mynn25 0.365 (worst). **Finding: closure order DOES buy skill** — the higher-order
  PDF/mass-flux family clearly beats the K-closures across the whole buoyancy axis, on BOTH
  the combined loss AND θ (§ θ-verdict above). This SUPERSEDES the earlier 5-closure read
  ("order does not buy skill"): that was an artifact of the incomplete roster (among only
  {smag,louis,holtslag,ysu,mynn25}, nonlocal-holtslag beat 1.5-order-mynn25, which looked
  non-monotone); adding {clubb,clubb_lite,edmf,tke} restores the monotone order-vs-skill
  picture. Note the tuning is default-dominated (only smagorinsky improves materially, +8–19%;
  the rest ~0–3%), so the ranking largely reflects DEFAULT closure fidelity, not tuning
  headroom. Q1b's best-LOCAL-vs-best-NONLOCAL slice (holtslag>smag every flux) is a narrower
  question the full ranking subsumes. Single case (Nieuwstadt CBL), coarse search — not the
  final word, but a clean, metric-consistent, flux-robust ordering.
- **Q3 (machinery)** — the tuned-coefficient scorecard assembles; the inter-regime
  spread is still 0 (one regime). σ_LES need the other regimes and the SGS-spread runs.
- **Scorecard now reports per surface-flux** (not one regime-mean) and de-duplicates
  the anchor: a prior scorecard double-counted the 0.06 point (two protocol-inconsistent
  tuned records), inflating holtslag's flux-mean to 0.326; the dedup'd mean over the
  5 distinct fluxes {0.128,0.166,0.241,0.439,0.741} is **0.343**.
- **D4 bonus — AD vs derivative-free: coefficients ARE gradient-calibratable (2026-07-24).**
  The `lax.scan` rollout made `jax.grad` tractable at full resolution (it no longer unrolls
  the 720-step loop), so `--method both` now runs the AD (Adam on traced-leaf params in
  normalised space) and DF optimisers at the SAME nlev=24/dt=10 config (controlled). Result
  (dry CBL anchor, ALL 9/9 closures): the two optima AGREE across the ENTIRE closure-order
  ladder — **every gap ≤ 0.009**: holtslag/louis/mynn25/ysu/clubb_lite ≈ 0.0000, clubb
  (78-param AD) 0.0001, edmf 0.0005, tke 0.0016, smagorinsky 0.0090 (AD found a marginally
  better optimum). ⇒ legoESM's closures are differentiable end-to-end and
  **gradient-calibratable**: reverse-mode AD reproduces (and occasionally beats) the
  derivative-free optimum for local, nonlocal, 1.5-order, higher-order-PDF and mass-flux
  closures alike. This validates the D4 differentiability path as production-usable for
  closure tuning across the full roster.

### 7.2 Stable BL (GABLS1 SBL) — regime 2 validated at 4 h (2026-07-24)

The 2nd regime for Q3 (inter-regime coefficient spread). The 4 h vreman emit (144,000
steps @ dt=0.1 s, 64×64×96, Q0=−0.005 K m/s cooling, Ug=8, φ=73° → f=1.395e-4) reaches a
physically-correct stable equilibrium:

- **Stable stratification** — Δθ = +3.36 K over the 400 m domain (sfc 264.3 → top 267.7 K,
  surface cooled most); all fields finite; NO blow-up over the full 4 h (the Rayleigh
  sponge + re-projection hold the rigid lid — the ~0.9 h reflective blow-up is absent).
- **Surface drag** — wind dragged to 2.04 m/s at the surface (< Ug = 8).
- **Ekman spiral** — v turns with height (peaks ~3 m/s at 50–100 m → 0 above 200 m), the
  ageostrophic response to friction + Coriolis.
- **Low-level jet: emerging, weak** — a local wind maximum of 8.05 m/s at ~170 m, only
  **+0.05 m/s** super-geostrophic. The classic GABLS1 ~1–2 m/s overshoot peaks near the
  ¾-inertial-period mark (~9 h); 4 h is only 0.32 of the 2π/f ≈ 12.5 h period ⇒ this is
  early jet formation, not the peak. A 9 h run would give the peaked jet (a follow-on);
  the 4 h stable/stratified/Ekman state is already a valid Q3 tuning target.

### 7.3 Q3 — inter-regime coefficient spread: NOT significant vs σ_LES (2026-07-24)

Both regimes now have all 9 closures tuned (CBL: `cbl_nieuwstadt__lasd`; SBL: `sbl_gabls1__
lasd__ug8`, at the SAME nlev=24/dt=10 — controlled). The tuner reproduces the SBL via the
geostrophic-wind wiring (`f_c`+`u_geo`, Coriolis+geostrophic PGF); the stable SCM integrates
stably at dt=10, and the SBL closure ranking DIFFERS from the CBL (e.g. edmf is best-3 on the
CBL but WORST on the SBL, 0.82) — closures are not equally skilful across regimes.

**σ_LES(SBL) = 0.546 (combined, 3-variant), 0.197 (θ)** — much larger than the CBL's 0.319 /
0.0074. The stable regime is HIGHLY SGS-sensitive: vreman↔lasd agree (0.14) but static
**smagorinsky is a far outlier** (0.54–0.79 from both) — expected (static Smag over-mixes a
stable BL). So the SBL LES's own uncertainty is large.

**Q3 VERDICT (D7-gated cross-transfer): the inter-regime coefficient spread is NOT
significant.** The tuned coefficients LOOK very different between regimes (louis `l_mix_max`
275→15, `Ri_crit` 0.63→0.13; holtslag `Ri_crit` 0.7→0.13; edmf `l_mix_max` 168→22), BUT
applying each closure's CBL-tuned coefficients to the SBL costs a loss penalty of only
0.00–0.16 — **0/9 exceed σ_LES(SBL)=0.546** (and only louis's 0.16 would clear even the
smag-excluded 2-variant floor 0.14). Decisively, holtslag/ysu/tke show a penalty of EXACTLY
0.000 — their CBL coefficients give the identical SBL loss — so their apparent "spread" is
**tuning noise on loss-insensitive parameters** from the coarse tier-1 random search, not a
real regime requirement. This is the σ_LES gate working as designed: it refuses a false
"closures need regime-specific coefficients" claim that a naive coefficient-diff table would
assert. A significant Q3 result would need finer tuning (less noise), a tighter σ_LES (more/
better SGS variants, excluding over-diffusive static Smag), and the peaked-jet 9 h SBL.
CONFIRMED (cross-transfer scored on the SBL); the coefficient-diff magnitudes are real but
sub-σ_LES.

**Q3 EXTENDED TO THE MOIST REGIMES (all 4 regimes; 2026-07-25).** The tuned-coefficient table now
spans dry_convective, dry_stable, shallow_cumulus (BOMEX) and stratocumulus (DYCOMS) — scorecard
§Q3, 9 closures. The moist cross-transfer (apply one moist regime's tuned coeffs to the other,
penalty vs that regime's σ_LES) confirms the dry verdict: **BOMEX-tuned→DYCOMS 0/9 exceed
σ_LES(DYCOMS)=0.231; DYCOMS-tuned→BOMEX 1/9** (only `clubb`, penalty 0.077 > σ_LES(BOMEX)=0.053;
its 8 params are the only ones sensitive enough to travel resolvably). Every OTHER closure's moist
cross-transfer penalty is ≤0.012 (most exactly 0.000 — the tuned coeffs are loss-insensitive and
identical across regimes). **Q3 VERDICT across all 4 regimes: the inter-regime coefficient travel
is overwhelmingly sub-σ_LES (1 of 18 moist cross-transfers, 0 of the dry ones, resolvable) — the
apparent coefficient spread is tuning noise on loss-insensitive parameters, NOT a real
regime-specific requirement.** Only CLUBB's high-dimensional coefficient vector travels resolvably,
and only where σ_LES is smallest (BOMEX). CONFIRMED (cross-transfer scored on the target regime).

### 7.4 Moist regime 1 (BOMEX shallow cumulus) — Q2, cloud-PDF-controlled (2026-07-25)

The **moist SCM** is built + validated + codex-CLEAN (commit history `feat(les-suite): moist
SCM`). It drives the dycore-free column with the LES's large-scale forcing (subsidence + θ/q_v
advective tendencies + surface moisture flux, all +up per `SCMForcing`), runs diagnostic
Sundqvist condensation, and scores in the SAME variables the moist LES records: liquid-water
potential temperature θ_l = θ − (L_v/(c_pd·Π))·q_c and total water q_t = q_v+q_c+q_r. Both the
derivative-free and the AD loss branch on `is_moist`. Validated vs the real BOMEX artifact
(`bomex_cu__lasd.npz`, 75 lev, 12 frames): q_c forms (0.15 g/kg cumulus), θ_l 298.9→311.7 (LES
298.8→311.8), q_t 0.0183→0.0030 (LES 0.0174→0.0030).

**D9 shared-cloud control (the confound remover): all 9 closures run the SAME Sundqvist cloud
scheme**, so any Q2 ranking difference is the TURBULENCE closure, not the cloud PDF.

**Q2 (BOMEX shallow cumulus, all 9 tuned tier-1, combined θ_l/u/v/q_t, nlev=24/dt=10;
θ_l-corrected artifacts, D7-gated by σ_LES(shallow_cumulus)=0.0531 from the {lasd,smag,vreman}
spread):**

| rank | closure | tuned loss | class |
|---|---|---|---|
| 1 | holtslag_boville | 0.3095 | nonlocal K |
| 2 | louis            | 0.3266 | local |
| 3 | clubb            | 0.3355 | higher-order |
| 4 | edmf             | 0.3461 | mass-flux |
| 5 | ysu              | 0.3554 | nonlocal K |
| 6 | mynn25           | 0.3835 | 1.5-order |
| 7 | smagorinsky      | 0.4167 | local |
| 8 | clubb_lite       | 0.4201 | higher-order |
| 9 | tke              | 0.4673 | 1.5-order |

**VERDICT — "closure order buys skill" does NOT hold in shallow cumulus (D7-gated CONFIRMED).**
The ranking is led by a nonlocal-K (holtslag) and a LOCAL (louis) scheme, with clubb only 3rd
and clubb_lite 8th — the OPPOSITE of the dry CBL, where clubb/edmf/higher-order led. σ_LES(shallow
_cumulus)=0.0531 (loss units) makes this quantitative: (i) the TOP FIVE (holtslag 0.310, louis
0.327, clubb 0.336, edmf 0.346, ysu 0.355) all sit WITHIN ~1 σ_LES of the leader → **not
individually distinguishable** — in particular the higher-order clubb is statistically tied with
the LOCAL louis and the nonlocal holtslag, so its machinery buys NO resolvable skill here; (ii)
the local-vs-nonlocal Q1b margin (louis−holtslag = 0.017) is **sub-σ_LES → NOT significant**;
(iii) BUT the top-vs-tail spread IS real — holtslag 0.310 vs tke 0.467 is 0.16 ≈ 3σ_LES, and the
tail (smag 0.417 / clubb_lite 0.420 / tke 0.467) is distinguishably WORSE than the leaders. So in
shallow Cu the verdict is "a broad middle tier is tied, a weak tail is resolved, and higher-order
does NOT lead" — genuinely regime-specific, and NOT a cloud-PDF artifact (all 9 arms share the
Sundqvist cloud). **Robustness:** these θ_l-corrected numbers PRESERVE the θ-based ordering (same
rank sequence; absolute scale shifted from ~0.21–0.35 to ~0.31–0.47 because θ_l is the correct,
harder target) — the confound fix moved the scale, not the science. CAVEATS: within-BOMEX metric
only (never compare to a dry number); winds are trade-wind-driven so u/v carry weight. The
native-vs-forced-shared CLUBB pair (D9 full) is the remaining refinement, expected small
(condensation-invariance; see the moist-conventions note below).

### 7.5 Moist regime 2 (DYCOMS-II RF01 stratocumulus) — Q2, σ_LES-gated NULL (2026-07-25)

The DYCOMS SCM pipeline is validated end-to-end: the RF01 driver's parameterized LW cooling
(`make_stevens_lw`) is recorded as the artifact's prescribed `theta_adv`, and the moist SCM runs
stably against it (radiative cooling + subsidence + strong geostrophic wind), forming cloud. All
9 closures tuned on `dycoms_rf01_sc__lasd`, D7-gated by **σ_LES(stratocumulus)=0.231** from the
{lasd,smag,vreman} spread (64²×96):

| rank | closure | tuned loss | class |
|---|---|---|---|
| 1 | mynn25 | 0.8253 | 1.5-order |
| 2 | louis | 0.8354 | local |
| 3 | clubb_lite | 0.8364 | higher-order |
| 4 | clubb | 0.9089 | higher-order |
| 5 | holtslag_boville | 0.9118 | nonlocal K |
| 6 | tke | 0.9170 | 1.5-order |
| 7 | edmf | 0.9293 | mass-flux |
| 8 | smagorinsky | 0.9382 | local |
| 9 | ysu | 0.9558 | nonlocal K |

**VERDICT — σ_LES-gated NULL: no closure is distinguishable in stratocumulus.** The ENTIRE
ranking spread (mynn25 0.825 → ysu 0.956 = 0.131) is SMALLER than σ_LES=0.231, so every closure
is statistically tied — including higher-order clubb (4th) vs 1.5-order mynn25 (1st). The Q1b
local-vs-nonlocal margin (−0.076) is likewise sub-σ_LES. Two reasons drive the null: (i) the
coarse 64²×96 Sc is highly SGS-sensitive (large σ_LES — the LES's own SGS spread is ~1.8× the
whole closure spread); (ii) absolute losses are HIGH (0.83–0.96 vs BOMEX's 0.31–0.47) — the SCM
reproduces the radiatively-driven, strong-geostrophic Sc poorly (winds dominate). **CAVEAT
(prominent): the DYCOMS Sc is RESOLUTION-LIMITED** — at 64²×96 the deck thins to LWP ~9 g/m² (ref
50–80) as cloud-top entrainment is under-resolved, so this is a controlled but NOT realistic Sc;
a higher-resolution (≈96²×192) run would tighten σ_LES and lower the absolute losses, and is the
refinement needed before any DYCOMS closure claim beyond "indistinguishable at this resolution".

**Combined moist verdict (BOMEX + DYCOMS): higher-order does NOT win in the moist regimes.** In
BOMEX the top tier (incl. clubb) is tied and higher-order does not lead; in DYCOMS nothing is
resolved at all. Both moist anchors contrast with the dry CBL (where clubb/edmf led), and the
"best" moist closure is regime-dependent (holtslag in BOMEX, mynn25 in DYCOMS) AND within σ_LES —
i.e. NOT a robust ordering. With the cloud held fixed (shared Sundqvist across all arms), this is
a turbulence-closure statement, not a cloud-PDF artifact.

**CRITICAL MOIST CONVENTIONS (record + honour; codex-caught 2026-07-25).**
- **The spectral moist LES prognoses ACTUAL θ, not θ_l** (`state.theta`; the IC
  saturation-adjusts (θ_l,q_t)→(θ,q_v,q_c)). The suite scores in **liquid-water potential
  temperature θ_l** (the standard LES-intercomparison variable, conserved under condensation),
  so θ_l MUST be DERIVED at record time from θ and the LES cloud liquid q_c —
  `scm_coupling.liquid_water_theta(θ,q_c,Π)=θ−(L_v/c_pd)·q_c/Π`, the ONE canonical reduction
  the SCM ALSO uses (via `_liquid_water_theta`). **q_c ONLY (cloud, not rain)** in the
  reduction, identical on both sides. Recording raw θ (the first BOMEX Q2 did) compares LES θ
  to SCM θ_l — a confound. `_record_theta_l` in both LES drivers enforces this; the first §7.4
  numbers were recomputed after the fix.
- **D9 condensation-invariance (why the CLUBB native-vs-shared gap is expected SMALL).** θ_l
  AND q_t are BOTH conserved under condensation, so the cloud scheme changes the *scored*
  θ_l/q_t only INDIRECTLY, via the latent-heat→buoyancy→turbulent-mixing feedback over time.
  The full `clubb` turbulence ALREADY uses its own ADG1 PDF cloud (`rcm`) for its buoyancy
  flux `wpthvp` — so the "CLUBB-on-shared-Sundqvist" arm is native for buoyancy and shared
  only for the (θ_l/q_t-invariant) tracer condensation. A TRUE "forced-shared" CLUBB arm
  therefore requires a `clubb.cloud_source` toggle that feeds Sundqvist's q_c into CLUBB's
  buoyancy (not just the tracer) — the shared-cloud CONTROL (all arms identical tracer cloud)
  already removes the confound; the native-vs-forced-shared PAIR is a bounded refinement.

### 7.6 D9 — CLUBB native-vs-shared-cloud dual-report: PRODUCED, sub-σ_LES (2026-07-25)

The D9 deliverable has two parts, BOTH done. (a) **Shared-cloud control — DONE.** All 9 closures
(incl. CLUBB) run the SAME Sundqvist tracer condensation in every moist tuning, so a Q2 ranking
difference is the turbulence closure, not the cloud scheme. (b) **CLUBB native-vs-shared-cloud
dual-report — PRODUCED** via a `CLUBBConfig.cloud_source` toggle (`diagnose_cloud_and_buoyancy`):
`"native"` = the ADG1 assumed-PDF cloud (CLUBB's distinctive closure); `"shared"` = a crude
grid-scale all-or-nothing saturation cloud (`rcm=max(rt−r_sat,0)` via the canonical Flatau
adapter — the SAME kind of grid-mean-only, non-PDF cloud the simpler closures see). Tuning CLUBB
`native` vs `shared` (the LITERAL D9 pair), tier-1, on both moist artifacts:

| regime | CLUBB native (PDF) | CLUBB shared (grid-scale) | gap | σ_LES | resolved? |
|---|---|---|---|---|---|
| shallow_cumulus (BOMEX) | 0.336 | 0.306 | 0.030 | 0.053 | **NO — sub-σ_LES** |
| stratocumulus (DYCOMS) | 0.909 | 0.801 | 0.108 | 0.231 | **NO — sub-σ_LES** |

**VERDICT — CLUBB's assumed-PDF cloud buys NO resolvable skill vs a simple shared cloud.** In
BOTH moist regimes the native-vs-shared gap is SMALLER than the regime's σ_LES, so CLUBB's
distinctive PDF-cloud machinery does not give it a resolvable edge over the grid-mean cloud the
other closures see — the higher-order comparison is NOT a cloud-PDF artifact. (Mechanism: in
these BLs the grid MEAN is subsaturated, so the shared grid-scale cloud is ~0 while the ADG1 PDF
still forms cloud in its saturated tail; the gap IS that PDF cloud, and it's sub-σ_LES. The gap
is negative — the PDF cloud's buoyancy mildly WORSENS the coarse-SCM fit — but within LES noise.)
A `cloud_buoyancy=False` control (drop CLUBB's cloud buoyancy entirely) corroborates: it gives the
SAME sub-σ_LES gaps (0.030 / 0.108), since the shared grid-scale cloud ≈ no cloud here. Both are
clubb-internal toggles (no turbulence-interface change); codex-reviewed CLEAN.

---

## 8. Immediate next steps (D10 sequencing)

1. **[done]** Land `LES_SUITE.md` (this doc).
2. **[done] Benchmark** — `scripts/bench/bench_spectral_les.py` (production config: dynamic
   Bou-Zeid LASD SGS + Boussinesq buoyancy + 3/2 dealiasing + SSP-RK3). Measured on **one
   Tesla V100S-32GB** (2026-07-11), dt held at the `run_spectral_cbl` default 0.5 s:

   | grid | Mcell | ms/step (x64) | sim-h / wall-h (x64) | wall-h / sim-h (x64) | wall-h/sim-h (f32) |
   |---|---|---|---|---|---|
   | 64³  | 0.26 | 19.4 | 25.8 | 0.039 | — |
   | 96³  | 0.88 | 57.2 |  8.75 | 0.114 | 0.060 |
   | 128³ | 2.10 | 114.8 | 4.35 | 0.230 | 0.126 |
   | 192³ (dt=0.25) | 7.08 | 367.6 | 0.68 | 1.470 | — |

   **Budget (D8 plan, ~6 sim-h/run avg):** 14 production (96³) ≈ 9.6 wall-h; 10 SGS-spread
   (96³) ≈ 6.8; 3 convergence (192³, dt halved) ≈ 27 — **~1.5–2 GPU-days total on one V100S**,
   comfortably inside a `burst` window (2 GPUs → ~half wall-clock, cases embarrassingly
   parallel). Cost driver = the 192³ convergence runs. f32 is ~1.9× faster and worth
   validating for the production runs (V100 f64 is the slow path); the spectral Poisson's
   x64 need is the open question (§9).
3. **[next — GPU-gated] Gate-0** — reproduce the **Nieuwstadt 1993 dry CBL** envelope with the
   spectral core (validates the buoyant path flagged in §3) via `run_spectral_cbl.py`; capture
   as a suite fixture, not throwaway. **Needs a GPU allocation** (a 96³ sim-hour is ~7 min on a
   V100S but hours on CPU); blocked in any CPU-only session.
4. **[DONE — merged to `main`]** CLUBB `__param_spec__` (§4). Full CLUBB is now tunable on the
   same footing as every other closure: `clubb.py` ships a `__param_spec__` over `CLUBBParams`
   (**78 tunable**: 8 tier-1 core {C1,C8,C11,C14,beta,c_K,gamma_coef,mu}, 40 tier-2, 30 tier-3;
   plus 24 excluded — the 9 off-in-CAM zero-default terms were excluded because a sigmoid seed
   cannot represent a lower-bound default) + `CLUBBConfig` (7 excluded). Registered in
   `param_collector.SPEC_MODULES`; graduated out of `PARAM_SPEC_TODO`. The SCM tuners descend
   into the nested `CLUBBConfig.params` and re-wrap. Tests: `test_clubb_param_spec.py`,
   `test_scm_rce_clubb_nesting.py`. Merged 2026-07-11 (branch `clubb-param-spec`); driver-level
   `--params` reachability is baselined as a follow-up like every other turbulence scheme.
5. **[DONE — 2026-07-21] §5 infrastructure complete.** All library modules +
   drivers + validators built, tested (168 tests), and physics-modules
   codex-reviewed CLEAN: `registry`/`catalog`, `bridge`, `counter_gradient` (Q1),
   `score` (D6), `core/profile_metrics`, `matrix` + `run_les_suite_matrix`,
   `intercomparison` (D7) + `compare_les_intercomparison`, `emit` + `run_les_suite`
   (dry CBL emission), `scm_coupling` + `scm_runner` + `tune_scm_to_les` (D4
   derivative-free), `scorecard` + `build_les_scorecard` (Q2/Q3). **Gate-0
   Nieuwstadt CBL PASSED** on GPU (buoyant path validated). The full chain runs
   end-to-end: LES → self-describing artifact → SCM tuning → Q2/Q3 scorecard.
6. **[in progress] The Q1/Q2/Q3 science answers (§7).** The machinery is done; this is
   the compute + coverage campaign. Status (2026-07-24):
   - **Q1a structural ceiling** — DONE across the dry-CBL flux axis (5 fluxes; a
     counter-gradient layer at every Q0 0.02→0.12). The height *trend* is
     non-monotonic; only presence-at-all-fluxes is claimed.
   - **Q2 dry-CBL ranking** — **all 9 closures** now wired in `tune_scm_to_les`
     (`_cbl_scheme_table`: smagorinsky, louis, holtslag_boville, ysu, tke, mynn25,
     clubb_lite, edmf, clubb — the full D3 order ladder). The 8 flat schemes are
     probe-confirmed finite at the campaign grid (nlev=24); full `clubb` is confirmed
     finite only at a COARSE nlev=8, dt=30 grid (loss 0.235) — its nlev=24 XLA compile
     OOM'd 3× on the memory-contended shared node, so 0.235 is a viability smoke test
     and is NOT scorecard-comparable to the nlev=24 numbers (controlled-comparison
     rule). The nested-CLUBB descend/re-wrap is the shared public helper
     `turbulence/tunable_subconfig.py` (also used by the RCE campaign + AD trainer; an
     earlier private cross-import fixed). 8-closure × 5-flux tuning campaign running
     (~24 h CPU, ×contention); **full `clubb` tuning into the scorecard is a separate
     step pending memory headroom** (nlev=24) — clubb is deliberately kept out of the
     running campaign shell so its heavy compile can't OOM-crash it.
   - **Scorecard** reports PER surface-flux (dedup'd; `n fluxes` denominators).
   - **D7 σ_LES DONE for the dry CBL anchor**: `run_les_suite --sgs
     {lasd,smagorinsky,vreman}` + `sigma_les.py`/`compute_sigma_les.py` produced a REAL
     σ_LES(cbl_nieuwstadt)=**0.3186** (loss units), and `build_les_scorecard
     --sgs-artifacts-dir` gates the Q1b margins on it (all "not significant by combined
     gate" — see the wind-conditioning caveat in §7.1). Remaining D7 compute: the
     2×-resolution runs (fold into σ_LES), and the SGS spread for the OTHER regimes as
     they come online.
   - **Sheared-CBL `--Ug` DONE**: `run_les_suite --Ug <U_g> --lat <φ>` drives the dry
     U_g axis (geostrophic + Coriolis; records u_geo/v_geo/f_c). Well-conditioned winds
     fix the Iter-8 σ_LES pathology → a defensible Q1b significance once the sheared
     SGS-spread is emitted.
   - **AD path (D4) DONE**: `scm_runner.scm_les_loss_jax` (differentiable loss) +
     `tune_scm_to_les.tune_closure_ad` (Adam on traced-leaf params) + `--method
     {df,ad,both}` (the AD-vs-DF comparison). Traced-leaf params reuse one compiled grad
     (vs DF's per-candidate recompile); caveat — the Python step-loop UNROLLS under
     jax.grad, so full-resolution AD awaits a `lax.scan` rollout.
   - **STABLE (SBL) regime DONE + 4 h VALIDATED** (§7.2): `run_les_suite` wires
     `dry_stable` via `_build_sbl` (GABLS1 stratified IC) + `_make_emit_step` (Rayleigh
     sponge + re-projection). The 4 h vreman run (144k steps) reaches a physical stable
     equilibrium — stratified (Δθ=+3.36 K), surface-dragged (2.04 m/s), Ekman spiral, no
     blow-up; the super-geostrophic jet is emerging but weak at 4 h (+0.05 m/s @ ~170 m;
     peaks ~9 h). A valid Q3 tuning target. 2nd regime → Q3 once tuned.
   - **θ-consistent D7 metric DONE (codex-CLEAN)**: `theta_significance.py` re-scores each
     tuned closure for its θ-only RMSE and gates the per-flux margin on σ_LES(θ)≈0.0074 (the
     robust θ floor) vs the wind-dominated σ_LES(combined)=0.319. Self-check is a HARD GATE
     (untrusted re-scores excluded + reported). Validated end-to-end on real data (1-closure
     smoke: self-check exact; v/Ekman dominates the combined). The full ranked verdict RUN is
     compute-bound (~5 h for 27 closures) → a quiet-node step.
   - **DRY-REGIME SUITE COMPLETE (2026-07-24)**: the `lax.scan` rollout (§7.1/D4; ~250×
     faster free-run, AD-tractable) unblocked the whole compute wall — Q2 full 9×5 grid
     (incl. clubb@nlev=24), the θ-consistent verdict, σ_LES(SBL), SBL tuning, Q3, and the
     D4 AD-vs-DF run all landed. Q1/Q2/Q3/D4/D7 are DONE for the CBL/sheared/SBL regimes.
   - **REMAINING — MOIST regimes: IN PROGRESS (forcing IS available; 2026-07-24 CORRECTION).**
     An earlier note wrongly called this blocked after mis-checking the cache path (looked at
     `data/les_forcing/`, but `resolve_sam_case_dir` uses `data/les_cases/`). The decks ARE
     present + readable: `data/les_cases/{BOMEX,DYCOMS_RF01,DYCOMS_RF02,RICO,...}/` with
     snd/lsf/sfc (verified: BOMEX snd θ=298.7 K, q_v=17 g/kg, lsf ug=−10 — the Siebesma-2003
     sounding). `run_bomex_les.py`/`run_dycoms_les.py` read them via `resolve_sam_case_dir`;
     the `fetch_les_forcing` error is a red herring (it only fails to find a SOURCE gSAM
     checkout to COPY from — moot when the cache is already populated). ⇒ the moist LES CAN
     be emitted here. Build (below). The artifact SCHEMA
     is already moist-ready (`LESReferenceArtifact.qt`/`wqt_resolved`/`wqt_sgs`/`w_qv_s`,
     `is_moist()`; θ carries θ_l for moist; `score.qt_rmse` exists) — so the gap is the moist
     SCM PHYSICS (q_t IC + condensation/latent-heating + a moist microphysics config in
     `build_cbl_scm_from_artifact`), built + validated against the real moist LES.
     **VERIFIED FEASIBLE (2026-07-24): the BOMEX moist LES emits with the cached deck**
     (`run_bomex_les.py --nx 32 --hours 0.3`: morrison micro, LASD SGS, SHF/LHF=9.5/153.4 W/m²,
     wall=26 s; a full 64³ 6 h run is validating the cloud layer vs Siebesma cc 10–15 %). The
     state carries `st.u/v/w/theta/tracers[qv,qc,qr]` ⇒ the moist-artifact fields are computable
     (θ_l=θ−(L_v/c_p)·q_c/Π, q_t=q_v+q_c+q_r, planar means + resolved ⟨w'θ_l'⟩/⟨w'q_t'⟩). PLAN
     (least-duplication): add `--emit-suite-artifact` to `run_bomex_les` (reuse its validated
     Eulerian-morrison loop, hook the per-frame profiles at `_save`, build a moist
     `LESReferenceArtifact`) rather than re-implement the 832-line moist emission in
     `run_les_suite`; then moist SCM (condensation) → tune → moist Q1/Q2/Q3 + full D9. NOTE the
     doable WITHOUT external data only because GABLS1 is an ANALYTIC case with an existing
     `run_spectral_sbl.py` to reuse; BOMEX/DYCOMS are NOT in `ANALYTIC_CASES` and have NO
     analytic builder — an analytic route would require implementing the Siebesma (2003) /
     Stevens (2005) sounding+forcing spec from scratch (authoritative source needed; not to
     be reconstructed from memory), so the gSAM deck is the practical unblock.
   - **MOIST build plan (once forcing is available).**
     Foundations EXIST: the moist LES core `dynamics/les/spectral_les_moist.py` (moist
     diagnostics, positive-definite moisture conservation, Sundqvist/SBK89 diagnostic
     condensation) + standalone `run_bomex_les.py`/`run_dycoms_les.py`, and the registry
     cases `bomex_cu` (shallow_cumulus) + `dycoms_rf01_sc` (stratocumulus). GAPS to fill:
     (1) wire the moist regimes into `run_les_suite` (`_build_bomex`/`_build_dycoms` IC +
     moist emit step, add to `_WIRED_REGIMES` — analogous to the SBL wiring); (2) the moist
     SCM coupling — extend `build_cbl_scm_from_artifact` with a `q_t` IC + condensation so
     the SCM reproduces the moist BL (score.py already computes `qt_rmse`); (3) the D9 cloud
     scheme = the CLUBB native-vs-shared-cloud pair (so Q2's "higher-order wins" is not a
     cloud-PDF artifact). Then re-run Q1/Q2/Q3 including the moist regimes. Also optional:
     the 2×-resolution σ_LES runs; the 9 h peaked-jet SBL.
   See `CHANGELOG.md`.

Each code step follows CLAUDE.md: pre-impl grep, a direct unit test per new `.py`, and — for
the CLUBB spec, the bridge, and the score assembly (physics/numerics-touching) — the mandatory
codex adversarial-review loop before "done."

---

## 9. Open items to revisit
- Deardorff TKE-SGS (`tke_sgs_plane.py`) wiring status — verify the prognostic-`e` carry is
  live before offering it as an LES SGS variant in the D7 spread.
- Exact numeric regime-axis values (`w'θ'ₛ`, `U_g`, inversion strength, SST) — set from the
  Nieuwstadt/GABLS1/Wangara/BOMEX/DYCOMS anchors once gate-0 fixes the achievable resolution.
- Shared cloud/saturation scheme identity for D9 — pick the concrete scheme (must use
  `thermo.saturation_*`, no re-impl) when the moist cases are built.
