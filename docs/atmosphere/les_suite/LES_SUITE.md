# LES_SUITE — an LES-truth suite for tuning and comparing SCM turbulence closures

Status: **scope + design (living doc)**. Author: A. Connolly. Started 2026-07-10.
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
  stated margin, in both diagnostic and prognostic scoring.
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
  flux** (nonlocal already wins at 0.02) and the local handicap broadly deepens with
  buoyancy — consistent with the Q1a structural ceiling (a counter-gradient layer at
  every flux). CAVEAT (D7): significance vs σ_LES is pending the SGS-spread runs — a
  margin below σ_LES is not a result; the +0.005 margin at 0.02 is small.
- **Q2 (tuned ranking, coarse tier-1), flux-robust** — `holtslag_boville` (nonlocal)
  is best and `mynn25` (1.5-order) worst at **every** flux point (best tuned loss,
  Q0=0.06 anchor shown): holtslag 0.241 < smagorinsky (tuned) 0.261 < louis 0.269 <
  ysu 0.363 < mynn25 0.365; the ordering is identical at 0.02/0.04/0.08/0.12 (see the
  per-flux scorecard). Finding: **closure order does NOT buy skill here** — consistent
  with Q1a. Coarse search; only smagorinsky responds to tuning (+8–19%), the rest
  ~0% (default-dominated); single case/regime/metric — not definitive.
- **Q3 (machinery)** — the tuned-coefficient scorecard assembles; the inter-regime
  spread is still 0 (one regime). σ_LES need the other regimes and the SGS-spread runs.
- **Scorecard now reports per surface-flux** (not one regime-mean) and de-duplicates
  the anchor: a prior scorecard double-counted the 0.06 point (two protocol-inconsistent
  tuned records), inflating holtslag's flux-mean to 0.326; the dedup'd mean over the
  5 distinct fluxes {0.128,0.166,0.241,0.439,0.741} is **0.343**.
- **D4 bonus** — the AD-vs-derivative-free comparison awaits the AD path (which also
  fixes the tuner's recompile-per-candidate cost; the RCE trainer's `lax.scan` +
  `eqx.filter_value_and_grad` pattern is the reuse target).

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
   - **D7 σ_LES enabler DONE**: `run_les_suite --sgs {lasd,smagorinsky,vreman}` now
     emits the SGS-spread variants (was hardcoded LASD). Still to build: the emission
     (3 GPU runs/case) AND a multi-artifact σ_LES aggregator (spread of the mean
     profiles across the variant + 2×-resolution artifacts) — `intercomparison.py`
     only checks a single run vs published bands, it does NOT yet compute σ_LES. σ_LES
     gates the Q1b margin significance + Q3 spread.
   - **Remaining**: wire the stable/moist regime IC builders + sheared-CBL `--Ug` in
     `run_les_suite.py` (only dry-convective wired) → Q3 inter-regime + Q2 per-regime;
     the AD path (D4 comparison + perf); run the D7 σ_LES SGS-spread + 2×-resolution.
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
