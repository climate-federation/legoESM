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
  flux** (nonlocal already wins at 0.02), consistent with the Q1a structural ceiling.
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
  is the best closure, not its calibration.
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
  sensible ordering by closure sophistication, robust across the whole buoyancy axis. This
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
   - **REMAINING — MOIST regimes: BLOCKED on external forcing data (2026-07-24).** The moist
     LES emission (`run_bomex_les.py`/`run_dycoms_les.py`) requires the third-party gSAM
     `CASES/BOMEX` + `CASES/DYCOMS_RF01` decks (Khairoutdinov SAM input: snd/lsf/sfc),
     which are NOT bundled in the repo (correct — third-party) and are NOT present on this
     system: `fetch_les_forcing.py --only BOMEX` → "Could not find a gSAM checkout";
     `LEGOESM_GSAM_ROOT` is unset and neither autodetect path exists; the fetcher only
     COPIES from a local gSAM checkout (no download). **⇒ the moist LES artifacts cannot be
     produced in this environment** — the user must provide a gSAM checkout via
     `LEGOESM_GSAM_ROOT=<dir containing CASES/>` (or `--gsam-root`), then
     `python scripts/data/fetch_les_forcing.py --only BOMEX DYCOMSII`. Once the forcing is
     available, the code path below is the build (a fresh-session task). NOTE the SBL was
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
