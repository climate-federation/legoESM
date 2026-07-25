# LES_SUITE — build CHANGELOG (append-only; condensed every 10 iterations)

Detailed, iteration-level progress log for the LES-truth suite. The living design
doc is `LES_SUITE.md`; this file is consulted only when the design doc's summary is
insufficient. Newest entries at the bottom of each section.

Deliverable tracker (LES_SUITE.md §5 architecture + §7 science deliverables):

**§5 INFRASTRUCTURE: COMPLETE** (all les_suite modules built, tested, codex-CLEAN).

| Deliverable | Status (2026-07-24, iter 15) |
|---|---|
| §5 infrastructure (registry…scorecard, gate-0) | **DONE** (codex-CLEAN) |
| Q1a structural ceiling (dry CBL flux sweep) | **DONE** — CG layer at every Q0 0.02→0.12 |
| Q1b skill threshold (dry CBL, prognostic) | **DONE** — combined-gate NOT significant (wind-dominated); **θ-consistent gate: SIGNIFICANT at every flux** (margins ≫ σ_LES(θ)=0.0074) ⇒ closures distinguishable on θ |
| Q2 nine-closure ranking | **DONE — FULL 9×5 grid** (45/45 tuned via scan, incl. clubb@nlev=24). Higher-order clubb/clubb_lite/edmf best at every flux > tke > holtslag > smag/louis > ysu/mynn25 ⇒ **closure ORDER buys skill** (both combined & θ) |
| Q3 inter-regime coefficient spread | **DONE — NOT significant vs σ_LES** (both regimes tuned 9/9; σ_LES(SBL)=0.546; CBL→SBL coeff-transfer penalty 0–0.16 < σ_LES ⇒ apparent spread is tuning noise on loss-insensitive params). §7.3 |
| D4 AD path (+ AD-vs-DF) | **DONE + AD-vs-DF RUN** — scan rollout makes jax.grad tractable at full res; AD reproduces the DF optimum (gaps ≤0.009, AD occasionally better) ⇒ closures gradient-calibratable |
| D7 σ_LES | **DONE for dry CBL** (free 0.319 + sheared 0.283 real, gate wired); moist/2×-res remain |

## Durable science results (dry-convective CBL; see LES_SUITE.md §7.1 for the full write-up)
- **Q1a (structural ceiling)** — a counter-gradient layer is present at **every** flux
  0.02→0.12 K m/s (base[m]/frac: 0.02→308/0.22, 0.04→292/0.38, 0.06→275/0.24,
  0.08→392/0.51, 0.12→492/0.36) ⇒ a tuning-independent ceiling on local K≥0 closures
  across the whole buoyancy axis. The layer-base *trend* is non-monotonic (a 2-point
  "drops with flux" read, commit ec2fa1cd6, is NOT supported); only presence-at-all-fluxes
  is claimed.
- **Q2 (tuned ranking, coarse tier-1, per surface-flux)** — best tuned loss by Q0:
  | closure | type | 0.02 | 0.04 | 0.06 | 0.08 | 0.12 | flux-mean |
  |---|---|---|---|---|---|---|---|
  | holtslag_boville | nonlocal | **0.128** | **0.166** | **0.241** | **0.439** | **0.741** | **0.343** |
  | smagorinsky (tuned) | local | 0.133 | 0.180 | 0.261 | 0.458 | 0.767 | 0.360 |
  | louis | local | 0.134 | 0.184 | 0.269 | 0.468 | 0.783 | 0.367 |
  | ysu | nonlocal | 0.152 | 0.240 | 0.363 | 0.579 | 0.962 | 0.459 |
  | mynn25 | 1.5-order | 0.152 | 0.241 | 0.365 | 0.580 | 0.965 | 0.461 |
  Finding: nonlocal (holtslag) best, 1.5-order (mynn25) worst at **every** flux ⇒
  "closure order buys skill" NOT supported for the dry CBL (consistent with Q1a).
  Caveats: coarse tier-1 search; only smagorinsky responds to tuning (+8–19%); the
  flux-mean de-dups the 0.06 anchor (a prior scorecard read 0.326). The tke/clubb_lite/
  edmf/clubb (closures 6–9) tunes are the running-campaign / deferred follow-on.
- **Q1b (skill threshold)** — nonlocal (holtslag) beats best-tuned local (smag) at every
  flux, margins +0.0053,+0.0142,+0.0207,+0.0196,+0.0257 (generally widening, not strictly
  monotone). **D7 gate:** σ_LES(cbl_nieuwstadt)=0.3186 ⇒ NONE of the margins are
  significant — but σ_LES(combined) is wind-dominated (σ_θ=0.0074 ≪ σ_u=0.43/σ_v=0.34)
  and ill-conditioned for the Ug=0 CBL ⇒ a defensible threshold needs the sheared cases.

## Conventions locked during the build
- LES reference artifacts are **self-describing**: each carries both the truth profiles
  AND the exact forcing the LES received, so the SCM bridge reconstructs `SCMForcing`
  from the artifact (the controlled-comparison "same forcing" rule) rather than
  re-deriving it from the case.
- Reused profile-metric primitives live in `training/scm_rce_metrics.py` (`weighted_std`,
  `weighted_rmse`, `safe_sqrt`) — re-exported via `core/profile_metrics.py`. No new numerics.
- Sign conventions (match `scm_forcing.py`): `subsidence_w` +up; surface kinematic heat
  flux `w_th_s` [K m/s] +up (into the BL); `theta_adv` [K/s], `qv_adv` [(kg/kg)/s].
- Coriolis: `f = 2Ω sinφ` (Ω from `legoesm.constants`); NH geostrophic/Ekman
  `du/dt=+f(v−vg)`, `dv/dt=−f(u−ug)` toward `u_geo`.
- σ_LES lives in the SCM tuner's EXACT loss units (final-snapshot `prognostic_profile_score`
  via the shared `scm_runner.final_prognostic_truth`) so it gates the Q1b/Q2 margins.

---

## Iteration log (CONDENSED 2026-07-24 at iter 10 — full prose in each iter's commit
message; the science numbers are above + in LES_SUITE.md §7.1)

**Iters 1–2 (2026-07-21..22) — infrastructure + first science.** CPU library
(`bridge`/`counter_gradient`/`score`/`matrix`, codex-CLEAN, 6 round-1 defects fixed);
GPU pipeline (`emit`/`run_les_suite`, `scm_coupling`/`scm_runner`/`tune_scm_to_les`/
`scorecard`); gate-0 Nieuwstadt CBL PASSED (σ_w/w_*=0.68); 5 closures wired; the
`safe_sqrt(NaN)=0` tuner bug fixed (+inf on non-finite; instrument-don't-infer). Perf
limit identified: the DF tuner recompiles per candidate (~11–12 min/eval) → the AD path
is the fix.

**Iters 3–9 (2026-07-24, one commit each):**
- **Iter 3 `18e0bd5de`** — per-flux scorecard + anchor dedup (typed flux key, n-flux
  denominator); Q2 roster 5→8 (tke/clubb_lite/edmf, probe-finite); 8-closure campaign
  launched. Q2 flux-robust ranking established.
- **Iter 4 `cc7a26350`** — 9th closure `clubb` via a shared `tunable_subconfig.py`
  (extracted the RCE campaign's private nested-CLUBB descend/re-wrap, fixing a private
  cross-import); apply-site factored to `apply_overrides_to_base`. clubb probe-finite
  0.235 at nlev=8 (nlev=24 OOMs on the contended node → tuning deferred). D3 ladder complete.
- **Iter 5 `2e2921637`** — Q1b local→nonlocal skill threshold in the scorecard.
- **Iter 6 `f27da3608`** — `--sgs` selector (D7 enabler): emit the CBL under
  {lasd,smagorinsky,vreman}.
- **Iter 7 `70058135d`** — σ_LES aggregator (`sigma_les.py` + `compute_sigma_les` CLI)
  in the tuner's exact loss units (factored `scm_runner.final_prognostic_truth`).
- **Iter 8 `9ee24ead7`** — first REAL σ_LES (GPU SGS spread): 0.3186 combined; D7 gate
  wired into the scorecard (`--sgs-artifacts-dir`); all Q1b margins NOT significant
  (wind-dominated caveat).
- **Iter 9 `d7bcd867b`** — sheared CBL driver (U_g axis): `--Ug`/`--lat` + Coriolis
  (`_coriolis_f`) + geostrophic IC; artifact records u_geo/v_geo/f_c. GPU-validated
  Ekman (u 4.69→8.0, v 0.63→0.02); winds now well-conditioned. Sign convention
  codex-verified.

- **Iter 10 (this commit)** — CHANGELOG condensed (this file). Emitted the sheared
  {lasd,smagorinsky,vreman} SGS spread at Ug=8 (GPU) and computed the sheared σ_LES:
  combined 0.283, θ 0.0073, u **0.114**, v 0.476 — vs free-conv 0.319/0.0074/**0.431**/
  0.344. Shear well-conditions the streamwise wind as hypothesised (σ_u 0.43→0.11, 4×),
  but the weak Ekman cross-wind keeps σ_v large (0.34→0.48) → σ_combined only drops
  0.32→0.28; σ_LES(θ)≈0.0074 is a robust metric-invariant floor. ⇒ a clean D7 verdict
  needs a θ-CONSISTENT metric (score the closure loss AND σ_LES on θ alone — don't mix a
  combined margin with a θ-only σ). Doc-only + already-reviewed σ_LES code (codex-exempt).

- **Iters 11–20 (2026-07-24, one commit each) — the science campaign + the scan breakthrough**
  (durable results in the tracker above + LES_SUITE.md §7; per-iter prose in each commit):
  - **11** — AD path (D4): `scm_les_loss_jax` + `tune_closure_ad` (Adam, traced leaves) + `--method {df,ad,both}`.
  - **12** — STABLE SBL regime wired (`_build_sbl` GABLS1 sounding + `_make_emit_step` Rayleigh sponge/re-projection).
  - **13** — resumable science campaign launched (flux sweep + tuning).
  - **14 `8e47a6dc2`** — flock single-instance guard on the campaign (orphan+resubmit race).
  - **15 `2211a7065`+`f02769564`** — SBL 4 h VALIDATED (stratified/Ekman, jet emerging) + `scm_les_final_score` (θ/u/v breakdown) + nlev/dt recorded.
  - **16 `3a7d081a1`** — θ-consistent D7 tool (`theta_significance.py`, self-check HARD GATE; codex-CLEAN).
  - **17 `f88a980fa`** — **lax.scan SCM rollout (~250×)** (`pure_step`+`scm_scan_final_state`; machine-precision vs `run`, AD-tractable) — the breakthrough that collapsed the compute wall.
  - **18 `2de7c6260`** — **Q2 COMPLETE (full 9×5)** via scan (clubb@nlev=24 unblocked) ⇒ closure ORDER buys skill (higher-order best; combined + θ agree).
  - **19 `c1a677e3b`** — **Q3 DONE + geostrophic wiring** (SBL tuned 9/9; σ_LES(SBL)=0.546; CBL→SBL coeff-transfer <σ_LES ⇒ spread NOT significant = tuning noise).
  - **20 `72d7c278e`** — **D4 bonus RUN** (AD≈DF, gaps ≤0.009 ⇒ closures gradient-calibratable) + this condense.
  - **MILESTONE: Q1/Q2/Q3/D4/D7 all DONE for the dry regimes (CBL/sheared/SBL). Remaining: moist (BOMEX/DYCOMS) + D9 cloud.**

- **Iter 21 — MOIST regimes BLOCKED on external forcing data; D4 9/9 confirmed.** clubb's
  78-param AD finished (df=0.2308/ad=0.2309, gap 0.0001) → D4 complete 9/9, all AD≈DF.
  Investigated the last deliverable (moist BOMEX/DYCOMS): it is BLOCKED on the third-party
  gSAM `CASES/` forcing decks, which are NOT in the repo and NOT on this system
  (`fetch_les_forcing --only BOMEX` → "Could not find a gSAM checkout"; `LEGOESM_GSAM_ROOT`
  unset; the fetcher only copies from a local checkout, no download). The moist LES CANNOT
  be emitted here — the user must supply a gSAM checkout (`LEGOESM_GSAM_ROOT=<dir with
  CASES/>`). Documented in §8 with the build plan for when the forcing is available.
  **⇒ every deliverable PRODUCIBLE in this environment is DONE (dry-regime Q1/Q2/Q3/D4/D7);
  the moist regime is externally blocked, not incomplete-by-effort.**

- **Iter 22 — sheared CBL Q1b (well-conditioned): closure ranking is SHEAR-DEPENDENT.** While
  moist stays externally blocked, did a producible enrichment the docs flagged as the Q1b
  "next step": tuned all 9 closures on the sheared CBL (Ug=8, via the geostrophic wiring —
  validates it end-to-end on the CBL) + θ-gate (σ_LES(θ)=0.0073). Ranking clubb_lite≈clubb≈
  mynn25 best, smagorinsky worst — vs the FREE CBL (mynn25 worst, higher-order best), shear
  REORDERS the mid tier: mynn25 (MYNN, shear-production) jumps worst→3rd-best, local smag
  drops to worst; higher-order clubb/clubb_lite stay robust in BOTH. ⇒ order-buys-skill holds
  for the higher-order family across shear; the mid-tier ranking does not transfer. §7.1.

- **Iter 23 — Q3-shear axis, scorecard hardening, session regression VALIDATED.** (a) Q3-shear
  cross-transfer (free-CBL coeffs → sheared CBL): 0/9 exceed σ_LES(sheared)=0.283 ⇒ coefficient
  portability holds on the SHEAR axis too (same as CBL→SBL) — the shear-dependent ranking is
  default-closure-skill, not a tuned-coeff requirement (§7.1). (b) Hardened the scorecard: a
  `(regime,q0)` slice mixing >1 distinct artifact (e.g. free vs sheared CBL) now WARNS loudly
  (the per-scheme dedup could silently drop/replace a config); sheared records isolated in
  `tuned/sheared_analysis/` (2 tests). (c) Full les_suite regression **224 passed** — the
  session's scan-rollout + geostrophic + scorecard + score-factoring changes are green, no
  cross-interaction breakage. **Dry-regime suite fully complete + validated; moist externally
  blocked (gSAM forcing).**

- **Iter 24 — CORRECTION: moist regimes NOT blocked; forcing IS cached. Moist build STARTED.**
  Earlier iters wrongly called moist "blocked on external gSAM forcing" after mis-checking the
  cache path (`data/les_forcing/` vs the real `data/les_cases/`). The decks ARE present +
  readable: `data/les_cases/{BOMEX,DYCOMS_RF01,DYCOMS_RF02,RICO,...}/` (snd/lsf/sfc; BOMEX
  θ=298.7/q_v=17 g/kg/ug=−10). `run_bomex_les` reads them via `resolve_sam_case_dir`. Starting
  the moist build: wire moist into `run_les_suite` → emit BOMEX/DYCOMS → moist SCM q_t coupling
  → moist Q1/Q2/Q3 + full D9.
- **Iter 25 — moist build PROGRESSING (emission validated; artifact fields wired).** (a) Full
  64³ 6 h BOMEX validated: a cumulus layer forms (cc≈0.067 2nd-half-mean, LWP≈2.4 g/m²,
  qc_max~1e-3; under Siebesma cc 10–15 % as expected at coarse 200 m res). (b) Extended the
  shared `les_record._profiles` to emit ⟨w'θ'⟩ (=⟨w'θ_l'⟩ moist) always + q_t/⟨w'q_t'⟩ when
  q_v is passed (4 tests; dry set unchanged). (c) `run_bomex_les` both _save paths now pass
  qv3 → prof_NNN.npz carry qt/wqt/wtheta (verified end-to-end). NEXT: the prof→
  `LESReferenceArtifact` converter (`build()` exposes ug/vg/sfc-fluxes/f_c) via an
  `--emit-suite-artifact` flag → moist SCM (condensation) → tune → moist Q1/Q2/Q3 + D9.
- **Iter 26 — moist DATA PIPELINE COMPLETE (LES→artifact); real BOMEX artifact emitted.**
  Built + tested the moist LES→`LESReferenceArtifact` converter (`run_bomex_les
  --emit-suite-artifact`): valid self-describing moist artifact (θ_l, q_t, u, v, resolved
  ⟨w'θ_l'⟩/⟨w'q_t'⟩ + Coriolis/geostrophic/subsidence/advection forcing; qt∈[0.003,0.017]
  kg/kg = BOMEX). sgs labelled `lasd` when `--dynamic`. Real 64³ 6 h `bomex__lasd.npz`
  emitting. MOIST-SCM DESIGN (next, one coherent unit): `_moist_cbl_physics(turb)` =
  PhysicsConfig with `microphysics=MicrophysicsConfig(scheme="sundqvist")` (diagnostic
  condensation, shallow-cu) + turbulence; `build_cbl_scm_from_artifact` moist branch
  (`is_moist`) sets q_v IC = q_t (q_c≈0 at BOMEX t0 ⇒ q_v=q_t), T IC from θ_l (≈θ at t0),
  SCMForcing with subsidence_w/theta_adv/qv_adv/u_geo/w_qv_s from the artifact; moist score
  extracts q_t=Σ(q_v,q_c,q_r) from the SCM state + scores via `score.qt_rmse`. Then tune →
  moist Q1/Q2/Q3 + D9.

- **Iter 27 — MOIST SCM BUILT + VALIDATED + BOMEX Q2 DELIVERED (codex-CLEAN).** Wired the
  moist regime end-to-end (`scm_runner.py`): `_moist_cbl_physics` (Sundqvist condensation),
  moist branch in `build_cbl_scm_from_artifact` (q_v IC=q_t(t0); subsidence/θ-adv/q_v-adv/
  w_qv_s forcing, all +up = SCMForcing convention, sign-checked), `scm_final_moist_on` →
  (θ_l,u,v,q_t) with θ_l=θ−(L_v/(c_pd·Π))·q_c (helper `_liquid_water_theta`), q_t=q_v+q_c+q_r.
  BOTH DF (`scm_les_final_score`) AND AD (`scm_les_loss_jax`) branch on `is_moist`. `scm.py`
  create() now pre-allocates q_g UNCONDITIONALLY for active microphysics (shared
  MicrophysicsOutput always emits dq_g_dt) → fixes lax.scan carry-pytree stability.
  Codex adversarial review: found+fixed ONE high-sev defect (AD path missing scm_qt →
  ValueError on moist), added regression tests (θ_l sign+decrement, combined-folds-qt, moist
  jax.grad); round-2 VERDICT CLEAN. 17 scm_runner tests + 117 CPU-suite green. Aligned the
  BOMEX emit case-label to the registry name `bomex_cu` (scorecard get_case resolution).
  **BOMEX shallow-cumulus Q2 (9 closures tuned, shared Sundqvist cloud = D9 shared-cloud
  control): holtslag_boville 0.212 > louis 0.229 > clubb 0.239 > edmf 0.248 > ysu 0.250 >
  mynn25 0.276 > smag 0.304 > clubb_lite 0.308 > tke 0.351.** VERDICT: "order buys skill"
  does NOT hold in shallow Cu (nonlocal+local lead, higher-order bunched) — regime-specific,
  NOT a cloud-PDF artifact (cloud held fixed across arms). CAVEATS: different metric than dry
  (compare within-BOMEX only); wind-weighted; σ_LES(shallow_cumulus) not yet available (1 SGS
  variant) → margins UNGATED/PLAUSIBLE. See LES_SUITE.md §7.4.

- **Iter 28 — θ_l CONFOUND FIXED (codex HIGH) + BOMEX Q2 D7-GATED + DYCOMS converter.** Codex
  caught that the spectral moist LES prognoses ACTUAL θ (IC saturation-adjusts θ_l→θ), but the
  artifacts recorded/labelled it θ_l while the SCM produces θ_l → the first BOMEX Q2 compared θ
  to θ_l. FIXED: `scm_coupling.liquid_water_theta` = ONE canonical θ_l reduction (SCM delegates);
  `_record_theta_l` in both LES drivers records θ_l from prognostic θ + cloud q_c (q_c only, not
  rain, matching SCM); `sigma_les_prognostic` extended to moist (θ_l,u,v,q_t + sigma_qt). Codex
  re-review: production CLEAN. Re-emitted 3 BOMEX θ_l artifacts (lasd/smag/vreman, tight: θ_l sfc
  ~299.05, top ~311.77) → re-tuned 9 → σ_LES(shallow_cumulus)=0.0531. **CORRECTED + GATED BOMEX
  Q2: holtslag 0.310 > louis 0.327 > clubb 0.336 > edmf 0.346 > ysu 0.355 > mynn25 0.384 > smag
  0.417 > clubb_lite 0.420 > tke 0.467.** Ordering PRESERVED vs the θ-based run (confound moved
  the scale, not the science). GATED VERDICT (CONFIRMED): top-5 tied within σ_LES (higher-order
  clubb statistically tied with LOCAL louis — buys no resolvable skill), Q1b local-vs-nonlocal
  margin 0.017 sub-σ_LES (not significant), but top-vs-tail 0.16≈3σ_LES IS resolved (weak tail:
  smag/clubb_lite/tke). "Order buys skill" does NOT hold in shallow Cu — regime-specific, not a
  cloud-PDF artifact. Also: DYCOMS `--emit-suite-artifact` converter (RF01 LW cooling→theta_adv,
  sign-checked; records 2nd-half-mean radiative θ-tendency; case-label→registry `dycoms_rf01_sc`;
  q_t incl rain) + unit tests; end-to-end θ_l recording test (closes codex LOW). D9-native fully
  scoped (clubb `cloud_source` toggle; CONFIRMED it's a turbulence-interface change since
  clubb_turbulence takes q_v not q_c; gap expected small by condensation-invariance).

- **Iter 29 — DYCOMS (moist regime 2) DELIVERED, σ_LES-gated NULL.** DYCOMS pipeline validated
  end-to-end: RF01 LW cooling (make_stevens_lw) → artifact theta_adv; moist SCM runs stably
  (radiative + subsidence + strong geostrophic), forms cloud. Emitted 3 SGS artifacts (64²×96,
  θ_l sfc~289.4 top~306 tight) → tuned 9 → σ_LES(stratocumulus)=0.231. **DYCOMS Q2: mynn25 0.825 >
  louis 0.835 > clubb_lite 0.836 > clubb 0.909 > holtslag 0.912 > tke 0.917 > edmf 0.929 > smag
  0.938 > ysu 0.956.** GATED VERDICT: NULL — whole spread 0.131 < σ_LES 0.231 → NO closure
  distinguishable (higher-order clubb 4th, tied with everything). High absolute losses (SCM
  reproduces the radiative/geostrophic Sc poorly; winds dominate). CAVEAT: RESOLUTION-LIMITED —
  64²×96 Sc thins to LWP ~9 (ref 50-80); 96²×192 is the refinement. **COMBINED MOIST VERDICT
  (BOMEX+DYCOMS): higher-order does NOT win in moist regimes** (BOMEX top-tier tied, DYCOMS all
  tied), contrasting the dry-CBL higher-order win; best moist closure is regime-dependent
  (holtslag/mynn25) AND within σ_LES. Cloud held fixed (shared Sundqvist) → turbulence statement,
  not cloud-PDF. See LES_SUITE.md §7.5.

**NEXT (remaining for DONE):** (1) full-D9 NATIVE leg — the CLUBB turbulence-interface change (add
cloud q_c to the scheme input + `cloud_source` toggle + validator loop; OR the smaller
clubb-internal `cloud_buoyancy` toggle; scoped in memory `les-suite-moist-status`). (2) OPTIONAL
higher-res DYCOMS (96²×192) for a realistic Sc + tighter σ_LES. (3) clubb 2×-res σ_LES.
