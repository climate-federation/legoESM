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
| Q1b skill threshold (dry CBL, prognostic) | **DONE** — nonlocal beats local every flux; NOT significant vs σ_LES (wind-dominated, Ug=0) |
| Q2 nine-closure ranking | **7/9 tuned at the anchor** (clubb_lite 0.231 BEATS holtslag 0.241 at q0=0.06; +tke); edmf tuning, clubb nlev=24 deferred (OOM); flux-sweep for closures 6–9 pending |
| Q3 inter-regime coefficient spread | **regimes ready + SBL validated at 4 h** (stable/stratified/Ekman, jet emerging weak — peaks ~9 h); needs SBL SGS spread → σ_LES(SBL), then SBL tuning |
| D4 AD path (+ AD-vs-DF) | **DONE** (`scm_les_loss_jax` + `tune_closure_ad` + `--method both`); full-res needs the lax.scan rollout |
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

- **Iter 11 (this commit) — AD path (D4).** Feasibility confirmed by probe: the SCM
  forward IS differentiable end-to-end w.r.t. traced turbulence params (the blocker was
  `scm_les_final_loss`'s `float()`/`bool()` concretization). Built:
  `scm_runner.scm_les_loss_jax` (a pure-JAX, traced-scalar form of the SAME final-snapshot
  objective — jax.grad'able; smagorinsky C_s grad −2.3e-4 finite/nonzero, mynn25 A1 ~0
  matching its known negligible θ-leverage); `tune_scm_to_les.tune_closure_ad` (Adam
  gradient descent in normalised [0,1] param space; TRACED-leaf params → one jaxpr shape
  so XLA reuses the compiled grad, vs the DF NEW-static-config-per-candidate recompile —
  not jit'd since build's float() grid-setup is jit-incompatible with a traced config);
  `--method {df,ad,both}`
  in the CLI (`both` writes the D4 AD-vs-DF comparison: loss gap + which optimum wins).
  Tests: `test_scm_les_loss_jax_is_differentiable`, `test_tune_closure_ad_reduces_or_
  matches_default`. NOTE: `run` is a Python step-loop that jax.grad UNROLLS, so AD is
  practical only at modest nsteps until a `lax.scan` rollout lands (the full-resolution
  perf follow-on).

- **Iter 12 (this commit) — STABLE (SBL) regime wired → the 2nd regime for Q3.**
  `run_les_suite` gains `_build_sbl` (GABLS1 stratified sounding — the smooth θ ramp
  zi=100/Δ=25/γ=0.01, Ug=8 geostrophic IC + lower-half perturbations, projected) and
  `_make_emit_step` (a regime factory: dry CBL = plain `sl.step`; SBL adds the Rayleigh
  sponge in the top 25% + a re-projection so the low-level jet / GWs are absorbed at the
  rigid lid, not reflected → no ~0.9 h blow-up). `dry_stable` added to `_WIRED_REGIMES`;
  the SBL uses the case grid dt (0.1 s, stiffer stratification), `nu_floor` background
  viscosity, and negative surface flux (cooling). All faithful to the validated
  `run_spectral_sbl.py`. Coriolis via `--lat` (GABLS1 73° → f=1.39e-4). Tests:
  `test_build_sbl_stratified_ic_and_geostrophic_wind` (θ increases upward, u≈Ug, Q0<0),
  `test_emit_step_factory_rejects_unknown_regime`, wired-regimes. GPU VALIDATION (0.3 h,
  64×64×96, vreman): runs STABLY — no blow-up (the sponge holds the lid), θ stays stratified
  (sfc 264.8 → top 267.9 K, cooled from 265), the surface wind is dragged to 2.62 m/s
  (< Ug=8), forcing recorded (u_geo=8, f=1.39e-4, w'θ'=−0.005), all finite. The
  super-geostrophic low-level JET has not formed yet — expected, it develops over HOURS
  (inertial period 2π/f≈12.5 h); 0.3 h is early spin-up. So the wiring runs correctly;
  the equilibrium jet + full validation await the 4 h run. This unlocks Q3
  (dry_convective vs dry_stable inter-regime coefficient spread) once both are tuned.
  Moist regimes (shallow_cumulus/stratocumulus) still need their IC builders + D9 cloud.

- **Iter 14 (this commit) — campaign single-instance guard (Q2 throughput fix).**
  Diagnosed a real drag on Q2: TWO `run_les_suite_campaign.sh` shells were running at
  once — the original (17:27) had been orphaned from its dead harness task (reparented to
  init, kept running) and an iter-13 resubmit (19:02) then duplicated it. Both landed on
  the heavy `clubb_lite` (nlev=24) tune, racing on the same `__clubb_lite__df.json` and
  halving each other's CPU (the "1h50m clubb_lite" was contention, not a deadlock — both
  at 136% CPU). They self-heal (idempotent skip-if-exists) but waste a core and can clobber.
  FIX: a `flock -n` single-instance guard (fd 9 on `campaign/.campaign.lock`, auto-released
  on exit) so any concurrent resubmit is a clean no-op. Self-tested (2nd instance refuses)
  + `bash -n`. Shell orchestration only (no numerics/AD) → codex-exempt. Does NOT stop the
  two pre-guard shells (kill is classifier-blocked); they finish idempotently, and future
  resubmits can't re-duplicate. AD scan-rollout re-scoped as NOT on the critical path: D4's
  AD-vs-DF agreement is a controlled `--method both` run at EQUAL config, not full-res AD
  (reduced-res AD vs full-res DF would be a resolution confound); the scan-rollout is a pure
  perf nicety, deferred. SBL 4 h vreman emit still integrating (jet develops over ~12.5 h).

- **Iter 15 (this commit) — SBL 4 h equilibrium VALIDATED + θ-breakdown score exposed.**
  The GABLS1 4 h vreman emit (144,000 steps @ dt=0.1 s, 64×64×96, wall=2821 s) completed
  and OVERWROTE the 0.3 h validation. Final-frame equilibrium (LES_SUITE.md §7.2): stably
  stratified (Δθ=+3.36 K, sfc 264.3 → top 267.7 K), surface wind dragged to 2.04 m/s, a
  clear Ekman spiral (v peaks ~3 m/s at 50–100 m → 0 above 200 m), all finite, NO blow-up
  over the full 4 h (the sponge holds at the rigid lid). The super-geostrophic low-level
  JET is EMERGING but WEAK — a local wind max of 8.05 m/s at ~170 m, only +0.05 m/s over
  Ug=8; the classic GABLS1 ~1–2 m/s overshoot peaks near the ¾-inertial-period (~9 h), and
  4 h is only 0.32 of the 2π/f≈12.5 h period ⇒ early jet formation, not the peak. So the
  SBL regime is a physically-correct stable/stratified/Ekman target — usable for Q3 tuning
  (a 9 h run would give the peaked jet, a follow-on). Also (commit 2211a7065): factored
  `scm_les_final_score` (full θ/u/v + combined PrognosticScore) out of `scm_les_final_loss`
  (now a thin wrapper; a test asserts `.combined == loss` bit-exactly), and recorded
  nlev/dt in the tuner JSON — both enabling the θ-consistent D7 gate (rank closures on
  θ_rmse, gate margins on σ_LES(θ)=0.0074 vs the wind-dominated σ_LES(combined)=0.319).
  Campaign single-instance `flock` guard added (commit 8e47a6dc2) after two orphaned
  campaigns raced clubb_lite. Q2 anchor now 7/9 (added tke + clubb_lite=0.231, which BEATS
  holtslag=0.241 — a higher-order PDF closure edging the best nonlocal at the anchor flux;
  edmf still tuning, clubb deferred). **NEXT-SBL:** emit the SBL SGS spread (lasd+smag) →
  σ_LES(SBL) for the D7 gate on regime 2.

**NEXT:** (1) emit the SBL SGS spread + tune the closures on it → Q3 inter-regime spread +
Q2 per-regime (SBL vs CBL). (2) the θ-consistent D7 metric. (3) AD scan-rollout for
full-resolution AD tuning. (4) MOIST (BOMEX/DYCOMS) IC builders + D9 cloud scheme.
(5) clubb tuning at nlev=24 (memory); the 2×-resolution σ_LES.
