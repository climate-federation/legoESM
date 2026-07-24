# LES_SUITE — build CHANGELOG (append-only; condensed every 10 iterations)

Detailed, iteration-level progress log for the LES-truth suite. The living design
doc is `LES_SUITE.md`; this file is consulted only when the design doc's summary is
insufficient. Newest entries at the bottom of each section.

Deliverable tracker (LES_SUITE.md §5 architecture + §7 science deliverables):

**§5 INFRASTRUCTURE: COMPLETE** (168 tests green; physics modules codex-reviewed CLEAN).

| Component | Path | Status |
|---|---|---|
| LESCase registry + catalog | `les_suite/{registry,catalog}.py` | **done** (49959ec94) |
| LES→SCM bridge | `les_suite/bridge.py` | **done** (codex CLEAN) |
| Counter-gradient diagnostic (Q1) | `les_suite/counter_gradient.py` | **done** (codex CLEAN) |
| Score assembly (D6) | `les_suite/score.py` | **done** (codex CLEAN) |
| Shared profile primitives | `core/profile_metrics.py` | **done** |
| Matrix wiring | `les_suite/matrix.py` + `scripts/matrix/run_les_suite_matrix.py` | **done** |
| Intercomparison gate (D7) | `les_suite/intercomparison.py` + `scripts/validate/les_suite/compare_les_intercomparison.py` | **done** |
| LES ensemble driver | `les_suite/emit.py` + `scripts/run/run_les_suite.py` | **done** (dry CBL; codex CLEAN) |
| SCM↔LES coupling | `les_suite/scm_coupling.py` | **done** (regrid + θ↔T; codex CLEAN) |
| SCM forward eval | `les_suite/scm_runner.py` | **done** (codex CLEAN) |
| SCM→LES tuner (D4) | `scripts/run/tune_scm_to_les.py` | **done** (derivative-free; codex CLEAN) |
| Scorecard (Q2/Q3) | `les_suite/scorecard.py` + `scripts/validate/les_suite/build_les_scorecard.py` | **done** |
| Gate-0 Nieuwstadt CBL run | `results/les_suite/gate0_nieuwstadt/` + `docs/.../gate0_nieuwstadt_result.json` | **PASS** (buoyant path validated) |
| **Q1/Q2/Q3 science answers** | (need the full GPU ensemble + tuning campaign) | **pending compute** |

**Remaining for the §7 science deliverables** (all machinery built; this is a compute
+ coverage campaign, not new infrastructure):
1. Emit the full LES ensemble on GPU — the dry (w'θ'ₛ×U_g) grid + anchors across
   regimes (§6). Needs: the stable/moist regime IC builders in `run_les_suite.py`
   (only dry-convective CBL is wired), + the sheared-CBL driver (`--Ug`).
2. Wire the remaining closures' base configs in `tune_scm_to_les._base_turbulence`
   (only mynn25 today) + the AD path (D4 comparison).
3. Run the tuning campaign (all closures × regimes) → scorecard = Q2/Q3 answers.
4. Q1: run `counter_gradient.diagnose_truth` across the flux grid → the structural
   threshold; the skill threshold from the tuned rankings.
5. D7 σ_LES: the SGS-spread + 2×-resolution runs → the Q3 error bars.

## First real science outputs (dry-convective anchor, 2026-07-21)
Demonstrating the §7 machinery produces real numbers (full answers need the ensemble):
- **Q1a (structural ceiling)** on the science-quality 2 h 96³ x64 CBL artifact
  (`cbl_nieuwstadt__lasd.npz`, 12 frames, model-exact resolved+SGS flux):
  `counter_gradient.diagnose_truth` finds a **counter-gradient layer at 275–292 m**
  (24% of levels counter-gradient at the final time). ⇒ in the developed CBL there is
  an up-gradient transport layer where ANY non-negative eddy-diffusivity (local
  down-gradient) closure is STRUCTURALLY unable to match the LES flux, regardless of
  tuning — the tuning-independent ceiling on local closures.
  **Flux sweep COMPLETE (campaign 9157232, 5 points 2026-07-24)** →
  `results/les_suite/q1_counter_gradient_sweep.json`. A counter-gradient layer is
  present at **every** flux 0.02→0.12 K m/s (base[m]/frac): 0.02→308/0.22,
  0.04→292/0.38, 0.06→275/0.24, 0.08→392/0.51, 0.12→492/0.36. **CORRECTION** to the
  earlier 2-point read (commit ec2fa1cd6, "CG layer drops with flux"): with all 5
  points the layer-base trend is **non-monotonic** — it descends only through 0.06,
  then rises steeply at 0.08–0.12; the fraction is noisy (0.22→0.51→0.36), not
  monotone. The height/fraction *trend* is NOT a supported result on this data. The
  robust, defensible Q1a claim is the **structural** one: an up-gradient layer exists
  at every flux across the buoyancy axis ⇒ the tuning-independent ceiling on local
  K≥0 closures holds throughout, not just at the anchor.
- **Q2 (per-regime closure skill)** — mynn25 tier-1 derivative-free tuning on the 2 h
  CBL (17 candidates, dt=5, nlev=32): default loss **0.397**, best **0.397** (A1/A2
  mixing coefficients have negligible leverage on the dry-CBL θ fit → the default is
  already near-optimal). A real Q2 data point for the dry-convective anchor; the full
  ranking needs the other 8 closures wired. Scorecard generated end-to-end.
- **Q2 TUNED RANKING — dry_convective CBL, FULL FLUX SWEEP (5 closures × 5 fluxes,
  2026-07-24)** — derivative-free tier-1 tuning of each closure against each flux
  artifact; scorecard now reports **per surface-flux** (`results/les_suite/scorecard.md`),
  not a single flux-mean. Best tuned loss (lower = better) by Q0 [K m/s]:
  | closure | type | 0.02 | 0.04 | 0.06 | 0.08 | 0.12 | flux-mean |
  |---|---|---|---|---|---|---|---|
  | holtslag_boville | nonlocal | **0.128** | **0.166** | **0.241** | **0.439** | **0.741** | **0.343** |
  | smagorinsky (tuned) | local | 0.133 | 0.180 | 0.261 | 0.458 | 0.767 | 0.360 |
  | louis | local | 0.134 | 0.184 | 0.269 | 0.468 | 0.783 | 0.367 |
  | ysu | nonlocal | 0.152 | 0.240 | 0.363 | 0.579 | 0.962 | 0.459 |
  | mynn25 | 1.5-order | 0.152 | 0.241 | 0.365 | 0.580 | 0.965 | 0.461 |
  **Q2 finding (now flux-robust, not a single anchor):** `holtslag_boville` (nonlocal)
  is best and `mynn25` (1.5-order) worst at **every** flux across a 6× range ⇒
  "closure order buys skill" is NOT supported for the dry CBL — consistent with the
  Q1a counter-gradient result. **Caveats (precision rule):** COARSE tuning — tier-1
  only, small search (n_random=3), nlev=24, single case/regime, single metric (final
  θ/u/v prognostic RMSE). Only **smagorinsky** responds to tuning (+8% to +19%,
  descending with flux); the other four show ~0% improvement, so their rank is
  default-dominated (smag's rank-2 is *entirely* tuning-driven — its default 0.402
  is worse than louis). NOT a definitive calibration; Q3 inter-regime spread is still
  0 (one regime). The flux-mean row de-duplicates the anchor — a prior scorecard
  double-counted the 0.06 point (two protocol-inconsistent records), inflating it to
  0.326; the correct dedup'd flux-mean is 0.343.
- **5 closures now wired for the CBL tuner** (smagorinsky, louis, holtslag_boville,
  ysu, mynn25 — spanning local→nonlocal→1.5-order). **Preliminary DEFAULT-parameter**
  θ-loss on the 2 h CBL (NOT the tuned ranking — Q2 requires tuning each; labeled
  per the precision rule): holtslag_boville **0.241** (nonlocal, best), louis 0.281
  (local), smagorinsky 0.334 (local), ysu 0.395 (nonlocal), mynn25 0.397. The
  nonlocal holtslag_boville leading at default is consistent with the Q1a
  counter-gradient finding (nonlocal transport in the CBL). All 5 run cleanly (finite
  losses). The tuned Q2 ranking is the next campaign step.
  NOTE: single-column tuning runs on CPU (`JAX_PLATFORMS=cpu`) — trivial there and
  avoids GPU contention (the GPU is for the LES ensemble emission).
- **Perf limitation identified (tuner recompiles per candidate).** The derivative-free
  tuner rebuilds the `TurbulenceConfig` for each candidate, and the SCM's jitted step
  bakes the config in → each candidate pays a full XLA COMPILE (~40–60 s), so a
  single closure's tier-1 search is ~10 min on CPU and a full 5-closure ranking is
  ~50 min. The RIGHT fix (CLAUDE.md SegmentForcing doctrine): drive the tuned params
  as TRACED leaves via `apply_param_overrides` INSIDE a jitted/AD loss, so ONE
  compilation serves every param value — i.e. build the AD path (D4's other half),
  which also gives the AD-vs-derivative-free comparison. Until then the tuned ranking
  is run with reduced candidates / a persistent background batch.
- **BUG found + fixed (CLAUDE.md precision rule — instrument, don't infer)**: the FIRST
  Q2 run reported "100% improvement, best_loss=0.0". Instrumenting it (not trusting it)
  showed the "best" candidate's SCM had DIVERGED to NaN θ, yet `scm_les_final_loss`
  returned 0.0 — the score's `safe_sqrt` maps NaN→0 (a perfect fit), so a blown-up SCM
  was selected as best. Fixed: the loss returns +inf on non-finite SCM output
  (01792620c); regression test locks it. A runtime-only failure codex's static review
  could not see — exactly the case the precision rule exists for.

## Conventions locked during the build
- LES reference artifacts are **self-describing**: each carries both the truth
  profiles AND the exact forcing the LES received, so the SCM bridge reconstructs
  `SCMForcing` from the artifact (guaranteeing the controlled-comparison "same
  forcing" rule, CLAUDE.md critical rule) rather than re-deriving it from the case.
- Reused profile-metric primitives live in `packages/ml/legoesm/training/scm_rce_metrics.py`
  (`weighted_std`, `weighted_rmse`, `safe_sqrt`). NOTE: LES_SUITE.md §2 cites a
  `core/profile_metrics.py` that does **not** exist on this tree; the primitives are
  in `scm_rce_metrics.py`. Score assembly reuses those, no new numerics.
- Sign conventions (match `scm_forcing.py`): `subsidence_w` positive **upward**;
  surface kinematic heat flux `w_th_s` [K m/s] positive **upward** (into the BL);
  `theta_adv` [K/s], `qv_adv` [(kg/kg)/s].

---

## Iteration log (condensed 2026-07-22 — detail in git history + the tracker above)

**Iter 1 (2026-07-21) — CPU library.** Built + codex-CLEAN: `bridge` (self-describing
artifact + forcing/truth extraction), `counter_gradient` (Q1), `score` (D6),
`core/profile_metrics` (extracted shared primitives from `scm_rce_metrics`, avoids
atmosphere→training cycle), `matrix` wiring. Codex round-1 found 6 bridge/score
defects (forcing orientation, dry/moist + prescribe validation gaps, unvalidated
weights, prognostic t0 semantics, silent moist→dry score) — all fixed; round-2 CLEAN.

**Iter 2 (2026-07-21..22) — GPU unblocked, full pipeline + science.**
- `intercomparison` (D7) + gate-0: **Nieuwstadt CBL PASSED** on GPU (σ_w/w_*=0.68,
  well-mixed, surface flux 0.86, entrainment −0.156). Metric fix found by
  instrumentation: entrainment = flux MINIMUM (inversion base), not θ-grad-max height.
- `emit` + `run_les_suite` (dry-CBL emission). Codex found 2 issues → 3 rounds to
  CLEAN: SGS flux must use the core's FACE discretization (bit-exact vs c2f/ddz_c2f/
  f2c), and frame timing (true IC + step-index scheduling). Full pipeline validated
  end-to-end on GPU (surface total 1.01·Q0).
- `scm_coupling` (regrid + θ↔T, added the thermo inverse) + `scm_runner` (SCM forward
  eval; auto-sizes the SCM lid above the LES top or T collapses) + `tune_scm_to_les`
  (D4 derivative-free) + `scorecard` (Q2/Q3). Coupling/tuner codex-CLEAN (orientation
  verified — no upside-down scoring).
- **BUG (instrument-don't-infer):** a diverged SCM scored 0.0 (perfect) because
  `safe_sqrt(NaN)=0`, so the tuner selected blown-up candidates. Fixed → +inf on
  non-finite output; regression test locks it.
- 5 closures wired (local→nonlocal→1.5-order). Perf limit: the tuner recompiles per
  candidate (~8 h for a 5-closure tier-1 CBL ranking on CPU) → the AD-traced-params
  path (D4's other half) is the fix + the AD-vs-DF deliverable.
- Science outputs recorded above (Q1a ceiling, Q2 tuned ranking, Q3 machinery).

**Iter 3 (2026-07-24) — per-flux scorecard + Q2 roster 5→8 closures.**
- **Scorecard reworked to PER-SURFACE-FLUX + anchor dedup** (codex CLEAN, 3 rounds):
  `scorecard.py` `rank_closures_per_flux` (one ranking per Q0 slice) + typed canonical
  flux key `("q0", round(q0,6))` (float32/exact merge; distinct campaign fluxes stay
  separate) + non-finite-`best_loss` rejection + per-closure `n fluxes` denominator
  (⚠ on incomplete matrix). `tune_scm_to_les.py` stamps `artifact`/`sgs`/`q0`
  provenance; default output filename now keys off `artifact.stem` (case_name would
  overwrite across fluxes). `build_les_scorecard.py` `_backfill_provenance` for legacy
  records. Deleted 5 protocol-inconsistent pre-`__lasd__` stale anchor tunings; the
  dedup'd holtslag flux-mean is 0.343 (a prior double-count read 0.326). Q2 ranking is
  flux-robust: holtslag(nonlocal) best, mynn25(1.5-order) worst at every flux 0.02→0.12.
- **Q2 CLOSURE ROSTER 5→8** (`tune_scm_to_les._cbl_scheme_table`): added `tke`
  (1.5-order k-l), `clubb_lite` (higher-order), `edmf` (mass-flux) — the D3 order
  ladder is now local(2)→nonlocal(2)→1.5-order(2)→higher-order/MF(2). **Probe-confirmed
  finite SCM losses** on the 0.06 CBL anchor (nlev=24, dt=10, single eval ~12 min CPU):
  tke 0.266, clubb_lite 0.236, edmf 0.233 — all sensible vs the wired set (holtslag
  0.241, louis 0.269, smag 0.261). clubb_lite/edmf even beat holtslag at default.
  Build test extended to all 8 (each builds + has tier-1 metas). **Full `clubb` (9th)
  DEFERRED**: its tier-1 coefficients live in nested `CLUBBConfig.params` (scheme_key
  `atm.turb.CLUBBParams`), so it needs the descend/re-wrap path (`_tunable_subconfig`
  pattern from `run_scm_rce_campaign.py`), not the flat `config_cls(surface=)` table —
  a focused follow-up.
- **8-closure tuning campaign LAUNCHED** (`run_les_suite_campaign.sh`, closure loop now
  8-wide; idempotent → tunes only the 15 new (closure,flux) combos, ~24 h CPU). On
  completion the per-flux scorecard extends to an 8-closure Q2 ranking across the flux
  sweep. NOTE: single CPU SCM eval is ~11–12 min (recompile-per-candidate); the AD path
  remains the perf fix.

**Iter 7 (2026-07-24) — σ_LES aggregator (D7) in the tuner's loss units.**
- NEW `les_suite/sigma_les.py` (`sigma_les_prognostic`, `SigmaLES`, `SigmaLESError`):
  aggregates a set of same-case SGS-variant artifacts (the `run_les_suite --sgs` spread
  ± 2×-resolution) into σ_LES = RMS over ordered variant PAIRS of
  `score.prognostic_profile_score(truth=A, B-regridded-to-A).combined`. Because it
  reuses the EXACT normalized weighted-RMSE the SCM tuner scores with, σ_LES is DIRECTLY
  comparable to a closure's tuned loss and to a Q1b margin — the D7 significance gate
  ("a margin below σ_LES is not a result") is finally quantifiable. Regrids B→A's
  heights (so 2×-resolution compares to coarse), requires matched output-time schedules,
  rejects <2 / mixed-case / moist inputs (loud errors). Pure assembly (no LES, no SCM,
  no new numerics — reuses `prognostic_truth`/`prognostic_profile_score`/`interp_profile`).
  Exported from `les_suite/__init__`. Tests: `test_sigma_les.py` (7 — identical→0,
  θ-only spread isolates σ_θ, pair count, all error paths).
- WIRED via `scripts/validate/les_suite/compute_sigma_les.py` (loads `{case}__{sgs}.npz`
  variants + prints σ_LES + per-pair distances + the interpretation). Test:
  `test_compute_sigma_les.py` (2). NEXT to fully close D7: emit the 3 SGS variants (GPU)
  + the 2×-resolution runs, run this driver, then gate the Q1b margins / Q3 spread on the
  resulting σ_LES in `build_les_scorecard`.

**Iter 6 (2026-07-24) — `--sgs` selector in `run_les_suite` (D7 σ_LES enabler).**
- `run_les_suite.py` gains `_sgs_les_config(sgs)` + a `--sgs` flag: emit the dry-CBL
  with any of the case's `sgs_variants` — `lasd` (scale-dependent dynamic Smagorinsky),
  `smagorinsky` (static Mason-capped), `vreman` (Vreman 2004). Previously the SGS was
  hardcoded to LASD, so the D7 σ_LES SGS-spread (which needs the SAME case run under
  {lasd, smagorinsky, vreman}) could not be emitted. `--sgs` is validated against the
  case's declared `sgs_variants` (hard error on a stray closure — dispatch hardening),
  drives the artifact's `sgs` tag + filename (`cbl_nieuwstadt__{lasd,smagorinsky,
  vreman}.npz`), and the emit SGS-flux reconstruction reads the same `eddy_viscosity`
  the core integrates so it stays bit-consistent per variant. Tests: `test_sgs_*`,
  `test_build_cbl_selects_sgs_in_config` (builds each config on CPU — no GPU). This is
  the CODE enabler for D7; still to build are (a) the SGS-spread EMISSION (3 GPU
  runs/case) and (b) a **multi-artifact σ_LES aggregator** (the spread of the mean
  profiles across the {lasd,smagorinsky,vreman} artifacts + the 2×-resolution runs) —
  `intercomparison.py` today only checks a single run vs published bands, it does NOT
  compute σ_LES. σ_LES then gates the Q1b margin significance + the Q3 inter-regime
  spread.

**Iter 5 (2026-07-24) — Q1b skill-threshold analysis in the scorecard.**
- Added `local_vs_nonlocal_skill` + `CLOSURE_FAMILY` (D3 families) + `SkillCrossing`
  to `scorecard.py`, rendered as a new **Q1b** scorecard section: per (regime, flux),
  best-tuned LOCAL vs best-tuned NONLOCAL and their margin (local−nonlocal; >0 ⇒
  nonlocal wins). Reuses the deduped per-flux rankings; families not in {local,
  nonlocal} are ignored. Tests: `test_skill_*` (4). This is the **Q1b deliverable**
  (§7 Q1's skill-threshold half; the structural half Q1a was already done).
- **Q1b RESULT (dry-CBL flux sweep, prognostic, 5-closure results):** nonlocal
  (`holtslag_boville`) beats best-tuned local (`smagorinsky`) at EVERY flux 0.02→0.12,
  margins +0.0053, +0.0142, +0.0207, +0.0196, +0.0257 — generally widening with flux
  but NOT strictly monotonic (a slight dip at Q0=0.08). So the local→nonlocal skill
  crossover is below the lowest sampled flux, and the local handicap broadly deepens
  with buoyancy (consistent with the Q1a counter-gradient ceiling). CAVEAT: σ_LES
  significance is a D7 follow-up; the +0.005 margin at 0.02 is small. NOTE: computed
  from the existing per-flux campaign tuned results under gitignored `results/` (the
  8-closure campaign's new-closure tunes are still running); the local/nonlocal
  families are unchanged by the new closures so Q1b is stable.

**Iter 4 (2026-07-24) — Q2 roster 8→9: full `clubb` wired + shared nested-leaf helper.**
- **Extracted the nested-CLUBB descend/re-wrap to a shared public module**
  `atmosphere/physics/turbulence/tunable_subconfig.py` (`tunable_subconfig` /
  `rewrap_tunable_subconfig`). The logic FORMERLY lived, privately, in
  `run_scm_rce_campaign.py` (as `_tunable_subconfig`/`_rewrap_tunable_subconfig`) and
  `train_scm_rce_params.py` imported it as `campaign._tunable_subconfig` — a private
  cross-module import (CLAUDE.md violation). Now a single public source of truth reused
  by the RCE campaign, the RCE AD trainer, AND the LES-suite tuner; no live code
  references the old private names, and the cross-import is gone. Direct unit test
  `test_tunable_subconfig.py`; excluded from the physics-contract gate (config plumbing,
  not a scheme); existing `test_scm_rce_clubb_nesting.py` repointed.
- **Full `clubb` wired into the LES tuner** (`_cbl_scheme_table` → 9 closures): entry
  `"clubb": (CLUBBConfig, "atm.turb.CLUBBParams")`. The tuner's single apply-site was
  factored to a module function `apply_overrides_to_base` that DESCENDS via the shared
  helper before `apply_param_overrides`, so an override lands on the nested
  `CLUBBParams` (flat schemes: the helper is identity) — directly unit-tested
  (`test_clubb_override_descends_into_nested_params`, `test_apply_overrides_flat_scheme`).
  `CLUBBConfig(surface=)` constructs the base like the flat schemes; build test → 9
  schemes. **Finite-SCM-loss probe CONFIRMED**: full CLUBB runs to loss **0.2349**
  (finite) on the 0.06 CBL anchor — sensible vs the wired set (clubb_lite 0.236, edmf
  0.233, holtslag 0.241). CAVEAT (controlled-comparison): this probe is at **nlev=8,
  dt=30** (a coarse grid), NOT the campaign's nlev=24 — CLUBB's full nlev=24 XLA
  compile OOM'd 3× on the memory-contended shared node (other users' 50–192 GB jobs),
  so the 0.235 is a viability smoke test, NOT a scorecard-comparable number. Full
  CLUBB tuning into the per-flux scorecard therefore stays a **separate step pending
  memory headroom** (CLUBB's higher-order compile is far heavier than the flat
  schemes'); `clubb` is deliberately NOT yet in the running 8-closure campaign shell
  so its nlev=24 compile can't OOM-crash that campaign.
- The D3 order ladder is now COMPLETE: local(2)→nonlocal(2)→1.5-order(2)→
  higher-order/MF(2)→full-higher-order(clubb). Q2 nine-closure roster wired.

**NEXT:** (1) AD path — traced params via `apply_param_overrides` inside a jitted/AD
loss (one compile serves all candidates; fixes the per-candidate recompile + gives the
D4 AD-vs-DF comparison; needs a scan-based differentiable SCM segment). (2) Wire the
stable/sheared/moist regime emission in `run_les_suite` (only dry-convective CBL today)
→ the §6 flux×shear grid + BOMEX/DYCOMS (unlocks Q3 inter-regime spread). (3) D7 σ_LES
SGS-spread + 2×-resolution runs → the error bars.
