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
  tuning — the tuning-independent ceiling on local closures. The full Q1 threshold
  needs this across the w'θ'ₛ grid (the unwired dry-grid emission).

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

## Iteration log

### Iter 1 (2026-07-21)
- Read LES_SUITE.md + existing registry/catalog/tests (all passing, 23 tests).
- Confirmed registry + catalog complete and merged (commit 49959ec94).
- Created this CHANGELOG.
- **`bridge.py` (22 tests)**: `LESReferenceArtifact` self-describing schema
  (truth profiles + applied forcing), `artifact_to_scm_forcing`,
  `diagnostic_truth`/`prognostic_truth`, `total_turbulent_flux`. Orientation
  contract documented (artifact surface-first; SCM top-to-bottom; flip deferred to
  the SCM-coupling layer, not silently buried in the bridge).
- **`counter_gradient.py` (10 tests)**: Q1 structural diagnostic —
  `flux·dθ/dz > gate` over a contiguous layer ⇒ any K≥0 down-gradient closure must
  fail. `centered_dtheta_dz` (non-uniform grid), `counter_gradient_diagnostic`,
  `diagnose_truth`. Single-cell blips rejected (min-layer guard).
- **`score.py` (11 tests)**: D6 diagnostic (`diagnostic_flux_score`) + prognostic
  (`prognostic_profile_score`) assemblies, std-normalized mass-weighted RMSE,
  AD-safe (finite grad at perfect fit). No new numerics — reuses core primitives.
- **`core/profile_metrics.py` (8 tests)**: extracted `safe_sqrt`/`weighted_std`/
  `weighted_rmse` from `scm_rce_metrics.py` (now re-exports them) into the shared
  low-level home the design doc names, + `layer_weights_from_heights`. Avoids the
  atmosphere→training circular import and the "no dup numerics" doctrine violation.
  `test_scm_rce_metrics.py` still green after the re-point.
- Fixed a series-path NaN gradient in `score._normalized_rmse_series` (single
  `safe_sqrt` over the time-mean-square, no intermediate bare sqrt).
- Pre-existing unrelated failure noted: `test_no_private_cross_imports` red on the
  branch baseline (fv3/grids/ocean private imports) — NOT caused by this work; out
  of scope for the LES suite.
- Guardrails run: `test_no_hardcoded_constants`, `test_federation_plan` green;
  les_suite modules are outside `*/physics/*` so the inline-coeff/physics-contract
  gates do not apply (floors/gates are numerics one-offs, documented in-line).
- **`matrix.py` + `run_les_suite_matrix.py` (16 tests)**: `_les_suite_matrix_spec()`
  (routes through `core.setup_selector.MatrixRunnerSpec`, `valid_grids` = registry
  resolution labels), `build_test_matrix()` (every (name,grid) real),
  `select_cases()` (exact/substring/grid filters; empty exact/grid selection →
  `SystemExit` = dispatch hardening). Grid axis = LES resolution label (`96x96x96`);
  core locked to spectral (D1), not a selection axis.
- **npz artifact I/O** added to `bridge.py` (`save_artifact`/`load_artifact`, 2
  round-trip tests) — self-describing `.npz` the LES driver writes and the tuner
  reads; scalars/strings ride a JSON `__meta__` key, optional None channels omitted.

- **Codex adversarial review round 1** (mandatory, CLAUDE.md) ran on bridge/
  counter_gradient/score/core.profile_metrics. Verdict: ISSUES FOUND (6). All fixed
  same iteration + non-vacuous tests added:
  1. `artifact_to_scm_forcing` returned surface-first profiles to a top-to-bottom
     SCM (latent reversal bug). FIX: `scm_top_to_bottom=True` default reverses every
     vertical forcing profile; surface time-series left unreversed. Same-grid
     assumption documented; cross-grid regrid still deferred to coupling layer.
  2. Dry artifact could carry silently-discarded moisture fields. FIX: `validate()`
     forbids wqt_resolved/wqt_sgs/qv_adv/w_qv_s when qt is None; wqt_sgs requires
     wqt_resolved.
  3. `prescribe` mode allowed stored-but-ignored surface fields (broke the
     self-describing/exact-forcing contract). FIX: each mode requires exactly its
     field(s) and forbids the others.
  4. Explicit score weights unvalidated (a x100 or negative weight silently
     corrupted the metric; negatives could report a false perfect). FIX:
     `_resolve_weights` rejects negative + non-unit-sum weights.
  5. `prognostic_truth(t0_s)` used nearest-sample argmin, not "at or after". FIX:
     `searchsorted(side='left')`; raises if t0 past the last sample.
  6. Moist score silently became a dry score when the caller omitted the SCM
     moisture output (an incomplete run could out-score a complete one). FIX: raise
     when moist truth lacks the matching SCM moisture arg.
  Plus a mathematically-wrong docstring comment in counter_gradient.py corrected.
- **Codex round 2: VERDICT CLEAN.** All 6 round-1 fixes verified correct + complete;
  matrix.py selection/dispatch-hardening verified across all 16 cases; no new
  defects. The mandatory iterate-with-codex loop converged in 2 iterations.
- Full suite at iter-1 close: 109 les_suite/core/scm-rce tests green; ruff clean.
- **Committed** as `feat(les-suite): LES→SCM bridge, Q1 counter-gradient, D6 score,
  matrix wiring` (branch les-suite-optimization).

### Iter 2 (2026-07-21) — GPU unblocked
- **GPU became available** (2× Tesla V100S-32GB, JAX backend=gpu). This unblocks the
  D10 gate-0 and the LES ensemble (§8 step 3 was CPU-blocked).
- **`intercomparison.py` (10 tests) + CLI**: the D7 buoyant-path credibility gate —
  `cbl_diagnostics` extracts the universal dry-CBL convective scaling (z_i, w_*, peak
  σ_w/w_*, mixed-layer ∂θ/∂z, surface + entrainment ⟨w'θ'⟩/Q0), banded against the
  Nieuwstadt-1993 + entrainment-ratio envelope; `gate_from_cbl_profiles`,
  `evaluate_cbl_gate`; CLI exits non-zero on any band failure. Committed de7672463.
  (`g` from `legoesm.constants` — the local banned-literal hook caught a hardcoded
  9.80616 in the test and forced `constants.g`.)
- **Gate-0 launched** on GPU: `run_spectral_cbl.py` 96³ x64 (D8 production res),
  Q0=0.06, z_i0=800 m, 2 h (~10 t*), LASD SGS, adaptive CFL. A 48³ smoke test first
  confirmed the buoyant path develops correctly (σ_w/w_*=0.56, surface flux ratio
  0.89, well-mixing). Verdict recorded once the production run + validator finish.
- Codex review of `intercomparison.py` running in parallel with the LES run.

- **Gate-0 PASS** (validator on the 96³ x64 run): σ_w/w_*=0.679, mixed-layer
  ∂θ/∂z=0.070 mK/m, surface flux 0.861, entrainment -0.156 — all four bands. The
  buoyant path of the spectral truth core is validated against Nieuwstadt-1993; the
  §3 named risk (buoyant path least-validated) is retired. Fixture: JSON committed.
- **`emit.py` (12 tests) + `run_les_suite.py` (3 dispatch tests)**: the ensemble
  driver. emit does the horizontal-mean + resolved/SGS flux reductions (SGS flux =
  -<(ν_t/Pr)∂θ/∂z> from the closure's `eddy_viscosity`); SGS-flux SIGN convention
  gets explicit analytic tests. Driver: registry case → spectral LES → validated
  artifact. Dry CBL wired; stable/moist regimes raise (honest dispatch).
- **SGS-flux model-consistency CONFIRMED** (read the core, not inferred):
  `spectral_les_plane.rhs` sets `nu_t = eddy_viscosity(...)` (line 1116) and
  `scalar_rhs` uses `Kh = nu_t/pr_sgs` — identical to emit. The `sgs_buoyancy` Lilly
  rescale (1126-7) is skipped (driver pins `sgs_buoyancy=False`), so the emitted SGS
  flux equals the flux the model integrated. Driver comments the invariant.
- **Full pipeline VALIDATED end-to-end on GPU** (96³ CBL, 0.35 h f32): emit →
  `load_artifact` → `diagnostic_truth` → surface flux ratio resolved 0.999 + SGS
  0.011 = **1.010** (~1, physically correct); the Q1 `diagnose_truth` counter-gradient
  diagnostic **detects a structural-ceiling layer at 675-758 m** — a real Q1a result.
- 132 les_suite/core tests green; counter-gradient gradient tests made
  precision-robust (x32/x64).
- **Codex review of emit/run_les_suite: ISSUES FOUND (2), both fixed** (798434caf):
  (1) emit's SGS flux used a cell-centred `K_h·gradient`, not the core's FACE
  discretization (`c2f(K_h)·ddz_c2f(θ)`, surface face = Q0, lid = 0) — a plausible
  approximation, wrong at the surface. Rewrote to replicate `scalar_rhs` exactly;
  a new test asserts bit-agreement (atol 1e-10) with the core's own c2f/ddz_c2f/f2c
  ops. (2) the driver stepped once then labelled it t=0 (dropped the true IC) and
  could drop the final frame — now records the real t=0 IC before stepping and
  drives every frame target through T; added --frames/--hours/--dt validation.
  Re-run GPU round-trip: true IC at t=0 (resolved flux exactly 0), surface total
  1.04·Q0, k=1 = 1.01·Q0.
  (Codex invocation note: the positional-arg form hung on stdin this session; the
  working form is `codex exec [flags] < prompt.md` — feed the prompt via stdin.)
- **Codex round 2**: Fix 1 (SGS face flux) CONFIRMED bit-exact (maxerr=0.0 vs the
  core's own operators); true-IC recording correct. One remaining issue: the frame
  loop still dropped/overshot frames when dt was large vs the frame spacing. FIXED
  (f70a7b46a): step-index sampling via the tested pure helper `frame_step_schedule`
  — strictly-increasing distinct times, final at n_steps·dt≈T, dt≥T guarded; 3
  regression tests cover codex's failing cases. GPU re-run: [0,252,504,756,1008,1260].
- **Codex round 3: VERDICT CLEAN** — frame_step_schedule verified over exhaustive
  n_steps=1..500 × frames=2..700 (distinct, final=n_steps, no silent shorten); driver
  records len(schedule)+1 frames, times strictly increasing, no off-by-one; the
  removed dedup/sel logic is safe (duplicates no longer constructible). The
  iterate-with-codex loop on emission converged (2→1→0 issues).
- Committed: intercomparison gate (de7672463), gate-0 fix (8d1bdc81b), emit +
  run_les_suite (218129efd).

- **`scm_coupling.py` (12 tests)**: the tuner's grid bridge — `interp_profile`
  (linear height regrid, surface-first, clamps out-of-range), `regrid_truth` (LES
  truth onto a fixed eval grid held identical across closures), and the θ↔T pair
  (added the missing `potential_temperature_from_temperature` inverse to
  `thermodynamics.py`, reusing the canonical Exner constants). Physics guardrails
  green (3807). This de-risks the hardest tuner sub-problem (SCM sigma/T ↔ LES
  height/θ). Committed 0e9095433.

**NEXT:** `tune_scm_to_les.py` composes coupling + SCM run + score: init the SCM from
LES(t=0) (θ→T via `T_from_theta` on the SCM pressure; `interp_profile` LES→SCM
heights), drive it with `artifact_to_scm_forcing`, score with `regrid_truth` +
`score.*`, then tune each closure's `__param_spec__` via AD (`eqx.filter_value_and_grad`
+ `param_collector.apply_param_overrides` in the loss) AND derivative-free (D4). Then
`build_les_scorecard.py`, the full GPU ensemble, and the Q1/Q2/Q3 answers.

--- earlier NEXT (superseded) ---
`tune_scm_to_les.py` — the per-(closure,
regime) AD + derivative-free tuner (D4). Reuse (do NOT fork): `scm.py` +
`scm_forcing.py` (single-column integration, CPU-cheap), `training/param_collector`
(`build_trainable_params`/`apply_param_overrides`), `ml/training.create_optimizer`,
and the RCE-tuning precedents `scripts/run/{run_scm_rce_params,train_scm_rce_params,
run_scm_rce_campaign}.py`. It consumes cached LES artifacts (`bridge.load_artifact`)
→ `artifact_to_scm_forcing` + `diagnostic_truth`/`prognostic_truth` → `score.*`.
Then the two validators (`build_les_scorecard.py`, `compare_les_intercomparison.py`)
and, GPU-gated, `run_les_suite.py` + gate-0. Science answers (Q1/Q2/Q3) stay blocked
until a GPU allocation runs the LES ensemble (§8 step 3).
