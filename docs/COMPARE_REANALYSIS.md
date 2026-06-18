# Compare-to-Reanalysis + LES-Informed Column Correction

**Branch:** `feat/compare-reanalysis`
**Status (iter 40, compressed):** The whole offline loop is built, tested + Codex-reviewed, and **structurally complete** end-to-end — every stage `run (AMIP/CMIP) → time-mean → compare to ERA5 → rank → cluster → LES-diagnose → feedback → inject (turbulence_override) → RE-RUN` is real and tested (capstone iter 37; the injection provably changes the simulation, iter 35; feedback↔physics column ordering locked, iter 36). A **runnable HPC entry point** exists — `scripts/run/run_correction_campaign.py` (`make_base_driver_builder` for AMIP **and** CMIP, real-ERA5 reference loading, per-column `clubb_lite.C_K` correction, per-round bias report). The per-iteration table below records each stage's module + tests (compressed at iters 10/20/30/40; full detail in git log). **Remaining (the done-criterion):** ONLY the EMPIRICAL demonstration — execute the multi-day campaign on real ERA5 at HPC scale and observe the LES-informed C_K *lower* the bias (not unit-testable here, not honestly fakeable; the result is genuinely unknown until run). Breadth follow-ups: distributed/MPI feedback sharding, cubed-sphere geostrophic rotation, Gaussian/Voronoi extractors.
**Date:** 2026-06-15 (design); 2026-06-17 (impl began)
**Scope:** Atmosphere component only. ERA5 reanalysis only. **Not** supervised learning.

> ## Implementation status (compressed at iter 10; full history in git log `feat/compare-reanalysis`)
>
> The pipeline is built bottom-up as tested, Codex-reviewed components. Done so far:
>
> | Stage / gap | Module (all tested + Codex-clean) | Iter |
> |---|---|---|
> | 2 / #1 per-column ERA5 metric | `training/column_era5_metrics.py` (`score_columns`, `rank_worst_columns`; +`safe_sqrt` in `scm_rce_metrics`) | 1 |
> | 3 / #2 worst-column manifest + env tags | `training/column_manifest.py` (`build_worst_column_manifest`, `compute_column_environment` SST/CAPE/shear, JSON I/O) | 2 |
> | 2 orchestration | `training/compare_reanalysis.py` (`compare_state_to_reference`; ERA5→model regrid decision) | 3 |
> | 2 driver | `scripts/validate/compare_amip_era5.py` (real `main`: load restart+ERA5→regrid→compare→manifest) | 4 |
> | 4 / #3 GCM-col→SCMForcing (assembly) | `atmosphere/column_forcing.py` (`build_column_scm_forcing`, ω→w, Coriolis) | 5 |
> | 5 / #5 LES resolved fluxes | `atmosphere/dynamics/rce_diagnostics.py::resolved_turbulent_fluxes_plane` (w'θ'/w'q'/w'u'/w'v'/w'θ_v') | 6 |
> | 6 / #6 closure-coeff diagnosis | `atmosphere/dynamics/les_closure_diagnosis.py` (eddy K, mixing length, **entrainment w_e**) | 7 |
> | 7 / #7 parameter-field assembly | `training/parameter_field.py` (`scatter_column_field` static; `environment_kernel_field` N-W regression) | 8 |
> | 7 / #7 feedback application | `training/feedback.py` (`build_parameter_field` dispatch; `apply_column_parameter_field` column-promotion-gated splice, traced-in-loss) | 9 |
>
> **Conventions held every iter:** pre-impl search + reuse (no re-derived numerics);
> every new `.py` gets a direct unit test (analytic where possible); AD-safe
> (double-where masked divisions, gradient-safe sqrt); dispatch raises on unknown;
> mandatory Codex adversarial-review loop to clean before commit.
>
> | 4 grid-side extractor (lat-lon) | `column_large_scale_extract.py` (`extract_column_forcing_latlon`: ω from continuity, −V·∇θ/−V·∇q → `ColumnLargeScaleState`) | 10 |
> | 4 1.5-order TKE SGS closure | `tke_sgs_plane.py` (Deardorff/Lilly ν_t=C_kℓ√e, ε, Pr_t; eqm↔Smagorinsky) | 11 |
> | 5 LES regime selection | `les_regime.py` (CAPE→shallow/deep + per-regime resolution; raise on unknown) | 12 |
> | 5 LES vertical mapping | `les_vertical_mapping.py` (interp + `build_top_relaxation` reusing `sponge_profile`) | 13 |
> | 6 LES→coefficient | `column_les_diagnosis.py` (`diagnose_column_coefficient`: eddy-K/entrainment-w_e; top-down→ascending reversal) | 14 |
> | 4–6 column-LES driver | `scripts/run/run_column_les.py` (`build_column_les_setup`→`run_forced_les`→diagnose; `process_column`/`extract_gcm_column`) | 15 |
> | 7 diagnoses→feedback field | `feedback_assembly.py` (`reduce_column_diagnosis`, `assemble_feedback_field`) | 16 |
> | verify bias↓ | `bias_metrics.py` (`aggregate_combined_bias`, `bias_improvement`, `worst_column_bias_change`) | 17 |
> | 7 per-scheme promotion | `promotable_params.py` (`apply_feedback_to_scheme`; gray `tau_equator`/`tau_pole`) | 18 |
> | CLOSED-LOOP orchestrator | `correction_loop.py` (`run_correction_iteration`: compare→diagnose→assemble→apply→re-compare→bias; heavy steps injected) | 19 |
> | real-dycore integration + moist-IC fix | `test_run_column_les.py` runs the real plane LES; fixed `run_forced_les` moisture seeding (`q_v_init`) | 20 |
> | real-comparison loop adapter | `correction_loop.make_compare_fn` (wraps the real `compare_state_to_reference`) | 21 |
> | AMIP+CMIP bridge | `compare_reanalysis.column_state_from_hydrostatic` (HydrostaticState→ColumnState, unwraps Field; SST prescribed/coupled) | 22 |
> | real coupled-run (CMIP)→compare | `test_cmip_compare_integration.py` (real `CoupledESMDriver`, C8/L5) | 23 |
> | physically-coherent promotion | `_shared.broadcast_column_param` + `clubb_lite` C_K wrapped (registered `clubb_lite_C_K`) | 24 |
> | iterative multi-round campaign | `correction_loop.run_correction_campaign` (accumulates the feedback field across rounds) | 25 |
> | FULL mock-free end-to-end gate | `test_correction_e2e_integration.py` (real coupled→compare→real LES→K→feedback→`clubb_lite_C_K`; non-vacuous valid-K) | 26 |
> | cubed-sphere extractor | `extract_column_forcing_cubed_sphere` + `extract_column_forcing` dispatcher (4D-native ops, no `vmap(pad_halo)`, raise on unknown grid) | 27 |
> | cubed wired into LES spin-off | `extract_gcm_column` grid-agnostic gather `arr[tuple(col_index)]`; cubed worst column spins off its LES | 28 |
> | geostrophic wind (lat-lon) + LES Coriolis | `geostrophic_wind_from_gradients` (two-term σ-PGF, sign-correct both hemispheres) → `u_geo0/v_geo0`; cubed/equator→`f×V` | 29 |
> | time-mean state accumulator | `column_state_accumulator.py` (scan-friendly, AD-safe `ColumnState` climatology mean feeding `compare_state_to_reference`) | 30 |
> | run→time-mean→compare wiring | `run_to_column_mean.py` (sample column state per `diag_days` segment via `segment_callback` → time mean; `amip`/`cmip_column_state`, AMIP threads segment `day`). `CoupledESMDriver.run` gained optional `segment_callback` + public `q_v` | 31 |
> | real `compare_fn(config)` | `make_run_fn(build_driver, extract)` → `run_fn(config)` (config→fresh driver→time-mean) ∘ `make_compare_fn` → a real `compare_fn` (run+mean+score); dominance-margin worst-column test; last compare-path mock retired | 32 |
> | env-clustering (LES cost) | `column_clustering.cluster_columns_by_environment` — deterministic farthest-first (k-center) over normalised SST/CAPE/shear, anchored at the worst column → K representative REAL columns + labels; finiteness/scale validation, duplicate-env early stop | 33 |
> | clustering WIRED into the loop | `run_correction_iteration`/`campaign` `les_budget=K` (+`env_scales`): diagnose only K env-reps, map each to its cluster members → `n_diagnosed=K` (new field) LES, all corrected; None=diagnose-all | 34 |
> | **CONFIG-INJECTION** (param-update mechanism) | `ExperimentConfig.turbulence_override` + shared `turbulence_config_for` (all backends FV/MPAS/spectral) → corrected `clubb_lite` config (incl. per-column C_K array) reaches the kernel; validate_strict scheme+isinstance, serialization drops it. **PROVEN**: real runs with different C_K → different T; per-column array runs a real step | 35 |
> | column-ordering CONTRACT | `test_feedback_column_ordering.py` locks feedback-flat order == physics `ColumnAdapter` flatten (REAL pipeline + REAL driver, lat-lon + cubed, elementwise) — per-column C_K lands on the right cell. Residual: MPI rank-local sharding | 36 |
> | **CAPSTONE: full loop + REAL re-run** | `test_correction_loop_real_rerun.py` — real coupled→time-mean→compare→worst col→[mock diagnose]→feedback→`apply_feedback_to_scheme`→`run_correction_iteration` RE-RUNS the real model with the C_K via `turbulence_override`. NON-VACUOUS: re-run driver's `turbulence_config.C_K` == the (128,) array. Only the empirical bias-sign (real ERA5@HPC) unasserted | 37 |
> | **HPC CAMPAIGN DRIVER** | `scripts/run/run_correction_campaign.py` — `make_clubb_build_driver` + `make_les_diagnose_fn`(→`process_column`) + `build_correction_campaign` (wires run∘compare∘diagnose→`run_correction_campaign`); `main()` CLI loads config+ERA5 ref, grid/sigma from the driver. Smoke tests (mock driver+LES→real `process_column`) | 38 |
> | LES library → package | column-LES orchestration moved verbatim from the script to `legoesm.atmosphere.dynamics.column_les` (source-under-packages); script keeps only the CLI (no re-export shim). Faithful (lint==baseline, federation+e2e green) | 39 |
> | CMIP CLI (**AMIP and CMIP**) | `run_correction_campaign.make_base_driver_builder(mode)` → (build_base_driver, extractor): `amip`=ModelDriver+prescribed SST, `cmip`=CoupledESMDriver+coupled SST (`ocean_grid=None`→same-grid, ocean on atm grid). `main()` `--mode`/`--coupled-preset` (guarded). Dispatch-hardened; real AMIP + CMIP build tests | 40 |
> | restartable campaign (§1) | `run_correction_campaign` gained `initial_field`/`start_round`/`checkpoint_callback` (resume from the accumulated field; persist {round,C_K,field} per round) + `build_correction_campaign` passthrough + CLI `--checkpoint`/`--resume` — a multi-day HPC campaign survives a job timeout. Tests: checkpoint fires per round (start_round offset), **resume == uninterrupted** (split 2-round == straight 2-round: final C_K/field/round-1 config match), zero-iter resume returns the resume base, initial_field-forwarded | 41 |
> | **foundations for SIMULTANEOUS multi-coefficient correction** (PR1+PR2 of a sequenced plan) | Codex design-reviewed the simultaneous C_K+Pr_t correction and recommended SPLITTING it (isolate the behavior-preserving refactor from the new multi loop); landed the two foundations here, multi loop (run_multi_correction_*) sequenced next. **PR1 — factored line search:** extracted `_run_line_search(compare_fn, baseline, fractions, make_candidate)` from `run_correction_iteration` (the backtracking gate-candidate loop) so single + future multi share it with no duplicated numerics; the `frac==1.0` short-circuit / pre-clamp / post-blend-clamp / dtype-matched base stay INSIDE the caller's `make_candidate` closure → byte-identical single-coefficient path (39 loop tests pass unchanged + a direct helper test). **PR2 — LES-once-diagnose-many:** `run_column_les_pipeline` gains `methods=[...]` (run the dominant-cost LES ONCE, diagnose every method on the same final state → `{method: diagnosis}`); `ColumnLESConfig.diagnosis_methods` + per-method `clubb_l_mix_max` validation (fires when clubb_coefficient is one of several); `process_column` threads it; scalar `method=` path unchanged. **Codex review:** PR1 byte-identical confirmed; PR2 bug found+fixed — `build_correction_campaign` now REJECTS a `diagnosis_methods` config up front (would otherwise return a dict and crash the single-coefficient reduce; the multi campaign is the deferred PR4). Tests: helper largest-improving/fallback, one-LES-run-many-diagnose + per-method l_mix_max guard + empty-list raise, process_column multi dict, campaign multi-methods rejection. Lint == HEAD | 48 |
> | **second coefficient: turbulent Prandtl number Pr_t = K_m/K_h** (multi-param) | Extends the correction from one coefficient (C_K) to a second — the done-criterion's "parameters" (plural). Pr_t is a DIMENSIONLESS ratio of the LES momentum (K_m) and heat (K_h) diffusivities, so (unlike C_K) it carries NO wp2-identification assumption (the ratio cancels the shared ℓ·√wp2). New leaf `prandtl_number_from_diffusivities` (AD-safe masked division, valid where both diffusivities valid + K_h>floor) + `diagnose_prandtl_number` (K_h reuses `diagnose_eddy_diffusivity`, K_m the iter-46 momentum inversion, co-located; min-valid guard). One-line body change `Kh_full = Km_full/broadcast_column_param(config.Pr_t, l_mix)` makes Pr_t per-column-safe (scalar path byte-identical — tested). New reduce `"prandtl_number"`, `coefficient_value`, dispatch; registered `clubb_lite_Pr_t` promotion (bounds (0.3,1.5) → iter-45 clamp works). `build_correction_campaign` now MAPS the diagnosis method → (promotion_key, background): prandtl_number→clubb_lite_Pr_t, else clubb_lite_C_K; CLI `--diagnosis-method prandtl_number` + generic checkpoint/output field. **Codex-reviewed** (physics/sign/co-location/broadcast clean; 2 CLI-checkpoint bugs found+fixed: hardcoded "C_K" key → real `corrected_field` name, + a resume-method-mismatch guard). Tests: analytic Pr_t=K_m/K_h, low-K_h/both-valid masks, AD-safety, manual-composition wiring, body per-column-Kh + uniform==scalar byte-identical, reduce, promotion+clamp, dispatch, e2e campaign in-bounds per-column Pr_t. Full sweep 189 passed | 47 |
> | **dimensionally-correct dimensionless C_K diagnosis** (units-bug fix) | The loop injected a DIMENSIONAL eddy diffusivity `K [m²/s]` (~1–100) as the DIMENSIONLESS GCM `C_K` (~0.1–1.2) — a unit mismatch the iter-45 clamp only masked. New `diagnose_clubb_coefficient` returns the ACTUAL GCM parameter: `C_K = K_m/(ℓ·√wp2)`, the exact inverse of the GCM closure `Km = C_K·ℓ·√wp2`. `K_m` = shear-projected down-gradient MOMENTUM diffusivity `−(F·S)/|S|²` from the resolved `⟨w'u'⟩,⟨w'v'⟩` (momentum, not heat → no `Pr_t` conflation); `ℓ` = the SAME shared Blackadar `mixing_length(z, l_mix_max)` the GCM uses; `√wp2` from the resolved `w'²`. New leaf inversions `momentum_diffusivity_from_fluxes` + `clubb_coefficient_from_diffusivity` (double-where AD-safe at the |S|² and √wp2/division sites); all 7 arrays reversed top-down→ascending + co-located; `<min_valid_levels` (3) ⇒ column invalid. New reduce method `"clubb_coefficient"` (factored `_valid_profile_mean`), `coefficient_value` + `ColumnLESConfig.clubb_l_mix_max` + dispatch (raises w/o `l_mix_max`); `build_correction_campaign` keeps reduce==LES method + auto-populates `l_mix_max`; CLI `--diagnosis-method` defaults to the units-correct `clubb_coefficient`. **Design codex-reviewed** (physics/sign/AD/co-location validated, corrections applied). Tests: analytic `K_m`/`C_K` exactness, manual-composition wiring equality, AD-safety grads, counter-gradient/low-wp2/zero-shear invalid, dispatch hardening, min-valid guard, e2e campaign produces valid in-bounds per-column `C_K`. KNOWN LIMITATION (design): `C_K` uses LES `w'²` while the GCM evaluates its OWN prognostic `wp2` — an O(1) offset possible in deep convection; the gate/line-search/bias-monitor are safeguards, and co-tuning `C_eps` to align the budgets is a follow-up. (Codex implementation-level pass deferred one iteration — quota.) | 46 |
> | **physical-bounds guard on the diagnosed coefficient** | Two complementary guards so a degenerate/blown-up LES cannot inject an unphysical, destabilizing `C_K`. (1) `reduce_column_diagnosis` now marks a **non-finite** reduced coefficient INVALID (keeps background) — always on, backward-compatible (finite diagnoses unchanged). (2) `clip_to_bounds` clamps the diagnosed field to the coefficient's registered `__param_spec__` bounds (`C_K ∈ (0.1, 1.2)`), reusing the param-hygiene system: new `feedback.param_field_bounds` (resolves `(lo,hi)` via the existing spec helpers) + `promotable_params.clip_field_to_promotable_bounds` (raises on unknown key). Clamp is applied to `raw_field` BEFORE the line search (distinct in-range sub-steps) AND **post-blend** (Codex: the invariant — bounds the result even when `base`/`initial_field` is itself out of range), and on the no-op round (so the accumulated base stays in-range). Library default off (compat); `build_correction_campaign` default ON + CLI `--allow-unphysical-coeff` opt-out. Tests: non-finite→invalid (eddy+entrainment+end-to-end background), clamp clamps C_K + gray tau generically + unknown-key raises + no-bounds passthrough, iteration clamps 5.0→1.2 + keeps line-search sub-steps in range + out-of-range-base post-blend clamp + no-op clamp, campaign never injects unphysical. Codex re-review: CLEAN (all Q1–Q5 clear, Q6 no-op gap fixed) | 45 |
> | **line search over correction magnitude** (robust to the LES↔GCM overshoot) | `run_correction_iteration` gained `step_fractions` (+ `CorrectionResult.step_fraction`): the LES-diagnosed `C_K` is a *different-model* estimate that can overshoot the bias-minimizing GCM value, so the round backtracks the injected field `base + s·(raw − base)` over descending `s`, keeping the LARGEST `s` whose re-run lowers the global bias (Armijo-style, ≤1 re-run/fraction; stops at first improving). None improve ⇒ largest `s` reported with `improved=False` so the iter-43 gate rejects it. Generalizes the gate (gate = degenerate `(1.0,)`); composes with it (line search finds an improving sub-step → gate accepts). `None` default = full single step (byte-identical to prior). Wired through `run_correction_campaign`/`build_correction_campaign` + CLI `--step-fractions`. **Codex-found BUG fixed:** `environment_kernel_field` excluded `background` from its result_type → an accumulated float64 base was downcast at no-neighbor columns, drifting untouched columns under a partial step; now `background` is in the dtype + `base_field` built at `raw_field.dtype` (blend provably identity at untouched cols, both strategies). Tests: validation, largest-improving-step (full overshoots→rejected, half kept, **logged call-order proves full tried first**), full-step-when-improving, none-improve→full+reject, campaign **line search rescues a round the bare gate would reject**, float64-background no-downcast + scalar-bg neutrality. Codex re-review: CLEAN | 44 |
> | **monotonic acceptance gate** (the "improve the biases" done-criterion) | `run_correction_campaign` gained `accept_only_if_improved` (+ `CampaignResult.accepted`): a round is KEPT only if its correction lowered the area-weighted global bias (`result.bias.improved`), else REJECTED — config + field discarded, next round restarts from the prior accepted (best-so-far) base, so the accumulated `clubb_lite.C_K` field NEVER regresses; no-op rounds vacuously accepted. `final_field` unified to derive from the last-accepted `base`. `build_correction_campaign` defaults the gate ON (production) + CLI `--keep-worsening-rounds` opt-out; library default OFF (compat). **Codex-found HIGH bug fixed:** the CLI checkpoint persisted `res.updated_config.C_K` (the REJECTED config) beside the accepted `field` → resume desync; now both derive from the accepted `field` (single source of truth; config rebuilt from `field.reshape(-1)` on resume). Tests: gate rejects worsening / keeps improving / default-off keeps worsening / no-op accepted / 2-round monotonic no-regression / **rejected-round checkpoint field ≠ discarded updated_config** (locks the desync fix) / build-campaign default-gate rejects non-improving mock. Codex re-review: all fixes CONFIRMED | 43 |
> | **env-kernel generalization wired through the loop** (§6) | `assemble_feedback_field` gained `strategy="static"\|"environment"` (env = N-W-regress diagnosed C_K onto SST/CAPE/shear over the FULL grid via `environment_kernel_field`, so K LES on cluster reps generalize to all env-similar columns) + `column_environment_grid(model, sigma, *, env_config, p_full, p_half)` (full-grid predictors via the SAME `compute_column_environment` the manifest used — Codex: parameterized so `grid_env`≡`sample_env` under custom env_config/hybrid pressure). Wired `feedback_strategy`/`env_grid_fn` through `run_correction_iteration`/`run_correction_campaign`/`build_correction_campaign` + CLI `--feedback-strategy`. **Codex-found real bug fixed:** `environment_kernel_field` now accepts an ARRAY background (accumulated round-k field, scalar OR grid-shaped/flat, flattened C-order + size-checked, symmetric with `scatter_column_field`) — multi-round env accumulation would otherwise crash on `(ncol,)` vs `grid_shape` broadcast. Tests: env generalizes to non-worst similar columns, requires `grid_env`/`env_grid_fn`, unknown-strategy raises, per-column array-bg fallback + C-order ordering + size-mismatch raise, **2-round campaign accumulation** (round-0 env-A C_K persists into the round-1 env-B field). Codex re-review: PASS, no confirmed bugs | 42 |
>
> **Feedback loop closes in code:** LES diagnoses → `assemble_feedback_field` (16)
> → `apply_column_parameter_field` (9) → updated scheme config; iterated by
> `correction_loop` (19/25). Both lat-lon and native cubed-sphere worst columns
> spin off real LES; forcing = subsidence + advection + (lat-lon) geostrophic.
>
> **Remaining to reach the done-criterion:**
> - **Empirical bias-reduction demo** (the actual success criterion) — needs an
>   HPC-scale run: real `compare_fn` (AMIP + `compare_amip_era5`) and `diagnose_fn`
>   (`run_column_les.process_column`) on real ERA5 over a multi-day run + many
>   LES. Orchestrator (19/25) ready + mock-tested; not unit-testable here. NB a
>   self-consistent perfect-model OSSE cannot honestly prove it (the LES and the
>   GCM closure are different models — that gap is the method's whole point).
> - **AMIP/CMIP run → time-mean → compare → score**: DONE (iter 31–32,
>   `run_to_column_mean` + `make_run_fn` composed with `make_compare_fn` → a real
>   `compare_fn(config)`, integration-tested on a real coupled run). The ONLY
>   remaining piece for the HPC demo is a production `build_driver(scheme_config)`
>   that injects the corrected per-column scheme config (e.g. the `clubb_lite_C_K`
>   field) into the dycore/physics — the config→model wiring is intentionally
>   injected so this stays generic; that wiring + real ERA5 + the multi-day run is
>   the HPC-scale bias-reduction demo (not unit-testable here).  **Injection-point
>   finding (iter 33):** `clubb_lite` IS production-wired (`turbulence="clubb_lite"`
>   → `clubb_lite_turbulence` with `config.clubb_lite`), but `ExperimentConfig`
>   carries only the scheme *string* and the driver builds
>   `TurbulenceConfig(scheme=cfg.turbulence)` with DEFAULT sub-configs at ~3 backend
>   sites (MPAS/spectral/FV). So `build_driver` needs a shared physics-config
>   builder + an `ExperimentConfig` turbulence override threaded through those
>   sites — an invasive, HPC-validated driver refactor.  **DONE (iter 35):**
>   `ExperimentConfig.turbulence_override` + `turbulence_config_for` thread the
>   corrected config (scalar OR per-column-array C_K) through all backends; a real
>   run with a different C_K provably changes the model state. So `build_driver`
>   is now `ExperimentConfig(..., turbulence=clubb_lite,
>   turbulence_override=apply_feedback_to_scheme(...))`.  Two pieces remain for
>   the empirical demo: (a) the **column-ordering contract** — the feedback
>   field's flat (grid row-major) order must match the physics `ColumnAdapter`'s
>   row-major flatten — DONE (iter 36): `test_feedback_column_ordering.py` locks
>   feedback-flat == physics `ColumnAdapter` order via the REAL
>   `build_physics_pipeline(grid).adapter` + a real coupled driver (lat-lon +
>   cubed); distributed/MPI rank-local sharding is a separate MPI item; (b) the
>   run itself on real ERA5 at scale.
> - **LES batch cost**: DONE (iter 33 clustering + iter 34 wired into
>   `run_correction_iteration` via `les_budget`); the HPC harness sets `les_budget`
>   to its affordable LES count.
> - **Extractor follow-ups** — cubed-sphere geostrophic (metric-correct
>   east/north↔grid rotation, visually verified for cube-edge artifacts);
>   Gaussian/Voronoi grids (the dispatcher raises on those).
---

## 1. Goal

Add a capability that:

1. Runs an **AMIP-style** atmosphere simulation (prescribed observed SST/SIC).
2. **Compares** the model state to **ERA5 reanalysis** at the reanalysis output cadence.
3. **Diagnoses** which atmospheric *columns* in legoESM are performing worst.
4. **Spins off a full 3-D LES** forced like each worst-performing GCM column.
5. Uses the **LES↔GCM discrepancy** to **diagnose a physical closure coefficient**
   (e.g. an entrainment rate) and feed that high-resolution information back to
   **correct the column bias** — as a spatially-varying (and possibly
   height-varying) parameter field in the GCM.

This is an **offline, iterative** loop (not online/in-the-loop, not
superparameterization).

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  AMIP run (prescribed SST/SIC)                                    │
  │        │  state @ 6-hourly                                        │
  │        ▼                                                          │
  │  Compare vs ERA5 (6-hourly)  → per-column error metrics           │
  │        │                                                          │
  │        ▼                                                          │
  │  Rank worst columns (T/q profile RMSE + precip + wind error)      │
  │        │  top-N (configurable)                                    │
  │        ▼                                                          │
  │  Extract each column's large-scale state → SCMForcing             │
  │        │                                                          │
  │        ▼                                                          │
  │  Full 3-D LES (standalone, offline) forced like the column        │
  │        │  resolved fluxes w'T', w'q', w'u'                        │
  │        ▼                                                          │
  │  Diagnose closure coefficient (e.g. entrainment) from LES         │
  │        │  per-column, possibly height-varying                     │
  │        ▼                                                          │
  │  Update spatially-varying parameter field → next AMIP iteration   │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Decisions (interview answers)

### Architecture
| Decision | Choice | Notes |
|---|---|---|
| **LES engine** | **Existing 3-D** `compressible_euler_plane.py` (already 3-D; Smagorinsky SGS). **Add** an optional 1.5-order TKE SGS closure + run **standalone at true LES resolution**. | The plane dycore is already a full 3-D doubly-periodic non-hydrostatic core (prognostic `u,v,w`, 3-D strain-rate Smagorinsky) — **no 2D→3D extension needed**. "plane" = Cartesian/tangent-plane geometry, *not* 2-D. The real work is (a) LES-resolution validation and (b) a new 1.5-order TKE SGS option. *Not* an embedded plane-CRM superparameterization. |
| **Loop topology** | **Offline / iterative** | AMIP → diagnose → batch LES → update params → re-run. No LES inside the GCM time loop. Restartable. |
| **Feedback mechanism** | **Spatially-varying parameter field** | A scalar physics coefficient becomes a per-column (possibly height-varying) field. *New infra — none exists today.* |
| **Worst-column metric** | T & q profile RMSE **+** precipitation error **+** wind/circulation error | Mass-weighted profile RMSE (reuse `scm_rce_metrics`). **Not** using time-tendency growth. |

### Data & forcing
| Decision | Choice | Notes |
|---|---|---|
| **ERA5 cadence / run type** | **6-hourly ERA5, AMIP first** | WeatherBench2 ERA5 default cadence; instantaneous snapshots. `diag_days → 0.25`. AMIP with prescribed observed SST/SIC. |
| **LES forcing source** | **From the GCM column's large-scale state** | Extract subsidence, horizontal advective tendencies, geostrophic wind, surface fluxes from the flagged GCM column; impose via `SCMForcing`. CRM/LES sees the GCM's own environment (so discrepancy ⇒ physics/closure error, given the same large-scale env). |
| **First target parameter** | **Pick after diagnosis** | Keep the target-parameter generic in the interface; choose the scheme/coefficient the worst-column diagnosis most implicates. The user's leading example is an **entrainment rate**. |

### LES configuration
| Decision | Choice | Notes |
|---|---|---|
| **LES regime** | **Configurable per column** | Shallow (dx ~25–100 m, domain ~5–10 km, top ~3–4 km) vs deep-convection (dx ~100–500 m, domain ~50–100 km, top ~20 km) chosen from the diagnosis. |
| **LES budget (N columns)** | **Configurable N** | Default small (top 5–20), scale up on HPC. (Clustering by environment recorded as a future option — see §6.) |
| **Vertical mapping** | **Interpolate + relax above LES top** | Interpolate GCM column to LES fine grid below LES top; force within the domain; Newtonian-relax near LES top to the GCM profile. Standard LES-from-GCM practice. |
| **Correction target** | **Resolved turbulent/convective fluxes** → **diagnose a closure coefficient** | Use LES-resolved `w'T'`, `w'q'`, `w'u'` (and condensate/precip as needed) to **diagnose a physically-meaningful coefficient (e.g. entrainment rate)** that, in the GCM, is a **constant or height-varying coefficient**. The diagnosed coefficient becomes the per-column field. |

### Deferred (decide later — see §6)
- Exact target parameter (after diagnosis).
- Generalization strategy: regress correction onto environmental predictors
  (SST/CAPE/shear) **vs** tie to geographic (lat,lon). Depends on LES budget.
- Whether the coefficient is set by **direct diagnosis** from LES fluxes,
  **gradient-based** matching (differentiable SCM vs LES target), or both.

---

## 3–5, 8. Build plan — REALIZED (see the iter-1..19 status table above)

The original §3 (infrastructure to reuse), §4 (pipeline stages), §5 (new-infra
gaps), and §8 (proposed file layout) described *what to build*; all of it is now
built, tested, and Codex-reviewed — see the status table at the top of this file
and the git history on `feat/compare-reanalysis`. Key reuse anchors that the
build honoured: ERA5 ingestion (`ml/data/era5_loader.py`, `training/era5_to_state.py`),
AMIP driver (`scripts/run/run_amip.py`, `OutputConfig.diag_days`), comparison
metrics (`ml/loss.py`, `training/scm_rce_metrics.py`), SCM forcing
(`atmosphere/scm_forcing.py` + `plane_large_scale_forcing.py`), the plane NH LES
dycore (`atmosphere/dynamics/compressible_euler_plane.py`, `scripts/run/run_les_plane.py`),
plane diagnostics (`rce_diagnostics.py`), and the param registry
(`training/param_collector.py`, `feedback.apply_column_parameter_field`). File
layout followed repo rules (source under `packages/<pkg>/legoesm/`, scripts in the
right bucket, a direct test per `.py`, dispatch raises on unknown).

---

## 6. Open questions / deferred

- **Target parameter** — fixed after the diagnosis stage implicates a scheme.
  Leading candidate: convection/turbulence **entrainment rate**.
- **Generalization** — regress correction onto environmental predictors (SST,
  CAPE, shear) for space/time generalization **vs** static `(lat,lon)` field.
  Depends on affordable LES count. (Clustering worst columns by environment and
  running one LES per cluster representative is the natural bridge.)
- **Coefficient inference method** — direct diagnosis from LES fluxes vs
  gradient-based matching (differentiable SCM vs LES target, extending
  `scm_rce` param training) vs both.
- **Regrid direction** — ERA5→model grid vs model→ERA5 lat-lon for the
  comparison. Pick once; document; keep area/mass weighting consistent.
- **LES lateral BCs** — doubly-periodic plane assumes horizontally homogeneous
  large-scale forcing; valid for a single representative column, revisit for
  strongly heterogeneous columns.

---

## 7. Risks & considerations

- **LES cost dominates.** A single deep-convection LES (dx 100–500 m, ~50–100 km,
  ~20 km top) is far more expensive than the whole AMIP run. Keep N small;
  consider clustering; LES stage must be checkpointed and embarrassingly
  parallel across columns.
- **The gap is resolution, not dimensionality.** The plane dycore is already
  fully 3-D (see §3) — no 2D→3D work is needed. What *is* needed: it was only
  validated at dx=2 km (CRM), not dx≤100 m (LES). Finer dx changes the acoustic
  CFL (N_ACOUSTIC), Smagorinsky behavior, and stability. Expect re-tuning + fresh
  stability validation at LES resolution before trusting the resolved fluxes.
  (See `docs/specs/CRM_implementation.md` for stabilization findings F7/F10/F11.)
- **SGS closure realism at LES scale.** Smagorinsky is adequate for CRM scale but
  a 1.5-order TKE closure (new — does not exist yet) is generally preferred for
  boundary-layer LES; the diagnosed coefficient may depend on the SGS choice, so
  validate against a published case (§9) for each closure.
- **Forcing consistency.** Forcing the LES from the *GCM* column's large-scale
  state (chosen) means the LES↔GCM discrepancy attributes error to the
  **physics/closure given the same environment** — *not* to large-scale errors.
  This is the intended attribution but must be stated when interpreting results
  (a column wrong because its large-scale state is wrong won't be fixed here).
- **Vertical-top relaxation** can contaminate the diagnosed coefficient if the
  relaxation layer overlaps the diagnosis region; keep the diagnosis below the
  sponge.
- **Differentiability / JIT** of the spatially-varying field is a first-class
  requirement (autodiff + mass conservation, per CLAUDE.md). The field must be a
  clean pytree leaf with stable shape.
- **Conservation.** Any feedback correction must not silently break column mass/
  energy/moisture conservation; validate with conservation diagnostics.

---

## 9. Validation plan

- **Single-rank → MPI** for the LES stage (per CLAUDE.md parallel rule).
- **LES at LES resolution** must re-pass the RCE realism gate (CWV plateau,
  precip ~3 mm/day, MSE drift) before its fluxes are trusted.
- **Visual verification** of any spatial parameter field (cube-imprint / grid
  artifacts not caught by norms).
- **Conservation diagnostics** after the feedback correction.
- **Mandatory Codex adversarial review** on each major stage (new module /
  >~50 LOC / numerics / AD / conservation), per CLAUDE.md iterate-with-codex loop.
- Idealized check: force the LES *and* the GCM column with an identical known
  large-scale state for a case with a published entrainment rate (e.g. a
  BOMEX/DYCOMS-class setup) and confirm the diagnosed coefficient recovers it.

---

## 10. Interview log

Three rounds of structured questions were used to capture the decisions in §2.
Key clarifications volunteered by the user:

1. *"I don't want to do a plane CRM as in superparameterization but a full 3-D
   LES."* → The plane dycore is the engine, but run **standalone at LES
   resolution**, offline, forced by the GCM column — not embedded per-column.
2. *"Something like the entrainment rate that can be diagnosed from the LES, but
   must be a constant or possibly a height-varying coefficient in the GCM."* →
   The correction is a **diagnosed closure coefficient**, realized as a
   spatially-varying (and possibly height-varying) parameter field.
