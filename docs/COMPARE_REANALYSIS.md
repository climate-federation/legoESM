# Compare-to-Reanalysis + LES-Informed Column Correction

**Branch:** `feat/compare-reanalysis`
**Status (iter 50, compressed):** The whole offline loop is built, tested + Codex-reviewed, and **structurally complete** end-to-end — every stage `run (AMIP/CMIP) → time-mean → compare to ERA5 → rank → cluster → LES-diagnose → feedback → inject (turbulence_override) → RE-RUN` is real and tested (capstone iter 37; the injection provably changes the simulation, iter 35; feedback↔physics column ordering locked, iter 36). A **runnable HPC entry point** exists — `scripts/run/run_correction_campaign.py` (`make_base_driver_builder` for AMIP **and** CMIP, real-ERA5 reference loading, per-column `clubb_lite` C_K **and** Pr_t correction (single or SIMULTANEOUS multi-coefficient), per-round bias report). The per-iteration table below records each stage's module + tests (compressed at iters 10/20/30/40/50; full detail in git log). **Remaining (the done-criterion):** ONLY the EMPIRICAL demonstration — execute the multi-day campaign on real ERA5 at HPC scale and observe the LES-informed C_K *lower* the bias (not unit-testable here, not honestly fakeable; the result is genuinely unknown until run). Breadth follow-ups: distributed/MPI feedback sharding, cubed-sphere geostrophic rotation, Gaussian/Voronoi extractors.
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
> | restartable campaign (§1) | `initial_field`/`start_round`/`checkpoint_callback` + CLI `--checkpoint`/`--resume`; **resume == uninterrupted** (a multi-day HPC campaign survives a job timeout) | 41 |
> | env-kernel generalization wired | `assemble_feedback_field` `strategy="environment"` N-W-regresses diagnosed C_K onto SST/CAPE/shear over the full grid (`column_environment_grid`); K LES on cluster reps generalize to all env-similar columns. Codex bug fixed: `environment_kernel_field` array background (C-order, size-checked) for multi-round accumulation | 42 |
> | **monotonic acceptance gate** ("improve the biases") | `run_correction_campaign(accept_only_if_improved)`: keep a round only if the global bias fell, else REJECT (revert to best-so-far base) → the field never regresses. Codex HIGH fix: checkpoint persisted the rejected config beside the accepted field → resume desync; both now derive from the accepted field | 43 |
> | **line search over correction magnitude** | `step_fractions` backtracks `base + s·(raw−base)` (Armijo, largest improving s), robust to the LES↔GCM overshoot; composes with the gate. Codex bug fixed: `environment_kernel_field` excluded background from its dtype → float64 base downcast/drift; base_field now at raw_field.dtype | 44 |
> | physical-bounds guard on the diagnosed coefficient | (1) non-finite reduced diagnosis → INVALID (keeps background); (2) `clip_to_bounds` clamps to the registered `__param_spec__` bounds (reusing `param_field_bounds` + `clip_field_to_promotable_bounds`), pre-line-search AND post-blend (the invariant) AND on no-op. Codex CLEAN (no-op gap fixed) | 45 |
> | **dimensionally-correct C_K diagnosis** (units-bug fix) | the loop injected a DIMENSIONAL K [m²/s] as the DIMENSIONLESS C_K; new `diagnose_clubb_coefficient` returns C_K = K_m/(ℓ·√wp2), the exact inverse of the GCM `Km=C_K·ℓ·√wp2` (shear-projected MOMENTUM K_m, the SAME shared `mixing_length`, resolved w'²). Leaf inversions `momentum_diffusivity_from_fluxes`/`clubb_coefficient_from_diffusivity` (double-where AD-safe, all arrays co-located). Design+impl Codex-reviewed. KNOWN LIMITATION: C_K uses LES w'² vs the GCM's own prognostic wp2 — possible O(1) offset in deep convection (C_eps co-tuning a follow-up) | 46 |
> | **second coefficient: Pr_t = K_m/K_h** | a DIMENSIONLESS ratio (no wp2-identification issue). `prandtl_number_from_diffusivities` + `diagnose_prandtl_number` (K_h reuses `diagnose_eddy_diffusivity`, K_m the iter-46 momentum inversion); one-line body change `Kh=Km/broadcast_column_param(Pr_t,l_mix)` (scalar byte-identical); registered `clubb_lite_Pr_t` (bounds (0.3,1.5)). Codex fixed 2 CLI-checkpoint bugs | 47 |
> | **SIMULTANEOUS multi-coefficient correction** (sequenced PR1–4) | PR1 factored `_run_line_search` out of `run_correction_iteration` (byte-identical single path); PR2 `run_column_les_pipeline(methods=)` runs the LES ONCE and diagnoses many (`{method: diagnosis}`); PR3 `CorrectionSpec` + `run_multi_correction_iteration` (per-spec assemble+clamp, ONE shared line-search fraction, COMBINED bias, applied to independent config slots); PR4 `run_multi_correction_campaign` (per-coefficient base DICT; gate advances/reverts ALL coefficients atomically). Shared helpers `_diagnose_columns`/`_env_grid_predictors`/`_raw_and_base_field`. Limitation: one fraction + one gate ⇒ a round mixing a helping + a hurting coeff is rejected whole. Codex-reviewed each (single path byte-identical, atomic revert, non-dict-diagnose guard) | 48–49 |
> | **multi-coefficient campaign wiring** (PR5) | `build_multi_correction_campaign(coefficients=("C_K","Pr_t"))` maps each coefficient → CorrectionSpec, sets `les_config.diagnosis_methods` (one LES run, many diagnoses), auto-populates l_mix_max; CLI `--coefficients C_K,Pr_t` routes to a DICT checkpoint/resume/output. Factored shared compose helpers (`_compose_compare_fn`/`_maybe_env_grid_fn`/`_grid_latlon_deg`). Codex: no wiring bugs (2 nits fixed). Multi-coefficient correction is now complete end-to-end | 50 |
> | **campaign health verdict + persisted summary** (the actionable complement to iter-54) | `campaign_health(summary) → CampaignHealth(status, message)` classifies a finished campaign into a one-look, actionable verdict: `"improved"` (bias fell ≥ 2%, `.ok`; a binding clamp is surfaced as a NOTE — a good run is never demoted), `"clamp_limited"` (did NOT improve AND ≥20% of columns pinned at a coefficient's `__param_spec__` bounds → the LES wants a value outside the calibratable range; widen the bounds / check the diagnosis), `"stalled"` (no improvement, clamp not the cause → check the LES↔GCM transfer or config), `"no_rounds"`. `CoefficientSummary` gained `n_columns` + a `clamp_fraction` property (capped at 1 for degenerate bounds). Both CLI paths now COMPUTE the summary+health, PERSIST them into the output JSON (`"summary"`, `"health"`), and print the verdict — so an HPC run's output is self-describing + tells the user IF the parameters improved the bias and WHY not. Investigated + kept-deferred the cubed-sphere geostrophic (grid-frame computation is simpler than the iter-29 framing but still carries an O(skew) corner error on the non-orthogonal metric) and the MPI per-column override (a real but future-facing gap; the override already fails LOUDLY, not silently, under a column-count mismatch). **Codex-reviewed** (PASS; the one concern — a strongly-improving run with clamp-binding being mis-demoted — fixed: improvement is judged FIRST, clamp surfaced as a note). Tests: improved / stalled / clamp-limited-when-not-improving / improved-takes-precedence-over-clamp / no-rounds / clamp_fraction | 55 |
> | **campaign diagnostics summary** (monitor whether the corrections work) | New `legoesm.training.campaign_summary` condenses a single- or multi-coefficient campaign into the numbers an HPC user reads to judge it (the iter-46 review's "monitor coefficient drift"): `summarize_campaign(result)` → `CampaignSummary` with the bias trajectory (initial→final via the LAST ACCEPTED round, absolute + fractional reduction), acceptance rate, `stop_reason`, and per-coefficient `CoefficientSummary` (field min/max/mean/std + **n columns pinned at the registered `__param_spec__` bounds** — a binding clamp signals the LES wants a coefficient MORE extreme than the calibratable range, an actionable signal). Bounds resolved via the registry (`PROMOTABLE_FIELDS` + `feedback.param_field_bounds` on the result's `final_config`); pure host-side NumPy, duck-types the result (no `correction_loop` import → no cycle). `.report()` prints a concise multi-line roll-up; both CLI paths emit it after the run. Decided AGAINST the cubed-sphere geostrophic + Gaussian extractors this iteration: both confirmed genuinely complex/risky (cubed needs non-orthogonal-metric handling + visual verification; Gaussian needs the spectral SH transforms) — kept deferred. **Codex-reviewed: PASS, all clean** (last-accepted final-bias walk, multi/single dispatch, bound-count tolerance, array-config bounds lookup, layering). Tests: single + multi summaries, bound-hitting count, single-needs-promotion_key raise, zero-round safety, report() | 54 |
> | **convergence early-stopping + perfect-model loop-convergence validation** | (a) `bias_tol`/`patience` on both campaigns: stop once `patience` (default 2) consecutive rounds make no meaningful progress (rejected, or improving by < `bias_tol`) so a multi-day HPC campaign doesn't waste expensive rounds after the bias plateaus; `CampaignResult`/`MultiCampaignResult.stop_reason` = `"converged"`\|`"max_iterations"`; factored `_round_made_progress`; wired through `build_*` + CLI `--bias-tol`/`--patience`. (b) **Perfect-model OSSE for the LOOP mechanics** (`test_campaign_real_compare_multi_round_converges_to_zero_bias`): a MULTI-ROUND `run_correction_campaign` over the REAL `compare_state_to_reference` (only the LES replaced by a perfect diagnose_fn) drives the per-column coefficient so the bias DECREASES MONOTONICALLY to ~0 and the campaign converges + early-stops — the strongest in-environment demonstration that updating the parameters lowers the bias (the LES↔GCM transfer remains the separate HPC-only empirical question, per §9). **Codex-reviewed: CLEAN** except one bug fixed — `patience<=0` was an immediate-stop footgun (`no_progress>=0` always true), now raises when `bias_tol` is set. Tests: early-stop converges (single + multi), `bias_tol=None` runs all rounds, the perfect-model monotone-to-zero convergence, the patience guard, build-level early-stop wiring | 53 |
> | **sequential (staged) multi-coefficient mode** (block coordinate descent) | Addresses the iter-49/51 limitation that the COMBINED mode (one shared fraction + one gate on the combined bias) rejects a round whole when one coefficient helps + another hurts — acute for the COUPLED C_eps↔C_K (C_eps sets the GCM wp2 that C_K then uses, Codex iter-51 concern). `run_multi_correction_iteration(sequential=True)` corrects the coefficients IN SPEC ORDER, each with its OWN line search + gate against the RUNNING state (so a later coefficient sees the prior accepted ones — recommended order C_eps→C_K→Pr_t lets wp2 converge before C_K); a coefficient is kept only if IT improves, the others untouched. `MultiCorrectionResult.step_fractions_by_key` reports the accepted per-coefficient fraction (0.0 = rejected). The round can never worsen (each step gated) so the campaign accepts + accumulates normally; feedback_fields stays consistent with updated_config (rejected coeff = base). One LES run still shared (diagnose once, then per-coefficient apply+compare). Wired through `run_multi_correction_campaign`/`build_multi_correction_campaign` + CLI `--staged`. Default `sequential=False` = combined (byte-identical via the factored `_blend`). **Codex-reviewed: all CLEAN** (closure cfg-capture, monotone guarantee, feedback/config consistency, campaign accumulation, combined byte-identity). Tests: combined rejects help+hurt whole, sequential keeps the helper + rejects the hurter (per-coeff fractions), sequential campaign accumulates, build_multi `--staged` C_eps,C_K,Pr_t | 52 |
> | **third coefficient: C_eps — closes the iter-46 wp2-identification gap** | The iter-46 LIMITATION (C_K = K_m/(ℓ·√wp2) uses the LES w'² but the GCM evaluates its OWN prognostic wp2 → O(1) offset in convection) is closed by co-diagnosing **C_eps** (the wp2-dissipation coefficient) so the GCM's equilibrium wp2 tracks the LES w'². New leaf `c_eps_from_budget`: inverts the steady-state lite wp2 budget `C_eps = P·ℓ/wp2^{3/2}`, net production `P = K_m·S² − K_h·N²` (shear − buoyancy), valid where both diffusivities valid + wp2>floor + ℓ>0 + **P>0** + a raw-value sanity ceiling (transport-dominated blow-ups flagged, not clamped-in); double-where AD-safe at `wp2^{3/2}`. New `diagnose_c_eps_coefficient` (reuses K_m/K_h inversions, S² from mean-wind shear, `N²=(g/⟨θ_v⟩)∂⟨θ_v⟩/∂z` with co-located interface θ_v, shared `mixing_length`, resolved w'²). One-line body change `diss_wp2 = broadcast_column_param(config.C_eps, l_mix)·√wp2/l` (scalar byte-identical — tested). Registered `clubb_lite_C_eps` (bounds (0.06,0.6)) + reduce/`coefficient_value`/dispatch (`c_eps` needs l_mix_max) + `_COEFFICIENT_SPEC_MAP`/`_METHOD_PROMOTION`; the multi campaign now co-corrects **C_K + Pr_t + C_eps**; CLI `--diagnosis-method c_eps` + `--coefficients`. **Design + impl Codex-reviewed** (physics/sign/AD/co-location/byte-identity all CLEAN; 2 single-path wiring bugs found+fixed — the build_correction_campaign + CLI c_eps branch). DOCUMENTED LIMITATION: the inversion drops the GCM's wp2-transport term, so the local production=dissipation balance can be off in convective BLs (P>0 + sanity + bounds + the gate are the backstops — a rejected C_eps correction does no harm). Tests: analytic C_eps + P<0/blow-up invalid + AD-safety, manual-composition wiring, body per-column wp2 + uniform==scalar byte-identical, reduce, registry+clamp, dispatch, single + 3-coefficient campaigns in-bounds | 51 |
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
