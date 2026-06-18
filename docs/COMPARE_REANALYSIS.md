# Compare-to-Reanalysis + LES-Informed Column Correction

**Branch:** `feat/compare-reanalysis`
**Status (compressed at iters 50 + 60 + 70 + 80 + 90 + 100):** The whole offline loop is built, tested + Codex-reviewed, and **structurally complete** end-to-end — every stage `run (AMIP/CMIP) → time-mean → compare to ERA5 → rank → cluster → LES-diagnose → feedback → inject (turbulence_override) → RE-RUN` is real and tested (capstone iter 37; the injection provably changes the simulation, iter 35; feedback↔physics column ordering locked, iter 36). A **runnable HPC entry point** exists — `scripts/run/run_correction_campaign.py` (`make_base_driver_builder` for AMIP **and** CMIP, real-ERA5 reference loading, per-column `clubb_lite` C_K/Pr_t/C_eps correction (single or SIMULTANEOUS multi-coefficient), per-round bias report, self-describing summary+health verdict). The campaign's saved JSON now **deploys back into a fresh production run** via `legoesm.training.deploy_correction.corrected_turbulence_override` (iter 57, grid-identity-guarded iter 58) — the literal "update the parameters in the AMIP/CMIP simulation" plumbing is closed and `validate_strict`-checked. A **cross-resolution (grid-AGNOSTIC) deploy** path also exists end-to-end: a campaign run with `--feedback-strategy environment` now EXPORTS a `<out>.env_kernel.json` (the RAW env→coefficient regression, iters 69–70 producer wiring) that `apply_env_kernel_override` re-evaluates on ANY target grid by environmental similarity (so a cheap low-res campaign deploys on an expensive high-res run), with a coverage/in-hull domain-shift diagnostic. A **perfect-model (identical-twin) OSSE harness** (`legoesm.training.perfect_model_osse`, iters 59–60) demonstrates clause 5's LOGIC in a controlled twin — the model's own run with a KNOWN coefficient is the pseudo-truth, and the loop is checked to LOWER the bias AND RECOVER the known parameter (single AND simultaneous multi-coefficient C_K+Pr_t+C_eps) — the cheap go/no-go an HPC user runs before the real-ERA5 campaign. A **cross-resolution OSSE** (iter 72, `run_cross_resolution_osse`) extends this to the grid-AGNOSTIC env-kernel deploy: it LEARNS the env→coefficient correction on a coarse grid and DEPLOYS it on a finer grid by environmental similarity, demonstrating (in the twin) that the kernel TRANSFERS and lowers the finer grid's bias — the first measured bias-reduction from the iter-69/70 deploy, gated by an out-of-hull (extrapolation) trust check. The per-iteration table below records each stage's module + tests (compressed at iters 10/20/30/40/50/60/70/80, folding iters 1–50 into summary rows; full detail in git log). **Remaining (the done-criterion):** ONLY the EMPIRICAL demonstration — execute the multi-day campaign on real ERA5 at HPC scale and observe the LES-informed coefficients *lower* the bias (not unit-testable here, not honestly fakeable; the result is genuinely unknown until run). **Breadth — ALL DONE through iter 90:** the LES spin-off works on ALL 4 grid families (lat-lon, cubed-sphere, MPAS/Voronoi, Gaussian/spectral — extractors + compare-side handoffs + capstones, iters 73–85); distributed/MPI FEEDBACK sharding is real-mpirun-validated on all 3 structured families (iters 61–64); and the **distributed-MPAS LES-correction campaign now runs END-TO-END under real MPI** (owned-cell mask 86 + cross-rank global top-k 87 + collective correction loop 88 + `build_distributed_correction_campaign` 89, all `mpirun -np 2`-validated). Geostrophic LES-forcing wind is now supplied on ALL 4 grid families — lat-lon + Voronoi (iter 82) + Gaussian (iter 90, via the shared `spectral_gradient_3d`) + cubed-sphere (iter 93, the metric-correct 2×2 basis solve `_gradient_cubed_geographic_3d`, CONVERGING incl. panel edges after iters 91/92's naive attempts were Codex-rejected). The one-shot CLI spectral-CHECKPOINT restart load is DONE (iter 92, `load_restart` reconstructs the spectral keys via `reconstruct_spectral_state_from_npz`). The distributed-MPAS campaign is runnable for SINGLE + simultaneous MULTI-coefficient correction (iters 89/95) AND now has a ONE-CALL runnable entry point `build_distributed_mpas_campaign` (iter 96) that does the MPI-aware setup (partition the global mesh → wire the rank-local mesh as grid + driver) so an HPC user supplies only the global mesh + a `build_local_driver`. The distributed slice now FAILS LOUDLY on a global/reference/area mesh-count mismatch (iter 97 — exact `nCells_global` guards replacing a JAX gather's silent OOB-clamp), and a DEFAULT-ON collective pre-flight (`assert_partition_covers_global`, iter 98) verifies the rank partition covers the global mesh EXACTLY once (no silently-uncorrected or double-counted cell) before a multi-day run. A DEFAULT-ON physical-plausibility pre-flight (`validate_reference_physical`, iter 99) catches a units/sign error in the loaded ERA5 reference (T/p_s/q_v wrong units) before it drives the loop to 'correct' a fake bias. The campaign now reports the LES-diagnosis VALIDITY rate (`n_diagnoses_valid` per round, `n_diagnoses_valid_total` in the summary) and a `no_valid_diagnoses` health verdict (iter 100), so a run that makes no correction because the spin-off LES never developed turbulence says so instead of looking like a broken correction; it also EARLY-ABORTS (collective-safe, `stop_reason="no_valid_diagnoses"`, default-on, CLI `--keep-dry-rounds` to disable) after `patience` consecutive dry rounds rather than burning multi-day HPC time (iter 101). The CLI launch chain (flag → parser → shared `_campaign_knobs_from_args` → builder → loop) is now fully unit-tested (iter 102), so a wiring regression fails in CI rather than on a multi-day HPC launch. The spin-off LES now FAILS FAST on an acoustically-unstable timestep (iter 103, `run_forced_les` horizontal acoustic-CFL pre-flight reusing `cfl_diagnostic`; all 3 LES CLIs default to the CFL-safe `--les-dt 0.5`) instead of blowing up partway through a multi-day run. A CI import-resolution guard (iter 104, `test_cli_imports_resolve.py`) now catches a stale function-scope import in any of the 4 compare-reanalysis HPC CLIs (the OSSE-class dead-entry-point bug) before a launch, and the campaign→deploy JSON contract is round-trip-tested (iter 105, `build_campaign_output_dict` → `corrected_clubb_config`) so the on-disk format that updates a production run cannot silently drift. The real-ERA5 loader now FAILS LOUDLY on a missing/misnamed required variable (iter 106, `load_era5_slice` required-raise + bidirectional `resolve_var`) instead of silently feeding zeros into the bias, and the full real ERA5-input→reference chain (load→regrid→interp→carry→ColumnState) is now exercised end-to-end by an integration test (iter 107) instead of being bypassed by the compare test's monkeypatch. **All documented breadth is now COMPLETE; the ONLY remaining item is the done-criterion's empirical real-ERA5 HPC demonstration.**
**Date:** 2026-06-15 (design); 2026-06-17 (impl began)
**Scope:** Atmosphere component only. ERA5 reanalysis only. **Not** supervised learning.

> ## Implementation status (compressed at iter 10; full history in git log `feat/compare-reanalysis`)
>
> The pipeline is built bottom-up as tested, Codex-reviewed components. Done so far:
>
> | Stage / gap | Module (all tested + Codex-clean) | Iter |
> |---|---|---|
> | **Foundational pipeline build (iters 1–9)** — COMPRESSED (header narrates; git log has per-iter detail) | per-column ERA5 metric (`training/column_era5_metrics.py`), worst-column manifest + SST/CAPE/shear env tags (`column_manifest.py`), compare orchestration + driver (`compare_reanalysis.py`, `scripts/validate/compare_amip_era5.py`), GCM-col→SCMForcing (`atmosphere/column_forcing.py`), LES resolved fluxes (`rce_diagnostics.py`), closure-coeff diagnosis incl entrainment w_e (`les_closure_diagnosis.py`), parameter-field assembly (`parameter_field.py`: static scatter + N-W env regression), promotion-gated feedback splice (`feedback.py`) | 1–9 |
>
> **Conventions held every iter:** pre-impl search + reuse (no re-derived numerics);
> every new `.py` gets a direct unit test (analytic where possible); AD-safe
> (double-where masked divisions, gradient-safe sqrt); dispatch raises on unknown;
> mandatory Codex adversarial-review loop to clean before commit.
>
> | **Real-pipeline + grid-coverage build (iters 10–30)** — COMPRESSED (git log has per-iter detail) | lat-lon large-scale extractor (`column_large_scale_extract.py`), 1.5-order TKE SGS closure (`tke_sgs_plane.py`), LES regime selection + vertical mapping (`les_regime.py`, `les_vertical_mapping.py`), LES→coefficient (`column_les_diagnosis.py`), column-LES driver (`run_column_les.py`), diagnoses→feedback (`feedback_assembly.py`), bias verification (`bias_metrics.py`), per-scheme promotion (`promotable_params.py`), the CLOSED-LOOP orchestrator (`correction_loop.run_correction_iteration`), real-dycore LES + moist-IC fix, real compare adapter (`make_compare_fn`), AMIP+CMIP bridge (`column_state_from_hydrostatic`) + real CMIP compare test, physically-coherent `clubb_lite_C_K` promotion (`broadcast_column_param`), multi-round campaign (`run_correction_campaign`), mock-free e2e gate, cubed-sphere extractor + LES spin-off, geostrophic wind / LES Coriolis, time-mean accumulator (`column_state_accumulator.py`) | 10–30 |
> | **Closed-loop + multi-coefficient build (iters 31–50)** — COMPRESSED at iter 80 (header narrates; git log has per-iter detail) | run→time-mean→compare wiring (`run_to_column_mean.py`, `amip`/`cmip_column_state`) + real `compare_fn` (`make_run_fn`∘`make_compare_fn`); env-clustering for LES cost (`cluster_columns_by_environment` k-center, `les_budget=K`); **CONFIG-INJECTION** (`turbulence_override`+`turbulence_config_for`, proven: different C_K→different T); column-ordering CONTRACT (`test_feedback_column_ordering`); **CAPSTONE real re-run** (`test_correction_loop_real_rerun`, the (128,) C_K reaches the re-run kernel); **HPC CAMPAIGN DRIVER** (`run_correction_campaign.py`: `make_clubb_build_driver`+`make_les_diagnose_fn`+`build_correction_campaign`); LES library→package (`column_les.py`); CMIP CLI (`make_base_driver_builder` AMIP+CMIP); restartable campaign (`initial_field`/`start_round`/`checkpoint`/`--resume`); env-kernel generalization (`assemble_feedback_field strategy="environment"`); **monotonic acceptance gate** (`accept_only_if_improved`, field never regresses); **line search** (`step_fractions`, Armijo); physical-bounds guard (`clip_to_bounds`); **dimensionally-correct C_K** (`diagnose_clubb_coefficient`=K_m/(ℓ√wp2), the exact GCM inverse); **Pr_t** (`diagnose_prandtl_number`, K_m/K_h); **SIMULTANEOUS multi-coefficient** (`CorrectionSpec`+`run_multi_correction_iteration/campaign`, one LES→many diagnoses, atomic gate) + wiring (`build_multi_correction_campaign(coefficients=…)`). All Codex-reviewed (several real bugs fixed: env-kernel dtype/background, checkpoint resume desync, no-op clamp gap) | 31–50 |
> | **End-to-end integration test of the real ERA5-input→reference chain (the gap that hid iter 106's bug)** (iter 107 — exercises load→regrid→interp→carry→ColumnState with real source) | The compare test MONKEYPATCHES `load_era5_slice`, so the REAL input chain (`load_era5_slice` → `era5_to_latlon_carry` [scipy regrid + log-p `interp_pressure_to_sigma` + q→mixing-ratio] → `column_state_from_carry`) was UNTESTED end-to-end — exactly the integration gap that let iter 106's silent-zeros bug hide. First AUDITED the rest of the input path: `interp_pressure_to_sigma` is clean (correct log-p interp + hold-constant extrapolation via searchsorted/clip/alpha-clamp), the carry's q→mixing-ratio + the `fill_value=None` regrid are fine for the standard convention (model cell-centred lat ⊂ ERA5 pole-inclusive lat; model lon ⊂ ERA5 lon → no out-of-bounds extrapolation; verified the actual `create_latlon_grid` vs ERA5 ranges). NEW integration test: a synthetic GLOBAL ERA5 dataset (real `load_era5_slice`) → real `era5_to_latlon_carry` onto `create_latlon_grid(3,4)` → `column_state_from_carry` → a reference that is correctly shaped on the MODEL grid+sigma AND passes the iter-99 `validate_reference_physical`. **Codex 2-round review: round 1 — every source/integration concern CLEAN (descending-lat OK in scipy≥1.10, no out-of-bounds extrapolation, validate has real teeth, monkeypatch target correct, shape ordering correct) + ONE MEDIUM: the test fields were spatially UNIFORM, so the horizontal regrid was effectively untested (a lat/lon axis swap would still pass); FIXED — `u` now varies LINEARLY by latitude (10+0.5·lat/90) and the test asserts the EXACT bilinear-interpolated profile at model lats (±60°) that fall BETWEEN the ERA5 nodes (±45/±90), so an axis swap fails by >0.4 m/s and the zeros-bug by ~10; round 2 CLEAN (bilinear math exact, level-constant field passes the vertical interp unchanged, atol calibrated, non-vacuous).** Tests: the integration test + the existing load/resolve tests adapted (T level-varying, u lat-varying — both interps now genuinely exercised). 7 green; ruff-clean. **The real ERA5→reference chain the empirical run consumes is now exercised end-to-end, not bypassed by a monkeypatch.** | 107 |
> | **Fail-fast ERA5 loader: no silent-zeros on a missing required variable + bidirectional name resolve** (iter 106 — hardens the real-ERA5 INPUT path that feeds the whole compare) | `load_era5_slice` (the real-ERA5 input to the entire pipeline) SILENTLY loaded a missing/misnamed REQUIRED variable as `np.zeros(...)` — a zeros T/u/v/q/p_s corrupts every column's bias so the loop "corrects" against GARBAGE with no error (a CLAUDE.md no-silent-coerce catastrophe on the empirical-run input). And `resolve_var` resolved names ONE-WAY (long ERA5/WeatherBench → short ECMWF/GRIB) with a dead redundant block, so a short-named store with a long-name request fell through to the silent zeros. FIX: `resolve_var` is now BIDIRECTIONAL (long↔short, robust to either store convention; dead block removed, alias table promoted to a module constant); the `load_era5_slice` getters gained `required` so a missing REQUIRED field (T/u/v/q/p_s) RAISES with the requested name + alias + the store's available variables, while optional surface fields (skin_temperature/geopotential_at_surface) keep the zero-fill. (Both were entirely UNTESTED — the compare test monkeypatches `load_era5_slice`.) **Codex 2-round review: round 1 — NO source bugs (resolver bidirectionality + no long/short key collision, geopotential vs geopotential_at_surface isolation, required-raise composition, the `resolved[0]→resolved` list-fix, no circular import from the ruff-moved import — all CLEAN) + ONE test-fidelity flag (the synthetic 3D fields were level-constant so the descending→ascending-pressure reversal was not actually validated); FIXED — temperature now varies by level (300/250/200 K at 1000/500/100 hPa) and the test asserts the reversed `T=[200,250,300]` co-located with `plev_Pa=[10000,50000,100000]`; round 2 CLEAN (non-vacuous — fails if the reversal is removed).** Tests: NEW `test_era5_load_slice.py` — resolve_var bidirectional (long→short, short→long, absent→None); load via a synthetic in-memory dataset for long-names + level-ordering, short-names (bidirectional), missing-required-T / missing-required-p_s → raise, missing-optional-sst → zero-fill. 6 green; ruff-clean. **A missing/misnamed required ERA5 variable now FAILS LOUDLY at load instead of silently feeding zeros into the bias.** | 106 |
> | **Round-trip-test the campaign→deploy JSON contract (the literal clause-5 plumbing)** (iter 105 — extract + test the on-disk format the deploy path reads to update a production run) | The campaign CLI writes a corrected-config JSON that `deploy_correction.corrected_clubb_config` reads to update a production AMIP/CMIP run ("update the parameters in the simulation"), but the `json.dump` was INLINE in two `# pragma: no cover` `main()`s, DUPLICATED, and NEVER round-trip-tested against the deploy loader — a format drift would surface only at the empirical run. EXTRACTED a pure `build_campaign_output_dict(result, *, grid_provenance, summary, health, corrected_field|coefficients)` (the ONE place the on-disk format is built; both CLI sites now call it; raises unless EXACTLY one shape is given) and added a write→read→apply round-trip test: build the dict → `json.dumps`/`loads` → `corrected_clubb_config` → assert the per-column C_K (single) / C_K+Pr_t (multi) are recovered + the summary/health/grid survive. **Codex 2-round review: round 1 found a CRITICAL — my `.reshape(-1)` on the SINGLE path silently promoted a 0-D scalar `final_config` field (a no-op / zero-round campaign that corrected NOTHING) to a length-1 array, MASKING the deploy loader's 1-D rejection that the prior `.tolist()` triggered loudly; REVERTED the single to behavior-preserving `.tolist()` (multi keeps its original `.reshape(-1)` — its fields are grid-shaped) + added a test locking the un-masked loud rejection of a scalar single-field; round 2 CLEAN (byte-identical to the prior inline bodies, scalar test faithful + non-vacuous, key order unchanged).** Tests: 4 new (single + multi round-trip, dispatch neither/both, scalar-rejection). 184 campaign+deploy+cli-import green; ruff-clean. **The on-disk campaign→deploy format is now locked by a real round-trip test, so a serialization drift fails in CI, not on a multi-day launch.** | 105 |
> | **Static import-resolution guard for the HPC CLIs (catch the OSSE-class dead-entry-point bug)** (iter 104 — generalizes the iter-103 finding into a CI tripwire) | Iter 103 found a STALE function-scope import in the OSSE CLI (`from rce_diagnostics import run_forced_les` — it lives in `column_les`), INVISIBLE because the heavy `main()` is `# pragma: no cover`: an HPC launch would crash at import. AUDITED all 99 absolute `from X import Y` (module-level + function-scope) across the 4 compare-reanalysis CLIs (`run_correction_campaign`, `run_column_les`, `run_perfect_model_osse`, `compare_amip_era5`) — all resolve now (the only flags were `from mpi4py import MPI` submodule false-positives of `hasattr`). NEW `tests/run/test_cli_imports_resolve.py`: parses each CLI's AST, parametrizes over every absolute from-import, and asserts each symbol resolves on its module WITHOUT running the heavy paths — in-repo (`legoesm`/`scripts`) modules MUST import + expose the symbol (a wrong path or stale symbol fails LOUDLY), optional third-party deps absent from the env are skipped. The resolver is submodule-aware (`from mpi4py import MPI` imports the MPI *submodule*) and rejects a non-package module's missing attribute (the OSSE class). **Codex 2-round review**: round 1 found the core sound + a MEDIUM — `find_spec` proves DISCOVERABILITY not IMPORTABILITY (a discoverable-but-broken submodule would false-pass); FIXED to actually `import_module` the submodule (distinguishing a genuinely-absent candidate from a missing transitive dep) + 2 LOW polish (docstring "absolute"; non-vacuity test uses a synthetic sentinel instead of hard-coding the historical `rce_diagnostics` absence, so a future legitimate re-export cannot churn it) — all applied per Codex's exact recommendations; the non-vacuity test was strengthened to exercise BOTH the non-package and package rejection paths. Tests: 99 parametrized resolution checks + 1 non-vacuity self-test (rejects a sentinel on both a non-package module AND a package submodule path). 100 green; ruff-clean. **A stale import in any compare-reanalysis HPC entry point now fails in CI, not on a multi-day launch.** | 104 |
> | **Fail-fast acoustic-CFL pre-flight on the spin-off LES (don't blow up mid-run)** (iter 103 — closes a §7-flagged clause-3 stability gap) | §7 flags that the plane LES runs at dx≤100 m (vs the dycore's dx=2 km validation); a too-large LES timestep violates the acoustic CFL and BLOWS UP mid-run → an invalid diagnosis + wasted multi-day HPC compute, with nothing checking it. REUSED the existing `cfl_diagnostic` (no re-derivation): EXTRACTED a shared `column_sound_speed_upper_bound(height_coord, theta_prime=None)` (DRY'd `compute_courant_numbers_plane` + `suggest_stable_dt` onto it) and added `acoustic_courant_horizontal(...)` — the HORIZONTAL acoustic Courant `c_s·(dt/n_acoustic)/min(dx,dy)`, distinct from the general `acoustic` (which uses `min(dx,dy,dz)`) because `run_forced_les` solves the VERTICAL acoustic SEMI-IMPLICITLY (vertical unconditionally stable → only the explicit horizontal mode binds). NEW pre-flight in `run_forced_les`: raises if the rest-state horizontal acoustic Courant × a convective-warming margin exceeds the limit, BEFORE the time loop (cheap), with the recommended stable dt. **Codex 3-round adversarial review**: round 1 confirmed the core CORRECT (horizontal `min(dx,dy)` length scale right for the semi-implicit-vertical config; refactor behaviour-preserving; dry `c_s` EXACT for the dry-EOS dycore) but flagged the threshold (RISK); round 2 → tightened `_LES_ACOUSTIC_CFL_MAX` 1.5→**1.0** (the documented-safe `C_a<1` region; the real-LES test runs at 0.57) + added a **1.05 convective-warming margin** (rest `c_s` under-estimates run-time `c_s` ~5% as the column warms) + clarified the message is ACOUSTIC-only; round 3 caught **2 missed `run_forced_les` CLI callers** with unsafe `--les-dt`/`--dt` defaults (1.0 → blow up; + the OSSE CLI's 20.0 AND its pre-existing BROKEN import `from rce_diagnostics import run_forced_les`, fixed to `column_les`). EXHAUSTIVELY audited all 3 `run_forced_les` CLIs → all now default to the CFL-safe **0.5** (C_a≈0.58). Tests: 5 new (sound-speed bound, horizontal-acoustic uses dx-not-dz + substep scaling + n<1 guard, run_forced_les rejects an unstable dt, the warming-margin rejects a computed marginal dt, CLI `--les-dt` default 0.5); the real-dycore LES (dt=0.5) still passes the guard + runs. 51 cfl+campaign + the LES-CFL + OSSE tests green; sources ruff-clean. **A misconfigured LES timestep now FAILS LOUDLY before the run instead of blowing up partway through a multi-day campaign.** | 103 |
> | **Make the CLI wiring testable: shared `_campaign_knobs_from_args` + parser/forwarding tests** (iter 102 — closes a 20-arg untested HPC entry point, the exact bug class Codex caught in iter 101) | The HPC user's launch path — `_build_arg_parser` (20+ args) + `main()`'s arg→`build_*` mapping — was `# pragma: no cover` and UNTESTED; iter 101 already proved a forgotten/inverted flag silently breaks a multi-day run. Both CLI build calls (single + multi) DUPLICATED ~9 arg→kwarg mappings incl. boolean NEGATIONS (`--keep-dry-rounds`→`stop_on_no_valid_diagnoses=False`, `--allow-unphysical-coeff`→`clip_to_bounds=False`, `--keep-worsening-rounds`→`accept_only_if_improved=False`) + a `--step-fractions` CSV parse. EXTRACTED a pure shared `_campaign_knobs_from_args(args)` (the ONE place the flags map; both call sites now `**`-splat it so they cannot drift) and splatted it into both `build_correction_campaign`/`build_multi_correction_campaign` calls (no duplicate kwargs; `n_worst` stays out — it routes to `compose_compare_fn`, a different hop). **Codex 2-round review: round 1 CLEAN on 5/6 axes (negation polarity, CSV parse incl. whitespace, every dest a real parser arg, les_config split, no duplicate/missing kwarg) with one INFO test-gap — the subset test proved CLI→builder but not builder→run_*; round 2 CONFIRMED the added forwarding test sound + non-vacuous (sentinels non-default so a hardcoded default mismatches; module-top vs function-scope monkeypatch targets correct).** Tests: 5 new — parser defaults (abort/clamp/gate ON, mode=amip), required-args + bad-choice `SystemExit`, the knob negation/parse mapping (default + all-flags-flipped), `set(knobs) ⊆ BOTH builders' signatures` (the guard that would have caught the iter-101 drop), and a monkeypatch forwarding test pinning builder→`run_*` for all 8 run-bound knobs. 35 non-MPI campaign green; ruff-clean. **The full CLI chain (flag → parser → knobs → builder → loop) is now unit-tested, so a wiring regression fails in CI, not on a multi-day HPC launch.** | 102 |
> | **Dry-LES early-abort + restore x64 test precision** (iter 101 — builds on iter 100's validity count; also fixes 3 pre-existing-on-HEAD broken tests) | TWO pieces. (a) BRANCH-HEALTH: `tests/unit/test_correction_loop.py` had 3 FAILING clip-to-bounds tests on HEAD (`float(np.float32(1.2))=1.2000000476 > 1.2`) — the file ran float32 (no x64) while the correction loop is a SCIENTIFIC path; enabled `jax.config.update("jax_enable_x64", True)` (matching the sibling `test_run_correction_campaign.py`) so the registered-bound clamp is at production precision (verify-first: confirmed it's a float32-representation artifact, NOT a clip bug; no source change). (b) FEATURE: a DRY-LES early-abort — `run_correction_campaign` + `run_multi_correction_campaign` gained `stop_on_no_valid_diagnoses=True`; once `patience` consecutive rounds FLAG worst columns but produce ZERO globally-valid LES diagnoses (the spin-off LES develops no turbulence — too short/unforced), the campaign STOPS with `stop_reason="no_valid_diagnoses"` instead of wasting multi-day rounds. COLLECTIVE-SAFE: the dry test is on the GLOBAL corrected + valid counts via `_global_count` (allreduce SUM) — every rank breaks identically (a rank-local count would diverge + deadlock, iter-88/98 lesson); checked BEFORE the bias_tol convergence stop so the cause is reported correctly; a no-op round (no columns flagged) is NOT dry. **Codex 2-round review**: round 1 confirmed the loop logic collective-safe + correct but found a MEDIUM threading gap — `build_correction_campaign`/`build_multi_correction_campaign` + the CLI did not forward the flag (distributed/CLI callers couldn't opt out); FIXED (both builders forward it, distributed wrappers pass it via `**campaign_kwargs`, NEW CLI `--keep-dry-rounds`) + a false-green collective-safety test replaced with a `global_reduce=λx:x+1` (peer-rank-has-valid) test that genuinely distinguishes GLOBAL from rank-local counting; round 2: CLEAN on all fixes. Tests: 7 new (single + multi dry-abort, no-abort-when-valid, disable flag, no-op-not-dry, global-vs-local + globally-dry); a pre-existing convergence test was found ACCIDENTALLY dry (default `eddy_diffusivity` is invalid for the uniform-θ mock) and fixed to use `clubb_coefficient` (valid) so it tests convergence not the abort. 113 non-MPI campaign + 68 correction-loop (now x64) + 4 np=2 MPI green. **A campaign whose spin-off LES never develops turbulence now ABORTS early with the true cause instead of silently burning HPC time.** | 101 |
> | **Surface the LES-diagnosis VALIDITY rate (clause-5 diagnostics) + `no_valid_diagnoses` health verdict** (iter 100) | On a real run many worst-column LES spin-offs fail the realism gate (too short to develop turbulence) → `valid=False` → no correction, so a user saw `n_diagnosed=N` with no bias improvement and could NOT tell 'LES too short' from 'correction logic broken'. NEW `feedback_assembly.count_valid_diagnoses(diagnoses, method)` + `count_valid_multi_diagnoses(diagnoses, methods)` reuse the canonical `reduce_column_diagnosis` validity (≥1 valid level + finite — the EXACT criterion `assemble_feedback_field` uses). `CorrectionResult`/`MultiCorrectionResult` gain `n_diagnoses_valid` (counted over the LES RUNS, so ≤ `n_diagnosed` even when env-clustering expands K runs across more columns); `_diagnose_columns` now also returns the representative `run_diagnoses`. `CampaignSummary` gains `n_diagnosed_total`/`n_diagnoses_valid_total` (+ `report()` 'X/Y valid'), and `campaign_health` gains a `no_valid_diagnoses` status (checked before `clamp_limited`/`stalled` — with no valid diagnosis nothing is corrected so the clamp can't be the cause) telling the user to lengthen/force the LES. **Codex 2-round review**: round 1 found (a) clustered double-count — `n_diagnoses_valid` counted expanded COLUMNS (could exceed `n_diagnosed`) → fixed to count LES RUNS; (b) the multi-path counted validity from non-spec EXTRA methods → fixed to only the spec-required `methods`; round 2: CLEAN (added the sharp negative clustered test: one invalid rep ⇒ `n_diagnoses_valid=1 < n_diagnosed=2`). Tests: count helpers (incl. extra-method exclusion), the two result fields, clustered run-level counting, summary totals + 'X/Y valid' report, `no_valid_diagnoses` status + its precedence over clamp + the 'some valid' negative, np=2 e2e `n_diagnoses_valid==1`. (Pre-existing-on-HEAD float32 clip-bounds failures + N815/UP035 lint are unrelated, untouched.) **A campaign that makes no correction because the LES never developed turbulence now SAYS SO.** | 100 |
> | **Geostrophic completion + spectral restart + distributed-campaign hardening (iters 90–99)** — COMPRESSED (header narrates; git log has per-iter detail) | Gaussian geostrophic forcing + promoted shared `gaussian.spectral_gradient_3d` (90); cubed-sphere geostrophic investigated → Codex NO-SHIP → reverted with the exact contravariant-metric diagnosis (91) → METRIC-CORRECT 2×2 basis-solve `_gradient_cubed_geographic_3d` CONVERGES incl. panel edges ⇒ geostrophic on ALL 4 grid families (93); one-shot CLI spectral-CHECKPOINT restart loader (`reconstruct_spectral_state_from_npz`, x64/tracer-set/shape guards) (92); guardrail-harness AUDIT caught + fixed 2 formula-reimpl violations — NEW `physics._shared.exner_to_pressure` + `brunt_vaisala_n_squared_from_gradient` (bit-identical) (94). DISTRIBUTED-CAMPAIGN HARDENING (all Codex-reviewed, mpirun -np 2-validated): symmetric multi-coefficient wrapper `build_distributed_multi_correction_campaign` via shared `_distributed_campaign_kwargs` (95); ONE-CALL runnable entry point `build_distributed_mpas_campaign` (MPI-aware partition→local-mesh setup) (96); fail-fast mesh-identity guards on the distributed reference/area slice (no silent JAX gather-clamp; exact `nCells_global`) (97); collective pre-flight `assert_partition_covers_global` (owned sets tile the mesh once, collective-safe after a HIGH-deadlock fix) (98); fail-fast ERA5-reference physical-plausibility pre-flight `validate_reference_physical` (units/sign guard) (99) | 90–99 |
> | **MPAS + spectral LES spin-off + distributed-MPAS campaign (iters 73–89)** — COMPRESSED (header narrates; git log has per-iter detail) | MPAS/Voronoi extractor (73, `extract_column_forcing_voronoi`, TRiSK+Perot) + compare-side cell-wind reconstruction (74, `column_state_from_hydrostatic(mesh=)`) + ERA5→MPAS-cell regridder (75, `era5_to_mpas_carry`) + `make_les_diagnose_fn` u_edge caller wiring (76) + MPAS campaign CAPSTONE + env-kernel-producer crash fix (77–78) + multi-coeff/restart/CMIP×MPAS coverage (79–80) + C_K exact-inverse round-trip (81) + Voronoi geostrophic (82) + Gaussian/spectral extractor (83, `extract_column_forcing_gaussian`) + spectral compare-side handoff (84, `grid_winds_from_spectral`) + spectral campaign CAPSTONE (85). DISTRIBUTED-MPAS: owned-cell mask (86, `owned_cell_valid_mask`) + cross-rank global top-k (87, `distributed_manifest`) + COLLECTIVE correction loop (88, `global_reduce=global_sum_mpi`, no-deadlock gate/line-search/accept) + the campaign DRIVER (89, `build_distributed_correction_campaign`). All Codex-reviewed; MPI paths `mpirun -np 2`-validated end-to-end. LES spin-off works on ALL 4 grid families; the distributed-MPAS correction campaign runs end-to-end under real MPI | 73–89 |
> | **Deploy + cross-resolution + OSSE + conservation (iters 51–72)** — COMPRESSED (header narrates; git log has per-iter detail) | third coefficient C_eps (51); the saved-JSON DEPLOY back into a fresh run (`deploy_correction.corrected_turbulence_override`, grid-identity-guarded) (57–58); perfect-model identical-twin OSSE — the loop LOWERS the bias AND RECOVERS the known coefficient, single + simultaneous multi (59–60); distributed/MPI FEEDBACK sharding across lat-lon/cubed/MPAS (`turbulence_config_for` + `active_column_layout`, real-mpirun) (61–64); cross-resolution env-kernel DEPLOY (`EnvKernel` / `apply_env_kernel_override`, grid-agnostic by env similarity + in-hull diagnostic) (69) + its PRODUCER wiring (70); conservation gate on the param-update deploy (flux-form ⇒ identical column integrals + synthetic-violation self-test) (71); cross-resolution OSSE (`run_cross_resolution_osse` — the first MEASURED bias-reduction from the env-kernel transfer, in the twin) (72); + the LES-realism RH-supersaturation cap (68) and validation hardening. All Codex-reviewed | 51–72 |
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
> - **Extractor follow-ups** — Voronoi extractor DONE (iter 73); remaining:
>   cubed-sphere + Voronoi geostrophic (metric-correct east/north↔grid rotation,
>   visually verified for cube-edge artifacts — the Voronoi Perot gradient is
>   already geographic so it is low-risk there); the upstream MPAS-state→u_edge
>   handoff (compare-side cell-velocity reconstruction); the Gaussian/spectral
>   extractor (the dispatcher still raises on Gaussian).
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
- **Conservation diagnostics** after the feedback correction. **DONE (iter 71):**
  an analytic flux-form gate (`test_clubb_lite_ck_promotion.py`) proves a per-column
  C_K/Pr_t deploy leaves the mass-weighted column-integrated q/u/v/θ tendencies
  unchanged (it cannot leak mass/moisture/momentum/energy), with a
  synthetic-violation self-test.
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
