# Compare-to-Reanalysis + LES-Informed Column Correction

**Branch:** `feat/compare-reanalysis`
**Status (compressed at iters 50 + 60 + 70 + 80 + 90 + 100 + 110):** The whole offline loop is built, tested + Codex-reviewed, and **structurally complete** end-to-end — every stage `run (AMIP/CMIP) → time-mean → compare to ERA5 → rank → cluster → LES-diagnose → feedback → inject (turbulence_override) → RE-RUN` is real and tested (capstone iter 37; the injection provably changes the simulation, iter 35; feedback↔physics column ordering locked, iter 36). A **runnable HPC entry point** exists — `scripts/run/run_correction_campaign.py` (`make_base_driver_builder` for AMIP **and** CMIP, real-ERA5 reference loading, per-column `clubb_lite` C_K/Pr_t/C_eps correction (single or SIMULTANEOUS multi-coefficient), per-round bias report, self-describing summary+health verdict). The campaign's saved JSON now **deploys back into a fresh production run** via `legoesm.training.deploy_correction.corrected_turbulence_override` (iter 57, grid-identity-guarded iter 58) — the literal "update the parameters in the AMIP/CMIP simulation" plumbing is closed and `validate_strict`-checked. A **cross-resolution (grid-AGNOSTIC) deploy** path also exists end-to-end: a campaign run with `--feedback-strategy environment` now EXPORTS a `<out>.env_kernel.json` (the RAW env→coefficient regression, iters 69–70 producer wiring) that `apply_env_kernel_override` re-evaluates on ANY target grid by environmental similarity (so a cheap low-res campaign deploys on an expensive high-res run), with a coverage/in-hull domain-shift diagnostic. A **perfect-model (identical-twin) OSSE harness** (`legoesm.training.perfect_model_osse`, iters 59–60) demonstrates clause 5's LOGIC in a controlled twin — the model's own run with a KNOWN coefficient is the pseudo-truth, and the loop is checked to LOWER the bias AND RECOVER the known parameter (single AND simultaneous multi-coefficient C_K+Pr_t+C_eps) — the cheap go/no-go an HPC user runs before the real-ERA5 campaign. A **cross-resolution OSSE** (iter 72, `run_cross_resolution_osse`) extends this to the grid-AGNOSTIC env-kernel deploy: it LEARNS the env→coefficient correction on a coarse grid and DEPLOYS it on a finer grid by environmental similarity, demonstrating (in the twin) that the kernel TRANSFERS and lowers the finer grid's bias — the first measured bias-reduction from the iter-69/70 deploy, gated by an out-of-hull (extrapolation) trust check. The per-iteration table below records each stage's module + tests (compressed at iters 10/20/30/40/50/60/70/80, folding iters 1–50 into summary rows; full detail in git log). **Remaining (the done-criterion):** ONLY the EMPIRICAL demonstration — execute the multi-day campaign on real ERA5 at HPC scale and observe the LES-informed coefficients *lower* the bias (not unit-testable here, not honestly fakeable; the result is genuinely unknown until run). **Breadth — ALL DONE through iter 90:** the LES spin-off works on ALL 4 grid families (lat-lon, cubed-sphere, MPAS/Voronoi, Gaussian/spectral — extractors + compare-side handoffs + capstones, iters 73–85); distributed/MPI FEEDBACK sharding is real-mpirun-validated on all 3 structured families (iters 61–64); and the **distributed-MPAS LES-correction campaign now runs END-TO-END under real MPI** (owned-cell mask 86 + cross-rank global top-k 87 + collective correction loop 88 + `build_distributed_correction_campaign` 89, all `mpirun -np 2`-validated). Geostrophic LES-forcing wind is now supplied on ALL 4 grid families — lat-lon + Voronoi (iter 82) + Gaussian (iter 90, via the shared `spectral_gradient_3d`) + cubed-sphere (iter 93, the metric-correct 2×2 basis solve `_gradient_cubed_geographic_3d`, CONVERGING incl. panel edges after iters 91/92's naive attempts were Codex-rejected). The one-shot CLI spectral-CHECKPOINT restart load is DONE (iter 92, `load_restart` reconstructs the spectral keys via `reconstruct_spectral_state_from_npz`). The distributed-MPAS campaign is runnable for SINGLE + simultaneous MULTI-coefficient correction (iters 89/95) AND now has a ONE-CALL runnable entry point `build_distributed_mpas_campaign` (iter 96) that does the MPI-aware setup (partition the global mesh → wire the rank-local mesh as grid + driver) so an HPC user supplies only the global mesh + a `build_local_driver`. The distributed slice now FAILS LOUDLY on a global/reference/area mesh-count mismatch (iter 97 — exact `nCells_global` guards replacing a JAX gather's silent OOB-clamp), and a DEFAULT-ON collective pre-flight (`assert_partition_covers_global`, iter 98) verifies the rank partition covers the global mesh EXACTLY once (no silently-uncorrected or double-counted cell) before a multi-day run. A DEFAULT-ON physical-plausibility pre-flight (`validate_reference_physical`, iter 99) catches a units/sign error in the loaded ERA5 reference (T/p_s/q_v wrong units) before it drives the loop to 'correct' a fake bias. The campaign now reports the LES-diagnosis VALIDITY rate (`n_diagnoses_valid` per round, `n_diagnoses_valid_total` in the summary) and a `no_valid_diagnoses` health verdict (iter 100), so a run that makes no correction because the spin-off LES never developed turbulence says so instead of looking like a broken correction; it also EARLY-ABORTS (collective-safe, `stop_reason="no_valid_diagnoses"`, default-on, CLI `--keep-dry-rounds` to disable) after `patience` consecutive dry rounds rather than burning multi-day HPC time (iter 101). The CLI launch chain (flag → parser → shared `_campaign_knobs_from_args` → builder → loop) is now fully unit-tested (iter 102), so a wiring regression fails in CI rather than on a multi-day HPC launch. The spin-off LES now FAILS FAST on an acoustically-unstable timestep (iter 103, `run_forced_les` horizontal acoustic-CFL pre-flight reusing `cfl_diagnostic`; all 3 LES CLIs default to the CFL-safe `--les-dt 0.5`) instead of blowing up partway through a multi-day run. A CI import-resolution guard (iter 104, `test_cli_imports_resolve.py`) now catches a stale function-scope import in any of the 4 compare-reanalysis HPC CLIs (the OSSE-class dead-entry-point bug) before a launch, and the campaign→deploy JSON contract is round-trip-tested (iter 105, `build_campaign_output_dict` → `corrected_clubb_config`) so the on-disk format that updates a production run cannot silently drift. The real-ERA5 loader now FAILS LOUDLY on a missing/misnamed required variable (iter 106, `load_era5_slice` required-raise + bidirectional `resolve_var`) instead of silently feeding zeros into the bias, and the full real ERA5-input→reference chain (load→regrid→interp→carry→ColumnState) is now exercised end-to-end by an integration test (iter 107) instead of being bypassed by the compare test's monkeypatch. A SEVERE ERA5→cubed-sphere regrid bug (the input weights were built from a Gaussian PROXY of the uniform lat-lon ERA5 grid, mis-indexing source cells by up to ~154° of latitude) was found + fixed (iter 109, `compute_latlon_to_cs_weights` from the ACTUAL source nodes; content-fingerprinted weight cache; T=lat regression test) — the cubed-sphere real-ERA5 reference is now spatially correct. A follow-up audit (iter 110) verified the OTHER three ERA5→grid input paths: spectral (`regrid_latlon_to_gaussian`, scipy from actual coords, descending-lat OK) and MPAS (`era5_to_mpas_carry`, never a proxy, spatial-fidelity tested) are CORRECT; the MPAS weight CACHE had the same iter-109 key weakness and was re-keyed onto the shared content-fingerprint (collision/GC-safe). All four ERA5→grid input paths are verified spatially correct with collision/GC-safe weight caches, and their specific-humidity→mixing-ratio conversion is now factored onto the single canonical `thermo` helper (iter 111, which also fixed a latent float32 divide-by-zero in that helper), and the most-shared input numeric `interp_pressure_to_sigma` is locked by a real numerical regression suite (iter 112, replacing a vacuous constant-field test; verify-first confirmed the routine is numerically correct). The clause-3 worst-column ranking's cross-variable commensurability is locked too (iter 113 — a mis-set per-variable normalization that hid a variable from selection now fails in CI; verify-first confirmed the comparison/metric/LES-inverse/deploy code is correct and well-tested), and the LES-closure + ranking-metric unit tests are now deterministic at fp32 (iter 114 — no longer silently dependent on a cross-file x64 leak; xdist/isolation-safe). **All documented breadth is COMPLETE; the ONLY remaining item is the done-criterion's empirical real-ERA5 HPC demonstration.**
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
> | **Make the LES-closure + metrics unit tests DETERMINISTic at fp32 (kill the x64-leak dependence)** (iter 114 — branch-health, extends iter 108 across the clause-4 + clause-3 test files) | CI runs `tests/unit/` WITHOUT `JAX_ENABLE_X64` (the fp32-by-default tier; `ci.yml` documents the module-level x64 leak as a "known wart"). A float32-isolation sweep of the compare-reanalysis unit files found 7 tests asserting analytic LES-closure inverses (K_m recovery, `C_K=K_m/(ℓ√wp2)` exact inverse, Pr_t ratio, C_eps-from-budget, entrainment velocity) + a constant-offset RMSE with x64-ONLY tolerances (`rtol=1e-12`/`1e-9`, `rel=1e-12`, `abs=1e-10`). At fp32 these recover to ~1.3e-8–2e-7 relative, so the tests FAILED in isolation / under xdist and PASSED only via the session-wide x64 leak from a `test_correction_loop` import during collection — order-dependent / xdist-fragile (the exact class iter 108 fixed for one test). FIX: loosened only the 7 genuinely-failing tolerances to `rtol/rel=1e-5` / `abs=1e-5` (≈670× the measured fp32 noise, ≥1000× below any plausible formula regression); LEFT the 3 already-deterministic tight tolerances (gradient=0.5, K0=5, ℓ0=50 — exactly representable, residual 0.0) UNTOUCHED (minimal diff). **Codex adversarial review: CLEAN — "the change is sound, no blocking issues"; verified each loosened assertion still catches concrete bug classes (sign/projection/missing-term/broadcasting), the root cause is fp32 rounding NOT catastrophic cancellation (so loosening hides no model-fidelity problem), and no test is now vacuous.** Both files pass at float32 ISOLATION (46, was 7 failing) AND x64; no net-new ruff; `test_training_modules` (iter-112 interp tests) confirmed already fp32-clean. **The compare-reanalysis LES-closure + ranking-metric unit tests are now deterministic under xdist/isolation instead of silently depending on a cross-file x64 leak.** | 114 |
> | **Lock the cross-variable COMMENSURABILITY of the worst-column ranking score** (iter 113 — the clause-3 selection numeric: which columns get LES spin-offs) | Audited the COMPARISON side after the input-path sweep. Verify-first CONFIRMED correct + well-tested (no source bug): the mass weights are a p_s-independent mass FRACTION (`dsigma/Σdsigma`, identical for model+reference — no surface-pressure confound; ERA5-reference-on-its-own-p_s is the documented methodology); `score_columns` normalizes each variable by a config scale before the weighted RMS (reuses `safe_sqrt`/`weighted_rmse` from `scm_rce_metrics` — no duplication); NaN-gradient safety (all-invalid column, zero-error `sqrt`) is already tested; the LES→coefficient inverse (`momentum_diffusivity_from_fluxes` least-squares projection, `clubb_coefficient_from_diffusivity`) has 28 analytic tests; the deploy slicing is reassembly-tested. GAP FOUND: the per-variable normalization makes T/qv/wind COMMENSURABLE (a 3 K / 1.5e-3 kg·kg⁻¹ / 5 m·s⁻¹ error each → identical combined score √⅓), but NO test exercised qv or wind DRIVING the ranking — a mis-set `*_norm` (e.g. `qv_norm` 1.5e-3→1.0) would silently make qv invisible to worst-column selection and nothing would catch it. Added 2 contract tests: commensurability (each one-scale error → equal √⅓ score) + qv/wind ranking-participation (large qv/wind errors outrank a tiny-T column). **Codex 2-round adversarial review**: round 1 FAIL — the tests injected the error AS `cfg.*_norm` and divided by the same field, so the ratio CANCELED → vacuous against a norm drift (the exact bug they claimed to catch); FIXED by locking the default scales (explicit value-lock asserts) + injecting HARD-CODED absolute errors (proven: mis-setting `qv_norm`→1.0 now drops the qv score to 8.7e-4≠√⅓ → fails); round 2 confirmed test 1 non-vacuous for all 3 norms + math correct, flagged a test-2 docstring OVERCLAIM (ranking alone only catches >~50× drift; the value-lock is the real guard) → docstring corrected to state the ranking catches a FORMULA regression (variable dropped from the combine) while the value-lock catches constant drift. 20 metrics tests green at x64; no net-new ruff. **The clause-3 worst-column ranking's cross-variable commensurability is now locked: a mis-set normalization that hid a variable from selection fails in CI.** | 113 |
> | **De-vacuum the vertical-interpolation tests — real numerical regression suite for `interp_pressure_to_sigma`** (iter 112 — the single most-shared ERA5-input numeric had only happy-path coverage) | `interp_pressure_to_sigma` (log-p interpolation of ERA5 pressure-level data onto model sigma levels) is called by ALL FOUR grid carries + the NMC DA path, but its only tests asserted SHAPE and DIFFERENTIABILITY on a CONSTANT field (T≡250) — vacuous: a constant field hides bracketing, log-p-vs-linear-p, axis-swap, and extrapolation-direction bugs (any interpolator passes). Verify-first NUMERICAL audit first CONFIRMED the function is correct: a log-p-linear field f=a+b·ln(p) is reproduced EXACTLY at interior targets (err 0.0 at x64), both hold-constant extrapolations (above model top → f[0]; below surface → f[-1]) are correct, gradients finite. (No source bug; the routine is sound. A fail-loud ascending-plev precondition was considered + rejected: the routine is intentionally `jax.grad`-able and all callers are setup-time, so a Python assert on a possibly-traced coordinate would be the wrong trade.) Then REPLACED the vacuous coverage with 7 contract tests: log-p exactness (interior), exact reproduction AT source levels, hold-constant ABOVE top + BELOW surface, per-column p_s + DISTINCT per-column profiles (catches a take_along_axis target-broadcast AND source column-mixing bug), monotone no-overshoot, and a gradient test asserting the EXACT log-p bracket weight α=(ln p_t−ln p_lo)/(ln p_hi−ln p_lo) on the two bracketing levels only (distinct from the linear-in-p weight by >0.05, so it pins LOG-p not linear-p; convex, sums to 1, non-bracket levels get zero gradient). **Codex adversarial review: all expected values mathematically CORRECT and tests NON-VACUOUS; 2 strengthening suggestions (per-column test used identical profiles → now distinct (a,b) per column; gradient test only checked convexity → now asserts the exact log-p weight) — both APPLIED.** 9 TestVerticalInterp green at float32 AND x64; no new ruff. **The most-shared ERA5-input numeric is now locked by a real numerical regression suite instead of a constant-field happy-path test.** | 112 |
> | **Factor the ERA5 specific-humidity→mixing-ratio conversion onto the canonical thermo helper + fix its float32 divide guard** (iter 111 — DRY/no-re-derived-numerics + a latent divide-by-zero in the canonical helper) | Verify-first audit of the ERA5 INPUT path's moisture handling. First confirmed the cubed-sphere WIND frame is correct (ERA5 geographic east/north → `HydrostaticState.u/v`, which the cubed-sphere dycore rotates to geographic at cell centres, `cubed_sphere.py:1379`, and the compare side assumes geographic — all frame-consistent; NOT a bug) and the q→r conversion is applied CONSISTENTLY across all four carries. BUT all four (`era5_to_{spectral,cubedsphere,mpas,latlon}_carry`) RE-DERIVED `q_specific=clip(...,0,0.99); r=q/(1−q)` inline — a CLAUDE.md "never re-derive shared utilities" violation: the canonical `legoesm.thermo.specific_humidity_to_mixing_ratio` already exists (and is used by `dephy_scm.py`). FIX: all four now call the canonical helper (removes the unjustified silent 0.99 clamp in favor of its divide-by-zero floor; `ml`→`core` import is federation-legal; behavior-identical for physical ERA5 q≤~0.04). **Codex 2-round adversarial review**: round 1 — NO production bugs (all 4 call sites correctly wired q/p_s/sigma, behavior-equivalent, negative-undershoot still floored to r=0, federation-clean) + ONE real LATENT bug it surfaced: the canonical helper's `denominator_floor=1e-12` is BELOW float32 precision near 1.0, so `1−1e-12` ROUNDS to `1.0` in float32 → corrupt q≥1 divides by zero → **inf** (confirmed: `q=1.0f32→inf`). Since the refactor routes 4 FLOAT32 ERA5 paths through this helper and the old inline 0.99-clamp was divide-safe, this was in-scope; FIXED in `thermo.py` by explicitly flooring the denominator `denom=jnp.maximum(1−q, floor)` (matching the helper's already-correct tendency sibling) → corrupt float32 q now gives a large-but-FINITE r. Round 2 CLEAN: both fixes correct, no new bugs, behavior-preserving for all physical inputs, the float32 no-inf test non-vacuous. Tests: NEW `tests/unit/test_humidity_conversions.py` (5 — formula, round-trip, negative→0, float32 divide-safety no-inf, jit+grad-safe) — the first DIRECT unit test of the canonical core helper — + a q→r exactness assertion in the lat-lon integration test. 43 humidity+era5+mpas+regrid+dephy green; touched files ruff-clean. **The ERA5 moisture conversion now uses ONE canonical helper across all 4 grids, and that helper is genuinely divide-safe at float32.** | 111 |
> | **Harden the MPAS ERA5→cell regrid weight cache (the same bug class as iter-109's cubed-sphere fix)** (iter 110 — also the 10-iter compression checkpoint) | Verify-first follow-up to iter 109: AUDITED the OTHER three ERA5→grid input paths for the same Gaussian-proxy / mis-indexing class. SPECTRAL (`regrid_latlon_to_gaussian`) is CORRECT — `scipy.RegularGridInterpolator` is built from the ACTUAL ERA5 coords and (verified) handles descending ERA5 latitude (scipy 1.17). MPAS (`era5_to_mpas_carry` via `compute_latlon_to_voronoi_weights`) is also CORRECT and well-tested for spatial fidelity (equator-pole, longitude-gradient no-dateline-scramble, descending=ascending) — it never used a proxy. BUT its weight CACHE (`_get_voronoi_weights`) still had the exact iter-109 key weakness: keyed on `(size, endpoints, id(mesh))`, which COLLIDES on same-extent/different-INTERIOR source grids and is `id(mesh)` GC-reuse unsafe. FIX: re-keyed onto the shared `_coord_fingerprint` (content `(shape, hash(tobytes))`) of both source coords AND `mesh.latCell`/`mesh.lonCell` — collision- and GC-safe, structurally identical to `_get_cs_weights`. **Codex review: CLEAN — "No concrete correctness bugs found"; tests non-vacuous, the interior-perturbation test genuinely catches the bug, consistent with the cubed-sphere path.** Tests: 3 new (content-hit, same-endpoint/different-interior non-collision, different-mesh separation) on top of the existing spatial-fidelity suite. 12 MPAS-carry green; ruff-clean. **All four ERA5→grid input paths are now verified spatially correct AND their weight caches collision/GC-safe.** | 110 |
> | **Real-ERA5 input-path + HPC-launch hardening (iters 100–109)** — COMPRESSED (header narrates; git log has per-iter detail) | Hardened the empirical-run path end-to-end, all Codex-reviewed: LES-diagnosis VALIDITY rate + `no_valid_diagnoses` health verdict (100); collective-safe DRY-LES early-abort + restored x64 test precision (101); testable CLI wiring via shared `_campaign_knobs_from_args` + parser/forwarding tests (102); fail-fast acoustic-CFL pre-flight on the spin-off LES, all 3 CLIs default to the CFL-safe `--les-dt 0.5` (103); static import-resolution CI guard for the 4 HPC CLIs, catching the OSSE dead-import (104); round-trip test of the campaign→deploy JSON contract via extracted `build_campaign_output_dict`, fixing a scalar-promotion that masked the deploy loader's rejection (105); fail-fast ERA5 loader — no silent-zeros on a missing required variable + bidirectional `resolve_var` (106); end-to-end integration test of the real `load_era5_slice`→`era5_to_latlon_carry`→`column_state_from_carry` chain the monkeypatch had hidden (107); fixed an order-dependent x64-leak flaky test in the compare suite (108); and the SEVERE cubed-sphere ERA5 regrid bug — weights built from a Gaussian PROXY of the uniform lat-lon ERA5 grid mis-indexed source cells by up to ~154° latitude → fixed with `compute_latlon_to_cs_weights` from the ACTUAL nodes + a content-fingerprinted, collision/GC-safe weight cache + a T=lat regression test (109). | 100–109 |
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
