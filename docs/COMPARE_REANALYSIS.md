# Compare-to-Reanalysis + LES-Informed Column Correction

**Branch:** `feat/compare-reanalysis`
**Status (compressed at iters 50 + 60 + 70 + 80 + 90):** The whole offline loop is built, tested + Codex-reviewed, and **structurally complete** end-to-end — every stage `run (AMIP/CMIP) → time-mean → compare to ERA5 → rank → cluster → LES-diagnose → feedback → inject (turbulence_override) → RE-RUN` is real and tested (capstone iter 37; the injection provably changes the simulation, iter 35; feedback↔physics column ordering locked, iter 36). A **runnable HPC entry point** exists — `scripts/run/run_correction_campaign.py` (`make_base_driver_builder` for AMIP **and** CMIP, real-ERA5 reference loading, per-column `clubb_lite` C_K/Pr_t/C_eps correction (single or SIMULTANEOUS multi-coefficient), per-round bias report, self-describing summary+health verdict). The campaign's saved JSON now **deploys back into a fresh production run** via `legoesm.training.deploy_correction.corrected_turbulence_override` (iter 57, grid-identity-guarded iter 58) — the literal "update the parameters in the AMIP/CMIP simulation" plumbing is closed and `validate_strict`-checked. A **cross-resolution (grid-AGNOSTIC) deploy** path also exists end-to-end: a campaign run with `--feedback-strategy environment` now EXPORTS a `<out>.env_kernel.json` (the RAW env→coefficient regression, iters 69–70 producer wiring) that `apply_env_kernel_override` re-evaluates on ANY target grid by environmental similarity (so a cheap low-res campaign deploys on an expensive high-res run), with a coverage/in-hull domain-shift diagnostic. A **perfect-model (identical-twin) OSSE harness** (`legoesm.training.perfect_model_osse`, iters 59–60) demonstrates clause 5's LOGIC in a controlled twin — the model's own run with a KNOWN coefficient is the pseudo-truth, and the loop is checked to LOWER the bias AND RECOVER the known parameter (single AND simultaneous multi-coefficient C_K+Pr_t+C_eps) — the cheap go/no-go an HPC user runs before the real-ERA5 campaign. A **cross-resolution OSSE** (iter 72, `run_cross_resolution_osse`) extends this to the grid-AGNOSTIC env-kernel deploy: it LEARNS the env→coefficient correction on a coarse grid and DEPLOYS it on a finer grid by environmental similarity, demonstrating (in the twin) that the kernel TRANSFERS and lowers the finer grid's bias — the first measured bias-reduction from the iter-69/70 deploy, gated by an out-of-hull (extrapolation) trust check. The per-iteration table below records each stage's module + tests (compressed at iters 10/20/30/40/50/60/70/80, folding iters 1–50 into summary rows; full detail in git log). **Remaining (the done-criterion):** ONLY the EMPIRICAL demonstration — execute the multi-day campaign on real ERA5 at HPC scale and observe the LES-informed coefficients *lower* the bias (not unit-testable here, not honestly fakeable; the result is genuinely unknown until run). **Breadth — ALL DONE through iter 90:** the LES spin-off works on ALL 4 grid families (lat-lon, cubed-sphere, MPAS/Voronoi, Gaussian/spectral — extractors + compare-side handoffs + capstones, iters 73–85); distributed/MPI FEEDBACK sharding is real-mpirun-validated on all 3 structured families (iters 61–64); and the **distributed-MPAS LES-correction campaign now runs END-TO-END under real MPI** (owned-cell mask 86 + cross-rank global top-k 87 + collective correction loop 88 + `build_distributed_correction_campaign` 89, all `mpirun -np 2`-validated). Geostrophic LES-forcing wind is supplied on lat-lon + Voronoi + Gaussian (iter 90, via the shared `spectral_gradient_3d` for the spectral grid). **Remaining (minor breadth):** cubed-sphere geostrophic — iter 91 investigated + Codex-rejected a naive `grid.angle` rotation (it omits the covariant→contravariant metric inverse `G^x=(g_x−c·g_y)/(1−c²)`, `c=ê_x·ê_y` from `1−area/(dx·dy)`, → a non-convergent ~10–17% near-edge error in a Coriolis-reference wind); the EXACT fix (metric-correct gradient OR an orthogonality-deficit cell gate) is now scoped, so cubed stays on f×V meanwhile. The one-shot CLI spectral-CHECKPOINT restart load is now DONE (iter 92, `load_restart` reconstructs the spectral keys via `reconstruct_spectral_state_from_npz`). The remaining breadth is just the cubed-sphere geostrophic (the metric-correct gradient).
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
> | **One-shot CLI spectral-CHECKPOINT restart loader** (iter 92 — `load_restart` now reconstructs a spectral run's checkpoint offline) | The offline compare CLI (`compare_amip_era5.py`) could not load a `discretization='spectral'` run's restart: `ModelDriver.save_checkpoint` writes the five `*_hat` complex coefficient arrays + a `spectral_layout` marker via `np.savez` (NOT the grid layout `load_checkpoint_auto` reads), so `load_restart` failed. NEW shared `spectral_pe.reconstruct_spectral_state_from_npz(d, template=)` (the SINGLE canonical reconstruction): `template=None` → plain `Field`-wrapped coefficients (offline), `template=self.state` → reuse the configured Field metadata + validate shapes + refuse to drop water (in-driver). `load_restart` gained a `spectral_layout` branch (returns the `SpectralHydrostaticState` + grid-space q_v + step/day; the CLI's existing `grid_winds_from_spectral` then synthesizes grid winds), and the ModelDriver's in-driver spectral load was DEDUP'd onto the same helper (bit-exact restart-continuation test still green). **Codex 3-round review (round 1: 8 findings → round 2: 2 MEDIUM → round 3: CLEAN-TO-SHIP)**: HARDENED — an x64 guard (complex128 coeffs silently downcast to complex64 with x64 off → raise), byte-string tracer-name decode, EXACT tracer key-set match (no silent add/drop water in EITHER direction), per-tracer shape validation, strict coeff-shape validation vs the grid's `n_sh`/sigma `nlev` (a non-spectral grid → raise), NpzFile `with`-closed. Tests: 11 units (no-template/template reconstruction, shape + tracer-set + dtype + byte-name + non-spectral-grid guards, `load_restart` spectral branch) + the bit-exact ModelDriver spectral restart. 24 spectral/restart/federation/private-import green (x64); ruff-clean (zero net new errors; the dedup net-removed code). **The one-shot CLI is now grid-general (loads + compares a spectral restart).** | 92 |
> | **Cubed-sphere geostrophic: investigated → Codex NO-SHIP → reverted with a precise metric diagnosis** (iter 91 — verify-first prevented a wrong reference wind) | Attempted the LAST grid family's geostrophic forcing by rotating the cubed grid-axis Φ gradient to geographic via `grid.angle` + the tested `rotate_winds_grid_to_geo`. Numerically characterised it: for Φ=C·sin(lat) the rotated EAST component (true 0) and the north MAGNITUDE both carry a SYSTEMATIC, NON-convergent ~10–17% error near panel edges/corners (median <0.5%) — 10.5%→11.8% as n=24→96, ~23% of interior cells >2%. **Codex adversarial review = NO-SHIP (revert)**, decisive root cause: `gradient_x_3d`/`gradient_y_3d` return COVARIANT directional derivatives, so on the NON-orthogonal equiangular grid the metric-inverse step is missing — `G^x=(g_x−c·g_y)/(1−c²)`, `c=ê_x·ê_y` — and the naive rotation drops the `c/(1−c²)` cross-coupling. Unlike an advection tendency (one diffuse channel), a geostrophic REFERENCE wind feeds the LES plane Coriolis DIRECTLY (a persistent `f·ΔV_geo` departure biasing the whole momentum budget), so the error is NOT tolerable. REVERTED to `u_geo=v_geo=None`; the regression guard + code comment now record the EXACT fix (the contravariant inverse; `c` derivable from `1−area/(dx·dy)`) + the alternative (a calibrated orthogonality-deficit cell gate) — a far more actionable "blocked" status than the prior vague "needs metric rotation". Meaningful work = the rigorous correctness investigation that stopped a wrong geostrophic shipping (CLAUDE.md verify-first / no-silent-coerce / cubed-metric-first). 11 cubed + 35 extractor green (x64); ruff-clean. | 91 |
> | **Gaussian geostrophic forcing + shared spectral gradient** (iter 90 — the spectral LES column now gets a geostrophic reference wind) | The 4th-grid-family extractor previously deferred the geostrophic wind (`u_geo=v_geo=None` → the column LES plane Coriolis fell back to f×V); now `extract_column_forcing_gaussian` supplies it via the shared `_geostrophic_wind_column(grad_fn=_gradient_gaussian_3d)` (like the Voronoi extractor iter 82), except within the equatorial cutoff. To avoid DUPLICATE gradient numerics, PROMOTED the dycore's private `spectral_nh._spectral_gradient_3d` to a public `legoesm.grids.gaussian.spectral_gradient_3d` (the geographic east/north SH gradient `dfdx=(1/(a cosφ))∂/∂λ`, `dfdy=(1/a)∂/∂φ`, sibling of `uv_from_vordiv_3d`) + dedup'd spectral_nh onto it; `_gradient_gaussian_3d` is a thin `sh_analysis_3d → spectral_gradient_3d` wrapper. **Codex 2-round review (round 1: NO bugs, flagged the duplicate-numerics — my first Helmholtz draft equalled the private gradient to ~3e-23 ⇒ CLAUDE.md requires promotion → round 2: CLEAN)**. Tests: the promoted operator analytic (Φ=C·sinφ → dfdx=0, dfdy=(C/a)cosφ to rtol 1e-8); Gaussian geostrophic anchors (uniform state → wind EXACTLY 0 atol 1e-9; meridional T gradient → finite predominantly-ZONAL jet >1 m/s; equatorial column → None). 13 Gaussian + 23 spectral_nh dycore + vector-calc green (x64); ruff-clean; the iter-85 capstone (equatorial bias column) unaffected. **Geostrophic forcing now on lat-lon + Voronoi + Gaussian (cubed-sphere still f×V — grid-axis operators need a metric rotation).** | 90 |
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
