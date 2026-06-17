# Compare-to-Reanalysis + LES-Informed Column Correction

**Branch:** `feat/compare-reanalysis`
**Status (iter 30, compressed):** Whole pipeline built, tested + Codex-reviewed, composed into the closed-loop orchestrator (`correction_loop`, iter 19/25) AND covered by a single FULL mock-free end-to-end gate (iter 26). Both lat-lon and native cubed-sphere worst columns spin off real plane LES (iter 27–28); forcing includes lat-lon geostrophic wind wired into the LES Coriolis (iter 29); iter 30–32 add the time-mean `ColumnState` accumulator + the run→time-mean→compare→score wiring (`run_to_column_mean` + `make_run_fn` composed with `make_compare_fn` → a real `compare_fn(config)`, integration-tested on a real coupled run); iter 33–34 add env-clustering of worst columns wired into the loop via `les_budget`; iter 35 lands the **build_driver config-injection** (`ExperimentConfig.turbulence_override` → corrected per-column `clubb_lite` C_K reaches all dycore backends; a real run with a changed C_K provably changes the simulation); iter 36 locks the **column-ordering contract** (feedback-flat order == physics `ColumnAdapter` order, real pipeline + real driver); iter 37 is the **CAPSTONE** — the full offline loop closes end-to-end with a REAL model re-run, proven (non-vacuously) to consume the LES-informed per-column C_K. **Remaining (the done-criterion):** ONLY the EMPIRICAL bias-reduction demonstration on real ERA5 at HPC scale — every structural stage (run→compare→rank→cluster→LES→diagnose→feedback→inject→**re-run**) is now real and tested; what's left is the multi-day run on real ERA5 where the LES-informed C_K actually *lowers* the bias (not unit-testable here, and not honestly fakeable) — plus distributed/MPI feedback sharding, cubed-sphere geostrophic rotation, and Gaussian/Voronoi extractors. (Per-iteration history compressed at iters 10/20/30; full detail in git log.)
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
> | run→time-mean→compare wiring | `run_to_column_mean.py` (`run_to_column_mean` samples the column state at each `diag_days` segment via the driver's `segment_callback`, folds into the iter-30 mean; `cmip_column_state`/`amip_column_state` extractors — AMIP threads the segment `day` for time-dependent prescribed SST). `CoupledESMDriver.run` gained an optional composed `segment_callback` + public `q_v` property (the real `run_amip_fn`/`run_cmip_fn` for `make_compare_fn`) | 31 |
> | real `compare_fn(config)` adapter | `run_to_column_mean.make_run_fn(build_driver, extract)` → `run_fn(config)` (config→fresh driver→time-mean ColumnState) = the real `run_amip_fn`/`run_cmip_fn` for `make_compare_fn`. Integration test composes it with `make_compare_fn` over a real tiny coupled run → a real `compare_fn(config)` that runs+time-means+scores vs a synthetic ERA5 reference, with a DOMINANCE-MARGIN worst-column assertion (biased column ≥2× runner-up). The last mock in the compare path is now real | 32 |
> | env-clustering worst columns (LES cost) | `column_clustering.py::cluster_columns_by_environment` — deterministic farthest-first (Gonzalez k-center) over the manifest's normalised env tags (SST/CAPE/shear), anchored at the worst-scoring column, picks K REPRESENTATIVE REAL columns + per-column labels so one LES per cluster covers many worst columns (§6/§7 cost path). Finiteness + env_scales validation, near-constant-feature suppression, duplicate-env early stop | 33 |
> | clustering WIRED into the loop | `run_correction_iteration`/`run_correction_campaign` gained `les_budget=K` (+`env_scales`): when `K < len(manifest)`, diagnose only the K env-representatives and map each coefficient to its cluster members before the feedback splice — so `n_corrected` columns corrected but only `n_diagnosed=K` (new `CorrectionResult` field) LES run. `les_budget=None` = unchanged (diagnose all). Tests: K diagnose calls (not N), n_diagnosed==K<n_corrected, cluster members get the RIGHT representative's coefficient (distinct CAPE-based K), None/≥N/≤0 paths | 34 |
> | **build_driver CONFIG-INJECTION** (the parameter-update mechanism) | `ExperimentConfig.turbulence_override` (a `TurbulenceConfig`, default None) + shared `physics_pipeline.turbulence_config_for(config)` used by ALL backends (FV `_resolve_turbulence`, MPAS, spectral) → the corrected `clubb_lite` config (incl. a **per-column C_K array** from the feedback field) reaches the turbulence kernel WITHOUT a new driver signature. `validate_strict` enforces `override.scheme==turbulence` + isinstance; serialization drops it (runtime-only). **PROVEN**: a real coupled run with different C_K (0.2 vs 1.2) gives a different T field, and a per-column (ncol,) C_K array runs a real forward step (finite output) — i.e. "updating these parameters in the AMIP/CMIP simulation" genuinely changes the simulation | 35 |
> | column-ordering CONTRACT guard | `test_feedback_column_ordering.py` LOCKS the invariant that the feedback field's flat (grid row-major) order == the physics `ColumnAdapter` flatten, so a per-column C_K lands on the RIGHT grid cell. Tests: end-to-end (manifest flat_index → `assemble_feedback_field` → `apply_feedback_to_scheme` → the diagnosed K sits at the adapter's column index for its grid cell, lat-lon + cubed shapes); feedback flatten == REAL `build_physics_pipeline(grid).adapter.flatten_2d` (lat-lon + cubed); a REAL coupled driver's compare `ColumnState` shape + elementwise `p_s` flatten == physics `adapter` (the compare-vs-physics orientation). Residual: distributed/MPI rank-local column sharding (separate MPI item) | 36 |
> | **CAPSTONE: full loop closes with a REAL re-run** | `tests/run/test_correction_loop_real_rerun.py` runs the ENTIRE loop with real components + the re-run: real coupled run → time-mean → `make_compare_fn` score vs synthetic-ERA5 reference (deterministic +6 K worst column, dominance-checked) → mock diagnose (real LES is iter 20/26) → `assemble_feedback_field` → `apply_feedback_to_scheme` (per-column clubb C_K) → `run_correction_iteration`'s 2nd `compare_fn` RE-RUNS the real model with that C_K via `turbulence_override`. NON-VACUOUS: captures the re-run's driver and asserts `driver._atm.physics.turbulence_config.C_K` == the (128,) per-column array (had the injection been ignored it would be scalar 0.4). The offline loop is closed end-to-end with real model runs; only the EMPIRICAL bias-sign (real ERA5 @ HPC) is unasserted | 37 |
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
