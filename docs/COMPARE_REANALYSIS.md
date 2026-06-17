# Compare-to-Reanalysis + LES-Informed Column Correction

**Branch:** `feat/compare-reanalysis`
**Status (iter 20):** Whole pipeline built, tested + Codex-reviewed, and composed into the closed-loop orchestrator (`correction_loop`, iter 19) — every stage from AMIP↔ERA5 compare → worst-column manifest → forced column-LES → closure-coefficient diagnosis → feedback field → applied to a real production scheme (gray `tau_equator`) → bias-improvement measurement. **Remaining:** the empirical bias-reduction demonstration from an HPC-scale run (real `compare_fn`/`diagnose_fn` on real ERA5) — not unit-testable here — plus AMIP/CMIP `diag_days=0.25` run wiring and per-grid extractor follow-ups.
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
> | 4 / #3 grid-side forcing extractor (lat-lon) | `atmosphere/dynamics/column_large_scale_extract.py` (`extract_column_forcing_latlon`: ω from continuity, −V·∇θ/−V·∇q advection → `ColumnLargeScaleState`) | 10 |
> | 4 / #4 1.5-order TKE SGS closure | `atmosphere/dynamics/tke_sgs_plane.py` (Deardorff/Lilly: ν_t=C_k ℓ√e, ε, Pr_t, TKE tendency, equilibrium↔Smagorinsky `C_s=(C_k³/C_ε)¼≈0.19`) | 11 |
> | 5 LES regime selection | `atmosphere/dynamics/les_regime.py` (`les_resolution_for_column`: CAPE→shallow/deep dispatch + per-regime plane-LES resolution; raise on unknown/non-finite/invalid box) | 12 |
> | 5 LES vertical mapping | `atmosphere/dynamics/les_vertical_mapping.py` (`interpolate_column_to_les`, `top_relaxation_rate` reusing `sponge_profile`, `relaxation_tendency`, `build_top_relaxation`) | 13 |
> | 6 LES→coefficient composition | `atmosphere/dynamics/column_les_diagnosis.py` (`diagnose_column_coefficient`: dispatch eddy-K / entrainment-w_e; top-down→ascending reversal of fluxes+gradient) | 14 |
> | 4–6 column-LES driver | `scripts/run/run_column_les.py` (`build_column_les_setup`→`run_forced_les`→`diagnose`; `process_column`/`extract_gcm_column`/`run_column_les_pipeline`; real `main` loops the manifest) | 15 |
> | 7 LES diagnoses→feedback field | `training/feedback_assembly.py` (`reduce_column_diagnosis` K-profile→scalar / w_e; `assemble_feedback_field` scatters at worst-column flat indices) | 16 |
> | verify bias↓ (success metric) | `training/bias_metrics.py` (`aggregate_combined_bias` area-weighted; `bias_improvement` baseline-vs-updated; `worst_column_bias_change`) | 17 |
> | 7 per-scheme promotion | `training/promotable_params.py` (`apply_feedback_to_scheme`: registry of (ncol,)-capable coefficients; gray `tau_equator`/`tau_pole` promoted end-to-end, NO body change) | 18 |
> | CLOSED LOOP orchestrator | `training/correction_loop.py` (`run_correction_iteration`: compare→LES-diagnose→assemble→apply→re-compare→`bias_improvement`; heavy AMIP/LES steps injected; empty-manifest no-op) | 19 |
> | REAL-dycore integration test + moist-IC fix | `tests/run/test_run_column_les.py::test_process_column_real_dycore_integration` runs the actual plane-NH LES (mock-free); caught + fixed `run_forced_les` carrying no moisture (now seeds GCM q_v on the LES grid → `ColumnLESSetup.q_v_init`) | 20 |
> | real-comparison loop adapter | `correction_loop.make_compare_fn` wraps the REAL `compare_state_to_reference` into the loop's `compare_fn` (only the AMIP/CMIP model run injected); loop now closes with real scoring+manifest, not a mocked score field | 21 |
> | AMIP **and CMIP** mode bridge | `compare_reanalysis.column_state_from_hydrostatic` (driver `HydrostaticState`→`ColumnState`, unwraps `Field`s; mode-agnostic, SST = prescribed/coupled); fixed a latent `Field`-not-unwrapped bug also in iter-4 `model_state_from_restart` (now delegates) | 22 |
> | REAL coupled-run (CMIP) → compare smoke | `tests/run/test_cmip_compare_integration.py` runs a real `CoupledESMDriver` (slab-ocean aquaplanet, C8/L5, 1 day) and feeds its actual atmosphere state + coupled SST through the real compare → finite scores + worst-column manifest (the AMIP/CMIP half, mock-free; complements iter-20's real-LES half) | 23 |
>
> **Feedback loop now closes in code:** LES diagnoses → `assemble_feedback_field`
> (iter 16) → `apply_column_parameter_field` (iter 9) → updated scheme config.
>
> **Remaining to reach the done-criterion (AMIP/CMIP vs reanalysis → LES → params → bias↓):**
> 1. **Extractor follow-ups** — geostrophic wind (∇Φ) + cubed-sphere/Gaussian
>    grids (iter-10 covers lat-lon ω + advective tendencies, ERA5-native).
> 2. **Standalone LES driver** — DONE (iter 15): `scripts/run/run_column_les.py`
>    chains manifest→regime→θ/forcing interp→`run_forced_les`→coefficient. The
>    forced plane-dycore run loop (`run_forced_les`) + `main` are real but
>    HPC-validated, not unit-tested (the orchestration helpers are). Optional
>    follow-up: wire the iter-11 TKE closure into the dycore as the SGS option.
> 3. **Per-scheme promotion** — DONE (iter 18) for the cleanest case: gray
>    `tau_equator`/`tau_pole` accept a per-column field with NO body change
>    (already `(ncol,)`-vectorised), applied via the `promoted_fields` allowlist
>    (no `__param_spec__`/collector change). A convection-entrainment or
>    turbulence-diffusivity promotion follows the same pattern once that scheme
>    body is wrapped in a column-broadcast (the physically-targeted follow-up).
> 4. **AMIP/CMIP run wiring** (`diag_days=0.25`) + a tiny real end-to-end smoke.
> 5. **End-to-end bias-reduction demo** — the actual success criterion. The
>    *orchestrator* now exists (`correction_loop.run_correction_iteration`, iter
>    19) and is unit-tested with mocks (a correction that lowers worst-column
>    scores yields `improved=True`); the remaining work is supplying the real
>    HPC-scale `compare_fn` (AMIP run + `compare_amip_era5`) and `diagnose_fn`
>    (`run_column_les.process_column`) and running it on real ERA5 to produce
>    the empirical bias drop. That run is HPC-scale (not unit-testable here).
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
