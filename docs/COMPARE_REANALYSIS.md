# Compare-to-Reanalysis + LES-Informed Column Correction

**Branch:** `feat/compare-reanalysis`
**Status:** Implementation in progress — diagnosis→force→estimate→assemble→apply chain built + tested (iters 1–9); grid-side forcing extractor + LES driver + end-to-end demo remain.
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
>
> **Remaining to reach the done-criterion (AMIP/CMIP vs reanalysis → LES → params → bias↓):**
> 1. **Extractor follow-ups** — geostrophic wind (∇Φ) + cubed-sphere/Gaussian
>    grids (iter-10 covers lat-lon ω + advective tendencies, ERA5-native).
> 2. **Standalone LES driver** — orchestrate manifest→`les_resolution_for_column`
>    (iter 12)→column θ/q interp to the LES grid + `ColumnLargeScaleState` forcing
>    (iters 5/10)→run plane LES (`run_les_plane` machinery, optionally the iter-11
>    TKE closure wired into the dycore)→resolved-flux output (iter 6)→coefficient
>    (iter 7). Heavy run step; the config/selection pieces are now in place.
> 3. **Per-scheme promotion** of a real production coefficient (entrainment) —
>    `shape`-keyed `__param_spec__` + scheme-body `(ncol,)` broadcast (physics-validated).
> 4. **AMIP/CMIP run wiring** (`diag_days=0.25`) + a tiny real end-to-end smoke.
> 5. **End-to-end bias-reduction demo** — the actual success criterion.
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

## 3. Existing infrastructure to reuse

> Pre-impl search done (per CLAUDE.md). The pipeline reuses these; **do not
> re-derive** metrics, forcing, ERA5 loading, or LES dynamics.

### ERA5 ingestion & regridding
- `packages/ml/legoesm/ml/data/era5_loader.py` — ERA5 from WeatherBench2 GCS,
  Zarr cache, variable aliasing, configurable levels/times/cadence.
  Functions: `create_era5_dataset`, `load_era5_batch`, `load_era5_ic`.
- `packages/ml/legoesm/training/era5_to_state.py` — regrid ERA5 lat-lon → model
  grid (`era5_to_spectral_carry`, `era5_to_cubedsphere_carry`,
  `era5_to_latlon_carry`), pressure→sigma interp (`interp_pressure_to_sigma`),
  specific-humidity → mixing-ratio conversion.

### AMIP driver & output
- `scripts/run/run_amip.py` — AMIP driver (prescribed SST/SIC; analytical/COBE/
  HadISST forcing; configurable physics).
- `packages/coupler/legoesm/driver/config.py::OutputConfig` — `diag_days`,
  `checkpoint_days`, `monthly_means`, `cmip_output`, `cmip_resolution_deg`.
  **Set `diag_days = 0.25` for 6-hourly comparison.**

### Comparison metrics (area/mass weighted)
- `packages/ml/legoesm/ml/loss.py` — `area_weighted_mse`,
  `latitude_weighted_rmse`, `latitude_weighted_bias`, `per_variable_mse`.
- `packages/ml/legoesm/training/scm_rce_metrics.py` — mass-weighted vertical
  RMSE: `weighted_rmse`, `score_profiles_jax`, `precip_score_jax`,
  `score_profiles_precip_jax`. **Reuse for per-column profile scoring.**

### SCM + large-scale forcing (for LES forcing extraction)
- `packages/atmosphere/legoesm/atmosphere/scm.py::SingleColumnModel` — column
  integration reusing the canonical physics pipeline.
- `packages/atmosphere/legoesm/atmosphere/scm_forcing.py::SCMForcing` — the
  **forcing API the LES will be driven through**: `f_c`, `u_geo`, `v_geo`,
  `subsidence_w`, `theta_adv`, `qv_adv`, `prescribe` (`none`/`T_s`/`fluxes`),
  `T_s`, `w_th_s`, `w_qv_s`.

### LES / plane dynamics
- `src/legoesm/atmosphere/dynamics/compressible_euler_plane.py` — doubly-periodic                                                                                                          
-  non-hydrostatic compressible-Euler dycore + Smagorinsky LES (c_s=0.2). **The                                                                                                             
-  LES engine.** Validated to RCE at dx=2 km (CRM regime); will need to run at                                                                                                              
-  finer dx for true turbulence-resolving LES (see §5/§7 risks).                                                                                                                            
- `src/legoesm/atmosphere/dynamics/compressible_euler_plane_halo.py`,                                                                                                                      
-  `src/legoesm/parallel/plane_mpi.py` — MPI (2D pencil) support.                                                                                                                           
- `packages/core/legoesm/grids/plane.py::PlaneGrid` — plane grid.
- `packages/atmosphere/legoesm/atmosphere/dynamics/compressible_euler_plane.py` —
  3-D doubly-periodic non-hydrostatic compressible-Euler dycore + Smagorinsky SGS
  (c_s=0.2). **The LES engine.** Validated to RCE at dx=2 km (CRM regime); needs
  finer dx + stability re-validation for true turbulence-resolving LES (§5/§7).
  **A 1.5-order TKE SGS closure does NOT exist yet** (only Smagorinsky) — adding
  it is the one genuine new dynamics-side piece.
- `packages/atmosphere/legoesm/atmosphere/dynamics/compressible_euler_plane_halo.py`,
  `packages/core/legoesm/parallel/plane_mpi.py` — MPI (2D pencil) support.
- `packages/atmosphere/legoesm/atmosphere/dynamics/plane_operators.py`,
  `plane_operators_halo.py`, `plane_large_scale_forcing.py` — plane operators +
  large-scale forcing application.
- `packages/core/legoesm/grids/plane.py::PlaneGrid` — 3-D plane grid
  (`nx, ny, nlev`).
- `packages/atmosphere/legoesm/atmosphere/dynamics/rce_diagnostics.py` —
  `domain_mean_profiles_plane`, `column_water_vapor_plane`,
  `cloud_fraction_profile_plane`, etc. **Extend here for resolved-flux
  diagnostics** (`w'T'`, `w'q'`, `w'u'`) — see §5.
- `scripts/run/run_rce.py`, `run_rcemip_plane.py`, `run_rce_mpi_long.py` —
  existing plane drivers (templates for the LES driver).

### Parameter plumbing
- `packages/ml/legoesm/training/param_collector.py`,
  `packages/ml/legoesm/training/trainable_params.py` — spec-based parameter
  registry; `__param_spec__` per `*Config`. **All atmosphere params are scalar
  today (no `shape_key`)** — the spatially-varying field is new infra.

---

## 4. Pipeline stages (proposed)

1. **AMIP-run stage** — `run_amip.py` with `diag_days=0.25`, writing the 3-D
   state at 6-hourly cadence to a run directory.
2. **Compare-to-ERA5 stage** — load ERA5 (6-hourly) via `era5_loader`, regrid
   ERA5→model grid (or model→ERA5; decide once, document), compute per-column
   error fields (T/q mass-weighted profile RMSE, precip error, u/v error).
3. **Diagnose-worst-columns stage** — rank columns by a combined score; select
   top-N (config). Emit a manifest of `(lat, lon, time, large-scale state)`.
4. **LES-forcing-extraction stage** — for each flagged column build an
   `SCMForcing` from that column's GCM large-scale state (subsidence, advective
   tendencies, geostrophic wind, surface fluxes), plus the GCM column profile
   for vertical interpolation + top relaxation.
5. **LES-run stage** — standalone 3-D LES (dycore at LES resolution),
   regime chosen per column; output resolved fluxes + profiles.
6. **Coefficient-diagnosis stage** — diagnose the closure coefficient (e.g.
   entrainment rate) from the LES resolved fluxes; produce a per-column
   (possibly height-varying) value.
7. **Feedback stage** — assemble the spatially-varying parameter field; apply to
   the next AMIP iteration.

---

## 5. New infrastructure needed (gaps)

1. **Per-column comparison metric module** — column-resolved (not zonal-mean)
   error fields vs ERA5. Builds on `scm_rce_metrics` + `loss.py` (no new profile
   numerics). Likely `packages/.../diagnostics/` + a `scripts/validate/` driver.
2. **Worst-column ranking + manifest** — selection of top-N, environment tagging
   (SST/CAPE/shear) for later generalization. **DONE (iter 2)** —
   `legoesm.training.column_manifest` (`build_worst_column_manifest`,
   `compute_column_environment`, JSON I/O).
3. **GCM-column → SCMForcing extractor** — derive large-scale forcing terms from
   a single GCM column's neighborhood (subsidence from continuity / ω, advective
   tendencies, geostrophic wind, surface fluxes). New, but uses existing
   `SCMForcing` schema.
4. **LES-resolution config + standalone LES driver** — configure the (already
   3-D) plane dycore for a true turbulence-resolving LES regime (finer dx,
   acoustic-substep/Smagorinsky re-tuning, stability re-validation — *not* a
   dimensionality change); add an **optional 1.5-order TKE SGS closure** as an
   alternative to Smagorinsky; a `scripts/run/` driver that ingests the forcing
   manifest. Closure + regime selection must dispatch-error on unknown values.
5. **Resolved-flux diagnostics** — add `w'T'`, `w'q'`, `w'u'` (and entrainment
   diagnosis) to `rce_diagnostics.py` (extend, don't duplicate). **DONE (iter 6)**
   — `resolved_turbulent_fluxes_plane` (w'θ'/w'q'/w'u'/w'v'/w'θ_v').
6. **Closure-coefficient diagnosis** — map LES resolved fluxes → coefficient
   (entrainment rate / eddy diffusivity / mixing length). Physically-grounded,
   with units/sign checks. **DONE (iter 7)** — `les_closure_diagnosis`
   (`eddy_diffusivity_from_flux`, `mixing_length_from_momentum_diffusivity`,
   `entrainment_velocity_from_buoyancy_flux`).
7. **Spatially-varying parameter field infra** — promote a chosen scalar physics
   coefficient to a per-column (height-varying) field. Needs:
   - `__param_spec__` `shape` key support already exists (currently unused for
     atmosphere) — extend a target `*Config` field to carry a spatial shape.
   - Apply overrides via the config pytree (`apply_param_overrides`), traced
     inside any loss, static floats in production (SegmentForcing doctrine).
   - **Differentiability preserved** — field must flow through `jax.grad`/JIT
     without retrace; no Python control flow on traced columns.

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

## 8. Proposed file layout (respecting repo rules)

> New scripts go in the correct `scripts/` bucket; source under
> `packages/<pkg>/legoesm/`; every new `.py` gets a direct test; dispatch raises
> on unknown selection.

- `scripts/run/run_amip_reanalysis_compare.py` — orchestrates AMIP + compare.
- `scripts/validate/compare_amip_era5.py` — per-column ERA5 comparison + ranking
  (non-matrix validator). **DONE (iter 4)** — real `main()` + tested helpers;
  uses `compare_reanalysis` (iter 3) core.
- `scripts/run/run_column_les.py` — standalone forced 3-D LES from the manifest.
- `packages/ml/legoesm/training/column_era5_metrics.py` — per-column metrics
  (reusing `scm_rce_metrics`/`loss.py`). **DONE (iter 1).** Lives in the **ml**
  package (not `tools/diagnostics`) because it reuses `scm_rce_metrics`, which
  `legoesm.diagnostics` sits below in the FEDERATION.md dependency DAG.
- `packages/atmosphere/.../dynamics/rce_diagnostics.py` — **extend** with
  resolved-flux + entrainment diagnostics.
- `packages/atmosphere/.../<scheme>_config.py` — target `*Config` gains a
  spatially-shaped field + `__param_spec__` entry.
- Tests mirror under `tests/<component>/...`; LES-resolution dispatch + manifest
  parsing get direct unit tests.

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
