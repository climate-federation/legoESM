# Compare-to-Reanalysis + LES-Informed Column Correction

**Branch:** `feat/compare-reanalysis`
**Status:** Implementation in progress — Stage 2 (per-column comparison metric) landed.
**Date:** 2026-06-15 (design); 2026-06-17 (impl began)
**Scope:** Atmosphere component only. ERA5 reanalysis only. **Not** supervised learning.

> ## Implementation progress (newest first)
>
> ### Iter 9 (2026-06-17) — Stage 7 (gap #7, application): feedback dispatch + apply ✅
> Added `legoesm.training.feedback` — the feedback **application** layer:
> - `build_parameter_field(strategy, ...)` — hardened dispatch over the iter-8
>   strategies (`"static"` scatter vs `"environment"` kernel); raises on unknown
>   strategy, missing required inputs, AND **cross-strategy input leakage** (a
>   static call passing env-only inputs, or vice-versa, is rejected — not
>   silently ignored).
> - `apply_column_parameter_field(config, field_name, field, ...)` — flattens the
>   grid field to `(ncol,)`, validates length, and splices it into a scheme
>   `*Config` via `apply_param_overrides` (traced-in-loss). **Requires the field
>   be authorized column-promoted** — explicit `promoted_fields` allowlist OR a
>   non-None `shape` key in the config's `__param_spec__` (nested
>   `{Class:{"params":{field:{shape}}}}` or flat layout) — so a per-column field
>   can never silently overwrite a scalar parameter. Rejects scalar/0-d fields.
> - Tests: 17 cases incl. dispatch + cross-strategy-leak raises, nested & flat
>   spec authorization, unpromoted/unknown-field/scalar rejects, non-1D env
>   guard, and an end-to-end `diagnosed-values → field → config-leaf` `jax.grad`.
> - **Codex adversarial review: clean** (6 real findings fixed over 3 rounds,
>   incl. the nested-`__param_spec__` layout bug and the scalar-overwrite footgun).
>
> **Next:** the per-scheme **promotion** of a real production coefficient
> (e.g. a convection entrainment rate) — add the `shape`-keyed `__param_spec__`
> entry + make the scheme body broadcast a `(ncol,)` field (physics-validated),
> then the end-to-end bias-reduction demo. Still open: grid-side
> `ColumnLargeScaleState` extractor; LES driver + 1.5-order TKE SGS.
>
> ### Iter 8 (2026-06-17) — Stage 7 (gap #7, assembly): parameter-field assembly ✅
> Added `legoesm.training.parameter_field` — assemble the iter-7 per-column
> diagnosed coefficients into a full-grid GCM parameter field (the feedback
> field), both §6 generalization strategies, pure-JAX + differentiable:
> - `scatter_column_field(grid_shape, flat_indices, values, background, valid)` —
>   **static (lat,lon)**: background + diagnosed values at the worst columns
>   (distinct indices per `rank_worst_columns`; differentiable w.r.t. values).
> - `environment_kernel_field(grid_env, sample_env, sample_values, length_scales,
>   ...)` — **regress onto environment predictors**: Nadaraya–Watson Gaussian
>   kernel regression in normalized (SST/CAPE/shear) space → every grid column
>   gets a value from environmentally-similar diagnosed columns; columns with no
>   sample within ~3 normalized σ fall back to background.
> - **Hardening (Codex, clean after 2 rounds):** invalid samples sanitized to
>   finite zero BEFORE arithmetic (kills 0·NaN contamination + NaN adjoints),
>   length-scale floor (no 0/0), `>=` threshold, conservative 3σ default,
>   dtype promotion, exact-zero background gradient.
> - Tests: 13 cases incl. analytic N-W recovery/averaging, threshold boundary,
>   NaN-invalid non-contamination, AD-safety, jit.
> - The *application* to a scheme `*Config` (shape-keyed field +
>   `apply_param_overrides`, traced-in-loss) is the companion next step — the
>   `shape_key` infra already exists in `param_collector`.
>
> **Next:** wire the assembled field into a target scheme `*Config` as a
> `shape`-keyed `__param_spec__` param applied via `apply_param_overrides`
> (traced-in-loss), then the end-to-end bias-reduction demo. Still open:
> grid-side `ColumnLargeScaleState` extractor; LES driver + 1.5-order TKE SGS.
>
> ### Iter 7 (2026-06-17) — Stage 6 (gap #6): closure-coefficient diagnosis ✅
> Added `atmosphere.dynamics.les_closure_diagnosis` — the **inverse** of the
> forward turbulence schemes: diagnose a closure coefficient from the iter-6
> resolved fluxes. Public API:
> - `eddy_diffusivity_from_flux(flux, phi_full, z_full)` → `K=-w'φ'/(∂⟨φ⟩/∂z)`
>   per interior interface + `valid` mask (rejects ill-posed near-zero gradient
>   and counter-gradient `K<0`).
> - `mixing_length_from_momentum_diffusivity(K_m, shear)` → Prandtl `ℓ=√(K_m/|∂U/∂z|)`.
> - `entrainment_velocity_from_buoyancy_flux(w_θv, θv_full, z_iface)` →
>   `w_e=-(w'θ_v')_inv/Δθ_v` at the inversion (argmin buoyancy flux), valid only
>   for a stable interior inversion with negative entrainment flux — **the doc's
>   headline coefficient**.
> - All AD-safe (double-where masked divisions; gradient-safe sqrt), jit/vmap-
>   friendly; thresholds are documented diagnostic regularizers.
> - Tests: 13 cases incl. **analytic K recovery** (`flux=-K0·∂φ/∂z⇒K=K0`),
>   mixing-length round-trip, exact entrainment from a crafted inversion, ill-
>   posed/counter-gradient/boundary-min rejection, sqrt(0)+entrainment grad.
> - **Codex adversarial review: clean** (4 real findings fixed: sqrt(0) AD leak,
>   false fail-safe claim, unguarded boundary inversion, tie convention).
>
> **Next:** the **feedback half** — map the diagnosed per-column coefficient onto
> a spatially-varying GCM parameter field (gap #7) and apply it via the config
> pytree (`apply_param_overrides`, traced-in-loss). Still open: grid-side
> `ColumnLargeScaleState` extractor; standalone LES driver + 1.5-order TKE SGS;
> end-to-end bias-reduction demo.
>
> ### Iter 6 (2026-06-17) — Stage 5 (gap #5): LES resolved-flux diagnostics ✅
> **Extended** `atmosphere.dynamics.rce_diagnostics` (no new module) with the
> LES-resolved turbulent fluxes the closure diagnosis (stage 6) consumes:
> - `resolved_turbulent_fluxes_plane(state, height_coord, qv_slot)` →
>   `ResolvedTurbulentFluxes` = `w'θ'`, `w'q_v'`, `w'u'`, `w'v'`, `w'θ_v'`
>   (buoyancy) profiles at the `nlev-1` interior interfaces + `z_half_interior`.
> - `_resolved_flux_interfaces` — domain-mean eddy covariance `<w'φ'>` with the
>   **correct staggering**: `w` stays native on its half-level grid (no
>   smoothing, matching `vertical_velocity_variance_plane`), the full-level
>   scalar is averaged to the interior interfaces; perturbations from the
>   per-level horizontal mean (so `theta_ref` cancels). Buoyancy flux uses
>   `θ_v=θ(1+(1/ε−1)q_v)` (reuses `constants.epsilon`, same convention as
>   `compute_cape`).
> - Validates w-on-half-levels, u/v full-level shapes, z_half length, qv_slot.
> - Tests: 11 new cases in `tests/unit/test_rce_diagnostics.py` incl. **analytic
>   checkerboard covariances** (`<w'θ'>=W·A`, `<w'q'>=W·B`, `<w'u'>=W·C`) and the
>   **exact nonlinear buoyancy cross-term** `W·(A(1+c·q0)+c·θ_ref·B)`; zero-flux
>   for uniform w/scalar; bad-shape raises; jit + grad.
> - **Codex adversarial review: clean** (confirmed staggering/perturbation/θ_v
>   algebra correct; 2 validation gaps fixed).
>
> **Next:** Stage 6 — closure-coefficient diagnosis (entrainment / eddy
> diffusivity / mixing length) from these resolved fluxes (e.g. K from
> `−w'φ'/(∂<φ>/∂z)`, entrainment from the flux-jump at inversion). Still also
> open: the grid-side `ColumnLargeScaleState` extractor (iter 5 follow-up) and
> the standalone LES driver + 1.5-order TKE SGS.
>
> ### Iter 5 (2026-06-17) — Stage 4 (assembly): GCM-column → SCMForcing ✅
> Added `legoesm.atmosphere.column_forcing` — assembles a **steady** `SCMForcing`
> from a flagged column's extracted large-scale state, so the LES is forced
> exactly like that column's environment. Public API:
> - `ColumnLargeScaleState` — the extracted quantities (subsidence_w **or** omega,
>   u/v_geo, theta/qv advective tendencies, surface forcing, lat).
> - `build_column_scm_forcing(ls, *, allow_no_subsidence=False)` → validated
>   `SCMForcing` (constant-in-time callables; Coriolis from lat; ω→w conversion).
> - `subsidence_w_from_omega` (reuses `_shared.diagnose_grid_w_from_omega`),
>   `coriolis_f_c` (reuses `coriolis_parameter_fv3`).
> - **Hardening (Codex-driven, clean after 3 rounds):** raises on both-/neither-
>   subsidence (neither needs explicit `allow_no_subsidence` — silently dropping
>   subsidence biases the forced LES), surface-prescription **exclusivity** guards
>   (reject conflicting extras, not just missing required channels), rank-1/nlev
>   profile-shape validation, and dtype promotion incl. q_v.
> - Reuse-only physics (no re-derived ω→w, Coriolis, or forcing schema); lives in
>   the **atmosphere** package (ml pipeline calls it per-record).
> - Tests: `tests/atmosphere/test_column_forcing.py` (16 cases).
>
> **Remaining for Stage 4:** the **grid-side extractor** — derive `omega` (from
> `∇·v_h` via `grids.vertical` continuity), geostrophic wind (`∇Φ`), and
> theta/qv advective tendencies (`-V·∇·`) at a flagged column's neighbourhood,
> producing `ColumnLargeScaleState`. Then Stage 5 — standalone LES driver +
> 1.5-order TKE SGS + resolved-flux diagnostics.
>
> ### Iter 4 (2026-06-17) — Stage 2 driver: `compare_amip_era5.py` ✅
> Added `scripts/validate/compare_amip_era5.py` — the non-matrix validator that
> loads a saved AMIP restart + an ERA5 slice, regrids ERA5 → model grid+sigma,
> and writes the worst-column JSON manifest. **Real `main()`** wired on confirmed
> APIs (`grids.factory.create_grid` / `grids.vertical.create_sigma_coordinate` /
> `driver.restart.load_restart` (10-tuple) / `era5_to_state.load_era5_slice` +
> `era5_to_*_carry`), not a stub. Importable, unit-tested helpers:
> - `canonical_grid_type` / `select_era5_regrid` — grid-type dispatch (aliases;
>   raises on unknown; spectral↔gaussian token mapping between regrid + factory).
> - `grid_lat_lon_deg` — uniform `grid.grid_lat`/`grid_lon` (works for
>   cubed-sphere `(6,n,n)` and lat-lon/Gaussian `(n_lat,n_lon)`).
> - `sigma_levels`, `model_state_from_restart`, `compare_and_write`.
> - **Hardening (Codex-driven, clean after 2 rounds):** `load_restart(strict=True)`
>   + explicit post-load shape/nlev checks vs the built grid (resolution/nlev
>   mismatch fails loudly); `--sst-npz` for prescribed SST with a **loud
>   UserWarning** when absent (SST tag → surface-air proxy; CAPE/shear/ranking
>   unaffected) instead of silent degradation.
> - Tests: `tests/validate/test_compare_amip_era5.py` (8 cases incl. a
>   fully-monkeypatched `main()` wiring test asserting factory token, strict=True,
>   regrid call order, manifest written, proxy warning fires).
>
> **Remaining for Stage 2/Stage 1:** thread real prescribed SST + segment precip
> into the manifest; AMIP/CMIP run wiring with `diag_days=0.25`; an end-to-end
> smoke on a tiny real checkpoint+ERA5 slice. **Next major:** Stage 4 — the
> GCM-column → `SCMForcing` extractor (subsidence / advective tendencies /
> geostrophic wind / surface fluxes from a flagged column's neighborhood).
>
> ### Iter 3 (2026-06-17) — Stage 2 orchestration: model↔ERA5 compare entry point ✅
> Added `legoesm.training.compare_reanalysis` — ties the iter-1 metric and
> iter-2 manifest into one entry point on aligned model + ERA5 states.
> **Regrid-direction decided: ERA5 → model grid + model sigma** (the supported
> `era5_to_state` direction; comparison stays native to the model grid/columns).
> Public API:
> - `ColumnState` (T/q_v/u/v/p_s, optional precip_mm_day/sst_K) + `ColumnComparison`.
> - `compare_state_to_reference(...)` — mass weights from dsigma → `score_columns`
>   → env tags from the **model** column → `build_worst_column_manifest`.
>   Precip enters only when **both** states carry it (avoids the exactly-one
>   raise; ERA5 precip is often absent).
> - `build_pressure_from_sigma` (pure-sigma `p=σ·p_s`; hybrid coords pass explicit
>   `p_full`/`p_half` overrides), `precip_mm_day_from_accum`,
>   `column_state_from_carry` (duck-typed `SegmentCarry` adapter — no coupler import).
> - **Hardening (Codex-driven):** strict surface-last increasing-σ check (rejects
>   reversed coords that would flip mass weights / CAPE), both-or-neither +
>   shape + monotonic-pressure validation on hybrid overrides, full column-shape
>   validation of all surface fields, and `rank_worst_columns` now routes
>   non-finite scores to −∞ (never selected, flagged invalid).
> - Tests: `tests/unit/test_compare_reanalysis.py` (13 cases incl. biased-column
>   flagging, precip-drop, reversed-σ / reversed-pressure / mismatched-shape /
>   partial-override raises, accum→rate, carry adapter).
> - **Codex adversarial review: clean** (6 findings over 3 rounds; all real
>   correctness/validation bugs fixed).
>
> **Next:** the `scripts/validate/compare_amip_era5.py` driver — load a saved
> AMIP snapshot + ERA5 slice (`era5_to_state` / `era5_loader`), call
> `compare_state_to_reference`, write the manifest. Needs the AMIP snapshot
> on-disk format inspected first. Then Stage 1 AMIP wiring (`diag_days=0.25`).
>
> ### Iter 2 (2026-06-17) — Stage 3 / gap #2: worst-column manifest + env tags ✅
> Added `legoesm.training.column_manifest` (ml package). Public API:
> - `compute_bulk_shear(u, v, sigma_full, config)` — vector bulk wind shear
>   `|V(upper)-V(lower)|` between the model levels nearest config sigma refs
>   (JAX-native `jnp.argmin`+`jnp.take`, jit-safe with traced sigma).
> - `compute_column_environment(...)` → `ColumnEnvironmentFields` (SST, CAPE,
>   bulk shear) per column. CAPE reuses canonical
>   `atmosphere.physics.thermodynamics.parcel_profile_and_cape` (surface-last
>   `(ncol,nlev)`, virtual-T CAPE); reshapes leading column dims and back.
> - `build_worst_column_manifest(...)` → `list[ColumnRecord]` — host-side
>   assembly on top of `rank_worst_columns`; unravels flat→grid index, drops
>   invalid (padded) columns, broadcasts lat/lon (grid-shaped or 1-D
>   rectilinear via meshgrid; raises on cubed-sphere 1-D coords).
> - `ColumnRecord` / `ColumnEnvironment` NamedTuples + JSON I/O
>   (`write_manifest`/`read_manifest`/`manifest_to_dicts`/`dicts_to_manifest`),
>   round-trip tested. Manifest is the lightweight index (grid_index, lat/lon,
>   time_index, scores, env tags); full column profiles are re-extracted from
>   the saved AMIP state in stage 4, not stored here.
> - `EnvironmentConfig` — shear reference sigma levels (diagnostic, not physics).
> - Tests: `tests/unit/test_column_manifest.py` (9 cases: shear math + level
>   selection, jit-with-traced-sigma, CAPE sign/shape, manifest selection +
>   invalid-drop, coord broadcasting + bad-shape raise, JSON round-trip).
> - **Codex adversarial review: clean** (1 real bug: host-side `np.argmin` broke
>   the jit contract → fixed to JAX-native level selection).
>
> **Next:** Stage 2 completion — the model↔ERA5 regrid + comparison driver
> (`scripts/validate/compare_amip_era5.py`) that produces `score_columns` inputs
> from a saved AMIP run + ERA5 (reuse `era5_to_state` / `era5_loader`), then
> emits the manifest. Then Stage 1 AMIP driver wiring (`diag_days=0.25`).
>
> ### Iter 1 (2026-06-17) — Stage 2 / gap #1: per-column comparison metric ✅
> Added `legoesm.training.column_era5_metrics` (in the **ml** package — it must
> sit above `legoesm.diagnostics` (tools), which cannot import the reused
> `scm_rce_metrics`/`loss`/`era5_loader`; FEDERATION.md DAG). Public API:
> - `normalized_mass_weights(dsigma)` — σ-coordinate per-level mass weights (Σ=1).
> - `per_column_weighted_rmse(model, ref, weights)` — mass-weighted vertical RMSE
>   per column, NaN-level masked + renormalized; reuses `scm_rce_metrics.weighted_rmse`.
> - `per_column_vector_wind_rmse(...)` — `sqrt(Σ w_k(Δu²+Δv²))` vector-wind RMS.
> - `score_columns(...)` → `ColumnErrorFields` (T_rmse_K, qv_rmse, wind_rmse_m_s,
>   precip_err_mm_day, dimensionless `combined_score`). Precip reuses
>   `precip_score_jax`; raises on exactly-one-of precip args.
> - `rank_worst_columns(score, n, valid_mask=)` → `(flat_idx, scores, valid)`;
>   `valid` flags padded slots when `n` > valid-column count (JIT-static shape).
> - `ColumnErrorConfig` — diagnostic normalization scales (module `UPPER_SNAKE`
>   constants; not trainable physics → no `__param_spec__`).
> - **Autodiff-safe**: added shared `safe_sqrt` to `scm_rce_metrics` (double-where
>   idiom) so `weighted_rmse` and all combined scores differentiate cleanly at
>   zero error (perfect-match / all-masked columns) — fixes a `sqrt(0)` NaN-grad
>   hazard that also latently affected the SCM-RCE training path. Primal unchanged
>   (SCM-RCE tests still green).
> - Tests: `tests/unit/test_column_era5_metrics.py` (18 cases incl. AD-safety,
>   NaN masking, ranking validity, partial-precip guard).
> - **Codex adversarial review: clean** (3 real bugs found+fixed across 3 rounds:
>   sqrt(0) NaN-grad ×2, ranking validity flag).
>
> **Next:** Stage 3 — worst-column manifest (lat/lon/time + environment tags
> SST/CAPE/shear) built on `rank_worst_columns`; then Stage 1 AMIP driver wiring
> (`diag_days=0.25`) + the model↔ERA5 regrid that feeds `score_columns`.

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
