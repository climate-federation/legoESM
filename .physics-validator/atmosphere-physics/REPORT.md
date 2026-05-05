# Atmosphere Physics Validator — Audit Report

**Scope:** `src/legoesm/atmosphere/physics/` — 97 files, ~14k LOC
**Date:** 2026-05-01
**Mode:** Static analysis (8 modules) + Codex adversarial review (3 high-risk modules)
**Source code modified:** none (per protocol — fixes require failing regression test first)

## Module status

| Subpackage | LOC | Status | P0 findings |
|------------|----:|--------|------------:|
| convection/ | ~3700 | needs-work | 4 critical / 17 total |
| microphysics/ | ~2200 | needs-work | 3 critical / 11 total |
| turbulence/ | ~3000 | minor | 0 / hardcoded tunables |
| radiation/ (gray + solar) | ~660 | minor | 0 / `S_0` default mismatch |
| radiation/rrtmgp/ | ~6500 | deferred | needs separate cycle |
| gravity_wave_drag/ | ~750 | needs-work | 1 critical / 9 total |
| clouds/ | ~265 | clean | 0 |
| top-level (combined, _shared, thermo, physics_state) | — | minor | `compute_moist_adiabat` saturated-below-LCL |
| ml (learned_column, neural_physics, ml_param) | — | not deeply audited | NN plumbing only |

## Top 10 findings (severity × probability)

1. **CRITICAL — convection LNB index confusion** `convection/_plume.py:244-251`
   Mask intended to select "above LFC altitude" mixes surface-last and surface-first conventions. LNB diagnosed near surface in many columns. Affects ZM / KF / Emanuel / Tiedtke / Bechtold cloud-depth diagnosis.

2. **CRITICAL — convection CIN window inverted** `convection/_plume.py:311-317`
   `window_above_lfc * window_below_lcl` is empty (signs flipped). CIN ≈ 0 for all columns.

3. **CRITICAL — CAPE/τ used as M_b** `convection/zhang_mcfarlane.py:131-148`, `kain_fritsch.py:166-176`, `emanuel.py:125-131`, `tiedtke.py:196-210`, `bechtold.py:188-223`
   CAPE/τ has units m²/s³, not kg/(m²·s). Magnitude masked by `M_b_max` cap; AD gradient w.r.t. CAPE off by a density factor.

4. **CRITICAL — convection mass_flux/edmf q_c_u from saturation excess** `convection/mass_flux.py:319-331,437-451`
   Plume cloud water built as `dilution * (q_sat_base - q_sat_moist)` independent of actual q_v ⇒ produces cloud water in dry columns. Shared `_plume.entraining_detraining_plume` does this correctly; the bulk paths use the non-conservative shortcut.

5. **CRITICAL — Sundqvist column-water non-conservation** `microphysics/sundqvist.py:150-152`
   `dq_v + dq_c + dq_r = 0` with no sedimentation/precip sink, but `precipitation` field is computed and emitted ⇒ column water grows without bound (or surface flux is fictitious).

6. **CRITICAL — Morrison/Thompson mixed-phase donor over-extraction** `microphysics/morrison.py:105-117,157-161`, `microphysics/thompson.py:126-136,182-187`
   Default `bergeron_rate=1e-3 /s × dt=1200s = 1.2` ⇒ removes 1.2 × q_c per step. Morrison clamps melt with `min(..., q_X/dt)`; Thompson does not even for melt.

7. **CRITICAL — Hines WKB amplitude growth compounded incorrectly** `gravity_wave_drag/hines.py:83-108`
   Per-step multiplied by cumulative `rho_ratio[k]=sqrt(rho_sfc/rho_k)` instead of inter-level `sqrt(rho[k+1]/rho[k])` ⇒ amplitude overshoots WKB by a product of cumulative ratios. Saturation cap masks runaway in practice.

8. **MAJOR — Morrison/Thompson missing L_f for Bergeron + riming** `morrison.py:149-154`, `thompson.py:174-179`
   Liquid → ice releases L_f ≈ 333 kJ/kg. Magnitude estimate at default rates: ~2.8 K/day in mixed-phase clouds.

9. **MAJOR — `delta_0_eff/delta_deep` rescale also scales subsidence** `tiedtke.py:243-260`, `bechtold.py:259-272`, `emanuel.py:180-189`
   Should rescale only the detrainment piece returned by `_apply_mass_flux_kernel`, not the whole tendency.

10. **MAJOR — Tiedtke saturation-deficit MC proxy has wrong sign** `tiedtke.py:185-193`
    `max(q_sat - q_v, 0)` is large in dry columns; convection should fire in moist columns. Used as fallback when `moisture_convergence` arg is None.

## Codex review highlights

- **Convection** (`convection/review-1.md`): codex executed JAX evaluations to verify LNB / CIN / crossing-index behavior. Confirmed all 4 critical findings; identified additional bugs (CIN window inversion, plume q_c_u accumulation without entrainment dilution).
- **Microphysics** (`microphysics/review-1.md`): confirmed missing L_f, donor over-extraction, ice-nucleation-without-mass. Sundqvist column-water leak flagged as the most actionable finding.
- **GWD** (`gravity_wave_drag/review-1.md`): confirmed Hines WKB compounding, sigma_sat inverted scaling, McFarlane dimensionally suspect stress.

Codex / static analysis agreement: ~80%. Codex correctly upgraded several "style" findings to numerical bugs (e.g., Bechtold downdraft `0.05` cooling — not just hardcoded but dimensionally wrong AND non-water-conserving).

## Differentiability test coverage gaps

- **Convection:** `mass_flux`, `edmf` not directly grad-tested.
- **Turbulence:** only `smagorinsky`, `louis` are grad-tested; missing `holtslag_boville`, `tke`, `clubb_lite`, `edmf`, `ysu`.
- **Microphysics:** only `kessler`, `sundqvist`; missing `seifert_beheng`, `morrison`, `thompson`.
- **GWD:** none. `tests/unit/test_physics_gwd.py` has no `jax.grad` calls.

## Suggested PR plan (priority-ordered)

- **P0:** ~~fix convection LNB & CIN; Sundqvist column-water budget; Thompson melt clamp; Morrison/Thompson Bergeron+riming donor clamps~~ ✅ **DONE 2026-05-01.**
- **P0 (extended scope):** ~~fix Hines WKB compounding; mass_flux/edmf q_c_u dry-column bug; CAPE/τ→M_b unit issue~~ ✅ **DONE 2026-05-01** (initial scope expanded to include all 7 P0s after audit triage).
- **P1:** add `+L_f` to Morrison/Thompson dT_dt; fix `delta_0_eff` rescale.
- **P2:** move hardcoded tunables to configs; fix `solar.py` `S_0` default; resolve specific-humidity vs mixing-ratio doc/formula drift.
- **P3:** add `tests/unit/test_diff_gwd.py`; extend AD parametrize lists to all schemes; add L_f regression tests for Morrison/Thompson.

## P0 fixes applied (2026-05-01)

| # | Bug | File(s) | Test(s) | Diff |
|---|-----|---------|---------|------|
| P0-1 | LNB diagnosed at LFC altitude (surface-last/first index confusion) | `convection/_plume.py:244-260` | `tests/unit/test_convection_plume.py::test_compute_lfc_lnb_lnb_finds_upper_zero_crossing` | threshold flip removed; below-LFC region driven strongly negative via additive offset |
| P0-2 | CIN window factors selected the wrong sides — empty intersection | `convection/_plume.py:307-322` | `tests/unit/test_convection_plume.py::test_compute_cin_integrates_negative_buoyancy_between_lcl_and_lfc` | `above_LFC * below_LCL` → `below_LFC * above_LCL` |
| P0-3 | CAPE/τ→M_b dimensionally wrong (m²/s³ vs kg/m²/s) in 5 schemes | `convection/{zhang_mcfarlane,kain_fritsch,emanuel,tiedtke,bechtold}.py` | `tests/unit/test_zhang_mcfarlane.py::test_zm_M_b_matches_dimensional_formula` | M_b = ρ_BL × CAPE / (g × τ) per Kain (2004) §3 |
| P0-4 | Plume q_c_u in dry columns (mass_flux, edmf) | `convection/mass_flux.py` (2 sites) | `tests/unit/test_physics_convection.py::test_no_cloud_water_in_dry_column` | use actual q_v_sfc instead of q_sat_sfc as plume's water reservoir |
| P0-5 | Sundqvist column-water leak (precip emitted while q_r also accumulates); semantics not propagated to ML rebuild path | `microphysics/sundqvist.py:150-180`, `atmosphere/physics/ml_parameterization.py::apply_predicted_sundqvist_rain_survival_fraction`, `ml/physics/data.py` (teacher snapshot) | `tests/unit/test_physics_microphysics.py::test_sundqvist_column_water_budget_closes`, `test_sundqvist_drains_incoming_qr_to_precipitation`; `tests/unit/test_ml_physics_parameterization.py::test_sundqvist_rain_survival_rebuild_drains_incoming_qr` | full diagnostic-rain semantics: `dq_r_dt = -q_r/dt` (drain), drained mass added to `precipitation` flux. Codex stop-time review caught two gaps: (1) initial `dq_r_dt = 0` froze any incoming q_r; (2) the ML rebuild + teacher-snapshot paths still emitted `dq_r_dt = autoconv - evap`, so a model trained against the snapshot would learn to leak rain. Both paths now mirror the leaf. |
| P0-6 | Morrison/Thompson Bergeron+riming over-extract q_c; Thompson melt unclamped | `microphysics/{morrison,thompson}.py` | `tests/unit/test_physics_microphysics.py::test_qc_does_not_go_negative_at_long_dt[*]`, `test_thompson_qi_does_not_go_negative_warm` | proportional q_c-sink scaling; Thompson melt clamps `min(rate*q*frac, q/dt)` |
| P0-7 | Hines WKB amplitude compounded (cumulative ρ-ratio per step instead of inter-level); also smooth limiter produced negative drag | `gravity_wave_drag/hines.py` | `tests/unit/test_physics_gwd.py::test_hines_no_saturation_zero_drag` | inter-level ratio in scan; drag clipped to [0, Fmax] |

All 156 physics unit tests pass post-fix.  Per protocol, each P0 has at least one new failing-test → minimum-diff fix → re-run cycle in `tests/unit/`.

## Pre-existing failures (NOT caused by these fixes)

- `tests/atmosphere/hydrostatic/unit/test_microphysics.py::TestCheckpointWithHydrometeors::*` (3 tests) — `load_checkpoint` returns 9 fields; tests expect 8. Reproduces on `main` without these patches.

## P1 / P3 / P2 fixes applied (2026-05-01, follow-up pass)

| # | Bug | File(s) | Test(s) | Diff |
|---|-----|---------|---------|------|
| P1-1 | Morrison / Thompson dT_dt missing `+ L_f * (bergeron + riming_i + riming_s) / c_pd` (cloud → ice phase change) | `microphysics/{morrison,thompson}.py` | `tests/unit/test_physics_microphysics.py::test_freezing_releases_latent_heat_of_fusion[*]` | added L_f freezing term to latent-heating assembly |
| P1-2 | `delta_0_eff/delta_deep` rescale also rescaled subsidence in Tiedtke / Bechtold / Emanuel (subsidence is ``M/ρ × dT/dz`` — independent of delta_0) | `convection/{tiedtke,bechtold,emanuel}.py` | covered by existing scheme test suites + `tests/unit/test_emanuel.py::test_emanuel_downdraft_toggle_changes_subcloud_dT` | pass per-column / per-level `delta_0_eff` directly to the kernel; only the detrainment terms in the kernel use it |
| P3 | `tests/unit/test_physics_gwd.py` had no `jax.grad` calls (audit gap) | `tests/unit/test_physics_gwd.py` | `test_gwd_grad_through_T_finite_and_nonzero[*]`, `test_gwd_grad_through_wind_finite_and_nonzero[*]`, `test_prognostic_spectral_grad_through_launch_flux` | added grad-finite & grad-nonzero coverage for Rayleigh / Lindzen / McFarlane / Hines / prognostic_spectral.  Mcfarlane T/u tests xfailed pending the dimensional-stress fix (audit's "McFarlane stress dimensionally suspect"). |
| P2 | `solar.py` default `S_0=1360` while `constants.S_0=1361` | `atmosphere/physics/radiation/solar.py` | covered by 71-test radiation suite — no functional regression | use `constants.S_0` as the default in `daily_mean_insolation`, `perpetual_equinox_insolation`. |

## Follow-up cycle (2026-05-01) — deferred items closed

| # | Bug | File(s) | Test(s) | Diff |
|---|-----|---------|---------|------|
| Tiedtke MC proxy | Saturation-deficit formula was *inverted*: large in dry columns (suppressed convection where it should fire), small in moist columns. | `convection/tiedtke.py`, `convection/config.py` (new `mc_proxy_RH_crit=0.6` field) | `tests/unit/test_tiedtke.py::test_tiedtke_mc_proxy_is_larger_in_moist_columns` | Replaced `max(q_sat - q_v, 0)` with saturation-EXCESS `max(q_v - RH_crit * q_sat, 0)` — now positive in moist columns, vanishing in dry. |
| McFarlane stress dimensional | Both `tau_0 = G_0 * U * h^2 * N * ρ` (units `kg²/(m²·s⁴)`) AND `tau_sat = eff * ρ * U³ / (N * envelope)` (units `kg/s²`) were missing the `k_wave` factor; neither was a proper stress.  Operational drag was ~1e-21 m/s².  (Codex stop-time review caught the `tau_sat` half after the initial `tau_0` fix.) | `gravity_wave_drag/mcfarlane.py`, `gravity_wave_drag/config.py` (G_0 docstring) | `tests/unit/test_physics_gwd.py::test_gwd_grad_through_*[mcfarlane]` (xfail removed; tests now active), `test_mcfarlane_stress_units_match_lindzen` | Inserted missing `k_wave` factor in BOTH stresses: `tau_0 = G_0 * ρ * N * k * h² * U` and `tau_sat = eff * ρ * U³ * k / (N * envelope)` (both proper Pa, matching Lindzen).  Updated docstring to mark `G_0` as dimensionless prefactor.  Test column updated to provide a critical level so the saturation path actually exercises (orographic schemes need shear or critical level). |
| RRTMGP audit cycle | (Deferred from initial scope) | `atmosphere/physics/radiation/rrtmgp/` (~6500 LOC, 21 files) | (No new tests; 71 existing rad tests pass) | **Static analysis only — no critical findings.** `rrtmgp/constants.py` correctly re-exports from `legoesm.constants`; no hardcoded duplicates of physical constants; `clip`/`where`/`maximum` patterns are limit-guards (not dead-gradient cuts); two Pa↔hPa conversions are documented and consistent.  Sub-report at `.physics-validator/atmosphere-physics/rrtmgp/REPORT.md`. |
| P2 hardcoded tunables | `edmf.py` `theta_u_init = theta + 0.5` (BL-parcel perturbation hardcoded); `holtslag_boville.py` `b_louis = 5.0` and a duplicated `5.0 * Ri` literal. | `turbulence/{edmf,holtslag_boville}.py`, `turbulence/config.py` (new `EDMFConfig.parcel_dT=0.5`, `HoltslagBovilleConfig.b_louis=5.0`) | covered by full physics-smoke + GWD suite (55 tests pass) | Lifted hardcoded scheme-defining numerics to config NamedTuples per the audit's tunable-discipline rule. |
| P2 specific-humidity vs mixing-ratio doc drift | Codebase variable `q_v` is documented inconsistently: `saturation_mixing_ratio` returns mixing ratio (`r = ε e/(p-e)`) while many docstrings call `q_v` "specific humidity" (`q ≈ r / (1+r)`, ~1% drift). | `src/legoesm/thermo.py` (module docstring expanded with conventions section) | n/a (documentation only) | Added a "Conventions — water-vapor mass variables" section to the `thermo.py` module docstring noting that the codebase uses `r ≈ q` interchangeably and pointing physics that needs the distinction (e.g., saturated tropical columns, q_c bookkeeping) to convert explicitly. |
| Prognostic-spectral GWD test gap | (Codex stop-time review) The diff test only asserted `isfinite(g)`; a zero gradient (broken graph) would pass silently. | `tests/unit/test_physics_gwd.py::test_prognostic_spectral_grad_through_launch_flux` | own test | Tightened: also require `abs(g) > 1e-30`. |

## Four-invariant sign-off

- **Units:** ❌ multiple unit errors (CAPE/τ→M_b in 5 convection schemes; Hines drag; McFarlane stress; Tiedtke/Bechtold downdraft cooling).
- **Signs:** ⚠ CIN window inverted; Tiedtke MC proxy wrong sign; Hines smooth limiter can produce negative drag.
- **Differentiability:** ⚠ design is AD-friendly but `q ** 0.525` NaN at q=0; explicit Euler in mass_flux/edmf prognostics.
- **Test cases:** not run — would require regression suite re-run post-fix.

**Status: cannot sign off as clean.** Substantial P0/P1 backlog generated for follow-up PRs.

## Per-module artifacts

```
.physics-validator/atmosphere-physics/
├── REPORT.md                          (this file)
├── convection/        inventory.md, static.md, packet-1.md, review-1.md, rebuttals.md
├── microphysics/      inventory.md, static.md, packet-1.md, review-1.md, rebuttals.md
├── gravity_wave_drag/ inventory.md, static.md, packet-1.md, review-1.md
├── turbulence/        inventory.md, static.md
├── radiation/         inventory.md, static.md
├── clouds/            inventory.md, static.md
├── top_level/         inventory.md, static.md
└── ml/                inventory.md
```

## Important caveat

These findings come from static analysis + one codex iteration per module. **Each P0 should be verified with a failing regression test before fixing**, per the physics-validator protocol. False positives are possible — particularly for sign-convention findings on schemes with intentional smooth-limiter design choices.
