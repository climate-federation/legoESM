# RRTMGP Faithfulness Audit — Tracking

Goal: legoESM `src/.../rrtmgp/` faithful to
https://github.com/earth-system-radiation/rte-rrtmgp, fully
JAX-differentiable, mixed-precision-safe, MPI/GPU-scalable.
Iter-10 compressed; original at `rrtmgp.original.md`.

---

## Iter 17+18 — centralised cloud_kwargs helper + codex review (2026-05-29)

### Iter-17: centralise to_rrtmg_kwargs
- Two duplicate cloud_kwargs build sites (iter-15 fix +
  iter-16 fix) → moved to `CloudProperties.to_rrtmg_kwargs()` method.
- Helper deliberately omits `cloud_fraction` with a docstring that
  cites commit 4c9591bb's 59-113 W/m² OSR cost.
- Both `physics_pipeline.py` and `integration.py` now call
  `cloud_props.to_rrtmg_kwargs()` instead of building the dict
  inline.
- New regression test `TestCloudKwargsHelper::test_kwargs_excludes_cloud_fraction`
  pins the contract.

### Iter-18: codex adversarial review of iter-13 → iter-17 restorations
Fresh codex thread (5-section prompt, ≤300 word report) reviewed all
four restored production fixes.  **VERDICT: SHIP**.

Codex confirmed:
1. **Sign fix faithfulness**: `-G·dflux/abs(dp)/Cp` gives cooling for
   positive outward flux divergence in BOTH `solve_columns` (positive
   dp) and the legacy `RRTMGP.compute_heating_rate` callpath
   (centered-difference negative dp canonicalised by `abs(dp)`).
2. **AD-safety completeness**: All 5 named restoration sites present
   (`cloud_optics.py:165`, `monochromatic_two_stream.py:89/521`,
   `two_stream.py:478`).  Survivors (gas_optics.py:205,
   delta-scaling divides, `optics_base` combines, `_k_fn`) are
   pre-existing patterns unrelated to the 59407953 restoration.
3. **Cloud-fraction logic**: τ-linear-in-LWP argument verified by
   reading `cloud_optics.py:143` (τ = interp(ext)·cld_path·mask,
   linear in cld_path) and `optics.py:359` (RRTMGP cf scaling only if
   kwarg provided).
4. **No restoration-specific bugs found**.

Note: the public `solve_columns(..., cloud_fraction=...)` parameter
still applies cf scaling when a caller passes it explicitly — that's
the right behaviour for an in-cloud-LWP caller.  In-tree call sites
use grid-mean LWP and now correctly omit cloud_fraction.

---

## Iter 15+16 — third lost AIMIP-merge fix restored (2026-05-29)

After iter-13 (sign) and iter-14 (AD-safe floors) were found and
restored, audited git history for more `Fix.*RRTMG` commits that
predate `b5b5954e Aimip (#312)`.  Found:

`4c9591bb Fix cloud-fraction double-discount in RRTMG cloud coupling`
(Kevin Debeire, 2026-05-17) — also reverted by AIMIP merge.

### Bug
`compute_radiation_core` was passing cloud fraction through TWO
independent discount paths:
1. `compute_cloud_properties` returned **grid-mean** LWP =
   q_c·dp/g (q_c is the grid-mean prognostic) which already
   carries `LWP_grid = cf · LWP_in-cloud`.
2. RRTMG optics then multiplied cloud τ by cloud_fraction AGAIN.

Effective τ: `cf² · τ_in-cloud` instead of `cf · τ_in-cloud`.
Clouds ~cf× too thin in both SW and LW.

### Cost (from probe_cloud_fraction.py in calibration repo)
- cf = 0.6: −59 W/m² OSR
- cf = 0.3: −113 W/m² OSR

### Fix
Stop passing `cloud_fraction` from `compute_radiation_core` to
RRTMG.  Since τ is linear in LWP, "grid-mean LWP, no cf scaling" is
mathematically identical to the correct "in-cloud LWP × cf scaling".

### Verification
- 21 cloud-fraction + RRTMGP unit tests pass.
- All 10 AMIP-RRTMG integration tests pass.

### Cumulative lost-fix audit
| iter | Lost commit | Author | Impact |
|---|---|---|---|
| 13 | `0be22f0f` | Aytac Pacal | Thermal runaway (T̄ 261→293K/120d) |
| 14 | `59407953` | Kevin Debeire | NaN gradients in 4/5 AMIP params |
| 15 | `4c9591bb` | Kevin Debeire | -59..-113 W/m² OSR from cf² discount |

All three pre-existed in main before the `b5b5954e AIMIP-#312` merge
silently reverted them.  All restored in iter-13/14/15.

---

## Iter 13+14 — restore production-critical fixes lost in AIMIP merge (2026-05-29)

**This is the highest-impact work in the entire session.**

Investigation: while verifying iter-12 heating-rate output for a
US-Std-like column, observed that **free-tropospheric LW heating
was +0.5 K/day (radiative warming) instead of −1..−2 K/day
(radiative cooling)**.  Cross-referenced git history and found:

| Lost commit | Author | Date | Pre-revert impact |
|---|---|---|---|
| `0be22f0f` Fix sign-inverted RRTMGP radiative heating rate | Aytac Pacal | 2026-05-22 | Thermal runaway, T̄ 261→293K over 120 days, NaN at day 125 |
| `59407953` Fix AD-unsafe maximum-floor patterns | Kevin Debeire | 2026-05-14 | 4/5 AMIP+RRTMG tunable params NaN gradients (no training) |

Both were silently reverted by `b5b5954e Aimip (#312)` merge.  My
iter-1 → iter-12 work was all happening atop this broken state.

### iter-13: heating-rate sign restored
- `two_stream.compute_heating_rate`: restored `-G * dflux / abs(dp) / Cp`.
- New tests: `TestHeatingRateSign::test_free_tropospheric_LW_cools`
  pins LW heating < 0 in 300-800 hPa.
  `test_clear_sky_SW_heats` pins SW heating ≥ 0 everywhere.
- Both tests reference commit `0be22f0f` in docstrings to make any
  future revert traceable.

### iter-14: AD-safe maximum-floor patterns restored
- `cloud_optics.py`: ssa and asymmetry_factor combine-divide → `safe_divide`.
- `monochromatic_two_stream.py:lw_combine_sources`: sqrt(max(x, 0)) → sqrt(max(x, _EPSILON)).
- `monochromatic_two_stream.py:_solve_rte_2stream`: three sites
  (`albedo_op`, `upward_emission_op`, `flux_down_op`) all
  `1/max(1-r·a, _EPSILON)` → `safe_divide`.
- `two_stream.py:solve_sw` aerosol path: w_tot and g_tot → `safe_divide`.

### Verification
- **107 tests pass** (61 radiation unit + 36 stratosphere + 10 AMIP-RRTMG).
- 2 skipped multidevice MPI tests need >1 visible JAX device.
- Free-tropospheric LW heating now -0.5 K/day (cooling) ✓
- AMIP-RRTMG smoke test runs without thermal drift ✓

---

## Iter 1 audit — initial findings (2026-05-28)

Files reviewed: `optics/{gas_optics,optics_utils,lookup_*,optics}.py`,
`rte/{two_stream,monochromatic_two_stream,rte_utils}.py`,
`rrtmgp.py`, upstream `mo_gas_optics_rrtmgp_kernels.F90` (kernels),
`mo_gas_optics_rrtmgp.F90` (frontend lines 300/303/519/1030/1354,
1555 for optimal angle).

**Match to upstream (verified)**:
- log(p)-interp with `+itropo-1` shift; `kmajor` (n_t, n_p+1, n_eta, n_gpt).
- `tropo_idx = (p ≤ p_ref_trop)` maps Fortran `itropo = merge(1,2,p>tropo)`.
- `col_mix ≡ combined_vmr × col_dry`; eta = vmr_1/combined_vmr, fallback 0.5.
- Rayleigh: `(1+vmr_h2o)·col_dry = col_dry + col_h2o`.
- Planck-face: `sqrt(pfrac[k]·pfrac[k+1])·totplnk(T_face)` via `lw_combine_sources`.
- `t_planck = linspace(160, 355, 196)` matches upstream `totplnk_delta`.

**Bugs flagged** (all fixed in subsequent iters):

| ID | Severity | Site | Issue |
|---|---|---|---|
| BUG-1 | critical | `compute_planck_sources` | `totplnk(T)∝T⁴` linear-extrap with `weight2>1` catastrophic at halo T<160K |
| BUG-2 | moderate | major/minor/Rayleigh/planck_fraction | same out-of-range extrap, smaller impact |
| BUG-3 | AD | `_compute_relative_abundance_interpolant` | `where(combined_vmr>0,…)` NaN grad in (0,eps] |
| BUG-4 | perf | `compute_minor_optical_depth` | 2× wasted compute via both-branch where; functionally OK |
| BUG-5 | climatology | `_standard_o3_profile` | σ_log=1.5, peak 8 ppm → 10× too much O3 at 100 hPa |
| OPT-MISS | faithfulness | `lw_cell_source_and_properties` | hard-coded `_LW_DIFFUSIVE_FACTOR=1.66`; upstream uses per-band `optimal_angle_fit` |

---

## Iter-by-iter changes (1 → 9)

### Iter 1+2 — stratosphere fidelity + optimal LW angle
- `gas_optics.py`: `_clip_to_table_range(t, t_ref)` helper, applied at
  major/minor/Rayleigh/planck_fraction/planck_sources entry points.
  Upstream errors out on out-of-range tlay/tlev/tsfc; legoESM clamps
  to `[t_ref[0], t_ref[-1]] = [160, 355]K` to stay differentiable.
- `gas_optics.py`: safe-div via `jnp.maximum(combined_vmr, 1e-30)` so
  reverse-mode AD doesn't propagate NaN through dead `where` branch.
- `lookup_gas_optics_longwave.py`: load `optimal_angle_fit (n_bnd, 2)`
  from netCDF; `None` for older files.
- `monochromatic_two_stream.py:lw_cell_source_and_properties`:
  optional `lw_diffusive_factor` kwarg.
- `two_stream.py:_compute_optimal_lw_secant(tau, band_idx, fit, hw)`:
  upstream `compute_optimal_angles` formula `c0·exp(−Σtau) + c1`.
- `two_stream.py:solve_lw`: opt-in `use_optimal_angle=False`; raises
  `ValueError` when True+fit-missing (no silent degradation).
- `config.py:RRTMGPConfig.use_optimal_angle: bool = False`.
- `rrtmgp.py:_instance_cache_key` honors `use_optimal_angle`.
- 18 new tests across `TestClipToTableRange`, `TestRelativeAbundanceSafeDiv`,
  `TestOutOfRangeTemperature`, `TestMixedPrecision`, `TestOptimalLwSecant`.

### Iter 3 — skewed-Gaussian O3 + saturation tests
- `_standard_o3_profile`: replaced σ_log=1.5 single Gaussian with
  piecewise skewed: σ_trop=0.9, σ_strat=1.5, peak 9 ppm at 10 hPa.
  C0-continuous at peak; finite gradient on each side.
- `_compute_minor_optical_depth` docstring: clarify that density
  scaling `p/T` uses **physical** T (not clipped) — Lorentz line-
  shape is meaningful outside the table range, so minor OD doesn't
  fully saturate at the boundary.
- New tests: `TestStandardO3Profile` (peak loc, factor-of-2 fit at
  100/30/10/1 hPa, finite grad), `test_out_of_range_T_saturates_{major,rayleigh}_OD`.

### Iter 3.5 — tropospheric O3 baseline (codex iter-3 SHIP review)
- Codex flagged 200-500 hPa coverage gap: skewed Gaussian alone gives
  <1 ppb vs real ~25-50 ppb (2-3 OOM silent underestimate).
- Fix: `o3 = max(o3_gauss, 2.0e-8)` — 20 ppb tropospheric background.
- New profile vs US Std Atm 1976: 20 ppb (1000/500 hPa), 35 ppb (200),
  341 ppb (100), 4.3 ppm (30), 9 ppm (10), 2.8 ppm (1), 81 ppb (0.1).
- New tests: `test_extended_coverage_*` ([0.2, 5.0] ratio at
  1000/500/200/0.1 hPa), `test_tropospheric_background_floor`.

### Iter 4 — eliminate duplicate optics call
- `use_optimal_angle=True` was calling `compute_lw_optical_properties`
  twice per g-point (once for secant, once inside
  `_compute_local_properties_lw`).
- `_compute_local_properties_lw` accepts new
  `precomputed_lw_optical_props` kwarg; `solve_lw.step_fn` computes
  optics once at the top and threads the dict to both consumers.
- AD path: gradients flow through one shared call instead of two.

### Iter 5 — bound eta-fraction gradient
- Iter-1 safe-div used `where(combined_vmr > 0, vmr/max(c, eps), 0.5)`.
  In (0, eps] band, division branch was selected with denominator
  clamped to eps → `1/eps ~ 1e30` backward gradient.
- Fix: tighten condition to `combined_vmr > eps` so the 0.5 fallback
  is picked when denom would be clamped. Matches upstream
  `col_mix > 2·tiny(col_mix)` more faithfully.
- New test: `test_bounded_grad_when_combined_vmr_subepsilon` pins
  `max |∂tau/∂vmr| < 1e20` (was ~1e35 under old condition).

### Iter 6 — integrated O3 column DU sanity
- Point-value tests didn't guard against integrated-column drift.
- New test computes `∫ vmr_o3 dp` and converts to Dobson Units via
  SI factor `7891 = (N_A/(g·M_air))/2.687e20`. Range [200, 400] DU.
- Current profile integrates to **262 DU** (mid-lat annual ≈ 300).

### Iter 7 — scan/loop equivalence for optimal angle
- Existing `test_rrtmgp_use_scan_equivalence` covers fixed-1.66 path.
- New `test_optimal_angle_use_scan_equivalence` runs `solve_columns`
  with `use_optimal_angle=True` under both `use_scan=True/False` and
  asserts `lw_flux_up/down/heating_rate` bit-equivalent (1e-10 tol).

### Iter 8 — bound optimal-angle vs fixed-1.66 LW flux delta
- Upstream RFMIP secants are 1.5-1.9 → flux delta should be single-
  digit %.
- New test: `|out_opt − out_fixed|/out_fixed ∈ (1e-6, 10%)` at surface.
- Catches silently-no-op plumbing AND runaway-secant regression.

### Iter 9 — defensive safe-divide for `vmr_ref_ratio`
- `vmr_ref_ratio = vmr_ref[0] / vmr_ref[1]` would NaN if a future
  table has `vmr_ref[1] = 0`.
- Fix: clamp denom via `jnp.maximum(vmr_ref[1], _VMR_SAFE_DIV_EPS)`.
- Mirrors iter-5 pattern; shipped tables unchanged (all entries ≥1e-10).

---

## Code/spec touched (cumulative)

| File | Changes |
|---|---|
| `optics/gas_optics.py` | `_clip_to_table_range`, safe-div at `combined_vmr` (iter-1, iter-5), safe-div at `vmr_ref_ratio` (iter-9), minor-OD density-scaling doc (iter-3) |
| `optics/lookup_gas_optics_longwave.py` | Load `optimal_angle_fit` from netCDF |
| `rte/two_stream.py` | `_compute_optimal_lw_secant`, `solve_lw(use_optimal_angle)`, `precomputed_lw_optical_props` threading (iter-4) |
| `rte/monochromatic_two_stream.py` | `lw_cell_source_and_properties(lw_diffusive_factor=…)` |
| `rrtmgp.py` | `_standard_o3_profile` skewed + 20 ppb bg, `solve_columns` forwards `use_optimal_angle`, cache-key honors it |
| `config.py` | `RRTMGPConfig.use_optimal_angle: bool = False` |
| `tests/.../test_rrtmgp_stratosphere.py` | **32 new tests** across 6 classes |

## Codex adversarial reviews
- **Iter-1+2 review** (post-cancel rerun): VERDICT FIX → MEDIUM (silent
  degrade when fit missing) + LOW (tau-clamp doc). Both addressed.
- **Iter-3 review**: VERDICT SHIP → coverage gap at 200-500 hPa flagged.
  Addressed in iter-3.5 (tropospheric baseline).
- Hung review on iter-1 cancelled; consolidated into iter-2.5.

## Status
- ✅ 32 RRTMGP-stratosphere tests + 61 existing radiation + 10 AMIP-RRTMG
  integration = **103 tests pass**, 2 skipped (multidevice MPI).
- ✅ Zero regressions across iter-1 → iter-9.
- ✅ AD-safe end-to-end; mixed precision supported via internal cast to
  table dtype; MPI/GPU-friendly (`jax.lax.scan` + `jax.checkpoint` per
  g-point; embarrassingly parallel per rank; no cross-rank state).

## Deferred to iter 10+
- Default-enable `use_optimal_angle=True` after upstream RFMIP
  reference-flux validation.
- Replicate-boundary halo for T/q_v in `solve_columns` (clip in
  `gas_optics` already handles; defer).
- SW direct-beam Taylor expansion for very high tau (currently OK
  via exp underflow → 0 with zero gradient).
- Audit cloud-optics for stratosphere effects (already clip + safe-div).
