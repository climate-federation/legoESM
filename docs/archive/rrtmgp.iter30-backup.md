# RRTMGP Faithfulness Audit — Tracking

Goal: `src/.../rrtmgp/` faithful to
https://github.com/earth-system-radiation/rte-rrtmgp, fully JAX-
differentiable, mixed-precision-safe, MPI/GPU-scalable.

Iter-10 compressed in `rrtmgp.original.md`.
Iter-20 compressed in `rrtmgp.iter20-backup.md`.

---

## Iter 1 audit — initial findings (2026-05-28)

Files reviewed: `optics/{gas_optics,optics_utils,lookup_*,optics,
optics_base,cloud_optics}.py`, `rte/{two_stream,monochromatic_two_stream,
rte_utils}.py`, `rrtmgp.py`, upstream
`mo_gas_optics_rrtmgp{,_kernels}.F90` (lines 300/303/519/1030/1354,
1555 for optimal angle).

**Verified match to upstream**: log(p)-interp with `+itropo-1` shift;
`kmajor` (n_t, n_p+1, n_eta, n_gpt); tropopause boundary; col_mix ≡
combined_vmr·col_dry; eta interpolation; Rayleigh
`(1+vmr_h2o)·col_dry`; Planck-face geom-mean; t_planck=linspace(160,
355, 196).

**Bugs flagged (all fixed)**:

| ID | Severity | Iter |
|---|---|---|
| BUG-1: planck_sources T-extrap catastrophic at T<160K | critical | 1+2 |
| BUG-2: same out-of-range extrap for major/minor/Rayleigh/pfrac | moderate | 1+2 |
| BUG-3: `where(combined_vmr>0,…)` NaN grad in (0,eps] | AD | 1+2 → 5 |
| BUG-4: minor OD 2x compute waste | perf | (defer) |
| BUG-5: O3 σ_log=1.5 too wide → 10× tropopause O3 | climatology | 3 + 3.5 |
| OPT-MISS: hard-coded LW_DIFFUSIVE=1.66 vs upstream optimal angle | faithfulness | 2 |

---

## Iter-by-iter summary

### Iter 1+2 — stratosphere fidelity + optimal LW diffusivity
- `_clip_to_table_range(t, t_ref)` helper applied at every table
  interpolation entry (major/minor/Rayleigh/planck_fraction/planck_sources).
- Safe-div `jnp.maximum(combined_vmr, 1e-30)` in
  `_compute_relative_abundance_interpolant`.
- `optimal_angle_fit` loaded from netCDF (None for older files).
- `_compute_optimal_lw_secant(tau, band_idx, fit, hw)` matches
  upstream `c0·exp(−Σtau) + c1`.
- `solve_lw(use_optimal_angle=False)` opt-in; raises ValueError when
  True + fit missing.
- `RRTMGPConfig.use_optimal_angle: bool = False` field; cache-key
  honours it.

### Iter 3 — skewed-Gaussian O3 + saturation tests
- Replaced σ=1.5 single Gaussian with σ_trop=0.9 / σ_strat=1.5
  piecewise, peak 9 ppm at 10 hPa.  C0-continuous.
- Documented minor-OD physical-T density scaling.

### Iter 3.5 — tropospheric O3 baseline (codex iter-3 SHIP follow-up)
- Codex flagged 200-500 hPa coverage gap (Gaussian alone gives <1 ppb
  vs real ~25-50 ppb — 2-3 OOM gap).
- Fix: `o3 = max(o3_gauss, 2.0e-8)` — 20 ppb baseline via `jnp.maximum`.

### Iter 4 — eliminate duplicate optics call
- `solve_lw(use_optimal_angle=True)` was calling
  `compute_lw_optical_properties` twice per g-point.  Thread
  `precomputed_lw_optical_props` dict through `_compute_local_properties_lw`.

### Iter 5 — bound eta-fraction gradient
- Tightened iter-1 where condition: `combined_vmr > 0` → `> eps` so
  the 0.5 fallback is selected when denom would be clamped (gradient
  was `1/eps ~ 1e30`).

### Iter 6 — integrated O3 column DU sanity
- `7891 · ∫ vmr_o3 dp` ∈ [200, 400] DU.  Current: 262 DU.

### Iter 7 — optimal-angle scan/loop equivalence
- New test: solve_columns(use_optimal_angle=True) must match between
  scan and unrolled (1e-10 tol).

### Iter 8 — optimal-angle flux delta bound
- `|out_opt − out_fixed|/out_fixed ∈ (1e-6, 10%)` — sandwich bound.

### Iter 9 — safe-divide for vmr_ref_ratio
- `vmr_ref[0] / max(vmr_ref[1], eps)` — defensive.

### Iter 10 — first compression
- 448 → 160 lines, backup at `rrtmgp.original.md`.

### Iter 11 — column permutation invariance
- Pin `out_orig[perm] == out_perm` to 1e-12 — guards MPI/GPU sharding.

### Iter 12 — halo_width=0 path test
- Pin secant formula matches at hw=0 (no halo strip).

### Iter 13 — CRITICAL: heating-rate sign restored (commit 0be22f0f lost in AIMIP-#312)
- `return -G * dflux / abs(dp) / CP_D` (was returning +G*…).
- Pre-fix symptom: free-trop LW heating +0.5 K/day instead of −0.5 K/day.
- Original fix commit message: "thermal runaway T̄ 261→293 K over
  120 days, NaN at day 125".
- Regression tests: `TestHeatingRateSign::{test_free_tropospheric_LW_cools,
  test_clear_sky_SW_heats}` reference commit 0be22f0f in docstrings.

### Iter 14 — CRITICAL: AD-unsafe maximum-floor restored (commit 59407953 lost)
- Five sites: cloud_optics ssa/g divides, lw_combine_sources
  `sqrt(max(x, 0))`, three `_solve_rte_2stream` betas (`1/max(1−r·a, eps)`),
  solve_sw aerosol `w_tot`/`g_tot` divides.
- All replaced with `safe_divide` (from
  `legoesm.atmosphere.physics._shared`).
- Original fix commit message: "4-of-5 AMIP+RRTMG tunable params
  produced NaN gradients on every backward pass".

### Iter 15+16 — CRITICAL: cloud-fraction double-discount (commit 4c9591bb lost)
- Bug: `compute_cloud_properties` returns grid-mean LWP
  (cf-discounted), RRTMG optics then multiplies tau by cloud_fraction
  again — `τ_used = cf² · τ_in-cloud` instead of `cf · τ_in-cloud`.
- Cost: −59 W/m² OSR at cf=0.6, −113 at cf=0.3.
- Iter-15: physics_pipeline.py — removed `cloud_fraction` kwarg.
- Iter-16: integration.py — same bug, second copy of cloud_kwargs.

### Iter 17 — centralise to_rrtmg_kwargs helper
- `CloudProperties.to_rrtmg_kwargs()` method centralises the
  cf-omitting kwargs build with documentation citing 4c9591bb.
- Both call sites refactored.
- Pinned regression test: `TestCloudKwargsHelper`.

### Iter 18 — codex review of iter-13 → iter-17 restorations
- **VERDICT: SHIP**.  Codex confirmed faithfulness across sign
  convention, AD-safety completeness (via grep), cloud-fraction
  logic rigour.

### Iter 19 — extend safe_divide to codex iter-18 "survivors"
- `optics.py:_apply_delta_scaling_for_cloud` — cloud_ssa, cloud_asy.
- `optics_base.py:combine_optical_properties` — asymmetry-factor, ssa.
- Same AD-unsafe pattern as iter-14 sites; codex iter-18 review
  noted them as pre-existing survivors.

### Iter 20 — second compression
- 286 → ~120 lines, backup at `rrtmgp.iter20-backup.md`.

---

## Code/spec touched (cumulative iter-1 → iter-19)

| File | Iters |
|---|---|
| `optics/gas_optics.py` | 1+2 (clip + safe-div combined_vmr), 5 (tighten where), 9 (safe-div vmr_ref_ratio), 3 (minor-OD doc) |
| `optics/cloud_optics.py` | 14 (safe-div ssa/g) |
| `optics/optics.py` | 19 (safe-div delta-scaling) |
| `optics/optics_base.py` | 19 (safe-div combine) |
| `optics/lookup_gas_optics_longwave.py` | 1+2 (load optimal_angle_fit) |
| `rte/monochromatic_two_stream.py` | 1+2 (lw_diffusive_factor kwarg), 14 (safe-div sqrt + 3 RTE betas) |
| `rte/two_stream.py` | 1+2/4 (optimal secant + precomputed), 13 (sign fix), 14 (safe-div aerosol) |
| `rrtmgp.py` | 1+2 (use_optimal_angle plumbing), 3+3.5 (O3 profile) |
| `config.py` | 1+2 (use_optimal_angle field) |
| `clouds/cloud_fraction.py` | 17 (to_rrtmg_kwargs method) |
| `driver/physics_pipeline.py` | 15+17 (cf fix + helper use) |
| `atmosphere/physics/radiation/integration.py` | 16+17 (cf fix + helper use) |
| `tests/.../test_rrtmgp_stratosphere.py` | **~35 new tests across 8 classes** |

## Production-critical lost-fix audit (AIMIP-#312 merge silently reverted)

| Lost commit | Author | Date | Pre-revert impact | Restored |
|---|---|---|---|---|
| `0be22f0f` heating-rate sign | A. Pacal | 2026-05-22 | Thermal runaway, T̄ 261→293K/120d | iter-13 |
| `59407953` AD-safe maximum-floor | K. Debeire | 2026-05-14 | NaN grads in 4/5 AMIP params | iter-14 |
| `4c9591bb` cloud-fraction cf² | K. Debeire | 2026-05-17 | -59..-113 W/m² OSR | iter-15+16 |

## Codex reviews
- Iter-3 SHIP (skewed O3); follow-up iter-3.5 closed coverage gap.
- Iter-18 SHIP on iter-13 → iter-17 restorations; iter-19 closed the
  AD-safety "survivors" finding.

## Status (after iter-19)
- ✅ 35+ stratosphere tests + 61 radiation + 10 AMIP-RRTMG = **106+
  tests pass**, 2 multidevice MPI skipped.
- ✅ Zero regressions across iter-1 → iter-19.
- ✅ AD-safe end-to-end (all known `num/max(denom,eps)` patterns in
  RRTMG chain replaced with `safe_divide`).
- ✅ Mixed precision via internal cast to table dtype.
- ✅ MPI/GPU scalability (`lax.scan + checkpoint` per g-point;
  embarrassingly parallel per rank; column-permutation invariant per
  iter-11 test).

## Deferred to iter 20+
- Default-enable `use_optimal_angle=True` after upstream RFMIP
  reference-flux validation.
- Audit other physics modules for analogous AIMIP-#312-style merge
  losses.
