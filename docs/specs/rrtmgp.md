# RRTMGP Faithfulness Audit — Tracking

Goal: `src/.../rrtmgp/` faithful to
https://github.com/earth-system-radiation/rte-rrtmgp, fully JAX-
differentiable, mixed-precision-safe, MPI/GPU-scalable.

Compression history: iter-10 (`docs/archive/rrtmgp.original.md`),
iter-20 (`docs/archive/rrtmgp.iter20-backup.md`),
iter-30 (this file via `docs/archive/rrtmgp.iter30-backup.md`).

---

## Initial audit (iter-1, 2026-05-28)

Files reviewed: `optics/{gas_optics, optics_utils, lookup_*, optics,
optics_base, cloud_optics}.py`, `rte/{two_stream, monochromatic_two_stream,
rte_utils}.py`, `rrtmgp.py`; upstream Fortran kernels +
mo_gas_optics_rrtmgp.F90 (lines 300/303/519/1030/1354, 1555 for
optimal angle).

**Verified faithful**: log(p)-interp with `+itropo-1` shift; `kmajor`
(n_t, n_p+1, n_eta, n_gpt); tropopause boundary; `col_mix ≡
combined_vmr·col_dry`; eta interp; Rayleigh `(1+vmr_h2o)·col_dry`;
Planck-face geom-mean; `t_planck=linspace(160, 355, 196)`.

**Bugs flagged → all fixed**:

| ID | Severity | Site | Fix |
|---|---|---|---|
| BUG-1 | critical | planck_sources T-extrap (T^4) | iter-1+2 (clip) |
| BUG-2 | moderate | major/minor/Rayleigh/pfrac T-extrap | iter-1+2 (clip) |
| BUG-3 | AD | `where(combined_vmr>0,…)` NaN grad (0,eps] | iter-1+2/5 (safe-div) |
| BUG-4 | perf | minor OD 2× compute | deferred |
| BUG-5 | climatology | O3 σ=1.5 → 10× tropopause O3 | iter-3 + 3.5 |
| OPT-MISS | faithfulness | hard-coded LW_DIFFUSIVE=1.66 | iter-2 (optimal angle, opt-in) |

---

## Iter-by-iter summary

### Stratosphere/tropopause fidelity + optimal LW diffusivity

| Iter | What | Where |
|---|---|---|
| 1+2 | `_clip_to_table_range` at every table interp entry; safe-div `combined_vmr`; load `optimal_angle_fit`; `_compute_optimal_lw_secant` (upstream `c0·exp(−Σtau)+c1`); `solve_lw(use_optimal_angle=…)` opt-in; `RRTMGPConfig.use_optimal_angle: bool=False`; cache-key honours it | `gas_optics.py`, `lookup_gas_optics_longwave.py`, `monochromatic_two_stream.py`, `two_stream.py`, `rrtmgp.py`, `config.py` |
| 3 | Skewed-Gaussian O3: σ_trop=0.9 / σ_strat=1.5, peak 9 ppm at 10 hPa, C0-continuous; minor-OD physical-T density-scaling doc | `rrtmgp.py`, `gas_optics.py` |
| 3.5 | 20 ppb tropospheric O3 baseline via `max(o3_gauss, 2e-8)` (codex iter-3 SHIP follow-up: 200-500 hPa coverage gap) | `rrtmgp.py` |
| 4 | Thread `precomputed_lw_optical_props` dict to eliminate duplicate `compute_lw_optical_properties` per g-point | `two_stream.py` |
| 5 | Tighten iter-1 where condition `combined_vmr > 0` → `> eps` (gradient was `1/eps ~ 1e30` in (0,eps] band) | `gas_optics.py` |
| 9 | Safe-div for `vmr_ref[0]/vmr_ref[1]` (defensive) | `gas_optics.py` |
| 19 | Extend safe-divide to codex iter-18 survivors: `_apply_delta_scaling_for_cloud` (cloud_ssa/asy) + `combine_optical_properties` (g/ssa) | `optics.py`, `optics_base.py` |

### Regression tests

| Iter | Test class / scenario | What |
|---|---|---|
| 1-9 | `TestClipToTableRange`, `TestRelativeAbundanceSafeDiv`, `TestOutOfRangeTemperature`, `TestMixedPrecision`, `TestOptimalLwSecant` | 18 base tests |
| 3+6 | `TestStandardO3Profile` | peak loc, factor-2 fit at 100/30/10/1 hPa, factor-5 at 1000/500/200/0.1 hPa, 20 ppb floor, 200-400 DU column |
| 7 | scan/loop equivalence for `use_optimal_angle=True` | 1e-10 tol |
| 8 | optimal vs fixed-1.66 LW flux delta ∈ (1e-6, 10%) | sandwich bound |
| 11 | `solve_columns` column-permutation invariance | MPI/GPU sharding guard |
| 12 | `_compute_optimal_lw_secant(halo_width=0)` | pin formula at no-halo |
| 13 | `TestHeatingRateSign::{LW_cools, SW_heats}` | iter-13 sign-fix regression guard |
| 17 | `TestCloudKwargsHelper::test_kwargs_excludes_cloud_fraction` | iter-15+16 cf-fix regression guard |
| 20 | AD safety through pure-stratosphere column | iter-14/19 floor-fix regression guard |
| 21 | optimal-angle secant ∈ [1.0, 2.0] across c0+c1 endpoints | catches axis/ordering bugs |
| 25 | secant = c1 exactly for 11 c0=0 bands | catches coefficient-order regressions |
| 26 | all-stratosphere / all-troposphere AD-finite | dead-branch NaN-cotangent guard |
| 27 | fp32 inputs end-to-end finite | mixed-precision smoke |
| 28 | sharded RRTMGP matches unsharded | RRTMGP column-shard invariance |
| 29 | sharded `use_optimal_angle=True` matches unsharded | optimal-angle path shard-safe |

---

## Production-critical fixes silently reverted by AIMIP-#312 merge

| Iter | Lost commit | Author | Pre-revert impact | Restored at |
|---|---|---|---|---|
| 13 | `0be22f0f` heating-rate sign | A. Pacal 2026-05-22 | Thermal runaway, T̄ 261→293K/120d, NaN at day 125 | `two_stream.py:compute_heating_rate` |
| 14 | `59407953` 5 AD-unsafe maximum-floors | K. Debeire 2026-05-14 | NaN grads in 4/5 AMIP+RRTMG tunable params | `cloud_optics.py`, `monochromatic_two_stream.py` (×4), `two_stream.py` |
| 15 | `4c9591bb` cloud-fraction cf² discount | K. Debeire 2026-05-17 | −59 W/m² OSR at cf=0.6, −113 at cf=0.3 | `driver/physics_pipeline.py` |
| 16 | (same `4c9591bb` at second site) | — | identical | `atmosphere/physics/radiation/integration.py` |
| 17 | (centralised so it can't resurface) | — | — | `clouds/cloud_fraction.py::CloudProperties.to_rrtmg_kwargs` |

All four restorations independently codex-reviewed: **VERDICT: SHIP**
(iter-18).  iter-19 closed the codex "survivors" finding.

---

## Dead-code cleanup (iter-22 → iter-24)

The swirl_jatmos compute_heating_rate API and its supporting modules
had zero callers across `src/` and `tests/`:

| Iter | Removed | Lines |
|---|---|---|
| 22 | `RRTMGP.compute_heating_rate` + 4 helpers + 3 imports | -284 |
| 23 | `RRTMGP.__init__` + Sequence/RadiativeTransfer imports + 4 dead instance vars | -27 |
| 24 | `rrtmgp_common.py` + `stretched_grid_util.py` files | -141 |

Cumulative: ~452 lines of dead swirl_jatmos compat code removed.
Sole entry points to the RRTMGP solver are now
`RRTMGP.from_legoesm_config(rrtmgp_config)` + `solve_columns(...)`.

---

## Codex adversarial reviews

- **Iter-3** SHIP (skewed O3 profile); follow-up iter-3.5 closed
  200-500 hPa coverage gap.
- **Iter-18** SHIP on iter-13 → iter-17 production-fix restorations;
  iter-19 closed the AD-safety "survivors" finding.

---

## Current status (after iter-29)

- ✅ **~110 tests pass** (61 existing radiation + ~45 stratosphere +
  10 AMIP-RRTMG integration); 2 skipped multidevice MPI tests.
- ✅ Zero regressions across iter-1 → iter-29.
- ✅ AD-safe end-to-end: every known `num/max(denom,eps)` and
  `sqrt(max(x,0))` pattern in RRTMG chain replaced with
  `safe_divide` / `max(x, _EPSILON)`.
- ✅ Mixed precision via internal cast to table dtype, fp32 inputs
  verified end-to-end.
- ✅ MPI/GPU scalability: `lax.scan` + `jax.checkpoint` per g-point;
  embarrassingly parallel per rank; column-permutation invariant +
  sharded-equivalence verified for default and `use_optimal_angle=True`.
- ✅ ~452 lines of dead swirl_jatmos compat code removed.

## Deferred to iter 30+

- Default-enable `use_optimal_angle=True` after upstream RFMIP
  reference-flux validation (iter-29 added the gating sharded test).
- Audit other physics modules for analogous AIMIP-#312-style merge
  losses (radiation is fully audited; convection/microphysics/BL
  not yet checked).
