# RRTMGP Faithfulness Audit — Tracking

Goal: `src/.../rrtmgp/` faithful to
https://github.com/earth-system-radiation/rte-rrtmgp, fully JAX-
differentiable, mixed-precision-safe, MPI/GPU-scalable.

Compression history: iter-10 / 20 / 30 / 40 (this file via
`docs/archive/rrtmgp.{original, iter20-backup, iter30-backup,
iter40-backup}.md`).

---

## Initial audit (iter-1, 2026-05-28)

Reviewed `optics/{gas_optics, optics_utils, lookup_*, optics,
optics_base, cloud_optics}.py`, `rte/{two_stream,
monochromatic_two_stream, rte_utils}.py`, `rrtmgp.py` against upstream
Fortran kernels + `mo_gas_optics_rrtmgp.F90` (lines 300/303/519/1030
/1354/1555).

**Verified faithful**: log(p)-interp with `+itropo-1` shift; `kmajor`
(n_t, n_p+1, n_eta, n_gpt); tropopause boundary; `col_mix ≡
combined_vmr·col_dry`; eta interp; Rayleigh `(1+vmr_h2o)·col_dry`;
Planck-face geom-mean; `t_planck=linspace(160, 355, 196)`.

**Bugs flagged → all fixed**:

| ID | Severity | Site | Fix |
|---|---|---|---|
| BUG-1 | critical | planck_sources T-extrap (T^4) | iter-1+2 (clip) |
| BUG-2 | moderate | major/minor/Rayleigh/pfrac T-extrap | iter-1+2 (clip) |
| BUG-3 | AD | `where(combined_vmr>0,…)` NaN grad in (0,eps] | iter-1+2/5 (safe-div) |
| BUG-4 | perf | minor OD 2× compute via both-branch where | deferred (functional) |
| BUG-5 | climatology | O3 σ=1.5 → 10× tropopause O3 | iter-3 + 3.5 |
| OPT-MISS | faithfulness | hard-coded LW_DIFFUSIVE=1.66 vs upstream optimal_angle_fit | iter-2 (opt-in) |

---

## Production-critical fixes silently reverted by AIMIP-#312 merge

Three pre-merge `Fix RRTMG*` commits were wholesale-clobbered when
the `b5b5954e Aimip (#312)` merge replaced files instead of cherry-
picking AIMIP additions onto current main:

| Iter | Lost commit | Pre-revert impact | Restored at |
|---|---|---|---|
| 13 | `0be22f0f` heating-rate sign | Thermal runaway T̄ 261→293K/120d, NaN day 125 | `two_stream.py:compute_heating_rate` |
| 14 | `59407953` 5 AD-unsafe max-floors | NaN grads 4/5 AMIP tunable params | `cloud_optics.py`, `monochromatic_two_stream.py` (×4), `two_stream.py` |
| 15 | `4c9591bb` cloud-fraction cf² discount | −59 W/m² OSR at cf=0.6, −113 at cf=0.3 | `driver/physics_pipeline.py` |
| 16 | (same `4c9591bb` at 2nd site) | identical | `radiation/integration.py` |
| 17 | (centralised cf-omit) | regression at 3rd site impossible | `clouds/cloud_fraction.py::CloudProperties.to_rrtmg_kwargs` |

Codex iter-18 reviewed iter-13 → 17 → **VERDICT: SHIP**.
Codex iter-3 reviewed iter-3 → SHIP (with iter-3.5 closing 200-500 hPa gap).

---

## Iter-by-iter summary (compressed)

### Stratosphere fidelity + optimal LW diffusivity

| Iter | What | Where |
|---|---|---|
| 1+2 | `_clip_to_table_range` at every table interp entry; safe-div `combined_vmr`; load `optimal_angle_fit`; `_compute_optimal_lw_secant` (upstream `c0·exp(−Σtau)+c1`); `solve_lw(use_optimal_angle=…)` opt-in; cache-key honours it | `gas_optics.py`, `lookup_gas_optics_longwave.py`, `monochromatic_two_stream.py`, `two_stream.py`, `rrtmgp.py`, `config.py` |
| 3 / 3.5 | Skewed-Gaussian O3 + 20 ppb tropospheric baseline (codex iter-3 follow-up: closed 200-500 hPa coverage gap) | `rrtmgp.py:_standard_o3_profile` |
| 4 | Thread `precomputed_lw_optical_props` to eliminate duplicate optics call per g-point | `two_stream.py` |
| 5 | Tighten where `combined_vmr > eps` (was `> 0` → 1/eps grad in (0,eps]) | `gas_optics.py` |
| 9 | Defensive safe-div for `vmr_ref[0]/vmr_ref[1]` | `gas_optics.py` |
| 19 | Extend safe_divide to codex iter-18 survivors: `_apply_delta_scaling_for_cloud` + `combine_optical_properties` | `optics.py`, `optics_base.py` |

### Cache + dead code (iter-22 → 24, 32 → 39)

| Iter | What | Δ |
|---|---|---|
| 22-24 | Delete dead swirl_jatmos `compute_heating_rate` API: method (-240), `__init__` (-22), 4 helpers + 3 imports (-32), `rrtmgp_common.py` + `stretched_grid_util.py` (-141), 4 dead instance vars (-4) | -452 lines |
| 32 | **Cache-key bug fix**: `_instance_cache_key` was missing `S_0, aerosol_ssa, aerosol_g, sfc_emissivity, sfc_albedo` → silent stale-solver reuse on config tweaks.  Added `_hashable` shim for AIMIP arrays | +28 lines |
| 33 | Drop redundant `instance.atmospheric_state` field (7 fields stored to expose `.vmr`) | -11 lines |
| 36 | Move `include_clouds` from optics key to instance key (RRTMOptics loads cloud tables unconditionally → 2× dedup) | -1 +2 |
| 39 | 0-D arrays in `_hashable` → value-hash instead of `id()` (prevents fresh cache entry per `jnp.array(0.07)` call) | +8 |

### Regression tests (cumulative ~55 tests across 10 classes)

| Iter | Class / test | Pins |
|---|---|---|
| 1-9 | `TestClipToTableRange`, `TestRelativeAbundanceSafeDiv`, `TestOutOfRangeTemperature`, `TestMixedPrecision`, `TestOptimalLwSecant` | 18 base tests for clip / safe-div / optimal-angle |
| 3+6 | `TestStandardO3Profile` | peak loc, factor-2 fit at 100/30/10/1 hPa, 200-400 DU column |
| 7 | scan/loop equivalence for `use_optimal_angle=True` | 1e-10 tol |
| 8 | optimal vs fixed-1.66 LW flux delta ∈ (1e-6, 10%) | sandwich bound |
| 11 | column-permutation invariance | MPI/GPU shard guard |
| 12 | `halo_width=0` formula match | hw=1 hard-code regression |
| 13 | `TestHeatingRateSign` | iter-13 sign-fix guard |
| 17 | `TestCloudKwargsHelper` | iter-15+16 cf-fix guard |
| 20 | AD safety through pure-stratosphere column | iter-14/19 floor-fix guard |
| 21 / 25 | secant ∈ [1.0, 2.0]; c0=0 collapses to c1 | catches axis/order/coef bugs |
| 26 | all-stratosphere / all-troposphere AD-finite | dead-branch NaN guard |
| 27 | fp32 inputs end-to-end finite | mixed-precision smoke |
| 28 / 29 | sharded RRTMGP matches unsharded (default + `use_optimal_angle=True`) | MPI/GPU |
| 31 | per-class iter-coverage table in file docstring | self-doc |
| 32 / 34 | cache-key fix (key + end-to-end aerosol_ssa flip pin) | iter-32 guard |
| 36 / 37 | include_clouds: optics key dedup + end-to-end flux flip pin | iter-36 guard |
| 39 | 0-D array value-hash | iter-39 guard |

---

## Status (after iter-39)

- ✅ **~115 tests pass** (61 existing radiation + ~45 stratosphere
  + 10 AMIP-RRTMG integration); 2 multidevice MPI skipped.
- ✅ Zero regressions across iter-1 → iter-39.
- ✅ AD-safe end-to-end (codex iter-18 SHIP; iter-30 documented two
  legacy `max(d,eps)` sites kept on purpose).
- ✅ Mixed precision via internal table-dtype cast; fp32 inputs OK.
- ✅ MPI/GPU: `lax.scan + jax.checkpoint` per g-point; embarrassingly
  parallel; column-permutation invariant; shard-equivalent for both
  default and `use_optimal_angle=True`.
- ✅ ~452 lines dead swirl_jatmos compat code removed.
- ✅ Cache: optics key dedup'd; instance key complete; 0-D arrays
  value-hash; AIMIP array `id()` for N-D.

## Deferred to iter 40+

- Default-enable `use_optimal_angle=True` after upstream RFMIP
  reference-flux validation (iter-29 added the gating sharded test).
- Audit other physics modules for analogous AIMIP-#312-style
  feature-merge losses (radiation fully audited;
  convection / microphysics / BL not checked).
- Possibly skip `cloud_optics_lw/sw` loading when
  `include_clouds=False` (memory save ~MB per solver; iter-36 made
  this a possible follow-up since the load is now decoupled from
  the optics cache key).
