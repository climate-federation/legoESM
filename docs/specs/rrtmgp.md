# RRTMGP Faithfulness Audit — Tracking

Goal: `src/.../rrtmgp/` faithful to
https://github.com/earth-system-radiation/rte-rrtmgp, fully JAX-
differentiable, mixed-precision-safe, MPI/GPU-scalable.

Compression history: iter-10 / 20 / 30 / 40 / 50 (this file via
`docs/archive/rrtmgp.{original,iter20,iter30,iter40,iter50}-backup.md`).

---

## Initial audit (iter-1, 2026-05-28)

Reviewed `optics/{gas_optics, optics_utils, lookup_*, optics,
optics_base, cloud_optics}.py`, `rte/{two_stream,
monochromatic_two_stream, rte_utils}.py`, `rrtmgp.py` against
upstream Fortran kernels + `mo_gas_optics_rrtmgp.F90` (lines
300/303/519/1030/1354/1555).

**Verified faithful**: log(p)-interp + `+itropo-1` shift; `kmajor`
(n_t, n_p+1, n_eta, n_gpt); tropopause boundary; `col_mix`; eta
interp; Rayleigh; Planck-face geom-mean; `t_planck`.

**Bugs flagged → all fixed**:

| ID | Sev | Site | Fix |
|---|---|---|---|
| BUG-1 | critical | planck_sources T-extrap (T^4) | iter-1+2 (clip) |
| BUG-2 | moderate | major/minor/Rayleigh/pfrac T-extrap | iter-1+2 (clip) |
| BUG-3 | AD | `where(combined_vmr>0,…)` NaN grad | iter-1+2/5 |
| BUG-5 | climatology | O3 σ=1.5 → 10× tropopause O3 | iter-3 + 3.5 |
| OPT-MISS | faithfulness | hard-coded LW_DIFFUSIVE=1.66 | iter-2 (opt-in) |

---

## Production-critical fixes silently reverted by AIMIP-#312 merge

Three pre-merge `Fix RRTMG*` commits were wholesale-clobbered when
`b5b5954e Aimip (#312)` replaced files instead of cherry-picking
AIMIP additions onto current main:

| Iter | Lost commit | Pre-revert impact | Restored at |
|---|---|---|---|
| 13 | `0be22f0f` heating-rate sign | Thermal runaway T̄ 261→293K/120d | `two_stream.py:compute_heating_rate` |
| 14 | `59407953` 5 AD-unsafe max-floors | NaN grads 4/5 AMIP params | `cloud_optics.py`, `monochromatic_two_stream.py` (×4), `two_stream.py` |
| 15 | `4c9591bb` cloud-fraction cf² | -59 W/m² OSR at cf=0.6, -113 at cf=0.3 | `driver/physics_pipeline.py` |
| 16 | (same `4c9591bb` 2nd site) | identical | `radiation/integration.py` |
| 17 | (centralised cf-omit) | regression at 3rd site impossible | `clouds/cloud_fraction.py::to_rrtmg_kwargs` |

Codex iter-18 SHIP on iter-13 → 17 restorations.  iter-3 codex SHIP
on skewed O3; iter-3.5 closed 200-500 hPa gap.  iter-40 codex SHIP
on iter-32 → 40 cache work after iter-41 addressed the two FIX
findings (cloud-optics None-guard, _hashable dtype gate).

---

## Iter buckets

### Stratosphere fidelity + optimal LW diffusivity (iter-1 → 9, 19)
- `_clip_to_table_range` at every table interp; safe-div for
  `combined_vmr` (iter-1+2/5) and `vmr_ref_ratio` (iter-9); load
  `optimal_angle_fit`; `_compute_optimal_lw_secant`;
  `use_optimal_angle` config flag opt-in; `precomputed_lw_optical_props`
  to dedup optics call (iter-4); safe_divide extended to delta-scaling
  + combine_optical_properties (iter-19).

### O3 climatology (iter-3 + 3.5)
- Skewed-Gaussian (σ_trop=0.9, σ_strat=1.5, peak 9 ppm at 10 hPa) +
  20 ppb tropospheric baseline.

### Cache + dead code (iter-22 → 24, 32 → 47)
- Delete dead swirl_jatmos `compute_heating_rate` API (-456 lines).
- iter-32 cache-key bug: missing 5 baked-in config fields → silent
  stale-solver reuse.  Added `_hashable` shim.
- iter-33: drop redundant `instance.atmospheric_state` field.
- iter-36+40: `include_clouds` cache-key dance — eventually moved
  back to optics key when iter-40 made cloud-table load conditional.
- iter-39+42: 0-D arrays in `_hashable` → value-hash with dtype-kind
  gate (`f`/`i`/`b`/`u` only).
- iter-41: cloud-optics None-guard + clear ValueError on direct-
  bypass.
- iter-45+46: trim 537 lines of dead `interpolation.py` utilities +
  5 unused `kernel_ops` functions.
- iter-47: trim 2 unused constants re-exports.
- iter-48: ruff F401 sweep — 4 real unused imports.

### Regression tests (~60 tests across 10+ classes)

`test_rrtmgp_stratosphere.py` — `TestClipToTableRange`,
`TestRelativeAbundanceSafeDiv`, `TestOutOfRangeTemperature`,
`TestMixedPrecision`, `TestOptimalLwSecant`, `TestStandardO3Profile`,
`TestTropopauseBoundary`, `TestADSafetyAtStratosphereTau`,
`TestCloudKwargsHelper`, `TestHeatingRateSign`.

`test_radiation.py::{TestRRTMGP,TestColumnShardedRadiation}` —
cache-key tests (iter-32/36/37/39/41/42), sharded-equivalence
(iter-28/29), iter-13 sign / iter-15 cf end-to-end pins.

---

## Status (after iter-49)

- ✅ ~120 tests pass (61 existing radiation + ~46 stratosphere + 14
  cache+config + 10 AMIP-RRTMG integration); 2 multidevice MPI
  skipped.
- ✅ Zero regressions across iter-1 → iter-49.
- ✅ AD-safe end-to-end (codex iter-18 + iter-40 SHIP).
- ✅ Mixed precision via internal table-dtype cast; fp32 inputs OK.
- ✅ MPI/GPU: `lax.scan + jax.checkpoint` per g-point;
  embarrassingly parallel; column-permutation invariant; shard-
  equivalent for default and `use_optimal_angle=True`.
- ✅ ~1030 lines of dead swirl_jatmos code removed.
- ✅ Memory: optics cache dedupes via include_clouds → cloud_optics
  load skipped when False (~MB per cached entry).

## Deferred to iter 50+

- Default-enable `use_optimal_angle=True` after upstream RFMIP
  reference-flux validation (iter-29 added the gating sharded test).
- Audit other physics modules for analogous AIMIP-#312-style
  feature-merge losses (radiation fully audited;
  convection / microphysics / BL not checked).
