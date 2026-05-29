# RRTMGP Faithfulness Audit — Tracking

Goal: `src/.../rrtmgp/` faithful to
https://github.com/earth-system-radiation/rte-rrtmgp, fully JAX-
differentiable, mixed-precision-safe, MPI/GPU-scalable.

Compression history: iter-10 / 20 / 30 / 40 / 50 / 60 (this file via
`docs/archive/rrtmgp.{original,iter20,iter30,iter40,iter50,iter60}-backup.md`).

---

## Initial audit (iter-1, 2026-05-28)

Reviewed `optics/{gas_optics, optics_utils, lookup_*, optics,
optics_base, cloud_optics}.py`, `rte/{two_stream,
monochromatic_two_stream, rte_utils}.py`, `rrtmgp.py` against
upstream Fortran kernels.  **Faithful**: log(p) interp, `+itropo-1`
shift, `kmajor` (n_t, n_p+1, n_eta, n_gpt), tropopause boundary,
`col_mix`, eta interp, Rayleigh, Planck-face geom-mean, `t_planck`.

| ID | Sev | Fix iter |
|---|---|---|
| T-extrap T^4 in planck/major/minor/Rayleigh/pfrac | crit/mod | 1+2 (`_clip_to_table_range`) |
| `where(combined_vmr>0,…)` NaN grad | AD | 1+2/5 (safe-div) |
| σ=1.5 O3 → 10× tropopause | clim | 3 + 3.5 (skewed-Gaussian σ_trop=0.9 σ_strat=1.5, 20 ppb baseline) |
| Hard-coded LW_DIFFUSIVE=1.66 | fidelity | 2 (opt-in via `use_optimal_angle` config) |

---

## AIMIP-#312 silently reverted 3 production fixes (iter-13 → 17)

| Iter | Lost commit | Impact | Restored site |
|---|---|---|---|
| 13 | `0be22f0f` heating-rate sign | T̄ 261→293K/120d | `two_stream.compute_heating_rate` |
| 14 | `59407953` 5 AD-unsafe max-floors | NaN grads 4/5 AMIP | `cloud_optics`, `monochromatic_two_stream` ×4, `two_stream` |
| 15+16 | `4c9591bb` cloud-fraction cf² (2 sites) | -59 to -113 W/m² OSR | `driver/physics_pipeline`, `radiation/integration` |
| 17 | centralised cf-omit | 3rd-site regression impossible | `clouds.cloud_fraction.to_rrtmg_kwargs` |

Iter-59 audit of `git diff --name-status b5b5954e^..b5b5954e`
confirms the merge touched ZERO convection/microphysics/BL files;
no analogous losses possible outside radiation footprint.

Codex SHIP verdicts: iter-3 (O3), iter-18 (iter-13→17), iter-40
(cache work), iter-58 (dead-code series iter-50→57).

---

## Iter buckets

### Stratosphere fidelity + optimal LW diffusivity (iter-1 → 9, 19)
`_clip_to_table_range` at every table interp.  Safe-divide for
`combined_vmr` (iter-1+2/5), `vmr_ref_ratio` (iter-9), delta-scaling
+ `combine_optical_properties` (iter-19).  Load `optimal_angle_fit`,
add `_compute_optimal_lw_secant`, opt-in via `use_optimal_angle`
config flag; `precomputed_lw_optical_props` dedups optics call
(iter-4).

### Cache integrity (iter-22 → 47)
- iter-22: delete dead swirl_jatmos `compute_heating_rate` API
  (-456 LOC).
- iter-32: cache-key bug missing 5 baked-in fields → silent stale-
  solver reuse.  `_hashable` shim.
- iter-33: drop redundant `instance.atmospheric_state` field.
- iter-36+40: `include_clouds` keyed on optics-cache; iter-40
  conditionalised cloud-table load.
- iter-39+42: 0-D arrays → value-hash with dtype-kind gate
  (`f`/`i`/`b`/`u` only).
- iter-41: cloud-optics None-guard + direct-bypass ValueError.
- iter-45+46+47+48: trim 537 LOC dead `interpolation.py` + 5
  `kernel_ops` + 2 constants re-exports + 4 F401 imports.

### Dead-code trim (iter-50 → 61)
- iter-51: clear-sky `solve_columns` regression pin (no cloud kwargs).
- iter-52: `reconstruct_vmr_fields_from_pressure` path.
- iter-53: `from_config` factories on `AtmosphericState`/`LookupVMR`.
- iter-54: `AtmosphericStateCfg` + `RadiativeTransfer` config classes.
- iter-55: dead `utils/` subdirectory.
- iter-56: `interpolate_orig` + `evaluate_weighted_lookup` (-72 LOC).
- iter-57: deprecated `RRTMGP._cache_key` alias.
- iter-58: codex review of iter-50→57 — one docstring cross-ref fix.
- iter-59: AIMIP-#312 cross-module audit (vacuously empty) +
  `@_skip_if_metal_broken` for 3 JAX-Metal env failures.
- iter-60: `recurrent_op_1d` + `recurrent_op_1d_scan` in
  `rte_utils.py` (-83 LOC).
- iter-61: `GrayAtmosphereOptics` class (impl + config) — swirl_jatmos
  RRTMGP-internal gray-radiation impl that legoESM never used
  (legoESM has `legoesm.atmosphere.physics.radiation.gray`).  Drops
  155 LOC class + 12 LOC config + 4 import lines + 2 cast tweaks +
  `optics_factory` dispatch elif branch.  Net -172 LOC.  Factory's
  unknown-scheme `raise ValueError` (per CLAUDE.md dispatch audit)
  preserved + made the error message more informative.
- iter-62: dead `_shift_up` in `optics_base.py` (0 callers; sibling
  `_shift_down` still used by `reconstruct_face_values`).  Plus codex
  HOLD→SHIP follow-up: added comment in `optics_factory` clarifying
  that the removed gray-Planck (Schneider 2004 / O'Gorman 2008) is
  NOT the RRTMGP correlated-k Planck path
  (`RRTMOptics.compute_planck_sources` → `gas_optics.planck_source`
  is unchanged).

### Regression tests (~60 across 10+ classes)
`test_rrtmgp_stratosphere.py` — `TestClipToTableRange`,
`TestRelativeAbundanceSafeDiv`, `TestOutOfRangeTemperature`,
`TestMixedPrecision` (iter-1+2/27/63: planck-fp32 + solve_columns-
fp32-finite + iter-63 **fp32-vs-fp64 numerical equivalence within
fp32 round-off** + iter-63 **bfloat16 input acceptance**),
`TestOptimalLwSecant`, `TestStandardO3Profile`,
`TestTropopauseBoundary`, `TestADSafetyAtStratosphereTau`,
`TestCloudKwargsHelper`, `TestHeatingRateSign`.

`test_radiation.py::{TestRRTMGP,TestColumnShardedRadiation}` —
cache-key (iter-32/36/37/39/41/42), sharded-equivalence
(iter-28/29), iter-13 sign / iter-15 cf end-to-end pins.

---

## Status (after iter-63)

- ✅ 115 pass + 5 skip (3 Metal-broken + 2 multidevice) — 0 fail.
  iter-63 added 2 mixed-precision tests (fp32-vs-fp64 numerical
  equivalence, bfloat16-input acceptance).
- ✅ AD-safe end-to-end; mixed precision via table-dtype cast;
  fp32 inputs OK.
- ✅ MPI/GPU: `lax.scan + jax.checkpoint` per g-point;
  column-permutation invariant; shard-equivalent for default and
  `use_optimal_angle=True` paths.
- ✅ ~1113 LOC dead swirl_jatmos code removed across iter-22 → 60.

## Deferred

- Default-enable `use_optimal_angle=True` pending upstream RFMIP
  reference-flux validation (iter-29 gating sharded test exists).
