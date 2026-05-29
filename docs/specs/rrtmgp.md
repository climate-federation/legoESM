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
`TestTropopauseBoundary`, `TestADSafetyAtStratosphereTau`
(iter-20 + iter-65 mesospheric-boundary),
`TestCloudKwargsHelper`, `TestHeatingRateSign`,
`TestAerosolPath` (iter-66: AOD-zero ≡ AOD-None, sign of AOD on
SW, ∂(SW)/∂(AOD) finite),
`TestEnergyConservation` (iter-67: combined and LW-only column
flux divergence ≡ ∑(hr · dp) × c_p / g),
`TestCloudPath` (iter-68: LWP-zero ≡ LWP-None, cloud-albedo on
SW, cloud-greenhouse on LW, ∂(F)/∂(LWP) finite + sign-correct).

`test_radiation.py::{TestRRTMGP,TestColumnShardedRadiation}` —
cache-key (iter-32/36/37/39/41/42), sharded-equivalence
(iter-28/29), iter-13 sign / iter-15 cf end-to-end pins.

---

## Status (after iter-69)

- ✅ 126 pass + 6 skip (3 Metal-broken + 2 multidevice + 1
  fp32-vs-fp64 needs x64) — 0 fail.
  iter-69: codex adversarial-review of iter-63 → iter-68 returned
  HOLD with 2 real blockers (1 prompt-only confusion ignored).
  Both fixed in iter-69 commit:
  1. iter-66 ``test_aerosol_reduces_toa_sw_down`` used
     ``argmax(p_full)`` as the surface index — but that's a full-
     level index in [0, nlev-1] when flux arrays are interface-
     dimensioned (ncol, nlev+1).  Fixed to ``[0, -1]`` (the
     surface interface), matching the cloud/energy tests.
  2. iter-63 ``test_solve_columns_fp32_matches_fp64_inputs``
     needed an x64 guard: without ``JAX_ENABLE_X64=1`` JAX
     silently downcasts ``jnp.float64`` to fp32, making the test
     a trivial fp32-vs-fp32 comparison that would pass while
     masking a real missing-cast bug.  Added
     ``pytest.skip`` when x64 is unavailable.
  iter-68 added new class `TestCloudPath` (4 tests): LWP=zeros ≡
  LWP=None bit-for-bit, low cloud reduces surface SW (albedo),
  low cloud increases surface LW down (greenhouse), AD through
  cloud_path_liq finite + sign-correct for both SW and LW.
  Pre-iter-68 cloud coverage was structural (kwargs schema,
  on-vs-off-changes-flux) — no physical-sign or AD pin.
  iter-67 added new class `TestEnergyConservation` (2 tests):
  combined LW+SW and LW-only column flux-divergence ≡
  ∑(hr · dp) × c_p / g.  Pre-iter-67 only `gray_radiation` had
  this pin; RRTMGP energy conservation was unpinned despite the
  iter-13 sign-fix territory.
  iter-63 added 2 mixed-precision tests.  iter-64 added 1 MPI-
  shard correctness test (subset call ≡ slice of full call).
  iter-65 added 1 mesospheric-pressure AD test —
  `test_jax_grad_finite_at_mesosphere_pressure_boundary` — pinning
  that `jax.grad` is finite for columns extending below
  `p_ref[-1] = 1.005 Pa` (≈ 65 km altitude).
  iter-66 added new class `TestAerosolPath` (3 tests):
  AOD=zeros bit-equivalence vs AOD=None, aerosol reduces surface
  SW (sign), AD safety through AOD.  Pre-iter-66 the only aerosol
  coverage was a cache-key test in `test_radiation.py` (iter-32) —
  no value/sign/diff pin.
- ✅ AD-safe end-to-end; mixed precision via table-dtype cast;
  fp32 inputs OK.
- ✅ MPI/GPU: `lax.scan + jax.checkpoint` per g-point;
  column-permutation invariant; shard-equivalent for default and
  `use_optimal_angle=True` paths.
- ✅ ~1113 LOC dead swirl_jatmos code removed across iter-22 → 60.

## Deferred

- Default-enable `use_optimal_angle=True` pending upstream RFMIP
  reference-flux validation (iter-29 gating sharded test exists).
