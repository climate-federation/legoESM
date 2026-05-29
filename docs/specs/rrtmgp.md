# RRTMGP Faithfulness Audit — Tracking

Goal: `src/.../rrtmgp/` faithful to
https://github.com/earth-system-radiation/rte-rrtmgp, fully JAX-
differentiable, mixed-precision-safe, MPI/GPU-scalable.

Compression history: iter-10 / 20 / 30 / 40 / 50 / 60 / 70 (this file via
`docs/archive/rrtmgp.{original,iter20,iter30,iter40,iter50,iter60,iter70}-backup.md`).

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

Iter-59 audit confirms #312 touched ZERO convection/microphysics/BL
files; no analogous losses possible outside radiation footprint.

Codex SHIP verdicts: iter-3 (O3), iter-18 (iter-13→17), iter-40
(cache work), iter-58 (dead-code series iter-50→57), iter-69
(test series iter-63→68 — 2 blockers found + fixed).

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
- iter-39+42: 0-D arrays → value-hash with dtype-kind gate.
- iter-41: cloud-optics None-guard + direct-bypass ValueError.
- iter-45→48: trim 537 LOC dead `interpolation.py` + 5
  `kernel_ops` + 2 constants re-exports + 4 F401 imports.

### Dead-code trim (iter-50 → 62)
~1113 LOC of dead swirl_jatmos code removed across iter-22→62:
clear-sky pin (iter-51), reconstruct_vmr_fields path (iter-52),
from_config factories (iter-53), AtmosphericStateCfg +
RadiativeTransfer (iter-54), utils/ subdirectory (iter-55),
interpolate_orig + evaluate_weighted_lookup (iter-56),
deprecated `_cache_key` alias (iter-57), AIMIP-#312 audit +
Metal-env skip (iter-59), `recurrent_op_1d{,_scan}` (iter-60),
`GrayAtmosphereOptics` class + config + factory branch
(iter-61), `_shift_up` + Planck-path clarifier (iter-62).

### Test coverage expansion (iter-63 → 72)
- iter-63: `TestMixedPrecision` — fp32-vs-fp64 numerical
  equivalence within fp32 round-off (x64-gated, skips otherwise);
  bfloat16-input finiteness.
- iter-64: column-local invariance —
  `solve_columns(global)[k:k+M] ≡ solve_columns(global[k:k+M])`.
  Combined with iter-11 permutation-invariance, pins RRTMGP as
  embarrassingly parallel over the column axis.
- iter-65: `test_jax_grad_finite_at_mesosphere_pressure_boundary`
  — AD safety for columns extending below `p_ref[-1] = 1.005 Pa`
  (≈ 65 km altitude), exercising `_pressure_interpolant`'s
  log-space extrapolation path.
- iter-66: `TestAerosolPath` — AOD=zeros ≡ AOD=None, aerosol
  reduces surface SW (sign), ∂(SW)/∂(AOD) finite + negative.
- iter-67: `TestEnergyConservation` — combined LW+SW and LW-only
  column flux divergence ≡ ∑(hr · dp) × c_p / g.
- iter-68: `TestCloudPath` — LWP=zeros ≡ LWP=None, cloud-albedo
  on SW, cloud-greenhouse on LW (sign), ∂(F)/∂(LWP) finite +
  sign-correct for both SW and LW.
- iter-69: codex review of iter-63→68 series.  HOLD on 2 real
  blockers (1 prompt-only confusion ignored): (a) aerosol surface
  index off-by-one (`argmax(p_full)` is full-level idx but flux
  arrays are interface-dim) → fixed to `[0, -1]`; (b) fp32-vs-fp64
  test needs x64 guard → added `pytest.skip` when x64 off.
- iter-70: `TestCloudPath` ice extension — cirrus IWP reduces TOA
  OLR (cold-cloud greenhouse), ∂(OLR)/∂(IWP) finite + column-sum
  negative.  Closes the cloud_path_ice gap (codex iter-69 Q5).
- iter-71: `TestRteRecurrenceScanEquivalence` — GPU(`lax.scan`) ≡
  CPU(for-loop) for `rte_utils.recurrent_op_with_halos`: forward +
  reverse recurrence, reverse-mode AD gradient, and end-to-end
  `monochromatic_two_stream.{lw,sw}_transport` flux_{up,down,net}.
  Pins the `rrtmgp_use_scan` GPU perf knob as numerically inert
  (cross-backend safety); `test_production_blockers` only pinned
  its config routing, not flux/gradient equality.
- iter-72: `TestSolverCompilationStability` — `solve_columns` under
  `jax.jit` compiles once per shape; a value-only change does NOT
  retrace (only an `nlev` change does).  Pins the #1 GPU-throughput
  property (a value-dependent retrace recompiles every step); a
  trace-counter idiom catches it.

### Test classes in `test_rrtmgp_stratosphere.py` (~66 tests)
`TestClipToTableRange`, `TestRelativeAbundanceSafeDiv`,
`TestOutOfRangeTemperature`, `TestMixedPrecision`,
`TestOptimalLwSecant`, `TestStandardO3Profile`,
`TestTropopauseBoundary`, `TestADSafetyAtStratosphereTau`,
`TestCloudKwargsHelper`, `TestHeatingRateSign`,
`TestEnergyConservation`, `TestCloudPath`, `TestAerosolPath`,
`TestRteRecurrenceScanEquivalence`, `TestSolverCompilationStability`.
`test_radiation.py::{TestRRTMGP,TestColumnShardedRadiation}` —
cache-key + sharded-equivalence + iter-13/15 end-to-end pins.

---

## Status (after iter-72)

- ✅ 132 pass + 6 skip (3 Metal-broken + 2 multidevice +
  1 fp32-vs-fp64 x64-gated) — 0 fail.
- ✅ AD-safe end-to-end; mixed precision via table-dtype cast;
  fp32 inputs OK; bfloat16 inputs accepted.
- ✅ MPI/GPU: `lax.scan + jax.checkpoint` per g-point;
  column-permutation invariant; column-local invariance pinned
  (subset ≡ slice of full); shard-equivalent for default and
  `use_optimal_angle=True` paths.  Vertical-recurrence GPU path
  (`use_scan=True`, `recurrent_op_with_halos`) pinned numerically
  identical to the CPU for-loop for fluxes **and** gradients
  (iter-71) — the perf knob never changes answers.  `solve_columns`
  under `jax.jit` is compilation-stable: one compile per shape, no
  value-dependent retrace (iter-72) — no per-step recompile on GPU.
- ✅ ~1113 LOC dead swirl_jatmos code removed across iter-22 → 62.
- ✅ Physical sign pinned: cloud-albedo on SW, cloud-greenhouse
  on LW, aerosol reduces surface SW, scattering aerosol has
  negative ∂(SW)/∂(AOD).
- ✅ Energy conservation pinned: ∑(hr · dp) ≡ (g/c_p) ·
  (F_net_TOA − F_net_sfc) for both combined and LW-only.

## Deferred

- Default-enable `use_optimal_angle=True` pending upstream RFMIP
  reference-flux validation (iter-29 gating sharded test exists).
- LW aerosol path (currently only SW; upstream rte-rrtmgp
  supports both).
- cloud_fraction / solar_spectral_fraction / o3_vmr /
  ghg_vmr_override coverage gaps (codex iter-69 Q5; cloud_path_ice
  closed iter-70).
