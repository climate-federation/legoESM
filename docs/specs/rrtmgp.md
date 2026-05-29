# RRTMGP Faithfulness Audit — Tracking

Goal: `src/.../rrtmgp/` faithful to
https://github.com/earth-system-radiation/rte-rrtmgp, fully JAX-
differentiable, mixed-precision-safe, MPI/GPU-scalable.

Compression history: iter-10 / 20 / 30 / 40 / 50 / 60 / 70 / 80 (this file
via `docs/archive/rrtmgp.{original,iter20,…,iter70,iter80}-backup.md`).

Status: **codex-APPROVE at iter-81** — the iter-70→81 RRTMGP surface is
shippable; core faithfulness audit complete (see Status / Deferred).

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

Codex verdicts: SHIP iter-3 (O3), iter-18 (iter-13→17), iter-40 (cache),
iter-58 (dead-code iter-50→57), iter-69 (tests iter-63→68 — 2 blockers
fixed); iter-79→80 (tests iter-70→78 — needs-attention, 3 medium fixed
across passes); **APPROVE iter-81** (iter-70→81, no material findings).

---

## Iter buckets

### Stratosphere fidelity + optimal LW diffusivity (iter-1 → 9, 19)
`_clip_to_table_range` at every table interp.  Safe-divide for
`combined_vmr` (iter-1+2/5), `vmr_ref_ratio` (iter-9), delta-scaling
+ `combine_optical_properties` (iter-19).  Load `optimal_angle_fit`,
add `_compute_optimal_lw_secant`, opt-in via `use_optimal_angle`
config flag; `precomputed_lw_optical_props` dedups optics call (iter-4).

### Cache integrity (iter-22 → 47)
iter-22 delete dead swirl_jatmos `compute_heating_rate` API (-456 LOC);
iter-32 cache-key bug missing 5 baked fields → stale-solver reuse
(`_hashable` shim); iter-33 drop redundant `atmospheric_state` field;
iter-36+40 `include_clouds` keyed on optics-cache + conditional cloud-table
load; iter-39+42 0-D arrays → value-hash w/ dtype-kind gate; iter-41
cloud-optics None-guard + direct-bypass ValueError; iter-45→48 trim 537 LOC
dead `interpolation.py` + 5 `kernel_ops` + re-exports + F401.

### Dead-code trim (iter-50 → 62)
~1113 LOC dead swirl_jatmos removed (iter-22→62): clear-sky pin (51),
reconstruct_vmr_fields (52), from_config factories (53), AtmosphericStateCfg
+ RadiativeTransfer (54), utils/ (55), interpolate_orig +
evaluate_weighted_lookup (56), `_cache_key` alias (57), #312 audit +
Metal-env skip (59), `recurrent_op_1d{,_scan}` (60), `GrayAtmosphereOptics`
(61), `_shift_up` + Planck clarifier (62).

### Test coverage expansion + LW-aerosol feature (iter-63 → 81)
All in `test_rrtmgp_stratosphere.py` unless noted.  ✅ = codex-approved.

| iter | class / change | invariant pinned |
|---|---|---|
| 63 | TestMixedPrecision | fp32≡fp64 within fp32 round-off (x64-gated); bf16 inputs finite |
| 64 | column-local invariance | `solve(global)[k:k+M] ≡ solve(global[k:k+M])`; with iter-11 perm-inv ⇒ embarrassingly parallel / column |
| 65 | mesosphere AD | grad finite for columns below `p_ref[-1]=1.005 Pa` (log-extrap path) |
| 66 | TestAerosolPath (SW) | AOD=0≡None; aerosol ↓ sfc SW; ∂SW/∂AOD finite<0 |
| 67 | TestEnergyConservation | ∑(hr·dp) ≡ (g/cp)(F_net_TOA−F_net_sfc), combined + LW-only |
| 68 | TestCloudPath | LWP=0≡None; cloud albedo(SW)+greenhouse(LW) signs; ∂F/∂LWP finite |
| 69 | codex review 63→68 | fixed aerosol sfc index→`[0,-1]`; fp32-vs-fp64 x64 skip-guard |
| 70 | TestCloudPath ice | cirrus IWP ↓OLR; ∂OLR/∂IWP finite, col-sum<0 |
| 71 | TestRteRecurrenceScanEquivalence | scan≡for-loop (fwd/rev/AD + lw/sw_transport); `rrtmgp_use_scan` numerically inert (GPU↔CPU) |
| 72 | TestSolverCompilationStability | `solve_columns` under jit compiles 1×/shape; no value-dependent retrace |
| 73 | TestGasVmrOverride | ghg_vmr_override + o3_vmr reach **major+minor** optics (5×CH4↓OLR guards minor path); ∂OLR/∂CO2<0 |
| 74 | TestCloudFractionCoverage | cf linear-OD ×1 (cf=0≡clear, cf=1≡None exact, cf=0.5 monotone), greenhouse/albedo ∂signs |
| 75 | TestSolarSpectralFraction | table-weights≡default; sum-norm invariant; spectral; len-check ValueError; AD |
| 76 | **FEATURE** TestLwAerosolPath | LW aerosol (pure-absorbing ssa=0) added to OD in `solve_lw`, injected before optimal-angle secant; opt-in, byte-identical when unused |
| 77 | TestDeltaScaling | SW cloud delta-Eddington f=ωg²: τ'=(1−f)τ, ω'=ω(1−g²)/(1−f), g'=g/(1+g); conservative ω=1 + isotropic g=0 + AD |
| 78 | TestRayleighScattering | Rayleigh OD≥0; ∝ molecules (exact); spectral 1/λ⁴ (max/min≫10); ∂/∂molecules>0 |
| 79 | codex review 70→78 (needs-attn) | (a) LW aerosol param→`aerosol_absorption_optical_depth_lw` + "absorption NOT extinction" docs; (b) replaced vacuous optimal-angle test |
| 80 | codex re-review | optimal-angle test → white-box capture: spy `_compute_optimal_lw_secant`, assert its Στ grows by exactly the injected aerosol OD per g-point |
| 81 | codex 3rd pass → **APPROVE** | `rrtmgp_radiation()` shim now forwards `aerosol_absorption_optical_depth_lw` + `cloud_fraction` (+ reachability test); no material findings |

Test classes: `TestClipToTableRange`, `TestRelativeAbundanceSafeDiv`,
`TestOutOfRangeTemperature`, `TestMixedPrecision`, `TestOptimalLwSecant`,
`TestStandardO3Profile`, `TestTropopauseBoundary`,
`TestADSafetyAtStratosphereTau`, `TestCloudKwargsHelper`,
`TestHeatingRateSign`, `TestEnergyConservation`, `TestCloudPath`,
`TestAerosolPath`, `TestRteRecurrenceScanEquivalence`,
`TestSolverCompilationStability`, `TestGasVmrOverride`,
`TestCloudFractionCoverage`, `TestSolarSpectralFraction`,
`TestLwAerosolPath`, `TestDeltaScaling`, `TestRayleighScattering`.
`test_radiation.py::{TestRRTMGP,TestColumnShardedRadiation}` — cache-key +
shard-equivalence + iter-13/15 end-to-end pins.

---

## Status (after iter-81 — codex APPROVE)

- ✅ 160 pass + 6 skip (3 Metal-broken + 2 multidevice + 1 fp32-vs-fp64
  x64-gated) — 0 fail.  `test_radiation.py` 68p/5s.
- ✅ AD-safe end-to-end; mixed precision via table-dtype cast; fp32 + bf16
  inputs OK.
- ✅ MPI/GPU: `lax.scan + jax.checkpoint` per g-point; column-permutation +
  column-local invariant; shard-equivalent (default + `use_optimal_angle`).
  Vertical-recurrence GPU path (`use_scan=True`) numerically ≡ CPU for-loop
  for fluxes **and** gradients (iter-71).  `solve_columns` jit
  compilation-stable: 1 compile/shape, no value-retrace (iter-72).
- ✅ Physical signs: cloud albedo(SW)/greenhouse(LW); aerosol ↓ sfc SW,
  scattering aerosol ∂SW/∂AOD<0; cloud-fraction OD scaling ×1 (iter-74);
  delta-Eddington forward-peak removal (iter-77); Rayleigh ∝ air, 1/λ⁴
  (iter-78).
- ✅ Energy conservation: ∑(hr·dp) ≡ (g/cp)(F_net_TOA−F_net_sfc).
- ✅ Gas-VMR overrides faithful (major+minor, `get_vmr` precedence), AD-diff
  (iter-73).  All codex iter-69-Q5 coverage gaps closed (cloud_path_ice 70,
  o3/ghg 73, cloud_fraction 74, solar_spectral_fraction 75).
- ✅ LW aerosol supported (iter-76/79–81): `aerosol_absorption_optical_depth_lw`
  (absorption OD, ssa=0) in `solve_lw` + `rrtmgp_radiation()` shim; opt-in,
  byte-identical when unused, AD-safe, secant sees aerosol τ (pinned).
- ✅ ~1113 LOC dead swirl_jatmos code removed (iter-22→62).

## Deferred

- Default-enable `use_optimal_angle=True` — blocked on upstream RFMIP
  reference-flux validation data (iter-29 gating sharded test exists).
- 3-D driver lacks an LW-aerosol data source: `physics_pipeline` sources SW
  `aerosol_od_col` + `cloud_fraction` but not LW AOD; auto-sourcing it
  (config/state field, symmetric to SW) is future work — the path is
  reachable via the public API/shim.
- LW aerosol scattering (ssa/asymmetry inputs) + per-band/species optical-
  property table (only the pure-absorbing ssa=0 limit is wired).
