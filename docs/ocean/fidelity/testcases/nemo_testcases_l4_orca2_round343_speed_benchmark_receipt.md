# ORCA2 round 343 — speed and 8-day accuracy benchmark of legoESM ORCA2 (exact vs normal mode)

- **Round**: `orca2_r343`
- **Author**: Argon
- **Branch**: `fidelity/nemo-testcases-orca2-rounds`
- **Preregistration**: `docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l4_orca2_round343_speed_benchmark.md`
- **Benchmark script**: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py:133-1750`
- **Direct unit test**: `tests/ocean/fidelity/test_orca2_round343_speed_benchmark.py:29-200`
- **Evidence bundle**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r343/orca2_round343_speed_benchmark_report.json` (`arm_exact_cpu_fp64.json`, `arm_normal_cpu_fp64.json`, `arm_normal_gpu_fp64.json`, `arm_normal_gpu_fp32.json`)
- **Independent review**: `independent review: operator`

---

## 1. Exact-Mode vs Normal-Mode Switch Inventory (No Defaults Changed)

By reading the production ORCA2 K0 standalone card builder `packages/ocean/legoesm/ocean/bulk_flux_omip.py:854-927` (`build_nemo_orca2_k0_ocean_card`), the ORCA2 testcase recipe `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1942-2222`, and the core precision/transcendental/rounding modules, exact-NEMO mode is governed by two primary runtime mechanisms and 46 cataloged switch entries across four categories:

1. **CPU-only scalar `libm` callback loop**: `packages/core/legoesm/core/precision.py:117-125` (`PrecisionPolicy.fp64(transcendentals="libm")`) routes `tx.exp`, `tx.log`, `tx.log10`, `tx.sin`, `tx.cos`, `tx.tanh`, `tx.atan`, and `tx.pow` through `_scalar_libm` and `_scalar_libm_binary` in `packages/core/legoesm/core/transcendentals.py:48-85`. That path invokes `jax.pure_callback` and executes a Python loop over `range(source_flat.size)` calling `ctypes.CDLL("libm.so.6")` element-by-element on the host CPU (raising `RuntimeError` if the backend is not CPU). Normal mode uses `PrecisionPolicy.fp64(transcendentals="native")` (`packages/core/legoesm/core/precision.py:103-114`) or `PrecisionPolicy.fp32(transcendentals="native")`, which dispatches directly to vectorized XLA `jnp` primitives on CPU or GPU.
2. **Statement-level XLA optimization barriers and signed-zero preservation**: `packages/core/legoesm/core/source_rounding.py:14-35` (`nemo_source_round`, `nemo_source_barrier`) wraps intermediate arithmetic in `jax.lax.optimization_barrier` and `jnp.where(jnp.isfinite(value), value, jnp.copysign(jnp.abs(value), value))` to prevent XLA fused multiply-add (FMA) contraction and operand re-association.

### Complete Switch Table (`build_orca2_exact_vs_normal_switch_inventory`)

All 46 inventory entries verified by `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py:775-837` (`verify_switch_inventory_against_card`) against both `build_nemo_orca2_k0_ocean_card` and fresh class defaults in `packages/` (zero defaults modified):

| # | Category | Owner | Switch / Field | Exact-Mode Value | Normal Default Value | Switched in Normal Mode? | Code Citations (Exact -> Normal Default) |
|---|---|---|---|---|---|---|---|
| 1 | **A** (`libm`-guarded) | `PrecisionPolicy` | `transcendentals` | `"libm"` | `"native"` | Yes | `packages/core/legoesm/core/precision.py:117-125` -> `packages/core/legoesm/core/precision.py:103-114` |
| 2 | **A** (`libm`-guarded) | `NEMOTestcaseCard` | `ocean.transcendentals` | `"libm"` | `"native"` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:66` -> `packages/core/legoesm/core/precision.py:103` |
| 3 | **A** (`libm`-guarded) | `TKEConfig` | `tke_htau_evaluation` | `"nemo_scalar_libm"` | `"jax_expression"` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2152` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:654` |
| 4 | **A** (`libm`-guarded) | `TKEConfig` | `tke_etau_exponential_evaluation` | `"nemo_scalar_libm"` | `"jax_expression"` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2153` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:648` |
| 5 | **A** (`libm`-guarded) | `DoubleDiffusionConfig` | `(enabled, form)` | `(True, "nemo_literal")` | `(False, "hybrid")` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2174-2175` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py:245-251` |
| 6 | **A** (`libm`-guarded) | `IWMConfig` | `(form, tsdiff)` | `("nemo_literal", True)` | `("generic", False)` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2167-2173` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py:187-203` |
| 7 | **B** (rounding/order) | `TKEConfig` | `nemo_bn2_source_rounded` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2137` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:671` |
| 8 | **B** (rounding/order) | `TKEConfig` | `nemo_tke_solve_source_rounded` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2139` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:678` |
| 9 | **B** (rounding/order) | `TKEConfig` | `nemo_tke_matrix_rhs_source_rounded` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2141` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:685` |
| 10 | **B** (rounding/order) | `TKEConfig` | `nemo_sh2_r3_source_rounded` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2142` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:689` |
| 11 | **B** (rounding/order) | `TKEConfig` | `nemo_sh2_r3_stored_reciprocal` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2143` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:693` |
| 12 | **B** (rounding/order) | `TKEConfig` | `tke_mxl_raw_evaluation` | `"nemo_literal"` | `"factored"` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2156` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:658` |
| 13 | **B** (rounding/order) | `TKEConfig` | `tke_solver_evaluation` | `"nemo_literal"` | `"shared_thomas"` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:422` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:642` |
| 14 | **B** (rounding/order) | `TKEConfig` | `tke_langmuir_evaluation` | `"nemo_literal"` | `"vectorized"` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1946` -> `packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:698` |
| 15 | **B** (rounding/order) | `GMRediConfig` | `slope_uv_source_rounding` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2934` -> `packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:417` |
| 16 | **B** (rounding/order) | `GMRediConfig` | `redi_a33_source_ordering` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2935` -> `packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:489` |
| 17 | **B** (rounding/order) | `GMRediConfig` | `redi_a33_evaluation` | `"nemo_literal"` | `"normalized_square"` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2912` -> `packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:485` |
| 18 | **B** (rounding/order) | `GMRediConfig` | `redi_vertical_skew_evaluation` | `"nemo_literal"` | `"normalized_sums"` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2911` -> `packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:480` |
| 19 | **B** (rounding/order) | `GMRediConfig` | `slope_metric_evaluation` | `"nemo_reciprocal"` | `"division"` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2898` -> `packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:412` |
| 20 | **B** (rounding/order) | `MLEConfig` | `nemo_association` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2990` -> `packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py:121` |
| 21 | **B** (rounding/order) | `ZStarCoordinate` | `nemo_r3t_stored_product` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2209` -> `packages/ocean/legoesm/ocean/vertical.py:1531` |
| 22 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `nemo_eos_gdept_rounded_product` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1983` -> `packages/ocean/legoesm/ocean/state.py:3224` |
| 23 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `zdf_tracer_nemo_rounded_thickness` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1976` -> `packages/ocean/legoesm/ocean/state.py:3233` |
| 24 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `barotropic_rhs_ascending_level_sum` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1977` -> `packages/ocean/legoesm/ocean/state.py:3237` |
| 25 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `rk3_stage1_rhs_nemo_literal` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1978` -> `packages/ocean/legoesm/ocean/state.py:3179` |
| 26 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `rk3_stage_velocity_written_order` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1979` -> `packages/ocean/legoesm/ocean/state.py:3184` |
| 27 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `rk3_stage_tracer_written_order` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1991` -> `packages/ocean/legoesm/ocean/state.py:3264` |
| 28 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `rk3_stage_r3f_nemo_literal` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1981` -> `packages/ocean/legoesm/ocean/state.py:3214` |
| 29 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `rk3_stage_r3t_nemo_literal` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1982` -> `packages/ocean/legoesm/ocean/state.py:3219` |
| 30 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `rk3_stage_r3uv_nemo_literal` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1984` -> `packages/ocean/legoesm/ocean/state.py:3229` |
| 31 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `nemo_bottom_drag_written_statements` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2002` -> `packages/ocean/legoesm/ocean/state.py:3301` |
| 32 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `rk3_vector_rhs_nemo_order` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2014` -> `packages/ocean/legoesm/ocean/state.py:3342` |
| 33 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `een_vorticity_written_order` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2015` -> `packages/ocean/legoesm/ocean/state.py:3190` |
| 34 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `nemo_wzv_stored_r3_ratios` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2017` -> `packages/ocean/legoesm/ocean/state.py:3321` |
| 35 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `nemo_zad_written_association` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2018` -> `packages/ocean/legoesm/ocean/state.py:3327` |
| 36 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `nemo_stage2_momentum_wzv_half_step` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2019` -> `packages/ocean/legoesm/ocean/state.py:3333` |
| 37 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `nemo_stage_momentum_wzv_stored_kmm_ratios` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2020` -> `packages/ocean/legoesm/ocean/state.py:3338` |
| 38 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `nemo_ldf_stored_f_area` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2026` -> `packages/ocean/legoesm/ocean/state.py:3348` |
| 39 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `nemo_ldf_stage_r3f` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2027` -> `packages/ocean/legoesm/ocean/state.py:3367` |
| 40 | **B** (rounding/order) | `LatLonCGridOceanConfig` | `rk3_stage1_ldf_source_order` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2028` -> `packages/ocean/legoesm/ocean/state.py:3361` |
| 41 | **B** (rounding/order) | `BarotropicConfig` | `barotropic_substep_dt_nemo_literal` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1957` -> `packages/ocean/legoesm/ocean/state.py:1646` |
| 42 | **B** (rounding/order) | `BarotropicConfig` | `barotropic_een_ratio_written_order` | `True` | `False` | Yes | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2053` -> `packages/ocean/legoesm/ocean/state.py:1667` |
| 43 | **C** (unconditional) | `source_rounding` | `nemo_source_round` / `nemo_source_barrier` | `optimization_barrier + copysign` | `optimization_barrier + copysign` | No (no config flag) | `packages/core/legoesm/core/source_rounding.py:14-35` |
| 44 | **D** (structural) | `LatLonCGridOceanConfig` | `zdf_implicit_solver_evaluation` (+ companion zdf flags) | `"nemo_literal"` | `"shared_thomas"` | No (WS-RK3 + partial-cell required) | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:544` -> `packages/ocean/legoesm/ocean/state.py:3163` |
| 45 | **D** (structural) | `LatLonCGridOceanConfig` | `orca2_l8_krhs_selection` (+ stage-3 tracer/EIV/Treguier bundle) | `True` | `False` | No (WS-RK3 constructor required) | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2954-2975` -> `packages/ocean/legoesm/ocean/state.py:3423` |
| 46 | **D** (structural) | `GMRediConfig` | `kappa_redi_horizontal_evaluation` (+ tripolar/WS-RK3 companions) | `"nemo_metric_literal"` | `"cosine_scaled"` | No (tripolar constructor required) | `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2909` -> `packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:473` |

---

## 2. Speed Benchmark Results (Standalone ORCA2 K0 Card)

All four arms run `packages/ocean/legoesm/ocean/bulk_flux_omip.py:983-1118` (`nemo_orca2_k0_ocean_step`) on the identical `(148, 180, 30)` ORCA2 K0 standalone card and inputs.
- **Why warm-up takes 2 steps (`kt=1..2`)**: Step `kt=1` compiles `LatLonCGridOceanModel._step_jitted` (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:15444-15453`) with `state.eta_rk3_after=None` and `state.u_before=None`; step `kt=2` compiles the steady-state pytree signature where `state.eta_rk3_after` and `state.{u,v,T,S,eta}_before` are populated `Field` leaves (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:15610-15631`). From `kt=3` onward, zero recompilations occur.
- **Why odd and even `kt` steps have distinct times on GPU**: Inside `nemo_orca2_k0_ocean_step`, the 3-D ocean step `model.step` is `@jax.jit`-compiled (`~0.17–0.19 s` on GPU), whereas the outer surface boundary wrapper `nemo_blk_oce_ncar` runs in eager JAX (`packages/ocean/legoesm/ocean/bulk_flux_omip.py:1219-1459`). With `nn_fsbc = 2`, odd steps (`kt = 1, 3, 5, ...`) execute the 5-iteration NCAR turbulent transfer loop, `blk_oce_2`, `nemo_sbc_ice_if`, and `nemo_orca2_s2` as eager 2-D kernel dispatches (`~0.73–0.76 s` total step time on GPU), whereas even steps (`kt = 2, 4, 6, ...`) take the early-return retention branch (`packages/ocean/legoesm/ocean/bulk_flux_omip.py:1280-1352`, `~0.17–0.19 s` total step time on GPU).
- **`float32` support**: Supported out of the box with zero model changes by setting `JAX_ENABLE_X64=0` and `PrecisionPolicy.fp32(transcendentals="native")`.

### Speed & Memory Table

| Metric | (a) `exact_cpu_fp64` | (b) `normal_cpu_fp64` | (c) `normal_gpu_fp64` | (d) `normal_gpu_fp32` |
|---|---|---|---|---|
| **Backend / Device** | CPU | CPU | 1 × GPU (`Tesla V100S-PCIE-32GB`) | 1 × GPU (`Tesla V100S-PCIE-32GB`) |
| **Precision / Transcendentals** | `float64` / `libm` | `float64` / `native` | `float64` / `native` | `float32` (`JAX_ENABLE_X64=0`) / `native` |
| **Total Steps Run (Warm-up + Steady)** | `22` (`2 + 20`) | `64` (`2 + 62`) | `64` (`2 + 62`) | `64` (`2 + 62`) |
| **Compile + Warm-up `t(kt=1) + t(kt=2)`** | **`239.44 s`** (`118.84s + 120.60s`) | **`175.42 s`** (`89.55s + 85.87s`) | **`263.73 s`** (`136.29s + 127.44s`) | **`264.92 s`** (`136.98s + 127.95s`) |
| **Net Compile Overhead Est.** | `196.32 s` | `172.79 s` | `262.80 s` | `264.03 s` |
| **Steady-State Median (`s/step`)** | **`21.5587 s/step`** | **`1.3185 s/step`** | **`0.4680 s/step`** | **`0.4473 s/step`** |
| **Steady-State Mean (`s/step`)** | `21.0865 s/step` | `1.3281 s/step` | `0.4796 s/step` | `0.4643 s/step` |
| **Odd-`kt` Median (`nn_fsbc=2` update)** | `22.5248 s/step` | `1.3837 s/step` | `0.7576 s/step` | `0.7327 s/step` |
| **Even-`kt` Median (`nn_fsbc=2` hold)** | `19.5232 s/step` | `1.2021 s/step` | `0.1898 s/step` | `0.1683 s/step` |
| **Steady-State IQR / `[min, max]` (`s`)** | `2.9780 s` / `[17.86, 24.13]` | `0.2054 s` / `[0.90, 1.96]` | `0.5675 s` / `[0.18, 0.89]` | `0.5639 s` / `[0.16, 1.20]` |
| **Speedup vs Exact CPU (Median)** | `1.00×` | **`16.35×`** | **`46.07×`** | **`48.20×`** |
| **Host Peak RSS Memory** | `11,290.5 MiB` (`11.03 GiB`) | `9,847.1 MiB` (`9.62 GiB`) | `6,739.1 MiB` (`6.58 GiB`) | `5,681.8 MiB` (`5.55 GiB`) |
| **GPU Peak Memory In-Use** | N/A | N/A | `2,534.7 MiB` (`2.48 GiB`) | `2,006.3 MiB` (`1.96 GiB`) |
| **Projected 1-Year (`2,920` steps, Steady)** | **`1,049.19 min` (`17.49 hr`)** | **`64.16 min` (`1.07 hr`)** | **`22.77 min` (`0.38 hr`)** | **`21.77 min` (`0.36 hr`)** |
| **Projected 1-Year (`2,920` steps, + Compile)** | **`1,052.46 min` (`17.54 hr`)** | **`67.04 min` (`1.12 hr`)** | **`27.15 min` (`0.45 hr`)** | **`26.17 min` (`0.44 hr`)** |

---

## 3. Accuracy Cost Over 8 Days (`64` Steps) vs Exact-Mode Reference (`orca2_r342`)

Each arm is compared at the end of every simulated day (`kt = 8, 16, 24, 32, 40, 48, 56, 64`) against the saved exact-mode snapshots in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step/candidate_step{kt:08d}.npz`.

- **Exact-mode control (arm `exact_cpu_fp64`)**: At Day 1 (`kt=8`, SHA-256 `ce932543...`) and Day 2 (`kt=16`, SHA-256 `cbee4521...`), `exact_cpu_fp64` reproduces the `orca2_r342` snapshots with **`0.0` SST RMS**, **`0.0` SST max**, **`0.0` 3-D `T` RMS**, and **`0.0` 3-D `T` max**.
- **Normal `float64` CPU vs GPU consistency**: Arms **(b) `normal_cpu_fp64`** and **(c) `normal_gpu_fp64`** produce identical daily RMS and max differences to `~10⁻¹² °C` across all 8 days. Their `~1.4–5.3 × 10⁻³ °C` wet SST RMS difference relative to exact mode is dominated by the two Category A vertical-mixing parameterizations whose `"nemo_literal"` forms are hard-guarded by `transcendentals == "libm"` (`DoubleDiffusionConfig` at `packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py:325-329` and `IWMConfig` at `packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py:411-415`) and therefore revert to their normal class defaults (`ddm.enabled=False`, `iwm.form="generic", tsdiff=False`).

### Daily SST (`T[..., 0]`) and 3-D `T` Difference Table (`°C`)

| Day (`kt`) | Reference `candidate_step*.npz` SHA-256 (prefix) | Arm | SST RMS (All Cells) [`°C`] | SST RMS (Wet Cells) [`°C`] | SST Max Abs [`°C`] | 3-D `T` RMS (All Cells) [`°C`] | 3-D `T` RMS (Wet Cells) [`°C`] | 3-D `T` Max Abs [`°C`] |
|---|---|---|---|---|---|---|---|---|
| **Day 1** (`kt=8`) | `ce932543cdad` | **(a)** `exact_cpu_fp64` | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` |
| | | **(b)** `normal_cpu_fp64` | `1.373095e-03` | `1.748272e-03` | `8.255179e-02` | `1.287237e-03` | `1.753773e-03` | `2.450638e-01` |
| | | **(c)** `normal_gpu_fp64` | `1.373095e-03` | `1.748272e-03` | `8.255179e-02` | `1.287237e-03` | `1.753773e-03` | `2.450638e-01` |
| | | **(d)** `normal_gpu_fp32` | `7.121244e-03` | `9.067019e-03` | `3.116683e-01` | `5.329369e-03` | `7.260905e-03` | `6.751881e-01` |
| **Day 2** (`kt=16`) | `cbee45219d68` | **(a)** `exact_cpu_fp64` | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` | `0.000000e+00` |
| | | **(b)** `normal_cpu_fp64` | `2.215070e-03` | `2.820306e-03` | `2.027637e-01` | `2.418589e-03` | `3.295163e-03` | `5.977165e-01` |
| | | **(c)** `normal_gpu_fp64` | `2.215070e-03` | `2.820306e-03` | `2.027637e-01` | `2.418589e-03` | `3.295163e-03` | `5.977165e-01` |
| | | **(d)** `normal_gpu_fp32` | `1.384412e-02` | `1.762683e-02` | `4.295891e-01` | `9.852466e-03` | `1.342332e-02` | `7.429568e-01` |
| **Day 3** (`kt=24`) | `a501cebb4046` | **(b)** `normal_cpu_fp64` | `3.448027e-03` | `4.390149e-03` | `2.998329e-01` | `3.480597e-03` | `4.742078e-03` | `6.653729e-01` |
| | | **(c)** `normal_gpu_fp64` | `3.448027e-03` | `4.390149e-03` | `2.998329e-01` | `3.480597e-03` | `4.742078e-03` | `6.653729e-01` |
| | | **(d)** `normal_gpu_fp32` | `2.109090e-02` | `2.685367e-02` | `6.862853e-01` | `1.460851e-02` | `1.990310e-02` | `1.069800e+00` |
| **Day 4** (`kt=32`) | `667c871cfb4d` | **(b)** `normal_cpu_fp64` | `3.406203e-03` | `4.336898e-03` | `1.894681e-01` | `3.882112e-03` | `5.289115e-03` | `5.840148e-01` |
| | | **(c)** `normal_gpu_fp64` | `3.406203e-03` | `4.336898e-03` | `1.894681e-01` | `3.882112e-03` | `5.289115e-03` | `5.840148e-01` |
| | | **(d)** `normal_gpu_fp32` | `2.692585e-02` | `3.428295e-02` | `7.960351e-01` | `1.843919e-02` | `2.512215e-02` | `8.702997e-01` |
| **Day 5** (`kt=40`) | `6af045e1f164` | **(b)** `normal_cpu_fp64` | `3.506055e-03` | `4.464033e-03` | `1.508169e-01` | `4.049011e-03` | `5.516503e-03` | `6.699001e-01` |
| | | **(c)** `normal_gpu_fp64` | `3.506055e-03` | `4.464033e-03` | `1.508169e-01` | `4.049011e-03` | `5.516503e-03` | `6.699001e-01` |
| | | **(d)** `normal_gpu_fp32` | `3.221834e-02` | `4.102153e-02` | `8.186378e-01` | `2.247023e-02` | `3.061417e-02` | `1.137342e+00` |
| **Day 6** (`kt=48`) | `1bfd037dec42` | **(b)** `normal_cpu_fp64` | `4.177718e-03` | `5.319218e-03` | `3.493624e-01` | `4.251643e-03` | `5.792576e-03` | `7.019607e-01` |
| | | **(c)** `normal_gpu_fp64` | `4.177718e-03` | `5.319218e-03` | `3.493624e-01` | `4.251643e-03` | `5.792576e-03` | `7.019607e-01` |
| | | **(d)** `normal_gpu_fp32` | `3.873546e-02` | `4.931936e-02` | `1.014475e+00` | `2.730125e-02` | `3.719610e-02` | `1.614134e+00` |
| **Day 7** (`kt=56`) | `bd317d47b249` | **(b)** `normal_cpu_fp64` | `3.797956e-03` | `4.835692e-03` | `1.743547e-01` | `4.587784e-03` | `6.250545e-03` | `7.039534e-01` |
| | | **(c)** `normal_gpu_fp64` | `3.797956e-03` | `4.835692e-03` | `1.743547e-01` | `4.587784e-03` | `6.250545e-03` | `7.039534e-01` |
| | | **(d)** `normal_gpu_fp32` | `4.370857e-02` | `5.565129e-02` | `1.006025e+00` | `3.156460e-02` | `4.300463e-02` | `1.295309e+00` |
| **Day 8** (`kt=64`) | `85169ea892ea` | **(b)** `normal_cpu_fp64` | `4.156920e-03` | `5.292738e-03` | `1.603794e-01` | `5.057543e-03` | `6.890559e-03` | `8.072074e-01` |
| | | **(c)** `normal_gpu_fp64` | `4.156920e-03` | `5.292738e-03` | `1.603794e-01` | `5.057543e-03` | `6.890559e-03` | `8.072074e-01` |
| | | **(d)** `normal_gpu_fp32` | `4.882567e-02` | `6.216657e-02` | `9.703472e-01` | `3.590351e-02` | `4.891611e-02` | `1.526749e+00` |
