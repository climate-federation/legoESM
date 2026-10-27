# SI3 Ladder Round 2 (`si3_r2`) Receipt — Rung D5 (C1D Subnormal-Snow Platform Classification & ZDF Layer-Reciprocal Closure)

## 1. Scope, Preregistration, and Provenance

This round executes rung **D5** of the SI3 sea-ice fidelity ladder (`docs/ice/fidelity/si3_ladder_design.md:220`, `docs/ice/fidelity/si3_ladder_design.md:344-350`) under **DECISION 156**: classify and close lane 3's C1D subnormal-snow defect per platform before starting the ORCA2 rungs, reusing the immutable 8,760-step C1D North Greenland ERA5 SAS record (`c1d_omip_l3_sasice_scalarmath_v2_a`) from `docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase6_receipt.md:194-209` and `docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md:30-43`.

* **Preregistration commit (before measurement):** `59fd8f2395ccba60b5b04f3c2c68946835feea12` (`docs/ice/fidelity/si3_r2_d5_preregister.md:1-74`)
* **Implementation & probe commit:** `3a99d0bfabbdfa2782cc31cc4c7b693d81996cad` (updated with compiled `jax.jit` full-year evaluation in `8ea398af75cdb53adad8f1febeb79d89ab664e97`)
* **Oracle record root:** `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a`
  * `oracle_si3_thd_frames.bin` SHA-256: `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b`
  * `oracle_si3_zdf_inputs.bin` SHA-256: `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd`
* **Committed probe:** `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:144-2072`
* **Committed unit & gradient tests:** `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:61-2776`
* **Evidence artifacts under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round2/` (eager `si3_column_step_arrays`; inner `fori_loop` compiled):**
  * `d5_platform_probe_cpu.json` (CPU default, `XLA_FLAGS=""`) SHA-256: `8e9de0b3de4648be43eb40275b6520cf40156ddac308940b736d39f00af9dbcf`
  * `d5_platform_probe_cpu_avx_nofma.json` (CPU FMA disabled, `XLA_FLAGS="--xla_cpu_max_isa=AVX"`) SHA-256: `c8be84fa73ea213136967fba05e5f8821bb2d00c1c1ecdc23e26b14c4bf3aaf9`
  * `d5_platform_probe_gpu.json` (GPU 1 pinned via `CUDA_VISIBLE_DEVICES=1`) SHA-256: `3f2c622e5c0751337d306f096d30a4b6b6a44148eb57d9fdd4e5f52574ceb388`
  * `gpu_nvidia_smi_before.txt` SHA-256: `c72f77ea02607b1183b6245635a2244381554b83e426f4f356b89329feefb757`
  * `gpu_nvidia_smi_after.txt` SHA-256: `af76586d8e9d0257eaa8b4e259080df666f0ad50054fe427e0f784245dd0f472`
* **Compiled `jax.jit` evidence artifacts under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round3/` (`r2_pre_reciprocals` and `r2_baseline` arms):**
  * `cpu_default_probe.json` SHA-256: `3b500ddf849202a66143bc0cbfa6c4524830a4f00847793415fe7fb4acd36c69`
  * `cpu_nofma_probe.json` SHA-256: `6a83b03454b6907995084dad93c574a89f757d6d9ea8c5c3ba32436b4ad9d9ca`
  * `gpu_probe.json` SHA-256: `4cd0025943a9f87301551c875f2451557f8c327fb18c84f41ae93f3a12838feb`

### Idle-GPU Provenance (`CUDA_VISIBLE_DEVICES=1`)
Before launching the GPU probe, `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` and `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` were recorded to `gpu_nvidia_smi_before.txt`, showing GPU 0 occupied by another job (`pid = 3784548`, `24,706 MiB`, GPU 0 at `56 %` utilization) and GPU 1 completely idle (`0 MiB`, `0 %`, zero compute apps on GPU 1). The probe was pinned exclusively to GPU 1 via `CUDA_VISIBLE_DEVICES=1`. Immediately after completion, the same queries were recorded to `gpu_nvidia_smi_after.txt`, confirming GPU 1 returned to `0 MiB` (`5 %` transient cooldown) with zero compute apps on GPU 1.

---

## 2. Stage A & Stage B: Platform Subnormal Probe and Corrected FTZ/DAZ Classification

`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:144-252` executes four compiled `@jax.jit` binary64 probes under `PrecisionPolicy.fp64()` (`JAX_ENABLE_X64=1`) on the smallest positive binary64 subnormal `x_sub = 5e-324` (`0x0000000000000001`) and normal factors `a = 2^-1022` (`0x0010000000000000`), `b = 2^-52` (`0x3cb0000000000000`), followed by the compiled ZDF snow-present predicate `h_s > 0.0` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:131-141`) across all 26 C1D steps (`kt = 5842..5867`) where NEMO's ZDF-entry snow thickness is subnormal (`0 < h_s < 2.2250738585072014e-308`).

### Separation of Output Flushing (FTZ) vs Input Flushing (DAZ)
In `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:172-252` and `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:1985-2008`:
* `x_sub * 1.0` and `x_sub + 0.0` have a subnormal mathematical result (`5e-324`), so they flush to `0.0` under output flush-to-zero (`FTZ`) alone even if input denormals-are-zero (`DAZ`) is disabled (`ftz_or_daz_flushes_mul_add`). They therefore cannot classify `DAZ`.
* `ftz_flushes_subnormal_output` is classified from `_jit_produce_subnormal(2^-1022, 2^-52)`, whose inputs are normal binary64 numbers (`0x0010000000000000` and `0x3cb0000000000000`) and whose product is subnormal (`0x0000000000000001`).
* `daz_flushes_subnormal_input` is classified **exclusively** from Stage B's `_jit_snow_pred(h_s)` (`h_s > 0.0`) on positive subnormal inputs (`inputs_are_positive_subnormal = True`), because the comparison output is a normal boolean (`True`/`False`, or `1.0`/`0.0` in `_jit_isnow`) that output `FTZ` alone cannot flush.
* `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1026-1074` verifies that `classify_ftz_daz` distinguishes `FTZ_ONLY_FLUSH_TO_ZERO` (`daz_flushes_subnormal_input = False`) from `FTZ_DAZ_FLUSH_TO_ZERO` (`daz_flushes_subnormal_input = True`) and includes a planted control (`plant_invalid_daz_from_mul_add=True`) proving the former `x * 1.0` / `x + 0.0` rule fails on an `FTZ`-only state.

| Probe metric | CPU (`x86_64`, `cpu:0`, default & `--xla_cpu_max_isa=AVX`) | GPU (`Tesla V100S-PCIE-32GB`, `cuda:0` via `CUDA_VISIBLE_DEVICES=1`) |
|---|---|---|
| Input `x_sub` (`float64` / bits) | `5e-324` / `0x0000000000000001` | `5e-324` / `0x0000000000000001` |
| `_jit_identity(x_sub)` | `5e-324` / `0x0000000000000001` | `5e-324` / `0x0000000000000001` |
| `_jit_mul_one(x_sub, 1.0)` | `0.0` / `0x0000000000000000` | `5e-324` / `0x0000000000000001` |
| `_jit_add_zero(x_sub, 0.0)` | `0.0` / `0x0000000000000000` | `5e-324` / `0x0000000000000001` |
| `_jit_produce_subnormal(2^-1022, 2^-52)` (FTZ test) | `0.0` / `0x0000000000000000` | `5e-324` / `0x0000000000000001` |
| `ftz_flushes_subnormal_output` (from `a * b`) | `True` | `False` |
| Recorded `kt = 5842` entry `h_s` | `5.562684646268003e-309` (`0x0004000000000000`) | `5.562684646268003e-309` (`0x0004000000000000`) |
| Recorded `kt = 5843..5867` entry `h_s` | `5e-324` (`0x0000000000000001`) | `5e-324` (`0x0000000000000001`) |
| `_jit_snow_pred(h_s)` true count (`kt = 5842..5867`, DAZ test) | `0 / 26` (`predicate_all_false = True`) | `26 / 26` (`predicate_all_true = True`) |
| `_jit_isnow(h_s)` (`kt = 5842..5867`) | `0.0` (`0x0000000000000000`) | `1.0` (`0x3ff0000000000000`) |
| `daz_flushes_subnormal_input` (from `h_s > 0.0`) | `True` | `False` |
| `ftz_daz_mode` stamp | `FTZ_DAZ_FLUSH_TO_ZERO` | `IEEE754_GRADUAL_UNDERFLOW` |
| **Platform disposition** | **`D5_PLATFORM_FLOOR`** | **`SUBNORMAL_PRESERVED`** |

### CPU Disposition (`D5_PLATFORM_FLOOR`)
On `x86_64` CPU (`JAX_PLATFORMS=cpu`, both with default `XLA_FLAGS=""` and with `XLA_FLAGS="--xla_cpu_max_isa=AVX"`), `_jit_identity` is a buffer pass-through that executes no floating-point instruction, whereas compiled floating-point arithmetic (`_jit_produce_subnormal`) flushes subnormal outputs (`FTZ = True`) and the compiled comparison `_jit_snow_pred` (`vucomisd`) flushes positive subnormal inputs to zero (`DAZ = True`). Consequently, across all 26 subnormal-snow steps `kt = 5842..5867`, `h_s > 0.0` evaluates to `False` (`isnow = 0.0`) on CPU and enters the snow-free 4-equation BL99 tridiagonal branch instead of NEMO's snow-present 7-equation branch. Per **DECISION 156** and `docs/ice/fidelity/si3_ladder_design.md:344-350`, CPU is declared **`D5_PLATFORM_FLOOR`**, no subnormal workaround is introduced, and the 26 subnormal steps (`75` `POST_ZDF` over-bar rows, `629` all-stage over-bar rows in `kt = 5842..5867`, led by `kt = 5860 POST_ZDF.e_s` at normalised error `0.02510099530281747` and `190,081,022,848,368` ULPs) remain a stated platform floor on CPU.

---

## 3. Stage C: BL99 Layer-Reciprocal & Frozen-Conductivity Transcription on the Subnormal-Preserving Platform (GPU)

On `Tesla V100S-PCIE-32GB` (`cuda:0`), IEEE-754 gradual underflow is preserved (`SUBNORMAL_PRESERVED`) and `h_s > 0.0` evaluates to `True` (`isnow = 1.0`) for all 26 steps `kt = 5842..5867`. However, before the statement fix (`_nemo_zdf_layer_reciprocals=False`), `_si3_zdf_bl99_step` still disagreed with NEMO across `kt = 5841..5867` (`58` over-bar `POST_ZDF` rows and `81` nonzero-ULP `POST_ZDF` rows in `kt = 5842..5867`) and across snow-free melting steps (`kt = 4242..5778`, where `POST_DH..EXIT` accumulated `18,280` over-bar rows from `qcn_ice_top_1d`).

### Statement Departures in `icethd_zdf_bl99` (NEMO vs legoESM)

In NEMO's BL99 vertical heat diffusion (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:170-186` and compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:172-188`):
1. **Layer thicknesses and precomputed reciprocals:** NEMO computes `zh_i(ji) = MAX( zh_min , h_i_1d(ji) ) * r1_nlay_i` (`r1_nlay_i = 1.0_wp / 3.0_wp`), `z1_h_i(ji) = 1._wp / zh_i(ji)`, `zh_s(ji) = MAX( zh_min , h_s_1d(ji) ) * r1_nlay_s`, and `z1_h_s(ji) = 1._wp / zh_s(ji)`.
   * Solar radiation transmission uses `zh_s(ji)` and `zh_i(ji)` (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:217-230`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:219-232`).
   * Thermal conductances multiply by `z1_h_s(ji)` and `z1_h_i(ji)` and form the snow-ice harmonic interface conductance with `0.5_wp * ( ztcond_i(ji,0) * zh_s(ji) + rcnd_s * zh_i(ji) )` (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:326-348`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:321-343`).
   * Tridiagonal prefactors multiply by `z1_h_i(ji)` and `z1_h_s(ji)`: `zeta_i(ji,jk) = rDt_ice * r1_rhoi * z1_h_i(ji) / zcpi` and `zeta_s(ji,jk) = rDt_ice * r1_rhos * r1_rcpi * z1_h_s(ji)` (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:366-376`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:353-364`).
   * Under `NEMO_SI3_CONSTANTS_CONFIG` (`rho_snow = 330.0`, `c_ice = 2096.7`, `dt = 3600.0`) on clamped snow (`h_s <= 1.0e-3 m`), NEMO's `zh_s = 0x3f35d867c3ece2a5`, `z1_h_s = 3000.0` (`0x40a7700000000000`), and `zeta_s = 0x402f37c77f84d1ee` (`15.608943924998249`), whereas legacy `dt / (rho_snow * dz_s * c_ice) = 0x402f37c77f84d1ef` (`15.608943924998251`, 1 ULP larger). Multiplied by `zkappa_s = 1500.0` (`eta_s * k_s ~ 2.34e4`) in the stiff clamped-snow tridiagonal rows, that 1-ULP difference in `eta_s` amplifies into `470..718` ULPs in `POST_ZDF.e_s` (`norm_err` up to `9.55e-14 > 1e-15`).
2. **Post-solve conductive fluxes (`qcn_ice_bot_1d`, `qcn_ice_top_1d`) reuse iteration-frozen `zkappa`:**
   * While `qcn_ice_top` and `qcn_ice_bottom` are not stored directly in the `POST_ZDF` frame of `oracle_si3_thd_frames.bin`, they are **scored downstream** across `POST_DH..EXIT` (`oracle_si3_thd_frames.bin`) and in `zq_top` of `oracle_si3_dh_operands.bin`, because `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1927`) passes `zdf.qcn_ice_top` and `zdf.qcn_ice_bottom` directly into `_dh_step` (`qml = qns_ice + qsr_ice - qtr_ice_top - qcn_ice_top` for surface melt when `T_surface >= T0` on `1,160` snow-free melting steps `kt = 4242..5778`, and `zf` for basal growth/melt on all `8,760` steps).
   * In NEMO (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:326-348` and `scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:821-823`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:321-343` and `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:782-784`), `zkappa_s(ji,0)`, `zkappa_i(ji,0)`, and `zkappa_i(ji,nlay_i)` are computed inside `IF ( .NOT. l_T_converged(ji) )` from the iterate's entry temperatures (`T_i^{(m)}`, `T_s^{(m)}`), frozen once the Picard loop converges, and reused after the loop in:
     * `qcn_ice_bot_1d(ji) = - zkappa_i(ji,nlay_i) * zg1 * ( t_bo_1d(ji ) - t_i_1d (ji,nlay_i) )`
     * `qcn_ice_top_1d(ji) = - isnow(ji) * zkappa_s(ji,0) * zg1s * ( t_s_1d(ji,1) - t_su_1d(ji) ) - ( 1._wp - isnow(ji) ) * zkappa_i(ji,0) * zg1 * ( t_i_1d(ji,1) - t_su_1d(ji) )`
   * Recomputing `p07_conductivity` from post-solve `Ti[..., 0]` instead of reusing iteration-frozen `k_i_frozen[..., 0]` caused a `634,976,763`-ULP error in `zq_top` at `kt = 4242` (`249909.11100362349` vs oracle `249909.09252336744` = `0x410e81a8bd7ce42a`) and a `140,889`-ULP error (`norm_err = 2.8322650931157343e-11 > 1e-15`) in `kt = 4242 POST_DH.h_i`; transcribing the frozen `zkappa` reuse literally drops both to **`0` ULPs (`0.0`)** and closes an additional **`18,280`** over-bar rows across `POST_DH..EXIT` on GPU (`22,775 -> 4,495` eager; `35,091 -> 4,961` under `jax.jit`).

### Selectable SI3 Statement Fix
We added the selectable SI3 statement flag `_nemo_zdf_layer_reciprocals: bool = True` in `packages/ice/legoesm/ice/bitz_lipscomb.py`:
* `_radiation` (`packages/ice/legoesm/ice/bitz_lipscomb.py:475-528`): computes `dzs` and `dzi` via `jnp.maximum(_H_MIN, h_s) * r1_nlay` and `jnp.maximum(_H_MIN, h_i) * r1_nlay` gated by `h_s > 0.0` and `h_i > 0.0`.
* `_si3_zdf_bl99_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:545-566`, `packages/ice/legoesm/ice/bitz_lipscomb.py:592-661`, `packages/ice/legoesm/ice/bitz_lipscomb.py:759-804`): computes `zh_i`, `z1_h_i`, `zh_s`, `z1_h_s`, freezes `k_i_frozen` and `k_s_frozen` on active Picard iterates, and uses NEMO's written multiplication order for `k_i`, `k_s`, `k_si`, `eta_i`, `eta_s`, `qcn_bottom`, and `qcn_top`.
* `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1927`): forwards `_nemo_zdf_layer_reciprocals`.
* No configuration classes (`SeaIceConfig`, `SI3ThermoConfig`) were modified, and `bitz_lipscomb_conduction_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:336-366`) and all non-SI3 callers are completely untouched.

### Before/After Comparison on Sampled Clamped-Snow Steps (`POST_ZDF`)

| Step & field | Oracle bits (`float64`) | Before fix (`_nemo_zdf_layer_reciprocals=False`) | After fix (`_nemo_zdf_layer_reciprocals=True`) |
|---|---|---|---|
| `kt = 5841` `POST_ZDF.e_s[0]` (`h_s = 5.01e-293 m`, normal) | `0x419aa3e83425a050` (`111737357.03674436`) | CPU: `0x419aa3e83425a31c` (`716` ULP, `9.548e-14`)<br>GPU: `0x419aa3e83425a316` (`710` ULP, `9.468e-14`) | CPU: `0x419aa3e83425a050` (**`0` ULP, `0.0`**)<br>GPU: `0x419aa3e83425a050` (**`0` ULP, `0.0`**) |
| `kt = 5841` `POST_ZDF.t_su` | `0x4070ea37438e23b1` (`270.6384921600648`) | CPU: `271` ULP (`5.692e-14`)<br>GPU: `269` ULP (`5.650e-14`) | CPU: **`0` ULP (`0.0`)**<br>GPU: **`0` ULP (`0.0`)** |
| `kt = 5841` `POST_ZDF.e_i[0]` | `0x41adc715e6db4a95` (`249793267.4283034`) | CPU: `553` ULP (`6.179e-14`)<br>GPU: `553` ULP (`6.179e-14`) | CPU: **`0` ULP (`0.0`)**<br>GPU: **`0` ULP (`0.0`)** |
| `kt = 5842` `POST_ZDF.e_s[0]` (`h_s = 5.56e-309 m`, subnormal) | `0x419a91b93c5fb981` (`111439439.09348108`) | CPU: `0x419a3a6803ffffff` (`95749191638402` ULP, `0.01278`, `D5_PLATFORM_FLOOR`)<br>GPU: `0x419a91b93c5fbb57` (`470` ULP, `6.285e-14`) | CPU: unchanged (`D5_PLATFORM_FLOOR`)<br>GPU: `0x419a91b93c5fb981` (**`0` ULP, `0.0`**) |
| `kt = 5860` `POST_ZDF.e_s[0]` (`h_s = 5e-324 m`, subnormal) | `0x419ae748b3d86d6f` (`112841260.96135496`) | CPU: `0x419a3a6803ffffff` (`190081022848368` ULP, `0.02510099530281747`, `D5_PLATFORM_FLOOR`)<br>GPU: `0x419ae748b3d86fc4` (`597` ULP, `7.884e-14`) | CPU: unchanged (`D5_PLATFORM_FLOOR`)<br>GPU: `0x419ae748b3d86d6f` (**`0` ULP, `0.0`**) |
| `kt = 5860` `POST_ZDF.t_su` | `0x4070d0881092576e` (`269.0332189289537`) | CPU: `1250293771752` ULP (`2.642e-4`, `D5_PLATFORM_FLOOR`)<br>GPU: `225` ULP (`4.754e-14`) | CPU: unchanged (`D5_PLATFORM_FLOOR`)<br>GPU: `0x4070d0881092576e` (**`0` ULP, `0.0`**) |
| `kt = 5860` `POST_ZDF.e_i[0]` | `0x41af05f35b2c5d49` (`260241837.5866492`) | CPU: `765562252607` ULP (`8.538e-5`, `D5_PLATFORM_FLOOR`)<br>GPU: `0` ULP (`0.0`) | CPU: unchanged (`D5_PLATFORM_FLOOR`)<br>GPU: `0x41af05f35b2c5d49` (**`0` ULP, `0.0`**) |
| `kt = 5867` `POST_ZDF.e_s[0]` (`h_s = 5e-324 m`, subnormal) | `0x419afe4be6272c4d` (`113218297.59641379`) | CPU: `214286246759502` ULP (`0.02820`, `D5_PLATFORM_FLOOR`)<br>GPU: `718` ULP (`9.442e-14`) | CPU: unchanged (`D5_PLATFORM_FLOOR`)<br>GPU: `0x419afe4be6272c4d` (**`0` ULP, `0.0`**) |

Across all 26 subnormal ZDF-entry steps (`kt = 5842..5867`) on GPU, `POST_ZDF` over-bar rows drop from **`57` to `1`** under Round 2's entry bridge (the single remaining row is `kt = 5857 POST_ZDF.e_i` at `65` ULPs, `norm_err = 7.251983965966999e-15`, caused by a 2-ULP entry-bridge difference in `ENTRY.e_i[0]` from `(e_i * 3.0) / v_i` vs `(e_i / v_i) * 3.0`; when `_si3_zdf_bl99_step` is called with the exact `oracle_si3_zdf_inputs.bin` operand at `kt = 5857`, all `POST_ZDF` outputs are `0` ULPs), and all-stage over-bar rows in `kt = 5842..5867` drop from **`280` to `17`** (`eager`) and **`273` to `7`** (`jax.jit`; and to **`0`** in Round 3 once `_nemo_salprof` is transcribed in full). In the boundary gate (`tests/ocean/fidelity/test_nemo_si3thd_phase2_gate.py:208-219`), closing `kt = 5` and `kt = 6` `POST_ZDF` advances the first over-bar step from `kt = 5` (`kt5.POST_ZDF.e_s`) to `kt = 7` (`kt7.POST_DH.e_i`, `sweep_steps_examined = 7`).

---

## 4. Stage D: Full-Year (8,760-Step) C1D Rescore (`jax.jit` vs Eager), CPU FMA Analysis, and Next D5 Upstream Input Items

In Round 2's initial probe commit (`3a99d0bfabbdfa2782cc31cc4c7b693d81996cad`, `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round2/*.json`), `run_year_rescore` called `si3_column_step_arrays` **eagerly** (without outer `jax.jit`, so only inner `jax.lax.fori_loop` bodies compiled), whereas the full-year gate (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1836`) compiles `si3_column_step_arrays` under `jax.jit`. In `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883` (`8ea398af75cdb53adad8f1febeb79d89ab664e97`, `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round3/{cpu_default_probe.json,cpu_nofma_probe.json,gpu_probe.json}`), every arm — including Round 2's before (`r2_pre_reciprocals`, `_nemo_zdf_layer_reciprocals=False`) and after (`r2_baseline`, `_nemo_zdf_layer_reciprocals=True`) arms — is evaluated under **`jax.jit`**. Both the **compiled `jax.jit`** numbers and the **earlier eager** numbers are reported below, explicitly labeled.

### 4.1 Compiled `jax.jit` Full-Year Rescore for Round 2 (`r2_pre_reciprocals -> r2_baseline`, Unrounded `_nemo_source_round=False` & Source-Rounded `step4_source_round_only`)

| Stage (`rows`) | GPU (`CUDA_VISIBLE_DEVICES=1`, `jax.jit`) Before -> After (Delta; Changed / Closed / Opened / To `0` ULP / From `0` ULP) | CPU AVX Diagnostic (`--xla_cpu_max_isa=AVX`, `jax.jit`) Before -> After (Delta; Changed / Closed / Opened / To `0` ULP / From `0` ULP) | CPU Production JIT (`XLA_FLAGS=""`, `jax.jit`) Before -> After (Delta; Changed / Closed / Opened / To `0` ULP / From `0` ULP) | CPU Production JIT + `nemo_source_round=True` (`step4_source_round_only`, `jax.jit`) Over-bar (`0` ULP) |
|---|---:|---:|---:|---:|
| `ENTRY` (`105,120`) | `0 -> 0` (`0`; `0`/`0`/`0`/`0`/`0`)<br>`0`-ULP: `84,373 -> 84,373` | `0 -> 0` (`0`; `0`/`0`/`0`/`0`/`0`)<br>`0`-ULP: `84,373 -> 84,373` | `0 -> 0` (`0`; `0`/`0`/`0`/`0`/`0`)<br>`0`-ULP: `84,373 -> 84,373` | `0` (`84,373`) |
| `POST_ZDF` (`61,320`) | `2,039 -> 497` (**`-1,542`**; `9,502`/`1,567`/`25`/`4,232`/`795`)<br>`0`-ULP: `45,551 -> 48,988` | `4,487 -> 3,228` (**`-1,259`**; `9,664`/`1,320`/`61`/`3,390`/`899`)<br>`0`-ULP: `43,007 -> 45,498` | `4,581 -> 3,380` (**`-1,201`**; `9,521`/`1,291`/`90`/`2,030`/`1,040`)<br>`0`-ULP: `40,161 -> 41,151` | **`551` (`49,005`)** |
| `POST_DH` (`61,320`) | `4,614 -> 676` (**`-3,938`**; `11,322`/`4,034`/`96`/`4,538`/`907`)<br>`0`-ULP: `36,618 -> 40,249` | `6,568 -> 3,489` (**`-3,079`**; `11,542`/`3,231`/`152`/`4,108`/`996`)<br>`0`-ULP: `36,305 -> 39,417` | `6,652 -> 3,728` (**`-2,924`**; `11,399`/`3,115`/`191`/`2,457`/`1,032`)<br>`0`-ULP: `35,204 -> 36,629` | **`717` (`40,245`)** |
| `POST_TEMP1` (`61,320`) | `4,614 -> 676` (**`-3,938`**; `11,322`/`4,034`/`96`/`4,538`/`907`)<br>`0`-ULP: `36,618 -> 40,249` | `6,568 -> 3,489` (**`-3,079`**; `11,542`/`3,231`/`152`/`4,108`/`996`)<br>`0`-ULP: `36,305 -> 39,417` | `6,652 -> 3,728` (**`-2,924`**; `11,399`/`3,115`/`191`/`2,457`/`1,032`)<br>`0`-ULP: `35,204 -> 36,629` | **`717` (`40,245`)** |
| `POST_SAL` (`61,320`) | `4,623 -> 680` (**`-3,943`**; `11,375`/`4,039`/`96`/`4,568`/`913`)<br>`0`-ULP: `38,146 -> 41,801` | `6,622 -> 3,540` (**`-3,082`**; `11,593`/`3,235`/`153`/`4,129`/`1,006`)<br>`0`-ULP: `37,302 -> 40,425` | `6,711 -> 3,779` (**`-2,932`**; `11,457`/`3,125`/`193`/`2,481`/`1,041`)<br>`0`-ULP: `36,204 -> 37,644` | **`719` (`41,800`)** |
| `POST_TEMP2` (`61,320`) | `4,623 -> 680` (**`-3,943`**; `11,375`/`4,039`/`96`/`4,568`/`913`)<br>`0`-ULP: `38,146 -> 41,801` | `6,622 -> 3,540` (**`-3,082`**; `11,593`/`3,235`/`153`/`4,129`/`1,006`)<br>`0`-ULP: `37,302 -> 40,425` | `6,711 -> 3,779` (**`-2,932`**; `11,457`/`3,125`/`193`/`2,481`/`1,041`)<br>`0`-ULP: `36,204 -> 37,644` | **`719` (`41,800`)** |
| `POST_DO` (`105,120`) | `7,289 -> 876` (**`-6,413`**; `13,743`/`6,549`/`136`/`5,548`/`923`)<br>`0`-ULP: `77,901 -> 82,526` | `9,377 -> 3,849` (**`-5,528`**; `13,987`/`5,723`/`195`/`5,037`/`1,030`)<br>`0`-ULP: `76,876 -> 80,883` | `9,468 -> 4,167` (**`-5,301`**; `13,838`/`5,563`/`262`/`3,412`/`1,078`)<br>`0`-ULP: `75,761 -> 78,095` | **`975` (`82,507`)** |
| `EXIT` (`105,120`) | `7,289 -> 876` (**`-6,413`**; `13,743`/`6,549`/`136`/`5,548`/`923`)<br>`0`-ULP: `77,901 -> 82,526` | `9,377 -> 3,849` (**`-5,528`**; `13,987`/`5,723`/`195`/`5,037`/`1,030`)<br>`0`-ULP: `76,876 -> 80,883` | `9,468 -> 4,167` (**`-5,301`**; `13,838`/`5,563`/`262`/`3,412`/`1,078`)<br>`0`-ULP: `75,761 -> 78,095` | **`975` (`82,507`)** |
| **Total (`621,960` rows, `jax.jit`)** | **`35,091 -> 4,961` (`-30,130`)**<br>(`82,382`/`30,811`/`681`/`33,540`/`6,281`)<br>`0`-ULP: **`435,254 -> 462,513` (`+27,259`)** | **`49,621 -> 24,984` (`-24,637`)**<br>(`83,908`/`25,698`/`1,061`/`29,938`/`6,963`)<br>`0`-ULP: **`428,346 -> 451,321` (`+22,975`)** | **`50,243 -> 26,728` (`-23,515`)**<br>(`82,909`/`24,897`/`1,382`/`18,730`/`7,342`)<br>`0`-ULP: **`418,872 -> 430,260` (`+11,388`)** | **`5,373` (`462,482`)**<br>(normal `POST_ZDF` 3-field: `476` over-bar / `17,497` `0`-ULP) |

Why did `r2_baseline` under `jax.jit` have `26,728` over-bar rows on CPU default (`24,984` on CPU `--xla_cpu_max_isa=AVX`) versus `6,584` (`5,107` on `--xla_cpu_max_isa=AVX`) in eager execution?
* When `si3_column_step_arrays` was called without outer `jax.jit`, only the Picard `jax.lax.fori_loop` body inside `_si3_zdf_bl99_step` compiled under XLA; the pre-loop temperature conversions (`ice_temperature_from_enthalpy`, `snow_temperature_from_enthalpy`), the post-loop enthalpy conversions (`ice_enthalpy_from_temperature`, `snow_enthalpy_from_temperature`), and the entire `POST_DH..EXIT` chain executed eagerly one primitive op at a time.
* Under full `jax.jit` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1836`), XLA's target-independent `HloAlgebraicSimplifier` reassociated unguarded expressions in `ice_temperature_from_enthalpy` and `ice_enthalpy_from_temperature` (even under `--xla_cpu_max_isa=AVX`) and XLA CPU's instruction selector emitted FMA (`vfmasd`) instructions across both `POST_ZDF` and `POST_DH..EXIT`. Applying Round 3's in-code `nemo_source_round=True` to `r2_baseline` (`step4_source_round_only`) eliminates that compiler reassociation and FMA contraction under `jax.jit`, reducing CPU production-JIT over-bar rows on `r2_baseline` from `26,728` to **`5,373`** (`462,482` rows at `0` ULP).

### 4.2 Earlier Eager Full-Year Rescore (`si3_column_step_arrays` Unjitted; Inner `fori_loop` Compiled — `round2/*.json`)

For provenance with the Round 2 evidence files (`d5_platform_probe_gpu.json`, `d5_platform_probe_cpu_avx_nofma.json`, `d5_platform_probe_cpu.json`), the **eager** tables originally recorded in Round 2 are retained below and explicitly labeled **eager**:

#### 4.2.1 GPU (`Tesla V100S-PCIE-32GB`, `CUDA_VISIBLE_DEVICES=1`, `SUBNORMAL_PRESERVED`) — **Eager** (`d5_platform_probe_gpu.json`)

| Stage (`eager`) | Over-bar rows before (`False`, `eager`) | Over-bar rows after (`True`, `eager`) | Net delta (`eager`) | Touched rows (`ulp_change > 0`) | Closed to `AT-BAR` | Opened to `DEBT` | Moved to `0` ULP vs oracle | Moved from `0` ULP vs oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ENTRY` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| `POST_ZDF` | `2,070` | `471` | **`-1,599`** | `9,932` | `1,646` | `47` | `6,474` | `498` |
| `POST_DH` | `4,608` | `626` | **`-3,982`** | `11,769` | `4,055` | `73` | `6,592` | `557` |
| `POST_TEMP1` | `4,608` | `626` | **`-3,982`** | `11,769` | `4,055` | `73` | `6,592` | `557` |
| `POST_SAL` | `4,615` | `628` | **`-3,987`** | `11,846` | `4,061` | `74` | `6,636` | `564` |
| `POST_TEMP2` | `4,615` | `628` | **`-3,987`** | `11,846` | `4,061` | `74` | `6,636` | `564` |
| `POST_DO` | `7,256` | `758` | **`-6,498`** | `14,227` | `6,585` | `87` | `7,735` | `569` |
| `EXIT` | `7,256` | `758` | **`-6,498`** | `14,227` | `6,585` | `87` | `7,735` | `569` |
| **Total (`621,960` rows, `eager`)** | **`35,028`** | **`4,495`** | **`-30,533`** | **`85,616`** (`7,462` steps) | **`31,048`** | **`515`** | **`48,400`** | **`3,878`** |

On GPU (both in `eager` and under `jax.jit`), the `0.02510099530281747` subnormal-snow defect at `kt = 5860 POST_ZDF.e_s` is completely eliminated (`597` ULPs / `7.883635078992678e-14` before -> **`0` ULPs / `0.0`** after), and the maximum normalised error across the entire 8,760-step year drops from `2.5926465229707996e-08` (`eager`, `202,041,682` ULPs; `2.5926463176550252e-08` / `202,041,666` ULPs under `jax.jit` at `kt = 4943 POST_DO.e_s`) to **`5.52937990098289e-13`** (`kt = 5247 POST_ZDF.e_i`, `4,709` ULPs in both `eager` and `jax.jit`). On GPU **eager** touched rows (`85,616 + 536,344 == 621,960`):
* `POST_ZDF.e_s` (`5,932` changed): `713` closed to `AT-BAR`, `3` opened, `3,163` moved to `0` ULP, `max_ulp: 869 -> 103`.
* `POST_ZDF.t_su` (`3,027` changed): `530` closed to `AT-BAR`, `1` opened, `2,588` moved to `0` ULP, `max_ulp: 329 -> 39`.
* `POST_ZDF.e_i` (`973` changed): `403` closed to `AT-BAR`, `43` opened, `723` moved to `0` ULP, `max_ulp: 1,636 -> 505`.
* `POST_DH.h_i` / `POST_TEMP1.h_i` / `POST_SAL.h_i` / `POST_TEMP2.h_i` (`1,216` changed each): `1,188` closed to `AT-BAR`, `0` opened, `685` moved to `0` ULP, `max_ulp: 67,563,669 -> 7`.
* `POST_DH.e_i` / `POST_TEMP1.e_i` / `POST_SAL.e_i` / `POST_TEMP2.e_i` (`2,146` changed each): `1,538` closed to `AT-BAR`, `44` opened, `47` moved to `0` ULP, `max_ulp: 15,764,203 -> 4,700`.
* `POST_DH.h_s` / `POST_TEMP1.h_s` / `POST_SAL.h_s` / `POST_TEMP2.h_s` (`147` changed each): `38` closed to `AT-BAR`, `0` opened, `135` moved to `0` ULP, `max_ulp: 205,257,104 -> 55`.
* `POST_DO.e_s` / `EXIT.e_s` (`5,179` changed each): `878` closed to `AT-BAR`, `38` opened, `3,125` moved to `0` ULP, `max_ulp: 202,041,682 -> 61`.
* `POST_DO.v_i` / `EXIT.v_i` (`1,214` changed each): `1,188` closed to `AT-BAR`, `0` opened, `724` moved to `0` ULP, `max_ulp: 60,807,302 -> 6`.
* `POST_DO.sv_i` / `EXIT.sv_i` (`1,264` changed each): `1,196` closed to `AT-BAR`, `0` opened, `784` moved to `0` ULP, `max_ulp: 60,966,878 -> 6`.
* `POST_DO.szv_i` / `EXIT.szv_i` (`1,255` changed each): `1,195` closed to `AT-BAR`, `1` opened, `305` moved to `0` ULP, `max_ulp: 81,289,174 -> 19`.
* `POST_DO.v_s` / `EXIT.v_s` (`144` changed each): `36` closed to `AT-BAR`, `0` opened, `134` moved to `0` ULP, `max_ulp: 184,731,394 -> 49`.

#### 4.2.2 CPU (`--xla_cpu_max_isa=AVX` vs Default `XLA_FLAGS=""`) — **Eager** (`d5_platform_probe_cpu_avx_nofma.json`, `d5_platform_probe_cpu.json`)

| Stage (`eager`) | CPU FMA Off (`--xla_cpu_max_isa=AVX`, `eager`) Before -> After (Delta) | CPU FMA Off (`eager`) Closed / Opened / To `0` ULP / From `0` ULP | CPU Default (`XLA_FLAGS=""`, `eager`) Before -> After (Delta) | CPU Default (`eager`) Closed / Opened / To `0` ULP / From `0` ULP |
|---|---:|---:|---:|---:|
| `ENTRY` | `0 -> 0` (`0`) | `0` / `0` / `0` / `0` | `0 -> 0` (`0`) | `0` / `0` / `0` / `0` |
| `POST_ZDF` | `2,087 -> 545` (**`-1,542`**) | `1,589` / `47` / `6,418` / `498` | `2,090 -> 646` (**`-1,444`**) | `1,519` / `75` / `3,601` / `1,147` |
| `POST_DH` | `4,648 -> 698` (**`-3,950`**) | `4,023` / `73` / `6,568` / `557` | `4,630 -> 899` (**`-3,731`**) | `3,885` / `154` / `4,118` / `1,266` |
| `POST_TEMP1` | `4,648 -> 698` (**`-3,950`**) | `4,023` / `73` / `6,568` / `557` | `4,630 -> 899` (**`-3,731`**) | `3,885` / `154` / `4,118` / `1,266` |
| `POST_SAL` | `4,655 -> 700` (**`-3,955`**) | `4,029` / `74` / `6,612` / `564` | `4,640 -> 901` (**`-3,739`**) | `3,894` / `155` / `4,158` / `1,277` |
| `POST_TEMP2` | `4,655 -> 700` (**`-3,955`**) | `4,029` / `74` / `6,612` / `564` | `4,640 -> 901` (**`-3,739`**) | `3,894` / `155` / `4,158` / `1,277` |
| `POST_DO` | `7,342 -> 883` (**`-6,459`**) | `6,546` / `87` / `7,712` / `569` | `7,326 -> 1,169` (**`-6,157`**) | `6,362` / `205` / `5,252` / `1,292` |
| `EXIT` | `7,342 -> 883` (**`-6,459`**) | `6,546` / `87` / `7,712` / `569` | `7,326 -> 1,169` (**`-6,157`**) | `6,362` / `205` / `5,252` / `1,292` |
| **Total (`621,960` rows, `eager`)** | **`35,377 -> 5,107` (`-30,270`)** | **`30,785` / `515` / `48,202` / `3,878`** | **`35,282 -> 6,584` (`-28,698`)** | **`29,801` / `1,103` / `30,657` / `8,817`** |

#### 4.2.3 Touched Field Breakdown on Default CPU (`eager`, `84,981` rows touched, `536,979` rows untouched in `d5_platform_probe_cpu.json`)
* **`POST_ZDF` (`9,865` touched rows, `eager`):**
  * `POST_ZDF.e_s`: `5,961` changed rows, `666` closed to `AT-BAR`, `11` opened to `DEBT`, `1,452` moved to `0` ULP vs oracle, `386` moved from `0` ULP; max ULP vs oracle on touched rows dropped from `869` to `80`.
  * `POST_ZDF.t_su`: `2,930` changed rows, `499` closed to `AT-BAR`, `2` opened to `DEBT`, `1,576` moved to `0` ULP vs oracle, `638` moved from `0` ULP.
  * `POST_ZDF.e_i`: `974` changed rows, `354` closed to `AT-BAR`, `62` opened to `DEBT`, `573` moved to `0` ULP vs oracle, `123` moved from `0` ULP.
* **`POST_DH` and `POST_TEMP1` (`11,676` touched rows each, `eager`):**
  * `e_s`: `5,260` changed, `701` closed, `79` opened, `1,726` moved to `0` ULP.
  * `t_su`: `2,930` changed, `499` closed, `2` opened, `1,576` moved to `0` ULP.
  * `e_i`: `2,141` changed, `1,479` closed, `62` opened, `45` moved to `0` ULP.
  * `h_i`: `1,216` changed, `1,178` closed, `4` opened, `679` moved to `0` ULP.
  * `h_s`: `129` changed, `28` closed, `7` opened, `92` moved to `0` ULP (max ULP on touched rows `205,257,104 -> 102,609`).
* **`POST_SAL` and `POST_TEMP2` (`11,754` touched rows each, `eager`):**
  * Same as `POST_DH` plus `sz_i`: `78` changed, `9` closed, `1` opened, `40` moved to `0` ULP.
* **`POST_DO` and `EXIT` (`14,128` touched rows each, `eager`):**
  * `e_s`: `5,206` changed, `797` closed, `109` opened, `1,734` moved to `0` ULP (max ULP on touched rows `202,041,682 -> 101,580`).
  * `t_su`: `2,930` changed, `499` closed, `2` opened, `1,576` moved to `0` ULP.
  * `e_i`: `2,140` changed, `1,493` closed, `73` opened, `68` moved to `0` ULP.
  * `sv_i`: `1,262` changed, `1,186` closed, `5` opened, `769` moved to `0` ULP.
  * `szv_i`: `1,253` changed, `1,183` closed, `5` opened, `298` moved to `0` ULP.
  * `v_i`: `1,213` changed, `1,178` closed, `4` opened, `715` moved to `0` ULP.
  * `v_s`: `124` changed, `26` closed, `7` opened, `92` moved to `0` ULP (max ULP on touched rows `184,731,394 -> 92,348`).

### Operator note on reviewer-measured numbers

The `1,811 / 7,385` snow-temperature counts, the `26,202` POST_ZDF rows and the `1,625 -> 0` over-bar result quoted in 4.4 were measured by the operator's independent Claude review (in eager mode for `snow_temperature_from_enthalpy` and `_si3_zdf_bl99_step`), not by Round 2's initial probe. Its probe scripts and output are kept, uncommitted, under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round2/operator_review_claude/` (`si3r2_*.py`, `si3r2_bridge_probe_out.json`). Round 3 reproduces them in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:404-699` and `docs/ice/fidelity/si3_r3_d5_receipt.md:36-65` under both `jax.jit` and eager execution.

### 4.4 Next D5 Items (Upstream Entry-Input Mismatches, Recorded for Next Round)

Per the one-statement-per-round rule, the two upstream input mismatches that explain all remaining normal-step `POST_ZDF` over-bar rows (`471` `POST_ZDF` over-bar rows in `eager` / `497` under `jax.jit` on GPU) are recorded here as the next D5 items:
1. **Harness global-to-equivalent entry conversion (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-286`):** `_exact_1d_entry` converted layer enthalpy and salinity from global frame fields as `(g["e_i"] * 3.0) / vi`, `(g["e_s"] * 3.0) / vs`, and `g["szv_i"] * 3.0 / vi`, whereas NEMO's `ice_var_glo2eqv` computes `(e_i / v_i) * zlay_i` and `(e_s / v_s) * zlay_s` (`src/ICE/icevar.F90:390`, `src/ICE/icevar.F90:412-413`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:404`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`) and computes layer salinity `sz_i` via `ice_var_salprof` (`src/ICE/icevar.F90:488-506`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:502-520`).
2. **Solver-entry snow temperature formula (`packages/ice/legoesm/ice/bitz_lipscomb.py:417-426`, called at `packages/ice/legoesm/ice/bitz_lipscomb.py:1841-1843`):** `snow_temperature_from_enthalpy` evaluated `-e * inverse_rho_snow * inverse_c_ice + constants.latent_fusion * inverse_c_ice` from `src/ICE/icethd_dh.F90:498-503` (compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:508-513`), whereas NEMO at solver entry (`ice_var_glo2eqv`) evaluates `rt0 + MAX( -100._wp , MIN( r1_rcpi*( -r1_rhos*( e_s / v_s * zlay_s ) + rLfus ) , 0._wp ) )` (`src/ICE/icevar.F90:412-413`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`).
* **Measured impact:** Recorded ZDF-entry snow temperature is reproduced to `0` ULP in `1,811` of `7,385` snow steps as shipped in `eager` mode (`409 / 7,385` under `jax.jit` on CPU default `XLA_FLAGS=""`, `398 / 7,385` under `jax.jit` on CPU `--xla_cpu_max_isa=AVX` and GPU), and in all **`7,385` of `7,385` snow steps** under both `jax.jit` and `eager` when both (1) and (2) are aligned with NEMO. Fed NEMO-exact ZDF inputs (`e / v * 3`, `ice_var_salprof` `sz_i`, and `src/ICE/icevar.F90:412-413` entry snow temperature) with in-code `nemo_source_round`, all **`26,202` normal-step `POST_ZDF` rows** across the 8,760-step year match NEMO to **`0` ULP** under `jax.jit` (and in `eager`: `1,625 -> 0` over-bar on GPU and CPU `--xla_cpu_max_isa=AVX`; under `jax.jit`: `4,313 -> 0` on CPU default `XLA_FLAGS=""`, `4,149 -> 0` on CPU `--xla_cpu_max_isa=AVX`, and `1,545 -> 0` on GPU).

---

## 5. Verification Tests, Planted Controls, and Autodiff Check

`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:61-2776` pins all five required properties:
1. **One-step bit-exact statement test + revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:61-313`):**
   * Pins the scalar NEMO replay of `zh_s`, `z1_h_s`, `zkappa_s = 1500.0`, and `zeta_s = 0x402f37c77f84d1ee` against legacy `0x402f37c77f84d1ef` (1 ULP), and pins the iteration-frozen `zkappa` conductive-flux transcription (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:782-784`) at `kt = 4242` (`zq_top = 0x410e81a8bd7ce42a`, `0` ULP vs `634,976,763` ULP when conductivity is recomputed from post-solve temperatures).
   * Verifies on `kt = 5841` (and `kt = 5860` with snow present) that `_si3_zdf_bl99_step(..., _nemo_zdf_layer_reciprocals=True)` matches NEMO's `POST_ZDF` `t_su`, `e_s`, and `e_i` to **0 ULPs (`0.0`)**, and **fails** when reverted (`_nemo_zdf_layer_reciprocals=False`), jumping to `710..716` ULPs (`norm_err > 9.4e-14 > 1e-15`) at `kt = 5841` and `597` ULPs (`norm_err = 7.883635078992678e-14 > 1e-15`) at `kt = 5860`.
2. **Autodiff check (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:316-472`):**
   * Evaluates `jax.value_and_grad` through `_si3_zdf_bl99_step` and `_radiation` across clamped-snow (`kt = 5841`), thick-snow (`kt = 1`), and snow-free regimes, verifying that forward outputs under `value_and_grad` (`T_surface`, `e_snow`, `e_ice`, `qcn_ice_top`, `qcn_ice_bottom`) are bit-identical (`0` ULPs) to the standalone primal call, all reverse-mode gradients with respect to `(e_ice, e_snow, h_ice, h_snow, T_surface, qns_ice)` are finite and nonzero, and reverse-mode `jax.grad` matches forward-mode `jax.jvp` to `< 1e-11` relative error. Includes a planted `jax.lax.stop_gradient` control that fires when snow-branch gradients are zeroed.
3. **Non-SI3 caller bit-for-bit invariance against parent commit + planted code-path controls (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:475-1023`):**
   * Loads the parent-commit (`59fd8f2395ccba60b5b04f3c2c68946835feea12`) modules via `git show` and compares `bitz_lipscomb_conduction_step`, `thomas_solve(..., "normalised")`, `IceColumnModel.step()`, and `step_sea_ice` (both slab and dynamic-v2) bit-for-bit (`uint64`) against the current commit across multiple thicknesses, and proves the test fails (`pytest.raises(AssertionError)`) when any of the four non-SI3 code paths is perturbed.
4. **FTZ vs DAZ classification test + planted control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1026-1074`):**
   * Verifies `classify_ftz_daz` reports `daz_flushes_subnormal_input=False` under `FTZ`-only (`x * 1.0 == 0.0` and `x + 0.0 == 0.0` while `h_s > 0.0` is `True`) and proves the planted invalid formula (`plant_invalid_daz_from_mul_add=True`) fails.
5. **Probe & planted failing controls (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1077-1232`):**
   * Verifies `--plant zero_subnormal_input`, `--plant flush_recorded_operand`, and `--plant revert_statement_fix` all cause `nemo_si3thd_d5_platform_probe.py` to exit with code `1`.

---

## 6. Choices List and Review Disposition

* **CHOICES:**
  * `ASKED`: **DECISION 156** (taken by user before round start: classify and close D5 per platform first, then start ORCA2 rungs).
  * `ASKED`: **Certification platform for the SI3 sea-ice ladder** (surfaced in `DECISION_NEEDED` and resolved by user in **DECISION 158**: `(1)` CPU (`fp64`, production JIT, `XLA_FLAGS=""`) with XLA's FMA prevented in code via `nemo_source_round`, where the 26 subnormal steps `kt = 5842..5867` remain a stated `D5_PLATFORM_FLOOR` of `629` rows; each rung spot-checked on an idle GPU).
  * `UNASKED`: **NONE**. `_nemo_zdf_layer_reciprocals` defaulting to `True` on the internal SI3 path (`_radiation`, `_si3_zdf_bl99_step`, `si3_column_step_arrays`) follows the existing private `_nemo_*` flag pattern in `packages/ice/legoesm/ice/bitz_lipscomb.py` and Rule 3 (the fixed value is the default); the operator accepts it. No configuration classes (`SeaIceConfig`, `SI3ThermoConfig`) were modified. Note per DECISION 156: SI3 ladder design decisions 2–8 (`docs/ice/fidelity/si3_ladder_design.md:354-365`) remain queued alongside the platform certification decision.
* **Independent review:** `independent review: operator`
