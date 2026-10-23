# SI3 Ladder Round 2 (`si3_r2`) Receipt — Rung D5 (C1D Subnormal-Snow Platform Classification & ZDF Layer-Reciprocal Closure)

## 1. Scope, Preregistration, and Provenance

This round executes rung **D5** of the SI3 sea-ice fidelity ladder (`docs/ice/fidelity/si3_ladder_design.md:220`, `docs/ice/fidelity/si3_ladder_design.md:344-350`) under **DECISION 156**: classify and close lane 3's C1D subnormal-snow defect per platform before starting the ORCA2 rungs, reusing the immutable 8,760-step C1D North Greenland ERA5 SAS record (`c1d_omip_l3_sasice_scalarmath_v2_a`) from `docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase6_receipt.md:194-209` and `docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md:30-43`.

* **Preregistration commit (before measurement):** `59fd8f2395ccba60b5b04f3c2c68946835feea12` (`docs/ice/fidelity/si3_r2_d5_preregister.md:1-74`)
* **Implementation & probe commit:** `852f85571fba437241e7745372aeb20bb7afdb82`
* **Oracle record root:** `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a`
  * `oracle_si3_thd_frames.bin` SHA-256: `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b`
  * `oracle_si3_zdf_inputs.bin` SHA-256: `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd`
* **Committed probe:** `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:115-774`
* **Committed unit & gradient tests:** `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:55-486`
* **Evidence artifacts under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round2/`:**
  * `d5_platform_probe_cpu.json` SHA-256: `c91a4fa8eb4f905c6ca4e8c521200aa202dca29794e8b36b5b242144cbe7d7d9`
  * `d5_platform_probe_gpu.json` SHA-256: `10e2df24563920c8b4eb0f0b4c31bb2584fb6d3d248cff3a8233f9efcaaff93c`

---

## 2. Stage A & Stage B: Platform Subnormal Probe and FTZ/DAZ Stamp

`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:115-176` executes four compiled `@jax.jit` binary64 probes under `PrecisionPolicy.fp64()` (`JAX_ENABLE_X64=1`) on the smallest positive binary64 subnormal `x_sub = 5e-324` (`0x0000000000000001`) and normal factors `a = 2^-1022` (`0x0010000000000000`), `b = 2^-52` (`0x3cb0000000000000`), followed by the compiled ZDF snow-present predicate `h_s > 0.0` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:102-112`) across all 26 C1D steps (`kt = 5842..5867`) where NEMO's ZDF-entry snow thickness is subnormal (`0 < h_s < 2.2250738585072014e-308`):

| Probe metric | CPU (`x86_64`, `cpu:0`, `jax 0.10.0`) | GPU (`Tesla V100S-PCIE-32GB`, `cuda:0`, `jax 0.10.0`) |
|---|---|---|
| Input `x_sub` (`float64` / bits) | `5e-324` / `0x0000000000000001` | `5e-324` / `0x0000000000000001` |
| `_jit_identity(x_sub)` | `5e-324` / `0x0000000000000001` | `5e-324` / `0x0000000000000001` |
| `_jit_mul_one(x_sub, 1.0)` | `0.0` / `0x0000000000000000` | `5e-324` / `0x0000000000000001` |
| `_jit_add_zero(x_sub, 0.0)` | `0.0` / `0x0000000000000000` | `5e-324` / `0x0000000000000001` |
| `_jit_produce_subnormal(2^-1022, 2^-52)` | `0.0` / `0x0000000000000000` | `5e-324` / `0x0000000000000001` |
| `daz_flushes_subnormal_input` | `True` | `False` |
| `ftz_flushes_subnormal_output` | `True` | `False` |
| `ftz_daz_mode` stamp | `FTZ_DAZ_FLUSH_TO_ZERO` | `IEEE754_GRADUAL_UNDERFLOW` |
| Recorded `kt = 5842` entry `h_s` | `5.562684646268003e-309` (`0x0004000000000000`) | `5.562684646268003e-309` (`0x0004000000000000`) |
| Recorded `kt = 5843..5867` entry `h_s` | `5e-324` (`0x0000000000000001`) | `5e-324` (`0x0000000000000001`) |
| `_jit_snow_pred(h_s)` true count (`kt = 5842..5867`) | `0 / 26` (`predicate_all_false = True`) | `26 / 26` (`predicate_all_true = True`) |
| `_jit_isnow(h_s)` (`kt = 5842..5867`) | `0.0` (`0x0000000000000000`) | `1.0` (`0x3ff0000000000000`) |
| **Platform disposition** | **`D5_PLATFORM_FLOOR`** | **`SUBNORMAL_PRESERVED`** |

### CPU Disposition (`D5_PLATFORM_FLOOR`)
On `x86_64` CPU (`JAX_PLATFORMS=cpu`), `_jit_identity` is a buffer pass-through that executes no floating-point instruction, whereas every compiled floating-point instruction (`_jit_mul_one`, `_jit_add_zero`, `_jit_produce_subnormal`, and `_jit_snow_pred`) executes with XLA CPU's MXCSR `FTZ=1` / `DAZ=1` state, flushing subnormal inputs and outputs to `0x0000000000000000` (`0.0`). Consequently, across all 26 subnormal-snow steps `kt = 5842..5867`, `h_s > 0.0` evaluates to `False` (`isnow = 0.0`) on CPU and enters the snow-free 4-equation BL99 tridiagonal branch instead of NEMO's snow-present 7-equation branch. Per **DECISION 156** and `docs/ice/fidelity/si3_ladder_design.md:344-350`, CPU is declared **`D5_PLATFORM_FLOOR`**, no subnormal workaround is introduced, and the 26 subnormal steps (`75` `POST_ZDF` over-bar rows, `629` all-stage over-bar rows in `kt = 5842..5867`, led by `kt = 5860 POST_ZDF.e_s` at normalised error `0.02510099530281747` and `190,081,022,848,368` ULPs) remain reported as unresolved platform-floor rows on CPU.

---

## 3. Stage C: First Statement Departure on the Subnormal-Preserving Platform (GPU) and Selectable SI3 Fix

On `Tesla V100S-PCIE-32GB` (`cuda:0`), IEEE-754 gradual underflow is preserved (`SUBNORMAL_PRESERVED`) and `h_s > 0.0` evaluates to `True` (`isnow = 1.0`) for all 26 steps `kt = 5842..5867`. However, before the statement fix (`_nemo_zdf_layer_reciprocals=False`), `_si3_zdf_bl99_step` still disagreed with NEMO across `kt = 5841..5867`, producing `57` over-bar `POST_ZDF` rows (`80` nonzero-ULP `POST_ZDF` rows) across `kt = 5842..5867`, including `597` ULPs (`norm_err = 7.883635078992678e-14 > 1e-15`) at `kt = 5860 POST_ZDF.e_s` and `710` ULPs (`716` ULPs on CPU, `norm_err = 9.468475654165245e-14 > 1e-15`) at `kt = 5841 POST_ZDF.e_s` (where `h_s = 5.010420900022432e-293 m` = `0x0340000000000000` is a normal binary64 number clamped to `zh_min = 1.0e-3 m`).

### First Statement Departure (NEMO vs legoESM)

In NEMO's BL99 vertical heat diffusion (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:170-186` and compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:172-188`):
* Layer thicknesses multiply by the precomputed reciprocals `r1_nlay_i = 1.0_wp / 3.0_wp` and `r1_nlay_s = 1.0_wp / 3.0_wp` and precompute layer-thickness reciprocals `z1_h_i(ji) = 1._wp / zh_i(ji)` and `z1_h_s(ji) = 1._wp / zh_s(ji)`.
* Solar radiation transmission uses `zh_s(ji)` and `zh_i(ji)` (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:217-230`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:219-232`).
* Thermal conductances multiply by `z1_h_s(ji)` and `z1_h_i(ji)` and form the snow-ice harmonic interface conductance with `0.5_wp * ( ztcond_i(ji,0) * zh_s(ji) + rcnd_s * zh_i(ji) )` (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:332-341`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:327-336`).
* Tridiagonal prefactors multiply by `z1_h_i(ji)` and `z1_h_s(ji)`: `zeta_i(ji,jk) = rDt_ice * r1_rhoi * z1_h_i(ji) / zcpi` and `zeta_s(ji,jk) = rDt_ice * r1_rhos * r1_rcpi * z1_h_s(ji)` (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:366-376`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:353-364`).
* Post-solve conductive fluxes multiply by `zkappa_i` / `zkappa_s`, which carry `z1_h_i(ji)` / `z1_h_s(ji)` (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:821-823`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:782-784`).

In `packages/ice/legoesm/ice/bitz_lipscomb.py`, legacy legoESM instead divided by `3.0` (`dz_i = jnp.maximum(h_ice, _H_MIN) / 3.0`, `dz_s = jnp.maximum(h_snow, _H_MIN) / 3.0`) and divided directly by `dz_i` and `dz_s` (`eta_i = dt / (constants.rho_ice * dz_i[..., None] * ci)`, `eta_s = dt / (constants.rho_snow * dz_s[..., None] * constants.c_ice)`). Under `NEMO_SI3_CONSTANTS_CONFIG` (`rho_snow = 330.0`, `c_ice = 2096.7`, `dt = 3600.0`) on clamped snow (`h_s <= 1.0e-3 m`):
* `zh_s = 1.0e-3 * (1.0 / 3.0) = 0x3f35d867c3ece2a5`, `z1_h_s = 1.0 / zh_s = 3000.0` (`0x40a7700000000000`).
* NEMO's `zeta_s = rDt_ice * r1_rhos * r1_rcpi * z1_h_s = 0x402f37c77f84d1ee` (`15.608943924998249`), whereas legacy `dt / (rho_snow * dz_s * c_ice) = 0x402f37c77f84d1ef` (`15.608943924998251`, 1 ULP larger).
* Multiplied by `zkappa_s = 1500.0` (`eta_s * k_s ~ 2.34e4`) in the stiff clamped-snow tridiagonal rows, that 1-ULP difference in `eta_s` (together with `zh_i` / `z1_h_i`) amplifies into `470..718` ULPs in `POST_ZDF.e_s` (`norm_err` up to `9.55e-14 > 1e-15`).

### Selectable SI3 Statement Fix
We added the selectable SI3 statement flag `_nemo_zdf_layer_reciprocals: bool = True` in `packages/ice/legoesm/ice/bitz_lipscomb.py`:
* `_radiation` (`packages/ice/legoesm/ice/bitz_lipscomb.py:355-398`): computes `dzs` and `dzi` via `jnp.maximum(_H_MIN, h_s) * (1.0 / 3.0)` and `jnp.maximum(_H_MIN, h_i) * (1.0 / 3.0)` gated by `h_s > 0.0` and `h_i > 0.0`.
* `_si3_zdf_bl99_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:401-417`, `packages/ice/legoesm/ice/bitz_lipscomb.py:465-524`, `packages/ice/legoesm/ice/bitz_lipscomb.py:682-712`): computes `zh_i`, `z1_h_i`, `zh_s`, `z1_h_s` and uses NEMO's written multiplication order for `k_i`, `k_s`, `k_si`, `eta_i`, `eta_s`, `k_top_i`, `k_base_i`, and `k_top`.
* `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1100-1127`): forwards `_nemo_zdf_layer_reciprocals`.
* No configuration classes (`SeaIceConfig`, `SI3ThermoConfig`) were modified, and `bitz_lipscomb_conduction_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:280-310`) and all non-SI3 callers are completely untouched.

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

Across all 26 subnormal ZDF-entry steps (`kt = 5842..5867`) on GPU, `POST_ZDF` over-bar rows drop from **`58` to `1`** (the single remaining row is `kt = 5857 POST_ZDF.e_i` at `65` ULPs, `norm_err = 7.251983965966999e-15`, caused by a 2-ULP entry-bridge difference in `ENTRY.e_i[0]` from `(e_i * 3.0) / v_i` vs `(e_i / v_i) * 3.0`; when `_si3_zdf_bl99_step` is called with the exact `oracle_si3_zdf_inputs.bin` operand at `kt = 5857`, all `POST_ZDF` outputs are `0` ULPs), and all-stage over-bar rows in `kt = 5842..5867` drop from **`280` to `17`**. In the boundary gate (`tests/ocean/fidelity/test_nemo_si3thd_phase2_gate.py:208-219`), closing `kt = 5` and `kt = 6` `POST_ZDF` advances the first over-bar step from `kt = 5` (`kt5.POST_ZDF.e_s`) to `kt = 7` (`kt7.POST_DH.e_i`, `sweep_steps_examined = 7`).

---

## 4. Stage D: Full-Year (8,760-Step) C1D Exact-Entry Rescore

`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:329-618` re-scores all `8,760` steps (`621,960` stage-field rows) of `C1D_OMIP_L3` before (`_nemo_zdf_layer_reciprocals=False`) vs after (`_nemo_zdf_layer_reciprocals=True`) on both GPU (`d5_platform_probe_gpu.json`, where subnormals are preserved) and CPU (`d5_platform_probe_cpu.json`, where `kt = 5842..5867` hit `D5_PLATFORM_FLOOR`):

### GPU (`Tesla V100S-PCIE-32GB`, `SUBNORMAL_PRESERVED`) Full-Year Rescore

| Stage | Over-bar rows before (`False`) | Over-bar rows after (`True`) | Net delta | Touched rows (`ulp_change > 0`) | Closed to `AT-BAR` | Opened to `DEBT` | Moved to `0` ULP vs oracle | Moved from `0` ULP vs oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ENTRY` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| `POST_ZDF` | `2,070` | `471` | **`-1,599`** | `9,932` | `1,646` | `47` | `6,474` | `498` |
| `POST_DH` | `4,608` | `2,905` | **`-1,703`** | `9,463` | `1,776` | `73` | `5,920` | `557` |
| `POST_TEMP1` | `4,608` | `2,905` | **`-1,703`** | `9,463` | `1,776` | `73` | `5,920` | `557` |
| `POST_SAL` | `4,615` | `2,907` | **`-1,708`** | `9,540` | `1,782` | `74` | `5,964` | `564` |
| `POST_TEMP2` | `4,615` | `2,907` | **`-1,708`** | `9,540` | `1,782` | `74` | `5,964` | `564` |
| `POST_DO` | `7,256` | `5,340` | **`-1,916`** | `9,609` | `2,003` | `87` | `6,045` | `569` |
| `EXIT` | `7,256` | `5,340` | **`-1,916`** | `9,609` | `2,003` | `87` | `6,045` | `569` |
| **Total (`621,960` rows)** | **`35,028`** | **`22,775`** | **`-12,253`** | **`67,156`** (`6,314` steps) | **`12,768`** | **`515`** | **`42,332`** | **`3,878`** |

On GPU (`67,156 + 554,804 == 621,960`), the `0.02510099530281747` subnormal-snow defect at `kt = 5860 POST_ZDF.e_s` is completely eliminated (`597` ULPs / `7.883635078992678e-14` before -> **`0` ULPs / `0.0`** after), and the maximum normalised error across the entire 8,760-step year is `2.5926465229707996e-08` (`kt = 4943 POST_DO.e_s`). On GPU touched rows:
* `POST_ZDF.e_s` (`5,932` changed): `713` closed to `AT-BAR`, `3` opened, `3,163` moved to `0` ULP, `max_ulp: 869 -> 103`.
* `POST_ZDF.t_su` (`3,027` changed): `530` closed to `AT-BAR`, `1` opened, `2,588` moved to `0` ULP, `max_ulp: 329 -> 39`.
* `POST_ZDF.e_i` (`973` changed): `403` closed to `AT-BAR`, `43` opened, `723` moved to `0` ULP, `max_ulp: 1,636 -> 505`.
* `POST_DH.h_s` / `POST_TEMP1.h_s` / `POST_SAL.h_s` / `POST_TEMP2.h_s` (`145` changed each): `36` closed to `AT-BAR`, `0` opened, `133` moved to `0` ULP, `max_ulp: 205,883 -> 55`.
* `POST_DO.e_s` / `EXIT.e_s` (`5,177` changed each): `876` closed to `AT-BAR`, `38` opened, `3,123` moved to `0` ULP, `max_ulp: 202,506 -> 61`.
* `POST_DO.v_s` / `EXIT.v_s` (`142` changed each): `34` closed to `AT-BAR`, `0` opened, `132` moved to `0` ULP, `max_ulp: 185,295 -> 49`.

### CPU (`x86_64`, `D5_PLATFORM_FLOOR`) Full-Year Rescore

| Stage | Over-bar rows before (`False`) | Over-bar rows after (`True`) | Net delta | Touched rows (`ulp_change > 0`) | Closed to `AT-BAR` | Opened to `DEBT` | Moved to `0` ULP vs oracle | Moved from `0` ULP vs oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ENTRY` | `0` | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| `POST_ZDF` | `2,090` | `646` | **`-1,444`** | `9,865` | `1,519` | `75` | `3,601` | `1,147` |
| `POST_DH` | `4,630` | `3,173` | **`-1,457`** | `9,369` | `1,611` | `154` | `3,446` | `1,266` |
| `POST_TEMP1` | `4,630` | `3,173` | **`-1,457`** | `9,369` | `1,611` | `154` | `3,446` | `1,266` |
| `POST_SAL` | `4,640` | `3,175` | **`-1,465`** | `9,447` | `1,620` | `155` | `3,486` | `1,277` |
| `POST_TEMP2` | `4,640` | `3,175` | **`-1,465`** | `9,447` | `1,620` | `155` | `3,486` | `1,277` |
| `POST_DO` | `7,326` | `5,746` | **`-1,580`** | `9,503` | `1,785` | `205` | `3,563` | `1,292` |
| `EXIT` | `7,326` | `5,746` | **`-1,580`** | `9,503` | `1,785` | `205` | `3,563` | `1,292` |
| **Total (`621,960` rows)** | **`35,282`** | **`24,834`** | **`-10,448`** | **`66,503`** (`6,305` steps) | **`11,551`** | **`1,103`** | **`24,591`** | **`8,817`** |

### Touched Field Breakdown Across the Year (`66,503` rows touched, `555,457` rows untouched)

Every touched row in `d5_platform_probe_cpu.json` is accounted for by stage and field (`66,503 + 555,457 == 621,960`):
* **`POST_ZDF` (`9,865` touched rows):**
  * `POST_ZDF.e_s`: `5,961` changed rows, `666` closed to `AT-BAR`, `11` opened to `DEBT`, `1,452` moved to `0` ULP vs oracle, `386` moved from `0` ULP; max ULP vs oracle on touched rows dropped from `869` to `80`.
  * `POST_ZDF.t_su`: `2,930` changed rows, `499` closed to `AT-BAR`, `2` opened to `DEBT`, `1,576` moved to `0` ULP vs oracle, `638` moved from `0` ULP.
  * `POST_ZDF.e_i`: `974` changed rows, `354` closed to `AT-BAR`, `62` opened to `DEBT`, `573` moved to `0` ULP vs oracle, `123` moved from `0` ULP.
* **`POST_DH` and `POST_TEMP1` (`9,369` touched rows each):**
  * `e_s`: `5,258` changed, `701` closed, `79` opened, `1,724` moved to `0` ULP.
  * `t_su`: `2,930` changed, `499` closed, `2` opened, `1,576` moved to `0` ULP.
  * `e_i`: `997` changed, `365` closed, `62` opened, `28` moved to `0` ULP.
  * `h_s`: `127` changed, `26` closed, `7` opened, `90` moved to `0` ULP (max ULP on touched rows `205,883 -> 102,609`).
  * `h_i`: `57` changed, `20` closed, `4` opened, `28` moved to `0` ULP.
* **`POST_SAL` and `POST_TEMP2` (`9,447` touched rows each):**
  * Same as `POST_DH` plus `sz_i`: `78` changed, `9` closed, `1` opened, `40` moved to `0` ULP.
* **`POST_DO` and `EXIT` (`9,503` touched rows each):**
  * `e_s`: `5,204` changed, `795` closed, `109` opened, `1,732` moved to `0` ULP (max ULP on touched rows `202,506 -> 101,580`).
  * `t_su`: `2,930` changed, `499` closed, `2` opened, `1,576` moved to `0` ULP.
  * `e_i`: `996` changed, `394` closed, `73` opened, `37` moved to `0` ULP.
  * `v_s`: `122` changed, `24` closed, `7` opened, `90` moved to `0` ULP (max ULP on touched rows `185,295 -> 92,348`).
  * `sv_i`: `103` changed, `28` closed, `5` opened, `64` moved to `0` ULP.
  * `szv_i`: `94` changed, `25` closed, `5` opened, `36` moved to `0` ULP.
  * `v_i`: `54` changed, `20` closed, `4` opened, `28` moved to `0` ULP.
* **Subnormal window (`kt = 5842..5867`) on CPU:** `629` over-bar rows before and `629` after (unchanged because CPU is `D5_PLATFORM_FLOOR`; `49` ice-side rows in `kt = 5842..5867` change by 1-2 ULPs from `z1_h_i` without crossing the bar). The largest normal row on CPU remains `kt = 5860 POST_ZDF.e_s` (`norm_err = 0.02510099530281747`, `max_abs = 2832427.961354971`, `scale = 112841260.96135496`, `190,081,022,848,368` ULPs), while on GPU `kt = 5860 POST_ZDF.e_s` closes to `0` ULPs (`0.0`).

---

## 5. Verification Tests, Planted Controls, and Autodiff Check

`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:55-486` pins all four required properties:
1. **One-step bit-exact statement test + revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:55-216`):**
   * Pins the scalar NEMO replay of `zh_s`, `z1_h_s`, `zkappa_s = 1500.0`, and `zeta_s = 0x402f37c77f84d1ee` against legacy `0x402f37c77f84d1ef` (1 ULP).
   * Verifies on `kt = 5841` (and `kt = 5860` with snow present) that `_si3_zdf_bl99_step(..., _nemo_zdf_layer_reciprocals=True)` matches NEMO's `POST_ZDF` `t_su`, `e_s`, and `e_i` to **0 ULPs (`0.0`)**, and **fails** when reverted (`_nemo_zdf_layer_reciprocals=False`), jumping to `710..716` ULPs (`norm_err > 9.4e-14 > 1e-15`) at `kt = 5841` and `597` ULPs (`norm_err = 7.883635078992678e-14 > 1e-15`) at `kt = 5860`.
2. **Autodiff check (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:219-373`):**
   * Evaluates `jax.value_and_grad` through `_si3_zdf_bl99_step` and `_radiation` at `kt = 5841`, verifying that forward outputs under `value_and_grad` are bit-identical (`0` ULPs) to the primal call and to NEMO's oracle, all reverse-mode gradients with respect to `(e_ice, e_snow, S_layers, h_ice, h_snow, T_surface)` are finite and nonzero, and reverse-mode `jax.grad` matches forward-mode `jax.jvp` to `< 1e-11` relative error. Includes a planted `jax.lax.stop_gradient` control that fires when snow-branch gradients are zeroed.
3. **Non-SI3 caller invariance (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:376-389`):**
   * Verifies `bitz_lipscomb_conduction_step` outputs are finite and bit-identical.
4. **Probe & planted failing controls (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:392-486`):**
   * Verifies `--plant zero_subnormal_input`, `--plant flush_recorded_operand`, and `--plant revert_statement_fix` all cause `nemo_si3thd_d5_platform_probe.py` to exit with code `1`.

---

## 6. Choices List and Review Disposition

* **CHOICES:**
  * `ASKED`: **DECISION 156** (taken by user before round start: classify and close D5 per platform first, then start ORCA2 rungs).
  * `UNASKED`: **NONE**. No configuration change was made (`SeaIceConfig` and `SI3ThermoConfig` untouched). Note per DECISION 156: SI3 ladder design decisions 2–8 (`docs/ice/fidelity/si3_ladder_design.md:354-365`) remain queued for operator/user confirmation before O1 starts.
* **Independent review:** `independent review: operator`
