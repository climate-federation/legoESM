# SI3 Ladder Round 3 (`si3_r3`) Receipt — Rung D5 Continuation (Harness Entry Conversion, `ice_var_glo2eqv` Snow Temperature, and In-Code CPU FMA Prevention)

## 1. Scope, Preregistration, and Provenance

This round continues rung **D5** of the SI3 sea-ice fidelity ladder (`docs/ice/fidelity/si3_ladder_design.md:220`, `docs/ice/fidelity/si3_ladder_design.md:344-350`) under **DECISION 156** and **DECISION 158**:
* **DECISION 158 (user):** the SI3 ladder certifies on the CPU (`fp64`, production JIT, `XLA_FLAGS=""`) with XLA's fused multiply-add (`FMA`) prevented **in code** at the statements where NEMO's `-O3` build does not fuse — reusing the ocean ladder's helper `packages/core/legoesm/core/source_rounding.py:14-27` (`nemo_source_round`), never a global XLA flag; binary64 subnormal cases (`kt = 5842..5867`) stay a stated platform floor (`D5_PLATFORM_FLOOR`); each rung is spot-checked on an idle GPU.

Four ordered steps were preregistered in `docs/ice/fidelity/si3_r3_d5_preregister.md:1-54` before any measurement and evaluated both cumulatively and in isolation across the immutable 8,760-step C1D North Greenland ERA5 SAS record (`621,960` stage-field rows across 8 stages) under **compiled `jax.jit`** (matching the full-year gate `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1836`) on CPU production JIT (`XLA_FLAGS=""`), CPU no-FMA diagnostic (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`), and an idle GPU (`CUDA_VISIBLE_DEVICES=1`):
1. **Step 1 — Reproduce the Round 2 reviewer's numbers with a committed probe** (`docs/ice/fidelity/si3_r2_d5_receipt.md:194-203`, `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:404-699`).
2. **Step 2 — Harness entry conversion (mimicry glue, not model):** make `_nemo_salprof` and `_entry_arrays` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-286` (and `_exact_1d_entry` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:717-732`) follow NEMO's `(e / v) * 3.0` (`src/ICE/icevar.F90:390`, `src/ICE/icevar.F90:412-413`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:404`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`) and the complete option-2 salinity profile (`src/ICE/icevar.F90:354-360`, `src/ICE/icevar.F90:488-506`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:368-374`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:502-520`), including both SSS-dependent guards.
3. **Step 3 — Model statement (`snow_temperature_from_enthalpy`):** transcribe NEMO's solver-entry snow temperature statement (`src/ICE/icevar.F90:412-413`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`) in `packages/ice/legoesm/ice/bitz_lipscomb.py:371-413` (called at `packages/ice/legoesm/ice/bitz_lipscomb.py:814-819`) under the selectable SI3 flag `_nemo_glo2eqv_snow_temp: bool = True`, keeping non-SI3 callers bit-for-bit identical.
4. **Step 4 — In-code CPU fusion prevention across both `POST_ZDF` and `POST_DH..EXIT` (`DECISION 158`):** prevent XLA CPU FMA and cross-statement algebraic simplification in code with `nemo_source_round` (`packages/core/legoesm/core/source_rounding.py:14-27`) in `packages/core/legoesm/timestepping/tridiagonal.py:137-180` and `packages/ice/legoesm/ice/bitz_lipscomb.py:150-198`, `:201-233`, `:236-298`, `:356-368`, `:371-413`, `:416-429`, `:462-515`, `:778-876`, `:1008-1028`, `:1031-1090`, `:1624-2246`, and `:2249-2367`, controlled by `_nemo_source_round: bool = True`, and prove that under compiled `jax.jit`, CPU production JIT (`XLA_FLAGS=""`) equals the `--xla_cpu_max_isa=AVX` diagnostic **bitwise across the entire year** (identical full-year SHA-256 output hash `389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf` and identical SHA-256 digests on all 8 stages).

### Provenance & Evidence Digests
* **Preregistration commit (before measurement):** `501a51a8a0a10d808f41838922db514db55f3800` (`docs/ice/fidelity/si3_r3_d5_preregister.md:1-54`)
* **Implementation & probe commits:** `0ad9f4b62ee8236d9b5086a31a928f8b0a372da2`, `0ff50e908343c599fb3fe7791144fcd356a49f84`, `8ea398af75cdb53adad8f1febeb79d89ab664e97`
* **Oracle record root:** `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a`
  * `oracle_si3_thd_frames.bin` SHA-256: `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b`
  * `oracle_si3_zdf_inputs.bin` SHA-256: `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd`
* **Committed probe:** `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:144-2072`
* **Committed unit & gradient tests:** `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:61-2776`
* **Evidence artifacts under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round3/` (`jax.jit` evaluation, `worktree.clean = true` at `8ea398af75cdb53adad8f1febeb79d89ab664e97`):**
  * `cpu_default_probe.json` (CPU production JIT, `XLA_FLAGS=""`) SHA-256: `3b500ddf849202a66143bc0cbfa6c4524830a4f00847793415fe7fb4acd36c69`
  * `cpu_nofma_probe.json` (CPU diagnostic JIT, `XLA_FLAGS="--xla_cpu_max_isa=AVX"`) SHA-256: `6a83b03454b6907995084dad93c574a89f757d6d9ea8c5c3ba32436b4ad9d9ca`
  * `gpu_probe.json` (GPU 1 spot-check via `CUDA_VISIBLE_DEVICES=1`) SHA-256: `4cd0025943a9f87301551c875f2451557f8c327fb18c84f41ae93f3a12838feb`
  * `gpu_nvidia_smi_before.txt` SHA-256: `4a9ff082832dc79dc3f7a7966bd6b7a8b9aec07cb1774976d39f991a1773a2cd`
  * `gpu_nvidia_smi_after.txt` SHA-256: `5ab893e6a1ca0c82cf972b4954b0c6f7a9055624659af4204cade8801b872662`

### Idle-GPU Spot-Check Provenance (`CUDA_VISIBLE_DEVICES=1`)
Before launching the GPU spot-check, `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` and `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` were recorded to `gpu_nvidia_smi_before.txt`, showing both GPU 0 (`0 MiB`, `0 %`) and GPU 1 (`0 MiB`, `0 %`) completely idle with zero active compute processes. The probe was pinned to GPU 1 via `CUDA_VISIBLE_DEVICES=1`. Immediately after completion, the same queries were recorded to `gpu_nvidia_smi_after.txt`, confirming both GPUs at `0 MiB` (`83 %` transient utilization sample on GPU 1 at process teardown, `0 MiB` used memory) with zero compute apps.

---

## 2. Step 1: Committed Reproduction of the Round 2 Reviewer's Numbers (`jax.jit` vs Eager)

`run_step1_reviewer_reproduction` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:404-699`) evaluates every Step 1 arm under compiled `jax.jit` (and we compare against the earlier eager numbers from `0ff50e908343c599fb3fe7791144fcd356a49f84` where `snow_temperature_from_enthalpy` and `_si3_zdf_bl99_step` were called without outer `jax.jit`):

1. **ZDF-entry snow temperature (`exact["t_s"]` in `oracle_si3_zdf_inputs.bin`) across all `7,385` snow-present steps (`h_s > 0.0`):**
   * **Round 2 as shipped** (`(e_s * 3.0) / v_s` + `src/ICE/icethd_dh.F90:498-503` formula, `_nemo_glo2eqv_snow_temp=False`, `_nemo_source_round=False`):
     * **Compiled `jax.jit`:** **`409` of `7,385` steps at `0` ULP** (`6,976` mismatching steps, `max_ulp = 3`) on CPU `XLA_FLAGS=""`, and **`398` of `7,385` steps at `0` ULP** (`6,987` mismatching steps, `max_ulp = 3`) on CPU `--xla_cpu_max_isa=AVX` and GPU (where XLA constant-folds `inverse_rho_snow * inverse_c_ice` before multiplying by `-e`).
     * **Eager (`0ff50e9083`):** **`1,811` of `7,385` steps at `0` ULP** (`5,574` mismatching steps, `max_ulp = 3`).
   * **Step 2 harness conversion only** (`(e_s / v_s) * 3.0` + `src/ICE/icethd_dh.F90:498-503` formula, `_nemo_glo2eqv_snow_temp=False`, `_nemo_source_round=False`):
     * **Compiled `jax.jit`:** **`287` of `7,385` steps at `0` ULP** (`7,098` mismatching steps, `max_ulp = 2`) on CPU `XLA_FLAGS=""`, and **`315` of `7,385` steps at `0` ULP** (`7,070` mismatching steps, `max_ulp = 2`) on CPU `--xla_cpu_max_isa=AVX` and GPU.
     * **Eager (`0ff50e9083`):** **`3,547` of `7,385` steps at `0` ULP** (`3,838` mismatching steps, `max_ulp = 1`).
   * **Step 2 + Step 3 model statement** (`(e_s / v_s) * 3.0` + `src/ICE/icevar.F90:412-413` formula, `_nemo_glo2eqv_snow_temp=True`, `_nemo_source_round=True`):
     * **Both compiled `jax.jit` and eager:** **`7,385` of `7,385` steps at `0` ULP** (`0` mismatching steps, `max_ulp = 0`) across all three platforms.
2. **ZDF-entry salinity profile (`frames[1]["sz_i"]`) across all `8,760` steps:**
   * **Round 2 as shipped** (`szv_i * 3.0 / v_i`): **`5,225` of `8,760` steps at `0` ULP** (`3,535` mismatching steps by `1` ULP).
   * **Step 2 harness conversion** (`_nemo_salprof(s_i, sss)` from `src/ICE/icevar.F90:488-506`): **`8,760` of `8,760` steps at `0` ULP** (`0` mismatching steps).
3. **Normal-step `POST_ZDF` solved fields (`t_su`, `e_i`, `e_s` across the `8,734` normal steps outside `kt = 5842..5867` = `26,202` rows):**

| Step 1 `POST_ZDF` evaluation arm (`26,202` rows = `8,734` normal steps × 3 fields) | Mode (`jax.jit` vs `eager`) | CPU production (`XLA_FLAGS=""`) Over-bar / `0` ULP / Nonzero ULP (`max_ulp`) | CPU diagnostic (`--xla_cpu_max_isa=AVX`) Over-bar / `0` ULP / Nonzero ULP (`max_ulp`) | GPU (`CUDA_VISIBLE_DEVICES=1`) Over-bar / `0` ULP / Nonzero ULP (`max_ulp`) |
|---|---|---:|---:|---:|
| **Round 2 as shipped** (`_nemo_entry_conversion=False`, `_nemo_glo2eqv_snow_temp=False`, `_nemo_source_round=False`) | **`jax.jit`**<br>*(`eager`)* | **`3,307` / `11,407` / `14,795` (`5,056`)**<br>*(`571` / `15,873` / `10,329` (`4,709`))* | **`3,153` / `13,990` / `12,212` (`4,709`)**<br>*(`470` / `20,054` / `6,148` (`4,709`))* | **`496` / `17,410` / `8,792` (`4,709`)**<br>*(`470` / `20,054` / `6,148` (`4,709`))* |
| **NEMO-exact ZDF inputs before Round 2 reciprocals** (`_nemo_zdf_layer_reciprocals=False`, `_nemo_source_round=False`) | **`jax.jit`**<br>*(`eager`)* | **`4,313` / `11,685` / `14,517` (`5,056`)**<br>*(`1,650` / `15,431` / `10,771` (`1,636`))* | **`4,149` / `13,347` / `12,855` (`3,358`)**<br>*(`1,625` / `16,292` / `9,910` (`1,636`))* | **`1,545` / `16,714` / `9,488` (`1,636`)**<br>*(`1,625` / `16,292` / `9,910` (`1,636`))* |
| **NEMO-exact ZDF inputs after Round 2 reciprocals, unrounded** (`_nemo_zdf_layer_reciprocals=True`, `_nemo_source_round=False`) | **`jax.jit`**<br>*(`eager`)* | **`3,120` / `13,739` / `12,463` (`5,056`)**<br>*(`130` / `19,171` / `7,031` (`330`))* | **`2,938` / `21,294` / `4,908` (`3,358`)**<br>*(`0` / `26,202` / `0` (`0`))* | **`24` / `26,154` / `48` (`147`)**<br>*(`0` / `26,202` / `0` (`0`))* |
| **NEMO-exact ZDF inputs with Step 4 `nemo_source_round`** (`_nemo_zdf_layer_reciprocals=True`, `_nemo_source_round=True`) | **`jax.jit` & `eager`** | **`0` / `26,202` / `0` (`0`)**<br>(`e_i`: `0`/`8,734`, `e_s`: `0`/`8,734`, `t_su`: `0`/`8,734`) | **`0` / `26,202` / `0` (`0`)**<br>(`e_i`: `0`/`8,734`, `e_s`: `0`/`8,734`, `t_su`: `0`/`8,734`) | **`0` / `26,202` / `0` (`0`)**<br>(`e_i`: `0`/`8,734`, `e_s`: `0`/`8,734`, `t_su`: `0`/`8,734`) |

This **CONFIRMS** Prediction **P1** (`docs/ice/fidelity/si3_r3_d5_preregister.md:40-42`) and shows why compiling the outer step under `jax.jit` matters: without outer `jax.jit` (`eager`), only the inner Picard `fori_loop` body compiled while pre-loop `ice_temperature_from_enthalpy` and post-loop `ice_enthalpy_from_temperature` ran eagerly (`0` over-bar on `--xla_cpu_max_isa=AVX` and GPU); under full `jax.jit`, XLA's `HloAlgebraicSimplifier` reassociates the unrounded pre/post-loop enthalpy conversions (`2,938` over-bar on `--xla_cpu_max_isa=AVX`, `24` on GPU), and Step 4's `nemo_source_round=True` closes all **`26,202 / 26,202` normal-step `POST_ZDF` rows to `0` ULP (`0` over-bar)** on all three platforms under `jax.jit`.

---

## 3. Step 2 (Harness), Step 3 (Model), and Step 4 (In-Code CPU Fusion Prevention)

### 3.1 Step 2: Complete Harness Entry Conversion & Option-2 Salinity Profile (`nemo_si3thd_phase2_gate.py`)
In NEMO's `ice_var_glo2eqv` and `ice_var_salprof` (`src/ICE/icevar.F90:354-360`, `src/ICE/icevar.F90:390`, `src/ICE/icevar.F90:412-413`, `src/ICE/icevar.F90:488-506`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:368-374`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:404`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:502-520`), global fields are converted to equivalent 1-D layer fields before `ice_thd`:
* `ze_i = e_i(ji,jj,jk,jl) / v_i(ji,jj,jl) * zlay_i` (`zlay_i = 3.0_wp`)
* `e_s(ji,jj,jk,jl) / v_s(ji,jj,jl) * zlay_s` (`zlay_s = 3.0_wp`)
* For `nn_icesal = 2` (C1D's salinity scheme), `s_i = sv_i / v_i` followed by `CALL ice_var_salprof` (`src/ICE/icevar.F90:354-360`, `src/ICE/icevar.F90:488-506`), whereas the `nn_icesal == 4` branch (`src/ICE/icevar.F90:362-379`) is not taken.
* `_nemo_salprof(s_i, sss)` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-249` (called by `_entry_arrays` at `:252-286` and `_exact_1d_entry` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:717-732`) transcribes the complete `src/ICE/icevar.F90:488-506` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:502-520`) option-2 salinity profile, including both SSS-dependent guards:
  1. `WHERE( s_i(:,:) >= 0.5_wp * rn_sinew * sss_m(:,:) )` (`zsi0 = 0.5_wp * rn_sinew * sss_m`, `zsi1 = rn_simax`, `z1_dS = 1._wp / ( zsi1 - zsi0 )`, linear interpolation `zargtemp = ( zsi1 - s_i ) * z1_dS` when `s_i < rn_simax`, `zarg = 1._wp` when `s_i >= rn_simax`, and `zarg = 0._wp` in `ELSEWHERE`).
  2. `IF( s_i(ji,jj,jl) <= REAL( nlay_i, wp ) * MIN( rn_simin , rn_sinew * sss_m(ji,jj) ) ) sz_i(ji,jj,jk,jl) = s_i(ji,jj,jl)` (constant bulk salinity across all `nlay_i` layers when `s_i <= 3.0 * min(rn_simin, rn_sinew * sss_m)`).
* Transcribing both SSS-dependent guards in `_nemo_salprof` also closes all `10` remaining subnormal-window (`kt = 5842..5867`) `POST_DH..EXIT` over-bar rows on GPU (`10 -> 0` over-bar in `kt = 5842..5867` on GPU) and reduces normal-step `POST_DH..EXIT` over-bar rows across the year from `568` to **`336`**.
* In `ENTRY` (`105,120` rows across `8,760` steps), `0`-ULP rows increase from `84,373` to **`85,157` (`+784` rows moved to `0` ULP)** with `0` over-bar rows.

### 3.2 Step 3: Solver-Entry Snow Temperature Statement (`bitz_lipscomb.py`)
At solver entry (`ice_var_glo2eqv`), NEMO computes snow layer temperature via `src/ICE/icevar.F90:412-413` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`):
* `t_s(ji,jj,jk,jl) = rt0 + MAX( -100._wp , MIN( r1_rcpi*( -r1_rhos*( e_s(ji,jj,jk,jl) / v_s(ji,jj,jl) * zlay_s ) + rLfus ) , 0._wp ) )`
* Previously, `snow_temperature_from_enthalpy` (`packages/ice/legoesm/ice/bitz_lipscomb.py:371-413`, called at `packages/ice/legoesm/ice/bitz_lipscomb.py:814-819`) evaluated the expanded `src/ICE/icethd_dh.F90:498-503` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:508-513`) form `constants.T0 + (-e * inverse_rho_snow * inverse_c_ice + constants.latent_fusion * inverse_c_ice)`.
* Because `r1_rcpi * (-r1_rhos * e_s + rLfus)` rounds `-r1_rhos * e_s + rLfus` before multiplying by `r1_rcpi`, the two parenthesizations differ by `1..2` ULPs on `7,070..7,098` of the `7,385` snow steps under `jax.jit` (`3,838` in eager mode; and at `kt = 1334`, that `1`-ULP entry `T_s` difference amplifies through the stiff clamped-snow tridiagonal solve into a `9`-ULP / `1.0269e-15 > 1e-15` over-bar error in `POST_ZDF.e_s`).
* Transcribing `src/ICE/icevar.F90:412-413` under `_nemo_glo2eqv_snow_temp: bool = True` (`packages/ice/legoesm/ice/bitz_lipscomb.py:371-413`) brings ZDF-entry `t_s` to **`7,385 / 7,385` steps at `0` ULP** under both `jax.jit` and eager execution.

### 3.3 Step 4: In-Code CPU Fusion Prevention Across `POST_ZDF` and `POST_DH..EXIT` via `nemo_source_round` (`DECISION 158`)
NEMO 5.0.2's C1D oracle binary (`nemo.exe`, compiled with gfortran `-O3` without `-march`, targeting baseline `x86-64` SSE2) contains **0 FMA instructions** and evaluates each parenthesized Fortran sub-expression with separate IEEE-754 binary64 roundings. By contrast, when `si3_column_step_arrays` is compiled under `jax.jit` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1836`), XLA performs two classes of transformations on `x86_64` CPU: (a) AVX2/AVX-512 fused multiply-add (`vfmasd`) contraction of `a * b + c` / `a * b - c`, and (b) target-independent `HloAlgebraicSimplifier` reassociation (e.g., factoring `2.0 * a + 2.0 * b -> (a + b) * 2.0` or reassociating linear combinations). Under **DECISION 158**, every such statement across both the `POST_ZDF` BL99 solver (items 1–7) and the `POST_DH..EXIT` thermodynamics chain (items 8–12) is guarded in code with `nemo_source_round` (`packages/core/legoesm/core/source_rounding.py:14-27`), gated by `_nemo_source_round: bool = True`:

1. **Thomas unnormalised backward substitution (`packages/core/legoesm/timestepping/tridiagonal.py:137-180`):**
   * NEMO source: `scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:571-595` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:544-567`), where `t_i_1d(ji,jk) = ( zindtbis(jm) - ztrid(jm,3) * t_i_1d(ji,jk+1) ) / zdiagbis(jm)` and `t_s_1d(ji,jk) = ( zindtbis(jm) - ztrid(jm,3) * t_s_1d(ji,jk+1) ) / zdiagbis(jm)` evaluate the product `ztrid(jm,3) * t(...)` before subtracting from `zindtbis(jm)`.
   * In `unnormalised_backward` (`packages/core/legoesm/timestepping/tridiagonal.py:170-175`), `c[..., k] * current[..., k + 1]` is wrapped in `nemo_source_round` for `operation_order == "nemo_unnormalised"` (while `"nemo_unnormalised_unrounded"` preserves the unguarded path for `_nemo_source_round=False` ablation).
2. **Ice temperature from enthalpy (`packages/ice/legoesm/ice/bitz_lipscomb.py:236-298`):**
   * NEMO source: `src/ICE/icevar.F90:390-395` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:404-409`), where `ztmelts = - sz_i * rTmlt`, `zbbb = ( rcp - rcpi ) * ztmelts + ze_i * r1_rhoi - rLfus`, `zccc = SQRT( MAX( zbbb * zbbb - 4._wp * rcpi * rLfus * ztmelts , 0._wp) )`, and `t_i = MAX( -100._wp , MIN( -( zbbb + zccc ) * 0.5_wp * r1_rcpi , ztmelts ) ) + rt0` are evaluated without FMA across the products in `zbbb`, `zccc`, and `t_i`.
3. **Solver-entry snow temperature from enthalpy (`packages/ice/legoesm/ice/bitz_lipscomb.py:371-413`):**
   * NEMO source: `src/ICE/icevar.F90:412-413` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`), where `-r1_rhos * e_s` is rounded before `+ rLfus` and before `* r1_rcpi`.
4. **Ice & snow enthalpy from temperature (`packages/ice/legoesm/ice/bitz_lipscomb.py:201-233`, `packages/ice/legoesm/ice/bitz_lipscomb.py:356-368`):**
   * NEMO source: `src/ICE/icevar.F90:938-953` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:952-967`), called at `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:813`, where `e_i_1d` and `e_s_1d = rhos * ( rcpi * ( rt0 - t_s_1d(ji,jk) ) + rLfus )` round each parenthesized product/difference separately.
5. **Pringle et al. (2007) thermal conductivity (`packages/ice/legoesm/ice/bitz_lipscomb.py:416-429`, `packages/ice/legoesm/ice/bitz_lipscomb.py:600-611`):**
   * NEMO source: `scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:276-290` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:271-285`), where `0.5_wp * (sz_i_1d(ji,jk) + sz_i_1d(ji,jk+1))`, `0.5_wp * (t_i_1d(ji,jk) + t_i_1d(ji,jk+1)) - rt0`, `rcnd_i + 0.09_wp * ... / MIN(...)`, and `- 0.011_wp * (...)` are rounded without FMA.
6. **Solar radiation transmission (`packages/ice/legoesm/ice/bitz_lipscomb.py:462-515`):**
   * NEMO source: `scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:217-230` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:219-232`), where `zh_s * REAL(jk,wp)`, `zh_i * REAL(jk,wp)`, and the two canopy/open-sky transmission terms in `zradtr_i` are rounded before addition/subtraction.
7. **BL99 kappa interface, specific heat `zcpi`, surface flux `qns_ice_1d`, tridiagonal matrix assembly, melting-point clip, and post-solve `qcn_ice_top_1d` (`packages/ice/legoesm/ice/bitz_lipscomb.py:621-856`):**
   * NEMO source: `scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:334-335`, `:365-366`, `:399-529`, `:620`, `:821-823` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:329-330`, `:353-354`, `:387-512`, `:588`, `:782-784`). Wrapping `ztcond_i(ji,0) * zh_s(ji)` and `rcnd_s * zh_i(ji)` in `zkappa_s(ji,nlay_s)`, the quotient `zgamma * sz_i / MAX((t_i - rt0) * (ztiold - rt0), epsi20)` added to `rcpi` in `zcpi` (the denominator product `(t_i - rt0) * (ztiold - rt0)` is not an FMA pattern and is not wrapped), `dqns * (t_su - ztsub)` in `qns_ice_1d`, the `zeta * zkappa` products and `2.0 * k` factors in `ztrid`/`zindterm` (preventing `HloAlgebraicSimplifier` from refactoring `1.0 + eta * (k_1 + 2.0 * k_0)`), `-rTmlt * sz_i + rt0`, and the two terms of `qcn_ice_top_1d` eliminates all `12,463` FMA/simplifier-induced nonzero-ULP normal-step `POST_ZDF` rows (`3,120` over-bar rows) under CPU production `jax.jit`.
8. **Option-2 post-salinity linear profile `option2_salinity_profile` (`packages/ice/legoesm/ice/bitz_lipscomb.py:150-198`, called at `:2321-2326`):**
   * NEMO source: `src/ICE/icevar.F90:575-576` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:589-590`), where `sz_i_1d(ji,jk) = zalpha * s_i_1d(ji) * 2._wp * ( REAL(jk,wp) - 0.5_wp ) * r1_nlay_i + ( 1._wp - zalpha ) * s_i_1d(ji)` evaluates the linear profile weight product before adding `1.0 - alpha`. Wrapping `alpha[..., None] * z` in `nemo_source_round` prevents XLA CPU FMA in `POST_SAL.sz_i`, `POST_TEMP2.sz_i`, `POST_DO.szv_i`, and `EXIT.szv_i` (e.g., at `kt = 3961`).
9. **Piecewise vertical enthalpy/salt remapping `_piecewise_remap` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1008-1028`, called at `:2214-2230`):**
   * NEMO source: `src/ICE/icevar.F90:1019-1023` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:1033-1037`, called from `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:486`, `:492`), where cumulative layer enthalpy/salt integrals accumulate `overlap * enthalpy` products before summing. Wrapping `overlap * enthalpy[..., None, :]` in `nemo_source_round` prevents XLA CPU from fusing the 5-term dot product in `jnp.sum` into a chain of `vfmasd` instructions across `POST_DH..EXIT.e_i` (e.g., at `kt = 2899`).
10. **Snow enthalpy cumulative remapping `_nemo_snow_enthalpy_remap` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1031-1090`, called at `:2224-2230`):**
    * NEMO source: `src/ICE/icethd_dh.F90:577` and `src/ICE/icethd_dh.F90:598-600` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:587` and `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:608-610`), where `zeh_cum0(jk0) = zeh_cum0(jk0-1) + pe_old(jk0-1) * ph_old(jk0-1)` and the linear interpolation numerator `zeh_cum0(jk0-1) * ( zh_cum0(jk0) - zh_cum1(jk1) ) + zeh_cum0(jk0) * ( zh_cum1(jk1) - zh_cum0(jk0-1) )` evaluate each product before addition. Wrapping `enthalpy[..., layer] * thickness[..., layer]` and the two interpolation products in `nemo_source_round` prevents XLA CPU FMA in `POST_DH..EXIT.e_s` (e.g., at `kt = 2899`).
11. **Surface snow melt available-heat update in `_dh_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1795-1797`):**
    * NEMO source: `src/ICE/icethd_dh.F90:219` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:229`), where `zq_top = MAX( 0._wp , zq_top + zdum * ze_s(jk) )` evaluates `zdum * ze_s(jk)` before adding to `zq_top`. Wrapping `delta * eseg_s[..., k]` in `nemo_source_round` prevents XLA CPU FMA during snow surface melt (e.g., at `kt = 4237`).
12. **Surface & basal ice melt specific enthalpy difference `zdE = zEi - zEw` in `_dh_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1819-1828`, `:1925-1931`, `:1968-1977`):**
    * NEMO source: `src/ICE/icethd_dh.F90:254-256` and `src/ICE/icethd_dh.F90:391-393` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:264-266` and `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:401-403`), where NEMO writes `zEi = - e_i_1d(ji,jk) * r1_rhoi`, `zEw = rcp * ztmelts`, and `zdE = zEi - zEw`, whereas Round 3's `_dh_step` wrote `ice_specific = round_fn(-zdf.e_ice[..., k] / constants.rho_ice)` (dividing by `constants.rho_ice` instead of multiplying by `r1_rhoi = 1.0 / constants.rho_ice`; transcribed to `-zdf.e_ice[..., k] * r1_rhoi` in Round 5). Wrapping `ice_specific` and `water_specific` in `nemo_source_round` prevents XLA CPU from fusing `water_specific` into `ice_specific - water_specific` (e.g., at `kt = 4237` and `kt = 4738`).

---

## 4. Full-Year (`8,760`-Step, `621,960`-Row) C1D Rescore Under `jax.jit` (vs Eager), Bitwise CPU Proof, and GPU Step 3 vs Step 4 Bitwise Comparison

`run_year_rescore` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883`) evaluates all `621,960` stage-field rows across the 8 stages (`ENTRY`, `POST_ZDF`, `POST_DH`, `POST_TEMP1`, `POST_SAL`, `POST_TEMP2`, `POST_DO`, `EXIT`) under **compiled `jax.jit`** (`execution_mode = "jax.jit"`, matching `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1836`) for each cumulative step and each isolated single-step arm from the Round 2 baseline, recording both aggregate counts and the SHA-256 digest of all stage-field float64 outputs (`full_year_output_sha256` and `stage_output_sha256`).

### 4.1 Summary of Over-Bar Rows, `0`-ULP Rows, and Full-Year Output SHA-256 Across All Arms (**Compiled `jax.jit`**, with Earlier **Eager** Counts Labeled)

| Evaluation arm | Mode (`jax.jit` vs `eager`) | CPU Production (`XLA_FLAGS=""`) Over-bar / `0` ULP / Nonzero ULP (Normal `POST_ZDF` 3-field over-bar / `0` ULP; Full-Year SHA-256 prefix) | CPU Diagnostic (`--xla_cpu_max_isa=AVX`) Over-bar / `0` ULP / Nonzero ULP (Normal `POST_ZDF` 3-field over-bar / `0` ULP; Full-Year SHA-256 prefix) | GPU Spot-Check (`CUDA_VISIBLE_DEVICES=1`) Over-bar / `0` ULP / Nonzero ULP (Normal `POST_ZDF` 3-field over-bar / `0` ULP; Full-Year SHA-256 prefix) |
|---|---|---:|---:|---:|
| **Pre-Round-2 (`r2_pre_reciprocals`)** | **`jax.jit`**<br>*(`eager`)* | **`50,243` / `418,872` / `203,088`**<br>(`4,506` / `8,653`; `d95b3e0fe7d2e4ed`)<br>*(`35,282` / `442,662` / `179,298`; `2,015`/`13,419`)* | **`49,621` / `428,346` / `193,614`**<br>(`4,412` / `11,499`; `7d78b10313b35512`)<br>*(`35,377` / `447,311` / `174,649`; `2,012`/`14,134`)* | **`35,091` / `435,254` / `186,706`**<br>(`1,982` / `14,028`; `a2ba07f07555759b`)<br>*(`35,028` / `447,427` / `174,533`; `2,012`/`14,134`)* |
| **Round 2 baseline (`r2_baseline`, before Step 2)** | **`jax.jit`**<br>*(`eager`)* | **`26,728` / `430,260` / `191,700`**<br>(`3,305` / `9,643`; `3f734303337eaff8`)<br>*(`6,584` / `464,502` / `157,458`; `571`/`15,873`)* | **`24,984` / `451,321` / `170,639`**<br>(`3,153` / `13,990`; `b2e1f7fea11f3e71`)<br>*(`5,107` / `491,635` / `130,325`; `470`/`20,054`)* | **`4,961` / `462,513` / `159,447`**<br>(`496` / `17,410`; `c608b723ea4e2073`)<br>*(`4,495` / `491,949` / `130,011`; `470`/`20,054`)* |
| **After Step 2 (`step2_harness_only`, cumulative & isolated)** | **`jax.jit`**<br>*(`eager`)* | **`25,446` / `441,771` / `180,189`**<br>(`3,139` / `9,911`; `25a67a5dad87fe40`)<br>*(`3,161` / `482,471` / `139,489`; `134`/`18,199`)* | **`23,320` / `463,619` / `158,341`**<br>(`2,943` / `14,837`; `db98c7f196a44b44`)<br>*(`1,416` / `517,233` / `104,727`; `1`/`23,717`)* | **`1,354` / `476,845` / `145,115`**<br>(`30` / `19,258`; `342fe312ac9b0967`)<br>*(`797` / `517,544` / `104,416`; `1`/`23,717`)* |
| **After Step 2 + Step 3 (`step2_plus_step3`, cumulative)** | **`jax.jit`**<br>*(`eager`)* | **`24,650` / `450,323` / `171,637`**<br>(`3,117` / `11,569`; `a10301c62922e5f3`)<br>*(`2,931` / `488,873` / `133,087`; `130`/`19,171`)* | **`22,503` / `506,156` / `115,804`**<br>(`2,938` / `21,294`; `49557319d46eddcd`)<br>*(`1,197` / `532,390` / `89,570`; `0`/`26,202`)* | **`516` / `521,577` / `100,383`**<br>(`24` / `26,154`; `1724818b5baadeba`)<br>*(`578` / `532,701` / `89,259`; `0`/`26,202`)* |
| **After Step 2 + Step 3 + Step 4 (`step2_plus_step3_plus_step4`, final)** | **`jax.jit`**<br>*(`eager`)* | **`965` / `521,368` / `100,592`**<br>(**`0` / `26,202`; `389490e20dbd86cb`**)<br>*(`1,197` / `532,390` / `89,570`; `0`/`26,202`)* | **`965` / `521,368` / `100,592`**<br>(**`0` / `26,202`; `389490e20dbd86cb`**)<br>*(`1,197` / `532,390` / `89,570`; `0`/`26,202`)* | **`336` / `521,679` / `100,281`**<br>(**`0` / `26,202`; `46b20957b001c3b7`**)<br>*(`578` / `532,701` / `89,259`; `0`/`26,202`)* |
| *Isolated Step 3 only (`step3_model_only`, from `r2_baseline`)* | **`jax.jit`**<br>*(`eager`)* | `25,995` / `436,288` / `185,672`<br>(`3,282` / `10,735`; `5156c70a920d019f`)<br>*(`6,407` / `467,851` / `154,109`; `568`/`16,376`)* | `24,254` / `476,216` / `145,744`<br>(`3,151` / `17,569`; `11275bd2f2d36950`)<br>*(`4,962` / `499,354` / `122,606`; `471`/`21,259`)* | `4,196` / `488,536` / `133,424`<br>(`489` / `21,231`; `e59ff01ed2fd6284`)<br>*(`4,350` / `499,668` / `122,292`; `471`/`21,259`)* |
| *Isolated Step 4 only (`step4_source_round_only`, from `r2_baseline`)* | **`jax.jit`**<br>*(`eager`)* | `5,373` / `462,482` / `159,478`<br>(`476` / `17,497`; `e66f5493064bf4fe`)<br>*(`5,107` / `491,635` / `130,325`; `470`/`20,054`)* | `5,439` / `462,231` / `159,729`<br>(`478` / `17,436`; `84d49ac319ab103c`)<br>*(`5,107` / `491,635` / `130,325`; `470`/`20,054`)* | `4,817` / `462,545` / `159,415`<br>(`478` / `17,436`; `73a60209a9be902b`)<br>*(`4,495` / `491,949` / `130,011`; `470`/`20,054`)* |

### 4.2 Stage-by-Stage Progression Under Compiled `jax.jit` on CPU Production JIT (`XLA_FLAGS=""`, `cpu_default_probe.json`)

| Stage (`rows`) | `r2_pre_reciprocals` Over-bar / `0` ULP (`jax.jit`) | `r2_baseline` Over-bar / `0` ULP (`jax.jit`) | After Step 2 (`step2_harness_only`) Over-bar / `0` ULP (`jax.jit`) | After Step 2+3 (`step2_plus_step3`) Over-bar / `0` ULP (`jax.jit`) | After Step 2+3+4 (`step2_plus_step3_plus_step4`) Over-bar / `0` ULP (`jax.jit`) | Net Over-bar Delta (`r2_baseline -> final`) | Net `0`-ULP Delta (`r2_baseline -> final`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ENTRY` (`105,120`) | `0` / `84,373` | `0` / `84,373` | `0` / `85,157` | `0` / `85,157` | `0` / `85,157` | `0` | **`+784`** |
| `POST_ZDF` (`61,320`) | `4,581` / `40,161` | `3,380` / `41,151` | `3,214` / `44,954` | `3,192` / `46,612` | **`75` / `61,245`** | **`-3,305`** | **`+20,094`** |
| `POST_DH` (`61,320`) | `6,652` / `35,204` | `3,728` / `36,629` | `3,542` / `40,144` | `3,430` / `41,291` | **`106` / `50,209`** | **`-3,622`** | **`+13,580`** |
| `POST_TEMP1` (`61,320`) | `6,652` / `35,204` | `3,728` / `36,629` | `3,542` / `40,144` | `3,430` / `41,291` | **`106` / `50,209`** | **`-3,622`** | **`+13,580`** |
| `POST_SAL` (`61,320`) | `6,711` / `36,204` | `3,779` / `37,644` | `3,599` / `37,614` | `3,487` / `38,761` | **`106` / `48,281`** | **`-3,673`** | **`+10,637`** |
| `POST_TEMP2` (`61,320`) | `6,711` / `36,204` | `3,779` / `37,644` | `3,599` / `37,614` | `3,487` / `38,761` | **`106` / `48,281`** | **`-3,673`** | **`+10,637`** |
| `POST_DO` (`105,120`) | `9,468` / `75,761` | `4,167` / `78,095` | `3,975` / `78,072` | `3,812` / `79,225` | **`233` / `88,993`** | **`-3,934`** | **`+10,898`** |
| `EXIT` (`105,120`) | `9,468` / `75,761` | `4,167` / `78,095` | `3,975` / `78,072` | `3,812` / `79,225` | **`233` / `88,993`** | **`-3,934`** | **`+10,898`** |
| **Total (`621,960`, `jax.jit`)** | **`50,243` / `418,872`** | **`26,728` / `430,260`** | **`25,446` / `441,771`** | **`24,650` / `450,323`** | **`965` / `521,368`** | **`-25,763`** | **`+91,108`** |

On CPU production `jax.jit` (`145,857` touched rows + `476,103` untouched rows == `621,960` across `8,735` steps), `25,797` rows closed to `AT-BAR` and `34` rows opened to `DEBT` (net `-25,763`), while `95,400` rows moved to `0` ULP vs oracle and `4,292` rows moved from `0` ULP (`3,107` of which are `ENTRY.sz_i` rows where the stored global `ENTRY` frame holds `sz_i = szv_i * 3.0 / v_i`, while `ice_var_glo2eqv` overwrites `sz_i` with `ice_var_salprof` before `POST_ZDF`, where `0` rows moved away from `0` ULP and `20,094` rows moved to `0` ULP).

### 4.3 Stage-by-Stage Progression Under Compiled `jax.jit` on CPU Diagnostic (`--xla_cpu_max_isa=AVX`) and GPU (`CUDA_VISIBLE_DEVICES=1`)

| Stage (`rows`) | CPU `--xla_cpu_max_isa=AVX` (`jax.jit`) `r2_baseline -> Step 2 -> Step 2+3 -> Step 2+3+4` Over-bar (`0` ULP) | GPU `CUDA_VISIBLE_DEVICES=1` (`jax.jit`) `r2_baseline -> Step 2 -> Step 2+3 -> Step 2+3+4` Over-bar (`0` ULP) |
|---|---:|---:|
| `ENTRY` (`105,120`) | `0 -> 0 -> 0 -> 0`<br>(`84,373 -> 85,157 -> 85,157 -> 85,157`) | `0 -> 0 -> 0 -> 0`<br>(`84,373 -> 85,157 -> 85,157 -> 85,157`) |
| `POST_ZDF` (`61,320`) | `3,228 -> 3,018 -> 3,013 -> 75`<br>(`45,498 -> 49,880 -> 56,337 -> 61,245`) | **`497 -> 30 -> 24 -> 0`**<br>(**`48,988 -> 54,376 -> 61,272 -> 61,320`**) |
| `POST_DH` (`61,320`) | `3,489 -> 3,242 -> 3,136 -> 106`<br>(`39,417 -> 42,962 -> 48,983 -> 50,209`) | `676 -> 164 -> 55 -> 31`<br>(`40,249 -> 43,924 -> 50,238 -> 50,243`) |
| `POST_TEMP1` (`61,320`) | `3,489 -> 3,242 -> 3,136 -> 106`<br>(`39,417 -> 42,962 -> 48,983 -> 50,209`) | `676 -> 164 -> 55 -> 31`<br>(`40,249 -> 43,924 -> 50,238 -> 50,243`) |
| `POST_SAL` (`61,320`) | `3,540 -> 3,302 -> 3,196 -> 106`<br>(`40,425 -> 40,434 -> 46,455 -> 48,281`) | `680 -> 165 -> 56 -> 31`<br>(`41,801 -> 41,991 -> 48,305 -> 48,315`) |
| `POST_TEMP2` (`61,320`) | `3,540 -> 3,302 -> 3,196 -> 106`<br>(`40,425 -> 40,434 -> 46,455 -> 48,281`) | `680 -> 165 -> 56 -> 31`<br>(`41,801 -> 41,991 -> 48,305 -> 48,315`) |
| `POST_DO` (`105,120`) | `3,849 -> 3,607 -> 3,413 -> 233`<br>(`80,883 -> 80,895 -> 86,893 -> 88,993`) | `876 -> 333 -> 135 -> 106`<br>(`82,526 -> 82,741 -> 89,031 -> 89,043`) |
| `EXIT` (`105,120`) | `3,849 -> 3,607 -> 3,413 -> 233`<br>(`80,883 -> 80,895 -> 86,893 -> 88,993`) | `876 -> 333 -> 135 -> 106`<br>(`82,526 -> 82,741 -> 89,031 -> 89,043`) |
| **Total (`621,960`, `jax.jit`)** | **`24,984 -> 23,320 -> 22,503 -> 965`**<br>(**`451,321 -> 463,619 -> 506,156 -> 521,368`**) | **`4,961 -> 1,354 -> 516 -> 336`**<br>(**`462,513 -> 476,845 -> 521,577 -> 521,679`**) |

### 4.4 Bitwise Full-Year SHA-256 Proof: CPU Production JIT (`XLA_FLAGS=""`) == CPU Diagnostic JIT (`--xla_cpu_max_isa=AVX`)
Before inserting the 5 `POST_DH..EXIT` `nemo_source_round` guards (items 8–12 of Section 3.3), `ENTRY` and `POST_ZDF` matched under `jax.jit` (`POST_ZDF` `26,202 / 26,202` at `0` ULP), but XLA CPU fused multiply-adds inside `POST_DH..EXIT` caused CPU production `jax.jit` (`963` over-bar / `499,840` at `0` ULP; `EXIT` `0`-ULP `85,461`) to diverge from `--xla_cpu_max_isa=AVX` (`965` over-bar / `521,368` at `0` ULP; `EXIT` `0`-ULP `88,993`). With items 8–12 in place, `step2_plus_step3_plus_step4` on CPU production `jax.jit` (`XLA_FLAGS=""`) and on CPU diagnostic `jax.jit` (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`) are **100 % bit-for-bit identical across all `8,760` steps (`621,960` stage-field rows)**:
* **Full-year output SHA-256 (`step2_plus_step3_plus_step4`):**
  * CPU production `jax.jit` (`XLA_FLAGS=""`): `389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf`
  * CPU diagnostic `jax.jit` (`--xla_cpu_max_isa=AVX`): `389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf`
* **Per-stage output SHA-256 (`step2_plus_step3_plus_step4`, identical on both):**
  * `ENTRY`: `c80c6970402c109a9ee7f28d3767dbc9c12d2d0d9a3335bc0e59e2cfdfa713ff` (`0` over-bar, `85,157` at `0` ULP)
  * `POST_ZDF`: `7dbcf2bfe80eacd552cd500e6f8e813c14635e4c9c1b78a9123d90cc55abfdd5` (`75` over-bar, `61,245` at `0` ULP)
  * `POST_DH`: `073910c8f1e9e5d8a262f23a16ae2cdfd4acfbec6a8478094c1074180534d85b` (`106` over-bar, `50,209` at `0` ULP)
  * `POST_TEMP1`: `073910c8f1e9e5d8a262f23a16ae2cdfd4acfbec6a8478094c1074180534d85b` (`106` over-bar, `50,209` at `0` ULP)
  * `POST_SAL`: `b8a4161873f6e0a3794be6e6e286a4064142ab4eb3938e9bbd639fac6edf09a4` (`106` over-bar, `48,281` at `0` ULP)
  * `POST_TEMP2`: `b8a4161873f6e0a3794be6e6e286a4064142ab4eb3938e9bbd639fac6edf09a4` (`106` over-bar, `48,281` at `0` ULP)
  * `POST_DO`: `4fcaff4f10f5792b24e72c5a709af9f899b25523e89b98b5434f14b4ae3bc8de` (`233` over-bar, `88,993` at `0` ULP)
  * `EXIT`: `1e2aab5dfcb8e879cb4e2c0e04ff1405020d8bdcca56cc78100d2d0de5689c5d` (`233` over-bar, `88,993` at `0` ULP)

### 4.5 Row-by-Row and Full-Output Hash Comparison Between Step 3 (`_nemo_source_round=False`) and Step 4 (`_nemo_source_round=True`) Across All Platforms

`step3_vs_step4_bitwise_comparison` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:1745-1758` compares `step2_plus_step3` (`_nemo_source_round=False`) against `step2_plus_step3_plus_step4` (`_nemo_source_round=True`) row by row (`uint64` bit equality and ULP distance) and by full-year SHA-256 hash across all `621,960` stage-field rows under `jax.jit`:

| Platform (`jax.jit`) | Step 3 (`_nemo_source_round=False`) Full-Year SHA-256 | Step 4 (`_nemo_source_round=True`) Full-Year SHA-256 | Bitwise Identical? | Changed / Unchanged Rows (`621,960` total) | Changed Rows by Stage (`ENTRY` / `POST_ZDF` / `POST_DH` / `POST_TEMP1` / `POST_SAL` / `POST_TEMP2` / `POST_DO` / `EXIT`) | Max ULP Diff (Step 4 vs Step 3) | Over-Bar Change (`Step 3 -> Step 4`) | `0`-ULP Change (`Step 3 -> Step 4`) |
|---|---|---|---:|---:|---|---:|---:|---:|
| **CPU Production (`XLA_FLAGS=""`)** | `a10301c62922e5f3...` | `389490e20dbd86cb...` | `False` | `112,799` / `509,161` | `0` / `14,663` / `15,660` / `15,660` / `16,406` / `16,406` / `17,002` / `17,002` | `102,609` | `24,650 -> 965` (`-23,685`) | `450,323 -> 521,368` (`+71,045`) |
| **CPU Diagnostic (`--xla_cpu_max_isa=AVX`)** | `49557319d46eddcd...` | `389490e20dbd86cb...` | `False` | `40,729` / `581,231` | `0` / `4,933` / `5,215` / `5,215` / `5,918` / `5,918` / `6,765` / `6,765` | `16,927` | `22,503 -> 965` (`-21,538`) | `506,156 -> 521,368` (`+15,212`) |
| **GPU (`CUDA_VISIBLE_DEVICES=1`)** | `1724818b5baadeba...` | `46b20957b001c3b7...` | `False` | `368` / `621,592` | `0` / `48` / `48` / `48` / `53` / `53` / `59` / `59` | `148` | `516 -> 336` (`-180`) | `521,577 -> 521,679` (`+102`) |

* **Why GPU is not bitwise invariant between Step 3 and Step 4 under `jax.jit` (and instead improves in Step 4):**
  * In Round 3's earlier **eager** run, `step2_plus_step3` and `step2_plus_step3_plus_step4` had the same aggregate counts (`578` over-bar / `532,701` at `0` ULP) only because `si3_column_step_arrays` was called without outer `jax.jit`.
  * Under **compiled `jax.jit`** on GPU (`CUDA_VISIBLE_DEVICES=1`), XLA's `HloAlgebraicSimplifier` and `nvptx` fusion altered `48` `POST_ZDF` rows (`45` in `e_i`, `3` in `e_s`; `24` over-bar rows, `max_ulp = 147`) and `320` downstream `POST_DH..EXIT` rows when `_nemo_source_round=False` (`step2_plus_step3`: `516` total over-bar rows, `521,577` at `0` ULP).
  * Enabling Step 4 (`_nemo_source_round=True`) changes those `368` rows (`621,592` rows remain bit-for-bit identical) and **closes all `48` nonzero-ULP `POST_ZDF` rows (`24` over-bar rows) to `0` ULP**, bringing GPU `POST_ZDF` to **`0` over-bar rows and `61,320 / 61,320` rows at `0` ULP** across the entire 8,760-step year (including all 26 subnormal steps `kt = 5842..5867`!) and reducing total GPU over-bar rows from `516` to **`336`**.
  * Moreover, outside the 26 subnormal steps (`8,734` normal steps = `620,114` rows), CPU production `jax.jit` (`965 - 629 = 336` over-bar, `521,368 - 975 = 520,393` at `0` ULP), CPU `--xla_cpu_max_isa=AVX` (`965 - 629 = 336` over-bar, `520,393` at `0` ULP), and GPU `jax.jit` (`336 - 0 = 336` over-bar, `521,679 - 1,286 = 520,393` at `0` ULP) all agree on the exact same **`336` normal-step over-bar rows and `520,393` normal-step `0`-ULP rows**.

---

## 5. Remaining Owner Ranking After Step 4 (Compiled `jax.jit`)

With `POST_ZDF` completely closed to `0` ULPs (`61,138 / 61,138` normal-step `POST_ZDF` rows at `0` ULP on CPU, and `61,320 / 61,320` all-step `POST_ZDF` rows at `0` ULP on GPU, plus `0` over-bar rows in the 26 subnormal steps `kt = 5842..5867` on GPU), the remaining over-bar rows in `step2_plus_step3_plus_step4` under `jax.jit` (**`965`** on CPU = `629` floor + `336` normal; **`336`** on GPU = `0` subnormal + `336` normal) rank as follows (`cpu_default_probe.json`, `cpu_nofma_probe.json`, `gpu_probe.json`):

| Rank | Owner | CPU Over-bar Rows (`cpu_default` & `cpu_avx`, `jax.jit`) | GPU Over-bar Rows (`CUDA_VISIBLE_DEVICES=1`, `jax.jit`) | Nonzero-ULP Rows (CPU / GPU) | Max Normalised Error | Worst Row (`kt`, Stage, Field, ULP) |
|---:|---|---:|---:|---:|---:|---|
| 1 | `D5_PLATFORM_FLOOR_KT5842_5867_SUBNORMAL_SNOW` (`kt = 5842..5867`, 26 subnormal steps) | **`629`** | **`0`** (`AT-BAR`) | `871` / `560` | CPU: `0.02510099530281747`<br>GPU: `9.515654828503087e-16` (`AT-BAR`) | CPU: `kt = 5860 POST_ZDF.e_s` (`190,081,022,848,368` ULP, `D5_PLATFORM_FLOOR`)<br>GPU: `kt = 5844 POST_DO.e_i` (`6` ULP, `norm_err = 9.516e-16 <= 1e-15`) |
| 2 | `POST_DO.e_i` (normal steps, threshold-edge `<= 1.756e-15`) | **`99`** | **`99`** | `8,216` / `8,216` | `1.75568772769754e-15` | `kt = 4413 POST_DO.e_i` (`10` ULP, `max_ulp = 11`) |
| 3 | `EXIT.e_i` (same as `POST_DO.e_i`) | **`99`** | **`99`** | `8,216` / `8,216` | `1.75568772769754e-15` | `kt = 4413 EXIT.e_i` (`10` ULP, `max_ulp = 11`) |
| 4 | `POST_DH.e_i` (normal steps, threshold-edge `<= 1.488e-15`) | **`29`** | **`29`** | `8,269` / `8,269` | `1.487975650337286e-15` | `kt = 4413 POST_DH.e_i` (`7` ULP, `max_ulp = 11`) |
| 5 | `POST_TEMP1.e_i` (same as `POST_DH.e_i`) | **`29`** | **`29`** | `8,269` / `8,269` | `1.487975650337286e-15` | `kt = 4413 POST_TEMP1.e_i` (`7` ULP, `max_ulp = 11`) |
| 6 | `POST_SAL.e_i` (same as `POST_DH.e_i`) | **`29`** | **`29`** | `8,269` / `8,269` | `1.487975650337286e-15` | `kt = 4413 POST_SAL.e_i` (`7` ULP, `max_ulp = 11`) |
| 7 | `POST_TEMP2.e_i` (same as `POST_DH.e_i`) | **`29`** | **`29`** | `8,269` / `8,269` | `1.487975650337286e-15` | `kt = 4413 POST_TEMP2.e_i` (`7` ULP, `max_ulp = 11`) |
| 8 | `POST_DO.e_s` (normal steps, threshold-edge `<= 2.123e-15`) | **`7`** | **`7`** | `147` / `147` | `2.1226890857966868e-15` | `kt = 5460 POST_DO.e_s` (`12` ULP, `max_ulp = 16`) |
| 9 | `EXIT.e_s` (same as `POST_DO.e_s`) | **`7`** | **`7`** | `147` / `147` | `2.1226890857966868e-15` | `kt = 5460 EXIT.e_s` (`12` ULP, `max_ulp = 16`) |
| 10 | `POST_DH.e_s` (normal steps, threshold-edge `<= 1.615e-15`) | **`2`** | **`2`** | `137` / `137` | `1.6145373645668299e-15` | `kt = 5460 POST_DH.e_s` (`12` ULP) |
| 11 | `POST_TEMP1.e_s` (same as `POST_DH.e_s`) | **`2`** | **`2`** | `137` / `137` | `1.6145373645668299e-15` | `kt = 5460 POST_TEMP1.e_s` (`12` ULP) |
| 12 | `POST_SAL.e_s` (same as `POST_DH.e_s`) | **`2`** | **`2`** | `137` / `137` | `1.6145373645668299e-15` | `kt = 5460 POST_SAL.e_s` (`12` ULP) |
| 13 | `POST_TEMP2.e_s` (same as `POST_DH.e_s`) | **`2`** | **`2`** | `137` / `137` | `1.6145373645668299e-15` | `kt = 5460 POST_TEMP2.e_s` (`12` ULP) |
| — | **All other `POST_ZDF..EXIT` fields outside `kt = 5842..5867`** | **`0`** | **`0`** | — | `<= 6.66e-16` (`AT-BAR`) | — |
| **Total** | **All `621,960` stage-field rows (`jax.jit`)** | **`965`** (`629` floor + `336` normal) | **`336`** (`0` subnormal + `336` normal) | `100,592` / `100,281` | CPU: `0.02510` (`floor`) / `2.123e-15` (normal)<br>GPU: `2.1226890857966868e-15` | — |

Outside the stated 26-step CPU subnormal floor (`D5_PLATFORM_FLOOR`, `kt = 5842..5867`), the maximum normalised error across the entire 8,760-step C1D year under `jax.jit` is **`2.1226890857966868e-15`** (`12` ULPs at `kt = 5460 POST_DO.e_s`), and every remaining normal-step over-bar row (`336` rows: `29` steps in `POST_DH..POST_TEMP2.e_i` at `<= 11` ULP / `1.488e-15`, `99` steps in `POST_DO..EXIT.e_i` at `<= 11` ULP / `1.756e-15`, `2` steps in `POST_DH..POST_TEMP2.e_s` at `<= 12` ULP / `1.615e-15`, and `7` steps in `POST_DO..EXIT.e_s` at `<= 16` ULP / `2.123e-15`) is `<= 2.13e-15` (`1..2.1x` the `1e-15` bar) from `ice_var_vremap` enthalpy remapping in `_dh_step` / `_do_step` (`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md:30-43`).

---

## 6. Verification Tests, Planted Controls, and Autodiff Check

`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:61-2776` pins every Round 2 and Round 3 requirement:
1. **Step 2 bit-for-bit test + SSS-guard & revert controls (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1235-1358`):**
   * Verifies on `kt = 2` and `kt = 15` (inexact C1D operands) that `_exact_1d_entry(..., _nemo_entry_conversion=True)` reproduces `frames[1]["sz_i"]` and `exact["t_s"]` to **`0` ULPs** and **fails** (`1` ULP) when reverted (`_nemo_entry_conversion=False`).
   * Explicitly exercises both SSS-dependent guards in `_nemo_salprof(s_i, sss)` (`s_i >= 0.5 * rn_sinew * sss` and `s_i <= 3.0 * min(rn_simin, rn_sinew * sss)`) and proves that omitting either guard or hardcoding `3.0 * 0.1` fails.
2. **Step 3 bit-for-bit test + revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1361-1458`):**
   * Verifies on `kt = 2`, `15`, and `1334` that `snow_temperature_from_enthalpy(..., _nemo_glo2eqv_snow_temp=True)` reproduces `exact["t_s"]` to **`0` ULPs**, and **fails** when reverted (`_nemo_glo2eqv_snow_temp=False`), pushing `kt = 1334 POST_ZDF.e_s` over the `1e-15` bar (`norm_err = 1.0269e-15 > 1e-15`, `9` ULPs).
3. **Step 4 CPU fusion-prevention bit-for-bit test across `POST_ZDF` and `POST_DH..EXIT` + revert controls (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1461-1599`):**
   * Verifies on `kt = 1`, `3`, `2906`, `4242`, and `5841` that `_si3_zdf_bl99_step(..., _nemo_source_round=True)` matches all `POST_ZDF` fields (`t_su`, `e_i`, `e_s`) to **`0` ULPs** under compiled `@jax.jit`, and **fails** on CPU default JIT when reverted (`_nemo_source_round=False`), pushing `kt = 2906 POST_ZDF.e_s` over the `1e-15` bar (`norm_err = 1.1393e-15 > 1e-15`, `9` ULPs).
   * Verifies on `kt = 3961` (`option2_salinity_profile`), `kt = 2899` (`_piecewise_remap` and `_nemo_snow_enthalpy_remap`), `kt = 4237` (snow surface melt `zq_top` and ice surface melt `zdE`), and `kt = 4738` (ice basal melt `zdE`) that `jax.jit(si3_column_step_arrays)(..., _nemo_source_round=True)` matches the NEMO-exact reference and **fails** (`ulp > 0`) on CPU default JIT when `_nemo_source_round=False`.
4. **Autodiff & forward-bit preservation across all touched functions (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:316-472`, `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1602-1737`):**
   * Verifies `jax.value_and_grad` and `jax.jvp` through `option2_salinity_profile`, `snow_temperature_from_enthalpy`, `ice_temperature_from_enthalpy`, `ice_enthalpy_from_temperature`, `snow_enthalpy_from_temperature`, `p07_conductivity`, `_radiation`, `_si3_zdf_bl99_step`, `_piecewise_remap`, `_nemo_snow_enthalpy_remap`, and `thomas_solve(..., "nemo_unnormalised")`: primal outputs under `value_and_grad` and `jvp` are bit-for-bit (`uint64`) identical (`0` ULP) to direct primal evaluation, all gradients are finite and nonzero, and reverse-mode `jax.grad` matches forward-mode `jax.jvp` to `< 1e-12` relative error.
5. **Non-SI3 caller bit-for-bit invariance against parent commit (`59fd8f2395`) + code-path planted controls (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:475-1023`):**
   * Loads `tridiagonal.py`, `bitz_lipscomb.py`, `snow.py`, `sea_ice.py`, and `scm.py` directly from parent commit `59fd8f2395ccba60b5b04f3c2c68946835feea12` via `git show` (`_load_parent_commit_non_si3_modules`), executes the real non-SI3 callers (`bitz_lipscomb_conduction_step`, `thomas_solve(..., "normalised")`, `IceColumnModel.step()`, and `step_sea_ice` slab and dynamic-v2) on both the parent-commit modules and current-commit modules, asserts bit-for-bit (`uint64`) equality across all outputs, and proves that planting a code-path change in any of those four non-SI3 execution paths (`bitz_lipscomb_conduction_step`, `thomas_solve` unnormalised forward elimination, `simple_bulk_fluxes`, or `update_salinity_and_salt_flux`) causes the comparison to fail (`pytest.raises(AssertionError)`).
6. **Probe CLI expectations and planted failing controls (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1077-1232`):**
   * Verifies `--expect-step1-exact-snow-temp` and `--expect-zero-post-zdf-over-bar` pass, and that `--plant zero_subnormal_input`, `--plant flush_recorded_operand`, `--plant revert_statement_fix`, and `--plant revert_snow_temp_statement` all cause `nemo_si3thd_d5_platform_probe.py` to exit with code `1`.

---

## 7. Choices List and Review Disposition

* **CHOICES:**
  * `ASKED`: **DECISION 156** (taken by user before Round 2: classify and close D5 per platform before starting the ORCA2 rungs) and **DECISION 158** (taken by user before Round 3: certify the SI3 ladder on CPU `fp64` production JIT with XLA FMA prevented in code via `nemo_source_round`, subnormal cases as a stated platform floor, and each rung spot-checked on an idle GPU).
  * `UNASKED`: **NONE**. `_nemo_entry_conversion: bool = True` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:252-286` and `_nemo_glo2eqv_snow_temp: bool = True`, `_nemo_source_round: bool = True` in `packages/ice/legoesm/ice/bitz_lipscomb.py` (with `"nemo_unnormalised_unrounded"` in `packages/core/legoesm/timestepping/tridiagonal.py:137-180` for `_nemo_source_round=False` ablation) follow the existing private `_nemo_*` flag pattern and Rule 3 (the NEMO-literal value is the default on the SI3 path; non-SI3 callers are bit-for-bit unchanged). No configuration classes (`SeaIceConfig`, `SI3ThermoConfig`) were modified.
* **Independent review:** `independent review: operator`


## Operator amendments after independent review (Claude LAND, codex HOLD on mapping/receipt items)

Both reviewers independently reproduced the core result under jax.jit: the full-year output SHA-256 of CPU production
jit equals the --xla_cpu_max_isa=AVX diagnostic's (389490...), 965 over-bar rows, 521,368 rows at 0 ULP; the
salinity-profile guards match icevar.F90:495-505; the non-SI3 control runs parent vs current source with real plants.
Corrections and debts recorded by the operator:
- The year gate's jitted step starts at nemo_si3thd_phase2b_year_gate.py:1819 (:1814 is eager_step); citations fixed.
- The round-2 receipt's gpu_nvidia_smi_before.txt SHA-256 was malformed; replaced with the recomputed value.
- GPU: the Step 3 -> Step 4 change is reported by COUNTS only (368 rows changed; over-bar 516 -> 336). The evidence
  holds no per-row oracle-error direction, so "improvements only" is NOT claimed.
- DEBT (next D5/O1 round), three nemo_source_round sites wrap legoESM statements that are not NEMO's own form:
  (a) division by ice density where NEMO multiplies by its reciprocal (icethd_dh.F90:254,391; this receipt's line
  quoting NEMO's form as the model's is wrong); (b) the ice-layer remap (_piecewise_remap, bitz_lipscomb.py:902) uses an
  overlap sum, not NEMO's cumulative-content interpolation (icevar.F90:957-1034) — this remap owns the remaining
  over-bar rows and must be transcribed literally, replacing that insertion; (c) the salinity profile factors the bulk
  salinity differently from NEMO (icevar.F90:500-501).
- DEBT: the harness's two salinity-profile transition constants are named after namelist parameters (record values
  5.0 and 2.0) while NEMO uses fixed 4.5 and 3.5 — rename to match what is used; and the year gate silently defaults
  sea-surface salinity to 34 when absent (no effect on this record, constant 34.0) — a hidden default to be made a
  refusal.
