# SI3 Ladder Round 3 (`si3_r3`) Receipt — Rung D5 Continuation (Harness Entry Conversion, `ice_var_glo2eqv` Snow Temperature, and In-Code CPU FMA Prevention)

## 1. Scope, Preregistration, and Provenance

This round continues rung **D5** of the SI3 sea-ice fidelity ladder (`docs/ice/fidelity/si3_ladder_design.md:220`, `docs/ice/fidelity/si3_ladder_design.md:344-350`) under **DECISION 156** and **DECISION 158**:
* **DECISION 158 (user):** the SI3 ladder certifies on the CPU (`fp64`, production JIT, `XLA_FLAGS=""`) with XLA's fused multiply-add (`FMA`) prevented **in code** at the statements where NEMO's `-O3` build does not fuse — reusing the ocean ladder's helper `packages/core/legoesm/core/source_rounding.py:14-27` (`nemo_source_round`), never a global XLA flag; binary64 subnormal cases (`kt = 5842..5867`) stay a stated platform floor (`D5_PLATFORM_FLOOR`); each rung is spot-checked on an idle GPU.

Four ordered steps were preregistered in `docs/ice/fidelity/si3_r3_d5_preregister.md:1-54` before any measurement and evaluated both cumulatively and in isolation across the immutable 8,760-step C1D North Greenland ERA5 SAS record (`621,960` stage-field rows across 8 stages) on CPU production JIT (`XLA_FLAGS=""`), CPU no-fusion diagnostic (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`), and an idle GPU (`CUDA_VISIBLE_DEVICES=1`):
1. **Step 1 — Reproduce the Round 2 reviewer's numbers with a committed probe** (`docs/ice/fidelity/si3_r2_d5_receipt.md:180-189`, `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:395-690`).
2. **Step 2 — Harness entry conversion (mimicry glue, not model):** make `_entry_arrays` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-252` (and `_exact_1d_entry` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:712-721`) follow NEMO's `(e / v) * 3.0` (`src/ICE/icevar.F90:390`, `src/ICE/icevar.F90:412-413`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:404`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`) and option-2 salinity profile (`src/ICE/icevar.F90:354-360`, `src/ICE/icevar.F90:488-506`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:368-374`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:502-520`).
3. **Step 3 — Model statement (`snow_temperature_from_enthalpy`):** transcribe NEMO's solver-entry snow temperature statement (`src/ICE/icevar.F90:412-413`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`) in `packages/ice/legoesm/ice/bitz_lipscomb.py:349-391` (called at `packages/ice/legoesm/ice/bitz_lipscomb.py:503-509`) under the selectable SI3 flag `_nemo_glo2eqv_snow_temp: bool = True`, keeping non-SI3 callers bit-for-bit identical.
4. **Step 4 — In-code CPU fusion prevention (`DECISION 158`):** prevent XLA CPU FMA and cross-statement algebraic simplification in code with `nemo_source_round` (`packages/core/legoesm/core/source_rounding.py:14-27`) in `packages/core/legoesm/timestepping/tridiagonal.py:137-180` and `packages/ice/legoesm/ice/bitz_lipscomb.py:179-212`, `:215-276`, `:334-346`, `:349-391`, `:394-409`, `:412-458`, `:461-871`, and `:1259-1290`, controlled by `_nemo_source_round: bool = True`, and prove that the CPU production-JIT (`XLA_FLAGS=""`) full-year re-score equals the `--xla_cpu_max_isa=AVX` no-fusion diagnostic across every stage and field.

### Provenance & Evidence Digests
* **Preregistration commit (before measurement):** `501a51a8a0a10d808f41838922db514db55f3800` (`docs/ice/fidelity/si3_r3_d5_preregister.md:1-54`)
* **Implementation & probe commits:** `0ad9f4b62ee8236d9b5086a31a928f8b0a372da2`, `0ff50e908343c599fb3fe7791144fcd356a49f84`
* **Oracle record root:** `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a`
  * `oracle_si3_thd_frames.bin` SHA-256: `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b`
  * `oracle_si3_zdf_inputs.bin` SHA-256: `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd`
* **Committed probe:** `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:136-1439`
* **Committed unit & gradient tests:** `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:56-1241`
* **Evidence artifacts under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round3/`:**
  * `cpu_default_probe.json` (CPU production JIT, `XLA_FLAGS=""`) SHA-256: `a6a8f45234557e5df42034534cedeb67b63efcc527f53bc34f3246eaf7d1f449`
  * `cpu_nofma_probe.json` (CPU diagnostic, `XLA_FLAGS="--xla_cpu_max_isa=AVX"`) SHA-256: `42b958345f9ae85c7ac840c9e2e4c7ec66a82f22580247e8d10c26d091d5f596`
  * `gpu_probe.json` (GPU 1 spot-check via `CUDA_VISIBLE_DEVICES=1`) SHA-256: `ef379a59e8514448808e57397a2cb16ab675317368515785b8231f1fbf756947`
  * `gpu_nvidia_smi_before.txt` SHA-256: `4a9ff082832dc79dc3f7a7966bd6b7a8b9aec07cb1774976d39f991a1773a2cd`
  * `gpu_nvidia_smi_after.txt` SHA-256: `dca3367dcbda999503db5fe37c8013c39cd0d55f657e11465810bc4c2284de59`

### Idle-GPU Spot-Check Provenance (`CUDA_VISIBLE_DEVICES=1`)
Before launching the GPU spot-check, `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` and `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` were recorded to `gpu_nvidia_smi_before.txt`, showing both GPU 0 (`0 MiB`, `0 %`) and GPU 1 (`0 MiB`, `0 %`) completely idle with zero active compute processes. The probe was pinned to GPU 1 via `CUDA_VISIBLE_DEVICES=1`. Immediately after completion, the same queries were recorded to `gpu_nvidia_smi_after.txt`, confirming both GPUs at `0 MiB` (`5 %` transient cooldown on GPU 1) with zero compute apps.

---

## 2. Step 1: Committed Reproduction of the Round 2 Reviewer's Numbers

`run_step1_reviewer_reproduction` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:395-690`) reproduces from committed code every reviewer-measured number cited in `docs/ice/fidelity/si3_r2_d5_receipt.md:180-189`:

1. **ZDF-entry snow temperature (`exact["t_s"]` in `oracle_si3_zdf_inputs.bin`) across all `7,385` snow-present steps (`h_s > 0.0`):**
   * **Round 2 as shipped** (`(e_s * 3.0) / v_s` + `src/ICE/icethd_dh.F90:498-503` formula): **`1,811` of `7,385` steps at `0` ULP** (`5,574` mismatching steps, `max_ulp = 3`).
   * **Step 2 harness conversion only** (`(e_s / v_s) * 3.0` + `src/ICE/icethd_dh.F90:498-503` formula): **`3,547` of `7,385` steps at `0` ULP** (`3,838` mismatching steps, `max_ulp = 1`).
   * **Step 2 + Step 3 model statement** (`(e_s / v_s) * 3.0` + `src/ICE/icevar.F90:412-413` formula): **`7,385` of `7,385` steps at `0` ULP** (`0` mismatching steps, `max_ulp = 0`).
2. **ZDF-entry salinity profile (`frames[1]["sz_i"]`) across all `8,760` steps:**
   * **Round 2 as shipped** (`szv_i * 3.0 / v_i`): **`5,225` of `8,760` steps at `0` ULP** (`3,535` mismatching steps by `1` ULP).
   * **Step 2 harness conversion** (`_nemo_salprof(s_i)` from `src/ICE/icevar.F90:488-506`): **`8,760` of `8,760` steps at `0` ULP** (`0` mismatching steps).
3. **Normal-step `POST_ZDF` solved fields (`t_su`, `e_i`, `e_s` across the `8,734` normal steps outside `kt = 5842..5867` = `26,202` rows):**

| Step 1 `POST_ZDF` evaluation arm (`26,202` rows = `8,734` normal steps × 3 fields) | CPU production JIT (`XLA_FLAGS=""`) Over-bar / `0` ULP / Nonzero ULP (`max_ulp`) | CPU no-fusion (`--xla_cpu_max_isa=AVX`) Over-bar / `0` ULP / Nonzero ULP (`max_ulp`) | GPU (`CUDA_VISIBLE_DEVICES=1`) Over-bar / `0` ULP / Nonzero ULP (`max_ulp`) |
|---|---:|---:|---:|
| **Round 2 as shipped** (`_nemo_entry_conversion=False`, `_nemo_glo2eqv_snow_temp=False`, `_nemo_source_round=False`) | `571` / `15,873` / `10,329` (`4,709`)<br>(`e_i`: `530`/`6,677`, `e_s`: `36`/`3,256`, `t_su`: `5`/`5,940`) | `470` / `20,054` / `6,148` (`4,709`)<br>(`e_i`: `455`/`6,931`, `e_s`: `12`/`5,220`, `t_su`: `3`/`7,903`) | `470` / `20,054` / `6,148` (`4,709`)<br>(`e_i`: `455`/`6,931`, `e_s`: `12`/`5,220`, `t_su`: `3`/`7,903`) |
| **NEMO-exact ZDF inputs before Round 2 reciprocals** (`_nemo_zdf_layer_reciprocals=False`, `_nemo_source_round=False`) | `1,650` / `15,431` / `10,771` (`1,636`)<br>(`e_i`: `463`/`7,724`, `e_s`: `681`/`2,607`, `t_su`: `506`/`5,100`) | **`1,625` / `16,292` / `9,910` (`1,636`)**<br>(`e_i`: `433`/`7,788`, `e_s`: `687`/`2,753`, `t_su`: `505`/`5,751`) | **`1,625` / `16,292` / `9,910` (`1,636`)**<br>(`e_i`: `433`/`7,788`, `e_s`: `687`/`2,753`, `t_su`: `505`/`5,751`) |
| **NEMO-exact ZDF inputs after Round 2 reciprocals, unrounded** (`_nemo_zdf_layer_reciprocals=True`, `_nemo_source_round=False`) | `130` / `19,171` / `7,031` (`330`)<br>(`e_i`: `103`/`8,373`, `e_s`: `23`/`4,601`, `t_su`: `4`/`6,197`) | **`0` / `26,202` / `0` (`0`)**<br>(`e_i`: `0`/`8,734`, `e_s`: `0`/`8,734`, `t_su`: `0`/`8,734`) | **`0` / `26,202` / `0` (`0`)**<br>(`e_i`: `0`/`8,734`, `e_s`: `0`/`8,734`, `t_su`: `0`/`8,734`) |
| **NEMO-exact ZDF inputs with Step 4 `nemo_source_round`** (`_nemo_zdf_layer_reciprocals=True`, `_nemo_source_round=True`) | **`0` / `26,202` / `0` (`0`)**<br>(`e_i`: `0`/`8,734`, `e_s`: `0`/`8,734`, `t_su`: `0`/`8,734`) | **`0` / `26,202` / `0` (`0`)**<br>(`e_i`: `0`/`8,734`, `e_s`: `0`/`8,734`, `t_su`: `0`/`8,734`) | **`0` / `26,202` / `0` (`0`)**<br>(`e_i`: `0`/`8,734`, `e_s`: `0`/`8,734`, `t_su`: `0`/`8,734`) |

This **CONFIRMS** Prediction **P1** (`docs/ice/fidelity/si3_r3_d5_preregister.md:40-42`): the committed probe reproduces `1,811 / 7,385` snow steps as shipped, `7,385 / 7,385` with Steps 2+3, and `1,625 -> 0` over-bar rows (`26,202 / 26,202` at `0` ULP) on GPU and CPU-without-fusion (and now also on CPU production JIT with Step 4 `nemo_source_round=True`).

---

## 3. Step 2 (Harness), Step 3 (Model), and Step 4 (In-Code CPU Fusion Prevention)

### 3.1 Step 2: Harness Entry Conversion (`nemo_si3thd_phase2_gate.py`)
In NEMO's `ice_var_glo2eqv` (`src/ICE/icevar.F90:354-360`, `src/ICE/icevar.F90:390`, `src/ICE/icevar.F90:412-413`, `src/ICE/icevar.F90:488-506`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:368-374`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:404`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:502-520`), global fields are converted to equivalent 1-D layer fields before `ice_thd`:
* `ze_i = e_i(ji,jj,jk,jl) / v_i(ji,jj,jl) * zlay_i` (`zlay_i = 3.0_wp`)
* `e_s(ji,jj,jk,jl) / v_s(ji,jj,jl) * zlay_s` (`zlay_s = 3.0_wp`)
* For `nn_icesal = 2` (C1D's salinity scheme), `s_i = sv_i / v_i` followed by `CALL ice_var_salprof` (`src/ICE/icevar.F90:354-360`, `src/ICE/icevar.F90:488-506`), whereas the `nn_icesal == 4` branch (`src/ICE/icevar.F90:362-379`) is not taken.
* `_entry_arrays` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-252` (and `_exact_1d_entry` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:712-721`) now computes `(g["e_i"] / vi) * 3.0`, `(g["e_s"] / vs) * 3.0`, and `_nemo_salprof(s_i)` by default (`_nemo_entry_conversion: bool = True`), with `_nemo_entry_conversion=False` retained for before/after ablation.
* In `ENTRY` (`105,120` rows across `8,760` steps), `0`-ULP rows increase from `84,373` to **`85,157` (`+784` rows moved to `0` ULP)** with `0` over-bar rows.

### 3.2 Step 3: Solver-Entry Snow Temperature Statement (`bitz_lipscomb.py`)
At solver entry (`ice_var_glo2eqv`), NEMO computes snow layer temperature via `src/ICE/icevar.F90:412-413` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`):
* `t_s(ji,jj,jk,jl) = rt0 + MAX( -100._wp , MIN( r1_rcpi*( -r1_rhos*( e_s(ji,jj,jk,jl) / v_s(ji,jj,jl) * zlay_s ) + rLfus ) , 0._wp ) )`
* Previously, `snow_temperature_from_enthalpy` (`packages/ice/legoesm/ice/bitz_lipscomb.py:349-391`, called at `packages/ice/legoesm/ice/bitz_lipscomb.py:503-509`) evaluated the expanded `src/ICE/icethd_dh.F90:498-503` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:508-513`) form `constants.T0 + (-e * inverse_rho_snow * inverse_c_ice + constants.latent_fusion * inverse_c_ice)`.
* Because `r1_rcpi * (-r1_rhos * e_s + rLfus)` rounds `-r1_rhos * e_s + rLfus` before multiplying by `r1_rcpi`, the two parenthesizations differ by `1` ULP on `3,838` of the `7,385` snow steps (and at `kt = 1334`, that `1`-ULP entry `T_s` difference amplifies through the stiff clamped-snow tridiagonal solve into a `9`-ULP / `1.0269e-15 > 1e-15` over-bar error in `POST_ZDF.e_s`).
* Transcribing `src/ICE/icevar.F90:412-413` under `_nemo_glo2eqv_snow_temp: bool = True` (`packages/ice/legoesm/ice/bitz_lipscomb.py:349-391`) brings ZDF-entry `t_s` to **`7,385 / 7,385` steps at `0` ULP** and closes the remaining `1` normal-step `POST_ZDF` over-bar row (`kt = 1334`) and all `2,485` remaining nonzero-ULP normal-step `POST_ZDF` rows on GPU and CPU `--xla_cpu_max_isa=AVX`.

### 3.3 Step 4: In-Code CPU Fusion Prevention via `nemo_source_round` (`DECISION 158`)
Comparing default CPU production JIT (`XLA_FLAGS=""`) against `XLA_FLAGS="--xla_cpu_max_isa=AVX"` and NEMO's `-O3` gfortran build (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90` and `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90`) identified two classes of compiler transformations on `x86_64` CPU: (a) AVX2/AVX-512 fused multiply-add (`FMA`) contraction of `a * b + c` / `a * b - c`, and (b) XLA's `HloAlgebraicSimplifier` factoring `2.0 * a + 2.0 * b -> (a + b) * 2.0` before emitting `vfmasd`. Under **DECISION 158**, every such statement on the SI3 BL99 path is guarded in code with `nemo_source_round` (`packages/core/legoesm/core/source_rounding.py:14-27`), gated by `_nemo_source_round: bool = True`:

1. **Thomas unnormalised backward substitution (`packages/core/legoesm/timestepping/tridiagonal.py:137-180`):**
   * NEMO source: `scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:571-595` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:544-567`), where `t_i_1d(ji,jk) = ( zindtbis(jm) - ztrid(jm,3) * t_i_1d(ji,jk+1) ) / zdiagbis(jm)` and `t_s_1d(ji,jk) = ( zindtbis(jm) - ztrid(jm,3) * t_s_1d(ji,jk+1) ) / zdiagbis(jm)` evaluate the product `ztrid(jm,3) * t(...)` before subtracting from `zindtbis(jm)`.
   * In `unnormalised_backward` (`packages/core/legoesm/timestepping/tridiagonal.py:170-175`), `c[..., k] * current[..., k + 1]` is wrapped in `nemo_source_round` for `operation_order == "nemo_unnormalised"` (while `"nemo_unnormalised_unrounded"` preserves the unguarded path for `_nemo_source_round=False` ablation).
2. **Ice temperature from enthalpy (`packages/ice/legoesm/ice/bitz_lipscomb.py:215-276`):**
   * NEMO source: `src/ICE/icevar.F90:390-395` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:404-409`), where `ztmelts = - sz_i * rTmlt`, `zbbb = ( rcp - rcpi ) * ztmelts + ze_i * r1_rhoi - rLfus`, `zccc = SQRT( MAX( zbbb * zbbb - 4._wp * rcpi * rLfus * ztmelts , 0._wp) )`, and `t_i = MAX( -100._wp , MIN( -( zbbb + zccc ) * 0.5_wp * r1_rcpi , ztmelts ) ) + rt0` are evaluated without FMA across the products in `zbbb`, `zccc`, and `t_i`.
3. **Solver-entry snow temperature from enthalpy (`packages/ice/legoesm/ice/bitz_lipscomb.py:349-391`):**
   * NEMO source: `src/ICE/icevar.F90:412-413` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`), where `-r1_rhos * e_s` is rounded before `+ rLfus` and before `* r1_rcpi`.
4. **Ice & snow enthalpy from temperature (`packages/ice/legoesm/ice/bitz_lipscomb.py:179-212`, `packages/ice/legoesm/ice/bitz_lipscomb.py:334-346`):**
   * NEMO source: `src/ICE/icevar.F90:938-953` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:952-967`), called at `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:813`, where `e_i_1d` and `e_s_1d = rhos * ( rcpi * ( rt0 - t_s_1d(ji,jk) ) + rLfus )` round each parenthesized product/difference separately.
5. **Pringle et al. (2007) thermal conductivity (`packages/ice/legoesm/ice/bitz_lipscomb.py:394-409`, `packages/ice/legoesm/ice/bitz_lipscomb.py:567-591`):**
   * NEMO source: `scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:276-290` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:271-285`), where `0.5_wp * (sz_i_1d(ji,jk) + sz_i_1d(ji,jk+1))`, `0.5_wp * (t_i_1d(ji,jk) + t_i_1d(ji,jk+1)) - rt0`, `rcnd_i + 0.09_wp * ... / MIN(...)`, and `- 0.011_wp * (...)` are rounded without FMA.
6. **Solar radiation transmission (`packages/ice/legoesm/ice/bitz_lipscomb.py:412-458`):**
   * NEMO source: `scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:217-230` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:219-232`), where `zh_s * REAL(jk,wp)`, `zh_i * REAL(jk,wp)`, and the two canopy/open-sky transmission terms in `zradtr_i` are rounded before addition/subtraction.
7. **BL99 kappa interface, specific heat `zcpi`, surface flux `qns_ice_1d`, tridiagonal matrix assembly, melting-point clip, and post-solve `qcn_ice_top_1d` (`packages/ice/legoesm/ice/bitz_lipscomb.py:604-842`):**
   * NEMO source: `scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:334-335`, `:365-366`, `:399-529`, `:620`, `:821-823` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:329-330`, `:353-354`, `:387-512`, `:588`, `:782-784`). Wrapping `ztcond_i(ji,0) * zh_s(ji)` and `rcnd_s * zh_i(ji)` in `zkappa_s(ji,nlay_s)`, `(t_i - rt0) * (ztiold - rt0)` and `zgamma * sz_i / MAX(...)` in `zcpi`, `dqns * (t_su - ztsub)` in `qns_ice_1d`, the `zeta * zkappa` products and `2.0 * k` factors in `ztrid`/`zindterm` (preventing `HloAlgebraicSimplifier` from refactoring `1.0 + eta * (k_1 + 2.0 * k_0)`), `-rTmlt * sz_i + rt0`, and the two terms of `qcn_ice_top_1d` eliminates all `7,031` FMA-induced nonzero-ULP `POST_ZDF` rows (`130` over-bar rows) on CPU production JIT.

---

## 4. Full-Year (`8,760`-Step, `621,960`-Row) C1D Rescore Across All Steps and Platforms

`run_year_rescore` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:797-1255`) evaluates all `621,960` stage-field rows across the 8 stages (`ENTRY`, `POST_ZDF`, `POST_DH`, `POST_TEMP1`, `POST_SAL`, `POST_TEMP2`, `POST_DO`, `EXIT`) for each cumulative step and each isolated single-step arm from the Round 2 baseline:

### 4.1 Summary of Over-Bar Rows and `0`-ULP Rows Before and After Each Step (`621,960` Total Rows)

| Evaluation arm | CPU Production JIT (`XLA_FLAGS=""`) Over-bar / `0` ULP / Nonzero ULP (Normal `POST_ZDF` 3-field over-bar / `0` ULP) | CPU No-Fusion Diagnostic (`--xla_cpu_max_isa=AVX`) Over-bar / `0` ULP / Nonzero ULP (Normal `POST_ZDF` 3-field over-bar / `0` ULP) | GPU Spot-Check (`CUDA_VISIBLE_DEVICES=1`) Over-bar / `0` ULP / Nonzero ULP (Normal `POST_ZDF` 3-field over-bar / `0` ULP) |
|---|---:|---:|---:|
| **Pre-Round-2 (`r2_pre_reciprocals`)** | `35,282` / `442,662` / `179,298`<br>(`2,015` / `13,419`) | `35,377` / `447,311` / `174,649`<br>(`2,012` / `14,134`) | `35,028` / `447,427` / `174,533`<br>(`2,012` / `14,134`) |
| **Round 2 baseline (`r2_baseline`, before Step 2)** | **`6,584` / `464,502` / `157,458`**<br>(`571` / `15,873`) | **`5,107` / `491,635` / `130,325`**<br>(`470` / `20,054`) | **`4,495` / `491,949` / `130,011`**<br>(`470` / `20,054`) |
| **After Step 2 (`step2_harness_only`, cumulative & isolated)** | **`3,161` / `482,471` / `139,489`**<br>(`134` / `18,199`) | **`1,416` / `517,233` / `104,727`**<br>(`1` / `23,717`) | **`797` / `517,544` / `104,416`**<br>(`1` / `23,717`) |
| **After Step 2 + Step 3 (`step2_plus_step3`, cumulative)** | **`2,931` / `488,873` / `133,087`**<br>(`130` / `19,171`) | **`1,197` / `532,390` / `89,570`**<br>(**`0` / `26,202`**) | **`578` / `532,701` / `89,259`**<br>(**`0` / `26,202`**) |
| **After Step 2 + Step 3 + Step 4 (`step2_plus_step3_plus_step4`, final)** | **`1,197` / `532,390` / `89,570`**<br>(**`0` / `26,202`**) | **`1,197` / `532,390` / `89,570`**<br>(**`0` / `26,202`**) | **`578` / `532,701` / `89,259`**<br>(**`0` / `26,202`**) |
| *Isolated Step 3 only (`step3_model_only`, from `r2_baseline`)* | `6,407` / `467,851` / `154,109`<br>(`568` / `16,376`) | `4,962` / `499,354` / `122,606`<br>(`471` / `21,259`) | `4,350` / `499,668` / `122,292`<br>(`471` / `21,259`) |
| *Isolated Step 4 only (`step4_source_round_only`, from `r2_baseline`)* | `5,107` / `491,635` / `130,325`<br>(`470` / `20,054`) | `5,107` / `491,635` / `130,325`<br>(`470` / `20,054`) | `4,495` / `491,949` / `130,011`<br>(`470` / `20,054`) |

### 4.2 Stage-by-Stage Progression on CPU Production JIT (`XLA_FLAGS=""`, `cpu_default_probe.json`)

| Stage (`rows`) | `r2_baseline` Over-bar / `0` ULP | After Step 2 (`step2_harness_only`) Over-bar / `0` ULP | After Step 2+3 (`step2_plus_step3`) Over-bar / `0` ULP | After Step 2+3+4 (`step2_plus_step3_plus_step4`) Over-bar / `0` ULP | Net Over-bar Delta (`r2_baseline -> final`) | Net `0`-ULP Delta (`r2_baseline -> final`) |
|---|---:|---:|---:|---:|---:|---:|
| `ENTRY` (`105,120`) | `0` / `84,373` | `0` / `85,157` | `0` / `85,157` | `0` / `85,157` | `0` | **`+784`** |
| `POST_ZDF` (`61,320`) | `646` / `47,381` | `209` / `53,242` | `205` / `54,214` | **`75` / `61,245`** | **`-571`** | **`+13,864`** |
| `POST_DH` (`61,320`) | `899` / `40,270` | `416` / `44,466` | `384` / `45,372` | **`152` / `51,438`** | **`-747`** | **`+11,168`** |
| `POST_TEMP1` (`61,320`) | `899` / `40,270` | `416` / `44,466` | `384` / `45,372` | **`152` / `51,438`** | **`-747`** | **`+11,168`** |
| `POST_SAL` (`61,320`) | `901` / `41,810` | `418` / `42,518` | `386` / `43,424` | **`152` / `49,510`** | **`-749`** | **`+7,700`** |
| `POST_TEMP2` (`61,320`) | `901` / `41,810` | `418` / `42,518` | `386` / `43,424` | **`152` / `49,510`** | **`-749`** | **`+7,700`** |
| `POST_DO` (`105,120`) | `1,169` / `84,294` | `642` / `85,052` | `593` / `85,955` | **`257` / `92,046`** | **`-912`** | **`+7,752`** |
| `EXIT` (`105,120`) | `1,169` / `84,294` | `642` / `85,052` | `593` / `85,955` | **`257` / `92,046`** | **`-912`** | **`+7,752`** |
| **Total (`621,960`)** | **`6,584` / `464,502`** | **`3,161` / `482,471`** | **`2,931` / `488,873`** | **`1,197` / `532,390`** | **`-5,387`** | **`+67,888`** |

On CPU production JIT (`91,159 + 530,801 == 621,960`), `5,395` rows closed to `AT-BAR` and `8` rows opened to `DEBT` (net `-5,387`), while `71,193` rows moved to `0` ULP vs oracle and `3,305` rows moved from `0` ULP (`3,107` of which are `ENTRY.sz_i` rows where the stored global `ENTRY` frame holds `sz_i = szv_i * 3.0 / v_i`, while `ice_var_glo2eqv` overwrites `sz_i` with `ice_var_salprof` before `POST_ZDF`, where `0` rows moved away from `0` ULP and `13,864` rows moved to `0` ULP).

### 4.3 Stage-by-Stage Progression on CPU No-Fusion Diagnostic (`--xla_cpu_max_isa=AVX`) and GPU (`CUDA_VISIBLE_DEVICES=1`)

| Stage (`rows`) | CPU `--xla_cpu_max_isa=AVX` `r2_baseline -> Step 2 -> Step 2+3 -> Step 2+3+4` Over-bar (`0` ULP) | GPU `CUDA_VISIBLE_DEVICES=1` `r2_baseline -> Step 2 -> Step 2+3 -> Step 2+3+4` Over-bar (`0` ULP) |
|---|---:|---:|
| `ENTRY` (`105,120`) | `0 -> 0 -> 0 -> 0`<br>(`84,373 -> 85,157 -> 85,157 -> 85,157`) | `0 -> 0 -> 0 -> 0`<br>(`84,373 -> 85,157 -> 85,157 -> 85,157`) |
| `POST_ZDF` (`61,320`) | `545 -> 76 -> 75 -> 75`<br>(`51,562 -> 58,760 -> 61,245 -> 61,245`) | **`471 -> 1 -> 0 -> 0`**<br>(**`51,632 -> 58,835 -> 61,320 -> 61,320`**) |
| `POST_DH` (`61,320`) | `698 -> 180 -> 152 -> 152`<br>(`44,080 -> 49,322 -> 51,438 -> 51,438`) | `626 -> 107 -> 79 -> 79`<br>(`44,115 -> 49,356 -> 51,472 -> 51,472`) |
| `POST_TEMP1` (`61,320`) | `698 -> 180 -> 152 -> 152`<br>(`44,080 -> 49,322 -> 51,438 -> 51,438`) | `626 -> 107 -> 79 -> 79`<br>(`44,115 -> 49,356 -> 51,472 -> 51,472`) |
| `POST_SAL` (`61,320`) | `700 -> 180 -> 152 -> 152`<br>(`45,635 -> 47,394 -> 49,510 -> 49,510`) | `628 -> 107 -> 79 -> 79`<br>(`45,670 -> 47,428 -> 49,544 -> 49,544`) |
| `POST_TEMP2` (`61,320`) | `700 -> 180 -> 152 -> 152`<br>(`45,635 -> 47,394 -> 49,510 -> 49,510`) | `628 -> 107 -> 79 -> 79`<br>(`45,670 -> 47,428 -> 49,544 -> 49,544`) |
| `POST_DO` (`105,120`) | `883 -> 310 -> 257 -> 257`<br>(`88,135 -> 89,942 -> 92,046 -> 92,046`) | `758 -> 184 -> 131 -> 131`<br>(`88,187 -> 89,992 -> 92,096 -> 92,096`) |
| `EXIT` (`105,120`) | `883 -> 310 -> 257 -> 257`<br>(`88,135 -> 89,942 -> 92,046 -> 92,046`) | `758 -> 184 -> 131 -> 131`<br>(`88,187 -> 89,992 -> 92,096 -> 92,096`) |
| **Total (`621,960`)** | **`5,107 -> 1,416 -> 1,197 -> 1,197`**<br>(**`491,635 -> 517,233 -> 532,390 -> 532,390`**) | **`4,495 -> 797 -> 578 -> 578`**<br>(**`491,949 -> 517,544 -> 532,701 -> 532,701`**) |

### 4.4 Exact Equivalence of CPU Production JIT (`XLA_FLAGS=""`) and `--xla_cpu_max_isa=AVX` After Step 4
Compare the final `step2_plus_step3_plus_step4` column across all three platforms:
* **CPU production JIT (`XLA_FLAGS=""`) vs CPU no-fusion diagnostic (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`):**
  * Total over-bar rows: **`1,197 == 1,197`**
  * Total `0`-ULP rows: **`532,390 == 532,390`**
  * Total nonzero-ULP rows: **`89,570 == 89,570`**
  * Per-stage over-bar counts: `ENTRY: 0`, `POST_ZDF: 75`, `POST_DH: 152`, `POST_TEMP1: 152`, `POST_SAL: 152`, `POST_TEMP2: 152`, `POST_DO: 257`, `EXIT: 257` — **identical on every stage**.
  * Per-stage `0`-ULP counts: `ENTRY: 85,157`, `POST_ZDF: 61,245`, `POST_DH: 51,438`, `POST_TEMP1: 51,438`, `POST_SAL: 49,510`, `POST_TEMP2: 49,510`, `POST_DO: 92,046`, `EXIT: 92,046` — **identical on every stage**.
  * Even isolated `step4_source_round_only` on CPU production JIT (`5,107` over-bar, `491,635` at `0` ULP) matches `r2_baseline` on `--xla_cpu_max_isa=AVX` (`5,107` over-bar, `491,635` at `0` ULP) to the exact row count on every stage.
* **GPU (`CUDA_VISIBLE_DEVICES=1`):**
  * Across all `8,760` steps (including all 26 subnormal-snow steps `kt = 5842..5867`, where Step 2 also closed the single remaining `kt = 5857 POST_ZDF.e_i` row from Round 2), **`POST_ZDF` has `0` over-bar rows and `61,320 / 61,320` rows at `0` ULP** (`100 %` bit-exact across the entire year!).
  * Outside the 26 subnormal steps (`8,734` normal steps), CPU production JIT (`1,197 - 629 = 568`), CPU `--xla_cpu_max_isa=AVX` (`1,197 - 629 = 568`), and GPU (`578 - 10 = 568`) all have **the exact same `568` over-bar rows**.

This **CONFIRMS** Predictions **P2**, **P3**, and **P4** (`docs/ice/fidelity/si3_r3_d5_preregister.md:44-54`). Note on P2's intermediate count (`docs/ice/fidelity/si3_r3_d5_preregister.md:45`): Step 2 alone performed even better than the `470 -> 60` estimate, dropping normal-step `POST_ZDF` over-bar rows all the way from `470` to **`1`** (`kt = 1334 POST_ZDF.e_s`), which Step 3 then closed to **`0`**.

---

## 5. Remaining Owner Ranking After Step 4

With `POST_ZDF` completely closed to `0` ULPs (`61,138 / 61,138` normal-step `POST_ZDF` rows at `0` ULP on CPU, and `61,320 / 61,320` all-step `POST_ZDF` rows at `0` ULP on GPU), the remaining over-bar rows in `step2_plus_step3_plus_step4` (`1,197` on CPU, `578` on GPU) rank as follows (`cpu_default_probe.json`, `cpu_nofma_probe.json`, `gpu_probe.json`):

| Rank | Owner | CPU Over-bar Rows (`cpu_default` & `cpu_avx`) | GPU Over-bar Rows (`CUDA_VISIBLE_DEVICES=1`) | Nonzero-ULP Rows (CPU / GPU) | Max Normalised Error | Worst Row (`kt`, Stage, Field, ULP) |
|---:|---|---:|---:|---:|---:|---|
| 1 | `D5_PLATFORM_FLOOR_KT5842_5867_SUBNORMAL_SNOW` (`kt = 5842..5867`, 26 subnormal steps) | **`629`** | **`10`** | `871` / `560` | CPU: `0.02510099530281747`<br>GPU: `1.2742122061897727e-15` | CPU: `kt = 5860 POST_ZDF.e_s` (`190,081,022,848,368` ULP, `D5_PLATFORM_FLOOR`)<br>GPU: `kt = 5858 POST_DO.e_i` (`8` ULP, `norm_err = 1.274e-15`) |
| 2 | `POST_DO.e_i` (normal steps, threshold-edge `<= 1.463e-15`) | **`128`** | **`128`** | `8,312` / `8,312` | `1.462719571719126e-15` | `kt = 4334 POST_DO.e_i` (`9` ULP, `max_ulp = 13`) |
| 3 | `EXIT.e_i` (same as `POST_DO.e_i`) | **`128`** | **`128`** | `8,312` / `8,312` | `1.462719571719126e-15` | `kt = 4334 EXIT.e_i` (`9` ULP, `max_ulp = 13`) |
| 4 | `POST_DH.e_i` (normal steps, threshold-edge `<= 1.461e-15`) | **`77`** | **`77`** | `8,444` / `8,444` | `1.4608864626350778e-15` | `kt = 44 POST_DH.e_i` (`7` ULP, `max_ulp = 13`) |
| 5 | `POST_TEMP1.e_i` (same as `POST_DH.e_i`) | **`77`** | **`77`** | `8,444` / `8,444` | `1.4608864626350778e-15` | `kt = 44 POST_TEMP1.e_i` (`7` ULP, `max_ulp = 13`) |
| 6 | `POST_SAL.e_i` (same as `POST_DH.e_i`) | **`77`** | **`77`** | `8,444` / `8,444` | `1.4608864626350778e-15` | `kt = 44 POST_SAL.e_i` (`7` ULP, `max_ulp = 13`) |
| 7 | `POST_TEMP2.e_i` (same as `POST_DH.e_i`) | **`77`** | **`77`** | `8,444` / `8,444` | `1.4608864626350778e-15` | `kt = 44 POST_TEMP2.e_i` (`7` ULP, `max_ulp = 13`) |
| 8 | `POST_DO.e_s` (normal steps, threshold-edge `<= 2.025e-15`) | **`2`** | **`2`** | `5` / `5` | `2.0252397989846502e-15` | `kt = 5960 POST_DO.e_s` (`18` ULP) |
| 9 | `EXIT.e_s` (same as `POST_DO.e_s`) | **`2`** | **`2`** | `5` / `5` | `2.0252397989846502e-15` | `kt = 5960 EXIT.e_s` (`18` ULP) |
| — | **All other `POST_ZDF..EXIT` fields outside `kt = 5842..5867`** | **`0`** | **`0`** | — | `<= 9.99e-16` (`AT-BAR`) | — |
| **Total** | **All `621,960` stage-field rows** | **`1,197`** (`629` floor + `568` normal) | **`578`** (`10` subnormal `POST_DH..EXIT` + `568` normal) | `89,570` / `89,259` | CPU: `0.02510` (`floor`) / `2.025e-15` (normal)<br>GPU: `2.0252397989846502e-15` | — |

Outside the stated 26-step CPU subnormal floor (`D5_PLATFORM_FLOOR`, `kt = 5842..5867`), the maximum normalised error across the entire 8,760-step C1D year is now **`2.0252397989846502e-15`** (`18` ULPs at `kt = 5960 POST_DO.e_s`), and every remaining normal-step over-bar row (`568` rows: `77` steps in `POST_DH..POST_TEMP2.e_i` at `<= 13` ULP / `1.461e-15`, `128` steps in `POST_DO..EXIT.e_i` at `<= 13` ULP / `1.463e-15`, and `2` steps in `POST_DO..EXIT.e_s` at `<= 18` ULP / `2.025e-15`) is `<= 2.03e-15` (`1..2x` the `1e-15` bar) from `ice_var_vremap` enthalpy remapping in `_dh_step` / `_do_step` (`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md:30-43`).

---

## 6. Verification Tests, Planted Controls, and Autodiff Check

`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:56-1241` pins every Round 2 and Round 3 requirement:
1. **Step 2 bit-for-bit test + revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:855-934`):**
   * Verifies on `kt = 2` and `kt = 15` (inexact C1D operands) that `_exact_1d_entry(..., _nemo_entry_conversion=True)` reproduces `frames[1]["sz_i"]` and `exact["t_s"]` to **`0` ULPs** and **fails** (`1` ULP) when reverted (`_nemo_entry_conversion=False`).
2. **Step 3 bit-for-bit test + revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:937-1034`):**
   * Verifies on `kt = 2`, `15`, and `1334` that `snow_temperature_from_enthalpy(..., _nemo_glo2eqv_snow_temp=True)` reproduces `exact["t_s"]` to **`0` ULPs**, and **fails** when reverted (`_nemo_glo2eqv_snow_temp=False`), pushing `kt = 1334 POST_ZDF.e_s` over the `1e-15` bar (`norm_err = 1.0269e-15 > 1e-15`, `9` ULPs).
3. **Step 4 CPU fusion-prevention bit-for-bit test + revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1037-1122`):**
   * Verifies on `kt = 1`, `3`, `2906`, `4242`, and `5841` that `_si3_zdf_bl99_step(..., _nemo_source_round=True)` matches all `POST_ZDF` fields (`t_su`, `e_i`, `e_s`) to **`0` ULPs** under compiled `@jax.jit`, and **fails** on CPU default JIT when reverted (`_nemo_source_round=False`), pushing `kt = 2906 POST_ZDF.e_s` over the `1e-15` bar (`norm_err = 1.1393e-15 > 1e-15`, `9` ULPs).
4. **Autodiff & forward-bit preservation across all touched functions (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:305-459`, `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1125-1241`):**
   * Verifies `jax.value_and_grad` and `jax.jvp` through `snow_temperature_from_enthalpy`, `ice_temperature_from_enthalpy`, `ice_enthalpy_from_temperature`, `snow_enthalpy_from_temperature`, `p07_conductivity`, `_radiation`, `_si3_zdf_bl99_step`, and `thomas_solve(..., "nemo_unnormalised")`: primal outputs under `value_and_grad` and `jvp` are bit-for-bit (`uint64`) identical (`0` ULP) to direct primal evaluation, all gradients are finite and nonzero, and reverse-mode `jax.grad` matches forward-mode `jax.jvp` to `< 1e-13` relative error.
5. **Non-SI3 caller bit-for-bit invariance (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:462-655`):**
   * Verifies `bitz_lipscomb_conduction_step`, `IceColumnModel`, `step_sea_ice` (slab and dynamic-v2), and `thomas_solve(..., "normalised")` are bit-for-bit (`uint64`) invariant against the pre-D5 baseline, with planted 1-ULP controls that fire.
6. **Probe CLI expectations and planted failing controls (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:709-852`):**
   * Verifies `--expect-step1-exact-snow-temp` and `--expect-zero-post-zdf-over-bar` pass, and that `--plant zero_subnormal_input`, `--plant flush_recorded_operand`, `--plant revert_statement_fix`, and `--plant revert_snow_temp_statement` all cause `nemo_si3thd_d5_platform_probe.py` to exit with code `1`.

---

## 7. Choices List and Review Disposition

* **CHOICES:**
  * `ASKED`: **DECISION 156** (taken by user before Round 2: classify and close D5 per platform before starting the ORCA2 rungs) and **DECISION 158** (taken by user before Round 3: certify the SI3 ladder on CPU `fp64` production JIT with XLA FMA prevented in code via `nemo_source_round`, subnormal cases as a stated platform floor, and each rung spot-checked on an idle GPU).
  * `UNASKED`: **NONE**. `_nemo_entry_conversion: bool = True` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:224-252` and `_nemo_glo2eqv_snow_temp: bool = True`, `_nemo_source_round: bool = True` in `packages/ice/legoesm/ice/bitz_lipscomb.py` (with `"nemo_unnormalised_unrounded"` in `packages/core/legoesm/timestepping/tridiagonal.py:137-180` for `_nemo_source_round=False` ablation) follow the existing private `_nemo_*` flag pattern and Rule 3 (the NEMO-literal value is the default on the SI3 path; non-SI3 callers are bit-for-bit unchanged). No configuration classes (`SeaIceConfig`, `SI3ThermoConfig`) were modified.
* **Independent review:** `independent review: operator`
