# SI3 Ladder Round 5 (`si3_r5`) — Rung D5 Debt Closure Receipt (`ice_var_vremap`, `r1_rhoi` in `icethd_dh`, `ice_var_salprof` Factoring, and Harness `zsi1`/`zsi0` + Missing-SSS Refusal)

* **Round:** `si3_r5` (SI3 sea-ice fidelity ladder, rung D5 debts recorded in `docs/ice/fidelity/si3_r3_d5_receipt.md:247-266`)
* **Branch:** `fidelity/nemo-testcases-si3-ladder`
* **Parent commit (Round 3 landed + operator amendments):** `a572f4b2540351bf49328d1533211427f9c9d772`
* **Preregistration document:** `docs/ice/fidelity/si3_r5_d5_preregister.md:1-48`
* **Probe & gate code commit:** `575fea9634e8225d9084e4364e0bf488bd90c831` (`clean: true`)
* **Decisions binding this round:** **DECISION 156**, **DECISION 158**, **DECISION 159** (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_si3/standing_note.md`)
* **Independent review:** `independent review: operator`

---

## 1. Executive Summary & Verdict

Round 5 (`si3_r5`) closes the four D5 debt items recorded in the Round 3 receipt (`docs/ice/fidelity/si3_r3_d5_receipt.md:247-266`) one statement at a time, each measured on the full `8,760`-step (`621,960` stage-field row) C1D year under **compiled `jax.jit`** (matching `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1836`) across CPU production JIT (`XLA_FLAGS=""`), the CPU no-FMA diagnostic (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`), and an idle GPU spot-check (`CUDA_VISIBLE_DEVICES=0`, `Tesla V100S-PCIE-32GB`):

* **Headline hashes & counts (both reported explicitly):**
  * **Remap (`r5_step1_ice_remap`) and remap + density reciprocal (`r5_step1_plus_step2_density_reciprocal`) intermediate arms:** CPU production JIT (`XLA_FLAGS=""`) & CPU `--xla_cpu_max_isa=AVX` are **bitwise identical** (`CPU == AVX`) at **`663` over-bar / `561,810` at `0` ULP** (`60,150` nonzero-ULP; `34` normal-step over-bar + `629` stated platform floor `kt = 5842..5867`) with full-year output SHA-256 **`732613e9505e0a2a396fce90532e18942f5bbde545fa5baf840308d0474a1737`** (GPU spot-check: **`34` over-bar / `562,185` at `0` ULP**, SHA-256 `51c47f486cb200509a24c6b332a599446956c315c072bc711b52a6138719ee5b`).
  * **Shipped final arm (`r5_all_four_items`, all four items):** CPU production JIT (`XLA_FLAGS=""`) & CPU `--xla_cpu_max_isa=AVX` are **bitwise identical** (`CPU == AVX`) at **`663` over-bar / `567,586` at `0` ULP** (`54,374` nonzero-ULP; `34` normal-step over-bar + `629` stated platform floor `kt = 5842..5867`) with full-year output SHA-256 **`62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`** (GPU spot-check: **`34` over-bar / `567,973` at `0` ULP**, SHA-256 `482f7bae496cb7ffacec1d244965533839f86a30586567cb1a0ab4add3004de2`).
* **Round 3 baseline (`965`) vs in-run `r3_baseline` comparison arm (`953` on CPU production, `965` on AVX):**
  * The **`965` over-bar / `521,368` at `0` ULP** (`336` normal over-bar, SHA-256 `389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf`) baseline is Round 3's landed commit (`a572f4b2540351bf49328d1533211427f9c9d772`), where `_piecewise_remap` carried the non-NEMO `_nemo_source_round` insertion.
  * In this Round 5 run, removing that old rounding from `_piecewise_remap` moves the in-run Round 3 comparison arm (`r3_baseline`, `_nemo_ice_remap=False`) to **`953` over-bar / `521,432` at `0` ULP / `100,528` nonzero-ULP** (`324` normal over-bar, SHA-256 `65d5669690b9af43cc90e6ee2182d23915b45f46fbaff4efba7ec4ed160f5351`) on CPU production (`XLA_FLAGS=""`), while `--xla_cpu_max_isa=AVX` stays at **`965` over-bar / `521,368` at `0` ULP / `100,592` nonzero-ULP** (`336` normal over-bar, SHA-256 `389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf`).

1. **Statement 1 — Literal transcription of NEMO's ice-layer vertical remapping `ice_var_vremap` (`src/ICE/icevar.F90:957-1034`, compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:971-1048`, called from `src/ICE/icethd_dh.F90:511` / compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:521`):**
   * Transcribed in `_nemo_ice_enthalpy_remap` (`packages/ice/legoesm/ice/bitz_lipscomb.py:980-1011`, called at `packages/ice/legoesm/ice/bitz_lipscomb.py:1249-1252` under `_nemo_ice_remap: bool = True`), replacing the overlap-sum `_piecewise_remap` (`packages/ice/legoesm/ice/bitz_lipscomb.py:980-1012`) on the SI3 ice-enthalpy path and removing the non-NEMO `_nemo_source_round` insertion from `_piecewise_remap`.
   * **Effect on full-year C1D (`jax.jit`):**
     * **CPU production JIT (`XLA_FLAGS=""`) & CPU `--xla_cpu_max_isa=AVX` (bitwise identical full-year output SHA-256 `732613e9505e0a2a396fce90532e18942f5bbde545fa5baf840308d0474a1737`):** Over-bar rows drop from **`965` (Round 3 landed / AVX `r3_baseline`) and `953` (in-run CPU production `r3_baseline`) to `663` (`-302` vs Round 3 landed / AVX; `-290` vs in-run CPU production `r3_baseline`)**; normal-step over-bar rows outside `kt = 5842..5867` drop from **`336` (`324` in-run CPU) to `34` (`-89.9%`)**, closing **all `29 × 4 = 116` normal-step over-bar rows** in `POST_DH.e_i`, `POST_TEMP1.e_i`, `POST_SAL.e_i`, and `POST_TEMP2.e_i` to **`0`**, and reducing `POST_DO.e_i` and `EXIT.e_i` normal-step over-bar rows from `99` (`93` in-run CPU) each to **`6` each** (`kt = 3996, 4014, 4092, 4400, 4413, 5241`, `max_norm = 1.062677e-15`, `max_ulp = 10`). `0`-ULP rows increase from **`521,368` (`521,432` in-run CPU) to `561,810` (`+40,442` vs Round 3 landed / AVX; `+40,378` vs in-run CPU production)**, with nonzero-ULP rows dropping from `100,592` (`100,528` in-run CPU) to `60,150`.
     * **GPU spot-check (`CUDA_VISIBLE_DEVICES=0`):** Over-bar rows drop from **`336` to `34` (`-302` over-bar rows)**; `0`-ULP rows increase from **`521,679` to `562,185` (`+40,506` new `0`-ULP rows**; nonzero-ULP rows drop from `100,281` to `59,775`; full-year output SHA-256 `51c47f486cb200509a24c6b332a599446956c315c072bc711b52a6138719ee5b`).
2. **Statement 2 — Ice density reciprocal `r1_rhoi = 1.0 / constants.rho_ice` in `_dh_step` (`src/ICE/icethd_dh.F90:254,391`, compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:264,401`) and correction of the Round 3 receipt's misquoted line (`docs/ice/fidelity/si3_r3_d5_receipt.md:109-110`):**
   * Transcribed in `_dh_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1039,1125-1132,1182-1190,1205-1209` under `_nemo_dh_ice_density_reciprocal: bool = True`), evaluating `ice_specific = sr(-zdf.e_ice[..., k] * inverse_ice_density)` instead of dividing by `constants.rho_ice`.
   * Corrected `docs/ice/fidelity/si3_r3_d5_receipt.md:109-110`, which previously quoted NEMO's `zEi = - e_i_1d(ji,jk) * r1_rhoi` while Round 3's `_dh_step` actually executed `-zdf.e_ice[..., k] / constants.rho_ice`.
   * **Effect on full-year C1D (`jax.jit`):**
     * Against true IEEE-754 division (`_nemo_dh_ice_density_reciprocal=False`, which uses `jax.lax.optimization_barrier` on `constants.rho_ice` so XLA's `AlgebraicSimplifier` cannot rewrite `/ 917.0` into `* (1.0 / 917.0)`), `-e_i * r1_rhoi` differs from `-e_i / rho_ice` by `1` ULP on **`4,969` of `8,760` steps** (`4,958` steps for `zdE = zEi - zEw`, `1,535` active melt steps), and changes **`6` stage-field rows** at `kt = 4893` (`POST_DH.e_i`, `POST_TEMP1.e_i`, `POST_SAL.e_i`, `POST_TEMP2.e_i` from `4` ULP to `2` ULP vs oracle, and `POST_DO.e_i`, `EXIT.e_i` from `2` ULP to `1` ULP vs oracle; full-year SHA-256 `828693712a1c21945b878bf6dc2eaf3deb460d3d3a769ef3ead19059a76ef261` -> `732613e9505e0a2a396fce90532e18942f5bbde545fa5baf840308d0474a1737` on CPU/AVX, and `5862a292fabb72b15d8248822b4590bd50acb862bbbd407eef5eebb59664066b` -> `51c47f486cb200509a24c6b332a599446956c315c072bc711b52a6138719ee5b` on GPU).
     * Relative to Round 3's compiled `jax.jit` baseline (where XLA's `AlgebraicSimplifier` already folded `/ 917.0` into `* (1.0 / 917.0)`), full-year aggregate counts are unchanged at **`663` over-bar / `561,810` `0`-ULP** on CPU/AVX and **`34` over-bar / `562,185` `0`-ULP** on GPU, while making the reciprocal multiplication explicit in source (`eager` == `jax.jit`).
3. **Statement 3 — Option-2 salinity profile bulk-salinity factoring (`src/ICE/icevar.F90:504-505,575-576`, compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:518-519,589-590`):**
   * Transcribed in `option2_salinity_profile` (`packages/ice/legoesm/ice/bitz_lipscomb.py:163-211`, called at `packages/ice/legoesm/ice/bitz_lipscomb.py:1750-1755` under `_nemo_salprof_factoring: bool = True`), evaluating `sr(sr(sr(sr(alpha[..., None] * S_bulk[..., None]) * 2.0) * jk_minus_half) * r1_nlay_i) + sr((1.0 - alpha[..., None]) * S_bulk[..., None])` instead of `S_bulk[..., None] * (sr(alpha[..., None] * z) + (1.0 - alpha[..., None]))`. (Note: in `src/ICE/icevar.F90`, the 1-D expression is at `:575-576` and the 2-D expression is at `:504-505`, 4 lines below lines 500–501 due to the guard/comment lines at `:499-503`.)
   * **Effect on full-year C1D (`jax.jit`):**
     * **CPU production JIT (`XLA_FLAGS=""`) & CPU `--xla_cpu_max_isa=AVX` (bitwise identical full-year output SHA-256 `62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`):** Over-bar rows remain **`663`** (`34` normal + `629` floor), and `0`-ULP rows increase from **`561,810` to `567,586` (`+5,776` new `0`-ULP rows**; nonzero-ULP rows drop from `60,150` to `54,374`, and normal-step nonzero-ULP rows drop from `59,279` to `53,555`):
       * `POST_SAL.sz_i` normal-step `0`-ULP rows increase from `6,832` to **`8,727` of `8,734` (`+1,895` normal + `+26` subnormal = `+1,921` rows**, leaving only `7` nonzero-ULP normal steps!).
       * `POST_TEMP2.sz_i` normal-step `0`-ULP rows increase from `6,832` to **`8,727` of `8,734` (`+1,921` rows**, leaving only `7` nonzero-ULP normal steps!).
       * `POST_DO.szv_i` and `EXIT.szv_i` `0`-ULP rows each increase by **`+967` rows** (`988` moved to `0` ULP, `21` moved from `0` ULP; normal nonzero-ULP rows drop from `3,032` to `2,065` each).
     * **GPU spot-check (`CUDA_VISIBLE_DEVICES=0`):** Over-bar rows remain **`34`**, and `0`-ULP rows increase from **`562,185` to `567,973` (`+5,788` new `0`-ULP rows**; nonzero-ULP rows drop from `59,775` to `53,987`; full-year output SHA-256 `482f7bae496cb7ffacec1d244965533839f86a30586567cb1a0ab4add3004de2`).
4. **Statement 4 — Harness `_nemo_salprof` transition constant naming (`zsi1 = 4.5, zsi0 = 3.5`) and missing-SSS refusal (`GateError`):**
   * Renamed `rn_sal_gd: float = 4.5, rn_sal_fl: float = 3.5` in `_nemo_salprof` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-249`) to `zsi1: float = 4.5, zsi0: float = 3.5` (`src/ICE/icevar.F90:471-472,546-547`), and fixed the inline comment line citations from lines 495/496 to `src/ICE/icevar.F90:499-500`.
   * Removed the silent `sss_m = 34.0` default from `_nemo_salprof` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-249`), `_entry_arrays` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:252-286`), and `_exact_1d_entry` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:717-732`), and added explicit `require(...)` checks in `_nemo_salprof`, `_entry_arrays`, `_exact_1d_entry`, and `_forcing` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:289-301`, `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:705-714`) plus `--plant-missing-sss` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1802,1898-1904,2161-2179`) so the gates refuse with `GateError` whenever sea-surface salinity is absent or non-finite.
   * **Effect on full-year C1D (`jax.jit`):** Bit-identical to Statement 3 when valid `sss_m=exact["sss"]` is supplied (**`663` over-bar / `567,586` `0`-ULP**, SHA `62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320` on CPU/AVX; **`34` over-bar / `567,973` `0`-ULP**, SHA `482f7bae496cb7ffacec1d244965533839f86a30586567cb1a0ab4add3004de2` on GPU), and raises `GateError` when `sss_m` or `"sss"` is absent.

---

## 2. Provenance & Evidence Artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| C1D V2 thermo stream | `/data/abyssal/dbalwada/nemo-testcases-l2/phase2b_c1d_v2/round9_scalarmath_v2/oracle_si3_thd_frames.bin` | `51d4cb2122ad9accdbc3bbde75c813fa2bd4220f08f891cd69745f7b405a61f6` |
| C1D V2 ZDF input stream | `/data/abyssal/dbalwada/nemo-testcases-l2/phase2b_c1d_v2/round9_scalarmath_v2/oracle_si3_zdf_inputs.bin` | `099fd7b863e3a324d8ac692fb9fb8c9f8b55d69ea62c7be96574c048c0ab3fc3` |
| CPU production JIT probe (`XLA_FLAGS=""`) | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round5/d5_platform_probe_cpu.json` | `6b7ee3fceeb7e4e8000f6bdac46c1f04d78b31cf85d43bcd47f4323c391b65a0` |
| CPU diagnostic no-FMA probe (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`) | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round5/d5_platform_probe_cpu_avx_nofma.json` | `0ee86a8c6b79434d01c42b27e9207d5d09af0deef77ccb4c97d2067541bec060` |
| GPU spot-check probe (`CUDA_VISIBLE_DEVICES=0`) | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round5/d5_platform_probe_gpu.json` | `2a02863699f8f69eafbcd0efb69c10d561d4beff7058d675ceea433402e1228d` |
| GPU `nvidia-smi` before | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round5/gpu_nvidia_smi_before.txt` | `8bf8117625ecc4949dd04d5251972fb45aea968d45d028585ec6f67bd1e8f841` |
| GPU `nvidia-smi` after | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round5/gpu_nvidia_smi_after.txt` | `74bf2d4d1bb0405431e42de0c81e831aef1d537b9165be39a569ecb76f8d8bd3` |

---

## 3. Statement-by-Statement NEMO Source Mapping and Transcription

### 3.1 Statement 1: Ice-Layer Vertical Remapping `ice_var_vremap` (`src/ICE/icevar.F90:957-1034`)

* **NEMO Fortran source (`src/ICE/icevar.F90:957-1034`, compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:971-1048`, called from `src/ICE/icethd_dh.F90:511` / compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:521`):**
  ```fortran
  zts_cum0(0) = 0._wp 
  zh_cum0 (0) = 0._wp
  DO jk0 = 1, nlay_i+2
     zts_cum0(jk0) = zts_cum0(jk0-1) + pts_old(jk0-1)
     zh_cum0 (jk0) = zh_cum0 (jk0-1) + ph_old(jk0-1)
  END DO

  zhnew = SUM( ph_old(0:nlay_i+1) ) * r1_nlay_i  

  zh_cum1(0) = 0._wp
  DO jk1 = 1, nlay_i
     zh_cum1(jk1) = zh_cum1(jk1-1) + zhnew
  END DO

  zts_cum1(0:nlay_i) = 0._wp 
  DO jk0 = 1, nlay_i+2
     DO jk1 = 1, nlay_i-1
        IF( zh_cum1(jk1) <= zh_cum0(jk0) .AND. zh_cum1(jk1) > zh_cum0(jk0-1) )   THEN
           zts_cum1(jk1) = ( zts_cum0(jk0-1) * ( zh_cum0(jk0) - zh_cum1(jk1  ) ) +  &
              &              zts_cum0(jk0  ) * ( zh_cum1(jk1) - zh_cum0(jk0-1) ) )  &
              &            / ( zh_cum0(jk0) - zh_cum0(jk0-1) )
        ENDIF
     END DO
  END DO
  zts_cum1(nlay_i) = zts_cum0(nlay_i+2) 

  DO jk1 = 1, nlay_i
     pts_i(jk1) = MAX( 0._wp, zts_cum1(jk1) - zts_cum1(jk1-1) ) / MAX( zhnew, epsi20 )
  END DO
  ```
* **legoESM transcription (`packages/ice/legoesm/ice/bitz_lipscomb.py:980-1011`, called at `packages/ice/legoesm/ice/bitz_lipscomb.py:1249-1252`):**
  * `_nemo_ice_enthalpy_remap` accumulates `old_h_cumulative` (`zh_cum0(0:5)`) and `old_e_cumulative` (`zts_cum0(0:5)` from `pts_old = sr(thickness * enthalpy)`), computes `new_layer = total * (1.0 / 3.0)` (`zhnew = SUM(ph_old) * r1_nlay_i`) and `new_h_cumulative` (`zh_cum1(0:3)`), interpolates `new_e_cumulative[..., 1:3]` with `nemo_source_round` on both two-product terms of `zts_cum0(jk0-1) * (zh_cum0(jk0) - zh_cum1(jk1)) + zts_cum0(jk0) * (zh_cum1(jk1) - zh_cum0(jk0-1))` before dividing by `zh_cum0(jk0) - zh_cum0(jk0-1)`, sets `new_e_cumulative[..., 3] = old_e_cumulative[..., 5]`, and divides `jnp.maximum(0.0, new_e_cumulative[..., layer] - new_e_cumulative[..., layer - 1]) / jnp.maximum(new_layer, _EPS20)`.
  * `_piecewise_remap` (`packages/ice/legoesm/ice/bitz_lipscomb.py:980-1012`) is restored to its unrounded form without `_nemo_source_round`.

### 3.2 Statement 2: Ice Density Reciprocal `r1_rhoi` in `_dh_step` (`src/ICE/icethd_dh.F90:254,391`) & Round 3 Receipt Correction

* **NEMO Fortran source (`src/ICE/icethd_dh.F90:254-256,391-393`, compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:264-266,401-403`):**
  ```fortran
  zEi            = - e_i_1d(ji,jk) * r1_rhoi             ! Specific enthalpy of layer k [J/kg, <0]
  zEw            =    rcp * ztmelts                      ! Specific enthalpy of resulting meltwater [J/kg, <0]
  zdE            =    zEi - zEw                          ! Specific enthalpy difference < 0
  ```
* **legoESM transcription (`packages/ice/legoesm/ice/bitz_lipscomb.py:1039,1125-1132,1182-1190,1205-1209`):**
  * Defines `inverse_ice_density = 1.0 / constants.rho_ice` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1039`) and evaluates `ice_specific = sr(-zdf.e_ice[..., k] * inverse_ice_density)` under `_nemo_dh_ice_density_reciprocal: bool = True` in both the surface ice melt loop (`:1125-1132`) and both basal ice melt loops (`:1205-1209` and `:1182-1190`).
  * Corrected `docs/ice/fidelity/si3_r3_d5_receipt.md:109-110` to state explicitly that Round 3's `_dh_step` wrote `-zdf.e_ice[..., k] / constants.rho_ice` whereas NEMO multiplies by `r1_rhoi` (transcribed in Round 5).

### 3.3 Statement 3: Option-2 Salinity Profile Bulk-Salinity Factoring (`src/ICE/icevar.F90:504-505,575-576`)

* **NEMO Fortran source (`src/ICE/icevar.F90:504-505,575-576`, compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:518-519,589-590`):**
  ```fortran
  DO jk = 1, nlay_i
     sz_i_1d(ji,jk) = zalpha * s_i_1d(ji) * 2._wp * ( REAL(jk,wp) - 0.5_wp ) * r1_nlay_i + ( 1._wp - zalpha ) * s_i_1d(ji)
  END DO
  ```
* **legoESM transcription (`packages/ice/legoesm/ice/bitz_lipscomb.py:163-211`, called at `packages/ice/legoesm/ice/bitz_lipscomb.py:1750-1755`):**
  * Under `_nemo_salprof_factoring: bool = True`, `option2_salinity_profile` evaluates:
    ```python
    jk_minus_half = jnp.asarray((0.5, 1.5, 2.5), dtype=S_bulk.dtype)
    r1_nlay_i = jnp.asarray(1.0 / 3.0, dtype=S_bulk.dtype)
    sloped = sr(
        sr(
            sr(sr(alpha[..., None] * S_bulk[..., None]) * 2.0)
            * jk_minus_half
        )
        * r1_nlay_i
    )
    constant = sr((1.0 - alpha[..., None]) * S_bulk[..., None])
    return sloped + constant
    ```

### 3.4 Statement 4: Harness `zsi1`/`zsi0` Naming and Missing-SSS Refusal

* **NEMO Fortran source (`src/ICE/icevar.F90:471-472,499-500,546-547`):**
  * Option-2 (`nn_icesal == 2`) uses local constants `zsi0 = 3.5_wp, zsi1 = 4.5_wp` (`src/ICE/icevar.F90:471-472,546-547`), whereas namelist parameters `rn_sal_gd = 5.0, rn_sal_fl = 2.0` belong to option 3.
* **Harness transcription (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-301`, `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:705-752,1802,1898-1904,2161-2179`):**
  * Renamed `rn_sal_gd: float = 4.5, rn_sal_fl: float = 3.5` in `_nemo_salprof` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-249`) to `zsi1: float = 4.5, zsi0: float = 3.5`, and updated inline comments from lines 495/496 to `src/ICE/icevar.F90:499-500`.
  * Replaced `sss_m = 34.0` default with `sss_m: np.ndarray | float | None = None` and added explicit `require(sss_m is not None, "missing sea-surface salinity (sss_m)")` and finite checks in `_nemo_salprof`, `_entry_arrays`, `_exact_1d_entry`, and `_forcing`, plus `--plant-missing-sss` in `nemo_si3thd_phase2b_year_gate.py`.

---

## 4. Full-Year (`8,760`-Step, `621,960`-Row) C1D Rescore Under `jax.jit`

### 4.1 Progression Across Ordered Statements (`run_year_rescore`, `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883`)

| Evaluation Arm (`jax.jit`) | CPU Production (`XLA_FLAGS=""`) Over-bar / `0` ULP / Nonzero ULP (Normal over-bar; Full-Year SHA-256) | CPU Diagnostic (`--xla_cpu_max_isa=AVX`) Over-bar / `0` ULP / Nonzero ULP (Normal over-bar; Full-Year SHA-256) | GPU Spot-Check (`CUDA_VISIBLE_DEVICES=0`) Over-bar / `0` ULP / Nonzero ULP (Normal over-bar; Full-Year SHA-256) |
|---|---:|---:|---:|
| **Round 3 landed baseline (`a572f4b2540351bf49328d1533211427f9c9d772`, before Statement 1)** | **`965` / `521,368` / `100,592`**<br>(`336`; `389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf`) | **`965` / `521,368` / `100,592`**<br>(`336`; `389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf`) | **`336` / `521,679` / `100,281`**<br>(`336`; `46b20957b001c3b71e8e43befd70e3f1f03d2096068a4f26ac583948fc192e24`) |
| *In-run `r3_baseline` comparison arm (`_nemo_ice_remap=False`, old rounding removed from `_piecewise_remap`)* | **`953` / `521,432` / `100,528`**<br>(`324`; `65d5669690b9af43cc90e6ee2182d23915b45f46fbaff4efba7ec4ed160f5351`) | **`965` / `521,368` / `100,592`**<br>(`336`; `389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf`) | **`336` / `521,679` / `100,281`**<br>(`336`; `46b20957b001c3b71e8e43befd70e3f1f03d2096068a4f26ac583948fc192e24`) |
| **After Statement 1 (`r5_step1_ice_remap`, `src/ICE/icevar.F90:957-1034`)** | **`663` / `561,810` / `60,150`**<br>(**`34`; `732613e9505e0a2a396fce90532e18942f5bbde545fa5baf840308d0474a1737`**) | **`663` / `561,810` / `60,150`**<br>(**`34`; `732613e9505e0a2a396fce90532e18942f5bbde545fa5baf840308d0474a1737`**) | **`34` / `562,185` / `59,775`**<br>(**`34`; `51c47f486cb200509a24c6b332a599446956c315c072bc711b52a6138719ee5b`**) |
| *Statement 2 true-`fdiv` control (`r5_step1_ice_remap_true_fdiv`, `/ rho_ice`)* | `663` / `561,810` / `60,150`<br>(`34`; `828693712a1c21945b878bf6dc2eaf3deb460d3d3a769ef3ead19059a76ef261`) | `663` / `561,810` / `60,150`<br>(`34`; `828693712a1c21945b878bf6dc2eaf3deb460d3d3a769ef3ead19059a76ef261`) | `34` / `562,185` / `59,775`<br>(`34`; `5862a292fabb72b15d8248822b4590bd50acb862bbbd407eef5eebb59664066b`) |
| **After Statement 1 + 2 (`r5_step1_plus_step2_density_reciprocal`, `src/ICE/icethd_dh.F90:254,391`)** | **`663` / `561,810` / `60,150`**<br>(**`34`; `732613e9505e0a2a396fce90532e18942f5bbde545fa5baf840308d0474a1737`**) | **`663` / `561,810` / `60,150`**<br>(**`34`; `732613e9505e0a2a396fce90532e18942f5bbde545fa5baf840308d0474a1737`**) | **`34` / `562,185` / `59,775`**<br>(**`34`; `51c47f486cb200509a24c6b332a599446956c315c072bc711b52a6138719ee5b`**) |
| **After Statement 1 + 2 + 3 (`r5_step1_plus_step2_plus_step3_salprof`, `src/ICE/icevar.F90:504-505,575-576`)** | **`663` / `567,586` / `54,374`**<br>(**`34`; `62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`**) | **`663` / `567,586` / `54,374`**<br>(**`34`; `62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`**) | **`34` / `567,973` / `53,987`**<br>(**`34`; `482f7bae496cb7ffacec1d244965533839f86a30586567cb1a0ab4add3004de2`**) |
| **After Statement 1 + 2 + 3 + 4 (`r5_all_four_items`, shipped final Round 5)** | **`663` / `567,586` / `54,374`**<br>(**`34`; `62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`**) | **`663` / `567,586` / `54,374`**<br>(**`34`; `62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`**) | **`34` / `567,973` / `53,987`**<br>(**`34`; `482f7bae496cb7ffacec1d244965533839f86a30586567cb1a0ab4add3004de2`**) |
| *Isolated Statement 3 from `r3_baseline` (`r5_isolated_step3_salprof`)* | `953` / `527,208` / `94,752`<br>(`324`; `ebd8cfd237aa01242707fcdc3420eadcc777e647234cd3cae1aa37b63b6f766f`) | `965` / `527,144` / `94,816`<br>(`336`; `3910f8cdc09f2501fe51df7da93c2cbc0729b26ddacd48010817d4ebb5341933`) | `336` / `527,467` / `94,493`<br>(`336`; `01b640dc10a794f1c6d80436a6785bae15b7e392dd09cac2b69951f349323aec`) |

*Note on `965` (Round 3 landed baseline) vs `953` (in-run CPU production `r3_baseline` comparison arm):* The `965` over-bar / `521,368` `0`-ULP (`389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf`) baseline is Round 3's landed commit (`a572f4b2540351bf49328d1533211427f9c9d772`), where `_piecewise_remap` had the non-NEMO `_nemo_source_round` insertion. In this Round 5 run, removing that old rounding from `_piecewise_remap` moves the in-run `r3_baseline` comparison arm (`_nemo_ice_remap=False`) to **`953` over-bar / `521,432` `0`-ULP** (`324` normal over-bar, SHA `65d5669690b9af43cc90e6ee2182d23915b45f46fbaff4efba7ec4ed160f5351`) on CPU production (`XLA_FLAGS=""`), while `--xla_cpu_max_isa=AVX` stays at **`965` over-bar / `521,368` `0`-ULP** (`389490e20dbd86cb923cd693933487f55f8f4a05bb052888393103eb32d6ebdf`). On all Round 5 `_nemo_ice_remap=True` arms (`r5_step1_ice_remap` and `r5_step1_plus_step2_density_reciprocal` at SHA `732613e9505e0a2a396fce90532e18942f5bbde545fa5baf840308d0474a1737`, and `r5_step1_plus_step2_plus_step3_salprof` and shipped final `r5_all_four_items` at SHA `62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`), CPU production JIT (`XLA_FLAGS=""`) and `--xla_cpu_max_isa=AVX` produce **bitwise identical full-year output SHA-256 digests (`CPU == AVX`)**.

### 4.2 Stage-by-Stage Progression Under Compiled `jax.jit` (CPU Production JIT & `--xla_cpu_max_isa=AVX`)

| Stage (`rows`) | Round 3 Landed (`a572f4b254` / AVX `r3_baseline`) Over-bar / `0` ULP | In-Run CPU `r3_baseline` (`XLA_FLAGS=""`) Over-bar / `0` ULP | After Statement 1 & 1+2 (`r5_step1_ice_remap`, SHA `732613e9...`) Over-bar / `0` ULP | Shipped Final Arm (`r5_all_four_items`, SHA `62280a6d...`) Over-bar / `0` ULP | Net Over-bar Delta (`R3 -> R5`, Landed / In-Run CPU) | Net `0`-ULP Delta (`R3 -> R5`, Landed / In-Run CPU) |
|---|---:|---:|---:|---:|---:|---:|
| `ENTRY` (`105,120`) | `0` / `85,157` | `0` / `85,157` | `0` / `85,157` | `0` / `85,157` | `0` / `0` | `0` / `0` |
| `POST_ZDF` (`61,320`) | `75` / `61,245` | `75` / `61,245` | `75` / `61,245` | `75` / `61,245` | `0` / `0` | `0` / `0` |
| `POST_DH` (`61,320`) | `106` / `50,209` | `106` / `50,224` | **`77` / `57,775`** | **`77` / `57,775`** | **`-29` / `-29`** | **`+7,566` / `+7,551`** |
| `POST_TEMP1` (`61,320`) | `106` / `50,209` | `106` / `50,224` | **`77` / `57,775`** | **`77` / `57,775`** | **`-29` / `-29`** | **`+7,566` / `+7,551`** |
| `POST_SAL` (`61,320`) | `106` / `48,281` | `106` / `48,296` | **`77` / `55,847`** | **`77` / `57,768`** | **`-29` / `-29`** | **`+9,487` / `+9,472`** |
| `POST_TEMP2` (`61,320`) | `106` / `48,281` | `106` / `48,296` | **`77` / `55,847`** | **`77` / `57,768`** | **`-29` / `-29`** | **`+9,487` / `+9,472`** |
| `POST_DO` (`105,120`) | `233` / `88,993` | `227` / `88,995` | **`140` / `94,082`** | **`140` / `95,049`** | **`-93` / `-87`** | **`+6,056` / `+6,054`** |
| `EXIT` (`105,120`) | `233` / `88,993` | `227` / `88,995` | **`140` / `94,082`** | **`140` / `95,049`** | **`-93` / `-87`** | **`+6,056` / `+6,054`** |
| **Total (`621,960`)** | **`965` / `521,368`** | **`953` / `521,432`** | **`663` / `561,810`** | **`663` / `567,586`** | **`-302` / `-290`** | **`+46,218` / `+46,154`** |

---

## 5. Remaining Owner Ranking After Each Statement

### 5.1 Evolution of Normal-Step Over-Bar Owners Across Statements 1–4 (`kt != 5842..5867`)

| Stage.Field Owner | Before Statement 1 (Round 3 Landed `a572f4b254` / AVX `r3_baseline`, Over-bar / Nonzero ULP) | In-Run CPU `r3_baseline` (`XLA_FLAGS=""`, Over-bar / Nonzero ULP) | After Statement 1 & 1+2 (`r5_step1_ice_remap`, SHA `732613e9...`, Over-bar / Nonzero ULP) | Shipped Final Arm (`r5_all_four_items`, SHA `62280a6d...`, Over-bar / Nonzero ULP) |
|---|---:|---:|---:|---:|
| `POST_DO.e_i` | `99` / `8,206` | `93` / `8,204` | **`6` / `3,127`** | **`6` / `3,127`** |
| `EXIT.e_i` | `99` / `8,206` | `93` / `8,204` | **`6` / `3,127`** | **`6` / `3,127`** |
| `POST_DH.e_i` | `29` / `8,269` | `29` / `8,254` | **`0` / `703`** | **`0` / `703`** |
| `POST_TEMP1.e_i` | `29` / `8,269` | `29` / `8,254` | **`0` / `703`** | **`0` / `703`** |
| `POST_SAL.e_i` | `29` / `8,269` | `29` / `8,254` | **`0` / `703`** | **`0` / `703`** |
| `POST_TEMP2.e_i` | `29` / `8,269` | `29` / `8,254` | **`0` / `703`** | **`0` / `703`** |
| `POST_DO.e_s` | `7` / `147` | `7` / `147` | `7` / `147` | `7` / `147` |
| `EXIT.e_s` | `7` / `147` | `7` / `147` | `7` / `147` | `7` / `147` |
| `POST_DH.e_s` | `2` / `137` | `2` / `137` | `2` / `137` | `2` / `137` |
| `POST_TEMP1.e_s` | `2` / `137` | `2` / `137` | `2` / `137` | `2` / `137` |
| `POST_SAL.e_s` | `2` / `137` | `2` / `137` | `2` / `137` | `2` / `137` |
| `POST_TEMP2.e_s` | `2` / `137` | `2` / `137` | `2` / `137` | `2` / `137` |
| `POST_SAL.sz_i` | `0` / `1,902` | `0` / `1,902` | `0` / `1,902` | **`0` / `7`** |
| `POST_TEMP2.sz_i` | `0` / `1,902` | `0` / `1,902` | `0` / `1,902` | **`0` / `7`** |
| `POST_DO.szv_i` | `0` / `3,032` | `0` / `3,032` | `0` / `3,032` | **`0` / `2,065`** |
| `EXIT.szv_i` | `0` / `3,032` | `0` / `3,032` | `0` / `3,032` | **`0` / `2,065`** |

### 5.2 Complete Ranked Remaining Owners After Statement 4 (`r5_all_four_items`, `remaining_owner_ranking_after_r5` in `d5_platform_probe_cpu.json` & `d5_platform_probe_gpu.json`)

| Rank | Owner / `Stage.field` | CPU Over-Bar Rows | GPU Over-Bar Rows | CPU Nonzero-ULP Rows | GPU Nonzero-ULP Rows | Max ULP (CPU) | Max Normalised Error (CPU) | Worst Step & Field (`kt`, `Stage.field`) | Physical / Arithmetic Cause |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `D5_PLATFORM_FLOOR_KT5842_5867_SUBNORMAL_SNOW` | `629` | `0` | `819` | `150` | `190,081,022,848,368` | `2.510100e-02` | `kt = 5860`, `POST_ZDF.e_s` | CPU XLA LLVM `FTZ/DAZ` hardware subnormal flush in `kt = 5842..5867` (DECISION 158 stated platform floor; `0` over-bar on GPU) |
| 2 | `POST_DO.e_s` | `7` | `7` | `147` | `139` | `16` | `2.122689e-15` | `kt = 5460`, `POST_DO.e_s` | Already `1`–`12` ULP per layer at `POST_TEMP2` from snow melt / snow-ice path (`kt = 4943, 4945, 5460, 5644, 5822, 5960, 5962`); conversion to NEMO whole-cell content (`src/ICE/icethd.F90:445`) pushes over `1e-15` bar |
| 3 | `EXIT.e_s` | `7` | `7` | `147` | `139` | `16` | `2.122689e-15` | `kt = 5460`, `EXIT.e_s` | Same as `POST_DO.e_s` |
| 4 | `POST_DO.e_i` | `6` | `6` | `3,127` | `3,075` | `10` | `1.062677e-15` | `kt = 5241`, `POST_DO.e_i` | Already `3`–`7` ULP per layer at `POST_TEMP2` (plausible owner: `ze_i_old` accumulation in `src/ICE/icethd_dh.F90:134,284,306,361,421,476,482`; `kt = 3996, 4014, 4092, 4400, 4413, 5241`); conversion to NEMO whole-cell content (`src/ICE/icethd.F90:442`) pushes over `1e-15` bar |
| 5 | `EXIT.e_i` | `6` | `6` | `3,127` | `3,075` | `10` | `1.062677e-15` | `kt = 5241`, `EXIT.e_i` | Same as `POST_DO.e_i` |
| 6 | `POST_DH.e_s` | `2` | `2` | `137` | `129` | `2,563,146,750,041,342,108` | `1.614537e-15` | `kt = 5460`, `POST_DH.e_s` | Snow-melt / snow-ice enthalpy update in `_dh_step` (`kt = 5379, 5460`) |
| 7 | `POST_TEMP1.e_s` | `2` | `2` | `137` | `129` | `2,563,146,750,041,342,108` | `1.614537e-15` | `kt = 5460`, `POST_TEMP1.e_s` | Inherited from `POST_DH.e_s` |
| 8 | `POST_SAL.e_s` | `2` | `2` | `137` | `129` | `2,563,146,750,041,342,108` | `1.614537e-15` | `kt = 5460`, `POST_SAL.e_s` | Inherited from `POST_DH.e_s` |
| 9 | `POST_TEMP2.e_s` | `2` | `2` | `137` | `129` | `2,563,146,750,041,342,108` | `1.614537e-15` | `kt = 5460`, `POST_TEMP2.e_s` | Inherited from `POST_DH.e_s` |
| 10 | `ENTRY.e_i` | `0` | `0` | `5,447` | `5,447` | `3` | `4.673354e-16` | `kt = 7628`, `ENTRY.e_i` | Global-to-1D-to-global round-trip `(e_i / v_i * 3) * h_i * a_i * (1/3)` (`<= 3` ULP, all at-bar) |
| 11 | `ENTRY.e_s` | `0` | `0` | `4,489` | `4,489` | `3` | `4.349022e-16` | `kt = 1914`, `ENTRY.e_s` | Global-to-1D-to-global round-trip `(e_s / v_s * 3) * h_s * a_i * (1/3)` (`<= 3` ULP, all at-bar) |
| 12 | `ENTRY.v_i` | `0` | `0` | `2,626` | `2,626` | `1` | `2.219566e-16` | `kt = 7937`, `ENTRY.v_i` | Global-to-1D-to-global `(v_i / a_i) * a_i` (`1` ULP, all at-bar) |
| 13 | `ENTRY.v_s` | `0` | `0` | `2,526` | `2,526` | `1` | `5.551115e-17` | `kt = 859`, `ENTRY.v_s` | Global-to-1D-to-global `(v_s / a_i) * a_i` (`1` ULP, all at-bar) |
| 14 | `POST_DH.h_i` | `0` | `0` | `2,524` | `2,524` | `3` | `4.245777e-16` | `kt = 4359`, `POST_DH.h_i` | `icethd_dh.F90` incremental layer thickness sum (`<= 3` ULP, all at-bar) |
| 15 | `POST_TEMP1.h_i` | `0` | `0` | `2,524` | `2,524` | `3` | `4.245777e-16` | `kt = 4359`, `POST_TEMP1.h_i` | Inherited from `POST_DH.h_i` (`<= 3` ULP, all at-bar) |
| 16 | `POST_SAL.h_i` | `0` | `0` | `2,524` | `2,524` | `3` | `4.245777e-16` | `kt = 4359`, `POST_SAL.h_i` | Inherited from `POST_DH.h_i` (`<= 3` ULP, all at-bar) |
| 17 | `POST_TEMP2.h_i` | `0` | `0` | `2,524` | `2,524` | `3` | `4.245777e-16` | `kt = 4359`, `POST_TEMP2.h_i` | Inherited from `POST_DH.h_i` (`<= 3` ULP, all at-bar) |
| 18 | `ENTRY.sv_i` | `0` | `0` | `2,481` | `2,481` | `2` | `3.992932e-16` | `kt = 3966`, `ENTRY.sv_i` | Global-to-1D-to-global `(sv_i / v_i) * (h_i * a_i)` (`<= 2` ULP, all at-bar) |
| 19 | `ENTRY.szv_i` | `0` | `0` | `2,354` | `2,354` | `8` | `5.240056e-16` | `kt = 4368`, `ENTRY.szv_i` | Global-to-1D-to-global `_nemo_salprof(s_i) * v_i * (1/3)` (`<= 8` ULP, all at-bar) |
| 20 | `POST_DO.v_i` | `0` | `0` | `2,288` | `2,288` | `4` | `4.717530e-16` | `kt = 4359`, `POST_DO.v_i` | Inherited from `POST_DH.h_i` + `h_i * a_i` whole-cell volume write (`<= 4` ULP, all at-bar) |
| 21 | `EXIT.v_i` | `0` | `0` | `2,288` | `2,288` | `4` | `4.717530e-16` | `kt = 4359`, `EXIT.v_i` | Same as `POST_DO.v_i` (`<= 4` ULP, all at-bar) |
| 22 | `POST_DO.sv_i` | `0` | `0` | `2,213` | `2,213` | `3` | `4.538521e-16` | `kt = 4758`, `POST_DO.sv_i` | `s_i * v_i` whole-cell salt-volume write (`<= 3` ULP, all at-bar) |
| 23 | `EXIT.sv_i` | `0` | `0` | `2,213` | `2,213` | `3` | `4.538521e-16` | `kt = 4758`, `EXIT.sv_i` | Same as `POST_DO.sv_i` (`<= 3` ULP, all at-bar) |
| 24 | `POST_DO.szv_i` | `0` | `0` | `2,065` | `2,059` | `5` | `5.623522e-16` | `kt = 5070`, `POST_DO.szv_i` | `sz_i * v_i * (1/3)` global write product (`<= 5` ULP, all at-bar; reduced from `3,032` by Statement 3) |
| 25 | `EXIT.szv_i` | `0` | `0` | `2,065` | `2,059` | `5` | `5.623522e-16` | `kt = 5070`, `EXIT.szv_i` | Same as `POST_DO.szv_i` (`<= 5` ULP, all at-bar; reduced from `3,032` by Statement 3) |
| 26 | `POST_DH.e_i` | `0` | `0` | `703` | `677` | `7` | `8.216414e-16` | `kt = 5241`, `POST_DH.e_i` | Incremental `ze_i_old` accumulation (`src/ICE/icethd_dh.F90:134,284,306,361,421,476,482`; **all `0` over-bar**, reduced from `29` over-bar / `8,269` nonzero-ULP by Statement 1) |
| 27 | `POST_TEMP1.e_i` | `0` | `0` | `703` | `677` | `7` | `8.216414e-16` | `kt = 5241`, `POST_TEMP1.e_i` | Inherited from `POST_DH.e_i` (**all `0` over-bar**) |
| 28 | `POST_SAL.e_i` | `0` | `0` | `703` | `677` | `7` | `8.216414e-16` | `kt = 5241`, `POST_SAL.e_i` | Inherited from `POST_DH.e_i` (**all `0` over-bar**) |
| 29 | `POST_TEMP2.e_i` | `0` | `0` | `703` | `677` | `7` | `8.216414e-16` | `kt = 5241`, `POST_TEMP2.e_i` | Inherited from `POST_DH.e_i` (**all `0` over-bar**) |
| 30 | `POST_DH.h_s` | `0` | `0` | `81` | `81` | `4,314,448,443,020,935,169` | `5.551115e-17` | `kt = 652`, `POST_DH.h_s` | `h_s` incremental layer update (`<= 5.55e-17`, all at-bar) |
| 31 | `POST_TEMP1.h_s` | `0` | `0` | `81` | `81` | `4,314,448,443,020,935,169` | `5.551115e-17` | `kt = 652`, `POST_TEMP1.h_s` | Inherited from `POST_DH.h_s` (`<= 5.55e-17`, all at-bar) |
| 32 | `POST_SAL.h_s` | `0` | `0` | `81` | `81` | `4,314,448,443,020,935,169` | `5.551115e-17` | `kt = 652`, `POST_SAL.h_s` | Inherited from `POST_DH.h_s` (`<= 5.55e-17`, all at-bar) |
| 33 | `POST_TEMP2.h_s` | `0` | `0` | `81` | `81` | `4,314,448,443,020,935,169` | `5.551115e-17` | `kt = 652`, `POST_TEMP2.h_s` | Inherited from `POST_DH.h_s` (`<= 5.55e-17`, all at-bar) |
| 34 | `POST_DO.v_s` | `0` | `0` | `79` | `79` | `4,313,547,723,095,461,071` | `5.551115e-17` | `kt = 652`, `POST_DO.v_s` | `h_s * a_i` (`<= 5.55e-17`, all at-bar) |
| 35 | `EXIT.v_s` | `0` | `0` | `79` | `79` | `4,313,547,723,095,461,071` | `5.551115e-17` | `kt = 652`, `EXIT.v_s` | Same as `POST_DO.v_s` (`<= 5.55e-17`, all at-bar) |
| 36 | `POST_SAL.sz_i` | `0` | `0` | `7` | `7` | `6` | `5.106329e-16` | `kt = 6239`, `POST_SAL.sz_i` | Inherited from `1`-ULP `S_bulk` difference in `_dh_step` (`<= 6` ULP, all at-bar; reduced from `1,902` nonzero-ULP rows by Statement 3) |
| 37 | `POST_TEMP2.sz_i` | `0` | `0` | `7` | `7` | `6` | `5.106329e-16` | `kt = 6239`, `POST_TEMP2.sz_i` | Same as `POST_SAL.sz_i` (`<= 6` ULP, all at-bar; reduced from `1,902` nonzero-ULP rows by Statement 3) |

### 5.3 Retraction of Open-Water Growth Attribution & Recorded Next D5 Debts (Do Not Fix in Round 5)

* **Correction / Retraction of open-water growth attribution:**
  * Attributing the remaining `34` normal-step over-bar rows (`13` at `POST_DO`: `6` `POST_DO.e_i` at `kt = 3996, 4014, 4092, 4400, 4413, 5241` + `7` `POST_DO.e_s` at `kt = 4943, 4945, 5460, 5644, 5822, 5960, 5962`; `13` identical rows at `EXIT`; plus `2 × 4 = 8` 1-D snow rows `POST_DH..POST_TEMP2.e_s` at `kt = 5379, 5460`) to open-water growth (`icethd_do.F90`) is **WRONG**: open-water growth never runs in this C1D record (`qlead < 0` on **`0` of `8,760` steps** — heat flux to leads is never negative; reviewer probe `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round5/operator_review_claude/do_owner.py`).
  * All `13` over-bar rows at the growth stage `POST_DO` (`6` `e_i` + `7` `e_s`) already carry **`1`–`12` ULP per layer at `POST_TEMP2` before it** (`3`–`7` ULP on the `6` `e_i` steps and `1`–`12` ULP on the `7` `e_s` steps; reviewer probe `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round5/operator_review_claude/do_owner2.py`), and the conversion from 1-D volumetric enthalpy (`e_i_1d`, `e_s_1d`) to NEMO's whole-cell content (`e_i = e_i_1d * h_i * a_i * r1_nlay_i`, `e_s = e_s_1d * h_s * a_i * r1_nlay_s`, `src/ICE/icethd.F90:442,445`) pushes them over the `1e-15` bar.
* **Recorded Next D5 Debts (do not fix in Round 5):**
  1. **Incremental heat-content accumulation `ze_i_old` in `icethd_dh` (`src/ICE/icethd_dh.F90:134,284,306,361,421,476,482`) — PLAUSIBLE owner of the remaining normal-step ice-enthalpy rows:**
     * `_nemo_ice_enthalpy_remap` (`packages/ice/legoesm/ice/bitz_lipscomb.py:980-1011`) currently builds its input content as accumulated layer thickness times layer enthalpy (`pts_old = sr(thickness * enthalpy)`, `packages/ice/legoesm/ice/bitz_lipscomb.py:1074`), whereas NEMO accumulates the content `ze_i_old` itself one melt/growth/sublimation/flooding event at a time (`src/ICE/icethd_dh.F90:134,284,306,361,421,476,482`) and passes `ze_i_old` directly as `pts_old` to `ice_var_vremap` (`src/ICE/icethd_dh.F90:511`). This is the **PLAUSIBLE** owner of the remaining normal-step ice-enthalpy rows (`703` nonzero-ULP rows at `POST_DH..POST_TEMP2.e_i` with `0` over-bar, `max_ulp = 7`, `max_norm = 8.216414e-16` at `kt = 5241`, plus the `6` `POST_DO.e_i` and `6` `EXIT.e_i` over-bar rows at `kt = 3996, 4014, 4092, 4400, 4413, 5241`).
  2. **Snow melt / snow-ice enthalpy update path in `icethd_dh` (`src/ICE/icethd_dh.F90:204-225,441-496`) — owner of the remaining normal-step snow-enthalpy rows:**
     * The remaining normal-step snow-enthalpy over-bar rows (`2` steps `kt = 5379, 5460` across `POST_DH..POST_TEMP2.e_s` = `8` rows, worst at step `5460` with `12` ULP / `max_norm = 1.614537e-15`, plus `7` steps `kt = 4943, 4945, 5460, 5644, 5822, 5960, 5962` across `POST_DO.e_s` and `EXIT.e_s` = `14` rows, worst at step `5460` with `12` ULP / `max_norm = 2.122689e-15`) come from the snow melt / snow-ice path in `_dh_step`.

---

## 6. Unit Tests, Revert Controls, Gradients, and Non-SI3 Caller Invariance

1. **Statement 1 bit-for-bit & revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1740-1808`):**
   * Verifies via `inspect.signature(bl_mod._piecewise_remap)` that `_nemo_source_round` is removed from `_piecewise_remap`.
   * Pins `_nemo_ice_enthalpy_remap` (`_nemo_ice_remap=True`) bit-for-bit (`0` ULP, `uint64` equality) against oracle `POST_DH.e_i` on inexact C1D steps (`kt = 43, 704, 3898, 4230, 4922`), and proves that reverting `_nemo_ice_remap=False` fails bit-exactness (`>= 5` ULP) and crosses the `1e-15` bar on all five steps.
2. **Statement 2 bit-for-bit & revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1811-1900`):**
   * Pins the exact IEEE-754 `uint64` bits of `zEi = -e_i * r1_rhoi` (`0xc10ee367890ddc03`) and `zdE = zEi - zEw` (`0xc10edecdb63bb14a`) vs `-e_i / rho_ice` (`0xc10ee367890ddc02` and `0xc10edecdb63bb149`) at `kt = 4893`, and proves that reverting `_nemo_dh_ice_density_reciprocal=False` under `jax.jit` changes all 6 downstream `e_i` stage-field rows (`POST_DH..EXIT.e_i`) and increases the ULP distance from the oracle.
3. **Statement 3 bit-for-bit & revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1903-1964`):**
   * Pins `option2_salinity_profile` (`_nemo_salprof_factoring=True`) bit-for-bit (`0` ULP, `uint64` equality) against oracle `POST_SAL.sz_i` and `POST_TEMP2.sz_i` at `kt = 3961, 3963, 4413`, and proves that reverting `_nemo_salprof_factoring=False` fails bit-exactness (`1` ULP diff).
4. **Statement 4 harness `zsi1`/`zsi0` & missing-SSS refusal (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1967-2044`, `tests/ocean/fidelity/test_nemo_si3thd_phase2b_year_gate.py:64-85`):**
   * Verifies `_nemo_salprof` signature uses `zsi1=4.5, zsi0=3.5` (no `rn_sal_gd` or `rn_sal_fl`) with `sss_m=None` default, proves that planting the namelist values `(5.0, 2.0)` fails bit-exactness at `s_i = 4.1`, and proves that omitting `sss_m`, passing `sss_m=np.nan`, or omitting `"sss"` from the ZDF input dict raises `GateError` in `_nemo_salprof`, `_entry_arrays`, `_exact_1d_entry`, and `_forcing`.
5. **Autodiff (`jax.value_and_grad` & `jax.jvp`) and literal `0`-ULP forward bits (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2047-2209`):**
   * Verifies `_nemo_ice_enthalpy_remap`, `option2_salinity_profile`, and `si3_column_step_arrays`: forward primal outputs under `jax.value_and_grad` and `jax.jvp` are `0`-ULP (`uint64`) identical to standalone forward evaluation, gradients are finite and nonzero, reverse-mode agrees with forward-mode to `< 1e-10`, and a planted `jax.lax.stop_gradient` control fails.
6. **Parent-commit (`a572f4b2540351bf49328d1533211427f9c9d772`) control across non-SI3 callers and the production lane-3 SI3 dispatcher (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:475-1023`, `packages/ice/legoesm/ice/sea_ice.py:286-298,500-517`):**
   * Loads `tridiagonal.py`, `bitz_lipscomb.py`, `snow.py`, `sea_ice.py`, and `scm.py` from Round 5's actual parent commit `a572f4b2540351bf49328d1533211427f9c9d772` via `git show`, and runs both:
     1. All non-SI3 callers (`bitz_lipscomb_conduction_step`, `thomas_solve(..., "normalised")`, `IceColumnModel.step()` slab, and `step_sea_ice` dynamic-v2), proving every non-SI3 output stays bit-for-bit (`uint64`) identical to parent with 4 planted code-path controls that fail; and
     2. The production lane-3 SI3 dispatcher (`step_sea_ice` with `thermo_scheme="si3_bl99"` -> `_si3_step_with_trace`, `packages/ice/legoesm/ice/sea_ice.py:286-298,500-517`) on C1D step `kt = 43`, proving that ONLY `new_state.e_ice.data` changes (from parent's `_piecewise_remap` to current's literal `_nemo_ice_enthalpy_remap`, `0` ULP vs oracle `POST_DH.e_i`), while all other 8 `SI3ColumnState` fields (`concentration`, `h_ice`, `h_snow`, `T_surface`, `e_snow`, `S_bulk`, `S_layers`, `age_volume`) and all 22 `TileResponse` fields stay bit-for-bit (`uint64`) identical to parent, with two code-path plants inside `_si3_step_with_trace` / `si3_column_step_arrays` (`packages/ice/legoesm/ice/sea_ice.py:297,500-517`) that make the control fail.

---

## 7. Choices List and Review Disposition

* **CHOICES:**
  * `ASKED`: **DECISION 156** (classify and close D5 per platform before starting the ORCA2 rungs), **DECISION 158** (certify the SI3 ladder on CPU `fp64` production JIT with XLA FMA prevented in code via `nemo_source_round`, subnormal cases as a stated platform floor, and each rung spot-checked on an idle GPU), and **DECISION 159** (run one round per debt group with preregistered before/after counts and remaining owner rankings).
  * `UNASKED`: **NONE**. `_nemo_ice_remap: bool = True`, `_nemo_dh_ice_density_reciprocal: bool = True`, and `_nemo_salprof_factoring: bool = True` in `packages/ice/legoesm/ice/bitz_lipscomb.py` follow the existing private `_nemo_*` flag pattern and Rule 3 (the NEMO-literal value is the default on the SI3 path; non-SI3 callers are bit-for-bit unchanged). No configuration classes (`SeaIceConfig`, `SI3ThermoConfig`) were modified.
* **Independent review:** `independent review: operator`
