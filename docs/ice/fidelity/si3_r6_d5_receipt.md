# SI3 Ladder Round 6 (`si3_r6`) — Rung D5 Remaining Debts Receipt (`icethd_dh` Event-by-Event `ze_i_old`/`zh_i_old` Accumulation and Snow Precipitation/Sublimation Update)

* **Round:** `si3_r6` (SI3 sea-ice fidelity ladder, rung D5 remaining debts recorded in `docs/ice/fidelity/si3_r5_d5_receipt.md:248-258`)
* **Branch:** `fidelity/nemo-testcases-si3-ladder`
* **Parent commit (Round 5 landed):** `071dfa8443699a44e68904caaf3d02e17417e757`
* **Preregistration document:** `docs/ice/fidelity/si3_r6_d5_preregister.md:1-37` (committed at `10aa116fc65879dae32a054096d8fbc4b19e6d72` prior to probe execution)
* **Probe & gate code commit:** `d3cdcf21381e0f72b2f34575f348536a3b163338` (`clean: true`; supersedes initial Round 6 commit `fdd8a96c8d3ae682f1d35161ef515a47b2620ba9` after removing non-literal denominator guards and including `frq_m`)
* **Decisions binding this round:** **DECISION 156**, **DECISION 158**, **DECISION 159** (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_si3/standing_note.md`)
* **Independent review:** `independent review: operator`

---

## 1. Executive Summary & Verdict

Round 6 (`si3_r6`) closes the two remaining D5 debt families recorded in `docs/ice/fidelity/si3_r5_d5_receipt.md:248-258` one statement family at a time, each measured on the full `8,760`-step (`621,960` stage-field row) C1D year under **compiled `jax.jit`** (matching `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1836`) across CPU production JIT (`XLA_FLAGS=""`), the CPU no-FMA diagnostic (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`), and an idle GPU spot-check (`CUDA_VISIBLE_DEVICES=0`, `Tesla V100S-PCIE-32GB`):

* **Headline hashes & counts (each arm stated explicitly; 100% unchanged after making forward arithmetic NEMO-literal in `d3cdcf21381e0f72b2f34575f348536a3b163338`):**
  * **Round 5 baseline arm (`r5_all_four_items`, `_nemo_dh_ice_content_accumulation=False, _nemo_dh_snow_update=False`):**
    * **CPU production JIT (`XLA_FLAGS=""`) & CPU `--xla_cpu_max_isa=AVX` (bitwise identical `CPU == AVX`):** **`663` over-bar / `567,586` at `0` ULP** (`54,374` nonzero-ULP; `34` normal-step over-bar + `629` stated platform floor `kt = 5842..5867`) with full-year output SHA-256 **`62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`**.
    * **GPU spot-check (`CUDA_VISIBLE_DEVICES=0`):** **`34` over-bar / `567,973` at `0` ULP** (`53,987` nonzero-ULP; `34` normal-step over-bar + `0` in `kt = 5842..5867`) with full-year output SHA-256 **`482f7bae496cb7ffacec1d244965533839f86a30586567cb1a0ab4add3004de2`**.
  * **After Statement Family 1 (`r6_step1_ice_content_accumulation`, `_nemo_dh_ice_content_accumulation=True, _nemo_dh_snow_update=False`) — CONFIRMED:**
    * **CPU production JIT (`XLA_FLAGS=""`) & CPU `--xla_cpu_max_isa=AVX` (bitwise identical `CPU == AVX`):** **`651` over-bar (`-12` vs Round 5) / `598,638` at `0` ULP (`+31,052` vs Round 5)** (`23,322` nonzero-ULP; **`22` normal-step over-bar** + `629` stated platform floor `kt = 5842..5867`) with full-year output SHA-256 **`564460bd93b4a37f6d493dfbc5285088bd50f8e785334b860507cd0e73b700d7`**.
    * **GPU spot-check (`CUDA_VISIBLE_DEVICES=0`):** **`22` over-bar (`-12` vs Round 5) / `599,247` at `0` ULP (`+31,274` vs Round 5)** (`22,713` nonzero-ULP; **`22` normal-step over-bar** + `0` in `kt = 5842..5867`) with full-year output SHA-256 **`a26626ce17205839ffeaf81aa1636a3214fd112a7fb3bab02e53c6c9fdae3a3c`**.
  * **After Statement Family 1 + Statement Family 2 (`r6_step1_plus_step2_snow_update`, `_nemo_dh_ice_content_accumulation=True, _nemo_dh_snow_update=True`, shipped final Round 6 arm) — CONFIRMED:**
    * **CPU production JIT (`XLA_FLAGS=""`) & CPU `--xla_cpu_max_isa=AVX` (bitwise identical `CPU == AVX`):** **`629` over-bar (`-22` vs Statement 1, `-34` vs Round 5) / `599,956` at `0` ULP (`+1,318` vs Statement 1, `+32,370` vs Round 5)** (`22,004` nonzero-ULP; **`0` normal-step over-bar rows outside `kt = 5842..5867`** + `629` stated platform floor `kt = 5842..5867`) with full-year output SHA-256 **`b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`**.
    * **GPU spot-check (`CUDA_VISIBLE_DEVICES=0`):** **`0` over-bar (`-22` vs Statement 1, `-34` vs Round 5 — `0` over-bar rows across the entire 8,760-step, `621,960`-row C1D year!) / `600,737` at `0` ULP (`+1,490` vs Statement 1, `+32,764` vs Round 5)** (`21,223` nonzero-ULP; **100.0% (`61,320 / 61,320` rows) at `0` ULP in every 1-D operator stage `POST_ZDF`, `POST_DH`, `POST_TEMP1`, `POST_SAL`, and `POST_TEMP2`**) with full-year output SHA-256 **`9bcc933e8f3ef5b8e39f3f87d4edf63cfa8232aa857e3a4e6cbff38c570a659b`**.

1. **Statement Family 1 — Event-by-event ice heat-content (`ze_i_old`) and layer-thickness (`zh_i_old`, `zh_i`, `h_i_1d`) accumulation in `icethd_dh` feeding `ice_var_vremap` (`src/ICE/icethd_dh.F90:118-120,128-137,233-315,330-364,368-424,452-453,470-485,511` and `src/ICE/icevar.F90:957-1034`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:128-130,138-147,243-325,340-374,378-434,462-463,480-495,521` and `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:971-1048`) — CONFIRMED:**
   * Transcribed in `_nemo_ice_enthalpy_remap` (`packages/ice/legoesm/ice/bitz_lipscomb.py:980-1011`), `_dh_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1121-1209,1231-1252`), and `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1939`) under `_nemo_dh_ice_content_accumulation: bool = True`, accumulating `zh_i(1:3)`, `zh_i_old(0:4)`, `ze_i_old(0:4)`, and `h_i_1d` event-by-event across initial equal-layer setup (`icethd_dh.f90:138-147`), surface melt and ice sublimation (`icethd_dh.f90:243-325`), basal growth (`icethd_dh.f90:128-130,340-374`), basal melt (`icethd_dh.f90:378-434`), and snow-ice flooding (`icethd_dh.f90:462-463,480-495`), and passing `content=ze_i_old` directly as `pts_old` to `_nemo_ice_enthalpy_remap` (`icethd_dh.f90:521`, `icevar.f90:971-1048`).
   * **Effect on full-year C1D (`jax.jit`):**
     * **CPU production JIT (`XLA_FLAGS=""`) & CPU `--xla_cpu_max_isa=AVX` (`CPU == AVX`, SHA-256 `564460bd93b4a37f6d493dfbc5285088bd50f8e785334b860507cd0e73b700d7`):** Over-bar rows drop from **`663` to `651` (`-12`)** and normal-step over-bar rows drop from **`34` to `22` (`-12`)**, closing **all `6` `POST_DO.e_i` and all `6` `EXIT.e_i` over-bar rows** (`kt = 3996, 4014, 4092, 4400, 4413, 5241`) to **`0` over-bar** (`max_norm = 4.520989e-16`, `max_ulp = 3`). `0`-ULP rows increase from **`567,586` to `598,638` (`+31,052` rows; `0` rows moved from `0` ULP)**, closing **100% of normal-step nonzero-ULP rows** in `POST_DH.e_i`, `POST_TEMP1.e_i`, `POST_SAL.e_i`, and `POST_TEMP2.e_i` (`703 -> 0` each), `POST_DH.h_i`, `POST_TEMP1.h_i`, `POST_SAL.h_i`, and `POST_TEMP2.h_i` (`2,524 -> 0` each), `POST_SAL.sz_i` and `POST_TEMP2.sz_i` (`7 -> 0` each), `POST_DO.v_i` and `EXIT.v_i` (`2,288 -> 0` each), `POST_DO.sv_i` and `EXIT.sv_i` (`2,213 -> 0` each), and `POST_DO.szv_i` and `EXIT.szv_i` (`2,065 -> 0` each), and reducing `POST_DO.e_i` and `EXIT.e_i` normal-step nonzero-ULP rows from `3,127` to `628` (`-2,499` each).
     * **GPU spot-check (`CUDA_VISIBLE_DEVICES=0`, SHA-256 `a26626ce17205839ffeaf81aa1636a3214fd112a7fb3bab02e53c6c9fdae3a3c`):** Over-bar rows drop from **`34` to `22` (`-12`)**; `0`-ULP rows increase from **`567,973` to `599,247` (`+31,274` rows: `+31,052` normal-step rows + `+222` `kt = 5842..5867` ice rows)**.

2. **Statement Family 2 — Snow precipitation and sublimation thickness update in `icethd_dh` (`src/ICE/icethd_dh.F90:168-177,184-185,221`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:178-187,194-195,231`) — CONFIRMED:**
   * Localised the remaining `22` normal-step snow over-bar rows (`2 × 4 = 8` in `POST_DH..POST_TEMP2.e_s` at `kt = 5460, 5644` and `7 × 2 = 14` in `POST_DO.e_s, EXIT.e_s` at `kt = 4943, 4945, 5460, 5644, 5822, 5960, 5962`) to how `_dh_step` computed the 4-segment snow thickness `zh_s(0:3)` before `snw_ent` (`_nemo_snow_enthalpy_remap`):
     1. In `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:178-187` (`src/ICE/icethd_dh.F90:168-177`), `zh_s(0)` is initialized to `0._wp` and assigned only inside `IF( sprecip_1d(ji) > 0._wp )` with left-to-right association `zh_s(0) = zsnw(ji) * sprecip_1d(ji) * rDt_ice * r1_rhos / at_i_1d(ji)`. Without the `sprecip_1d > 0._wp` guard on `zh_s(0)`, negative numerical noise (`sprecip = -2.238e-19` at `kt = 5960`, `-1.199e-20` at `kt = 5962`) injected a negative top segment `zh_s(0) < 0` that shifted `zh_cum0` in `snw_ent`. Without `nemo_source_round` on the left-to-right products before `* rDt_ice * r1_rhos`, XLA's `AlgebraicSimplifier` constant-folded `dt * (1.0 / rho_snow)`, causing a `1`-ULP error in `zh_s(0)` (`kt = 4943, 4945, 5644, 5822`).
     2. In `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:194-195` (`src/ICE/icethd_dh.F90:184-185`), snow sublimation evaluates `zdeltah = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) )` and `zevap_rema = evap_ice_1d(ji) * rDt_ice + zdeltah * rhos`. Without `nemo_source_round` on `(-evaporation * inverse_snow_density) * dt`, XLA constant-folded `inverse_snow_density * dt`, shifting `zdeltah` by `1` ULP at `kt = 5460` (`10` ULP in `POST_DH.h_s` and `12` ULP / `1.614537e-15` in `POST_DH.e_s` and `2.122689e-15` in `POST_DO.e_s`) and forming a cancelling pair with `zh_s(0)` at `kt = 5733, 5932`.
     3. In `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:231` (`src/ICE/icethd_dh.F90:221`), snow surface melt clamps `zh_s(jk) = MAX( 0._wp , zh_s(jk) + zdum )`, preventing `-0.0` when `zdum == -zh_s(jk)` (`kt = 5379`).
   * Transcribed in `_dh_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1090-1095,1098-1108,1111-1119`) and `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1902,1904`) under `_nemo_dh_snow_update: bool = True`.
   * **Effect on full-year C1D (`jax.jit`):**
     * **CPU production JIT (`XLA_FLAGS=""`) & CPU `--xla_cpu_max_isa=AVX` (`CPU == AVX`, SHA-256 `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`):** Over-bar rows drop from **`651` to `629` (`-22` vs Statement 1; `-34` vs Round 5)**, reducing normal-step over-bar rows outside `kt = 5842..5867` from **`22` to `0` (`-100%`)**. `0`-ULP rows increase from **`598,638` to `599,956` (`+1,318` vs Statement 1; `+32,370` vs Round 5; `0` rows moved from `0` ULP)**, closing **100% of normal-step nonzero-ULP rows** in `POST_DH.e_s`, `POST_TEMP1.e_s`, `POST_SAL.e_s`, and `POST_TEMP2.e_s` (`137 -> 0` each) and `POST_DO.e_s` and `EXIT.e_s` (`147 -> 0` each), and reducing normal-step nonzero-ULP rows in `POST_DH..POST_TEMP2.h_s` (`81 -> 1` each) and `POST_DO..EXIT.v_s` (`79 -> 1` each) to the single subnormal precursor step `kt = 5841` (`h_s_1d = 5.562685e-309`).
     * **GPU spot-check (`CUDA_VISIBLE_DEVICES=0`, SHA-256 `9bcc933e8f3ef5b8e39f3f87d4edf63cfa8232aa857e3a4e6cbff38c570a659b`):** Over-bar rows drop from **`22` to `0` (`-22` vs Statement 1; `-34` vs Round 5 — `0` over-bar rows across the entire 8,760-step year!)**; `0`-ULP rows increase from **`599,247` to `600,737` (`+1,490` vs Statement 1: `+1,324` normal + `+166` subnormal-window rows; `+32,764` vs Round 5)**. (Note on `docs/ice/fidelity/si3_r6_d5_preregister.md:32,36`: the preregistration estimated GPU `0`-ULP as `599,051 -> 600,375` by adding the normal-step `+31,052` and `+1,324` deltas to `567,999`; in the actual GPU run, Statement 1 and Statement 2 also move `+222` and `+166` rows inside `kt = 5842..5867` to `0` ULP, yielding `567,973 -> 599,247 -> 600,737` and **100.0% (`61,320 / 61,320`) `0`-ULP rows in every 1-D stage `POST_ZDF..POST_TEMP2` on GPU**.)

---

## 2. Provenance & Evidence Artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| C1D V2 thermo stream (lane-3) | `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a/oracle_si3_thd_frames.bin` | `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| C1D V2 ZDF input stream (lane-3) | `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a/oracle_si3_zdf_inputs.bin` | `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd` |
| CPU production JIT probe (`XLA_FLAGS=""`) | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round6/d5_platform_probe_cpu.json` | `45f31fb9a5142af62c6af9862083877c836bc47bb719e9d1fb575d753a16fa29` |
| CPU diagnostic no-FMA probe (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`) | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round6/d5_platform_probe_cpu_avx_nofma.json` | `4440dc6238a8993f5d008716700b534e905fe355738f8f1644cb2cdfa09c0bd3` |
| GPU spot-check probe (`CUDA_VISIBLE_DEVICES=0`) | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round6/d5_platform_probe_gpu.json` | `04869ff5458bf531701621e8ac183faf382cbe3162bd70f024bfc52fbe595f09` |
| GPU `nvidia-smi` before | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round6/gpu_nvidia_smi_before.txt` | `4f3f3ef2ac8a5ff4752784db43a0f19b4afc7c1cca607250e427c5111cff7097` |
| GPU `nvidia-smi` after | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round6/gpu_nvidia_smi_after.txt` | `55bd659c19f587af41150785b9ca3007575206e04d4a77046d66fe34f5500ec4` |

---

## 3. Statement-by-Statement NEMO Source Mapping and Transcription

### 3.1 Statement Family 1: Event-by-Event `ze_i_old` and `zh_i_old` Accumulation in `icethd_dh` Feeding `ice_var_vremap`

* **Verbatim compiled NEMO Fortran source (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:128-130,138-147,243-253,262-274,288-295,300,309-317,322-325,340-355,365-374,378-389,399-411,425-434,462-463,480-495,521`, `C1D_OMIP_L3/BLD/ppsrc/nemo/sbcssm.f90:180-184`, and `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:971-1048`; original `src/ICE/icethd_dh.F90:118-120,128-137,233-315,330-364,368-424,452-453,470-485,511`, `src/SAS/sbcssm.F90:122-126`, and `src/ICE/icevar.F90:957-1034`):**
  1. **Surface/bottom available heat, `frq_m`, and initial equal-layer setup (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:128-130,138-147` and `C1D_OMIP_L3/BLD/ppsrc/nemo/sbcssm.f90:180-184`):**
     ```fortran
         zq_top = MAX( 0._wp, qml_ice_1d(ji) * rDt_ice )
         zf_tt  = qcn_ice_bot_1d(ji) + qsb_ice_bot_1d(ji) + fhld_1d(ji) + qtr_ice_bot_1d(ji) * frq_m_1d(ji)
         zq_bot = MAX( 0._wp, zf_tt * rDt_ice )
     ```
     ```fortran
         IF( ln_read_frq ) THEN
            frq_m(:,:) = sf_ssm_2d(jf_frq)%fnow(:,:,1) * tmask(:,:,1) ! solar penetration
         ELSE
            frq_m(:,:) = 1._wp
         ENDIF
     ```
     ```fortran
         zs_i_old(0:nlay_i+1) = 0._wp
         ze_i_old(0:nlay_i+1) = 0._wp
         zh_i_old(0:nlay_i+1) = 0._wp
         zh_i    (0:nlay_i+1) = 0._wp
         DO jk = 1, nlay_i
            zs_i_old(jk) = h_i_1d(ji) * r1_nlay_i * zs_i  (jk)
            ze_i_old(jk) = h_i_1d(ji) * r1_nlay_i * e_i_1d(ji,jk)
            zh_i_old(jk) = h_i_1d(ji) * r1_nlay_i
            zh_i    (jk) = h_i_1d(ji) * r1_nlay_i
         END DO
     ```
  2. **Surface ice melt and ice sublimation loop `jk = 1..nlay_i` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:243-253,262-274,288-295,300,309-317,322-325`):**
     ```fortran
         DO jk = 1, nlay_i
            ztmelts = - rTmlt * sz_i_1d(ji,jk)   ! Melting point of layer k [C]

            IF( t_i_1d(ji,jk) >= (ztmelts+rt0) ) THEN   !-- Internal melting

               zEi            = - e_i_1d(ji,jk) * r1_rhoi             ! Specific enthalpy of layer k [J/kg, <0]
               zdE            =   0._wp                               ! Specific enthalpy difference (J/kg, <0)
               !                                                          set up at 0 since no energy is needed to melt water...(it is already melted)
               zdum           = MIN( 0._wp , - zh_i(jk) )             ! internal melting occurs when the internal temperature is above freezing
               !                                                          this should normally not happen, but sometimes, heat diffusion leads to this
               zfmdt          = - zdum * rhoi                         ! Recompute mass flux [kg/m2, >0]
     ```
     ```fortran
            ELSE                                        !-- Surface melting

               zEi            = - e_i_1d(ji,jk) * r1_rhoi             ! Specific enthalpy of layer k [J/kg, <0]
               zEw            =    rcp * ztmelts                      ! Specific enthalpy of resulting meltwater [J/kg, <0]
               zdE            =    zEi - zEw                          ! Specific enthalpy difference < 0

               zfmdt          = - zq_top / zdE                        ! Mass flux to the ocean [kg/m2, >0]

               zdum           = - zfmdt * r1_rhoi                     ! Melt of layer jk [m, <0]

               zdum           = MIN( 0._wp , MAX( zdum , - zh_i(jk) ) )    ! Melt of layer jk cannot exceed the layer thickness [m, <0]

               zq_top         = MAX( 0._wp , zq_top - zdum * rhoi * zdE ) ! update available heat
     ```
     ```fortran
            ! update thickness
            zh_i  (jk) = MAX( 0._wp, zh_i  (jk) + zdum )
            h_i_1d(ji) = MAX( 0._wp, h_i_1d(ji) + zdum )
            !
            ! update heat content (J.m-2), salt content and layer thickness
            zs_i_old(jk) = zs_i_old(jk) + zdum * zs_i(jk)
            ze_i_old(jk) = ze_i_old(jk) + zdum * e_i_1d(ji,jk)
            zh_i_old(jk) = zh_i_old(jk) + zdum
     ```
     ```fortran
            zdum               = MAX( - zh_i(jk) , - zevap_rema * r1_rhoi )
     ```
     ```fortran
            zevap_rema   = zevap_rema + zdum * rhoi
            zh_i  (jk)   = MAX( 0._wp, zh_i  (jk) + zdum )
            h_i_1d(ji)   = MAX( 0._wp, h_i_1d(ji) + zdum )
            dh_i_sub(ji) = dh_i_sub(ji) + zdum

            ! update heat content (J.m-2), salt content and layer thickness
            zs_i_old(jk) = zs_i_old(jk) + zdum * zs_i(jk)
            ze_i_old(jk) = ze_i_old(jk) + zdum * e_i_1d(ji,jk)
            zh_i_old(jk) = zh_i_old(jk) + zdum
     ```
     ```fortran
            IF( zh_i(jk) > 0._wp ) THEN ; icount(jk) = 0
            ELSE                        ; icount(jk) = 1 ; ENDIF

         END DO
     ```
  3. **Basal growth at slot `nlay_i+1 = 4` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:340-355,365-374`):**
     ```fortran
         IF(  zf_tt < 0._wp  ) THEN

            zs_i_new       = zswitch_sal * rn_sinew * sss_1d(ji) + ( 1. - zswitch_sal ) * zs_i(1)          ! New ice salinity
            
            ztmelts        = - rTmlt * zs_i_new                                                            ! New ice melting point (C)
            
            zt_i_new       = zswitch_sal * t_bo_1d(ji) + ( 1. - zswitch_sal) * t_i_1d(ji, nlay_i)
            
            zEi            = rcpi * ( zt_i_new - (ztmelts+rt0) ) &                                         ! Specific enthalpy of forming ice (J/kg, <0)
               &             - rLfus * MAX( 0._wp, 1.0 - ztmelts / ( MIN( zt_i_new - rt0, -epsi10 ) ) ) + rcp * ztmelts ! clem: max to deal with different eq. freezing in ice and ocean (but useless for now)
            
            zEw            = rcp  * ( t_bo_1d(ji) - rt0 )                                                  ! Specific enthalpy of seawater (J/kg, < 0)
            
            zdE            = zEi - zEw                                                                     ! Specific enthalpy difference (J/kg, <0)
            
            dh_i_bog(ji)   = rDt_ice * MAX( 0._wp , zf_tt / ( zdE * rhoi ) )
     ```
     ```fortran
            ! update thickness
            zh_i(nlay_i+1) = zh_i(nlay_i+1) + dh_i_bog(ji)
            h_i_1d(ji)     = h_i_1d(ji)     + dh_i_bog(ji)

            ! update heat content (J.m-2), salt content and layer thickness
            zs_i_old(nlay_i+1) = zs_i_old(nlay_i+1) + dh_i_bog(ji) * zs_i_new
            ze_i_old(nlay_i+1) = ze_i_old(nlay_i+1) + dh_i_bog(ji) * (-zEi * rhoi)
            zh_i_old(nlay_i+1) = zh_i_old(nlay_i+1) + dh_i_bog(ji)

         ENDIF
     ```
  4. **Basal melt loop `jk = nlay_i..1` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:378-389,399-411,425-434`):**
     ```fortran
         DO jk = nlay_i, 1, -1
            IF(  zf_tt  >  0._wp  .AND. jk > icount(jk) ) THEN   ! do not calculate where layer has already disappeared by surface melting

               ztmelts = - rTmlt * sz_i_1d(ji,jk)  ! Melting point of layer jk (C)

               IF( t_i_1d(ji,jk) >= (ztmelts+rt0) ) THEN   !-- Internal melting

                  zEi            = - e_i_1d(ji,jk) * r1_rhoi     ! Specific enthalpy of melting ice (J/kg, <0)
                  zdE            = 0._wp                         ! Specific enthalpy difference   (J/kg, <0)
                  !                                                  set up at 0 since no energy is needed to melt water...(it is already melted)
                  zdum           = MIN( 0._wp , - zh_i(jk) )  ! internal melting occurs when the internal temperature is above freezing
     ```
     ```fortran
               ELSE                                        !-- Basal melting

                  zEi            = - e_i_1d(ji,jk) * r1_rhoi                       ! Specific enthalpy of melting ice (J/kg, <0)
                  zEw            = rcp * ztmelts                                   ! Specific enthalpy of meltwater (J/kg, <0)
                  zdE            = zEi - zEw                                       ! Specific enthalpy difference   (J/kg, <0)

                  zfmdt          = - zq_bot / zdE                                  ! Mass flux x time step (kg/m2, >0)

                  zdum           = - zfmdt * r1_rhoi                               ! Gross thickness change

                  zdum           = MIN( 0._wp , MAX( zdum, - zh_i(jk) ) )          ! bound thickness change

                  zq_bot         = MAX( 0._wp , zq_bot - zdum * rhoi * zdE )       ! update available heat. MAX is necessary for roundup errors
     ```
     ```fortran
               ! update thickness
               zh_i  (jk) = MAX( 0._wp, zh_i  (jk) + zdum )
               h_i_1d(ji) = MAX( 0._wp, h_i_1d(ji) + zdum )
               !
               ! update heat content (J.m-2), salt content and layer thickness
               zs_i_old(jk) = zs_i_old(jk) + zdum * zs_i(jk)
               ze_i_old(jk) = ze_i_old(jk) + zdum * e_i_1d(ji,jk)
               zh_i_old(jk) = zh_i_old(jk) + zdum
            ENDIF
         END DO
     ```
  5. **Snow-ice flooding at slot `0` and call to `ice_var_vremap` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:462-463,480-495,521` and `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:971-1048`):**
     ```fortran
         zfmdt          = ( rhos - rhoi ) * dh_snowice(ji)    ! <0
         zEw            = rcp * sst_1d(ji)
     ```
     ```fortran
         zh_i(0) = zh_i(0) + dh_snowice(ji)
         zdeltah =           dh_snowice(ji)

         ! update heat content (J.m-2), salt content and layer thickness
         zs_i_old(0) = zs_i_old(0) - zfmdt * sss_1d(ji) * r1_rhoi      ! clem: s(0) could be > rn_sinew*sss
         zh_i_old(0) = zh_i_old(0) + dh_snowice(ji)
         ze_i_old(0) = ze_i_old(0) + zfmdt * zEw           ! 1st part (sea water enthalpy)

         !
         DO jk = nlay_s, 0, -1   ! flooding of snow starts from the base
            zdum        = MIN( zdeltah, zh_s(jk) )         ! amount of snw that floods, > 0
            zh_s(jk)    = MAX( 0._wp, zh_s(jk) - zdum )    ! remove some snow thickness
            ze_i_old(0) = ze_i_old(0) + zdum * ze_s(jk)    ! 2nd part (snow enthalpy)
            ! update dh_snowice
            zdeltah     = MAX( 0._wp, zdeltah - zdum )
         END DO
     ```
     ```fortran
                                CALL ice_var_vremap( zh_i_old, ze_i_old, e_i_1d (ji,:) )
     ```
* **legoESM transcription (`packages/ice/legoesm/ice/bitz_lipscomb.py:41,980-1011,1121-1209,1231-1252,1811-1939`):**
  * `_nemo_ice_enthalpy_remap` accepts `content=None` and uses `pts_old = content if content is not None else sr(thickness * enthalpy)` (`packages/ice/legoesm/ice/bitz_lipscomb.py:980-992`).
  * Under `_nemo_dh_ice_content_accumulation: bool = True`, `_dh_step` accumulates `zh_i_layers`, `zh_i_old_layers`, `ze_i_old_layers`, `h_i_work`, `ze_i_old_bog`, and `flood_content` event-by-event with `nemo_source_round`, evaluates `zf = sr(sr(sr(zdf.qcn_ice_bottom + forcing.qsb_ice_bottom) + forcing.fhld) + sr(zdf.qtr_ice_bottom * frq_m))` (`packages/ice/legoesm/ice/bitz_lipscomb.py:41,1067-1068`, matching `icethd_dh.f90:129` and `sbcssm.f90:183`), evaluates NEMO's literal direct divisions `zfmdt = sr(-surface_energy / energy_difference)` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1129`, matching `icethd_dh.f90:268`) and `zfmdt = sr(-basal_energy / energy_difference)` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1130`, matching `icethd_dh.f90:405`) without any denominator guard (`zdE = zEi - zEw <= -225,630.87 J/kg < 0` across all `8,760 × 3 = 26,280` layer-steps of the C1D record, proved in `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2659-2776`), stacks `ze_i_old = [flood_content, ze_i_old_layers[0], ze_i_old_layers[1], ze_i_old_layers[2], ze_i_old_bog]` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1071-1077`), and passes `content=ze_i_old` into `_nemo_ice_enthalpy_remap` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1249-1251`).

### 3.2 Statement Family 2: Snow Precipitation and Sublimation Thickness Update in `icethd_dh`

* **Verbatim compiled NEMO Fortran source (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:178-187,194-195,231`; original `src/ICE/icethd_dh.F90:168-177,184-185,221`):**
  ```fortran
         IF( sprecip_1d(ji) > 0._wp ) THEN
            zh_s(0) = zsnw(ji) * sprecip_1d(ji) * rDt_ice * r1_rhos / at_i_1d(ji)   ! thickness of precip
            ze_s(0) = MAX( 0._wp, - qprec_ice_1d(ji) )                              ! enthalpy of the precip (>0, J.m-3)
            !
            hfx_spr_1d(ji) = hfx_spr_1d(ji) + ze_s(0) * zh_s(0) * a_i_1d(ji) * r1_Dt_ice   ! heat flux from snow precip (>0, W.m-2)
            wfx_spr_1d(ji) = wfx_spr_1d(ji) - rhos    * zh_s(0) * a_i_1d(ji) * r1_Dt_ice   ! mass flux, <0
            !
            ! update thickness
            h_s_1d(ji) = h_s_1d(ji) + zh_s(0)
         ENDIF
  ```
  ```fortran
         zdeltah    = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) )   ! amount of snw that sublimates (<0) or deposition (>0)
         zevap_rema =        evap_ice_1d(ji)           * rDt_ice + zdeltah * rhos  ! remaining evap in kg.m-2 (used for ice sublimation later on)
  ```
  ```fortran
               zh_s    (jk) = MAX( 0._wp , zh_s    (jk) + zdum )
  ```
* **legoESM transcription (`packages/ice/legoesm/ice/bitz_lipscomb.py:1090-1095,1098-1108,1111-1119,1902,1904`):**
  * Under `_nemo_dh_snow_update: bool = True`, `_dh_step` evaluates `h_precip_raw = sr(sr(sr(snow_partition * forcing.snow_precipitation) * dt) * inverse_snow_density) / a` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1090-1091`) with NEMO's literal `/ a` division (`at_i_1d(ji) == 0.9 > 0` across all `8,760` steps of the C1D record, proved in `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2659-2776`), gates `h_precip` on `forcing.snow_precipitation > 0.0` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1091-1092`), associates `snow_delta = jnp.maximum(sr(sr(-evaporation * inverse_snow_density) * dt), -h_s_work)` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1098-1103`), and clamps `jnp.maximum(0.0, hseg_s[..., k] + delta)` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1112-1119`).
* **Exact step localisation and correction to Round 5 receipt line 215 (`docs/ice/fidelity/si3_r5_d5_receipt.md:215,251,257`):**
  * In Round 5 (`r5_all_four_items`), the two `POST_DH..POST_TEMP2.e_s` over-bar steps were **`kt = 5460`** (`12` ULP, `norm_err = 1.614537e-15`) and **`kt = 5644`** (`9` ULP, `norm_err = 1.077274e-15`), whereas `kt = 5379` had `-0.0` (`0x8000000000000000`) vs `+0.0` (`0x0000000000000000`) on `e_s_1d(0)` (`norm_err = 0.0`, at-bar, large integer ULP count across the sign bit).
  * Across all `7` steps where `POST_DO.e_s` and `EXIT.e_s` crossed `1e-15` (`kt = 4943, 4945, 5460, 5644, 5822, 5960, 5962`):
    * At `kt = 4943, 4945, 5644, 5822`: XLA's `AlgebraicSimplifier` constant-folded `dt * inverse_snow_density` in `h_precip`, shifting `zh_s(0)` by `1` ULP and amplifying through `snw_ent`'s `zts_cum0(jk0) / dz` interpolation to `7`–`9` ULP on `e_s_1d`.
    * At `kt = 5460`: XLA constant-folded `inverse_snow_density * dt` in `snow_delta` (`zdeltah`), shifting `zdeltah` by `1` ULP (`-1.5720604029915320e-05` vs `-1.5720604029915323e-05`), `h_s_1d` by `10` ULP (`2.852968608124234e-07`), and `e_s_1d` by `12` ULP (`norm_err = 1.614537e-15` at `POST_DH..POST_TEMP2.e_s` and `2.122689e-15` at `POST_DO..EXIT.e_s`).
    * At `kt = 5960, 5962`: `forcing.snow_precipitation` had tiny negative values (`-2.238e-19` and `-1.199e-20`), and because `h_precip` lacked NEMO's `IF( sprecip_1d(ji) > 0._wp )` guard, `zh_s(0) < 0` shifted `h_s_1d` and `zh_cum0` in `snw_ent` (`8` ULP on `POST_DO.e_s`).
  * **Cancelling-pair verification (`kt = 5733, 5932`):** At `kt = 5733` and `kt = 5932`, XLA constant-folding of `dt * r1_rhos` produced a `+1` ULP error in `zh_s(0)` (`icethd_dh.f90:179`) and a `-1` ULP error in `zdeltah` (`icethd_dh.f90:194`) that cancelled each other in `zh_s(0) + zdeltah`. Transcribing both statements in NEMO's left-to-right order in `_nemo_dh_snow_update=True` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1090-1095,1098-1108,1111-1119`) makes both `zh_s(0)` and `zdeltah` `0` ULP individually and preserves `0` ULP on `POST_DH.h_s` and `POST_DH.e_s`.

---

## 4. Full-Year (`8,760`-Step, `621,960`-Row) C1D Rescore Under `jax.jit`

### 4.1 Progression Across Ordered Statement Families (`run_year_rescore`, `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883`)

| Evaluation Arm (`jax.jit`) | CPU Production (`XLA_FLAGS=""`) Over-bar / `0` ULP / Nonzero ULP (Normal over-bar; Full-Year SHA-256) | CPU Diagnostic (`--xla_cpu_max_isa=AVX`) Over-bar / `0` ULP / Nonzero ULP (Normal over-bar; Full-Year SHA-256) | GPU Spot-Check (`CUDA_VISIBLE_DEVICES=0`) Over-bar / `0` ULP / Nonzero ULP (Normal over-bar; Full-Year SHA-256) |
|---|---:|---:|---:|
| **Round 5 baseline (`r5_all_four_items`, before Statement Family 1)** | **`663` / `567,586` / `54,374`**<br>(`34`; `62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`) | **`663` / `567,586` / `54,374`**<br>(`34`; `62280a6d119ce8c74f4c758f5dfc7aec37f33a2c90c7764f3b1e93c2cbf4c320`) | **`34` / `567,973` / `53,987`**<br>(`34`; `482f7bae496cb7ffacec1d244965533839f86a30586567cb1a0ab4add3004de2`) |
| **After Statement Family 1 (`r6_step1_ice_content_accumulation`, `icethd_dh.f90:128-130,138-147,243-325,340-374,378-434,462-463,480-495,521`)** | **`651` / `598,638` / `23,322`**<br>(**`22`; `564460bd93b4a37f6d493dfbc5285088bd50f8e785334b860507cd0e73b700d7`**) | **`651` / `598,638` / `23,322`**<br>(**`22`; `564460bd93b4a37f6d493dfbc5285088bd50f8e785334b860507cd0e73b700d7`**) | **`22` / `599,247` / `22,713`**<br>(**`22`; `a26626ce17205839ffeaf81aa1636a3214fd112a7fb3bab02e53c6c9fdae3a3c`**) |
| **After Statement Family 1 + 2 (`r6_step1_plus_step2_snow_update`, shipped final Round 6, `icethd_dh.f90:178-187,194-195,231`)** | **`629` / `599,956` / `22,004`**<br>(**`0`; `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`**) | **`629` / `599,956` / `22,004`**<br>(**`0`; `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`**) | **`0` / `600,737` / `21,223`**<br>(**`0`; `9bcc933e8f3ef5b8e39f3f87d4edf63cfa8232aa857e3a4e6cbff38c570a659b`**) |
| *Isolated Statement Family 2 from Round 5 (`r6_isolated_step2_snow_update`)* | `641` / `568,904` / `53,056`<br>(`12`; `a1604df3db1f5296dba0a0f3d4a3b2d6bbc42a200d0358d7f7953ad1f847514a`) | `641` / `568,904` / `53,056`<br>(`12`; `a1604df3db1f5296dba0a0f3d4a3b2d6bbc42a200d0358d7f7953ad1f847514a`) | `12` / `569,451` / `52,509`<br>(`12`; `664a2cc10670b452b442b233b40c7b222598cc20e020d7f1ba71e5e7dc77d032`) |

### 4.2 Stage-by-Stage Progression Under Compiled `jax.jit` (CPU `XLA_FLAGS=""` == `--xla_cpu_max_isa=AVX`, and GPU)

| Stage (`rows`) | Round 5 Baseline (`r5_all_four_items`) CPU Over-bar / `0` ULP | After Statement Family 1 (`r6_step1_ice_content_accumulation`) CPU Over-bar / `0` ULP | Shipped Final Arm (`r6_step1_plus_step2_snow_update`) CPU Over-bar / `0` ULP | Shipped Final Arm (`r6_step1_plus_step2_snow_update`) GPU Over-bar / `0` ULP | Net CPU Delta (`R5 -> R6`, Over-bar / `0` ULP) | Net GPU Delta (`R5 -> R6`, Over-bar / `0` ULP) |
|---|---:|---:|---:|---:|---:|---:|
| `ENTRY` (`105,120`) | `0` / `85,157` | `0` / `85,157` | `0` / `85,157` | `0` / `85,157` | `0` / `0` | `0` / `0` |
| `POST_ZDF` (`61,320`) | `75` / `61,245` | `75` / `61,245` | `75` / `61,245` | **`0` / `61,320`** | `0` / `0` | `0` / `0` |
| `POST_DH` (`61,320`) | `77` / `57,775` | **`77` / `61,002`** | **`75` / `61,219`** | **`0` / `61,320`** | **`-2` / `+3,444`** | **`-2` / `+3,498`** |
| `POST_TEMP1` (`61,320`) | `77` / `57,775` | **`77` / `61,002`** | **`75` / `61,219`** | **`0` / `61,320`** | **`-2` / `+3,444`** | **`-2` / `+3,498`** |
| `POST_SAL` (`61,320`) | `77` / `57,768` | **`77` / `61,002`** | **`75` / `61,219`** | **`0` / `61,320`** | **`-2` / `+3,451`** | **`-2` / `+3,505`** |
| `POST_TEMP2` (`61,320`) | `77` / `57,768` | **`77` / `61,002`** | **`75` / `61,219`** | **`0` / `61,320`** | **`-2` / `+3,451`** | **`-2` / `+3,505`** |
| `POST_DO` (`105,120`) | `140` / `95,049` | **`134` / `104,114`** | **`127` / `104,339`** | **`0` / `104,490`** | **`-13` / `+9,290`** | **`-13` / `+9,379`** |
| `EXIT` (`105,120`) | `140` / `95,049` | **`134` / `104,114`** | **`127` / `104,339`** | **`0` / `104,490`** | **`-13` / `+9,290`** | **`-13` / `+9,379`** |
| **Total (`621,960`)** | **`663` / `567,586`** | **`651` / `598,638`** | **`629` / `599,956`** | **`0` / `600,737`** | **`-34` / `+32,370`** | **`-34` / `+32,764`** |

---

## 5. Remaining Owner Ranking After Each Statement Family

### 5.1 Ranked Remaining Owners After Statement Family 1 (`r6_step1_ice_content_accumulation`, `_nemo_dh_ice_content_accumulation=True, _nemo_dh_snow_update=False`)

| Rank | Owner / `Stage.field` | CPU Over-Bar Rows | GPU Over-Bar Rows | CPU Nonzero-ULP Rows | GPU Nonzero-ULP Rows | Max ULP (CPU) | Max Normalised Error (CPU) | Worst Step & Field (`kt`, `Stage.field`) | Physical / Arithmetic Cause |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `D5_PLATFORM_FLOOR_KT5842_5867_SUBNORMAL_SNOW` | `629` | `0` | `819` | `210` | `190,081,022,848,368` | `2.510100e-02` | `kt = 5860`, `POST_ZDF.e_s` | CPU XLA LLVM `FTZ/DAZ` hardware subnormal flush in `kt = 5842..5867` (DECISION 158 stated platform floor; `0` over-bar on GPU) |
| 2 | `POST_DO.e_s` | `7` | `7` | `147` | `147` | `16` | `2.122689e-15` | `kt = 5460`, `POST_DO.e_s` | Snow-precip/sublimation `zh_s` update before `snw_ent` (`icethd_dh.f90:178-187,194-195,231`; closed in Statement Family 2) |
| 3 | `EXIT.e_s` | `7` | `7` | `147` | `147` | `16` | `2.122689e-15` | `kt = 5460`, `EXIT.e_s` | Same as `POST_DO.e_s` (closed in Statement Family 2) |
| 4 | `POST_DH.e_s` | `2` | `2` | `137` | `137` | `2,563,146,750,041,342,108` | `1.614537e-15` | `kt = 5460`, `POST_DH.e_s` | Snow-precip/sublimation `zh_s` update (`kt = 5460, 5644` over-bar; closed in Statement Family 2) |
| 5 | `POST_TEMP1.e_s` | `2` | `2` | `137` | `137` | `2,563,146,750,041,342,108` | `1.614537e-15` | `kt = 5460`, `POST_TEMP1.e_s` | Inherited from `POST_DH.e_s` (closed in Statement Family 2) |
| 6 | `POST_SAL.e_s` | `2` | `2` | `137` | `137` | `2,563,146,750,041,342,108` | `1.614537e-15` | `kt = 5460`, `POST_SAL.e_s` | Inherited from `POST_DH.e_s` (closed in Statement Family 2) |
| 7 | `POST_TEMP2.e_s` | `2` | `2` | `137` | `137` | `2,563,146,750,041,342,108` | `1.614537e-15` | `kt = 5460`, `POST_TEMP2.e_s` | Inherited from `POST_DH.e_s` (closed in Statement Family 2) |
| 8 | `ENTRY.e_i` | `0` | `0` | `5,447` | `5,447` | `3` | `4.673354e-16` | `kt = 7628`, `ENTRY.e_i` | Global-to-1D-to-global round-trip `(e_i / v_i * 3) * h_i * a_i * (1/3)` (`<= 3` ULP, all at-bar) |
| 9 | `ENTRY.e_s` | `0` | `0` | `4,489` | `4,489` | `3` | `4.349022e-16` | `kt = 1914`, `ENTRY.e_s` | Global-to-1D-to-global round-trip `(e_s / v_s * 3) * h_s * a_i * (1/3)` (`<= 3` ULP, all at-bar) |
| 10 | `ENTRY.v_i` | `0` | `0` | `2,626` | `2,626` | `1` | `2.219566e-16` | `kt = 7937`, `ENTRY.v_i` | Global-to-1D-to-global `(v_i / a_i) * a_i` (`1` ULP, all at-bar) |
| 11 | `ENTRY.v_s` | `0` | `0` | `2,526` | `2,526` | `1` | `5.551115e-17` | `kt = 859`, `ENTRY.v_s` | Global-to-1D-to-global `(v_s / a_i) * a_i` (`1` ULP, all at-bar) |
| 12 | `ENTRY.sv_i` | `0` | `0` | `2,481` | `2,481` | `2` | `3.992932e-16` | `kt = 3966`, `ENTRY.sv_i` | Global-to-1D-to-global `(sv_i / v_i) * (h_i * a_i)` (`<= 2` ULP, all at-bar) |
| 13 | `ENTRY.szv_i` | `0` | `0` | `2,354` | `2,354` | `8` | `5.240056e-16` | `kt = 4368`, `ENTRY.szv_i` | Global-to-1D-to-global `_nemo_salprof(s_i) * v_i * (1/3)` (`<= 8` ULP, all at-bar) |
| 14 | `POST_DO.e_i` | `0` | `0` | `628` | `628` | `3` | `4.520989e-16` | `kt = 4200`, `POST_DO.e_i` | Harness global write `e_i_1d * h_i_1d * a_i_1d * (1/3)` vs NEMO `e_i_1d * v_i * r1_nlay_i` (`<= 3` ULP, **all `0` over-bar**; reduced from `6` over-bar / `3,127` nonzero-ULP by Statement 1) |
| 15 | `EXIT.e_i` | `0` | `0` | `628` | `628` | `3` | `4.520989e-16` | `kt = 4200`, `EXIT.e_i` | Same as `POST_DO.e_i` (`<= 3` ULP, **all `0` over-bar**; reduced from `6` over-bar / `3,127` nonzero-ULP by Statement 1) |
| 16 | `POST_DH.h_s` | `0` | `0` | `81` | `81` | `4,314,448,443,020,935,169` | `5.551115e-17` | `kt = 652`, `POST_DH.h_s` | `h_s` update (`<= 5.55e-17`, all at-bar; reduced to `1` on CPU / `0` on GPU in Statement 2) |
| 17 | `POST_TEMP1.h_s` | `0` | `0` | `81` | `81` | `4,314,448,443,020,935,169` | `5.551115e-17` | `kt = 652`, `POST_TEMP1.h_s` | Inherited from `POST_DH.h_s` |
| 18 | `POST_SAL.h_s` | `0` | `0` | `81` | `81` | `4,314,448,443,020,935,169` | `5.551115e-17` | `kt = 652`, `POST_SAL.h_s` | Inherited from `POST_DH.h_s` |
| 19 | `POST_TEMP2.h_s` | `0` | `0` | `81` | `81` | `4,314,448,443,020,935,169` | `5.551115e-17` | `kt = 652`, `POST_TEMP2.h_s` | Inherited from `POST_DH.h_s` |
| 20 | `POST_DO.v_s` | `0` | `0` | `79` | `79` | `4,313,547,723,095,461,071` | `5.551115e-17` | `kt = 652`, `POST_DO.v_s` | `h_s * a_i` (`<= 5.55e-17`, all at-bar; reduced to `1` on CPU / `0` on GPU in Statement 2) |
| 21 | `EXIT.v_s` | `0` | `0` | `79` | `79` | `4,313,547,723,095,461,071` | `5.551115e-17` | `kt = 652`, `EXIT.v_s` | Same as `POST_DO.v_s` |

### 5.2 Complete Ranked Remaining Owners After Statement Family 2 (`r6_step1_plus_step2_snow_update`, Shipped Final Round 6 Arm)

| Rank | Owner / `Stage.field` | CPU Over-Bar Rows | GPU Over-Bar Rows | CPU Nonzero-ULP Rows | GPU Nonzero-ULP Rows | Max ULP (CPU) | Max Normalised Error (CPU) | Worst Step & Field (`kt`, `Stage.field`) | Physical / Arithmetic Cause |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `D5_PLATFORM_FLOOR_KT5842_5867_SUBNORMAL_SNOW` | `629` | `0` | `819` | `44` | `190,081,022,848,368` | `2.510100e-02` | `kt = 5860`, `POST_ZDF.e_s` | CPU XLA LLVM `FTZ/DAZ` hardware subnormal flush in `kt = 5842..5867` (DECISION 158 stated platform floor; `0` over-bar and `<= 3` ULP on GPU) |
| 2 | `ENTRY.e_i` | `0` | `0` | `5,447` | `5,447` | `3` | `4.673354e-16` | `kt = 7628`, `ENTRY.e_i` | Harness global-to-1D-to-global round-trip `(e_i / v_i * 3) * h_i * a_i * (1/3)` (`<= 3` ULP, all at-bar) |
| 3 | `ENTRY.e_s` | `0` | `0` | `4,489` | `4,489` | `3` | `4.349022e-16` | `kt = 1914`, `ENTRY.e_s` | Harness global-to-1D-to-global round-trip `(e_s / v_s * 3) * h_s * a_i * (1/3)` (`<= 3` ULP, all at-bar) |
| 4 | `ENTRY.v_i` | `0` | `0` | `2,626` | `2,626` | `1` | `2.219566e-16` | `kt = 7937`, `ENTRY.v_i` | Harness global-to-1D-to-global `(v_i / a_i) * a_i` (`1` ULP, all at-bar) |
| 5 | `ENTRY.v_s` | `0` | `0` | `2,526` | `2,526` | `1` | `5.551115e-17` | `kt = 859`, `ENTRY.v_s` | Harness global-to-1D-to-global `(v_s / a_i) * a_i` (`1` ULP, all at-bar) |
| 6 | `ENTRY.sv_i` | `0` | `0` | `2,481` | `2,481` | `2` | `3.992932e-16` | `kt = 3966`, `ENTRY.sv_i` | Harness global-to-1D-to-global `(sv_i / v_i) * (h_i * a_i)` (`<= 2` ULP, all at-bar) |
| 7 | `ENTRY.szv_i` | `0` | `0` | `2,354` | `2,354` | `8` | `5.240056e-16` | `kt = 4368`, `ENTRY.szv_i` | Harness global-to-1D-to-global `_nemo_salprof(s_i) * v_i * (1/3)` (`<= 8` ULP, all at-bar) |
| 8 | `POST_DO.e_i` | `0` | `0` | `628` | `628` | `3` | `4.520989e-16` | `kt = 4200`, `POST_DO.e_i` | Harness global write `e_i_1d * h_i_1d * a_i_1d * (1/3)` vs NEMO `e_i_1d * v_i * r1_nlay_i` (`<= 3` ULP, all at-bar; `POST_TEMP2.e_i` is `0` ULP on all `8,734` normal steps) |
| 9 | `EXIT.e_i` | `0` | `0` | `628` | `628` | `3` | `4.520989e-16` | `kt = 4200`, `EXIT.e_i` | Same as `POST_DO.e_i` (`<= 3` ULP, all at-bar) |
| 10 | `POST_DH.h_s` | `0` | `0` | `1` | `0` | `1,125,899,906,842,624` | `5.562685e-309` | `kt = 5841`, `POST_DH.h_s` | Single subnormal output `h_s_1d = 5.562685e-309` flushed to `0.0` by CPU FTZ at `kt = 5841` (`0` ULP on GPU) |
| 11 | `POST_TEMP1.h_s` | `0` | `0` | `1` | `0` | `1,125,899,906,842,624` | `5.562685e-309` | `kt = 5841`, `POST_TEMP1.h_s` | Inherited from `POST_DH.h_s` (`0` ULP on GPU) |
| 12 | `POST_SAL.h_s` | `0` | `0` | `1` | `0` | `1,125,899,906,842,624` | `5.562685e-309` | `kt = 5841`, `POST_SAL.h_s` | Inherited from `POST_DH.h_s` (`0` ULP on GPU) |
| 13 | `POST_TEMP2.h_s` | `0` | `0` | `1` | `0` | `1,125,899,906,842,624` | `5.562685e-309` | `kt = 5841`, `POST_TEMP2.h_s` | Inherited from `POST_DH.h_s` (`0` ULP on GPU) |
| 14 | `POST_DO.v_s` | `0` | `0` | `1` | `0` | `1,013,309,916,158,362` | `5.006416e-309` | `kt = 5841`, `POST_DO.v_s` | Subnormal `v_s = h_s * a_i = 5.006416e-309` flushed to `0.0` by CPU FTZ at `kt = 5841` (`0` ULP on GPU) |
| 15 | `EXIT.v_s` | `0` | `0` | `1` | `0` | `1,013,309,916,158,362` | `5.006416e-309` | `kt = 5841`, `EXIT.v_s` | Same as `POST_DO.v_s` (`0` ULP on GPU) |

---

## 6. Unit Tests, Revert Controls, Gradients, Denominator/`frq_m` Proofs, and Parent-Commit (`071dfa844369`) Control

1. **Statement Family 1 bit-for-bit & revert control (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2212-2352`):**
   * Pins `_dh_step` (`_nemo_dh_ice_content_accumulation=True`) bit-for-bit (`0` ULP, `uint64` equality) against oracle `POST_DH.e_i` on all `6` former over-bar steps (`kt = 3996, 4014, 4092, 4400, 4413, 5241`) and against oracle `POST_DH.h_i` at `kt = 4359`, and proves that reverting `_nemo_dh_ice_content_accumulation=False` fails bit-exactness (`>= 3` ULP on `POST_DH.e_i`, `2` ULP on `POST_DH.h_i` at `kt = 4359`) and crosses `BAR = 1e-15` on `POST_DO.e_i` and `EXIT.e_i` on all `6` steps. Also pins the snow-ice flooding branch (`flood > 0`, `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:480-495`) on an active flooding column and proves reverting `_nemo_dh_ice_content_accumulation=False` changes `post_dh.e_ice`.
2. **Statement Family 2 bit-for-bit, revert control & cancelling-pair verification (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2355-2484`):**
   * Pins `_dh_step` (`_nemo_dh_snow_update=True`) bit-for-bit (`0` ULP, `uint64` equality) against oracle `POST_DH..POST_TEMP2.e_s`, `POST_DH..POST_TEMP2.h_s`, `POST_DO..EXIT.e_s`, and `POST_DO..EXIT.v_s` on all `7` former snow over-bar steps (`kt = 4943, 4945, 5460, 5644, 5822, 5960, 5962`), and proves that reverting `_nemo_dh_snow_update=False` fails bit-exactness and crosses `BAR = 1e-15` on `POST_DH..POST_TEMP2.e_s` at `kt = 5460, 5644` and on `POST_DO..EXIT.e_s` on all `7` steps. Also verifies `0`-ULP agreement on the cancelling-pair steps `kt = 5733, 5932`.
3. **Autodiff (`jax.value_and_grad` & `jax.jvp`) and literal `0`-ULP forward bits (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2487-2656`):**
   * Verifies `_nemo_ice_enthalpy_remap(..., content=ze_i_old)` and `si3_column_step_arrays(..., _nemo_dh_ice_content_accumulation=True, _nemo_dh_snow_update=True)` across `kt = 5241` and `kt = 5460`: forward primal outputs under `jax.value_and_grad` and `jax.jvp` are `0`-ULP (`uint64`) identical to standalone forward evaluation, gradients with respect to `(h_ice, h_snow, e_ice, e_snow, snow_precipitation, evaporation)` are finite and nonzero, reverse-mode agrees with forward-mode to `< 1e-10`, and a planted `jax.lax.stop_gradient` control on `content` fails.
4. **NEMO-literal denominator non-zero proof (`at_i_1d`, `zdE`) and `frq_m` verification (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2659-2776`):**
   * Scans all `8,760` steps (`26,280` ice layer-steps) of the C1D V2 record and proves that `at_i_1d(ji)` (`icethd_dh.f90:179`, `a = state.concentration`) satisfies `min(a) == max(a) == 0.9 > 0` (`0` zeros) and `zdE = zEi - zEw` (`icethd_dh.f90:268, 405`, `energy_difference`) satisfies `zdE <= -225,630.87 J/kg < 0` (`min(zdE) = -343,220.53 J/kg`, `0` zeros), with planted zero controls on both arrays that fail. Also verifies `frq_m` (`icethd_dh.f90:129`, `sbcssm.f90:180-184` where `frq_m = 1.0` under `ln_read_frq = .FALSE.`): at `kt = 1` (`qtr_ice_top = 0.0`), `frq_m = 1.0` and `frq_m = 0.0` are bitwise identical, whereas on sunlit steps (`1,375` steps with `qtr_ice_top > 0`, tested at `kt = 4242` where `qtr_ice_top = 31.775 W/m^2` and `qtr_ice_bot = 3.849 W/m^2`), `frq_m = 1.0` matches oracle `POST_DH.h_i` and `POST_DH.e_i` at `0` ULP while `frq_m = 0.0` fails bit-exactness (`112,845,198,105` ULP on `POST_DH.h_i`).
5. **Parent-commit (`071dfa8443699a44e68904caaf3d02e17417e757`) control across non-SI3 callers and the production lane-3 SI3 dispatcher (`tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:475-1023`, `packages/ice/legoesm/ice/sea_ice.py:286-298,500-517`):**
   * Loads `tridiagonal.py`, `bitz_lipscomb.py`, `snow.py`, `sea_ice.py`, and `scm.py` from Round 6's actual parent commit `071dfa8443699a44e68904caaf3d02e17417e757` (and Round 5's parent `a572f4b2540351bf49328d1533211427f9c9d772`) via `git show`, and runs both:
     1. All non-SI3 callers (`bitz_lipscomb_conduction_step`, `thomas_solve(..., "normalised")`, `IceColumnModel.step()` slab, and `step_sea_ice` dynamic-v2), proving every non-SI3 output stays bit-for-bit (`uint64`) identical to parent with 4 planted code-path controls that fail; and
     2. The production lane-3 SI3 dispatcher (`step_sea_ice` with `thermo_scheme="si3_bl99"` -> `_si3_step_with_trace`, `packages/ice/legoesm/ice/sea_ice.py:286-298,500-517`) under `jax.jit` on C1D steps `kt = 5241` (Statement Family 1: ONLY `new_state.e_ice.data` changes vs `071dfa844369`, moving to `0` ULP vs oracle `POST_DH.e_i`), `kt = 3996` (Statement Family 1: ONLY `new_state.h_ice.data` and `new_state.e_ice.data` change vs `071dfa844369`, moving to `0` ULP vs oracle `POST_DH.h_i` and `POST_DH.e_i`), `kt = 5460` (Statement Family 2: ONLY `new_state.h_snow.data` and `new_state.e_snow.data` change vs `071dfa844369`, moving to `0` ULP vs oracle `POST_DH.h_s` and `POST_DH.e_s`), and `kt = 43` (bit-for-bit identical vs `071dfa844369` across all 9 `SI3ColumnState` fields and all 22 `TileResponse` fields), with three code-path plants inside `_si3_step_with_trace` / `si3_column_step_arrays` (`packages/ice/legoesm/ice/sea_ice.py:297,500-517`) that make the control fail.

---

## 7. Choices List and Review Disposition

* **CHOICES:**
  * `ASKED`: **DECISION 156** (classify and close D5 per platform before starting the ORCA2 rungs), **DECISION 158** (certify the SI3 ladder on CPU `fp64` production JIT with XLA FMA prevented in code via `nemo_source_round`, subnormal cases as a stated platform floor, and each rung spot-checked on an idle GPU), and **DECISION 159** (run one round per debt group with preregistered before/after counts and remaining owner rankings).
  * `UNASKED`:
    1. In the initial Round 6 code commit (`fdd8a96c8d3ae682f1d35161ef515a47b2620ba9`), three non-literal choices were introduced unasked: (a) a denominator guard `safe_dE = jnp.where(internal_melt, -1.0e-20, energy_difference)` in surface and basal ice melt (`icethd_dh.f90:268, 405`), (b) an ice-concentration floor `/ jnp.maximum(a, _EPS10)` in `_nemo_dh_snow_update=True` (`icethd_dh.f90:179`), and (c) omitting `frq_m` from the transmitted-flux product `qtr_ice_bot_1d(ji) * frq_m_1d(ji)` in `zf_tt` (`icethd_dh.f90:129`). In commit `d3cdcf21381e0f72b2f34575f348536a3b163338` (review disposition for Claude and codex), both unasked denominator guards were removed in favor of NEMO's literal `/ energy_difference` and `/ a` divisions (`packages/ice/legoesm/ice/bitz_lipscomb.py:1091,1129,1130`, proved non-zero across all `8,760` steps in `tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2659-2776`), and `frq_m: jax.Array | float = 1.0` (`packages/ice/legoesm/ice/bitz_lipscomb.py:41,1067-1068,1901`, matching `sbcssm.f90:183` and `icethd_dh.f90:129`) was included in `SI3SurfaceForcing` and `zf`.
    2. `_nemo_dh_ice_content_accumulation: bool = True` and `_nemo_dh_snow_update: bool = True` in `packages/ice/legoesm/ice/bitz_lipscomb.py:1021-1022,1902-1904` follow the existing private `_nemo_*` flag pattern and Rule 3 (the NEMO-literal value is the default on the SI3 path; non-SI3 callers are bit-for-bit unchanged). No configuration classes (`SeaIceConfig`, `SI3ThermoConfig`) were modified.
* **Independent review:** `independent review: operator`

### 7.1 Operator Rebase Record (`071dfa8443` -> `d4d4cdb08d`)

The fast-forward landing rebase replayed the five approved Round 6 commits onto landed Round 4 O1 tip `d4d4cdb08df9866f28e295aad73a0f59f49eb473`. Five conflict hunks were resolved:

1. `docs/ice/fidelity/si3_r5_d5_receipt.md`, unit-test/parent-control hunk: retained Round 6's moved D5 probe ranges (`1740-2209` and parent helper `475-1023`) and re-anchored the landed O1-shifted SI3 dispatcher from `497-514` to `499-516`; no assertion or statement was dropped.
2. `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py`, `FILES` hunk: retained both the landed O1 compiled `ORCA2_SAS_ICE_O1_REC/.../icevar.f90` entry and Round 6's C1D `icevar.f90` alias, then preserved every other O1 and D5/Round 6 file entry exactly once.
3. The same gate, `CITATION_MAP` hunk: retained the complete expanded D5/Round 6 map and the complete O1 map, replaced only the superseded pre-O1 `sea_ice.py:...497-514` keys with `...499-516`, and introduced no duplicate key.
4. The same gate, `_NAME` hunk: retained the O1 path forms (`ocean.output.excerpt`, `cfgs/ref_cfgs.txt`, acquisition `run.sh`, and both `si3_r4_o1_*` receipts) together with both `si3_r6_d5_*` receipt names.
5. The final Round 6 documentation-fix commit's `CITATION_MAP` cleanup hunk: applied every approved Round 6 removal, replacement, and added exact citation to the already-combined map while preserving every O1-only key and value. Semantic comparison against both rebase sides found no missing entry, no altered retained value, and no newly duplicated key.
