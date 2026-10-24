# SI3 Ladder Rung O2 Preregistration — Round 7 (`si3_r7`): Scalar Initial State & Levitating-Ice Initial SSH (`ORCA2_SAS_ICE`)

- **Date**: 2026-10-24
- **Lane**: `si3` (Phase-3 NEMO 5.0.2 Sea-Ice Ladder, Round 7 / `si3_r7`)
- **Rung**: **O2** (`DECISION 153`, `DECISION 159`, `docs/ice/fidelity/si3_ladder_design.md:222`)
- **Parent commit**: `fe19fb330ecbe42737ac71b2fe152a492f82ac15` (Rung O1 / `si3_r4`, built on `a572f4b2540325343f71225777438411f1cb0803`)
- **Reference configuration**: `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE` (`jpl = 5`, `nlay_i = 10`, `nlay_s = 5`, `Ni0glo = 180`, `Nj0glo = 148`, `nn_hls = 2`, `jpi = 184`, `jpj = 152`)

---

## 1. Scope and Design (`DECISION 153`, `DECISION 159`)

Per `docs/ice/fidelity/si3_ladder_design.md:222` and `DECISION 159`:

- **Rung O2** uses the same `ORCA2_SAS_ICE` diagnostic deck as Rung O1 (`ln_icedyn = .false.`, `ln_icethd = .false.`, `nn_it000 = 1`, `nn_itend = 2`, `nn_fsbc = 1`, `nn_stock = 1`), initialized from the recorded `sst_m`/`sss_m` and `&namini` namelist scalars:
  1. **SST / freezing-point and hemisphere-dependent scalar initial state** (`src/ICE/iceistate.F90:124-126`, `:203-207`, `:262-304`; `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:174-175`, `:252-256`, `:309-338`) right before category fitting (`CALL ice_var_itd` at `src/ICE/iceistate.F90:311-316`, which belongs to Rung **O3**).
  2. **Levitating-ice initial SSH adjustment and snow/ice/pond mass** (`src/ICE/iceistate.F90:398-430`; `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:440-460`), reusing the ORCA2 lane's round-329 recipe function `orca2_initial_ice_ssh` (`ORCA2InitialIce`, `ORCA2_DECK_INITIAL_ICE`, `orca2_initial_ice_ssh` in `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` from commit `02acc8a1b10f2cc8511a6ba4d7fc233b34d73299` on `fidelity/nemo-testcases-orca2-rounds`).
- **Deck separation note**: `ORCA2_SAS_ICE`'s reference deck is **five-category (`jpl = 5`) with ponds (`ln_pnd = .true.`, `ln_pnd_LEV = .true.`, `ln_pnd_lids = .true.`)** and uses **TEOS-10 (`ln_teos10 = .true.`)** (`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:146`, `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg:9-50`). The ORCA2 ocean records (`ORCA2_OMIP_L4`) used `ln_eos80 = .true.`. They are never mixed: `orca2_initial_ice_ssh` keeps its exact `ln_eos80` default for `build_orca2_zps_card` (`0` ULP bit-identical on `ORCA2_OMIP_L4`) and supports `ORCA2_SAS_ICE`'s `ln_teos10 = .true.` freezing point and `DDPDD` global reduction (`glob_2Dsum`) for O2.
- **This round's deliverables**:
  1. Preregister O2 in this document and commit before any code edits.
  2. Port `ORCA2InitialIce`, `ORCA2_DECK_INITIAL_ICE`, and `orca2_initial_ice_ssh` into `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py` (keeping `build_orca2_zps_card` `0`-ULP identical) and transcribe O2's scalar initial state and levitating-ice SSH adjustment in the selectable SI3 path (`packages/ice/legoesm/ice/si3_orca2_state.py`, `packages/ice/legoesm/ice/config.py`, `packages/ice/legoesm/ice/sea_ice.py`, `packages/ice/legoesm/ice/__init__.py`).
  3. Build the O2 acquisition tool (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/`) and gate (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py`) extending O1's (importing O1's helpers from `nemo_si3_o1_orca2_sas_gate.py`) with observers of the scalar initial state right after NEMO builds it (`iceistate.F90:306` and `:438`), run its read-only `--preflight-only`, and stop with `ACQUISITION_NEEDED` without ever running NEMO.

---

## 2. NEMO 5.0.2 Oracle Source Analysis (`file:line` Citations)

All citations below are verified against `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`:

1. **Initialization call sequence in `ORCA2_SAS_ICE`**:
   - `src/SAS/nemogcm.F90:226-267`: `nemo_init` calls `dom_init( Kbb, Kmm, Kaa )` (`:239`) followed by `sbc_init( Kbb, Kmm, Kaa )` (`:261`).
   - `src/OCE/DOM/domain.F90:175-176`: in `dom_init`, `IF( l_SAS ) THEN ssh(:,:,:) = 0._wp` initializes `ssh(:,:,Kbb)` and `ssh(:,:,Kmm)` to `0.0_wp`.
   - `src/OCE/DOM/dommsk.F90:199-206`: `dom_msk` sets `ssmask(:,:) = MAXVAL( tmask(:,:,:), DIM=3 )` (equal to `tmask(:,:,1)` on ORCA2), `smask0(:,:) = tmask(A2D(0),1)` on the interior `(Nis0:Nie0, Njs0:Nje0)`, calls `dom_uniq( smask0_i, 'T' )` (`src/OCE/DOM/domutl.F90:91-121`), and sets `smask0_i(:,:) = ssmask(:,:) * smask0_i(:,:)`.
   - `src/OCE/SBC/sbcmod.F90:315-341`: `sbc_init` calls `sbc_ssm_init( Kbb, Kmm )` (`:315`, which reads `sst_m` and `sss_m` via `sbc_ssm` in `src/SAS/sbcssm.F90:66-169`, `:293-331`) and then `CALL ice_init( Kbb, Kmm, Kaa )` (`:340`).
   - `src/ICE/icestp.F90:230-314`: `ice_init` calls `ice_itd_init` (`:263`), `ice_thd_init` (`:267`), `ice_sbc_init` (`:273`), `ice_istate_init` (`:280`), `ice_istate( nit000, Kbb, Kmm )` (`:284`, when `.NOT. ln_rstart` and `nn_iceini_file /= 2`), `ice_var_glo2eqv(2)` (`:288`), `ice_var_agg(2)` (`:289`), and `ice_dyn_init` (`:291`).

2. **`&namini` parameters and `rn_amax_2d` in `ice_istate_init`**:
   - `src/ICE/iceistate.F90:453-534` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:483-562`) and `cfgs/SHARED/namelist_ice_ref:65-95`, `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg:22-37`:
     - `ln_iceini = .true.`, `nn_iceini_file = 0`
     - `rn_thres_sst = 2.0_wp`
     - `rn_hti_ini_n = 2.0_wp`, `rn_hti_ini_s = 1.0_wp`
     - `rn_hts_ini_n = 0.2_wp`, `rn_hts_ini_s = 0.2_wp`
     - `rn_ati_ini_n = 0.9_wp`, `rn_ati_ini_s = 0.9_wp`
     - `rn_smi_ini_n = 6.3_wp`, `rn_smi_ini_s = 6.3_wp`
     - `rn_tmi_ini_n = 270.0_wp`, `rn_tmi_ini_s = 270.0_wp`
     - `rn_tsu_ini_n = 270.0_wp`, `rn_tsu_ini_s = 270.0_wp`
     - `rn_tms_ini_n = 270.0_wp`, `rn_tms_ini_s = 270.0_wp`
     - `rn_apd_ini_n = 0.2_wp`, `rn_apd_ini_s = 0.2_wp`
     - `rn_hpd_ini_n = 0.05_wp`, `rn_hpd_ini_s = 0.05_wp`
     - `rn_hld_ini_n = 0.0_wp`, `rn_hld_ini_s = 0.0_wp`
   - `src/ICE/iceistate.F90:203-207` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:252-256`):
     `WHERE( gphit(:,:) >= 0._wp ) ; rn_amax_2d(:,:) = rn_amax_n ; ELSEWHERE ; rn_amax_2d(:,:) = rn_amax_s ; END WHERE` (`rn_amax_n = 0.997_wp`, `rn_amax_s = 0.997_wp` in `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg:14-15`).

3. **Freezing point `eos_fzp` (`t_bo` and `ztfrz`)**:
   - `src/ICE/iceistate.F90:125-126` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:174-175`):
     ```fortran
     CALL eos_fzp( sss_m(:,:), t_bo(:,:), kbnd=0 )
     t_bo(:,:) = ( t_bo(:,:) + rt0 ) * smask0(:,:)
     ```
     Note that `kbnd=0` (`DO_2D( 0, 0, 0, 0 )` in `src/OCE/TRA/eosbn2.F90:1668`) computes `t_bo` only on the interior `Nis0:Nie0, Njs0:Nje0`, and `smask0` is `tmask(Nis0:Nie0, Njs0:Nje0, 1)` (`0._wp` on halos).
   - `src/ICE/iceistate.F90:263` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:310`):
     ```fortran
     CALL eos_fzp( sss_m(:,:), ztfrz(:,:), kbnd=nn_hls )
     ```
     With `kbnd=nn_hls`, `ztfrz` is evaluated across the full `(1:jpi, 1:jpj)` array including halos.
   - `src/OCE/TRA/eosbn2.F90:1650-1710`:
     - Under `ln_teos10 = .true.` (`neos = np_teos10`, `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:146`, `src/OCE/TRA/eosbn2.F90:1674-1683`):
       `zs = SQRT( ABS( psal(ji,jj) ) * r1_S0 )` where `r1_S0 = 0.875_wp / 30.76941_wp = 1._wp / 35.16504_wp` (`src/OCE/TRA/eosbn2.F90:69`), and
       `pfzp(ji,jj) = ((((1.46873e-03_wp * zs - 9.64972e-03_wp) * zs + 2.28348e-02_wp) * zs - 3.12775e-02_wp) * zs + 2.07679e-02_wp) * zs - 5.87701e-02_wp`
       `pfzp(ji,jj) = pfzp(ji,jj) * psal(ji,jj)`.
     - Under `ln_eos80 = .true.` (`neos = np_eos80`, `src/OCE/TRA/eosbn2.F90:1691-1696`):
       `pfzp(ji,jj) = ( -0.0575_wp + 1.710523e-3_wp * SQRT( psal(ji,jj) ) - 2.154996e-4_wp * psal(ji,jj) ) * psal(ji,jj)`.

4. **SST threshold mask `zmsk` and 10-field scalar initial state (`iceistate.F90:264-304`)**:
   - `src/ICE/iceistate.F90:264-266` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:311-313`):
     ```fortran
     WHERE( ( sst_m(:,:) - ztfrz(:,:) ) * tmask(:,:,1) >= rn_thres_sst )   ;   zmsk(:,:) = 0._wp
     ELSEWHERE                                                             ;   zmsk(:,:) = tmask(:,:,1)
     END WHERE
     ```
   - `src/ICE/iceistate.F90:275-303` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:320-338`):
     For `(ji, jj)` in `1:jpi, 1:jpj`:
     if `ff_t(ji,jj) >= 0._wp`, select `(_n)` scalars, else `(_s)` scalars; multiply each by `zmsk(ji,jj)` to form `zht_i_ini`, `zht_s_ini`, `zat_i_ini`, `zsm_i_ini`, `ztm_i_ini`, `zt_su_ini`, `ztm_s_ini`, `zapnd_ini`, `zhpnd_ini`, `zhlid_ini`; then scale pond concentration by ice concentration:
     ```fortran
     zapnd_ini(:,:) = zapnd_ini(:,:) * zat_i_ini(:,:)
     ```

5. **Levitating-ice initial SSH branch (`iceistate.F90:398-430`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:440-460`)**:
   - After `ice_var_itd` distributes the scalar state across categories (`:311-316`) and `ice_var_agg(1)` sums `vt_i`, `vt_s`, `at_i`, `at_ip`, `vt_ip`, `vt_il` on `A2D(0)` (`:401`) followed by `lbc_lnk` on `vt_i, vt_s, vt_ip, vt_il` (`src/ICE/iceistate.F90:407`, removed in `ORCA2_OMIP_L4`'s older preprocessed file or idempotent when inputs already satisfy `lbc_lnk`):
     ```fortran
     snwice_mass  (:,:) = tmask(:,:,1) * ( rhos * vt_s(:,:) + rhoi * vt_i(:,:) + rhow * ( vt_ip(:,:) + vt_il(:,:) ) )
     snwice_mass_b(:,:) = snwice_mass(:,:)
     IF( .NOT.ln_ice_embd ) THEN
        IF( .NOT.ln_rstart ) THEN
           area = glob_2Dsum( 'iceistate', e1e2t(:,:) * ssmask(:,:) )
           zsshadj = glob_2Dsum( 'iceistate', snwice_mass(:,:) * r1_rho0 * e1e2t(:,:) ) / area
           WHERE( ssmask(:,:) == 1._wp )
              ssh(:,:,Kmm) = ssh(:,:,Kmm) - zsshadj(:,:)
              ssh(:,:,Kbb) = ssh(:,:,Kmm)
           END WHERE
        ENDIF
     ENDIF
     ```
   - For `ORCA2_SAS_ICE`'s `&namini` parameters (`rn_hti_ini_n = 2.0`, `rn_hti_ini_s = 1.0`, `rn_hts_ini = 0.2`, `rn_ati_ini = 0.9`, `rn_apd_ini = 0.2`, `rn_hpd_ini = 0.05`, `rn_hld_ini = 0.0`), the snow-to-ice excess in `ice_var_itd_1cMc` (`src/ICE/icevar.F90:1266`: `zdh = MAX(0._wp, (rhos * ph_s + (rhoi - rho0) * ph_i) * r1_rho0)`) is identically `0._wp` in every category (`(330 * 0.2/1.0 - 109) = -43 < 0`), no category thickness clips at `hi_max(5) = 99.0`, and `at_i = 0.9 <= rn_amax = 0.997`. Thus `vt_i = zht_i_ini * zat_i_ini`, `vt_s = zht_s_ini * zat_i_ini`, `vt_ip = zhpnd_ini * zapnd_ini`, and `vt_il = zhlid_ini * zapnd_ini`, so `snwice_mass` and `zsshadj` depend solely on the O2 scalar initial state and are reproduced bit-for-bit (`0` ULP) by `orca2_initial_ice_ssh`.

---

## 3. Frozen Record Schema for O2 (`O2SI3I01` + `O1SI3R01`)

The O2 acquisition tool records two binary snapshot families under `o2_si3_records/`:

1. **`o2_si3_istate_rankXXXX.bin` (magic `O2SI3I01`, version `1`)**:
   - Written once per MPI rank at the end of `ice_istate` (`src/ICE/iceistate.F90:438`), capturing:
     - **Integer header (22 `int32` scalars)**: `kt`, `mpprank`, `narea`, `jpi`, `jpj`, `jpl`, `nlay_i`, `nlay_s`, `nn_hls`, `Ni0glo`, `Nj0glo`, `nimpp`, `njmpp`, `Nis0`, `Nie0`, `Njs0`, `Nje0`, `Kbb`, `Kmm`, `nn_iceini_file`, `ln_iceini`, `ln_ice_embd`.
     - **Real header (26 `float64` scalars)**: `rn_thres_sst`, `rn_hti_ini_n`, `rn_hti_ini_s`, `rn_hts_ini_n`, `rn_hts_ini_s`, `rn_ati_ini_n`, `rn_ati_ini_s`, `rn_smi_ini_n`, `rn_smi_ini_s`, `rn_tmi_ini_n`, `rn_tmi_ini_s`, `rn_tsu_ini_n`, `rn_tsu_ini_s`, `rn_tms_ini_n`, `rn_tms_ini_s`, `rn_apd_ini_n`, `rn_apd_ini_s`, `rn_hpd_ini_n`, `rn_hpd_ini_s`, `rn_hld_ini_n`, `rn_hld_ini_s`, `rn_amax_n`, `rn_amax_s`, `rt0`, `area`, `zsshadj`.
     - **37 Fortran-ordered 2D `(jpi, jpj)` `float64` arrays** (`898,968` bytes per rank on `184 x 152`; `66,520` bytes per rank on the `10 x 9` 4-rank layout fixture):
       - Domain & surface inputs (13): `tmask`, `umask`, `vmask`, `ssmask`, `smask0`, `smask0_i`, `e1e2t`, `ff_t`, `gphit`, `sst_m`, `sss_m`, `ssh_before_kbb`, `ssh_before_kmm`.
       - Pre-`ice_var_itd` O2 scalar initial state (14, captured at `src/ICE/iceistate.F90:306`): `t_bo`, `ztfrz`, `zmsk`, `zht_i_ini`, `zht_s_ini`, `zat_i_ini`, `zsm_i_ini`, `ztm_i_ini`, `zt_su_ini`, `ztm_s_ini`, `zapnd_ini`, `zhpnd_ini`, `zhlid_ini`, `rn_amax_2d`.
       - Post-`ice_istate` aggregated volumes, snow/ice mass, and levitating-ice SSH (10, captured at `src/ICE/iceistate.F90:438`): `vt_i`, `vt_s`, `at_i`, `at_ip`, `vt_ip`, `vt_il`, `snwice_mass`, `snwice_mass_b`, `ssh_kbb`, `ssh_kmm`.

2. **`o2_si3_state_{entry,step0001,step0002}_rankXXXX.bin` (magic `O1SI3R01`, version `1`)**:
   - Identical schema to Rung O1 (`docs/ice/fidelity/si3_r4_o1_preregister.md`, `868,548` bytes per rank on `184 x 152`), read and stitched by importing O1's helpers (`read_o1_binary_snapshot`, `stitch_o1_binary_snapshots`, `o1_snapshot_to_si3_state`, `_compare_si3_states_wet_and_full`) from `scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py`.

---

## 4. Quantitative Fidelity Bar and Pre-Registered Negative-Control Plants (`P1–P7`)

### 4.1 Acceptance Bar (`DECISION 159`)

- **Bit-for-bit bar**: `max_ulp == 0` (`max|Delta| / max(|ref|, 1.0) <= 1e-15`) across all wet cells (`tmask(:,:,1) == 1`) and full `(184, 152)` arrays for every O2 scalar initial state field (`t_bo`, `ztfrz`, `zmsk`, `zht_i_ini`, `zht_s_ini`, `zat_i_ini`, `zsm_i_ini`, `ztm_i_ini`, `zt_su_ini`, `ztm_s_ini`, `zapnd_ini`, `zhpnd_ini`, `zhlid_ini`, `rn_amax_2d`, `snwice_mass`, `snwice_mass_b`, `ssh_kbb`, `ssh_kmm`), `0` ULP on the scalar reductions `area` and `zsshadj`, and `0` ULP across all 114 O1 state/moment/diagnostic fields at `entry`, `kt = 1`, `kt = 2`, and `rst_1step`.
- **Dry signed-zero tolerance**: Dry cells (`tmask == 0`) are excluded from wet-masked scoring; a dry `-0.0` fixture proves dry signed-zero differences do not fail wet-masked scoring while a wet `-0.0` / `1`-ULP perturbation at the first wet cell `(wi, wj)` (`wet_index.size > 0`) fires.

### 4.2 Pre-Registered Negative-Control Plants (`P1–P7`)

| Plant ID | Target Check | Planted Violation | Required Outcome |
|---|---|---|---|
| **P1** | Parent (`fe19fb330ecb` and `a572f4b25403`) vs current commit caller invariance | Perturb `bitz_lipscomb_conduction_step`, `_si3_step_with_trace`, `si3_o1_step`, or `orca2_initial_ice_ssh` (`build_orca2_zps_card`) by 1 ULP | Fails `0`-ULP invariance assertion |
| **P2** | 1-ULP O2 scalar initial state and SSH adjustment gate | Perturb `zht_i_ini[wi, wj]`, `ztfrz[wi, wj]`, or `ssh_kmm[wi, wj]` at first wet cell `(wi, wj)` by 1 ULP (`np.nextafter(..., np.inf)`) | Fails gate with `STATUS PLANT-FIRED` |
| **P3** | O2 physics formula plants (`eos-wrong-branch`, `tbo-halo-leak`, `pond-unscaled`, `hemisphere-swap`, `ssh-sign-or-div`) | Swap `teos10`/`eos80` in `eos_fzp`; omit `smask0` interior mask on `t_bo`; omit `zapnd_ini *= zat_i_ini`; swap NH/SH thicknesses; or flip sign / omit ponds in `orca2_initial_ice_ssh` | Fails gate with `STATUS PLANT-FIRED` |
| **P4** | North-fold `dom_uniq` (`smask0_i`) and `lbc_lnk` sign plants | Omit T-fold right-half deduplication in `smask0_i` (`domutl.F90:91-121`) or flip north-fold sign (`psgn`) | Fails gate with `STATUS PLANT-FIRED` |
| **P5** | Compiled `gfortran` 4-rank layout fixture (`O2SI3I01` + `O1SI3R01`) | `rank-collapse`, `wrong-offset`, `field-swap`, `truncation`, `bad-magic`, `missing-frame` | Fails layout verifier with `STATUS PLANT-FIRED` |
| **P6** | End-to-end `jax.jit` / `jax.value_and_grad` / `jax.jvp` differentiability | Wrap differentiable O2 inputs (`sss_m`, `sst_m`, `&namini` thickness/concentration/pond parameters) in `jax.lax.stop_gradient` | Fails AD gate with `STATUS PLANT-FIRED` (`grad_norm == 0.0`) |
| **P7** | Preflight checklist items 1–7 and admission provenance gates | Continuation `&`, undeclared identifier, bare `USE iso_fortran_env`, `ref_cfgs.txt`/`work_cfgs.txt`, namelist group/var/stem/fsbc/deviation, missing output dir, UTF-8, disk/RAM floors, target dir, excerpt SHA, log tokens, `same_program`, `EXIT` trap, and admission plants (`stamp`, `same-program`, `binary-sha`, `input-sha`, `missing-complete`, `missing-stop0`, `ref-rec-diff`, `1ulp-istate`, `1ulp-ssh`, `1ulp-state`, `rst-missing`, `rst-1ulp`) | Fails preflight / admission with `STATUS PLANT-FIRED` |

## Operator amendment after the landed O1 review series

This preregistration was originally committed on the pre-fix O1 head. The O2
implementation was subsequently rebased onto lane commit
`042c7bd74c2944dab8c39318687ab0cef54f9d2b`; the landed O1 helpers are imported
instead of copied. The following source verification supersedes any conflicting
prediction above:

- The configuration's own compiled source is
  `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iceistate.f90:308-338,439-460`.
- Its levitating-ice branch has no inner `ln_rstart` condition. It subtracts
  `zsshadj` independently from the existing `ssh(:,:,Kmm)` and
  `ssh(:,:,Kbb)` values (`:456-459`); it does not assign Kbb from
  the already-updated Kmm field.
- Restart NetCDF payloads are halo-free. O2 scores their wet interior against
  the corresponding model interior and scores NEMO-defined wet seam/fold halos
  only in rank-local writer records.
- The round remains acquisition-tool-only: preflight is permitted, NEMO is not.
