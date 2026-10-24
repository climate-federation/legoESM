# SI3 Ladder Round 8 (`si3_r8`) Receipt — Rung O3 Model Preparation (ORCA2 Prather Advection Across Cyclic Seam and North Fold, Without Acquisition)

* **Round:** `si3_r8` (SI3 sea-ice fidelity ladder, Rung O3 model preparation without acquisition and without running NEMO)
* **Branch:** `fidelity/nemo-testcases-si3-ladder`
* **Parent commit (O1 + D5 landed):** `042c7bd74c2944dab8c39318687ab0cef54f9d2b`
* **Preregistration document:** `docs/ice/fidelity/si3_r8_o3_preregister.md:1-96` (committed at `9e544a3124b946a88de9519dbdf713972c5e61e9` prior to model edits and test execution)
* **Design reference:** `docs/ice/fidelity/si3_ladder_design.md:187-218` (Rung O3 design row) and `docs/ice/fidelity/si3_r4_o1_receipt.md:1-142` (Rung O1 ORCA2 SI3 state, cyclic seam, and T-pivot north fold)
* **Decisions binding this round:** **DECISION 153**, **DECISION 158**, **DECISION 159** (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_si3/standing_note.md`)
* **Independent review:** `independent review: operator`

---

## 1. Outcome and Diagnosis — CONFIRMED versus PLAUSIBLE

### 1.1 Outcome

Rung O3 model preparation (`si3_r8`) is **LANDED**. NEMO 5.0.2's multidimensional second-order moment-conserving Prather (1986) sea-ice advection scheme (`src/ICE/icedyn_adv_pra.F90` / `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:46-1588`) and its surrounding `ice_dyn` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:126-265`), `ice_dyn_adv` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv.f90:112-178`), `ice_var_zapsmall` / `ice_var_zapneg` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:615-853`), and `ice_stp` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:181-272,429-599`) wrapper sequence are transcribed as a selectable SI3 path (`scheme="si3_orca2_o3"` / `build_si3_orca2_o3_config`) in the ONE shared ice package (`packages/ice/legoesm/ice/`).

In accordance with the round brief, no NEMO build or execution was performed and no O1/O2 acquisition tools (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/`), `iceistate` code, or initial-state paths were touched.

### 1.2 Search-Before-Build Inventory

Before writing any code, the existing sea-ice package and Rung O1 / Lane-3 implementations were searched:

1. **Rung O1 ORCA2 state, cyclic east-west seam, and T-pivot north fold (`packages/ice/legoesm/ice/si3_orca2_state.py:87-2196`):**
   Found `SI3State` (`packages/ice/legoesm/ice/si3_orca2_state.py:328-615`), `SI3PratherMoments` (`:164-176`), `SI3AllPratherMoments` (`:179-224`), `si3_lbc_lnk_orca2` (`:1244-1384`), `si3_apply_state_halos_orca2` (`:1412-1470`), `si3_var_glo2eqv` (`:883-1054`), `si3_var_agg` (`:1057-1212`), `si3_update_snwice_mass` (`:1215-1241`), and `si3_o1_step` (`:1482-1565`). Reused `si3_lbc_lnk_orca2` and `si3_apply_state_halos_orca2` directly for every subcycle and post-dynamics halo/fold exchange without duplicating any LBC logic.
2. **Lane-3 periodic Prather kernels (`packages/ice/legoesm/ice/transport.py:91-1944`):**
   Found `_si3_prather_limit_x` (`packages/ice/legoesm/ice/transport.py:608-638`), `_si3_prather_limit_y` (`:641-671`), `_si3_prather_merge_from_left` (`:674-717`), `_si3_prather_merge_from_right` (`:720-763`), `_si3_prather_merge_from_below` (`:766-809`), `_si3_prather_merge_from_above` (`:812-855`), `_si3_prather_x_substep` (`:858-1229`), and `_si3_prather_y_substep` (`:1232-1607`). Reused the elemental Prather slope limiters (`_si3_prather_limit_x`, `_si3_prather_limit_y`) and receiver moment merges (`_si3_prather_merge_from_left`, `_si3_prather_merge_from_right`, `_si3_prather_merge_from_below`, `_si3_prather_merge_from_above`) inside the new non-periodic stencil-indexed `si3_adv_x` (`packages/ice/legoesm/ice/transport.py:2014-2480`) and `si3_adv_y` (`:2483-2938`) while leaving every Lane-3 function untouched.

### 1.3 Diagnosis — CONFIRMED versus PLAUSIBLE

* **CONFIRMED (Test a — Statement-level `0`-ULP pins and negative-control plants `O3-A1..A4`):**
  Every transcribed statement in `si3_adv_x` (`packages/ice/legoesm/ice/transport.py:2014-2480`), `si3_adv_y` (`:2483-2938`), `si3_icemax2d_pra` (`:2941-2971`), `si3_icemax3d_pra` (`:2974-3003`), `si3_hbig_pra` (`:3018-3283`), `si3_hsnow_pra` (`:3286-3385`), `si3_var_zapneg` (`:3388-3581`), `si3_var_zapsmall` (`:3588-3793`), `si3_dyn_adv_pra_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:2350-2953`), `si3_dyn_orca2` (`:2956-3334`), and `si3_o3_step` (`:3337-3463`) matches its independent scalar/reference Fortran transcription to `0` ULP on inexact `float64` operands (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:583-1279`). All four families of statement-reversal/reordering plants (`Plant O3-A1`, `Plant O3-A2`, `Plant O3-A3`, `Plant O3-A4`) fire and fail the `0`-ULP check when enabled.
* **CONFIRMED (Test b — Lane-3 `ICE_ADV1D` / `ICE_ADV2D` and production dispatcher invariance at `0` ULP vs parent `042c7bd74c29` with plants `O3-B1..B2`):**
  Loading `packages/ice/legoesm/ice/*` directly from parent commit `042c7bd74c2944dab8c39318687ab0cef54f9d2b` via `git show` and comparing against the current workspace across `advect_si3_prather_1d`, `advect_si3_prather_2d`, `step_ice_adv1d_card`, `step_ice_adv2d_card`, `step_sea_ice` (`slab`, `dynamic-v2`, `si3_bl99`, `si3_orca2_o1`), and `si3_o1_step` yields `0` ULP across every output leaf (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:1329-1572`), and existing Lane-3 (`tests/ice/fidelity/test_nemo_si3_phase2_gate.py:1-131`, `tests/ice/fidelity/test_nemo_si3_phase2_adv2d_gate.py:1-215`) and O1 (`tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py:1-545`) gate suites pass unchanged. Code-path plants `Plant O3-B1` and `Plant O3-B2` fail with `max ULP = 1 != 0`.
* **CONFIRMED (Test c — Conservation of all 11 extensive families on ORCA2 across the cyclic seam to `< 2e-16` relative error and residual budget closure with plant `O3-C1`):**
  On the closed ORCA2 `(184, 152)` domain with active east-west cyclic seam transport (`max |u_ice[2, :]| > 0.1 m/s`), the interior area-weighted integral `SUM_{ji=3..182, jj=3..150} field(ji, jj) * e1e2t(ji, jj)` of all 11 extensive families (`v_i, v_s, a_i, oa_i, e_s, e_i, szv_i, sv_i, a_ip, v_ip, v_il`) is conserved to `max_rel_err = 1.583392961495183e-16` at `kt = 1` (`x -> y`, `icycle = 1`) and `max_rel_err = 1.7546464555576054e-16` at `kt = 2` (`y -> x`, `icycle = 2`) — over `50x` tighter than the preregistered `1.0e-14` threshold (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:1707-1818`). When `Hbig_pra` / `Hsnow_pra` / `ice_var_zapneg` clipping is triggered, `(diag_vice + diag_vsnw + diag_vpnd - diag_adv_mass) + wfx_res` and `(diag_heat - diag_adv_heat) + hfx_res` close cellwise to floating-point round-off (`<= 1.0e-13`). `Plant O3-C1` (`_plant_corrupt_receiver_flux=True`) breaks conservation (`rel_err > 1.0e-8`) and fires.
* **CONFIRMED (Test d — End-to-end `jax.jit` `0`-ULP primal parity and `reverse == forward` AD through `si3_o3_step` with plant `O3-D1`):**
  Through a full ORCA2 Prather advection step (`si3_o3_step`), `jax.jit` primal outputs match eager execution to `0` ULP across all 11 extensive families, `ato_i`, all 55 Prather moment arrays, and all diagnostics (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:1827-2009`). Reverse-mode gradients (`jax.value_and_grad`) with respect to all 10 active extensive fields (`v_i, v_s, a_i, oa_i, e_s, e_i, szv_i, a_ip, v_ip, v_il`), ice velocities (`u_ice, v_ice`), and Prather moments (`sx, sy, sxx, syy, sxy`) are finite and non-zero, and match forward-mode `jax.jvp` to relative error `1.3341694060629237e-15 <= 1.0e-11`. `Plant O3-D1` (`_plant_stop_gradient_vi=True`) zeros the gradient and fires.
* **PLAUSIBLE:** None. Every claim in this receipt is established by a committed test or source inspection.

---

## 2. Statement-by-Statement NEMO 5.0.2 `ORCA2_SAS_ICE` Transcription Ledger

Every statement transcribed in `si3_r8` is pinned to the compiled `ORCA2_SAS_ICE` preprocessed sources under `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/`:

### 2.1 Pre-Advection Thickness Recovery and `ice_dyn` Dispatch (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:126-265`)

1. **Pre-advection thickness recovery (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:156-170`):**
   ```fortran
         WHERE( a_i(:,:,:) >= epsi20 )
            h_i(:,:,:) = v_i(:,:,:) / a_i(:,:,:)
            h_s(:,:,:) = v_s(:,:,:) / a_i(:,:,:)
         ELSEWHERE
            h_i(:,:,:) = 0._wp
            h_s(:,:,:) = 0._wp
         END WHERE
         !
         WHERE( a_ip(:,:,:) >= epsi20 )
            h_ip(:,:,:) = v_ip(:,:,:) / a_ip(:,:,:)
            h_il(:,:,:) = v_il(:,:,:) / a_ip(:,:,:)
         ELSEWHERE
            h_ip(:,:,:) = 0._wp
            h_il(:,:,:) = 0._wp
         END WHERE
   ```
   * **legoESM transcription:** `si3_dyn_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:3010-3018`) computes `h_i`, `h_s`, `h_ip`, and `h_il` on the full `(184, 152, jpl)` domain before entering `si3_dyn_adv_pra_orca2`, using gradient-safe denominator guards (`jnp.where(ai_ge_eps20, state.a_i, _SI3_PRA_ONE)`) that preserve `0`-ULP NEMO-literal forward bits wherever `a_i >= epsi20` or `a_ip >= epsi20` while keeping reverse-mode AD finite on dry/ice-free cells. Because `ph_i, ph_s, ph_ip` are `INTENT(in)` to `ice_dyn_adv_pra` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:86-88`), they remain fixed at these pre-advection values across all subcycles `jt = 1..icycle` (`packages/ice/legoesm/ice/si3_orca2_state.py:2496-2501`).
2. **Prescribed 2-D advection selector `CASE ( np_dynADV2D )` and end-of-`ice_dyn` `lbc_lnk` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:187-228,253-260`):**
   ```fortran
         CASE ( np_dynADV2D )         !==  pure advection ==!   (2D w prescribed velocities)
            !
            u_ice(:,:) = rn_uice * umask(:,:,1)
            v_ice(:,:) = rn_vice * vmask(:,:,1)
            !CALL RANDOM_NUMBER(u_ice(:,:)) ; u_ice(:,:) = u_ice(:,:) * 0.1 + rn_uice * 0.9 * umask(:,:,1)
            !CALL RANDOM_NUMBER(v_ice(:,:)) ; v_ice(:,:) = v_ice(:,:) * 0.1 + rn_vice * 0.9 * vmask(:,:,1)
            ! ---
            CALL ice_dyn_adv   ( kt )                                          ! -- advection of ice
            CALL ice_var_zapsmall                                              ! -- zap small areas
   ```
   ```fortran
         IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN
            CALL lbc_lnk( 'icedyn', a_i , 'T', 1._wp, v_i , 'T', 1._wp, v_s , 'T', 1._wp, sv_i, 'T', 1._wp, oa_i, 'T', 1._wp, &
               &                    t_su, 'T', 1._wp, a_ip, 'T', 1._wp, v_ip, 'T', 1._wp, v_il, 'T', 1._wp, ldfull = .TRUE. )
         ELSE
            CALL lbc_lnk( 'icedyn', a_i , 'T', 1._wp, v_i , 'T', 1._wp, v_s , 'T', 1._wp, sv_i, 'T', 1._wp, oa_i, 'T', 1._wp, &
               &                    t_su, 'T', 1._wp, ldfull = .TRUE. )
         ENDIF
         CALL lbc_lnk( 'icedyn', e_i, 'T', 1._wp, e_s, 'T', 1._wp, szv_i, 'T', 1._wp, ldfull = .TRUE. )
   ```
   * **legoESM transcription:** `si3_dyn_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:3021-3221`) supports both prescribed state velocities (`u_ice, v_ice`) and optional `rn_uice * umask, rn_vice * vmask` override (`:3021-3026`), calls `si3_dyn_adv_pra_orca2` (`:3029-3081`), computes `diag_trp_*` (`:3084-3132`), runs `si3_var_zapsmall` (`:3135-3200`), and applies `si3_lbc_lnk_orca2(..., "T", _SI3_PRA_ONE, nn_hls=nn_hls, ldfull=True)` to `a_i, v_i, v_s, sv_i, oa_i, t_su, a_ip, v_ip, v_il, e_i, e_s, szv_i` (`:3203-3221`).

### 2.2 Advection Wrapper and Transport Diagnostics (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv.f90:112-178`)

```fortran
      SELECT CASE( nice_adv )
      !                                !-----------------------!
      CASE( np_advUMx )                ! ULTIMATE-MACHO scheme !
         !                             !-----------------------!
         CALL ice_dyn_adv_umx( nn_UMx, kt, u_ice, v_ice, h_i, h_s, h_ip, &
            &                          ato_i, v_i, v_s, sv_i, oa_i, a_i, a_ip, v_ip, v_il, e_s, e_i, szv_i )
         !                             !-----------------------!
      CASE( np_advPRA )                ! PRATHER scheme        !
         !                             !-----------------------!
         CALL ice_dyn_adv_pra(         kt, u_ice, v_ice, h_i, h_s, h_ip, &
            &                          ato_i, v_i, v_s, sv_i, oa_i, a_i, a_ip, v_ip, v_il, e_s, e_i, szv_i )
      END SELECT

      !------------
      ! diagnostics
      !------------
      diag_trp_ei(:,:) =    SUM(SUM( e_i  (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:,:) - e_i_b  (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:,:), dim=4 ), dim=3 ) * r1_Dt_ice
      diag_trp_es(:,:) =    SUM(SUM( e_s  (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:,:) - e_s_b  (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:,:), dim=4 ), dim=3 ) * r1_Dt_ice
      IF( nn_icesal == 4 ) THEN
         diag_trp_sv(:,:) = SUM(SUM( szv_i(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:,:) - szv_i_b(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:,:), dim=4 ), dim=3 ) * r1_Dt_ice
      ELSE
         diag_trp_sv(:,:) = SUM(     sv_i (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:)   - sv_i_b (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:)           , dim=3 ) * r1_Dt_ice
      ENDIF
      diag_trp_vi(:,:) =    SUM(     v_i  (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:)   - v_i_b  (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:)           , dim=3 ) * r1_Dt_ice
      diag_trp_vs(:,:) =    SUM(     v_s  (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:)   - v_s_b  (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:)           , dim=3 ) * r1_Dt_ice
```
* **legoESM transcription:** `si3_dyn_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:3084-3132`) evaluates the sequential `dim=4` (`jl = 1..jpl`) then `dim=3` (`jk = 1..nlay`) sums of `(field - field_b)` on `A2D(0)` (`2:182, 2:150`) multiplied by `r1_Dt_ice = sr(1.0 / rDt_ice)` with `nemo_source_round`, matching `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv.f90:156-164` and `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:438-457` (`store_fields`).

### 2.3 CFL Evaluation, Subcycle Loop, and Odd/Even Directional Split (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:46-510`)

1. **Module-saved Prather moment arrays (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:46-56`):**
   The 11 moment families (`sxice..sxyice`, `sxsn..sxysn`, `sxa..sxya`, `sxsal..sxysal`, `sxage..sxyage`, `sxc0..sxyc0`, `sxe..sxye`, `sxsi..sxysi`, `sxap..sxyap`, `sxvp..sxyvp`, `sxvl..sxyvl`, totaling 55 arrays) live in `SI3AllPratherMoments` (`packages/ice/legoesm/ice/si3_orca2_state.py:179-224`) inside `SI3State.prather`.
2. **CFL calculation, `icycle`, `zdt`, `z1_dt`, and face transports `zudy, zvdx` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:133-150`):**
   ```fortran
         zcfl =            MAXVAL( ABS( pu_ice(:,:) ) * rDt_ice * r1_e1u(:,:) )
         zcfl = MAX( zcfl, MAXVAL( ABS( pv_ice(:,:) ) * rDt_ice * r1_e2v(:,:) ) )

         CALL mpp_max( 'icedyn_adv_pra', zcfl, cdelay = 'cflice' )

         IF    ( zcfl > 1.5 ) THEN   ;   icycle = 3
         ELSEIF( zcfl >  .5 ) THEN   ;   icycle = 2
         ELSE                        ;   icycle = 1
         ENDIF
   !!$      !!test clem
   !!$      icycle=3
   !!$      !!test clem      
         zdt = rDt_ice / REAL(icycle)
         z1_dt = 1._wp / zdt

         ! --- transport --- !
         zudy(:,:) = pu_ice(:,:) * e2u(:,:)
         zvdx(:,:) = pv_ice(:,:) * e1v(:,:)
   ```
   * **legoESM transcription:** `si3_dyn_adv_pra_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:2462-2484`) computes `zcfl` with left-to-right `sr(sr(jnp.abs(u_ice) * rDt_ice) * r1_e1u)` and `sr(sr(jnp.abs(v_ice) * rDt_ice) * r1_e2v)`, selects `icycle in {1, 2, 3}` (or accepts an explicit `subcycles` override to replay delayed MPI `cdelay = 'cflice'` when needed), and computes `zdt = sr(rDt_ice / float(icycle))`, `zudy = sr(u_ice * e2u)`, and `zvdx = sr(v_ice * e1v)`.
3. **Subcycle halo margin `ihls` and pre-advection 9-point maxima (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:156-213,1519-1588`):**
   ```fortran
         DO jt = 1, icycle
            
            ! record at_i before advection (for open water)
            zati1(:,:) = SUM( pa_i(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:), dim=3 )

            IF( icycle == 1 ) THEN   ;   ihls = 0                   ! optimization
            ELSE                     ;   ihls = MAX( 0, nn_hls - jt )
            ENDIF
   ```
   ```fortran
      SUBROUTINE icemax2D_pra( ihls, pice , pmax )
   ...
         DO ji = Nis0-ihls, Nie0+ihls
            zmax1(ji) = MAX( epsi20, pice(ji,Njs0-1-ihls), pice(ji-1,Njs0-1-ihls), pice(ji+1,Njs0-1-ihls) )
            zmax2(ji) = MAX( epsi20, pice(ji,Njs0  -ihls), pice(ji-1,Njs0  -ihls), pice(ji+1,Njs0  -ihls) )
         END DO
         DO jj = ntsj-( ihls), ntej+(  ihls ) ; DO ji = ntsi-( ihls), ntei+(  ihls)
            zmax3 = MAX( epsi20, pice(ji,jj+1), pice(ji-1,jj+1), pice(ji+1,jj+1) )
            pmax(ji,jj) = MAX( epsi20, zmax1(ji), zmax2(ji), zmax3 )
            zmax1(ji) = zmax2(ji)
            zmax2(ji) = zmax3
         END DO   ;   END DO
      END SUBROUTINE icemax2D_pra
   ```
   * **legoESM transcription:** `si3_icemax2d_pra` (`packages/ice/legoesm/ice/transport.py:2941-2971`) and `si3_icemax3d_pra` (`:2974-3003`) evaluate the exact two-stage row-wise 3-point then 3-row `MAX(epsi20, ...)` reduction over `ji = Nis0-ihls .. Nie0+ihls, jj = Njs0-ihls .. Nje0+ihls`, called in `si3_dyn_adv_pra_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:2496-2523`) for `zhi_max, zhs_max, zhip_max, zei_max, zes_max` and `zszi_max` (`nn_icesal == 4`) or `zsi_max` (`nn_icesal /= 4`).
4. **Extensive field packing, odd/even sweep order, and intensive recovery (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:233-397`):**
   ```fortran
               IF( MOD( (kt - 1) / nn_fsbc , 2 ) ==  MOD( (jt - 1) , 2 ) ) THEN   !==  odd ice time step:  adv_x then adv_y  ==!
                  !                                                               !--------------------------------------------!
                  ! ---------------------- !
                  ! == mandatory fields == !
                  ! ---------------------- !
                  CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0ice , sxice , sxxice , syice , syyice , sxyice )    !--- ice volume
                  CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0ice , sxice , sxxice , syice , syyice , sxyice )
   ```
   ```fortran
               ! --- Recover the properties from their contents --- !
               DO jj = ntsj-( ihls), ntej+(  ihls ) ; DO ji = ntsi-( ihls), ntei+(  ihls)
                  pv_i (ji,jj,jl) = z0ice(ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
                  pv_s (ji,jj,jl) = z0snw(ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
                  poa_i(ji,jj,jl) = z0oi (ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
                  pa_i (ji,jj,jl) = z0ai (ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)
               END DO   ;   END DO
   ```
   * **legoESM transcription:** `si3_dyn_adv_pra_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:2540-2688`) packs `z0* = sr(field * e1e2t)` on `A2D(ihls+1)`, tests `(((kt - 1) // cfg.nn_fsbc) % 2) == ((jt - 1) % 2)` (`:2567`), runs `si3_adv_x(..., pcrh=1.0)` followed by `si3_adv_y(..., pcrh=0.0)` on odd parity or `si3_adv_y(..., pcrh=1.0)` followed by `si3_adv_x(..., pcrh=0.0)` on even parity across all 11 extensive families (`:2571-2655`), and recovers `field = sr(sr(z0 * r1_e1e2t) * tmask)` on `A2D(ihls)` (`:2658-2688`).
5. **Directional operators `adv_x` and `adv_y` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:513-957`):**
   * `si3_adv_x` (`packages/ice/legoesm/ice/transport.py:2014-2480`) implements `ji0 = 1 + ihls, jj0 = NINT(pcrh) + ihls` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:545-546`), Loop 1 over `A2D(ji0, jj0)` (`:550-630`), Loop 2 over `ji = ntsi-ji0 .. ntei+ji0-1, jj = ntsj-jj0 .. ntej+jj0` (`:632-651`, preserving NEMO's exact omission of `zfx` accumulation in the `put(ji,jj) < 0._wp` branch so `zfx(ji,jj)` remains `0._wp`), and Loop 3 over `ji = ntsi-(ji0-1) .. ntei+(ji0-1), jj = ntsj-jj0 .. ntej+jj0` (`:653-731`).
   * `si3_adv_y` (`packages/ice/legoesm/ice/transport.py:2483-2938`) implements `ji0 = NINT(pcrh) + ihls, jj0 = 1 + ihls` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:768-769`), Loop 1 over `A2D(ji0, jj0)` (`:773-853`), Loop 2 over `ji = ntsi-ji0 .. ntei+ji0, jj = ntsj-jj0 .. ntej+jj0-1` (`:855-875`, where `zfy` IS accumulated in the `pvt(ji,jj) < 0._wp` branch at `:868`), and Loop 3 over `ji = ntsi-ji0 .. ntei+ji0, jj = ntsj-(jj0-1) .. ntej+(jj0-1)` (`:877-955`).
   * **Differentiability & gradient-only guards (`DECISION 159`):** All forward branches (`zslpmax > 0._wp`, `put >= 0._wp`, `put < 0._wp`, `pvt >= 0._wp`, `pvt < 0._wp`) are evaluated with NEMO-literal IEEE-754 arithmetic; in untaken `jnp.where` arms (outside the active stencil or where `psm == 0._wp`), denominators (`psm`, `zpsm`) are guarded with `jnp.where(active & (psm != 0.0), psm, 1.0)` so reverse-mode cotangents never propagate `0.0 * NaN = NaN` from inactive outer halo cells (`packages/ice/legoesm/ice/transport.py:2167-2377,2629-2837`).

### 2.4 Post-Advection Regularization, Open-Water Evolution, and Subcycle Halo/Fold Exchanges (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:400-498,960-1156` and `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:615-853`)

1. **Advection rate diagnostics `diag_adv_mass, diag_adv_heat, diag_adv_salt` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:216-229,400-417`):**
   Transcribed in `si3_dyn_adv_pra_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:2526-2537,2691-2720`) on `A2D(0)` (`2:182, 2:150`).
2. **Overshoot limiter `Hbig_pra` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:421-422,960-1093`):**
   Transcribed in `si3_hbig_pra` (`packages/ice/legoesm/ice/transport.py:3018-3283`) on `A2D(ihls)` with `pdt = rDt_ice`, capping `h_ip`, `h_i` (bounded by `MIN(phi_max(ji,jj), hi_max(jpl))` from `state.bounds.hi_max[-1]`), `h_s`, `szv_i/v_i` (`nn_icesal == 4`) or `sv_i/v_i` (`nn_icesal /= 4`), `e_i/v_i`, and `e_s/v_s` when `pa_i(ji,jj,jcat) < 0.15` (or `pa_ip < 0.15` for `h_ip`), preserving NEMO's exact two-step `zfra = cap / zsi ; field = field * zfra` association and accumulating `wfx_res, hfx_res, sfx_res` on `A2D(0)`.
3. **Snow-load and pond-fraction limiter `Hsnow_pra` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:425,1096-1156`):**
   Transcribed in `si3_hsnow_pra` (`packages/ice/legoesm/ice/transport.py:3286-3385`) on `A2D(ihls)` with `pdt = rDt_ice`, removing snow excess `zvs_excess = MAX(0._wp, pv_s - pv_i * (rho0 - rhoi) * r1_rhos)` into `wfx_res` and `hfx_res` and enforcing `pa_ip = MIN(pa_ip, pa_i)`.
4. **Negative-value zapper `ice_var_zapneg` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:435`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:740-853`):**
   Transcribed in `si3_var_zapneg` (`packages/ice/legoesm/ice/transport.py:3388-3581`), executing the whole-array `WHERE( pv_i(:,:,:) <= 0._wp ) pa_i(:,:,:) = 0._wp` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:774`), zapping negative or orphan salt (`pszv_i` or `psv_i`), ice enthalpy (`pe_i`), snow enthalpy (`pe_s`), ice volume (`pv_i`), snow volume (`pv_s`), pond lid volume (`pv_il`, tested while `pv_ip` still holds its pre-zap value at `:831-834`), and pond volume (`pv_ip`, `:835-838`) on `A2D(ihls)`, accumulating `wfx_res, hfx_res, sfx_res` on `A2D(0)` (`:843-847`), and clamping `poa_i, pa_i, pa_ip < 0._wp` to `0._wp` (`:849-851`).
5. **Open-water fraction evolution `pato_i` and end-of-subcycle `lbc_lnk` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:438-494`):**
   ```fortran
         ! derive open water from ice concentration
         DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)
            zati2 = SUM( pa_i(ji,jj,:) )
            pato_i(ji,jj) = MAX( 0._wp, pato_i(ji,jj) - ( zati2 - zati1(ji,jj) )            &
               &                                      - (   ( zudy(ji,jj) - zudy(ji-1,jj) ) &   ! ad () for NP repro
               &                                          + ( zvdx(ji,jj) - zvdx(ji,jj-1) ) ) * r1_e1e2t(ji,jj) * zdt )
         END DO   ;   END DO
   ```
   * **legoESM transcription:** `si3_dyn_adv_pra_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:2841-2929`) updates `ato_i` on `A2D(0)` with the exact parenthesized divergence `((zudy(ji,jj) - zudy(ji-1,jj)) + (zvdx(ji,jj) - zvdx(ji,jj-1))) * r1_e1e2t(ji,jj) * zdt` and then calls O1's `si3_lbc_lnk_orca2(..., ldfull=True)` across all extensive fields (`'T', +1.0`), first-order moments `sx*, sy*` (`'T', -1.0`), second-order moments `sxx*, syy*, sxy*` (`'T', +1.0`), `ato_i` (`'T', +1.0`), and `wfx_res, hfx_res, sfx_res` (`'T', +1.0`).
6. **Small-ice zapper `ice_var_zapsmall` and post-dynamics `ice_stp` sequence (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:615-737`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:181-272,429-599`):**
   * `si3_var_zapsmall` (`packages/ice/legoesm/ice/transport.py:3588-3793`) implements `ice_var_zapsmall` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:615-737`) on `A2D(0)`.
   * `si3_o3_step` (`packages/ice/legoesm/ice/si3_orca2_state.py:3337-3463`) and `_si3_orca2_o3_step_with_trace` (`packages/ice/legoesm/ice/sea_ice.py:3563-3600`, dispatched from `step_sea_ice` at `packages/ice/legoesm/ice/sea_ice.py:366-367` when `config.scheme == "si3_orca2_o3"`, validated by `_validate_si3_orca2_o3_config` and constructed by `build_si3_orca2_o3_config` in `packages/ice/legoesm/ice/config.py:578-626`, exported in `packages/ice/legoesm/ice/__init__.py:226-262`) execute the full `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:181-257` dynamics-step sequence: `si3_dyn_orca2` (`ice_dyn` + `diag_trends(1)`), `si3_var_glo2eqv`, `si3_var_agg`, `si3_var_glo2eqv`, `si3_var_agg`, and `si3_update_snwice_mass`.

---

## 3. Precise Rung O3 Oracle Record Specification for the Acquisition Round

The subsequent Codex acquisition round (`si3_r9` / Rung O3 acquisition) must build an `ORCA2_SAS_ICE` observer (`ln_icedyn = .TRUE.`, `ln_dynADV2D = .TRUE.`, `ln_adv_Pra = .TRUE.`, `ln_icethd = .FALSE.`, `jpl = 5`, `nlay_i = 10`, `nlay_s = 5`, `nn_icesal = 4`, `nn_hls = 2`, domain `184 x 152`) that writes the following binary64 and NetCDF artifacts for two consecutive sea-ice steps `kt = 1` (`MOD((kt-1)/nn_fsbc, 2) == 0`, odd sweep `adv_x -> adv_y` on `jt = 1`) and `kt = 2` (`MOD((kt-1)/nn_fsbc, 2) == 1`, even sweep `adv_y -> adv_x` on `jt = 1`):

1. **Grid & Horizontal Metric Frame (`o3_orca2_grid_metrics.bin` / JSON manifest, Fortran-order IEEE-754 `float64` on `(184, 152)`):**
   * Horizontal scale factors and areas required by `SI3ORCA2AdvectionGrid` (`packages/ice/legoesm/ice/si3_orca2_state.py:2226-2303`):
     `e1u(184,152)`, `e2v(184,152)`, `r1_e1u(184,152)`, `r1_e2v(184,152)`, `e2u(184,152)`, `e1v(184,152)`, `e1e2t(184,152)`, `r1_e1e2t(184,152)`, and surface masks `tmask(184,152,1)`, `umask(184,152,1)`, `vmask(184,152,1)`.
2. **Step Forcing & CFL Frame at `kt = 1` and `kt = 2` (`o3_step_forcing_kt0001.bin`, `o3_step_forcing_kt0002.bin`):**
   * `u_ice(184,152)`, `v_ice(184,152)` (at `ice_dyn_adv_pra` entry, after `u_ice = rn_uice * umask(:,:,1)` and `v_ice = rn_vice * vmask(:,:,1)` at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:220-221`), `sss_m(184,152)`, `sst_m(184,152)`, `frq_m(184,152)`, scalar `zcfl` (`float64`), and integer `icycle` (`int32`, after `mpp_max` at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:136-141`).
   * Note: to exercise both single-subcycle (`icycle = 1`) and multi-subcycle (`icycle >= 2`, where `ihls = MAX(0, nn_hls - jt) = 1` on `jt = 1` and `0` on `jt = 2`) paths if desired, or the standard `rn_uice, rn_vice` namelist values, the recorded `icycle` and `zcfl` at each `kt` must be stamped in the step manifest.
3. **Full `(184, 152)` Stage Snapshots at `kt = 1` and `kt = 2`:**
   * **`stage = pre_dyn` (immediately after thickness recovery `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:156-170`, before `ice_dyn_adv`):**
     All 107 Rung O1 state/moment/equivalent/aggregated arrays (`v_i, v_s, sv_i, a_i, oa_i, a_ip, v_ip, v_il, t_su, e_s, e_i, szv_i, h_i, h_s, h_ip, h_il, s_i, t_i, t_s, sz_i, u_ice, v_ice, ato_i, at_i, vt_i, vt_s, st_i, et_s, et_i, tm_i, tm_s, tm_su, sm_i, tm_si, hm_i, hm_s, at_ip, vt_ip, vt_il, hm_ip, hm_il, at_ip_eff, snwice_mass, snwice_mass_b, snwice_fmass, hi_max, hi_mean` + all 55 Prather moment arrays `sxice..sxyvl`) plus the `store_fields` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:429-467`) before-step copies (`a_i_b, v_i_b, v_s_b, v_ip_b, v_il_b, sv_i_b, e_s_b, e_i_b, szv_i_b`).
   * **`stage = post_adv_pra` (immediately upon exit from `ice_dyn_adv_pra` at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:494` and `diag_trp_*` at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv.f90:156-164`, before `ice_var_zapsmall`):**
     All 11 extensive families (`v_i, v_s, sv_i, a_i, oa_i, a_ip, v_ip, v_il, e_s, e_i, szv_i`), `ato_i`, all 55 Prather moment arrays (`sxice..sxyvl`), the 6 `ice_dyn_adv_pra` diagnostics (`diag_adv_mass, diag_adv_salt, diag_adv_heat, wfx_res, hfx_res, sfx_res`), and the 5 `ice_dyn_adv` transport diagnostics (`diag_trp_vi, diag_trp_vs, diag_trp_ei, diag_trp_es, diag_trp_sv`).
   * **`stage = post_dyn` (after `ice_var_zapsmall` at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:226`, end-of-`ice_dyn` `lbc_lnk` at `:253-260`, and `diag_trends(1)` at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:215,545-599`):**
     All 107 Rung O1 state/moment/equivalent/aggregated arrays plus all 17 `SI3AdvectionDiagnostics` fields (`wfx_res, hfx_res, sfx_res, diag_adv_mass, diag_adv_salt, diag_adv_heat, diag_trp_vi, diag_trp_vs, diag_trp_ei, diag_trp_es, diag_trp_sv, diag_heat, diag_sice, diag_vice, diag_vsnw, diag_vpnd, diag_aice`, `packages/ice/legoesm/ice/si3_orca2_state.py:2306-2329`).
   * **`stage = post_step` (at the end of `ice_stp` at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:257`, after `ice_var_glo2eqv(2)`, `ice_var_agg(2)`, and `ice_update_flx`):**
     All 107 Rung O1 state/moment/equivalent/aggregated arrays + all 17 `SI3AdvectionDiagnostics` fields.
4. **NetCDF Ice Restart Files at `kt = 1` and `kt = 2`:**
   * Written by `ice_rst_write` (`src/ICE/icerst.F90`) and `adv_pra_rst('WRITE', kt)` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:1398-1515`), containing the 40 base SI3 restart variables plus all 50 active Prather moment restart variables for `nn_icesal == 4` and `ln_pnd_LEV = .TRUE.`:
     * Mandatory 3-D `(jpi, jpj, jpl)` moments (20 variables): `sxice, syice, sxxice, syyice, sxyice`, `sxsn, sysn, sxxsn, syysn, sxysn`, `sxa, sya, sxxa, syya, sxya`, `sxage, syage, sxxage, syyage, sxyage` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:1411-1433`).
     * Mandatory snow-layer 3-D `(jpi, jpj, jpl)` moments for `jk = 01..05` (25 variables): `sxc0_l01..05, syc0_l01..05, sxxc0_l01..05, syyc0_l01..05, sxyc0_l01..05` (`:1435-1447`).
     * Mandatory ice-layer 3-D `(jpi, jpj, jpl)` moments for `jk = 01..10` (50 variables): `sxe_l01..10, sye_l01..10, sxxe_l01..10, syye_l01..10, sxye_l01..10` (`:1449-1461`).
     * Optional ice-salt-layer 3-D `(jpi, jpj, jpl)` moments for `nn_icesal == 4`, `jk = 01..10` (50 variables): `sxsi_l01..10, sysi_l01..10, sxxsi_l01..10, syysi_l01..10, sxysi_l01..10` (`:1467-1481`), or bulk `sxsal, sysal, sxxsal, syysal, sxysal` when `nn_icesal /= 4` (`:1483-1491`).
     * Optional melt-pond 3-D `(jpi, jpj, jpl)` moments for `ln_pnd_LEV .OR. ln_pnd_TOPO` (15 variables): `sxap, syap, sxxap, syyap, sxyap`, `sxvp, syvp, sxxvp, syyvp, sxyvp`, `sxvl, syvl, sxxvl, syyvl, sxyvl` (`:1493-1513`).
     * Total Prather moment NetCDF restart variables per step: `20 + 25 + 50 + 50 + 15 = 160` 3-D `(184, 152, 5)` variables (mapping 1-to-1 to the 50 active 3-D/4-D arrays in `SI3AllPratherMoments`).
   * Note on `iom_get` sign convention at restart read (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:1260-1373`): when reading a restart in `adv_pra_rst('READ')`, NEMO passes `psgn = -1._wp` on all `sx*` and `sy*` variables and default `+1._wp` on `sxx*`, `syy*`, `sxy*`, matching `si3_apply_state_halos_orca2` (`packages/ice/legoesm/ice/si3_orca2_state.py:1412-1470`).

---

## 4. Verification Suite Results and Provenance

### 4.1 Test Suite Summary (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:1-2009`)

All 7 test functions in `tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py` pass on CPU (`CUDA_VISIBLE_DEVICES=""`, `JAX_ENABLE_X64=1`, `PrecisionPolicy.fp64()`):

1. `test_o3_a1_adv_x_and_adv_y_statement_pins_and_plants` (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:583-743`):
   `0` ULP across `ps0, psm, psx, psxx, psy, psyy, psxy` for both `si3_adv_x` and `si3_adv_y` across `ihls in {0, 1}` and `pcrh in {1.0, 0.0}` against independent scalar reference loops (`_ref_adv_x_scalar_loop`, `_ref_adv_y_scalar_loop`); `Plant O3-A1` (`_plant_swap_1malf_order`, `_plant_swap_alf_1malf_sxx`, `_plant_neg_branch_zfx` on `si3_adv_x` and `si3_adv_y`) fires and fails `_assert_zero_ulp`.
2. `test_o3_a2_odd_even_sweep_order_and_cfl_subcycling_pins_and_plant` (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:746-911`):
   `0` ULP for CFL `zcfl`, `icycle in {1, 2, 3}`, `ihls` schedule (`ihls = 0` when `icycle == 1`; `ihls = max(0, nn_hls - jt)` when `icycle > 1`), and odd/even directional split (`MOD((kt-1)//nn_fsbc, 2) == MOD(jt-1, 2)`); `Plant O3-A2` (`_plant_flip_sweep_order=True`) fires and fails `_assert_zero_ulp`.
3. `test_o3_a3_icemax_hbig_hsnow_zapneg_zapsmall_pato_diag_pins_and_plants` (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:914-1197`):
   `0` ULP across `si3_icemax2d_pra`, `si3_icemax3d_pra`, `si3_hbig_pra` (`nn_icesal == 4` and `nn_icesal == 1`), `si3_hsnow_pra`, `si3_var_zapneg` (`nn_icesal == 4` and `nn_icesal == 1`), `si3_var_zapsmall` (`nn_icesal == 4` and `nn_icesal == 1`), and `ato_i` / `diag_adv_*` / `diag_trp_*` / `diag_trends`; all 7 sub-plants of `Plant O3-A3` fire and fail `_assert_zero_ulp`.
4. `test_o3_a4_orca2_seam_and_north_fold_subcycle_lbc_lnk_pins_and_plants` (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:1200-1279`):
   `0` ULP across the ORCA2 cyclic east-west seam (`ji = 0..1` and `182..183`) and T-pivot north fold (`jj = 149..151`) for both `extensives` (`psgn = +1.0`), `sx, sy` (`psgn = -1.0`), and `sxx, syy, sxy` (`psgn = +1.0`) across `icycle = 2`; `Plant O3-A4` (`_plant_wrong_moment_fold_sign=True` and `_plant_skip_subcycle_lbc=True`) fires and fails `_assert_zero_ulp`.
5. `test_o3_b_parent_vs_current_lane3_and_production_dispatcher_zero_ulp` (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:1329-1572`):
   `0` ULP between parent commit `042c7bd74c2944dab8c39318687ab0cef54f9d2b` and the current workspace across `advect_si3_prather_1d`, `advect_si3_prather_2d`, `step_ice_adv1d_card`, `step_ice_adv2d_card`, `step_sea_ice` (`slab`, `dynamic-v2`, `si3_bl99`, `si3_orca2_o1`), and `si3_o1_step`; `Plant O3-B1` and `Plant O3-B2` fire and fail `_assert_zero_ulp`.
6. `test_o3_c_orca2_extensive_conservation_and_residual_budget_closure` (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:1707-1818`):
   All 11 extensive families conserved on `(184, 152)` ORCA2 across `kt in {1, 2}` and `icycle in {1, 2, 3}` (`max_rel_err = 1.583392961495183e-16` at `kt = 1, icycle = 1`; `max_rel_err = 1.7546464555576054e-16` at `kt = 2, icycle = 2`), residual mass and heat budgets close to `<= 1.0e-13`, and `Plant O3-C1` (`_plant_corrupt_receiver_flux=True`) fires.
7. `test_o3_d_jit_primal_zero_ulp_and_forward_equals_reverse_ad` (`tests/ocean/fidelity/test_nemo_si3_o3_orca2_prather.py:1827-2009`):
   `0` ULP between `jax.jit(si3_o3_step)` and eager `si3_o3_step`; finite non-zero gradients with respect to all 10 active extensive fields, `u_ice, v_ice`, and all 5 Prather moments; `|jvp - vjp| / max(|jvp|, 1e-30) = 1.3341694060629237e-15 <= 1.0e-11`; `Plant O3-D1` (`_plant_stop_gradient_vi=True`) fires.

### 4.2 Evidence Artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| Rung O3 model-prep verification report | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8/o3_model_prep_verification.json` | `0999d279ea770bbe4525ebd96206b05af444367a61f82aa58f55a5f6a2778555` |
| Evidence directory digest manifest | `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8/SHA256SUMS` | `66a65604d7c12d68d9d71988565df68fd0a3918cf044314fa321c56d983eb950` |

---

## 5. Choices and Review

* **CHOICES:**
  * `ASKED` = transcribe NEMO 5.0.2's Prather advection (`icedyn_adv_pra.F90` / `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90`, `icedyn.f90`, `icedyn_adv.f90`, `icevar.f90`, `icestp.f90`) in the ONE shared ice package as a selectable SI3 path (`scheme="si3_orca2_o3"` / `build_si3_orca2_o3_config`) applied to the full Rung O1 `SI3State` registry and all 55 Prather moment arrays across the ORCA2 cyclic seam and north fold reusing O1's `si3_lbc_lnk_orca2` / `si3_apply_state_halos_orca2`; prove Lane-3 and production dispatcher `0`-ULP invariance against parent `042c7bd74c2944dab8c39318687ab0cef54f9d2b` with code-path plants; verify Tests (a)–(d) with negative controls; specify the Rung O3 oracle record contract for the subsequent acquisition round without running NEMO.
  * `UNASKED` = none. No existing configuration default, selector, bar, mask, or Lane-3 / O1 / D5 behavior was modified.
* **Review:** `independent review: operator`
