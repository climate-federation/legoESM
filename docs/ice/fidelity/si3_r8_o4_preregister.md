# SI3 Ladder Rung O4 (`si3_r8`) Preregistration — ORCA2 Prather Advection Across Cyclic Seam and North Fold (Model Preparation, No Acquisition)

- **Campaign / Lane**: NEMO 5.0.2 SI3 Sea-Ice Ladder (`docs/ice/fidelity/si3_ladder_design.md:224`)
- **Round**: `si3_r8` — Model Preparation for Rung O4 (`icedyn_adv_pra.F90` on ORCA2 `(184, 152)` across cyclic east-west seam and T-pivot north fold, without acquisition and without running NEMO)
- **Parent Commit**: `042c7bd74c2944dab8c39318687ab0cef54f9d2b` (`fidelity/nemo-testcases-si3-ladder`, Rung O1 + Rung D5 landed)
- **Working Clone**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/si3_r8/repo`
- **Evidence Root**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8`

---

## 1. Scope and Hard Boundaries

1. **Single-rung model preparation (`DECISION 159`, Rung O4, `docs/ice/fidelity/si3_ladder_design.md:224`)**:
   Transcribe NEMO 5.0.2's multidimensional second-order moment-conserving Prather (1986) sea-ice advection scheme (`src/ICE/icedyn_adv_pra.F90` / `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90`) and its `icedyn.F90` / `icedyn_adv.F90` / `icevar.F90` / `icestp.F90` wrapper sequence (`si3_dyn_adv_pra_orca2`, `si3_dyn_orca2`, `si3_o4_step`, `build_si3_orca2_o4_config`) inside the ONE shared ice package (`packages/ice/legoesm/ice/`), operating on the full Rung O1 `SI3State` registry (`jpl = 5` categories, `nlay_i = 10` ice thermal/salinity layers, `nlay_s = 5` snow thermal layers, all extensive fields and their `11 x 5 = 55` Prather moment arrays `SI3AllPratherMoments`, plus carried `cflice_delayed`) on the full ORCA2 `(184, 152)` tripolar grid (`Ni0glo = 180, Nj0glo = 148, nn_hls = 2`).
2. **Reuse O1's halo and north-fold implementation**:
   All lateral boundary conditions (`lbc_lnk` between/after subcycles and at the end of `ice_dyn`) MUST call O1's existing `si3_lbc_lnk_orca2` and `si3_apply_state_halos_orca2` in `packages/ice/legoesm/ice/si3_orca2_state.py` (`src/OCE/LBC/lbclnk.F90:31-425`, `src/OCE/LBC/lbc_nfd_generic.h90:57-350`), never re-implementing cyclic east-west halos or T-pivot north-fold reflections.
3. **Generalize and reuse Lane-3 Prather kernels without duplication**:
   Generalize `_si3_prather_x_substep` and `_si3_prather_y_substep` in `packages/ice/legoesm/ice/transport.py` (`si3_adv_x`, `si3_adv_y`) and share `si3_icemax2d_pra`, `si3_icemax3d_pra`, `si3_hbig_pra`, `si3_hsnow_pra`, `si3_var_zapneg`, `si3_zapsmall_mask`, and `si3_var_zapsmall` across ORCA2 and Lane 3 (`packages/ice/legoesm/ice/fidelity/nemo_adv2d_testcase_recipe.py`) while keeping `advect_si3_prather_1d`, `advect_si3_prather_2d`, `step_ice_adv1d_card`, `step_ice_adv2d_card`, `step_sea_ice`, and `si3_o1_step` 0-ULP bit-identical to parent `042c7bd74c2944dab8c39318687ab0cef54f9d2b`.
4. **Parallel round boundary (`si3_r7`)**:
   Do NOT touch O1/O2 acquisition tools (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/`), `iceistate` code, or the initial-state path.
5. **No NEMO execution and no acquisition**:
   Do NOT run NEMO or build an acquisition runner in `si3_r8`. Specify in the receipt the exact binary/NetCDF record contract that the subsequent O4 acquisition round must produce.

---

## 2. Exact NEMO 5.0.2 `ORCA2_SAS_ICE` Preprocessed Source Target

Every statement transcribed in `si3_r8` is pinned to the clean uninstrumented `ORCA2_SAS_ICE` build at `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/` and raw `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/ICE/`:

1. **Pre-advection intensive thickness recovery (`src/ICE/icedyn.F90:155-171`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:155-170`)**:
   - `WHERE( a_i(:,:,:) >= epsi20 )` / `h_i(:,:,:) = v_i(:,:,:) / a_i(:,:,:)` / `h_s(:,:,:) = v_s(:,:,:) / a_i(:,:,:)` / `ELSEWHERE` / `h_i(:,:,:) = 0._wp` / `h_s(:,:,:) = 0._wp` / `END WHERE` (`icedyn.f90:156-162`)
   - `WHERE( a_ip(:,:,:) >= epsi20 )` / `h_ip(:,:,:) = v_ip(:,:,:) / a_ip(:,:,:)` / `h_il(:,:,:) = v_il(:,:,:) / a_ip(:,:,:)` / `ELSEWHERE` / `h_ip(:,:,:) = 0._wp` / `h_il(:,:,:) = 0._wp` / `END WHERE` (`icedyn.f90:164-170`)
   - Note: `ph_i, ph_s, ph_ip` are `INTENT(in   )` to `ice_dyn_adv_pra` (`icedyn_adv_pra.f90:86-88`) and remain fixed at their pre-advection values across all subcycles `jt = 1..icycle`.
2. **Advection wrapper and transport diagnostics (`src/ICE/icedyn_adv.F90:86-166`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv.f90:112-178`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:429-467`)**:
   - Pre-step `store_fields` copies `a_i_b, v_i_b, v_s_b, v_ip_b, v_il_b, sv_i_b, e_s_b, e_i_b, szv_i_b` (`icestp.f90:438-457`)
   - Dispatch `CASE( np_advPRA )` calling `CALL ice_dyn_adv_pra(         kt, u_ice, v_ice, h_i, h_s, h_ip, &` / `&                          ato_i, v_i, v_s, sv_i, oa_i, a_i, a_ip, v_ip, v_il, e_s, e_i, szv_i )` (`icedyn_adv.f90:147-150`)
   - Post-advection `diag_trp_ei, diag_trp_es, diag_trp_sv, diag_trp_vi, diag_trp_vs` difference multiplied by `r1_Dt_ice` (`icedyn_adv.f90:156-164`).
3. **CFL evaluation, delayed `mpp_max`, subcycle count `icycle`, and face transports (`src/ICE/icedyn_adv_pra.F90:137-159`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:130-150`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:789-846`)**:
   - `zcfl = MAXVAL( ABS( pu_ice(:,:) ) * rDt_ice * r1_e1u(:,:) )` (`icedyn_adv_pra.f90:133`)
   - `zcfl = MAX( zcfl, MAXVAL( ABS( pv_ice(:,:) ) * rDt_ice * r1_e2v(:,:) ) )` (`icedyn_adv_pra.f90:134`)
   - `CALL mpp_max( 'icedyn_adv_pra', zcfl, cdelay = 'cflice' )` (`icedyn_adv_pra.f90:136`). The admitted O1 `ORCA2_SAS_ICE` record resolves `use delayed global communications?             ln_mppdelay =  T` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round4/refused/oracle_o1_orca2_sas_ice_admission_refused_20261024T161410Z_1010784/ref_2step/ocean.output`, line 43; the round-7 O2 record prints the same line). In `mppmax1d_real_dp` (`lib_mpp.f90:789-846`), when `llisdelay = ln_mppdelay .AND. PRESENT(cdelay)` is true: on the first call without restart (`ndelayid(idvar) == ndlrstoff`, `lib_mpp.f90:823-828`), `mpi_allreduce` initializes `todelay(idvar)%zdpbuffout` from the current call's `ptab(:)` so call 1 uses call 1's `zcfl` and saves it; on every subsequent call (or first call from a restart where `DELAY__r8_cflice` was read by `iom_delay_rst` at `iom.f90:1396-1401,1424` and `icerst.f90:138,353`), `todelay(idvar)%zdpbuffin(:) = ptab(:)` and `ptab(:) = todelay(idvar)%zdpbuffout(:)` (`lib_mpp.f90:834-840`) return the PREVIOUS call's `zcfl` and store the current call's `zcfl` into carried state (`cflice_delayed`).
   - `IF ( zcfl > 1.5 ) THEN ; icycle = 3` / `ELSEIF( zcfl > .5 ) THEN ; icycle = 2` / `ELSE ; icycle = 1` / `ENDIF` (`icedyn_adv_pra.f90:138-141`), bounded in `{1, 2, 3}`; all three cycle counts are traced and the one selected by the traced `icycle` is kept, with no host-side conversion of `zcfl`.
   - `zdt = rDt_ice / REAL(icycle)` and `z1_dt = 1._wp / zdt` (`icedyn_adv_pra.f90:145-146`)
   - `zudy(:,:) = pu_ice(:,:) * e2u(:,:)` and `zvdx(:,:) = pv_ice(:,:) * e1v(:,:)` (`icedyn_adv_pra.f90:149-150`).
4. **Subcycle loop `DO jt = 1, icycle` (`src/ICE/icedyn_adv_pra.F90:164-530`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:156-498`)**:
   - `zati1(:,:) = SUM( pa_i(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:), dim=3 )` (`icedyn_adv_pra.f90:159`, sequential category sum `jl = 1..jpl`)
   - `IF( icycle == 1 ) THEN ; ihls = 0` / `ELSE ; ihls = MAX( 0, nn_hls - jt )` / `ENDIF` (`icedyn_adv_pra.f90:161-163`)
   - **Category loop `DO jl = 1, jpl` (`icedyn_adv_pra.f90:168-427`)**:
     - `icemax2D_pra` (`icedyn_adv_pra.f90:1519-1550`) and `icemax3D_pra` (`icedyn_adv_pra.f90:1552-1588`) over `ji = Nis0-ihls : Nie0+ihls, jj = Njs0-ihls : Nje0+ihls` for `zhi_max, zhs_max, zhip_max, zei_max, zes_max` and `zszi_max` (`nn_icesal == 4`) or `zsi_max` (`nn_icesal /= 4`) (`icedyn_adv_pra.f90:172-213`).
     - Pre-advection `zdiag_adv_mass, zdiag_adv_heat, zdiag_adv_salt` on `A2D(0)` (`icedyn_adv_pra.f90:216-230`).
     - Extensive field initialization `zarea(ji,jj) = e1e2t(ji,jj)` and, per family, e.g. `z0ice(ji,jj) = pv_i (ji,jj,jl) * e1e2t(ji,jj)` on `A2D(ihls+1)` (`icedyn_adv_pra.f90:233-261`).
     - Alternating directional split (`icedyn_adv_pra.f90:267-367`):
       - When `IF( MOD( (kt - 1) / nn_fsbc , 2 ) ==  MOD( (jt - 1) , 2 ) ) THEN`: for each family, e.g. `CALL adv_x( ihls, jl, zdt , zudy , 1._wp , zarea , z0ice , sxice , sxxice , syice , syyice , sxyice )` followed by `CALL adv_y( ihls, jl, zdt , zvdx , 0._wp , zarea , z0ice , sxice , sxxice , syice , syyice , sxyice )` (`icedyn_adv_pra.f90:267-316`).
       - Otherwise: for each family, e.g. `CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0ice , sxice , sxxice , syice , syyice , sxyice )` followed by the matching `adv_x` call with `0._wp` (`icedyn_adv_pra.f90:318-367`).
     - `adv_x` (`icedyn_adv_pra.f90:513-733`, `ji0 = 1 + ihls`, `jj0 = NINT(pcrh) + ihls`) and `adv_y` (`icedyn_adv_pra.f90:736-957`, `ji0 = NINT(pcrh) + ihls`, `jj0 = 1 + ihls`), preserving the exact negative-branch omission in `adv_x`, where `zfx (ji,jj) = 0._wp` is set and no `zfx` accumulation follows, (`icedyn_adv_pra.f90:615, 642-649`) vs `zfy` negative-branch accumulation in `adv_y` (`icedyn_adv_pra.f90:840, 865-872`).
     - Intensive recovery on `A2D(ihls)` (`icedyn_adv_pra.f90:370-397`): `pv_i (ji,jj,jl) = z0ice(ji,jj) * r1_e1e2t(ji,jj) * tmask(ji,jj,1)` (left-to-right product `(z0* * r1_e1e2t) * tmask`).
     - Post-advection `diag_adv_mass, diag_adv_heat, diag_adv_salt` accumulation on `A2D(0)` (`icedyn_adv_pra.f90:400-417`).
     - `Hbig_pra` (`icedyn_adv_pra.f90:421-422, 960-1093`) on `A2D(ihls)` with `pdt = rDt_ice`, using `hi_max(jpl)` (`state.bounds.hi_max[-1]`, `iceitd.f90:770`) and exact `zfra = cap / zsi ; field = field * zfra` association.
     - `Hsnow_pra` (`icedyn_adv_pra.f90:425, 1096-1156`) on `A2D(ihls)` with `pdt = rDt_ice`, using `zvs_excess = MAX( 0._wp, pv_s(ji,jj,jcat) - pv_i(ji,jj,jcat) * (rho0-rhoi) * r1_rhos )`, `wfx_res`, `hfx_res`, and `pa_ip(ji,jj,jcat) = MIN( pa_ip(ji,jj,jcat), pa_i(ji,jj,jcat) )`.
   - `ice_var_zapneg` (`icedyn_adv_pra.f90:435`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:740-853`) on `A2D(ihls)` with whole-array `WHERE( pv_i(:,:,:) <= 0._wp )   pa_i(:,:,:) = 0._wp` (`icevar.f90:774`) and `WHERE( poa_i (:,:,:) < 0._wp )   poa_i (:,:,:) = 0._wp` / `WHERE( pa_i  (:,:,:) < 0._wp )   pa_i  (:,:,:) = 0._wp` / `WHERE( pa_ip (:,:,:) < 0._wp )   pa_ip (:,:,:) = 0._wp` (`icevar.f90:849-851`).
   - Open-water fraction update `pato_i` on `A2D(0)` (`icedyn_adv_pra.f90:438-443`):
     `pato_i(ji,jj) = MAX( 0._wp, pato_i(ji,jj) - ( zati2 - zati1(ji,jj) )            &` / `&                                      - (   ( zudy(ji,jj) - zudy(ji-1,jj) ) &   ! ad () for NP repro` / `&                                          + ( zvdx(ji,jj) - zvdx(ji,jj-1) ) ) * r1_e1e2t(ji,jj) * zdt )`.
   - End-of-subcycle `lbc_lnk` with `ldfull = .TRUE.` (`icedyn_adv_pra.f90:450-494`) via `si3_lbc_lnk_orca2` across the ORCA2 cyclic seam and north fold (`'T', 1._wp` for extensives, `pato_i`, and `sxx, syy, sxy, wfx_res, hfx_res, sfx_res`; `'T', -1._wp` for `sx, sy`).
5. **Post-advection `ice_dyn` and `ice_stp` completion (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:218-261`, `icestp.f90:181-257, 526-540`, `ice.f90:465`)**:
   - Under `CASE ( np_dynADV2D )` (namelist `ln_dynADV2D` true): `CALL ice_var_zapsmall` (`icedyn.f90:226`, `icevar.f90:615-737`).
   - End-of-`ice_dyn` `lbc_lnk` on `a_i, v_i, v_s, sv_i, oa_i, t_su, a_ip, v_ip, v_il, e_i, e_s, szv_i` (`icedyn.f90:253-260`).
   - In `ice_stp` (`icestp.f90:181-257, 526-540`): `diag_set0` sets `t_si(ji,jj,jl) = rt0` on `A2D(0)` (`Nis0:Nie0, Njs0:Nje0`, since `t_si` is allocated on `A2D(0)` with no halo cells at `ice.f90:465` and zero-padded outside `A2D(0)` in `o1_pad_a2d_3d` at `o1_si3_state_writer.F90:35-40,166`), followed by `ice_dyn`, `diag_trends(1)`, `ice_var_glo2eqv(2)`, `ice_var_agg(1)`, `store_fields`, `ice_sbc_flx` (rung O7), `diag_trends(2)`, `ice_var_glo2eqv(2)`, `ice_var_agg(2)`, and `ice_update_flx` (`icestp.f90:211-257`; only its snow-ice mass part, `iceupdate.f90:203-208`, belongs to O4).

---

## 3. Preregistered Falsifiable Predictions and Negative-Control Plants

### Test (a): Bit-for-Bit (`0` ULP) Statement Pins with Inexact Operands and Reversal/Reordering Plants
- **Prediction**: Every transcribed statement in `icedyn_adv_pra.f90`, `icedyn.f90`, `icedyn_adv.f90`, `lib_mpp.f90`, `lbcnfd.f90`, and `icevar.f90` matches its independent scalar/reference Fortran evaluation to `0` ULP on inexact `float64` operands.
- **Plants**:
  - `Plant O4-A1` (`adv_x` / `adv_y` moment & flux statement order): Reordering `(1.0 - zalf) * psx` or `zalf * z1malf` or restoring `zfx` in `adv_x`'s negative branch produces `max_ulp >= 1` and fails `_assert_zero_ulp`.
  - `Plant O4-A2` (`odd/even` sweep order & delayed CFL `cflice`): Flipping `MOD((kt - 1) // nn_fsbc, 2) == MOD(jt - 1, 2)` (`x -> y` vs `y -> x`) or ignoring the 1-call `ln_mppdelay` lag (`_plant_ignore_mppdelay=True`) produces `max_ulp > 0` and fails `_assert_zero_ulp`.
  - `Plant O4-A3` (`icemax2D_pra` / `icemax3D_pra`, `Hbig_pra`, `Hsnow_pra`, `ice_var_zapneg`, `pato_i`): Reordering `zfra = cap / zsi ; field = field * zfra` to `(field * cap) / zsi` on an inexact operand, or omitting `(rho0 - rhoi) * r1_rhos`, or testing `pv_il` after zeroing `pv_ip` in `ice_var_zapneg`, or omitting the velocity divergence term in `pato_i`, produces `max_ulp >= 1` and fails `_assert_zero_ulp`.
  - `Plant O4-A4` (`lbc_lnk` between subcycles across seam and north fold against independent `lbcnfd.f90` replica): Flipping the T-fold sign `psgn` on EACH of `sx`, `sy`, `sxx`, `syy`, `sxy` individually or omitting the inter-subcycle `lbc_lnk` when `icycle >= 2` produces `max_ulp > 0` at the north fold/seam and fails `_assert_zero_ulp`.

### Test (b): Lane-3 `ICE_ADV1D` / `ICE_ADV2D` Invariance and Real Parent-vs-Current (`042c7bd74c29`) Control
- **Prediction**:
  1. All existing Lane-3 `ICE_ADV1D` and `ICE_ADV2D` gate tests (`tests/ice/fidelity/test_nemo_si3_phase2_gate.py`, `tests/ice/fidelity/test_nemo_si3_phase2_adv2d_gate.py`, `tests/ice/fidelity/test_nemo_si3_phase2_adv2d_replay.py`) and O1/D5 gates (`tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py`, `tests/ocean/fidelity/test_nemo_si3_d5_c1d_pnd_gate.py`) pass unchanged at `0` ULP.
  2. Loading `packages/ice/legoesm/ice/*` from parent commit `042c7bd74c2944dab8c39318687ab0cef54f9d2b` via `git show` and comparing against the current workspace across `advect_si3_prather_1d`, `advect_si3_prather_2d`, `step_ice_adv1d_card`, `step_ice_adv2d_card`, `step_sea_ice` (`slab`, `dynamic-v2`, `si3_bl99`, `si3_orca2_o1`), and `si3_o1_step` yields `0` ULP across every output leaf.
- **Plants**:
  - `Plant O4-B1`: Perturbing `advect_si3_prather_2d` or `step_ice_adv2d_card` by `1` ULP (`jnp.nextafter(..., jnp.inf)`) fails the parent-vs-current control with `max ULP = 1 != 0`.
  - `Plant O4-B2`: Perturbing `step_sea_ice` (`_si3_step_with_trace`) by `1` ULP fails the production dispatcher control with `max ULP = 1 != 0`.

### Test (c): Conservation of Every Extensive to Round-Off on a Closed ORCA2 Test Field Including Ice at and Next to the North Fold
- **Prediction**: On the full `(184, 152)` ORCA2 grid with active cyclic east-west seam transport and ice present all the way to and at the T-pivot north fold (`jj = 148, 149`), under a north-fold-antisymmetric (`F`, `psgn = -1`) streamfunction whose face transports `zudy, zvdx` satisfy exact T-fold symmetry (`zudy` `U`-fold `-1`, `zvdx` `V`-fold `-1`) so the unique cell-weighted global integral (`w = 1` on `jj = 2..148`, `w = 1` on `ji = 2..90, jj = 149`, `w = 0.5` on T-pivot `ji in {2, 92}, jj = 149`) has zero net boundary flux, the domain-integrated total content of every extensive family (`v_i`, `v_s`, `a_i`, `oa_i`, `e_s`, `e_i`, `szv_i`, `sv_i`, `a_ip`, `v_ip`, `v_il`) is conserved to `float64` round-off (`relative_error <= 1e-14`) across both `x -> y` and `y -> x` sweeps and `icycle in {1, 2, 3}`, and when capping/zapping is active the residual diagnostics (`wfx_res`, `hfx_res`, `sfx_res`) close the mass/heat/salt budget (`diag_adv_mass`, `diag_adv_heat`, `diag_adv_salt`).
- **Plant**:
  - `Plant O4-C1`: Dropping the donor flux subtraction `- zf0(ji, jj)` in a single cell breaks extensive conservation and fails the conservation assertion.

### Test (d): End-to-End Differentiability (`jax.jit`, `jax.value_and_grad`, `jax.jvp`, and Central Finite Differences) Without Passing `subcycles`
- **Prediction**: Through one full ORCA2 Prather advection step (`si3_dyn_adv_pra_orca2` / `si3_o4_step`) without passing `subcycles`, `jax.jit` primal output is `0` ULP identical to eager execution, reverse-mode gradients (`jax.value_and_grad`) with respect to all extensive fields (`v_i, v_s, a_i, oa_i, e_s, e_i, szv_i, a_ip, v_ip, v_il`), ice velocities (`u_ice, v_ice`), and Prather moments (`sx, sy, sxx, syy, sxy`) are finite and non-zero, directional derivatives from `jax.value_and_grad` match `jax.jvp` to `|jvp - vjp| / max(|jvp|, |vjp|, 1e-30) <= 1e-11`, and central finite differences `(L(x + eps*d) - L(x - eps*d)) / (2*eps)` match `jax.jvp` / `jax.value_and_grad` with ratio `~ 1.0` (`|ratio - 1.0| <= 1e-6` at `float64`).
- **Plant**:
  - `Plant O4-D1`: Inserting `jax.lax.stop_gradient` on `v_i` or `prather.ice.sx` zeros the gradient and fails the AD gate with `STATUS PLANT-FIRED`.

---

## 4. Amendment (resume 4, 2026-10-24) — changes made after review, stated so they are not read as preregistered

1. The CFL count is NEMO's delayed one: the step carries `cflice_delayed` and there is no subcycle override argument; the cycle count comes only from that carried value (or, on the first call from rest, the current one).
2. Test (b) no longer loads parent modules in-process. A clean `git archive` of `042c7bd74c2944dab8c39318687ab0cef54f9d2b` runs the committed dump `scripts/validate/ocean_fidelity/testcases/nemo_si3_lane3_prather_dump.py` in a fresh interpreter, and every array (ICE_ADV1D/ICE_ADV2D cards, both correction orders, the shared correction ledger with every clip active, both sweep kernels, `step_sea_ice`) is compared at 0 ULP with the current tree. Plant O4-B1 is now a real arithmetic-order change inside the lane-3 ledger, not a 1-ULP nudge of an output. Plant O4-B2 is dropped: the production dispatcher is inside the dumped set.
3. Test (d) also runs one compiled step with a traced `cflice_delayed` selecting each of `icycle = 1, 2, 3`.
4. Test (c) selects `icycle` through `cflice_delayed`, not an override.
5. Plant O4-C1 as built multiplies the summed face flux by `1.000001` (not a dropped donor term); Plant O4-D1 stops the gradient on the whole input state (not on `v_i` or `sx` only). Both still fail their gates.
