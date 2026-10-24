# SI3 Ladder Rung O3 (`si3_r8`) Preregistration — ORCA2 Prather Advection Across Cyclic Seam and North Fold (Model Preparation, No Acquisition)

- **Campaign / Lane**: NEMO 5.0.2 SI3 Sea-Ice Ladder (`docs/ice/fidelity/si3_ladder_design.md`)
- **Round**: `si3_r8` — Model Preparation for Rung O3 (`icedyn_adv_pra.F90` on ORCA2 `(184, 152)` across cyclic east-west seam and T-pivot north fold, without acquisition and without running NEMO)
- **Parent Commit**: `042c7bd74c29713be111033b8685c30d9209f1f9` (`fidelity/nemo-testcases-si3-ladder`, Rung O1 + Rung D5 landed)
- **Working Clone**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/si3_r8/repo`
- **Evidence Root**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8`

---

## 1. Scope and Hard Boundaries

1. **Single-rung model preparation (`DECISION 159`, Rung O3)**:
   Transcribe NEMO 5.0.2's multidimensional second-order moment-conserving Prather (1986) sea-ice advection scheme (`src/ICE/icedyn_adv_pra.F90` / `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90`) and its `icedyn.F90` / `icedyn_adv.F90` / `icevar.F90` / `icestp.F90` wrapper sequence as a selectable SI3 path inside the ONE shared ice package (`packages/ice/legoesm/ice/`), operating on the full Rung O1 `SI3State` registry (`jpl = 5` categories, `nlay_i = 10` ice thermal/salinity layers, `nlay_s = 5` snow thermal layers, all extensive fields and their `11 x 5 = 55` Prather moment arrays `SI3AllPratherMoments`) on the full ORCA2 `(184, 152)` tripolar grid (`Ni0glo = 180, Nj0glo = 148, nn_hls = 2`).
2. **Reuse O1's halo and north-fold implementation**:
   All lateral boundary conditions (`lbc_lnk` between/after subcycles and at the end of `ice_dyn`) MUST call O1's existing `si3_lbc_lnk_orca2` and `si3_apply_state_halos_orca2` in `packages/ice/legoesm/ice/si3_orca2_state.py` (`src/OCE/LBC/lbclnk.F90:31-425`, `src/OCE/LBC/lbc_nfd_generic.h90:57-350`), never re-implementing cyclic east-west halos or T-pivot north-fold reflections.
3. **Reuse and preserve Lane-3 Prather kernels**:
   Reuse the elemental Prather slope limiters, donor flux evaluations, and receiver cell moment merges in `packages/ice/legoesm/ice/transport.py` (`_si3_prather_limit_x`, `_si3_prather_limit_y`, `_si3_prather_merge_from_left`, `_si3_prather_merge_from_right`, `_si3_prather_merge_from_below`, `_si3_prather_merge_from_above`, `_si3_prather_x_substep`, `_si3_prather_y_substep`) while keeping `advect_si3_prather_1d`, `advect_si3_prather_2d`, `step_ice_adv1d_card`, `step_ice_adv2d_card`, `step_sea_ice`, and `si3_o1_step` 0-ULP bit-identical to parent `042c7bd74c29713be111033b8685c30d9209f1f9`.
4. **Parallel round boundary (`si3_r7`)**:
   Do NOT touch O1/O2 acquisition tools (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/`), `iceistate` code, or the initial-state path.
5. **No NEMO execution and no acquisition**:
   Do NOT run NEMO or build an acquisition runner in `si3_r8`. Specify in the receipt the exact binary/NetCDF record contract that the subsequent O3 acquisition round must produce.

---

## 2. Exact NEMO 5.0.2 `ORCA2_SAS_ICE` Preprocessed Source Target

Every statement transcribed in `si3_r8` is pinned to the clean uninstrumented `ORCA2_SAS_ICE` build at `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/` and raw `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/ICE/`:

1. **Pre-advection intensive thickness recovery (`src/ICE/icedyn.F90:155-171`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:155-170`)**:
   - `WHERE( a_i(:,:,:) >= epsi20 ) h_i(:,:,:) = v_i(:,:,:) / a_i(:,:,:) ; h_s(:,:,:) = v_s(:,:,:) / a_i(:,:,:) ELSEWHERE 0._wp` (`icedyn.f90:156-162`)
   - `WHERE( a_ip(:,:,:) >= epsi20 ) h_ip(:,:,:) = v_ip(:,:,:) / a_ip(:,:,:) ; h_il(:,:,:) = v_il(:,:,:) / a_ip(:,:,:) ELSEWHERE 0._wp` (`icedyn.f90:164-170`)
   - Note: `ph_i, ph_s, ph_ip` are `INTENT(in)` to `ice_dyn_adv_pra` (`icedyn_adv_pra.f90:86-88`) and remain fixed at their pre-advection values across all subcycles `jt = 1..icycle`.
2. **Advection wrapper and transport diagnostics (`src/ICE/icedyn_adv.F90:86-166`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv.f90:85-153`)**:
   - Pre-advection `diag_trp_vi, diag_trp_vs, diag_trp_ei, diag_trp_es, diag_trp_sv` accumulation (`icedyn_adv.f90:103-112`)
   - Dispatch `CASE( np_advPRA )` calling `ice_dyn_adv_pra` (`icedyn_adv.f90:124-129`)
   - Post-advection `diag_trp_*` difference multiplied by `r1_Dt_ice` (`icedyn_adv.f90:139-148`).
3. **CFL evaluation, subcycle count `icycle`, and face transports (`src/ICE/icedyn_adv_pra.F90:137-159`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:132-150`)**:
   - `zcfl = MAXVAL( ABS( pu_ice(:,:) ) * rDt_ice * r1_e1u(:,:) )` (`icedyn_adv_pra.f90:133`)
   - `zcfl = MAX( zcfl, MAXVAL( ABS( pv_ice(:,:) ) * rDt_ice * r1_e2v(:,:) ) )` (`icedyn_adv_pra.f90:134`)
   - `IF( zcfl > 1.5_wp ) icycle = 3 ELSEIF( zcfl > 0.5_wp ) icycle = 2 ELSE icycle = 1` (`icedyn_adv_pra.f90:138-141`), with an optional explicit `icycle_override` parameter to support delayed MPI `mpp_max(..., cdelay='cflice')` (`icedyn_adv_pra.f90:136`) replay when needed.
   - `zdt = rDt_ice / REAL(icycle)` and `z1_dt = 1._wp / zdt` (`icedyn_adv_pra.f90:145-146`)
   - `zudy(:,:) = pu_ice(:,:) * e2u(:,:)` and `zvdx(:,:) = pv_ice(:,:) * e1v(:,:)` (`icedyn_adv_pra.f90:149-150`).
4. **Subcycle loop `DO jt = 1, icycle` (`src/ICE/icedyn_adv_pra.F90:164-530`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:156-498`)**:
   - `zati1(:,:) = SUM( pa_i(Nis0:Nie0,Njs0:Nje0,:), dim=3 )` (`icedyn_adv_pra.f90:159`, sequential category sum `jl = 1..jpl`)
   - `IF( icycle == 1 ) THEN ihls = 0 ELSE ihls = MAX( 0, nn_hls - jt ) ENDIF` (`icedyn_adv_pra.f90:161-163`)
   - **Category loop `DO jl = 1, jpl` (`icedyn_adv_pra.f90:168-427`)**:
     - `icemax2D_pra` (`icedyn_adv_pra.f90:1519-1550`) and `icemax3D_pra` (`icedyn_adv_pra.f90:1552-1588`) over `ji = Nis0-ihls : Nie0+ihls, jj = Njs0-ihls : Nje0+ihls` for `zhi_max, zhs_max, zhip_max, zei_max, zes_max` and `zszi_max` (`nn_icesal == 4`) or `zsi_max` (`nn_icesal /= 4`) (`icedyn_adv_pra.f90:172-213`).
     - Pre-advection `zdiag_adv_mass, zdiag_adv_heat, zdiag_adv_salt` on `A2D(0)` (`icedyn_adv_pra.f90:216-230`).
     - Extensive field initialization `zarea = e1e2t` and `z0* = p* * e1e2t` on `A2D(ihls+1)` (`icedyn_adv_pra.f90:233-261`).
     - Alternating directional split (`icedyn_adv_pra.f90:267-367`):
       - When `MOD( (kt - 1) / nn_fsbc, 2 ) == MOD( (jt - 1), 2 )`: `adv_x(ihls, zdt, zudy, 1._wp, zarea, z0*, sx*, sxx*, sy*, syy*, sxy*)` followed by `adv_y(ihls, zdt, zvdx, 0._wp, zarea, z0*, sx*, sxx*, sy*, syy*, sxy*)` (`icedyn_adv_pra.f90:271-316`).
       - Otherwise: `adv_y(ihls, zdt, zvdx, 1._wp, zarea, z0*, sx*, sxx*, sy*, syy*, sxy*)` followed by `adv_x(ihls, zdt, zudy, 0._wp, zarea, z0*, sx*, sxx*, sy*, syy*, sxy*)` (`icedyn_adv_pra.f90:321-366`).
     - `adv_x` (`icedyn_adv_pra.f90:513-733`) on `A2D(ihls+1)` and `adv_y` (`icedyn_adv_pra.f90:736-957`) on `A2D(ihls)` (first loop `:782-892`) and `A2D(ihls)` (second loop `:895-955`), preserving the exact `zfx = 0._wp` negative-branch omission in `adv_x` (`icedyn_adv_pra.f90:615, 642-649`) vs `zfy` negative-branch accumulation in `adv_y` (`icedyn_adv_pra.f90:839, 865-871`).
     - Intensive recovery on `A2D(ihls)` (`icedyn_adv_pra.f90:370-397`): `p* = z0* * r1_e1e2t(ji,jj) * tmask(ji,jj,1)` (left-to-right product `(z0* * r1_e1e2t) * tmask`).
     - Post-advection `diag_adv_mass, diag_adv_heat, diag_adv_salt` accumulation on `A2D(0)` (`icedyn_adv_pra.f90:400-417`).
     - `Hbig_pra` (`icedyn_adv_pra.f90:421-422, 960-1093`) on `A2D(ihls)` using `hi_max(jpl)` (`state.bounds.hi_max[-1]`, `iceitd.f90:770`) and exact `zfra = cap / zsi ; field = field * zfra` association.
     - `Hsnow_pra` (`icedyn_adv_pra.f90:425, 1096-1156`) on `A2D(ihls)` using `zvs_excess = MAX( 0._wp, pv_s - pv_i * (rho0-rhoi) * r1_rhos )`, `wfx_snw_sni`, `hfx_res`, `wfx_res`, and `pa_ip = MIN( pa_ip, pa_i )`.
   - `ice_var_zapneg` (`icedyn_adv_pra.f90:435`, `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:740-853`) on `A2D(ihls)` with whole-array `WHERE( pv_i <= 0._wp ) pa_i = 0._wp` (`icevar.f90:774`) and `WHERE( poa_i/pa_i/pa_ip < 0._wp ) = 0._wp` (`icevar.f90:849-851`).
   - Open-water fraction update `pato_i` on `A2D(0)` (`icedyn_adv_pra.f90:438-443`):
     `pato_i(ji,jj) = MAX( 0._wp, pato_i(ji,jj) - ( zati2 - zati1(ji,jj) ) - ( ( zudy(ji,jj) - zudy(ji-1,jj) ) + ( zvdx(ji,jj) - zvdx(ji,jj-1) ) ) * r1_e1e2t(ji,jj) * zdt )`.
   - End-of-subcycle `lbc_lnk(..., ldfull = .TRUE.)` (`icedyn_adv_pra.f90:450-494`) via `si3_lbc_lnk_orca2` across the ORCA2 cyclic seam and north fold (`'T', +1.0` for extensives, `pato_i`, and `sxx, syy, sxy, wfx_res, hfx_res, sfx_res`; `'T', -1.0` for `sx, sy`).
5. **Post-advection `ice_dyn` and `ice_stp` completion (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:215-261`, `icestp.f90:208-256`)**:
   - When `nice_dyn == np_dynADV2D` (`ln_dynADV2D = .TRUE.`): `CALL ice_var_zapsmall` (`icedyn.f90:226`, `icevar.f90:615-737`).
   - End-of-`ice_dyn` `lbc_lnk` on `a_i, v_i, v_s, sv_i, oa_i, h_i, a_ip, v_ip, v_il, t_su, e_i, e_s, szv_i` (`icedyn.f90:253-260`).
   - In `ice_stp` (`icestp.f90:226-256`): `ice_var_glo2eqv(2)`, `ice_var_agg(1)`, `ice_var_glo2eqv(2)`, `ice_var_agg(2)`, and `ice_update_flx` (`snwice_mass, snwice_mass_b, snwice_fmass`).

---

## 3. Preregistered Falsifiable Predictions and Negative-Control Plants

### Test (a): Bit-for-Bit (`0` ULP) Statement Pins with Inexact Operands and Reversal/Reordering Plants
- **Prediction**: Every transcribed statement in `icedyn_adv_pra.f90`, `icedyn.f90`, `icedyn_adv.f90`, and `icevar.f90` matches its independent scalar/reference Fortran evaluation to `0` ULP on inexact `float64` operands.
- **Plants**:
  - `Plant O3-A1` (`adv_x` / `adv_y` moment & flux statement order): Reordering `(1.0 - zalf) * psx` or `zalf * z1malf` or restoring `zfx` in `adv_x`'s negative branch produces `max_ulp >= 1` and fails `_assert_zero_ulp`.
  - `Plant O3-A2` (`odd/even` sweep order): Flipping `MOD((kt - 1) // nn_fsbc, 2) == MOD(jt - 1, 2)` (`x -> y` vs `y -> x`) produces `max_ulp > 0` on non-divergent curvilinear flow and fails `_assert_zero_ulp`.
  - `Plant O3-A3` (`icemax2D_pra` / `icemax3D_pra`, `Hbig_pra`, `Hsnow_pra`, `ice_var_zapneg`, `pato_i`): Reordering `zfra = cap / zsi ; field = field * zfra` to `(field * cap) / zsi` on an inexact operand, or omitting `(rho0 - rhoi) * r1_rhos`, or testing `pv_il` after zeroing `pv_ip` in `ice_var_zapneg`, or omitting the velocity divergence term in `pato_i`, produces `max_ulp >= 1` and fails `_assert_zero_ulp`.
  - `Plant O3-A4` (`lbc_lnk` between subcycles across seam and north fold): Using `+1.0` instead of `-1.0` on `sx, sy` at the T-pivot north fold or omitting the inter-subcycle `lbc_lnk` when `icycle >= 2` produces `max_ulp > 0` and fails `_assert_zero_ulp`.

### Test (b): Lane-3 `ICE_ADV1D` / `ICE_ADV2D` Invariance and Real Parent-vs-Current (`042c7bd74c29`) Control
- **Prediction**:
  1. All existing Lane-3 `ICE_ADV1D` and `ICE_ADV2D` gate tests (`tests/ice/fidelity/test_nemo_si3_phase2_gate.py`, `tests/ice/fidelity/test_nemo_si3_phase2_adv2d_gate.py`, `tests/ice/fidelity/test_nemo_si3_oracle_gate.py`) and O1/D5 gates (`tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py`, `tests/ocean/fidelity/test_nemo_si3_d5_c1d_pnd_gate.py`) pass unchanged at `0` ULP.
  2. Loading `packages/ice/legoesm/ice/*` from parent commit `042c7bd74c29713be111033b8685c30d9209f1f9` via `git show` and comparing against the current workspace across `advect_si3_prather_1d`, `advect_si3_prather_2d`, `step_ice_adv1d_card`, `step_ice_adv2d_card`, `step_sea_ice` (`slab`, `dynamic-v2`, `si3_bl99`), and `si3_o1_step` yields `0` ULP across every output leaf.
- **Plants**:
  - `Plant O3-B1`: Perturbing `advect_si3_prather_2d` or `step_ice_adv2d_card` by `1` ULP (`jnp.nextafter(..., jnp.inf)`) fails the parent-vs-current control with `max ULP = 1 != 0`.
  - `Plant O3-B2`: Perturbing `step_sea_ice` (`_si3_step_with_trace`) by `1` ULP fails the production dispatcher control with `max ULP = 1 != 0`.

### Test (c): Conservation of Every Extensive to Round-Off on a Closed ORCA2 Test Field
- **Prediction**: On the full `(184, 152)` ORCA2 grid with active cyclic east-west seam transport and zero flux across the closed southern boundary and land boundaries (and zero net flux across the north fold / isolated interior patch where `Hbig_pra`, `Hsnow_pra`, and `ice_var_zapneg` do not clip), the domain-integrated total content `SUM_{ji=3..182, jj=3..150} field(ji, jj) * e1e2t(ji, jj)` of every extensive family (`v_i`, `v_s`, `a_i`, `oa_i`, `e_s`, `e_i`, `szv_i`, `sv_i`, `a_ip`, `v_ip`, `v_il`) is conserved to `float64` round-off (`relative_error <= 1e-14`) across both `x -> y` and `y -> x` sweeps and `icycle in {1, 2, 3}`, and when capping/zapping is active the residual diagnostics (`wfx_res`, `hfx_res`, `sfx_res`, `wfx_snw_sni`) close the mass/heat/salt budget (`diag_adv_mass`, `diag_adv_heat`, `diag_adv_salt`).
- **Plant**:
  - `Plant O3-C1`: Dropping the donor flux subtraction `- zf0(ji, jj)` in a single cell breaks extensive conservation and fails the conservation assertion.

### Test (d): End-to-End Differentiability (`jax.jit`, `jax.value_and_grad`, `jax.jvp`)
- **Prediction**: Through one full ORCA2 Prather advection step (`si3_dyn_adv_pra_orca2` / `si3_o3_step`), `jax.jit` primal output is `0` ULP identical to eager execution, reverse-mode gradients (`jax.value_and_grad`) with respect to all extensive fields (`v_i, v_s, a_i, oa_i, e_s, e_i, szv_i, a_ip, v_ip, v_il`), ice velocities (`u_ice, v_ice`), and Prather moments (`sx, sy, sxx, syy, sxy`) are finite and non-zero, and directional derivatives from `jax.value_and_grad` match `jax.jvp` to `|jvp - vjp| / max(|jvp|, |vjp|, 1e-30) <= 1e-11`.
- **Plant**:
  - `Plant O3-D1`: Inserting `jax.lax.stop_gradient` on `v_i` or `prather.ice.sx` zeros the gradient and fails the AD gate with `STATUS PLANT-FIRED`.
