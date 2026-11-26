# SI3 Ladder Rung O1 (`si3_r4`) — Preregistration

## 1. Scope and Reference Configuration (`DECISION 153`, `DECISION 159`)

Rung **O1** (`si3_r4`) is the first ORCA2 entry rung of the standalone SI3 sea-ice fidelity ladder (`docs/ice/fidelity/si3_ladder_design.md` Section 2 & Section 5). Per `DECISION 159`, O1 targets NEMO 5.0.2's shipped `ORCA2_SAS_ICE` reference configuration (`cfgs/ORCA2_SAS_ICE/cpp_ORCA2_SAS_ICE.fcm`: `bld::tool::fppkeys key_si3 key_linssh key_vco_1d3d key_xios key_RK3`, built with `del_key key_xios` for standalone NetCDF I/O without an external XIOS server) with sea-ice dynamics (`ln_icedyn = .false.`) and thermodynamics (`ln_icethd = .false.`) turned **OFF**, and the exact NEMO 5.0.2 ORCA2 SI3 state injected, read, aggregated, halo-exchanged, and written to restart.

This round (`si3_r4`) is **DESIGN + BUILD + ACQUISITION TOOL, then STOP**:
1. Preregister (this document, committed first): record schema, exact bridge/restart/aggregation/halo bar, and all plants.
2. Extend the **ONE** shared ice package (`packages/ice/legoesm/ice/`) with the selectable full ORCA2 SI3 state (`jpl = 5` categories, `nlay_i = 10` ice layers, `nlay_s = 5` snow layers, age, melt pond and lid volumes, all five Prather moments `sx, sy, sxx, syy, sxy` of every extensive variable, and the HFN category bounds `hi_max(0:jpl)` and `hi_mean(1:jpl)`), generalizing lane 3's 1-category `3+3` SI3 state without changing existing defaults or callers. No SST initialization or thickness fit in `legoesm` (that belongs to rung O2).
3. Write the committed O1 acquisition tool (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/`) to the ladder standard, run its read-only preflight (`--preflight-only`), and stop with `ACQUISITION_NEEDED`.

## 2. Oracle Source Citations (`NEMO 5.0.2`)

All statement ordering, branching, and conventions follow `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`:
- **Top-level SI3 init and stepping (`src/ICE/icestp.F90`)**:
  - `ice_init` (`src/ICE/icestp.F90:241-325`): calls `ice_itd_init` (`:282`), `ice_thd_init` (`:284`), `ice_sbc_init` (`:287`), `ice_istate` (`:294`), `ice_var_glo2eqv(1)` (`:295`), `ice_var_agg(1)` (`:296`), `ice_dyn_init` (`:298`), `ice_update_init` (`:300`).
  - `ice_stp` (`src/ICE/icestp.F90:96-238`): calls `store_fields` (`:148`, `:361-394`), `IF( lrst_ice ) CALL ice_rst_opn( kt )` (`:154`), skips `ice_dyn` when `ln_icedyn = .false.` (`:167`) and `ice_thd` when `ln_icethd = .false.` (`:205`), then executes `CALL ice_var_glo2eqv(2)` (`:218`), `CALL ice_var_agg(2)` (`:219`), `CALL ice_update_flx( kt )` (`:221`), and `IF( lrst_ice ) CALL ice_rst_write( kt )` (`:224`).
- **HFN category thickness bounds (`src/ICE/iceitd.F90:715-780`)**:
  - `ice_itd_init` (`opt_hips == 2`, `ln_cat_hfn = .true.` in `cfgs/SHARED/namelist_ice_ref:67-70`, `rn_himean = 2.0_wp` in `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg:23`): solves for `zhmax` by bisection (`src/ICE/iceitd.F90:739-755`) so that $\frac{1}{jpl}\sum_{jl=1}^{jpl} \frac{h_{\max}(jl) + h_{\max}(jl-1)}{2} = \text{rn\_himean}$ where $h_{\max}(jl) = \sqrt{h_{\max}(jl-1)^2 + (2\cdot jl - 1)\,zh_{\max}^2 / (jpl^2 + 1)}$, then sets `hi_mean(jl) = 0.5_wp * (hi_max(jl) + hi_max(jl-1))` (`:757-761`) and `hi_max(jpl) = 999.99_wp` (`:763`).
- **Global-to-equivalent conversion (`src/ICE/icevar.F90:255-428`, `:1564-1587`)**:
  - `ice_var_glo2eqv` computes intensive thicknesses (`h_i`, `h_s`), specific age (`o_i`), bulk and profile salinities (`s_i`, `sz_i`), ice temperatures (`t_i` via quadratic root of enthalpy `e_i` and salinity `sz_i` with `rLfus`, `rcpi`, `rcpw`, `rTmlt`), snow temperatures (`t_s` from `e_s`), snow coverage fraction (`ice_var_snwfra_3d`), and melt pond / lid thicknesses and effective fractions (`h_ip`, `h_il`, `a_ip_frac`, `a_ip_eff`).
- **Category aggregation (`src/ICE/icevar.F90:105-252`)**:
  - `ice_var_agg(kn)`: for `kn >= 1`, sums extensive quantities across `jl = 1..jpl` (`vt_i`, `vt_s`, `at_i`, `at_ip`, `vt_ip`, `vt_il`, `ato_i = MAX(1 - at_i, 0)`, `et_s`, `et_i`, `at_ip_eff`, `tm_su`, `st_i`); for `kn > 1` (`kn == 2` in `ice_stp`), also computes mean thickness/temperature/salinity/age diagnostics (`hm_i`, `hm_s`, `tm_si`, `om_i`, `sm_i`, `tm_i`, `tm_s`, `hm_ip`, `hm_il`).
- **Prognostic restart read/write (`src/ICE/icerst.F90:111-378`, `src/ICE/iceupdate.F90:189-194`, `:438-485`)**:
  - `ice_rst_write` (`src/ICE/icerst.F90:111-197`) and `ice_rst_read` (`:200-378`) read/write `nn_fsbc`, `kt_ice`, `v_i`, `v_s`, `sv_i`, `a_i`, `t_su`, `u_ice`, `v_ice`, `oa_i`, `a_ip`, `v_ip`, `v_il`, `e_s_l01..05`, `e_i_l01..10`, `szv_i_l01..10`, and (when `ln_cpl`) `cnd_ice`, `t1_ice`.
  - `update_rst` (`src/ICE/iceupdate.F90:438-485`) reads/writes `snwice_mass` and `snwice_mass_b`.
- **Prather moment restart read/write (`src/ICE/icedyn_adv_pra.F90:36-46`, `:1145-1503`)**:
  - `adv_pra_init` (`src/ICE/icedyn_adv_pra.F90:1145-1200`) allocates the five second-order moments (`sx, sy, sxx, syy, sxy`) for `ice`, `sn`, `a`, `age`, `c0` (`1..nlay_s`), `e` (`1..nlay_i`), `sal`, `si` (`1..nlay_i`, when `nn_icesal == 4`), and `ap`, `vp`, `vl` (when `ln_pnd_LEV .OR. ln_pnd_TOPO`), and calls `adv_pra_rst('READ')` (`:1203-1381`).
  - `adv_pra_rst('WRITE', kt)` (`:1383-1503`) writes all allocated Prather moments to `numriw`.
- **Lateral boundary conditions — cyclic east-west seam and ORCA2 T-pivot north fold (`src/OCE/LBC/lbc_lnk_pt2pt_generic.h90:99-243`, `src/OCE/LBC/lbc_nfd_generic.h90:22-191`, `src/ICE/icerst.F90:281-282`, `:307-356`, `src/ICE/icedyn_adv_pra.F90:438-482`)**:
  - East-west periodicity (`l_Iperio = .true.`, `nn_hls = 2`, `Ni0glo = 180`, `jpi = 184`): west halo columns `1:2` copy interior columns `181:182`; east halo columns `183:184` copy interior columns `3:4`.
  - South closed boundary (`jpso`): rows `1:2` zeroed (`0.0_wp`).
  - North fold (`c_NFtype == 'T'`, `jpj = 152`): folds rows `151:152` and the right half of row `150` (`ldfull = .true.`) with sign `psgn = +1.0_wp` for scalar T-grid fields (`v_i, v_s, sv_i, a_i, t_su, oa_i, a_ip, v_ip, v_il, e_s, e_i, szv_i, cnd_ice, t1_ice` and second-order moments `sxx*, syy*, sxy*`) and `psgn = -1.0_wp` for first-order gradient moments (`sx*, sy*`) on `'T'` and velocities `u_ice` (`'U'`) and `v_ice` (`'V'`).

## 3. Preregistered Record Schema

The O1 oracle acquisition record (`o1_si3_state_entry_rankXXXX.bin`, `o1_si3_state_stepNNNN_rankXXXX.bin`, and NEMO's NetCDF `ORCA2_00000001_restart_ice.nc` / `ORCA2_00000002_restart_ice.nc`) must contain the full ORCA2 SI3 state on `(jpi, jpj) = (184, 152)` (`Ni0glo = 180, Nj0glo = 148, nn_hls = 2`), `jpl = 5`, `nlay_i = 10`, `nlay_s = 5`:
1. **Header & subdomain metadata**: magic `O1SI3R01`, version `1`, `kt`, `mpprank`, `narea`, `jpi, jpj, jpl, nlay_i, nlay_s, nn_hls, Ni0glo, Nj0glo, nimpp, njmpp, Nis0, Nie0, Njs0, Nje0`, `nn_fsbc, nn_icesal, ln_pnd_LEV, ln_pnd_TOPO, ln_pnd_lids, ln_cpl, lrst_ice`, `rn_Dt, rDt_ice, rn_himean`.
2. **HFN category bounds**: `hi_max(0:5)` (`6` float64 values) and `hi_mean(1:5)` (`5` float64 values).
3. **Grid masks**: `tmask(:,:,1)`, `umask(:,:,1)`, `vmask(:,:,1)`.
4. **Prognostic state and restart variables**:
   - 3D `(jpi, jpj, jpl)`: `v_i`, `v_s`, `sv_i`, `a_i`, `t_su`, `oa_i`, `a_ip`, `v_ip`, `v_il`, `cnd_ice`, `t1_ice`.
   - 2D `(jpi, jpj)`: `u_ice`, `v_ice`, `snwice_mass`, `snwice_mass_b`, `snwice_fmass`.
   - 4D `(jpi, jpj, nlay_s, jpl)`: `e_s`.
   - 4D `(jpi, jpj, nlay_i, jpl)`: `e_i`, `szv_i`.
5. **All five Prather moments (`sx, sy, sxx, syy, sxy`) of every extensive variable**:
   - 3D `(jpi, jpj, jpl)`: `sxice..sxyice`, `sxsn..sxysn`, `sxa..sxya`, `sxage..sxyage`, `sxsal..sxysal`, `sxap..sxyap`, `sxvp..sxyvp`, `sxvl..sxyvl`.
   - 4D `(jpi, jpj, nlay_s, jpl)`: `sxc0..sxyc0`.
   - 4D `(jpi, jpj, nlay_i, jpl)`: `sxe..sxye`, `sxsi..sxysi`.
6. **`ice_var_glo2eqv` intensive outputs**: `h_i`, `h_s`, `o_i`, `h_ip`, `h_il`, `a_ip_frac`, `a_ip_eff`, `s_i`, `sz_i`, `t_i`, `t_s`.
7. **`ice_var_agg` category-aggregated outputs**: `vt_i`, `vt_s`, `at_i`, `at_ip`, `vt_ip`, `vt_il`, `ato_i`, `et_s`, `et_i`, `at_ip_eff`, `tm_su`, `st_i`, `hm_i`, `hm_s`, `tm_si`, `om_i`, `sm_i`, `tm_i`, `tm_s`, `hm_ip`, `hm_il`.

## 4. Preregistered Fidelity Bar

- **Exact bridge, restart, category bounds, `ice_var_glo2eqv`, `ice_var_agg`, `snwice_mass`, and ORCA2 seam/north-fold halo bar**: **`0` ULP** (`max|Δ| == 0.0` in IEEE-754 float64) between `legoesm` and NEMO 5.0.2 on every prognostic field, Prather moment, HFN category bound, equivalent variable, aggregated variable, and halo cell.
- **Trajectory bar (`DECISION 159`, `docs/ice/fidelity/si3_ladder_design.md` Section 3)**: `max|Δ| <= 1e-15` across every prognostic field, aggregated diagnostic, and Prather moment at every recorded step (`kt = 1..2`).
- **Existing caller invariance bar**: **`0` ULP** (`max|Δ| == 0.0`) comparing parent commit (`a572f4b25403`) vs current commit across all real non-SI3 callers (`zero_column_thermodynamics`, `bitz_lipscomb_step`, `c1d_omip_l3_step`, `adv2d_prather_step`) and lane-3 callers (`SI3Config()` default `jpl=1, nlay_i=3, nlay_s=3`, `si3_thd_step`).
- **Differentiability (`jax.jit` + `jax.grad`) bar**: end-to-end `jax.jit`, `jax.value_and_grad`, and `jax.jvp` through the full O1 SI3 state step, `si3_var_glo2eqv`, `si3_var_agg`, and `si3_lbc_lnk_orca2` with finite, non-zero gradients and machine-precision JVP/VJP agreement.

## 5. Preregistered Plants and Negative Controls

Every check ships a planted violation that proves the check fires:
1. **Plant P1 (Code-path caller invariance)**: Perturbing the legacy/default path in `packages/ice/legoesm/ice/` by `+1` ULP must fail the parent-vs-current bit-identity test for existing non-SI3 and lane-3 callers.
2. **Plant P2 (1-ULP state/restart/moment perturbation)**: Perturbing any prognostic field (`v_i`, `e_i`, `szv_i`) or Prather moment (`sxice`, `sxe`) by `+1` ULP must fail the `0` ULP bridge/restart gate.
3. **Plant P3 (Swapped field order / truncated record)**: Swapping `v_i` and `v_s` (or `sxice` and `syice`) or truncating the binary record must fail the schema/layout gate.
4. **Plant P4 (North-fold sign flip)**: Using `psgn = +1.0` instead of `-1.0` on `u_ice`, `v_ice`, or first-order Prather gradients (`sx*`, `sy*`) across the ORCA2 T-pivot north fold must fail the halo gate.
5. **Plant P5 (Aggregation / equivalent variable formula error)**: Perturbing the `ice_var_glo2eqv` ice temperature quadratic root or `ice_var_agg` thickness-weighted mean must fail the `0` ULP aggregation gate.
6. **Plant P6 (`stop_gradient` AD plant)**: Inserting `jax.lax.stop_gradient` into the differentiable O1 forward/aggregation/halo operator must zero the derivative and fail the AD gate.
7. **Plant P7 (Acquisition preflight checklist items 1–7)**:
   - Item 1 (namelist group check): planted unknown group `&namrun_typo` in `namelist_cfg` -> `FAIL`.
   - Item 2 (`sn_*` input file stem check): planted missing stem `missing_forcing_stem` -> `FAIL`.
   - Item 3 (`MY_SRC` scratch compile check): planted undeclared identifier in `MY_SRC` -> `FAIL`.
   - Item 4 (writer output directory check): planted missing output directory -> `FAIL`.
   - Item 5 (ASCII-only check): planted UTF-8 em-dash `—` in `MY_SRC` / `namelist_cfg` -> `FAIL`.
   - Item 6 (compiled `gfortran` multi-rank layout fixture): 4 MPI ranks with distinct rank-dependent data (`1000*(mpprank+1) + ...`); planted rank-independent / collapsed 1D array or wrong subdomain offset (`nimpp/njmpp` off by 1) -> `FAIL`.
   - Item 7 (whitespace-tolerant log check & `EXIT` trap): planted `E R R O R` or missing `AAAAAAAA` in `ocean.output` -> `FAIL` and moves run directory to `refused/`.
