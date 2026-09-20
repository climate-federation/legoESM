# NEMO testcase Lane 4 — ORCA2 Phase-2h complete writer audit

Date: 2026-09-06

Config copy audited:
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/MY_SRC`.
The source-set digest file is
`/data/abyssal/dbalwada/nemo-testcases-l4/build/phase2h_systematic_MY_SRC.sha256`
(SHA-256 `4ea891a7810f5818b6f3c6d15bb5dbcd49a44a4b8d808f6756fda6e9460106ff`).
No shipped NEMO source was edited.

## Census rule

`rg -n '\bWRITE\s*\(' MY_SRC/*.F90` was walked statement by statement,
including continued Fortran statements.  Every `WRITE` belongs to one of four
classes:

1. filename construction into a local character scalar;
2. formatted `numout`, namelist, or error-text output consisting of literals,
   scalars, and already-defined diagnostics;
3. an oracle stream header consisting of magic, integer/scalar metadata, and
   `STORAGE_SIZE`; or
4. an oracle binary payload listed below.

Classes 1--3 contain no array storage and need no canonical view.  The table is
therefore the exhaustive array-payload census.  `full 2-D`, `full 3-D`, and
`full 4-D` mean respectively `jpi*jpj`, `jpi*jpj*jpk` (or category in the
third dimension), and `jpi*jpj*category*layer`.  Every such payload can contain
unowned MPI halo bands; physical fields can additionally contain masked land,
bottom, or inactive category/layer storage.  It is now written only through a
zero-first `l4_canon_2d`, `l4_canon_3d`, or `l4_canon_4d` temporary.  These
functions copy only the rank-owned wet cells at the named T/U/V/F staggering
and never assign their source.

`A2D(0)` is NEMO's already-owned interior slice (`90*148` on rank 0), and
`1:npti` is SI3's packed active-point list.  Neither contains halo or land
slots.  Fixed local vectors are completely assigned by array constructors
before their write.  Those three shapes are intentionally not mask-expanded.

## Ocean and forcing payload statements

| source / stream | array operands and storage shape | undefined-risk audit and disposition |
|---|---|---|
| `stprk3.F90`: `oracle_step_entry` | `ts(T,S,Kbb)` full 4-D; `uu(Kbb)`, `vv(Kbb)` full 3-D; `ssh(:,:,Nbb)` full 2-D | halo, land, bottom; all canonical T/U/V |
| `stprk3.F90`: `oracle_bt_frames`, `oracle_stage`, `oracle_rhs` | `uu_b`, `vv_b`, `ssh` full 2-D; `ts` full 4-D; `uu`, `vv` full 3-D | halo and land/bottom; all canonical |
| `stprk3.F90`: `oracle_zdf_entry` | `avm`, `avt`, `dissl`, `utau_b`, `vtau_b` full 3-D/2-D | halo, land, inactive bottom; canonical T/T/T/U/V |
| `stprk3.F90`: `oracle_ocean_surface_input` | `utau`, `vtau`, `qsr`, `qns`, `qns_b`, `emp`, `emp_b`, `sfx`, `sfx_b`, `fwfice`, `rnf`, `snwice_mass`, `snwice_mass_b`, `snwice_fmass`, `rnf_tsc`, `rnf_tsc_b` full 2-D/3-D; iceberg `calving`, `calving_hflx`, `utau_icb`, `vtau_icb` inactive | halo/land; all active arrays canonical. Iceberg-off components use four fully assigned zero 2-D temporaries and are never dereferenced |
| `stp2d.F90`: `oracle_slow_forcing` | `e3u_3d`, `uu(Krhs)`, `umask`, `e3v_3d`, `vv(Krhs)`, `vmask` full 3-D; `Ue_rhs`, `Ve_rhs`, `utauU`, `vtauV` full 2-D | halo/land/bottom; canonical U/V; integer `SIZE` and scalar `r1_rho0` metadata have no risk |
| `stprk3_stg.F90`: `oracle_rkstage1_transport_operands` | `e2u`, `zub`, `e1v`, `zvb` full 2-D; `e3u`, `uu(Kmm)`, `umask`, `zFu`, `e3v`, `vv(Kmm)`, `vmask`, `zFv` full 3-D | halo/land/bottom. The Phase-2g mismatch was 232 `zub` and 198 `zvb` east-halo doubles; all operands canonical U/V |
| `stprk3_stg.F90`: `oracle_transport` and `oracle_tracer_transport` | `zFu`, `zFv`, `zFw` full 3-D | halo/land/bottom; canonical U/V/T |
| `stprk3_stg.F90`: stage-2 term, preupdate, and operand streams | `uu`, `vv` at `Kbb/Kmm/Krhs/Kaa` full 3-D, repeated at registered boundaries | halo/land/bottom; every occurrence canonical U/V |
| `stprk3_stg.F90`: HPG operand stream | each `rhd(:,:,jk)`, `e3w(:,:,jk,Kmm)`, `gdept_z0(:,:,jk,Kmm)` full 2-D slice | halo/land and dry levels; every loop write canonical T |
| `stprk3_stg.F90`: tracer-stage streams | T/S at `Kbb/Kmm/Krhs/Kaa`, and `zFu/zFv/zFw`, all full 3-D | halo/land/bottom; every pre/post-advection, qsr, and update occurrence canonical T/U/V |
| `stprk3_stg.F90`: `oracle_qsr_stage3` | `qsr`, `r3t(:,:,Kmm)` full 2-D; temperature `Krhs` before/after full 3-D | halo/land/bottom; canonical T |
| `eosbn2.F90`: `oracle_rkstage2_eos_operands` | `T`, `S`, `zh`, `zrau`, `zrau0`, `zrau1`, `zrau2`, `zrau3`, `zrau4`, `zrau5`, `zrau6`, `zr1`, `zr2` full 3-D | all 13 automatic work arrays may retain halo/land/bottom storage. They are now allocated only while the dump is armed and every payload is canonical T. EOS coefficient vectors and six constants are fully defined scalars/literals |
| `dynhpg.F90`: `oracle_rkstage2_hpg_literal` | `zhpi_u/v`, `zuap_u/v`, `sum_u/v` full 3-D; `r1_e1u`, `r1_e2v` full 2-D | halo/land/bottom; canonical U/V |
| `dynvor.F90`: `oracle_rkstage2_ene_operands` | per-level `zwz`, `zwx`, `zwy` full 2-D | loop kernels do not own all halos/land; canonical F/U/V |
| `dynspg_ts.F90`: `oracle_bt_ene_coeff` | `ffu_nw/ne/sw/se`, `ffv_nw/ne/sw/se` full 2-D | halo/land; canonical U/V |
| `dynspg_ts.F90`: `oracle_bt_substeps` | per-substep `sshn_e`, `un_e`, `vn_e`, intermediate eta/u/v, next eta/u/v, `sshbb_e`, `ubb_e`, `vbb_e`, `ssha_e`, `ua_e`, `va_e`, depth/transport work arrays, all full 2-D | halo/land at every external-mode time slot; every array canonical at T/U/V |
| `dynspg_ts.F90`: `oracle_bt_drag_operands` | `zCdU_u/v`, `un_e/vn_e`, `hur_e/hvr_e`, products and drag increments full 2-D | halo/land; canonical U/V; substep index scalar |
| `dynspg_ts.F90`: `oracle_bt_advmean_operands` | `r1_e2u`, `r1_e1v`, before/mid/transport/after `u/v`, `zhU/zhV`, `zhup2/zvhp2`, `un_adv/vn_adv` full 2-D; `wgtbtp2(1:icycle)` A1D | all full fields canonical U/V; the fully assigned active weight slice and scalar weights need no masking |
| `dynspg_ts.F90`: `oracle_bt_ordered_operands` | external-mode u/v/eta/depth/transport fields across the first two substeps, full 2-D | halo/land; every array canonical at T/U/V; stage weights and `rDt_e` are scalars |
| `traadv.F90`: `oracle_rkstage3_wzv` | `ww` before/after and `pFw` full 3-D; optional `wi` full 3-D only if allocated/active | halo/land/bottom; canonical T. Resolved `ln_zad_Aimp=.false.`, so `wi` has no schema slot and is not dereferenced |
| `traqsr.F90`: `oracle_rgb_chl` | one-component `sf_chl(1)%fnow` and `qsr` full 2-D horizontal storage; `r3t(Kmm)` `A2D(0)`; `gdepw_1d(1:jpk)` A1D | chlorophyll/qsr have halo/land slots and are canonical T; `r3t` is already owned. The complete vertical-coordinate vector is defined and remains A1D |
| `sbcblk.F90`: `oracle_sbcblk_o1` frame 0 | nine `sf(*)%fnow(:,:,1)` arrays, each `A2D(0)` | already the owned `90*148` slice; fld_read defines all cells; no canonical expansion |
| `sbcblk.F90`: `oracle_sbcblk_o1` frame 1 | `theta_air_zt`, `q_air_zt`, `precip` local zero-first `A2D(0)` copies; `sst_m`, `ssu_m`, `ssv_m`, `tsk`, `ssq`, bulk outputs, `qsr/qns/emp/utau/vtau/taum/wndm` as `A2D(0)` | all are owned slices. `pcd_du` and `qlwn` are wholly unowned in resolved NCAR and are written from zero-first local fields; no model array is changed |

## SI3 payload statements

| source / stream | array operands and storage shape | undefined-risk audit and disposition |
|---|---|---|
| `icedyn_adv_pra.F90`: every `oracle_si3_prather` write | category moments `sx/sy/sxx/syy/sxy` for ice volume, snow, area, age, salinity and optional pond/lead families full 3-D; layered ice/snow energy and optional layer salinity moments full 4-D | halo, land, inactive category/layer. Every operand is canonical T; optional groups remain behind the allocation-owning resolved selectors |
| `icestp.F90`: `oracle_si3_exchange_frames` | category heat/albedo/temperature/evaporation fields full 3-D; ocean/ice stress, freshwater, heat, snow, drag, aggregate exchange fields full 2-D | halo/land/inactive categories; every payload canonical T. The source-undefined implicit-drag field is first zeroed and copied only if `ln_drgice_imp`; nit000 aggregate halos use zero-first owned-slice locals |
| `icethd.F90`: `oracle_si3_thd_frames`, global frame | `a_i`, `v_i`, `v_s`, `sv_i`, `oa_i`, `t_su`, `a_ip`, `v_ip`, `v_il` full 3-D; `e_i`, `e_s`, `szv_i` full 4-D | halo/land/inactive category/layer; every operand canonical T |
| `icethd.F90`: thermodynamic packed frames | `a_i_1d`, `h_i_1d`, `h_s_1d`, `t_su_1d`, fluxes and thermodynamic work fields as `1:npti`; `e_i_1d`, `e_s_1d`, `sz_i_1d`, `t_i_1d`, `t_s_1d` as `1:npti,1:nlay_*` | packed active-point slices only; no halo/land/unallocated tail is written. Header counts are derived as `(13+nlay_s)*npti` and `(3+3*nlay_i+2*nlay_s)*npti` from the write lists |
| `icesbc.F90`: `oracle_si3_bulk_operands` | `z(17)`, `z1(50)`, `z2(39)` fixed local vectors | each vector is completely assigned by one array constructor from the registered active point before `WRITE`; no full-domain or unallocated storage |

## Result and acceptance boundary

The only uncensored full-domain array writes remaining in MY_SRC are model
restart/history writes inherited from NEMO itself; they are not new oracle
streams and are verified separately by the ordinary-output identity control.
All config-local oracle stream array writes now satisfy one of the three safe
forms: zero-first canonical full-domain view, already-owned `A2D(0)`/packed
slice, or completely assigned fixed local vector.

The scalar-math rebuild is
`nemo_ORCA2_OMIP_L4_phase2h_systematic.exe`, SHA-256
`c47a1a6bd9b1873f28cf3eaa34a6264cd3ed65e24d16f8d145c1137e2671a308`.
Its successful build log has SHA-256
`7146ac801ee88981a4dac1115970588550eec3062200d436bf60e0ad50e198b3`,
and `nm -D` found zero `_ZGV*` symbols.  This audit is necessary but not
sufficient: VARIANT V2 remains unpinned until the replacement twins produce
**92 / 92 raw-byte-identical oracle records** and each arm passes schema,
ordinary-output identity, and planted controls.
