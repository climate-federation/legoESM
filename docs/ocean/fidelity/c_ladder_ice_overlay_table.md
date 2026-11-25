# Coupled C-ladder ICE5 overlay table (Decision 217)

Decision 217 replaces the coupled R5FULLENTRY ice physics/numerics with the
certified ORCA2 SAS five-category configuration while leaving the ocean deck,
keys, forcing, start method, boundaries, run lengths, and two-rank layout
unchanged.

Compared resolved files:

- SAS: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step/record_100step/output.namelist.ice`
- coupled: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round30/oracle_coupled_c_ladder/REC_C3/output.namelist.ice`

The comparison uses `legoesm.core.nemo_namelist.parse_namelist`, so case,
Fortran logical spelling, whitespace, and float formatting do not create false
differences. The union has 48 differing rows. `SAS` means the overlay takes the
SAS value. `KEEP` means Decision 217 keeps the coupled value. `ASK` means the
row is inactive for this analytic start and is not changed without a separate
decision.

## Every resolved `output.namelist.ice` difference

| group | row | SAS | coupled | class | action and reason |
|---|---|---:|---:|---|---|
| `namdyn` | `ln_landfast_L16` | `F` | `T` | ice physics/numerics | SAS — landfast-ice dynamics option. |
| `namdyn_rdgrft` | `ln_distf_exp` | `F` | `T` | ice physics/numerics | SAS — ridging redistribution numerics. |
| `namdyn_rdgrft` | `ln_distf_lin` | `T` | `F` | ice physics/numerics | SAS — ridging redistribution numerics. |
| `namdyn_rdgrft` | `ln_str_smooth` | `T` | `F` | ice physics/numerics | SAS — ridge-strength smoothing. |
| `namdyn_rdgrft` | `rn_fpndrdg` | `1.0` | `0.5` | ice physics/numerics | SAS — pond fraction retained by ridging. |
| `namdyn_rdgrft` | `rn_fpndrft` | `1.0` | `0.5` | ice physics/numerics | SAS — pond fraction retained by rafting. |
| `namdyn_rdgrft` | `rn_porordg` | `0.3` | `0.0` | ice physics/numerics | SAS — ridge porosity. |
| `namini` | `rn_apd_ini_n` | `0.2` | `0.0` | ice initial state | SAS — live analytic-start pond fraction once SAS ponds are enabled. |
| `namini` | `rn_ati_ini_n` | `0.9` | `0.997` | ice initial state | SAS — live analytic-start NH concentration. |
| `namini` | `rn_ati_ini_s` | `0.9` | `0.8` | ice initial state | SAS — live analytic-start SH concentration. |
| `namini` | `rn_hpd_ini_n` | `0.05` | `0.0` | ice initial state | SAS — live analytic-start NH pond depth once SAS ponds are enabled. |
| `namini` | `rn_hti_ini_n` | `2.0` | `2.5` | ice initial state | SAS — live analytic-start NH ice thickness. |
| `namini` | `rn_hts_ini_s` | `0.2` | `0.1` | ice initial state | SAS — live analytic-start SH snow thickness. |
| `namini` | `rn_smi_ini_n` | `6.3` | `9.0` | ice initial state | SAS — live analytic-start NH ice salinity. |
| `namini` | `rn_smi_ini_s` | `6.3` | `4.0` | ice initial state | SAS — live analytic-start SH ice salinity. |
| `namini` | `rn_tmi_ini_n` | `270.0` | `260.0` | ice initial state | SAS — live analytic-start NH ice temperature. |
| `namini` | `rn_tmi_ini_s` | `270.0` | `272.0` | ice initial state | SAS — live analytic-start SH ice temperature. |
| `namini` | `rn_tms_ini_n` | `270.0` | `255.0` | ice initial state | SAS — live analytic-start NH snow temperature. |
| `namini` | `rn_tms_ini_s` | `270.0` | `271.0` | ice initial state | SAS — live analytic-start SH snow temperature. |
| `namini` | `rn_tsu_ini_n` | `270.0` | `250.0` | ice initial state | SAS — live analytic-start NH surface temperature. |
| `namini` | `sn_ati%clvar` | `ati` | `at_i` | ice initial state | ASK — file metadata is dormant with `nn_iceini_file=0`. |
| `namini` | `sn_hti%clvar` | `hti` | `ht_i` | ice initial state | ASK — file metadata is dormant with `nn_iceini_file=0`. |
| `namini` | `sn_hts%clvar` | `hts` | `ht_s` | ice initial state | ASK — file metadata is dormant with `nn_iceini_file=0`. |
| `namini` | `sn_smi%clvar` | `smi` | `sm_i` | ice initial state | ASK — file metadata is dormant with `nn_iceini_file=0`. |
| `namini` | `sn_tmi%clname` | `Ice_initialization` | `NOT USED` | ice initial state | ASK — file metadata is dormant with `nn_iceini_file=0`. |
| `namini` | `sn_tmi%clvar` | `tmi` | `tm_i` | ice initial state | ASK — file metadata is dormant with `nn_iceini_file=0`. |
| `namini` | `sn_tms%clvar` | `tms` | `tm_s` | ice initial state | ASK — file metadata is dormant with `nn_iceini_file=0`. |
| `namini` | `sn_tsu%clvar` | `tsu` | `tmsu` | ice initial state | ASK — file metadata is dormant with `nn_iceini_file=0`. |
| `namitd` | `rn_himin` | `0.1` | `0.05` | ice physics/numerics | SAS — minimum category thickness. |
| `nampar` | `cn_icerst_in` | `restart_ice` | `restart_ice_r4.2` | ice initial state | ASK — input restart name is dormant because `ln_rstart=F` and `nn_iceini_file=0`. |
| `nampar` | `jpl` | `5` | `1` | ice physics/numerics | SAS — five thickness categories required by Decision 217. |
| `nampar` | `nlay_i` | `10` | `3` | ice physics/numerics | SAS — certified ice thermodynamic layering. |
| `nampar` | `nlay_s` | `5` | `3` | ice physics/numerics | SAS — certified snow thermodynamic layering. |
| `nampar` | `rn_amax_n` | `0.997` | `0.99999` | ice physics/numerics | SAS — NH maximum ice concentration. |
| `nampar` | `rn_amax_s` | `0.997` | `0.99999` | ice physics/numerics | SAS — SH maximum ice concentration. |
| `namthd` | `ln_iceda` | `T` | `F` | ice physics/numerics | SAS — activate lateral melting. |
| `namthd_da` | `rn_beta` | `1.0` | absent | ice physics/numerics | SAS — lateral-melting parameter required by the SAS arm. |
| `namthd_da` | `rn_dmin` | `8.0` | absent | ice physics/numerics | SAS — lateral-melting parameter required by the SAS arm. |
| `namthd_do` | `rn_hinew` | `0.1` | `0.05` | ice physics/numerics | SAS — new-ice thickness. |
| `namthd_pnd` | `ln_pnd` | `T` | `F` | ice physics/numerics | SAS — activate melt ponds. |
| `namthd_pnd` | `ln_pnd_LEV` | `T` | `F` | ice physics/numerics | SAS — melt-pond level-ice option. |
| `namthd_sal` | `nn_icesal` | `4` | `2` | ice physics/numerics | SAS — certified salinity scheme. |
| `namthd_sal` | `rn_alpha_cw` | `5.0e-7` | `9.0e-7` | ice physics/numerics | SAS — salinity-profile coefficient. |
| `namthd_sal` | `rn_alpha_gn` | `4.6e-4` | `5.4e-4` | ice physics/numerics | SAS — salinity-profile coefficient. |
| `namthd_sal` | `rn_rc_gn` | `8.1` | `6.4` | ice physics/numerics | SAS — salinity-profile coefficient. |
| `namthd_sal` | `rn_rc_rjw` | `8.0` | `9.0` | ice physics/numerics | SAS — salinity-profile coefficient. |
| `namthd_sal` | `rn_time_gd` | `172800` | `1730000` | ice physics/numerics | SAS — gravity-drainage timescale. |
| `namthd_zdf` | `rn_cnd_s` | `0.31` | `0.5` | ice physics/numerics | SAS — snow conductivity. |

The 13 numeric `namini` rows marked SAS are live in this coupled OMIP start:
both runs resolve `ln_rstart=F`, `ln_iceini=T`, and `nn_iceini_file=0`.
NEMO therefore takes the analytic namelist branch
(`iceistate.f90:307-338`) and distributes the resulting one-category state
across `jpl` categories (`iceistate.f90:356-373`). The eight `sn_*` rows are
read only by the file branch (`nn_iceini_file=1`, `iceistate.f90:245-304`), and
`cn_icerst_in` is read only by the restart branch selected by `ln_rstart` or
`nn_iceini_file=2` (`icestp.f90:395-399`).

## Ocean/SBC rows that directly act on sea ice

These are the differing direct sea-ice interface rows in `namelist_cfg` or the
coupled resolved dynamic namelist. They are not copied from SAS: SAS has no
prognostic ocean, while Decision 217 says the coupled ocean deck and forcing
remain unchanged.

| group | row | SAS | coupled | class | action and reason |
|---|---|---:|---:|---|---|
| `namsbc` | `nn_fsbc` | `1` | `2` | coupling/forcing-interface | KEEP — coupled ocean/ice cadence is fixed by Decision 217 and the C1 contract. |
| `namdrg` | `ln_drgice_imp` | absent | `T` | coupling/forcing-interface | KEEP — ocean-side implicit ice drag has no SAS analogue. |
| `namzdf_tke` | `nn_mxlice` | absent | `2` | coupling/forcing-interface | KEEP — ocean TKE scaling under sea ice has no SAS-ocean analogue. |
| `namzdf_tke` | `rn_mxlice` | absent | `10.0` | coupling/forcing-interface | KEEP — ocean TKE scaling parameter has no SAS-ocean analogue. |
| `namzdf_tke` | `nn_eice` | absent | `1` | coupling/forcing-interface | KEEP — ocean TKE ice-mixing option has no SAS-ocean analogue. |
| `namsbc_sas` | `l_sasread` | `T` | absent | coupling/forcing-interface | KEEP absent — SAS-only ocean-state reader. |
| `namsbc_sas` | `ln_3d_uve` | `F` | absent | coupling/forcing-interface | KEEP absent — SAS-only ocean-state reader. |
| `namsbc_sas` | `ln_read_frq` | `F` | absent | coupling/forcing-interface | KEEP absent — SAS-only ocean-state reader. |
| `namsbc_sas` | `cn_dir` | `./` | absent | coupling/forcing-interface | KEEP absent — SAS-only ocean-state input location. |
| `namsbc_sas` | `sn_usp` | `sas_grid_U, 120 h, uos, ...` | absent | coupling/forcing-interface | KEEP absent — SAS-only ocean velocity input. |
| `namsbc_sas` | `sn_vsp` | `sas_grid_V, 120 h, vos, ...` | absent | coupling/forcing-interface | KEEP absent — SAS-only ocean velocity input. |
| `namsbc_sas` | `sn_tem` | `sas_grid_T, 120 h, sosstsst, ...` | absent | coupling/forcing-interface | KEEP absent — SAS-only ocean temperature input. |
| `namsbc_sas` | `sn_sal` | `sas_grid_T, 120 h, sosaline, ...` | absent | coupling/forcing-interface | KEEP absent — SAS-only ocean salinity input. |
| `namsbc_sas` | `sn_ssh` | `sas_grid_T, 120 h, sossheig, ...` | absent | coupling/forcing-interface | KEEP absent — SAS-only sea-surface-height input. |
| `namsbc_sas` | `sn_e3t` | `sas_grid_T, 120 h, e3t_m, ...` | absent | coupling/forcing-interface | KEEP absent — SAS-only ocean thickness input. |
| `namsbc_sas` | `sn_frq` | `sas_grid_T, 120 h, frq_m, ...` | absent | coupling/forcing-interface | KEEP absent — SAS-only shortwave-fraction input. |

Direct controls specifically named in Decision 217's audit are otherwise
identical: `ln_blk=T`, `nn_ice=2`, `rn_cd_io=0.005`, `nn_flxdist=-1`, and all
seven `namalb` rows. The two NEMO 5 resolved namelists contain no rows named
`rn_cio`, `nn_iceflx`, or an ice emissivity control; no value was inferred for
an absent name.

## Counts and unresolved rows

Counts include differing rows only; the 48 resolved ice rows and 16 direct
ocean/SBC rows are disjoint.

| class | differing rows | applied by ICE5 overlay | retained by Decision 217 | decision needed |
|---|---:|---:|---:|---:|
| ice physics/numerics | 26 | 26 | 0 | 0 |
| ice initial state | 22 | 13 | 0 | 9 |
| coupling/forcing-interface | 16 | 0 | 16 | 0 |
| diagnostics/output | 0 | 0 | 0 | 0 |
| **total** | **64** | **39** | **16** | **9** |

The nine ASK rows are dormant in the registered analytic start. Keeping their
current coupled spellings changes neither the initial state nor any C0/C1
record, but changing them would define future file/restart behavior not covered
by Decision 217.
