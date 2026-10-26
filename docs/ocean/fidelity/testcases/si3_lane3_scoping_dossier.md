# SI3 lane 3 — scoping dossier (research + planning only)

Status: **SCOPING. No implementation, no model run, no oracle build.**  This
document seeds the lane-3 codex dispatches for tracker #1699.  Every
`file:line` below was read in this session from the pinned sources; anything
inferred is labelled **PLAUSIBLE**.  Corrections to two earlier agent reports
are recorded in §7 (Rule 11).

## Provenance

| item | value |
|---|---|
| Oracle source | `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`, commit `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796` (same tree lane 1 certified) |
| Target ice deck | `/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_ice_cfg` SHA256 `9be4731d1835ebd042e590421fb43cec6d984f114f049728acf1a150edc90470` over `namelist_ice_ref` SHA256 `d18c3f3c147f65c60da65baae6f84084113c63afb86a8f70166e359b0afb5245` |
| Target ocean-side ice rows | `EXPREF/namelist_cfg:106` `nn_fsbc=4`; `:111` `nn_ice=2`; `:115` `ln_ice_embd=.FALSE.`; `:139-142` `ln_Cx_ice_cst=T, rn_Cd_ia=rn_Ce_ia=rn_Ch_ia=1e-3`; `:261` `ln_drgice_imp=.true.`; `namelist_ref:666` `nn_fwb_voltype=1` |
| Coverage matrix being retired | `docs/ocean/fidelity/omip_orca1_coverage_matrix.md` on `origin/fidelity/omip-orca1-coverage-audit` (rows CPP-SI3, SBC-ICE, ICE-*) |
| Lane-1 pattern | `origin/fidelity/nemo-testcases-l1-codex:docs/ocean/fidelity/testcases/nemo_testcases_l1_{preregistered_dossier,phase1_receipt,phase2_preregister,phase3_preregister,phase3_receipt}.md` |
| Method | `.claude/skills/oracle-fidelity/SKILL.md` (Rules 0, 1, 1b, 1c, 1d, 6, 9, 11) |

Resolution order is `namelist_ice_cfg` over `namelist_ice_ref`
(`icestp.F90:259-260` loads both; each `READ_NML_REF`/`READ_NML_CFG` pair,
e.g. `icestp.F90:345-346`).

---

## 1. Resolved ORCA1 ice configuration

"cfg" = set in `namelist_ice_cfg`; "ref" = inherited from `namelist_ice_ref`
because the cfg is silent.  Source column = where the selector is read and
the concrete routine/branch it selects.

| Block | Selector → resolved value | Origin | SI3 source (read) |
|---|---|---|---|
| `nampar` | `jpl=1`, `nlay_i=3`, `nlay_s=3` | cfg :24-26 | read `icestp.F90:345-346`; `rDt_ice = nn_fsbc*rn_Dt = 4*3600 = 14400 s` at `icestp.F90:377` |
| `nampar` | `ln_icedyn=T` | ref :29 | gate `icestp.F90:167-168` (`.AND. .NOT.ln_c1d`) |
| `nampar` | `ln_icethd=T` | cfg :32 | gate `icestp.F90:206` |
| `nampar` | `ln_virtual_itd=F` | ref :27 | `icethd.F90:158-159` (`ice_thd_mono` INACTIVE); `icethd_zdf_bl99.F90:290-300` G(he) enhancement INACTIVE |
| `nampar` | `rn_amax_n=rn_amax_s=0.99999` | cfg :27-28 | clipped to `1-epsi10` at `icestp.F90:364-365`; 2-D field `icestp.F90:277-279`; used `icecor.F90:78-81`, `icethd_do.F90:241-245` |
| `namitd` | `ln_cat_hfn=T`, `rn_himean=2.0`, `rn_himin=0.05`, `rn_himax=99` (ref :46) | cfg :37-39 | `iceitd.F90:694-769`; with `jpl=1`: `hi_max(0)=0`, `hi_max(1)` from `:741-746` then **overwritten to `rn_himax=99`** at `:761`; `ice_itd_rem` never called (`icethd.F90:179` guard), `ice_itd_reb` never called (`icecor.F90:85` guard) |
| `namdyn` | `ln_dynALL=T` | cfg :44 | `icedyn.F90:286` → `np_dynALL`; branch `:130-135` = rhg → adv → rdgrft → `ice_cor(kt,1)` |
| `namdyn` | `rn_ishlat=2` (no-slip ice) | ref :57 | `icedyn_rhg_evp.F90:210-225` builds `fimask` |
| `namdyn` | `ln_landfast_L16=T`, `rn_lf_depfra=0.125`, `rn_lf_bfr=15`, `rn_lf_relax=1e-5`, `rn_lf_tensile=0.05` | cfg :46; params ref :59-63 | tensile `icedyn_rhg_evp.F90:258`; basal stress `:335-361`; velocity solve `:554-568,605-619` (`zTauB`, `rn_lf_relax`) |
| `namdyn` | `sn_icbmsk`, `sn_fastmsk` = `'NOT USED'` | ref :69-70 | zeroed `icedyn.F90:319-320`; still `fld_read` every step `:115-116` |
| `namdyn_rdgrft` | `ln_str_H79=T`, `rn_pstar=2e4`, `rn_crhg=20`, `ln_str_smooth=F` | cfg :52-55 | `ice_strength` `icedyn_rdgrft.F90:908`; called ONCE per ice step from `icedyn_rhg_evp.F90:255`; upstream-advected re-evaluation inside subcycles `:758-777` (formula `:774`) |
| `namdyn_rdgrft` | `ln_distf_exp=T` (`rn_murdg=3.0` ref :87), `ln_partf_exp=T` (`rn_astar=0.03` ref :93), `ln_ridging=T` (`rn_hstar=25` ref :95), `ln_rafting=T` (`rn_hraft=0.75`, `rn_craft=5` ref :100-101), `rn_porordg=0`, `rn_fpndrdg=rn_fpndrft=0.5`, `rn_fsnwrdg=rn_fsnwrft=0.5` (ref :97,102), `rn_csrdg=0.5` (ref :88) | cfg :57-65 | participation `icedyn_rdgrft.F90:458-470`; ridge/raft split `:475-478` (`tanh(rn_craft*(h-rn_hraft))`); `hrexp=rn_murdg*sqrt(h)` `:571`; closing `:243-252`; iteration `:289-335` (`jp_itermax=20` `:169`); porosity `:708-710`; snow loss `:754-758`; pond loss `:762-766` (inert, ponds off) |
| `namdyn_rhg` | `ln_rhg_EVP=T`; **`ln_aEVP=T`**; `rn_creepl=2e-9`; `rn_ecc=2`; `nn_nevp=100`; `rn_relast=0.333` (inert under aEVP); `nn_rhg_chkcvg=0` | cfg :70; rest **ref :110-116** | dispatch `icedyn_rhg.F90:157,77-79`; aEVP: `zdtevp = rDt_ice` `icedyn_rhg_evp.F90:244` (not `rDt_ice/nn_nevp` `:237`), per-cell `zalph1` `:446`, `zbeta` `:466-470`, F-point `zalph2` `:476`; loop `DO jter=1,nn_nevp` `:381`; stresses `:458-461` (T), `:488` (F) |
| `namdyn_adv` | `ln_adv_Pra=T`, `ln_adv_UMx=F` | cfg :75-76 | `icedyn_adv.F90:155,88-91`; Prather `icedyn_adv_pra.F90:56`; CFL sub-cycling 1/2/3 `:119-132`; alternating x/y sweep order `:258-300` vs `:308-336`; moments restart-written `:487` → `:1383-1497` |
| ice `namsbc` | `rn_Cd_io=5e-3`, `nn_snwfra=2`, `rn_snwblow=0.66`, `nn_flxdist=-1`, `nn_qtrice=0` | ref :136-150 | read `icesbc.F90:453`; `nn_flxdist=-1` ⇒ `ice_flx_dist` never called (`:171,187`); `nn_qtrice=0` ⇒ constant snow extinction `icethd_zdf_bl99.F90:151-152` and GM77 transmission `sbcblk.F90:1323-1337` |
| ice `namsbc` | `ln_cndflx=F`, `ln_cndemulate=F` | cfg :81-82 | `icethd_zdf.F90:61` → `ice_thd_zdf_BL99(np_cnd_OFF)`; `ice_update_flx` non-conduction branch `iceupdate.F90:109-111` |
| `namthd` | `ln_icedH=T`, `ln_icedO=T`, `ln_leadhfx=T` | ref :157,159,160 | `icethd.F90:150,181`; `icesbc.F90:421,427` |
| `namthd` | `ln_icedA=F` | cfg :87 | `icethd.F90:161` INACTIVE; `ice_thd_da_init` skipped `icethd.F90:578` |
| `namthd_zdf` | `ln_zdf_BL99=T`, `ln_cndi_P07=T`, `ln_cndi_U64=F`, `rn_kappa_i=1`, `rn_kappa_s=10`, `ln_zdf_chkcvg=F` | ref :165-174; cfg :94 | P07 `icethd_zdf_bl99.F90:261-275` (`k = rcnd_i + 0.09*S/T - 0.011*T`, `rcnd_i=2.034396` `phycst.F90:60`); Picard loop `:241` (`iconv_max=200` `:76`); `zh_min=1e-3` `:90`, `zhi_ssl=0.10` `:89` |
| `namthd_zdf` | `rn_cnd_s=0.5` | cfg :92 | `rcnd_s=rn_cnd_s` `icethd_zdf.F90:112`; used `icethd_zdf_bl99.F90:317-320` |
| `namthd_do` | `rn_hinew=0.05`, `ln_frazil=F` | cfg :103; ref :191 | `icethd_do.F90:375-379` (`ht_i_new=rn_hinew` where `qlead<0`); frazil block `:381` INACTIVE |
| `namthd_sal` | `nn_icesal=2`, `ln_flushing=T`, `ln_drainage=T`, `rn_simin=0.1`, **`rn_sinew=0.75`**, `rn_sal_gd=5`, `rn_time_gd=1.73e6`, `rn_sal_fl=2`, `rn_time_fl=8.64e5` | cfg :108-124 | `icethd_sal.F90:207-248`: flushing if `t_su>=rt0` `:221-222`, drainage if `t_su<=t_bo` `:223-224`, `sfx_bri` `:231`, clamp `[rn_simin, rn_sinew*sss]` `:237`, linear profile `ice_var_salprof1d` `:248`; new-ice salinity `rn_sinew*sss` `icethd_do.F90:176`; snow-ice salinity update `icethd_dh.F90:513-515`.  Option-4 rows (`nn_liquidus`, `nn_drainage`, ...) are read but inert. |
| `namthd_pnd` | `ln_pnd=F` | cfg :151 | `icethd.F90:176-177` INACTIVE; `ice_thd_pnd_init` still called `:581` |
| `namini` | `ln_iceini=T`, `nn_iceini_file=1`; file vars `ht_i, ht_s, at_i, sm_i, tmsu`; `tm_i`/`tm_s` `NOT USED` | cfg :158-189 | `iceistate.F90:198-257`; T-fill rules for missing `tm_i/tm_s` `:219-235`; `ice_var_itd_1c1c` path (jpl=1) `icevar.F90:1087` via `:311-315`; `s_i` clamp `:337`; `at_i` cap `:389-393` |
| `namalb` | `rn_alb_sdry=.85, smlt=.75, idry=.64, imlt=.53, dpnd=.18, lpnd=.30, hpiv=1.0` | ref :306-312 | `icealb.F90:49-193`; thin-ice log blend `:159-162`; snow exp blend `:167-169`; clear/overcast `:181-185` with `cloud_fra=pp_cldf` (`sbcmod.F90:234`) |
| `namdia` | `ln_icediachk=F`, `ln_icectl=F` | cfg :198; ref :321 | conservation checks and debug prints INACTIVE (we will turn `ln_icediachk=T` in oracle runs — see §3) |
| ocean `namsbc_blk` | `ln_Cx_ice_cst=T`, `rn_Cd_ia=rn_Ce_ia=rn_Ch_ia=1e-3` | ocean cfg :139-142 | `sbcblk.F90:1101-1103`; stress `:1143-1147` (`rhoa*Cd*|U10|*U10`, wind NOT relative to ice: `wndm_ice` from `pwndi/pwndj` only, `:1082-1084`; `rn_vfac=0` ref :319) |
| ocean `namdrg` | `ln_drgice_imp=T` | ocean cfg :261 | `rCdU_ice` set `iceupdate.F90:391`; consumed `zdfdrg.F90:117-126`, `dynzdf.F90:161,298,472` |

Two ORCA1 deck facts the coverage matrix's SI3 table did not carry:
(a) **aEVP is active** (matrix row said "EVP; 100 subcycles and relaxation
.333") — see `namdyn_rhg` row; (b) `rn_sinew=0.75` sits next to the ref's own
comment "must be set to 0.30 if nn_icesal=2" (`namelist_ice_cfg:116-117`).
Both are TARGET facts to replicate, not to fix (Rule 9); (b) is flagged in §6.

---

## 2. SI3 step call graph (ORCA1 disposition)

Entry: `sbcmod.F90:474-477` `SELECT CASE(nn_ice) ... CASE(2) CALL ice_stp(kt,Kbb,Kmm,nsbc)`,
inside `sbc()`, which `stprk3.F90:138` calls as `sbc(kstp, Nbb, Nbb)` (RK3;
MLF: `stpmlf.F90:170`).  Within `sbc()` the order is `sbc_ssm` (`:403`) →
bulk `sbc_blk` (`:419`) → ice `ice_stp` (`:477`) → icebergs → `sbc_rnf`
(`:494`) → `sbc_ssr` (`:496`) → `sbc_fwb` (`:498`).  So the ice step sees the
ocean's `Nbb` surface (nn_fsbc-averaged) and its outputs feed runoff/SSS/FWB
and then stage 1 of the same RK3 step.

`ice_stp` (`icestp.F90:96-236`), gated `MOD(kt-1,nn_fsbc)==0` (`:126`) — i.e.
every 4th ocean step for ORCA1.  Ordered CALL list:

| # | line | CALL | role | ORCA1 |
|---:|---|---|---|---|
| 1 | :130-134 | `u_oce=ssu_m; v_oce=ssv_m` | ocean surface current seen by ice = nn_fsbc mean (`sbcssm.F90:141-170`) | active |
| 2 | :136 | `eos_fzp(sss_m, t_bo, kbnd=0)` | freezing point from mean SSS; TEOS-10 branch `eosbn2.F90:1672-1682` (no depth term: `pdep` absent) | active |
| 3 | :151 | `store_fields` | copy `*_b` (before) ice state `:388-426` | active |
| 4 | :156 | `ice_frm` | form drag | INACTIVE (`ln_Cx_ice_frm=F` ref :263) |
| 5 | :159 | `ice_sbc_tau(kt,ksbc,utau_ice,vtau_ice)` | air-ice stress; `ksbc=jp_blk` → `blk_ice_1` (`icesbc.F90:83-87`) | active |
| 6 | :164 | `diag_set0` | zero all `wfx_*`, `sfx_*`, `hfx_*`, `tau_icebfr`, `qsb_ice_bot`, `fhld` (`:429-501`) | active |
| 7 | :165 | `ice_rst_opn` | restart file open | infra |
| 8 | :168 | `ice_dyn(kt,Kmm)` | see 8a-8d | active |
| 8a | `icedyn.F90:132` | `ice_dyn_rhg` → `ice_dyn_rhg_evp` (`icedyn_rhg.F90:79`) | aEVP momentum + stress, landfast | active |
| 8b | `icedyn.F90:133` | `ice_dyn_adv` → `ice_dyn_adv_pra` (`icedyn_adv.F90:90`) | Prather transport of `v_i,v_s,a_i,oa_i,e_s,e_i,sv_i` | active |
| 8c | `icedyn.F90:134` | `ice_dyn_rdgrft` | ridging/rafting | active |
| 8d | `icedyn.F90:135` | `ice_cor(kt,1)` | thin-ice/`amax` corrections, `zapsmall` (`icecor.F90:67-114`) | active |
| 9 | :171 | `diag_trends(1)` | dyn trends (only if `ln_icediachk` or output) | inert |
| 10 | :176 | `bdy_ice` | open-boundary ice | INACTIVE (global; **PLAUSIBLE** `ln_bdy=F`, ocean `nambdy` not read this session) |
| 11 | :182-184 | `ice_var_glo2eqv(2)`, `ice_var_agg(1)`, `store_fields` | h_i/h_s/t from v/e; `at_i`; re-store before | active |
| 12 | :201 | `ice_sbc_flx(kt,ksbc)` | `ice_alb` (`icesbc.F90:150`) → `blk_ice_2` (`:162`) → `ice_flx_other` (`:195`: `qsb_ice_bot`, `qlead`, `fhld`) | active |
| 13 | :206 | `ice_thd(kt)` | see 13a-13j | active |
| 13a | `icethd.F90:115` | `ice_thd_frazil` | `ht_i_new=rn_hinew` where `qlead<0` (`icethd_do.F90:375-379`) | active (frazil off) |
| 13b | `:140` | `ice_thd_1d2d(jl,1)` | pack ice-covered points | active |
| 13c | `:148` | `ice_thd_zdf` → `ice_thd_zdf_BL99(np_cnd_OFF)` | 3+3 layer heat diffusion, P07 | active |
| 13d | `:150` | `ice_thd_dh` | precip/sublimation/surface+bottom growth-melt/snow-ice (`icethd_dh.F90:166-475`) | active |
| 13e | `:152,:156` | `ice_thd_temp` | enthalpy→T (twice) | active |
| 13f | `:154` | `ice_thd_sal` | option-2 flushing/drainage | active |
| 13g | `:158-161` | `ice_thd_mono`, `ice_thd_da` | virtual-ITD melt, lateral melt | INACTIVE |
| 13h | `:163` | `ice_thd_1d2d(jl,2)` | unpack | active |
| 13i | `:176-183` | `ice_thd_pnd` (off), `ice_itd_rem` (jpl=1 → off), `ice_thd_do` (on), `ice_cor(kt,2)` (on) | ponds / remap / open-water growth / corrections | as marked |
| 13j | `:192-201` | zero `u_ice/v_ice` where ice vanished; `lbc_lnk` | halo | active |
| 14 | :209-211 | `diag_trends(2)`, `ice_var_glo2eqv(2)`, `ice_var_agg(2)` | thermo trends; equivalents | active |
| 15 | :213 | `ice_update_flx(kt)` | **ocean fluxes** `qsr,qns,emp,sfx,fr_i,tn_ice,snwice_mass` (§5) | active |
| 16 | :218-226 | `ice_dia`, `ice_drift_wri`(off), `ice_wri`, `ice_rst_write`, `ice_ctl`(off) | output | infra |
| 17 | :233 | `ice_update_tau(kt, uu(:,:,1,Kbb), vv(:,:,1,Kbb))` | **outside the nn_fsbc gate: every ocean step**; `utau/vtau` and `rCdU_ice` (§5) | active |

Init (`ice_init`, `icestp.F90:241-325`): `par_init` → `ice_itd_init` →
`ice_thd_init` → `ice_sbc_init` → `ice_istate_init` → `ice_istate`
(file init) → `ice_var_glo2eqv(1)`/`agg(1)` → `ice_dyn_init`
(→ `rdgrft_init`, `rhg_init`, `adv_init`; `icedyn.F90:323-325`) →
`ice_update_init` → `ice_alb_init` → `ice_dia_init` → `ice_drift_init`;
`fr_i=at_i`, `tn_ice=t_su`, `drag_io=rn_Cd_io` (`:308-312`).

Coverage discipline (Rule 1): the lane-3 registry must list all 17 top-level
CALLs and the 8a-8d / 13a-13j children with a disposition each.  Items 4, 10,
13g, 13i(ponds, remap) are WAIVED-INACTIVE by the resolved deck; every other
row is UNVERIFIED until a receipt names it.

---

## 3. Testcase ladder within lane 3

Shipped SI3 test cases (`tests/demo_cfgs.txt`): `ICE_ADV1D OCE SAS ICE`,
`ICE_ADV2D OCE SAS ICE`, `ICE_RHEO OCE SAS ICE`.  All three compile
`key_si3 key_linssh key_vco_1d` (+`key_xios`, which lane 1 removed) — the
"ocean" is the Stand-Alone-Surface module with `l_sasread=.false.` (surface
fields set to 0; `namelist_cfg:82` in each), `ln_usr=T`, `nn_fsbc=1`,
`nn_ice=2`, **`ln_icethd=.false.`** in all three ice namelists.  So the
shipped cases exercise **dynamics only** against a resting, ice-free-forcing
ocean.  None forces BL99 thermodynamics; the shipped forced column is
`cfgs/C1D/EXP_SASICE` (`C1D OCE ICE TOP`, `cpp_C1D.fcm`: `key_si3 key_RK3
key_vco_1d3d key_linssh`), which needs an ERA5 station file not in the tree
(`EXP_SASICE/namelist_cfg:122-163`).

Analogue of OVERFLOW→LOCK→GYRE, ascending by subsystem and by how much new
legoESM machinery each needs.  Effort classes follow the coverage matrix
(S/M/L/XL/XXL).  "Overlay" = namelist rows we must change from the shipped
case to select the ORCA1 arm; each overlay is a declared deviation from the
documented test, kept minimal so the case stays identifiable (lane-1 rule).

| rung | oracle case | certifies (SI3 routines) | shipped → ORCA1 overlay | legoESM new/reuse | effort |
|---:|---|---|---|---|---|
| 3.1 | **ICE_ADV1D** (Schär & Smolarkiewicz 1996 / Lipscomb 2004; closed 1-D box, `dx=4 m`, `dt=2 s`, 40 steps; velocity `u = 1.5*rn_uice*ξ` converging at centre, `icedyn.F90:145-154`; IC ramp/step shapes `EXPREF/make_initice.py` (Lipscomb-2004 block, lines ~96-118)) | `ice_dyn_adv_pra` alone (`np_dynADV1D` branch); `ice_var_zapsmall` | `ln_adv_Pra=T, ln_adv_UMx=F` (shipped UMx, `namelist_ice_cfg:54-56`); `nlay_i=3, nlay_s=3` (shipped `nlay_i=2`) | NEW: Prather second-order-moment transport on a Cartesian C-grid (reuse `create_beta_plane_cgrid_geometry` from lane 1; existing `transport.py` is PPM/donor-cell, no moments). Restart moment arrays give the exact-state gate | **M** |
| 3.2 | **ICE_ADV2D, purpose 1a** (bi-periodic 300 km, `dx=3 km`, `dt=1200 s`, 485 steps, `rn_uice=rn_vice=0.5`; Gaussian h / square a IC `make_INITICE.py` (lines ~98-121)) | Prather 2-D with alternating sweep order + CFL sub-cycling (`icedyn_adv_pra.F90:119-132,258-336`); `zapsmall` | same as 3.1 | same Prather kernel, periodic halos; **h overshoot** is the documented diagnostic (`tests/README.rst:181-186`) | **S** on top of 3.1 |
| 3.3 | **ICE_ADV2D, purpose 1b** (`ln_dynRHGADV=T`, constant `utau_ice=1.3 N/m²` `MY_SRC/usrdef_sbc.F90:93`) | `ice_dyn_rhg_evp` + Prather + `Hpiling` (`icedyn.F90:139-142`) — first rheology receipt, no ridging | `ln_dynADV2D=F, ln_dynRHGADV=T`; `ln_aEVP=T` (ref default holds), `nn_nevp=100`; H79 `rn_pstar=2e4, rn_crhg=20` | NEW: **C-grid** aEVP (existing `rheology.py`/`dynamics.py` are A-grid: `rheology.py:238`, `dynamics.py:129`), incl. `fimask`/no-slip (`rn_ishlat=2`), sea-surface-tilt term (`zsshdyn`, zero here), ice-ocean drag vs `ssu_m=0` | **L** |
| 3.4 | **ICE_RHEO** (Heorton et al. 2018; closed 2000 km box, `dx=2 km`, `dt=30 s`, 720 steps; analytic spin-up wind `MY_SRC/usrdef_sbc.F90:98-145`, `Rwind=-0.8`, `Cd_atm=1.4e-3`, wind relative to ice) | full `ln_dynALL`: aEVP + Prather + `ice_dyn_rdgrft` (H79, exp participation, exp redistribution, rafting) + `ice_cor(kt,1)`; LKF shear pattern = documented behaviour (`EXPREF/README`, `sishea_EVP.png`) | **shipped default is EAP** (`namelist_ice_cfg:50-58`: `ln_rhg_EVP=F, ln_rhg_EAP=T, ln_aEVP=F, nn_nevp=200, rn_pstar=2700`) → overlay `ln_rhg_EVP=T, ln_rhg_EAP=F, ln_aEVP=T, nn_nevp=100, rn_pstar=2e4`; rdgrft rows to ORCA1 cfg :52-65; `ln_landfast_L16=T` needs nonzero bathymetry (`hu,hv` in `icedyn_rhg_evp.F90:342,349`) — `rn_bathy`/`usrdef_zgr` decision, §8 | NEW: ridging+rafting per SI3 (`ridging.py` has Lipscomb-2007 participation but no rafting, different redistribution constants); `ice_cor`; landfast basal stress; reuse C-grid aEVP from 3.3 | **L** (XL if landfast on) |
| 3.5 | **C1D EXP_SASICE column** (`cfgs/C1D`, `ln_c1d=T` ⇒ `ice_dyn` skipped `icestp.F90:167`; `nn_fsbc=1`, `dt=3600`, ECMWF bulk, EOS-80) | `ice_sbc_flx` (`blk_ice_2`, `ice_alb`, `ice_flx_other`), `ice_thd` chain 13a-13j, `ice_update_flx` | ice namelist: **whole ORCA1 ice deck** (shipped C1D ice cfg is empty ⇒ ref defaults `nlay_i=10, nlay_s=5, nn_icesal=4, ln_pnd=T`); ocean side: `ln_NCAR=T` + `ln_Cx_ice_cst` 1e-3 (ORCA1), TEOS-10; forcing: either fetch the ERA5 station file or replace by an analytic `usrdef_sbc` (decision §8) | NEW: 3+3-layer BL99 with P07 (`_future/bitz_lipscomb.py` has a tridiagonal core with **Untersteiner** conductivity, no snow layers, unwired — `bitz_lipscomb.py:1-19,28,126-135,147`); NEW option-2 salinity; NEW `ice_thd_dh` sequence; NEW open-water growth per `icethd_do.F90`; reuse constants/tridiagonal solver; reuse NCAR bulk + ice-IC loader | **XL** |
| 3.6 | **coupled slab** (ICE_ADV2D purpose 2 = `ln_icedyn=F` with edited `usrdef_sbc`, `README`; or a 1-layer OCE+ICE variant of 3.5 with SAS replaced by the real ocean) | `sbc_ssm` averaging with `nn_fsbc=4`, `eos_fzp`, `ice_update_flx` → `tra_sbc_RK3`/`ssh`, `ice_update_tau` every step with `ln_drgice_imp`, `sbc_fwb` `nn_fwb_voltype=1` | build a new work config (lane-1 precedent `OVERFLOW_OMIP_L1`) | NEW: legoESM ice↔ocean exchange card matching §5 (existing `coupler/ocean_forcing.py` exchanges stress/heat/fresh water but with its own conventions, `:37,89-90,221-249`) | **L** |
| 3.7 | ORCA1 ice-on (after lanes 2 and 4) | composition | — | — | out of lane 3 |

Phase structure per rung mirrors lane 1: (1) oracle build with the exact
overlay committed as `tests/<CASE>_OMIP_L3/EXPREF/namelist_ice_cfg`, no
fallback on failure; (2) fp64 write-only dumps at ice-step entry (before
`store_fields`) and after each of `ice_dyn_rhg / adv / rdgrft / ice_cor /
ice_thd / ice_update_flx`, plus `ln_icediachk=T` for the oracle's own
conservation ledger (`ice_cons_hsm` around every block, e.g.
`icedyn_rhg.F90:63,101`); (3) geometry + **restart coverage gate**
(Appendix A inventory, every variable VERIFIED/WAIVED); (4) legoESM card =
pure config over canonical blocks; (5) `kt=1` entry match; (6)
first-divergence sweep in the §2 order, stop at first over-bar row.

Time-level note (Rule 1d): SI3 has no leapfrog; the ice "before" state is
the `*_b` copy made at `icestp.F90:151` and re-made at `:184` after dynamics.
Ocean fields the ice reads are `Nbb` means (`sbc(kstp,Nbb,Nbb)`), and
`ice_update_tau` uses instantaneous `uu(:,:,1,Kbb)` (`icestp.F90:233`).  The
registry must distinguish `a_i_b` (post-dynamics, pre-thermo) from the
step-entry `a_i` — `ice_update_flx` weights atmospheric fluxes by `at_i_b`
(`iceupdate.F90:133,150-158`) but `qsr` by post-thermo `at_i` (`:150`).

---

## 4. legoESM reuse vs new

Base inventory from the prior agent (`packages/ice/legoesm/ice/`), with the
citations I re-verified this session marked ✓.

| SI3 requirement (ORCA1) | legoESM today | disposition |
|---|---|---|
| Prognostic state `a_i, v_i, v_s, e_i(nlay_i), e_s(nlay_s), sv_i, oa_i, t_su, u_ice, v_ice, stress1/2/12` per `icerst.F90:143-180` | `DynamicSeaIceState` ✓ `state.py:33-56`: `h_ice, T_ice, concentration, u_ice, v_ice, sigma_11/22/12, h_snow, S_ice, pond_*`; no layer enthalpies, no age | **NEW** layered `e_i/e_s`, `oa_i`; mapping `v=h·a` in a bridge (harness-only) |
| Grid: C-grid (u at U, v at V, `s1,s2` at T, `s12` at F; `icedyn_rhg_evp.F90` header "EVP-C-grid") | dynamics A-grid ✓ `rheology.py:238`, `dynamics.py:129,216`; C-grid support is transport-only, donor-cell ✓ `sea_ice.py:71-103`, `transport.py:182-211` | **NEW** C-grid strain/stress-divergence/momentum on `LatLonCGridGeometry`/beta-plane C-grid; A-grid path stays for other users |
| aEVP with landfast, `fimask` no-slip, ssh tilt | EVP/mEVP (Hunke-Dukowicz / Bouillon-Kimmritz) ✓ `rheology.py:436,533`, `dynamics.py:377,599`; `N_evp=120`, `T_evp=0.36` ✓ `config.py:356-361`; no aEVP per-cell α/β, no landfast (grep zero), no tilt term | **EXTEND** rheology with aEVP α/β (`icedyn_rhg_evp.F90:446,468,476`) as a selectable scheme; **NEW** landfast; tilt term needed for 3.6+ |
| H79 strength (`P*=2e4, C=20`) | `ice_strength` ✓ `rheology.py:178`; `P_star=2.75e4` default ✓ `config.py:358` | reuse; card pins ORCA1 values |
| Ridging + rafting, exp participation/redistribution | `ridging.py:81,116,338` ✓ (participation, column kernel, apply); `RidgingConfig` ✓ `config.py:224-249` (`e_star=0.36, mu_rdg=4, H_star=100`); rafting absent ✓ (grep) | **EXTEND** to SI3 formulas (`a*=0.03`, `μ=3`, `H*=25`, rafting tanh, porosity, snow loss 0.5) with the iteration loop `icedyn_rdgrft.F90:289-335` |
| Prather moments transport (+ restart of 5 moments × tracer) | PPM / donor-cell ✓ `transport.py:4,182,278`; no moments (grep zero) | **NEW** |
| `ice_cor` (thin-ice rescale, `amax` cap, zapsmall) | `_cap_multicat_concentration` `sea_ice.py:1416` (not re-read) | **PLAUSIBLE partial**; needs term-by-term match to `icecor.F90:67-114` |
| BL99 3 ice + 3 snow layers, P07 conductivity, `np_cnd_OFF` surface BC | `_future/bitz_lipscomb.py` ✓ (`thomas_solve`, U64 `k=k0+βS/T` with `beta_ice_cond=0.13` `constants.py:69`; `NOT YET INTEGRATED`); active thermo is zero-layer ✓ `sea_ice.py:988,1694` | **NEW** scheme (promote from `_future`, add snow layers, P07, Picard iteration with `dqns_ice` linearisation `icethd_zdf_bl99.F90:377`, SSL transmission) behind a `thermo_scheme` selector |
| `ice_thd_dh` sequence (precip → sublimation → surface melt → bottom growth/melt → snow-ice) | `snow.py` ✓ (`accumulate_snowfall:131`, `consume_*:176,254`, `snow_ice_flooding:316`); zero-layer melt in `_thermo_v2` | **NEW** ordered enthalpy-conserving sequence; snow-ice Archimedes rule reusable (`icethd_dh.F90:446` vs `snow.py:316`) |
| Salinity option 2 (bulk S, flushing/drainage restoring, `sfx_bri`, linear profile) | bulk salinity + salt budget ✓ `brine.py:67,268`; `BrineConfig` ✓ `config.py:203-221` | **EXTEND**: restoring terms and clamp `icethd_sal.F90:221-237`; SI3 option 2 is itself bulk (`s_i_1d`) with a diagnostic profile — smaller gap than the matrix's "layered" wording implies |
| Open-water growth (`qlead` → new ice at `rn_hinew`, `rn_sinew·sss`, `amax` residual) | `open_water_freeze` case exists in matrix (`run_sea_ice_test_matrix.py:1549`); formula not compared | **PLAUSIBLE partial**; match `icethd_do.F90:172-269` |
| Lead/bottom heat (`qsb_ice_bot`, `zch=0.0057`, `zqld/zqfr`, `fhld`) | `ocean_heat_transfer_coeff=20 W/m²/K` ✓ `config.py:324` (linear, not friction-velocity) | **NEW** `ice_flx_other` (`icesbc.F90:321-430`) |
| Albedo (`icealb.F90` log/exp blends, cloud mix) | delta-Eddington / MU / constant ✓ `config.py:384-390` | **NEW** `si3` albedo arm (small) |
| Bulk ice fluxes (`blk_ice_1/2`, constant Cd/Ch/Ce, `q_sat` over ice, LW linearisation) | `Cd_ice=Ch_ice=1.5e-3` ✓ `config.py:307-308`, `bulk_scheme="constant"|"most"` ✓ `:341` | **EXTEND** with SI3 formulas `sbcblk.F90:1219-1317` |
| Single category, HFN bounds (`hi_max(1)=99` for jpl=1) | `n_categories=1` ✓ `config.py:367`; `itd.py` remaps (`lipscomb_2001_remap:640`) unused at jpl=1 | reuse; ITD remap out of scope for ORCA1 |
| File init `ht_i/ht_s/at_i/sm_i/tmsu` | `nemo_native_fields.py:211-218` ✓ hard-codes exactly those ORCA1 names (`_ICE_INIT_REQUIRED/_OPTIONAL`) | reuse for ORCA1; **cannot read testcase files** (`hti/hts/ati/smi/tmi/tsu`, `make_INITICE.py`) — add name mapping in harness |
| Ocean exchange (§5) | `coupler/ocean_forcing.py` ✓ (`tau = -(f_ice·ocean_stress)`, heat/fresh-water channels `:37,89-90,221-249`); `_ocean_exchange_from_diag` ✓ `sea_ice.py:1379` | **NEW** SI3-exact card; existing path stays default |
| Constants | `constants.py:35-69` vs `phycst.F90:39-66`: `c_pi 2106` vs `rcpi 2096.7`; `L_f 3.337e5` vs `rLfus 3.333601e5`; `k_ice 2.04` vs `rcnd_i 2.034396`; `rho_ocean 1025` vs `rho0 1026` (`rho_ocean_nemo` exists `:49`); `albedo_ocean 0.06` vs `ralb_oce 0.066`; `beta_ice_cond 0.13` (U64) vs P07 `0.09/0.011` | **`ConstantsConfig` rows** on the SI3 card (per CLAUDE.md oracle rule: constants are config, never monkey-patch) |
| Tripole EVP | driver substitutes free drift ✓ `run_omip_core2.py:6746-6754` | resolved by the C-grid dynamics row; lane 4 composes the fold |

Net: reusable as-is — grid factories, constants plumbing, tridiagonal solver,
NCAR bulk, SI3 ice-IC loader, config/param-spec machinery, single-category
state. Everything the matrix lists as ABSENT is genuinely new; the two
"PARTIAL" items (BL99 core, ridging) are extensions, not rewrites.

---

## 5. Coupling surface (levitating `nn_ice=2`)

**What SI3 reads** (all nn_fsbc means from `sbcssm.F90:141-170`, instantaneous
when `nn_fsbc=1` `:73-92`): `ssu_m, ssv_m` (`icestp.F90:131-132`), `sss_m` →
`t_bo` (`:136`), `sst_m, e3t_m, frq_m` (`icesbc.F90:358-364`; `frq_m` =
fraction of solar absorbed in level 1, `sbcssm.F90:92`), `ssh_m` (tilt,
`icedyn_rhg_evp.F90:268`), `uu/vv(:,:,1,Kbb)` (`icestp.F90:233`).  Atmosphere
side via `sbc_blk` → `blk_ice_1/2`: `utau_ice, vtau_ice, qsr_ice, qns_ice,
dqns_ice, evap_ice, devap_ice, qemp_ice/oce, qprec_ice, emp_ice/oce,
qns_tot/qsr_tot, qtr_ice_top` (`sbc_ice.F90:42-73`).

**What SI3 writes to the ocean** (`ice_update_flx`, `iceupdate.F90:63-317`,
once per ice step):

| ocean field | formula | line |
|---|---|---|
| `qt_atm_oi` | `qns_tot + qsr_tot` | :111 |
| `zqsr` | `qsr_tot − Σ a_i_b (qsr_ice − qtr_ice_bot)` | :133 |
| `qt_oce_ai` | `qt_atm_oi − hfx_sum − hfx_bom − hfx_bog − hfx_dif − hfx_opw − hfx_snw + hfx_thd + hfx_dyn + hfx_res + hfx_sub − Σ qevap_ice a_i_b + hfx_spr` | :139-142 |
| `qsr` | if `fhld>0 & at_i>0`: `(1−at_i_b) qsr_oce (1−frq_m) + Σ a_i_b qtr_ice_bot (1−frq_m)`; else `zqsr` | :150-158 |
| `qns` | `qt_oce_ai − qsr` | :161 |
| `emp` | `emp_oce − wfx_ice − wfx_snw − wfx_pnd − wfx_err_sub` | :178 |
| `sfx` | `sfx_bog+bom+sum+sni+opw+res+dyn+bri+sub+lam` | :182-183 |
| `snwice_mass(_b)`, `snwice_fmass` | `rhos v_s + rhoi v_i (+ ponds)` | :189-194 |
| `fr_i`, `tn_ice` | `at_i`, `t_su` | :198-202 |

**Stress** (`ice_update_tau`, `:320-411`): once per ice step `tmod_io =
rho0·drag_io·|U_ice−U_oce|` at T (`:369`), `taum` blend (`:379`); every ocean
step `utau = (1−at_i)·utau_oce + at_i·½ tmod_io (u_ice + u_ice(i−1) −
zflagi(u_oce+u_oce(i−1)))` (`:400-406`) with `zflagi=0` under
`ln_drgice_imp=T` (`:388-396`), i.e. the ocean-velocity part of the ice-ocean
drag is applied implicitly through `rCdU_ice = −tmod_io·at_i/rho0` (`:391`)
in the vertical momentum solve (`zdfdrg.F90:117-126`; `dynzdf.F90:161,298,472`).

**Levitating**: `ln_ice_embd=.FALSE.` (ocean cfg :115) ⇒ no ice-mass
pressure on the ocean (`dynspg.F90:146-155`, `stp2d.F90:211-222` are inside
`IF(ln_ice_embd)`), and the EVP tilt uses bare `ssh_m` (`icevar.F90:1036`,
ELSE branch).  Ice mass still enters the **global freshwater budget** because
`nn_fwb_voltype=1` (ref :666) subtracts `snwice_fmass` from the corrected
`emp` (`sbcfwb.F90:237`).  Halo/fold: `lbc_lnk` on `utau/vtau/emp/qns` after
runoff (`sbcmod.F90:533-537`).

**What this means for the legoESM card**: (i) the ice step must consume
nn_fsbc-averaged surface fields and run every 4th ocean step; (ii) the four
ocean inputs are exactly `qsr, qns, emp, sfx` at T plus `utau/vtau` at T
(then ocean-side interpolation), plus `rCdU_ice` for the implicit top drag;
(iii) `qns` is a residual of a heat-conserving total, so the card's ice heat
ledger must reproduce every `hfx_*` bucket, not just the net; (iv) ice mass
is a diagnostic for FWB only; (v) legoESM's existing exchange
(`ocean_forcing.py:37,89-90`) uses `tau = −f_ice·ocean_stress` with the
ice-side concentration weighting — sign/weighting conventions must be pinned
against `:400-406` before any coupled receipt.  Which RK3 stage sees these: all
three, via `tra_sbc_RK3` (`stprk3_stg.F90:521`) and the barotropic RHS; lane 2
owns that placement.

---

## 6. Risks and unknowns, ranked

1. **C-grid dynamics is a new dycore for the ice package** (A-grid EVP cannot be
   bridged; strain rates at T/F and momentum at U/V are structurally
   different).  Owns rungs 3.3-3.4 and the tripole free-drift substitution.
   Largest single item; must land as a selectable grid arm with the A-grid
   path untouched.
2. **aEVP, not EVP, is the ORCA1 arm** (`namelist_ice_ref:110` inherited).
   Coverage-matrix row was wrong; any card that ports classical EVP with
   `rn_relast` will never match.  Also the shipped ICE_RHEO sets
   `ln_aEVP=.false.` — overlay required.
3. **No shipped oracle forces BL99+salinity.**  ICE_* cases run
   `ln_icethd=F`; `C1D/EXP_SASICE` needs an off-tree ERA5 file and ships an
   empty ice cfg (ref defaults: 10+5 layers, `nn_icesal=4`, ponds on).  Rung
   3.5 requires either fetching that file (provenance to pin) or an analytic
   `usrdef_sbc` column (a Frankenstein-risk composition — must be declared).
4. **ICE_RHEO ships a stale `MY_SRC/icedyn_rhg_evp.F90`** (1226-line diff vs
   `src/ICE`: old `USE dom_oce`, `zrhoco`, no `par_ice`).  Building the case
   as shipped overrides the 5.0.2 EVP with an older one; the receipt must
   delete or replace that override and record the choice.
5. **Landfast needs bathymetry**: `rn_lf_depfra·hu/hv` (`icedyn_rhg_evp.F90:342,349`)
   — ICE_RHEO/ADV2D use `key_vco_1d` slab depth; the L16 term is effectively
   a constant-depth test there.  Real coverage of landfast only on ORCA
   topography (lane 4).
6. **Prather restart moments** (`sxice...sxysal`, `icedyn_adv_pra.F90:1383-1497`)
   are state.  A restart-based exact gate must carry 5 moments × 7+ tracers ×
   `jpl`; the ice restart inventory (Appendix A) is the coverage list.
7. **ICE_ADV2D README warns UMx needs `ll_neg=.FALSE.`** — irrelevant for
   Prather but signals the case is lightly maintained; expect build/run
   surprises (lane-1 hit the same class: `Text::Balanced`, `key_xios`).
8. **`rn_sinew=0.75` with `nn_icesal=2`** contradicts the ref's own comment
   (`namelist_ice_cfg:116-117`).  Target fact; replicate, but flag to the user
   as a possible configuration defect in the reference run.
9. **Ice-IC loader names** hard-code ORCA1's `ht_i/at_i/...`; testcases and
   the ref default use `hti/ati/...`.  Harness mapping needed; no production
   change.
10. **`frq_m` and `e3t_m` semantics** (`sbcssm.F90:90-92`) tie the lead-heat
    budget to the ocean's level-1 thickness and solar penetration — the
    coupled rung inherits lane 1's `tra_qsr` RGB debt.
11. **Constants drift** (Appendix B): five constants differ at 1e-3..1e-2
    relative; under an exact bar every one shows up.  `ConstantsConfig` rows,
    not edits to `constants.py`.
12. **PLAUSIBLE, unverified this session**: `ln_bdy=F` for ORCA1 (call 10);
    `pp_cldf` value used for `cloud_fra` in albedo; whether `key_si3_1D`
    changes arithmetic order (it only changes loop packing per
    `icethd.F90:119-136`, PLAUSIBLE no numerical effect).

---

## 7. Corrections to prior agent reports (Rule 11)

| claim (prior report) | status | evidence |
|---|---|---|
| "Prather moments are NOT in the restart" | **RETRACTED** | `icedyn_adv_pra.F90:487` `IF(lrst_ice) CALL adv_pra_rst('WRITE',kt)` → `:1383-1497` `iom_rstput` of all moment arrays; the earlier grep looked only in `icerst.F90` |
| "SI3 init file uses `hti/hts/ati`, NOT `at_i/ht_i`" | **PARTLY WRONG** | the variable name is the 4th column of `sn_*` (`iceistate.F90:462-479`); ORCA1 pins `ht_i/ht_s/at_i/sm_i/tmsu` (`namelist_ice_cfg:183-189`), the ref and the testcases pin `hti/hts/ati/smi/tmi/tsu`.  legoESM's loader matches ORCA1, not the testcases (§4, risk 9) |
| "ln_ice_embd default not confirmed" | **RESOLVED** | explicit `.FALSE.` at ocean `namelist_cfg:115` |
| coverage matrix: "EVP; 100 EVP subcycles and relaxation .333 (ref)" | **INCOMPLETE** | `ln_aEVP=.true.` also inherited (`namelist_ice_ref:110`); `rn_relast` inert (`icedyn_rhg_evp.F90:236-247`) |
| "nn_qtrice lives in namthd_zdf" (task text) | confirmed correction | `icesbc.F90:453` (`namsbc`) |
| this session's earlier relative line numbers for `ice_flx_other` | superseded | true lines: `zch` :321, `qsb_ice_bot` :371/:377, `zqld` :358, `zqfr` :364, `qlead/fhld` :391-404, landfast stop :413, `ln_leadhfx` :421/:427 |

---

## 8. Decisions needed before dispatch (numbered; my pick marked)

1. Rung 3.5 oracle forcing: (a) fetch the ERA5 `EXP_SASICE` station file and
   pin its hash, or (b) analytic `usrdef_sbc` column declared as an
   experimental composition.  **Pick (a)** — real reference, no Frankenstein.
2. Landfast in ICE_RHEO: (a) overlay `ln_landfast_L16=T` on the slab depth
   (constant-depth test), or (b) leave off and certify landfast only in lane
   4.  **Pick (b)**; record L16 as UNVERIFIED-deferred.
3. ICE_RHEO stale `MY_SRC/icedyn_rhg_evp.F90`: (a) delete the override (build
   against 5.0.2 `src/ICE`), or (b) keep as shipped.  **Pick (a)**; hash both.
4. Which legoESM grid hosts the C-grid ice dynamics first: (a) lane-1 beta-plane
   C-grid (`create_beta_plane_cgrid_geometry`) then tripole, or (b) tripole
   first.  **Pick (a)** — matches oracle cases exactly.
5. Ordering: 3.1→3.2→3.3→3.4 (dynamics) before 3.5 (thermo), or thermo in
   parallel.  **Pick parallel**: 3.5 is independent of the C-grid work and is
   the XL item.

No production default changes are proposed by this dossier.

---

## Appendix A — coverage seeds (Rule 1)

**Ice restart** (`icerst.F90:137-180` + module writers), the enumeration a
lane-3 restart gate must dispose 1:1: `nn_fsbc, kt_ice, v_i, v_s, a_i, t_su,
u_ice, v_ice, oa_i, a_ip, v_ip, v_il, t_s(l), e_s(l), sv_i, e_i(l) [szv_i(l)
if nn_icesal=4], cnd_ice, t1_ice` (`:137-180`); `stress1_i, stress2_i,
stress12_i` (`icedyn_rhg_evp.F90:1110-1112`); `snwice_mass, snwice_mass_b`
(`iceupdate.F90:479-480`); Prather `sx/sy/sxx/syy/sxy` × {ice, sn, a, age,
c0(l), e(l), sal [si(l), ap, vp, vl]} (`icedyn_adv_pra.F90:1397-1497`).
Ponds/`szv_i`/`sxsi` are WAIVED-INACTIVE for ORCA1.

**Ice↔ocean exchange arrays** (`sbc_ice.F90:42-105`): `qns_ice, qsr_ice,
qla_ice, dqla_ice, dqns_ice, tn_ice, alb_ice, qml_ice, qcn_ice, qtr_ice_top,
utau_ice, vtau_ice, emp_ice, evap_ice, devap_ice, qns_oce, qsr_oce, qemp_oce,
qemp_ice, qevap_ice, qprec_ice, emp_oce, wndm_ice, sstfrz, rCdU_ice,
snwice_mass, snwice_mass_b, snwice_fmass` (+ CICE-only `topmelt, botmelt,
ss_iou/v, ...` WAIVED).

**Ocean fields written by ice**: `utau, vtau, taum, qsr, qns, emp, sfx, fr_i,
tn_ice` (`icestp.F90:53-55` header; §5).

## Appendix B — constants to pin on the SI3 card

| quantity | legoESM `packages/core/legoesm/constants.py` | NEMO `phycst.F90` |
|---|---|---|
| ice specific heat | `c_pi = 2106.0` (:36) | `rcpi = 2096.7` (:62) |
| fusion | `L_f = 3.337e5` (:43) | `rLfus = 0.3333601e6` (:64) |
| sublimation | `L_s = 2.834e6` (:42) | `rLsub = 2.8344e6` (:63) |
| ice conductivity | `k_ice_default = 2.04` (:53) | `rcnd_i = 2.034396` (:60) |
| snow conductivity | `k_snow = 0.31` (:54) | `rn_cnd_s = 0.5` (ORCA1 cfg :92) |
| snow density | `rho_snow = 330` (:46) | `rhos = 330` (:57) |
| ice density | `rho_ice = 917` (:45) | `rhoi = 917` (:58) |
| ocean ref density | `rho_ocean = 1025` (:48); `rho_ocean_nemo = 1026` (:49) | `rho0 = 1026` |
| liquidus slope | `mu_ice_freeze = 0.054` (:68) | `rTmlt = 0.054` (:65) |
| lead albedo | `albedo_ocean = 0.06` (`config.py:304`) | `ralb_oce = 0.066` (:66) |
| brine conductivity | `beta_ice_cond = 0.13` U64 (:69) | P07 `0.09`, `0.011` (`icethd_zdf_bl99.F90:264-275`) |
| ice emissivity | `emissivity_ice = 0.97` (`config.py:305`) | `emiss_i` (sbcblk; not read this session — PLAUSIBLE 0.97) |
