"""Reviewed policy data for :mod:`full_step_coverage`.

The CALL list is deliberately not copied here: the gate parses the selected
DINO ``MY_SRC/stpmlf.F90`` and fails if a parsed call has no policy below.
This file contains only the configuration resolutions, one-level dispatcher
descent, waivers, legoESM joins, leverage ordering, and receipt joins.
"""

ORACLE_REL = "cfgs/DINO"
RUN_REL = "RUN_SEQDUMP_D180_1R"
EXPECTED_DIRECT_CALLS = 148
SOURCE_SHA256 = {
    "MY_SRC/stpmlf.F90": "26849c4fa34305317881959dd5a1e41af9b13aa4f96a290a998add955a73a283",
    "cpp_DINO.fcm": "f72d84ceee6ec63a2bdc65354ba13fbde58b6b5611962d482d49b49b12a963dc",
    f"{RUN_REL}/namelist_cfg": "0e75f4cb32dad3ed3fc17b721651ec12e0366c2978df8c65a6e95ae80b48d68b",
    f"{RUN_REL}/ocean.output": "66162e1d54cc0c8648b0c220fe902b6a75f7b19a0e8c2349f5d11928809c0557",
}

# Parser-UNRESOLVED guards, resolved against the selected run and source.
# ACTIVE here means the call executes on at least one timestep in the run.
RESOLVED = {
    (144, "iom_init"): ("ACTIVE", "first step: stpmlf.F90:143 IF(kstp == nit000)"),
    (145, "dia_mlr_iom_init"): (
        "INACTIVE",
        "key_xios absent sets l_diamlr=.FALSE.; WORK/diamlr.F90:47-56; cpp_DINO.fcm:1",
    ),
    (146, "iom_init_closedef"): ("ACTIVE", "first step: stpmlf.F90:143 IF(kstp == nit000)"),
    (147, "dia_hth_init"): ("ACTIVE", "first-step diagnostic initialisation; stpmlf.F90:143-147"),
    (148, "dia_ptr_init"): ("ACTIVE", "first-step diagnostic initialisation; stpmlf.F90:143-148"),
    (149, "dia_ar5_init"): ("ACTIVE", "first-step diagnostic initialisation; stpmlf.F90:143-149"),
    (150, "dia_hsb_init"): ("ACTIVE", "first-step diagnostic initialisation; stpmlf.F90:143-150"),
    (151, "dia_25h_init"): ("ACTIVE", "first-step diagnostic initialisation; stpmlf.F90:143-151"),
    (152, "mlf_dia"): ("ACTIVE", "first-step diagnostic flag resolution; stpmlf.F90:143-152"),
    (155, "iom_swap"): ("INACTIVE", "lwxios=F: key_xios absent; cpp_DINO.fcm:1"),
    (156, "iom_init_closedef"): ("INACTIVE", "lwxios=F: key_xios absent; cpp_DINO.fcm:1"),
    (157, "iom_setkt"): ("INACTIVE", "lwxios=F: key_xios absent; cpp_DINO.fcm:1"),
    (180, "day"): ("ACTIVE", "all non-initial steps: stpmlf.F90:180 IF(kstp /= nit000)"),
    (189, "isf_stp"): ("INACTIVE", "ln_isf=.false.; RUN_SEQDUMP_D180_1R/namelist_ref:520"),
    (197, "sto_par"): ("INACTIVE", "ln_sto_eos=.false.; RUN_SEQDUMP_D180_1R/namelist_ref:1564"),
    (198, "sto_pts"): ("INACTIVE", "ln_sto_eos=.false.; RUN_SEQDUMP_D180_1R/namelist_ref:1564"),
    (218, "eos"): ("ACTIVE", "l_ldfslp=T from np_lap_i; ln_traldf_iso=T at ocean.output:831"),
    (233, "ldf_slp"): (
        "ACTIVE",
        "l_ldfslp=T and triad arm excluded by ln_traldf_triad=F; ocean.output:832",
    ),
    (237, "ldf_tra"): ("ACTIVE", "time-varying aht/eiv: ocean.output:839,857"),
    (238, "ldf_dyn"): ("ACTIVE", "time-varying ahm: ocean.output:880"),
    (295, "dyn_asm_inc"): ("INACTIVE", "lk_asminc=F: key_asminc absent; cpp_DINO.fcm:1"),
    (296, "asm_bkg_wri"): ("INACTIVE", "ln_bkgwri=.false.; RUN_SEQDUMP_D180_1R/namelist_ref:1475"),
    (422, "diurnal_layers"): (
        "INACTIVE",
        "ln_diurnal=.false.; RUN_SEQDUMP_D180_1R/namelist_ref:1388",
    ),
    (428, "ldf_eke"): (
        "INACTIVE",
        "l_ldfeke initialized .FALSE.; MY_SRC/ldftra.F90:98; nn_aei_ijk_t=21, ocean.output:857",
    ),
    (440, "dia_hth"): (
        "INACTIVE",
        "l_hth is iom_use OR at WORK/diahth.F90:353-357; key_xios absent "
        "makes iom_use=.FALSE. at src/OCE/IOM/iom.F90:3031-3038; cpp_DINO.fcm:1",
    ),
    (442, "dia_ptr"): ("INACTIVE", "l_diaptr=F; ocean.output:1106"),
    (452, "dia_detide"): ("INACTIVE", "l_diadetide=F; ocean.output:1096"),
    (453, "dia_mlr"): (
        "INACTIVE",
        "key_xios absent forces l_diamlr=.FALSE.; src/OCE/DIA/diamlr.F90:47-56; cpp_DINO.fcm:1",
    ),
    (497, "tra_asm_inc"): ("INACTIVE", "lk_asminc=F: key_asminc absent; cpp_DINO.fcm:1"),
    (509, "tra_isf"): ("INACTIVE", "ln_isf=.false.; RUN_SEQDUMP_D180_1R/namelist_ref:520"),
    (627, "dia_hsb"): (
        "INACTIVE",
        "l_diahsb is iom_use OR at WORK/diahsb.F90:459-462; key_xios absent "
        "makes iom_use=.FALSE. at src/OCE/IOM/iom.F90:3031-3038; cpp_DINO.fcm:1",
    ),
    (634, "rst_write"): ("ACTIVE", "scheduled restart writes are present; ocean.output:1427"),
    (635, "sto_rst_write"): (
        "INACTIVE",
        "ln_sto_eos=.false.; RUN_SEQDUMP_D180_1R/namelist_ref:1564",
    ),
    (659, "dia_obs"): ("INACTIVE", "ln_diaobs=.false.; RUN_SEQDUMP_D180_1R/namelist_ref:1424"),
    (665, "iom_close"): ("ACTIVE", "first step: stpmlf.F90:664-665"),
    (666, "iom_context_finalize"): ("INACTIVE", "lrxios=F: key_xios absent; cpp_DINO.fcm:1"),
    (667, "FLUSH"): ("ACTIVE", "first step and lwm=T; stpmlf.F90:664-667"),
    (668, "FLUSH"): ("INACTIVE", "numoni=-1 without SI3; key_si3 absent; cpp_DINO.fcm:1"),
    (674, "sbc_cpl_snd"): ("INACTIVE", "lk_oasis=F: key_oasis3 absent; cpp_DINO.fcm:1"),
}

# Calls that execute but are bookkeeping, diagnostics, receipt probes, or I/O.
# A WAIVED call is covered by the registry denominator but makes no fidelity
# claim about a physical tendency.
WAIVED_NAMES = {
    "iom_init",
    "iom_init_closedef",
    "dia_hth_init",
    "dia_ptr_init",
    "dia_ar5_init",
    "dia_hsb_init",
    "dia_25h_init",
    "mlf_dia",
    "day",
    "iom_setkt",
    "stp_dump_krhs",
    "stp_dump_state_and_bt",
    "stp_dump_ts_krhs",
    "trddump_acc_baro",
    "trddump_acc_plant",
    "trddump_acc_state",
    "dia_ar5",
    "dia_wri",
    "rst_write",
    "stp_ctl",
    "iom_close",
    "FLUSH",
}

# Each target is (routine, concrete CALL file:line, resolving quote,
# ACTIVE|INACTIVE|WAIVED). Selectable calls which DINO excludes remain rows.
DISPATCH = {
    (190, "sbc"): [
        ("usrdef_sbc_oce", "WORK/sbcmod.F90:417", "ln_usr=T; ocean.output:688", "ACTIVE"),
        ("sbc_flx", "WORK/sbcmod.F90:418", "ln_flx=F; ocean.output:685", "INACTIVE"),
        ("sbc_blk", "WORK/sbcmod.F90:419", "ln_blk=F; ocean.output:687", "INACTIVE"),
        ("sbc_abl", "WORK/sbcmod.F90:420", "ln_abl=F; ocean.output:691", "INACTIVE"),
        ("sbc_cpl_rcv", "WORK/sbcmod.F90:421", "ln_cpl=F; ocean.output:694", "INACTIVE"),
        ("sbc_cpl_rcv(none)", "WORK/sbcmod.F90:423", "nsbc=jp_usr; ocean.output:688", "INACTIVE"),
        ("sbc_cpl_rcv(mixed)", "WORK/sbcmod.F90:426", "ln_mixcpl=F; ocean.output:695", "INACTIVE"),
    ],
    (210, "zdf_phy"): [
        ("zdf_sh2", "WORK/zdfphy.F90:268", "ln_zdftke=T; ocean.output:731", "ACTIVE"),
        ("zdf_drg(BOTTOM)", "WORK/zdfphy.F90:277", "unconditional", "ACTIVE"),
        (
            "zdf_drg(TOP)",
            "WORK/zdfphy.F90:278",
            "ln_isfcav=F and ln_drgice_imp=F; ocean.output:427,806",
            "INACTIVE",
        ),
        ("zdf_mxl", "WORK/zdfphy.F90:280", "unconditional", "ACTIVE"),
        ("zdf_ric", "WORK/zdfphy.F90:285", "ln_zdfric=F; ocean.output:730", "INACTIVE"),
        ("zdf_tke", "WORK/zdfphy.F90:286", "ln_zdftke=T; ocean.output:731", "ACTIVE"),
        ("zdf_gls", "WORK/zdfphy.F90:287", "ln_zdfgls=F; ocean.output:732", "INACTIVE"),
        ("zdf_osm", "WORK/zdfphy.F90:288", "ln_zdfosm=F; ocean.output:733", "INACTIVE"),
        ("zdf_evd", "WORK/zdfphy.F90:323", "ln_zdfevd=T; ocean.output:736", "ACTIVE"),
        ("zdf_ddm", "WORK/zdfphy.F90:327", "ln_zdfddm=F; ocean.output:742", "INACTIVE"),
        ("zdf_swm", "WORK/zdfphy.F90:335", "ln_zdfswm=F; ocean.output:746", "INACTIVE"),
        ("zdf_iwm", "WORK/zdfphy.F90:336", "ln_zdfiwm=F; ocean.output:747", "INACTIVE"),
        ("zdf_mxl_turb", "WORK/zdfphy.F90:338", "diagnostic; no active consumer", "WAIVED"),
        ("lbc_lnk(zdfphy)", "WORK/zdfphy.F90:344", "unconditional", "ACTIVE"),
        ("tke_rst", "WORK/zdfphy.F90:347", "lrst_oce schedule active; ocean.output:1427", "WAIVED"),
        ("gls_rst", "WORK/zdfphy.F90:348", "ln_zdfgls=F; ocean.output:732", "INACTIVE"),
        ("ric_rst", "WORK/zdfphy.F90:349", "ln_zdfric=F; ocean.output:730", "INACTIVE"),
    ],
    (309, "dyn_adv"): [
        ("dyn_keg", "MY_SRC/dynadv.F90:89", "ln_dynadv_vec=T; ocean.output:988", "ACTIVE"),
        ("dyn_zad", "MY_SRC/dynadv.F90:97", "ln_dynadv_vec=T; ocean.output:988", "ACTIVE"),
        ("dyn_adv_cen2", "MY_SRC/dynadv.F90:108", "ln_dynadv_vec=T; ocean.output:988", "INACTIVE"),
        ("dyn_adv_up3", "MY_SRC/dynadv.F90:111", "ln_dynadv_vec=T; ocean.output:988", "INACTIVE"),
    ],
    (315, "dyn_vor"): [
        ("vor_ens(ncor)", "MY_SRC/dynvor.F90:150", "ln_dynvor_ens=F; ocean.output:999", "INACTIVE"),
        (
            "vor_ene(ncor)",
            "MY_SRC/dynvor.F90:151",
            "ln_dynvor_ene=F; ocean.output:1000",
            "INACTIVE",
        ),
        (
            "vor_enT(ncor)",
            "MY_SRC/dynvor.F90:152",
            "ln_dynvor_enT=F; ocean.output:1001",
            "INACTIVE",
        ),
        (
            "vor_een(ncor)",
            "MY_SRC/dynvor.F90:153",
            "dyn_vor_3D; ln_dyn_trd=T and EEN=T; ocean.output:1075,1002",
            "ACTIVE",
        ),
        (
            "vor_enT(nrvm)",
            "MY_SRC/dynvor.F90:167",
            "ln_dynvor_enT=F; ocean.output:1001",
            "INACTIVE",
        ),
        (
            "vor_ene(nrvm)",
            "MY_SRC/dynvor.F90:168",
            "ln_dynvor_ene=F; ocean.output:1000",
            "INACTIVE",
        ),
        ("vor_ens(nrvm)", "MY_SRC/dynvor.F90:169", "ln_dynvor_ens=F; ocean.output:999", "INACTIVE"),
        (
            "vor_een(nrvm)",
            "MY_SRC/dynvor.F90:170",
            "dyn_vor_3D; ln_dyn_trd=T and EEN=T; ocean.output:1075,1002",
            "ACTIVE",
        ),
        ("vor_enT(ntot)", "MY_SRC/dynvor.F90:200", "l_trddyn=T; ocean.output:1075", "INACTIVE"),
        ("vor_ene(ntot)", "MY_SRC/dynvor.F90:207", "l_trddyn=T; ocean.output:1075", "INACTIVE"),
        ("vor_ens(ntot)", "MY_SRC/dynvor.F90:214", "l_trddyn=T; ocean.output:1075", "INACTIVE"),
        ("vor_ens(nrvm,mix)", "MY_SRC/dynvor.F90:223", "l_trddyn=T; ocean.output:1075", "INACTIVE"),
        ("vor_ene(ncor,mix)", "MY_SRC/dynvor.F90:225", "l_trddyn=T; ocean.output:1075", "INACTIVE"),
        ("vor_een(ntot)", "MY_SRC/dynvor.F90:229", "l_trddyn=T; ocean.output:1075", "INACTIVE"),
    ],
    (319, "dyn_ldf"): [
        ("dynldf_lev_lap", "MY_SRC/dynldf.F90:83", "lap=T, lev=T; ocean.output:873,876", "ACTIVE"),
        ("dyn_ldf_iso", "MY_SRC/dynldf.F90:85", "ln_dynldf_iso=F; ocean.output:878", "INACTIVE"),
        ("dynldf_lev_blp", "MY_SRC/dynldf.F90:88", "ln_dynldf_blp=F; ocean.output:874", "INACTIVE"),
    ],
    (324, "dyn_hpg"): [
        ("hpg_zco", "MY_SRC/dynhpg.F90:118", "ln_hpg_zco=F; ocean.output:1023", "INACTIVE"),
        ("hpg_sco", "MY_SRC/dynhpg.F90:119", "ln_hpg_sco=T; ocean.output:1024", "ACTIVE"),
        ("hpg_djc", "MY_SRC/dynhpg.F90:120", "ln_hpg_djc=F; ocean.output:1026", "INACTIVE"),
        ("hpg_prj", "MY_SRC/dynhpg.F90:121", "ln_hpg_prj=F; ocean.output:1027", "INACTIVE"),
        ("hpg_isf", "MY_SRC/dynhpg.F90:122", "ln_hpg_isf=F; ocean.output:1025", "INACTIVE"),
    ],
    (332, "dyn_spg"): [
        ("dyn_spg_exp", "WORK/dynspg.F90:180", "ln_dynspg_exp=F; ocean.output:1035", "INACTIVE"),
        ("dyn_spg_ts", "WORK/dynspg.F90:181", "ln_dynspg_ts=T; ocean.output:1036", "ACTIVE"),
    ],
    (505, "tra_qsr"): [
        ("qsr_RGBc", "WORK/traqsr.F90:175", "ln_qsr_rgb=F; ocean.output:902", "INACTIVE"),
        ("qsr_RGB", "WORK/traqsr.F90:177", "ln_qsr_rgb=F; ocean.output:902", "INACTIVE"),
        ("qsr_2BD", "WORK/traqsr.F90:179", "ln_qsr_2bd=T; ocean.output:903", "ACTIVE"),
        ("qsr_5BDc", "WORK/traqsr.F90:181", "ln_qsr_5bd=F; ocean.output:904", "INACTIVE"),
        ("qsr_5BD", "WORK/traqsr.F90:183", "ln_qsr_5bd=F; ocean.output:904", "INACTIVE"),
    ],
    (528, "tra_adv"): [
        (
            "ldf_eiv_trp_MLF",
            "WORK/traadv.F90:344",
            "MLF generic; ln_ldfeiv=T, triad=F; ocean.output:855,832",
            "ACTIVE",
        ),
        ("tra_adv_cen", "WORK/traadv.F90:358", "ln_traadv_cen=F; ocean.output:947", "INACTIVE"),
        ("tra_adv_fct", "WORK/traadv.F90:361", "ln_traadv_fct=T; ocean.output:950", "ACTIVE"),
        (
            "tra_adv_cen(FCT fallback)",
            "WORK/traadv.F90:364",
            "ll_dofct=.TRUE.; WORK/traadv.F90:278",
            "INACTIVE",
        ),
        ("tra_adv_mus", "WORK/traadv.F90:367", "ln_traadv_mus=F; ocean.output:954", "INACTIVE"),
        ("tra_adv_ubs", "WORK/traadv.F90:369", "ln_traadv_ubs=F; ocean.output:956", "INACTIVE"),
        ("tra_adv_qck", "WORK/traadv.F90:371", "ln_traadv_qck=F; ocean.output:958", "INACTIVE"),
    ],
    (548, "tra_ldf"): [
        ("traldf_lev_lap", "WORK/traldf.F90:93", "ln_traldf_iso=T; ocean.output:831", "INACTIVE"),
        ("traldf_iso_lap", "WORK/traldf.F90:95", "lap=T, iso=T; ocean.output:826,831", "ACTIVE"),
        (
            "traldf_triad_lap",
            "WORK/traldf.F90:97",
            "ln_traldf_triad=F; ocean.output:832",
            "INACTIVE",
        ),
        ("traldf_lev_blp", "WORK/traldf.F90:101", "ln_traldf_blp=F; ocean.output:827", "INACTIVE"),
        ("traldf_iso_blp", "WORK/traldf.F90:103", "ln_traldf_blp=F; ocean.output:827", "INACTIVE"),
        (
            "traldf_triad_blp",
            "WORK/traldf.F90:105",
            "blp=F, triad=F; ocean.output:827,832",
            "INACTIVE",
        ),
    ],
    (551, "tra_zdf"): [
        ("tra_zdf_imp", "WORK/trazdf.F90:82", "unconditional MLF implicit solve", "ACTIVE"),
    ],
    (275, "wzv"): [
        (
            "wzv_MLF",
            "MY_SRC/sshwzv.F90:168",
            "MLF generic; key_RK3 absent, cpp_DINO.fcm:1",
            "ACTIVE",
        )
    ],
    (412, "wzv"): [
        (
            "wzv_MLF",
            "MY_SRC/sshwzv.F90:168",
            "MLF generic; key_RK3 absent, cpp_DINO.fcm:1",
            "ACTIVE",
        )
    ],
    (276, "wAimp"): [
        ("wAimp_MLF", "MY_SRC/sshwzv.F90:555", "ln_zad_Aimp=F; ocean.output:727", "INACTIVE")
    ],
    (417, "wAimp"): [
        ("wAimp_MLF", "MY_SRC/sshwzv.F90:555", "ln_zad_Aimp=F; ocean.output:727", "INACTIVE")
    ],
}

LEGO = {
    "usrdef_sbc_oce": "MISSING",
    "eos_rab": "legoesm.ocean.eos.nemo_seos_alpha_beta",
    "bn2": "legoesm.ocean.eos.compute_buoyancy_frequency_nemo_bn2",
    "zdf_sh2": "legoesm.ocean.physics.vertical_mixing.tke.tke_vertical_mixing",
    "zdf_drg(BOTTOM)": "legoesm.ocean.dynamics.ocean_pe_latlon_cgrid.nemo_bottom_drag_rate_faces",
    "zdf_mxl": "legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid._nemo_mld",
    "zdf_tke": "legoesm.ocean.physics.vertical_mixing.tke.tke_vertical_mixing",
    "zdf_evd": "legoesm.ocean.physics.vertical_mixing.k_profiles._enhanced_diffusion_K",
    "zdf_mxl_turb": "MISSING",
    "lbc_lnk(zdfphy)": "MISSING",
    "eos": "legoesm.ocean.eos.compute_ocean_rho_and_pressure",
    "ldf_slp": (
        "legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid.compute_nemo_native_slopes"
    ),
    "ldf_tra": "MISSING",
    "ldf_dyn": "MISSING",
    "ssh_nxt": "MISSING",
    "dom_qco_r3c": "MISSING",
    "wzv_MLF": "MISSING",
    "dyn_keg": "MISSING",
    "dyn_zad": "MISSING",
    "vor_een(ncor)": "MISSING",
    "vor_een(nrvm)": "MISSING",
    "dynldf_lev_lap": "MISSING",
    "hpg_sco": "MISSING",
    "dyn_spg_ts": "MISSING",
    "div_hor": "MISSING",
    "dyn_zdf": "MISSING",
    "ssh_atf": "MISSING",
    "tra_sbc": "MISSING",
    "qsr_2BD": "legoesm.ocean.physics.shortwave_penetration.shortwave_penetration_tendency",
    "ldf_eiv_trp_MLF": (
        "legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid.nemo_eiv_bolus_transport"
    ),
    "tra_adv_fct": "legoesm.ocean.advection.fct_tracer_advection",
    "traldf_iso_lap": (
        "legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid."
        "nemo_iso_lap_tracer_tendency_latlon_cgrid"
    ),
    "tra_zdf_imp": "MISSING",
    "mlf_baro_corr": "MISSING",
    "finalize_lbc": "MISSING",
    "tra_atf_qco": "MISSING",
    "dyn_atf_qco": "MISSING",
}

RECEIPT_ARTIFACT = "docs/ocean/fidelity/dino_full_step_zdf_receipts.json"
RECEIPT_ARTIFACT_SHA256 = "2af4004e30936c608259c8c133988e8b4b76cf50249e62a59e524d49137bf0b2"
RECEIPTS = {
    (204, "eos_rab"): "eos_rab_before",
    (206, "bn2"): "bn2_before",
    (207, "bn2"): "bn2_now",
    (210, "zdf_sh2"): "zdf_sh2",
    (210, "zdf_drg(BOTTOM)"): "bottom_drag_coefficient",
    (210, "zdf_mxl"): "native_mld",
}

SOURCE_ASSERTIONS = {
    "eos_rab_before": [
        ("rows.1_eos_rab_before.disposition", "VERIFIED"),
        ("rows.1_eos_rab_before.alpha.max_column_error", 3.0874669417540936e-16),
        ("rows.1_eos_rab_before.alpha.bar", 1e-15),
        ("rows.1_eos_rab_before.alpha.pass", True),
        ("rows.1_eos_rab_before.beta.max_column_error", 0.0),
        ("rows.1_eos_rab_before.beta.bar", 1e-15),
        ("rows.1_eos_rab_before.beta.pass", True),
    ],
    "bn2_before": [
        ("rows.2_bn2_before.disposition", "VERIFIED"),
        ("rows.2_bn2_before.output.max_column_error", 5.968673256056548e-16),
        ("rows.2_bn2_before.output.bar", 1e-15),
        ("rows.2_bn2_before.output.pass", True),
    ],
    "bn2_now": [
        ("rows.3_eos_rab_bn2_now.disposition", "VERIFIED"),
        ("rows.3_eos_rab_bn2_now.output.max_column_error", 5.968545325803916e-16),
        ("rows.3_eos_rab_bn2_now.output.bar", 1e-15),
        ("rows.3_eos_rab_bn2_now.output.pass", True),
    ],
    "zdf_sh2": [
        ("rows.4_zdf_sh2.disposition", "VERIFIED"),
        ("rows.4_zdf_sh2.output.max_column_error", 0.0),
        ("rows.4_zdf_sh2.output.bar", 1e-15),
        ("rows.4_zdf_sh2.output.pass", True),
    ],
    "bottom_drag_coefficient": [
        ("rows.5_bottom_drag_coefficient.disposition", "VERIFIED"),
        ("rows.5_bottom_drag_coefficient.output.max_column_error", 0.0),
        ("rows.5_bottom_drag_coefficient.output.bar", 1e-15),
        ("rows.5_bottom_drag_coefficient.output.pass", True),
    ],
    "native_mld": [
        ("rows.6_native_mld_index.disposition", "VERIFIED"),
        ("rows.6_native_mld_index.output.max_column_error", 0.0),
        ("rows.6_native_mld_index.output.bar", 1e-12),
        ("rows.6_native_mld_index.output.pass", True),
        ("rows.7_native_mld_depth.disposition", "VERIFIED"),
        ("rows.7_native_mld_depth.output.max_column_error", 5.67046062233333e-16),
        ("rows.7_native_mld_depth.output.bar", 1e-15),
        ("rows.7_native_mld_depth.output.pass", True),
    ],
}

# Lower number is higher plausible climate leverage. This is explicitly a
# planning judgment, never a measured fidelity score.
LEVERAGE = {
    "eos_rab": (
        0,
        "NOW-level expansion/contraction coefficients feed buoyancy and all downstream mixing",
    ),
    "dyn_spg_ts": (1, "global barotropic transport and free-surface update"),
    "tra_adv_fct": (2, "moves heat and salt in all three dimensions"),
    "zdf_tke": (
        3,
        "sets vertical heat/momentum exchange; first ZDF divergence is inside this call",
    ),
    "dyn_zdf": (4, "implicit vertical momentum solve and drag coupling"),
    "tra_zdf_imp": (5, "implicit vertical heat/salt solve"),
    "hpg_sco": (6, "basin-scale pressure-gradient acceleration"),
    "vor_een(ncor)": (7, "planetary-vorticity momentum tendency"),
    "vor_een(nrvm)": (7, "relative-vorticity momentum tendency"),
    "dyn_zad": (8, "vertical momentum transport"),
    "dyn_keg": (9, "horizontal kinetic-energy-gradient tendency"),
    "wzv_MLF": (10, "common vertical velocity input to momentum and tracer advection"),
    "traldf_iso_lap": (11, "isoneutral heat/salt redistribution"),
    "ldf_eiv_trp_MLF": (12, "GM overturning transport"),
    "dynldf_lev_lap": (13, "lateral momentum dissipation"),
    "ldf_slp": (14, "shared slope input to Redi and GM"),
    "ldf_tra": (15, "time-varying Redi/GM coefficients"),
    "ldf_dyn": (16, "time-varying momentum viscosity"),
    "qsr_2BD": (17, "vertical distribution of solar heat"),
    "tra_sbc": (18, "surface heat/freshwater tracer tendency"),
    "usrdef_sbc_oce": (19, "surface momentum, heat, and freshwater forcing"),
    "ssh_nxt": (20, "free-surface continuity update"),
    "div_hor": (21, "depth-integrated continuity"),
    "dom_qco_r3c": (22, "z-star thickness ratios feed most transport operators"),
    "eos": (23, "density input to slopes and pressure gradient"),
    "zdf_evd": (24, "convective enhancement of vertical diffusivity"),
    "lbc_lnk(zdfphy)": (25, "halo values of vertical viscosity feed implicit solve"),
    "mlf_baro_corr": (26, "reconciles 3-D and barotropic depth means"),
    "finalize_lbc": (27, "commits boundary/halo state before filtering"),
    "tra_atf_qco": (28, "filters tracer content carried to next step"),
    "dyn_atf_qco": (29, "filters momentum carried to next step"),
    "ssh_atf": (30, "filters free surface carried to next step"),
    "zdf_mxl_turb": (31, "diagnostic turbocline depth; no active physics consumer found"),
}
