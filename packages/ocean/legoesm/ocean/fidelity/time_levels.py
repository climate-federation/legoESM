"""Which NEMO TIME LEVEL each oracle dump belongs to (#1226).

NEMO's modified-leapfrog step carries three time levels — ``Nbb`` (before),
``Nnn`` (now), ``Naa`` (after) — and a routine is frequently called with T/S at
ONE level and geometry at ANOTHER::

    CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nnn )   ! stpmlf.F90:184
    CALL eos_rab( ts(:,:,:,:,Nnn), rab_n, Nnn )   ! stpmlf.F90:185

Comparing a dump against the WRONG level silently substitutes
``|T_now - T_before|`` for "error".  That difference is largest in the
thermocline, so it does not look like noise — it looks like a beautifully
depth-structured defect.  It cost this campaign a full false diagnosis: a
7074-cell "tail", three 1.8% alpha "outliers" and a "levels 6-8 structure",
all of which evaporated at the correct level.

The mistake is easy precisely because the state LOOKS right — same field, same
shape, same units, plausible magnitude.  So the correct level is recorded here
ONCE, next to its NEMO source line, and :func:`select_ts` makes picking it the
default action rather than a thing to remember.

HARNESS glue: it selects comparison inputs and changes no answer.
"""
from __future__ import annotations

__all__ = ["TimeLevel", "time_level_for_dump", "select_ts", "register_dump"]

TimeLevel = str  # "before" | "now" | "after"

_VALID: frozenset[str] = frozenset({"before", "now", "after"})

# dump basename -> (time level of its T/S, NEMO source line that proves it).
# The GEOMETRY level is separate and usually Nnn — see the module docstring.
_DUMP_TIME_LEVEL: dict[str, tuple[TimeLevel, str]] = {
    # eos_rab / bn2 family: T/S at Nbb, geometry at Nnn.
    "dump_alpha_b.bin": ("before", "stpmlf.F90:184 eos_rab(ts(...,Nbb), rab_b, Nnn)"),
    "dump_beta_b.bin": ("before", "stpmlf.F90:184 eos_rab(ts(...,Nbb), rab_b, Nnn)"),
    "tke_dump_rn2b.bin": ("before", "rn2b = bn2(ts(...,Nbb)); zdfmxl.F90:98 integrates rn2b"),
    # #1226 item-11 Prandtl-stage instrumentation (zdftke.F90:206-244, the
    # SAME "IF(kt==nit000.AND.nn_pdl==1)" dump block as tke_dump_rn2b.bin
    # above): sh2/avm_in are the INPUT p_sh2/p_avm arrays tke_tke receives
    # (dumped verbatim at :218-219, not recomputed), and pdlr/zri are the
    # Prandtl-branch outputs computed FROM rn2b (zdftke.F90:462-478 for pdlr
    # itself; the dump's own recomputation at :229-238 for zri) -- all four
    # are governed by the SAME before-level rn2b as tke_dump_rn2b.bin, so
    # they are registered "before" for the identical reason.
    "tke_dump_sh2.bin": ("before", "zdftke.F90:218 WRITE(8845) p_sh2 -- the "
                                    "shear-production INPUT to the rn2b-"
                                    "governed Prandtl branch, same dump block "
                                    "as tke_dump_rn2b.bin"),
    "tke_dump_avm_in.bin": ("before", "zdftke.F90:219 WRITE(8848) p_avm -- "
                                       "the INPUT avm (pre tke_avn update), "
                                       "same dump block as tke_dump_rn2b.bin"),
    "tke_dump_pdlr.bin": ("before", "zdftke.F90:223/477 p_pdlr computed from "
                                     "rn2b (before) in tke_tke's nn_pdl==1 "
                                     "branch, zdftke.F90:462-478"),
    "tke_dump_zri.bin": ("before", "zdftke.F90:229-238 zzri recomputed "
                                    "verbatim from rn2b/p_avm/p_sh2 (before), "
                                    "the SAME formula as tke_tke:462-474"),
    "eiv_dump_rn2b.bin": ("before", "same rn2b; NOTE only 35 levels — missing the deepest interface"),
    # zdf_mxl outputs are computed FROM rn2b, hence also before-level inputs.
    "dump_nmln.bin": ("before", "zdfmxl.F90:96-101, integrand is rn2b (before)"),
    "dump_hmlp.bin": ("before", "zdfmxl.F90:104, gdepw(nmln) from the rn2b integral"),
    # ldf_slp slopes are built on the before-level density field.
    "eiv_dump_uslp.bin": ("before", "ldfslp.F90 uses rn2b/rab_b"),
    "eiv_dump_vslp.bin": ("before", "ldfslp.F90 uses rn2b/rab_b"),
    "eiv_dump_wslpi.bin": ("before", "ldfslp.F90 uses rn2b/rab_b"),
    "eiv_dump_wslpj.bin": ("before", "ldfslp.F90 uses rn2b/rab_b"),
    # RAW pre-Shapiro w-point slopes (zwz/zww captured at ldfslp.F90:335-336,
    # written to units 8840/8841 at :378-379).  These split ldf_slp into the
    # slope FORMULA (stage A) and the 16-point smoother (stage B) -- internals,
    # not endpoints.
    "eiv_dump_zwz_raw.bin": ("before", "ldfslp.F90:335 zwz pre-Shapiro (unit 8840)"),
    "eiv_dump_zww_raw.bin": ("before", "ldfslp.F90:336 zww pre-Shapiro (unit 8841)"),
    # FULL j-direction intermediate chain (units 8842-8848).  Lets ldf_slp be
    # walked stage by stage in NEMO's own order:
    #   prd_arg -> zgrv_iik/iikm1 -> zaj -> zbw -> zbj -> zfk -> zww_raw -> wslpj
    # so the FIRST diverging stage can be identified instead of inferred.
    "eiv_dump_zaj.bin": ("before", "ldfslp.F90:312 zaj, 4-pt zgrv/zcj (unit 8842)"),
    "eiv_dump_zbj.bin": ("before", "ldfslp.F90:317 zbj = MIN(zbw, -100|zaj|, -7e3/e3w|zaj|) (unit 8843)"),
    "eiv_dump_zbw.bin": ("before", "ldfslp.F90:303 zbw = zm1_2g*pn2*(prd(k)+prd(k-1)+2) (unit 8844)"),
    "eiv_dump_zfk.bin": ("before", "ldfslp.F90:320 zfk ML step from nmln, INTEGER division (unit 8845)"),
    "eiv_dump_zgrv_iik.bin": ("before", "ldfslp.F90 zgrv rolling buffer, current level (unit 8846)"),
    "eiv_dump_zgrv_iikm1.bin": ("before", "ldfslp.F90 zgrv rolling buffer, level above (unit 8847)"),
    "eiv_dump_prd_arg.bin": ("before", "ldfslp.F90 prd argument as received (unit 8848)"),
    # Asselin filter dumps are explicit about their own level.
    "atf_dump_tem_before.bin": ("before", "traatf_qco.F90, pre-filter state"),
    "atf_dump_sal_before.bin": ("before", "traatf_qco.F90, pre-filter state"),
    "atf_dump_ssh_before.bin": ("before", "dynatf_qco.F90, pre-filter state"),
    "atf_dump_uu_before.bin": ("before", "dynatf_qco.F90, pre-filter state"),
    "atf_dump_vv_before.bin": ("before", "dynatf_qco.F90, pre-filter state"),
    "atf_dump_tem_after.bin": ("after", "traatf_qco.F90, post-filter state"),
    "atf_dump_sal_after.bin": ("after", "traatf_qco.F90, post-filter state"),
    "atf_dump_ssh_after.bin": ("after", "dynatf_qco.F90, post-filter state"),
    "atf_dump_uu_after.bin": ("after", "dynatf_qco.F90, post-filter state"),
    "atf_dump_vv_after.bin": ("after", "dynatf_qco.F90, post-filter state"),

    # --- zdftke chain instrumentation (#1226 term-by-term TKE walk). ---
    # zdf_tke(kt, Kbb, Kmm, ...) is called from stpmlf.F90:190 as
    # CALL zdf_phy(kstp, Nbb, Nnn, Nrhs) -> zdfphy.F90:286
    # CALL zdf_tke(kt, Kbb, Kmm, sh2, avm_k, avt_k), i.e. Kbb=Nbb, Kmm=Nnn.
    # These are TKE-closure OUTPUTS, not T/S fields themselves -- en/zmxlm/
    # zmxld/avt/avm have no leapfrog time-level dimension of their own
    # (zdf_oce.F90:53 declares en as a single "now" SAVE array, no Nbb/Nnn/Naa
    # index). Each is registered by the time level of its DOMINANT governing
    # N^2 input, exactly as hmlp/nmln above were registered by their rn2b
    # integrand rather than by their own (non-existent) leapfrog index. Where
    # a field mixes both (the Prandtl branch), the citation says so
    # explicitly -- do not read the "now" label as "purely now-derived".
    "tke_dump_en.bin": ("now", "zdftke.F90:496 en RHS stratification-destruction "
                                "term uses rn2 (now, ts(...,Nnn) via stpmlf.F90:187); "
                                "geometry (gdepw/e3t) also Kmm=Nnn. The Prandtl/Ri "
                                "branch feeding p_pdlr (zdftke.F90:429-441 in the "
                                "rebuilt file) uses rn2b (before) but does NOT modify "
                                "en itself -- en's own RHS is now-only. Captured after "
                                "tke_tke returns (zdftke.F90 zdf_tke, right after "
                                "CALL tke_tke), i.e. fully final incl. nn_etau=1 "
                                "penetration (zdftke.F90 tke_tke, nn_etau==1 branch)."),
    "tke_dump_zmxlm.bin": ("now", "zdftke.F90:740 zrn2 = MAX(rn2(ji,jj,jk), rsmall) "
                                   "in tke_avn's mixing-length buoyancy-scale calc "
                                   "(rn2 = now, ts(...,Nnn)); captured after ALL "
                                   "nn_mxl constraint sweeps (tke_avn nn_mxl SELECT "
                                   "CASE, DINO nn_mxl=3 branch) have run."),
    "tke_dump_zmxld.bin": ("now", "same rn2(now) dependency as tke_dump_zmxlm.bin "
                                   "(zdftke.F90:740); captured after ALL nn_mxl "
                                   "constraint sweeps, same insertion point."),
    "tke_dump_avt_final.bin": ("now", "base closure avt=MAX(zav,avtb) (tke_avn, "
                                        "zsqen=SQRT(en)/zmxlm branch) is now-derived "
                                        "(en, zmxlm both now per above); the nn_pdl==1 "
                                        "correction multiplies by p_pdlr, which DOES "
                                        "depend on rn2b (before) via the Prandtl/Ri "
                                        "branch in tke_tke -- avt_final is therefore a "
                                        "MIXED now/before composite, registered 'now' "
                                        "for its dominant (zmxlm, en) dependency. "
                                        "Captured at zdf_tke routine exit, after "
                                        "CALL tke_avn returns, i.e. what zdfphy.F90:"
                                        "313-314 copies into avt_k immediately after."),
    "tke_dump_avm_final.bin": ("now", "avm=MAX(zav,avmb) has no Prandtl correction "
                                        "(nn_pdl only touches avt, tke_avn's "
                                        "IF(nn_pdl==1) block) -- purely now-derived "
                                        "(en, zmxlm). Captured at zdf_tke routine "
                                        "exit alongside tke_dump_avt_final.bin."),

    # --- #1226 batched-rebuild dumps (2026-07-30, MY_SRC line numbers). ---
    "tke_dump_rn2.bin": ("now", "rn2 = bn2(ts(...,Nnn), rab_n, Nnn) at MY_SRC "
                                 "stpmlf.F90:187 (NOW N^2, unlike rn2b at unit "
                                 "8849). WRITE at MY_SRC zdftke.F90:236, in the "
                                 "post-tke_tke/pre-tke_avn block: rn2 is read "
                                 "(never written) by tke_tke's EN stratification "
                                 "term (zdftke.F90:514) and tke_avn's mixing "
                                 "length (:758), so the dump equals what BOTH "
                                 "consumed/consume. Interior 52x199, jk=1..jpk."),
    "tke_dump_dissl.bin": ("now", "dissl = SAVE'd dissipation carry (restart-"
                                   "read, zdftke.F90 tke_rst) consumed by the "
                                   "semi-implicit split (zdiag MY_SRC zdftke."
                                   "F90:510, RHS :515) BEFORE the dump and only "
                                   "rewritten by tke_avn at :837 AFTER it. WRITE "
                                   "at MY_SRC zdftke.F90:237 (post-tke_tke, "
                                   "pre-tke_avn) => exactly the carried-in value "
                                   "the split used at this kt, PRE-overwrite "
                                   "side. No leapfrog index of its own; 'now' "
                                   "for its governing en/rn2 stage, as for "
                                   "tke_dump_en.bin. Interior 52x199, jk=1..jpk."),
    "wzv_dump_ww_call1.bin": ("now", "wzv_MLF diagnoses NOW w: pww integrated "
                                      "from hdiv with e3t(:,:,:,Kmm) (sshwzv."
                                      "F90:211), Kmm=Nnn. WRITE at MY_SRC "
                                      "sshwzv.F90:295, end of wzv_MLF after ALL "
                                      "pww writes. CALL 1 of 2 = stpmlf.F90:244 "
                                      "(pre-dynamics, pre-barotropic hdiv). "
                                      "Full 56x203, jk=1..jpk, w-grid."),
    "wzv_dump_ww_call2.bin": ("now", "same WRITE site (MY_SRC sshwzv.F90:295), "
                                      "CALL 2 of 2 = stpmlf.F90:315 under "
                                      "ln_dynspg_ts, AFTER dyn_spg_ts + the 2nd "
                                      "div_hor call -- this is the ww the SAME "
                                      "step's tra_adv consumes. A single file "
                                      "would have silently kept only this call; "
                                      "hence two files. Full 56x203, jk=1..jpk."),
    "stp_dump_22_before_traldf_tem.bin": ("now", "ts(:,:,:,jp_tem,Nrhs) -- the "
                                                  "accumulated tracer RHS -- "
                                                  "dumped at MY_SRC stpmlf.F90:"
                                                  "436, immediately BEFORE CALL "
                                                  "tra_ldf (:437). Nrhs is a "
                                                  "TENDENCY accumulator, not a "
                                                  "leapfrog state; registered "
                                                  "'now' for its Nnn evaluation "
                                                  "stage. Full 56x203, jk=1.."
                                                  "jpkm1 (35 levels)."),
    "stp_dump_22_before_traldf_sal.bin": ("now", "salinity twin of stp_dump_22_"
                                                  "before_traldf_tem.bin (same "
                                                  "WRITE, MY_SRC stpmlf.F90:436)."),
    "stp_dump_23_after_traldf_tem.bin": ("now", "ts(:,:,:,jp_tem,Nrhs) dumped at "
                                                 "MY_SRC stpmlf.F90:438, "
                                                 "immediately AFTER CALL tra_ldf "
                                                 "(:437). NOTHING writes ts(Nrhs) "
                                                 "between dumps 22 and 23 except "
                                                 "tra_ldf itself (dumps adjacent "
                                                 "to the call), so 23-22 == the "
                                                 "traldf_iso lap increment. Full "
                                                 "56x203, jk=1..jpkm1."),
    "stp_dump_23_after_traldf_sal.bin": ("now", "salinity twin of stp_dump_23_"
                                                 "after_traldf_tem.bin (same "
                                                 "WRITE, MY_SRC stpmlf.F90:438)."),
    # mlf_baro_corr before/after (W3/W1a, #388 nemo_faithful_ocean_implementation_
    # plan.md): full 3-D u/v captured at kt==nit000 only, BEFORE (stpmlf.F90:592-
    # 604, zu_bc_before snapshot of puu(:,:,:,Kaa) as it arrives POST-dyn_zdf) and
    # AFTER (stpmlf.F90:621-632, puu(:,:,:,Kaa) once the depth-mean replacement at
    # :616-619 has run) the barotropic/baroclinic reconciliation. Both dumps are
    # the Kaa ("after") time level -- mlf_baro_corr never touches Kbb/Kmm here
    # (the ln_bt_fw=F Kmm branch at :634-643 is a SEPARATE array section, not
    # dumped). Full haloed jpi x jpj, jk=1..jpkm1 (35 levels), no A2D restriction
    # in the Fortran WRITE loop.
    "baro_dump_u_before.bin": ("after", "stpmlf.F90:592-604 zu_bc_before = "
                                         "puu(:,:,:,Kaa) pre-correction snapshot"),
    "baro_dump_v_before.bin": ("after", "stpmlf.F90:592-604 zv_bc_before = "
                                         "pvv(:,:,:,Kaa) pre-correction snapshot"),
    "baro_dump_u_after.bin": ("after", "stpmlf.F90:621-632 puu(:,:,:,Kaa) after "
                                        "the depth-mean replacement (:616-619)"),
    "baro_dump_v_after.bin": ("after", "stpmlf.F90:621-632 pvv(:,:,:,Kaa) after "
                                        "the depth-mean replacement (:616-619)"),
    "zdf_dump_u1_prestress.bin": ("after", "puu(:,:,1,Kaa) captured per j-slab at "
                                            "MY_SRC dynzdf.F90:350, immediately "
                                            "BEFORE the MLF surface-stress add "
                                            "(stock dynzdf.F90:333). Kaa mid-"
                                            "solve RHS state entering the second "
                                            "recurrence. Interior 52x199, 2-D."),
    "zdf_dump_u1_poststress.bin": ("after", "puu(:,:,1,Kaa) captured at MY_SRC "
                                             "dynzdf.F90:369, immediately AFTER "
                                             "the stress add and BEFORE the "
                                             "downward sweep / third-recurrence "
                                             "upward sweep that rewrites level 1 "
                                             "-- post-pre == zDt_2*(utau_b+utauU)"
                                             "/(e3u(:,:,1,Kaa)*rho0)*umask, the "
                                             "isolated stress application "
                                             "(centred MLF branch; key_RK3 "
                                             "undefined). Interior 52x199, 2-D."),
    "zdf_dump_v1_prestress.bin": ("after", "v twin of zdf_dump_u1_prestress.bin "
                                            "(capture MY_SRC dynzdf.F90:538, "
                                            "stress line stock :507)."),
    "zdf_dump_v1_poststress.bin": ("after", "v twin of zdf_dump_u1_poststress.bin "
                                             "(capture MY_SRC dynzdf.F90:555)."),
    "wnd_dump_zu_frc_inc.bin": ("now", "zu_frc wind-only increment: snapshot at "
                                        "MY_SRC dynspg_ts.F90:432 immediately "
                                        "before the wind IF/ELSE, dump of "
                                        "(zu_frc - prewind) at :451 immediately "
                                        "after -- brackets ONLY the wind add "
                                        "(stock :420-433; DINO ln_bt_fw=F => "
                                        "CENTRED branch zztmp*(utau_b+utauU)*"
                                        "r1_hu(:,:,Kmm), Kmm=Nnn depth). Mirrors "
                                        "the drag bracket (drg_dump_zu_frc_inc). "
                                        "Interior 52x199, 2-D."),
    "wnd_dump_zv_frc_inc.bin": ("now", "v twin of wnd_dump_zu_frc_inc.bin (same "
                                        "bracket, vtau_b+vtauV, r1_hv(:,:,Kmm))."),

    # --- FACE10 task (2026-08-01): w1400 face-flux + EIV bolus 10-day
    # measurement. Additive WRITE-only per-day dumps (kt = nit000 + 32*iday,
    # iday=1..10 -> DAY suffix NN=01..10), gated MOD(kt-nit000,32)==0 in
    # MY_SRC/sshwzv.F90 and MOD(kt-kit000,32)==0 in MY_SRC/ldftra.F90 (nit000
    # == kit000, both are the FACE10 run's first-step index 230401; 32
    # steps/day at rn_rdt=2700s per namelist_cfg nn_it000/nn_itend). Distinct
    # per-day file, never REPLACE'd across days (same anti-provenance-trap
    # reasoning as wzv_dump_ww_call1/call2 above). ---

    # --- #1455 A5 (2026-08-02): NEMO per-cell isoneutral(Redi) tracer trend
    # ttrd_ldf, dumped into the per-rank restart via iom_rstput (MY_SRC/
    # trddump.F90 trddump_write, called from restart.F90:187). Registered by
    # its BEFORE (Kbb) time level: traldf_iso_lap's tracer gradients
    # zdit/zdjt/zdkt read pt_in(...,Kbb) at traldf_iso_scheme.h90:26-30, i.e.
    # the diffusive operator differentiates the before-level tracer field, not
    # the now-level one. The trend is captured in trddump_tra
    # (trdtra.F90:348) from ptrdx = the Krhs delta traldf.F90:95-112
    # accumulated across the single traldf_iso_lap call (horizontal A11/A22 +
    # the K33 vertical diagonal combined; trdtra.F90:376 case jptra_ldf). This
    # is a NetCDF restart variable, not a ".bin" dump, but its level is
    # recorded here for the same reason as every entry above: comparing it
    # against a now-level lego field silently substitutes a T_now-T_before
    # difference for "error". lego's box_heat_budget evaluates iso_redi+k33 on
    # the state's OWN T/S (2-level scheme, no NEMO before-twin) -- flagged, not
    # asserted equivalent (see scripts/tmp/redi_localize_1455_a5.py).
    "ttrd_ldf": ("before", "traldf_iso_scheme.h90:26-30 zdit/zdjt/zdkt read "
                            "pt_in(...,Kbb); accumulated to Krhs traldf.F90:95, "
                            "captured trdtra.F90:348/376 (jptra_ldf), written "
                            "restart.F90:187 -> MY_SRC/trddump.F90 trddump_write"),
    "strd_ldf": ("before", "salinity twin of ttrd_ldf (same traldf_iso_lap "
                            "Kbb read, trddump_tra ptrdy)"),

    # --- #1492 EIV-DIAG task (2026-08-06): additive annual dump of the
    # Treguier eddy-induced-velocity coefficient (aeiu/aeiv) plus its
    # isoneutral-slope/N^2 inputs, for RUN_EIV_DIAG (y10->y20 continuation)
    # in MY_SRC/ldftra.F90. WRITE block inserted at ldftra.F90 end of
    # ldf_tra (after the existing aeiu_3d/aeiv_3d iom_put calls), gated
    # MOD(kt,11520)==0 .AND. kt>=nit000 (11520 steps/year at rn_rdt=2700s).
    # Registered "before": aeiu/aeiv are ldf_eiv's OWN output (ldftra.F90:
    # 668-768), and ldf_eiv's zah/zn accumulation reads rn2b (before-level
    # N^2, ldftra.F90:699/713) and wslpi/wslpj (also before-level per the
    # existing "eiv_dump_wslpi.bin"/"eiv_dump_wslpj.bin"/"eiv_dump_rn2b.bin"
    # entries above, ldfslp.F90 uses rn2b/rab_b) -- aeiu/aeiv inherit the
    # SAME before-level dependency as their own governing inputs, exactly as
    # dump_nmln.bin/dump_hmlp.bin above are registered by their rn2b
    # integrand rather than by their own (non-existent) leapfrog index.
    "eivdiag_aeiu_yNN_rankRR.bin": ("before", "ldftra.F90:668-768 ldf_eiv "
        "aeiu output; zah/zn accumulation (:699-723) reads rn2b (before, "
        "same dependency as eiv_dump_rn2b.bin above). WRITE at ldftra.F90 "
        "end of ldf_tra (new block), gated MOD(kt,11520)==0, year index = "
        "kt/11520."),
    "eivdiag_aeiv_yNN_rankRR.bin": ("before", "v twin of eivdiag_aeiu (same "
        "WRITE block, same ldf_eiv aeiv output)."),
    "eivdiag_wslpi_yNN_rankRR.bin": ("before", "ldfslp.F90 wslpi, same "
        "before-level slope as eiv_dump_wslpi.bin (uses rn2b/rab_b); "
        "re-dumped alongside aeiu/aeiv at the annual cadence so the bolus "
        "streamfunction psi_uw = -1/4*e2u*(wslpi_k+wslpi_k+1)*(aeiu_k+"
        "aeiu_k+1) (ldftra.F90:828-829) can be reconstructed offline at "
        "matched years."),
    "eivdiag_wslpj_yNN_rankRR.bin": ("before", "v twin of eivdiag_wslpi "
        "(ldfslp.F90 wslpj, same before-level dependency as "
        "eiv_dump_wslpj.bin)."),
    "eivdiag_rn2b_yNN_rankRR.bin": ("before", "before-level N^2, identical "
        "quantity/citation as eiv_dump_rn2b.bin above (rn2b = bn2(ts(..., "
        "Nbb), rab_b, Nnn)); re-dumped at the annual cadence for the same "
        "offline bolus reconstruction as the other eivdiag_* entries."),

    # --- dyn_cor_2D (#1226 item 5): the barotropic EEN Coriolis, substep 1 ---
    # WRITE block at MY_SRC dynspg_ts.F90:794-806, guarded by
    # "ll_spg_dump .AND. jn == 1", immediately after CALL dyn_cor_2D(ua_e,
    # va_e, zu_trd, zv_trd) at :783 and BEFORE the tidal / bottom-stress
    # additions overwrite zu_trd/zv_trd in place.
    #
    # The dumped ua_e/va_e are the BEFORE-level barotropic transport, proved
    # by the config's own switches (RUN_GDB/namelist_cfg: ln_bt_fw=.false.,
    # nn_bt_flt=2), not assumed:
    #   1. nn_bt_flt /= 3          => ll_bt_av = .TRUE.   (:202-203)
    #   2. ll_init  = ll_bt_av     => ll_init  = .TRUE.   (:208)
    #   3. ln_bt_fw=.F.            => CENTRED branch: un_e = puu_b(:,:,Kbb),
    #                                 vn_e = pvv_b(:,:,Kbb)              (:570-571)
    #   4. ll_init                 => ub_e = ubb_e = vb_e = vbb_e = 0    (:546-553)
    #   5. jn=1 and (jn<3).AND.ll_init => za1=1, za2=za3=0               (:630-632)
    #   6. ua_e = za1*un_e + za2*ub_e + za3*ubb_e                        (:645)
    #      => ua_e == un_e == puu_b(Kbb) exactly at jn=1.
    # Independently confirmed by measurement: max|ua_e - un_e_init| = 0.000e+00
    # against spg_dump_un_e_init.bin (which is itself puu_b(Kbb), :570).
    # zu_trd/zv_trd are dyn_cor_2D OF that before-level transport, so they
    # carry the same level (registered by their governing input, exactly as
    # dump_nmln/dump_hmlp are registered by their rn2b integrand).
    "cor2d_dump_ua_e_in_substep1.bin": ("before", "dynspg_ts.F90:804 WRITE(8975) "
        "ua_e at jn=1; ua_e == un_e == puu_b(:,:,Kbb) there via :202-203 "
        "(ll_bt_av) -> :208 (ll_init) -> :570 (ln_bt_fw=.F. CENTRED seed) -> "
        ":546-553 (ub_e=ubb_e=0) -> :630-632 (za1=1,za2=za3=0) -> :645."),
    "cor2d_dump_va_e_in_substep1.bin": ("before", "dynspg_ts.F90:805 WRITE(8976) "
        "va_e at jn=1; v twin of cor2d_dump_ua_e_in_substep1.bin, same proof "
        "chain via :571 (vn_e = pvv_b(:,:,Kbb)) and :648."),
    "cor2d_dump_zu_trd_substep1.bin": ("before", "dynspg_ts.F90:802 WRITE(8973) "
        "zu_trd at jn=1, the OUTPUT of CALL dyn_cor_2D(ua_e, va_e, ...) at "
        ":783 (pre tidal/bottom-stress overwrite). No leapfrog index of its "
        "own; registered by its governing input ua_e/va_e = puu_b/pvv_b(Kbb) "
        "(see cor2d_dump_ua_e_in_substep1.bin)."),
    "cor2d_dump_zv_trd_substep1.bin": ("before", "dynspg_ts.F90:803 WRITE(8974) "
        "zv_trd at jn=1; v twin of cor2d_dump_zu_trd_substep1.bin, same "
        "dyn_cor_2D call at :783 and same governing before-level input "
        "(pvv_b(:,:,Kbb))."),
    # --- #1455 sec-D JOB 1: the barotropic time-mean advective transport
    # un_adv/vn_adv (dynspg_ts.F90).  Accumulated OVER the substep window
    # (:736 ``un_adv += za2*zhU*r1_e2u``, wgtbtp2 boxcar weights) and
    # normalised at :999 (``/r1_wgt2s``).  It is a SUBSTEP TIME MEAN, not a
    # bare leapfrog level; NEMO reconciles the 3-D momentum depth-mean ONTO it
    # at Kmm=NOW (dynspg_ts.F90:1172, ``un_adv*r1_hu(Kmm)``), so it is
    # registered "now" -- the level of the reconcile that consumes it (the same
    # transport_avg target the DINO card selects, dino.py:1487).  Written as a
    # SINGLETON (first-step-only, ll_spg_dump=kt==nit000) at :1046 WRITE(8862).
    # Units: [m^2/s] transport per unit width = <SUM_k e3u*u>_substep (map to
    # full transport [m^3/s] via *e2u).
    "spg_dump_un_adv_final.bin": ("now", "dynspg_ts.F90:1046 WRITE(8862) un_adv; "
        "substep time-mean advective transport (:736 accumulate, :999 /r1_wgt2s) "
        "reconciled at Kmm=NOW (:1172 un_adv*r1_hu(Kmm))."),
    "spg_dump_vn_adv_final.bin": ("now", "dynspg_ts.F90:1047 WRITE(8863) vn_adv; "
        "v twin of spg_dump_un_adv_final.bin, same substep-mean accumulation "
        "(:737) and Kmm=NOW reconcile."),
    # --- #1455 sec-D PHASE 2: the FULL per-barotropic-substep trajectory
    # substep_dump.bin (dynspg_ts.F90:926-940, guarded "IF(kt==nit000)", one
    # record per jn=1..icycle).  Each record holds sshn_e/ssha_e/zsshp2_e/un_e/
    # vn_e/ua_e/va_e at that substep -- the barotropic-mode fast fields, which
    # have NO leapfrog time level of their own (un_e etc. are the split-explicit
    # sub-cycle state, seeded at jn=1 from puu_b(:,:,Kbb) via the ln_bt_fw=F
    # CENTRED branch, dynspg_ts.F90:570; proven by un_e[jn=1] == un_e_init to
    # 0.0e0).  Registered "before" for the SAME reason as
    # cor2d_dump_ua_e_in_substep1.bin above (the sub-cycle seed is the Kbb
    # barotropic transport); this dump is a superset (all icycle substeps, not
    # just jn=1).  jpi x jpj = 56 x 203, per-substep records, fp64.
    "substep_dump.bin": ("before", "dynspg_ts.F90:926-940 WRITE(799) per-jn "
        "barotropic sub-cycle fields (sshn_e/ssha_e/zsshp2_e/un_e/vn_e/ua_e/"
        "va_e); un_e[jn=1] == puu_b(:,:,Kbb) (dynspg_ts.F90:570, ln_bt_fw=F "
        "CENTRED seed), same Kbb sub-cycle seed as cor2d_dump_ua_e_in_substep1.bin"),
    # --- #1226 SEQDUMP intra-step seam walk (RUN_SEQDUMP_Y20_1R, y20 kt=
    # 230401..230404). WRITE sites in MY_SRC/stpmlf.F90 (oracle commit
    # ebeb8a5), each proved by reading the CALL that produces the dumped
    # field one line above the WRITE (Rule 1d cite = the producing CALL, not
    # the WRITE). These localise WHERE the barotropic/longitude-uniform
    # eta(Naa) injection first appears un-inherited.
    #
    # ssh/r3 CHAIN (r3t = ssh/H_0 ratio; ssh = H_0*r3t, so the r3 chain IS the
    # ssh commit chain). Naa (after) time level throughout -- these are the
    # SAME leap-frog after-ssh at three successive commit points:
    "seq_dump_rhd_bbb": ("before", "stpmlf.F90:212 CALL eos(ts,Nbb,rhd) "
        "-> WRITE(8930) :222; before in-situ density feeding ldf_slp."),
    "seq_dump_rhd_nnn": ("now", "stpmlf.F90:271 CALL eos(ts,Nnn,rhd,rhop) "
        "-> WRITE(8931) :280; now in-situ density feeding dyn_hpg."),
    "seq_dump_hdiv_nnn": ("now", "stpmlf.F90:340 CALL div_hor(kstp,Nbb,Nnn) "
        "2nd (time-split) call -> WRITE(8932) :362; hdiv(Nnn). NOTE outermost "
        "1-cell ring is zero-filled, valid after the standard 2-cell strip."),
    # r3(Naa) AFTER the SECOND dom_qco_r3c (stpmlf.F90:367), i.e. built from
    # the POST-barotropic (dyn_spg-replaced) ssh(Naa). This is the ssh the
    # step COMMITS pre-filter. Naa.
    "seq_dump_r3t_aaa": ("after", "stpmlf.F90:367 CALL dom_qco_r3c(ssh(Naa),"
        "r3t(Naa),...) 2nd call, post dyn_spg -> WRITE(8933) :377; r3t(Naa) "
        "= ssh(Naa)/H_0 from the barotropic-replaced ssh."),
    "seq_dump_r3u_aaa": ("after", "stpmlf.F90:367 dom_qco_r3c -> WRITE(8934) "
        ":378; r3u(Naa) u-point twin of seq_dump_r3t_aaa."),
    "seq_dump_r3v_aaa": ("after", "stpmlf.F90:367 dom_qco_r3c -> WRITE(8935) "
        ":379; r3v(Naa) v-point twin of seq_dump_r3t_aaa."),
    "seq_dump_r3f": ("after", "stpmlf.F90:367 dom_qco_r3c (the ONLY call "
        "producing r3f) -> WRITE(8936) :380; r3f(Naa) f-point ratio."),
    # r3_f from the ASSELIN-FILTERED now ssh (stpmlf.F90:442). This is what
    # the NEXT step carries as its geometry -- registered by traatf/ssh_atf's
    # own after-filter level exactly as atf_dump_ssh_after is "after".
    "seq_dump_r3t_f": ("after", "stpmlf.F90:442 CALL dom_qco_r3c(ssh(Nnn),"
        "r3t_f,...) from ssh_atf-filtered ssh -> WRITE(8937) :450; r3t_f."),
    "seq_dump_r3u_f": ("after", "stpmlf.F90:442 dom_qco_r3c filtered -> "
        "WRITE(8938) :451; r3u_f u-point twin of seq_dump_r3t_f."),
    "seq_dump_r3v_f": ("after", "stpmlf.F90:442 dom_qco_r3c filtered -> "
        "WRITE(8939) :452; r3v_f v-point twin of seq_dump_r3t_f."),
    # post-finalize_lbc Naa state (stpmlf.F90:562 CALL finalize_lbc; dumps at
    # :576-579). u/v/T/S at Naa BEFORE the Asselin swap.
    "seq_dump_postlbc_u_aaa": ("after", "stpmlf.F90:562 CALL finalize_lbc -> "
        "WRITE(8940) :576; uu(Naa) post-lbc."),
    "seq_dump_postlbc_v_aaa": ("after", "stpmlf.F90:562 finalize_lbc -> "
        "WRITE(8941) :577; vv(Naa) post-lbc."),
    "seq_dump_postlbc_tem_aaa": ("after", "stpmlf.F90:562 finalize_lbc -> "
        "WRITE(8942) :578; ts(jp_tem,Naa) post-lbc."),
    "seq_dump_postlbc_sal_aaa": ("after", "stpmlf.F90:562 finalize_lbc -> "
        "WRITE(8943) :579; ts(jp_sal,Naa) post-lbc."),
    # r3c_dump_r3t: r3t(Naa) after the FIRST dom_qco_r3c (stpmlf.F90:244),
    # built from ssh_nxt's FIRST-GUESS ssh(Naa) (pre dyn_spg). Paired with
    # seq_dump_r3t_aaa this brackets the barotropic ssh replacement.
    "r3c_dump_r3t": ("after", "stpmlf.F90:244 CALL dom_qco_r3c(ssh(Naa),"
        "r3t(Naa),...) 1st call, from ssh_nxt first-guess ssh -> WRITE :256; "
        "r3t(Naa) pre-barotropic."),
    "r3c_dump_r3u": ("after", "stpmlf.F90:244 dom_qco_r3c 1st -> :257; r3u(Naa) "
        "pre-barotropic."),
    "r3c_dump_r3v": ("after", "stpmlf.F90:244 dom_qco_r3c 1st -> :258; r3v(Naa) "
        "pre-barotropic."),
}
# SEQDUMP families carry a _ktNNNNNNNN suffix (per step). Register the exact
# per-step filenames (fail-closed lookup does not strip the suffix), y20
# window kt=230401..230404.
_SEQDUMP_BASES = {k: v for k, v in _DUMP_TIME_LEVEL.items()
                  if k.startswith(("seq_dump_", "r3c_dump_"))}
for _b, _lv in _SEQDUMP_BASES.items():
    for _kt in range(230401, 230405):
        _DUMP_TIME_LEVEL[f"{_b}_kt{_kt:08d}.bin"] = _lv
del _SEQDUMP_BASES, _b, _lv, _kt
_EIVDIAG_SOURCE_AEIU = _DUMP_TIME_LEVEL["eivdiag_aeiu_yNN_rankRR.bin"][1]
_EIVDIAG_SOURCE_AEIV = _DUMP_TIME_LEVEL["eivdiag_aeiv_yNN_rankRR.bin"][1]
_EIVDIAG_SOURCE_WSLPI = _DUMP_TIME_LEVEL["eivdiag_wslpi_yNN_rankRR.bin"][1]
_EIVDIAG_SOURCE_WSLPJ = _DUMP_TIME_LEVEL["eivdiag_wslpj_yNN_rankRR.bin"][1]
_EIVDIAG_SOURCE_RN2B = _DUMP_TIME_LEVEL["eivdiag_rn2b_yNN_rankRR.bin"][1]
del _DUMP_TIME_LEVEL["eivdiag_aeiu_yNN_rankRR.bin"]
del _DUMP_TIME_LEVEL["eivdiag_aeiv_yNN_rankRR.bin"]
del _DUMP_TIME_LEVEL["eivdiag_wslpi_yNN_rankRR.bin"]
del _DUMP_TIME_LEVEL["eivdiag_wslpj_yNN_rankRR.bin"]
del _DUMP_TIME_LEVEL["eivdiag_rn2b_yNN_rankRR.bin"]
# RUN_EIV_DIAG covers years 11-20 (kt/11520 = 11..20), 16 MPI ranks (jpni=2
# x jpnj=8, same tiling as RUN_20Y/RUN_ENS_M*).
for _yr in range(11, 21):
    _yy = f"{_yr:02d}"
    for _rank in range(16):
        _rr = f"{_rank:02d}"
        _DUMP_TIME_LEVEL[f"eivdiag_aeiu_y{_yy}_rank{_rr}.bin"] = ("before", _EIVDIAG_SOURCE_AEIU)
        _DUMP_TIME_LEVEL[f"eivdiag_aeiv_y{_yy}_rank{_rr}.bin"] = ("before", _EIVDIAG_SOURCE_AEIV)
        _DUMP_TIME_LEVEL[f"eivdiag_wslpi_y{_yy}_rank{_rr}.bin"] = ("before", _EIVDIAG_SOURCE_WSLPI)
        _DUMP_TIME_LEVEL[f"eivdiag_wslpj_y{_yy}_rank{_rr}.bin"] = ("before", _EIVDIAG_SOURCE_WSLPJ)
        _DUMP_TIME_LEVEL[f"eivdiag_rn2b_y{_yy}_rank{_rr}.bin"] = ("before", _EIVDIAG_SOURCE_RN2B)
del _yr, _yy, _rank, _rr


_FACE10_WZV_SOURCE = (
    "Per-day twin of wzv_dump_ww_call1/2.bin: SAME post-wzv_MLF pww (bottom "
    "BC + upward hdiv integration + bdy/AGRIF masking all done), same WRITE "
    "site MY_SRC sshwzv.F90:295-ff (new block inserted directly after the "
    "existing call1/call2 dump, sshwzv.F90:299-321). pww is the NOW "
    "(Kmm=Nnn) diagnosed w; 'now' time level, matching call1/call2. Full "
    "jpi x jpj (halo incl.), jk=1..jpk, w-grid. Gated MOD(kt-nit000,32)==0, "
    "kt in nit000+32..nit000+320 (FACE10 run's 10-day window)."
)
_FACE10_EIV_W_SOURCE = (
    "EIV/bolus contribution to pww, captured EXACTLY as the increment added "
    "to pww (MY_SRC ldftra.F90 ldf_eiv_trp_MLF, add statement at the (now) "
    "line directly below the new dump block, ldftra.F90:952-954; dump block "
    "ldftra.F90:922-951). zw_eiv_incr = (zpsi_uw(:,:,1)-zpsi_uw(i-1,:,1)) + "
    "(zpsi_vw(:,:,1)-zpsi_vw(:,j-1,1)), captured into a local BEFORE the "
    "production ADD so the add statement itself is untouched -- this is the "
    "TRUE increment, not an approximation (ideal quantity from step 2b, "
    "achievable additively; no traadv.F90 restructuring needed since "
    "ldf_eiv_trp_MLF folds EIV into pww BEFORE tra_adv consumes it -- read "
    "traadv.F90:208-210,343-344 and ldftra.F90:923-926 to confirm this "
    "ordering). 'now' time level (Kmm=Nnn slopes/coeffs feed this add, same "
    "stage as wslpi/aeiu used elsewhere in ldf_eiv_trp_MLF). Interior "
    "T2D(nn_hls), jk=1..jpkm1, w-grid. Gated MOD(kt-kit000,32)==0, kt in "
    "kit000+32..kit000+320."
)
_FACE10_EIV_U_SOURCE = (
    "CORRECTED LABEL (2026-08-01, caught in post-processing): contains the "
    "raw bolus STREAMFUNCTION zpsi_uw(:,:,1) -- psi at the interface at the "
    "TOP of cell jk -- NOT the transport increment (WRITE at MY_SRC "
    "ldftra.F90:957, inside the per-day dump block). The TRUE added "
    "increment u_eiv(jk) = psi(top of jk+1) - psi(top of jk), i.e. vertical "
    "differencing of this dump (last level: -psi, since psi at the bottom "
    "interface is 0 via wumask); validated by reproducing legoESM's own "
    "(total - Eulerian) face numbers to ~1% (face10_verdict.py). An earlier "
    "registry entry mislabeled this as the increment "
    "zpsi_uw(1)-zpsi_uw(2); that quantity is what the ORIGINAL single-shot "
    "eiv_dump_u.bin (ldftra.F90:864) dumps, not this per-day file. 'now' "
    "time level. Full jpi x jpj, jk=1..jpkm1, u-grid."
)
_FACE10_EIV_V_SOURCE = (
    "v twin of eiv_dump_u_dayNN.bin (WRITE ldftra.F90:958, raw "
    "zpsi_vw(:,:,1) streamfunction -- same corrected label, same vertical-"
    "differencing reconstruction for the true v_eiv increment, v-grid)."
)
# Filenames carry a _rankNN suffix (16 MPI ranks, jpni=2 x jpnj=8): a FIRST
# version of the Fortran dump used one shared filename across all ranks, so
# every rank's OPEN(...STATUS='REPLACE') raced on the SAME path and only the
# last writer's data survived (caught via file-size check -- 250560 bytes ==
# exactly one rank's local array, not 16 -- BEFORE any physics number was
# read from it). Fixed by suffixing narea (dom_oce, 1-based MPI rank) into
# the filename; registry entries below match the CORRECTED per-rank names.
for _iday in range(1, 11):
    _nn = f"{_iday:02d}"
    for _rank in range(16):
        _rr = f"{_rank:02d}"
        _DUMP_TIME_LEVEL[f"wzv_dump_ww_day{_nn}_rank{_rr}.bin"] = ("now", _FACE10_WZV_SOURCE)
        _DUMP_TIME_LEVEL[f"eiv_dump_w_incr_day{_nn}_rank{_rr}.bin"] = ("now", _FACE10_EIV_W_SOURCE)
        _DUMP_TIME_LEVEL[f"eiv_dump_u_day{_nn}_rank{_rr}.bin"] = ("now", _FACE10_EIV_U_SOURCE)
        _DUMP_TIME_LEVEL[f"eiv_dump_v_day{_nn}_rank{_rr}.bin"] = ("now", _FACE10_EIV_V_SOURCE)
del _iday, _nn, _rank, _rr


def register_dump(basename: str, level: TimeLevel, source: str) -> None:
    """Register a dump's time level, CITING the NEMO line that proves it.

    ``source`` is mandatory and must be non-empty: an unsourced entry is a
    guess, and a guess here is exactly the failure this module exists to stop.
    """
    if level not in _VALID:
        raise ValueError(
            f"unknown time level {level!r} for {basename!r}; "
            f"expected one of {sorted(_VALID)}")
    if not source.strip():
        raise ValueError(
            f"register_dump({basename!r}) needs a NEMO source citation "
            "(file:line showing which time level feeds it) — an unsourced "
            "entry is a guess")
    _DUMP_TIME_LEVEL[basename] = (level, source)


def time_level_for_dump(basename: str) -> TimeLevel:
    """Time level of ``basename``'s T/S; RAISES on an unregistered dump.

    Deliberately fail-CLOSED (dispatch hardening): defaulting an unknown dump
    to "now" is what produced the false diagnosis in the first place, and a
    silent default would reintroduce it for every dump added later.
    """
    basename = basename.rsplit("/", 1)[-1]
    try:
        return _DUMP_TIME_LEVEL[basename][0]
    except KeyError:
        raise ValueError(
            f"dump {basename!r} has no registered NEMO time level. Read the "
            "call site in the NEMO source, then register_dump(name, level, "
            "source) with the file:line. Do NOT assume 'now' — many routines "
            "run on Nbb T/S with Nnn geometry (stpmlf.F90:184), and comparing "
            "against the wrong level substitutes |T_now - T_before| for error."
        ) from None


def select_ts(basename: str, *, now, before, after=None):
    """Return the ``(T, S)`` pair that matches ``basename``'s time level.

    Make the right thing the default action: pass all the levels you loaded and
    let the registry choose, instead of picking one by hand at each call site.

    ``now``/``before``/``after`` are ``(T, S)`` tuples.
    """
    level = time_level_for_dump(basename)
    chosen = {"now": now, "before": before, "after": after}[level]
    if chosen is None:
        raise ValueError(
            f"dump {basename!r} needs the {level!r} time level, but no "
            f"{level!r} state was supplied to select_ts()")
    return chosen
