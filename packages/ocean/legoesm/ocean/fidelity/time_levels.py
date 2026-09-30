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
    # Certified idealised-testcase trajectory.  The write is the first action
    # in stp_RK3 and names Nbb explicitly; kt=1 is therefore the native initial
    # condition at first-step entry, not a now/after state.
    "oracle_step_entry_kt00000001.bin": (
        "before",
        "scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/"
        "stprk3.F90:88-100 (used byte-for-byte by testcase lanes 1 and 2) "
        "writes ts/uu/vv/ssh(...,Nbb) before forcing, stp_2D, and RK stages",
    ),
    "oracle_step_entry_kt00002160.bin": (
        "before",
        "scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/"
        "stprk3.F90:88-100 writes the GYRE midpoint ts/uu/vv/ssh(...,Nbb) "
        "before forcing, stp_2D, and RK stages",
    ),
    "oracle_step_entry_kt00004320.bin": (
        "before",
        "scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/"
        "stprk3.F90:88-100 writes the GYRE final ts/uu/vv/ssh(...,Nbb) "
        "before forcing, stp_2D, and RK stages",
    ),
    # eos_rab / bn2 family: T/S at Nbb, geometry at Nnn.
    "dump_alpha_b.bin": ("before", "stpmlf.F90:184 eos_rab(ts(...,Nbb), rab_b, Nnn)"),
    "dump_beta_b.bin": ("before", "stpmlf.F90:184 eos_rab(ts(...,Nbb), rab_b, Nnn)"),
    "tke_dump_rn2b.bin": ("before", "rn2b = bn2(ts(...,Nbb)); zdfmxl.F90:98 integrates rn2b"),
    "bn2_dump_zrw.bin": ("before", "eosbn2.F90:1459-1460 zrw in the first "
                                      "stpmlf bn2(ts(...,Nbb),rab_b,rn2b,Nnn) call"),
    "bn2_dump_zaw.bin": ("before", "eosbn2.F90:1462 thermal interpolation in "
                                      "the Nbb-tracer/Nnn-geometry bn2 call"),
    "bn2_dump_zbw.bin": ("before", "eosbn2.F90:1463 saline interpolation in "
                                      "the Nbb-tracer/Nnn-geometry bn2 call"),
    "bn2_dump_numerator.bin": ("before", "eosbn2.F90:1465-1467 numerator in "
                                            "the Nbb-tracer/Nnn-geometry bn2 call"),
    "bn2_dump_result.bin": ("before", "eosbn2.F90:1465-1468 assigned rn2b in "
                                         "the Nbb-tracer/Nnn-geometry bn2 call"),
    # Geometry paired with the before T/S above is nevertheless Kmm=Nnn.
    # ldftra's instrumentation writes the live arrays verbatim at the same
    # step; register geometry by its own level instead of inheriting the T/S
    # label from rn2b.
    "eiv_dump_gdept.bin": ("now", "ldftra.F90:951 gdept(...,Kmm); Kmm=Nnn"),
    "eiv_dump_e3w.bin": ("now", "ldftra.F90:951 e3w(...,Kmm); Kmm=Nnn"),
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
    # Row-30 U/V source-order write-only ladder (units 8900--8911). These
    # preserve the same BEFORE density/rn2b state as the existing slope dumps.
    "eiv_dump_zgru_iik.bin": (
        "before", "ldfslp.F90:203/217 U gradient, current rolling slot"),
    "eiv_dump_zgru_iikm1.bin": (
        "before", "ldfslp.F90:203/217 U gradient, preceding rolling slot"),
    "eiv_dump_zau.bin": ("before", "ldfslp.F90:242 metric-scaled U gradient"),
    "eiv_dump_zav.bin": ("before", "ldfslp.F90:243 metric-scaled V gradient"),
    "eiv_dump_zbu_pre.bin": (
        "before", "ldfslp.F90:244 U denominator before bounds"),
    "eiv_dump_zbv_pre.bin": (
        "before", "ldfslp.F90:245 V denominator before bounds"),
    "eiv_dump_zbu_post.bin": (
        "before", "ldfslp.F90:248 U denominator after bounds"),
    "eiv_dump_zbv_post.bin": (
        "before", "ldfslp.F90:249 V denominator after bounds"),
    "eiv_dump_uslp_raw.bin": (
        "before", "ldfslp.F90:269 raw U slope before Shapiro"),
    "eiv_dump_vslp_raw.bin": (
        "before", "ldfslp.F90:270 raw V slope before Shapiro"),
    "eiv_dump_uslp_postshapiro.bin": (
        "before", "ldfslp.F90:279-285 U after Shapiro, before LBC"),
    "eiv_dump_vslp_postshapiro.bin": (
        "before", "ldfslp.F90:286-292 V after Shapiro, before LBC"),
    # Row-30 raw-U composite continuation (units 8912--8919).
    "eiv_dump_iku.bin": ("before", "ldfslp.F90:251 U mixed-layer index"),
    "eiv_dump_zfi.bin": ("before", "ldfslp.F90:254 U integer ML switch"),
    "eiv_dump_e3u_miku.bin": (
        "now", "ldfslp.F90:261-264 live e3u(miku,Kmm) depth operand"),
    "eiv_dump_zdepu.bin": (
        "now", "ldfslp.F90:261-264 U live water-column depth"),
    "eiv_dump_zuslp_hml_pre.bin": (
        "before", "ldfslp.F90:269 carried U mixed-layer anchor before update"),
    "eiv_dump_sint_u.bin": (
        "before", "ldfslp.F90:269 U interior zau/(zbu-zeps)"),
    "eiv_dump_mlterm_u.bin": (
        "before", "ldfslp.F90:269 U mixed-layer depth-anchor product"),
    "eiv_dump_blend_u.bin": (
        "before", "ldfslp.F90:269 U raw blend before umask"),
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
    "tke_dump_zmxlm_raw.bin": ("now", "write-only row-19 slot captured at "
                                       "zdftke.F90:831-833 immediately after "
                                       "MAX(rmxl_min,SQRT(2*en/MAX(rn2,rsmall))) "
                                       "and before every nn_mxl=3 constraint scan; "
                                       "en/rn2 and geometry are Kmm=Nnn."),
    "tke_dump_zmxld.bin": ("now", "same rn2(now) dependency as tke_dump_zmxlm.bin "
                                   "(zdftke.F90:740); captured after ALL nn_mxl "
                                   "constraint sweeps, same insertion point."),
    "tke_dump_zsqen_base.bin": ("now", "write-only row-21 operand captured at "
                                         "patched MY_SRC/zdftke.F90:930 from "
                                         "the SQRT(en) evaluated at :924, "
                                         "after the TKE solve and "
                                         "row-20 mixing-length scans."),
    "tke_dump_zav_base.bin": ("now", "write-only row-21 operand captured at "
                                       "patched MY_SRC/zdftke.F90:931 from "
                                       "the :925 expression "
                                       "rn_ediff*zmxlm*SQRT(en), before either "
                                       "coefficient floor."),
    "tke_dump_avm_base.bin": ("now", "write-only row-21 base viscosity captured "
                                       "at patched MY_SRC/zdftke.F90:932 from "
                                       "the :926 assignment, before "
                                       "the later zdfphy EVD/LBC assembly."),
    "tke_dump_avt_base.bin": ("now", "write-only row-21 base diffusivity captured "
                                       "at patched MY_SRC/zdftke.F90:933 from "
                                       "the :927 assignment, before the nn_pdl "
                                       "Prandtl overwrite at :944."),
    "tke_dump_dissl_postavn.bin": ("now", "write-only row-21 dissipation carry "
                                            "captured at patched MY_SRC/zdftke.F90:"
                                            "934 from :928 SQRT(en)/zmxld, after the "
                                            "current tke_avn update."),
    "zdf_dump_hmld_turb.bin": ("now", "write-only row-28 turbocline depth "
                                         "persisted after zdf_mxl_turb from "
                                         "zdfmxl.F90:145-152; hmld uses the live "
                                         "Kmm gdepw ladder and current composed avt."),
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
    # Realized coefficients consumed after the current step's closure+EVD
    # assembly. stpmlf calls zdf_phy with Kmm=Nnn at MY_SRC stpmlf.F90:210;
    # zdfphy.F90:311-323 copies closure outputs then applies EVD; ldftra writes
    # those global arrays verbatim at MY_SRC ldftra.F90:955-956. EVD's
    # MIN(rn2,rn2b) trigger mixes now/before, but the field is registered
    # "now" for the current zdf_phy call that owns and publishes it.
    "dump_avt.bin": ("now", "MY_SRC stpmlf.F90:210 calls zdf_phy(Kbb=Nbb,Kmm=Nnn); "
                              "zdfphy.F90:311-323 copies avt_k then applies EVD; "
                              "MY_SRC ldftra.F90:955 writes realized avt verbatim"),
    "dump_avm.bin": ("now", "MY_SRC stpmlf.F90:210 calls zdf_phy(Kbb=Nbb,Kmm=Nnn); "
                              "zdfphy.F90:311-344 copies avm_k, applies EVD, then "
                              "the interior side of lbc_lnk; MY_SRC "
                              "ldftra.F90:956 writes realized avm verbatim"),

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
    "tke_dump_en_postlc.bin": ("now", "MY_SRC zdftke.F90:505 captures en "
                                      "immediately after the active base-source "
                                      "Langmuir block at :401-468 (line :463 "
                                      "updates en), before Prandtl/matrix/RHS; "
                                      "en is the current TKE state."),
    "tke_dump_zdiag_pre.bin": ("now", "MY_SRC zdftke.F90:583 captures zdiag "
                                      "after literal matrix/RHS assembly and "
                                      "before the Thomas forward recurrence."),
    "tke_dump_zlw_pre.bin": ("now", "MY_SRC zdftke.F90:584 captures zd_lw "
                                    "after literal matrix/RHS assembly and "
                                    "before the Thomas forward recurrence."),
    "tke_dump_en_pre.bin": ("now", "MY_SRC zdftke.F90:585 captures en after "
                                   "Langmuir and the TKE budget, immediately "
                                   "before the Thomas forward recurrence."),
    "tke_dump_zdiag_forward.bin": ("now", "MY_SRC zdftke.F90:591 captures "
                                          "the diagonal after NEMO's forward "
                                          "Thomas recurrence."),
    "tke_dump_zrhs_forward.bin": ("now", "MY_SRC zdftke.F90:599 captures "
                                         "the forward-recurring RHS work held "
                                         "in zd_lw."),
    "tke_dump_en_postsolve.bin": ("now", "MY_SRC zdftke.F90:609 captures en "
                                         "after back substitution, floor, and "
                                         "wmask, before nn_etau penetration."),
    "tke_dump_etau_argument.bin": ("now", "write-only row-18 operand slot for "
                                           "-gdepw(Kmm)/htau in the active "
                                           "nn_etau=1 block, zdftke.F90:590."),
    "tke_dump_etau_exp.bin": ("now", "write-only row-18 operand slot for "
                                      "EXP(-gdepw(Kmm)/htau) in the active "
                                      "nn_etau=1 block, zdftke.F90:590."),
    "tke_dump_etau_increment.bin": ("now", "write-only row-18 full additive "
                                            "increment at zdftke.F90:590-591, "
                                            "including ice/W/T masks."),
    "tke_dump_etau_gdepw.bin": ("now", "write-only row-18 direct operand "
                                         "gdepw(ji,jj,jk,Kmm) captured immediately "
                                         "before the active nn_etau=1 division."),
    "tke_dump_etau_htau.bin": ("now", "write-only row-18 direct operand "
                                        "htau(ji,jj) captured at the same read "
                                        "site and repeated over jk."),
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
    # --- #1455 sec-D JOB 2: the zu_frc forcing-assembly brackets.  Each entry
    # is levelled INDIVIDUALLY by its own governing input -- they do NOT share
    # one justification (an earlier version of this block claimed all of them
    # were Kmm=Nnn rates from puu(Krhs); that is true only of spg_dump_z?_frc).
    # Every write site read in MY_SRC before registering (#1455 audit).
    #
    # BEFORE, not "now": under DINO's ln_bt_fw=.false. the drag increment is
    # built from the BEFORE-level bottom baroclinic residual
    # ``puu(ji,jj,ikbu,Kbb) - puu_b(ji,jj,Kbb)`` (dynspg_ts.F90:1819-1822, the
    # CENTRED branch of dyn_drg_init), with only the r1_hu depth at Kmm.  Same
    # switch, same branch and same governing level as
    # cor2d_dump_zu_trd_substep1.bin above, which this file already registers
    # "before" -- so registering these "now" would contradict the module's own
    # rule (level by governing input, geometry noted separately).
    "drg_dump_zu_frc_inc.bin": ("before", "zu_frc bottom-drag-only increment: "
                                       "snapshot zu_frc_predrg at MY_SRC "
                                       "dynspg_ts.F90:379, dump of (zu_frc - "
                                       "zu_frc_predrg) at :398 immediately "
                                       "after CALL dyn_drg_init(Kbb,Kmm,...) "
                                       "at :382-383 -- brackets ONLY the drag "
                                       "add.  GOVERNING INPUT is the BEFORE "
                                       "residual puu(ikbu,Kbb)-puu_b(Kbb) "
                                       "(:1819-1822, ln_bt_fw=F CENTRED "
                                       "branch); GEOMETRY is r1_hu(Kmm) "
                                       "(:1828).  Interior 52x199, 2-D, "
                                       "[m/s^2]."),
    "drg_dump_zv_frc_inc.bin": ("before", "v twin of drg_dump_zu_frc_inc.bin "
                                       "(MY_SRC dynspg_ts.F90:399 WRITE(8972); "
                                       "same CENTRED branch, governing input "
                                       "pvv(ikbv,Kbb)-pvv_b(Kbb) at :1821)."),
    "drg_dump_rCdU_bot.bin": ("now", "rCdU_bot = -Cd*|U| [m/s], SIGN NEGATIVE "
                                     "(zdfdrg.F90:76 declares it '(<0) "
                                     "[m/s]'), the T-point drag "
                                     "coefficient*speed from zdf_drg_nonlin "
                                     "(zdfdrg.F90:172-189), which reads "
                                     "uu(:,:,:,Kmm) -- hence \"now\", unlike "
                                     "the drag INCREMENT above whose residual "
                                     "is Kbb.  Dumped at MY_SRC "
                                     "dynspg_ts.F90:397 WRITE(8970) in the "
                                     "same bracket.  NOT a rate.  FULL haloed "
                                     "jpi x jpj (ji=1,jpi / jj=1,jpj), NOT the "
                                     "interior slice its zu_frc_inc siblings "
                                     "use."),
    "spg_dump_zu_frc.bin": ("now", "FULLY-ASSEMBLED barotropic slow forcing "
                                   "zu_frc at MY_SRC dynspg_ts.F90:514/520 "
                                   "WRITE(8850) -- snapshot after the LAST "
                                   "write to zu_frc (the wind add, :437-443) "
                                   "and before the jn=1..icycle substep loop's "
                                   "own per-substep updates.  Governing input "
                                   "is this step's puu(:,:,:,Krhs) -> \"now\". "
                                   "NOTE the depth weighting at :337 is the "
                                   "REST metric e3u_0/r1_hu_0 (key_qco), NOT "
                                   "the live Kmm metric -- that difference is "
                                   "row 1 of the JOB-2 table.  Interior 52x199 "
                                   "(A2D(0), see the :511 comment), [m/s^2]."),
    "spg_dump_zv_frc.bin": ("now", "v twin of spg_dump_zu_frc.bin (MY_SRC "
                                   "dynspg_ts.F90:516/521 WRITE(8851))."),
    "spg_dump_ssh_frc.bin": ("now", "ssh_frc = r1_rho0*r1_2*(emp_b+emp), "
                                    "assembled at MY_SRC dynspg_ts.F90:470 "
                                    "(ln_bt_fw) / :477 (CENTRED, the DINO "
                                    "branch) and dumped at :518/522 "
                                    "WRITE(8852), the same snapshot point as "
                                    "spg_dump_zu_frc.bin.  Governing input is "
                                    "emp/emp_b at this step -> \"now\".  Units "
                                    "[m/s] (r1_rho0 already applied), NOT a "
                                    "momentum rate.  Written FULL haloed jpi x "
                                    "jpj (ji=1,jpi / jj=1,jpj), unlike its "
                                    "zu_frc/zv_frc siblings in the same WRITE "
                                    "block."),
    # NEMO's own wind stress at the U POINT, the isolating reference for the
    # JOB-2 row-6 wind term.  usrdef_sbc.F90:221/380 computes
    # utau(ji,jj) = znl_cbc(..., gphiu(ji,jj)) DIRECTLY at the U point -- there
    # is no 2-cell average in NEMO, so comparing legoESM's cell-centred tau
    # against this requires legoESM's own interpolation and that choice must be
    # stated wherever it is used.  Dumped at :432 WRITE(8810), FULL haloed.
    "sbc_dump_utau.bin": ("now", "utau at the U point, MY_SRC "
                                 "usrdef_sbc.F90:432 WRITE(8810); computed at "
                                 ":221/:380 as znl_cbc(znds_wnd_phi, "
                                 "znds_wnd_val, gphiu(ji,jj)) for THIS step's "
                                 "sbc call -> \"now\" (utau_b is the previous "
                                 "step's copy).  Stress ON the ocean [Pa], "
                                 "positive eastward.  FULL haloed jpi x jpj."),

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
    # registered "now" -- the level of the reconcile that consumes it, i.e.
    # legoESM's "transport_avg" target.  (The DINO kamm_mlf card no longer
    # SELECTS that target: since #1455 R6 it ships "velocity_avg" +
    # barotropic_after_reconcile="nemo_mlf_baro_corr", because NEMO undoes this
    # NOW-level reconcile at stpmlf.F90:787-790 before committing.  The
    # registration of THIS dump is unaffected -- it records which NEMO level
    # the artifact carries, not which target legoESM selects.)  Written as a
    # SINGLETON (first-step-only, ll_spg_dump=kt==nit000) at :1046 WRITE(8862).
    # Units: [m^2/s] transport per unit width = <SUM_k e3u*u>_substep (map to
    # full transport [m^3/s] via *e2u).
    # --- #1455 post-tendency stage bisection: the PRIMARY barotropic velocity
    # the split-explicit loop commits, and the two momentum STATE checkpoints
    # that bracket the implicit vertical solve.  All three read verbatim from
    # the instrumented DINO build's own source, not inferred.
    #
    # puu_b/pvv_b are written from the Kaa slot explicitly:
    #   WRITE(8859) ( ( puu_b(ji,jj,Kaa), ...   -- dynspg_ts.F90:1043
    #   WRITE(8860) ( ( pvv_b(ji,jj,Kaa), ...   -- dynspg_ts.F90:1044
    # (singleton, ll_spg_dump = kt==nit000; OPEN at :1033/:1035).  Under DINO's
    # ll_bt_av=.TRUE. (nn_bt_flt=2) this is the boxcar-weighted substep mean
    # already divided by r1_wgt1s -- a VELOCITY [m/s], unlike un_adv above,
    # which is a transport per unit width.  This is legoESM's "velocity_avg"
    # reconcile target and the quantity mlf_baro_corr installs.
    "spg_dump_puu_b_final.bin": ("after", "dynspg_ts.F90:1043 WRITE(8859) "
        "puu_b(ji,jj,Kaa); boxcar substep-mean barotropic velocity [m/s] "
        "committed at the AFTER level (OPEN at :1033, ll_spg_dump singleton)."),
    "spg_dump_pvv_b_final.bin": ("after", "dynspg_ts.F90:1044 WRITE(8860) "
        "pvv_b(ji,jj,Kaa); v twin of spg_dump_puu_b_final.bin."),
    "spg_dump_pssh_final.bin": ("after", "dynspg_ts.F90:1045 WRITE(8861) "
        "pssh(ji,jj,Kaa); the barotropic loop's committed AFTER sea surface "
        "height [m]."),
    # stp_dump_state_and_bt (cfgs/DINO/MY_SRC/stpmlf.F90:885-930) writes the
    # momentum STATE -- not an RHS -- at the level it is handed, and both call
    # sites hand it Naa:
    #   CALL stp_dump_state_and_bt( kstp, 7, 'dynspg', uu(:,:,:,Naa), ... )
    #                                                       -- stpmlf.F90:337
    #   CALL stp_dump_state_and_bt( kstp, 8, 'dynzdf', uu(:,:,:,Naa), ... )
    #                                                       -- stpmlf.F90:403
    # Filenames are built at :907-908 (u/v) and :920-921 (ub/vb) as
    # stp_dump_<NN>_<tag>_kt<KT8>_<field>.bin -- note the kt comes BEFORE the
    # field, which is the opposite order from the seq_dump_ families, so these
    # cannot ride the _SEQDUMP_KT_WINDOWS expansion and are registered per kt.
    # Stage 7 carries Nrhs's contents because Nrhs==Naa in this build; that
    # identity is what s17_dynzdf_bracket.py's control C1 proves bit-exactly.
    # Both are gated by MOD(kt-nit000, nn_stpdump_every) at :903.
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
    # CENTRED branch, dynspg_ts.F90:573; proven by un_e[jn=1] == un_e_init to
    # 0.0e0).  Registered "before" for the SAME reason as
    # cor2d_dump_ua_e_in_substep1.bin above (the sub-cycle seed is the Kbb
    # barotropic transport); this dump is a superset (all icycle substeps, not
    # just jn=1).  jpi x jpj = 56 x 203, per-substep records, fp64.
    "substep_dump.bin": ("before", "dynspg_ts.F90:926-940 WRITE(799) per-jn "
        "barotropic sub-cycle fields (sshn_e/ssha_e/zsshp2_e/un_e/vn_e/ua_e/"
        "va_e); un_e[jn=1] == puu_b(:,:,Kbb) (dynspg_ts.F90:573, ln_bt_fw=F "
        "CENTRED seed), same Kbb sub-cycle seed as cor2d_dump_ua_e_in_substep1.bin"),
    # --- #1226 SEQDUMP intra-step seam walk (RUN_SEQDUMP_Y20_1R, y20 kt=
    # 230401..230404). WRITE sites in MY_SRC/stpmlf.F90 (oracle commit
    # ebeb8a5), each proved by reading the CALL that produces the dumped
    # field one line above the WRITE (Rule 1d cite = the producing CALL, not
    # the WRITE).
    # The three r3?_f entries below cite their WRITE by UNIT NUMBER ONLY --
    # no line number. Those units are unique in the file, so the citation
    # survives drift; a line number here does not, and cannot currently be
    # checked. The pinned revision ebeb8a5 is not an object in the NEMO
    # repository and no copy on disk matches the cited layout, so the only
    # readable file is the working copy, which has moved. It moved by
    # DIFFERENT amounts in different blocks -- the post-lbc CALL below
    # shifted +17 (562 -> 579) while its own WRITE shifted +28 (576 -> 604)
    # -- so an offset measured at one anchor cannot be carried across a
    # block boundary to date a WRITE nine lines away. A previous edit did
    # exactly that and moved these three from :450/:451/:452 to
    # :451/:452/:453; that derivation is RETRACTED as circular (it assumed
    # the very block alignment in question) and neither trio is established.
    # Recover the pinned source from whatever archive produced the dump
    # binaries if a line number is ever needed again. These localise WHERE the barotropic/longitude-uniform
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
        "r3t_f,...) from ssh_atf-filtered ssh -> WRITE(8937); r3t_f."),
    "seq_dump_r3u_f": ("after", "stpmlf.F90:442 dom_qco_r3c filtered -> "
        "WRITE(8938); r3u_f u-point twin of seq_dump_r3t_f."),
    "seq_dump_r3v_f": ("after", "stpmlf.F90:442 dom_qco_r3c filtered -> "
        "WRITE(8939); r3v_f v-point twin of seq_dump_r3t_f."),
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
# The DAY-180 window (kt=5761..5764, RUN_SEQDUMP_D180_1R) is the SAME dump
# code at a different restart: the instrumented binary md5 ea0c113c writes
# each family from the identical WRITE site already cited in the entry above,
# only the kt in the filename changes.  This is therefore a kt-range
# extension of EXISTING registrations, not a new registration class -- no new
# call site is being asserted, so no new file:line proof is owed.  Keep the
# two windows in one tuple so a third window cannot be added without noticing
# the fail-closed lookup does not strip the suffix.
_SEQDUMP_KT_WINDOWS = (range(230401, 230405),   # y20  RUN_SEQDUMP_Y20_1R
                       range(5761, 5765))       # d180 RUN_SEQDUMP_D180_1R
_SEQDUMP_BASES = {k: v for k, v in _DUMP_TIME_LEVEL.items()
                  if k.startswith(("seq_dump_", "r3c_dump_"))}
for _b, _lv in _SEQDUMP_BASES.items():
    for _window in _SEQDUMP_KT_WINDOWS:
        for _kt in _window:
            _DUMP_TIME_LEVEL[f"{_b}_kt{_kt:08d}.bin"] = _lv
del _SEQDUMP_BASES, _b, _lv, _kt, _window

# stp_dump_state_and_bt's filename puts the kt BEFORE the field name
# (stpmlf.F90:907-908, :920-921), so these cannot ride the expansion above.
# Same two windows, same instrumented binary, same WRITE sites -- a kt-range
# extension of the registrations just above, not a new call site.
_STPDUMP_STATE = {
    ("07", "dynspg"): ("stpmlf.F90:337 CALL stp_dump_state_and_bt(kstp,7,"
                       "'dynspg',uu(:,:,:,Naa),vv(:,:,:,Naa),...) -- the "
                       "momentum state AFTER dyn_spg and BEFORE dyn_zdf; "
                       "Nrhs==Naa in this build, so it carries the "
                       "pre-dyn_zdf Krhs (written at :907-908/:920-921)."),
    ("08", "dynzdf"): ("stpmlf.F90:403 CALL stp_dump_state_and_bt(kstp,8,"
                       "'dynzdf',uu(:,:,:,Naa),vv(:,:,:,Naa)) -- the "
                       "momentum state AFTER the implicit vertical solve, "
                       "before finalize_lbc (written at :907-908)."),
}
for (_st, _tag), _src in _STPDUMP_STATE.items():
    for _fld in ("u", "v", "ub", "vb"):
        for _window in _SEQDUMP_KT_WINDOWS:
            for _kt in _window:
                _DUMP_TIME_LEVEL[
                    f"stp_dump_{_st}_{_tag}_kt{_kt:08d}_{_fld}.bin"] = (
                        "after", _src)
del _STPDUMP_STATE, _st, _tag, _src, _fld, _kt, _window
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

# NEMO testcase campaign whole-step RK3 records.  Lane 2 reuses lane 1's
# byte-for-byte MY_SRC instrumentation, but every emitted basename is
# registered here so gates cannot manufacture a time-level label from the
# filename.  stprk3.F90 writes the step-entry Nbb state at :92-100; the stage
# writer at :365-373 writes the explicit Kaa supplied after each stage.  The
# transport writer records the Kmm operands consumed at stprk3_stg.F90:257-319.
# The RHS and barotropic-frame writers capture their explicitly passed levels
# at stprk3.F90:344-389.  Header checks in the testcase gate independently pin
# the numeric Kaa/Kmm/Nrhs indices; these semantic labels select the comparison
# role and deliberately fail closed for any new basename.
# The ENTRY writer's OWN range, read off the card that runs it:
# cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90 gates it on
# ``kstp >= nit000 .AND. kstp <= nit000 + 59``, so a run longer than ten steps
# writes SIXTY of these and every one is the same whole-step Nbb entry state.
# The registry covered only ten, so a GYRE year's own days 0..10 -- already on
# disk -- could not be read at all.  NOTE the citation: the vendored copy at
# scripts/.../nemo502_MY_SRC/stprk3.F90:90 is lane 1's, whose gate is
# ``nit000 .OR. midpoint .OR. nitend``; the WRITE statement is byte-identical
# but the two gates are not, so the lane-2 range is cited from the CARD.
_STEP_ENTRY_LAST_KT = 60
for _kt in range(1, _STEP_ENTRY_LAST_KT + 1):
    _step = f"{_kt:08d}"
    _DUMP_TIME_LEVEL[f"oracle_step_entry_kt{_step}.bin"] = (
        "before",
        "cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90-101 (lane 2) and "
        "scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/"
        "stprk3.F90:92-100 (lane 1) write ts/uu/vv/ssh(...,Nbb) at whole-step "
        "entry; lane 2's gate is kstp <= nit000+59",
    )
for _kt in range(1, 11):
    _step = f"{_kt:08d}"
    _DUMP_TIME_LEVEL[f"oracle_bt_frames_kt{_step}.bin"] = (
        "after",
        "scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/"
        "stprk3.F90:344-352 writes uu_b/vv_b at the explicit klevel=Naa "
        "plus composed un_adv/vn_adv after stp_2D; NEMO stprk3.F90:213 "
        "swaps Naa into Nbb, making this pair the next-step seed",
    )
for _stage, _kaa, _kmm in ((1, 3, 1), (2, 2, 3), (3, 3, 2)):
    _DUMP_TIME_LEVEL[f"oracle_stage_kt00000001_s{_stage}.bin"] = (
        "after",
        "scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/"
        f"stprk3.F90:365-373 writes the stage-{_stage} Kaa={_kaa} state",
    )
    _DUMP_TIME_LEVEL[f"oracle_transport_kt00000001_s{_stage}.bin"] = (
        "now",
        "scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/"
        f"stprk3_stg.F90:257-319 writes stage-{_stage} Kmm={_kmm} "
        "advecting transports",
    )
    _DUMP_TIME_LEVEL[f"oracle_rkstage_ww_kt00000001_s{_stage}.bin"] = (
        "now",
        "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_l2_gyre_round21_oracle/traadv_round21.patch "
        f"writes stage-{_stage} Kmm={_kmm} ww after the executed "
        "tra_adv_trp wzv(np_transport) call at src/OCE/TRA/traadv.F90:220-235",
    )
_DUMP_TIME_LEVEL["oracle_rhs_kt00000001.bin"] = (
    "now",
    "scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/"
    "stprk3.F90:205-206,377-389 writes the explicitly passed Nrhs=3 "
    "momentum state after stp_2D and before the three RK stages",
)
_DUMP_TIME_LEVEL["oracle_rkstage2_terms_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3/MY_SRC/stprk3_stg.F90:339-367 writes the stage-2 "
    "Krhs accumulator before and after dyn_hpg/dyn_vor/dyn_adv",
)
_DUMP_TIME_LEVEL["oracle_rktracer_stage3_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3/MY_SRC/stprk3_stg.F90:565-713 writes the stage-3 "
    "Krhs accumulator in source order, with explicit Kbb/Kmm/Kaa headers",
)
for _stage, _kmm in ((1, 1), (2, 3)):
    _DUMP_TIME_LEVEL[f"oracle_rktracer_operands_kt00000001_s{_stage}.bin"] = (
        "now",
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
        "GYRE_OMIP_L2_P3/MY_SRC/stprk3_stg.F90:606-716 writes the "
        f"stage-{_stage} Kmm={_kmm} tracer Krhs checkpoints and the "
        "Kbb/Kmm/Kaa update operands",
    )
_DUMP_TIME_LEVEL["oracle_rkstage1_transport_operands_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3/MY_SRC/stprk3_stg.F90 writes the stage-1 "
    "e2u/e3u/uu/zub/umask/zFu and e1v/e3v/vv/zvb/vmask/zFv operands "
    "immediately after stprk3_stg.F90:265-278 materializes zFu/zFv",
)
_DUMP_TIME_LEVEL["oracle_stage1_wzv_operands_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_OMIP_L4/MY_SRC/traadv.F90 writes stage-1 Kmm=1 pFu/pFv, "
    "Kbb/Kaa QCO stretch operands, post-wzv ww and the resulting pFw; "
    "the binary header carries Kbb=1,Kmm=1,Kaa=3",
)
_DUMP_TIME_LEVEL["oracle_bt_advmean_operands_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3/MY_SRC/dynspg_ts.F90:695-698,928-939 writes "
    "the kt=1 substep transport accumulator and its normalized NOW-level "
    "un_adv/vn_adv handoff",
)
_DUMP_TIME_LEVEL["oracle_bt_advmean_operands_kt00000002.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:559-572,"
    "797-817 forms the substep transport accumulator and its normalized "
    "NOW-level un_adv/vn_adv handoff; the round-73 source card widens only "
    "this WRITE-only stream to kt=2",
)
_DUMP_TIME_LEVEL["oracle_bt_uamid_operands_kt00000002.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:484-493 "
    "forms kt=2 ua_e from un_e/ub_e/ubb_e inside the NOW-step external loop; "
    "the round-76 WRITE-only patch records those operands at that statement",
)
_DUMP_TIME_LEVEL["oracle_bt_step_operands_kt00000002.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:"
    "481-795 executes the kt=2 NOW-step external recurrence; the round-81 "
    "WRITE-only patch records its current/history/midpoint, continuity, "
    "pressure, trend, forcing, update, and swap operands in source order",
)
_DUMP_TIME_LEVEL["oracle_bt_step_operands_kt00001081.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90 "
    "records the step-1081 current/history/midpoint, continuity, pressure, "
    "trend, forcing, update, swap, and final pssh boundaries in compiled "
    "source order",
)
_DUMP_TIME_LEVEL["oracle_stage1_qco_operands_kt00001081.bin"] = (
    "after",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stprk3_stg.f90 records "
    "stage-1 ssha copied from the completed external solve and the direct "
    "after-step r3ta result of dom_qco_r3c_RK3",
)
_DUMP_TIME_LEVEL["oracle_slow_forcing_split_kt00001081.bin"] = (
    "before",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:289-325 "
    "copies the step-1081 Ue_rhs/Ve_rhs, evaluates dyn_cor_2D on the "
    "Kmm=Nbb step-entry depth means, and records both operands plus the "
    "masked frozen-forcing result around the executing subtraction",
)
_DUMP_TIME_LEVEL["oracle_bt_ordered_operands_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3_SM/MY_SRC/dynspg_ts.F90:599-917 writes the kt=1 "
    "substep-1/2 NOW-level external-mode histories, continuity, face-depth, "
    "pressure-gradient, trend, forcing, and velocity-update operands",
)
_DUMP_TIME_LEVEL["oracle_tracer_transport_kt00000001_s3.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3/MY_SRC/stprk3_stg.F90:550-565 writes stage-3 "
    "zFu/zFv/zFw after tra_adv_trp and immediately before tra_adv",
)
_DUMP_TIME_LEVEL["oracle_tracer_transport_kt00000002_s3.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/"
    "stprk3_stg.f90:792-834 calls tra_adv_trp on the live stage-3 Kmm "
    "state and writes its zFu/zFv/zFw immediately afterward",
)
_DUMP_TIME_LEVEL["oracle_rkstage2_ene_operands_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3/MY_SRC/dynvor.F90:448-535 writes the stage-2 "
    "Kmm=3 post-division zwz and e3u/e3v(Kmm) transport operands inside "
    "the executing np_CRV vor_ene call",
)
_DUMP_TIME_LEVEL["oracle_rkstage2_hpg_operands_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3/MY_SRC/stprk3_stg.F90:348-366 writes the stage-2 "
    "Kmm=3 rhd, e3w(Kmm), and gdept_z0(Kmm) immediately after eos and "
    "before the executing dyn_hpg call",
)
_DUMP_TIME_LEVEL["oracle_rkstage2_eos_operands_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3/MY_SRC/eosbn2.F90:260-348 writes the stage-2 "
    "Knn=Kmm=3 pts/gdept inputs and each local EOS intermediate from the "
    "executing eos_insitu call immediately before dyn_hpg",
)
_DUMP_TIME_LEVEL["oracle_rkstage3_wzv_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "GYRE_OMIP_L2_P3/MY_SRC/traadv.F90 writes the stage-3 Kmm vertical "
    "transport at the live wzv -> wAimp -> e1e2t*ww boundaries cited by "
    "src/OCE/TRA/traadv.F90:220-226",
)
_DUMP_TIME_LEVEL["oracle_rkstage3_preldf_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/stprk3_stg.F90:400 "
    "calls dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs ) inside CASE ( 3 ); the "
    "round-29 instrument writes uu/vv(:,:,:,Krhs) immediately before that "
    "line, and stprk3_stg.F90:218 fixes stage 3 as Kbb = N, Kmm = N+1/2, so "
    "the momentum operands accumulated into that Krhs are the stage's LIVE "
    "Kmm ones (eos/dyn_hpg/dyn_vor/dyn_adv at :324-333, all on Kmm)",
)
_DUMP_TIME_LEVEL["oracle_zdf_matrix_kt00000001.bin"] = (
    "now",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/stprk3_stg.F90:430 "
    "calls dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa ) at kstg == 3; the "
    "round-29 instrument writes that call's operands from inside "
    "src/OCE/DYN/dynzdf.F90, whose avm/e3uw(Kmm) operands are the stage's "
    "LIVE Kmm ones (dynzdf.F90:182-195)",
)
del _kt, _step, _stage, _kaa, _kmm


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
