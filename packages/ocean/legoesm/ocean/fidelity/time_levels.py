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
}

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
