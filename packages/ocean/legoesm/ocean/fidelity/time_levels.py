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
}


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
