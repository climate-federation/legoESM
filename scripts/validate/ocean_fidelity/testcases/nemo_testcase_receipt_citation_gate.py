#!/usr/bin/env python3
"""Every ``file:line`` cited in the receipt must point at the symbol named.

Three rounds in a row produced a wrong-citation finding -- round 25 corrected
NEMO line numbers by hand, round 26 corrected round 25's, and round 27
corrected round 26's (``stp2d.F90:126/:129/:190/:279`` were a comment banner,
a blank line and two off-by-two references, and a code comment called
``stprk3_stg.F90:168-243`` "tracers" when it is the ``r3t/r3u/r3v`` block).
Hand-checking has now failed three times, so this gate replaces it.

How it works, and what it can and cannot see:

* it reads the receipt from a named heading to the end of the file, drops
  fenced code blocks, and walks the inline code spans IN ORDER;
* a span of the form ``<file>:<lines>`` is a citation and also sets the
  current file; a span that is a bare ``<file>`` sets the current file; a span
  of the form ``:<lines>`` is a citation against the current file.  That is
  how the prose reads, and binding continuations wrongly is itself a defect
  the gate would report;
* every extracted citation must appear in ``CITATION_MAP``, which pins the
  RESOLVED path (the same basename exists in ``src/OCE`` and in several
  ``MY_SRC`` overrides) and the symbol the prose names.  An unmapped citation
  is a FAILURE, so a new claim cannot enter the receipt uninspected;
* each mapped symbol must occur in the cited line range of the real file.

Blind spots, written down per Rule 2: it checks that the named symbol IS at
the cited line, not that the line means what the prose says about it; it
cannot see a citation the prose renders without backticks; and a mapped
symbol that is too generic (a bare ``!``) would pass vacuously, which is why
every entry below names an identifier or a distinctive fragment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
REPO = Path(__file__).resolve().parents[4]
LOCK = NEMO / "tests/LOCK_EXCHANGE_OMIP_L1_P3"
OVERFLOW_RUN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10")

# path anchors, so one basename cannot silently resolve to a MY_SRC override
_OCE = NEMO / "src/OCE"
_DYN = _OCE / "DYN"
FILES = {
    "stprk3.F90": _OCE / "stprk3.F90",
    "stprk3_stg.F90": _OCE / "stprk3_stg.F90",
    "stp2d.F90": _OCE / "stp2d.F90",
    "oce.F90": _OCE / "oce.F90",
    "DOM/istate.F90": _OCE / "DOM/istate.F90",
    "domzgr_substitute.h90": _OCE / "DOM/domzgr_substitute.h90",
    "dynspg_ts.F90": _DYN / "dynspg_ts.F90",
    "dynhpg.F90": _DYN / "dynhpg.F90",
    "dynadv.F90": _DYN / "dynadv.F90",
    "trabbl.F90": _OCE / "TRA/trabbl.F90",
    "MY_SRC/stprk3.F90": LOCK / "MY_SRC/stprk3.F90",
    "domain.F90": _OCE / "DOM/domain.F90",
    "usrdef_hgr.F90": LOCK / "MY_SRC/usrdef_hgr.F90",
    "namelist_cfg": LOCK / "EXP00/namelist_cfg",
    "namelist_ref": NEMO / "cfgs/SHARED/namelist_ref",
    "ocean.output": OVERFLOW_RUN / "ocean.output",
    "overflow_kt1_10/ocean.output": OVERFLOW_RUN / "ocean.output",
    "lock_kt1_10/ocean.output": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10"
        "/ocean.output"),
    # The PREPROCESSED source LOCK compiles.  OVERFLOW's is the same body with
    # its own line offsets, which is why the eligibility gate greps both cards
    # programmatically instead of citing OVERFLOW's numbers here.
    "BLD/ppsrc/nemo/dynspg_ts.f90": LOCK / "BLD/ppsrc/nemo/dynspg_ts.f90",
    "BLD/ppsrc/nemo/dynhpg.f90": LOCK / "BLD/ppsrc/nemo/dynhpg.f90",
    "BLD/ppsrc/nemo/dynadv_up3.f90": LOCK / "BLD/ppsrc/nemo/dynadv_up3.f90",
    "BLD/ppsrc/nemo/dynvor.f90": LOCK / "BLD/ppsrc/nemo/dynvor.f90",
    "ocean_pe_latlon_cgrid.py":
        REPO / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py",
    "nemo_testcase_recipe.py":
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py",
}

# citation -> the symbol the receipt's prose says lives there.
CITATION_MAP = {
    # value: one symbol (checked on the FIRST and LAST cited line -- the same
    # line for a single-line citation), or [first_symbol, last_symbol].
    # Both endpoints are pinned so a shifted line number cannot pass; the
    # audit below REFUSES any entry that survives a +/-1 or +/-2 shift, which
    # is what forces a symbol specific enough to identify the line.
    "stprk3.F90:186": "CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )",
    "stprk3.F90:195": "stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )",
    "stprk3.F90:197": "Nrhs = Nnn   ;   Nnn  = Naa   ;   Naa  = Nrhs",
    "stprk3.F90:200": "stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )",
    "stprk3.F90:207": "stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )",
    "MY_SRC/stprk3.F90:204": "CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )",
    "MY_SRC/stprk3.F90:206": "CALL l1_dump_rhs( kstp, Nrhs )",
    "MY_SRC/stprk3.F90:215": "stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )",
    "stp2d.F90:126": "hydrostatic pressure gradient (HPG))",
    "stp2d.F90:128": "CALL dyn_hpg( kt, Kbb",
    "stp2d.F90:131": "CALL dyn_ldf( kt, Kbb, Kbb, uu, vv, Krhs )",
    "stp2d.F90:146": "CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )",
    "stp2d.F90:172": "CALL dyn_adv_up3 ( kt, Kbb, Kbb, uu, vv, Krhs, pUe=Ue_rhs",
    "stp2d.F90:185": "Ue_rhs(ji,jj) + SUM( e3u_0(ji,jj,1:jpkm1)*uu(ji,jj,1:jpkm1,Krhs)",
    "stp2d.F90:196": "CALL dyn_drg_init( Kbb, Kbb, uu, vv, uu_b, vv_b, Ue_rhs",
    "stp2d.F90:199-202": ["DO_2D( 0, 0, 0, 0 )", "END_2D"],
    "stp2d.F90:200": "r1_rho0 * utauU(ji,jj) * r1_hu(ji,jj,Kbb)",
    "stp2d.F90:281": "CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )",
    "stprk3_stg.F90:65": "SUBROUTINE stp_RK3_stg( kstg, kstp, Kbb, Kmm, Krhs, Kaa )",
    "stprk3_stg.F90:262,267": [
        "zub(ji,jj) = r1_Dt * rn_Dt * un_adv(ji,jj)",
        "zub(ji,jj) = un_adv(ji,jj)*r1_hu(ji,jj,Kmm) - uu_b(ji,jj,Kmm)"],
    "stprk3_stg.F90:309-334": ["SELECT CASE( kstg )", "ENDIF"],
    "stprk3_stg.F90:315": "IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu",
    "stprk3_stg.F90:315-430": [
        "IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu",
        "IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )"],
    "stprk3_stg.F90:324": "CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )",
    "stprk3_stg.F90:324-378": [
        "CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )",
        "/           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)"],
    "stprk3_stg.F90:327": "CALL    dyn_vor( kstp,      Kmm, uu, vv, Krhs )",
    "stprk3_stg.F90:331": "CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)",
    "stprk3_stg.F90:333": "CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )",
    "stprk3_stg.F90:367-368": [
        "uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) )",
        "vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )"],
    "stprk3_stg.F90:373-378": [
        "uu(ji,jj,jk,Kaa) = (         ( 1._wp + r3u(ji,jj,Kbb) )",
        "/           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)"],
    "stprk3_stg.F90:437-446": ["#endif", "END_3D"],
    "stprk3_stg.F90:453-598": [
        "Tracers : RHS computation + time-stepping",
        "CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )"],
    "stprk3_stg.F90:453-601": [
        "Tracers : RHS computation + time-stepping", "END DO"],
    "stprk3_stg.F90:463-599": [
        "CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )",
        "IF( ln_zdfnpc  )   CALL tra_npc( kstp,      Kmm, Krhs, ts    , Kaa  )"],
    "stprk3_stg.F90:655": "END SUBROUTINE stp_RK3_stg",
    "stprk3_stg.F90:168-243": [
        "r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)",
        "Dynamic : RHS computation + time-stepping"],
    "dynspg_ts.F90:282": "zu_frc(:,:) =   Ue_rhs(:,:)",
    "dynspg_ts.F90:296": "CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )",
    "dynspg_ts.F90:303": "#else",
    "dynspg_ts.F90:344-345": [
        "DO_3D( 0, 0, 0, 0, 1, jpkm1 )",
        "puu(ji,jj,jk,Krhs) = ( puu(ji,jj,jk,Krhs) - zu_frc(ji,jj) )"],
    "dynspg_ts.F90:487": "un_e  (:,:) =    puu_b(:,:,Kmm)",
    "dynspg_ts.F90:735-761": ["DO_2D( 0, 0, 0, 0 )", "END_2D"],
    "dynspg_ts.F90:750-753": [
        "z1_hv = ssvmask(ji,jj) / ( hv_0(ji,jj) + zsshv_a(ji,jj)",
        "rDt_e * (  zhu_bck        * zu_spg (ji,jj)"],
    "dynspg_ts.F90:752-761": [
        "ua_e(ji,jj) = (               hu_e  (ji,jj) *   un_e (ji,jj)", "END_2D"],
    "dynspg_ts.F90:825-847": [
        "ELSE                                       ! Sum transports",
        "pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s"],
    "dynspg_ts.F90:870-895": [
        "IF( (.NOT.(ln_dynadv_vec .OR. lk_linssh)) .AND. ll_bt_av ) THEN",
        "CALL lbc_lnk( 'dynspg_ts', puu_b, 'U', -1._wp, pvv_b, 'V', -1._wp )"],
    "dynspg_ts.F90:910": "#else",
    "dynspg_ts.F90:938-975": [
        "IF( ln_dynadv_vec .OR. lk_linssh ) THEN",
        "* ( pvv_b(:,:,Kaa) - pvv_b(:,:,Kbb) * hv(:,:,Kbb) ) * r1_Dt"],
    "dynhpg.F90:359": "puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap",
    "dynhpg.F90:383": "puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap",
    "dynhpg.F90:359,383": [
        "puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap",
        "puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap"],
    "dynadv.F90:144": "vector form : keg + zad + vor is used",
    "domzgr_substitute.h90:139": "define  gdept(i,j,k,t)",
    "domzgr_substitute.h90:145": "gdept_z0(i,j,k,t) (gdept(i,j,k,t)-ssh(i,j,t))",
    "domzgr_substitute.h90:139,145": [
        "define  gdept(i,j,k,t)",
        "gdept_z0(i,j,k,t) (gdept(i,j,k,t)-ssh(i,j,t))"],
    "oce.F90:39,99": [
        "ssh, uu_b,  vv_b", "ALLOCATE( ssh (jpi,jpj,jpt)  , uu_b(jpi,jpj,jpt)"],
    "DOM/istate.F90:149-155": [
        "uu_b(:,:,Kbb) = 0._wp   ;   vv_b(:,:,Kbb) = 0._wp",
        "vv_b(:,:,Kbb) = vv_b(:,:,Kbb) * r1_hv(:,:,Kbb)"],
    "usrdef_hgr.F90:103-104": [
        "pff_f(:,:) = 0._wp            ! here No earth rotation: f=0",
        "pff_t(:,:) = 0._wp"],
    "trabbl.F90:519-527": ["DO_2D( 1, 0, 1, 0 )", "END_2D"],
    "domain.F90:159": "r1_hu_0(:,:) = ssumask(:,:) / ( hu_0(:,:) + 1._wp",
    "namelist_cfg:85": "ln_dynadv_up3 = .true.",
    "namelist_cfg:105-106": [
        "nn_bt_flt     = 3", "rn_bt_alpha   = 0.07"],
    "namelist_ref:1092": "nn_bt_flt     = 1          ! Add dissipation",
    "namelist_ref:1177": "ln_zad_Aimp = .false.",
    "ocean.output:875": "Barotropic time filter => nn_bt_flt",
    "lock_kt1_10/ocean.output:615": "no explicit diffusion                ln_dynldf_OFF",
    "overflow_kt1_10/ocean.output:727": "no explicit diffusion                ln_dynldf_OFF",
    "ocean_pe_latlon_cgrid.py:2005-2006": ["z_coord,", "eta_safe,"],
    "ocean_pe_latlon_cgrid.py:2045-2046": [
        "requires the raw NEMO", "nemo_e3w_0 mesh field; midpoint reconstruction on"],
    "nemo_testcase_recipe.py:92,274,914": [
        'pgf_scheme="nemo_sco",', 'if cfg.pgf_scheme != "nemo_sco":'],
    "BLD/ppsrc/nemo/dynspg_ts.f90:1224": [
        "REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in   ) ::  puu, pvv"],
    "BLD/ppsrc/nemo/dynhpg.f90:378,397": [
        "DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value",
        "DO jk= 2, jpkm1"],
    "BLD/ppsrc/nemo/dynhpg.f90:393,412": [
        "puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap",
        "puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap"],
    "BLD/ppsrc/nemo/dynadv_up3.f90:138-141": [
        "IF( PRESENT( pUe ) ) THEN     ! 3D RHS cumulation : set 2D RHS to zero",
        "END DO   ;   END DO"],
    "BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355": [
        "ELSE                           !-  added the 3D RHS  -!",
        "ELSE                                !-  added the 3D RHS  -!"],
    "BLD/ppsrc/nemo/dynadv_up3.f90:213,336,357": [
        "puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp",
        "puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - zFu_t(ji,jj) * r1_e1e2u(ji,jj)"],
    "BLD/ppsrc/nemo/dynvor.f90:655-656": [
        "zwx(ji,jj) = e2u(ji,jj) * (e3t_1d(jk)",
        "zwy(ji,jj) = e1v(ji,jj) * (e3t_1d(jk)"],
    "BLD/ppsrc/nemo/dynvor.f90:665": "pu_rhs(ji,jj,jk) = pu_rhs(ji,jj,jk) + zuav * ( zwz(ji  ,jj-1) + zwz(ji,jj) )",
}

DEFAULT_RECEIPT = (
    REPO / "docs/ocean/fidelity/testcases"
    / "nemo_testcases_l2_gyre_phase3_round8_receipt.md")
DEFAULT_HEADING = "## Round 25 —"

_NAME = (r"[A-Za-z0-9_./]+\.(?:F90|f90|h90|py|fcm)"
         r"|[A-Za-z0-9_./]*(?:namelist_cfg|namelist_ref|ocean\.output)")
_SPAN = re.compile(r"`([^`\n]+)`")
_FULL = re.compile(rf"^(?P<f>{_NAME}):(?P<l>\d[\d,\-]*)$")
_ONLY = re.compile(rf"^(?P<f>{_NAME})$")
_BARE = re.compile(r"^:(?P<l>\d[\d,\-]*)$")


def extract(text: str) -> list[str]:
    """Ordered, de-duplicated ``file:lines`` citations, continuations bound."""
    text = re.sub(r"```.*?```", "\n", text, flags=re.S)
    current: str | None = None
    found: list[str] = []
    for match in _SPAN.finditer(text):
        span = match.group(1).strip()
        if (full := _FULL.match(span)):
            current = full.group("f")
            found.append(f"{current}:{full.group('l')}")
        elif (only := _ONLY.match(span)):
            current = only.group("f")
        elif (bare := _BARE.match(span)):
            if current is None:
                found.append(f"<UNBOUND>:{bare.group('l')}")
            else:
                found.append(f"{current}:{bare.group('l')}")
    return list(dict.fromkeys(found))


def line_numbers(spec: str) -> list[int]:
    numbers: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            first, last = part.split("-")
            numbers.extend(range(int(first), int(last) + 1))
        else:
            numbers.append(int(part))
    return numbers


def check(citation: str, symbols, shift: int = 0) -> dict:
    """Pin BOTH endpoints of the citation, so a shifted line number fails.

    The first draft asked ``symbol in "\n".join(cited lines)``, which for a
    range passes if ANY line holds the symbol.  36 of 78 mapped citations then
    survived a wrong line number within +/-6 -- the gate could not catch the
    defect class it exists for.  Now the FIRST and LAST cited line each carry
    their own symbol, and :func:`audit_shift_sensitivity` refuses any entry
    that still survives a shift.
    """
    path_key, spec = citation.rsplit(":", 1)
    path = FILES.get(path_key)
    if path is None or not path.is_file():
        return {"citation": citation, "status": "UNRESOLVED",
                "detail": f"no readable file for {path_key!r}"}
    body = path.read_text(errors="replace").splitlines()
    numbers = [n + shift for n in line_numbers(spec)]
    if any(n < 1 or n > len(body) for n in numbers):
        return {"citation": citation, "status": "OUT-OF-RANGE",
                "detail": f"{path} has {len(body)} lines"}
    wanted = [symbols, symbols] if isinstance(symbols, str) else list(symbols)
    if len(wanted) == 1:
        wanted = wanted * 2
    if len(wanted) != 2:
        return {"citation": citation, "status": "BAD-MAP-ENTRY",
                "detail": "expected a symbol or [first, last]"}
    for symbol, line in zip(wanted, (numbers[0], numbers[-1])):
        if symbol not in body[line - 1]:
            return {"citation": citation, "status": "SYMBOL-NOT-AT-LINE",
                    "symbol": symbol, "line": line,
                    "detail": body[line - 1].strip()[:160]}
    return {"citation": citation, "status": "OK", "symbols": wanted}


def audit_shift_sensitivity(shifts=(-2, -1, 1, 2)) -> list[dict]:
    """Every mapped citation must FAIL under a wrong line number.

    This is the gate's own non-vacuity control, run on every invocation: an
    entry whose symbols are too generic to identify the line is reported as
    SHIFT-BLIND and fails the gate, which forces a better symbol rather than
    letting the map quietly stop checking anything.
    """
    blind = []
    for citation, symbols in CITATION_MAP.items():
        survives = [s for s in shifts
                    if check(citation, symbols, shift=s)["status"] == "OK"]
        if survives:
            blind.append({"citation": citation, "symbols": symbols,
                          "passes_at_shifts": survives})
    return blind


def run(receipt: Path, heading: str, *, plant: str | None = None) -> dict:
    document = receipt.read_text()
    if heading not in document:
        raise SystemExit(f"heading {heading!r} not in {receipt}")
    citations = extract(document[document.index(heading):])
    unmapped = [c for c in citations if c not in CITATION_MAP]
    rows = [check(c, CITATION_MAP[c], shift=2 if c == plant else 0)
            for c in citations if c in CITATION_MAP]
    bad = [row for row in rows if row["status"] != "OK"]
    blind = audit_shift_sensitivity()
    return {
        "format": "nemo-testcase-receipt-citation-gate-v1",
        "receipt": str(receipt),
        "from_heading": heading,
        "citations_found": len(citations),
        "unmapped_citations": unmapped,
        "failures": bad,
        "shift_blind_map_entries": blind,
        "planted_shift": plant,
        "status": ("PASS" if not unmapped and not bad and not blind
                   else "FAIL"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--from-heading", default=DEFAULT_HEADING)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plant", metavar="CITATION",
        help="shift this citation's line numbers by 2; the gate MUST fail, "
             "which is what proves it is not vacuous")
    args = parser.parse_args(argv)
    report = run(args.receipt, args.from_heading, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if args.plant:
        # A planted control exits NONZERO: 1 when it correctly made the gate
        # fail, 2 when it did not fire (which would mean the gate is vacuous).
        return 1 if report["status"] == "FAIL" else 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
