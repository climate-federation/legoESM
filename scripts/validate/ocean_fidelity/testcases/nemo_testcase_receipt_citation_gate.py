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
    "stprk3.F90:186": "CALL stp_2D",
    "stprk3.F90:195": "stp_RK3_stg( 1",
    "stprk3.F90:197": "Nnn  = Naa",
    "stprk3.F90:200": "stp_RK3_stg( 2",
    "stprk3.F90:207": "stp_RK3_stg( 3",
    "MY_SRC/stprk3.F90:204": "CALL stp_2D",
    "MY_SRC/stprk3.F90:206": "l1_dump_rhs",
    "MY_SRC/stprk3.F90:215": "stp_RK3_stg( 1",
    "stp2d.F90:128": "CALL dyn_hpg",
    "stp2d.F90:131": "CALL dyn_ldf",
    "stp2d.F90:146": "CALL dyn_vor",
    "stp2d.F90:172": "dyn_adv_up3",
    "stp2d.F90:185": "Ue_rhs(ji,jj) + SUM",
    "stp2d.F90:196": "dyn_drg_init",
    "stp2d.F90:199-202": "utauU",
    "stp2d.F90:200": "utauU",
    "stp2d.F90:281": "CALL dyn_spg_ts",
    "stprk3_stg.F90:65": "SUBROUTINE stp_RK3_stg",
    "stprk3_stg.F90:262,267": "zub(ji,jj)",
    "stprk3_stg.F90:309-334": "CASE ( 2 , 3 )",
    "stprk3_stg.F90:315": "CALL dyn_adv",
    "stprk3_stg.F90:315-430": "CALL dyn_zdf",
    "stprk3_stg.F90:324": "dyn_hpg",
    "stprk3_stg.F90:324-378": "dyn_hpg",
    "stprk3_stg.F90:327": "dyn_vor",
    "stprk3_stg.F90:331": "dyn_adv",
    "stprk3_stg.F90:333": "zFu, zFv, zFw",
    "stprk3_stg.F90:367-368": "uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb)",
    "stprk3_stg.F90:373-378": "1._wp + r3u(ji,jj,Kaa)",
    "stprk3_stg.F90:437-446": "barotropic velocity correction",
    "stprk3_stg.F90:453-601": "tra_zdf",
    "stprk3_stg.F90:463-599": "tra_zdf",
    "stprk3_stg.F90:655": "END SUBROUTINE stp_RK3_stg",
    "dynspg_ts.F90:282": "zu_frc(:,:) =   Ue_rhs(:,:)",
    "dynspg_ts.F90:296": "dyn_cor_2D",
    "dynspg_ts.F90:303": "#else",
    "dynspg_ts.F90:344-345": "puu(ji,jj,jk,Krhs) = ( puu(ji,jj,jk,Krhs) - zu_frc",
    "dynspg_ts.F90:487": "un_e  (:,:) =    puu_b(:,:,Kmm)",
    "dynspg_ts.F90:735-761": "ua_e(ji,jj)",
    "dynspg_ts.F90:750-753": "ua_e(ji,jj)",
    "dynspg_ts.F90:752-761": "ua_e(ji,jj)",
    "dynspg_ts.F90:825-847": "puu_b  (:,:,Kaa) + za1 * ua_e",
    "dynspg_ts.F90:870-895": "puu_b(ji,jj,Kaa) = puu_b(ji,jj,Kaa) /",
    "dynspg_ts.F90:910": "#else",
    "dynspg_ts.F90:938-975": "puu(:,:,jk,Krhs) = puu(:,:,jk,Krhs)",
    "dynhpg.F90:359": "puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap",
    "dynhpg.F90:359,383": "zhpi(ji,jj) + zuap",
    "dynhpg.F90:383": "puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap",
    "dynadv.F90:144": "keg + zad + vor",
    "domzgr_substitute.h90:139": "define  gdept(",
    "domzgr_substitute.h90:145": "define  gdept_z0(",
    "domzgr_substitute.h90:139,145": "gdept",
    "oce.F90:39,99": "uu_b",
    "DOM/istate.F90:149-155": "uu_b(ji,jj,Kbb) + e3u",
    "usrdef_hgr.F90:103-104": "pff_f(:,:) = 0._wp",
    "trabbl.F90:519-527": "mgrhu(ji,jj) = INT(  SIGN",
    "namelist_cfg:85": "ln_dynadv_up3",
    "namelist_cfg:105-106": "rn_bt_alpha",
    "namelist_ref:1092": "nn_bt_flt",
    "namelist_ref:1177": "ln_zad_Aimp",
    "ocean.output:875": "nn_bt_flt",
    "ocean_pe_latlon_cgrid.py:2005-2006": "gdept_z0",
    "ocean_pe_latlon_cgrid.py:2045-2046": "dp_dx_sco",
    "nemo_testcase_recipe.py:92,274,914": "nemo_sco",
    # round 27
    "lock_kt1_10/ocean.output:615": "ln_dynldf_OFF",
    "overflow_kt1_10/ocean.output:727": "ln_dynldf_OFF",
    "BLD/ppsrc/nemo/dynspg_ts.f90:1224": "INTENT(in",
    "BLD/ppsrc/nemo/dynhpg.f90:393,412": "= zhpi(ji,jj) + zuap",
    "BLD/ppsrc/nemo/dynadv_up3.f90:138-141": "pUe(ji,jj) = 0._wp",
    "BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355": "ELSE",
    "BLD/ppsrc/nemo/dynadv_up3.f90:213,336,357": "puu(ji,jj",
    "BLD/ppsrc/nemo/dynvor.f90:655-656": "zwx(ji,jj) = e2u(ji,jj)",
    "BLD/ppsrc/nemo/dynvor.f90:665": "pu_rhs(ji,jj,jk) + zuav",
    "stprk3_stg.F90:453-598": "tra_zdf",
    "stp2d.F90:126": "hydrostatic pressure gradient",
    "stprk3_stg.F90:168-243": "r3v(:,:,Kaa)",
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


def check(citation: str, symbol: str, shift: int = 0) -> dict:
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
    text = "\n".join(body[n - 1] for n in numbers)
    if symbol in text:
        return {"citation": citation, "status": "OK", "symbol": symbol}
    return {"citation": citation, "status": "SYMBOL-NOT-AT-LINE",
            "symbol": symbol, "detail": text.strip()[:160]}


def run(receipt: Path, heading: str, *, plant: str | None = None) -> dict:
    document = receipt.read_text()
    if heading not in document:
        raise SystemExit(f"heading {heading!r} not in {receipt}")
    citations = extract(document[document.index(heading):])
    unmapped = [c for c in citations if c not in CITATION_MAP]
    rows = [check(c, CITATION_MAP[c], shift=2 if c == plant else 0)
            for c in citations if c in CITATION_MAP]
    bad = [row for row in rows if row["status"] != "OK"]
    return {
        "format": "nemo-testcase-receipt-citation-gate-v1",
        "receipt": str(receipt),
        "from_heading": heading,
        "citations_found": len(citations),
        "unmapped_citations": unmapped,
        "failures": bad,
        "planted_shift": plant,
        "status": "PASS" if not unmapped and not bad else "FAIL",
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
