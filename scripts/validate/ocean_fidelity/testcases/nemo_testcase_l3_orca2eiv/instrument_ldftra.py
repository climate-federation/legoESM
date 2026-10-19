#!/usr/bin/env python3
"""Add the write-only L3 EIV transport observer to NEMO ``ldftra``.

The transform is deliberately exact-context and refuses an anchor immediately
following a continued Fortran statement.  It works on both ``ldftra.F90`` and
NEMO's preprocessed ``ldftra.f90`` so preflight can syntax-check what the
compiler will see without creating a configuration.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MAGIC = "NEMO_L3_EIV_1   "
TAGS = ("BEFORE_U        ", "BEFORE_V        ", "AFTER_U         ",
        "AFTER_V         ", "DELTA_U         ", "DELTA_V         ")

DECL = """      INTEGER  ::   l3_unit, l3_ios
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE :: l3_before_u, l3_before_v
"""
BEFORE = """      IF( kt == kit000 .AND. lwp .AND. .NOT. l_istiled ) THEN
         ALLOCATE( l3_before_u(LBOUND(pFu,1):UBOUND(pFu,1),LBOUND(pFu,2):UBOUND(pFu,2),LBOUND(pFu,3):UBOUND(pFu,3)) )
         ALLOCATE( l3_before_v(LBOUND(pFv,1):UBOUND(pFv,1),LBOUND(pFv,2):UBOUND(pFv,2),LBOUND(pFv,3):UBOUND(pFv,3)) )
         l3_before_u(:,:,:) = pFu(:,:,:)
         l3_before_v(:,:,:) = pFv(:,:,:)
      ENDIF
"""
AFTER = f"""      IF( kt == kit000 .AND. lwp .AND. .NOT. l_istiled ) THEN
         OPEN( NEWUNIT=l3_unit, FILE='oracle_l3_eiv_kt00000001.bin', ACCESS='STREAM', &
            &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=l3_ios )
         IF( l3_ios /= 0 ) CALL ctl_stop( 'L3 EIV observer: open failed' )
         WRITE(l3_unit) '{MAGIC}'
         WRITE(l3_unit) 1, kt, Kmm, Krhs, SIZE(pFu,1), SIZE(pFu,2), SIZE(pFu,3), 64, 6
         WRITE(l3_unit) '{TAGS[0]}' ; WRITE(l3_unit) l3_before_u
         WRITE(l3_unit) '{TAGS[1]}' ; WRITE(l3_unit) l3_before_v
         WRITE(l3_unit) '{TAGS[2]}' ; WRITE(l3_unit) pFu
         WRITE(l3_unit) '{TAGS[3]}' ; WRITE(l3_unit) pFv
         WRITE(l3_unit) '{TAGS[4]}' ; WRITE(l3_unit) pFu - l3_before_u
         WRITE(l3_unit) '{TAGS[5]}' ; WRITE(l3_unit) pFv - l3_before_v
         CLOSE(l3_unit)
         DEALLOCATE(l3_before_u, l3_before_v)
      ENDIF
"""

DECL_ANCHOR = "      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   ztrpu, ztrpv\n"
BEFORE_ANCHOR = "      !!----------------------------------------------------------------------\n      !\n"
AFTER_ANCHOR = "      !                              ! diagnose the eddy induced velocity and associated heat transport\n"


def _insert_once(text: str, anchor: str, insertion: str, where: str) -> str:
    if text.count(anchor) != 1:
        raise ValueError(f"{where} anchor count is {text.count(anchor)}, expected 1")
    pos = text.index(anchor)
    previous = text[:pos].rstrip().splitlines()[-1]
    if previous.rstrip().endswith("&"):
        raise ValueError(f"{where} anchor follows a continued '&' statement")
    return text.replace(anchor, anchor + insertion, 1)


def instrument(text: str) -> str:
    if "oracle_l3_eiv_kt00000001.bin" in text:
        raise ValueError("ldftra is already instrumented")
    start_marker = "SUBROUTINE ldf_eiv_trp_RK3_t("
    end_marker = "END SUBROUTINE ldf_eiv_trp_RK3_t"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise ValueError("RK3 EIV transport routine is not unique")
    start = text.index(start_marker)
    end = text.index("\n", text.index(end_marker, start)) + 1
    prefix, routine, suffix = text[:start], text[start:end], text[end:]
    routine = _insert_once(routine, DECL_ANCHOR, DECL, "declaration")
    routine = _insert_once(routine, BEFORE_ANCHOR, BEFORE, "before-copy")
    # Place the writer immediately before diagnostics, after all EIV additions.
    if routine.count(AFTER_ANCHOR) != 1:
        raise ValueError("after-update anchor is not unique")
    pos = routine.index(AFTER_ANCHOR)
    previous = routine[:pos].rstrip().splitlines()[-1]
    if previous.rstrip().endswith("&"):
        raise ValueError("after-update anchor follows a continued '&' statement")
    routine = routine.replace(AFTER_ANCHOR, AFTER + AFTER_ANCHOR, 1)
    return prefix + routine + suffix


def validate_literals(text: str) -> None:
    expected = (MAGIC, *TAGS)
    found = re.findall(r"WRITE\(l3_unit\) '([^']*)'", text)
    if found != list(expected):
        raise ValueError(f"observer character frames differ: {found!r}")
    bad = [value for value in found if len(value) != 16]
    if bad:
        raise ValueError(f"observer character literal is not exactly 16 bytes: {bad!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plant-after-continuation", action="store_true")
    parser.add_argument("--plant-short-literal", action="store_true")
    args = parser.parse_args()
    text = args.source.read_text()
    if args.plant_after_continuation:
        text = text.replace(DECL_ANCHOR, "      l3_plant = 1 &\n" + DECL_ANCHOR, 1)
    out = instrument(text)
    if args.plant_short_literal:
        out = out.replace(MAGIC, MAGIC[:-1], 1)
    validate_literals(out)
    args.output.write_text(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
