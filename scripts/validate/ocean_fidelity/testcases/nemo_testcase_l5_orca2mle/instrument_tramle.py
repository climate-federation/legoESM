#!/usr/bin/env python3
"""Add the write-only L5 MLE transport observer to NEMO ``tramle``."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MAGIC = "NEMO_L3_EIV_1   "  # shared six-frame reader layout, not a claim label
TAGS = ("BEFORE_U        ", "BEFORE_V        ", "AFTER_U         ",
        "AFTER_V         ", "DELTA_U         ", "DELTA_V         ")

DECL_ANCHOR = "      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   zstreamu, zstreamv\n"
DECL = """      INTEGER  ::   l5_unit, l5_ios
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE :: l5_before_u, l5_before_v
"""
BODY_ANCHOR = "      IF( iom_use('psiu_mle') .OR. iom_use('psiv_mle') ) THEN\n"
BEFORE = """      IF( kt == nit000 .AND. lwp .AND. .NOT. l_istiled ) THEN
         ALLOCATE( l5_before_u(LBOUND(pFu,1):UBOUND(pFu,1),LBOUND(pFu,2):UBOUND(pFu,2),LBOUND(pFu,3):UBOUND(pFu,3)) )
         ALLOCATE( l5_before_v(LBOUND(pFv,1):UBOUND(pFv,1),LBOUND(pFv,2):UBOUND(pFv,2),LBOUND(pFv,3):UBOUND(pFv,3)) )
         l5_before_u(:,:,:) = l5_canon_3d(pFu,'U')
         l5_before_v(:,:,:) = l5_canon_3d(pFv,'V')
      ENDIF
"""
AFTER_ANCHOR = "      IF (ln_osm_mle.AND.ln_zdfosm) THEN\n"
AFTER = f"""      IF( kt == nit000 .AND. lwp .AND. .NOT. l_istiled ) THEN
         OPEN( NEWUNIT=l5_unit, FILE='oracle_l5_mle_kt00000001.bin', ACCESS='STREAM', &
            &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=l5_ios )
         IF( l5_ios /= 0 ) CALL ctl_stop( 'L5 MLE observer: open failed' )
         WRITE(l5_unit) '{MAGIC}'
         WRITE(l5_unit) 1, kt, Kmm, 3, SIZE(pFu,1), SIZE(pFu,2), SIZE(pFu,3), 64, 6
         WRITE(l5_unit) '{TAGS[0]}' ; WRITE(l5_unit) l5_before_u
         WRITE(l5_unit) '{TAGS[1]}' ; WRITE(l5_unit) l5_before_v
         WRITE(l5_unit) '{TAGS[2]}' ; WRITE(l5_unit) l5_canon_3d(pFu,'U')
         WRITE(l5_unit) '{TAGS[3]}' ; WRITE(l5_unit) l5_canon_3d(pFv,'V')
         WRITE(l5_unit) '{TAGS[4]}' ; WRITE(l5_unit) l5_canon_3d(pFu,'U') - l5_before_u
         WRITE(l5_unit) '{TAGS[5]}' ; WRITE(l5_unit) l5_canon_3d(pFv,'V') - l5_before_v
         CLOSE(l5_unit)
         DEALLOCATE(l5_before_u, l5_before_v)
      ENDIF
"""
END_ANCHOR = "   END SUBROUTINE tra_mle_trp_RK3_t\n"
CANON = """
   FUNCTION l5_canon_3d( pfield, cdgrid ) RESULT( zfield )
      !! Zero-first observer: read only rank-owned wet U/V cells.
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pfield
      CHARACTER(LEN=1), INTENT(in) :: cdgrid
      REAL(wp), DIMENSION(SIZE(pfield,1),SIZE(pfield,2),SIZE(pfield,3)) :: zfield
      INTEGER :: ji, jj, jk, jif, jjf, ioff, joff
      LOGICAL :: llwet
      zfield(:,:,:) = 0._wp
      ioff = ( jpi - SIZE(pfield,1) ) / 2
      joff = ( jpj - SIZE(pfield,2) ) / 2
      DO jk = 1, SIZE(pfield,3)
         DO jj = 1, SIZE(pfield,2)
            jjf = jj + joff
            DO ji = 1, SIZE(pfield,1)
               jif = ji + ioff
               IF( jif >= 1+nn_hls .AND. jif <= jpi-nn_hls .AND. &
                  & jjf >= 1+nn_hls .AND. jjf <= jpj-nn_hls ) THEN
                  SELECT CASE( cdgrid )
                  CASE('U')   ; llwet = umask(jif,jjf,jk) /= 0._wp
                  CASE('V')   ; llwet = vmask(jif,jjf,jk) /= 0._wp
                  CASE DEFAULT; llwet = .FALSE.
                  END SELECT
                  IF( llwet ) zfield(ji,jj,jk) = pfield(ji,jj,jk)
               ENDIF
            END DO
         END DO
      END DO
   END FUNCTION l5_canon_3d
"""


def _insert(text: str, anchor: str, insertion: str, label: str) -> str:
    if text.count(anchor) != 1:
        raise ValueError(f"{label} anchor count {text.count(anchor)} != 1")
    previous = text[:text.index(anchor)].rstrip().splitlines()[-1]
    if previous.rstrip().endswith("&"):
        raise ValueError(f"{label} anchor follows continued '&'")
    return text.replace(anchor, anchor + insertion, 1)


def _insert_before(text: str, anchor: str, insertion: str, label: str) -> str:
    if text.count(anchor) != 1:
        raise ValueError(f"{label} anchor count {text.count(anchor)} != 1")
    previous = text[:text.index(anchor)].rstrip().splitlines()[-1]
    if previous.rstrip().endswith("&"):
        raise ValueError(f"{label} anchor follows continued '&'")
    return text.replace(anchor, insertion + anchor, 1)


def instrument(text: str) -> str:
    if "oracle_l5_mle_kt00000001.bin" in text:
        raise ValueError("tramle is already instrumented")
    start_marker = "SUBROUTINE tra_mle_trp_RK3_t("
    end_marker = "END SUBROUTINE tra_mle_trp_RK3_t"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise ValueError("RK3 MLE transport routine is not unique")
    start = text.index(start_marker)
    end = text.index("\n", text.index(end_marker, start)) + 1
    prefix, routine, suffix = text[:start], text[start:end], text[end:]
    routine = _insert(routine, DECL_ANCHOR, DECL, "declaration")
    # The del_key key_xios build makes iom_use() unconditionally false.  The
    # snapshot must therefore precede this diagnostic-output IF; placing it
    # inside leaves the AFTER observer with unallocated arrays.
    routine = _insert_before(routine, BODY_ANCHOR, BEFORE, "before-copy")
    # This anchor is after the level loop and therefore after every pFu/pFv add.
    routine = routine.replace(AFTER_ANCHOR, AFTER + AFTER_ANCHOR, 1)
    routine = _insert(routine, END_ANCHOR, CANON, "canonical-view")
    out = prefix + routine + suffix
    validate(out)
    return out


def validate(text: str) -> None:
    found = re.findall(r"WRITE\(l5_unit\) '([^']*)'", text)
    if found != [MAGIC, *TAGS] or any(len(value) != 16 for value in found):
        raise ValueError(f"observer literals differ: {found!r}")
    required = (
        "zfield(:,:,:) = 0._wp",
        "IF( llwet ) zfield(ji,jj,jk) = pfield(ji,jj,jk)",
        "l5_before_u(:,:,:) = l5_canon_3d(pFu,'U')",
        "l5_before_v(:,:,:) = l5_canon_3d(pFv,'V')",
        "l5_canon_3d(pFu,'U') - l5_before_u",
        "l5_canon_3d(pFv,'V') - l5_before_v",
    )
    if any(text.count(value) != 1 for value in required):
        raise ValueError("observer bypasses its canonical zero-first view")
    before_marker = "      IF( kt == nit000 .AND. lwp .AND. .NOT. l_istiled ) THEN"
    before = text.index(before_marker)
    routine = text[text.rindex("SUBROUTINE tra_mle_trp_RK3_t(", 0, before):before]
    depth = 0
    for line in routine.splitlines():
        statement = line.split("!", 1)[0].strip().upper()
        if re.match(r"^IF\s*\(.*\)\s*THEN\s*$", statement):
            depth += 1
        elif re.match(r"^END\s*IF\s*$", statement):
            depth -= 1
        if depth < 0:
            raise ValueError("unbalanced IF before before-copy")
    if depth:
        raise ValueError("before-copy is nested inside an IF")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plant-after-continuation", action="store_true")
    parser.add_argument("--plant-short-literal", action="store_true")
    parser.add_argument("--plant-raw-observer", action="store_true")
    parser.add_argument("--plant-nested-before", action="store_true")
    args = parser.parse_args()
    text = args.source.read_text()
    if args.plant_after_continuation:
        text = text.replace(DECL_ANCHOR, "      l5_plant = 1 &\n" + DECL_ANCHOR, 1)
    out = instrument(text)
    if args.plant_short_literal:
        out = out.replace(MAGIC, MAGIC[:-1], 1)
    if args.plant_raw_observer:
        out = out.replace("l5_canon_3d(pFu,'U')", "pFu", 1)
    if args.plant_nested_before:
        out = out.replace(BEFORE + BODY_ANCHOR, BODY_ANCHOR + BEFORE, 1)
    validate(out)
    args.output.write_text(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
