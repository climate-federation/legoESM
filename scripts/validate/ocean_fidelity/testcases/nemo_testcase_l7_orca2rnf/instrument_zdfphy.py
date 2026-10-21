#!/usr/bin/env python3
"""Add a write-only before/after river-mouth observer to NEMO ``zdfphy``."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MAGIC = "NEMO_L7_RNFKZ_1 "
TAGS = ("CLOSURE_AVT     ", "RIVER_AVT       ", "EVD_DDM_AVT     ",
        "IWM_AVT         ")

DECL_ANCHOR = "      INTEGER ::   ji, jj, jk, jtile   ! dummy loop indice\n"
DECL = """      INTEGER :: l7_unit, l7_ios
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE :: l7_closure, l7_river, l7_evd_ddm
"""
BLOCK = """         IF( ln_rnf_mouth ) THEN                   !* increase diffusivity at rivers mouths
            DO_3D( 0, 0, 0, 0, 2, nkrnf )
               avt(ji,jj,jk) = avt(ji,jj,jk) + 2._wp * rn_avt_rnf * rnfmsk(ji,jj) * wmask(ji,jj,jk)
            END_3D
         ENDIF
"""
BEFORE = """         IF( kt == nit000 .AND. lwp .AND. .NOT. ln_tile ) THEN
            ALLOCATE( l7_closure(jpi,jpj,jpk), l7_river(jpi,jpj,jpk), l7_evd_ddm(jpi,jpj,jpk) )
            l7_closure(:,:,:) = l7_canon_avt(avt)
         ENDIF
"""
AFTER_RIVER = """         IF( kt == nit000 .AND. lwp .AND. .NOT. ln_tile ) THEN
            l7_river(:,:,:) = l7_canon_avt(avt)
         ENDIF
"""
WAVE_ANCHOR = """         !
         !                                         !* wave-induced mixing
"""
AFTER_DDM = """         IF( kt == nit000 .AND. lwp .AND. .NOT. ln_tile ) THEN
            l7_evd_ddm(:,:,:) = l7_canon_avt(avt)
         ENDIF
"""
IWM_BLOCK = """         IF( ln_zdfswm )   CALL zdf_swm( kt, Kmm, avm, avt, avs )   ! surface  wave (Qiao et al. 2004)
         IF( ln_zdfiwm )   CALL zdf_iwm( kt, Kmm, avm, avt, avs )   ! internal wave (de Lavergne et al 2017)
"""
AFTER_IWM = f"""         IF( kt == nit000 .AND. lwp .AND. .NOT. ln_tile ) THEN
            OPEN( NEWUNIT=l7_unit, FILE='oracle_l7_rnf_kt00000001.bin', ACCESS='STREAM', &
               & FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=l7_ios )
            IF( l7_ios /= 0 ) CALL ctl_stop( 'L7 river observer: open failed' )
            WRITE(l7_unit) '{MAGIC}'
            WRITE(l7_unit) 1, kt, jpi, jpj, jpk, 64, 4
            WRITE(l7_unit) '{TAGS[0]}' ; WRITE(l7_unit) l7_closure
            WRITE(l7_unit) '{TAGS[1]}' ; WRITE(l7_unit) l7_river
            WRITE(l7_unit) '{TAGS[2]}' ; WRITE(l7_unit) l7_evd_ddm
            WRITE(l7_unit) '{TAGS[3]}' ; WRITE(l7_unit) l7_canon_avt(avt)
            CLOSE(l7_unit)
            DEALLOCATE(l7_closure, l7_river, l7_evd_ddm)
         ENDIF
"""
END_ANCHOR = "   END SUBROUTINE zdf_phy\n"
CANON = """
   FUNCTION l7_canon_avt( pfield ) RESULT( zfield )
      !! Zero first; copy only cells NEMO defines in the river statement.
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pfield
      REAL(wp), DIMENSION(SIZE(pfield,1),SIZE(pfield,2),SIZE(pfield,3)) :: zfield
      INTEGER :: ji, jj, jk
      zfield(:,:,:) = 0._wp
      DO jk = 2, jpkm1
         DO jj = 1+nn_hls, jpj-nn_hls
            DO ji = 1+nn_hls, jpi-nn_hls
               IF( wmask(ji,jj,jk) /= 0._wp ) zfield(ji,jj,jk) = pfield(ji,jj,jk)
            END DO
         END DO
      END DO
   END FUNCTION l7_canon_avt
"""


def _once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{label} anchor count {text.count(old)} != 1")
    previous = text[:text.index(old)].rstrip().splitlines()[-1]
    if previous.rstrip().endswith("&"):
        raise ValueError(f"{label} anchor follows continued '&'")
    return text.replace(old, new, 1)


def instrument(text: str) -> str:
    if "oracle_l7_rnf_kt00000001.bin" in text:
        raise ValueError("zdfphy is already instrumented")
    start = text.index("   SUBROUTINE zdf_phy( kt, Kbb, Kmm, Krhs )")
    end = text.index(END_ANCHOR, start) + len(END_ANCHOR)
    prefix, routine, suffix = text[:start], text[start:end], text[end:]
    routine = _once(routine, DECL_ANCHOR, DECL_ANCHOR + DECL, "declaration")
    routine = _once(routine, BLOCK, BEFORE + BLOCK + AFTER_RIVER, "river block")
    routine = _once(routine, WAVE_ANCHOR, AFTER_DDM + WAVE_ANCHOR,
                    "post-EVD/DDM boundary")
    routine = _once(routine, IWM_BLOCK, IWM_BLOCK + AFTER_IWM,
                    "post-IWM boundary")
    routine = _once(routine, END_ANCHOR, END_ANCHOR + CANON, "canonical view")
    out = prefix + routine + suffix
    validate(out)
    return out


def validate(text: str) -> None:
    literals = re.findall(r"WRITE\(l7_unit\) '([^']*)'", text)
    if literals != [MAGIC, *TAGS] or any(len(value) != 16 for value in literals):
        raise ValueError(f"observer literals differ: {literals!r}")
    required = (
        "zfield(:,:,:) = 0._wp",
        "IF( wmask(ji,jj,jk) /= 0._wp ) zfield(ji,jj,jk) = pfield(ji,jj,jk)",
        "l7_closure(:,:,:) = l7_canon_avt(avt)",
        "l7_river(:,:,:) = l7_canon_avt(avt)",
        "l7_evd_ddm(:,:,:) = l7_canon_avt(avt)",
    )
    if any(text.count(value) != 1 for value in required):
        raise ValueError("observer bypasses its zero-first defined-cell view")
    before = text.index(BEFORE)
    river = text.index(BLOCK)
    after_river = text.index(AFTER_RIVER)
    evd = text.index("IF( ln_zdfevd )", after_river)
    after_ddm = text.index(AFTER_DDM, evd)
    iwm = text.index(IWM_BLOCK, after_ddm)
    after_iwm = text.index(AFTER_IWM, iwm)
    if not before < river < after_river < evd < after_ddm < iwm < after_iwm:
        raise ValueError("observer no longer pins closure/river/EVD-DDM/IWM order")
    if "iom_use" in text[before:after_iwm]:
        raise ValueError("observer is nested in an XIOS-dependent arm")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plant-after-continuation", action="store_true")
    parser.add_argument("--plant-short-literal", action="store_true")
    parser.add_argument("--plant-raw-observer", action="store_true")
    parser.add_argument("--plant-after-evd", action="store_true")
    args = parser.parse_args()
    text = args.source.read_text()
    if args.plant_after_continuation:
        text = text.replace(DECL_ANCHOR, "      l7_plant = 1 &\n" + DECL_ANCHOR, 1)
    out = instrument(text)
    if args.plant_short_literal:
        out = out.replace(MAGIC, MAGIC[:-1], 1)
    if args.plant_raw_observer:
        out = out.replace("l7_river(:,:,:) = l7_canon_avt(avt)",
                          "l7_river(:,:,:) = avt", 1)
    if args.plant_after_evd:
        out = out.replace(AFTER_RIVER, "", 1).replace(
            "         IF( ln_zdfevd )   CALL zdf_evd( kt, Kmm, Krhs, avm, avt )",
            "         IF( ln_zdfevd )   CALL zdf_evd( kt, Kmm, Krhs, avm, avt )\n"
            + AFTER_RIVER.rstrip("\n"), 1)
    validate(out)
    args.output.write_text(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
