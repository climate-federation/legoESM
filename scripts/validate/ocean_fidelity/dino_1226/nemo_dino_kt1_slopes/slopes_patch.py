#!/usr/bin/env python3
"""Add a WRITE-only, per-rank slope dump at ``tra_ldf``'s OWN call site.

WHY THIS EXISTS.  ``RUN_FROMREST_KT1``'s ``uslp_stg``/``vslp_stg``/
``wslpi_stg``/``wslpj_stg`` are identically ZERO on all sixteen tiles while
``rhd_stg``/``tn_stg``/``rn2_stg`` from the same snapshot carry data.  NEMO's
``tra_ldf`` cannot have run on zero slopes -- the A33 stage is 1.07x the whole
tendency -- so the DUMP is what is empty, and the slope routine has therefore
never been scored on this card.  Every ``tra_ldf`` number on this branch is a
number for the OPERATOR fed whatever slopes legoESM produced; nothing
separates a correct operator on wrong slopes from the converse.

WHERE.  Immediately after ``CALL traldf_iso_a33( Kmm, ah_wslp2, akz )`` inside
``traldf_iso_lap`` (``src/OCE/TRA/traldf_iso.F90:135``).  That is tra_ldf's own
frame: ``uslp``/``vslp``/``wslpi``/``wslpj`` are the ``ldfslp`` module arrays
the very next statements consume, and ``ah_wslp2``/``akz`` have just been
filled -- so the six fields are exactly the operands, with no intervening
writer to argue about.

ADDITIVE.  Nothing existing is edited: one USE-free block of declarations and
one OPEN/WRITE/CLOSE.  ``ACTION='WRITE'``, per-rank filenames, and the same
self-describing header ``nemo_dino_kt1_rankdump`` uses, so the reader never
guesses a stride or a halo width.
"""
from __future__ import annotations

import re
import sys

HEADER_FIELDS = ("jpi, jpj, narea, nimpp, njmpp, nn_hls, "
                 "Nis0, Nie0, Njs0, Nje0, jpiglo, jpjglo, jpk")

_ANCHOR = "      CALL traldf_iso_a33( Kmm, ah_wslp2, akz )"

_DECL_ANCHOR = ("      REAL(wp), DIMENSION(T2D(0)    ) ::   zfw"
                "          ! INNER     domain")

_DECLS = """      CHARACTER(LEN=64) ::   cl_slp   ! #1728 per-rank slope-dump name
"""

_DUMP = """      !
      ! ---------------------------------------------------------------- #1728
      ! WRITE-ONLY slope record at tra_ldf's OWN call site.  Reads nothing,
      ! changes nothing; the six fields below are the operands the statements
      ! immediately following consume.  kt==nit000 only.
      IF( kt == nit000 ) THEN
         WRITE(cl_slp,'(A,I4.4,A)') 'ldfslp_at_traldf_r', narea-1, '.bin'
         OPEN( UNIT=7281, FILE=TRIM(cl_slp), FORM='UNFORMATTED',            &
            &  ACCESS='STREAM', STATUS='REPLACE', ACTION='WRITE' )
         WRITE(7281) jpi, jpj, narea, nimpp, njmpp, nn_hls,                 &
            &        Nis0, Nie0, Njs0, Nje0, jpiglo, jpjglo, jpk
         WRITE(7281) uslp (:,:,:)
         WRITE(7281) vslp (:,:,:)
         WRITE(7281) wslpi(:,:,:)
         WRITE(7281) wslpj(:,:,:)
         WRITE(7281) ah_wslp2(:,:,:)
         WRITE(7281) akz     (:,:,:)
         CLOSE(7281)
      ENDIF
      ! -------------------------------------------------------------- end #1728
"""


def patch(path: str) -> None:
    src = open(path).read()
    if "ldfslp_at_traldf_r" in src:
        raise SystemExit(
            "REFUSING: %s already carries the slope dump. A second patch "
            "would write the block twice." % path)
    # There are TWO a33 call sites: one in traldf_iso_lap and one in
    # traldf_iso_blp.  DINO runs the LAP one -- ln_traldf_lap=.true.,
    # ln_traldf_blp=.false., ln_traldf_iso=.true. (RUN_FROMREST_KT1/
    # namelist_cfg) give nldf_tra = np_lap_i, and traldf.F90's dispatcher
    # calls traldf_iso_lap.  Patching the blp copy would produce an empty
    # record, so the window is located explicitly instead of counting.
    if src.count(_ANCHOR) != 2:
        raise SystemExit(
            "REFUSING: the traldf_iso_a33 call site occurs %d times in %s "
            "(expected exactly 2: traldf_iso_lap and traldf_iso_blp) -- this "
            "is not the file this patch was written against."
            % (src.count(_ANCHOR), path))
    lo = src.index("   SUBROUTINE traldf_iso_lap(")
    hi = src.index("   END SUBROUTINE traldf_iso_lap")
    a_at = src.index(_ANCHOR, lo, hi)          # raises if it is not in the lap body
    if src.count(_DECL_ANCHOR) < 1:
        raise SystemExit(
            "REFUSING: the declaration anchor is absent from %s" % path)
    d_at = src.index(_DECL_ANCHOR, lo, hi)
    # Insert after the END OF THE LINE, so the statement keeps its own
    # trailing comment instead of having the block spliced in front of it.
    a_eol = src.index("\n", a_at)
    # Insert the LATER one first so the earlier offset stays valid.
    src = src[:a_eol] + "\n" + _DUMP.rstrip("\n") + src[a_eol:]
    d_eol = src.index("\n", d_at)
    src = src[:d_eol] + "\n" + _DECLS.rstrip("\n") + src[d_eol:]
    # ``narea``/``nimpp``/``njmpp``/``jpiglo``/``jpjglo`` live in dom_oce, which
    # traldf_iso already USEs through oce; assert rather than assume.
    if not re.search(r"^\s*USE\s+dom_oce", src, re.M):
        raise SystemExit(
            "REFUSING: %s does not USE dom_oce, so narea/nimpp/njmpp are not "
            "in scope and the dump would not compile." % path)
    open(path, "w").write(src)
    print("patched %s: slope dump at tra_ldf's own call site" % path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: slopes_patch.py <traldf_iso.F90 in a COPY>")
    patch(sys.argv[1])
