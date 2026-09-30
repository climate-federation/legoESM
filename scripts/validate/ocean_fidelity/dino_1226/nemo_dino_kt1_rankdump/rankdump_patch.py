#!/usr/bin/env python
"""Additive-only patch: give every dyn_spg_ts debug stream a per-rank name.

WHY.  ``cfgs/DINO/MY_SRC/dynspg_ts.F90`` writes every debug stream to a FIXED
filename under two guards that never mention the MPI rank::

    ll_spg_dump = ( kt == nit000 ) .AND. ( .NOT. l_istiled .OR. ntile == 1 )
                                                    ! dynspg_ts.F90:206
    IF( kt == nit000 ) THEN ... OPEN( UNIT=799, FILE='substep_dump.bin' ...
                                                    ! dynspg_ts.F90:926-928

With ``jpni x jpnj = 2 x 8`` all sixteen ranks ``OPEN`` the same path with
``STATUS='REPLACE'`` and write concurrently, so the bytes on disk are an
interleave of sixteen writers.  ``spg_kt1_barotropic_ladder.py --audit-only``
proves it on the existing record three independent ways.

WHAT THIS CHANGES.  Filenames, and the substep stream's record framing.  It
adds declarations and string assignments; it reads no new array, writes no
production array, and moves no guard.  The physics the binary executes is
unchanged, which is what the twin byte-identity check in ``run.sh`` then
proves against the untouched record.

THE SUBSTEP STREAM becomes ONE FILE PER (rank, substep) with a self-describing
header, instead of one growing sequential file.  Record framing inside a single
unformatted file was the second thing that made the old dump unreadable: the
header said ``jpj=28`` while the file length was the ``jpj=29`` record length,
and no stride recovered substep markers 1..45.  A file per substep cannot have
that ambiguity, and the header carries everything the reader needs to place the
tile in the global domain.

Usage
-----
    python rankdump_patch.py <path to the COPY's MY_SRC/dynspg_ts.F90>

Refuses to touch a path under the pristine ``cfgs/DINO`` or under ``src/``.
"""
from __future__ import annotations

import os
import re
import sys

# The header every per-substep file starts with.  Keep in step with
# read_rankdump.py::HEADER_FIELDS -- the reader asserts the count.
_HEADER = ("jpi, jpj, icycle, jn, narea, nimpp, njmpp, nn_hls, "
           "Nis0, Nie0, Njs0, Nje0, jpiglo, jpjglo")

_OLD_SUBSTEP = """         IF( kt == nit000 ) THEN
            IF( jn == 1 ) THEN
               OPEN( UNIT=799, FILE='substep_dump.bin', FORM='UNFORMATTED', &
                  &  ACCESS='STREAM', STATUS='REPLACE', ACTION='WRITE' )
               WRITE(799) jpi, jpj, icycle
            ENDIF
            WRITE(799) jn
            WRITE(799) sshn_e(:,:)
            WRITE(799) ssha_e(:,:)
            WRITE(799) zsshp2_e(:,:)
            WRITE(799) un_e(:,:)
            WRITE(799) vn_e(:,:)
            WRITE(799) ua_e(:,:)
            WRITE(799) va_e(:,:)
            IF( jn == icycle )   CLOSE( 799 )
         ENDIF"""

_NEW_SUBSTEP = """         IF( kt == nit000 ) THEN
            ! #1728 rank + substep tagged, self-describing, WRITE-only.
            WRITE(cl_sub,'(A,I4.4,A,I3.3,A)') 'substep_r', narea-1,      &
               &                              '_s', jn, '.bin'
            OPEN( UNIT=799, FILE=TRIM(cl_sub), FORM='UNFORMATTED',       &
               &  ACCESS='STREAM', STATUS='REPLACE', ACTION='WRITE' )
            WRITE(799) {header}
            WRITE(799) sshn_e(:,:)
            WRITE(799) ssha_e(:,:)
            WRITE(799) zsshp2_e(:,:)
            WRITE(799) un_e(:,:)
            WRITE(799) vn_e(:,:)
            WRITE(799) ua_e(:,:)
            WRITE(799) va_e(:,:)
            CLOSE( 799 )
         ENDIF""".format(header=_HEADER)


def patch(path: str) -> None:
    real = os.path.realpath(path)
    nemo = os.path.realpath(os.path.join(
        os.path.dirname(real), "..", "..", ".."))
    for forbidden in (os.path.join(nemo, "cfgs", "DINO") + os.sep,
                      os.path.join(nemo, "src") + os.sep):
        if real.startswith(forbidden):
            raise SystemExit(
                f"REFUSING to patch {real}: it is inside the read-only oracle "
                f"tree {forbidden}. Patch the makenemo COPY, never the "
                f"original.")
    src = open(real).read()
    if "TRIM(cl_rk)" in src:
        raise SystemExit(f"{real} is already patched; refusing to double-apply")

    # 1. two declarations, next to the existing ll_spg_dump one
    anchor = "      LOGICAL  ::   ll_spg_dump "
    if anchor not in src:
        raise SystemExit("ll_spg_dump declaration not found -- "
                         "this is not DINO's instrumented dynspg_ts.F90")
    src = src.replace(
        anchor,
        "      CHARACTER(LEN=32) ::   cl_rk    ! #1728 per-rank name suffix\n"
        "      CHARACTER(LEN=64) ::   cl_sub   ! #1728 per-substep filename\n"
        + anchor, 1)

    # 2. build the suffix once, where the guard is set
    anchor2 = "      ll_spg_dump = ( kt == nit000 )"
    if anchor2 not in src:
        raise SystemExit("ll_spg_dump assignment not found")
    src = src.replace(
        anchor2,
        "      WRITE(cl_rk,'(A,I4.4,A)') '_r', narea-1, '.bin'\n" + anchor2, 1)

    # 3. the substep stream: one self-describing file per (rank, substep).
    #    BEFORE the filename regex below, which would otherwise rewrite this
    #    block's FILE= and the byte-for-byte match would silently stop firing.
    if _OLD_SUBSTEP not in src:
        raise SystemExit(
            "the substep_dump block does not match byte-for-byte; the oracle "
            "source moved. Re-read dynspg_ts.F90 around the UNIT=799 OPEN and "
            "update _OLD_SUBSTEP rather than loosening the match.")
    src = src.replace(_OLD_SUBSTEP, _NEW_SUBSTEP, 1)

    # 4. rank-tag every remaining once-per-step debug filename
    src, n_files = re.subn(r"FILE='([A-Za-z0-9_]+)\.bin'",
                           r"FILE='\1'//TRIM(cl_rk)", src)
    if "FILE='substep_dump" in src:
        raise SystemExit("the substep stream was not replaced before the "
                         "filename rewrite")

    # 5. WRITE-only, checked rather than asserted in prose
    for bad in ("ACTION='READ'", "ACTION='READWRITE'"):
        if bad in src:
            raise SystemExit(f"patched source contains {bad}: a debug stream "
                             "must never open for reading")
    # The stream this patch ADDS is WRITE-only and says so; the pre-existing
    # dumps are left exactly as the oracle wrote them (additive-only), so this
    # counts rather than rewrites them.
    if "FILE=TRIM(cl_sub), FORM='UNFORMATTED',       &\n" not in src or \
            "ACTION='WRITE'" not in _NEW_SUBSTEP:
        raise SystemExit("the added substep writer is not ACTION='WRITE'")
    n_open = src.count("OPEN( UNIT=")
    n_write = src.count("ACTION='WRITE'")
    print(f"  {n_open} OPEN statements, {n_write} declare ACTION='WRITE', "
          f"0 declare READ or READWRITE")

    open(real, "w").write(src)
    print(f"patched {real}: {n_files} rank-tagged filenames, "
          f"substep stream is one file per "
          f"(rank, substep)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    patch(sys.argv[1])
