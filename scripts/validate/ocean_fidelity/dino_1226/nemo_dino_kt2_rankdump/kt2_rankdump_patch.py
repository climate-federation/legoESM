#!/usr/bin/env python
"""Additive-only patch: the kt<=2 rank- AND step-tagged dyn_spg_ts record.

WHAT THIS ADDS OVER ``nemo_dino_kt1_rankdump/rankdump_patch.py``, which it
reuses rather than re-implements (RULE 4: the header layout, the refusals and
the WRITE-only checks already exist there and the two must not drift):

1. ``kt`` IN EVERY FILENAME, and the guard becomes ``kt <= nit000 + 1``.  The
   kt=1 patch tags by rank only, so a two-step run would have kt=2 overwrite
   kt=1 in the same directory and the record would silently be one step.

2. THE CROSS-STEP BAROTROPIC SUB-STATE, dumped twice per step: immediately
   after the ``IF( ll_init )`` block (dynspg_ts.f90:463-470) and immediately
   after the substep loop ends.  Those six arrays -- ``sshb_e``, ``sshbb_e``,
   ``ub_e``, ``ubb_e``, ``vb_e``, ``vbb_e`` -- are the ONLY barotropic state
   that can cross a baroclinic step on this card, and NEMO does NOT write
   them to the restart here: ``ts_rst`` guards the ``ub2_b``/``un_bf`` group
   on ``ln_bt_fw`` (FALSE on DINO) and the sub-state group on
   ``nn_bt_flt == 3`` (2 on DINO), so neither branch fires.  Without this
   dump the question "what does NEMO carry into kt=2" has no record at all.

TWO PREREGISTERED PREDICTIONS, both falsifiable by this record, both written
BEFORE it is acquired:

  P1  THE kt=2 RECORD HOLDS 68 SUBSTEP FILES PER RANK, NOT 45.  At
      ``kt == nit000`` with the Euler first step NEMO sets
      ``ll_fw_start = .TRUE.`` (dynspg_ts.f90:228-232) and ``ts_wgt`` centres
      the boxcar at ``jic = nn_e = 23``, giving ``icycle = 45``
      (ocean.output:1265).  At ``kt == nit000 + 1`` with ``ln_bt_fw = .FALSE.``
      NEMO RESETS ``ll_fw_start = .FALSE.`` and calls ``ts_wgt`` again
      (:245-250), which centres at ``jic = 2*nn_e = 46`` and gives
      ``icycle = 68``, with the primary boxcar nonzero on substeps 24..68.
      MEASURED on the legoESM side this round: it runs 91 substeps at kt=2.
      If the record shows 45 or 91 rather than 68, this prediction is wrong
      and the kt=2 window finding is retracted.

  P2  THE SUB-STATE AT THE START OF kt=2 IS IDENTICALLY ZERO.  ``ll_bt_av``
      is ``.TRUE.`` whenever ``nn_bt_flt /= 3`` (:208-209) and ``ll_init =
      ll_bt_av`` (:214), and BOTH statements sit OUTSIDE the
      ``IF( kt == nit000 )`` block, so the sub-state is re-zeroed at :463-470
      on EVERY step.  If the start-of-kt=2 dump is nonzero, then NEMO does
      carry a barotropic history on this card, decision 33 is live, and the
      end-of-kt=1 dump is exactly the value to substitute.

Usage
-----
    python kt2_rankdump_patch.py <path to the COPY's MY_SRC/dynspg_ts.F90>
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _kt1_patch_module():
    """The kt=1 patch, imported so the header layout has ONE definition."""
    p = os.path.join(_HERE, "..", "nemo_dino_kt1_rankdump", "rankdump_patch.py")
    p = os.path.normpath(p)
    if not os.path.exists(p):
        raise SystemExit(f"the kt=1 patch is missing: {p}")
    spec = importlib.util.spec_from_file_location("_kt1_rankdump_patch", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The six arrays that are the whole of the cross-step barotropic sub-state.
SUBSTATE = ("sshb_e", "sshbb_e", "ub_e", "ubb_e", "vb_e", "vbb_e")


def _substate_block(tag: str, unit: int) -> str:
    """One self-describing WRITE of the six sub-state arrays.

    The header's substep slot is written 0: neither dump belongs to a substep
    (one is before the loop, one after it), and 0 is not a legal ``jn``, so a
    reader cannot mistake either for a substep tile.
    """
    body = "\n".join(
        f"            WRITE({unit}) {nm}(:,:)" for nm in SUBSTATE)
    hdr = _kt1_patch_module()._HEADER.replace(", jn,", ", 0,")
    return f"""
         ! ---- #1728 kt<=2 sub-state dump ({tag}), WRITE-only ----
         IF( kt <= nit000 + 1 ) THEN
            WRITE(cl_st,'(A,I4.4,A,I8.8,A)') 'substate_{tag}_r', narea-1,  &
               &                             '_kt', kt, '.bin'
            OPEN( UNIT={unit}, FILE=TRIM(cl_st), FORM='UNFORMATTED',       &
               &  ACCESS='STREAM', STATUS='REPLACE', ACTION='WRITE' )
            WRITE({unit}) {hdr}
{body}
            CLOSE( {unit} )
         ENDIF
"""


def _unit(src: str, want: int) -> int:
    """Refuse a Fortran unit number the file already opens.

    The first draft of this patch reached for 8971 and 8972, which
    ``dynspg_ts.F90`` ALREADY opens for the bottom-drag forcing increments
    (``drg_dump_z[uv]_frc_inc``).  Two OPENs on one unit is not a compile
    error: the second silently closes the first stream, and the record would
    have been missing the very arrays it exists to capture -- while every
    check in this script still passed.  Caught by dry-running the patch on a
    scratch copy of the oracle source and listing the units, which is why the
    dry run is part of the workflow and not an optional extra.
    """
    used = {int(m) for m in re.findall(r"UNIT=(\d+)", src)}
    if want in used:
        raise SystemExit(
            f"REFUSING: unit {want} is already opened in this source "
            f"(units in use: {sorted(used)}). Pick a free one.")
    return want


def patch(path: str) -> None:
    kt1 = _kt1_patch_module()
    real = os.path.realpath(path)
    nemo = os.path.realpath(os.path.join(
        os.path.dirname(real), "..", "..", ".."))
    for forbidden in (os.path.join(nemo, "cfgs", "DINO") + os.sep,
                      os.path.join(nemo, "src") + os.sep):
        if real.startswith(forbidden):
            raise SystemExit(
                f"REFUSING to patch {real}: it is inside the read-only oracle "
                f"tree {forbidden}. Patch the makenemo COPY, never the "
                "original.")
    src = open(real).read()
    if "TRIM(cl_sub)" in src or "TRIM(cl_st)" in src:
        raise SystemExit(f"{real} is already patched; refusing to double-apply")

    # 1. declarations
    anchor = "      LOGICAL  ::   ll_spg_dump "
    if anchor not in src:
        raise SystemExit("ll_spg_dump declaration not found -- this is not "
                         "DINO's instrumented dynspg_ts.F90")
    src = src.replace(
        anchor,
        "      CHARACTER(LEN=32) ::   cl_rk    ! #1728 per-rank name suffix\n"
        "      CHARACTER(LEN=64) ::   cl_sub   ! #1728 per-substep filename\n"
        "      CHARACTER(LEN=64) ::   cl_st    ! #1728 sub-state filename\n"
        + anchor, 1)

    # 2. the suffix carries kt, so kt=2 cannot overwrite kt=1
    anchor2 = "      ll_spg_dump = ( kt == nit000 )"
    if anchor2 not in src:
        raise SystemExit("ll_spg_dump assignment not found")
    src = src.replace(
        anchor2,
        "      WRITE(cl_rk,'(A,I4.4,A,I8.8,A)') '_r', narea-1, '_kt', kt, "
        "'.bin'\n" + anchor2, 1)
    # ...and the once-per-step dumps must FIRE at kt=2, not only at kt=1.
    src = src.replace(anchor2 + " .AND.",
                      "      ll_spg_dump = ( kt <= nit000 + 1 ) .AND.", 1)
    if "( kt <= nit000 + 1 )" not in src:
        raise SystemExit(
            "the ll_spg_dump guard was not widened to kt <= nit000+1; the "
            "record would carry kt=1 only and P1 could not be tested")

    # 3. the substep stream: one self-describing file per (rank, kt, substep)
    if kt1._OLD_SUBSTEP not in src:
        raise SystemExit(
            "the substep_dump block does not match byte-for-byte; the oracle "
            "source moved. Re-read dynspg_ts.F90 around the UNIT=799 OPEN and "
            "update the kt=1 patch's _OLD_SUBSTEP rather than loosening it.")
    new_substep = (kt1._NEW_SUBSTEP
                   .replace("IF( kt == nit000 ) THEN",
                            "IF( kt <= nit000 + 1 ) THEN", 1)
                   .replace("'substep_r', narea-1,      &\n"
                            "               &                              "
                            "'_s', jn, '.bin'",
                            "'substep_r', narea-1,      &\n"
                            "               &                              "
                            "'_kt', kt, '_s', jn, '.bin'", 1)
                   .replace("'(A,I4.4,A,I3.3,A)'", "'(A,I4.4,A,I8.8,A,I3.3,A)'",
                            1))
    if "_kt', kt, '_s'" not in new_substep:
        raise SystemExit(
            "the substep filename did not gain its kt field; at kt=2 every "
            "file would overwrite its kt=1 twin and the record would be one "
            "step pretending to be two")
    src = src.replace(kt1._OLD_SUBSTEP, new_substep, 1)

    # 4. the cross-step sub-state, twice: after ll_init and after the loop.
    init_anchor = """      IF( ll_init )THEN
         sshbb_e(:,:) = 0._wp
         ubb_e  (:,:) = 0._wp
         vbb_e  (:,:) = 0._wp
         sshb_e (:,:) = 0._wp
         ub_e   (:,:) = 0._wp
         vb_e   (:,:) = 0._wp
      ENDIF"""
    if init_anchor not in src:
        raise SystemExit(
            "the ll_init re-initialisation block does not match byte for "
            "byte; P2 is a claim about THAT block, so the dump must sit "
            "immediately after it and nowhere else.")
    src = src.replace(init_anchor,
                      init_anchor + _substate_block("start", _unit(src, 9301)), 1)

    # The anchor must include the line AFTER `END DO`, and the dump must go
    # after the whole thing.  The first version anchored on the comment
    # immediately BEFORE `END DO` and inserted ahead of it, which put the
    # "end of loop" dump INSIDE the loop: 45 opens per rank per step, the
    # docstring's claim false, and only the last write happening to hold the
    # right state.  An independent review caught it in a dry run.
    loop_end = ("      END DO                                               "
                "!        end loop      !\n"
                "      !                                                    "
                "! ==================== !")
    if loop_end not in src:
        # fall back to the swap block's owner: the statement AFTER the last
        # swap is the loop's END DO, and anchoring on a comment that may not
        # exist is worse than refusing.
        raise SystemExit(
            "could not find the substep loop's END DO; refusing to guess "
            "where the end-of-loop sub-state dump belongs. Re-read "
            "dynspg_ts.F90 after the ubb_e/ub_e/un_e swap and update this "
            "anchor.")
    src = src.replace(loop_end,
                      loop_end + _substate_block("end", _unit(src, 9302)), 1)

    # 5. rank+kt tag every remaining once-per-step debug filename
    src, n_files = re.subn(r"FILE='([A-Za-z0-9_]+)\.bin'",
                           r"FILE='\1'//TRIM(cl_rk)", src)
    if "FILE='substep_dump" in src or "FILE='substate_" in src:
        raise SystemExit("a tagged stream was rewritten by the filename regex")

    # 6. WRITE-only, checked rather than asserted in prose
    for bad in ("ACTION='READ'", "ACTION='READWRITE'"):
        if bad in src:
            raise SystemExit(f"patched source contains {bad}: a debug stream "
                             "must never open for reading")
    n_open = src.count("OPEN( UNIT=")
    n_write = src.count("ACTION='WRITE'")
    print(f"  {n_open} OPEN statements, {n_write} declare ACTION='WRITE', "
          "0 declare READ or READWRITE")
    open(real, "w").write(src)
    print(f"patched {real}: {n_files} rank+kt-tagged filenames, substep "
          "stream is one file per (rank, kt, substep), sub-state dumped at "
          "the start and the end of every step <= nit000+1")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    patch(sys.argv[1])
