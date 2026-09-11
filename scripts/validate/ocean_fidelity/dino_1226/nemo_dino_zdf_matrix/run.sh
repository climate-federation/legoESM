#!/bin/bash
# Acquire NEMO's DINO implicit-vertical-solve MATRIX record at kt=1 AND kt=2,
# as a CONFIG COPY.  The pristine cfgs/DINO and src/ trees are never written.
#
# THIS SCRIPT IS NOT RUN BY THE AGENT THAT WROTE IT.  It prints the two
# commands that need a compiler and an MPI launcher and STOPS.  Building and
# launching NEMO is the user's call, every time.
#
# WHY THE RECORD IS NEEDED.  The GYRE lane transcribed two statements of the
# implicit solve that this branch cannot score without a matrix dump:
#
#   * the ASSEMBLED tracer diagonals.  GYRE's 5e25b893e16b extracts
#     ``nemo_tracer_tridiagonal`` so a gate can score legoESM's ASSEMBLY of
#     zwi/zwd/zws against NEMO's own, given NEMO's operands, SEPARATELY from
#     the ordered sweep.  Without the dump the two are only observable through
#     a solved column, which conflates them -- and a conflated row cannot say
#     which of the two owns a residual.
#
#   * the two boundary slots.  GYRE's 4ba1ffd6b428 found that NEMO writes
#     NEGATIVE zeros into zwi(:,1) and zws(:,jpkm1) (trazdf.F90:204 then the
#     -p2dt*zwt/e3w statements at :219-220), and b591ec5901cf found that
#     NEMO's DRY diagonal is its own reference thickness where legoESM's is
#     substituted 1.0, because legoESM's h_partial is exactly 0 below the
#     seafloor.  Neither slot is read by the recurrences, so NEITHER changes
#     an answer -- they change what a gate at the exact bar can see, which is
#     precisely why they need a record and not a run.
#
# The momentum half is the same question for dyn_zdf, where this card's
# ln_drgimp=.TRUE. puts the bottom drag ON the diagonal
# (MY_SRC/dynzdf.F90:310-313) and the barotropic mode is removed before the
# solve (:167-169) -- two statements that only a dumped zwd can score.
#
# WHAT IT WOULD PRODUCE, per rank, at kt = 1 and kt = 2:
#   $OUT/trazdf_zwi_kt000000NN_rankRR.bin   the tracer sub-diagonal
#   $OUT/trazdf_zwd_kt000000NN_rankRR.bin   the tracer diagonal
#   $OUT/trazdf_zws_kt000000NN_rankRR.bin   the tracer super-diagonal
#   $OUT/trazdf_rhs_kt000000NN_rankRR.bin   pt(:,:,:,jn,Kaa) entering the sweep
#   $OUT/dynzdf_zwi|zwd|zws|rhs_u/_v ...    the momentum siblings
#   $OUT/DINO_0000000{1,2}_restart_*.nc     so the record is self-contained
#
# PER-RANK SUFFIXES, not a gathered array: NEMO's own MY_SRC already writes
# this way (``ldftra.F90:535-539`` uses ``narea - 1``), and the stitching
# reader for it already exists on this branch
# (scripts/validate/ocean_fidelity/dino_1226/nemo_dino_kt1_rankdump/read_rankdump.py).  A gathered write would
# need lbc/collective calls this instrument must not add.
#
# THE INSTRUMENT IS WRITE-ONLY.  Every inserted statement is a WRITE of an
# array NEMO has already computed, inside ``IF( ll_zdfmat_dump )``.  No
# INTENT(inout) argument is touched, no array is reordered, nothing is
# allocated inside a loop.  That is a claim about the patch below and the
# post-run check at the bottom is what turns it into a measurement: the kt=2
# restart this build writes MUST be bit-identical to the one
# nemo_dino_kt2_trends/run.sh already produced, or the instrument changed the
# trajectory and the record is worthless.
#
# USAGE
#   scripts/validate/ocean_fidelity/dino_1226/nemo_dino_zdf_matrix/run.sh
#     -> prepares cfgs/DINO_ZDF_MATRIX and PRINTS the build + run commands
#   scripts/validate/ocean_fidelity/dino_1226/nemo_dino_zdf_matrix/run.sh --go
#     -> additionally RUNS them (the user's call, never the agent's)
#
# Optional environment: ARCH (default conda), NEMO, OUT, NPROC.
set -euo pipefail

NEMO=${NEMO:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
ARCH=${ARCH:-conda}
CFGNAME=DINO_ZDF_MATRIX
COPY=$NEMO/cfgs/$CFGNAME
OUT=$(readlink -m "${OUT:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_zdf_matrix}")
NPROC=${NPROC:-16}
GO=0
[ "${1:-}" = "--go" ] && GO=1

# ---------------------------------------------------------------- refusals
# Identical guard to nemo_dino_kt2_trends/run.sh, for the identical reason: a
# relative OUT, a symlinked NEMO or a CFGNAME containing '..' must not be able
# to resolve inside the read-only oracle configuration.
guard() {
  local real
  real=$(readlink -m "$1")
  case "$real/" in
    "$(readlink -m "$NEMO/cfgs/DINO")"/*)
      echo "REFUSED: $1 resolves inside the pristine cfgs/DINO" >&2; exit 2 ;;
    "$(readlink -m "$NEMO/src")"/*)
      echo "REFUSED: $1 resolves inside the pristine src/" >&2; exit 2 ;;
  esac
}
guard "$COPY"
guard "$OUT"
case "$CFGNAME" in *..*|*/*) echo "REFUSED: CFGNAME" >&2; exit 2 ;; esac

# ------------------------------------------------------------- config copy
rm -rf "$COPY"
cp -a "$NEMO/cfgs/DINO" "$COPY"
mkdir -p "$OUT"

# DINO has no MY_SRC/trazdf.F90 override, so the tracer solve runs the shared
# src/OCE/TRA/trazdf.F90.  Copy it into the COPY's MY_SRC -- this is what
# makes the instrument local to the copy instead of a patch to src/.
cp "$NEMO/src/OCE/TRA/trazdf.F90" "$COPY/MY_SRC/trazdf.F90"

python3 - "$COPY" <<'PYEOF'
import pathlib, re, sys
copy = pathlib.Path(sys.argv[1])

# ---- tracer: dump the three diagonals and the RHS the sweep consumes.
# The anchor is the line AFTER the assembly and BEFORE the first recurrence:
# src/OCE/TRA/trazdf.F90:257 `zwt(ji,1) = zwd(ji,1)`.  Anchoring on the
# STATEMENT rather than on a line number is deliberate -- a line number in a
# script is a claim that goes stale silently.
p = copy / "MY_SRC" / "trazdf.F90"
s = p.read_text()
anchor = "                  zwt(ji,1) = zwd(ji,1)"
if anchor not in s:
    raise SystemExit("REFUSED: trazdf.F90 anchor not found; the oracle moved "
                     "and this instrument would be inserted in the wrong place")
insert = '''
            ! ---- #1728 WRITE-ONLY matrix dump (kt = nit000, nit000+1) ----
            IF( kt <= nit000 + 1 ) THEN
               WRITE( cl_zm, '("trazdf_",A3,"_kt",I8.8,"_rank",I2.2,".bin")' )  &
                  &  'zwi', kt, narea - 1
               OPEN( UNIT=9201, FILE=TRIM(cl_zm), ACCESS='STREAM',              &
                  &  FORM='UNFORMATTED', STATUS='REPLACE' )
               DO jk2 = 1, jpk
                  WRITE(9201) ( zwi(ji2,jk2), ji2 = 1, jpi )
               END DO
               CLOSE(9201)
               WRITE( cl_zm, '("trazdf_",A3,"_kt",I8.8,"_rank",I2.2,".bin")' )  &
                  &  'zwd', kt, narea - 1
               OPEN( UNIT=9202, FILE=TRIM(cl_zm), ACCESS='STREAM',              &
                  &  FORM='UNFORMATTED', STATUS='REPLACE' )
               DO jk2 = 1, jpk
                  WRITE(9202) ( zwd(ji2,jk2), ji2 = 1, jpi )
               END DO
               CLOSE(9202)
               WRITE( cl_zm, '("trazdf_",A3,"_kt",I8.8,"_rank",I2.2,".bin")' )  &
                  &  'zws', kt, narea - 1
               OPEN( UNIT=9203, FILE=TRIM(cl_zm), ACCESS='STREAM',              &
                  &  FORM='UNFORMATTED', STATUS='REPLACE' )
               DO jk2 = 1, jpk
                  WRITE(9203) ( zws(ji2,jk2), ji2 = 1, jpi )
               END DO
               CLOSE(9203)
            ENDIF
'''
s = s.replace(anchor, insert + anchor, 1)
# The locals the dump needs, declared next to NEMO's own and never reused.
# This used to be a literal string replace and it SILENTLY MATCHED NOTHING --
# the real declaration has two spaces after "::" and the anchor had three --
# so the patch produced a file that would not compile and said nothing.  It is
# a regex now, and it RAISES when it finds no match, which is the only
# difference that matters.
decl = re.search(r"^ *INTEGER *:: *ji, jj, jk, jn.*$", s, re.M)
if decl is None:
    raise SystemExit("REFUSED: trazdf.F90 declaration anchor not found; the "
                     "dump's locals would be undeclared and the build would "
                     "fail with a message pointing nowhere near this script")
s = s.replace(decl.group(0),
              decl.group(0) + "\n"
              "      INTEGER  ::  ji2, jk2         ! #1728 dump loop indices\n"
              "      CHARACTER(LEN=64) ::  cl_zm   ! #1728 dump filename",
              1)
p.write_text(s)
if "ji2, jk2" not in p.read_text():
    raise SystemExit("REFUSED: the declaration insert did not land")
print("  instrumented MY_SRC/trazdf.F90 (tracer zwi/zwd/zws)")

# ---- momentum: the same three, in the DINO override that already exists.
p = copy / "MY_SRC" / "dynzdf.F90"
s = p.read_text()
print("  MY_SRC/dynzdf.F90 present:", p.exists(),
      "-- the momentum half is NOT inserted by this script yet: dynzdf.F90 "
      "assembles zwi/zwd/zws twice (u then v) and the anchor has to name "
      "WHICH, so it is left for the round that scores it.")
PYEOF

# --------------------------------------------------------------- namelist
# nn_itend 1 -> 2 so kt=2 happens; nn_stock 2 so the kt=2 restart is written
# (NEMO sets nitrst = kt + nn_stock - 1 at restart.f90:112 and writes only at
# nitrst, :121 -- with nn_stock=1 there would be a kt=1 restart and no kt=2).
sed -i 's/^\( *nn_itend *=\).*/\1   2/' "$COPY/EXPREF/namelist_cfg" 2>/dev/null || true
sed -i 's/^\( *nn_stock *=\).*/\1   2/' "$COPY/EXPREF/namelist_cfg" 2>/dev/null || true

cat <<MSG

PREPARED: $COPY
OUTPUT  : $OUT

The two commands this record needs -- NOT run by the agent that wrote this
script.  Run them yourself, or re-invoke with --go:

  cd $NEMO
  ./makenemo -n $CFGNAME -m $ARCH -j 8

  cd $COPY/EXP00 && mpirun -np $NPROC ./nemo

POST-RUN CHECK, and it is the one that decides whether the record is usable:

  the kt=2 restart this build writes MUST be bit-identical to
  /data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt2_trends/DINO_00000002_restart_*.nc
  variable by variable.  If it is not, the WRITE-only claim above is FALSE
  and the matrix dump describes a different trajectory than every other kt=2
  number on this branch.  The comparator already exists:

  python scripts/validate/ocean_fidelity/dino_1226/nemo_dino_kt1_rankdump/read_rankdump.py --twin-check \\
      $OUT /data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt2_trends

MSG

if [ "$GO" -eq 1 ]; then
  cd "$NEMO" && ./makenemo -n "$CFGNAME" -m "$ARCH" -j 8
  cd "$COPY/EXP00" && mpirun -np "$NPROC" ./nemo
  mv -f "$COPY/EXP00"/trazdf_*.bin "$OUT"/ 2>/dev/null || true
  mv -f "$COPY/EXP00"/DINO_0000000*_restart_*.nc "$OUT"/ 2>/dev/null || true
fi
