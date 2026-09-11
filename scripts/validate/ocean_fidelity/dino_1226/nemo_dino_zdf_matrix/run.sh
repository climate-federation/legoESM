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
# HOW THE CONFIG COPY IS MADE, and this is a FIX not a style choice.  This
# script used to build the copy with ``cp -a $NEMO/cfgs/DINO $COPY``.
# ``cfgs/DINO`` is 60 GB -- 63 entries, of which 52 are RUN_* directories full
# of this campaign's output -- and the oracle tree sits in a home directory at
# its quota, so that copy filled the disk and the acquisition never ran.  The
# copy is now ``makenemo -r DINO -n <COPY>``, which writes the SKELETON only
# (MY_SRC, EXP00, cpp_*.fcm, BLD): 137 MB measured on the two copies that
# already exist, and zero RUN_* directories.  The script CHECKS that rather
# than trusting it, and every other acquisition on this branch already built
# its copy this way -- this one was the outlier.
#
# USAGE -- ONE command, run by the user (it builds and launches NEMO):
#   scripts/validate/ocean_fidelity/dino_1226/nemo_dino_zdf_matrix/run.sh
#
# Optional environment: ARCH (default conda), NEMO, OUT, NPROC.
set -euo pipefail

NEMO=${NEMO:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
ARCH=${ARCH:-conda}
CFGNAME=DINO_ZDF_MATRIX
COPY=$NEMO/cfgs/$CFGNAME
OUT=$(readlink -m "${OUT:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_zdf_matrix}")
NPROC=${NPROC:-16}
SRCREF=$NEMO/cfgs/DINO/RUN_FROMREST_KT1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------- refusals
# Identical guard to nemo_dino_kt2_trends/run.sh, for the identical reason: a
# relative OUT, a symlinked NEMO or a CFGNAME containing '..' must not be able
# to resolve inside the read-only oracle configuration.
guard() {                       # guard <path being written> [cfgcopy]
  local real
  real=$(readlink -m "$1")
  case "$real/" in
    "$(readlink -m "$NEMO/cfgs/DINO")"/*)
      echo "REFUSED: $1 resolves inside the pristine cfgs/DINO" >&2; exit 2 ;;
    "$(readlink -m "$NEMO/src")"/*)
      echo "REFUSED: $1 resolves inside the pristine src/" >&2; exit 2 ;;
  esac
  if [ "${2:-}" != "cfgcopy" ]; then
    case "$real/" in
      "$(readlink -m "$NEMO")"/*)
        echo "REFUSED: $1 resolves inside the NEMO checkout $NEMO; the" >&2
        echo "  record must be written outside the read-only oracle." >&2
        exit 2 ;;
    esac
  fi
}
guard "$COPY" cfgcopy
guard "$OUT"
case "$CFGNAME" in *..*|*/*) echo "REFUSED: CFGNAME" >&2; exit 2 ;; esac
case "$OUT" in
  /|/tmp|/home|/data) echo "REFUSED: OUT=$OUT is a system directory" >&2
                      exit 2 ;;
esac
if [ -e "$COPY" ]; then
  echo "REFUSED: $COPY already exists." >&2
  echo "  Delete it by hand if you are sure it is stale -- this script will" >&2
  echo "  not decide that for you, because a stale copy still compiles." >&2
  exit 2
fi
if [ ! -d "$SRCREF" ]; then
  echo "REFUSED: $SRCREF not found (the namelists this record must reuse)" >&2
  exit 2
fi
if [ ! -f "$NEMO/arch/arch-$ARCH.fcm" ]; then
  echo "REFUSED: no arch/arch-$ARCH.fcm; set ARCH=<your makenemo -m arch>" >&2
  exit 2
fi

# ------------------------------------------------------------- config copy
# -j 0 creates the configuration WITHOUT compiling, so MY_SRC can be patched
# before a single object file exists.  This is the line that replaced
# ``cp -a $NEMO/cfgs/DINO $COPY``; see the header.
cd "$NEMO"
./makenemo -r DINO -n "$CFGNAME" -m "$ARCH" -j 0
mkdir -p "$OUT"

# THE COPY MUST CARRY NO RUN DIRECTORY.  A skeleton is ~137 MB; cfgs/DINO is
# 60 GB, essentially all of it RUN_* output, and copying that is what filled
# the quota.  Checked on the copy that actually exists, not on the command
# that made it, so any future edit that reintroduces a wholesale copy fails
# here instead of on the disk.
if find "$COPY" -maxdepth 1 -name 'RUN_*' -print -quit | grep -q .; then
  echo "REFUSED: $COPY contains a RUN_* directory, so it is a wholesale copy" >&2
  echo "  of cfgs/DINO and not a makenemo skeleton. Delete it and fix the" >&2
  echo "  line that built it." >&2
  exit 2
fi
echo "config copy: $(du -sh "$COPY" | cut -f1), $(find "$COPY" -maxdepth 1 \
  -name 'RUN_*' | wc -l) RUN_* directories (must be 0)"

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

# --------------------------------------------------------------- build
./makenemo -n "$CFGNAME" -m "$ARCH" -j 8

# The PREPROCESSOR decides whether the writer is in the binary.  A patch that
# compiled away would otherwise produce an empty record that looks like a
# successful run.  Each token must be ABSENT from the pristine source, or it
# discriminates nothing.
PP=$COPY/BLD/ppsrc/nemo/trazdf.f90
for token in "trazdf_" "cl_zm" "ji2, jk2"; do
  if ! grep -qF "$token" "$PP"; then
    echo "REFUSED: '$token' is not in the COMPILED source $PP -- the writer" >&2
    echo "  did not survive preprocessing, so the record would be empty." >&2
    exit 3
  fi
  if grep -qF "$token" "$NEMO/src/OCE/TRA/trazdf.F90"; then
    echo "REFUSED: ppsrc token '$token' also occurs in the PRISTINE source," >&2
    echo "  so finding it above proved nothing. Pick a token the patch adds." >&2
    exit 3
  fi
done
echo "ppsrc check: the matrix writer is in the compiled source, and each"
echo "  token checked is absent from the pristine source"

# --------------------------------------------------------------- namelist
# The run happens in $OUT with the CERTIFIED kt=1 record's namelists -- the
# same pattern as nemo_dino_kt2_trends/run.sh -- so the only difference
# between this record's trajectory and the certified one is the end step.
# nn_itend 1 -> 2 so kt=2 happens; nn_stock 2 so the kt=2 restart is written
# (NEMO sets nitrst = kt + nn_stock - 1 at restart.f90:112 and writes only at
# nitrst, :121 -- with nn_stock=1 there would be a kt=1 restart and no kt=2).
# The OLD value is checked before each edit: a namelist that has already moved
# must not be silently re-edited into something else.
cd "$OUT"
cp -f "$SRCREF/namelist_cfg" "$SRCREF/namelist_ref" .
for x in "$SRCREF"/*.xml; do [ -e "$x" ] && cp -f "$x" .; done
python3 - "$OUT/namelist_cfg" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
for key, want_old, new in (("nn_itend", "1", "2"), ("nn_stock", "1", "2")):
    m = re.search(r"^(\s*%s\s*=\s*)(\S+)(.*)$" % key, s, re.M)
    if m is None:
        raise SystemExit("REFUSED: %s not found in %s" % (key, p))
    if m.group(2).rstrip() != want_old:
        raise SystemExit(
            "REFUSED: %s is %r, expected %r -- this namelist is not the kt=1 "
            "record's, so editing it would produce a record that is not what "
            "this script claims" % (key, m.group(2), want_old))
    s = s[:m.start()] + m.group(1) + new + m.group(3) + s[m.end():]
open(p, "w").write(s)
print("namelist: nn_itend 1 -> 2, nn_stock 1 -> 2")
PY

# ------------------------------------------------------------------- run
ln -sf "$COPY/BLD/bin/nemo.exe" nemo
mpirun -np "$NPROC" ./nemo 2>&1 | tee run_zdf_matrix.log

# ----------------------------------------------------------- record check
# THE CHECK THAT DECIDES WHETHER THE RECORD IS USABLE.  Every inserted
# statement is a WRITE of an array NEMO has already computed, so the kt=2
# restart this build writes MUST be bit-identical to the one
# nemo_dino_kt2_trends/run.sh already produced.  If it is not, the WRITE-only
# claim in the header is FALSE and the matrix dump describes a different
# trajectory than every other kt=2 number on this branch.
KT2REF=${KT2REF:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt2_trends}
python3 "$HERE/../nemo_dino_kt1_rankdump/read_rankdump.py" \
    --twin-check "$OUT" --reference "$KT2REF" --kt 2

cat <<MSG

record written to $OUT
  the three tracer diagonals and the RHS, per rank, at kt = 1 and kt = 2:
    $OUT/trazdf_{zwi,zwd,zws}_kt000000{1,2}_rank??.bin
  the kt=2 restart, twin-checked against $KT2REF above:
    $OUT/DINO_00000002_restart_*.nc

NOT IN THIS RECORD, and named rather than omitted: the MOMENTUM diagonals.
dynzdf.F90 assembles zwi/zwd/zws twice (u then v) and the anchor has to name
WHICH, so the momentum half is left for the round that scores it.
MSG
