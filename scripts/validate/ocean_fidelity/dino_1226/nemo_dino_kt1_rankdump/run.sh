#!/bin/bash
# Acquire NEMO's DINO kt=1 barotropic record WITHOUT the 16-rank filename race,
# as a CONFIG COPY.  The pristine cfgs/DINO and src/ trees are never written.
#
# WHY A COPY.  The existing record (RUN_FROMREST_KT1) is unreadable at the
# substep level: dyn_spg_ts writes every debug stream to a FIXED filename under
# guards that test the timestep and never the MPI rank (dynspg_ts.F90:206,
# :926), and the run used jpni x jpnj = 2 x 8, so all sixteen ranks OPEN the
# same path with STATUS='REPLACE'.  spg_kt1_barotropic_ladder.py --audit-only
# proves the interleave three independent ways.  The previous fix edited
# cfgs/DINO/MY_SRC in place behind a restore trap; that is a backup-and-restore
# dance around the read-only oracle, and one aborted run leaves it rewritten.
# A makenemo copy cannot.
#
# WHAT IT PRODUCES
#   $OUT/substep_r<rank>_s<substep>.bin   16 x 45 self-describing tiles
#   $OUT/*_r<rank>.bin                    the once-per-step streams, rank-tagged
#   $OUT/DINO_00000001_restart_*.nc       the twin restart, checked byte-for-byte
#                                          against RUN_FROMREST_KT1's
#
# USAGE
#   scripts/validate/ocean_fidelity/dino_1226/nemo_dino_kt1_rankdump/run.sh
#
# Optional environment: ARCH (default conda), NEMO, OUT, NPROC.
# ~5 minutes: one makenemo build plus a ONE TIME STEP run.
set -euo pipefail

NEMO=${NEMO:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
ARCH=${ARCH:-conda}
CFGNAME=DINO_KT1_RANKDUMP
COPY=$NEMO/cfgs/$CFGNAME
# ABSOLUTE, resolved HERE. A relative OUT is canonicalised against the
# INVOCATION directory by the guard below but created after `cd "$NEMO"`, so
# `OUT=cfgs/SHARED` from anywhere else passed the guard and then landed inside
# the oracle anyway.
OUT=$(readlink -m "${OUT:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_rankdump}")
NPROC=${NPROC:-16}
SRCREF=$NEMO/cfgs/DINO/RUN_FROMREST_KT1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------- refusals
# 1. Never write inside the pristine oracle config or the shared source tree.
#    Checked on the RESOLVED paths, not on the strings, so a symlinked NEMO or
#    a CFGNAME containing '..' cannot slip past.
guard() {                       # guard <path being written> [allow-cfg-copy]
  # `readlink -m` canonicalises WITHOUT requiring the path to exist. `-f`
  # returns EMPTY when an intermediate directory is missing, and an empty
  # string matches none of the patterns below, so every not-yet-created path
  # walked straight through -- and `local real=$(...)` hides the non-zero
  # exit from `set -e`, so nothing said so.
  local real
  real=$(readlink -m "$1")
  case "$real/" in
    "$(readlink -m "$NEMO/cfgs/DINO")"/*)
      echo "REFUSING: $1 resolves inside cfgs/DINO (read-only oracle)" >&2
      exit 2 ;;
    "$(readlink -m "$NEMO/src")"/*)
      echo "REFUSING: $1 resolves inside src/ (read-only oracle)" >&2
      exit 2 ;;
  esac
  # The RECORD may not land anywhere inside the oracle checkout at all. An
  # earlier version only named cfgs/DINO and src/, and a reviewer showed
  # OUT=$NEMO/cfgs/SHARED walked straight through it and would have clobbered
  # the namelist_ref every configuration includes.
  if [ "${2:-}" != "cfgcopy" ]; then
    case "$real/" in
      "$(readlink -m "$NEMO")"/*)
        echo "REFUSING: $1 resolves inside the NEMO checkout $NEMO; the" >&2
        echo "  record must be written outside the read-only oracle." >&2
        exit 2 ;;
    esac
  fi
}
guard "$COPY" cfgcopy
guard "$OUT"
case "$OUT" in
  /|/tmp|/home|/data) echo "REFUSING: OUT=$OUT is a system directory" >&2
                      exit 2 ;;
esac
case "$(readlink -m "$NEMO")/" in
  "$OUT"/*) echo "REFUSING: OUT=$OUT CONTAINS the NEMO checkout" >&2
            exit 2 ;;
esac

# 2. Never reuse a config copy.  A half-built or previously-patched copy is the
#    one way this script could produce a record that is not what it claims.
if [ -e "$COPY" ]; then
  echo "REFUSING: $COPY already exists." >&2
  echo "  Delete it by hand if you are sure it is stale -- this script will" >&2
  echo "  not decide that for you, because a stale copy still compiles." >&2
  exit 2
fi
if [ ! -d "$SRCREF" ]; then
  echo "REFUSING: $SRCREF not found (the namelists this record must reuse)" >&2
  exit 2
fi
if [ ! -f "$NEMO/arch/arch-$ARCH.fcm" ]; then
  echo "REFUSING: no arch/arch-$ARCH.fcm; set ARCH=<your makenemo -m arch>" >&2
  exit 2
fi

# ---------------------------------------------------------------- build
cd "$NEMO"
# -j 0 creates the configuration without compiling, so MY_SRC can be patched
# before a single object file exists.
./makenemo -r DINO -n "$CFGNAME" -m "$ARCH" -j 0
guard "$COPY/MY_SRC/dynspg_ts.F90" cfgcopy
python3 "$HERE/rankdump_patch.py" "$COPY/MY_SRC/dynspg_ts.F90"
./makenemo -n "$CFGNAME" -m "$ARCH" -j 8

# The PREPROCESSOR is what decides whether the writer is in the binary.  A
# patch that compiled away (wrong guard, wrong key) would otherwise produce an
# empty record that looks like a successful run.
PP=$COPY/BLD/ppsrc/nemo/dynspg_ts.f90
# Each token must be ABSENT from the pristine source, or it discriminates
# nothing. ACTION='WRITE' was in this list and already occurs once unpatched.
for token in "TRIM(cl_rk)" "substep_r" "cl_sub"; do
  if ! grep -qF "$token" "$PP"; then
    echo "REFUSING: '$token' is not in the COMPILED source $PP -- the writer" >&2
    echo "  did not survive preprocessing, so the record would be empty." >&2
    exit 3
  fi
done
for token in "TRIM(cl_rk)" "substep_r" "cl_sub"; do
  if grep -qF "$token" "$NEMO/cfgs/DINO/MY_SRC/dynspg_ts.F90"; then
    echo "REFUSING: ppsrc token '$token' also occurs in the PRISTINE source," >&2
    echo "  so finding it above proved nothing. Pick a token the patch adds." >&2
    exit 3
  fi
done
echo "ppsrc check: the rank-tagged writer is in the compiled source, and each"
echo "  token checked is absent from the pristine source"

# ---------------------------------------------------------------- run
mkdir -p "$OUT"
cd "$OUT"
cp -f "$SRCREF/namelist_cfg" "$SRCREF/namelist_ref" .
for x in "$SRCREF"/*.xml; do [ -e "$x" ] && cp -f "$x" .; done
ln -sf "$COPY/BLD/bin/nemo.exe" nemo
mpirun -np "$NPROC" ./nemo 2>&1 | tee run_kt1_rankdump.log

# ---------------------------------------------------------------- twin check
# The instrumentation must not have changed the PHYSICS.  Proven, not asserted:
# every variable of every restart tile against the untouched record.
python3 "$HERE/read_rankdump.py" --twin-check "$OUT" --reference "$SRCREF"

echo
echo "record written to $OUT"
echo "  stitch a substep:  python3 $HERE/read_rankdump.py --run-dir $OUT --substep 1"
echo
echo "NOT YET A DROP-IN FOR THE LADDER GATE. spg_kt1_barotropic_ladder.py"
echo "  still opens the OLD fixed filenames (substep_dump.bin,"
echo "  spg_dump_*.bin, sbc_dump_qns.bin); every one of them is rank-tagged"
echo "  here, so pointing it at this directory would read nothing and its"
echo "  own audit would report INDETERMINATE. Extending it to the tagged"
echo "  names is the next change, and it needs this record to exist first."
