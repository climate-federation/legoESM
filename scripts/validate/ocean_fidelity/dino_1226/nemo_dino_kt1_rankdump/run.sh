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
OUT=${OUT:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_rankdump}
NPROC=${NPROC:-16}
SRCREF=$NEMO/cfgs/DINO/RUN_FROMREST_KT1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------- refusals
# 1. Never write inside the pristine oracle config or the shared source tree.
#    Checked on the RESOLVED paths, not on the strings, so a symlinked NEMO or
#    a CFGNAME containing '..' cannot slip past.
guard() {                       # guard <path being written>
  local real
  real=$(readlink -f "$1")
  case "$real/" in
    "$(readlink -f "$NEMO/cfgs/DINO")"/*)
      echo "REFUSING: $1 resolves inside cfgs/DINO (read-only oracle)" >&2
      exit 2 ;;
    "$(readlink -f "$NEMO/src")"/*)
      echo "REFUSING: $1 resolves inside src/ (read-only oracle)" >&2
      exit 2 ;;
  esac
}
guard "$COPY"
guard "$OUT"

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
guard "$COPY/MY_SRC/dynspg_ts.F90"
python3 "$HERE/rankdump_patch.py" "$COPY/MY_SRC/dynspg_ts.F90"
./makenemo -n "$CFGNAME" -m "$ARCH" -j 8

# The PREPROCESSOR is what decides whether the writer is in the binary.  A
# patch that compiled away (wrong guard, wrong key) would otherwise produce an
# empty record that looks like a successful run.
PP=$COPY/BLD/ppsrc/nemo/dynspg_ts.f90
for token in "TRIM(cl_rk)" "substep_r" "ACTION='WRITE'"; do
  if ! grep -qF "$token" "$PP"; then
    echo "REFUSING: '$token' is not in the COMPILED source $PP -- the writer" >&2
    echo "  did not survive preprocessing, so the record would be empty." >&2
    exit 3
  fi
done
echo "ppsrc check: the rank-tagged WRITE-only writer is in the compiled source"

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
echo "  re-score the ladder: spg_kt1_barotropic_ladder.py --run-dir $OUT"
