#!/bin/bash
# Acquire NEMO's DINO record for DAYS 1..5 from rest, as a CONFIG COPY.  The
# pristine cfgs/DINO and src/ trees are never written.
#
# WHY THIS RECORD IS NEEDED, and what question it answers.
# The from-rest 3-D temperature gap between legoESM and NEMO is, MEASURED
# (twin_nemo_ts_maps.py against RUN_TRAJ, matched protocol both sides):
#
#     step 1    1.1760e-07 K     (nemo_dino_step1_gate.py)
#     step 2    6.1224e-06 K     (kt2_leapfrog_gate.py)
#     day 10    7.6779e-04 K     (kt 320)
#     day 20    1.0215e-03 K     (kt 640)
#     day 30    9.1382e-04 K     (kt 960)
#
# It is FLAT from day 10 onward and 4 orders larger than at step 2, so the
# whole of it is established somewhere inside the first ten days -- and NEMO
# has NO record between kt=2 and kt=320.  The Phase-0 distinguishability floor
# is 1.2455e-05 K at day 30 and GROWS with time, so the gap is already >= 60x
# the floor at day 10 and "the earliest day the gap exceeds 10x the floor"
# cannot be resolved any finer than "<= day 10" from what exists.
#
# WHAT IT PRODUCES
#   $OUT/DINO_0000003[2]_restart_*.nc  and the same at kt 64, 96, 128, 160
#   i.e. one restart per DAY for days 1..5 (32 steps = 1 day at rn_Dt = 2700).
#
# The ONLY namelist edits are nn_itend 1 -> 160 and nn_stock 1 -> 32, both
# checked against their current value before being written, so a namelist that
# has already moved cannot be silently re-edited into something else.  No
# source patch: this record needs no instrumentation the certified binary
# lacks.
#
# NOT RUN BY THE SESSION THAT WROTE IT.  Producing it costs a makenemo build
# and a 160-step MPI run; the working rule for this branch is that the agent
# writes the config copy and STOPS when a new oracle record is needed.
#
# USAGE
#   scripts/validate/ocean_fidelity/dino_1226/nemo_dino_earlydays/run.sh
#
# Optional environment: ARCH (default conda), NEMO, OUT, NPROC.
set -euo pipefail

NEMO=${NEMO:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
ARCH=${ARCH:-conda}
CFGNAME=DINO_EARLYDAYS
COPY=$NEMO/cfgs/$CFGNAME
OUT=$(readlink -m "${OUT:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_earlydays}")
NPROC=${NPROC:-16}
SRCREF=$NEMO/cfgs/DINO/RUN_FROMREST_KT1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------- refusals
# Identical guard to nemo_dino_kt2_trends/run.sh, for the identical reason: a
# relative OUT, a symlinked NEMO or a CFGNAME containing '..' must not be able
# to resolve inside the read-only oracle.
guard() {                       # guard <path being written> [cfgcopy]
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
./makenemo -r DINO -n "$CFGNAME" -m "$ARCH" -j 8

# ---------------------------------------------------------------- namelist
mkdir -p "$OUT"
cd "$OUT"
cp -f "$SRCREF/namelist_cfg" "$SRCREF/namelist_ref" .
for x in "$SRCREF"/*.xml; do [ -e "$x" ] && cp -f "$x" .; done
python3 - "$OUT/namelist_cfg" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
for key, want_old, new in (("nn_itend", "1", "160"), ("nn_stock", "1", "32")):
    m = re.search(r"^(\s*%s\s*=\s*)(\S+)(.*)$" % key, s, re.M)
    if m is None:
        raise SystemExit("REFUSING: %s not found in %s" % (key, p))
    if m.group(2).rstrip() != want_old:
        raise SystemExit(
            "REFUSING: %s is %r, expected %r -- this namelist is not the "
            "kt=1 record's, so editing it would produce a record that is "
            "not what this script claims" % (key, m.group(2), want_old))
    s = s[:m.start()] + m.group(1) + new + m.group(3) + s[m.end():]
open(p, "w").write(s)
print("namelist: nn_itend 1 -> 160, nn_stock 1 -> 32")
PY
grep -E "^\s*(nn_itend|nn_stock|rn_Dt|ln_rstart|ln_tra_trd|ln_dyn_trd)\s*=" namelist_cfg

# ---------------------------------------------------------------- run
ln -sf "$COPY/BLD/bin/nemo.exe" nemo
mpirun -np "$NPROC" ./nemo 2>&1 | tee run_earlydays.log

# ----------------------------------------------------------- record check
python3 "$HERE/../nemo_dino_kt1_rankdump/read_rankdump.py" \
    --record-check "$OUT" --reference "$SRCREF" --kt 160 \
    --allow nn_itend,nn_stock

echo
echo "record written to $OUT"
echo "then score legoESM against each day in ONE command per day:"
for d in 1 2 3 4 5; do
  echo "  python scripts/validate/ocean_fidelity/dino_1226/twin_nemo_ts_maps.py \\"
  echo "    --run-dino-dir <a run with --snapshot-every-days 1> --day $d \\"
  echo "    --nemo-kt $((d * 32)) --nemo-run $OUT --output-dir <dir>"
done
