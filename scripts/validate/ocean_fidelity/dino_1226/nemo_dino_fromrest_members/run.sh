#!/bin/bash
# Acquire NEMO's FROM-REST 4-member ensemble for the spread floor, as CONFIG
# COPIES.  The pristine cfgs/DINO and src/ trees are never written.
#
# WHY.  legoESM and NEMO, both from rest on this card, differ by a 3-D
# temperature rms that grows from ~2.1e-3 K at day 30 to ~4.3e-3 K at day 360.
# Whether that is a remaining operator mismatch or simply chaos cannot be read
# off the number: it has to be compared with the spread each model generates on
# its own from an infinitesimally perturbed initial state.
#
# NO SOURCE PATCH IS NEEDED.  DINO already carries the perturbation:
# cfgs/DINO/MY_SRC/usrdef_istate.F90:177-183 (compiled at
# BLD/ppsrc/nemo/usrdef_istate.f90:188-194) adds, under nn_pert_seed /= 0,
#
#    1.e-10 * SIN( REAL( NINT(pdept)*73 + NINT(gphit*1000)*179
#                        + nn_pert_seed*997 ) ) * tmask
#
# and nn_pert_seed is already declared (MY_SRC/usrdef_nam.F90:81) and already
# in the namusr_def list (:126).  So each member is ONE NAMELIST LINE, and the
# binary is the certified one.  legoESM reproduces that same expression in
# verdict360_fromrest.nemo_istate_perturbation, which is checked against it.
#
# THIS SCRIPT REFUSES TO RUN UNTIL PHASE 0 HAS SPOKEN.
# A claim review showed the floor may be so small at day 360 that "the models
# are distinguishable" is fixed before any member runs -- the restart-seeded
# record has the spread growing 2200x between day 90 and day 360 while the
# from-rest gap grows 1.8x, which is the signature of a bounded deterministic
# offset rather than a diverging trajectory.  PHASE 0 (legoESM only, no NEMO
# time at all) measures the floor and classifies it against a preregistered
# window.  These NEMO members are spent only if that window is met.
#
# WHAT IT PRODUCES
#   $OUT/member_<s>/DINO_*_restart_*.nc   360 days from rest, seed s
#   $OUT/member_<s>/namelist_cfg          with nn_pert_seed = s, diffable
#
# USAGE
#   scripts/validate/ocean_fidelity/dino_1226/nemo_dino_fromrest_members/run.sh
#
# Optional environment: ARCH (default conda), NEMO, OUT, NPROC, SEEDS, PHASE0.
set -euo pipefail

NEMO=${NEMO:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
ARCH=${ARCH:-conda}
CFGNAME=DINO_FROMREST_MEMBERS
COPY=$NEMO/cfgs/$CFGNAME
OUT=$(readlink -m "${OUT:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_fromrest_members}")
NPROC=${NPROC:-16}
SEEDS=${SEEDS:-"1 2 3 4"}
SRCREF=$NEMO/cfgs/DINO/RUN_FROMREST_Y1
PHASE0=${PHASE0:-/data/abyssal/dbalwada/dino_fromrest_y1/verdict360_fromrest/phase0_floor.json}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ------------------------------------------------------- the phase-0 gate
if [ ! -f "$PHASE0" ]; then
  echo "REFUSING: $PHASE0 does not exist." >&2
  echo "  PHASE 0 (legoESM-only, no NEMO time) has not been scored, and its" >&2
  echo "  entire purpose is to say whether these members can produce an" >&2
  echo "  informative answer.  Run:" >&2
  echo "    python scripts/validate/ocean_fidelity/dino_1226/\\" >&2
  echo "      verdict360_fromrest.py --phase0        # writes the seeds" >&2
  echo "    ... run the four legoESM members ..." >&2
  echo "    python .../verdict360_fromrest.py --score-phase0" >&2
  exit 2
fi
# The decision is READ from phase 0's own artifact, never re-derived here.  A
# constant copied into two files is a constant that goes stale in one of them.
python3 - "$PHASE0" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if "phase1_worth_running" not in d:
    raise SystemExit(
        "REFUSING: %s predates the derived phase-1 test (no "
        "'phase1_worth_running' key).  Re-run verdict360_fromrest.py "
        "--score-phase0." % sys.argv[1])
if not d["phase1_worth_running"]:
    raise SystemExit(
        "REFUSING: phase 0 says these members cannot change the answer.\n"
        "  day-360 gap/(2*floor) = %.3f; NEMO's own spread would have to be "
        "%.3f x legoESM's for phase 1 to flip the verdict, and the "
        "restart-seeded record has NEMO 9-16600x TIGHTER.\n"
        "  Do not spend the NEMO time." % (d.get("ratio_360", float("nan")),
                                           d.get("required_nemo_spread_factor",
                                                 float("nan"))))
print("phase-0 gate: PROCEED (gap/(2*floor) = %.3f, required NEMO spread "
      "factor %.3f)" % (d.get("ratio_360", float("nan")),
                        d.get("required_nemo_spread_factor", float("nan"))))
PY

# ---------------------------------------------------------------- refusals
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
        echo "REFUSING: $1 resolves inside the NEMO checkout $NEMO" >&2
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
  echo "REFUSING: $COPY already exists; delete it by hand if it is stale." >&2
  exit 2
fi
if [ ! -d "$SRCREF" ]; then
  echo "REFUSING: $SRCREF not found (the 1-year from-rest namelists)" >&2
  exit 2
fi
if [ ! -f "$NEMO/arch/arch-$ARCH.fcm" ]; then
  echo "REFUSING: no arch/arch-$ARCH.fcm; set ARCH=<your makenemo -m arch>" >&2
  exit 2
fi

# ---------------------------------------------------------------- build
cd "$NEMO"
./makenemo -r DINO -n "$CFGNAME" -m "$ARCH" -j 8
# The perturbation must be IN the binary.  It is a namelist switch on an
# existing MY_SRC block, so this checks the COMPILED source rather than
# trusting that the block survived.
PP=$COPY/BLD/ppsrc/nemo/usrdef_istate.f90
grep -qF "nn_pert_seed" "$PP" || {
  echo "REFUSING: nn_pert_seed is not in the COMPILED $PP, so every member" >&2
  echo "  would be the SAME unperturbed run and the spread would be 0." >&2
  exit 3; }
echo "ppsrc check: the nn_pert_seed perturbation is in the compiled source"

# ---------------------------------------------------------------- members
for s in $SEEDS; do
  d=$OUT/member_$s
  guard "$d"
  mkdir -p "$d"
  cd "$d"
  cp -f "$SRCREF/namelist_cfg" "$SRCREF/namelist_ref" .
  for x in "$SRCREF"/*.xml; do [ -e "$x" ] && cp -f "$x" .; done
  python3 - "$d/namelist_cfg" "$s" <<'PY'
import re, sys
p, seed = sys.argv[1], int(sys.argv[2])
s = open(p).read()
m = re.search(r"^(\s*nn_pert_seed\s*=\s*)(\S+)(.*)$", s, re.M)
if m is None:
    # namusr_def carries the key with a default of 0; add it explicitly so the
    # member's own namelist RECORDS the choice instead of relying on a default.
    m2 = re.search(r"^(&namusr_def.*)$", s, re.M)
    if m2 is None:
        raise SystemExit("REFUSING: no &namusr_def block in %s" % p)
    s = s[:m2.end()] + ("\n   nn_pert_seed = %d   ! #1728 from-rest ensemble "
                        "member seed\n" % seed) + s[m2.end():]
else:
    s = s[:m.start()] + m.group(1) + str(seed) + m.group(3) + s[m.end():]
open(p, "w").write(s)
print("member seed %d written into %s" % (seed, p))
PY
  grep -E "^\s*(nn_pert_seed|nn_itend|rn_Dt|ln_rstart)\s*=" namelist_cfg
  ln -sf "$COPY/BLD/bin/nemo.exe" nemo
  mpirun -np "$NPROC" ./nemo 2>&1 | tee "run_member_$s.log"
  cd "$OUT"
done

# The members must actually DIFFER.  A seed that never reached the code would
# give four bit-identical runs and a floor of exactly zero, which would then be
# read as "the models are distinguishable" -- the exact false positive this
# whole harness exists to avoid.
python3 - "$OUT" $SEEDS <<'PY'
import glob, hashlib, os, sys
root, seeds = sys.argv[1], sys.argv[2:]
h = {}
for s in seeds:
    f = sorted(glob.glob(os.path.join(root, "member_%s" % s,
                                      "DINO_*_restart*.nc")))
    if not f:
        raise SystemExit("REFUSING: member %s produced no restart" % s)
    h[s] = hashlib.sha256(open(f[-1], "rb").read()).hexdigest()[:16]
for s in seeds:
    print("  member %s final restart sha256[:16] = %s" % (s, h[s]))
if len(set(h.values())) != len(seeds):
    raise SystemExit(
        "REFUSING: two or more members are bit-identical, so nn_pert_seed did "
        "not reach the initial state and the measured spread would be 0.")
print("all members differ, as the perturbation requires")
PY

echo
echo "members written to $OUT"
echo "then score with:"
echo "  python scripts/validate/ocean_fidelity/dino_1226/verdict360_fromrest.py \\"
echo "    --phase1 --nemo-root $OUT"
