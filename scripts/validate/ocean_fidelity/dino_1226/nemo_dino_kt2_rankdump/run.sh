#!/bin/bash
# Acquire NEMO's DINO kt<=2 barotropic record: every dyn_spg_ts stream tagged
# by RANK and by TIME STEP, plus the cross-step barotropic sub-state at the
# start and the end of both steps.  A CONFIG COPY: the pristine cfgs/DINO and
# src/ trees are never written, and the copy is a makenemo SKELETON (~137 MB),
# never `cp -a cfgs/DINO` (60 GB, 52 RUN_* directories, and the quota).
#
# WHY THIS RECORD EXISTS -- two predictions this round MEASURED on the legoESM
# side and could not test on NEMO's, written down before acquisition:
#
#   P1  kt=2 RUNS 68 SUBSTEPS, NOT 45 AND NOT 91.  ll_fw_start is TRUE only at
#       kt=nit000 with the Euler start (dynspg_ts.F90:228-232), and at
#       kt=nit000+1 with ln_bt_fw=.FALSE. NEMO resets it and calls ts_wgt again
#       (:245-250), moving the boxcar centre from nn_e=23 to 2*nn_e=46 and
#       giving icycle=68 with the primary window nonzero on substeps 24..68.
#       legoESM runs 91.  The record decides it two independent ways: the
#       NUMBER of substep files per rank at kt=2, and the ``icycle`` field
#       every substep file's own header carries.
#
#   P2  THE SUB-STATE AT THE START OF kt=2 IS IDENTICALLY ZERO, because
#       ll_bt_av is .TRUE. for nn_bt_flt/=3 (:208-209) and ll_init=ll_bt_av
#       (:214), both OUTSIDE the kt==nit000 block, so :463-470 re-zeroes the
#       sub-state every step.  If it is NOT zero, NEMO carries a barotropic
#       history on this card, decision 33 is live, and the END-of-kt=1 dump in
#       this same record is exactly the value to substitute.
#
# WHAT IT PRODUCES
#   $OUT/substep_r<rank>_kt<kt>_s<jn>.bin    self-describing tiles, kt = 1, 2
#   $OUT/substate_{start,end}_r<rank>_kt<kt>.bin
#                                            sshb_e/sshbb_e/ub_e/ubb_e/
#                                            vb_e/vbb_e, same header
#   $OUT/*_r<rank>_kt<kt>.bin                the once-per-step streams
#   $OUT/DINO_00000002_restart_*.nc          twin-checked below
#
# THE INSTRUMENT IS WRITE-ONLY -- every inserted statement writes an array
# NEMO has already computed -- and the post-run twin check is what turns that
# from a claim into a measurement.
#
# USAGE -- ONE command, run by the user (it builds and launches NEMO):
#   scripts/validate/ocean_fidelity/dino_1226/nemo_dino_kt2_rankdump/run.sh
#
# Optional environment: ARCH (default conda), NEMO, OUT, NPROC, KT2REF.
# ~6 minutes: one makenemo build plus a TWO TIME STEP run.
# THE ONE WRITE INTO THE ORACLE TREE, named rather than left to be found:
# `makenemo` appends the new configuration's name to $NEMO/cfgs/work_cfgs.txt
# and creates $NEMO/cfgs/<CFGNAME>/.  That is makenemo's own bookkeeping, it is
# how every acquisition on this branch already works, and there is no flag to
# turn it off.  The guards below are about everything else.
set -euo pipefail

NEMO=${NEMO:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
ARCH=${ARCH:-conda}
CFGNAME=DINO_KT2_RANKDUMP
COPY=$NEMO/cfgs/$CFGNAME
OUT=$(readlink -m "${OUT:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt2_rankdump}")
NPROC=${NPROC:-16}
SRCREF=$NEMO/cfgs/DINO/RUN_FROMREST_KT1
KT2REF=${KT2REF:-/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt2_trends}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------- refusals
# Identical to nemo_dino_kt1_rankdump/run.sh, for identical reasons: a
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
case "$CFGNAME" in *..*|*/*) echo "REFUSING: CFGNAME" >&2; exit 2 ;; esac
if [ -d "$OUT/.git" ] || git -C "$OUT" rev-parse --git-dir >/dev/null 2>&1
then
  echo "REFUSED: OUT=$OUT is inside a git checkout; NEMO writes hundreds of" >&2
  echo "  files there and they are not artifacts this repo should carry." >&2
  echo "  (OUT=. resolves here, which is how this was found.)" >&2
  exit 2
fi
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
# -j 0 creates the configuration WITHOUT compiling, so MY_SRC can be patched
# before a single object file exists.  NEVER `cp -a cfgs/DINO`: that is 60 GB
# of this campaign's RUN_* output and it fills the home quota.
./makenemo -r DINO -n "$CFGNAME" -m "$ARCH" -j 0

# THE COPY MUST CARRY NO RUN DIRECTORY, checked on the copy that exists rather
# than on the command that made it.
if find "$COPY" -maxdepth 1 -name 'RUN_*' -print -quit | grep -q .; then
  echo "REFUSING: $COPY contains a RUN_* directory, so it is a wholesale" >&2
  echo "  copy of cfgs/DINO and not a makenemo skeleton." >&2
  exit 2
fi
echo "config copy: $(du -sh "$COPY" | cut -f1), 0 RUN_* directories"

guard "$COPY/MY_SRC/dynspg_ts.F90" cfgcopy
python3 "$HERE/kt2_rankdump_patch.py" "$COPY/MY_SRC/dynspg_ts.F90"
./makenemo -n "$CFGNAME" -m "$ARCH" -j 8

# The PREPROCESSOR decides whether the writer is in the binary.  Each token
# must be ABSENT from the pristine source, or it discriminates nothing.
PP=$COPY/BLD/ppsrc/nemo/dynspg_ts.f90
for token in "TRIM(cl_st)" "substate_start_r" "substate_end_r" "_kt', kt, '_s'"; do
  if ! grep -qF "$token" "$PP"; then
    echo "REFUSING: '$token' is not in the COMPILED source $PP -- the writer" >&2
    echo "  did not survive preprocessing, so the record would be empty." >&2
    exit 3
  fi
  if [ ! -f "$NEMO/cfgs/DINO/MY_SRC/dynspg_ts.F90" ]; then
    echo "REFUSED: the pristine source is not where this check looks, so" >&2
    echo "  'absent from the pristine source' would pass vacuously." >&2
    exit 3
  fi
  if grep -qF "$token" "$NEMO/cfgs/DINO/MY_SRC/dynspg_ts.F90"; then
    echo "REFUSING: ppsrc token '$token' also occurs in the PRISTINE source," >&2
    echo "  so finding it above proved nothing. Pick a token the patch adds." >&2
    exit 3
  fi
done
echo "ppsrc check: the rank+kt-tagged writer and both sub-state dumps are in"
echo "  the compiled source, and each token checked is absent from the"
echo "  pristine source"

# --------------------------------------------------------------- namelist
# Run in $OUT with the CERTIFIED kt=1 record's namelists, two steps, and the
# kt=2 restart written (restart.f90:112 sets nitrst = kt + nn_stock - 1 and
# :121 writes only at nitrst).  The OLD value is checked before each edit.
mkdir -p "$OUT"
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
        raise SystemExit("REFUSING: %s not found in %s" % (key, p))
    if m.group(2).rstrip() != want_old:
        raise SystemExit(
            "REFUSING: %s is %r, expected %r -- this namelist is not the kt=1 "
            "record's, so editing it would produce a record that is not what "
            "this script claims" % (key, m.group(2), want_old))
    s = s[:m.start()] + m.group(1) + new + m.group(3) + s[m.end():]
open(p, "w").write(s)
print("namelist: nn_itend 1 -> 2, nn_stock 1 -> 2")
PY

# ------------------------------------------------------------------- run
ln -sf "$COPY/BLD/bin/nemo.exe" nemo
mpirun -np "$NPROC" ./nemo 2>&1 | tee run_kt2_rankdump.log

# ----------------------------------------------------------- twin check
# The instrumentation must not have changed the PHYSICS: the kt=2 restart this
# build writes must be bit-identical to the one nemo_dino_kt2_trends/run.sh
# produced with no source patch at all.
python3 "$HERE/../nemo_dino_kt1_rankdump/read_rankdump.py" \
    --twin-check "$OUT" --reference "$KT2REF" --kt 2

# --------------------------------------------------- the two predictions
python3 - "$OUT" <<'PY'
import glob, os, sys
d = sys.argv[1]
for kt in (1, 2):
    n = sorted({int(os.path.basename(f).split("_s")[1][:-4])
                for f in glob.glob(os.path.join(d, f"substep_r*_kt{kt:08d}_s*.bin"))})
    print(f"P1  kt={kt}: {len(n)} substeps per rank"
          + ("" if not n else f" (1..{max(n)})"))
print("P1 PREDICTED: 45 at kt=1, 68 at kt=2. legoESM runs 45 and 91.")
print("P2: read substate_start_r*_kt00000002.bin with read_rankdump.py's "
      "header layout; PREDICTED identically zero (ll_init is TRUE every step "
      "for nn_bt_flt=2, dynspg_ts.F90:208-214, :463-470).")
PY

echo
echo "record written to $OUT"
