#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run NEMO, makenemo or
# mpirun from its sandbox; it wrote this file and stopped.
#
#   Usage:  run.sh
#
# WHAT THIS ACQUIRES, AND WHY.  NEMO's GYRE from-rest record has restarts at
# kt = 10 and then NOTHING until kt = 180 (day 30).  The gap between the two
# models is 2.768e-03 K after ten steps and 1.4241e-02 K at day 30 -- so
# nineteen per cent of the day-30 gap is already present at step 10 and the
# remaining 170 steps multiply it by only 5.1.  Where the rest of it is made
# cannot be read off two points.  This fills days 1..30 at DAY resolution.
#
# Days 0..10 need NO acquisition: the certified card's own MY_SRC writer is
# gated on `kstp >= nit000 .AND. kstp <= nit000 + 59`
# (cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90), so the year run's
# pristine member already wrote SIXTY per-step entry states and they have been
# on disk since that run.  This script's run writes them again, which is the
# instrument's own reproducibility check.
#
# NO BUILD, AND THAT IS A DELIBERATE DEPARTURE.  The preregistration's
# ASKED/UNASKED table, item 1, records it: this stages a new RUN DIRECTORY
# against the ALREADY-CERTIFIED BINARY rather than making a `makenemo -r ... -n
# ...` config copy.  A second binary inside a comparison whose only intended
# variable is the time axis cannot be separated from a physics difference, and
# the check below is stronger than any build-provenance hash:
#
#   the kt = 180 restart written here must be BYTE-IDENTICAL to the year
#   control's kt = 180 restart.
#
# That single `cmp` proves the restart cadence changed no arithmetic AND that
# this record and the scored year are ONE trajectory.  If it ever fails, this
# record is not admissible and the script refuses.
#
# NEMO'S TREND DIAGNOSTICS ARE NOT ENABLED, also on the table (item 2).  Three
# reasons, none of them taste: &namtrd's 3-D outputs go through iom_put and
# this card compiles without key_xios (cpp keys are `key_qco key_vco_1d3d
# key_RK3`), so they would write NOTHING; NEMO 5 itself raises
# 'The trends diagnostics are a work in progress: they are not yet fully
# tested or functional' for every trend flag under RK3 (src/OCE/TRD/trdini.F90,
# trd_init); and the campaign's own rule forbids quoting an unclosed trend
# bucket at all.  The per-operator record this campaign uses is the card's OWN
# stage writer, which fires at kstp == nit000.  A per-operator record AT a day
# boundary is therefore acquired by RESTARTING from that day's restart and
# running ONE instrumented step -- a separate run, a separate choice, and not
# this one.

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R41ADVSP
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly YEAR_RUN=$L2/year_fromrest
readonly TARGET_RUN=$L2/year_owners
# The run directory is named nemo_seed0 because that is what it IS -- the
# unperturbed control, seed 0 by the year preregistration's own definition --
# and because the scoring harness's reader resolves <root>/nemo_seed<N>.  A
# differently-named directory would have needed a code change whose only
# content is a second naming convention.
readonly RUN_DIR=$TARGET_RUN/nemo_seed0
# The record's cadence.  Fixed by the preregistration, ASKED item 3.
readonly NN_ITEND=180           # 30 days at rn_Dt = 14400 s, nn_leapy = 30
readonly NN_STOCK=6             # a restart every DAY
readonly NN_WRITE=180           # one output file; scoring reads restarts
readonly ENTRY_DUMPS=60         # the card's writer covers nit000..nit000+59

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly HARNESS=$here/../nemo_testcase_l2_gyre_year_owners.py
source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
certified_binary=$source_cfg/BLD/bin/nemo.exe
reference_restart=$YEAR_RUN/nemo_pristine/GYRE_OMIP_L2_P3_00000180_restart.nc
control_restart=$YEAR_RUN/nemo_seed0/GYRE_OMIP_L2_P3_00000180_restart.nc

# ------------------------------------------------------------- preconditions
[[ -d "$source_cfg/EXP00" ]] || { printf 'REFUSE: no %s/EXP00\n' "$source_cfg" >&2; exit 64; }
[[ -x "$certified_binary" ]] || { printf 'REFUSE: no certified binary at %s\n' "$certified_binary" >&2; exit 64; }
[[ -f "$HARNESS" ]] || { printf 'REFUSE: no harness at %s\n' "$HARNESS" >&2; exit 64; }
for f in "$reference_restart" "$control_restart" \
         "$YEAR_RUN/nemo_pristine/nemo"; do
  [[ -f "$f" ]] || { printf 'REFUSE: %s is missing; the byte-identity check is what licenses this record and it cannot be skipped\n' "$f" >&2; exit 64; }
done
# The two year-run day-30 restarts must already agree with each other, or the
# reference this record is admitted against is not one thing.
cmp -s "$reference_restart" "$control_restart" || {
  printf 'REFUSE: the year run pristine and seed-0 day-30 restarts DIFFER; there is no single reference to admit against\n' >&2
  exit 65
}

# ------------------------------------------------------------------- refusals
guard() {
  local real
  real=$(readlink -m "$1")
  case "$real/" in
    "$(readlink -m "$NEMO_ROOT")"/*)
      printf 'REFUSE: %s resolves inside the NEMO checkout (read-only oracle)\n' "$1" >&2; exit 2 ;;
  esac
}
guard "$RUN_DIR"
[[ ! -e "$RUN_DIR" ]] || { printf 'REFUSE: %s already exists; delete it by hand if it is stale.\n' "$RUN_DIR" >&2; exit 64; }

free_kb=$(df -Pk "$TARGET_RUN" | awk 'NR==2 {print $4}')
[[ "$free_kb" -ge 2097152 ]] || { printf 'REFUSE: %s has %s kB free, under the 2 GB floor\n' "$TARGET_RUN" "$free_kb" >&2; exit 68; }

# ------------------------------------------------------------------ staging
mkdir -p "$RUN_DIR"
for name in namelist_ref namelist_top_cfg namelist_top_ref \
            namelist_pisces_cfg namelist_pisces_ref context_nemo.xml \
            file_def_nemo.xml iodef.xml axis_def_nemo.xml \
            domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml \
            field_def_nemo-pisces.xml; do
  cp -L "$source_cfg/EXP00/$name" "$RUN_DIR/$name"
done
# THE BINARY MUST BE THE ONE THE REFERENCE RUN USED, and that is checked
# against the binary itself, not against a provenance file.  The year round's
# run.sh copies ONE binary.sha256 into EVERY run directory, so
# nemo_pristine/binary.sha256 names the PATCHED (YRPERT) binary while
# nemo_pristine/nemo IS the certified (R41ADVSP) one -- a provenance file that
# does not describe the binary beside it.  Found by an independent review of
# this round.  So: hash the executable.
if ! cmp -s "$certified_binary" "$YEAR_RUN/nemo_pristine/nemo"; then
  printf 'REFUSE: %s differs from the binary the reference run used (%s).\n' \
    "$certified_binary" "$YEAR_RUN/nemo_pristine/nemo" >&2
  printf '  This record would not be the same model as the year it is admitted against.\n' >&2
  exit 65
fi
cp "$certified_binary" "$RUN_DIR/nemo"
sha256sum "$certified_binary" "$YEAR_RUN/nemo_pristine/nemo" \
  "$YEAR_RUN/nemo_seed0/nemo" >"$RUN_DIR/binary.sha256"
( cd "$source_cfg" && find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum ) \
  >"$RUN_DIR/source_cfg.sha256"

# The namelist writer.  It PRINTS the diff against the certified card as
# PARSED ASSIGNMENTS (a line-by-line zip shifts at the first insertion and then
# calls every later line changed -- that defect refused a correct namelist in
# the year round) and REFUSES if anything but the three cadence rows moved.
python3 - "$source_cfg/EXP00/namelist_cfg" "$RUN_DIR/namelist_cfg" \
         "$NN_ITEND" "$NN_STOCK" "$NN_WRITE" <<'PY'
import re, sys
src, dst, itend, stock, write = sys.argv[1:6]
text = open(src).read()
for key, value in (("nn_itend", itend), ("nn_stock", stock),
                   ("nn_write", write)):
    text, n = re.subn(r"^(\s*%s\s*=\s*)(\S+)" % key, r"\g<1>%s" % value,
                      text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("REFUSE: %s not found exactly once in %s" % (key, src))
open(dst, "w").write(text)

ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*(?:!.*)?$")

def rows(blob):
    out = {}
    for line in blob.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("&"):
            continue
        match = ASSIGN.match(line)
        if match:
            out[match.group(1)] = match.group(2)
    return out

before, after = rows(open(src).read()), rows(text)
changed = sorted(k for k in before if k in after and before[k] != after[k])
added = sorted(set(after) - set(before))
removed = sorted(set(before) - set(after))
print("  namelist rows changed vs the certified card:")
for key in changed:
    print("    %-14s %s -> %s" % (key, before[key], after[key]))
allowed = {"nn_itend", "nn_stock", "nn_write"}
if added or removed or set(changed) != allowed:
    raise SystemExit(
        "REFUSE: changed=%s added=%s removed=%s; only %s may move"
        % (changed, added, removed, sorted(allowed)))
PY

# --------------------------------------------------------------------- run
( cd "$RUN_DIR"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
    2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>run.user.time.log )

# ------------------------------------------- the instrument's own checks
# 1. THE ADMISSION CHECK.  Same binary, same physics namelist, different
#    restart cadence -- so the day-30 state must be byte-identical to the
#    scored year's.  If it is not, the cadence changed the arithmetic and this
#    record cannot be compared against the year's own numbers.
printf '\n=== day-30 byte identity against the scored year control ===\n'
if cmp -s "$RUN_DIR/GYRE_OMIP_L2_P3_00000180_restart.nc" "$control_restart"; then
  printf '  IDENTICAL GYRE_OMIP_L2_P3_00000180_restart.nc\n'
else
  printf 'REFUSE: the day-30 restart DIFFERS from the scored year control.\n' >&2
  printf '  The restart cadence changed the trajectory, so this record is NOT\n' >&2
  printf '  the same run as the one the year verdict was computed on.\n' >&2
  exit 71
fi

# 2. Every day must have a restart, or the day-by-day table would be computed
#    on a shorter list than it was asked for and say so nowhere.
python3 - "$RUN_DIR" "$NN_STOCK" "$NN_ITEND" <<'PY'
import glob, os, sys
run, stock, itend = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
missing = [s for s in range(stock, itend + 1, stock)
           if not glob.glob(os.path.join(run, "*_%08d_restart.nc" % s))]
if missing:
    raise SystemExit("REFUSE: missing restarts at kt %s" % missing)
print("every one of the 30 days has a restart")
PY

# 3. The sixty per-step entry dumps must be there, and the first one must match
#    the year run's, which is the same instrument writing the same state.
n_entry=$(ls "$RUN_DIR"/oracle_step_entry_kt*.bin 2>/dev/null | wc -l)
if [[ "$n_entry" -ne "$ENTRY_DUMPS" ]]; then
  printf 'REFUSE: %s per-step entry dumps, expected %s; the card writer did not run as compiled\n' "$n_entry" "$ENTRY_DUMPS" >&2
  exit 72
fi
for kt in 00000001 00000010 00000060; do
  cmp -s "$RUN_DIR/oracle_step_entry_kt$kt.bin" \
         "$YEAR_RUN/nemo_pristine/oracle_step_entry_kt$kt.bin" \
    || { printf 'REFUSE: entry dump kt=%s differs from the year run pristine member\n' "$kt" >&2; exit 72; }
done
printf 'all %s per-step entry dumps written; kt=1/10/60 byte-identical to the year run\n' "$ENTRY_DUMPS"

( cd "$RUN_DIR" && find . -type f \( -name '*_restart.nc' -o -name 'oracle_*.bin' \) \
    -print0 | sort -z | xargs -0 sha256sum > earlydays_artifacts.sha256 )

printf '\nNEMO early-days record written to %s\n' "$RUN_DIR"
printf 'score with:\n'
printf '  python %s --day-gap --days 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 \\\n' "$HARNESS"
printf '      --nemo-root %s --lego-root %s --lego-tag daily\n' "$TARGET_RUN" "$TARGET_RUN"
printf 'GYRE_EARLYDAYS_NEMO_READY %s\n' "$RUN_DIR"
