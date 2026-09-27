#!/usr/bin/env bash
set -Eeuo pipefail

# OPERATOR-EXECUTED ACQUISITION ONLY.  The agent wrote this file and stopped;
# mpirun's PMIx launcher is refused in the agent sandbox (operator's note AM).
#
#   Usage:  bash run.sh --run
#
# WHAT THIS ACQUIRES.  NEMO's side of the GYRE DECADE: one ten-year run from
# rest on the certified card, with a restart every 30 model days.  The design,
# the scored rows and the bar are in
#   docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_decade_climate.md
# and nothing here re-derives any of them.
#
# NO BUILD.  This is the one real difference from the round-132 pattern this
# script is copied from, and it is deliberate.  The decade differs from the
# certified year ONLY in three namelist integers, so there is nothing to
# compile; running the ALREADY-CERTIFIED binary means the year-1 admission
# check below can actually pass.  A fresh compilation of the same statements
# can differ in the last bit (the compiled-rounding floor this campaign has
# measured many times, notes L / AL / AS), and that is exactly what refused
# three acquisitions in a row.  The binary used here is the one the
# year-from-rest PRISTINE control ran, and its day-30 restart is byte-identical
# to round 132's.
#
# THE PATH MATTERS.  FCM is not called, but mpirun and the netCDF runtime come
# from the conda build environment, the same PATH every previous acquisition
# used.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R41ADVSP
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly TARGET_RUN=$L2/decade/nemo
readonly REFERENCE_RUN=$L2/round132/oracle_daily_restarts
# The decade, and the restart cadence.  Both are fixed by the preregistration
# and both follow from the card's own rn_Dt = 14400 s: 6 steps a day, 180 steps
# a month, 2160 steps a year, 21600 steps a decade.
readonly STEPS_PER_DAY=6
readonly NN_ITEND=21600
readonly NN_STOCK=180
readonly NN_WRITE=21600
readonly YEAR_STEPS=2160
# The certified binary, named by content so a swapped file cannot go unnoticed.
readonly BINARY_SHA=a759e8b478e3bda5ba731009fd353159db892c5ac2ba4c48351f5136411960cd

refuse_on_error() {
  local status=$?
  printf 'REFUSE: gyre decade acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

if [[ "${1:-}" != "--run" ]]; then
  printf 'This script runs NEMO for ten model years (~7 minutes) and writes to\n'
  printf '  %s\n' "$TARGET_RUN"
  printf 'Re-run with --run to proceed.\n'
  exit 0
fi

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SCORER=$here/../nemo_testcase_l2_gyre_decade_climate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_decade_climate.md
readonly source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly binary=$source_cfg/BLD/bin/nemo.exe

# ------------------------------------------------------------- preconditions
for path in "$binary" "$source_cfg/EXP00/namelist_cfg" "$SCORER" "$PREREG"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$REFERENCE_RUN" ]] || { printf 'REFUSE: no round-132 reference at %s\n' "$REFERENCE_RUN" >&2; exit 64; }
if [[ -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: %s already exists; delete it by hand if it is stale.\n' "$TARGET_RUN" >&2
  exit 64
fi
# The binary is the certified one, by content and not by path.
actual_sha=$(sha256sum "$binary" | awk '{print $1}')
if [[ "$actual_sha" != "$BINARY_SHA" ]]; then
  printf 'REFUSE: %s has sha256 %s, expected the certified %s\n' \
    "$binary" "$actual_sha" "$BINARY_SHA" >&2
  exit 65
fi
if nm -D "$binary" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in %s; not the scalar-math arch\n' "$binary" >&2
  exit 65
fi
# Space.  120 restarts x 1.4 MB, the copied 52 MB binary, and ~2 GB of the
# card's own per-step oracle_*.bin instrument streams (201 MB for 2160 steps,
# measured on the year run, and some of its writers fire every step).
free_kb=$(df -Pk "$(dirname "$TARGET_RUN")" | awk 'NR==2 {print $4}')
if [[ "$free_kb" -lt 8388608 ]]; then
  printf 'REFUSE: %s has %s kB free, under the 8 GB floor\n' "$TARGET_RUN" "$free_kb" >&2
  exit 68
fi

mkdir -p "$TARGET_RUN"
cd "$TARGET_RUN"
for name in namelist_ref namelist_top_cfg namelist_top_ref \
            namelist_pisces_cfg namelist_pisces_ref context_nemo.xml \
            file_def_nemo.xml iodef.xml axis_def_nemo.xml \
            domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml \
            field_def_nemo-pisces.xml; do
  cp -L "$source_cfg/EXP00/$name" "$TARGET_RUN/$name"
done
cp "$binary" "$TARGET_RUN/nemo"
sha256sum "$TARGET_RUN/nemo" > "$TARGET_RUN/binary.sha256"

# ------------------------------------------------------------------ namelist
# Copied from nemo_testcase_l2_gyre_year_fromrest_members/run.sh, which is the
# writer that has already been dry-run against this exact certified namelist.
# It compares PARSED ASSIGNMENTS, not lines, because a line-by-line zip shifts
# at the first insertion and then calls every later line changed.
python3 - "$source_cfg/EXP00/namelist_cfg" "$TARGET_RUN/namelist_cfg" \
         "$NN_ITEND" "$NN_STOCK" "$NN_WRITE" <<'PY'
import re, sys
src, dst, itend, stock, write = sys.argv[1:6]
text = open(src).read()
for key, value in (("nn_itend", itend), ("nn_stock", stock), ("nn_write", write)):
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
if removed or added or set(changed) != allowed:
    raise SystemExit(
        "REFUSE: changed=%s added=%s removed=%s; the preregistration allows "
        "exactly %s changed and nothing added or removed"
        % (changed, added, removed, sorted(allowed)))
# rn_Dt is the reason every step count in this script is what it is, so it is
# READ here rather than assumed.
match = re.search(r"^\s*rn_Dt\s*=\s*([0-9.eEdD+-]+)", text, re.M)
if match is None:
    raise SystemExit("REFUSE: no rn_Dt in the namelist")
dt = float(match.group(1).replace("d", "e").replace("D", "e"))
if abs(dt - 14400.0) > 0.0:
    raise SystemExit("REFUSE: rn_Dt = %s, not the certified 14400 s" % dt)
steps_per_day = 86400.0 / dt
if steps_per_day != 6.0:
    raise SystemExit("REFUSE: %s steps per day, not 6" % steps_per_day)
print("  rn_Dt = %g s -> %g steps/day; %s steps = %g days = %g years"
      % (dt, steps_per_day, itend, int(itend) / steps_per_day,
         int(itend) / steps_per_day / 360.0))
PY

# ----------------------------------------------------------------------- run
# No /usr/bin/time: GNU time is not installed on this box (operator's note AH),
# and calling it exited an earlier acquisition at 127.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > run.user.time.log
started=$SECONDS
set +e
mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
rc=${PIPESTATUS[0]}
set -e
printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nNEMO_EXIT=%s\n' \
  "$((SECONDS - started))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >> run.user.time.log
if [[ "$rc" -ne 0 ]]; then
  printf 'REFUSE: NEMO exited %s; see %s/run.user.stdout.log\n' "$rc" "$TARGET_RUN" >&2
  exit 70
fi

# ------------------------------------------------- the instrument's checks
# 1.  ADMISSION (operator's note AS).  A developed-state record is admitted
#     when its restarts are byte-identical to the UN-INSTRUMENTED round-132
#     reference at the shared steps.  Year 1 is where the two overlap: round
#     132 dumped every day, this run dumps every month, so all twelve of this
#     run's year-1 restarts have a round-132 twin.  A single difference means
#     this is not the certified trajectory and nothing downstream is admissible.
printf '\n=== admission: year-1 restarts against the round-132 reference ===\n'
mismatch=0
for (( step=NN_STOCK; step<=YEAR_STEPS; step+=NN_STOCK )); do
  name=$(printf 'GYRE_OMIP_L2_P3_%08d_restart.nc' "$step")
  if [[ ! -f "$TARGET_RUN/$name" ]]; then
    printf 'REFUSE: this run did not write %s\n' "$name" >&2; exit 71
  fi
  if [[ ! -f "$REFERENCE_RUN/$name" ]]; then
    printf 'REFUSE: the round-132 reference has no %s\n' "$name" >&2; exit 71
  fi
  if cmp -s "$REFERENCE_RUN/$name" "$TARGET_RUN/$name"; then
    printf '  IDENTICAL %s\n' "$name"
  else
    printf '  DIFFERS   %s\n' "$name"; mismatch=1
  fi
done
if [[ "$mismatch" -ne 0 ]]; then
  printf 'REFUSE: the decade run is not byte-identical to the certified year;\n' >&2
  printf '  it is a different trajectory and the decade record is not admissible.\n' >&2
  exit 71
fi
printf 'ADMISSION OK: all twelve year-1 months match round 132 byte for byte\n'

# 2.  Every scored month must have a restart, or the scorer would silently
#     compute a climatology over fewer months than the preregistration names.
printf '\n=== completeness: 120 monthly restarts ===\n'
missing=0
for (( step=NN_STOCK; step<=NN_ITEND; step+=NN_STOCK )); do
  name=$(printf 'GYRE_OMIP_L2_P3_%08d_restart.nc' "$step")
  [[ -f "$TARGET_RUN/$name" ]] || { printf '  MISSING %s\n' "$name"; missing=$((missing + 1)); }
done
if [[ "$missing" -ne 0 ]]; then
  printf 'REFUSE: %s of the 120 monthly restarts are missing\n' "$missing" >&2
  exit 72
fi
printf 'all 120 monthly restarts present\n'

( cd "$TARGET_RUN" && find . -maxdepth 1 -name '*_restart.nc' -print0 \
    | sort -z | xargs -0 sha256sum > nemo_decade_restarts.sha256 )

printf '\nNEMO decade written to %s\n' "$TARGET_RUN"
printf 'score with:\n'
printf '  python %s --score --months 120\n' "$SCORER"
printf '  python %s --figures --months 120\n' "$SCORER"
printf 'GYRE_DECADE_NEMO_READY %s\n' "$TARGET_RUN"
