#!/usr/bin/env bash
# ORCA2 round-67 acquisition: pinned ORCA1ICE from-rest 30-day comparator.
# The operator runs this file; the agent never invokes mpirun in the sandbox.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-67 month acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--resume-month|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--resume-month|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly RECORDS=/data/abyssal/dbalwada/nemo-testcases-l4/runs
readonly PINNED=$RECORDS/variant_orca1ice_phase2x_a_10step_np2
readonly BINARY_SOURCE=$RECORDS/uninstrumented_30day_np2/nemo
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round67/acquisition
readonly CALIBRATION=$EVIDENCE/orca1ice_uninstrumented_calibration_10step_np2
readonly MONTH=$EVIDENCE/orca1ice_uninstrumented_fromrest_30day_np2
readonly BINARY_SHA=c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343
readonly DECK_SHA=51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9
readonly INPUT_SHA=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly ICE_CFG_SHA=6b647863137b518b95ff97f83975d9afcb3944b7f6e8d63e45a05f494f5edc89
readonly CALIBRATION_COMMIT=61314622ebff810e988159566e3ec1990cc6a00c
readonly ACQUIRED_MONTH_COMMIT=61314622ebff810e988159566e3ec1990cc6a00c

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly GATE=$here/../nemo_testcase_l4_orca2_round67_month_record_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round67_month_scale.md

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 65; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2
    exit 66
  }
}

for path in "$GATE" "$PREREG" "$BINARY_SOURCE" "$PINNED/deck_files.sha256" \
  "$PINNED/input_files.sha256" "$PINNED/namelist_cfg" "$PINNED/namelist_ice_cfg" \
  "$PINNED/ORCA2_00000010_restart_0000.nc" \
  "$PINNED/ORCA2_00000010_restart_0001.nc" \
  "$PINNED/ORCA2_00000010_restart_ice_0000.nc" \
  "$PINNED/ORCA2_00000010_restart_ice_0001.nc"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing acquisition input %s\n' "$path" >&2; exit 65; }
done
pin "$BINARY_SHA" "$BINARY_SOURCE" 'scalar-math binary'
pin "$DECK_SHA" "$PINNED/deck_files.sha256" 'pinned deck manifest'
pin "$INPUT_SHA" "$PINNED/input_files.sha256" 'pinned input manifest'
pin "$ICE_CFG_SHA" "$PINNED/namelist_ice_cfg" 'pinned ORCA1ICE namelist'
( cd "$PINNED" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null )
grep -Eq 'number of the last time step.*nn_itend *= *10' "$PINNED/ocean.output" || {
  printf 'REFUSE: pinned source is not ten steps\n' >&2; exit 65;
}
grep -Eq 'number of ice  categories.*jpl *= *1' "$PINNED/ocean.output" || {
  printf 'REFUSE: pinned source is not the one-category deck\n' >&2; exit 65;
}

# Refuse a duplicate if a complete pinned 240-step comparator already exists.
while IFS= read -r -d '' manifest; do
  root=$(dirname "$manifest")
  [[ "$root" == "$MONTH" ]] && continue
  if [[ "$(sha256sum "$manifest" | awk '{print $1}')" == "$DECK_SHA" ]] && \
     [[ -f "$root/ocean.output" ]] && \
     grep -Eq 'number of the last time step.*nn_itend *= *240' "$root/ocean.output"; then
    printf 'REFUSE: an existing pinned 240-step comparator must be admitted: %s\n' "$root" >&2
    exit 67
  fi
done < <(find "$RECORDS" -mindepth 2 -maxdepth 2 -type f -name deck_files.sha256 -print0)

mkdir -p "$EVIDENCE"
for mount in /tmp "$EVIDENCE"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  }
done

# Validate the only authorized deck edit before any target is created.
"$PY" - "$PINNED/namelist_cfg" <<'PYPREFLIGHT'
import re
import sys

source = open(sys.argv[1], encoding="utf-8").read()
changed = source
for key in ("nn_itend", "nn_stock"):
    changed, count = re.subn(
        rf"^(\s*{key}\s*=\s*)(\S+)", rf"\g<1>240", changed,
        count=1, flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(f"REFUSE: {key} not found exactly once")

assignment = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*(?:!.*)?$")
def rows(text):
    return {match.group(1): match.group(2) for line in text.splitlines()
            if (match := assignment.match(line))}
before, after = rows(source), rows(changed)
delta = sorted(key for key in before if before[key] != after[key])
if delta != ["nn_itend", "nn_stock"] or set(before) != set(after):
    raise SystemExit(f"REFUSE: unauthorized namelist delta {delta}")
print("NAMELIST_PREFLIGHT_PASS nn_itend=240 nn_stock=240")
PYPREFLIGHT

"$PY" -m py_compile "$GATE"
bash -n "$0"
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND67_MONTH_PREFLIGHT_READY %s\n' "$MONTH"
  exit 0
fi

admit() {
  local calibration_commit=$1 month_commit=$2
  local plant
  for plant in calibration-ulp missing-shard hidden-deck-delta; do
    if "$PY" "$GATE" --pinned "$PINNED" --calibration "$CALIBRATION" \
      --month "$MONTH" --expect-calibration-commit "$calibration_commit" \
      --expect-month-commit "$month_commit" --plant "$plant" \
      >"$EVIDENCE/${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2
      exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/${plant}_plant.log"
  done
  "$PY" "$GATE" --pinned "$PINNED" --calibration "$CALIBRATION" \
    --month "$MONTH" --expect-calibration-commit "$calibration_commit" \
    --expect-month-commit "$month_commit" \
    --output "$EVIDENCE/month_record_admission.json"
  ( cd "$EVIDENCE" && sha256sum month_record_admission.json *_plant.log \
      "$CALIBRATION"/ORCA2_00000010_restart*.nc \
      "$MONTH"/ORCA2_00000240_restart*.nc >round67_outputs.sha256 )
  printf 'ORCA2_ROUND67_MONTH_ACQUISITION_PASS %s\n' "$MONTH"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -d "$CALIBRATION" && -d "$MONTH" ]] || {
    printf 'REFUSE: existing calibration or month target is absent\n' >&2; exit 68;
  }
  admit "$CALIBRATION_COMMIT" "$ACQUIRED_MONTH_COMMIT"
  exit 0
fi

stage() {
  local target=$1 step=$2
  mkdir "$target"
  while read -r digest name; do cp -a "$PINNED/$name" "$target/$name"; done \
    <"$PINNED/deck_files.sha256"
  while read -r digest name; do cp -a "$PINNED/$name" "$target/$name"; done \
    <"$PINNED/input_files.sha256"
  cp "$PINNED/input_files.sha256" "$target/input_files.sha256"
  cp "$BINARY_SOURCE" "$target/nemo"
  printf '%s\n' "$COMMIT" >"$target/producer_commit.txt"
  if [[ "$step" -eq 10 ]]; then
    cp "$PINNED/deck_files.sha256" "$target/deck_files.sha256"
  else
    "$PY" - "$target/namelist_cfg" <<'PYEDIT'
import re
import sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
for key in ("nn_itend", "nn_stock"):
    text, count = re.subn(
        rf"^(\s*{key}\s*=\s*)(\S+)", rf"\g<1>240", text,
        count=1, flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(f"REFUSE: {key} not found exactly once")
open(path, "w", encoding="utf-8").write(text)
PYEDIT
    while read -r digest name; do
      ( cd "$target" && sha256sum "$name" )
    done <"$PINNED/deck_files.sha256" >"$target/deck_files.sha256"
  fi
  ( cd "$target" && sha256sum -c deck_files.sha256 >/dev/null && \
      sha256sum -c input_files.sha256 >/dev/null )
}

run_target() {
  local target=$1
  (
    cd "$target"
    for output in ocean.output time.step run.user.stdout.log run.user.time.log; do
      [[ ! -e "$output" ]] || { printf 'REFUSE: output exists: %s/%s\n' "$target" "$output" >&2; exit 69; }
    done
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    started=$SECONDS
    set +e
    mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    pipe_rc=("${PIPESTATUS[@]}")
    set -e
    [[ "${pipe_rc[0]}" -eq 0 ]] || { printf 'REFUSE: mpirun exited %s\n' "${pipe_rc[0]}" >&2; exit 70; }
    [[ "${pipe_rc[1]:-0}" -eq 0 ]] || { printf 'REFUSE: tee exited %s\n' "${pipe_rc[1]}" >&2; exit 70; }
    printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  )
  grep -q 'STOP 0' "$target/run.user.stdout.log" || {
    printf 'REFUSE: NEMO did not report STOP 0: %s\n' "$target" >&2; exit 70;
  }
}

if [[ "$MODE" == --resume-month ]]; then
  [[ -d "$CALIBRATION" ]] || {
    printf 'REFUSE: completed calibration target is absent\n' >&2; exit 68;
  }
  [[ ! -e "$MONTH" ]] || {
    printf 'REFUSE: month target already exists; use --admit-existing\n' >&2; exit 68;
  }
  "$PY" "$GATE" --pinned "$PINNED" --calibration "$CALIBRATION" \
    --expect-calibration-commit "$CALIBRATION_COMMIT" --mode calibration \
    --output "$EVIDENCE/calibration_admission.json"
  stage "$MONTH" 240
  run_target "$MONTH"
  admit "$CALIBRATION_COMMIT" "$COMMIT"
  exit 0
fi

[[ ! -e "$CALIBRATION" && ! -e "$MONTH" ]] || {
  printf 'REFUSE: calibration or month target already exists\n' >&2; exit 68;
}

stage "$CALIBRATION" 10
run_target "$CALIBRATION"
"$PY" "$GATE" --pinned "$PINNED" --calibration "$CALIBRATION" \
  --expect-calibration-commit "$COMMIT" --mode calibration \
  --output "$EVIDENCE/calibration_admission.json"

stage "$MONTH" 240
run_target "$MONTH"
admit "$COMMIT" "$COMMIT"
