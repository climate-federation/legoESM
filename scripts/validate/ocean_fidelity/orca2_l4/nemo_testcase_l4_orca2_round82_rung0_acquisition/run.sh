#!/usr/bin/env bash
# ORCA2 round-82 acquisition: Decision-79 hierarchy rung 0.
# The operator runs this file; agents never invoke mpirun in the sandbox.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-%s rung-0 acquisition failed at line %s (exit %s)\n' \
    "${ORCA2_RUNG0_ROUND:-82}" "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly CPP=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm
readonly BASE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round69/acquisition/orca1ice_surface_only_240step_np2
readonly BINARY_SOURCE=/data/abyssal/dbalwada/nemo-testcases-l4/runs/uninstrumented_30day_np2/nemo
readonly ROUND=${ORCA2_RUNG0_ROUND:-82}
readonly EVIDENCE=${ORCA2_RUNG0_EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round82/acquisition}
readonly TWIN_A=$EVIDENCE/${ORCA2_RUNG0_TWIN_A:-orca2_rung0_10step_a_np2}
readonly TWIN_B=$EVIDENCE/${ORCA2_RUNG0_TWIN_B:-orca2_rung0_10step_b_np2}
readonly MONTH=$EVIDENCE/${ORCA2_RUNG0_MONTH:-orca2_rung0_240step_np2}
readonly CANONICAL=$EVIDENCE/rung0_namelist_cfg
readonly SOURCE_NAMELIST_SHA=036f3cec148b2e89cede910d13189db4cc8e2e74a9b9ecdede7a87a5c14a98ec
readonly CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly BINARY_SHA=c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343
readonly DECK_SHA=09a350860ff6eaef17d1f0e18aa8e16c4d929e994d9d6804d0976f531b06f66e
readonly INPUT_SHA=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly DECK_GATE=$here/../nemo_testcase_l4_orca2_round82_rung0_deck_gate.py
readonly RECORD_GATE=$here/../nemo_testcase_l4_orca2_round82_rung0_record_gate.py
readonly PREREG=${ORCA2_RUNG0_PREREG:-$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round82.md}

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
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 66;
  }
}

for path in "$DECK_GATE" "$RECORD_GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 65;
  }
done
pin "$SOURCE_NAMELIST_SHA" "$BASE/namelist_cfg" 'admitted shipped namelist'
pin "$CPP_SHA" "$CPP" 'ORCA2 CPP card'
pin "$BINARY_SHA" "$BINARY_SOURCE" 'scalar-math binary'
pin "$DECK_SHA" "$BASE/deck_files.sha256" 'shipped deck manifest'
pin "$INPUT_SHA" "$BASE/input_files.sha256" 'ORCA2 input manifest'
(cd "$BASE" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

mkdir -p "$EVIDENCE"
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence path is not a real directory: %s\n' "$EVIDENCE" >&2; exit 65;
}
bash -n "$0"
"$PY" -m py_compile "$DECK_GATE" "$RECORD_GATE"
"$PY" "$DECK_GATE" --source "$BASE/namelist_cfg" --render "$CANONICAL"
"$PY" "$DECK_GATE" --source "$BASE/namelist_cfg" --candidate "$CANONICAL" \
  --cpp "$CPP" --output "$EVIDENCE/rung0_deck_preflight.json"
"$PY" "$RECORD_GATE" --preflight-only --output "$EVIDENCE/rung0_record_preflight.json"
"$PY" "$RECORD_GATE" --render-run-deck --run-deck-source "$CANONICAL" \
  --run-deck-output "$EVIDENCE/rung0_10step_namelist_preflight" \
  --steps 10 --stock 1 --restart-list
"$PY" "$RECORD_GATE" --render-run-deck --run-deck-source "$CANONICAL" \
  --run-deck-output "$EVIDENCE/rung0_month_namelist_preflight" \
  --steps 240 --stock 240
for plant in extra-delta mixing-value live-module cpp; do
  if "$PY" "$DECK_GATE" --source "$BASE/namelist_cfg" --candidate "$CANONICAL" \
    --cpp "$CPP" --plant "$plant" >"$EVIDENCE/deck_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: deck %s plant stayed green\n' "$plant" >&2
    exit 72
  fi
  grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/deck_${plant}_plant.log"
done

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND%s_RUNG0_PREFLIGHT_READY %s\n' "$ROUND" "$TWIN_A"
  exit 0
fi

admit() {
  local plant
  for plant in restart-ulp missing-rank wrong-step nonfinite hidden-month-delta; do
    if "$PY" "$RECORD_GATE" --source "$BASE/namelist_cfg" --cpp "$CPP" \
      --twin-a "$TWIN_A" --twin-b "$TWIN_B" --month "$MONTH" \
      --expect-commit "$COMMIT" --plant "$plant" \
      >"$EVIDENCE/record_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: record %s plant stayed green\n' "$plant" >&2
      exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/record_${plant}_plant.log"
  done
  "$PY" "$RECORD_GATE" --source "$BASE/namelist_cfg" --cpp "$CPP" \
    --twin-a "$TWIN_A" --twin-b "$TWIN_B" --month "$MONTH" \
    --expect-commit "$COMMIT" --output "$EVIDENCE/rung0_record_admission.json"
  (cd "$EVIDENCE" && sha256sum rung0_*json deck_*_plant.log record_*_plant.log \
    "$TWIN_A"/ORCA2_00000010_restart_*.nc \
    "$TWIN_B"/ORCA2_00000010_restart_*.nc \
    "$MONTH"/ORCA2_00000240_restart_*.nc >"round${ROUND}_outputs.sha256")
  printf 'ORCA2_ROUND%s_RUNG0_ACQUISITION_PASS %s\n' "$ROUND" "$MONTH"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -d "$TWIN_A" && -d "$TWIN_B" && -d "$MONTH" ]] || {
    printf 'REFUSE: one or more rung-0 targets are absent\n' >&2; exit 68;
  }
  admit
  exit 0
fi

[[ ! -e "$TWIN_A" && ! -e "$TWIN_B" && ! -e "$MONTH" ]] || {
  printf 'REFUSE: rung-0 target already exists; use --admit-existing\n' >&2; exit 68;
}
for mount in "$EVIDENCE" /tmp; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 6291456 ]] || {
    printf 'REFUSE: %s has under 6 GiB free\n' "$mount" >&2; exit 67;
  }
done

stage() {
  local target=$1 steps=$2 stock=$3 restart_list=$4
  mkdir "$target"
  while read -r digest name; do cp -a "$BASE/$name" "$target/$name"; done <"$BASE/deck_files.sha256"
  while read -r digest name; do cp -a "$BASE/$name" "$target/$name"; done <"$BASE/input_files.sha256"
  cp "$CANONICAL" "$target/namelist_cfg"
  cp "$BINARY_SOURCE" "$target/nemo"
  printf '%s\n' "$COMMIT" >"$target/producer_commit.txt"
  run_deck_args=(
    "$PY" "$RECORD_GATE" --render-run-deck
    --run-deck-source "$CANONICAL" --run-deck-output "$target/namelist_cfg"
    --steps "$steps" --stock "$stock"
  )
  if [[ "$restart_list" == true ]]; then run_deck_args+=(--restart-list); fi
  "${run_deck_args[@]}"
  "$PY" - "$target/chlorophyll.nc" "$target/rung0_zero_flux.nc" <<'PYZERO'
import sys
import numpy as np
from netCDF4 import Dataset
source, target = sys.argv[1:]
with Dataset(source) as src, Dataset(target, "w", format="NETCDF4") as dst:
    dst.createDimension("time_counter", 1)
    dst.createDimension("y", 148)
    dst.createDimension("x", 180)
    for name in ("nav_lat", "nav_lon"):
        variable = dst.createVariable(name, "f8", ("y", "x"))
        variable[:] = np.asarray(src.variables[name][:], dtype=np.float64)
    for name in ("utau", "vtau", "qtot", "qsr", "emp"):
        variable = dst.createVariable(name, "f8", ("time_counter", "y", "x"))
        variable[:] = 0.0
    dst.setncattr("title", "ORCA2 hierarchy rung-0 exact-zero surface fluxes")
PYZERO
  while read -r digest name; do (cd "$target" && sha256sum "$name"); done \
    <"$BASE/deck_files.sha256" >"$target/deck_files.sha256"
  while read -r digest name; do (cd "$target" && sha256sum "$name"); done \
    <"$BASE/input_files.sha256" >"$target/input_files.sha256"
  (cd "$target" && sha256sum rung0_zero_flux.nc >>input_files.sha256)
  (cd "$target" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
  sha256sum "$target/nemo" >"$target/binary.sha256"
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

stage "$TWIN_A" 10 1 true
run_target "$TWIN_A"
stage "$TWIN_B" 10 1 true
run_target "$TWIN_B"
stage "$MONTH" 240 240 false
run_target "$MONTH"
admit
