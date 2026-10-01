#!/usr/bin/env bash
# Operator-executed hierarchy recorder repair: rung-6 calibration, then rung 5.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: ORCA2 hierarchy recorder repair failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_HIER_R7ABSENT
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly RUNG6=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung6/record
readonly RUNG6_ADMISSION=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung6/rung6_admission.json
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung5
readonly FAILED_RUN=$EVIDENCE/record
readonly CALIBRATION=$EVIDENCE/record_rung6_absent_repair_calibration
readonly TARGET_RUN=$EVIDENCE/record_absent_v2
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly SOURCE_MODULE_SHA=879b84c92c82da16e7c9a233b2f084ed02b4f17677d1e685c85a81e2bb3a22af
readonly SOURCE_STPRK3_SHA=2f6dd49cfd222022e76cd4d5577d1d315c84d5be599aac912c8380d1d7fae9f9
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly RUNG6_ADMISSION_SHA=45d76acb89ff7377563ae54734c9644441aae07b8a3bed82fae4a3ed7b8e1d46
readonly RUNG5_DECK_SHA=f537e3d29a6e2472f89cc9a8d23ec70d18756e3ad112088e7256804698bc1d59
readonly RUNG5_EXEC_SHA=53b45d1c98af7554e254e25e5b07acbab56cb09ed22896cf090415b371f293b8

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/l4_r69_surface_round7_absent.patch
readonly MANIFEST=$here/rung5_absent_manifest.json
readonly GATE=$here/../nemo_testcase_l4_orca2_hier_decks_round7_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_hier_decks_round7.md

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "$0" "$PATCH" "$MANIFEST" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
pin "$SOURCE_MODULE_SHA" "$SOURCE_ROOT/MY_SRC/l4_r69_surface.F90" 'source recorder'
pin "$SOURCE_STPRK3_SHA" "$SOURCE_ROOT/MY_SRC/stprk3.F90" 'source step program'
pin "$SOURCE_CPP_SHA" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source CPP keys'
pin "$RUNG6_ADMISSION_SHA" "$RUNG6_ADMISSION" 'admitted rung-6 gate'
pin "$RUNG5_DECK_SHA" "$FAILED_RUN/namelist_cfg.deck" 'exact rung-5 deck'
pin "$RUNG5_EXEC_SHA" "$FAILED_RUN/namelist_cfg" 'rung-5 execution deck'
grep -q '"status": "PASS_RUNG6_RECORD"' "$RUNG6_ADMISSION" || {
  printf 'REFUSE: rung-6 source is not admitted\n' >&2; exit 65;
}
(cd "$RUNG6" && sha256sum -c input_files.sha256 >/dev/null)

mkdir -p "$EVIDENCE"
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence path is not a real directory\n' >&2; exit 65;
}
bash -n "$0"
"$PY" -m py_compile "$GATE"
"$PY" "$GATE" --preflight-only >"$EVIDENCE/round7_preflight.json"

dry=$(mktemp -d /tmp/orca2-hier-r7.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/l4_r69_surface.F90" "$dry/l4_r69_surface.F90"
patch -s --fuzz=0 "$dry/l4_r69_surface.F90" <"$PATCH"
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: repair removes source lines\n' >&2; exit 66; }
cpp -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$dry/l4_r69_surface.F90" \
  -o "$dry/l4_r69_surface.f90"
"$FC" -fsyntax-only -ffree-line-length-none -J "$dry" \
  -I "$SOURCE_ROOT/BLD/inc" "$dry/l4_r69_surface.f90"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_HIERARCHY_RUNG5_ABSENT_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

check_calibration() {
  local source actual
  [[ "$(find "$CALIBRATION" -maxdepth 1 -type f -name 'oracle_r69_surface_*.bin' | wc -l)" -eq 480 ]] || {
    printf 'REFUSE: calibration does not contain 480 frames\n' >&2; exit 71;
  }
  for source in "$RUNG6"/oracle_r69_surface_*.bin "$RUNG6"/ORCA2_00000240_restart_????.nc; do
    actual=$CALIBRATION/$(basename "$source")
    cmp -s "$source" "$actual" || {
      printf 'REFUSE: repaired active-runoff build changed %s\n' "$(basename "$source")" >&2; exit 71;
    }
  done
  printf 'RUNG6_REPAIR_CALIBRATION_BIT_IDENTICAL\n'
}

admit() {
  local plant
  check_calibration
  for plant in field-name truncated frame-nonfinite absent-as-zero owner-on missing-frame \
    terminal-nonfinite terminal-step ice-sentinel-read tke-sentinel-read \
    resolved-consequence runoff-group-unread active-runoff-print sha-inventory \
    calibration-frame calibration-restart; do
    if "$PY" "$GATE" --record "$TARGET_RUN" --calibration "$CALIBRATION" \
      --expect-commit "$COMMIT" --plant "$plant" >"$EVIDENCE/round7_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/round7_${plant}_plant.log"
  done
  "$PY" "$GATE" --record "$TARGET_RUN" --calibration "$CALIBRATION" \
    --expect-commit "$COMMIT" --output "$EVIDENCE/rung5_absent_admission.json"
  (cd "$EVIDENCE" && sha256sum round7_preflight.json rung5_absent_admission.json \
    round7_*_plant.log >round7_SHA256SUMS)
  printf 'ORCA2_HIERARCHY_RUNG5_ABSENT_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -d "$CALIBRATION" && -d "$TARGET_RUN" ]] || {
    printf 'REFUSE: repaired calibration or rung-5 record is absent\n' >&2; exit 68;
  }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$CALIBRATION" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: repaired build or run target already exists\n' >&2; exit 68;
}
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67;
  }
done

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/l4_r69_surface.F90" <"$PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: repaired build produced no binary\n' >&2; exit 69; }
[[ "$(grep -Fc 'CALL put_absent' "$TARGET_ROOT/BLD/ppsrc/nemo/l4_r69_surface.f90")" -eq 2 ]] || {
  printf 'REFUSE: compiled repair lacks two ABSENT calls\n' >&2; exit 69;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 69
fi

stage_run() {
  local target=$1 config=$2 deck=$3 hierarchy_manifest=$4
  mkdir "$target"
  while read -r digest name; do
    case "$name" in
      namelist_cfg) cp -a "$config" "$target/namelist_cfg" ;;
      *) cp -a "$RUNG6/$name" "$target/$name" ;;
    esac
  done <"$RUNG6/deck_files.sha256"
  while read -r digest name; do cp -a "$RUNG6/$name" "$target/$name"; done <"$RUNG6/input_files.sha256"
  cp -a "$RUNG6/deck_files.sha256" "$RUNG6/input_files.sha256" "$target/"
  cp -a "$deck" "$target/namelist_cfg.deck"
  cp -a "$hierarchy_manifest" "$target/hierarchy_manifest.json"
  cp -a "$MANIFEST" "$target/recorder_repair_manifest.json"
  cp -a "$BINARY" "$target/nemo"
  cp -a "$TARGET_ROOT/BLD/ppsrc/nemo/l4_r69_surface.f90" "$target/compiled_l4_r69_surface.f90"
  cp -a "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90" "$target/compiled_stprk3.f90"
  printf '%s\n' "$COMMIT" >"$target/producer_commit.txt"
  (cd "$target" && sha256sum -c input_files.sha256 >/dev/null)
}

run_one() {
  local target=$1
  (
    cd "$target"
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
    grep -q 'STOP 0' run.user.stdout.log || { printf 'REFUSE: NEMO did not report STOP 0\n' >&2; exit 70; }
    [[ "$(tr -d '[:space:]' <time.step)" == 240 ]] || { printf 'REFUSE: NEMO did not reach step 240\n' >&2; exit 70; }
    find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\0' | sort -z | xargs -0 sha256sum >SHA256SUMS
  )
}

stage_run "$CALIBRATION" "$RUNG6/namelist_cfg" "$RUNG6/namelist_cfg.deck" "$RUNG6/hierarchy_manifest.json"
run_one "$CALIBRATION"
check_calibration
stage_run "$TARGET_RUN" "$FAILED_RUN/namelist_cfg" "$FAILED_RUN/namelist_cfg.deck" "$FAILED_RUN/hierarchy_manifest.json"
run_one "$TARGET_RUN"
admit
