#!/usr/bin/env bash
# ORCA2 hierarchy decks round 2: rung-9 no-sea-ice record, no NEMO rebuild.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: ORCA2 hierarchy rung-9 acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
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
readonly SOURCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung10/record
readonly SOURCE_ADMISSION=$SOURCE/../rung10_admission.json
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung9
readonly DECK=$EVIDENCE/deck
readonly RUN=$EVIDENCE/record
readonly BUILD=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE
readonly BINARY_SHA=450410b5c9b1960c4cc8d05d690decca3308f888e70987b444682315bb846572
readonly DECK_SHA=09a350860ff6eaef17d1f0e18aa8e16c4d929e994d9d6804d0976f531b06f66e
readonly INPUT_SHA=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly STPRK3_SHA=b9da36788bb5a3d8ccff03cae7c1123d1b3510c39a0cd86263fcc293da526422
readonly WRITER_SHA=b116511cbefb6d2b6f1d911a77079a4268f343ea3aa1b861d08ac0c8d2bc4bcf

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly GATE=$here/../nemo_testcase_l4_orca2_hier_decks_round2_gate.py
readonly MANIFEST=$here/rung9_manifest.json
readonly SENTINEL=$here/rung9_unread_namelist_ice_cfg
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_hier_decks_round2.md

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

for path in "$0" "$GATE" "$MANIFEST" "$SENTINEL" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2
    exit 65
  }
done
[[ -f "$SOURCE_ADMISSION" ]] || { printf 'REFUSE: admitted rung-10 evidence absent\n' >&2; exit 65; }
grep -q '"status": "PASS_RUNG10_RECORD"' "$SOURCE_ADMISSION" || {
  printf 'REFUSE: rung-10 evidence is not admitted\n' >&2; exit 66;
}
pin "$BINARY_SHA" "$SOURCE/nemo" 'instrumented binary'
pin "$DECK_SHA" "$SOURCE/deck_files.sha256" 'upper-rung deck manifest'
pin "$INPUT_SHA" "$SOURCE/input_files.sha256" 'upper-rung input manifest'
pin "$CPP_SHA" "$BUILD/cpp_ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE.fcm" 'CPP-key file'
pin "$STPRK3_SHA" "$BUILD/BLD/ppsrc/nemo/stprk3.f90" 'compiled stprk3'
pin "$WRITER_SHA" "$BUILD/BLD/ppsrc/nemo/l4_r69_surface.f90" 'compiled writer'
(cd "$SOURCE" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

mkdir -p "$EVIDENCE"
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence path is not a real directory: %s\n' "$EVIDENCE" >&2
  exit 65
}
bash -n "$0"
"$PY" -m py_compile "$GATE"
"$PY" "$GATE" --preflight-only --stage-deck "$DECK" --output "$EVIDENCE/preflight.json" >/dev/null
for plant in deck-extra build-pin ice-artifact; do
  if "$PY" "$GATE" --preflight-only --plant "$plant" >"$EVIDENCE/${plant}_preflight_plant.log" 2>&1; then
    printf 'REFUSE: %s preflight plant stayed green\n' "$plant" >&2
    exit 72
  fi
  grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/${plant}_preflight_plant.log"
done

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_HIERARCHY_RUNG9_PREFLIGHT_READY %s\n' "$RUN"
  exit 0
fi

admit() {
  local expected_commit plant
  expected_commit=$(<"$RUN/producer_commit.txt")
  for plant in field-name truncated missing-frame frame-nonfinite terminal-nonfinite terminal-step ice-sentinel-read resolved-consequence sha-inventory; do
    if "$PY" "$GATE" --record "$RUN" --expect-commit "$expected_commit" --plant "$plant" \
      >"$EVIDENCE/${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s admission plant stayed green\n' "$plant" >&2
      exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/${plant}_plant.log"
  done
  "$PY" "$GATE" --record "$RUN" --expect-commit "$expected_commit" \
    --output "$EVIDENCE/rung9_admission.json"
  (cd "$EVIDENCE" && sha256sum deck/SHA256SUMS preflight.json rung9_admission.json *_plant.log >SHA256SUMS)
  printf 'ORCA2_HIERARCHY_RUNG9_ACQUISITION_PASS %s\n' "$RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -d "$RUN" ]] || { printf 'REFUSE: existing rung-9 record is absent: %s\n' "$RUN" >&2; exit 68; }
  admit
  exit 0
fi

[[ ! -e "$RUN" ]] || { printf 'REFUSE: rung-9 record already exists: %s\n' "$RUN" >&2; exit 68; }
for mount in "$EVIDENCE" /tmp; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2
    exit 67
  }
done

mkdir "$RUN"
while read -r digest name; do
  case "$name" in
    namelist_cfg) cp -a "$DECK/namelist_cfg" "$RUN/$name" ;;
    namelist_ice_cfg) cp -a "$SENTINEL" "$RUN/$name" ;;
    *) cp -a "$SOURCE/$name" "$RUN/$name" ;;
  esac
done <"$SOURCE/deck_files.sha256"
while read -r digest name; do cp -a "$SOURCE/$name" "$RUN/$name"; done <"$SOURCE/input_files.sha256"
cp -a "$SOURCE/deck_files.sha256" "$SOURCE/input_files.sha256" "$RUN/"
cp -a "$SOURCE/nemo" "$RUN/nemo"
cp -a "$BUILD/BLD/ppsrc/nemo/stprk3.f90" "$RUN/compiled_stprk3.f90"
cp -a "$BUILD/BLD/ppsrc/nemo/l4_r69_surface.f90" "$RUN/compiled_l4_r69_surface.f90"
cp -a "$MANIFEST" "$RUN/hierarchy_manifest.json"
printf '%s\n' "$COMMIT" >"$RUN/producer_commit.txt"

# The inherited deck manifest pins the upper rung; verify every unchanged row
# explicitly and the two intentional execution-deck substitutions separately.
while read -r digest name; do
  case "$name" in
    namelist_cfg) pin 9177246fef7f2d9da1dcfcf13b1ccb8107b3004b45074159ffab67accc86eb15 "$RUN/$name" 'rung-9 ocean namelist' ;;
    namelist_ice_cfg) cmp -s "$SENTINEL" "$RUN/$name" || { printf 'REFUSE: execution sentinel differs\n' >&2; exit 66; } ;;
    *) pin "$digest" "$RUN/$name" "inherited deck file $name" ;;
  esac
done <"$SOURCE/deck_files.sha256"
(cd "$RUN" && sha256sum -c input_files.sha256 >/dev/null)

(
  cd "$RUN"
  for output in ocean.output time.step run.user.stdout.log run.user.time.log SHA256SUMS; do
    [[ ! -e "$output" ]] || { printf 'REFUSE: output exists: %s/%s\n' "$RUN" "$output" >&2; exit 70; }
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
  grep -q 'STOP 0' run.user.stdout.log || { printf 'REFUSE: NEMO did not report STOP 0\n' >&2; exit 70; }
  [[ "$(tr -d '[:space:]' <time.step)" == 240 ]] || { printf 'REFUSE: NEMO did not reach step 240\n' >&2; exit 70; }
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\0' | sort -z | xargs -0 sha256sum >SHA256SUMS
)
admit
