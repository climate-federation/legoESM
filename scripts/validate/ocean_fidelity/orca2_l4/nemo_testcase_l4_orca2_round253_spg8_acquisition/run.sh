#!/usr/bin/env bash
# Operator-executed, rank-complete OMT-4 kt=8 external-substep record.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-253 OMT4 SPG8 acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing|--plant-content) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing|--plant-content]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly BASE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly BINARY_RUN=$BASE/orca2_rounds/round166/acquisition/orca2_rung0_spgts_kt8_ranked_10step_np2
readonly DECK_RUN=$BASE/orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2
readonly EVIDENCE=$BASE/orca2_rounds/round253/acquisition
readonly SMOKE_RUN=$EVIDENCE/orca2_omt4_spg8_smoke_2step_np2
readonly TARGET_RUN=$EVIDENCE/orca2_omt4_spg8_ranked_10step_np2
readonly BINARY_SHA=23b20aa74dff4713ce225a3a8891a0b41a8110db79342a86db8fddff38337237
readonly DECK_NML_SHA=334a398b49c6361559be7327bec0278e481b29860f9020556d7e1247d8cb3b61
readonly WRITER_SHA=78abcc75c0d6c740e59a51e29bc0d44ff57d312cf3220d674b7bd3eafb41d368
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly CHECKER=$here/check_record.py
readonly WRITER=$here/../nemo_testcase_l4_orca2_round166_spg8_acquisition/l4_r95_spgts_frames.F90
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round253.md

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
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
for path in "$0" "$CHECKER" "$WRITER" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
pin "$BINARY_SHA" "$BINARY_RUN/nemo" 'admitted round-166 record binary'
pin "$DECK_NML_SHA" "$DECK_RUN/namelist_cfg" 'admitted OMT-4 namelist'
pin "$WRITER_SHA" "$WRITER" 'committed self-describing writer'
(cd "$DECK_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
grep -Eq '^ *nn_itend *= *10' "$DECK_RUN/namelist_cfg" || {
  printf 'REFUSE: OMT-4 deck is not the admitted ten-step protocol\n' >&2; exit 66;
}
grep -Eq '^ *nn_stock *= *10' "$DECK_RUN/namelist_cfg" || {
  printf 'REFUSE: OMT-4 terminal restart cadence moved\n' >&2; exit 66;
}
mkdir -p "$EVIDENCE"
bash -n "$0"

copy_deck() {
  local target=$1
  mkdir "$target"
  while read -r _ name; do cp -a "$DECK_RUN/$name" "$target/$name"; done <"$DECK_RUN/deck_files.sha256"
  while read -r _ name; do cp -a "$DECK_RUN/$name" "$target/$name"; done <"$DECK_RUN/input_files.sha256"
  cp "$DECK_RUN/deck_files.sha256" "$DECK_RUN/input_files.sha256" "$target/"
  cp "$BINARY_RUN/nemo" "$target/nemo"
}

run_nemo() {
  local target=$1 label=$2
  (
    cd "$target"
    set +e
    mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    pipe_rc=("${PIPESTATUS[@]}")
    set -e
    [[ "${pipe_rc[0]}" -eq 0 && "${pipe_rc[1]:-0}" -eq 0 ]] || {
      printf 'REFUSE: %s NEMO run failed (mpi=%s tee=%s)\n' "$label" "${pipe_rc[0]}" "${pipe_rc[1]:-0}" >&2; exit 69;
    }
    grep -Fxq 'STOP 0' run.user.stdout.log || {
      printf 'REFUSE: %s run lacks STOP 0\n' "$label" >&2; exit 69;
    }
  )
}

verify_manifest() {
  local manifest=$1 path name count digest
  [[ -f "$manifest" ]] || { printf 'REFUSE: missing producer content manifest\n' >&2; exit 70; }
  for path in "$0" "$CHECKER" "$WRITER" "$PREREG"; do
    name=$(basename "$path")
    count=$(awk -v name="$name" '{n=split($2,p,"/"); if(p[n]==name)c++} END{print c+0}' "$manifest")
    [[ "$count" -eq 1 ]] || { printf 'REFUSE: producer manifest count for %s is %s\n' "$name" "$count" >&2; exit 70; }
    digest=$(awk -v name="$name" '{n=split($2,p,"/"); if(p[n]==name)print $1}' "$manifest")
    pin "$digest" "$path" "recorded producer content $name"
  done
}

admit() {
  local plant
  verify_manifest "$TARGET_RUN/toolchain.sha256"
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r166_spg_rank????_kt00000008.bin' | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: expected exactly two rank kt=8 SPG records\n' >&2; exit 70;
  }
  for plant in header field-name field-dims truncation missing-frame swapped-rank restart-byte; do
    if "$PY" "$CHECKER" --root "$TARGET_RUN" --baseline "$DECK_RUN" --plant "$plant" \
      >"$TARGET_RUN/round253_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round253_${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  "$PY" "$CHECKER" --root "$TARGET_RUN" --baseline "$DECK_RUN" \
    --output "$TARGET_RUN/round253_spg8_admission.json"
  printf 'ORCA2_ROUND253_OMT4_SPG8_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --plant-content ]]; then
  plant_manifest=$(mktemp /tmp/orca2-r253-content.XXXXXX)
  sha256sum "$0" "$CHECKER" "$WRITER" "$PREREG" >"$plant_manifest"
  sed -i '1s/^[0-9a-f]/z/' "$plant_manifest"
  if (verify_manifest "$plant_manifest") >/dev/null 2>&1; then
    printf 'REFUSE: content plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED content\n'; exit 69
fi
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND253_OMT4_SPG8_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi
if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_RUN/nemo" ]] || { printf 'REFUSE: existing target lacks binary\n' >&2; exit 68; }
  admit
  exit 0
fi

[[ ! -e "$SMOKE_RUN" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: smoke or target run directory already exists\n' >&2; exit 68;
}
for mount in /tmp "$EVIDENCE"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67; }
done

copy_deck "$SMOKE_RUN"
sed -i -E 's/^( *nn_itend *= *).*/\1 2/' "$SMOKE_RUN/namelist_cfg"
sed -i -E 's/^( *nn_stock *= *).*/\1 2/' "$SMOKE_RUN/namelist_cfg"
sed -i -E 's/^( *nn_stocklist *= *).*/\1 2/' "$SMOKE_RUN/namelist_cfg"
grep -Eq '^ *nn_itend *= *2' "$SMOKE_RUN/namelist_cfg" || { printf 'REFUSE: smoke itend edit failed\n' >&2; exit 68; }
grep -Eq '^ *nn_stock *= *2' "$SMOKE_RUN/namelist_cfg" || { printf 'REFUSE: smoke stock edit failed\n' >&2; exit 68; }
run_nemo "$SMOKE_RUN" smoke

copy_deck "$TARGET_RUN"
sha256sum "$0" "$CHECKER" "$WRITER" "$PREREG" >"$TARGET_RUN/toolchain.sha256"
run_nemo "$TARGET_RUN" record
admit
