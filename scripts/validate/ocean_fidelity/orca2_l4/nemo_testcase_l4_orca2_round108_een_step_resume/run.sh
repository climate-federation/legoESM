#!/usr/bin/env bash
# Operator-executed fresh-target resume of the round-107 EEN acquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-108 EEN step resume failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing|--plant-environment|--plant-rank-log|--plant-toolchain) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing|--plant-environment|--plant-rank-log|--plant-toolchain]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly TARGET_CFG=ORCA2_OMIP_L4_R107EENSTEP
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly FAILED_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round107/acquisition/orca2_rung0_een_step_ranked_10step_np2
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round105/acquisition/orca2_rung0_een_accum_repair_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round108/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_een_step_ranked_resume_10step_np2
readonly BINARY_SHA=3557a0c3666538ff291a3716c948216422d52b3baabbadb726ffa4a0201b2a0e
readonly COMPILED_DYNSPG_SHA=ccb5b552356701ead51f65cd83a2a4cf426ded5644ad33c87086de350c066941
readonly NML_SHA=5192355842d9233d8356ab87b4ff8b65eac539e66dc135f77a07e26451d360e8
readonly DECK_MANIFEST_SHA=0e40688deddd7a22f8a6a7105ebc623c80e335d7bea5d88a80702bf447148312
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly GATE=$REPO/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round107_een_step_acquisition/check_record.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round108.md

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

check_environment_contract() {
  local source=$1
  [[ "$(grep -Fc 'export ORCA2_R105_EEN_ACCUM_DIR=$TARGET_RUN' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'export ORCA2_R107_EEN_STEP_DIR=$TARGET_RUN' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'both recorder directories are not absolute/pre-created' "$source")" -eq 1 ]]
}

marker_ranks() {
  local root=$1 recorder=$2 token=$3
  if [[ "$token" == INIT ]]; then
    find "$root" -maxdepth 1 -type f \
      \( -name 'ocean.output*' -o -name 'run.user.stdout.log' \) -print0 |
      sort -z | xargs -0 grep -h "ORCA2_${recorder}_${token}" 2>/dev/null |
      awk '{print $(NF-1)}' | sort
  else
    find "$root" -maxdepth 1 -type f \
      \( -name 'ocean.output*' -o -name 'run.user.stdout.log' \) -print0 |
      sort -z | xargs -0 grep -h "ORCA2_${recorder}_${token}" 2>/dev/null |
      awk '{print $NF}' | sort
  fi
}

check_rank_markers() {
  local root=$1 recorder token ranks
  for recorder in R105_EEN_ACCUM R107_EEN_STEP; do
    for token in INIT DUMP; do
      ranks=$(marker_ranks "$root" "$recorder" "$token")
      [[ "$ranks" == $'0\n1' ]] || return 1
    done
  done
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "$0" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$FAILED_RUN/nemo" "$FAILED_RUN/compiled_dynspg_ts.f90" \
  "$FAILED_RUN/namelist_cfg" "$FAILED_RUN/deck_files.sha256" \
  "$FAILED_RUN/input_files.sha256" "$FAILED_RUN/ocean.output"; do
  [[ -f "$path" ]] || { printf 'REFUSE: failed run is missing %s\n' "$path" >&2; exit 64; }
done
pin "$BINARY_SHA" "$FAILED_RUN/nemo" 'round-107 binary'
pin "$BINARY_SHA" "$TARGET_ROOT/BLD/bin/nemo.exe" 'round-107 built binary'
pin "$COMPILED_DYNSPG_SHA" "$FAILED_RUN/compiled_dynspg_ts.f90" 'compiled dynspg_ts'
pin "$NML_SHA" "$FAILED_RUN/namelist_cfg" 'rung-0 namelist'
pin "$DECK_MANIFEST_SHA" "$FAILED_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$FAILED_RUN/input_files.sha256" 'input manifest'
(cd "$FAILED_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
grep -Fq 'round105: missing EEN operand output directory' "$FAILED_RUN/ocean.output" || {
  printf 'REFUSE: failed run does not carry the diagnosed inherited-recorder error\n' >&2; exit 66;
}
[[ "$(find "$FAILED_RUN" -maxdepth 1 -type f -name 'oracle_r107_een_step_rank????_kt00000001.bin' -size 0c | wc -l)" -eq 2 ]] || {
  printf 'REFUSE: failed run no longer has exactly two zero-byte round-107 streams\n' >&2; exit 66;
}
grep -Fq 'CALL r105_een_accum_init' "$FAILED_RUN/compiled_dynspg_ts.f90" &&
grep -Fq 'CALL r107_een_step_init' "$FAILED_RUN/compiled_dynspg_ts.f90" || {
  printf 'REFUSE: compiled recorder chain moved\n' >&2; exit 66;
}
check_environment_contract "$0" || { printf 'REFUSE: two-recorder environment contract is incomplete\n' >&2; exit 66; }
bash -n "$0"

scratch=$(mktemp -d /tmp/orca2-r108-een.XXXXXX)
printf 'temporary control directory (retained): %s\n' "$scratch"
if [[ "$MODE" == --plant-environment ]]; then
  cp "$0" "$scratch/run.sh"
  sed -i '/export ORCA2_R105_EEN_ACCUM_DIR=/d' "$scratch/run.sh"
  if check_environment_contract "$scratch/run.sh"; then
    printf 'REFUSE: inherited-environment plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED environment\n'
  exit 69
fi
if [[ "$MODE" == --plant-rank-log ]]; then
  printf ' ORCA2_R105_EEN_ACCUM_INIT 1 0 /tmp/a\n ORCA2_R105_EEN_ACCUM_DUMP 1 0\n ORCA2_R107_EEN_STEP_INIT 1 0 /tmp/b\n ORCA2_R107_EEN_STEP_DUMP 1 0\n' >"$scratch/ocean.output"
  : >"$scratch/run.user.stdout.log"
  if check_rank_markers "$scratch"; then
    printf 'REFUSE: rank-log plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED rank-log\n'
  exit 69
fi

verify_recorded_tools() {
  local manifest=${1:-$TARGET_RUN/toolchain.sha256} digest path
  [[ -f "$manifest" && "$(wc -l <"$manifest")" -eq 3 ]] || {
    printf 'REFUSE: producer content manifest is missing or has wrong cardinality\n' >&2; exit 70;
  }
  while read -r digest path; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { printf 'REFUSE: malformed producer digest\n' >&2; exit 70; }
    pin "$digest" "$path" "recorded producer content $(basename "$path")"
  done <"$manifest"
}

if [[ "$MODE" == --plant-toolchain ]]; then
  sha256sum "$0" "$GATE" "$PREREG" >"$scratch/toolchain.sha256"
  sed -i '1s/^[0-9a-f]/z/' "$scratch/toolchain.sha256"
  if (verify_recorded_tools "$scratch/toolchain.sha256") >/dev/null 2>&1; then
    printf 'REFUSE: producer-content plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED toolchain\n'
  exit 69
fi

admit() {
  local plant record expected recorded rank
  verify_recorded_tools
  recorded=$(cat "$TARGET_RUN/producer_commit.txt")
  [[ "$recorded" =~ ^[0-9a-f]{40}$ ]] || { printf 'REFUSE: malformed producer token\n' >&2; exit 70; }
  check_rank_markers "$TARGET_RUN" || { printf 'REFUSE: recorder markers are not exactly once per rank\n' >&2; exit 70; }
  for rank in 0000 0001; do
    cmp -s "$TARGET_RUN/oracle_r104_een_accum_rank${rank}_kt00000001.bin" \
      "$SOURCE_RUN/oracle_r104_een_accum_rank${rank}_kt00000001.bin" || {
      printf 'REFUSE: inherited round-105 record moved on rank %s\n' "$rank" >&2; exit 70;
    }
  done
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r107_een_step_rank????_kt00000001.bin' -size +0c | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: expected exactly two nonempty round-107 records\n' >&2; exit 70;
  }
  for record in "$TARGET_RUN"/oracle_r107_een_step_rank????_kt00000001.bin; do
    expected="$(sha256sum "$record" | awk '{print $1}') $recorded $(basename "$record")"
    [[ "$(cat "$record.stamp")" == "$expected" ]] || { printf 'REFUSE: record stamp moved: %s\n' "$record" >&2; exit 70; }
  done
  for plant in header field-name field-dims truncation missing-field duplicate-rank bottom recurrence restart-byte; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --source-root "$SOURCE_RUN" --plant "$plant" >"$TARGET_RUN/round108_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round108_${plant}_plant.log" || { printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71; }
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --source-root "$SOURCE_RUN" --output "$TARGET_RUN/round108_een_step_admission.json"
  (cd "$TARGET_RUN" && sha256sum oracle_r104_een_accum_rank*.bin oracle_r107_een_step_rank*.bin oracle_r107_een_step_rank*.bin.stamp ORCA2_000000??_restart_????.nc round108_*_plant.log round108_een_step_admission.json >round108_outputs.sha256)
  printf 'ORCA2_ROUND108_EEN_STEP_RESUME_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_RUN/nemo" ]] || { printf 'REFUSE: existing target lacks binary\n' >&2; exit 68; }
  pin "$BINARY_SHA" "$TARGET_RUN/nemo" 'existing round-107 binary'
  pin "$NML_SHA" "$TARGET_RUN/namelist_cfg" 'existing rung-0 namelist'
  (cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
  admit
  exit 0
fi

mkdir -p "$EVIDENCE"
[[ ! -e "$TARGET_RUN" ]] || { printf 'REFUSE: fresh resume target already exists\n' >&2; exit 68; }
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67; }
done
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND108_EEN_STEP_RESUME_READY %s\n' "$TARGET_RUN"
  exit 0
fi

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$FAILED_RUN/$name" "$TARGET_RUN/$name"; done <"$FAILED_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$FAILED_RUN/$name" "$TARGET_RUN/$name"; done <"$FAILED_RUN/input_files.sha256"
cp "$FAILED_RUN/deck_files.sha256" "$FAILED_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$FAILED_RUN/nemo" "$TARGET_RUN/nemo"
cp "$FAILED_RUN/compiled_dynspg_ts.f90" "$TARGET_RUN/compiled_dynspg_ts.f90"
printf '%s\n' "$COMMIT" >"$TARGET_RUN/producer_commit.txt"
sha256sum "$0" "$GATE" "$PREREG" >"$TARGET_RUN/toolchain.sha256"
(cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
(
  cd "$TARGET_RUN"
  export ORCA2_R105_EEN_ACCUM_DIR=$TARGET_RUN
  export ORCA2_R107_EEN_STEP_DIR=$TARGET_RUN
  [[ "$ORCA2_R105_EEN_ACCUM_DIR" == /* && "$ORCA2_R107_EEN_STEP_DIR" == /* \
    && -d "$ORCA2_R105_EEN_ACCUM_DIR" && -d "$ORCA2_R107_EEN_STEP_DIR" ]] || {
    printf 'REFUSE: both recorder directories are not absolute/pre-created\n' >&2; exit 69;
  }
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  mpi_rc=${pipe_rc[0]}; tee_rc=${pipe_rc[1]:-0}
  printf 'MPIRUN_RC=%d\nWALL_SECONDS=%d\nRUN_FINISHED_UTC=%s\n' "$mpi_rc" "$((SECONDS-started))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  [[ "$mpi_rc" -eq 0 && "$tee_rc" -eq 0 ]] || { printf 'REFUSE: NEMO run failed (mpi=%s tee=%s)\n' "$mpi_rc" "$tee_rc" >&2; exit 69; }
  grep -Fxq 'STOP 0' run.user.stdout.log || { printf 'REFUSE: completed run lacks STOP 0\n' >&2; exit 69; }
  for record in oracle_r107_een_step_rank????_kt00000001.bin; do
    printf '%s %s %s\n' "$(sha256sum "$record" | awk '{print $1}')" "$COMMIT" "$record" >"$record.stamp"
  done
  printf 'RUN DONE\n' >>run.user.time.log
)
admit
