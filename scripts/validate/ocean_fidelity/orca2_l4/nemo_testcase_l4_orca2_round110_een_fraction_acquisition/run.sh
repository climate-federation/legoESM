#!/usr/bin/env bash
# Operator-executed rank-complete rung-0 EEN fraction acquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-110 EEN fraction acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing|--plant-layout|--plant-path|--plant-environment|--plant-toolchain|--plant-rank-log) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing|--plant-layout|--plant-path|--plant-environment|--plant-toolchain|--plant-rank-log]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4_R107EENSTEP
readonly TARGET_CFG=ORCA2_OMIP_L4_R110EENFRAC
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round108/acquisition/orca2_rung0_een_step_ranked_resume_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round110/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_een_fraction_ranked_10step_np2
readonly SOURCE_DYNSPG_SHA=4d8423abdc62b1eef3d36c7dd5779e1c350283d3f10527765a74f0d868c475cc
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly SOURCE_BINARY_SHA=3557a0c3666538ff291a3716c948216422d52b3baabbadb726ffa4a0201b2a0e
readonly SOURCE_NML_SHA=5192355842d9233d8356ab87b4ff8b65eac539e66dc135f77a07e26451d360e8
readonly DECK_MANIFEST_SHA=0e40688deddd7a22f8a6a7105ebc623c80e335d7bea5d88a80702bf447148312
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/dynspg_ts_round110.patch
readonly WRITER=$here/l4_r110_een_fraction.F90
readonly GATE=$here/check_record.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round110.md

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

check_writer() {
  local source=$1
  [[ "$(grep -Fc "GET_ENVIRONMENT_VARIABLE('ORCA2_R110_EEN_FRACTION_DIR'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "output_dir(1:1) /= '/'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "STATUS='NEW'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'ALLOCATABLE, SAVE' "$source")" -ge 1 ]] &&
  [[ "$(grep -Fc 'IF(dumped) RETURN' "$source")" -eq 1 ]]
}

check_layout() {
  local source=$1
  [[ "$(grep -Fc 'USE l4_r110_een_fraction' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r110_een_fraction_init' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r110_een_fraction_store' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r110_een_fraction_dump' "$source")" -eq 1 ]]
}

check_environment_contract() {
  local source=$1
  [[ "$(grep -Ec '^  export ORCA2_R105_EEN_ACCUM_DIR=\$TARGET_RUN$' "$source")" -eq 1 ]] &&
  [[ "$(grep -Ec '^  export ORCA2_R107_EEN_STEP_DIR=\$TARGET_RUN$' "$source")" -eq 1 ]] &&
  [[ "$(grep -Ec '^  export ORCA2_R110_EEN_FRACTION_DIR=\$TARGET_RUN$' "$source")" -eq 1 ]]
}

marker_ranks() {
  local root=$1 recorder=$2 token=$3
  find "$root" -maxdepth 1 -type f \
    \( -name 'ocean.output*' -o -name 'run.user.stdout.log' \) -print0 |
    sort -z | xargs -0 grep -h "ORCA2_${recorder}_${token}" 2>/dev/null |
    awk -v token="$token" '{if(token=="INIT") print $(NF-1); else print $NF}' | sort
}

check_rank_markers() {
  local root=$1 recorder token ranks
  for recorder in R105_EEN_ACCUM R107_EEN_STEP R110_EEN_FRACTION; do
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

for path in "$0" "$PATCH" "$WRITER" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" "$SOURCE_RUN/namelist_cfg" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned input %s\n' "$path" >&2; exit 64; }
done
pin "$SOURCE_DYNSPG_SHA" "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" 'source dynspg_ts'
pin "$SOURCE_CPP_SHA" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source run binary'
pin "$SOURCE_NML_SHA" "$SOURCE_RUN/namelist_cfg" 'source rung-0 namelist'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'input manifest'
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
[[ "$(find "$SOURCE_RUN" -maxdepth 1 -type f -name 'oracle_r104_een_accum_rank????_kt00000001.bin' -size +0c | wc -l)" -eq 2 ]] || {
  printf 'REFUSE: admitted inherited accumulator record is incomplete\n' >&2; exit 66;
}
[[ "$(find "$SOURCE_RUN" -maxdepth 1 -type f -name 'oracle_r107_een_step_rank????_kt00000001.bin' -size +0c | wc -l)" -eq 2 ]] || {
  printf 'REFUSE: admitted inherited per-level record is incomplete\n' >&2; exit 66;
}

mkdir -p "$EVIDENCE"
bash -n "$0"
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2; exit 66; }
scratch=$(mktemp -d /tmp/orca2-r110-een.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$scratch"
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$scratch/dynspg_ts.F90"
cp "$WRITER" "$scratch/l4_r110_een_fraction.F90"
patch -s --fuzz=0 -p0 -d "$scratch" <"$PATCH"

if [[ "$MODE" == --plant-layout ]]; then
  sed -i '/CALL r110_een_fraction_store/d' "$scratch/dynspg_ts.F90"
  if check_layout "$scratch/dynspg_ts.F90"; then printf 'REFUSE: layout plant stayed green\n' >&2; exit 69; fi
  printf 'STATUS PLANT-FIRED layout\n'; exit 69
fi
if [[ "$MODE" == --plant-path ]]; then
  sed -i "/output_dir(1:1) \/= '\/'/d" "$scratch/l4_r110_een_fraction.F90"
  if check_writer "$scratch/l4_r110_een_fraction.F90"; then printf 'REFUSE: path plant stayed green\n' >&2; exit 69; fi
  printf 'STATUS PLANT-FIRED path\n'; exit 69
fi
if [[ "$MODE" == --plant-environment ]]; then
  cp "$0" "$scratch/run.sh"
  sed -i '/export ORCA2_R110_EEN_FRACTION_DIR=/d' "$scratch/run.sh"
  if check_environment_contract "$scratch/run.sh"; then printf 'REFUSE: environment plant stayed green\n' >&2; exit 69; fi
  printf 'STATUS PLANT-FIRED environment\n'; exit 69
fi
if [[ "$MODE" == --plant-rank-log ]]; then
  printf ' ORCA2_R105_EEN_ACCUM_INIT 1 0 /tmp/a\n ORCA2_R105_EEN_ACCUM_DUMP 1 0\n ORCA2_R107_EEN_STEP_INIT 1 0 /tmp/b\n ORCA2_R107_EEN_STEP_DUMP 1 0\n ORCA2_R110_EEN_FRACTION_INIT 1 0 /tmp/c\n' >"$scratch/ocean.output"
  : >"$scratch/run.user.stdout.log"
  if check_rank_markers "$scratch"; then printf 'REFUSE: rank-log plant stayed green\n' >&2; exit 69; fi
  printf 'STATUS PLANT-FIRED rank-log\n'; exit 69
fi

check_writer "$scratch/l4_r110_een_fraction.F90" || { printf 'REFUSE: writer contract is incomplete\n' >&2; exit 66; }
check_layout "$scratch/dynspg_ts.F90" || { printf 'REFUSE: source layout is incomplete\n' >&2; exit 66; }
check_environment_contract "$0" || { printf 'REFUSE: three-recorder environment contract is incomplete\n' >&2; exit 66; }
cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$scratch/l4_r110_een_fraction.F90" -o "$scratch/l4_r110_een_fraction.f90"
"$FC" -c -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$scratch" \
  "$scratch/l4_r110_een_fraction.f90" -o "$scratch/l4_r110_een_fraction.o"
cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$scratch/dynspg_ts.F90" -o "$scratch/dynspg_ts.f90"
check_layout "$scratch/dynspg_ts.f90" || { printf 'REFUSE: compiled layout is incomplete\n' >&2; exit 66; }
"$FC" -fsyntax-only -ffree-line-length-none -I "$scratch" -I "$SOURCE_ROOT/BLD/inc" -J "$scratch" "$scratch/dynspg_ts.f90"
printf 'SYNTAX_PROOF_PASS l4_r110_een_fraction.f90 dynspg_ts.f90\n'

verify_recorded_tools() {
  local manifest=${1:-$TARGET_RUN/toolchain.sha256} digest path
  [[ -f "$manifest" && "$(wc -l <"$manifest")" -eq 5 ]] || {
    printf 'REFUSE: producer content manifest is missing or has wrong cardinality\n' >&2; exit 70;
  }
  while read -r digest path; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { printf 'REFUSE: malformed producer digest\n' >&2; exit 70; }
    pin "$digest" "$path" "recorded producer content $(basename "$path")"
  done <"$manifest"
}

if [[ "$MODE" == --plant-toolchain ]]; then
  sha256sum "$0" "$PATCH" "$WRITER" "$GATE" "$PREREG" >"$scratch/toolchain.sha256"
  sed -i '1s/^[0-9a-f]/z/' "$scratch/toolchain.sha256"
  if (verify_recorded_tools "$scratch/toolchain.sha256") >/dev/null 2>&1; then
    printf 'REFUSE: producer-content plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED toolchain\n'; exit 69
fi

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND110_EEN_FRACTION_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant rank record expected recorded
  verify_recorded_tools
  recorded=$(cat "$TARGET_RUN/producer_commit.txt")
  [[ "$recorded" =~ ^[0-9a-f]{40}$ ]] || { printf 'REFUSE: malformed producer token\n' >&2; exit 70; }
  check_rank_markers "$TARGET_RUN" || { printf 'REFUSE: recorder markers are not exactly once per rank\n' >&2; exit 70; }
  for rank in 0000 0001; do
    cmp -s "$TARGET_RUN/oracle_r104_een_accum_rank${rank}_kt00000001.bin" \
      "$SOURCE_RUN/oracle_r104_een_accum_rank${rank}_kt00000001.bin" || {
      printf 'REFUSE: inherited round-105 record moved on rank %s\n' "$rank" >&2; exit 70;
    }
    cmp -s "$TARGET_RUN/oracle_r107_een_step_rank${rank}_kt00000001.bin" \
      "$SOURCE_RUN/oracle_r107_een_step_rank${rank}_kt00000001.bin" || {
      printf 'REFUSE: inherited round-107 record moved on rank %s\n' "$rank" >&2; exit 70;
    }
  done
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r110_een_fraction_rank????_kt00000001.bin' -size +0c | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: expected exactly two nonempty round-110 records\n' >&2; exit 70;
  }
  for record in "$TARGET_RUN"/oracle_r110_een_fraction_rank????_kt00000001.bin; do
    expected="$(sha256sum "$record" | awk '{print $1}') $recorded $(basename "$record")"
    [[ "$(cat "$record.stamp")" == "$expected" ]] || { printf 'REFUSE: record stamp moved: %s\n' "$record" >&2; exit 70; }
  done
  for plant in header field-name field-dims truncation missing-field duplicate-rank bottom denominator quotient sum inherited restart-byte; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --source-root "$SOURCE_RUN" --plant "$plant" >"$TARGET_RUN/round110_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round110_${plant}_plant.log" || { printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71; }
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --source-root "$SOURCE_RUN" --output "$TARGET_RUN/round110_een_fraction_admission.json"
  (cd "$TARGET_RUN" && sha256sum oracle_r104_een_accum_rank*.bin oracle_r107_een_step_rank*.bin oracle_r110_een_fraction_rank*.bin oracle_r110_een_fraction_rank*.bin.stamp ORCA2_000000??_restart_????.nc round110_*_plant.log round110_een_fraction_admission.json >round110_outputs.sha256)
  printf 'ORCA2_ROUND110_EEN_FRACTION_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_ROOT/BLD/bin/nemo.exe" && -x "$TARGET_RUN/nemo" ]] || { printf 'REFUSE: existing target lacks binary\n' >&2; exit 68; }
  cmp -s "$TARGET_ROOT/BLD/bin/nemo.exe" "$TARGET_RUN/nemo" || { printf 'REFUSE: staged and built binaries differ\n' >&2; exit 68; }
  check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" || { printf 'REFUSE: compiled writer layout moved\n' >&2; exit 68; }
  pin "$SOURCE_NML_SHA" "$TARGET_RUN/namelist_cfg" 'existing rung-0 namelist'
  (cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || { printf 'REFUSE: target config or run directory already exists\n' >&2; exit 68; }
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67; }
done

manifest=$(mktemp -d /tmp/orca2-r110-een-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$0" "$PATCH" "$WRITER" "$GATE" "$PREREG" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l4_r110_een_fraction.F90"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
cmp -s "$scratch/dynspg_ts.F90" "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" || { printf 'REFUSE: target source differs from syntax-proved source\n' >&2; exit 68; }
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" || { printf 'REFUSE: compiled writer layout is incomplete\n' >&2; exit 69; }
if nm -D "$BINARY" | grep -q '_ZGV'; then printf 'REFUSE: vector-math symbol present\n' >&2; exit 69; fi

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" "$TARGET_RUN/compiled_dynspg_ts.f90"
cp "$manifest"/* "$TARGET_RUN/"
(cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
(
  cd "$TARGET_RUN"
  export ORCA2_R105_EEN_ACCUM_DIR=$TARGET_RUN
  export ORCA2_R107_EEN_STEP_DIR=$TARGET_RUN
  export ORCA2_R110_EEN_FRACTION_DIR=$TARGET_RUN
  [[ "$ORCA2_R105_EEN_ACCUM_DIR" == /* && "$ORCA2_R107_EEN_STEP_DIR" == /* \
    && "$ORCA2_R110_EEN_FRACTION_DIR" == /* && -d "$TARGET_RUN" ]] || {
    printf 'REFUSE: all recorder directories must be absolute and pre-created\n' >&2; exit 69;
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
  for record in oracle_r110_een_fraction_rank????_kt00000001.bin; do
    printf '%s %s %s\n' "$(sha256sum "$record" | awk '{print $1}')" "$COMMIT" "$record" >"$record.stamp"
  done
  printf 'RUN DONE\n' >>run.user.time.log
)
admit
