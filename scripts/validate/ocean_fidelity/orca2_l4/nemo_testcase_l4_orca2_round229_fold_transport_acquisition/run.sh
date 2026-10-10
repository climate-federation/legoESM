#!/usr/bin/env bash
# Operator-executed OMT-4 rank-complete tracer-consumer operand acquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-229 fold acquisition failed at line %s (exit %s)\n' \
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
readonly SOURCE_CFG=ORCA2_OMIP_L4_R210OMT1_P3
readonly TARGET_CFG=ORCA2_OMIP_L4_R229FOLDTRP
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round229/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_omt4_fold_transport_10step_np2
readonly WORK_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work
readonly SOURCE_STP_SHA=2b1636336acc5a5e3326291801c217f263f8ccc3ca875dbfec0d3b789d911e5b
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly SOURCE_BINARY_SHA=5b82a3254c40f71186af159b93cba419709ccf49cf4172ad3d44440b8fb1d895
readonly DECK_MANIFEST_SHA=539987382ab7cb038d268614cbf8b6644304e4a0bb8e1e0386197812b69a0624
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/stprk3_stg_round229.patch
readonly WRITER=$here/l4_r229_fold.F90
readonly CHECKER=$here/check_record.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round229.md

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

check_layout() {
  local source=$1 call_line consumer_line
  [[ "$(grep -Fc 'USE l4_r229_fold, ONLY : r229_dump_fold_operands' "$source")" -eq 1 ]] || return 1
  [[ "$(grep -Fc 'CALL r229_dump_fold_operands' "$source")" -eq 1 ]] || return 1
  call_line=$(grep -nF 'CALL r229_dump_fold_operands' "$source" | cut -d: -f1)
  consumer_line=$(grep -nF 'CALL tra_adv_trp' "$source" | tail -1 | cut -d: -f1)
  [[ -n "$call_line" && -n "$consumer_line" && "$call_line" -gt "$consumer_line" ]]
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "$0" "$PATCH" "$WRITER" "$CHECKER" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
  git diff --quiet HEAD -- "$path" || {
    printf 'REFUSE: acquisition artifact differs from HEAD: %s\n' "$path" >&2; exit 64;
  }
done
pin "$SOURCE_STP_SHA" "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" 'source stprk3_stg'
pin "$SOURCE_CPP_SHA" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source-run binary'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'input manifest'
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
grep -Fxq 'STOP 0' "$SOURCE_RUN/run.user.stdout.log" || {
  printf 'REFUSE: admitted OMT-4 source run did not complete\n' >&2; exit 65;
}
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
[[ "$removed" -eq 0 ]] || {
  printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2; exit 66;
}
mkdir -p "$EVIDENCE" "$WORK_ROOT"
bash -n "$0"
"$PY" -m py_compile "$CHECKER"
scratch=$(mktemp -d "$WORK_ROOT/orca2-r229-fold.XXXXXX")
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$scratch/stprk3_stg.F90"
patch -s --fuzz=0 -p0 -d "$scratch" <"$PATCH"
check_layout "$scratch/stprk3_stg.F90" || {
  printf 'REFUSE: tracer-consumer writer is misplaced\n' >&2; exit 66;
}
grep -Fq "STATUS='NEW'" "$WRITER" || {
  printf 'REFUSE: writer does not fail on an existing record\n' >&2; exit 66;
}
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND229_FOLD_TRANSPORT_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r229_fold_rank????_kt00000001_s1.bin' | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: expected exactly two rank-complete operand records\n' >&2; exit 70;
  }
  for plant in rank field-name truncation; do
    if "$PY" "$CHECKER" --root "$TARGET_RUN" --plant "$plant" >"$TARGET_RUN/round229_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round229_${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  "$PY" "$CHECKER" --root "$TARGET_RUN" --output "$TARGET_RUN/round229_fold_record_admission.json"
  for rank in 0000 0001; do
    cmp -s "$SOURCE_RUN/ORCA2_00000010_restart_${rank}.nc" \
      "$TARGET_RUN/ORCA2_00000010_restart_${rank}.nc" || {
      printf 'REFUSE: rank %s terminal restart moved under writer\n' "$rank" >&2; exit 72;
    }
  done
  (cd "$TARGET_RUN" && sha256sum oracle_r229_fold_rank*.bin ORCA2_00000010_restart_????.nc \
    round229_*_plant.log round229_fold_record_admission.json >round229_outputs.sha256)
  printf 'ORCA2_ROUND229_FOLD_TRANSPORT_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_RUN/nemo" ]] || { printf 'REFUSE: existing target lacks binary\n' >&2; exit 68; }
  grep -Fxq 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || {
    printf 'REFUSE: existing target lacks STOP 0\n' >&2; exit 68;
  }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 68;
}
for mount in "$WORK_ROOT" "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67;
  }
done

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l4_r229_fold.F90"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90" || {
  printf 'REFUSE: compiled tracer-consumer writer moved\n' >&2; exit 69;
}

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
(cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  mpi_rc=${pipe_rc[0]}
  tee_rc=${pipe_rc[1]:-0}
  printf 'MPIRUN_RC=%d\nWALL_SECONDS=%d\nRUN_FINISHED_UTC=%s\n' "$mpi_rc" \
    "$((SECONDS-started))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  [[ "$mpi_rc" -eq 0 && "$tee_rc" -eq 0 ]] || {
    printf 'REFUSE: NEMO run failed (mpi=%s tee=%s)\n' "$mpi_rc" "$tee_rc" >&2; exit 69;
  }
  grep -Fxq 'STOP 0' run.user.stdout.log || {
    printf 'REFUSE: completed run lacks STOP 0\n' >&2; exit 69;
  }
)
admit
