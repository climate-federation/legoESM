#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: unexpected failure at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

# USER-EXECUTED ACQUISITION ONLY. --run invokes makenemo/mpirun;
# --admit-existing never rebuilds or reruns NEMO.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R59TKE
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R101TKEW
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round59/oracle_tke_operands
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round101/oracle_tke_statement_walk
readonly ROUND101=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round101
readonly STAGE_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage
readonly RECORD=oracle_tke_statement_walk_kt00000002.bin
readonly LEGACY_RECORD=$SOURCE_RUN/oracle_tke_operands_kt00000002.bin
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly EXPECTED_SIZE=$((16 + 13 * 4 + 5 * 32 * 22 * 31 * 8))

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--plant-layout|--admit-existing|--plant-size) ;;
  *)
    printf 'REFUSE: usage: %s [--run|--preflight-only|--plant-layout|--admit-existing|--plant-size]\n' "$0" >&2
    exit 64
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BASE=$SOURCE_ROOT/MY_SRC/zdftke.F90
readonly WRITER=$here/l2_r101_tke_walk.F90
readonly PATCH=$here/zdftke_round101.patch
readonly STAGE_GATE=$here/../nemo_testcase_l2_gyre_round46_kt2_stage_gate.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round101.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/zdftke.f90

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$BASE" "$WRITER" "$PATCH" "$STAGE_GATE" "$ADMISSION" \
    "$PREREG" "$LEGACY_RECORD" "$STAGE_ROOT/oracle_rkstage3_terms_kt00000001.bin"; do
  [[ -f "$path" ]] || {
    printf 'REFUSE: missing required path %s\n' "$path" >&2
    exit 64
  }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]] || {
  printf 'REFUSE: source configuration is incomplete: %s\n' "$SOURCE_ROOT" >&2
  exit 64
}
[[ -f "$SOURCE_RUN/$FINAL_RESTART" && -f "$SOURCE_RUN/mesh_mask.nc" ]] || {
  printf 'REFUSE: source run lacks restart or mesh identity file\n' >&2
  exit 64
}
[[ "$EXPECTED_SIZE" -eq 873028 ]] || {
  printf 'REFUSE: registered byte-layout arithmetic moved from 873028\n' >&2
  exit 66
}
for pattern in 'number of the last time step.*nn_itend *= *10' \
  'Assimilation cycle.*nn_no *= *0' 'ln_tile *= *F' \
  'ocean time step.*rn_Dt *= *14400\.0+' \
  'prandl number flag.*nn_pdl.*= *1' \
  'mixing length type.*nn_mxl.*= *3' \
  'Langmuir cells parametrization.*ln_lc.*= *T' \
  'test param. to add tke induced by wind.*nn_etau.*= *0'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || {
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  }
done

dry=$(mktemp -d /tmp/gyre-r101-source.XXXXXXXX)
cp "$BASE" "$dry/zdftke.F90"
patch -s "$dry/zdftke.F90" <"$PATCH"
[[ -z "$(diff "$BASE" "$dry/zdftke.F90" | grep '^<' || true)" ]] || {
  printf 'REFUSE: Round-101 patch removes or replaces a source-card line\n' >&2
  exit 66
}
check_layout() {
  local source=$1
  [[ "$(grep -c 'CALL r101_tke_begin' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_entry' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_after_boundaries_row' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_after_langmuir_row' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_rhs_row' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_post_sweep_row' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_tke_finish' "$source")" -eq 1 ]]
}
check_layout "$dry/zdftke.F90" || {
  printf 'REFUSE: patched source lacks the registered seven calls\n' >&2
  exit 66
}
grep -Fq "CHARACTER(LEN=16), PARAMETER :: r101_magic = 'NEMO_L2_R101TKE'" \
  "$WRITER" || {
  printf 'REFUSE: writer magic/layout declaration moved\n' >&2
  exit 66
}

syntax=$(mktemp -d /tmp/gyre-r101-syntax.XXXXXXXX)
preprocess() {
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
    -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$1" -o "$2"
}
preprocess "$WRITER" "$syntax/l2_r101_tke_walk.f90"
preprocess "$dry/zdftke.F90" "$syntax/zdftke.f90"
if ! "$FC" -fsyntax-only -ffree-line-length-none \
     -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" \
     "$syntax/l2_r101_tke_walk.f90"; then
  printf 'REFUSE: Round-101 writer failed gfortran syntax proof\n' >&2
  exit 66
fi
if ! "$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" \
     -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/zdftke.f90"; then
  printf 'REFUSE: patched zdftke failed gfortran syntax proof\n' >&2
  exit 66
fi

if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/zdftke.planted.F90
  cp "$dry/zdftke.F90" "$planted"
  sed -i '/CALL r101_after_langmuir_row/d' "$planted"
  if check_layout "$planted"; then
    printf 'REFUSE: source-layout plant stayed green\n' >&2
    exit 2
  fi
  printf 'REFUSE: source-layout plant removed the post-Langmuir boundary\n' >&2
  exit 69
fi
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND101_TKE_STATEMENT_PREFLIGHT_READY %s\n' "$SOURCE_RUN"
  exit 0
fi

stage_gate() {
  "$PY" "$STAGE_GATE" --root "$STAGE_ROOT" \
    --expect-commit "$COMMIT" --mode stage-tke-record \
    --round40-kt1 "$STAGE_ROOT/oracle_rkstage3_terms_kt00000001.bin" \
    --round41-kt1 "$STAGE_ROOT/oracle_dynadv_split_kt00000001_s3.bin" \
    --tke-statement-root "$TARGET_RUN" --tke-operand-record "$LEGACY_RECORD" \
    "$@"
}

validate_existing() {
  for path in "$TARGET_ROOT" "$TARGET_RUN" "$BINARY" "$COMPILED" \
      "$TARGET_RUN/$RECORD" "$TARGET_RUN/$RECORD.stamp" \
      "$TARGET_RUN/nemo" "$TARGET_RUN/binary.sha256" \
      "$TARGET_RUN/producer_commit.txt" "$TARGET_RUN/source_cfg.sha256" \
      "$TARGET_RUN/toolchain.sha256" "$TARGET_RUN/run.user.stdout.log" \
      "$TARGET_RUN/run.user.time.log" "$TARGET_RUN/$FINAL_RESTART" \
      "$TARGET_RUN/mesh_mask.nc"; do
    [[ -e "$path" ]] || {
      printf 'REFUSE: existing acquisition lacks %s\n' "$path" >&2
      exit 64
    }
  done
  check_layout "$COMPILED" || {
    printf 'REFUSE: compiled zdftke lacks the registered seven calls\n' >&2
    exit 66
  }
  grep -Fq 'NEMO_L2_R101TKE' \
      "$TARGET_ROOT/BLD/ppsrc/nemo/l2_r101_tke_walk.f90" || {
    printf 'REFUSE: compiled build lacks the Round-101 writer magic\n' >&2
    exit 66
  }
  local actual_size
  actual_size=$(stat -c %s "$TARGET_RUN/$RECORD")
  if [[ "$MODE" == --plant-size ]]; then
    local planted_size=$((EXPECTED_SIZE + 8))
    if [[ "$actual_size" -eq "$planted_size" ]]; then
      printf 'REFUSE: byte-size plant stayed green\n' >&2
      exit 2
    fi
    printf 'REFUSE: byte-size plant expected %s but record has %s\n' \
      "$planted_size" "$actual_size" >&2
    exit 69
  fi
  [[ "$actual_size" -eq "$EXPECTED_SIZE" ]] || {
    printf 'REFUSE: record size %s != compiled layout %s\n' \
      "$actual_size" "$EXPECTED_SIZE" >&2
    exit 66
  }
  local producer
  producer=$(tr -d '[:space:]' <"$TARGET_RUN/producer_commit.txt")
  [[ "$producer" == "$COMMIT" ]] || {
    printf 'REFUSE: producer commit %s != current committed tree %s\n' \
      "$producer" "$COMMIT" >&2
    exit 66
  }
  local expected_binary recorded_binary built_binary copied_binary
  read -r expected_binary recorded_binary <"$TARGET_RUN/binary.sha256"
  [[ "$recorded_binary" == "$BINARY" ]] || {
    printf 'REFUSE: binary manifest names %s, not %s\n' \
      "$recorded_binary" "$BINARY" >&2
    exit 66
  }
  built_binary=$(sha256sum "$BINARY" | awk '{print $1}')
  copied_binary=$(sha256sum "$TARGET_RUN/nemo" | awk '{print $1}')
  [[ "$built_binary" == "$expected_binary" \
      && "$copied_binary" == "$expected_binary" ]] || {
    printf 'REFUSE: built/executed binary digest moved\n' >&2
    exit 66
  }
  if nm -D "$BINARY" | grep -q '_ZGV'; then
    printf 'REFUSE: vector-math symbol present in recorded binary\n' >&2
    exit 68
  fi
  grep -Fxq 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || {
    printf 'REFUSE: NEMO stdout lacks STOP 0\n' >&2
    exit 66
  }
  grep -Fxq 'RUN_DONE' "$TARGET_RUN/run.user.time.log" || {
    printf 'REFUSE: acquisition lacks RUN_DONE\n' >&2
    exit 66
  }
  local digest stamped_digest stamped_commit stamped_name
  digest=$(sha256sum "$TARGET_RUN/$RECORD" | awk '{print $1}')
  read -r stamped_digest stamped_commit stamped_name \
    <"$TARGET_RUN/$RECORD.stamp"
  [[ "$stamped_digest" == "$digest" && "$stamped_commit" == "$COMMIT" \
      && "$stamped_name" == "$RECORD" ]] || {
    printf 'REFUSE: record stamp disagrees with digest/commit/name\n' >&2
    exit 66
  }
  if ! (
    cd "$SOURCE_ROOT"
    sha256sum -c "$TARGET_RUN/source_cfg.sha256"
  ) >"$ROUND101/round101_source_cfg_check.log" 2>&1; then
    printf 'REFUSE: source-card manifest no longer verifies\n' >&2
    exit 66
  fi

  if ! stage_gate --output "$TARGET_RUN/round101_tke_record_validation.json"; then
    printf 'REFUSE: clean Round-101 TKE record gate failed\n' >&2
    exit 66
  fi
  local plant_status
  for plant in stage-tke-record-header stage-tke-record-truncation \
      stage-tke-record-stamp stage-tke-record-ulp stamp; do
    if stage_gate --plant "$plant" \
        >"$TARGET_RUN/round101_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: stage gate plant %s stayed green\n' "$plant" >&2
      exit 69
    else
      plant_status=$?
    fi
    [[ "$plant_status" -eq 1 ]] || {
      printf 'REFUSE: stage gate plant %s exited %s instead of 1\n' \
        "$plant" "$plant_status" >&2
      exit 69
    }
  done

  if ! "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
      --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
      --allowed-new "$RECORD" \
      --output "$TARGET_RUN/round101_admission.json"; then
    printf 'REFUSE: clean Round-101 twin admission failed\n' >&2
    exit 66
  fi
  if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
      --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
      --allowed-new "$RECORD" --plant-consumed \
      --output "$TARGET_RUN/round101_admission_plant.json" \
      >"$TARGET_RUN/round101_admission_plant.log" 2>&1; then
    printf 'REFUSE: twin-admission consumed-field plant stayed green\n' >&2
    exit 69
  else
    plant_status=$?
  fi
  [[ "$plant_status" -eq 1 ]] || {
    printf 'REFUSE: twin-admission plant exited %s instead of 1\n' \
      "$plant_status" >&2
    exit 69
  }
  (
    cd "$TARGET_RUN"
    sha256sum "$RECORD" "$RECORD.stamp" "$FINAL_RESTART" mesh_mask.nc \
      round101_*.json round101_*_plant.log \
      >round101_outputs.sha256
  )
  printf 'ROUND101_TKE_STATEMENT_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing || "$MODE" == --plant-size ]]; then
  validate_existing
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists; use --admit-existing without rebuilding: %s or %s\n' \
    "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
}
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  }
done

manifest=$(mktemp -d /tmp/gyre-r101-manifest.XXXXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$BASE" "$WRITER" \
  "$PATCH" "$STAGE_GATE" "$ADMISSION" "$PREREG" \
  "$SOURCE_RUN/ocean.output" "$LEGACY_RECORD" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) \
  -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) \
  -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_r101_tke_walk.F90"
patch -s "$TARGET_ROOT/MY_SRC/zdftke.F90" <"$PATCH"
cmp "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || {
  printf 'REFUSE: target build produced no executable\n' >&2
  exit 68
}
check_layout "$COMPILED" || {
  printf 'REFUSE: compiled zdftke lacks the registered seven calls\n' >&2
  exit 68
}
grep -Fq 'NEMO_L2_R101TKE' \
    "$TARGET_ROOT/BLD/ppsrc/nemo/l2_r101_tke_walk.f90" || {
  printf 'REFUSE: compiled build lacks the Round-101 writer magic\n' >&2
  exit 68
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in new binary\n' >&2
  exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  cp -L "$TARGET_ROOT/EXP00/$name" "$TARGET_RUN/$name"
done
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  set +e
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
    2>>run.user.time.log
  run_status=${PIPESTATUS[0]}
  set -e
  if [[ "$run_status" -ne 0 ]]; then
    printf 'REFUSE: NEMO acquisition exited %s\n' "$run_status" >&2
    exit "$run_status"
  fi
  [[ -s "$RECORD" ]] || {
    printf 'REFUSE: NEMO produced no %s\n' "$RECORD" >&2
    exit 66
  }
  actual_size=$(stat -c %s "$RECORD")
  [[ "$actual_size" -eq "$EXPECTED_SIZE" ]] || {
    printf 'REFUSE: record size %s != compiled layout %s\n' \
      "$actual_size" "$EXPECTED_SIZE" >&2
    exit 66
  }
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  digest=$(sha256sum "$RECORD" | awk '{print $1}')
  printf '%s %s %s\n' "$digest" "$COMMIT" "$RECORD" >"$RECORD.stamp"
)

validate_existing
