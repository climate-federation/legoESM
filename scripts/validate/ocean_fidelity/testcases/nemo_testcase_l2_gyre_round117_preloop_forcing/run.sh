#!/usr/bin/env bash
set -Eeuo pipefail

# USER-EXECUTED ACQUISITION ONLY.  --run invokes makenemo/mpirun.
# --admit-existing verifies and admits the already-completed run without
# invoking either command.
refuse_on_error() {
  local status=$?
  printf 'REFUSE: round117 acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

die() {
  printf 'REFUSE: %s\n' "$1" >&2
  exit "${2:-64}"
}

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing|--plant-layout|--plant-size) ;;
  *) die "usage: run.sh [--run|--preflight-only|--admit-existing|--plant-layout|--plant-size]" 62 ;;
esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R111FCTW
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R117PRELOOP
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round111/oracle_fct_writers
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round117/oracle_preloop_forcing
readonly RECOVERY_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round118/admission
readonly REFERENCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round81/oracle_btstep_kt2
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly SLOW_RECORD=oracle_slow_forcing_kt00000002.bin
readonly PRELOOP_RECORD=oracle_preloop_forcing_kt00000002.bin
readonly SLOW_EXPECTED_SIZE=1486548
readonly FULL_2D_COUNT=$((36 * 26))
readonly OWNED_2D_COUNT=$((32 * 22))
readonly PRELOOP_EXPECTED_SIZE=$((16 + 8 * 4 + (6 * FULL_2D_COUNT + 12 * OWNED_2D_COUNT) * 8))
readonly ACQUISITION_COMMIT=c8f5d513df453f3f12a3d5eb05b05be8c8d3a2fd

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90
readonly STP_PATCH=$here/stp2d_round117.patch
readonly SPG_PATCH=$here/dynspg_ts_round117.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round117_preloop_gate.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round117.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  die "acquisition requires a clean committed tree" 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$STP_PATCH" "$SPG_PATCH" "$GATE" "$ADMISSION" "$PREREG" \
            "$SOURCE_ROOT/MY_SRC/stp2d.F90" \
            "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" \
            "$SOURCE_RUN/$FINAL_RESTART" "$SOURCE_RUN/mesh_mask.nc" \
            "$SOURCE_RUN/ocean.output" \
            "$REFERENCE_RUN/oracle_bt_step_operands_kt00000002.bin"; do
  [[ -f "$path" ]] || die "missing source-card input $path"
done
for path in "$SOURCE_ROOT/EXP00" "$SOURCE_ROOT/MY_SRC" \
            "$SOURCE_ROOT/WORK" "$SOURCE_ROOT/BLD/inc"; do
  [[ -d "$path" ]] || die "missing source-card directory $path"
done

# Freeze the exact resolved source card.  The acquisition changes no namelist.
for pattern in 'number of the last time step.*nn_itend *= *10' \
  'Assimilation cycle.*nn_no *= *0' 'ln_tile *= *F' 'ln_traqsr *= *T' \
  'ln_bdy *= *F' 'ln_traadv_fct *= *T' 'nn_fct_h *= *2' \
  'nn_fct_v *= *2' 'nn_fct_imp *= *1' 'ln_zad_Aimp *= *F' \
  'ln_traldf_msc *= *F' 'ln_trabbc *= *F' 'ln_trabbl *= *F' \
  'ln_tradmp *= *F' 'ln_zdfmfc *= *F' 'ln_zdfosm *= *F' \
  'ln_zdfnpc *= *F'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || \
    die "source run lacks resolved row: $pattern" 65
done

# Patch exact R111 source copies with zero fuzz.  A line beginning '<' in a
# normal diff would mean that the supposedly additive instrument removed or
# replaced a source line, which is forbidden.
dry=$(mktemp -d /tmp/gyre-r117-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stp2d.F90" "$dry/stp2d.F90"
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90"
patch -s --fuzz=0 "$dry/stp2d.F90" <"$STP_PATCH"
patch -s --fuzz=0 "$dry/dynspg_ts.F90" <"$SPG_PATCH"
for name in stp2d dynspg_ts; do
  if diff "$SOURCE_ROOT/MY_SRC/$name.F90" "$dry/$name.F90" | grep -q '^<'; then
    die "round117 $name patch removes or replaces an R111 source line" 66
  fi
done
for check in \
  'stp2d.F90:NEMO_L2_SLOW_2' \
  'stp2d.F90:kt == nit000 + 1' \
  'dynspg_ts.F90:NEMO_L2_R117PF1' \
  'dynspg_ts.F90:SIZE(zu_frc)' \
  'dynspg_ts.F90:CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )'; do
  file=${check%%:*}
  needle=${check#*:}
  grep -Fq "$needle" "$dry/$file" || \
    die "dry source lacks $file statement: $needle" 66
done

check_writer_layout() {
  local source=$1
  grep -Eq 'ALLOCATE\( ffu_nw\((A2D\(0\)|Nis0-)' "$source" &&
  [[ "$(grep -c 'REAL(wp), DIMENSION(jpi,jpj) :: zu_trd, zu_spg' "$source")" -eq 1 ]] &&
  grep -Eq 'REAL\(wp\), DIMENSION\((A2D\(0\)|Nis0-)' "$source" &&
  [[ "$(grep -c 'WRITE(l2_r117_unit) l2_r117_magic' "$source")" -eq 1 ]] &&
  [[ "$(grep -c '& ffv_nw, ffv_ne, ffv_sw, ffv_se' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'WRITE(l2_r117_unit) zu_trd, zv_trd' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'WRITE(l2_r117_unit) zu_frc, zv_frc' "$source")" -eq 1 ]]
}
check_writer_layout "$dry/dynspg_ts.F90" || \
  die "dry source lacks the registered six-full/twelve-owned writer" 66
[[ "$PRELOOP_EXPECTED_SIZE" -eq 112560 ]] || \
  die "compiled-layout arithmetic moved from 112560" 66

if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/dynspg_ts.planted.F90
  cp "$dry/dynspg_ts.F90" "$planted"
  sed -i '/^[[:space:]]*& ffv_nw, ffv_ne, ffv_sw, ffv_se$/s/ffv_ne, //' \
    "$planted"
  if check_writer_layout "$planted"; then
    die "layout plant stayed green" 70
  fi
  printf 'STATUS PLANT-FIRED: layout: removed one compiled coefficient field\n' >&2
  die "round117 compiled-layout plant fired" 1
fi

# Use the exact preprocessor keys and include roots of the admitted R111 build,
# then make gfortran parse both complete translation units.
syntax=$(mktemp -d /tmp/gyre-r117-syntax.XXXXXX)
preprocess() {
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
    -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$1" -o "$2"
}
preprocess "$dry/stp2d.F90" "$syntax/stp2d.f90"
preprocess "$dry/dynspg_ts.F90" "$syntax/dynspg_ts.f90"
for name in stp2d dynspg_ts; do
  "$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
    -J "$syntax" "$syntax/$name.f90"
done
printf 'ROUND117_SYNTAX_PASS stp2d.f90 dynspg_ts.f90\n'
printf 'ROUND117_LAYOUT slow=%s preloop=%s full2=%s owned2=%s\n' \
  "$SLOW_EXPECTED_SIZE" "$PRELOOP_EXPECTED_SIZE" "$FULL_2D_COUNT" "$OWNED_2D_COUNT"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND117_PREFLIGHT_READY commit=%s target=%s\n' "$COMMIT" "$TARGET_CFG"
  exit 0
fi

postprocess_records() {
  local record_commit=$1
  r117_gate() {
    "$PY" "$GATE" --root "$TARGET_RUN" --reference-root "$REFERENCE_RUN" \
      --expect-commit "$COMMIT" --expect-record-commit "$record_commit" "$@"
  }
  r117_gate --output "$TARGET_RUN/round117_preloop_validation.json"
  local plant status
  for plant in stamp truncation header layout input-ulp reference-ulp; do
    if r117_gate --plant "$plant" \
         --output "$TARGET_RUN/round117_${plant}_plant.json" \
         >"$TARGET_RUN/round117_${plant}_plant.log" 2>&1; then
      die "round117 $plant plant stayed green" 70
    else
      status=$?
    fi
    [[ "$status" -eq 1 ]] || \
      die "round117 $plant plant exited $status instead of 1" 70
    grep -Fq "STATUS PLANT-FIRED: $plant" \
      "$TARGET_RUN/round117_${plant}_plant.log" || \
      die "round117 $plant plant exited without its named firing" 70
  done

  "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
    --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
    --allowed-new "$SLOW_RECORD" "$PRELOOP_RECORD" \
    --output "$TARGET_RUN/round117_admission.json"
  if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
       --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
       --allowed-new "$SLOW_RECORD" "$PRELOOP_RECORD" --plant-consumed \
       --output "$TARGET_RUN/round117_admission_plant.json" \
       >"$TARGET_RUN/round117_admission_plant.log" 2>&1; then
    die "round117 admission plant stayed green" 70
  else
    status=$?
  fi
  [[ "$status" -eq 1 ]] || \
    die "round117 admission plant exited $status instead of 1" 70

  (
    cd "$TARGET_RUN"
    sha256sum "$SLOW_RECORD" "$PRELOOP_RECORD" \
      "$SLOW_RECORD.stamp" "$PRELOOP_RECORD.stamp" \
      "$FINAL_RESTART" mesh_mask.nc round117_preloop_validation.json \
      round117_*_plant.log round117_admission.json \
      round117_admission_plant.log >round117_outputs.sha256
  )
  printf 'ROUND117_PRELOOP_RECORDS_READY %s\n' "$TARGET_RUN"
}

manifest_hash() {
  local suffix=$1
  awk -v suffix="$suffix" '
    $2 ~ (suffix "$") { value=$1; count++ }
    END { if (count != 1) exit 1; print value }
  ' "$TARGET_RUN/toolchain.sha256"
}

verify_git_tool() {
  local relative=$1
  local suffix=$2
  local expected actual
  expected=$(manifest_hash "$suffix") || \
    die "toolchain manifest lacks unique $suffix" 66
  actual=$(git show "$ACQUISITION_COMMIT:$relative" | sha256sum | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || \
    die "producer-commit tool $relative differs from manifest" 66
  printf '%s %s\n' "$actual" "$relative"
}

verify_live_tool() {
  local path=$1
  local suffix=$2
  local expected actual
  expected=$(manifest_hash "$suffix") || \
    die "toolchain manifest lacks unique $suffix" 66
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || \
    die "live acquisition input $path differs from manifest" 66
  printf '%s %s\n' "$actual" "$path"
}

admit_existing() {
  local path actual producer expected_binary recorded_binary
  local built_binary copied_binary digest
  for path in "$TARGET_ROOT" "$TARGET_RUN" "$BINARY" "$COMPILED" \
      "$TARGET_RUN/$SLOW_RECORD" "$TARGET_RUN/$PRELOOP_RECORD" \
      "$TARGET_RUN/nemo" "$TARGET_RUN/binary.sha256" \
      "$TARGET_RUN/producer_commit.txt" "$TARGET_RUN/source_cfg.sha256" \
      "$TARGET_RUN/toolchain.sha256" "$TARGET_RUN/run.user.stdout.log" \
      "$TARGET_RUN/run.user.time.log" "$TARGET_RUN/$FINAL_RESTART" \
      "$TARGET_RUN/mesh_mask.nc"; do
    [[ -e "$path" ]] || die "existing acquisition lacks $path" 64
  done
  check_writer_layout "$COMPILED" || \
    die "compiled writer differs from the registered layout" 66
  actual=$(stat -c %s "$TARGET_RUN/$PRELOOP_RECORD")
  if [[ "$MODE" == --plant-size ]]; then
    if [[ "$actual" -eq $((PRELOOP_EXPECTED_SIZE + 8)) ]]; then
      die "size plant stayed green" 70
    fi
    printf 'STATUS PLANT-FIRED: size: expected %s bytes but record has %s\n' \
      "$((PRELOOP_EXPECTED_SIZE + 8))" "$actual" >&2
    die "round117 byte-size plant fired" 1
  fi
  [[ "$actual" -eq "$PRELOOP_EXPECTED_SIZE" ]] || \
    die "$PRELOOP_RECORD has $actual bytes, compiled layout requires $PRELOOP_EXPECTED_SIZE" 66
  actual=$(stat -c %s "$TARGET_RUN/$SLOW_RECORD")
  [[ "$actual" -eq "$SLOW_EXPECTED_SIZE" ]] || \
    die "$SLOW_RECORD has $actual bytes, expected $SLOW_EXPECTED_SIZE" 66

  producer=$(tr -d '[:space:]' <"$TARGET_RUN/producer_commit.txt")
  [[ "$producer" == "$ACQUISITION_COMMIT" ]] || \
    die "producer commit $producer != registered $ACQUISITION_COMMIT" 66
  mkdir -p "$RECOVERY_ROOT"
  if ! (
    cd "$SOURCE_ROOT"
    sha256sum -c "$TARGET_RUN/source_cfg.sha256"
  ) >"$RECOVERY_ROOT/source_cfg_check.log" 2>&1; then
    die "source-card manifest no longer verifies" 66
  fi
  {
    verify_live_tool "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
      '/arch/arch-conda-scalarmath.fcm'
    verify_git_tool \
      'scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round117_preloop_forcing/stp2d_round117.patch' \
      '/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round117_preloop_forcing/stp2d_round117.patch'
    verify_git_tool \
      'scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round117_preloop_forcing/dynspg_ts_round117.patch' \
      '/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round117_preloop_forcing/dynspg_ts_round117.patch'
    verify_git_tool \
      'scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round117_preloop_gate.py' \
      '/nemo_testcase_l2_gyre_round117_preloop_forcing/../nemo_testcase_l2_gyre_round117_preloop_gate.py'
    verify_git_tool \
      'scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round21_admission.py' \
      '/nemo_testcase_l2_gyre_round117_preloop_forcing/../nemo_testcase_l2_gyre_round21_admission.py'
    verify_git_tool \
      'docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round117.md' \
      '/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round117.md'
    verify_live_tool "$SOURCE_ROOT/BLD/ppsrc/nemo/stp2d.f90" \
      '/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90'
    verify_live_tool "$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" \
      '/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90'
    verify_live_tool "$SOURCE_RUN/ocean.output" \
      '/round111/oracle_fct_writers/ocean.output'
    verify_live_tool "$REFERENCE_RUN/oracle_bt_step_operands_kt00000002.bin" \
      '/round81/oracle_btstep_kt2/oracle_bt_step_operands_kt00000002.bin'
  } >"$RECOVERY_ROOT/toolchain_check.log"

  read -r expected_binary recorded_binary <"$TARGET_RUN/binary.sha256"
  [[ "$recorded_binary" == "$BINARY" ]] || \
    die "binary manifest names $recorded_binary, not $BINARY" 66
  built_binary=$(sha256sum "$BINARY" | awk '{print $1}')
  copied_binary=$(sha256sum "$TARGET_RUN/nemo" | awk '{print $1}')
  [[ "$built_binary" == "$expected_binary" ]] || \
    die "built binary digest differs from binary.sha256" 66
  [[ "$copied_binary" == "$expected_binary" ]] || \
    die "executed binary digest differs from binary.sha256" 66
  if nm -D "$BINARY" | grep -q '_ZGV'; then
    die "vector-math symbol present in recorded binary" 68
  fi
  grep -Fxq 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || \
    die "existing NEMO stdout lacks STOP 0" 66

  for path in "$SLOW_RECORD" "$PRELOOP_RECORD"; do
    digest=$(sha256sum "$TARGET_RUN/$path" | awk '{print $1}')
    printf '%s %s %s\n' "$digest" "$producer" "$path" \
      >"$TARGET_RUN/$path.stamp"
  done
  if ! grep -Fxq 'RUN_DONE' "$TARGET_RUN/run.user.time.log"; then
    printf 'RUN_RECOVERED_UTC=%s\nRUN_DONE\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$TARGET_RUN/run.user.time.log"
  fi
  postprocess_records "$producer"
}

if [[ "$MODE" == --admit-existing || "$MODE" == --plant-size ]]; then
  admit_existing
  exit 0
fi

if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
  die "new target already exists; use --admit-existing: $TARGET_ROOT or $TARGET_RUN" 64
fi

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || die "$mount has under 4 GB free" 67
done

manifest=$(mktemp -d /tmp/gyre-r117-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$STP_PATCH" "$SPG_PATCH" "$GATE" "$ADMISSION" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/stp2d.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" \
  "$SOURCE_RUN/ocean.output" \
  "$REFERENCE_RUN/oracle_bt_step_operands_kt00000002.bin" \
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
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stp2d.F90" <"$STP_PATCH"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" <"$SPG_PATCH"
cmp "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg" || \
  die "source card changed namelist_cfg" 68
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || die "built binary is missing or not executable" 68
for check in \
  'stp2d.f90:NEMO_L2_SLOW_2' \
  'stp2d.f90:ROUND117_SLOW_FORCING_DUMP' \
  'dynspg_ts.f90:NEMO_L2_R117PF1' \
  'dynspg_ts.f90:SIZE(zu_frc)' \
  'dynspg_ts.f90:ROUND117_PRELOOP_FORCING_DUMP' \
  'dynspg_ts.f90:CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )'; do
  file=${check%%:*}
  needle=${check#*:}
  grep -Fq "$needle" "$TARGET_ROOT/BLD/ppsrc/nemo/$file" || \
    die "compiled writer check failed: $check" 68
done
if nm -D "$BINARY" | grep -q '_ZGV'; then
  die "vector-math symbol present in acquisition binary" 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  [[ -e "$TARGET_ROOT/EXP00/$name" ]] || die "prepared run input is missing: $name" 68
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
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | \
      tee run.user.stdout.log ; } 2>>run.user.time.log
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || die "NEMO acquisition process failed" 69
  grep -Fxq 'STOP 0' run.user.stdout.log || die "NEMO run lacks STOP 0" 69
  for item in "$SLOW_RECORD:$SLOW_EXPECTED_SIZE" \
              "$PRELOOP_RECORD:$PRELOOP_EXPECTED_SIZE"; do
    record=${item%%:*}
    expected=${item##*:}
    [[ -f "$record" ]] || die "NEMO did not emit $record" 69
    actual=$(stat -c %s "$record")
    [[ "$actual" -eq "$expected" ]] || \
      die "$record has $actual bytes, expected $expected" 69
    digest=$(sha256sum "$record" | awk '{print $1}')
    printf '%s %s %s\n' "$digest" "$COMMIT" "$record" >"$record.stamp"
  done
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)

postprocess_records "$COMMIT"
