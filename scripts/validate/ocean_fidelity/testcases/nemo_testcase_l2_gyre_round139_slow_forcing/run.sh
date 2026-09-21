#!/usr/bin/env bash
set -Eeuo pipefail

# Round-139 passive developed-state slow-forcing split.  This clones the
# admitted Round-137 source card, changes no namelist value, and adds one
# WRITE-only stream around the already executing initial dyn_cor_2D call.
refuse_on_error() {
  local status=$?
  printf 'REFUSE: round139 acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R137EXT
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R139SLOW
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round137/oracle_developed_external
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round139/oracle_developed_slow_forcing
readonly RESTART_1080=GYRE_OMIP_L2_P3_00001080_restart.nc
readonly PROCESS_1081=oracle_process_budget_kt00001081.bin
readonly EXTERNAL_RECORD=oracle_bt_step_operands_kt00001081.bin
readonly QCO_RECORD=oracle_stage1_qco_operands_kt00001081.bin
readonly SPLIT_RECORD=oracle_slow_forcing_split_kt00001081.bin
readonly SOURCE_BINARY_SHA=7adeef8aa59015b75a5bf04a6c1a5063e29bcbb712f7af0b22838e2e72be0d83
readonly RESTART_SHA=6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976
readonly PROCESS_SHA=526d1fc73faeda990c661f2363a5bb328168bae17d4aea315daf05c35b4cd7b0
readonly EXTERNAL_SHA=6bc0f990ccba183a48ad09549b72b549603694e917389eb4f8b9c03c88ceaf09
readonly QCO_SHA=626d21e229f7ced8f606f6385e04224fb2e086cef92d81d95dff7e2ef3e90878
readonly EXPECTED_SPLIT_SIZE=33852

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--plant-layout) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--plant-layout]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly DYN_PATCH=$here/dynspg_ts_round139.patch
readonly SPLIT_GATE=$here/../nemo_testcase_l2_gyre_round83_slow_forcing_walk.py
readonly BT_GATE=$here/../nemo_testcase_l2_gyre_round81_btstep_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round139.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$DYN_PATCH" "$SPLIT_GATE" "$BT_GATE" "$PREREG" \
            "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" \
            "$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" \
            "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
            "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
            "$SOURCE_RUN/$RESTART_1080" "$SOURCE_RUN/$PROCESS_1081" \
            "$SOURCE_RUN/$EXTERNAL_RECORD" "$SOURCE_RUN/$QCO_RECORD"; do
  if [[ ! -f "$path" ]]; then
    printf 'REFUSE: missing source-card input %s\n' "$path" >&2
    exit 64
  fi
done
for path in "$SOURCE_ROOT/EXP00" "$SOURCE_ROOT/MY_SRC"; do
  if [[ ! -d "$path" ]]; then
    printf 'REFUSE: missing source-card directory %s\n' "$path" >&2
    exit 64
  fi
done
if [[ "$MODE" == --run && ! -w "$NEMO_ROOT/cfgs" ]]; then
  printf 'REFUSE: NEMO configuration directory is not writable in this sandbox\n' >&2
  exit 64
fi

digest=$(sha256sum "$SOURCE_ROOT/BLD/bin/nemo.exe" | awk '{print $1}')
if [[ "$digest" != "$SOURCE_BINARY_SHA" ]] || \
   ! cmp -s "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo"; then
  printf 'REFUSE: Round-137 source binary ancestry changed (%s)\n' "$digest" >&2
  exit 65
fi
for pair in "$RESTART_1080:$RESTART_SHA" "$PROCESS_1081:$PROCESS_SHA" \
            "$EXTERNAL_RECORD:$EXTERNAL_SHA" "$QCO_RECORD:$QCO_SHA"; do
  name=${pair%%:*}
  expected=${pair#*:}
  digest=$(sha256sum "$SOURCE_RUN/$name" | awk '{print $1}')
  if [[ "$digest" != "$expected" ]]; then
    printf 'REFUSE: source control %s is %s, expected %s\n' \
      "$name" "$digest" "$expected" >&2
    exit 65
  fi
done
for pattern in 'number of the last time step.*nn_itend *= *1081' \
  'ocean time step.*rn_Dt *= *14400' \
  'Barotropic time steps => in seconds *= *288' \
  'in iterations nn_e *= *50' \
  'Barotropic time filter => nn_bt_flt *= *3' \
  'Tiling \(T\) or not \(F\).*ln_tile *= *F' \
  'open boundaries not used \(ln_bdy = F\)' \
  'Logical for Dir. Lim wd option.*ln_wd_dl *= *F'; do
  if ! grep -Eq "$pattern" "$SOURCE_RUN/ocean.output"; then
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  fi
done
for row in 'nn_itend *= *1081' 'nn_stock *= *180' 'nn_write *= *2160' \
           'nn_pert_seed *= *0'; do
  if ! grep -Eq "^[[:space:]]*$row" "$SOURCE_RUN/namelist_cfg"; then
    printf 'REFUSE: source namelist lacks row: %s\n' "$row" >&2
    exit 65
  fi
done

removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$DYN_PATCH")
if [[ "$removed" -ne 0 ]]; then
  printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2
  exit 66
fi

dry=$(mktemp -d /tmp/gyre-r139-source.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90"
patch -s --fuzz=0 "$dry/dynspg_ts.F90" <"$DYN_PATCH"

check_layout() {
  local dyn=$1
  [[ "$(grep -Fc "r139_magic = 'NEMO_L2_R139SLOW'" "$dyn")" -eq 1 ]] &&
  [[ "$(grep -Fc '& ntsi, ntei, ntsj, ntej, 6' "$dyn")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r139_unit) Ue_rhs(ntsi:ntei,ntsj:ntej), Ve_rhs(ntsi:ntei,ntsj:ntej)' "$dyn")" -eq 1 ]] &&
  [[ "$(grep -F -A1 'WRITE(r139_unit) Ue_rhs(ntsi:ntei,ntsj:ntej), Ve_rhs(ntsi:ntei,ntsj:ntej)' "$dyn" | grep -Fc '& zu_trd(ntsi:ntei,ntsj:ntej), zv_trd(ntsi:ntei,ntsj:ntej)')" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r139_unit) zu_frc(ntsi:ntei,ntsj:ntej), zv_frc(ntsi:ntei,ntsj:ntej)' "$dyn")" -eq 1 ]]
}
if ! check_layout "$dry/dynspg_ts.F90"; then
  printf 'REFUSE: dry source-card byte layout is incomplete\n' >&2
  exit 66
fi
reader_size=$("$PY" -c "import runpy; print(runpy.run_path('$SPLIT_GATE')['ROUND139_EXPECTED_SIZE'])")
if [[ "$reader_size" -ne "$EXPECTED_SPLIT_SIZE" ]]; then
  printf 'REFUSE: writer/reader byte layouts disagree: %s versus %s\n' \
    "$EXPECTED_SPLIT_SIZE" "$reader_size" >&2
  exit 66
fi
if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/dynspg_ts.planted.F90
  cp "$dry/dynspg_ts.F90" "$planted"
  sed -i 's/WRITE(r139_unit) zu_frc(ntsi:ntei,ntsj:ntej), zv_frc(ntsi:ntei,ntsj:ntej)/WRITE(r139_unit) Ue_rhs(ntsi:ntei,ntsj:ntej), Ve_rhs(ntsi:ntei,ntsj:ntej)/' "$planted"
  if check_layout "$planted"; then
    printf 'REFUSE: layout plant stayed green\n' >&2
  else
    printf 'STATUS PLANT-FIRED: layout\n'
    printf 'REFUSE: planted Round-139 layout violation was detected\n' >&2
  fi
  exit 69
fi

syntax=$(mktemp -d /tmp/gyre-r139-syntax.XXXXXX)
printf 'temporary syntax-proof directory (retained): %s\n' "$syntax"
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
  -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/dynspg_ts.F90" -o "$syntax/dynspg_ts.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/dynspg_ts.f90"
printf 'SYNTAX_PROOF_PASS dynspg_ts.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND139_DEVELOPED_SLOW_FORCING_PREFLIGHT_READY %s\n' "$SOURCE_RUN"
  exit 0
fi

if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: new target already exists: %s or %s\n' \
    "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
fi
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 4194304 ]]; then
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  fi
done

manifest=$(mktemp -d /tmp/gyre-r139-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$DYN_PATCH" "$SPLIT_GATE" \
  "$BT_GATE" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" \
  "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
if [[ ! -d "$TARGET_ROOT/EXP00" || ! -d "$TARGET_ROOT/MY_SRC" ]]; then
  printf 'REFUSE: makenemo could not create the new Round-139 configuration\n' >&2
  exit 68
fi
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" <"$DYN_PATCH"
if ! cmp -s "$dry/dynspg_ts.F90" "$TARGET_ROOT/MY_SRC/dynspg_ts.F90"; then
  printf 'REFUSE: target writer differs from syntax-proved source\n' >&2
  exit 68
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED_DYN=$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90
if [[ ! -x "$BINARY" || ! -f "$COMPILED_DYN" ]]; then
  printf 'REFUSE: target binary or compiled source is missing\n' >&2
  exit 68
fi
if ! cmp -s "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
     "$TARGET_ROOT/cpp_$TARGET_CFG.fcm" || ! check_layout "$COMPILED_DYN"; then
  printf 'REFUSE: compiled target keys or writer layout differ\n' >&2
  exit 68
fi
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in acquisition binary\n' >&2
  exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  if [[ ! -f "$SOURCE_RUN/$name" ]]; then
    printf 'REFUSE: admitted prepared input is missing: %s\n' "$name" >&2
    exit 68
  fi
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
if ! cmp -s "$SOURCE_RUN/namelist_cfg" "$TARGET_RUN/namelist_cfg"; then
  printf 'REFUSE: staged namelist differs from the admitted Round-137 run\n' >&2
  exit 68
fi
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"

(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >run.user.time.log
  started=$SECONDS
  mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  status=${PIPESTATUS[0]}
  elapsed=$((SECONDS - started))
  printf 'wall_seconds %s\n' "$elapsed" >>run.user.time.log
  if [[ "$status" -ne 0 ]]; then
    printf 'REFUSE: NEMO acquisition process failed\n' >&2
    exit 69
  fi
  if ! grep -Fxq 'STOP 0' run.user.stdout.log; then
    printf 'REFUSE: NEMO run did not terminate with STOP 0\n' >&2
    exit 69
  fi
  printf 'NEMO_FINISHED_UTC=%s\nNEMO_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)

count=$(find "$TARGET_RUN" -maxdepth 1 -type f -name "$SPLIT_RECORD" | wc -l)
if [[ "$count" -ne 1 || ! -f "$TARGET_RUN/$SPLIT_RECORD" ]]; then
  printf 'REFUSE: expected exactly one split record, found %s\n' "$count" >&2
  exit 70
fi
bytes=$(stat -c %s "$TARGET_RUN/$SPLIT_RECORD")
if [[ "$bytes" -ne "$EXPECTED_SPLIT_SIZE" ]]; then
  printf 'REFUSE: split record is %s bytes, expected %s\n' \
    "$bytes" "$EXPECTED_SPLIT_SIZE" >&2
  exit 70
fi

passive_admit() {
  local candidate_restart=$1
  local candidate_process=$2
  local candidate_external=$3
  local candidate_qco=$4
  for triple in \
    "$RESTART_1080:$SOURCE_RUN/$RESTART_1080:$candidate_restart" \
    "$PROCESS_1081:$SOURCE_RUN/$PROCESS_1081:$candidate_process" \
    "$EXTERNAL_RECORD:$SOURCE_RUN/$EXTERNAL_RECORD:$candidate_external" \
    "$QCO_RECORD:$SOURCE_RUN/$QCO_RECORD:$candidate_qco"; do
    local label=${triple%%:*}
    local remainder=${triple#*:}
    local reference=${remainder%%:*}
    local candidate=${remainder#*:}
    if ! cmp -s "$reference" "$candidate"; then
      printf 'REFUSE: instrument perturbed admitted %s\n' "$label" >&2
      return 1
    fi
  done
}
passive_admit "$TARGET_RUN/$RESTART_1080" "$TARGET_RUN/$PROCESS_1081" \
  "$TARGET_RUN/$EXTERNAL_RECORD" "$TARGET_RUN/$QCO_RECORD"
plant_dir=$(mktemp -d /tmp/gyre-r139-admission-plant.XXXXXX)
cp "$TARGET_RUN/$PROCESS_1081" "$plant_dir/$PROCESS_1081"
"$PY" -c "from pathlib import Path; p=Path('$plant_dir/$PROCESS_1081'); b=bytearray(p.read_bytes()); b[-1]^=1; p.write_bytes(b)"
if passive_admit "$TARGET_RUN/$RESTART_1080" "$plant_dir/$PROCESS_1081" \
     "$TARGET_RUN/$EXTERNAL_RECORD" "$TARGET_RUN/$QCO_RECORD" \
     >"$TARGET_RUN/round139_passive_admission_plant.log" 2>&1; then
  printf 'REFUSE: passive-admission plant stayed green\n' >&2
  exit 71
fi
printf 'STATUS PLANT-FIRED: passive-admission\n' \
  >>"$TARGET_RUN/round139_passive_admission_plant.log"

for record in "$SPLIT_RECORD" "$EXTERNAL_RECORD" "$QCO_RECORD"; do
  printf '%s %s %s\n' \
    "$(sha256sum "$TARGET_RUN/$record" | awk '{print $1}')" \
    "$COMMIT" "$record" >"$TARGET_RUN/$record.stamp"
done
printf '%s\n' "$COMMIT" >"$TARGET_RUN/producer_commit.txt"

"$PY" "$BT_GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
  --record "$EXTERNAL_RECORD" --expected-kt 1081 --qco-record "$QCO_RECORD" \
  --output "$TARGET_RUN/round139_parent_record_validation.json"

r139_gate() {
  "$PY" "$SPLIT_GATE" --round139-record-only \
    --expect-commit "$COMMIT" --expect-record-commit "$COMMIT" \
    --expect-krhs-commit "$COMMIT" --round139-root "$TARGET_RUN" "$@"
}
r139_gate --output "$TARGET_RUN/round139_record_validation.json"
for plant in record-stamp record-header record-truncation record-replay-ulp; do
  if r139_gate --plant "$plant" \
       --output "$TARGET_RUN/round139_${plant}_plant.json" \
       >"$TARGET_RUN/round139_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: Round-139 %s plant stayed green\n' "$plant" >&2
    exit 72
  fi
  if ! grep -Fq 'STATUS PLANT-FIRED' \
       "$TARGET_RUN/round139_${plant}_plant.log"; then
    printf 'REFUSE: Round-139 %s exited without its plant marker\n' \
      "$plant" >&2
    exit 72
  fi
done

(
  cd "$TARGET_RUN"
  sha256sum "$SPLIT_RECORD" "$SPLIT_RECORD.stamp" \
    "$EXTERNAL_RECORD" "$EXTERNAL_RECORD.stamp" \
    "$QCO_RECORD" "$QCO_RECORD.stamp" "$RESTART_1080" "$PROCESS_1081" \
    round139_parent_record_validation.json round139_record_validation.json \
    round139_*_plant.log round139_*_plant.json \
    >round139_outputs.sha256
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
printf 'ROUND139_DEVELOPED_SLOW_FORCING_READY %s\n' "$TARGET_RUN"
