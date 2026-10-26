#!/usr/bin/env bash
set -Eeuo pipefail

# Round-137 passive developed-state external-mode acquisition.  This clones
# the admitted Round-123 source card, stops after step 1081, and adds only two
# write-only streams.  It never modifies canonical NEMO source.
refuse_on_error() {
  local status=$?
  printf 'REFUSE: round137 acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R123PROC
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R137EXT
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round123/oracle_process_budget
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round137/oracle_developed_external
readonly RESTART_1080=GYRE_OMIP_L2_P3_00001080_restart.nc
readonly PROCESS_1081=oracle_process_budget_kt00001081.bin
readonly RECORD=oracle_bt_step_operands_kt00001081.bin
readonly QCO_RECORD=oracle_stage1_qco_operands_kt00001081.bin
readonly SOURCE_BINARY_SHA=f1984b87c4d0c6efea6d4bbb4ffbfee1eea1eb595f9952c909476e9e2245124a
readonly RESTART_SHA=6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976
readonly PROCESS_SHA=526d1fc73faeda990c661f2363a5bb328168bae17d4aea315daf05c35b4cd7b0
readonly EXPECTED_RECORD_SIZE=10444796
readonly EXPECTED_QCO_SIZE=22508

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--plant-layout) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--plant-layout]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly DYN_PATCH=$here/dynspg_ts_round137.patch
readonly STAGE_PATCH=$here/stprk3_stg_round137.patch
readonly NAMELIST_PATCH=$here/namelist_cfg_round137.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round81_btstep_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round137.md
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

for path in "$DYN_PATCH" "$STAGE_PATCH" "$NAMELIST_PATCH" "$GATE" \
            "$PREREG" "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" \
            "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" \
            "$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" \
            "$SOURCE_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90" \
            "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
            "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
            "$SOURCE_RUN/$RESTART_1080" "$SOURCE_RUN/$PROCESS_1081"; do
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
  printf 'REFUSE: Round-123 source binary ancestry changed (%s)\n' "$digest" >&2
  exit 65
fi
for pair in "$RESTART_1080:$RESTART_SHA" "$PROCESS_1081:$PROCESS_SHA"; do
  name=${pair%%:*}
  expected=${pair#*:}
  digest=$(sha256sum "$SOURCE_RUN/$name" | awk '{print $1}')
  if [[ "$digest" != "$expected" ]]; then
    printf 'REFUSE: source control %s is %s, expected %s\n' \
      "$name" "$digest" "$expected" >&2
    exit 65
  fi
done
for pattern in 'number of the last time step.*nn_itend *= *1440' \
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
for row in 'nn_itend *= *1440' 'nn_stock *= *180' 'nn_write *= *2160' \
           'nn_pert_seed *= *0'; do
  if ! grep -Eq "^[[:space:]]*$row" "$SOURCE_RUN/namelist_cfg"; then
    printf 'REFUSE: source namelist lacks row: %s\n' "$row" >&2
    exit 65
  fi
done

for patch_path in "$DYN_PATCH" "$STAGE_PATCH"; do
  removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$patch_path")
  if [[ "$removed" -ne 0 ]]; then
    printf 'REFUSE: writer patch removes %s source lines: %s\n' \
      "$removed" "$patch_path" >&2
    exit 66
  fi
done
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$NAMELIST_PATCH")
if [[ "$removed" -ne 1 ]] || ! grep -q '^-.*nn_itend.*1440' "$NAMELIST_PATCH"; then
  printf 'REFUSE: namelist patch does not replace exactly nn_itend=1440\n' >&2
  exit 66
fi

dry=$(mktemp -d /tmp/gyre-r137-source.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90"
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
patch -s --fuzz=0 "$dry/dynspg_ts.F90" <"$DYN_PATCH"
patch -s --fuzz=0 "$dry/stprk3_stg.F90" <"$STAGE_PATCH"
cp "$SOURCE_RUN/namelist_cfg" "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$NAMELIST_PATCH"

check_layout() {
  local dyn=$1
  local stg=$2
  [[ "$(grep -Fc "l2_magic = 'NEMO_L2_BTSTP_1'" "$dyn")" -eq 1 ]] &&
  [[ "$(grep -Fc '& ntsi, ntei, ntsj, ntej, 37, rDt_e' "$dyn")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r137_unit) jn, za1, za2, za3' "$dyn")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r137_unit) za0, za1, za2, za3, zsshp2_e' "$dyn")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r137_unit) pssh(ntsi:ntei,ntsj:ntej,Kaa)' "$dyn")" -eq 1 ]] &&
  [[ "$(grep -Fc "r137_qco_magic = 'NEMO_L2_R137QCO'" "$stg")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r137_qco_unit) ssha, r1_ht_0, r3ta' "$stg")" -eq 1 ]]
}
if ! check_layout "$dry/dynspg_ts.F90" "$dry/stprk3_stg.F90"; then
  printf 'REFUSE: dry source-card byte layout is incomplete\n' >&2
  exit 66
fi
reader_record_size=$("$PY" -c "import runpy; print(runpy.run_path('$GATE')['EXPECTED_DEVELOPED_SIZE'])")
reader_qco_size=$("$PY" -c "import runpy; print(runpy.run_path('$GATE')['EXPECTED_QCO_SIZE'])")
if [[ "$reader_record_size" -ne "$EXPECTED_RECORD_SIZE" || \
      "$reader_qco_size" -ne "$EXPECTED_QCO_SIZE" ]]; then
  printf 'REFUSE: writer/reader byte layouts disagree: %s/%s versus %s/%s\n' \
    "$EXPECTED_RECORD_SIZE" "$EXPECTED_QCO_SIZE" \
    "$reader_record_size" "$reader_qco_size" >&2
  exit 66
fi
if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/dynspg_ts.planted.F90
  cp "$dry/dynspg_ts.F90" "$planted"
  sed -i 's/WRITE(r137_unit) pssh(ntsi:ntei,ntsj:ntej,Kaa)/WRITE(r137_unit) ssha_e(ntsi:ntei,ntsj:ntej)/' "$planted"
  if check_layout "$planted" "$dry/stprk3_stg.F90"; then
    printf 'REFUSE: layout plant stayed green\n' >&2
  else
    printf 'STATUS PLANT-FIRED: layout\n'
  fi
  exit 69
fi

syntax=$(mktemp -d /tmp/gyre-r137-syntax.XXXXXX)
printf 'temporary syntax-proof directory (retained): %s\n' "$syntax"
for unit in dynspg_ts stprk3_stg; do
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
    -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$dry/$unit.F90" -o "$syntax/$unit.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
    -J "$syntax" "$syntax/$unit.f90"
  printf 'SYNTAX_PROOF_PASS %s.f90\n' "$unit"
done
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND137_DEVELOPED_EXTERNAL_PREFLIGHT_READY %s\n' "$SOURCE_RUN"
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

manifest=$(mktemp -d /tmp/gyre-r137-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$DYN_PATCH" "$STAGE_PATCH" \
  "$NAMELIST_PATCH" "$GATE" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90" \
  "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
if [[ ! -d "$TARGET_ROOT/EXP00" || ! -d "$TARGET_ROOT/MY_SRC" ]]; then
  printf 'REFUSE: makenemo could not create the new Round-137 configuration\n' >&2
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
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$STAGE_PATCH"
if ! cmp -s "$dry/dynspg_ts.F90" "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" || \
   ! cmp -s "$dry/stprk3_stg.F90" "$TARGET_ROOT/MY_SRC/stprk3_stg.F90"; then
  printf 'REFUSE: target writer differs from syntax-proved source\n' >&2
  exit 68
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED_DYN=$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90
readonly COMPILED_STAGE=$TARGET_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90
if [[ ! -x "$BINARY" || ! -f "$COMPILED_DYN" || ! -f "$COMPILED_STAGE" ]]; then
  printf 'REFUSE: target binary or compiled source is missing\n' >&2
  exit 68
fi
if ! cmp -s "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
     "$TARGET_ROOT/cpp_$TARGET_CFG.fcm" || \
   ! check_layout "$COMPILED_DYN" "$COMPILED_STAGE"; then
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
patch -s --fuzz=0 "$TARGET_RUN/namelist_cfg" <"$NAMELIST_PATCH"
if ! cmp -s "$dry/namelist_cfg" "$TARGET_RUN/namelist_cfg"; then
  printf 'REFUSE: staged namelist differs from preregistered truncation\n' >&2
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

for item in "$RECORD:$EXPECTED_RECORD_SIZE" "$QCO_RECORD:$EXPECTED_QCO_SIZE"; do
  name=${item%%:*}
  expected=${item#*:}
  count=$(find "$TARGET_RUN" -maxdepth 1 -type f -name "$name" | wc -l)
  if [[ "$count" -ne 1 || ! -f "$TARGET_RUN/$name" ]]; then
    printf 'REFUSE: expected exactly one %s record, found %s\n' "$name" "$count" >&2
    exit 70
  fi
  bytes=$(stat -c %s "$TARGET_RUN/$name")
  if [[ "$bytes" -ne "$expected" ]]; then
    printf 'REFUSE: %s is %s bytes, expected %s\n' "$name" "$bytes" "$expected" >&2
    exit 70
  fi
done

passive_admit() {
  local candidate_restart=$1
  local candidate_process=$2
  if ! cmp -s "$SOURCE_RUN/$RESTART_1080" "$candidate_restart"; then
    printf 'REFUSE: instrument perturbed admitted step-1080 restart\n' >&2
    return 1
  fi
  if ! cmp -s "$SOURCE_RUN/$PROCESS_1081" "$candidate_process"; then
    printf 'REFUSE: instrument perturbed admitted step-1081 process record\n' >&2
    return 1
  fi
}
passive_admit "$TARGET_RUN/$RESTART_1080" "$TARGET_RUN/$PROCESS_1081"
plant_dir=$(mktemp -d /tmp/gyre-r137-admission-plant.XXXXXX)
cp "$TARGET_RUN/$PROCESS_1081" "$plant_dir/$PROCESS_1081"
"$PY" -c "from pathlib import Path; p=Path('$plant_dir/$PROCESS_1081'); b=bytearray(p.read_bytes()); b[-1]^=1; p.write_bytes(b)"
if passive_admit "$TARGET_RUN/$RESTART_1080" "$plant_dir/$PROCESS_1081" \
     >"$TARGET_RUN/round137_passive_admission_plant.log" 2>&1; then
  printf 'REFUSE: passive-admission plant stayed green\n' >&2
  exit 71
fi
printf 'STATUS PLANT-FIRED: passive-admission\n' \
  >>"$TARGET_RUN/round137_passive_admission_plant.log"

printf '%s %s %s\n' "$(sha256sum "$TARGET_RUN/$RECORD" | awk '{print $1}')" \
  "$COMMIT" "$RECORD" >"$TARGET_RUN/$RECORD.stamp"
printf '%s %s %s\n' "$(sha256sum "$TARGET_RUN/$QCO_RECORD" | awk '{print $1}')" \
  "$COMMIT" "$QCO_RECORD" >"$TARGET_RUN/$QCO_RECORD.stamp"
printf '%s\n' "$COMMIT" >"$TARGET_RUN/producer_commit.txt"

r137_gate() {
  "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
    --record "$RECORD" --expected-kt 1081 --qco-record "$QCO_RECORD" "$@"
}
r137_gate --output "$TARGET_RUN/round137_record_validation.json"
for plant in stamp header truncation replay-ulp swap-ulp final-pssh-ulp qco-ulp; do
  if r137_gate --plant "$plant" \
       >"$TARGET_RUN/round137_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: round137 %s plant stayed green\n' "$plant" >&2
    exit 72
  fi
  if ! grep -Fq 'REFUSE:' "$TARGET_RUN/round137_${plant}_plant.log"; then
    printf 'REFUSE: round137 %s exited without a named refusal\n' "$plant" >&2
    exit 72
  fi
  printf 'STATUS PLANT-FIRED: %s\n' "$plant" \
    >>"$TARGET_RUN/round137_${plant}_plant.log"
done

(
  cd "$TARGET_RUN"
  sha256sum "$RECORD" "$RECORD.stamp" "$QCO_RECORD" "$QCO_RECORD.stamp" \
    "$RESTART_1080" "$PROCESS_1081" round137_record_validation.json \
    round137_*_plant.log >round137_outputs.sha256
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
printf 'ROUND137_DEVELOPED_EXTERNAL_READY %s\n' "$TARGET_RUN"
