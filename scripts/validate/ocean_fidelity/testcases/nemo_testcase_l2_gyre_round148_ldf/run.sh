#!/usr/bin/env bash
set -Eeuo pipefail

refuse_on_error() {
  local status=$?
  printf 'REFUSE: round148 developed-LDF acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R146RHSFAM
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R148LDF
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round146/oracle_developed_rhs_families
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round148/oracle_developed_ldf
readonly RECORD=oracle_developed_ldf_kt00001081.bin
readonly FAMILY_RECORD=oracle_developed_rhs_families_kt00001081.bin
readonly EXPECTED_SIZE=4717612
readonly SOURCE_BINARY_SHA=8270a36f619c46c196e7389bf063b4ac2c8ead418e66ffd3f1357b772de5d250
readonly MODE=${1:---run}
case "$MODE" in
  --run|--admit-existing|--preflight-only|--plant-layout) ;;
  *) printf 'REFUSE: usage: %s [--run|--admit-existing|--preflight-only|--plant-layout]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly CANONICAL_SOURCE=$NEMO_ROOT/src/OCE/DYN/dynldf_lev.F90
readonly PATCH=$here/dynldf_lev_round148.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round50_ldf_association_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round148.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$here/..:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$PATCH" "$GATE" "$PREREG" "$CANONICAL_SOURCE" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" "$SOURCE_RUN/namelist_cfg"; do
  if [[ ! -f "$path" ]]; then printf 'REFUSE: missing input %s\n' "$path" >&2; exit 64; fi
done
if [[ "$MODE" == --run && ! -w "$NEMO_ROOT/cfgs" ]]; then
  printf 'REFUSE: NEMO configuration directory is not writable\n' >&2
  exit 64
fi
digest=$(sha256sum "$SOURCE_ROOT/BLD/bin/nemo.exe" | awk '{print $1}')
if [[ "$digest" != "$SOURCE_BINARY_SHA" ]] || ! cmp -s "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo"; then
  printf 'REFUSE: Round-146 binary ancestry changed (%s)\n' "$digest" >&2
  exit 65
fi
for row in 'nn_itend *= *1081' 'nn_stock *= *180' 'nn_write *= *2160' 'nn_pert_seed *= *0'; do
  if ! grep -Eq "^[[:space:]]*$row" "$SOURCE_RUN/namelist_cfg"; then
    printf 'REFUSE: source namelist lacks row %s\n' "$row" >&2
    exit 65
  fi
done
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
if [[ "$removed" -ne 0 ]]; then printf 'REFUSE: patch removes %s lines\n' "$removed" >&2; exit 66; fi

dry=$(mktemp -d /tmp/gyre-r148-developed-ldf-source.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$CANONICAL_SOURCE" "$dry/dynldf_lev.F90"
patch -s --fuzz=0 "$dry/dynldf_lev.F90" <"$PATCH"
check_layout() {
  local source=$1
  [[ "$(grep -Fc "r148_ldf_magic = 'NEMO_L2_R148LDF'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "ROUND148_DEVELOPED_LDF_DUMP" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r148_ldf_unit)' "$source")" -eq 4 ]] &&
  [[ "$(grep -Fc 'DO jk = 1, jpkm1' "$source")" -ge 1 ]]
}
if ! check_layout "$dry/dynldf_lev.F90"; then
  printf 'REFUSE: writer layout is incomplete\n' >&2
  exit 66
fi
reader_size=$($PY -c "import runpy; print(runpy.run_path('$GATE')['DEVELOPED_EXPECTED_SIZE'])")
if [[ "$reader_size" -ne "$EXPECTED_SIZE" ]]; then
  printf 'REFUSE: writer/reader sizes disagree: %s versus %s\n' "$EXPECTED_SIZE" "$reader_size" >&2
  exit 66
fi
if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/dynldf_lev.planted.F90
  cp "$dry/dynldf_lev.F90" "$planted"
  sed -i '0,/ROUND148_DEVELOPED_LDF_DUMP/d' "$planted"
  if check_layout "$planted"; then
    printf 'REFUSE: layout plant stayed green\n' >&2
    exit 68
  fi
  printf 'STATUS PLANT-FIRED: layout\n'
  printf 'REFUSE: intentional layout-plant exit\n' >&2
  exit 69
fi

syntax=$(mktemp -d /tmp/gyre-r148-developed-ldf-syntax.XXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$dry/dynldf_lev.F90" \
  -o "$syntax/dynldf_lev.f90"
$FC -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/dynldf_lev.f90"
printf 'SYNTAX_PROOF_PASS dynldf_lev.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND148_DEVELOPED_LDF_PREFLIGHT_READY\n'
  exit 0
fi

readonly EXACT_INHERITED="GYRE_OMIP_L2_P3_00001080_restart.nc GYRE_OMIP_L2_P3_00001081_restart.nc oracle_process_budget_kt00001081.bin oracle_bt_step_operands_kt00001081.bin oracle_stage1_qco_operands_kt00001081.bin oracle_slow_forcing_split_kt00001081.bin"
admit_existing() {
  for path in "$TARGET_RUN/$RECORD" "$TARGET_RUN/$FAMILY_RECORD" \
    "$TARGET_RUN/oracle_developed_rhs_kt00001081.bin" \
    "$TARGET_RUN/binary.sha256" "$TARGET_RUN/source_cfg.sha256" \
    "$TARGET_RUN/producer_commit.txt" "$TARGET_RUN/nemo" "$BINARY"; do
    if [[ ! -f "$path" ]]; then printf 'REFUSE: existing admission lacks %s\n' "$path" >&2; exit 70; fi
  done
  if ! cmp -s "$TARGET_RUN/nemo" "$BINARY" || \
     ! (cd "$TARGET_RUN" && sha256sum -c binary.sha256 >/dev/null); then
    printf 'REFUSE: existing Round-148 binary changed\n' >&2
    exit 70
  fi
  if ! (cd "$SOURCE_ROOT" && sha256sum -c "$TARGET_RUN/source_cfg.sha256" >/dev/null); then
    printf 'REFUSE: Round-146 source-card ancestry changed\n' >&2
    exit 70
  fi
  local producer_commit
  producer_commit=$(tr -d '[:space:]' <"$TARGET_RUN/producer_commit.txt")
  if [[ ! "$producer_commit" =~ ^[0-9a-f]{40}$ ]]; then
    printf 'REFUSE: malformed producer commit %s\n' "$producer_commit" >&2
    exit 70
  fi
  printf '%s %s %s\n' "$(sha256sum "$TARGET_RUN/$RECORD" | awk '{print $1}')" \
    "$producer_commit" "$RECORD" >"$TARGET_RUN/$RECORD.stamp"
  gate() {
    $PY "$GATE" --admit-developed --expect-commit "$producer_commit" \
      --developed-record "$TARGET_RUN/$RECORD" \
      --developed-family-record "$TARGET_RUN/$FAMILY_RECORD" \
      --developed-family-baseline "$SOURCE_RUN/$FAMILY_RECORD" \
      --developed-parent-record "$TARGET_RUN/oracle_developed_rhs_kt00001081.bin" \
      --developed-parent-baseline "$SOURCE_RUN/oracle_developed_rhs_kt00001081.bin" \
      --developed-root "$TARGET_RUN" --developed-baseline-root "$SOURCE_RUN" \
      --developed-stamp "$TARGET_RUN/$RECORD.stamp" "$@"
  }
  local plant
  for plant in header truncation missing-field zcur-ulp post-ulp parent-wet-ulp parent-dry-ulp restart-byte; do
    if gate --developed-plant "$plant" >"$TARGET_RUN/round148_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2
      exit 72
    fi
    if ! grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round148_${plant}_plant.log"; then
      printf 'REFUSE: %s plant lacks marker\n' "$plant" >&2
      exit 72
    fi
  done
  gate --output "$TARGET_RUN/round148_developed_ldf_admission.json"
  (
    cd "$TARGET_RUN"
    sha256sum "$RECORD" "$RECORD.stamp" $EXACT_INHERITED \
      oracle_developed_rhs_kt00001081.bin "$FAMILY_RECORD" \
      round148_*_plant.log round148_developed_ldf_admission.json \
      >round148_outputs.sha256
    printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  )
  printf 'ROUND148_DEVELOPED_LDF_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  admit_existing
  exit 0
fi

if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: new target already exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 67
fi
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 4194304 ]]; then
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  fi
done

readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
manifest=$(mktemp -d /tmp/gyre-r148-developed-ldf-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
( cd "$SOURCE_ROOT"; find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum ) >"$manifest/source_cfg.sha256"
sha256sum "$PATCH" "$GATE" "$PREREG" "$CANONICAL_SOURCE" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/dynldf_lev.f90" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$CANONICAL_SOURCE" "$TARGET_ROOT/MY_SRC/dynldf_lev.F90"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/dynldf_lev.F90" <"$PATCH"
if ! cmp -s "$dry/dynldf_lev.F90" "$TARGET_ROOT/MY_SRC/dynldf_lev.F90"; then
  printf 'REFUSE: target writer differs from syntax-proved source\n' >&2
  exit 68
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath
if [[ ! -x "$BINARY" ]] || ! check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynldf_lev.f90"; then
  printf 'REFUSE: compiled writer is missing or incomplete\n' >&2
  exit 68
fi
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2
  exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"
mkdir "$TARGET_RUN"
for name in $PREPARED; do
  if [[ ! -f "$SOURCE_RUN/$name" ]]; then
    printf 'REFUSE: missing prepared input %s\n' "$name" >&2
    exit 68
  fi
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"

(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  status=${PIPESTATUS[0]}
  printf 'wall_seconds %s\n' "$((SECONDS-started))" >>run.user.time.log
  if [[ "$status" -ne 0 ]]; then printf 'REFUSE: NEMO process failed\n' >&2; exit 69; fi
  if ! grep -Fxq 'STOP 0' run.user.stdout.log; then printf 'REFUSE: NEMO lacks STOP 0\n' >&2; exit 69; fi
)

if [[ ! -f "$TARGET_RUN/$RECORD" ]]; then printf 'REFUSE: developed LDF record missing\n' >&2; exit 70; fi
bytes=$(stat -c %s "$TARGET_RUN/$RECORD")
if [[ "$bytes" -ne "$EXPECTED_SIZE" ]]; then
  printf 'REFUSE: record is %s bytes, expected %s\n' "$bytes" "$EXPECTED_SIZE" >&2
  exit 70
fi
for name in $EXACT_INHERITED; do
  if ! cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; then
    printf 'REFUSE: passive instrument moved inherited %s\n' "$name" >&2
    exit 71
  fi
done
admit_existing
