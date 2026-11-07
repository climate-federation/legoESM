#!/usr/bin/env bash
set -Eeuo pipefail

refuse_on_error() {
  local status=$?
  printf 'REFUSE: round146 RHS-family acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R140RHS
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R146RHSFAM
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round140/oracle_developed_rhs
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round146/oracle_developed_rhs_families
readonly RECORD=oracle_developed_rhs_families_kt00001081.bin
readonly ROUND140_RECORD=oracle_developed_rhs_kt00001081.bin
readonly EXPECTED_SIZE=2321368
readonly SOURCE_BINARY_SHA=6703dc6b6b2bd6431f78a8649d9ea15275be61a3243632aeeb7f20e9f71dd772
readonly RESUME_BUILD_COMMIT=44a0d6e8146ae8919b4cda26653fe979186a724c
if [[ $# -eq 0 && -d "$NEMO_ROOT/cfgs/$TARGET_CFG" && -d "$TARGET_RUN" ]]; then
  if [[ -f "$TARGET_RUN/$RECORD" ]]; then
    readonly MODE=--resume-admission
  else
    readonly MODE=--resume-run
  fi
else
  readonly MODE=${1:---run}
fi
case "$MODE" in
  --run|--resume-run|--resume-admission|--preflight-only|--plant-layout) ;;
  *) printf 'REFUSE: usage: %s [--run|--resume-run|--resume-admission|--preflight-only|--plant-layout]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly PATCH=$here/stp2d_round146.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round146_rhs_family_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round146.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$PATCH" "$GATE" "$PREREG" "$SOURCE_ROOT/MY_SRC/stp2d.F90" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" "$SOURCE_RUN/namelist_cfg"; do
  if [[ ! -f "$path" ]]; then printf 'REFUSE: missing input %s\n' "$path" >&2; exit 64; fi
done
if [[ "$MODE" == --run && ! -w "$NEMO_ROOT/cfgs" ]]; then
  printf 'REFUSE: NEMO configuration directory is not writable\n' >&2
  exit 64
fi
digest=$(sha256sum "$SOURCE_ROOT/BLD/bin/nemo.exe" | awk '{print $1}')
if [[ "$digest" != "$SOURCE_BINARY_SHA" ]] || ! cmp -s "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo"; then
  printf 'REFUSE: Round-140 binary ancestry changed (%s)\n' "$digest" >&2
  exit 65
fi
for row in 'nn_itend *= *1081' 'nn_stock *= *180' 'nn_write *= *2160' 'nn_pert_seed *= *0'; do
  if ! grep -Eq "^[[:space:]]*$row" "$SOURCE_RUN/namelist_cfg"; then
    printf 'REFUSE: source namelist lacks row %s\n' "$row" >&2; exit 65
  fi
done
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
if [[ "$removed" -ne 0 ]]; then printf 'REFUSE: patch removes %s lines\n' "$removed" >&2; exit 66; fi

dry=$(mktemp -d /tmp/gyre-r146-rhs-family-source.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/stp2d.F90" "$dry/stp2d.F90"
patch -s --fuzz=0 "$dry/stp2d.F90" <"$PATCH"
check_layout() {
  local source=$1
  [[ "$(grep -Fc "r146_family_magic = 'NEMO_L2_R146FAM'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r146_family_unit) uu(:,:,:,Krhs), vv(:,:,:,Krhs)' "$source")" -eq 5 ]] &&
  [[ "$(grep -Fc 'CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )' "$source")" -eq 1 ]]
}
if ! check_layout "$dry/stp2d.F90"; then printf 'REFUSE: writer layout is incomplete\n' >&2; exit 66; fi
reader_size=$("$PY" -c "import runpy; print(runpy.run_path('$GATE')['EXPECTED_SIZE'])")
if [[ "$reader_size" -ne "$EXPECTED_SIZE" ]]; then
  printf 'REFUSE: writer/reader sizes disagree: %s versus %s\n' "$EXPECTED_SIZE" "$reader_size" >&2; exit 66
fi
if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/stp2d.planted.F90
  cp "$dry/stp2d.F90" "$planted"
  sed -i '0,/WRITE(r146_family_unit) uu(:,:,:,Krhs), vv(:,:,:,Krhs)/d' "$planted"
  if check_layout "$planted"; then printf 'REFUSE: layout plant stayed green\n' >&2; else printf 'STATUS PLANT-FIRED: layout\n'; fi
  exit 69
fi

syntax=$(mktemp -d /tmp/gyre-r146-rhs-family-syntax.XXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$dry/stp2d.F90" -o "$syntax/stp2d.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/stp2d.f90"
printf 'SYNTAX_PROOF_PASS stp2d.f90\n'
if [[ "$MODE" == --preflight-only ]]; then printf 'ROUND146_RHS_FAMILY_PREFLIGHT_READY\n'; exit 0; fi

readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
if [[ "$MODE" == --resume-run || "$MODE" == --resume-admission ]]; then
  if [[ ! -x "$BINARY" || ! -x "$TARGET_RUN/nemo" || ! -f "$TARGET_RUN/binary.sha256" ]]; then
    printf 'REFUSE: resume target lacks its built binary or staged digest\n' >&2; exit 68
  fi
  if ! sha256sum -c "$TARGET_RUN/binary.sha256" >/dev/null || ! cmp -s "$BINARY" "$TARGET_RUN/nemo"; then
    printf 'REFUSE: resume binary differs from the syntax-proved staged binary\n' >&2; exit 68
  fi
  if [[ "$(cat "$TARGET_RUN/producer_commit.txt")" != "$RESUME_BUILD_COMMIT" ]]; then
    printf 'REFUSE: resume build commit changed\n' >&2; exit 68
  fi
  if ! check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/stp2d.f90"; then
    printf 'REFUSE: resume compiled writer layout changed\n' >&2; exit 68
  fi
  for name in $PREPARED; do
    if [[ ! -f "$TARGET_RUN/$name" ]] || ! cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; then
      printf 'REFUSE: resume prepared input changed: %s\n' "$name" >&2; exit 68
    fi
  done
  if [[ "$MODE" == --resume-run && -e "$TARGET_RUN/$RECORD" ]]; then
    printf 'REFUSE: resume record already exists; acquisition is not restartable\n' >&2; exit 68
  fi
  if [[ "$MODE" == --resume-run && -f "$TARGET_RUN/run.user.stdout.log" ]]; then
    mv "$TARGET_RUN/run.user.stdout.log" "$TARGET_RUN/run.user.sandbox_refusal.log"
  fi
  if [[ "$MODE" == --resume-run && -f "$TARGET_RUN/run.user.time.log" ]]; then
    mv "$TARGET_RUN/run.user.time.log" "$TARGET_RUN/run.user.sandbox_refusal.time.log"
  fi
  printf 'ROUND146_RESUME_BINARY_VERIFIED %s\n' "$(sha256sum "$BINARY" | awk '{print $1}')"
else
  if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
    printf 'REFUSE: new target already exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2; exit 64
  fi
  for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
    free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
    if [[ "$free_kb" -lt 4194304 ]]; then printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 67; fi
  done
  manifest=$(mktemp -d /tmp/gyre-r146-rhs-family-manifest.XXXXXX)
  printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
  ( cd "$SOURCE_ROOT"; find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum ) >"$manifest/source_cfg.sha256"
  sha256sum "$PATCH" "$GATE" "$PREREG" "$SOURCE_ROOT/BLD/ppsrc/nemo/stp2d.f90" >"$manifest/toolchain.sha256"
  cd "$NEMO_ROOT"
  ./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
  while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
    < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
  while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
    < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
  cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
  patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stp2d.F90" <"$PATCH"
  if ! cmp -s "$dry/stp2d.F90" "$TARGET_ROOT/MY_SRC/stp2d.F90"; then
    printf 'REFUSE: target writer differs from syntax-proved source\n' >&2; exit 68
  fi
  touch "$TARGET_ROOT/MY_SRC/"*.F90
  ./makenemo -n "$TARGET_CFG" -m conda-scalarmath
  if [[ ! -x "$BINARY" ]] || ! check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/stp2d.f90"; then
    printf 'REFUSE: compiled writer is missing or incomplete\n' >&2; exit 68
  fi
  if nm -D "$BINARY" | grep -q '_ZGV'; then printf 'REFUSE: vector-math symbol present\n' >&2; exit 68; fi
  sha256sum "$BINARY" >"$manifest/binary.sha256"
  mkdir "$TARGET_RUN"
  for name in $PREPARED; do
    if [[ ! -f "$SOURCE_RUN/$name" ]]; then printf 'REFUSE: missing prepared input %s\n' "$name" >&2; exit 68; fi
    cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
  done
  cp "$BINARY" "$TARGET_RUN/nemo"
  cp "$manifest"/* "$TARGET_RUN/"
fi
if [[ "$MODE" != --resume-admission ]]; then
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
else
  if ! grep -Fxq 'STOP 0' "$TARGET_RUN/run.user.stdout.log"; then
    printf 'REFUSE: completed acquisition lacks STOP 0\n' >&2; exit 69
  fi
  printf 'ROUND146_EXISTING_RUN_VERIFIED\n'
fi

if [[ ! -f "$TARGET_RUN/$RECORD" ]]; then printf 'REFUSE: RHS-family record missing\n' >&2; exit 70; fi
bytes=$(stat -c %s "$TARGET_RUN/$RECORD")
if [[ "$bytes" -ne "$EXPECTED_SIZE" ]]; then printf 'REFUSE: record is %s bytes, expected %s\n' "$bytes" "$EXPECTED_SIZE" >&2; exit 70; fi
readonly INHERITED="GYRE_OMIP_L2_P3_00001080_restart.nc GYRE_OMIP_L2_P3_00001081_restart.nc oracle_process_budget_kt00001081.bin oracle_bt_step_operands_kt00001081.bin oracle_stage1_qco_operands_kt00001081.bin oracle_slow_forcing_split_kt00001081.bin"
for name in $INHERITED; do
  if ! cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; then
    printf 'REFUSE: passive instrument moved inherited %s\n' "$name" >&2; exit 71
  fi
done
printf '%s %s %s\n' "$(sha256sum "$TARGET_RUN/$RECORD" | awk '{print $1}')" "$COMMIT" "$RECORD" >"$TARGET_RUN/$RECORD.stamp"
gate() {
  "$PY" "$GATE" --root "$TARGET_RUN" --round140-root "$SOURCE_RUN" --repo "$REPO" \
    --expect-commit "$COMMIT" "$@"
}
for plant in header truncation final-ulp missing-field parent-wet-ulp parent-dry-ulp parent-interior-ulp parent-halo-ulp restart-byte; do
  if gate --plant "$plant" --output "$TARGET_RUN/round146_${plant}_plant.json" \
       >"$TARGET_RUN/round146_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 72
  fi
  if ! grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round146_${plant}_plant.log"; then
    printf 'REFUSE: %s plant lacks marker\n' "$plant" >&2; exit 72
  fi
done
gate --output "$TARGET_RUN/round146_rhs_family_validation.json"
(
  cd "$TARGET_RUN"
  sha256sum "$RECORD" "$RECORD.stamp" $INHERITED "$ROUND140_RECORD" round146_*_plant.log \
    round146_*_plant.json round146_rhs_family_validation.json >round146_outputs.sha256
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
printf 'ROUND146_DEVELOPED_RHS_FAMILIES_READY %s\n' "$TARGET_RUN"
