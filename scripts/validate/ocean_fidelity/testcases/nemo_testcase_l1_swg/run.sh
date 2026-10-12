#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE_UNHANDLED: run.sh line %s exited %s\n' "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

# SWG stage-entry acquisition. Default is read-only preflight; --run is for the
# operator. --deck selects a committed namelist patch and its source case.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_NAME=SWG_OMIP_L1
readonly REFERENCE=${REFERENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round0_probe/kt10}

do_run=0
deck=shipped
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) do_run=1 ;;
    --deck)
      [[ $# -ge 2 ]] || { printf 'REFUSE_USAGE: --deck needs a value\n' >&2; exit 64; }
      deck=$2
      shift
      ;;
    *) printf 'REFUSE_USAGE: expected [--deck shipped|maddison] [--run]\n' >&2; exit 64 ;;
  esac
  shift
done

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$here/deck.sh"
swg_select_deck stage "$deck" "$here"
readonly MAKENEMO_SOURCE_CASE SOURCE_CASE_NAME RUN_NAME DECK_PATCH DEFAULT_EVIDENCE SOURCE_PATCH
readonly SOURCE_PATCH_SHA256
readonly EVIDENCE=${EVIDENCE:-$DEFAULT_EVIDENCE}
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE: acquisition tool is outside git\n' >&2; exit 66; }
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact acquisition tool first\n' >&2; exit 66; }
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)
readonly SRC_CASE=$NEMO_ROOT/tests/$SOURCE_CASE_NAME
readonly BASE_DECK=$NEMO_ROOT/tests/SWG/EXPREF
readonly REF_BIN=$NEMO_ROOT/tests/$REF_NAME/BLD/bin/nemo.exe
readonly STPRK3=$NEMO_ROOT/src/SWE/stprk3.F90
readonly STEP_PATCH=$here/stprk3_stage_entry_record.patch
readonly MODULE=$here/swg_r1_stage_entry_record.F90
readonly MESH_PATCH=$here/namelist_meshmask.patch
readonly CHECKER=$here/check_records.py

for f in "$BASE_DECK/namelist_cfg" "$STPRK3" "$STEP_PATCH" "$MODULE" \
         "$DECK_PATCH" "$MESH_PATCH" "$CHECKER" "$here/deck.sh"; do
  [[ -f "$f" ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$f" >&2; exit 67; }
done
if [[ "$deck" == shipped ]]; then
  for f in "$REF_BIN" "$REFERENCE/manifest.sha256"; do
    [[ -f "$f" ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$f" >&2; exit 67; }
  done
else
  [[ -x "$SRC_CASE/BLD/bin/nemo.exe" ]] || {
    printf 'REFUSE_MISSING_REFERENCE_BINARY: %s\n' "$SRC_CASE/BLD/bin/nemo.exe" >&2
    exit 67
  }
fi

dry=$(mktemp -d)
swg_verify_makenemo_source_case "$NEMO_ROOT" "$MAKENEMO_SOURCE_CASE"
swg_verify_source_case "$NEMO_ROOT" "$deck" "$dry"
cp "$STPRK3" "$BASE_DECK/namelist_cfg" "$dry/"
patch -s --fuzz=0 "$dry/stprk3.F90" <"$STEP_PATCH"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$DECK_PATCH"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$MESH_PATCH"
[[ "$(grep -c '^-[^-]' "$STEP_PATCH" || true)" -eq 0 ]] || {
  printf 'REFUSE_NONADDITIVE_INSTRUMENT: stprk3 patch removes a line\n' >&2; exit 68; }
[[ "$(grep -c 'CALL swg_open' "$dry/stprk3.F90")" -eq 3 ]] || {
  printf 'REFUSE_STAGE_COUNT: patched stepper does not have three entries\n' >&2; exit 68; }
[[ "$(swg_namelist_value "$dry/namelist_cfg" nn_itend)" == 10 \
      && "$(swg_namelist_value "$dry/namelist_cfg" nn_stock)" == 1 ]] || {
  printf 'REFUSE_STAGE_CLOCK: ten-step deck does not resolve 10/1\n' >&2; exit 68; }
if [[ "$deck" == shipped ]]; then
  expected=$(awk '$2=="nemo" {print $1}' "$REFERENCE/manifest.sha256")
  actual=$(sha256sum "$REF_BIN" | awk '{print $1}')
  [[ -n "$expected" && "$actual" == "$expected" ]] || {
    printf 'REFUSE_REFERENCE_BINARY: sha256 %s, expected %s\n' "$actual" "$expected" >&2
    exit 68
  }
else
  actual=$(sha256sum "$SRC_CASE/BLD/bin/nemo.exe" | awk '{print $1}')
fi
printf 'PREFLIGHT: deck %s source %s reference sha256 %s; additive patch; commit %s\n' \
  "$deck" "$SOURCE_CASE_NAME" "$actual" "$COMMIT"

if [[ "$do_run" -ne 1 ]]; then
  printf 'WOULD_BUILD: makenemo -a %s -n %s -m conda-scalarmath del_key key_xios\n' \
    "$MAKENEMO_SOURCE_CASE" "$RUN_NAME"
  printf 'WOULD_RUN: deck=%s instrumented binary; bit-compare every grid T/U/V/W field\n' "$deck"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi
[[ ! -e "$NEMO_ROOT/tests/$RUN_NAME" ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s\n' "$NEMO_ROOT/tests/$RUN_NAME" >&2; exit 69; }
[[ ! -e "$EVIDENCE" ]] || {
  printf 'REFUSE_EVIDENCE_EXISTS: %s\n' "$EVIDENCE" >&2; exit 69; }

if [[ "$deck" == shipped ]]; then
  instrumented_dir=$EVIDENCE/run
  reference_dir=$REFERENCE
  admission=$EVIDENCE/swg_round1_admission.json
else
  instrumented_dir=$EVIDENCE/instrumented
  reference_dir=$EVIDENCE/reference
  admission=$EVIDENCE/swg_dg_round1_stage_admission.json
fi
mkdir -p "$EVIDENCE/manifest" "$instrumented_dir"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
sha256sum "$STEP_PATCH" "$MODULE" "$DECK_PATCH" "$MESH_PATCH" "$CHECKER" \
  "$here/deck.sh" "$SOURCE_PATCH" \
  >"$EVIDENCE/manifest/tool.sha256"
if [[ "$deck" == maddison ]]; then
  mkdir -p "$reference_dir"
  cp -rL "$BASE_DECK/." "$reference_dir/"
  patch "$reference_dir/namelist_cfg" <"$DECK_PATCH"
  patch "$reference_dir/namelist_cfg" <"$MESH_PATCH"
  cp "$SRC_CASE/BLD/bin/nemo.exe" "$reference_dir/nemo"
  (
    cd "$reference_dir"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1
  )
fi
cd "$NEMO_ROOT"
cfg=$NEMO_ROOT/tests/$RUN_NAME
./makenemo -a "$MAKENEMO_SOURCE_CASE" -n "$RUN_NAME" -m conda-scalarmath del_key key_xios
swg_verify_target_cpp_keys "$cfg" "$RUN_NAME"
cp -r "$BASE_DECK/." "$cfg/EXP00/"
cp -r "$SRC_CASE/MY_SRC/." "$cfg/MY_SRC/"
cp -f "$STPRK3" "$MODULE" "$cfg/MY_SRC/"
patch "$cfg/EXP00/namelist_cfg" <"$DECK_PATCH"
patch "$cfg/EXP00/namelist_cfg" <"$MESH_PATCH"
patch "$cfg/MY_SRC/stprk3.F90" <"$STEP_PATCH"
touch "$cfg/MY_SRC/"*.F90
./makenemo -n "$RUN_NAME" -m conda-scalarmath
swg_verify_target_cpp_keys "$cfg" "$RUN_NAME"
[[ -x "$cfg/BLD/bin/nemo.exe" ]] || {
  printf 'REFUSE_BUILD_BINARY: instrumented executable missing\n' >&2; exit 70; }
grep -q 'CALL swg_open' "$cfg/BLD/ppsrc/nemo/stprk3.f90" || {
  printf 'REFUSE_COMPILED_WRITER: compiled stepper lacks writer\n' >&2; exit 70; }
cp -L "$cfg/EXP00/namelist_cfg" "$cfg/EXP00/namelist_ref" "$instrumented_dir/"
for xml in "$cfg"/EXP00/*.xml; do
  [[ ! -e "$xml" ]] || cp -L "$xml" "$instrumented_dir/"
done
cp "$cfg/BLD/bin/nemo.exe" "$instrumented_dir/nemo"
for f in "$reference_dir/namelist_cfg" "$instrumented_dir/namelist_cfg" \
         "$reference_dir/nemo" "$instrumented_dir/nemo"; do
  [[ -f "$f" ]] || { printf 'REFUSE_RECEIPT_INPUT: %s\n' "$f" >&2; exit 70; }
done
swg_write_resolved_namelist "$reference_dir/namelist_cfg" \
  "$EVIDENCE/manifest/reference_resolved_namelist_lines.txt"
swg_write_resolved_namelist "$instrumented_dir/namelist_cfg" \
  "$EVIDENCE/manifest/instrumented_resolved_namelist_lines.txt"
sha256sum "$reference_dir/nemo" "$reference_dir/namelist_cfg" \
  "$instrumented_dir/nemo" "$instrumented_dir/namelist_cfg" \
  >"$EVIDENCE/manifest/run.sha256"
(
  cd "$instrumented_dir"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1
)
if [[ "$deck" == maddison ]]; then
  [[ -s "$reference_dir/run.stat" && -s "$instrumented_dir/run.stat" ]] || {
    printf 'REFUSE_MISSING_RUNSTAT: Maddison reference/instrumented run.stat\n' >&2
    exit 70
  }
fi
python3 -I "$CHECKER" --evidence "$instrumented_dir" --reference "$reference_dir" \
  --json "$admission"
printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
