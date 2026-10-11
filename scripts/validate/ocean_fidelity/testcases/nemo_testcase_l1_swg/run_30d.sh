#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE_UNHANDLED: run_30d.sh line %s exited %s\n' "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

# SWG round-6 30-day acquisition. Default is read-only preflight; --run is
# operator-only. It runs the hash-pinned reference and a new additive-writer
# target on the same shipped physics deck, then admits daily entries/restarts
# only when all ordinary NEMO outputs are bit-identical.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_NAME=SWG_OMIP_L1
readonly RUN_NAME=SWG_OMIP_L1_P3D
readonly REFERENCE_MANIFEST=${REFERENCE_MANIFEST:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round0_probe/kt10/manifest.sha256}
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round6/oracle_swg_r6_30d}

do_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) do_run=1 ;;
    *) printf 'REFUSE_USAGE: expected no argument or --run\n' >&2; exit 64 ;;
  esac
  shift
done

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE: acquisition tool is outside git\n' >&2; exit 66; }
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact acquisition tool first\n' >&2; exit 66; }
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)
readonly SRC_CASE=$NEMO_ROOT/tests/SWG
readonly REF_CFG=$NEMO_ROOT/tests/$REF_NAME
readonly REF_BIN=$REF_CFG/BLD/bin/nemo.exe
readonly STPRK3=$NEMO_ROOT/src/SWE/stprk3.F90
readonly STEP_PATCH=$here/stprk3_daily_entry_record.patch
readonly MODULE=$here/swg_r1_stage_entry_record.F90
readonly DECK_PATCH=$here/namelist_cfg_30d.patch
readonly MESH_PATCH=$here/namelist_meshmask.patch
readonly CHECKER=$here/check_records.py

for f in "$SRC_CASE/cpp_SWG.fcm" "$SRC_CASE/EXPREF/namelist_cfg" "$STPRK3" \
         "$STEP_PATCH" "$MODULE" "$DECK_PATCH" "$MESH_PATCH" "$CHECKER" \
         "$REF_BIN" "$REFERENCE_MANIFEST"; do
  [[ -f "$f" ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$f" >&2; exit 67; }
done
[[ -d "$REF_CFG/EXP00" ]] || {
  printf 'REFUSE_MISSING_REFERENCE_DECK: %s\n' "$REF_CFG/EXP00" >&2; exit 67; }
expected=$(awk '$2=="nemo" {print $1}' "$REFERENCE_MANIFEST")
actual=$(sha256sum "$REF_BIN" | awk '{print $1}')
[[ -n "$expected" && "$actual" == "$expected" ]] || {
  printf 'REFUSE_REFERENCE_BINARY: sha256 %s, expected %s\n' "$actual" "$expected" >&2; exit 68; }
keys=$(sed -E 's/.*fppkeys *//' "$SRC_CASE/cpp_SWG.fcm" | xargs)
[[ "$keys" == 'key_xios key_qco key_vco_3d key_RK3' ]] || {
  printf 'REFUSE_SHIPPED_KEYS: found "%s"\n' "$keys" >&2; exit 68; }

dry=$(mktemp -d)
cp "$STPRK3" "$SRC_CASE/EXPREF/namelist_cfg" "$dry/"
patch -s --fuzz=0 "$dry/stprk3.F90" <"$STEP_PATCH"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$DECK_PATCH"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$MESH_PATCH"
[[ "$(grep -c '^-[^-]' "$STEP_PATCH" || true)" -eq 0 ]] || {
  printf 'REFUSE_NONADDITIVE_INSTRUMENT: stprk3 patch removes a line\n' >&2; exit 68; }
[[ "$(grep -c 'CALL swg_open' "$dry/stprk3.F90")" -eq 1 ]] || {
  printf 'REFUSE_DAILY_WRITER_COUNT: patched stepper lacks one entry writer\n' >&2; exit 68; }
grep -q 'nn_itend    =      720' "$dry/namelist_cfg" || {
  printf 'REFUSE_30D_LENGTH: patched deck is not 720 one-hour steps\n' >&2; exit 68; }
grep -q 'nn_stock    =      24' "$dry/namelist_cfg" || {
  printf 'REFUSE_DAILY_RESTART: patched deck lacks daily restarts\n' >&2; exit 68; }
printf 'PREFLIGHT: reference sha256 %s; additive daily writer; commit %s\n' "$actual" "$COMMIT"

[[ ! -e "$NEMO_ROOT/tests/$RUN_NAME" ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s\n' "$NEMO_ROOT/tests/$RUN_NAME" >&2; exit 69; }
[[ ! -e "$EVIDENCE" ]] || {
  printf 'REFUSE_EVIDENCE_EXISTS: %s\n' "$EVIDENCE" >&2; exit 69; }
if [[ "$do_run" -ne 1 ]]; then
  printf 'WOULD_BUILD: makenemo -a SWG -n %s -m conda-scalarmath del_key key_xios\n' "$RUN_NAME"
  printf 'WOULD_RUN: reference and instrumented binaries for 720 shipped one-hour steps; admit daily entries and restarts\n'
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/reference" "$EVIDENCE/instrumented"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
sha256sum "$STEP_PATCH" "$MODULE" "$DECK_PATCH" "$MESH_PATCH" "$CHECKER" \
  >"$EVIDENCE/manifest/tool.sha256"

cp -rL "$REF_CFG/EXP00/." "$EVIDENCE/reference/"
cp "$REF_BIN" "$EVIDENCE/reference/nemo"
patch "$EVIDENCE/reference/namelist_cfg" <"$DECK_PATCH"
patch "$EVIDENCE/reference/namelist_cfg" <"$MESH_PATCH"
(
  cd "$EVIDENCE/reference"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1
)

cd "$NEMO_ROOT"
./makenemo -a SWG -n "$RUN_NAME" -m conda-scalarmath del_key key_xios
cfg=$NEMO_ROOT/tests/$RUN_NAME
cp -r "$SRC_CASE/EXPREF/." "$cfg/EXP00/"
cp -r "$SRC_CASE/MY_SRC/." "$cfg/MY_SRC/"
cp -f "$STPRK3" "$MODULE" "$cfg/MY_SRC/"
patch "$cfg/EXP00/namelist_cfg" <"$DECK_PATCH"
patch "$cfg/EXP00/namelist_cfg" <"$MESH_PATCH"
patch "$cfg/MY_SRC/stprk3.F90" <"$STEP_PATCH"
touch "$cfg/MY_SRC/"*.F90
./makenemo -n "$RUN_NAME" -m conda-scalarmath
[[ -x "$cfg/BLD/bin/nemo.exe" ]] || {
  printf 'REFUSE_BUILD_BINARY: instrumented executable missing\n' >&2; exit 70; }
grep -q 'MOD(kstp - nit000, 24)' "$cfg/BLD/ppsrc/nemo/stprk3.f90" || {
  printf 'REFUSE_COMPILED_WRITER: compiled stepper lacks daily writer\n' >&2; exit 70; }
cp -rL "$cfg/EXP00/." "$EVIDENCE/instrumented/"
cp "$cfg/BLD/bin/nemo.exe" "$EVIDENCE/instrumented/nemo"
(
  cd "$EVIDENCE/instrumented"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1
)

record_kts=$(seq -s, 1 24 697)
restart_kts=$(seq -s, 24 24 720)
python3 -I "$CHECKER" --evidence "$EVIDENCE/instrumented" \
  --reference "$EVIDENCE/reference" \
  --record-kts "$record_kts" --stages 1 --restart-kts "$restart_kts" \
  --json "$EVIDENCE/swg_round6_30d_admission.json"
printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
