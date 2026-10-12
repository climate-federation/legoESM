#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE_UNHANDLED: run_30d.sh line %s exited %s\n' "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

# SWG 30-day acquisition. Default is read-only preflight; --run is operator-only.
# --deck selects a committed namelist patch and its source case.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_NAME=SWG_OMIP_L1
readonly REFERENCE_MANIFEST=${REFERENCE_MANIFEST:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round0_probe/kt10/manifest.sha256}

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
swg_select_deck month "$deck" "$here"
readonly SOURCE_CASE_NAME RUN_NAME DECK_PATCH DEFAULT_EVIDENCE SOURCE_PATCH
readonly SOURCE_PATCH_SHA256
readonly EVIDENCE=${EVIDENCE:-$DEFAULT_EVIDENCE}
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE: acquisition tool is outside git\n' >&2; exit 66; }
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact acquisition tool first\n' >&2; exit 66; }
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)
readonly SRC_CASE=$NEMO_ROOT/tests/$SOURCE_CASE_NAME
readonly BASE_DECK=$NEMO_ROOT/tests/SWG/EXPREF
readonly REF_CFG=$NEMO_ROOT/tests/$REF_NAME
if [[ "$deck" == shipped ]]; then
  readonly REF_BIN=$REF_CFG/BLD/bin/nemo.exe
  readonly REFERENCE_DECK=$REF_CFG/EXP00
else
  readonly REF_BIN=$SRC_CASE/BLD/bin/nemo.exe
  readonly REFERENCE_DECK=$BASE_DECK
fi
readonly STPRK3=$NEMO_ROOT/src/SWE/stprk3.F90
readonly STEP_PATCH=$here/stprk3_daily_entry_record.patch
readonly MODULE=$here/swg_r1_stage_entry_record.F90
readonly MESH_PATCH=$here/namelist_meshmask.patch
readonly CHECKER=$here/check_records.py

for f in "$BASE_DECK/namelist_cfg" "$STPRK3" \
         "$STEP_PATCH" "$MODULE" "$DECK_PATCH" "$MESH_PATCH" "$CHECKER" \
         "$REF_BIN" "$here/deck.sh"; do
  [[ -f "$f" ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$f" >&2; exit 67; }
done
[[ -d "$REFERENCE_DECK" ]] || {
  printf 'REFUSE_MISSING_REFERENCE_DECK: %s\n' "$REFERENCE_DECK" >&2; exit 67; }
actual=$(sha256sum "$REF_BIN" | awk '{print $1}')
if [[ "$deck" == shipped ]]; then
  [[ -f "$REFERENCE_MANIFEST" ]] || {
    printf 'REFUSE_MISSING_INPUT: %s\n' "$REFERENCE_MANIFEST" >&2; exit 67; }
  expected=$(awk '$2=="nemo" {print $1}' "$REFERENCE_MANIFEST")
  [[ -n "$expected" && "$actual" == "$expected" ]] || {
    printf 'REFUSE_REFERENCE_BINARY: sha256 %s, expected %s\n' "$actual" "$expected" >&2
    exit 68
  }
fi

dry=$(mktemp -d)
swg_verify_source_case "$NEMO_ROOT" "$deck" "$dry"
cp "$STPRK3" "$BASE_DECK/namelist_cfg" "$dry/"
patch -s --fuzz=0 "$dry/stprk3.F90" <"$STEP_PATCH"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$DECK_PATCH"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$MESH_PATCH"
[[ "$(grep -c '^-[^-]' "$STEP_PATCH" || true)" -eq 0 ]] || {
  printf 'REFUSE_NONADDITIVE_INSTRUMENT: stprk3 patch removes a line\n' >&2; exit 68; }
[[ "$(grep -c 'CALL swg_open' "$dry/stprk3.F90")" -eq 1 ]] || {
  printf 'REFUSE_DAILY_WRITER_COUNT: patched stepper lacks one entry writer\n' >&2; exit 68; }
swg_require_daily_clock "$dry/namelist_cfg"
printf 'PREFLIGHT: deck %s source %s reference sha256 %s; %s steps/day; additive daily writer; commit %s\n' \
  "$deck" "$SOURCE_CASE_NAME" "$actual" "$STEPS_PER_DAY" "$COMMIT"

if [[ "$do_run" -ne 1 ]]; then
  printf 'WOULD_BUILD: makenemo -a %s -n %s -m conda-scalarmath\n' \
    "$SOURCE_CASE_NAME" "$RUN_NAME"
  printf 'WOULD_RUN: reference and instrumented binaries for %s steps; admit %s daily entries and restarts\n' \
    "$((30 * STEPS_PER_DAY))" 30
  printf 'PREFLIGHT_OK\n'
  exit 0
fi
[[ ! -e "$NEMO_ROOT/tests/$RUN_NAME" ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s\n' "$NEMO_ROOT/tests/$RUN_NAME" >&2; exit 69; }
[[ ! -e "$EVIDENCE" ]] || {
  printf 'REFUSE_EVIDENCE_EXISTS: %s\n' "$EVIDENCE" >&2; exit 69; }

mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/reference" "$EVIDENCE/instrumented"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
sha256sum "$STEP_PATCH" "$MODULE" "$DECK_PATCH" "$MESH_PATCH" "$CHECKER" \
  "$here/deck.sh" "$SOURCE_PATCH" \
  >"$EVIDENCE/manifest/tool.sha256"

cp -rL "$REFERENCE_DECK/." "$EVIDENCE/reference/"
cp "$REF_BIN" "$EVIDENCE/reference/nemo"
patch "$EVIDENCE/reference/namelist_cfg" <"$DECK_PATCH"
patch "$EVIDENCE/reference/namelist_cfg" <"$MESH_PATCH"
(
  cd "$EVIDENCE/reference"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1
)

cd "$NEMO_ROOT"
if [[ "$deck" == shipped ]]; then
  ./makenemo -a "$SOURCE_CASE_NAME" -n "$RUN_NAME" -m conda-scalarmath del_key key_xios
else
  ./makenemo -a "$SOURCE_CASE_NAME" -n "$RUN_NAME" -m conda-scalarmath
fi
cfg=$NEMO_ROOT/tests/$RUN_NAME
cp -r "$BASE_DECK/." "$cfg/EXP00/"
cp -r "$SRC_CASE/MY_SRC/." "$cfg/MY_SRC/"
cp -f "$STPRK3" "$MODULE" "$cfg/MY_SRC/"
patch "$cfg/EXP00/namelist_cfg" <"$DECK_PATCH"
patch "$cfg/EXP00/namelist_cfg" <"$MESH_PATCH"
patch "$cfg/MY_SRC/stprk3.F90" <"$STEP_PATCH"
touch "$cfg/MY_SRC/"*.F90
./makenemo -n "$RUN_NAME" -m conda-scalarmath
[[ -x "$cfg/BLD/bin/nemo.exe" ]] || {
  printf 'REFUSE_BUILD_BINARY: instrumented executable missing\n' >&2; exit 70; }
grep -q 'MOD(kstp - nit000, NINT(rday / rn_Dt))' \
  "$cfg/BLD/ppsrc/nemo/stprk3.f90" || {
  printf 'REFUSE_COMPILED_WRITER: compiled stepper lacks daily writer\n' >&2; exit 70; }
cp -rL "$cfg/EXP00/." "$EVIDENCE/instrumented/"
cp "$cfg/BLD/bin/nemo.exe" "$EVIDENCE/instrumented/nemo"
swg_write_resolved_namelist "$EVIDENCE/reference/namelist_cfg" \
  "$EVIDENCE/manifest/reference_resolved_namelist_lines.txt"
swg_write_resolved_namelist "$EVIDENCE/instrumented/namelist_cfg" \
  "$EVIDENCE/manifest/instrumented_resolved_namelist_lines.txt"
sha256sum "$EVIDENCE/reference/nemo" "$EVIDENCE/reference/namelist_cfg" \
  "$EVIDENCE/instrumented/nemo" "$EVIDENCE/instrumented/namelist_cfg" \
  >"$EVIDENCE/manifest/run.sha256"
(
  cd "$EVIDENCE/instrumented"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1
)
if [[ "$deck" == maddison ]]; then
  [[ -s "$EVIDENCE/reference/run.stat" \
        && -s "$EVIDENCE/instrumented/run.stat" ]] || {
    printf 'REFUSE_MISSING_RUNSTAT: Maddison reference/instrumented run.stat\n' >&2
    exit 70
  }
fi

last_entry=$((29 * STEPS_PER_DAY + 1))
last_restart=$((30 * STEPS_PER_DAY))
record_kts=$(seq -s, 1 "$STEPS_PER_DAY" "$last_entry")
restart_kts=$(seq -s, "$STEPS_PER_DAY" "$STEPS_PER_DAY" "$last_restart")
if [[ "$deck" == shipped ]]; then
  admission=$EVIDENCE/swg_round6_30d_admission.json
else
  admission=$EVIDENCE/swg_dg_round1_30d_admission.json
fi
python3 -I "$CHECKER" --evidence "$EVIDENCE/instrumented" \
  --reference "$EVIDENCE/reference" \
  --record-kts "$record_kts" --stages 1 --restart-kts "$restart_kts" \
  --json "$admission"
printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
