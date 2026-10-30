#!/usr/bin/env bash
set -Eeuo pipefail

do_run=0
math_flavour=vector
while [[ $# -gt 0 ]]; do
  case $1 in
    --run) do_run=1 ;;
    --math-flavour)
      [[ $# -ge 2 ]] || { printf 'REFUSE_USAGE: --math-flavour needs a value\n' >&2; exit 64; }
      math_flavour=$2
      shift
      ;;
    *) printf 'REFUSE_USAGE: expected [--math-flavour vector|scalar] [--run]\n' >&2; exit 64 ;;
  esac
  shift
done
case $math_flavour in
  vector)
    readonly WRITER_NAME=GYRE_OMIP_L2_P3_DAILY
    readonly DEFAULT_ROUND=round6/oracle_gyre_r4_smag_daily_stage
    ;;
  scalar)
    readonly WRITER_NAME=GYRE_OMIP_L2_P3_SMW_DAILY
    readonly DEFAULT_ROUND=round7/oracle_gyre_r4_smag_daily_stage_scalar
    ;;
  *) printf 'REFUSE_MATH_FLAVOUR: expected vector or scalar\n' >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$here/deck.sh"
gyre_hires_select_deck r4_smag "$here"
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=$NEMO_ROOT/cfgs/$WRITER_NAME
readonly WRITER=$SOURCE_CFG/BLD/bin/nemo.exe
readonly COMPILED_STEPPER=$SOURCE_CFG/BLD/ppsrc/nemo/stprk3.f90
readonly BASE_EXP00=$NEMO_ROOT/cfgs/GYRE_OMIP_L2_P3/EXP00
readonly DURATION_PATCH=$here/namelist_cfg_r4_smag_30d.patch
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/$DEFAULT_ROUND}
readonly RANKS=${RANKS:-1}
[[ $RANKS == 1 ]] || {
  printf 'REFUSE_RANKS: stage dumps are lwp-only and ungathered; RANKS must be 1\n' >&2
  exit 64
}

repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE\n' >&2; exit 66; }
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact acquisition tool first\n' >&2; exit 66; }
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)
for path in "$BASE_DECK" "$DECK_PATCH" "$DURATION_PATCH" "$WRITER" \
  "$COMPILED_STEPPER" "$BASE_EXP00/namelist_ref"; do
  [[ -f $path ]] || {
    printf 'REFUSE_MISSING_INPUT: %s\n' "$path" >&2; exit 67; }
done
grep -q 'MOD( kstp - nit000, 24 ) == 0' "$COMPILED_STEPPER" || {
  printf 'REFUSE_COMPILED_WRITER: daily cadence is absent\n' >&2; exit 68; }
grep -q 'LANE1_STEP_ENTRY_DUMP' "$COMPILED_STEPPER" || {
  printf 'REFUSE_COMPILED_WRITER: entry writer is absent\n' >&2; exit 68; }

dry=$(mktemp -d)
trap 'rm -rf -- "$dry"' EXIT
cp "$BASE_DECK" "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$DECK_PATCH"
gyre_hires_verify_deck r4_smag "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$DURATION_PATCH"
[[ $(gyre_hires_namelist_value "$dry/namelist_cfg" nn_it000) == 1 \
   && $(gyre_hires_namelist_value "$dry/namelist_cfg" nn_itend) == 721 \
   && $(gyre_hires_namelist_value "$dry/namelist_cfg" nn_stock) == 721 \
   && $(gyre_hires_namelist_value "$dry/namelist_cfg" nn_write) == 24 ]] || {
  printf 'REFUSE_DAILY_CLOCK: expected kt=1..721 and 24-step output\n' >&2; exit 68; }
writer_hash=$(sha256sum "$WRITER" | awk '{print $1}')
if [[ $math_flavour == scalar ]] && nm -D "$WRITER" | grep -q '_ZGV'; then
  printf 'REFUSE_COMPILED_MATH: vector-math symbol in scalar writer\n' >&2; exit 68
fi
printf 'PREFLIGHT: deck=r4_smag math_flavour=%s cadence=kt1,25,...,721 ranks=1 evidence=%s writer_sha256=%s commit=%s\n' \
  "$math_flavour" "$EVIDENCE" "$writer_hash" "$COMMIT"
if [[ $do_run -ne 1 ]]; then
  printf 'WOULD_RUN: EVIDENCE=%s RANKS=1 %s --math-flavour %s --run\nPREFLIGHT_OK\n' \
    "$EVIDENCE" "$0" "$math_flavour"
  exit 0
fi

[[ ! -e $EVIDENCE ]] || {
  printf 'REFUSE_EVIDENCE_EXISTS: %s\n' "$EVIDENCE" >&2; exit 69; }
mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/run"
cp "$dry/namelist_cfg" "$BASE_EXP00/namelist_ref" "$EVIDENCE/run/"
for xml in "$BASE_EXP00"/*.xml; do
  [[ ! -e $xml ]] || cp "$xml" "$EVIDENCE/run/"
done
cp "$WRITER" "$EVIDENCE/run/nemo"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
printf '%s\n' "$math_flavour" >"$EVIDENCE/manifest/math_flavour"
gyre_hires_write_resolved_namelist "$EVIDENCE/run/namelist_cfg" \
  "$EVIDENCE/manifest/resolved_namelist_lines.txt"
sha256sum "$EVIDENCE/run/nemo" "$EVIDENCE/run/namelist_cfg" \
  "$EVIDENCE/run/namelist_ref" >"$EVIDENCE/manifest/run.sha256"
(
  cd "$EVIDENCE/run"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  mpirun -np 1 ./nemo >run.user.log 2>&1
)
expected=0
for day in $(seq 0 30); do
  kt=$((1 + 24 * day))
  record=$(printf '%s/run/oracle_step_entry_kt%08d.bin' "$EVIDENCE" "$kt")
  [[ -s $record ]] || {
    printf 'REFUSE_MISSING_RECORD: %s\n' "$record" >&2; exit 70; }
  expected=$((expected + 1))
done
actual=$(find "$EVIDENCE/run" -maxdepth 1 -type f \
  -name 'oracle_step_entry_kt*.bin' | wc -l)
[[ $actual -eq $expected ]] || {
  printf 'REFUSE_RECORD_COUNT: expected %s, found %s\n' "$expected" "$actual" >&2; exit 70; }
(
  cd "$EVIDENCE/run"
  sha256sum oracle_step_entry_kt*.bin \
    >../manifest/oracle_records.sha256
)
[[ $(wc -l <"$EVIDENCE/manifest/oracle_records.sha256") -eq $expected ]] || {
  printf 'REFUSE_RECORD_MANIFEST: expected %s hashes\n' "$expected" >&2; exit 70; }
printf 'ACQUISITION_DONE: %s (r4_smag day boundaries 0..30, kt=1,25,...,721)\n' \
  "$EVIDENCE"
