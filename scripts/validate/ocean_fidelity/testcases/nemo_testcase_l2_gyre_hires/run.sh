#!/usr/bin/env bash
set -Eeuo pipefail

# GYRE stage-entry acquisition.  The default is read-only preflight.  Only the
# operator passes --run; this tool deliberately reuses the already-built P3
# writer and never invokes makenemo or edits a NEMO source tree.
deck=shipped
do_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deck)
      [[ $# -ge 2 ]] || { printf 'REFUSE_USAGE: --deck needs a value\n' >&2; exit 64; }
      deck=$2
      shift
      ;;
    --run) do_run=1 ;;
    *) printf 'REFUSE_USAGE: expected [--deck shipped|r4|r4_smag] [--run]\n' >&2; exit 64 ;;
  esac
  shift
done

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$here/deck.sh"
gyre_hires_select_deck "$deck" "$here"
readonly DECK_TAG RUN_NAME DECK_PATCH DEFAULT_EVIDENCE BASE_DECK
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=$NEMO_ROOT/cfgs/GYRE_OMIP_L2_P3
readonly WRITER=$SOURCE_CFG/BLD/bin/nemo.exe
readonly COMPILED_STEPPER=$SOURCE_CFG/BLD/ppsrc/nemo/stprk3.f90
readonly EXP00=$SOURCE_CFG/EXP00
readonly EVIDENCE=${EVIDENCE:-$DEFAULT_EVIDENCE}
readonly RANKS=${RANKS:-8}

repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE: acquisition tool is outside git\n' >&2; exit 66; }
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact acquisition tool first\n' >&2; exit 66; }
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)
for path in "$BASE_DECK" "$WRITER" "$COMPILED_STEPPER" "$EXP00/namelist_ref"; do
  [[ -f "$path" ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$path" >&2; exit 67; }
done
[[ -z "$DECK_PATCH" || -f "$DECK_PATCH" ]] || {
  printf 'REFUSE_MISSING_INPUT: %s\n' "$DECK_PATCH" >&2; exit 67; }
[[ -x "$WRITER" && "$WRITER" -nt "$COMPILED_STEPPER" ]] || {
  printf 'REFUSE_STALE_WRITER: %s is absent or older than %s\n' "$WRITER" "$COMPILED_STEPPER" >&2
  exit 68
}
[[ "$(sha256sum "$WRITER" | awk '{print $1}')" == \
    974dc326b4f5237a0e8dfe50ba0ae8a4573507d57429fb82069ff3c20abfad41 ]] || {
  printf 'REFUSE_WRITER_IDENTITY: P3 writer hash changed\n' >&2; exit 68; }
grep -q 'LANE1_STEP_ENTRY_DUMP' "$COMPILED_STEPPER" || {
  printf 'REFUSE_WRITER_PROGRAM: compiled stepper lacks the entry writer\n' >&2; exit 68; }

dry=$(mktemp -d)
trap 'rm -rf -- "$dry"' EXIT
cp "$BASE_DECK" "$dry/namelist_cfg"
if [[ -n "$DECK_PATCH" ]]; then
  patch -s --fuzz=0 "$dry/namelist_cfg" <"$DECK_PATCH"
fi
gyre_hires_verify_deck "$deck" "$dry/namelist_cfg"
printf 'PREFLIGHT: deck=%s target=%s evidence=%s writer_sha256=%s commit=%s\n' \
  "$deck" "$RUN_NAME" "$EVIDENCE" \
  974dc326b4f5237a0e8dfe50ba0ae8a4573507d57429fb82069ff3c20abfad41 "$COMMIT"
if [[ "$do_run" -ne 1 ]]; then
  printf 'WOULD_RUN: EVIDENCE=%s RANKS=%s %s --deck %s --run\n' \
    "$EVIDENCE" "$RANKS" "$0" "$deck"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

[[ ! -e "$EVIDENCE" ]] || {
  printf 'REFUSE_EVIDENCE_EXISTS: %s\n' "$EVIDENCE" >&2; exit 69; }
mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/run"
cp "$dry/namelist_cfg" "$EXP00/namelist_ref" "$EVIDENCE/run/"
for xml in "$EXP00"/*.xml; do
  [[ ! -e "$xml" ]] || cp "$xml" "$EVIDENCE/run/"
done
cp "$WRITER" "$EVIDENCE/run/nemo"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
gyre_hires_write_resolved_namelist \
  "$EVIDENCE/run/namelist_cfg" "$EVIDENCE/manifest/resolved_namelist_lines.txt"
sha256sum "$EVIDENCE/run/nemo" "$EVIDENCE/run/namelist_cfg" \
  "$EVIDENCE/run/namelist_ref" >"$EVIDENCE/manifest/run.sha256"
(
  cd "$EVIDENCE/run"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  mpirun -np "$RANKS" --oversubscribe ./nemo >run.user.log 2>&1
)
for kt in $(seq 1 10); do
  record=$(printf '%s/run/oracle_step_entry_kt%08d.bin' "$EVIDENCE" "$kt")
  [[ -s "$record" ]] || { printf 'REFUSE_MISSING_RECORD: %s\n' "$record" >&2; exit 70; }
done
printf 'ACQUISITION_DONE: %s (%s, kt=1..10)\n' "$EVIDENCE" "$DECK_TAG"
