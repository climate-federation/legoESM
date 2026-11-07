#!/usr/bin/env bash
set -Eeuo pipefail

# Non-vacuity control for the ORCA2 write-only passivity gate.
#
# The gate in run.sh was corrected to skip seven streams the current source card
# can no longer emit, and to stop the stage-1 transport comparison before that
# stream's uninitialised third field (zFw).  Both are narrowings, so the gate
# has to be shown to still refuse when a byte that DOES carry signal moves.
#
# Each plant corrupts exactly one thing in the candidate run, runs --finalize,
# and requires the documented outcome.  Every plant restores the original bytes
# and is checked back to its recorded SHA-256, and the run is re-finalized at
# the end so its admission artifacts are left in the admitted state.
#
# The backup of the file under plant is deliberately NOT kept in the temporary
# work directory the cleanup trap removes: an interrupt would then delete the
# only copy of the original bytes and leave the certified run corrupted.  It is
# kept in a directory beside the run instead, the original is only corrupted
# after the backup and its digest are both on disk, and the first thing this
# script does is put back anything a previous interrupted attempt left pending.
# Restoring is therefore idempotent -- an interrupted plant is repaired by
# running this script again.

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly GATE=$here/run.sh
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/parallel/orca2/oracle_phase2v_tke_boundary_np2
readonly PARTIAL_STREAM=oracle_transport_kt00000001_s1.bin
readonly RECORD=oracle_tke_boundary_kt00000002.bin
readonly FIELD_BYTES=$((94 * 152 * 31 * 8))
readonly HEADER_BYTES=$((16 + 8 * 4))
readonly ZFU_OFFSET=$((HEADER_BYTES + 1000))
readonly ZFV_OFFSET=$((HEADER_BYTES + FIELD_BYTES + 1000))
readonly ZFW_OFFSET=$((HEADER_BYTES + 2 * FIELD_BYTES + 1000))
readonly STREAM_BYTES=$((HEADER_BYTES + 3 * FIELD_BYTES))
readonly BACKUP_DIR=${TARGET_RUN%/}.passivity-plant-backup

failures=0
mkdir -p "$BACKUP_DIR"

# Put back every backup found pending, verify it against the digest recorded
# when it was taken, and only then drop the backup.  Safe to call when nothing
# is pending, and safe to call twice.
restore_pending() {
  local backup name digest restored
  shopt -s nullglob
  for backup in "$BACKUP_DIR"/*.orig; do
    name=$(basename "$backup" .orig)
    if [[ ! -f "$BACKUP_DIR/$name.sha256" ]]; then
      printf 'PLANT FAIL restore: %s has no recorded digest; left in place\n' \
        "$name"
      failures=$((failures + 1))
      continue
    fi
    digest=$(cat "$BACKUP_DIR/$name.sha256")
    cp -f "$backup" "$TARGET_RUN/$name"
    restored=$(sha256sum "$TARGET_RUN/$name" | awk '{print $1}')
    if [[ "$restored" != "$digest" ]]; then
      printf 'PLANT FAIL restore: could not restore %s to %s\n' "$name" "$digest"
      failures=$((failures + 1))
      continue
    fi
    rm -f -- "$backup" "$BACKUP_DIR/$name.sha256"
  done
  shopt -u nullglob
}

restore_pending

work=$(mktemp -d /tmp/orca2-passivity-plant.XXXXXXXX)
trap 'restore_pending || true; rm -rf -- "$work"' EXIT

flip_byte() {
  local file=$1 offset=$2 current next
  current=$(dd if="$file" bs=1 skip="$offset" count=1 status=none \
    | od -An -tu1 | tr -d ' ')
  next=$(((current + 1) % 256))
  printf "$(printf '\\%03o' "$next")" \
    | dd of="$file" bs=1 seek="$offset" count=1 conv=notrunc status=none
}

# Run the gate against the currently planted run and require an outcome.
# expectation is PASS (exit 0) or REFUSE (any nonzero exit).  A refusal must
# also print the pattern the caller names, so that each plant is shown to trip
# the specific condition it was built for rather than any refusal at all.
check_plant() {
  local label=$1 expectation=$2 pattern=${3:-} status=0 log=$work/gate.log
  "$GATE" --finalize >"$log" 2>&1 || status=$?
  if [[ "$expectation" == REFUSE ]]; then
    if [[ "$status" -eq 0 ]]; then
      printf 'PLANT FAIL %s: gate admitted a corrupted run (exit 0)\n' "$label"
      failures=$((failures + 1))
      return
    fi
    if ! grep -q '^REFUSE: ' "$log"; then
      printf 'PLANT FAIL %s: gate exited %s with no named REFUSE line\n' \
        "$label" "$status"
      failures=$((failures + 1))
      return
    fi
    if ! grep -qF -- "$pattern" "$log"; then
      printf 'PLANT FAIL %s: gate refused without reporting %s\n' \
        "$label" "$pattern"
      failures=$((failures + 1))
      return
    fi
    printf 'PLANT PASS %s: exit=%s %s\n' "$label" "$status" \
      "$(grep -m1 -F -- "$pattern" "$log")"
  else
    if [[ "$status" -ne 0 ]]; then
      printf 'PLANT FAIL %s: gate refused an intact run (exit %s)\n' \
        "$label" "$status"
      failures=$((failures + 1))
      return
    fi
    printf 'PLANT PASS %s: exit=0 %s\n' "$label" \
      "$(grep -m1 '^ORCA2_TKE_BOUNDARY_PASSIVITY ' "$log")"
  fi
}

# Corrupt one stream, require an outcome, then restore it byte-for-byte.
plant_on_stream() {
  local label=$1 stream=$2 expectation=$3 pattern=$4 action=$5 argument=${6:-}
  local staged=$BACKUP_DIR/$stream.partial
  # Back the stream up and record its digest BEFORE corrupting it, and publish
  # the backup under its final name with a rename, so a backup that exists is
  # always complete and always has a digest beside it.
  cp -f "$TARGET_RUN/$stream" "$staged"
  sha256sum "$staged" | awk '{print $1}' >"$BACKUP_DIR/$stream.sha256"
  mv -f "$staged" "$BACKUP_DIR/$stream.orig"
  case "$action" in
    flip) flip_byte "$TARGET_RUN/$stream" "$argument" ;;
    truncate) truncate -s "-$argument" "$TARGET_RUN/$stream" ;;
    extend) truncate -s "+$argument" "$TARGET_RUN/$stream" ;;
    *)
      printf 'PLANT FAIL %s: unknown plant action %s\n' "$label" "$action"
      failures=$((failures + 1))
      restore_pending
      return
      ;;
  esac
  check_plant "$label" "$expectation" "$pattern"
  restore_pending
}

ordinary_stream=$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_*.bin' \
  -printf '%f\n' | sort \
  | grep -v -e "^$PARTIAL_STREAM\$" -e "^$RECORD\$" | head -n 1)
[[ -n "$ordinary_stream" ]] || {
  printf 'REFUSE: no ordinary inherited stream available to plant on\n' >&2
  exit 64
}

printf 'ORCA2 passivity plant; ordinary stream under test: %s\n' \
  "$ordinary_stream"

# 1. An ordinary inherited stream still carries full signal.
plant_on_stream 'ordinary-inherited-stream-byte' "$ordinary_stream" REFUSE \
  "ORCA2_TKE_BOUNDARY_DIFFERS $ordinary_stream" flip 2048

# 2. The transport stream's FIRST field (zFu) is inside the compared span, so
#    the narrowed comparison must still catch it.
plant_on_stream 'transport-zFu-byte' "$PARTIAL_STREAM" REFUSE \
  "ORCA2_TKE_BOUNDARY_DIFFERS $PARTIAL_STREAM" flip "$ZFU_OFFSET"

# 3. The SECOND field (zFv) is inside the compared span as well.
plant_on_stream 'transport-zFv-byte' "$PARTIAL_STREAM" REFUSE \
  "ORCA2_TKE_BOUNDARY_DIFFERS $PARTIAL_STREAM" flip "$ZFV_OFFSET"

# 4. A transport stream of the wrong length refuses under its own named
#    condition, short or padded, even though the changed bytes are in the
#    excluded tail; the length requirement is what stops a file hiding there,
#    and requiring the length refusal by name is what shows it is that check
#    firing rather than an ordinary byte difference.
plant_on_stream 'transport-truncated-tail' "$PARTIAL_STREAM" REFUSE \
  "REFUSE: inherited stream $PARTIAL_STREAM is not the registered $STREAM_BYTES bytes" \
  truncate 8
plant_on_stream 'transport-padded-tail' "$PARTIAL_STREAM" REFUSE \
  "REFUSE: inherited stream $PARTIAL_STREAM is not the registered $STREAM_BYTES bytes" \
  extend 8

# 5. Control: a byte in the excluded, never-assigned third field is admitted.
#    This is the documented exclusion behaving as described, not an accident.
plant_on_stream 'transport-zFw-byte-excluded' "$PARTIAL_STREAM" PASS '' \
  flip "$ZFW_OFFSET"

# Leave the run in its admitted state.
"$GATE" --finalize >"$work/final.log" 2>&1 || {
  printf 'REFUSE: run did not return to admitted state after the plants\n' >&2
  exit 70
}
grep -m1 '^ORCA2_TKE_BOUNDARY_PASSIVITY ' "$work/final.log"

if [[ "$failures" -ne 0 ]]; then
  printf 'REFUSE: %s ORCA2 passivity plant(s) did not behave as required\n' \
    "$failures" >&2
  exit 71
fi
printf 'ORCA2_PASSIVITY_PLANT PASS all controls behaved as required\n'
