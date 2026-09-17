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

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly GATE=$here/run.sh
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/parallel/orca2/oracle_phase2v_tke_boundary_np2
readonly PARTIAL_STREAM=oracle_transport_kt00000001_s1.bin
readonly RECORD=oracle_tke_boundary_kt00000002.bin
readonly FIELD_BYTES=$((94 * 152 * 31 * 8))
readonly HEADER_BYTES=$((16 + 8 * 4))
readonly ZFU_OFFSET=$((HEADER_BYTES + 1000))
readonly ZFW_OFFSET=$((HEADER_BYTES + 2 * FIELD_BYTES + 1000))

work=$(mktemp -d /tmp/orca2-passivity-plant.XXXXXXXX)
trap 'rm -rf -- "$work"' EXIT

failures=0

flip_byte() {
  local file=$1 offset=$2 current next
  current=$(dd if="$file" bs=1 skip="$offset" count=1 status=none \
    | od -An -tu1 | tr -d ' ')
  next=$(((current + 1) % 256))
  printf "$(printf '\\%03o' "$next")" \
    | dd of="$file" bs=1 seek="$offset" count=1 conv=notrunc status=none
}

# Run the gate against the currently planted run and require an outcome.
# expectation is PASS (exit 0) or REFUSE (any nonzero exit).
check_plant() {
  local label=$1 expectation=$2 status=0 log=$work/gate.log
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
    printf 'PLANT PASS %s: exit=%s %s\n' "$label" "$status" \
      "$(grep -m1 '^REFUSE: ' "$log")"
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
  local label=$1 stream=$2 expectation=$3 action=$4 argument=${5:-}
  local original=$work/$stream.orig digest restored
  cp -f "$TARGET_RUN/$stream" "$original"
  digest=$(sha256sum "$original" | awk '{print $1}')
  case "$action" in
    flip) flip_byte "$TARGET_RUN/$stream" "$argument" ;;
    truncate) truncate -s "-$argument" "$TARGET_RUN/$stream" ;;
    *)
      printf 'PLANT FAIL %s: unknown plant action %s\n' "$label" "$action"
      failures=$((failures + 1))
      return
      ;;
  esac
  check_plant "$label" "$expectation"
  cp -f "$original" "$TARGET_RUN/$stream"
  restored=$(sha256sum "$TARGET_RUN/$stream" | awk '{print $1}')
  [[ "$restored" == "$digest" ]] || {
    printf 'PLANT FAIL %s: could not restore %s\n' "$label" "$stream"
    failures=$((failures + 1))
  }
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
  flip 2048

# 2. The transport stream's FIRST field (zFu) is inside the compared span, so
#    the narrowed comparison must still catch it.
plant_on_stream 'transport-zFu-byte' "$PARTIAL_STREAM" REFUSE \
  flip "$ZFU_OFFSET"

# 3. Truncating the transport stream must refuse even though the lost bytes are
#    in the excluded tail; the length requirement is what stops a short file
#    hiding there.
plant_on_stream 'transport-truncated-tail' "$PARTIAL_STREAM" REFUSE \
  truncate 8

# 4. Control: a byte in the excluded, never-assigned third field is admitted.
#    This is the documented exclusion behaving as described, not an accident.
plant_on_stream 'transport-zFw-byte-excluded' "$PARTIAL_STREAM" PASS \
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
