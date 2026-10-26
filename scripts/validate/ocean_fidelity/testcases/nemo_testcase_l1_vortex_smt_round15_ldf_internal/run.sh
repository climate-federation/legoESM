#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: unhandled Round-227 acquisition failure at run.sh line %s (exit %s)\n' \
    "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
driver=$repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round227/oracle_vortex_smt3_ldf_internal

[[ -x "$driver" ]] \
  || { printf 'REFUSE: shared VORTEX acquisition driver is missing: %s\n' "$driver" >&2; exit 66; }

mode=--run
case "${1:-}" in
  '') ;;
  --run) mode=--run ;;
  --preflight) mode=--preflight ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight]\n' "$0" >&2; exit 64 ;;
esac
[[ $# -le 1 ]] \
  || { printf 'REFUSE: usage: %s [--run|--preflight]\n' "$0" >&2; exit 64; }

if [[ "$mode" == --preflight ]]; then
  EVIDENCE=$evidence "$driver" --variant smt3vecint
  printf 'ROUND227_LDF_INTERNAL_PREFLIGHT_PASS %s\n' "$evidence"
  exit 0
fi

[[ ! -e "$evidence" ]] \
  || { printf 'REFUSE: Round-227 evidence target already exists: %s\n' "$evidence" >&2; exit 70; }
started=$SECONDS
EVIDENCE=$evidence "$driver" --variant smt3vecint --run
for path in \
  "$evidence/oracle_ldf_slope_kt00000001.bin" \
  "$evidence/oracle_ldf_slope_kt00000001.bin.stamp" \
  "$evidence/oracle_ldf_iso_kt00000001.bin" \
  "$evidence/oracle_ldf_iso_kt00000001.bin.stamp" \
  "$evidence/vortex_round227_smt3_ldf_internal_admission.json"; do
  [[ -f "$path" ]] || { printf 'REFUSE: admitted acquisition lacks %s\n' "$path" >&2; exit 70; }
done
grep -q '"status": "ADMITTED"' \
  "$evidence/vortex_round227_smt3_ldf_internal_admission.json" \
  || { printf 'REFUSE: Round-227 admission did not report ADMITTED\n' >&2; exit 70; }
printf 'ROUND227_LDF_INTERNAL_SECONDS %s\n' "$((SECONDS-started))"
printf 'ROUND227_LDF_INTERNAL_RECORD_READY %s\n' "$evidence"
