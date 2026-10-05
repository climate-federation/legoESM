#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: unhandled SMT-3 acquisition failure at run.sh line %s (exit %s)\n' \
    "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
driver=$repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round224/oracle_vortex_smt3
kt_dir=$root/kt1_10
year_dir=$root/day100
kt_admission=$kt_dir/vortex_round224_smt3_vec_admission.json
year_admission=$year_dir/vortex_round224_smt3_vec_100d_admission.json

[[ -x "$driver" ]] \
  || { printf 'REFUSE: shared VORTEX acquisition driver is missing: %s\n' "$driver" >&2; exit 66; }

do_run=1
case "${1:-}" in
  '') ;;
  --run) do_run=1 ;;
  --preflight) do_run=0 ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight]\n' "$0" >&2; exit 64 ;;
esac
[[ $# -le 1 ]] \
  || { printf 'REFUSE: usage: %s [--run|--preflight]\n' "$0" >&2; exit 64; }

if [[ "$do_run" -eq 0 ]]; then
  EVIDENCE=$kt_dir "$driver" --variant smt3vec
  EVIDENCE=$year_dir "$driver" --variant smt3vec100d
  printf 'ROUND224_SMT3_PREFLIGHT_PASS %s\n' "$root"
  exit 0
fi

if [[ -f "$kt_admission" ]] && grep -q '"status": "ADMITTED"' "$kt_admission"; then
  [[ -f "$kt_dir/binaries.sha256" ]] \
    || { printf 'REFUSE: admitted ten-step arm lacks binaries.sha256\n' >&2; exit 70; }
  printf 'ROUND224_SMT3_KT1_10_ALREADY_ADMITTED %s\n' "$kt_dir"
elif [[ -e "$kt_dir" ]]; then
  printf 'REFUSE: ten-step evidence exists but is not admitted: %s\n' "$kt_dir" >&2
  exit 70
else
  started=$SECONDS
  EVIDENCE=$kt_dir "$driver" --variant smt3vec --run
  printf 'ROUND224_SMT3_KT1_10_SECONDS %s\n' "$((SECONDS-started))"
fi

if [[ -f "$year_admission" ]] && grep -q '"status": "ADMITTED"' "$year_admission"; then
  printf 'ROUND224_SMT3_DAY100_ALREADY_ADMITTED %s\n' "$year_dir"
elif [[ -e "$year_dir" ]]; then
  printf 'REFUSE: 100-day evidence exists but is not admitted: %s\n' "$year_dir" >&2
  exit 70
else
  started=$SECONDS
  EVIDENCE=$year_dir "$driver" --variant smt3vec100d --run
  printf 'ROUND224_SMT3_DAY100_SECONDS %s\n' "$((SECONDS-started))"
fi

printf 'ROUND224_SMT3_RECORD_READY %s\n' "$root"
