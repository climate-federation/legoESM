#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: unhandled SMT-6 acquisition failure at run.sh line %s (exit %s)\n' \
    "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

# SMT-RUNGS round 5, rungs SMT-6 and SMT-6b (Decision 110) -- OPERATOR-EXECUTED ONLY.
#   run.sh [--preflight]   check inputs, patches, decks, Fortran syntax; build nothing
#   run.sh --smoke         SMT-6: 2 steps on the SMT-5 executables (hash-proved);
#                          SMT-6b: build VORTEX_SMT6B_VEC_R8_OMIP_L1{,_P3}, 2 steps
#   run.sh --run           both decks: kt=1..10 per-stage record + 100-day daily
#                          record, reusing the smoke binaries by hash (needs --smoke)
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
driver=$repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/round5

[[ -x "$driver" ]] \
  || { printf 'REFUSE: shared VORTEX acquisition driver is missing: %s\n' "$driver" >&2; exit 66; }
[[ $# -le 1 ]] || { printf 'REFUSE: usage: %s [--preflight|--smoke|--run]\n' "$0" >&2; exit 64; }
mode=${1:---preflight}

admitted() {  # $1 = evidence dir, $2 = tag
  local json=$1/vortex_$2_admission.json
  [[ -f "$json" ]] && grep -q '"status": "ADMITTED"' "$json"
}

acquire() {   # $1 = variant, $2 = evidence dir, $3 = tag
  if admitted "$2" "$3"; then
    printf 'SMT6_%s_ALREADY_ADMITTED %s\n' "$1" "$2"
  elif [[ -e "$2" ]]; then
    printf 'REFUSE: evidence exists but is not admitted: %s\n' "$2" >&2
    exit 70
  else
    local started=$SECONDS
    EVIDENCE=$2 "$driver" --variant "$1" --run
    printf 'SMT6_%s_SECONDS %s\n' "$1" "$((SECONDS-started))"
  fi
}

case "$mode" in
  --preflight)
    for deck in smt6 smt6b; do
      for v in vecsmoke:smoke vec:kt1_10 vec100d:day100; do
        EVIDENCE=$root/oracle_vortex_$deck/${v#*:} "$driver" --variant "$deck${v%%:*}"
      done
    done
    printf 'SMTRUNGS_R5_SMT6_PREFLIGHT_PASS %s\n' "$root"
    ;;
  --smoke)
    for deck in smt6 smt6b; do
      dir=$root/oracle_vortex_$deck/smoke
      acquire "${deck}vecsmoke" "$dir" "smtrungs_r5_${deck}_vec_smoke"
      [[ -f "$dir/binaries.sha256" ]] \
        || { printf 'REFUSE: admitted %s smoke arm lacks binaries.sha256\n' "$deck" >&2; exit 70; }
    done
    printf 'SMTRUNGS_R5_SMT6_SMOKE_READY %s\n' "$root"
    ;;
  --run)
    for deck in smt6 smt6b; do
      admitted "$root/oracle_vortex_$deck/smoke" "smtrungs_r5_${deck}_vec_smoke" \
        || { printf 'REFUSE: run --smoke first (no admitted %s smoke arm)\n' "$deck" >&2; exit 70; }
    done
    for deck in smt6 smt6b; do
      acquire "${deck}vec" "$root/oracle_vortex_$deck/kt1_10" "smtrungs_r5_${deck}_vec"
      acquire "${deck}vec100d" "$root/oracle_vortex_$deck/day100" "smtrungs_r5_${deck}_vec_100d"
    done
    printf 'SMTRUNGS_R5_SMT6_RECORD_READY %s\n' "$root"
    ;;
  *) printf 'REFUSE: usage: %s [--preflight|--smoke|--run]\n' "$0" >&2; exit 64 ;;
esac
