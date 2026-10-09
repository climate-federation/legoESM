#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: unhandled SMT-5 acquisition failure at run.sh line %s (exit %s)\n' \
    "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

# SMT-RUNGS round 2, rung SMT-5 (Decision 107) -- OPERATOR-EXECUTED ONLY.
#   run.sh [--preflight]   check inputs, patches, deck, Fortran syntax; build nothing
#   run.sh --smoke         build VORTEX_SMT5_VEC_R8_OMIP_L1{,_P3}; 2-step run, both arms
#   run.sh --run           kt=1..10 per-stage record + 100-day daily record,
#                          reusing the smoke binaries by hash (needs --smoke first)
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
driver=$repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/round2/oracle_vortex_smt5
smoke_dir=$root/smoke
kt_dir=$root/kt1_10
year_dir=$root/day100

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
    printf 'SMT5_%s_ALREADY_ADMITTED %s\n' "$1" "$2"
  elif [[ -e "$2" ]]; then
    printf 'REFUSE: evidence exists but is not admitted: %s\n' "$2" >&2
    exit 70
  else
    local started=$SECONDS
    EVIDENCE=$2 "$driver" --variant "$1" --run
    printf 'SMT5_%s_SECONDS %s\n' "$1" "$((SECONDS-started))"
  fi
}

case "$mode" in
  --preflight)
    EVIDENCE=$smoke_dir "$driver" --variant smt5vecsmoke
    EVIDENCE=$kt_dir "$driver" --variant smt5vec
    EVIDENCE=$year_dir "$driver" --variant smt5vec100d
    printf 'SMTRUNGS_R2_SMT5_PREFLIGHT_PASS %s\n' "$root"
    ;;
  --smoke)
    acquire smt5vecsmoke "$smoke_dir" smtrungs_r2_smt5_vec_smoke
    [[ -f "$smoke_dir/binaries.sha256" ]] \
      || { printf 'REFUSE: admitted smoke arm lacks binaries.sha256\n' >&2; exit 70; }
    printf 'SMTRUNGS_R2_SMT5_SMOKE_READY %s\n' "$smoke_dir"
    ;;
  --run)
    admitted "$smoke_dir" smtrungs_r2_smt5_vec_smoke \
      || { printf 'REFUSE: run --smoke first (no admitted smoke arm in %s)\n' "$smoke_dir" >&2; exit 70; }
    acquire smt5vec "$kt_dir" smtrungs_r2_smt5_vec
    acquire smt5vec100d "$year_dir" smtrungs_r2_smt5_vec_100d
    printf 'SMTRUNGS_R2_SMT5_RECORD_READY %s\n' "$root"
    ;;
  *) printf 'REFUSE: usage: %s [--preflight|--smoke|--run]\n' "$0" >&2; exit 64 ;;
esac
