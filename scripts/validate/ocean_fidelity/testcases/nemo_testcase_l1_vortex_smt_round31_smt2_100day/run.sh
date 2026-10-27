#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: round-243 SMT-2 acquisition failed at run.sh line %s (exit %s)\n' \
    "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

# Round 243 needs the daily SMT-2 oracle trajectory before the shared
# round-210 scorer can run.  Reuse the existing VORTEX acquisition driver and
# its certified-binary hash checks; do not rebuild an unchanged NEMO source
# card.  The smt2vec100d arm changes only the run deck to NEMO's shipped
# 3000-step length and a 30-step restart cadence, then parses and admits every
# output itself.

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: commit the exact round-243 acquisition tool before running it\n' >&2
  exit 66
fi

driver=$repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round243/oracle_vortex_smt2/day100
[[ -x "$driver" ]] \
  || { printf 'REFUSE: common VORTEX acquisition driver is missing or not executable: %s\n' "$driver" >&2; exit 66; }
[[ ! -e "$evidence" ]] \
  || { printf 'REFUSE: round-243 evidence target already exists: %s\n' "$evidence" >&2; exit 66; }

printf 'ROUND243_SMT2_ACQUISITION_START %s\n' "$evidence"
EVIDENCE="$evidence" "$driver" --run --variant smt2vec100d
printf 'ROUND243_SMT2_RECORD_READY %s\n' "$evidence"
