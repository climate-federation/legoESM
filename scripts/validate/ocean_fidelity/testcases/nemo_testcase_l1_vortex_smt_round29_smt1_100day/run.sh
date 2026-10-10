#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: round-241 SMT-1 acquisition failed at run.sh line %s (exit %s)\n' \
    "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

# Round 241 needs the daily SMT-1 oracle trajectory before the common
# round-210 scorer can run.  Reuse the campaign's existing VORTEX acquisition
# driver and its certified-binary hash checks; do not create a second NEMO
# builder or recompile an unchanged source card.  The generic smt1vec100d arm
# changes only the run deck to NEMO's shipped 3000-step length and 30-step
# restart cadence, then parses and admits the output.

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: commit the exact round-241 acquisition tool before running it\n' >&2
  exit 66
fi

driver=$repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex/run.sh
evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round241/oracle_vortex_smt1/day100
[[ -x "$driver" ]] \
  || { printf 'REFUSE: common VORTEX acquisition driver is missing or not executable: %s\n' "$driver" >&2; exit 66; }
[[ ! -e "$evidence" ]] \
  || { printf 'REFUSE: round-241 evidence target already exists: %s\n' "$evidence" >&2; exit 66; }

printf 'ROUND241_SMT1_ACQUISITION_START %s\n' "$evidence"
EVIDENCE="$evidence" "$driver" --run --variant smt1vec100d
printf 'ROUND241_SMT1_RECORD_READY %s\n' "$evidence"
