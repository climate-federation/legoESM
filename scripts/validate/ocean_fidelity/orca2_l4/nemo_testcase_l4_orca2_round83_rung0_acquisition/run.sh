#!/usr/bin/env bash
# ORCA2 round-83 acquisition: fresh-target repair for rung-0 restart output.
set -Eeuo pipefail

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
driver=$here/../nemo_testcase_l4_orca2_round82_rung0_acquisition/run.sh

git -C "$repo" ls-files --error-unmatch "${driver#"$repo"/}" >/dev/null || {
  printf 'REFUSE: shared rung-0 acquisition driver is not committed: %s\n' "$driver" >&2
  exit 65
}

export ORCA2_RUNG0_ROUND=83
export ORCA2_RUNG0_EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition
export ORCA2_RUNG0_TWIN_A=orca2_rung0_restart_list_10step_a_np2
export ORCA2_RUNG0_TWIN_B=orca2_rung0_restart_list_10step_b_np2
export ORCA2_RUNG0_MONTH=orca2_rung0_restart_list_repair_240step_np2
export ORCA2_RUNG0_PREREG=$repo/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round83.md

exec "$driver" "$@"
