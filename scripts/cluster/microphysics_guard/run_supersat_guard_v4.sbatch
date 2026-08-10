#!/bin/bash
#SBATCH --job-name=ss_guard4
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs/ss_guard4_%j.log
# Round 4: PRE-EXISTING-FAILURE CONTROL. The same CLI + flag-reachability
# suites are run on origin/main (BASE) and on the branch, so a failure is only
# attributed to this change if it is absent on BASE (CLAUDE.md: a new failure
# is not yours until reproduced with the change stashed).
set -uo pipefail

PINNED_SHA="${PINNED_SHA:?set PINNED_SHA}"
BASE_SHA="${BASE_SHA:?set BASE_SHA}"
SRC=/burg-archive/glab/users/pg2328/legoESM
RUNS=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs
WT="$RUNS/wt4_${SLURM_JOB_ID}"; WTB="$RUNS/wt4base_${SLURM_JOB_ID}"
PY="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"

mkdir -p "$RUNS"
git -C "$SRC" worktree add --detach "$WT"  "$PINNED_SHA" >/dev/null 2>&1 || { echo "WT ADD FAILED"; exit 1; }
git -C "$SRC" worktree add --detach "$WTB" "$BASE_SHA"   >/dev/null 2>&1 || { echo "WTB ADD FAILED"; exit 1; }
echo "PINNED branch=$(git -C "$WT" rev-parse HEAD)"
echo "PINNED base  =$(git -C "$WTB" rev-parse HEAD)"
_pp_of() { local w="$1"; local p="$w"; for d in "$w"/packages/*/; do [ -d "${d}legoesm" ] && p="${p}:${d}"; done; echo "$p"; }
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for tag in base branch; do
  if [ "$tag" = base ]; then w="$WTB"; else w="$WT"; fi
  echo "===== [$tag] run_amip CLI + flag reachability ====="
  ( cd "$w" && PYTHONPATH="$(_pp_of "$w")" "$PY" -m pytest -q \
      "$w/tests/unit/test_run_amip_cli.py" \
      "$w/tests/unit/test_flag_reachability_audit.py" 2>&1 | tail -12 )
  echo "${tag^^}_CLI_RC=$?"
done

echo "===== [branch] full guard suites re-confirm ====="
( cd "$WT" && PYTHONPATH="$(_pp_of "$WT")" "$PY" -m pytest -q \
    "$WT/tests/test_microphysics_supersaturation_guard.py" \
    "$WT/tests/unit/test_supersaturation_guard_behaviour.py" \
    "$WT/tests/unit/test_hard_saturation_adjustment.py" \
    "$WT/tests/unit/test_hard_sat_ice_curve.py" \
    "$WT/tests/unit/test_physics_microphysics.py" \
    "$WT/tests/unit/test_params_reachability_audit.py" 2>&1 | tail -20 )
echo "SUITE_RC=$?"

echo "ALL DONE job=${SLURM_JOB_ID}"
