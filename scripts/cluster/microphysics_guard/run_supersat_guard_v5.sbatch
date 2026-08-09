#!/bin/bash
#SBATCH --job-name=ss_guard5
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs/ss_guard5_%j.log
# Round 5: re-confirm every suite at the CURRENT sha (round 3 ran at a stale
# one), plus a ruff BASE-vs-BRANCH comparison so a pre-existing repo-wide lint
# state is not misreported as introduced by this change.
set -uo pipefail

PINNED_SHA="${PINNED_SHA:?set PINNED_SHA}"
BASE_SHA="${BASE_SHA:?set BASE_SHA}"
SRC=/burg-archive/glab/users/pg2328/legoESM
RUNS=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs
WT="$RUNS/wt5_${SLURM_JOB_ID}"; WTB="$RUNS/wt5base_${SLURM_JOB_ID}"
PY="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"

mkdir -p "$RUNS"
git -C "$SRC" worktree add --detach "$WT"  "$PINNED_SHA" >/dev/null 2>&1 || { echo "WT ADD FAILED"; exit 1; }
git -C "$SRC" worktree add --detach "$WTB" "$BASE_SHA"   >/dev/null 2>&1 || { echo "WTB ADD FAILED"; exit 1; }
echo "PINNED branch=$(git -C "$WT" rev-parse HEAD)"
echo "PINNED base  =$(git -C "$WTB" rev-parse HEAD)"
_pp_of() { local w="$1"; local p="$w"; for d in "$w"/packages/*/; do [ -d "${d}legoesm" ] && p="${p}:${d}"; done; echo "$p"; }
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

echo "===== ALL guard suites at the CURRENT sha ====="
( cd "$WT" && PYTHONPATH="$(_pp_of "$WT")" "$PY" -m pytest -q \
    "$WT/tests/test_microphysics_supersaturation_guard.py" \
    "$WT/tests/unit/test_supersaturation_guard_behaviour.py" \
    "$WT/tests/unit/test_hard_saturation_adjustment.py" \
    "$WT/tests/unit/test_hard_sat_ice_curve.py" \
    "$WT/tests/unit/test_physics_microphysics.py" \
    "$WT/tests/unit/test_params_reachability_audit.py" \
    "$WT/tests/atmosphere/hydrostatic/unit/test_sundqvist_faithful.py" \
    "$WT/tests/atmosphere/hydrostatic/unit/test_microphysics.py" \
    "$WT/tests/unit/test_diff_atmosphere_convection_micro.py" 2>&1 | tail -25 )
echo "SUITE_RC=$?"

echo "===== ruff: SAME file list on base vs branch (pre-existing vs introduced) ====="
# Files that exist on BOTH sides, so the counts are comparable.
COMMON="packages/atmosphere/legoesm/atmosphere/physics/microphysics/sundqvist.py
packages/atmosphere/legoesm/atmosphere/physics/microphysics/ml_emulator.py
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py
packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py
packages/atmosphere/legoesm/atmosphere/physics/microphysics/kessler.py
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py
packages/coupler/legoesm/driver/run_config_yaml.py"
for tag in base branch; do
  if [ "$tag" = base ]; then w="$WTB"; else w="$WT"; fi
  n=$(cd "$w" && echo "$COMMON" | xargs "$PY" -m ruff check --statistics 2>/dev/null | tail -20 | awk '{s+=$1} END {print s+0}')
  echo "RUFF_${tag}_COMMON_FILES=$n"
done
echo "--- rule breakdown on the two files this change AUTHORED (branch) ---"
( cd "$WT" && "$PY" -m ruff check --statistics \
    "$WT/tests/test_microphysics_supersaturation_guard.py" \
    "$WT/tests/unit/test_supersaturation_guard_behaviour.py" 2>&1 | head -15 )
echo "--- same rules on an UNTOUCHED sibling test file (base) ---"
( cd "$WTB" && "$PY" -m ruff check --statistics \
    "$WTB/tests/unit/test_hard_saturation_adjustment.py" 2>&1 | head -15 )

echo "ALL DONE job=${SLURM_JOB_ID}"
