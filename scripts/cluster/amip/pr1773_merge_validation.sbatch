#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pr1773merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/pr1773_merge/validate_%j.out
#
# PR #1773: validate the MERGE COMMIT, and attribute every failure.
#
# The PR's own test plan has "full unit suite on the merged tree" as an
# unticked box, and the merge resolved 26 hunks across 12 files -- most of
# them the SAME feature arriving twice, because the peer PR #1756
# cherry-picked part of this branch and kept developing it on main. That is
# the shape where a resolution silently reverts or silently duplicates.
#
# THREE ARMS, so no failure is left unattributed:
#   MERGE  = the merge commit           (_pr1773_wt)
#   BRANCH = the pre-merge PR tip       (_pr1773_tip)
#   MAIN   = plain origin/main          (_main_ref, already built)
# A failure present on MAIN is not this PR's. A failure on MERGE and BRANCH
# but not MAIN is the PR's, pre-existing. A failure on MERGE alone is the
# RESOLUTION'S, and that is the only class that blocks.
#
# MODULES: the ones the resolution touched, plus the suites those files gate.
# `test_run_amip_cli.py` is included knowing main is already red on 11 of its
# tests (measured: 227 passed / 11 failed on plain main) -- the three-arm
# comparison is exactly how that stops being a confound.
#
# Each module runs ALONE. Learned on #1769: `test_fv3_duo_window_spmd.py`
# documents "Run ALONE (the device count must be set before jax is imported)"
# and sets XLA_FLAGS by `os.environ.setdefault`, so batching it with other
# modules silently fixes the device count at 1 and reports failures that are
# the harness, not the code.
set -uo pipefail
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
MERGE=/burg-archive/glab/users/pg2328/_pr1773_wt
BRANCH=/burg-archive/glab/users/pg2328/_pr1773_tip
MAIN=/burg-archive/glab/users/pg2328/_main_ref

if [ ! -d "$BRANCH" ]; then
  git -C "$MERGE" worktree add --detach "$BRANCH" origin/feat/raised-lid-l36 \
    >/dev/null 2>&1 || { echo "REFUSED: cannot build the pre-merge worktree"; exit 2; }
fi

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

MODULES="
tests/unit/test_gwd_background_source.py
tests/unit/test_sigma_top_config.py
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py
tests/test_validate_strict_coverage.py
tests/unit/test_cap_water.py
tests/unit/test_run_amip_cli.py
tests/unit/test_clubb_config.py
tests/unit/test_config_validation.py
"

run_tree () {   # $1 label, $2 worktree
  local label="$1" WT="$2"
  [ -d "$WT" ] || { echo "############ $label ABSENT ($WT) ############"; return; }
  echo "############ $label ($WT) ############"
  ( cd "$WT" && echo "sha=$(git rev-parse --short HEAD)" )
  export PYTHONPATH="$WT/packages/atmosphere:$WT/packages/ocean:$WT/packages/land:$WT/packages/ice:$WT/packages/coupler:$WT/packages/core:$WT/packages/ml:$WT/packages/tools:$WT/src:$WT:$WT/scripts/validate"
  export LEGOESM_MESH_CACHE_DIR="$WT/.meshcache"; mkdir -p "$LEGOESM_MESH_CACHE_DIR"
  export LEGOESM_ALLOW_BIG_MESH_BUILD=1
  for mod in $MODULES; do
    [ -f "$WT/$mod" ] || { echo "=== $label :: $mod ABSENT on this tree ==="; continue; }
    echo "=== $label :: $mod $(date -u) ==="
    ( cd "$WT" && timeout 3600 $PY -m pytest "$mod" -q -p no:randomly 2>&1 \
        | grep -E "^FAILED|^ERROR|passed|failed|error" | tail -25 )
    echo "=== $label :: $mod exit=${PIPESTATUS[0]} ==="
  done
}

run_tree MERGE  "$MERGE"
run_tree BRANCH "$BRANCH"
run_tree MAIN   "$MAIN"

echo "PR1773 MERGE VALIDATION DONE $(date -Is)"
