#!/bin/bash
#SBATCH --job-name=base_cli5
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/base_cli5.%j.out
# Five AMIP CLI tests came back red while the #1783 pin and the #1029 guard
# were both in the tree.  A new failure is not mine until it is reproduced
# with my changes removed, so run the same five at the committed HEAD in a
# SEPARATE checkout -- never by stashing in a shared worktree.
set -uo pipefail
SRC=/burg-archive/glab/users/pg2328/_sweep0923_wt
BASE=/burg-archive/glab/users/pg2328/_sweep0923_base
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
SEL='test_land_surface_scheme_flag_flows_to_config or test_surface_tiled_flags_flow_to_config or test_surface_tiled_accepts_flux_consuming_schemes'

rm -rf "$BASE"
git -C "$SRC" worktree remove --force "$BASE" 2>/dev/null || true
git -C "$SRC" worktree add --detach "$BASE" HEAD >/dev/null 2>&1 || {
  echo "FATAL: could not create the baseline worktree"; exit 2; }
echo "baseline HEAD: $(git -C "$BASE" log --oneline -1)"
echo "baseline dirty: $(git -C "$BASE" status --porcelain | wc -l) files"

run () {   # $1 = label, $2 = tree
  echo "########## $1 ##########"
  cd "$2"
  export PYTHONPATH="$2/src:$2/packages/atmosphere:$2/packages/core:$2/packages/coupler:$2/packages/ice:$2/packages/land:$2/packages/ml:$2/packages/ocean:$2/packages/tools:$2"
  export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
  "$PY" -m pytest -q tests/unit/test_run_amip_cli.py -k "$SEL" 2>&1 | tail -12
}

run "BASELINE (committed HEAD, none of today's uncommitted work)" "$BASE"
run "WORKING TREE (with the #1783 pin and the #1029 guard)" "$SRC"

git -C "$SRC" worktree remove --force "$BASE" 2>/dev/null || true
echo DONE
