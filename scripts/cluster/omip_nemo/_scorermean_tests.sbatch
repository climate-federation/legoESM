#!/bin/bash
#SBATCH --job-name=scorermean
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scorermean_%j.out
#
# Unit gate for the TKE step budget (TKEStepBudget / return_budget): the new
# closure test plus every existing TKE unit test, so the primal path is shown
# untouched before the budget is read on NEMO's state.
set -uo pipefail
WT=${WT:-/burg-archive/glab/users/pg2328/_omip_merge}
cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "HEAD: $(git -C $WT log --oneline -1)"
echo "dirty: $(git -C $WT status --porcelain | wc -l)"
$PY -m pytest -q -p no:cacheprovider tests/ocean/unit/test_compare_omip_nemo_bands.py tests/ocean/unit/test_mld_snapshot.py tests/ocean/fidelity/test_run_comparison_script.py 2>&1 | tail -15
rc1=${PIPESTATUS[0]}
echo "=== accumulator test rc=$rc1"
$PY -m pytest -q -p no:cacheprovider tests/ocean/unit/test_compare_omip_nemo_bands.py tests/ocean/unit/test_mld_snapshot.py tests/ocean/fidelity/test_run_comparison_script.py 2>&1 | tail -8
rc2=${PIPESTATUS[0]}
echo "=== accumulator + CLI tests rc=$rc2"
echo "=== card exit: $((rc1 + rc2)) ==="
exit $((rc1 + rc2))
