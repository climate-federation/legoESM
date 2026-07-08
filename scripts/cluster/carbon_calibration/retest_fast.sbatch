#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_retest
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/carbon_calibration/retest_%j.out
# Fast re-validation after the codex fixes (preset-aware om_to_oc + case-tolerant
# organic var lookup). Skips the slow --quick full-pipeline test (unaffected by
# the real-path edits, already green in the validate job).
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/lc_rebase
cd "$REPO" || exit 2
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO:$(ls -d "$REPO"/packages/*/ | sed 's:/$::' | tr '\n' ':')${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
$PY -m pytest tests/land/unit/test_soc_observations.py \
    tests/land/unit/test_train_carbon_params.py -q -k "not quick_dry_run"
echo "RETEST_EXIT=$?"
