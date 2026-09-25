#!/bin/bash
#SBATCH --job-name=regen_gold
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/regen_gold.%j.out
# Codex review finding: GridConfig.transition_exponent is serialized, but both
# Williamson goldens predate the field, so the exact-comparison golden test
# goes red.  The test documents its own regeneration path and asks for it "in
# a dedicated diff"; this is that regeneration, and it prints the diff so the
# change is a row added, not a silent rewrite.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

echo "########## BEFORE: the golden test should be red ##########"
"$PY" -m pytest -q tests/unit/test_yaml_to_experiment_config_golden.py 2>&1 | tail -8

echo "########## regenerate ##########"
LEGOESM_REGEN_GOLDEN=1 "$PY" -m pytest -q tests/unit/test_yaml_to_experiment_config_golden.py 2>&1 | tail -5

echo "########## what changed -- must be ONLY the new field ##########"
git -C "$WT" diff --stat tests/unit/golden/
git -C "$WT" diff tests/unit/golden/ | grep -E "^[+-]" | grep -v "^[+-][+-]" | sort | uniq -c | sort -rn | head -20

echo "########## AFTER: green ##########"
"$PY" -m pytest -q tests/unit/test_yaml_to_experiment_config_golden.py 2>&1 | tail -5
echo DONE
