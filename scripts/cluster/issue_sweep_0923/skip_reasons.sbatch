#!/bin/bash
#SBATCH --job-name=sweep0923_skips
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/sweep0923_skips.%j.out
# A skip is not a pass.  The #1729 and #1712 gates came back "3 passed, 2
# skipped" and "21 passed, 2 skipped"; print WHY each one skipped before either
# issue is closed on them.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

echo "########## #1729 skip reasons ##########"
"$PY" -m pytest -q -rs tests/ocean/fidelity/test_nemo_dino_fromrest_gate.py 2>&1 | tail -12
echo "########## #1712 skip reasons ##########"
"$PY" -m pytest -q -rs tests/unit/test_amip_production_subgrid_orography.py 2>&1 | tail -12
echo DONE
