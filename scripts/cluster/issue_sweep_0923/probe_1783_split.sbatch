#!/bin/bash
#SBATCH --job-name=p1783_split
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/p1783_split.%j.out
# #1783 re-examination. A reviewer REJECTED the pin that landed today, arguing
# the dry-adiabatic correction is required physics wherever the surface
# temperature is real -- and the unified lane's land slab IS real -- so pinning
# it off bakes in an opposite-sign bias rather than fixing anything.
# Reading the test showed its reference law calls the BARE surface-flux
# routine, with no reference height at all, which would make the test stale
# rather than the physics wrong. Measure before touching either.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
"$PY" -u scripts/tmp/_probe_1783_law_split.py
echo DONE
