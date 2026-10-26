#!/bin/bash
#SBATCH --job-name=i1675_cg
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/i1675_cg.%j.out
# #1675 site 1 acceptance: does promoting the five Helmholtz metric arrays to
# the control dtype clear the CG residual floor at kappa ~ 1e6, and does the
# FORWARD error follow?  The probe carries its own non-vacuity control (an arm
# with the promotion neutralised).
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "HEAD: $(git -C "$WT" log --oneline -1)  dirty=$(git -C "$WT" status --porcelain | wc -l)"

for K in 1e4 1e6; do
  echo "########## kappa = $K ##########"
  "$PY" -u scripts/validate/cg_helmholtz_mixed_precision_1675.py --n 24 --kappa "$K"
done

echo "########## regression: the existing CG suite must stay green ##########"
"$PY" -m pytest -q tests/atmosphere/hydrostatic/unit/test_semi_implicit_cdgrid.py 2>&1 | tail -8
echo DONE
