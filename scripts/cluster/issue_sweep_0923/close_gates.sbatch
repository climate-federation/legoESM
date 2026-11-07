#!/bin/bash
#SBATCH --job-name=sweep0923_gates
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/sweep0923_gates.%j.out
# Issue sweep 2026-09-23: three open issues look substantially delivered on
# main already.  Run each one's own acceptance gate so the issue is closed on a
# receipt, not on a code reading.
#   #1321 item 4 -- single-rank vs 2-rank bit-equivalence for MPAS land + qv filter
#   #1675 sites 2 -- mixed-precision storage dtype survives N steps
#   #1712        -- the subgrid-orography scale-decomposition guard
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "HEAD: $(git -C "$WT" log --oneline -1)"

echo "########## GATE #1675 -- mixed precision consistency (sites 1-3) ##########"
"$PY" -m pytest -q tests/unit/test_precision_modes.py tests/land/unit/test_carbon_storage_dtype.py \
     tests/unit/test_precision_leaf_dtypes.py 2>&1 | tail -12

echo "########## GATE #1675 site 1 -- CG Helmholtz residual at high kappa ##########"
"$PY" -m pytest -q tests/ -k "helmholtz or cg_helm" 2>&1 | tail -12

echo "########## GATE #1712 -- subgrid orography scale decomposition ##########"
"$PY" -m pytest -q tests/ -k "subgrid_orograph or sso_scale or scale_decomposition" 2>&1 | tail -12

echo "########## GATE #1321 item 4 -- MPAS land, single rank ##########"
"$PY" -m pytest -q tests/distributed/test_mpas_multilayer_land_mpi.py 2>&1 | tail -12

echo "########## GATE #1321 item 4 -- MPAS land + qv filter, 2 ranks ##########"
mpirun -np 2 "$PY" -m pytest -q -x tests/distributed/test_mpas_multilayer_land_mpi.py 2>&1 | tail -25

echo "########## GATE #1321 item 2 -- qv del2 filter distributed ##########"
"$PY" -m pytest -q tests/unit/test_mpas_qv_smoothing.py 2>&1 | tail -8

echo DONE
