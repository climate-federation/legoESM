#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scan354
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/_test_scan354_%j.out
set +e; cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q tests/ocean/unit/test_omip2_forcing_scan.py
echo "scan354_rc=$?"; echo DONE; date
