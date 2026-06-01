#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=convtest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/_test_convection_%j.out
set +e; cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
echo; echo "=== [1] grid-agnostic column physics ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q tests/ocean/unit/test_column_physics_grid_agnostic.py
echo "col_rc=$?"
echo; echo "=== [2] convection regression (implicit vertical mixing) ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q tests/ocean/unit/test_implicit_vertical_mixing.py
echo "imvm_rc=$?"
echo; echo "=== [3] runner --convection arg wired (argparse smoke) ==="
JAX_PLATFORMS=cpu $PY scripts/run_omip_core2.py --help 2>&1 | grep -A2 "\-\-convection" | head -8
echo "help_rc=$?"
echo DONE; date
