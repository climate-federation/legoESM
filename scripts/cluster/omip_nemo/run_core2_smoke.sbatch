#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_c2run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/c2run_%j.out

# Smoke-test the CORE-II legoESM tripole runner: validate the full chain
# (eORCA1 mesh -> mask/bathy -> tripole grid/model/state -> CORE-II forcing via
# compute_omip2_surface_forcing -> model.step(surface_forcing=...)) on the GPU and
# benchmark steps/s to size the real multi-year run.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1; date
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py --smoke --dt 600 --diag-every-days 1 2>&1 | tail -40
echo "c2run_rc=${PIPESTATUS[0]}"
echo "=== DONE ==="
