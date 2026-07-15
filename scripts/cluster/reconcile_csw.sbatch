#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=recon_csw
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/recon_csw_%j.log
set -euo pipefail
cd /burg-archive/glab/users/pg2328/legoESM_fv3grid
export PYTHONPATH=packages/core:packages/atmosphere:packages/ocean:packages/land:packages/seaice:src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  scripts/validate/fv3_native/reconcile_production_csw.py
echo "RECON_DONE"
