#!/bin/bash
#SBATCH --job-name=guardfix
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/guardfix_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -m pytest -q \
  tests/ocean/unit/test_tke_nemo_identity.py
echo "=== rc=$? ==="
