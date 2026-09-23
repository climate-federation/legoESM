#!/bin/bash
#SBATCH --job-name=teostol
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/teostol_%j.out
#
# Calibrate the TEOS-10 derivative-sweep tolerance from MEASUREMENT, and
# measure how strong a coefficient error the sweep can actually resolve.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u \
  scripts/validate/ocean_fidelity/teos10_coeff_sensitivity.py
echo "=== rc=$?"
