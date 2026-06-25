#!/bin/bash
#SBATCH --job-name=lego-amip-2yr
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/work/bd1083/b309178/logs/lego-amip-2yr-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/lego-amip-2yr-%j.err
#SBATCH --mail-type=FAIL,END

# 730-day (2-year) AMIP run with gray radiation.
# Gray: ~10.6 s/day → 730 days ≈ 130 min wall time.
# SST/SIC, GHG, solar, ozone, rayleigh GWD all active.
# Aerosol/volcanic inactive with gray radiation (files still checked at start).

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
OUTPUT=/scratch/b/b309178/amip_2yr_gray_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM 2-year AMIP (gray radiation)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Output:  $OUTPUT"
echo "Started: $(date)"
echo "=============================="

source /work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh
conda activate diffesm

cd "$REPO"

$PYTHON scripts/run_amip_levante.py \
    --year 1979 \
    --days 730 \
    --dt 150 \
    --ic-zarr "$IC_ZARR" \
    --radiation gray \
    --diag-days 5 \
    --checkpoint-days 73 \
    --gravity-wave-drag rayleigh \
    --cmip-output \
    --monthly-means \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
