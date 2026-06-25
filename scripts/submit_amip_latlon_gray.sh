#!/bin/bash
#SBATCH --job-name=lego-latlon-gray
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/work/bd1083/b309178/logs/lego-latlon-gray-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/lego-latlon-gray-%j.err
#SBATCH --mail-type=FAIL,END

# 120-day AMIP run on a 90×180 lat-lon C-grid (~2°, comparable to C48)
# with gray radiation.  dt=90s keeps CFL safe at high latitudes where
# the longitudinal grid spacing shrinks as cos(lat).
#
# Usage:
#   sbatch scripts/submit_amip_latlon_gray.sh

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
OUTPUT=/scratch/b/b309178/amip_latlon_gray_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM AMIP lat-lon gray 120d"
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
    --days 120 \
    --dt 90 \
    --grid-type latlon \
    --discretization latlon_cgrid \
    --resolution 90 \
    --ic-zarr "$IC_ZARR" \
    --radiation gray \
    --diag-days 5 \
    --checkpoint-days 60 \
    --monthly-means \
    --cmip-output \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
