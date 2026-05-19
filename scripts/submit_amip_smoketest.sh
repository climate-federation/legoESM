#!/bin/bash
#SBATCH --job-name=lego-smoketest
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/work/bd1083/b309178/logs/lego-smoketest-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/lego-smoketest-%j.err
#SBATCH --mail-type=FAIL,END

# 5-day AMIP smoke test validating post-calibration fixes:
#   - Cloud optical depth bug fix (cf² → cf scaling)
#   - rayleigh gravity-wave drag enabled
#   - hfss/hfls/rsdt/psl in regular output
#   - profile_u geographic rotation
#   - CMIP output pathway active

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
OUTPUT=/scratch/b/b309178/amip_smoketest_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM 5-day smoke test"
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
    --days 5 \
    --dt 150 \
    --ic-zarr "$IC_ZARR" \
    --diag-days 1 \
    --checkpoint-days 5 \
    --gravity-wave-drag rayleigh \
    --cmip-output \
    --monthly-means \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
