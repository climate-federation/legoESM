#!/bin/bash
#SBATCH --job-name=lego-75d-6node
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/work/bd1083/b309178/logs/lego-75d-6node-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/lego-75d-6node-%j.err
#SBATCH --mail-type=FAIL,END

# 75-day RRTMG AMIP run: 6 nodes × 4 A100-80GB = 24 GPUs total.
# Cubed-sphere: 6 faces / 6 MPI ranks = 1 face per rank.
# Expected speedup: ~5× over single-node (3 GPUs → 24 GPUs active on physics).
# 75-day RRTMG: ~10.5h (single-node) → ~2.1h (6-node estimate).
#
# Usage:
#   sbatch scripts/submit_amip_75d_6node.sh

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
OUTPUT=/scratch/b/b309178/amip_75d_6node_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM AMIP 75-day RRTMG 6-node MPI"
echo "Job ID:  $SLURM_JOB_ID"
echo "Nodes:   $SLURM_NODELIST"
echo "Output:  $OUTPUT"
echo "Started: $(date)"
echo "=============================="

source /work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh
conda activate diffesm

cd "$REPO"

$PYTHON scripts/run_amip_levante.py \
    --year 1979 \
    --days 75 \
    --dt 150 \
    --ic-zarr "$IC_ZARR" \
    --diag-days 5 \
    --checkpoint-days 75 \
    --cmip-output \
    --monthly-means \
    --distributed \
    --n-ranks 6 \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
