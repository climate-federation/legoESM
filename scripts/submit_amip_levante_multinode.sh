#!/bin/bash
#SBATCH --job-name=legoesm-amip-mn
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/work/bd1083/b309178/logs/legoesm-amip-mn-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/legoesm-amip-mn-%j.err
#SBATCH --mail-type=FAIL,END

# Multi-node AMIP run: 2 nodes × 3 active GPUs = 6 A100-80GB total (valid
# cubed-sphere count: 6 faces / 6 GPUs = 1 face per GPU).
# run_amip_levante.py --distributed spawns: srun --ntasks=2 python run_amip.py
# run_amip.py auto-detects SLURM_NTASKS and initialises MPI distributed mode.
#
# Expected speedup: ~1.8× over single-node (3 GPUs → 6 GPUs).
# 75-day RRTMG run: ~10.5h (single) → ~6h (2-node).
#
# Usage:
#   sbatch scripts/submit_amip_levante_multinode.sh
#   sbatch scripts/submit_amip_levante_multinode.sh 365   # 1-year run

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
DAYS=${1:-75}
OUTPUT=/scratch/b/b309178/amip_mn_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM AMIP multi-node (2 nodes, 6 GPUs)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Nodes:   $SLURM_NODELIST"
echo "Days:    $DAYS"
echo "Output:  $OUTPUT"
echo "Started: $(date)"
echo "=============================="

source /work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh
conda activate diffesm

cd "$REPO"

# run_amip_levante.py --distributed wraps the inner command with:
#   srun --ntasks=2 --ntasks-per-node=1 python run_amip.py --distributed ...
# run_amip.py then auto-detects SLURM_NTASKS=2 and initialises MPI.
$PYTHON scripts/run_amip_levante.py \
    --year 1979 \
    --days "$DAYS" \
    --dt 150 \
    --ic-zarr "$IC_ZARR" \
    --diag-days 5 \
    --checkpoint-days 30 \
    --cmip-output \
    --monthly-means \
    --distributed \
    --n-ranks 2 \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
