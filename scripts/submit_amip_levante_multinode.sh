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

# Multi-node AMIP run.  Number of MPI ranks is taken from SLURM_NTASKS,
# so the same script handles 2-, 3-, or 6-rank cubed-sphere layouts.
# Cubed-sphere constraint: ranks must divide 6 evenly (1, 2, 3, or 6).
#
# Usage:
#   # 2-node default (75 days):
#   sbatch scripts/submit_amip_levante_multinode.sh
#   # 2-node, 365 days:
#   sbatch scripts/submit_amip_levante_multinode.sh 365
#   # 6-node, 75 days, 4h limit (one face per rank, ~5× speedup):
#   sbatch --nodes=6 --ntasks=6 --time=04:00:00 \
#       scripts/submit_amip_levante_multinode.sh 75

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
DAYS=${1:-75}
OUTPUT=/scratch/b/b309178/amip_mn_${SLURM_NTASKS}n_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM AMIP multi-node ($SLURM_NTASKS ranks)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Nodes:   $SLURM_NODELIST"
echo "Days:    $DAYS"
echo "Output:  $OUTPUT"
echo "Started: $(date)"
echo "=============================="

source /work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh
conda activate diffesm

cd "$REPO"

# run_amip_levante.py --distributed wraps the inner command with srun;
# rank count is read from SLURM_NTASKS (no need to pass --n-ranks).
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
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
