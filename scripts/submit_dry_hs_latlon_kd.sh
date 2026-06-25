#!/bin/bash
#SBATCH --job-name=legoesm-dryhsll
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-dryhsll-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-dryhsll-%j.err
#SBATCH --mail-type=FAIL,END
set -euo pipefail
NLAT=${1:-64}; DAYS=${2:-100}
REPO=/work/bd1083/b309178/diffESM/legoesm_kd; CONDA=/work/bd1083/b309165/mambaforge; ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python
OUT=/scratch/b/b309165/dry_hs_latlon_${NLAT}_${DAYS}d_${SLURM_JOB_ID}.npz
mkdir -p /scratch/b/b309165/legoesm_logs
echo "Dry HS LATLON ${NLAT} ${DAYS}d  Job ${SLURM_JOB_ID}  $(date)"
source $CONDA/etc/profile.d/conda.sh; conda activate $ENV
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"; export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90; export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader || true
cd "$REPO"
$PYTHON scripts/run/run_dry_held_suarez_latlon.py --nlat "$NLAT" --nlev 30 --days "$DAYS" --out "$OUT"
echo "Finished $(date)  Output $OUT"
