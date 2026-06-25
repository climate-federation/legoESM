#!/bin/bash
#SBATCH --job-name=legoesm-dryhssp
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-dryhssp-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-dryhssp-%j.err
#SBATCH --mail-type=FAIL,END

# Dry Held-Suarez on the SPECTRAL dycore — grid-comparison control vs the
# cubed-sphere super-rotation. If spectral gives midlat jets (~28 m/s @45deg) +
# surface westerlies, the cube dycore is the culprit. T42 ~ C32-equivalent.
set -euo pipefail
NMAX=${1:-42}; DAYS=${2:-200}
REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge; ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python
OUT=/scratch/b/b309165/dry_hs_spectral_T${NMAX}_${DAYS}d_${SLURM_JOB_ID}.npz
mkdir -p /scratch/b/b309165/legoesm_logs
echo "Dry HS SPECTRAL T${NMAX} ${DAYS}d  Job ${SLURM_JOB_ID}  $(date)  Out $OUT"
source $CONDA/etc/profile.d/conda.sh; conda activate $ENV
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"; export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader || true
cd "$REPO"
$PYTHON scripts/run/run_dry_held_suarez_spectral.py --nmax "$NMAX" --nlev 30 --days "$DAYS" --out "$OUT"
echo "Finished $(date)  Output $OUT"
