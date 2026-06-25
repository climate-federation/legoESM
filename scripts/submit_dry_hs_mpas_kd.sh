#!/bin/bash
#SBATCH --job-name=legoesm-dryhsmp
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-dryhsmp-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-dryhsmp-%j.err
#SBATCH --mail-type=FAIL,END
set -euo pipefail
LEVEL=${1:-5}; DAYS=${2:-200}; ND2=${3:-0.0}; ND4DIV=${4:-86400.0}
REPO=/work/bd1083/b309178/diffESM/legoesm_kd; CONDA=/work/bd1083/b309165/mambaforge; ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python
OUT=/scratch/b/b309165/dry_hs_mpas_L${LEVEL}_${DAYS}d_nd2${ND2}_${SLURM_JOB_ID}.npz
mkdir -p /scratch/b/b309165/legoesm_logs
echo "Dry HS MPAS level${LEVEL} ${DAYS}d nu_del2_scale=${ND2}  Job ${SLURM_JOB_ID}  $(date)"
source $CONDA/etc/profile.d/conda.sh; conda activate $ENV
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"; export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90; export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name --format=csv,noheader || true
cd "$REPO"
$PYTHON scripts/run/run_dry_held_suarez_mpas.py --level "$LEVEL" --nlev 30 --days "$DAYS" --nu-del2-scale "$ND2" --nu-del4-div "$ND4DIV" --out "$OUT"
echo "Finished $(date)  Output $OUT"
