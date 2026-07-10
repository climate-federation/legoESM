#!/bin/bash
# WeatherBench scale training on DKRZ Levante (SLURM, 4x A100-80GB/node, IB-HDR200).
# Submit (N nodes):  sbatch --nodes=<N> train_wb.slurm
# One task (=1 GPU) per rank; ranks average gradients via mpi4jax (data-parallel).
#SBATCH --job-name=wb_scale_train
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=wb_scale_train_%j.out
set -euo pipefail

# Levante env: repo root (/work/bd1083/$USER/legoESM), modules, UCX CUDA-aware fabric.
source scripts/cluster/scaling_levante/_env.sh
module load python3/openmpi/cuda
conda activate legoesm-gpu
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

echo "[$(date)] Levante WB scale train: ${SLURM_NTASKS} ranks (1 GPU each)"

# one rank per GPU (--gpu-bind=single:1); SLURM_LOCALID pins rank->GPU in the code.
# WB_CORE selects the training core (#817): latlon (default) or spectral
# (Gaussian semi-implicit — bounded adjoint, no pole dt clamp).
srun --ntasks="${SLURM_NTASKS}" --gpu-bind=single:1 --cpu-bind=cores --mpi=pmix \
  python scripts/run/train_weatherbench_scale.py \
    --config config/wb/scale/train_07deg.yaml \
    --mode "${WB_MODE:-neural_gcm}" \
    --training-core "${WB_CORE:-latlon}" \
    --out "${SCRATCH:-.}/wb_scale_${WB_MODE:-neural_gcm}_${WB_CORE:-latlon}" \
    --resume --eval-wb2
echo "[$(date)] done"
