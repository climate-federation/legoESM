#!/bin/bash
#SBATCH --job-name=legoesm-atm-scale
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
##SBATCH --partition=compute
##SBATCH --account=pg2328
##SBATCH --gres=gpu:4

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/slurm

# Optional: load your site modules here.
# module purge
# module load python/3.11 cuda openmpi

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python in ${PWD}" >&2
  exit 2
fi

source .venv/bin/activate

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Optional pip args for site-specific wheels/pinning.
# Example (CUDA-enabled JAX wheels):
# export LEGOESM_PIP_INSTALL_ARGS="-U --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
export LEGOESM_PIP_INSTALL_ARGS="${LEGOESM_PIP_INSTALL_ARGS:-}"

# Example 1 (CPU + MPI scaling only):
# .venv/bin/python scripts/run_ginsburg_atmos_scaling.py \
#   --modes cpu \
#   --repeats 3 \
#   --cpu-scaling-devices 1,2,4,8 \
#   --cpu-mpi-ranks 1,2,4,8 \
#   --output-root "results/ginsburg_atm_scaling_cpu_${SLURM_JOB_ID}" \
#   --mpi-launcher mpirun \
#   --install-missing \
#   --pip-install-extra-args "${LEGOESM_PIP_INSTALL_ARGS}"

# Example 2 (GPU + MPI scaling on a GPU node):
.venv/bin/python scripts/run_ginsburg_atmos_scaling.py \
  --modes gpu \
  --repeats 3 \
  --gpu-scaling-devices 1,2,4 \
  --gpu-mpi-ranks 1,2,4 \
  --output-root "results/ginsburg_atm_scaling_gpu_${SLURM_JOB_ID}" \
  --mpi-launcher mpirun \
  --install-missing \
  --pip-install-extra-args "${LEGOESM_PIP_INSTALL_ARGS}"
