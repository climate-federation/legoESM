#!/bin/bash
#SBATCH --job-name=mpi_step_diag
#SBATCH --output=results/latlon_mpi/diag_%j.out
#SBATCH --error=results/latlon_mpi/diag_%j.err
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail
REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "${REPO_ROOT}"
mkdir -p results/latlon_mpi

PY="${HOME}/.venvs/legoesm-mpi/bin/python"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1

${PY} scripts/tmp/_diag_mpi_step_nans.py
