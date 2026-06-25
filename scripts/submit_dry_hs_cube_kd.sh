#!/bin/bash
#SBATCH --job-name=legoesm-dryhs
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/b/b309165/legoesm_logs/legoesm-dryhs-%j.out
#SBATCH --error=/scratch/b/b309165/legoesm_logs/legoesm-dryhs-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# DRY HELD-SUAREZ dynamical-core jet test (diagnose the AMIP wind collapse).
# Built on the SAME cd-grid dycore the AMIP runs use (create_atmosphere_dycore
# from the orogwd experiment config), HS forcing only, isothermal-rest IC.
#
# Question: does the dycore MAINTAIN a realistic eddy-driven jet (~28 m/s @45deg)?
#   - healthy jet  => dycore is fine; AMIP collapse is a physics/IC problem.
#   - weak jet     => dycore over-dissipates (resolution-test C32 vs C96).
#
# Pass resolution as $1 (default 32).  200 days for jet equilibration.
# Usage:  sbatch scripts/submit_dry_hs_cube_kd.sh 32
#         sbatch scripts/submit_dry_hs_cube_kd.sh 96
# ---------------------------------------------------------------------------
set -euo pipefail

N=${1:-32}
DAYS=${2:-200}
HSCALE=${3:-1.0}     # hyperdiff scale (over-dissipation probe)
DSCALE=${4:-1.0}     # div-damp scale
VCOORD=${5:-hybrid}  # hybrid | sigma (sigma matches spectral control)
AHSCALE=${7:-1.0}   # Laplacian viscosity A_h scale (eddy-killer test)
REPO=/work/bd1083/b309178/diffESM/legoesm_kd
CONDA=/work/bd1083/b309165/mambaforge
ENV=legoesm_kd
PYTHON=$CONDA/envs/$ENV/bin/python
NLEV=${6:-45}
TAG=c${N}_l${NLEV}_${DAYS}d_h${HSCALE}_d${DSCALE}_${VCOORD}_ah${AHSCALE}
OUT=/scratch/b/b309165/dry_hs_${TAG}_${SLURM_JOB_ID}.npz

mkdir -p /scratch/b/b309165/legoesm_logs

echo "=============================="
echo "Dry Held-Suarez  C${N} / L${NLEV}  ${DAYS}d  [dycore jet test]"
echo "Job: ${SLURM_JOB_ID}  Node: ${SLURMD_NODENAME}  Out: $OUT"
echo "Started: $(date)"
echo "=============================="

source $CONDA/etc/profile.d/conda.sh
conda activate $ENV

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
cd "$REPO"

$PYTHON scripts/run/run_dry_held_suarez_cube.py \
    --n "$N" --nlev "$NLEV" --days "$DAYS" \
    --hyperdiff-scale "$HSCALE" --div-damp-scale "$DSCALE" --vcoord "$VCOORD" --ah-scale "$AHSCALE" --out "$OUT"

echo "Finished: $(date)"
echo "Output: $OUT"
