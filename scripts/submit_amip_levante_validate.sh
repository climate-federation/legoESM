#!/bin/bash
#SBATCH --job-name=legoesm-amip-val
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/work/bd1083/b309178/logs/legoesm-amip-val-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/legoesm-amip-val-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# legoESM AMIP — 30-day VALIDATION run (Levante GPU node)
#
# Purpose: confirm the RRTMGP radiative-heating-rate sign fix (commit
# 9c7ff7f0) stops the thermal runaway.  The previous (broken) run reached
# T_mean = 281.9 K by day 30 on its way to a NaN blowup at day 125.  With
# correct radiation the free troposphere cools radiatively and convection
# balances it, so T_mean should stay near its ~261 K initial value and the
# day-by-day diagnostics should plateau instead of climbing.
#
# Identical physics/grid to the production submit_amip_levante.sh, only the
# run length differs (30 days vs 270).
#
# Usage:
#   sbatch scripts/submit_amip_levante_validate.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python

# --- run parameters ---
YEAR=1979
DAYS=30
DT=150
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
OUTPUT=/scratch/b/b309178/amip_${YEAR}_validate_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM AMIP 30-day validation"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Year:    $YEAR  Days: $DAYS  dt: ${DT}s"
echo "Output:  $OUTPUT"
echo "Started: $(date)"
echo "=============================="

# --- environment ---
source /work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh
conda activate diffesm

# JAX/XLA settings — must be set before any Python/JAX import
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo ""
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo ""
echo "--- JAX devices ---"
$PYTHON -c "import jax; print(f'JAX {jax.__version__}: {jax.devices()}')" || true
echo ""

cd "$REPO"

$PYTHON scripts/run_amip_levante.py \
    --year "$YEAR" \
    --days "$DAYS" \
    --dt "$DT" \
    --ic-zarr "$IC_ZARR" \
    --diag-days 5 \
    --checkpoint-days 30 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
