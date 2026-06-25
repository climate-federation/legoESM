#!/bin/bash
#SBATCH --job-name=legoesm-amip
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/work/bd1083/b309178/logs/legoesm-amip-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/legoesm-amip-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# legoESM AMIP run — Levante GPU node
#
# Physics stack:
#   RRTMG radiation, Sundqvist clouds (BL cloud fix rh_crit_bl=0.55/sigma=0.85),
#   Sundqvist microphysics, SBM convection, Louis turbulence, Rayleigh GWD
#
# Timing reference (C48/L40, dt=150s, 4×A100-80GB, --production-profile):
#   rad_update_steps = floor(3600/150) = 24  (1-hour radiation cadence)
#   ~26 simulated days per wall-clock hour (from job 25023151 baseline)
#   270-day run: ~10.4h → 12h limit gives ~1.6h margin
#   Remaining 95 days (days 271-365) submitted via submit_amip_levante_restart.sh
#
# Usage:
#   sbatch scripts/submit_amip_levante.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python

# --- run parameters ---
YEAR=1979
DAYS=270
DT=150
IC_ZARR=/scratch/b/b309178/era5_ic_1979-01-01.zarr
OUTPUT=/scratch/b/b309178/amip_${YEAR}_${DAYS}d_${SLURM_JOB_ID}

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM AMIP run"
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
# Disable pre-allocation so JAX grows memory on demand (avoids OOM on 4 GPUs)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# Deterministic GPU ordering
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Prevent CPU thread oversubscription from BLAS/OpenMP
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
