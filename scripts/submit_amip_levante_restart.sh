#!/bin/bash
#SBATCH --job-name=legoesm-amip-rst
#SBATCH --account=bd1083
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=6:00:00
#SBATCH --output=/work/bd1083/b309178/logs/legoesm-amip-%j.out
#SBATCH --error=/work/bd1083/b309178/logs/legoesm-amip-%j.err
#SBATCH --mail-type=FAIL,END

# ---------------------------------------------------------------------------
# legoESM AMIP restart — days 271-365 (continuation of submit_amip_levante.sh)
#
# Usage:
#   sbatch scripts/submit_amip_levante_restart.sh \
#       --export=RESTART_DIR=/scratch/b/b309178/amip_1979_270d_<JOBID>
#
# Or set RESTART_DIR before calling sbatch:
#   export RESTART_DIR=/scratch/b/b309178/amip_1979_270d_<JOBID>
#   sbatch scripts/submit_amip_levante_restart.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoESM
PYTHON=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python

# --- run parameters ---
YEAR=1979
START_DAY=270
DAYS=95
DT=150

# RESTART_DIR must be set by caller (the output dir from the 270-day run)
: "${RESTART_DIR:?ERROR: RESTART_DIR must be set to the 270-day output directory}"

RESTART_CHECKPOINT="${RESTART_DIR}/checkpoint_day270.npz"
OUTPUT="${RESTART_DIR}"   # append into same directory

mkdir -p /work/bd1083/b309178/logs

echo "=============================="
echo "legoESM AMIP restart run"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Year:    $YEAR  Start day: $START_DAY  Days: $DAYS  dt: ${DT}s"
echo "Restart: $RESTART_CHECKPOINT"
echo "Output:  $OUTPUT"
echo "Started: $(date)"
echo "=============================="

if [[ ! -f "$RESTART_CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $RESTART_CHECKPOINT"
    ls "$RESTART_DIR"/*.npz 2>/dev/null || echo "(no .npz files found)"
    exit 1
fi

# --- environment ---
source /work/bd1083/b309178/mambaforge/etc/profile.d/conda.sh
conda activate diffesm

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
    --restart-from "$RESTART_CHECKPOINT" \
    --restart-start-day "$START_DAY" \
    --diag-days 5 \
    --checkpoint-days 30 \
    --cmip-output \
    --monthly-means \
    --production-profile \
    --output "$OUTPUT"

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
