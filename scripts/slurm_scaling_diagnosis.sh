#!/bin/bash
#===============================================================================
# SLURM job script for legoESM scaling diagnostics on Levante (DKRZ)
#
# Runs systematic weak and strong scaling diagnostics across GPU counts,
# producing JSON reports for offline analysis by Claude Code.
#
# Usage:
#   # Full diagnostics on 1 node (4 A100s):
#   sbatch scripts/slurm_scaling_diagnosis.sh
#
#   # Strong scaling across 2 nodes (8 A100s):
#   sbatch --nodes=2 --ntasks=8 scripts/slurm_scaling_diagnosis.sh
#
#   # Quick sanity check:
#   DIAG_MODE=quick sbatch scripts/slurm_scaling_diagnosis.sh
#
#   # Sweep multiple GPU counts (submit array):
#   sbatch --array=1,2,3,6 scripts/slurm_scaling_diagnosis.sh
#
# Environment variables:
#   DIAG_MODE     - Diagnostic mode: full, quick, halo-only, reduction, roofline
#                   (default: full)
#   DIAG_N        - Per-face resolution (default: 48)
#   DIAG_NLEV     - Vertical levels (default: 40)
#   DIAG_PHYSICS  - Physics level: none, held_suarez (default: none)
#   DIAG_PREC     - Precision: float32, float64 (default: float64)
#   DIAG_GRID     - Grid type (default: cubed-sphere)
#   DIAG_OUTDIR   - Output directory (default: results/scaling_diagnosis)
#   XLA_PROFILE   - Set to 1 to capture XLA/TensorBoard traces
#===============================================================================

#SBATCH --job-name=legoesm-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/scaling_diag_%j_%a.out
#SBATCH --error=logs/scaling_diag_%j_%a.err
#SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration from environment (with defaults)
# ---------------------------------------------------------------------------
MODE="${DIAG_MODE:-full}"
N="${DIAG_N:-48}"
NLEV="${DIAG_NLEV:-40}"
PHYSICS="${DIAG_PHYSICS:-none}"
PRECISION="${DIAG_PREC:-float64}"
GRID="${DIAG_GRID:-cubed-sphere}"
OUTDIR="${DIAG_OUTDIR:-results/scaling_diagnosis}"
XLA_PROF="${XLA_PROFILE:-0}"

# For SLURM array jobs: use task ID as GPU count
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    NGPUS=$SLURM_ARRAY_TASK_ID
else
    NGPUS=$SLURM_NTASKS
fi

# Compute total tasks for multi-node
TOTAL_TASKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

echo "==============================================="
echo "  legoESM Scaling Diagnostics"
echo "==============================================="
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Nodes:       $SLURM_NNODES ($SLURM_NODELIST)"
echo "  Tasks:       $TOTAL_TASKS"
echo "  GPUs/node:   ${SLURM_GPUS_PER_NODE:-4}"
echo "  Mode:        $MODE"
echo "  Grid:        $GRID N=$N L=$NLEV"
echo "  Physics:     $PHYSICS"
echo "  Precision:   $PRECISION"
echo "  Output:      $OUTDIR"
echo "==============================================="

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
mkdir -p logs

# Load modules (adjust for your Levante setup)
module purge
module load python3/2024.01-gcc-11.2.0
module load openmpi/4.1.6-gcc-11.2.0
module load cuda/12.3.2

# Activate venv
source .venv/bin/activate

# JAX configuration.  Iter 25: ``gpu,cpu`` is the legacy alias and is
# rejected by JAX 0.10+ (``Backend 'rocm' is not in the list of
# known backends: ['cpu', 'tpu', 'cuda']``).  Default to ``cuda,cpu``
# on Levante's NVIDIA partition.  Iter 26: respect a value already
# exported by ``sbatch --export=JAX_PLATFORMS=...`` so ROCm sites
# don't have to edit this script — the documented override now
# actually works.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.90"
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"

if [ "$PRECISION" = "float64" ]; then
    export JAX_ENABLE_X64=1
fi

# Enable MPI profiling
export LEGOESM_PROFILE_MPI=1

# Iter 28: GPU binding is applied inside each ``srun`` task by SLURM
# (``--gpus-per-task=1``) and the Python entry point's
# ``_configure_env`` (which reads ``SLURM_LOCALID`` per-rank).
# Exporting ``CUDA_VISIBLE_DEVICES`` here at wrapper scope binds
# *every* rank to the launcher's local id (typically 0), pinning all
# ranks to GPU 0 and serialising the run.

# NCCL tuning for Levante
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
DIAG_CMD="python scripts/run_scaling_diagnosis.py \
    --mode $MODE \
    --grid $GRID \
    --n $N \
    --nlev $NLEV \
    --precision $PRECISION \
    --physics $PHYSICS \
    --output-dir $OUTDIR"

if [ "$XLA_PROF" = "1" ]; then
    DIAG_CMD="$DIAG_CMD --xla-profile"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo ""
echo "Running: srun $DIAG_CMD"
echo ""

srun --ntasks=$TOTAL_TASKS \
     --gpus-per-task=1 \
     $DIAG_CMD

echo ""
echo "==============================================="
echo "  Diagnostics complete."
echo "  Results: $OUTDIR"
echo "==============================================="

# ---------------------------------------------------------------------------
# Also run the standard scaling benchmark for reference
# ---------------------------------------------------------------------------
echo ""
echo "Running reference scaling benchmark..."
echo ""

srun --ntasks=$TOTAL_TASKS \
     --gpus-per-task=1 \
     python scripts/run_levante_gpu_scaling.py \
     --grid $GRID \
     --mode both \
     --precision $PRECISION \
     --n-gpus $TOTAL_TASKS \
     --n-levels $NLEV \
     --output-dir "$OUTDIR/scaling_benchmark"

echo ""
echo "All done. Transfer $OUTDIR/ to your local machine for analysis:"
echo "  scp -r levante:$PWD/$OUTDIR/ ."
echo "  python scripts/analyze_scaling_results.py $OUTDIR/<timestamp>/ --format markdown"
