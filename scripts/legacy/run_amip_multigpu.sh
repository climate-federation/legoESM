#!/bin/bash
# ==========================================================================
# Launch AMIP simulation on multiple GPUs using MPI
#
# Distributes the cubed-sphere grid across GPUs. The cubed-sphere has 6
# faces, so valid GPU counts must divide into the 6-face topology:
#   Valid: 1, 2, 3, 6, 12, 24, ...  (1-3 divide 6; >6 must be a multiple of 6)
#
# Usage:
#   ./scripts/run_amip_multigpu.sh                   # Auto-detect GPUs
#   ./scripts/run_amip_multigpu.sh --ngpus 4         # Use 4 GPUs (must be valid count)
#   ./scripts/run_amip_multigpu.sh --ngpus 6         # One GPU per cubed-sphere face
#   RESOLUTION=96 NLEV=60 ./scripts/run_amip_multigpu.sh  # Override via env vars
#
# Environment variable overrides:
#   NGPUS           Number of GPUs (default: auto-detect via nvidia-smi)
#   RESOLUTION      Cubed-sphere resolution N (default: 48)
#   NLEV            Number of vertical levels (default: 40)
#   DURATION_DAYS   Integration length in days (default: 365)
#   DT              Time step in seconds (default: 600)
#   OUTPUT_DIR      Output directory (default: results/amip_multigpu_C${RESOLUTION})
#   FORCING_PATH    Path to SST forcing NetCDF (default: analytical)
#   DATASET         Forcing dataset preset (default: analytical)
#
# Requirements:
#   - mpi4py, mpi4jax installed in Python environment
#   - CUDA-capable GPUs with appropriate CUDA/cuDNN
#   - OpenMPI or MPICH
#   - JAX with GPU support (jaxlib[cuda12] or similar)
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
log_info()  { echo "[INFO]  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo "[WARN]  $(date '+%H:%M:%S') $*" >&2; }
log_error() { echo "[ERROR] $(date '+%H:%M:%S') $*" >&2; }

usage() {
    cat <<'USAGE'
Usage: run_amip_multigpu.sh [OPTIONS] [-- EXTRA_ARGS_FOR_RUN_AMIP]

Options:
  --ngpus N          Number of GPUs to use (default: auto-detect)
  --resolution N     Cubed-sphere resolution (default: 48)
  --nlev N           Number of vertical levels (default: 40)
  --days N           Integration length in days (default: 365)
  --dt SECONDS       Time step (default: 600)
  --forcing PATH     Path to SST forcing NetCDF file
  --dataset NAME     Forcing dataset preset: cobe, hadisst, analytical (default: analytical)
  --output DIR       Output directory
  --dry-run          Print the launch command without executing
  -h, --help         Show this help message

Extra arguments after '--' are passed directly to run_amip.py.

Examples:
  # Auto-detect GPUs, 1-year analytical forcing:
  ./scripts/run_amip_multigpu.sh

  # 6 GPUs, C96 resolution, COBE-SST2 forcing, 10-year run:
  ./scripts/run_amip_multigpu.sh --ngpus 6 --resolution 96 --days 3650 \
      --forcing /data/MODEL.SST.COBE-SST2.nc --dataset cobe

  # Pass extra flags to run_amip.py:
  ./scripts/run_amip_multigpu.sh --ngpus 6 -- --radiation rrtmg --diurnal-cycle
USAGE
}

# -------------------------------------------------------------------------
# Validate GPU count for cubed-sphere topology
# -------------------------------------------------------------------------
validate_gpu_count() {
    local n=$1
    case $n in
        1|2|3|6) return 0 ;;
    esac
    if [ "$n" -gt 6 ] && [ $((n % 6)) -eq 0 ]; then
        return 0
    fi
    return 1
}

# -------------------------------------------------------------------------
# Detect number of available GPUs
# -------------------------------------------------------------------------
detect_gpus() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    else
        echo 0
    fi
}

# -------------------------------------------------------------------------
# Default settings (can be overridden by env vars or CLI args)
# -------------------------------------------------------------------------
_NGPUS="${NGPUS:-}"
_RESOLUTION="${RESOLUTION:-48}"
_NLEV="${NLEV:-40}"
_DURATION_DAYS="${DURATION_DAYS:-365}"
_DT="${DT:-600}"
_OUTPUT_DIR="${OUTPUT_DIR:-}"
_FORCING_PATH="${FORCING_PATH:-}"
_DATASET="${DATASET:-analytical}"
_DRY_RUN=0
_EXTRA_ARGS=""

# -------------------------------------------------------------------------
# Parse command-line arguments
# -------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngpus)
            _NGPUS="$2"; shift 2 ;;
        --resolution)
            _RESOLUTION="$2"; shift 2 ;;
        --nlev)
            _NLEV="$2"; shift 2 ;;
        --days)
            _DURATION_DAYS="$2"; shift 2 ;;
        --dt)
            _DT="$2"; shift 2 ;;
        --forcing)
            _FORCING_PATH="$2"; shift 2 ;;
        --dataset)
            _DATASET="$2"; shift 2 ;;
        --output)
            _OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)
            _DRY_RUN=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift; _EXTRA_ARGS="$*"; break ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1 ;;
    esac
done

# -------------------------------------------------------------------------
# Auto-detect GPU count if not specified
# -------------------------------------------------------------------------
if [ -z "$_NGPUS" ]; then
    _NGPUS=$(detect_gpus)
    if [ "$_NGPUS" -eq 0 ]; then
        log_error "No CUDA GPUs detected via nvidia-smi."
        log_error "Set --ngpus explicitly or ensure nvidia-smi is available."
        exit 1
    fi
    log_info "Auto-detected $_NGPUS GPU(s)"
fi

# -------------------------------------------------------------------------
# Validate GPU count
# -------------------------------------------------------------------------
if ! validate_gpu_count "$_NGPUS"; then
    log_error "$_NGPUS GPUs cannot tile the 6 cubed-sphere faces."
    echo ""
    echo "Valid GPU counts for cubed-sphere decomposition:"
    echo "  Small: 1, 2, 3, 6"
    echo "  Large: any multiple of 6 (12, 24, 48, ...)"
    echo ""
    echo "Suggestion: use --ngpus 6 (one GPU per face) or --ngpus 1 (single GPU)"
    exit 1
fi

# -------------------------------------------------------------------------
# Validate prerequisites
# -------------------------------------------------------------------------
if ! command -v mpirun &>/dev/null; then
    log_error "mpirun not found. Install OpenMPI or MPICH."
    log_error "  conda install -c conda-forge openmpi mpi4py"
    log_error "  pip install mpi4py"
    exit 1
fi

if ! python -c "import mpi4py" 2>/dev/null; then
    log_error "mpi4py not found. Install with: pip install mpi4py"
    exit 1
fi

if ! python -c "import jax; assert jax.devices('gpu')" 2>/dev/null; then
    log_warn "JAX GPU backend not detected. The run may fall back to CPU."
fi

# -------------------------------------------------------------------------
# Set output directory
# -------------------------------------------------------------------------
if [ -z "$_OUTPUT_DIR" ]; then
    _OUTPUT_DIR="results/amip_multigpu_C${_RESOLUTION}"
fi

# -------------------------------------------------------------------------
# Build forcing arguments
# -------------------------------------------------------------------------
DATASET_ARGS="--dataset $_DATASET"
if [ -n "$_FORCING_PATH" ]; then
    DATASET_ARGS="$DATASET_ARGS --forcing-path $_FORCING_PATH"
elif [ "$_DATASET" != "analytical" ]; then
    log_error "--forcing (or FORCING_PATH env) is required when dataset is not 'analytical'"
    exit 1
fi

# -------------------------------------------------------------------------
# Set environment variables for multi-GPU JAX
# -------------------------------------------------------------------------
export JAX_ENABLE_X64=1
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_enable_async_collectives=true"

# Prevent JAX from pre-allocating all GPU memory (useful for multi-process)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Ensure each MPI rank sees only its assigned GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# -------------------------------------------------------------------------
# Print launch summary
# -------------------------------------------------------------------------
echo "=========================================="
echo "  legoESM Multi-GPU AMIP Launch"
echo "=========================================="
echo "  GPUs:         $_NGPUS"
echo "  Resolution:   C${_RESOLUTION} / L${_NLEV}"
echo "  Duration:     $_DURATION_DAYS days"
echo "  Time step:    ${_DT}s"
echo "  Dataset:      $_DATASET"
if [ -n "$_FORCING_PATH" ]; then
echo "  Forcing:      $_FORCING_PATH"
fi
echo "  Output:       $_OUTPUT_DIR"
if [ -n "$_EXTRA_ARGS" ]; then
echo "  Extra args:   $_EXTRA_ARGS"
fi
echo "=========================================="
echo ""

# -------------------------------------------------------------------------
# Build the launch command
# -------------------------------------------------------------------------
LAUNCH_CMD="mpirun -np $_NGPUS \
    --bind-to none \
    --map-by slot \
    -x JAX_ENABLE_X64 \
    -x XLA_FLAGS \
    -x XLA_PYTHON_CLIENT_PREALLOCATE \
    -x XLA_PYTHON_CLIENT_MEM_FRACTION \
    -x CUDA_DEVICE_ORDER \
    -x CUDA_VISIBLE_DEVICES \
    -x PATH \
    -x LD_LIBRARY_PATH \
    python ${SCRIPT_DIR}/run_amip.py \
    $DATASET_ARGS \
    --resolution $_RESOLUTION \
    --nlev $_NLEV \
    --days $_DURATION_DAYS \
    --dt $_DT \
    --distributed \
    --n-ranks $_NGPUS \
    --vertical-coord hybrid \
    --radiation rrtmg \
    --rad-update-steps 3 \
    --diurnal-cycle \
    --ozone-source analytical \
    --clouds xu_randall \
    --dynamic-albedo \
    --monthly-means \
    --checkpoint-days 30 \
    --diag-days 5 \
    --output $_OUTPUT_DIR \
    $_EXTRA_ARGS"

if [ "$_DRY_RUN" -eq 1 ]; then
    echo "[DRY RUN] Would execute:"
    echo ""
    echo "  $LAUNCH_CMD"
    echo ""
    exit 0
fi

# -------------------------------------------------------------------------
# Launch
# -------------------------------------------------------------------------
log_info "Starting multi-GPU AMIP run..."
START_TIME=$(date +%s)

eval $LAUNCH_CMD
EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    log_info "Multi-GPU AMIP run completed successfully."
    log_info "Wall time: ${HOURS}h ${MINS}m ${SECS}s"
    log_info "Output: $_OUTPUT_DIR"

    # Auto-validate if output exists
    if [ -f "${_OUTPUT_DIR}/monthly_means.npz" ]; then
        log_info "Running validation..."
        python "${SCRIPT_DIR}/validate_amip.py" "$_OUTPUT_DIR" --spinup 30 || true
    fi
else
    log_error "Multi-GPU AMIP run failed with exit code $EXIT_CODE."
    log_error "Check logs in $_OUTPUT_DIR for details."
    exit $EXIT_CODE
fi
