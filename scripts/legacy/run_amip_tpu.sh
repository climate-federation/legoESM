#!/bin/bash
# ==========================================================================
# Launch AMIP simulation on TPU pods
#
# Supports single-host TPU VMs (e.g., v4-8) and multi-host TPU pods
# (e.g., v4-32, v4-128). JAX automatically detects the TPU topology
# and distributes computation across all available chips.
#
# Usage (single host, e.g., v4-8):
#   ./scripts/run_amip_tpu.sh
#
# Usage (multi-host TPU pod, e.g., v4-32):
#   gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all \
#     --command="cd ~/legoESM && ./scripts/run_amip_tpu.sh"
#
# Usage with options:
#   ./scripts/run_amip_tpu.sh --resolution 96 --nlev 60 --days 3650
#   RESOLUTION=48 NLEV=40 ./scripts/run_amip_tpu.sh
#
# Environment variable overrides:
#   RESOLUTION      Cubed-sphere resolution N (default: 48)
#   NLEV            Number of vertical levels (default: 40)
#   DURATION_DAYS   Integration length in days (default: 365)
#   DT              Time step in seconds (default: 600)
#   OUTPUT_DIR      Output directory (default: auto-generated)
#   FORCING_PATH    Path to SST forcing NetCDF (default: analytical)
#   DATASET         Forcing dataset preset (default: analytical)
#   TPU_PRECISION   float32 or bfloat16 (default: float32)
#
# Requirements:
#   - TPU VM with JAX pre-installed (or manually: pip install jax[tpu])
#   - For multi-host: gcloud CLI configured, SSH access to all workers
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
Usage: run_amip_tpu.sh [OPTIONS] [-- EXTRA_ARGS_FOR_RUN_AMIP]

Options:
  --resolution N     Cubed-sphere resolution (default: 48)
  --nlev N           Number of vertical levels (default: 40)
  --days N           Integration length in days (default: 365)
  --dt SECONDS       Time step (default: 600)
  --forcing PATH     Path to SST forcing NetCDF file
  --dataset NAME     Forcing dataset preset: cobe, hadisst, analytical (default: analytical)
  --output DIR       Output directory
  --precision MODE   TPU precision: float32 or bfloat16 (default: float32)
  --dry-run          Print the launch command without executing
  -h, --help         Show this help message

Extra arguments after '--' are passed directly to run_amip.py.

Examples:
  # Single-host TPU (v4-8), 1-year analytical:
  ./scripts/run_amip_tpu.sh

  # High-resolution 10-year run on TPU pod:
  ./scripts/run_amip_tpu.sh --resolution 96 --nlev 60 --days 3650

  # With COBE-SST2 forcing:
  ./scripts/run_amip_tpu.sh --dataset cobe --forcing /data/MODEL.SST.COBE-SST2.nc

  # Multi-host launch (v4-32 pod):
  gcloud compute tpus tpu-vm ssh my-tpu --worker=all \
      --command="cd ~/legoESM && ./scripts/run_amip_tpu.sh --resolution 96"
USAGE
}

# -------------------------------------------------------------------------
# Detect TPU topology
# -------------------------------------------------------------------------
detect_tpu() {
    if python -c "import jax; devs = jax.devices('tpu'); print(len(devs))" 2>/dev/null; then
        return 0
    fi
    return 1
}

# -------------------------------------------------------------------------
# Default settings
# -------------------------------------------------------------------------
_RESOLUTION="${RESOLUTION:-48}"
_NLEV="${NLEV:-40}"
_DURATION_DAYS="${DURATION_DAYS:-365}"
_DT="${DT:-600}"
_OUTPUT_DIR="${OUTPUT_DIR:-}"
_FORCING_PATH="${FORCING_PATH:-}"
_DATASET="${DATASET:-analytical}"
_PRECISION="${TPU_PRECISION:-float32}"
_DRY_RUN=0
_EXTRA_ARGS=""

# -------------------------------------------------------------------------
# Parse command-line arguments
# -------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --precision)
            _PRECISION="$2"; shift 2 ;;
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
# TPU-specific JAX configuration
# -------------------------------------------------------------------------
# TPU benefits from disabling x64 (float32 is native; bfloat16 for matmuls)
if [ "$_PRECISION" = "bfloat16" ]; then
    export JAX_ENABLE_X64=0
    export JAX_DEFAULT_MATMUL_PRECISION=bfloat16
    log_info "Using bfloat16 matmul precision (TPU-optimized)"
else
    export JAX_ENABLE_X64=0
    export JAX_DEFAULT_MATMUL_PRECISION=float32
    log_info "Using float32 precision"
fi

export JAX_PLATFORMS=tpu
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_data_parallel_all_reduce_opt=true"

# TPU memory management
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"

# Workaround for potential libtpu version mismatches
export TPU_LIBRARY_PATH="${TPU_LIBRARY_PATH:-}"

# -------------------------------------------------------------------------
# Validate TPU availability
# -------------------------------------------------------------------------
log_info "Detecting TPU devices..."

TPU_COUNT=$(python -c "
import jax
try:
    devs = jax.devices('tpu')
    print(len(devs))
except RuntimeError:
    print(0)
" 2>/dev/null || echo 0)

if [ "$TPU_COUNT" -eq 0 ]; then
    log_error "No TPU devices detected."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure you are on a TPU VM:  gcloud compute tpus describe <name>"
    echo "  2. Check JAX TPU install:       python -c 'import jax; print(jax.devices())'"
    echo "  3. Install JAX for TPU:         pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    echo ""
    exit 1
fi

log_info "Detected $TPU_COUNT TPU core(s)"

# Determine if this is a multi-host setup
PROCESS_COUNT=$(python -c "
import jax
print(jax.process_count())
" 2>/dev/null || echo 1)

PROCESS_INDEX=$(python -c "
import jax
print(jax.process_index())
" 2>/dev/null || echo 0)

if [ "$PROCESS_COUNT" -gt 1 ]; then
    log_info "Multi-host TPU pod: process $PROCESS_INDEX of $PROCESS_COUNT"
fi

# -------------------------------------------------------------------------
# Set output directory
# -------------------------------------------------------------------------
if [ -z "$_OUTPUT_DIR" ]; then
    _OUTPUT_DIR="results/amip_tpu_C${_RESOLUTION}"
fi

# Only create output dir on process 0
if [ "$PROCESS_INDEX" -eq 0 ]; then
    mkdir -p "$_OUTPUT_DIR"
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
# Print launch summary (only on process 0)
# -------------------------------------------------------------------------
if [ "$PROCESS_INDEX" -eq 0 ]; then
    echo "=========================================="
    echo "  legoESM TPU AMIP Launch"
    echo "=========================================="
    echo "  TPU cores:    $TPU_COUNT"
    echo "  Processes:    $PROCESS_COUNT"
    echo "  Precision:    $_PRECISION"
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
fi

# -------------------------------------------------------------------------
# Build the launch command
# -------------------------------------------------------------------------
# On TPU pods, JAX handles multi-host coordination automatically via
# the TPU runtime (no MPI needed). Each worker runs the same script
# and JAX discovers the topology.
LAUNCH_CMD="python ${SCRIPT_DIR}/run_amip.py \
    $DATASET_ARGS \
    --resolution $_RESOLUTION \
    --nlev $_NLEV \
    --days $_DURATION_DAYS \
    --dt $_DT \
    --distributed \
    --n-ranks $TPU_COUNT \
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
    echo "Environment:"
    echo "  JAX_ENABLE_X64=$JAX_ENABLE_X64"
    echo "  JAX_PLATFORMS=$JAX_PLATFORMS"
    echo "  JAX_DEFAULT_MATMUL_PRECISION=$JAX_DEFAULT_MATMUL_PRECISION"
    echo "  XLA_FLAGS=$XLA_FLAGS"
    exit 0
fi

# -------------------------------------------------------------------------
# Launch
# -------------------------------------------------------------------------
log_info "Starting TPU AMIP run (process $PROCESS_INDEX)..."
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
    if [ "$PROCESS_INDEX" -eq 0 ]; then
        log_info "TPU AMIP run completed successfully."
        log_info "Wall time: ${HOURS}h ${MINS}m ${SECS}s"
        log_info "Output: $_OUTPUT_DIR"

        # Auto-validate if output exists
        if [ -f "${_OUTPUT_DIR}/monthly_means.npz" ]; then
            log_info "Running validation..."
            python "${SCRIPT_DIR}/validate_amip.py" "$_OUTPUT_DIR" --spinup 30 || true
        fi
    fi
else
    log_error "TPU AMIP run failed with exit code $EXIT_CODE (process $PROCESS_INDEX)."
    exit $EXIT_CODE
fi
