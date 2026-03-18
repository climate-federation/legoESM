#!/bin/bash
# ==========================================================================
# Launch ensemble AMIP forecast
#
# Runs multiple ensemble members with perturbed initial conditions. Each
# member uses a different random seed to apply small temperature
# perturbations to the initial state, enabling spread/skill analysis and
# uncertainty quantification.
#
# Modes:
#   Serial:      Members run one-at-a-time on available hardware (default)
#   Parallel:    Members launched simultaneously as background processes
#   Distributed: Each member runs across multiple GPUs/TPUs via --distributed
#
# Usage:
#   ./scripts/run_ensemble.sh                            # 10 members, serial
#   ./scripts/run_ensemble.sh --members 50               # 50 members, serial
#   ./scripts/run_ensemble.sh --members 20 --parallel    # 20 members in parallel
#   ./scripts/run_ensemble.sh --members 6 --distributed --ngpus 6  # Distributed
#
# Environment variable overrides:
#   MEMBERS         Number of ensemble members (default: 10)
#   PERTURBATION    Initial condition perturbation magnitude [K] (default: 1e-4)
#   RESOLUTION      Cubed-sphere resolution N (default: 48)
#   NLEV            Number of vertical levels (default: 40)
#   DURATION_DAYS   Integration length in days (default: 365)
#   DT              Time step in seconds (default: 600)
#   OUTPUT_DIR      Base output directory (default: results/ensemble_C${RESOLUTION})
#   FORCING_PATH    Path to SST forcing NetCDF (default: analytical)
#   DATASET         Forcing dataset preset (default: analytical)
#   MAX_PARALLEL    Max concurrent members in parallel mode (default: all)
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
Usage: run_ensemble.sh [OPTIONS] [-- EXTRA_ARGS_FOR_RUN_AMIP]

Options:
  --members N         Number of ensemble members (default: 10)
  --perturbation F    Initial perturbation magnitude in K (default: 1e-4)
  --resolution N      Cubed-sphere resolution (default: 48)
  --nlev N            Number of vertical levels (default: 40)
  --days N            Integration length in days (default: 365)
  --dt SECONDS        Time step (default: 600)
  --forcing PATH      Path to SST forcing NetCDF file
  --dataset NAME      Forcing preset: cobe, hadisst, analytical (default: analytical)
  --output DIR        Base output directory
  --parallel          Run members in parallel as background processes
  --max-parallel N    Max concurrent members in parallel mode (default: all)
  --distributed       Use multi-GPU distributed mode per member
  --ngpus N           GPUs per member in distributed mode (default: 1)
  --dry-run           Print commands without executing
  -h, --help          Show this help message

Extra arguments after '--' are passed directly to run_amip.py.

Examples:
  # 10-member ensemble with analytical forcing:
  ./scripts/run_ensemble.sh

  # 50-member ensemble, 4 concurrent members:
  ./scripts/run_ensemble.sh --members 50 --parallel --max-parallel 4

  # High-resolution ensemble with COBE forcing:
  ./scripts/run_ensemble.sh --members 20 --resolution 96 \
      --dataset cobe --forcing /data/MODEL.SST.COBE-SST2.nc

  # Distributed ensemble (each member on 6 GPUs):
  ./scripts/run_ensemble.sh --members 5 --distributed --ngpus 6
USAGE
}

# -------------------------------------------------------------------------
# Default settings
# -------------------------------------------------------------------------
_MEMBERS="${MEMBERS:-10}"
_PERTURBATION="${PERTURBATION:-1e-4}"
_RESOLUTION="${RESOLUTION:-48}"
_NLEV="${NLEV:-40}"
_DURATION_DAYS="${DURATION_DAYS:-365}"
_DT="${DT:-600}"
_OUTPUT_DIR="${OUTPUT_DIR:-}"
_FORCING_PATH="${FORCING_PATH:-}"
_DATASET="${DATASET:-analytical}"
_PARALLEL=0
_MAX_PARALLEL="${MAX_PARALLEL:-0}"
_DISTRIBUTED=0
_NGPUS=1
_DRY_RUN=0
_EXTRA_ARGS=""

# -------------------------------------------------------------------------
# Parse command-line arguments
# -------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --members)
            _MEMBERS="$2"; shift 2 ;;
        --perturbation)
            _PERTURBATION="$2"; shift 2 ;;
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
        --parallel)
            _PARALLEL=1; shift ;;
        --max-parallel)
            _MAX_PARALLEL="$2"; shift 2 ;;
        --distributed)
            _DISTRIBUTED=1; shift ;;
        --ngpus)
            _NGPUS="$2"; shift 2 ;;
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
# Validate
# -------------------------------------------------------------------------
if [ "$_MEMBERS" -lt 1 ]; then
    log_error "Number of ensemble members must be >= 1"
    exit 1
fi

if [ "$_DISTRIBUTED" -eq 1 ] && ! command -v mpirun &>/dev/null; then
    log_error "mpirun not found but --distributed was specified."
    log_error "Install OpenMPI or MPICH: conda install -c conda-forge openmpi mpi4py"
    exit 1
fi

# -------------------------------------------------------------------------
# Set output directory
# -------------------------------------------------------------------------
if [ -z "$_OUTPUT_DIR" ]; then
    _OUTPUT_DIR="results/ensemble_C${_RESOLUTION}"
fi
mkdir -p "$_OUTPUT_DIR"

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
# Environment
# -------------------------------------------------------------------------
export JAX_ENABLE_X64=1

# -------------------------------------------------------------------------
# Print launch summary
# -------------------------------------------------------------------------
echo "=========================================="
echo "  legoESM Ensemble AMIP Launch"
echo "=========================================="
echo "  Members:      $_MEMBERS"
echo "  Perturbation: ${_PERTURBATION} K"
echo "  Resolution:   C${_RESOLUTION} / L${_NLEV}"
echo "  Duration:     $_DURATION_DAYS days"
echo "  Time step:    ${_DT}s"
echo "  Dataset:      $_DATASET"
if [ -n "$_FORCING_PATH" ]; then
echo "  Forcing:      $_FORCING_PATH"
fi
echo "  Output:       $_OUTPUT_DIR"
echo "  Mode:         $([ "$_PARALLEL" -eq 1 ] && echo "parallel" || echo "serial")"
if [ "$_PARALLEL" -eq 1 ] && [ "$_MAX_PARALLEL" -gt 0 ]; then
echo "  Max parallel: $_MAX_PARALLEL"
fi
if [ "$_DISTRIBUTED" -eq 1 ]; then
echo "  Distributed:  $_NGPUS GPUs per member"
fi
if [ -n "$_EXTRA_ARGS" ]; then
echo "  Extra args:   $_EXTRA_ARGS"
fi
echo "=========================================="
echo ""

# -------------------------------------------------------------------------
# Build base command for a single member
# -------------------------------------------------------------------------
build_member_cmd() {
    local member_id=$1
    local member_seed=$member_id
    local member_dir="${_OUTPUT_DIR}/member_$(printf '%03d' $member_id)"

    local cmd="python ${SCRIPT_DIR}/run_amip.py \
        $DATASET_ARGS \
        --resolution $_RESOLUTION \
        --nlev $_NLEV \
        --days $_DURATION_DAYS \
        --dt $_DT \
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
        --ensemble-size $_MEMBERS \
        --output $member_dir \
        $_EXTRA_ARGS"

    if [ "$_DISTRIBUTED" -eq 1 ]; then
        cmd="mpirun -np $_NGPUS \
            --bind-to none \
            --map-by slot \
            -x JAX_ENABLE_X64 \
            -x CUDA_VISIBLE_DEVICES \
            -x PATH \
            -x LD_LIBRARY_PATH \
            $cmd --distributed --n-ranks $_NGPUS"
    fi

    echo "$cmd"
}

# -------------------------------------------------------------------------
# Run a single member and track success/failure
# -------------------------------------------------------------------------
run_member() {
    local member_id=$1
    local member_dir="${_OUTPUT_DIR}/member_$(printf '%03d' $member_id)"
    local log_file="${member_dir}/run.log"

    mkdir -p "$member_dir"

    local cmd
    cmd=$(build_member_cmd "$member_id")

    if [ "$_DRY_RUN" -eq 1 ]; then
        echo "[DRY RUN] Member $member_id:"
        echo "  $cmd"
        echo ""
        return 0
    fi

    log_info "Starting member $member_id / $_MEMBERS -> $member_dir"

    # Set member-specific random seed via environment variable
    export LEGOESM_ENSEMBLE_SEED=$member_id

    if eval $cmd > "$log_file" 2>&1; then
        log_info "Member $member_id completed successfully."
        return 0
    else
        local exit_code=$?
        log_error "Member $member_id failed (exit code $exit_code). See $log_file"
        return $exit_code
    fi
}

# -------------------------------------------------------------------------
# Launch ensemble
# -------------------------------------------------------------------------
START_TIME=$(date +%s)
FAILED_MEMBERS=()
PIDS=()

if [ "$_DRY_RUN" -eq 1 ]; then
    for i in $(seq 1 "$_MEMBERS"); do
        run_member "$i"
    done
    exit 0
fi

if [ "$_PARALLEL" -eq 0 ]; then
    # --- Serial mode: one member at a time ---
    for i in $(seq 1 "$_MEMBERS"); do
        if ! run_member "$i"; then
            FAILED_MEMBERS+=("$i")
        fi
    done
else
    # --- Parallel mode: background processes with optional throttle ---
    ACTIVE_PIDS=()

    for i in $(seq 1 "$_MEMBERS"); do
        # Throttle if max-parallel is set
        if [ "$_MAX_PARALLEL" -gt 0 ]; then
            while [ "${#ACTIVE_PIDS[@]}" -ge "$_MAX_PARALLEL" ]; do
                # Wait for any one process to finish
                NEW_ACTIVE=()
                for pid in "${ACTIVE_PIDS[@]}"; do
                    if kill -0 "$pid" 2>/dev/null; then
                        NEW_ACTIVE+=("$pid")
                    fi
                done
                ACTIVE_PIDS=("${NEW_ACTIVE[@]}")
                if [ "${#ACTIVE_PIDS[@]}" -ge "$_MAX_PARALLEL" ]; then
                    sleep 2
                fi
            done
        fi

        member_dir="${_OUTPUT_DIR}/member_$(printf '%03d' $i)"
        mkdir -p "$member_dir"
        log_file="${member_dir}/run.log"

        cmd=$(build_member_cmd "$i")

        log_info "Launching member $i (background)..."
        LEGOESM_ENSEMBLE_SEED=$i eval $cmd > "$log_file" 2>&1 &
        pid=$!
        PIDS+=("$i:$pid")
        ACTIVE_PIDS+=("$pid")
    done

    # Wait for all members to finish
    log_info "Waiting for all ${_MEMBERS} members to complete..."
    for entry in "${PIDS[@]}"; do
        member_id="${entry%%:*}"
        pid="${entry##*:}"
        if wait "$pid"; then
            log_info "Member $member_id finished successfully (pid $pid)."
        else
            log_error "Member $member_id failed (pid $pid)."
            FAILED_MEMBERS+=("$member_id")
        fi
    done
fi

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "=========================================="
echo "  Ensemble Summary"
echo "=========================================="
echo "  Members:     $_MEMBERS"
echo "  Succeeded:   $(( _MEMBERS - ${#FAILED_MEMBERS[@]} ))"
echo "  Failed:      ${#FAILED_MEMBERS[@]}"
echo "  Wall time:   ${HOURS}h ${MINS}m ${SECS}s"
echo "  Output:      $_OUTPUT_DIR"

if [ ${#FAILED_MEMBERS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed members: ${FAILED_MEMBERS[*]}"
    echo "  Check logs: ${_OUTPUT_DIR}/member_NNN/run.log"
fi

echo "=========================================="
echo ""

# Run validation on ensemble output if all members succeeded
if [ ${#FAILED_MEMBERS[@]} -eq 0 ]; then
    # Check if we have monthly means from at least one member for validation
    FIRST_MEMBER="${_OUTPUT_DIR}/member_001"
    if [ -f "${FIRST_MEMBER}/monthly_means.npz" ]; then
        log_info "Running validation on member_001..."
        python "${SCRIPT_DIR}/validate_amip.py" "$FIRST_MEMBER" --spinup 30 || true
    fi

    # Print ensemble spread summary
    log_info "Ensemble output saved to $_OUTPUT_DIR"
    log_info "Member directories: ${_OUTPUT_DIR}/member_001 .. member_$(printf '%03d' $_MEMBERS)"
fi

# Exit with failure if any members failed
if [ ${#FAILED_MEMBERS[@]} -gt 0 ]; then
    exit 1
fi
