#!/bin/bash
# ==========================================================================
# Levante (DKRZ) GPU scaling benchmark -- SLURM submission driver
#
# Replicates CliMA JAMES 2026 Figures 11 (weak) and 12 (strong).
#
# Submits a series of SLURM jobs for 1, 2, 3, 6, 24, 54 GPUs covering
# both weak and strong scaling at float32 and float64 precision.
#
# Levante GPU partition:
#   - 60 nodes, each with 2x AMD EPYC 7763 + 4x NVIDIA A100
#   - 56 nodes with A100-80GB, 4 nodes with A100-40GB
#   - InfiniBand HDR 100G/200G interconnect
#
# Usage:
#   # Generate + submit all jobs:
#   ./scripts/run_levante_gpu_scaling.sh
#
#   # Dry run (print sbatch commands without submitting):
#   ./scripts/run_levante_gpu_scaling.sh --dry-run
#
#   # Override account / output directory:
#   ./scripts/run_levante_gpu_scaling.sh --account bb1234 --output results/my_scaling
#
#   # Run only weak scaling:
#   ./scripts/run_levante_gpu_scaling.sh --mode weak
#
#   # Custom GPU counts (comma-separated):
#   ./scripts/run_levante_gpu_scaling.sh --gpu-counts 1,2,3,6
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------------
ACCOUNT="${SLURM_ACCOUNT:-bm1183}"
PARTITION="gpu"
GPUS_PER_NODE=4
TIME_LIMIT="02:00:00"
OUTPUT_ROOT="${PROJECT_DIR}/output/scaling"
MODE="both"                    # weak, strong, or both
PRECISION="both"               # float32, float64, or both
N_WARMUP=3
N_TIMING=100
N_LEVELS=26
WEAK_BASE_N=24
# Iter 18 default: icosahedral is the only validated multi-rank MPI grid
# (see iter-13/17 ``_MPI_SUPPORTED_GRIDS`` in the Python driver).  The
# previous default (``--grid <unset>`` ⇒ spectral) silently downgraded
# multi-rank submissions to a no-MPI single-rank spectral run.
GRID="icosahedral"
STRONG_RESOLUTIONS="4,5,6"     # icosahedral subdivision levels
GPU_COUNTS="1,2,4,8"           # any positive count works for icosahedral
DRY_RUN=0
CONSTRAINT=""                  # e.g., "a100_80g" for 80GB nodes only

# -------------------------------------------------------------------------
# Parse CLI arguments
# -------------------------------------------------------------------------
usage() {
    cat <<'USAGE'
Usage: run_levante_gpu_scaling.sh [OPTIONS]

Options:
  --account ACCT         SLURM account (default: bm1183)
  --partition PART       SLURM partition (default: gpu)
  --time-limit HH:MM:SS Time limit per job (default: 02:00:00)
  --output DIR           Output root directory (default: output/scaling)
  --mode MODE            weak, strong, or both (default: both)
  --precision PREC       float32, float64, or both (default: both)
  --gpu-counts LIST      Comma-separated GPU counts (default: 1,2,3,6,24,54)
  --n-warmup N           Warmup steps (default: 3)
  --n-timing N           Timing steps (default: 100)
  --n-levels N           Vertical levels (default: 26)
  --weak-base-n N        Base resolution for weak scaling (default: 24)
  --strong-res LIST      Resolutions for strong scaling (default: 4,5,6)
  --grid GRID            Grid type (default: icosahedral; only validated
                         multi-rank MPI path)
  --constraint STR       SLURM constraint (e.g., a100_80g)
  --dry-run              Print sbatch commands without submitting
  -h, --help             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)       ACCOUNT="$2"; shift 2 ;;
        --partition)     PARTITION="$2"; shift 2 ;;
        --time-limit)    TIME_LIMIT="$2"; shift 2 ;;
        --output)        OUTPUT_ROOT="$2"; shift 2 ;;
        --mode)          MODE="$2"; shift 2 ;;
        --precision)     PRECISION="$2"; shift 2 ;;
        --gpu-counts)    GPU_COUNTS="$2"; shift 2 ;;
        --grid)          GRID="$2"; shift 2 ;;
        --n-warmup)      N_WARMUP="$2"; shift 2 ;;
        --n-timing)      N_TIMING="$2"; shift 2 ;;
        --n-levels)      N_LEVELS="$2"; shift 2 ;;
        --weak-base-n)   WEAK_BASE_N="$2"; shift 2 ;;
        --strong-res)    STRONG_RESOLUTIONS="$2"; shift 2 ;;
        --constraint)    CONSTRAINT="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# -------------------------------------------------------------------------
# Timestamp for this campaign
# -------------------------------------------------------------------------
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
CAMPAIGN_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"
JOBS_DIR="${CAMPAIGN_DIR}/jobs"
mkdir -p "${JOBS_DIR}"

echo "========================================================================="
echo "  legoESM Levante GPU Scaling Benchmark"
echo "========================================================================="
echo "  Account:      ${ACCOUNT}"
echo "  Partition:    ${PARTITION}"
echo "  GPU counts:   ${GPU_COUNTS}"
echo "  Mode:         ${MODE}"
echo "  Precision:    ${PRECISION}"
echo "  Levels:       ${N_LEVELS}"
echo "  Warmup:       ${N_WARMUP} steps"
echo "  Timing:       ${N_TIMING} steps"
echo "  Time limit:   ${TIME_LIMIT}"
echo "  Output:       ${CAMPAIGN_DIR}"
echo "  Dry run:      ${DRY_RUN}"
echo "========================================================================="

# -------------------------------------------------------------------------
# Generate and submit one SLURM job per GPU count
# -------------------------------------------------------------------------
IFS=',' read -ra GPU_LIST <<< "${GPU_COUNTS}"

JOB_IDS=""
JOB_COUNT=0

for N_GPUS in "${GPU_LIST[@]}"; do
    N_GPUS=$(echo "${N_GPUS}" | tr -d ' ')

    # Compute number of nodes needed (4 GPUs per Levante node)
    N_NODES=$(( (N_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
    if [ "${N_NODES}" -lt 1 ]; then
        N_NODES=1
    fi

    # For single-GPU runs, still request 1 full node for consistency
    ACTUAL_GPUS_REQUESTED=${N_GPUS}
    if [ "${N_GPUS}" -le "${GPUS_PER_NODE}" ]; then
        N_NODES=1
    fi

    # Determine ntasks-per-node for MPI (1 task per GPU)
    if [ "${N_GPUS}" -le "${GPUS_PER_NODE}" ]; then
        NTASKS_PER_NODE=${N_GPUS}
    else
        NTASKS_PER_NODE=${GPUS_PER_NODE}
    fi

    JOB_NAME="lego_scale_${N_GPUS}gpu"
    RUN_OUTPUT="${CAMPAIGN_DIR}/gpu_${N_GPUS}"
    SCRIPT_PATH="${JOBS_DIR}/${JOB_NAME}.sbatch"

    # Build the Python command.  Iter 18 fix: pass ``--grid`` (the
    # wrapper previously omitted it, so multi-rank jobs silently
    # ran the default ``spectral`` grid which has no MPI dispatch).
    PYTHON_CMD=".venv/bin/python scripts/run_levante_gpu_scaling.py"
    PYTHON_CMD="${PYTHON_CMD} --grid ${GRID}"
    PYTHON_CMD="${PYTHON_CMD} --mode ${MODE}"
    PYTHON_CMD="${PYTHON_CMD} --precision ${PRECISION}"
    PYTHON_CMD="${PYTHON_CMD} --n-gpus ${N_GPUS}"
    PYTHON_CMD="${PYTHON_CMD} --n-levels ${N_LEVELS}"
    PYTHON_CMD="${PYTHON_CMD} --n-warmup ${N_WARMUP}"
    PYTHON_CMD="${PYTHON_CMD} --n-timing ${N_TIMING}"
    PYTHON_CMD="${PYTHON_CMD} --weak-base-n ${WEAK_BASE_N}"
    PYTHON_CMD="${PYTHON_CMD} --strong-resolutions ${STRONG_RESOLUTIONS}"
    PYTHON_CMD="${PYTHON_CMD} --output-dir ${RUN_OUTPUT}"

    # Write the sbatch script
    cat > "${SCRIPT_PATH}" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=${N_NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=16
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${SCRIPT_PATH%.sbatch}.out
#SBATCH --error=${SCRIPT_PATH%.sbatch}.err
#SBATCH --exclusive
$([ -n "${CONSTRAINT}" ] && echo "#SBATCH --constraint=${CONSTRAINT}")

# ==========================================================================
# legoESM GPU scaling: ${N_GPUS} GPU(s) on ${N_NODES} node(s)
# ==========================================================================

set -euo pipefail

echo "========================================"
echo "  Job: \${SLURM_JOB_NAME}"
echo "  Job ID: \${SLURM_JOB_ID}"
echo "  Nodes: \${SLURM_NODELIST}"
echo "  GPUs requested: ${N_GPUS}"
echo "  Date: \$(date -u)"
echo "========================================"

# --- Environment setup ---
cd ${PROJECT_DIR}

# Load required modules
module purge
module load python3 cuda/12

# Activate the project venv
source .venv/bin/activate

# --- JAX / XLA configuration ---
# Iter 22: ``JAX_PLATFORMS="gpu,cpu"`` is the legacy alias and is
# rejected by JAX 0.10+ ("Backend 'rocm' is not in the list of known
# backends: ['cpu', 'tpu', 'cuda']").  Use ``cuda,cpu`` on NVIDIA
# nodes; for AMD ROCm nodes pass ``--export=JAX_PLATFORMS=rocm,cpu``
# via the SLURM submit command instead of changing this line.
export JAX_PLATFORMS="cuda,cpu"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# Prevent CPU thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# GPU ordering
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- Record environment ---
echo ""
echo "--- nvidia-smi ---"
nvidia-smi || true
echo ""
echo "--- Python / JAX versions ---"
python -c "import sys; print(f'Python {sys.version}')"
python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')" || true
echo ""

# --- Launch benchmark ---
mkdir -p ${RUN_OUTPUT}

if [ ${N_GPUS} -le ${GPUS_PER_NODE} ]; then
    # Single-node: run directly (JAX handles local device mesh)
    echo "Running single-node benchmark with ${N_GPUS} GPU(s)..."
    ${PYTHON_CMD}
else
    # Multi-node: use MPI (1 rank per GPU)
    echo "Running multi-node benchmark with ${N_GPUS} GPU(s) across ${N_NODES} nodes..."
    srun --mpi=pmix \\
        --ntasks=${N_GPUS} \\
        --ntasks-per-node=${NTASKS_PER_NODE} \\
        --gpus-per-node=${GPUS_PER_NODE} \\
        ${PYTHON_CMD}
fi

echo ""
echo "Benchmark complete: \$(date -u)"
echo "Results: ${RUN_OUTPUT}"
SBATCH_EOF

    chmod +x "${SCRIPT_PATH}"
    JOB_COUNT=$((JOB_COUNT + 1))

    echo ""
    echo "--- Job ${JOB_COUNT}: ${N_GPUS} GPU(s) on ${N_NODES} node(s) ---"
    echo "  Script: ${SCRIPT_PATH}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "  [DRY RUN] Would submit: sbatch ${SCRIPT_PATH}"
    else
        SUBMIT_OUTPUT=$(sbatch "${SCRIPT_PATH}" 2>&1) || true
        JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oP '\d+' | tail -1 || echo "unknown")
        echo "  Submitted: ${SUBMIT_OUTPUT}"

        if [ -n "${JOB_IDS}" ]; then
            JOB_IDS="${JOB_IDS},${JOB_ID}"
        else
            JOB_IDS="${JOB_ID}"
        fi
    fi
done

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo ""
echo "========================================================================="
echo "  Summary"
echo "========================================================================="
echo "  Jobs generated: ${JOB_COUNT}"
echo "  Campaign dir:   ${CAMPAIGN_DIR}"
echo "  Job scripts:    ${JOBS_DIR}/"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo ""
    echo "  Dry run complete. Re-run without --dry-run to submit."
else
    echo "  Job IDs:        ${JOB_IDS}"
    echo ""
    echo "  Monitor with:"
    echo "    squeue -u \$USER"
    echo ""
    echo "  After completion, collect results with:"
    echo "    ls ${CAMPAIGN_DIR}/gpu_*/all_scaling.csv"
    echo ""
    echo "  To generate combined plots across all GPU counts:"
    echo "    python scripts/run_levante_gpu_scaling.py \\"
    echo "        --mode ${MODE} --precision ${PRECISION} \\"
    echo "        --output-dir ${CAMPAIGN_DIR}/combined"
fi

echo "========================================================================="

# -------------------------------------------------------------------------
# Save campaign manifest
# -------------------------------------------------------------------------
cat > "${CAMPAIGN_DIR}/manifest.json" <<MANIFEST_EOF
{
  "timestamp_utc": "${TIMESTAMP}",
  "account": "${ACCOUNT}",
  "partition": "${PARTITION}",
  "gpu_counts": [$(echo "${GPU_COUNTS}" | sed 's/,/, /g')],
  "mode": "${MODE}",
  "precision": "${PRECISION}",
  "n_levels": ${N_LEVELS},
  "n_warmup": ${N_WARMUP},
  "n_timing": ${N_TIMING},
  "weak_base_n": ${WEAK_BASE_N},
  "strong_resolutions": "${STRONG_RESOLUTIONS}",
  "grid": "${GRID}",
  "time_limit": "${TIME_LIMIT}",
  "project_dir": "${PROJECT_DIR}",
  "campaign_dir": "${CAMPAIGN_DIR}",
  "n_jobs": ${JOB_COUNT},
  "dry_run": $([ "${DRY_RUN}" -eq 1 ] && echo "true" || echo "false")
}
MANIFEST_EOF

echo ""
echo "Manifest: ${CAMPAIGN_DIR}/manifest.json"
