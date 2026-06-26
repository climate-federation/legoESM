#!/bin/bash
# ===========================================================================
# Top-level submitter for the "full CPU node vs 1 A100" comparison, ONE GRID at
# a time, covering its resolutions.  Each resolution is fanned out into its own
# CPU job + GPU job (so a slow high-res case gets its own walltime and runs in
# parallel with the others).
#
# Dispatch (the two cube scripts differ from the rest -- cube needs the mpi4jax
# face-scatter path, the others use the native MPI driver):
#   cubed-sphere -> cube_fullnode_cpu.sh   + cube_strong_gpu.sh (RANKS=1)
#   latlon/ico/spectral -> fullnode_cpu.sh + fullnode_gpu.sh
#
# Usage:
#   scripts/cluster/scaling_derecho/submit_fullnode.sh <grid> [res ...]
#     <grid> : cubed-sphere | latlon | icosahedral | spectral
#     [res]  : resolutions to cover (default per-grid list if omitted)
#
#   # default resolutions for the grid:
#   scripts/cluster/scaling_derecho/submit_fullnode.sh latlon
#   # explicit resolutions:
#   scripts/cluster/scaling_derecho/submit_fullnode.sh cubed-sphere 48 96 192
#   # knobs (pass through to the jobs) + dry run / one tier:
#   PHYSICS=moist PRECISION=float32 DRYRUN=1 ... submit_fullnode.sh icosahedral
#   CPU_ONLY=1 ... submit_fullnode.sh spectral
# ===========================================================================
set -euo pipefail

# Resolve repo root from this script's location so each job's PBS_O_WORKDIR (=
# the dir we qsub from) resolves _env.sh and the relative job-script paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
SD="scripts/cluster/scaling_derecho"
QSUB="${QSUB:-qsub}"

GRID="${1:-}"
if [ -z "$GRID" ]; then
    echo "usage: $0 <cubed-sphere|latlon|icosahedral|spectral> [res ...]" >&2
    exit 2
fi
shift
RES=("$@")

# Default per-grid resolutions (match the underlying scripts) + reject unknowns.
if [ "${#RES[@]}" -eq 0 ]; then
    case "$GRID" in
        cubed-sphere) RES=(48 96 192) ;;
        latlon)       RES=(128 256) ;;
        icosahedral)  RES=(6 7) ;;
        spectral)     RES=(85 170) ;;
        *) echo "ERROR: unknown grid '$GRID'" >&2; exit 2 ;;
    esac
else
    case "$GRID" in
        cubed-sphere|latlon|icosahedral|spectral) ;;
        *) echo "ERROR: unknown grid '$GRID'" >&2; exit 2 ;;
    esac
fi

# Optional pass-through knobs (each prefixed with a comma so they append cleanly
# to the per-job -v var string).
EXTRA_VARS=""
if [ -n "${PHYSICS:-}" ];   then EXTRA_VARS="${EXTRA_VARS},PHYSICS=${PHYSICS}"; fi
if [ -n "${PRECISION:-}" ]; then EXTRA_VARS="${EXTRA_VARS},PRECISION=${PRECISION}"; fi

submit() {  # submit "<label>" <qsub args...>
    local label="$1"; shift
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "[DRYRUN] ${label}: ${QSUB} $*"
        return 0
    fi
    local jobid
    jobid="$(${QSUB} "$@")"
    echo "  submitted ${label}: ${jobid}"
}

echo "=== repo: ${REPO_ROOT} ==="
echo "=== full-node CPU-vs-A100: grid=${GRID}  resolutions=[${RES[*]}] ==="

for R in "${RES[@]}"; do
    case "$GRID" in
        cubed-sphere)
            CPU_SCRIPT="$SD/cube_fullnode_cpu.sh"; CPU_VARS="STRONG_RES=${R}${EXTRA_VARS}"
            GPU_SCRIPT="$SD/cube_strong_gpu.sh";   GPU_VARS="RANKS=1,STRONG_RES=${R}${EXTRA_VARS}"
            ;;
        *)  # latlon | icosahedral | spectral (validated above)
            CPU_SCRIPT="$SD/fullnode_cpu.sh"; CPU_VARS="GRID=${GRID},RESOLUTIONS=${R}${EXTRA_VARS}"
            GPU_SCRIPT="$SD/fullnode_gpu.sh"; GPU_VARS="GRID=${GRID},RESOLUTIONS=${R}${EXTRA_VARS}"
            ;;
    esac
    if [ "${GPU_ONLY:-0}" != "1" ]; then
        submit "cpu ${GRID} res=${R} (full node)" -v "$CPU_VARS" "$CPU_SCRIPT"
    fi
    if [ "${CPU_ONLY:-0}" != "1" ]; then
        submit "gpu ${GRID} res=${R} (1 A100)" -v "$GPU_VARS" "$GPU_SCRIPT"
    fi
done

echo "=== done ===  watch the queue with: qstat -u \$USER"
