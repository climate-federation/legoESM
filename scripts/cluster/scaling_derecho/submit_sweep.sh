#!/bin/bash
# ===========================================================================
# Submit the full Derecho MOIST scaling sweep in one shot.
#
# Dimensions: grid (latlon, icosahedral) x precision (float64, float32), all
# with --physics moist (Kessler warm-rain).  Each job sweeps ranks/GPUs
# internally.  CPU and GPU mirror the SAME grid x precision matrix.
#
#   CPU jobs  -> cpu_scaling.pbs       (one 128-core node; ranks 1,2,4,...,128)
#   GPU jobs  -> gpu_moist_scaling.pbs (one node, up to 4 A100; 1 GPU/rank)
#
# Grids: latlon + icosahedral only -- the two grids genuinely domain-decomposed
# for multi-rank/multi-GPU scaling.  cubed-sphere/spectral moist are single-
# device only (no scaling curve) and are intentionally excluded here.
#
# All single-node by design.  Run from anywhere -- the script cd's to the repo
# root so the relative qsub paths and each job's PBS_O_WORKDIR resolve.
#
# Usage:
#   scripts/cluster/scaling_derecho/submit_all.sh           # submit everything
#   DRYRUN=1 scripts/cluster/scaling_derecho/submit_all.sh  # print, don't submit
#   CPU_ONLY=1 ... / GPU_ONLY=1 ...                         # one tier only
# ===========================================================================

set -euo pipefail

# Resolve repo root from this script's location (.../scripts/cluster/scaling_derecho).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CPU_PBS="scripts/cluster/scaling_derecho/cpu_scaling.pbs"
GPU_PBS="scripts/cluster/scaling_derecho/gpu_moist_scaling.pbs"
QSUB="${QSUB:-qsub}"

submit() {
    # submit "<human label>" -v VARS pbs_script
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
echo "=== submitting full single-node scaling sweep ==="

# ---------------------------------------------------------------------------
# CPU (128-core node): latlon + icosahedral moist, genuine MPI weak+strong
# scaling, both precisions (grid x precision).
# ---------------------------------------------------------------------------
if [ "${GPU_ONLY:-0}" != "1" ]; then
    echo "--- CPU: moist float64 ---"
    submit "cpu latlon moist f64"      -v GRID=latlon,PHYSICS=moist,MODE=both,PRECISION=float64,MAX_RANKS=128      "${CPU_PBS}"
    submit "cpu icosahedral moist f64" -v GRID=icosahedral,PHYSICS=moist,MODE=both,PRECISION=float64,MAX_RANKS=128 "${CPU_PBS}"

    echo "--- CPU: moist float32 ---"
    submit "cpu latlon moist f32"      -v GRID=latlon,PHYSICS=moist,MODE=both,PRECISION=float32,MAX_RANKS=128      "${CPU_PBS}"
    submit "cpu icosahedral moist f32" -v GRID=icosahedral,PHYSICS=moist,MODE=both,PRECISION=float32,MAX_RANKS=128 "${CPU_PBS}"
fi

# ---------------------------------------------------------------------------
# GPU (1 node, up to 4 A100): latlon + icosahedral moist, route-A mpi4jax,
# both precisions (grid x precision) -- mirrors the CPU matrix.
# ---------------------------------------------------------------------------
if [ "${CPU_ONLY:-0}" != "1" ]; then
    echo "--- GPU: moist float64, 1->2->4 A100 ---"
    submit "gpu latlon moist f64"      -v GRID=latlon,PHYSICS=moist,MODE=both,PRECISION=float64 "${GPU_PBS}"
    submit "gpu icosahedral moist f64" -v GRID=icosahedral,PHYSICS=moist,MODE=both,PRECISION=float64 "${GPU_PBS}"

    echo "--- GPU: moist float32, 1->2->4 A100 ---"
    submit "gpu latlon moist f32"      -v GRID=latlon,PHYSICS=moist,MODE=both,PRECISION=float32 "${GPU_PBS}"
    submit "gpu icosahedral moist f32" -v GRID=icosahedral,PHYSICS=moist,MODE=both,PRECISION=float32 "${GPU_PBS}"
fi

echo "=== done ==="
echo "Watch the queue with: qstat -u \$USER"
