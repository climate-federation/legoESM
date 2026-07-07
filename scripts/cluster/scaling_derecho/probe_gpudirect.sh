#!/bin/bash -l
# ===========================================================================
# Run the GPU-direct fabric probe (probe_gpudirect.py) across >=2 nodes.
# Reproduces the 4-byte cross-node device sendrecv that aborts the multi-node
# GPU scaling job — in seconds, no dycore. See probe_gpudirect.py and
# docs/performance/multinode_gpu_direct_cxi.md.
#
# MUST run inside a >=2-node GPU allocation (a login-node mpiexec errors "No
# host list provided"):
#   qsub -I -A P08010000 -q main -l job_priority=premium \
#        -l select=2:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB -l walltime=00:30:00
#
# Usage (from the repo root, inside the allocation):
#   bash scripts/cluster/scaling_derecho/probe_gpudirect.sh              # current env (expect abort)
#   LOAD_GDRCOPY=1 bash scripts/cluster/scaling_derecho/probe_gpudirect.sh   # THE fix test
#   NPROC=8 PPN=4 bash .../probe_gpudirect.sh                            # override rank layout
#   MPI4JAX_USE_CUDA_MPI=0 bash .../probe_gpudirect.sh                   # sanity: host-staged should PASS
# ===========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"

# Pin the runtime MPICH explicitly (bare cray-mpich can default to 8.1.32).
module load cray-mpich/9.0.0 2>/dev/null || module load cray-mpich 2>/dev/null || true
# The knob under test: GDRCopy gives the NIC a device-memory registration path
# for the tiny inject send. Absent -> the 4-byte device pointer aborts on CXI.
if [ "${LOAD_GDRCOPY:-0}" = "1" ]; then
    if module load gdrcopy 2>/dev/null; then echo "gdrcopy: loaded"
    else echo "gdrcopy: MODULE NOT FOUND (try 'module avail gdrcopy')" >&2; fi
fi
export MPICH_GPU_SUPPORT_ENABLED=1
export MPI4JAX_USE_CUDA_MPI="${MPI4JAX_USE_CUDA_MPI:-1}"   # GPU-direct by default (the real test)
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

NPROC="${NPROC:-8}"     # >=2 nodes' worth of ranks so the ring crosses a node boundary
PPN="${PPN:-4}"         # 4 ranks/node (1 per A100)

# Escalation knobs (march the trivial ring toward the model halo):
#   JIT=1        -> --jit   (run mpi4jax inside jax.jit, the model's compiled path)
#   ITERS=100    -> --iters (repeat the exchange; repeated invocation)
#   BYTES=65536  -> --bytes (realistic message size)
PROBE_ARGS=()
[ "${JIT:-0}" = "1" ] && PROBE_ARGS+=(--jit)
[ -n "${ITERS:-}" ]   && PROBE_ARGS+=(--iters "${ITERS}")
[ -n "${BYTES:-}" ]   && PROBE_ARGS+=(--bytes "${BYTES}")

echo "=== GPU-direct probe: -n ${NPROC} --ppn ${PPN} | MPI4JAX_USE_CUDA_MPI=${MPI4JAX_USE_CUDA_MPI} | cray-mpich=${CRAY_MPICH_VERSION:-?} | gdrcopy=${LOAD_GDRCOPY:-0} | args=[${PROBE_ARGS[*]:-none}] ==="
mpiexec --ppn "${PPN}" -n "${NPROC}" \
    bash -c 'export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-0}; exec "$@"' _ \
    "${PY:-python}" "${SCRIPT_DIR}/probe_gpudirect.py" ${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"}
