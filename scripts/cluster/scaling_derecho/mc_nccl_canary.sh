#!/bin/bash -l
#PBS -N mc_nccl_canary
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=2:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# Route-B multi-controller NCCL CANARY (2 nodes x 4 A100 = 8 GPUs).
#
# THE question this answers: does jax.distributed multi-controller (one
# process per GPU, ppermute/psum over NCCL) initialize and communicate ACROSS
# NODES on Slingshot-11 — where the route-A cross-node GPU-direct MPI path
# aborts in the CXI inject (docs/performance/multinode_gpu_direct_cxi.md)?
# If YES, route-B is the multi-node GPU halo transport (all-NCCL, no mpi4jax,
# XLA latency-hiding applies) and the CXI ticket stops blocking GPU scaling.
#
# NO mpi4jax anywhere in this job: mixing the mpi4jax halo machinery with
# jax.distributed collectives in one program is the documented mixed-stack
# deadlock hazard. mpiexec (PALS) is ONLY the process launcher; MPICH GPU
# support / MPI4JAX_USE_CUDA_MPI / craype-accel are deliberately NOT set.
#
# Stages (fail-fast; each answers a narrower question):
#   A  mc_nccl_probe.py       — pure-JAX init + psum + ppermute ring + latency
#   B  gated parity tests     — atm + ocean lat-band SPMD multi-controller
#                               equivalence gates (LEGOESM_JAX_DISTRIBUTED_TEST)
#   C  MC benches (strong)    — bench_atm_latlon_spmd_scaling.py +
#                               bench_ocean_latlon_spmd_scaling.py
#                               --multicontroller, 8 GPU
#
# NCCL-over-Slingshot notes: NCCL needs the libfabric/OFI plugin to use the
# hsn interfaces; NCCL_DEBUG=INFO below prints the transport it selected
# (NET/OFI = fabric; NET/Socket = TCP fallback — WORKS but slow, still a
# PASS for correctness, footnote the numbers). NCCL_SOCKET_IFNAME pins the
# bootstrap to the high-speed interfaces.
#
# Direct run:   qsub scripts/cluster/scaling_derecho/mc_nccl_canary.sh
# ===========================================================================
set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ] \
        && [ -f "${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho/_env.sh" ]; then
    SCRIPT_DIR="${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "${REPO}"

N=8            # 2 nodes x 4 GPUs, one process per GPU
PPN=4
COORD_HOST="$(head -n 1 "${PBS_NODEFILE}")"
# Distinct coordinator port per stage: consecutive mpiexec invocations
# re-binding the same port on the head node is a TIME_WAIT flake source.
PORT_A=29788; PORT_B1=29789; PORT_B2=29790; PORT_C1=29791; PORT_C2=29792

# NCCL canary diagnostics: print the selected transport (NET/OFI vs
# NET/Socket) — that line IS a canary deliverable. Pin bootstrap ifaces to
# the Slingshot hsn NICs; a missing OFI plugin then falls back to sockets
# over hsn (correct-but-slow) instead of hanging on the mgmt network.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"

# Per-rank shim: pin one GPU per rank (PALS_LOCAL_RANKID) and bridge the PALS
# rank id to the OMPI env contract the probe/tests/benches read. The world
# SIZE is a job constant, exported once below.
export OMPI_COMM_WORLD_SIZE="$N"
SHIM='export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-0}; '\
'export OMPI_COMM_WORLD_RANK=${PALS_RANKID:-${OMPI_COMM_WORLD_RANK:-0}}; '\
'exec "$@"'

run_stage() {  # run_stage <label> <port> <cmd...>
    local label="$1" port="$2"; shift 2
    export LEGOESM_JAX_COORDINATOR="${COORD_HOST}:${port}"
    echo "=== [mc_nccl_canary] stage ${label} (coord ${LEGOESM_JAX_COORDINATOR}): $* ==="
    if mpiexec --ppn "$PPN" -n "$N" bash -c "$SHIM" _ "$@"; then
        echo "=== [mc_nccl_canary] stage ${label}: PASS ==="
    else
        echo "=== [mc_nccl_canary] stage ${label}: FAIL (rc=$?) — stopping ==="
        exit 1
    fi
}

# --- Stage A: pure-JAX probe (init + psum + ppermute + latency) -------------
run_stage A "$PORT_A" "$PY" scripts/cluster/scaling_derecho/mc_nccl_probe.py \
    --size "$N" --iters 100

# --- Stage B: gated multi-controller parity tests (atm, then ocean) ---------
export LEGOESM_JAX_DISTRIBUTED_TEST=1
run_stage B-atm "$PORT_B1" "$PY" -m pytest -x -q \
    tests/parallel/test_atm_latlon_spmd_multicontroller.py
run_stage B-ocean "$PORT_B2" "$PY" -m pytest -x -q \
    tests/parallel/test_latlon_ocean_spmd_multicontroller.py
unset LEGOESM_JAX_DISTRIBUTED_TEST

# --- Stage C: MC benches, strong scaling at 8 GPU ----------------------------
run_stage C-atm "$PORT_C1" "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
    --multicontroller --coordinator "${COORD_HOST}:${PORT_C1}" \
    --n-devices "$N" --n-lat 256 --n-lon 512 --nlev 30 --steps 12 --warmup 2 \
    --out results/a1/mc_canary_atm.jsonl
run_stage C-ocean "$PORT_C2" "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
    --multicontroller --coordinator "${COORD_HOST}:${PORT_C2}" \
    --n-devices "$N" --n-lat 256 --n-lon 512 --nlev 30 --steps 12 --warmup 2 \
    --out results/a1/mc_canary_ocean.jsonl

echo "=== [mc_nccl_canary] ALL STAGES PASS ==="
