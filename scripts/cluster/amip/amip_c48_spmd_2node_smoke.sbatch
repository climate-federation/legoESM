#!/usr/bin/env bash
#SBATCH --job-name=amip_c48_spmd_2node
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4
# --mem=0 = ALL node memory: 3 processes/node each run the full production-
# model XLA compile concurrently; the old 128G cap (sized for 1 process/node)
# blew up with std::bad_alloc + SIGSEGV inside xla_cuda_plugin (job 26037711).
#SBATCH --mem=0
#SBATCH --time=02:00:00
#
# =============================================================================
# First real >=2-node GPU run of run_amip via the cs_spmd lane (#693 / PR #749)
# =============================================================================
# The receipt Pierre left open on #693: "First real >=2-node GPU run on
# Levante/Derecho via the spmd route." PR #749 (merged to main 2026-07-03) put
# the exact harness on main; this smoke runs it on Levante.
#
# Route (Pierre's, adapted to the Levante cube-face shape):
#   srun --mpi=pmix ... run_amip.py --config amip_production.yaml \
#       --distributed --distributed-mode spmd
#
# spmd = ONE global device mesh over jax.devices() via jax.distributed + NCCL
# collectives — NO mpi4jax anywhere on the path, so the mpi4jax/jax.distributed
# coexistence hazard behind the multi-node day-5 divergence (#693) is absent.
#
# Shape: 2 nodes x 3 GPU = 6 processes, ONE cubed-sphere face per GPU (valid
# cube decompositions divide 6; 8 would not).
#
# NO --gpu-bind: SLURM's --gpu-bind=single:1 puts each task's GPU in its own
# cgroup, which blocks the CUDA IPC that NCCL needs between on-node peers —
# both 2x3-GPU attempts died with "Cuda failure 101 'invalid device ordinal'"
# (26030299 at ncclGroupEnd, 26030422 at ncclCommInitRankConfig). Instead all
# 3 allocated GPUs stay visible to every task and early_init's
# resolve_local_device_ids() binds each process to its GPU via SLURM_LOCALID
# (the standard JAX-on-SLURM recipe). The 2n x 1g variant worked with the old
# pinning only because it had no on-node peers.
#
# Short smoke: DAYS=2 (JIT + 2 integrated days + finite-state verdict). The
# harness-fixed run_amip (#748) exits non-zero on BLOWUP, so exit 0 here is a
# REAL receipt, not a masked one.
#
# Usage:
#   sbatch scripts/cluster/amip/amip_c48_spmd_2node_smoke.sbatch
# Overridable via --export: DAYS, OUTDIR.
# =============================================================================

set -uo pipefail

: "${REPO:=/work/bd1083/b309178/diffESM/legoesm_ap}"
: "${OUTDIR:=/scratch/b/b309178/amip_c48_spmd_2node_smoke}"
: "${DAYS:=2}"
PY=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python
PKG="${REPO}/packages"
CONFIG="${REPO}/config/amip/amip_production.yaml"

# shellcheck source=/dev/null
source "${REPO}/config/amip/amip_production.sh"

mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/slurm-${SLURM_JOB_ID:-local}.out") \
     2> >(tee -a "${OUTDIR}/slurm-${SLURM_JOB_ID:-local}.err" >&2)

for p in "${ERA5_IC}" "${ETOPO}" "${SST}" "${SIC}" "${SOLAR}" "${GHG}" "${OZONE}" "${AEROSOL}" "${VOLCANIC}"; do
    [[ -e "${p}" ]] || { echo "ERROR: missing ${p}"; exit 1; }
done

export PYTHONPATH="${REPO}/src:${PKG}/core:${PKG}/atmosphere:${PKG}/coupler:${PKG}/tools:${PKG}/ml:${PKG}/ocean:${PKG}/land:${PKG}/ice"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# Sharded autotuning distributes GEMM autotuning across the processes and
# exchanges results through the coordination service; with >1 process/node it
# segfaulted (addr 0x1) + threw std::bad_alloc inside xla_cuda_plugin while
# compiling the subcycled segment (jobs 26037711/26037749 — identical crash
# with 128G and 997G, so not host RAM). The 2n x 1-GPU run compiled clean.
# Autotune locally in every process instead.
export XLA_FLAGS="--xla_gpu_shard_autotuning=false ${XLA_FLAGS:-}"

BRANCH=$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "AMIP C48/L40 cs_spmd 2-NODE smoke (#693 receipt)  job=${SLURM_JOB_ID:-local}  branch=${BRANCH}  $(date -u)"
echo "  CONFIG: ${CONFIG}   nodes=${SLURM_NNODES:-?}  ntasks=${SLURM_NTASKS:-?}  mode=spmd"
echo "  OUTDIR: ${OUTDIR}  DAYS: ${DAYS}"

cd "${REPO}"
srun --mpi=pmix \
    "${PY}" -u scripts/run/run_amip.py \
        --config "${CONFIG}" \
        "${AMIP_PATH_FLAGS[@]}" \
        --distributed --distributed-mode spmd \
        --days "${DAYS}" \
        --diag-days 1 \
        --output "${OUTDIR}"
RC=$?
echo "[amip-spmd-2node] run_amip exit=${RC}  $(date -u)"
exit "${RC}"
