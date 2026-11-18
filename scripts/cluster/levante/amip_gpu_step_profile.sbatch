#!/usr/bin/env bash
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --job-name=amip_gpu_prof
#SBATCH --output=/scratch/b/b384341/amip_gpu_profile/%x-%a-%A.out
# Step-cost profile of the production AMIP deck on the production hardware
# (4 x A100, one MPI rank per GPU): one simulated day (768 steps at dt 112.5)
# with LEGOESM_PHASE_TIMERS=1 (model.step / land / other, every 24 steps) and
# one jax.profiler trace window on rank 0 covering steps 256-296 (one
# radiation step at rad_update_steps=32, three physics steps).  The
# steady-state s/step of the second half-day is the baseline every speed-up
# on perf/amip-speedup is measured against.
#   array 0: the deck as committed (rrtmgp_gpoint_batch_size 0 = the CPU
#            recipe: one g-point per scan step)
#   array 1: + --rrtmgp-gpoint-batch-size 16 (the code's GPU default)
#   array 2/3: the same two arms on ONE GPU, serial driver (user 2026-11-12:
#            single-GPU is a production target too); submit with
#            sbatch --ntasks=1 --ntasks-per-node=1 --gpus-per-node=1 \
#                   --cpus-per-task=16 --mem=120G --array=2-3 <this file>
#            (sbatch command-line resources override the header; N_GPUS is
#            read from the allocation, so the header's 4-rank set and the
#            1-rank set are the same script).
# Pre-registered: arm 1 radiation-step wall < arm 0; fluxes differ only by
# summation order (deck comment: <= 6.1e-4 W/m2 on CPU).
# Recorded deviations from amip_production.sh (same as the 2026-11-04
# segment-schedule receipt; this account cannot read the pinned paths): the
# surface data is the same file in the bb1596 checkout, the IC is the
# Levante-available 37-level ERA5 1979-01-01 store, the land IC is the
# bb1596 copy of the deck's res-6 spun-up state with ONE column (19808, 6.3 mm)
# lifted to the loader's Richards dry floor (regrid_land_ic.lift_to_dry_floor
# applied in place, nothing else touched: the unlifted copy is refused by the
# loader on rank 2 of 4, #2496 class).  Speed measurement only; not production.
set -uo pipefail
: "${REPO:=/home/b/b384341/projects/legoESM-perf}"
: "${ROOT:=/scratch/b/b384341/amip_gpu_profile}"
: "${DAYS:=1}"
: "${PY:=${REPO}/.venv/bin/python}"
: "${AMIP_SURFDATA:=/work/bb1596/b309178/diffESM/legoesm_ap/data/legoesm_surfdata_c260716.nc}"
: "${ERA5_IC:=/work/bb1596/b309178/diffESM/legoesm_ap_main/data/era5_ic_1979-01-01.zarr}"
: "${LAND_IC:=/scratch/b/b384341/amip_gpu_profile/data/soil_ic_mpas6_era5T_19790101_lifted.npz}"
export AMIP_SURFDATA ERA5_IC
: "${ARM_ID:=${SLURM_ARRAY_TASK_ID:-0}}"
ARMS=( "" "--rrtmgp-gpoint-batch-size 16" "" "--rrtmgp-gpoint-batch-size 16" )
ARM_NAMES=( deck4 gpt16_4 deck1 gpt16_1 )
[[ -n "${TRACE_SPEC:-}" ]] && ARM_NAMES=( "${ARM_NAMES[@]/%/_t${TRACE_SPEC%%:*}}" )
N_GPUS="${SLURM_NTASKS:-4}"
if (( ARM_ID >= 2 && N_GPUS != 1 )) || (( ARM_ID < 2 && N_GPUS != 4 )); then
  echo "arm ${ARM_ID} expects $(( ARM_ID >= 2 ? 1 : 4 )) rank(s), allocation has ${N_GPUS}" >&2; exit 64
fi
ARM_EXTRA="${ARMS[${ARM_ID}]}"
NAME="${ARM_NAMES[${ARM_ID}]}"
# EXTRA_FLAGS="..." (with ARM_NAME) replaces the arm's flags: ad-hoc ladders
# (e.g. the g-point batch-size ladder) without editing the arm table.
if [[ -n "${EXTRA_FLAGS:-}" ]]; then
  ARM_EXTRA="${EXTRA_FLAGS}"; NAME="${ARM_NAME:?EXTRA_FLAGS needs ARM_NAME}"
fi
[[ "${LAUNCHER:-srun}" == "hpcx" ]] && NAME="${NAME}_hpcx_cmpi${CUDA_MPI:-unset}"
OUTDIR="${ROOT}/${NAME}_${SLURM_JOB_ID:-local}"
mkdir -p "${OUTDIR}"
source "${REPO}/config/amip/amip_production.sh"
amip_require_forcing_files || exit 3
PKG="${REPO}/packages"
export PYTHONPATH="${REPO}/src:${PKG}/core:${PKG}/atmosphere:${PKG}/coupler:${PKG}/tools:${PKG}/ml:${PKG}/ocean:${PKG}/land:${PKG}/ice"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-/scratch/b/b384341/legoesm_jit_cache}"
mkdir -p "${LEGOESM_JIT_CACHE_DIR}"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
export LD_LIBRARY_PATH="/sw/spack-levante/openmpi-4.1.2-mnmady/lib:/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPI4JAX_USE_CUDA_MPI=0
export LEGOESM_PHASE_TIMERS=1
# HLO_DUMP=1: also dump the optimised HLO of the step and land modules (text),
# the map from a kernel name in the trace (loop_add_fusion_324) to the JAX
# source scope that produced it.  Compile-time cost only.
if [[ "${HLO_DUMP:-0}" == "1" ]]; then
  export XLA_FLAGS="--xla_dump_to=${OUTDIR}/hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit__step_jit|jit__land_step${XLA_FLAGS:+ $XLA_FLAGS}"
  NAME="${NAME}_hlo"; OUTDIR="${ROOT}/${NAME}_${SLURM_JOB_ID:-local}"; mkdir -p "${OUTDIR}"
  export XLA_FLAGS="${XLA_FLAGS/--xla_dump_to=*\/hlo/--xla_dump_to=${OUTDIR}\/hlo}"
  export LEGOESM_TRACE_STEPS="${TRACE_SPEC:-256:40}:${OUTDIR}/trace"
fi
# TRACE_SPEC="first:count" (default 256:40: one radiation step + held steps; the
# profiler keeps ~1e6 events, so a radiation step fills it and a 6-step window
# at 270 -- held steps + the no-radiation physics step 272 -- is the physics probe)
export LEGOESM_TRACE_STEPS="${TRACE_SPEC:-256:40}:${OUTDIR}/trace" LEGOESM_TRACE_RANKS=0
export PATH=/sw/spack-levante/git-2.43.7-2ofazl/bin:$PATH
cd "${REPO}"
echo "AMIP GPU step profile  arm=${NAME} extra='${ARM_EXTRA}' job=${SLURM_JOB_ID:-local} sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) $(date -u)"
nvidia-smi -L 2>&1 | head -4 || true
CMD=( "${PY}" -u scripts/run/run_amip.py
      --config config/amip/amip_production.yaml
      "${AMIP_PATH_FLAGS[@]}"
      --land-ic "${LAND_IC}"
      --days "${DAYS}"
      --output "${OUTDIR}" )
(( N_GPUS > 1 )) && CMD+=( --distributed )
[[ -n "${ARM_EXTRA}" ]] && CMD+=( ${ARM_EXTRA} )
echo "launch (${N_GPUS} rank(s)): ${CMD[*]}"
_CPUS="${SLURM_CPUS_PER_TASK:-8}"
unset SLURM_CPUS_PER_TASK
_GPU_MAP=$(seq -s, 0 $((N_GPUS - 1)))
if [[ "${LAUNCHER:-srun}" == "hpcx" ]]; then
  # NVIDIA HPC-X MPI (CUDA-aware) under its own mpirun: it cannot start under
  # srun on these nodes (probe job 29077883).  One GPU per local rank via the
  # wrapper.  CUDA_MPI=1 hands device buffers to MPI (GPU-direct halos),
  # CUDA_MPI=0 stages them through the host as the production stack does.
  # Every mpirun process inherits the ONE SLURM_PROCID of the step that
  # started mpirun's helper, and the driver reads SLURM_PROCID before its
  # OMPI_COMM_WORLD_RANK fallback when it picks the traced rank; the wrapper
  # unsets it so each process reports its own MPI rank.
  source "${REPO}/scripts/cluster/levante/hpcx_gdr_env.sh"
  export MPI4JAX_USE_CUDA_MPI="${CUDA_MPI:?LAUNCHER=hpcx needs CUDA_MPI=0|1}"
  CMD[0]="${REPO}/.venv-gdr/bin/python"
  _WRAP="${OUTDIR}/gpu_local_rank.sh"
  printf '#!/bin/bash\nunset SLURM_PROCID\nexport CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}\nexec "$@"\n' > "${_WRAP}"
  chmod +x "${_WRAP}"
  echo "launch via HPC-X mpirun, MPI4JAX_USE_CUDA_MPI=${MPI4JAX_USE_CUDA_MPI}: ${CMD[*]}"
  mpirun -np "${N_GPUS}" --bind-to none \
    -x PYTHONPATH -x JAX_PLATFORMS -x JAX_ENABLE_X64 -x XLA_PYTHON_CLIENT_PREALLOCATE \
    -x XLA_PYTHON_CLIENT_MEM_FRACTION -x LEGOESM_JIT_CACHE_DIR -x LEGOESM_PHASE_TIMERS \
    -x MPI4JAX_USE_CUDA_MPI -x LD_LIBRARY_PATH -x PATH -x OPAL_PREFIX \
    -x LEGOESM_TRACE_STEPS -x LEGOESM_TRACE_RANKS \
    -x AMIP_SURFDATA -x ERA5_IC "${_WRAP}" "${CMD[@]}"
else
  srun --ntasks="${N_GPUS}" --ntasks-per-node="${N_GPUS}" --gpus-per-node="${N_GPUS}" \
       --gpu-bind="map_gpu:${_GPU_MAP}" --cpus-per-task="${_CPUS}" \
       --kill-on-bad-exit=1 "${CMD[@]}"
fi
RC=$?
echo "run_amip exit=${RC}  $(date -u)"
grep -h "phase timers" "${OUTDIR}"/*.log 2>/dev/null | tail -5
exit "${RC}"
