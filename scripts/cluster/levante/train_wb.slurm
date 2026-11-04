#!/bin/bash
# WeatherBench training, data-parallel on DKRZ Levante (4x A100-80GB per node).
# One MPI rank per GPU; each rank trains on its own shard of the samples and the
# ranks average gradients every step (mpi4jax allreduce through host memory).
#
#   sbatch --nodes=<N> --export=ALL,WB_CONFIG=<deck>,WB_MODE=<sfno|neural_gcm>,WB_OUT=<dir> \
#     scripts/cluster/levante/train_wb.slurm
#
# Optional: WB_EXTRA (extra trainer flags, e.g. "--epochs 4"), WB_TIME_ONLY=1
# (timing run: no --resume, nothing else changes).  Resubmitting the same job
# with the same WB_OUT continues from the newest complete checkpoint.
#SBATCH --job-name=wb_dp_train
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=wb_dp_train_%j.out
set -uo pipefail
REPO="${LEGOESM_REPO:-${SLURM_SUBMIT_DIR:?}}"
cd "$REPO" || exit 1
PY="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export PYTHONPATH="$(ls -d "$REPO"/packages/*/ | tr '\n' ':')$REPO/src:$REPO"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# The persistent JAX compilation cache aborts under MPI on this cluster.
export LEGOESM_JIT_CACHE_DIR=""
# The OpenMPI the venv's mpi4py runs against; CUDA-aware MPI stays OFF
# (MPI4JAX_USE_CUDA_MPI=1 segfaults the gradient allreduce here).
export LD_LIBRARY_PATH="/sw/spack-levante/openmpi-4.1.2-mnmady/lib:/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPI4JAX_USE_CUDA_MPI=0
# srun refuses two different cpus-per-task values; state it on the srun line.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
RESUME=(--resume)
[ "${WB_TIME_ONLY:-0}" = "1" ] && RESUME=()
echo "[wb_dp] $(git -c safe.directory='*' rev-parse --short HEAD) ranks=${SLURM_NTASKS} deck=${WB_CONFIG:?} mode=${WB_MODE:?} start=$(date +%s)"
# shellcheck disable=SC2086
srun --ntasks="${SLURM_NTASKS}" --ntasks-per-node=4 --gpus-per-node=4 \
     --gpu-bind=map_gpu:0,1,2,3 --cpus-per-task=16 --kill-on-bad-exit=1 \
  "$PY" -u scripts/run/train_weatherbench_scale.py \
    --config "$WB_CONFIG" --mode "$WB_MODE" --training-core spectral \
    --out "${WB_OUT:?}" "${RESUME[@]}" ${WB_EXTRA:-}
rc=$?
echo "[wb_dp] exit=$rc end=$(date +%s)"
exit $rc
