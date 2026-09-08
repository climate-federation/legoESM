#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m6_gpu
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m6_gpu_gate_%j.log
# M6 on REAL GPU RANKS: one window per process, one A100 per process,
# jax.distributed via SLURM, NCCL collectives.  2 GPUs per node.
# COMPILE TIME (2026-09-08): the n_split-unrolled step takes HOURS to
# compile for CUDA -- jobs 9663001/9668306/9677460 were killed at 1-4 h
# and looked like NCCL hangs; a SIGUSR1 stack dump (the gate registers
# faulthandler) showed every rank inside backend_compile_and_load.  Hence
# the 8 h cap, and the persistent JIT cache is left ON here (unlike the
# many-rank CPU launcher) so a repeat run skips the compile.
# Usage: sbatch --nodes=N --ntasks=2N --ntasks-per-node=2 --export=ALL,KT=..,PAD=..,REF=<npz> this
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT" || exit 1
export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# one GPU per task: --gpus-per-task cgroups the other GPU away but the CUDA
# driver still counts it, and jax then fails on CUDA:1 (job 9650417)
export JAX_LOCAL_DEVICE_IDS=0
# NCCL must use the InfiniBand interface, not the management ethernet
# (duo_ranks_venue.sbatch, job 9610225); job 9663001 hung 4 h without it
export NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=WARN
N=${N:-48}; KM=${KM:-10}; KT=${KT:-2}; PAD=${PAD:-11}; NHFLAG=""; [ "${NH:-0}" = 1 ] && NHFLAG="--nh"
echo "host: $(hostname) ranks=$SLURM_NTASKS nodes=$SLURM_JOB_NUM_NODES sha: $(git rev-parse HEAD) dirty: $(git status --porcelain | wc -l)"
nvidia-smi -L || true
echo "===== M6 GPU gate C$N km=$KM ${NHFLAG:-hydro} kt=$KT pad=$PAD ranks=$SLURM_NTASKS ref=${REF:-none} ====="
srun --gpus-per-task=1 --kill-on-bad-exit=0 "$PY" scripts/validate/fv3_native/tiled_m6_model_gate.py --n "$N" --km "$KM" $NHFLAG --kt "$KT" --pad "$PAD" --n-split "${NSPLIT:-3}" --steps "${STEPS:-2}" --distributed ${REF:+--ref-npz $REF} ${GATE_EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^$|RuntimeError: jaxlib|Deprecat|shard_map import"
RC=${PIPESTATUS[0]}
echo "SRUN_RC=$RC"
echo DONE m6_gpu_gate
exit $RC
