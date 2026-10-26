#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m6_gpu
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
# The card model is pinned so a mixed allocation cannot turn a bitwise comparison
# into a comparison of GPU kernels. It must be rtx8000, NOT a100: only 8 a100
# nodes exist in `short` and the exclusion below removes five of them (g189-g193),
# leaving 3 against the 12 this gate needs -- an a100 pin here is unsatisfiable.
#SBATCH --constraint=rtx8000
#SBATCH --exclusive
# NODE EXCLUSION (user call 2026-09-16). These nodes compile the sharded
# step in full and then HANG in its first execution -- NCCL reaches
# "Connected all trees" on every rank and stops. Confirmed by a pinned-node
# A/B at six ranks: g[041-043] scored in 4 min 48 s (job 9777265), the
# identical job on g[188,190,193] was killed at 30 min (job 9777266); the
# hangs of jobs 9756806, 9756887, 9758308 and 9776325 all include nodes
# from these ranges and every completion ran on g041-044 + g094-101.
# A bare neighbour ppermute passes on them at every payload from 16 KB to
# 32 MB (jobs 9785525 vs 9785526, within 35 % of the good nodes), so the
# fault is not raw fabric bandwidth and the reproducer is the gate itself.
# DELETE THIS LINE once the nodes are fixed -- it is a workaround, not a
# property of the model.
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m6_gpu_gate_%j.log
# M6 on REAL GPU RANKS: one window per process, one A100 per process,
# jax.distributed via SLURM, NCCL collectives.  2 GPUs per node.
# Usage: sbatch --nodes=N --ntasks=2N --ntasks-per-node=2 --export=ALL,KT=..,PAD=..,REF=<npz> this
set -uo pipefail
WT=${WT:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT" || exit 1
export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 JAX_LOG_COMPILES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
# one GPU per task: --gpus-per-task cgroups the other GPU away but the CUDA
# driver still counts it, and jax then fails on CUDA:1 (job 9650417)
export JAX_LOCAL_DEVICE_IDS=0
# NCCL must use the InfiniBand interface, not the management ethernet
# (duo_ranks_venue.sbatch, job 9610225); job 9663001 hung 4 h without it
export NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
N=${N:-48}; KM=${KM:-10}; KT=${KT:-2}; PAD=${PAD:-11}; NHFLAG=""; [ "${NH:-0}" = 1 ] && NHFLAG="--nh"
echo "host: $(hostname) ranks=$SLURM_NTASKS nodes=$SLURM_JOB_NUM_NODES sha: $(git rev-parse HEAD) dirty: $(git status --porcelain | wc -l)"
nvidia-smi -L || true
echo "XLA_FLAGS=$XLA_FLAGS"
# PREFLIGHT, record-only (2026-09-13): two 24-rank runs (9756806, 9756887)
# compiled in ~100 s and then hung in the first EXECUTION; a third (9756793)
# on overlapping nodes ran through.  The MPI launcher refuses a slow fabric
# from one bare neighbour ppermute; here the number is only RECORDED next to
# the outcome until enough rows exist to justify a threshold (no threshold
# has been chosen -- ASK before adding one).
PREOUT=$(srun --gpus-per-task=1 "$PY" scripts/bench/bench_fv3_duo_spmd_scaling.py --mode bare --n-timed 30 2>&1)
echo "preflight: $(echo "$PREOUT" | grep -m1 'ONE neighbour ppermute' || echo "NO RESULT: $(echo "$PREOUT" | grep -m1 -iE 'error|Traceback' )") (nodes $SLURM_JOB_NODELIST)"
echo "$PREOUT" | grep -m1 "BARE-SLOWEST" || true
echo "===== M6 GPU gate C$N km=$KM ${NHFLAG:-hydro} kt=$KT pad=$PAD ranks=$SLURM_NTASKS ref=${REF:-none} ====="
# GPU rows are judged within GPU_MAX_REL_PEAK of each leaf's peak (user call
# 2026-09-15): two XLA:GPU programs differ at rounding, measured 1.5e-14.
GPU_MAX_REL_PEAK=${GPU_MAX_REL_PEAK:-1e-12}
echo "GPU_MAX_REL_PEAK=$GPU_MAX_REL_PEAK"
srun --gpus-per-task=1 --kill-on-bad-exit=0 "$PY" scripts/validate/fv3_native/tiled_m6_model_gate.py --n "$N" --km "$KM" $NHFLAG --kt "$KT" --pad "$PAD" --n-split "${NSPLIT:-3}" --steps "${STEPS:-2}" --distributed ${REF:+--ref-npz $REF} --gpu-max-rel-peak "$GPU_MAX_REL_PEAK" ${GATE_EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^$|RuntimeError: jaxlib|Deprecat|shard_map import"
RC=${PIPESTATUS[0]}
echo "SRUN_RC=$RC"
echo DONE m6_gpu_gate
exit $RC
