#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=gpuladder6
#SBATCH --time=00:40:00
#SBATCH --nodes=3 --ntasks=6 --ntasks-per-node=2 --cpus-per-task=8 --mem=64G --gres=gpu:2
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/compile_ladder_gpu6_%j.log
# Six GPUs, one whole cube face each. The ladder so far:
#   CPU  1 -> 24 devices: compile 11.6 -> 13.1 s, FLAT. Program shape exonerated.
#   GPU  1 device:        compile 78.3 s. The GPU compiler is exonerated as a
#                         12-hour cause.
# Job 9725446 ran this row for 90 minutes and printed NOTHING, because the
# driver piped every rank through `tail`, which holds all output until exit.
# So that row proved only that six ranks do not finish in 90 minutes; it did
# not say in which phase they are stuck.  Each rank now writes its own
# unbuffered file, and the script stamps lower/compile begin and end, so the
# last line of each rank file names the phase.
set -uo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/ladder6_${SLURM_JOB_ID}
mkdir -p "$OUT"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LEGOESM_JIT_CACHE_DIR=""
export JAX_LOG_COMPILES=1
export JAX_LOCAL_DEVICE_IDS=0
export NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO
export MODE="${MODE:-both}"
# Job 9735694: all six ranks entered compile within 2 s of each other and
# none left it in 38 minutes, at n=24 km=5 pad=5 n_split=1 -- a config that
# compiles in 78 s on ONE gpu with the collapsed (collective-free) program.
# So the stall is XLA:GPU compilation of the sharded program.  N and
# XLAFLAGS are env-tunable so the follow-up arms change ONE thing each.
N=${N:-24}
export XLA_FLAGS="${XLAFLAGS:-}"
cd "$REPO"
echo "host $(hostname)  sha $(git rev-parse HEAD)  dirty $(git status --porcelain | wc -l)"
echo "arm: N=$N XLA_FLAGS=${XLA_FLAGS:-none} TRITON_LOCAL=${TRITON_LOCAL:-0} MODE=$MODE"
echo "per-rank logs: $OUT/rank_*.log"
# GLM 2026-09-11: $HOME is LUSTRE at 94% full and the Triton/XLA kernel caches
# default there, so 6-24 ranks may be contending on one lock rather than
# deadlocking in the autotuner.  TRITON_LOCAL=1 moves every cache to per-rank
# node memory, leaving autotuning ON -- if that alone unblocks the compile,
# autotune_level=0 was masking a filesystem problem, not fixing a compiler one.
if [ "${TRITON_LOCAL:-0}" = 1 ]; then
  srun --gpus-per-task=1 --kill-on-bad-exit=0 \
    --output="$OUT/rank_%t.log" --error="$OUT/rank_%t.log" \
    bash -c 'export TRITON_CACHE_DIR=/dev/shm/triton_$SLURM_PROCID
             [ "${MODE:-both}" = triton ] || export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_per_fusion_autotune_cache_dir=/dev/shm/xlatune_$SLURM_PROCID"
             [ "${MODE:-both}" = xla ] && unset TRITON_CACHE_DIR
             mkdir -p ${TRITON_CACHE_DIR:-/dev/shm/unused_$SLURM_PROCID} /dev/shm/xlatune_$SLURM_PROCID
             echo "rank $SLURM_PROCID TRITON_CACHE_DIR=$TRITON_CACHE_DIR XLA_FLAGS=$XLA_FLAGS"
             exec "$0" "$@"' \
    "$PY" scripts/validate/fv3_native/compile_rank_ladder.py \
    --backend gpu --devices 6 --distributed --n "$N" --km 5 --pad 5 --n-split 1
else
  srun --gpus-per-task=1 --kill-on-bad-exit=0 \
    --output="$OUT/rank_%t.log" --error="$OUT/rank_%t.log" \
    "$PY" scripts/validate/fv3_native/compile_rank_ladder.py \
    --backend gpu --devices 6 --distributed --n "$N" --km 5 --pad 5 --n-split 1
fi
echo "rc=$?"
echo "--- last phase line per rank ---"
for f in "$OUT"/rank_*.log; do
  echo "$f: $(grep '\[compile-rank-ladder\]' "$f" | tail -1)"
done
echo GPULADDER6_DONE
