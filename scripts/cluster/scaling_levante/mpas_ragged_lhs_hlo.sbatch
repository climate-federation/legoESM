#!/bin/bash -l
#SBATCH --job-name=rag_lhs_hlo
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=rag_lhs_hlo.%j.log
# Does XLA's latency-hiding scheduler actually make the ragged-all-to-all
# ASYNC (start/done pair) in our step, or is the flag inert for this op?
# (GLM r2: without this check a NULL overlap A/B is uninterpretable —
# inert-flag and no-independent-work read identically in timings.)
# Method: compile+run 4 steps of s6@8 wide+ragged with the overlap flags
# and --xla_dump_to; then grep the post-scheduling HLO for the ragged
# custom call and async-start/done structure around it.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
DUMP="$SCRATCH/legoesm_scaling/lhs_hlo_j${SLURM_JOB_ID}"
mkdir -p "$DUMP"

env -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS -u NCCL_P2P_NET_CHUNKSIZE \
    XLA_FLAGS="--xla_gpu_shard_autotuning=false --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true --xla_dump_to=$DUMP" \
    LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=1 JAX_ENABLE_X64=0 \
  timeout 900 srun --export=ALL --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 8 --subdivision 6 --nlev 8 \
    --steps 4 --warmup 1 --lloyd 0 --partition-method sfc \
    --halo-strategy ppermute \
    --out "$DUMP/run.jsonl" || echo "run FAILED (dump may still exist)"

echo "=== HLO CENSUS (post-scheduling module) ==="
MOD=$(ls -S "$DUMP"/*jit__scan_run*after_optimizations.txt 2>/dev/null | head -1)
[ -z "$MOD" ] && MOD=$(ls -S "$DUMP"/*after_optimizations.txt 2>/dev/null | head -1)
echo "module: $(basename "$MOD" 2>/dev/null)"
if [ -n "$MOD" ]; then
  echo "--- ragged custom calls:"
  grep -c "ragged_all_to_all" "$MOD" || true
  echo "--- async start/done ops:"
  grep -oE "async-(start|done|update)" "$MOD" | sort | uniq -c || true
  echo "--- lines mentioning ragged (context):"
  grep -nE "ragged" "$MOD" | head -20
  echo "--- custom_call_schedule attrs:"
  grep -oE "schedule=[A-Z_]+" "$MOD" | sort | uniq -c || true
fi
echo "DONE dump=$DUMP"; exit 0
