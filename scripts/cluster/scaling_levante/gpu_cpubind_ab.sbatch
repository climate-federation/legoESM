#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=gpu_cpubind_ab_%j.log

# Host-core binding A/B for the GPU ladders (2026-09-21).  nature_ladder.sbatch
# launches every arm with `srun --ntasks-per-node=4` and no --cpus-per-task,
# which on the CPU lanes bound each rank to ONE core of its 32 (measured:
# affinity=2 hw threads).  The GPU lanes pass no --cpu-bind, so their host
# side is either unbound (256 cpus, one oversized XLA pool per rank) or
# 1-core bound; either way the NCCL proxy, CUDA driver and XLA host threads
# of 4 ranks share what SLURM gave them.  Arms at s9 (atm) and s9 (ocean),
# 32 GPUs, f32: the ladder launch vs `-c 64 --cpu-bind=cores`.
#   number: steady_median_ms per arm, plus each rank's affinity size
#   CONFIRMS: bound arm faster by more than the repeat spread (~1.5%)
#   REFUTES : equal within spread -> GPU ladders stand as measured
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cuda,cpu
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=131072
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/gpu_cpubind_ab}"
mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"; PPN=4; ND=$(( NODES * PPN ))
echo "nodes=$NODES nd=$ND outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
rc=0
arm() {  # arm <lane> <label> <srun extra args...>
  local lane="$1" label="$2"; shift 2
  local tag="${lane}9_d${ND}_${label}"
  grep -q '"valid": true' "$OUTDIR/${tag}.jsonl" 2>/dev/null && { echo "[have] $tag"; return 0; }
  rm -f "$OUTDIR/${tag}.jsonl"
  local bench args
  if [[ "$lane" == mpas ]]; then
    bench=scripts/bench/bench_mpas_spmd_scaling.py
    args=(--subdivision 9 --nlev 26 --lloyd 0 --wide-halo --partition-method sfc --steps 60 --warmup 10)
  else
    bench=scripts/bench/bench_ocean_mpas_spmd_scaling.py
    args=(--subdivision 9 --nlev 40 --lloyd 0 --block-steps 20 --blocks 3)
  fi
  echo "=== arm $tag $(date '+%F %T')  srun extra: $*"
  srun --export=ALL --nodes="$NODES" --ntasks="$ND" --ntasks-per-node="$PPN" --gpus-per-node=4 \
       --gpu-bind=none "$@" bash -c 'echo "affinity cpus=$(nproc)"' | sort | uniq -c
  if ! timeout --kill-after=30 1500 srun --export=ALL --nodes="$NODES" --ntasks="$ND" \
        --ntasks-per-node="$PPN" --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 "$@" \
        "$PY" "$bench" --multicontroller --n-devices "$ND" "${args[@]}" \
        --out "$OUTDIR/${tag}.jsonl" < /dev/null; then
    echo "!!! FAILED $tag"; rc=1
    [[ -e "$OUTDIR/${tag}.jsonl" ]] && mv -f "$OUTDIR/${tag}.jsonl" "$OUTDIR/${tag}.failed.jsonl"
  fi
  sleep 5
}
arm mpas ladder
arm mpas bound     -c 64 --cpu-bind=cores
arm mpas ladder_rep
arm mpas bound_rep -c 64 --cpu-bind=cores
arm omps ladder
arm omps bound     -c 64 --cpu-bind=cores
arm omps ladder_rep
arm omps bound_rep -c 64 --cpu-bind=cores
echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit "$rc"
