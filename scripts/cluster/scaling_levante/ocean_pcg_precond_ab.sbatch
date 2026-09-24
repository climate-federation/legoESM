#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=ocean_precond_ab_%j.log

# A/B of the communication-free polynomial preconditioner for the MPAS ocean
# barotropic PCG (MPASOceanConfig.barotropic_implicit_pcg_precond="poly").
#
# PRE-REGISTERED (CLAUDE.md compute discipline):
#   number produced  : steady_median_ms per arm at fixed device count, f32
#                      unless noted; everything but the preconditioner and
#                      the iteration count is identical between arms.
#   CONFIRMS         : poly4 at 20 iterations is faster than jacobi at 30 by
#                      more than 8% at 128 GPUs (the 10 saved iterations cost
#                      ~0.77 ms each there; the extra local mat-vecs must
#                      cost less than that).  The equal-count arm
#                      (poly4 at 30) measures the mat-vec overhead directly.
#   REFUTES          : poly4@20 within noise of, or slower than, jacobi@30 —
#                      the local sweeps cost as much as the communication
#                      they save.
#   why not cheaper  : convergence was measured offline on captured systems
#                      (barotropic_pcg_convergence.py); only the STEP TIME of
#                      the extra mat-vecs against 14-round halo exchanges
#                      needs GPUs, and the plateau only exists from 32 up.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cuda,cpu
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ocean_pcg_precond_ab}"
mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"; PPN=4
echo "nodes=$NODES outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
rc=0

arm() {  # arm <nd> <precond> <sweeps> <iters> <x64>
  local nd="$1" pre="$2" sw="$3" it="$4" x64="$5"
  local nn=$(( (nd + PPN - 1) / PPN ))
  (( nn > NODES )) && { echo "[skip] d$nd needs $nn nodes"; return 0; }
  local tag="omps9_d${nd}_${pre}${sw}_M${it}_x${x64}"
  grep -q '"valid": true' "$OUTDIR/${tag}.jsonl" 2>/dev/null && { echo "[have] $tag"; return 0; }
  rm -f "$OUTDIR/${tag}.jsonl"
  export JAX_ENABLE_X64="$x64"
  echo "=== arm $tag $(date '+%F %T')"
  if ! timeout --kill-after=30 1200 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
        --ntasks-per-node="$PPN" --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
        "$PY" scripts/bench/bench_ocean_mpas_spmd_scaling.py \
        --multicontroller --n-devices "$nd" --subdivision 9 --nlev 40 --lloyd 0 \
        --block-steps 20 --blocks 3 \
        --pcg-precond "$pre" --pcg-poly-sweeps "$sw" --pcg-fixed-iters "$it" \
        --out "$OUTDIR/${tag}.jsonl" < /dev/null; then
    echo "!!! FAILED $tag"; rc=1
    [[ -e "$OUTDIR/${tag}.jsonl" ]] && mv -f "$OUTDIR/${tag}.jsonl" "$OUTDIR/${tag}.failed.jsonl"
  fi
  sleep 5
}

for ND in ${ND_LIST:-128 32}; do
  arm "$ND" jacobi 1 30 0     # today's default
  arm "$ND" poly   4 20 0     # the proposed default
  arm "$ND" poly   4 30 0     # same count as today: isolates the mat-vec cost
  arm "$ND" poly   8 15 0     # more sweeps, fewer iterations
  arm "$ND" jacobi 1 30 1     # f64 pair
  arm "$ND" poly   4 20 1
done
echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit "$rc"
