#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=ocean_btr_probe_%j.log

# Which part of the ocean-MPAS strong-scaling plateau does the barotropic
# solve own?  The lane flattens at ~81 ms/step from 32 GPUs on (83.74 / 81.20
# / 82.41 at 32 / 64 / 128), and the distributed implicit_cn solve is the
# obvious suspect: at the default M=60 iterations it costs 1+2M = 121
# latency-serialised global allreduces per step plus M cell-halo exchanges.
#
# PRE-REGISTERED (CLAUDE.md compute discipline):
#   number produced  : steady_median_ms per arm at fixed device count.
#   CONFIRMS         : ms is linear in M with an INTERCEPT below ~35 ms at
#                      d=128, i.e. the solve owns more than half the plateau;
#                      and the single_reduce arm (same M, half the allreduces)
#                      recovers a visible part of the standard->M/2 gap.
#   REFUTES          : intercept above ~65 ms, i.e. the solve owns under 20%
#                      and the plateau lives in the baroclinic halo or in
#                      per-step fixed overhead.
#   why not cheaper  : the plateau does not exist below 32 GPUs, so no
#                      single-node or offline test can reproduce it.
#
# Lowering M CHANGES THE SOLVE and these rows never enter the figure; they are
# a slope measurement.  The single_reduce arm is different: it is the same
# solve in exact arithmetic (Chronopoulos-Gear) and is parity-gated, so if it
# wins it is a real lever rather than a probe.
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ocean_btr_probe}"
mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"; PPN=4
echo "nodes=$NODES outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
rc=0

arm() {  # arm <nd> <variant> <iters> <x64>
  local nd="$1" var="$2" it="$3" x64="$4"
  local nn=$(( (nd + PPN - 1) / PPN ))
  (( nn > NODES )) && { echo "[skip] d$nd needs $nn nodes"; return 0; }
  local tag="omps9_d${nd}_${var}_M${it}_x${x64}"
  grep -q '"valid": true' "$OUTDIR/${tag}.jsonl" 2>/dev/null && { echo "[have] $tag"; return 0; }
  rm -f "$OUTDIR/${tag}.jsonl"
  export JAX_ENABLE_X64="$x64"
  echo "=== arm $tag $(date '+%F %T')"
  if ! timeout --kill-after=30 1200 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
        --ntasks-per-node="$PPN" --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
        "$PY" scripts/bench/bench_ocean_mpas_spmd_scaling.py \
        --multicontroller --n-devices "$nd" --subdivision 9 --nlev 40 --lloyd 0 \
        --block-steps 20 --blocks 3 \
        --pcg-variant "$var" --pcg-fixed-iters "$it" \
        --out "$OUTDIR/${tag}.jsonl" < /dev/null; then
    echo "!!! FAILED $tag"; rc=1
    [[ -e "$OUTDIR/${tag}.jsonl" ]] && mv -f "$OUTDIR/${tag}.jsonl" "$OUTDIR/${tag}.failed.jsonl"
  fi
  sleep 5
}

for ND in ${ND_LIST:-128 32}; do
  arm "$ND" standard      60 0     # baseline: reproduces the plateau row
  arm "$ND" standard      30 0     # slope
  arm "$ND" standard      15 0     # slope
  arm "$ND" single_reduce 60 0     # same solve, half the allreduces
  arm "$ND" single_reduce 60 1     # and in f64, where the plateau is highest
  arm "$ND" standard      60 1
done
echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit "$rc"
