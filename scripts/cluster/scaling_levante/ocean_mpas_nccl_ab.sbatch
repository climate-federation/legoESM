#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
# STALL_DIAG=1 needs --time=03:00:00 at submission (two 2400 s arms + setup).
#SBATCH --output=ocean_mpas_nccl_ab_%j.log

# NCCL channel-count A/B for the MPAS-OCEAN lane (2026-09-21).  The
# atmosphere sweep (atm_mpas_nccl_sweep.sbatch) measured 8 -> 32 channels
# as -16.5% at 128 GPUs / -5.7% at 32 on a step that is one wide halo
# exchange + one allreduce.  The ocean step is (M+8) halo exchanges and
# 2M+9 allreduces from the fixed-iteration PCG (M=20): a different
# collective mix, so the gain need not transfer (both reviewers).  Both
# MPAS lanes are already pinned at 32 by decision; this decides whether
# the ocean pin is a gain, a wash, or a loss.
#   number it produces: steady_median_ms of omps s9 f32 at 32 (and 128 if
#                       allocated) GPUs for 8 ch, 32 ch, 8 ch repeat
#   CONFIRMS transfer : 32 ch beats 8 ch by more than the 8-vs-8 repeat
#                       spread plus 3%
#   REFUTES           : within the repeat spread, or slower -> the ocean
#                       lane goes back to 8 and the plotter guard becomes
#                       per-lane
#   why not cheaper   : env-only; the ocean ladder is re-measured after it
#                       either way, so this is the cheapest possible check.
# RESULT 32 GPUs (job 27582583, s9 f32): 8 ch 43.46 / 43.35 (repeat,
#   spread 0.25%), 32 ch 42.39 -> -2.4%.  Below the pre-registered +3%
#   transfer bar, so the atmosphere-sized gain does NOT transfer at 32; it
#   is a small real gain, not a loss, so the user's both-lanes pin (32)
#   stands and the ocean ladder is re-measured at 32.  128-GPU arms queued.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cuda,cpu
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ocean_mpas_nccl_ab}"
mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"; PPN=4
echo "nodes=$NODES outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
rc=0

# STALL_DIAG=1: the 128-GPU 32-channel arm (job 27582584) never reached its
# first collective in 1200 s while the 8-channel arm finished in ~600 s.
# Diagnostic mode: NCCL init logging, 2400 s limit, and a Python traceback
# of every rank dumped at 1500 s (faulthandler) to see where it sits.
STALL_DIAG="${STALL_DIAG:-0}"
LIMIT=1200; RUNNER=("$PY")
if [[ "$STALL_DIAG" == 1 ]]; then
  LIMIT=2400
  export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,NET
  RUNNER=("$PY" -c 'import faulthandler, runpy, sys; faulthandler.dump_traceback_later(1500, repeat=False); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")')
fi
arm() {  # arm <nd> <label> <env assignments...>
  local nd="$1" label="$2"; shift 2
  local nn=$(( (nd + PPN - 1) / PPN ))
  (( nn > NODES )) && { echo "[skip] d$nd needs $nn nodes"; return 0; }
  local tag="omps9_d${nd}_${label}"
  grep -q '"valid": true' "$OUTDIR/${tag}.jsonl" 2>/dev/null && { echo "[have] $tag"; return 0; }
  rm -f "$OUTDIR/${tag}.jsonl"
  echo "=== arm $tag $(date '+%F %T')  env: $*"
  if ! env "$@" timeout --kill-after=30 "$LIMIT" srun --export=ALL --nodes="$nn" --ntasks="$nd" \
        --ntasks-per-node="$PPN" --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
        "${RUNNER[@]}" scripts/bench/bench_ocean_mpas_spmd_scaling.py \
        --multicontroller --n-devices "$nd" --subdivision 9 --nlev 40 --lloyd 0 \
        --block-steps 20 --blocks 3 \
        --out "$OUTDIR/${tag}.jsonl" < /dev/null; then
    echo "!!! FAILED $tag"; rc=1
    [[ -e "$OUTDIR/${tag}.jsonl" ]] && mv -f "$OUTDIR/${tag}.jsonl" "$OUTDIR/${tag}.failed.jsonl"
  fi
  sleep 5
}

CH=131072
if [[ "$STALL_DIAG" == 1 ]]; then
  for ND in ${ND_LIST:-128}; do
    arm "$ND" ch32_diag  NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=$CH
    arm "$ND" ch32_diag2 NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=$CH  # reproducible?
  done
else
  for ND in ${ND_LIST:-32 128}; do
    arm "$ND" ch8     NCCL_MIN_NCHANNELS=8  NCCL_MAX_NCHANNELS=8  NCCL_P2P_NET_CHUNKSIZE=$CH   # old ladder pin
    arm "$ND" ch32    NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32 NCCL_P2P_NET_CHUNKSIZE=$CH   # new ladder pin
    arm "$ND" ch8_rep NCCL_MIN_NCHANNELS=8  NCCL_MAX_NCHANNELS=8  NCCL_P2P_NET_CHUNKSIZE=$CH   # noise floor
  done
fi
echo "=== DONE rc=$rc outdir=$OUTDIR ==="
exit "$rc"
