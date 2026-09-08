#!/bin/bash -l
# Everything the compiler will do for us, before we write anything ourselves.
#
# Both design reviews said the same thing about this lane's communication: the
# exchanges serialise at runtime on a single collective-library communicator,
# and the fix is to give them separate streams or communicators. That is a
# code change if we build it and a FLAG if the compiler already does it. This
# job tries the flags first, because they cost nothing to try and a win here
# retires the code change entirely.
#
# One flag per arm, against a shared baseline, interleaved and repeated, all in
# one allocation on the same nodes. Every flag below was checked to be accepted
# by this exact jaxlib build rather than assumed from a version's release notes;
# three plausible-looking ones (per-stream comms, pipelined collectives,
# persistent cliques) are NOT settable here and are therefore absent.
#
# Falsifiable, written before submit. Baseline at 64 devices is about 4.92 ms
# with 1.55 ms of communication:
#   any arm at or under 4.60 ms  -> that flag is worth adopting and is measured
#     again at 128 before it goes into the production environment.
#   every arm within 2% of baseline -> the compiler will not do this for us and
#     the explicit neighbour-transport build is the only remaining route.
#SBATCH --job-name=ll_flags
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_flags.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_XLA_OVERLAP=1
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ll_flags_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
NDEV=64; NODES=16
BASE_FLAGS="--xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"

# NCCL's own tuning is as free as the compiler's and was missing from the first
# version of this sweep: the collective library picks an algorithm and a
# protocol per message size and communicator, and those choices are a plausible
# home for a term that changes between 32 and 64 devices while the payload does
# not. Passed as environment rather than XLA flags, hence the separate hook.
run_arm () {   # tag  extra_flags   (NCCL_EXTRA env, if any, set by caller)
  local TAG=$1; shift
  local EXTRA="$*"
  echo "--- $TAG  flags: $EXTRA  nccl: ${NCCL_ALGO:-default}/${NCCL_PROTO:-default}  $(date +%T)"
  XLA_FLAGS="$BASE_FLAGS $EXTRA" \
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  JAX_ENABLE_X64=0 \
    timeout 1200 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 60 --warmup 10 \
      --out "$OUTDIR/${TAG}.jsonl" || echo "ARM $TAG FAILED"
  sleep 5
}

# The baseline is measured FIRST and LAST in each repetition, so a drift on the
# node shows up as a disagreement between its two readings instead of being
# silently attributed to whichever flag ran late.
run_arm warmup_discard
for REP in 1 2; do
  run_arm "base_first_r${REP}"
  run_arm "multistream_r${REP}"   --xla_gpu_experimental_enable_collective_multi_streaming=true
  run_arm "commsplit_r${REP}"     --xla_gpu_enable_nccl_comm_splitting=true
  run_arm "pipelinedp2p_r${REP}"  --xla_gpu_enable_pipelined_p2p=true
  run_arm "decompose_r${REP}"     --xla_gpu_collective_permute_decomposer_threshold=0
  run_arm "userbuf_r${REP}"       --xla_gpu_enable_nccl_user_buffers=true
  run_arm "symbuf_r${REP}"        --xla_gpu_experimental_enable_nccl_symmetric_buffers=true
  run_arm "hiprio_r${REP}"        --xla_gpu_enable_highest_priority_async_stream=true
  NCCL_PROTO=Simple run_arm "nccl_simple_r${REP}"
  NCCL_PROTO=LL128 run_arm "nccl_ll128_r${REP}"
  NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16 run_arm "nccl_chan16_r${REP}"
  run_arm "base_last_r${REP}"
done

"$PY" - "$OUTDIR" <<'PYSUM'
import json, glob, os, sys, collections, statistics
outdir = sys.argv[1]
vals = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(outdir, "*.jsonl"))):
    tag = os.path.basename(f)[:-6]
    if tag.startswith("warmup"):
        continue
    arm = tag.rsplit("_r", 1)[0]
    try:
        vals[arm].append(json.loads(open(f).read().splitlines()[-1])["steady_median_ms"])
    except (OSError, KeyError, IndexError):
        print(f"  {tag}: unreadable receipt")
base = vals.get("base_first", []) + vals.get("base_last", [])
if not base:
    print("NO BASELINE"); sys.exit(1)
b = statistics.median(base)
drift = (max(base) - min(base)) / b
print(f"baseline {b:.4f} ms from {len(base)} readings, "
      f"spread {100*drift:.1f}% (this is the noise floor)")
for arm, v in sorted(vals.items(), key=lambda kv: statistics.median(kv[1])):
    m = statistics.median(v)
    print(f"  {arm:<22} {m:8.4f} ms  {100*(m-b)/b:+6.1f}%  (n={len(v)}, "
          f"{['%.4f' % x for x in v]})")
PYSUM
echo DONE
