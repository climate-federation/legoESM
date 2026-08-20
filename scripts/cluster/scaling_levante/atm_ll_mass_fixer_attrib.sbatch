#!/bin/bash -l
# What is the 45% that is NOT communication?
#
# This lane's "local" work is measured as the step with every halo exchange
# deleted. At 32 devices that arm sits 0.2% above a perfect share of the
# single-device step, at 64 it is 7% above, and at 128 it is 45% above -- 0.71
# ms, nearly half the size of the communication term, and never looked inside.
#
# The no-communication arm is not free of communication. It replaces the HALO
# exchanges and nothing else, and the step also runs a global mass fixer: an
# area-weighted sum of surface pressure over the whole globe, which under
# lat-band sharding is an ALL-REDUCE over every device, on every step. An
# all-reduce's cost grows with the number of participants while a halo's does
# not. It is present in both arms, so it cancels out of the subtraction and
# lands in what gets reported as local work. Both design reviews independently
# said to look here.
#
# Falsifiable, written before submit. The comparison is the no-communication
# arm with the mass fixer on against the same arm with it off:
#   the gap grows with device count and removes most of the 0.71 ms at 128
#     -> the "local" excess is a global reduction, not slow kernels, and the
#        fix is to batch or drop it rather than to touch any kernel.
#   the gap is small and flat -> the excess is genuine local inefficiency at 16
#        latitude rows per device, and the next step is a kernel profile.
#
# Second, independent question in the same allocation: a channel-count sweep
# found 16 collective channels 12.7% faster than the 8 this campaign has always
# used, at 64 devices. That is a free change if it holds, so it is checked at
# 32 and 128 too before anyone adopts it.
#
# A run with the mass fixer off does NOT conserve mass. These arms are
# measurement only and must never be quoted as a model result.
#SBATCH --job-name=ll_massfix
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_massfix.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_XLA_OVERLAP=1
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ll_massfix_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"

run_arm () {   # ndev nocomm extra_args tag   (NCCL_* may be set by caller)
  local NDEV=$1 NC=$2 EXTRA=$3 TAG=$4
  local NODES=$(( NDEV / 4 ))
  echo "--- $TAG ndev=$NDEV nocomm=$NC extra='$EXTRA' chan=${NCCL_MAX_NCHANNELS:-8} $(date +%T)"
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=$NC JAX_ENABLE_X64=0 \
  NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8} \
  NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8} \
  NCCL_P2P_NET_CHUNKSIZE=131072 \
    timeout 1200 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 60 --warmup 10 $EXTRA \
      --out "$OUTDIR/${TAG}.jsonl" || echo "ARM $TAG FAILED"
  sleep 5
}

run_arm 32 0 "" warmup_discard
for REP in 1 2; do
  for NDEV in 32 64 128; do
    run_arm "$NDEV" 0 ""              "full_n${NDEV}_r${REP}"
    run_arm "$NDEV" 1 ""              "nocomm_n${NDEV}_r${REP}"
    run_arm "$NDEV" 1 "--no-fix-mass" "nocomm_nofix_n${NDEV}_r${REP}"
    run_arm "$NDEV" 0 "--no-fix-mass" "full_nofix_n${NDEV}_r${REP}"
    NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16 \
      run_arm "$NDEV" 0 ""            "full_chan16_n${NDEV}_r${REP}"
  done
done

"$PY" - "$OUTDIR" <<'PYSUM'
import json, glob, os, sys, collections, statistics
outdir = sys.argv[1]
SINGLE = 201.48   # measured single-device step, job 27051113, same build/flags
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
med = {k: statistics.median(v) for k, v in vals.items()}
print(f"{'arm':<26} {'ms':>8} {'vs ideal':>9}  readings")
for k in sorted(med, key=lambda k: (int(k.split('_n')[-1]), k)):
    n = int(k.split("_n")[-1])
    ideal = SINGLE / n
    print(f"{k:<26} {med[k]:8.4f} {100*(med[k]-ideal)/ideal:+8.1f}%  "
          f"{['%.4f' % x for x in vals[k]]}")
print()
for n in (32, 64, 128):
    a, b = med.get(f"nocomm_n{n}"), med.get(f"nocomm_nofix_n{n}")
    if a and b:
        print(f"{n:>4} devices: mass fixer costs {a-b:.4f} ms of the "
              f"'local' term ({100*(a-b)/a:.1f}% of it)")
    c, d = med.get(f"full_n{n}"), med.get(f"full_chan16_n{n}")
    if c and d:
        print(f"{n:>4} devices: 16 channels vs 8 = {100*(d-c)/c:+.1f}%")
PYSUM
echo DONE
