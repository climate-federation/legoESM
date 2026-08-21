#!/bin/bash -l
# The only free win on this lane, and it only appears at one device count.
#
# A one-flag-at-a-time sweep found sixteen collective channels instead of eight
# worth 11.3% at 64 devices. The same change is worth +0.1% at 32 devices and
# -1.0% at 128. Every compiler-side flag was a no-op or a regression. A win
# that exists at exactly one device count is either a real resonance worth
# understanding or an artifact worth not adopting, and the shape of the curve
# says which.
#
# Two things are swept together at 64 devices, one variable each:
#   * channels 4, 8, 16, 32 -- if the curve is monotone and saturating, the
#     channel count is masking a per-operation overhead, and the durable fix is
#     to coalesce the exchanges rather than to raise a tuning knob. If it is a
#     peak at 16, it is a resonance with this message size and this node count
#     and must be re-measured for any other configuration before use.
#   * the same sweep with the halo exchanges removed -- the channel count
#     accelerates a global reduction as much as a neighbour exchange, and this
#     lane has exactly one all-reduce. If the win survives without the halo it
#     never belonged to the halo.
#
# Falsifiable, written before submit:
#   monotone and saturating -> per-operation overhead, coalesce the exchanges.
#   peak at 16 -> configuration-specific, do not put in the production default
#     without re-measuring at every device count that will be run.
#   present with the halo removed -> it is the reduction, not the exchange.
#SBATCH --job-name=ll_chan
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_chan.%j.log
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
OUTDIR="$SCRATCH/legoesm_scaling/ll_chan${SLURM_JOB_NUM_NODES}n_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"

# Device count comes from the allocation, so the same sweep runs at 32, 64 or
# 128 GPUs with `sbatch --nodes=N` and nothing else edited.
NODES="${SLURM_JOB_NUM_NODES:-16}"; GPN=4; NDEV=$((NODES * GPN))
echo "sweep at ${NDEV} GPUs on ${NODES} nodes"

run_arm () {  # channels nocomm tag
  local CH=$1 NC=$2 TAG=$3
  echo "--- $TAG channels=$CH nocomm=$NC $(date +%T)"
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=$NC JAX_ENABLE_X64=0 \
  NCCL_MIN_NCHANNELS=$CH NCCL_MAX_NCHANNELS=$CH \
  NCCL_P2P_NET_CHUNKSIZE=131072 \
    timeout 1200 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 60 --warmup 10 \
      --out "$OUTDIR/${TAG}.jsonl" || echo "ARM $TAG FAILED"
  sleep 5
}

run_arm 8 0 warmup_discard
# Ascending in one repetition, descending in the other, so a drift on the node
# cancels rather than always favouring the channel counts that ran first.
for CH in 4 8 16 32; do run_arm "$CH" 0 "halo_c${CH}_r1"; done
for CH in 32 16 8 4; do run_arm "$CH" 0 "halo_c${CH}_r2"; done
for CH in 4 8 16 32; do run_arm "$CH" 1 "nohalo_c${CH}_r1"; done
for CH in 32 16 8 4; do run_arm "$CH" 1 "nohalo_c${CH}_r2"; done

"$PY" - "$OUTDIR" <<'PYSUM'
import json, glob, os, sys, collections, statistics
outdir = sys.argv[1]
vals = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(outdir, "*.jsonl"))):
    tag = os.path.basename(f)[:-6]
    if tag.startswith("warmup"):
        continue
    try:
        vals[tag.rsplit("_r", 1)[0]].append(
            json.loads(open(f).read().splitlines()[-1])["steady_median_ms"])
    except (OSError, KeyError, IndexError):
        print(f"  {tag}: unreadable receipt")
med = {k: statistics.median(v) for k, v in vals.items()}
for arm in ("halo", "nohalo"):
    base = med.get(f"{arm}_c8")
    print(f"\n{arm} arm, 8 channels = {base}")
    for ch in (4, 8, 16, 32):
        m = med.get(f"{arm}_c{ch}")
        if m and base:
            print(f"  {ch:>3} channels {m:8.4f} ms  {100*(m-base)/base:+6.1f}%  "
                  f"{['%.4f' % x for x in vals[f'{arm}_c{ch}']]}")
h8, n8 = med.get("halo_c8"), med.get("nohalo_c8")
h16, n16 = med.get("halo_c16"), med.get("nohalo_c16")
if all((h8, n8, h16, n16)):
    print(f"\ncommunication term: {h8-n8:.4f} ms at 8 channels, "
          f"{h16-n16:.4f} at 16  ({100*((h16-n16)-(h8-n8))/(h8-n8):+.1f}%)")
PYSUM
echo DONE
