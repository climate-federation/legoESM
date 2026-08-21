#!/bin/bash -l
# DOES THE CHANNEL COUNT REOPEN THE INTERIOR/RIM OVERLAP LEVER?
#
# On 2026-08-13 this campaign measured that XLA can hide ~79% of ONE
# collective behind independent compute, and ~0-20% of the same total payload
# split into 3, 6 or 12 collectives. The conclusion drawn was that many
# distinct-permutation rounds SERIALISE at runtime on the shared NCCL
# communicator, that no compiler flag opens the windows, and that the
# interior/rim overlap pattern used by MPAS-Fortran, ICON and FV3 is therefore
# not the lever for this model. A multi-week build was cancelled on it.
#
# That measurement ran at NCCL's default channel count. On 2026-08-21 the
# lat-lon lane's channel sweep found the channel count is the dominant free
# knob on communication (-40% on the communication term at 64 GPUs, with 8
# channels the WORST of {4,8,16,32}). Channels ARE the resource that many
# concurrent collectives contend for. So the 2026-08-13 conclusion may have
# been a channel-starvation artifact rather than a property of the stack, and
# the cancelled build may be worth reopening.
#
# ONE VARIABLE: NCCL_MIN_NCHANNELS = NCCL_MAX_NCHANNELS in {4,8,16,32}. Every
# other flag, the payload, the work sweep and the device count are fixed. The
# probe's own control (same program, collective replaced by the identity) is
# unchanged and runs inside every arm.
#
# Falsifiable, written before submit:
#   The number is the hidden fraction at n_collectives = 12 relative to the
#   hidden fraction at n_collectives = 1, in the same arm.
#   REOPENS THE LEVER: at 16 or 32 channels the 12-collective hidden fraction
#     rises above 50% while the 1-collective figure stays near 79%. Then the
#     rounds were contending for channels, an interior/rim split has something
#     to overlap with, and the cancelled build is back on the table.
#   CONFIRMS THE 2026-08-13 VERDICT: the 12-collective hidden fraction stays
#     under ~25% at every channel count. Then the serialisation is not about
#     channels, the August conclusion stands unchanged, and no further work
#     goes into overlap on this stack.
#   VOID: the 1-collective row does not reproduce ~79% hidden. The probe is
#     then measuring something other than what it measured in August and no
#     arm of this job is interpretable.
#SBATCH --job-name=ovl_chan
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ovl_chan.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ovl_chan_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"

# The August receipt's scheduler setting, held FIXED in every arm: the probe
# measures a different question without it, and it is not the variable here.
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true --xla_gpu_shard_autotuning=false"

rc=0
run_arm () {  # channels tag
  local CH=$1 TAG=$2
  echo "=== $TAG channels=$CH $(date +%T) ==="
  printf 'NCCL_MIN_NCHANNELS=%s NCCL_MAX_NCHANNELS=%s\n' "$CH" "$CH" > "$OUTDIR/${TAG}.env"
  NCCL_MIN_NCHANNELS=$CH NCCL_MAX_NCHANNELS=$CH JAX_ENABLE_X64=0 \
    timeout 400 srun --export=ALL --nodes=1 --ntasks=1 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/validate/collective_compute_overlap.py \
      --n-devices 4 --payload 2000000 \
      --flops 0 200 800 3200 \
      --n-collectives 1 12 \
      --reps 40 --warmup 10 \
      > "$OUTDIR/${TAG}.txt" 2>&1
  # Piping into tee would hand the shell TEE's exit status, so a python that
  # died would still read as a passing arm. Redirect, check, then show it.
  local ARM_RC=$?
  tail -n 40 "$OUTDIR/${TAG}.txt"
  if [ "$ARM_RC" -ne 0 ]; then echo "ARM $TAG FAILED (rc=$ARM_RC)"; rc=1; fi
  sleep 10
}

# Discarded warm-up arm, then the four channel counts up and back down so a
# drift on the node cancels instead of favouring whichever ran first.
run_arm 8 warmup_discard
for CH in 4 8 16 32; do run_arm "$CH" "c${CH}_r1"; done
for CH in 32 16 8 4; do run_arm "$CH" "c${CH}_r2"; done

echo "=== RECEIPTS ==="
for CH in 4 8 16 32; do
  for R in r1 r2; do
    F="$OUTDIR/c${CH}_${R}.txt"
    if [ ! -s "$F" ]; then echo "MISSING: c${CH}_${R}.txt"; rc=1; fi
  done
done
if [ "$rc" -ne 0 ]; then echo "SWEEP INCOMPLETE -- not a verdict"; exit 1; fi
echo "all arms present in $OUTDIR; read the per-arm tables, the probe prints no verdict"
echo DONE
