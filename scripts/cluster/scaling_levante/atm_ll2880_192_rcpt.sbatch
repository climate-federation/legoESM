#!/bin/bash -l
#SBATCH --job-name=atm_ll2880_rcpt
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=48
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009
#SBATCH --output=atm_ll2880_rcpt.%j.log
# Paper-grade receipt of the LL2880x5760 @192 point (job 27036060 measured
# 6.74 ms / 63.98 GC/s at 12-step blocks — 12 steps carries ~7.7 % spread on
# identical arms, 60 steps ~0.3 %). PROTOCOL NOTE (codex): --steps is the scan
# BLOCK length (median of two blocks / steps), and --warmup is ignored on the
# fused-scan lane — so this job defines a NEW 60-step-block series and
# SUPERSEDES the 12-step number rather than confirming it; A1/A2 palindrome is
# the internal drift control. A->B bundles fused+overlap deliberately
# (production-env pricing, not per-knob attribution); C prices the mcp2p pair
# as the single receipted unit lever. Packed exchange pinned OFF (banked
# default) in every arm; pricing packed@192 is a separate follow-up.
# Palindrome interleave, one variable per arm pair, 15-min timeout per arm
# (the thin-band NCCL-init deadlock presents as an infinite first call):
#   A: env of the banked point (overlap=0, fused-halo=0) — the receipt.
#   B: production-best env (overlap=1, fused-halo=1) — tests whether the
#      hang-era discriminator flags cost anything at 2880 (15-row bands);
#      at LL2304 overlap=1 hung, but that resolved to the thin-band trigger,
#      so 2880 with overlap on is untested, not unsafe.
#   C: B + the mcp2p NCCL pair (receipted -13..-18 % MPAS@64, -6.1 %
#      lat-lon@32; this receipt prices it at 192 for the adoption decision).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
# DKRZ scratch fallback BEFORE first use — _env.sh's own fallback comes too
# late for the cache line below, and set -u aborts on unbound SCRATCH (codex).
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
# Per-job JIT cache: a poisoned SHARED cache dir is the suspected spreader of
# the multi-shard autotune crash (2026-08-16); every >=64-rank job isolates.
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
# Required >=64 ranks on this jaxlib (autotune-cache sync crash); comm
# splitting off matches every working 192-rank receipt to date.
# NOTE: _env.sh consumes LEGOESM_XLA_OVERLAP at SOURCE time, so overlap is
# switched per-arm below by REPLACING XLA_FLAGS, not via the env knob.
BASE_XLA="--xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"
OVERLAP_XLA="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"
export XLA_FLAGS="$BASE_XLA"

cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_ll2880_rcpt_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
export LEGOESM_HANG_DEBUG=1

run_arm () { # tag overlap fused extra_env...
  local tag="$1" ovl="$2" fus="$3"; shift 3
  local arm_xla="$BASE_XLA"
  [ "$ovl" = 1 ] && arm_xla="$BASE_XLA $OVERLAP_XLA"
  echo "=== arm $tag overlap=$ovl fused=$fus extra='$*' ==="
  rm -f "$OUTDIR/$tag.jsonl"   # stale-receipt guard (codex): parse only THIS arm's write
  env -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS -u NCCL_P2P_NET_CHUNKSIZE \
      XLA_FLAGS="$arm_xla" LEGOESM_LATLON_SPMD_FUSED_HALO="$fus" \
      LEGOESM_LATLON_PACKED_EXCHANGE=0 "$@" \
      JAX_ENABLE_X64=0 \
  timeout 900 srun --export=ALL --ntasks=192 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices 192 --mode strong \
      --n-lat 2880 --n-lon 5760 --nlev 26 \
      --steps 60 --warmup 10 \
      --out "$OUTDIR/$tag.jsonl" || { echo "arm $tag FAILED/TIMEOUT"; rc=1; }
}

MCP2P=(NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072)
# palindrome: A B C C B A — each env measured once early, once late.
run_arm A1 0 0
run_arm B1 1 1
run_arm C1 1 1 "${MCP2P[@]}"
run_arm C2 1 1 "${MCP2P[@]}"
run_arm B2 1 1
run_arm A2 0 0

echo "=== RESULTS ==="
for T in A1 B1 C1 C2 B2 A2; do
  F="$OUTDIR/$T.jsonl"
  "$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$F').read().splitlines()[-1])
    ms=d['steady_median_ms']
    assert math.isfinite(ms) and ms > 0
except Exception as e:
    print('$T: MISSING/INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.2f} ms {d.get(\"mcells_per_s\",0)/1000:6.2f} GC/s')" \
    || { echo "$T receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc