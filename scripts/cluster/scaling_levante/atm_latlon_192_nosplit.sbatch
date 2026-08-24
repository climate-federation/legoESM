#!/bin/bash -l
#SBATCH --job-name=atm_ll192_nosplit
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=48
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=atm_ll192_nosplit.%j.log
# HUNDREDS-OF-GPUS lat-lon atmosphere (user directive 2026-08-02).
# LL2048 does not divide 192 ranks (2048/192 non-integer bands), so the
# >128 strong pair moves to LL2304 (96 and 192 both divide 2304) and the
# above-floor "hundreds" point is LL2880x5760 @192 = 86.4k cols/GPU.
#   arm 1: LL2304x4608 @ 96  f32 (110.6k cols/GPU)
#   arm 2: LL2304x4608 @192  f32 (55.3k cols/GPU — still ABOVE the ~30k
#          floor; codex r21 caught the first draft halving these — so
#          this pair tests device-count cost at healthy tiles)
#   arm 3: LL2880x5760 @192  f32 (86.4k cols/GPU — the honest 192-GPU
#          working point; predicted to hold ~existing GC/s levels)
# Context anchors (same bench, steps 12/warmup 3): LL2048@128 f32
# 5.58 ms 39.11 GC/s (job 26534060).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
# Validated combo (jobs 26681636/26681858: -20.3%/-15.0%, parity green):
export LEGOESM_LATLON_SPMD_FUSED_HALO=0  # DISCRIMINATOR 2: overlap=0 also hung (27013729), so the scheduler is refuted; this flips the fused-halo path off, one variable vs that arm.
export LEGOESM_XLA_OVERLAP=0   # DISCRIMINATOR: 27007255 + 26996572 + 26979367 all hung in first compile/exec at 192 with overlap=1; MPAS@192 (no overlap flag) compiles in 9 s. One variable.
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
# 192-rank fix (2026-08-16, jobs 26979367/68): multi-shard autotune-cache
# sync crashed ("No configuration found in cache after synchronizing
# results across all shards") at 192 ranks and plausibly hung the atm arm.
# Flag verified present in installed jaxlib (libjax_common.so).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"  # DISCRIMINATOR 3: NCCL INIT log froze mid-channel-setup for 28 min at 192 ranks (27014822); comm-splitting is the known large-scale init-hang suspect; flag verified in the installed plugin.

cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_ll192_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
export LEGOESM_HANG_DEBUG=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT
export NCCL_DEBUG_FILE=/scratch/b/b381103/legoesm_scaling/nccl_atm192_dbg.%h.%p
run_arm () { # nlat nlon np tag
  echo "=== atm LL$1x$2 @$3 f32 ==="
  JAX_ENABLE_X64=0 srun --ntasks="$3" --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$3" --mode strong \
      --n-lat "$1" --n-lon "$2" --nlev 26 \
      --steps 12 --warmup 3 \
      --out "$OUTDIR/$4.jsonl" || { echo "$4 FAILED"; rc=1; }
}
# @96 receipt banked in job 26628072 (7.872 ms); rerunning only the
# @192 arms, sick-8 nodes excluded at submit (see deadlock section).
run_arm 2304 4608 192 LL2304_f32_np192

echo "=== RESULTS ==="
for T in LL2304_f32_np192 LL2880_f32_np192; do
  F="$OUTDIR/$T.jsonl"
  "$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$F').readline())
    ms=d['steady_median_ms']
    assert math.isfinite(ms) and ms > 0
except Exception as e:
    print('$T: MISSING/INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.2f} ms {d.get(\"mcells_per_s\",0)/1000:6.2f} GC/s')" \
    || { echo "$T receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
