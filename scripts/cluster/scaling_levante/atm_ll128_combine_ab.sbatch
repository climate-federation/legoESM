#!/bin/bash -l
#SBATCH --job-name=ll128_comb
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=ll128_comb.%j.log
# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
# COMBINE independent CPs into fewer, larger messages.
# NOTE: the closed-levers null for these flags was the OCEAN lane
# (reduction-dominated); this is the first atm-latlon test — not a rerun
# of a closed null.
# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
#   A control (default flags)      : expect ~5.6 ms
#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
#   C +combine +pipelined-p2p      : scheduling interaction
#   A2 control repeat  : drift bracket (codex r7)
#   REFUTE if B,C within 2% of A/A2 -> refutes THIS threshold/impl only
#   (no GPU post-pass census per arm); next = scheduling/overlap lever.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
# Hang observability (jobs 26630576/26638830 hung in first-collective
# NCCL bootstrap on the same node block, all IB ports ACTIVE): per-rank
# NCCL debug files name the stalling pair next time. INFO logging costs
# only bootstrap-time lines; steady stepping unaffected (noted in doc).
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="$OUTDIR/nccl.%h.%p.log"
rc=0
run_arm () { # tag extra_xla_flags
  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices 128 --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
}
run_arm A_default ""
run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
run_arm A2_default ""
echo "=== RESULTS ==="
for T in A_default B_combine C_combine_pipelined A2_default; do
  "$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$OUTDIR/$T.jsonl').readline())
    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
except Exception as e:
    print('$T: MISSING/INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.3f} ms')" || { echo "$T receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
