#!/bin/bash -l
#SBATCH --job-name=ll_bound
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=ll_bound.%j.log
# Single-device SAME-PER-DEVICE-SIZE baselines to complete the calibrated
# T_bound for the LL2048 rows (roofline recipe: the nd=1 time must be at
# the PER-DEVICE tile, not the global grid — passing the global one gives
# measured/bound < 1, the known tell).  Tiles: 16x4096 (the @128 band),
# 32x4096 (the @64 band).  APPROXIMATION, recorded: an nd=1 run includes
# the pole tiles, so its operator mix slightly OVERCOUNTS an interior
# band's compute -> the bound is conservative (biased high).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_bound_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
run_arm () { # nlat x64 tag
  echo "=== nd=1 LL$1x4096 x64=$2 ($3) ==="
  JAX_ENABLE_X64=$2 srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --n-devices 1 --mode strong --n-lat "$1" --n-lon 4096 --nlev 26 \
      --steps 12 --warmup 3 \
      --out "$OUTDIR/$3.jsonl" || { echo "$3 FAILED"; rc=1; }
}
run_arm 16 0 tile16_f32
run_arm 32 0 tile32_f32
run_arm 16 1 tile16_f64
echo "=== RESULTS ==="
for T in tile16_f32 tile32_f32 tile16_f64; do
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
