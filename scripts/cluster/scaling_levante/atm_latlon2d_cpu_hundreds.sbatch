#!/bin/bash -l
#SBATCH --job-name=cpu_ll2d
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=cpu_ll2d.%j.log
# HUNDREDS-OF-CPUS lat-lon atmosphere (user directive 2026-08-02): the
# 2-D pencil path (wall poles, labelled throughput lane — NOT the pole
# fold; pole-matched A/B remains open) at r512 (512x1024 = 524k cols) so
# np512 still holds ~1k cols/rank.  Self-contained strong ladder
# np64->512, f64 moist, 32 rpn, block:cyclic per the placement receipt.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cpu_ll2d_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
for NP in 64 128 256 512; do
  NODES=$(( NP / 32 )); [ "$NODES" -lt 1 ] && NODES=1
  echo "--- atm latlon 2-D pencil r512 f64 np=$NP ---"
  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
      --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
    "$PY" scripts/bench/run_cpu_mpi_scaling.py \
      --grid latlon --latlon-2d --mode single --physics moist \
      --precision float64 --n-levels 26 --resolution 512 \
      --n-warmup 2 --n-timing 6 \
      --output-dir "$OUTDIR/np${NP}" < /dev/null || { echo "np$NP FAILED"; rc=1; }
done
echo "=== RESULTS ==="
for NP in 64 128 256 512; do
  F=$(find "$OUTDIR/np${NP}" -name "*.json" 2>/dev/null | head -1)
  "$PY" -c "
import json,math,sys
try:
    d=json.load(open('$F' or '/nonexistent'))
    r=d['results'][-1] if isinstance(d.get('results'),list) else d
    ms=r.get('ms_per_step') or r.get('time_per_step_ms')
    assert ms is not None and math.isfinite(ms) and ms > 0
except Exception as e:
    print('np$NP: MISSING/INVALID ->', e); sys.exit(1)
print(f'np$NP: {ms:9.2f} ms')" || { echo "np$NP receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
