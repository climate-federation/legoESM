#!/bin/bash -l
#SBATCH --job-name=mpas_s10_128
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=mpas_s10_128.%j.log
# HUNDREDS-OF-GPUS MPAS, rung 1: subdiv-10 (10.49M cells, lloyd=0
# synthetic, prewarmed job 26628074) at 128 GPUs = 81.9k cells/GPU —
# the SAME near-matched tile as s8@np8 (6.58 ms) and s9@np32 (12.47 ms),
# extending the 4x-devices weak series a third rung.
# Falsifiability, written BEFORE submit:
#   COMPOUNDING scale-out law (each 4x costs 1.5-1.9x): 18.7-23.6 ms
#   SATURATING (cost flattens past 32 devices):          ~12.5 ms
# Either outcome discriminates; protocol byte-matched to the s9 ladder
# (f32, sfc, --reorder-for 128, steps 12 / warmup 3).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s10_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 128 \
    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
    --partition-method sfc --reorder-for 128 \
    --out "$OUTDIR/np128.jsonl" || { echo "np128 FAILED"; rc=1; }
"$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$OUTDIR/np128.jsonl').readline())
    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
except Exception as e:
    print('np128: MISSING/INVALID ->', e); sys.exit(1)
print(f'np128: {ms:8.2f} ms  {d.get(\"mcells_per_s\",0)/1000:.2f} GC/s')" \
  || { echo "np128 receipt invalid"; rc=1; }
echo "DONE rc=$rc"; exit $rc
