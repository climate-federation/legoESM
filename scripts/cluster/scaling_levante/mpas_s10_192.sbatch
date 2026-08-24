#!/bin/bash -l
#SBATCH --job-name=mpas_s10_192
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=48
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=mpas_s10_192.%j.log
# HUNDREDS-OF-GPUS MPAS, rung 2: subdiv-10 at 192 GPUs = 54.6k cells/GPU
# synthetic, prewarmed job 26628074) at 128 GPUs = 81.9k cells/GPU —
# (above floor; own reorder-for 192 padding policy — strong contrast vs
# rung 1 carries a ~1e-5 padding difference, noted).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
# 192-rank fix (2026-08-16, jobs 26979367/68): multi-shard autotune-cache
# sync crashed ("No configuration found in cache after synchronizing
# results across all shards") at 192 ranks and plausibly hung the atm arm.
# Flag verified present in installed jaxlib (libjax_common.so).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"

cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s10_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
JAX_ENABLE_X64=0 srun --ntasks=192 --ntasks-per-node=4 \
    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 192 \
    --subdivision 10 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
    --partition-method sfc --reorder-for 192 \
    --out "$OUTDIR/np192.jsonl" || { echo "np192 FAILED"; rc=1; }
"$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$OUTDIR/np192.jsonl').readline())
    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
except Exception as e:
    print('np192: MISSING/INVALID ->', e); sys.exit(1)
print(f'np192: {ms:8.2f} ms  {d.get(\"mcells_per_s\",0)/1000:.2f} GC/s')" \
  || { echo "np192 receipt invalid"; rc=1; }
echo "DONE rc=$rc"; exit $rc
