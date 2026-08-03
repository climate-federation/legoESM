#!/bin/bash -l
#SBATCH --job-name=mpas_bound
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=mpas_bound.%j.log
# MPAS same-tile nd=1 compute anchor for the panel's calibrated bound:
# subdiv-6 lloyd0 GLOBAL mesh (40,962 natural cells) ~= the 41k
# cells/GPU tile of the measured rows s8-lloyd0@np16 (6.43 ms, job
# 26628076) and s9@np64 (9.60 ms, job 26600095).  A single-device run
# has zero inter-device traffic, so its fused step time IS the compute
# term at this tile (mesh-family lloyd0 matched; padding differs by
# construction — model-grade, recorded).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_bound_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
JAX_ENABLE_X64=0 srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --n-devices 1 --subdivision 6 --nlev 26 --steps 12 --warmup 3 \
    --lloyd 0 --partition-method sfc \
    --out "$OUTDIR/s6_nd1.jsonl" || { echo "s6_nd1 FAILED"; rc=1; }
"$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$OUTDIR/s6_nd1.jsonl').readline())
    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
except Exception as e:
    print('s6_nd1: MISSING/INVALID ->', e); sys.exit(1)
print(f's6_nd1: {ms:8.3f} ms')" || { echo "s6_nd1 receipt invalid"; rc=1; }
echo "DONE rc=$rc"; exit $rc
