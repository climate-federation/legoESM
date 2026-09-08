#!/bin/bash -l
#SBATCH --job-name=mpas_1gpu_base
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=mpas_1gpu_base.%j.log
# Single-GPU compute-track BASELINE (doc c3): the s9@64 owned-compute proxy
# is s6@1 nlev 26 (40,962 cells = the 64-GPU per-device load; historical
# control 2.00 ms). Re-pin on the CURRENT code before the batched-gather
# lever lands; the lever's A/B compares against THIS number, same job
# config, 60-step blocks. Two arms (a palindrome of one config) bound
# drift.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/c1gpu_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
export XLA_FLAGS="--xla_gpu_shard_autotuning=false"

arm () { # tag
  echo "=== s6@1 nlev26 [$1] ==="
  rm -f "$OUTDIR/$1.jsonl"
  JAX_ENABLE_X64=0 timeout 900 srun --ntasks=1 --gpus=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --n-devices 1 --subdivision 6 --nlev 26 \
      --steps 60 --warmup 5 --lloyd 0 --partition-method sfc \
      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
}
arm base_a
arm base_b

echo "=== RESULTS ==="
for T in base_a base_b; do
  "$PY" -c "
import json,math,sys
try:
    rows=[json.loads(l) for l in open('$OUTDIR/$T.jsonl')]
    assert len(rows)==1
    d=rows[0]
    assert d.get('n_devices')==1 and d.get('platform')=='gpu'
    assert d.get('subdivision')==6 and d.get('nlev')==26
    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
except Exception as e:
    print('$T: INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.3f} ms')" || { echo "$T invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
