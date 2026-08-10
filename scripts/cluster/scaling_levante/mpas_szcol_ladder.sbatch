#!/bin/bash -l
#SBATCH --job-name=szcol_ladder
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --exclude=l50100,l50000
#SBATCH --output=szcol_ladder.%j.log
# ===========================================================================
# POST-LANDING LADDER: MPAS strong scaling with size-aware colouring
# (default ON, landed at ratio 0.803 @ s9/64 — job 26857404). Rows for
# the "plot at each major improvement step" directive:
#   s9  @ 16 / 32 / 64   (strong-scaling curve, new default)
#   s10 @ 128            (the record row, new default)
# Protocol matches the existing ladder rows (f32, sfc, lloyd 0,
# steps 12 / warmup 3, reorder-for = the row's own device count as in
# the s9 ladder history).
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/szcol_ladder_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

rung () {  # subdiv nlev nd nodes tag
  local SD=$1 NL=$2 ND=$3 NN=$4 TAG=$5
  echo "=== s${SD}@${ND} [$TAG] (size-coloring default ON) ==="
  LEGOESM_MPAS_WIDE_HALO=0 LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
    srun --nodes=$NN --ntasks=$ND --ntasks-per-node=4 --gpus-per-node=4 \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices $ND --subdivision $SD --nlev $NL \
      --steps 12 --warmup 3 --lloyd 0 --partition-method sfc \
      --reorder-for $ND \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
}

rung 9 26 16 4 s9_np16
rung 9 26 32 8 s9_np32
rung 9 26 64 16 s9_np64
rung 10 26 128 32 s10_np128

echo "=== ROWS ==="
set +e
for f in "$OUTDIR"/*.jsonl; do
  "$PY" - "$f" <<'EOF'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])][0]
print(f"{sys.argv[1].split('/')[-1]}: {r['steady_median_ms']:.2f} ms "
      f"({r['n_devices']} dev, subdiv {r['subdivision']})")
EOF
done
echo "DONE rc=$rc"; exit $rc
