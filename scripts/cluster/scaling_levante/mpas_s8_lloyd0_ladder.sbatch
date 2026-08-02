#!/bin/bash -l
#SBATCH --job-name=mpas_s8_l0
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=mpas_s8_l0.%j.log
# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
#
# Falsifiability, written BEFORE submit:
#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
#             matched cells/GPU stay well above 1 (prior draft saw
#             1.80/1.35/1.41 on the CONFOUNDED pairs)
#   REFUTE  : ratios collapse toward ~1.0 -> the draft's "term" was the
#             Lloyd-mesh confound, and MPAS-GPU weak scaling is near
#             ideal at matched tile.
# Mesh: prewarmed into LEGOESM_MESH_CACHE_DIR (subdiv-8 is below the
# big-mesh refuse threshold, so a cache miss falls back to in-process
# builds — slower, still correct).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s8_l0_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
for NP in 8 16 32; do
  NODES=$(( NP / 4 ))
  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NP" \
      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
      --partition-method sfc --reorder-for 128 \
      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
done
echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
for NP in 8 16 32; do
  F="$OUTDIR/np${NP}.jsonl"
  "$PY" -c "
import json,sys
try:
    d=json.loads(open('$F').readline())
    print(f'np$NP: {d[\"steady_median_ms\"]:8.2f} ms')
except Exception as e:
    print('np$NP: MISSING/UNPARSEABLE ->', e); sys.exit(1)" \
    || { echo "np$NP receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
