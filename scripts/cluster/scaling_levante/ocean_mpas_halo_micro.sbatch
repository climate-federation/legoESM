#!/bin/bash -l
#SBATCH --job-name=oc_halomicro
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=oc_halomicro.%j.log
# Per-EPOCH halo cost at the two rank counts of the ocean-MPAS
# rank-count-term pair (s7@32 vs s8@128, 5120 cells/rank, geometric).
# Attribution state: scan-comm 30-41% (replicated), fabric REFUTED,
# fixer reductions REFUTED (D_off 54.0 > D_on 44.5, job 26820126).
# Question: does ONE halo epoch cost grow 32->128 at matched tile, and
# by how much? Uses the EXISTING instrument (bench_halo_exchange.py,
# voronoi mode) — not a new probe.
# Falsifiability BEFORE submit: per-epoch delta x 35 epochs/step
#   ~ D (=44 ms) CONFIRMS halo-epoch growth as the main term (then the
#   scan-share arithmetic must be re-read against per-epoch payload
#   differences: scan epochs are 2-D, non-scan are 3-D);
#   per-epoch delta x 35 << D REFUTES halo epochs; remaining candidate
#   = compute asymmetry from partition halo fraction (then measure
#   local (owned+halo) array sizes per arm).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/oc_halomicro_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
echo "--- voronoi halo micro s7 np32 (packed 1 node) ---"
JAX_ENABLE_X64=1 srun --nodes=1 --ntasks=32 --ntasks-per-node=32 \
    --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_halo_exchange.py --grid voronoi --level 7 \
    --nlev 10 --output "$OUTDIR/s7np32.json" < /dev/null \
  || { echo "s7np32 FAILED"; rc=1; }
echo "--- voronoi halo micro s8 np128 (32rpn x 4 nodes) ---"
JAX_ENABLE_X64=1 srun --nodes=4 --ntasks=128 --ntasks-per-node=32 \
    --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_halo_exchange.py --grid voronoi --level 8 \
    --nlev 10 --output "$OUTDIR/s8np128.json" < /dev/null \
  || { echo "s8np128 FAILED"; rc=1; }
echo "=== files ==="; ls -la "$OUTDIR"
echo "DONE rc=$rc"; exit $rc
