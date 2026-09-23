#!/bin/bash -l
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=prewarm_mesh_f64_%j.log
# Prewarm the f64 (x641) Voronoi mesh cache for s9 + s10 lloyd0 (single process).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu   # mesh build is numpy/CPU; no GPU needed
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_ALLOW_BIG_MESH_BUILD=1
export JAX_ENABLE_X64=1     # build the x641 (f64) variant
cd "$REPO"
for L in 9 10; do
  echo "=== prewarm level $L lloyd 0 (f64/x641) $(date) ==="
  "$PY" scripts/data/prewarm_voronoi_mesh.py --level "$L" --lloyd 0 || echo "prewarm L$L FAILED"
done
echo "=== DONE $(date) ==="
ls -la "$LEGOESM_MESH_CACHE_DIR"/scvt_v1_lvl{9,10}_*lloyd0_x641.npz 2>/dev/null
