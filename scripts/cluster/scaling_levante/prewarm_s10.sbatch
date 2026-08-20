#!/bin/bash -l
#SBATCH --job-name=prewarm_s10
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=prewarm_s10.%j.log
# Prewarm subdiv-10 lloyd=0 (10.5M cells, ~7 GB npz) into the shared
# mesh cache — unlocks MPAS at 128-224 GPUs ABOVE the ~30k tile floor
# (np128 = 81.9k, np224 = 46.8k cells/GPU).  s9 (2.62M) built in 67 min;
# s10 estimated ~4-5 h.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
"$PY" scripts/data/prewarm_voronoi_mesh.py --level 10 --lloyd 0
