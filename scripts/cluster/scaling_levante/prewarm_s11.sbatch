#!/bin/bash -l
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --mem=230G   # the node ceiling here: 256 cores x 940 MB. s10 asked 120G and completed in 3h23; this mesh is four times larger, so this is the most the partition can give it
#SBATCH --time=48:00:00
#SBATCH --output=prewarm_s11_%j.log

# Build the subdivision-11 icosahedral mesh: 41,943,042 cells, about 3.7 km,
# near 1/30 of a degree, and four times finer than the production mesh.
#
# It is the mesh the "four times finer" rung of the ocean scaling figure needs
# and the only one of the two finer rungs that is plausibly within reach; the
# next one after it is 168 million cells.
#
# Cost, scaled from what the smaller meshes actually took: subdivision 9, at
# 2.6 million cells, built in 67 minutes; subdivision 10, at 10.5 million,
# was estimated at four to five hours and produced files of seven and eleven
# gigabytes. This one is four times larger again, so budget most of a day and
# roughly forty gigabytes per precision. There is about two terabytes free.
#
# It writes into the shared cache, so it is a one-time cost that every later
# arm at this resolution reuses. Nothing else in the campaign depends on it,
# so a failure here costs only this job.
#
# It runs on the shared partition rather than the compute one because the
# build is a single long-running process and compute caps jobs at eight
# hours, which is less than this needs. Shared allows seven days and gives a
# slice of a node, which is the right shape for one process that wants a lot
# of memory and not many cores.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
ENV_SH="${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
[[ -f "$ENV_SH" ]] || ENV_SH="${SUBMIT_DIR}/_env.sh"
[[ -f "$ENV_SH" ]] || { echo "cannot find _env.sh from $SUBMIT_DIR" >&2; exit 2; }
source "$ENV_SH"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
df -h /work | tail -1
"$PY" scripts/data/prewarm_voronoi_mesh.py --level 11 --lloyd 0
echo "=== cache now ==="; du -sh "$LEGOESM_MESH_CACHE_DIR"
ls -la "$LEGOESM_MESH_CACHE_DIR"/scvt_v1_lvl11_* 2>/dev/null || echo "no level-11 file was written"
