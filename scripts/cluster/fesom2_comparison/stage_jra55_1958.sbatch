#!/bin/bash
# Stage DKRZ's per-variable JRA55-do v1.4.0 files (the ones FESOM2's
# namelist.forcing points at) into the merged source store, then build the
# legoESM 1-deg noleap OMIP-2 cache from it.
#
# Two steps, one job: the merged store is the only thing the cache builder
# can read, and it is large enough that we do not want it on a login node.
#
#   sbatch scripts/cluster/fesom2_comparison/stage_jra55_1958.sbatch
#
#SBATCH --job-name=jra55_stage
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --account=bb1596
#SBATCH --output=%x_%j.log

set -euo pipefail

# The conservative overlap weights are area ratios that must sum to 1 within
# 1e-6; in float32 their own roundoff sits at that tolerance.  Both steps run
# the regrid, so x64 is exported for the whole job.
export JAX_ENABLE_X64=1
# The venv carries the CUDA plugin for the GPU run; this job is on a CPU
# partition, where JAX would otherwise try cuInit and die with error 303.
export JAX_PLATFORMS=cpu

REPO=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/levante-ocean
PY=$REPO/.venv/bin/python
POOL=/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0
SCRATCH=/scratch/b/b381103/fesom2_comparison
YEAR_START=${YEAR_START:-1958}
YEAR_END=${YEAR_END:-1958}

mkdir -p "$SCRATCH"
cd "$REPO"

SRC_ZARR=$SCRATCH/jra55_do_fesom_${YEAR_START}_${YEAR_END}.zarr

echo "=== step 1/2: merge per-variable NetCDFs -> $SRC_ZARR ==="
$PY scripts/data/stage_jra55_do_fesom_pool.py \
    --pool-dir "$POOL" \
    --years "$YEAR_START" "$YEAR_END" \
    --out "$SRC_ZARR"

echo "=== step 2/2: conservative regrid -> 1 deg noleap cache ==="
$PY scripts/data/prepare_omip_forcing.py \
    --source "$SRC_ZARR" \
    --years "$YEAR_START" "$YEAR_END" \
    --target-resolution-deg 1.0 \
    --cache-dir "$SCRATCH" \
    --cache-filename "jra55_do_fesom_${YEAR_START}_${YEAR_END}_1deg_noleap.zarr" \
    --ref-year "$YEAR_START"

echo "=== done ==="
du -sh "$SRC_ZARR" "$SCRATCH/jra55_do_fesom_${YEAR_START}_${YEAR_END}_1deg_noleap.zarr"
