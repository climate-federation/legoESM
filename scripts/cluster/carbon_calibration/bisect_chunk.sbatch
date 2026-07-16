#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_bisect
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/carbon_calibration/bisect_%j.out
# =============================================================================
# Step-0 bisect: find the LARGEST crash-free CPU chunk size for the DifferLand
# coupled-model reverse-mode VJP (the batched VJP segfaults on CPU XLA at >=8
# archetypes).  Runs each chunk size N in {2,4,6,8} as an ISOLATED subprocess so
# a crash in one does not mask the smaller sizes.  JAX_TRACEBACK_FILTERING=off.
# =============================================================================
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/lc_rebase
cd "$REPO" || { echo "cannot cd $REPO"; exit 2; }
mkdir -p results/carbon_calibration

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO:$(ls -d "$REPO"/packages/*/ | sed 's:/$::' | tr '\n' ':')${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export JAX_TRACEBACK_FILTERING=off
export PYTHONUNBUFFERED=1

CLM5="${CLM5:-/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
CLIM="${CLIM:-results/land_forcing/era5_land_forcing_clim_2deg.nc}"

echo "=== node $(hostname) | python $($PY --version 2>&1) ==="
$PY -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())" 2>&1 | tail -1

$PY scripts/tmp/_bisect_carbon_chunk.py \
    --surf-path "$CLM5" \
    --climatology "$CLIM" \
    --om-to-oc 0.58 \
    --workdir results/carbon_calibration
RC=$?
echo "=== bisect exit=$RC ==="
