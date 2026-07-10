#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=carbon_exact
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=32:00:00
#SBATCH --output=results/carbon_calibration_exact/train_%j.out
# =============================================================================
# The EXACT coupled-model DifferLand SOM carbon calibration, run at useful scale
# by the chunked EXACT-gradient accumulation: each optimizer step is n_chunks
# SEQUENTIAL coupled-spin-up VJPs of <= GRAD_CHUNK archetypes (past the CPU
# batched-VJP crash at >=8), summed and normalised ONCE by the global cover-weight
# sum -> exactly the unchunked cover-weighted-MSE gradient.  CPU, x64, burst.
#
#   GRAD_CHUNK=<maxsafe> sbatch scripts/cluster/carbon_calibration/train_carbon_exact_chunked.sbatch
# =============================================================================
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/lc_rebase
cd "$REPO" || { echo "cannot cd $REPO"; exit 2; }
mkdir -p results/carbon_calibration_exact

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO:$(ls -d "$REPO"/packages/*/ | sed 's:/$::' | tr '\n' ':')${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
# Persistent JAX compilation cache on the SHARED filesystem (under .cache/, gitignored):
# train_carbon_params.py --compilation-cache-dir defaults to $JAX_COMPILATION_CACHE_DIR, so
# the ~20-min COLD compile of the per-archetype coupled-land-model graphs is written ONCE
# and reused across launches (a re-run with the same code/backend drops to seconds).
export JAX_COMPILATION_CACHE_DIR="$REPO/.cache/jax_compile_cache_carbon"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export PYTHONUNBUFFERED=1        # stream [target]/[preflight]/[train] progress

CLM5="${CLM5:-/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
CLIM="${CLIM:-results/land_forcing/era5_land_forcing_clim_2deg.nc}"
: "${GRAD_CHUNK:=4}"            # override with the Step-0 max-safe chunk size
: "${OUTDIR:=results/carbon_calibration_exact}"

echo "=== node $(hostname) | python $($PY --version 2>&1) ==="
$PY -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())" 2>&1 | tail -1
echo "=== EXACT chunked carbon calibration: grad_chunk=$GRAD_CHUNK ==="

$PY scripts/run/train_carbon_params.py \
    --slow-spinup-grad \
    --grad-chunk "$GRAD_CHUNK" \
    --rebuild \
    --surf-path "$CLM5" \
    --climatology "$CLIM" \
    --steps 12 \
    --max-archetypes 16 \
    --n-spinup 25 \
    --dt 7200 \
    --n-layers 8 \
    --patience 25 \
    --warmup-steps 3 \
    --om-to-oc 0.58 \
    --output "$OUTDIR"
RC=$?
echo "=== train_carbon_params exit=$RC ==="
if [ "$RC" -eq 0 ]; then
    echo "--- tuned_carbon_parameters.json ---"; cat "$OUTDIR/tuned_carbon_parameters.json"
    echo "--- scorecard.json ---"; cat "$OUTDIR/scorecard.json"
fi
