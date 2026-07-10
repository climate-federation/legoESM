#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_calibrate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --output=results/carbon_calibration/train_%j.out
# =============================================================================
# Stage-B (B2) DifferLand SOM carbon calibration on a GPU compute node.
# Gradient-tunes the 8 SOM CarbonConfig parameters against observed per-archetype
# SOC (CLM5 surfdata ORGANIC column) via the B1 differentiable forward map + MUON.
#
# Tractable TRAINING scale (documented approximation; the tuned params apply at
# production dt globally): a representative archetype subset + a short spin-up.
# Override any knob via env, e.g.:
#   STEPS=30 N_SPINUP=12 DT=7200 MAX_ARCH=20 \
#     sbatch scripts/cluster/carbon_calibration/train_carbon_params.sbatch
# =============================================================================
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/lc_rebase
cd "$REPO" || { echo "cannot cd $REPO"; exit 2; }
mkdir -p results/carbon_calibration

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO:$(ls -d "$REPO"/packages/*/ | sed 's:/$::' | tr '\n' ':')${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
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
: "${STEPS:=25}"
: "${N_SPINUP:=12}"
: "${N_VERIFY:=3}"
: "${DT:=7200}"
: "${MAX_ARCH:=16}"
: "${LR:=0.03}"
: "${WARMUP:=2}"                 # short warmup so a short run isn't wasted ramping LR
# CLM5 surfdata ORGANIC is organic MATTER density [kg/m3] with carbon fraction
# 0.58 gC/gOM (van Bemmelen 1/1.724; per the file's own units attribute) -> convert
# to organic CARBON so the observed target is carbon-consistent with som_total.
: "${OM_TO_OC:=0.58}"
: "${OUTDIR:=results/carbon_calibration}"

echo "=== node $(hostname) ==="
$PY -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())" 2>&1 | tail -2

echo "=== carbon calibration: steps=$STEPS n_spinup=$N_SPINUP dt=$DT max_arch=$MAX_ARCH ==="
$PY scripts/run/train_carbon_params.py \
    --rebuild \
    --surf-path "$CLM5" \
    --climatology "$CLIM" \
    --steps "$STEPS" \
    --n-spinup "$N_SPINUP" \
    --n-verify "$N_VERIFY" \
    --dt "$DT" \
    --max-archetypes "$MAX_ARCH" \
    --om-to-oc "$OM_TO_OC" \
    --lr "$LR" \
    --warmup-steps "$WARMUP" \
    --optimizer muon \
    --output "$OUTDIR"
RC=$?
echo "=== train_carbon_params exit=$RC ==="
if [ "$RC" -eq 0 ]; then
    echo "--- tuned_carbon_parameters.json ---"; cat "$OUTDIR/tuned_carbon_parameters.json"
    echo "--- scorecard.json ---"; cat "$OUTDIR/scorecard.json"
fi
