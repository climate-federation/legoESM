#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_sif_calib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/carbon_calibration_sif/train_sif_%j.out
# =============================================================================
# Stage-B (B2-fast) DifferLand carbon calibration WITH the SIF observation stream.
# Adds --with-sif --sif-obs <gridded SIF nc> to the fast closed-form SOM-SOC
# calibration: a per-archetype single-step simulated-SIF forward (spin-up-free) vs
# the observed gridded satellite SIF, summed with the SOC term (cover-weighted MSE),
# training the SIFConfig fluorescence params (kn0/kn_beta/kn_gamma/max_electron_yield/
# escape_probability/kf/kd/kp) alongside the 7 SOM fields.  The SIF term constrains
# photosynthesis/GPP (the boreal-productivity lever).
#
# PREREQUISITE: the --sif-obs nc, built by build_sif_observations.sbatch.
#
# Override via env, e.g.:
#   SIF_OBS=results/land_forcing/tropomi_sif740_growingseason_2deg.nc STEPS=60 \
#     sbatch scripts/cluster/carbon_calibration/train_carbon_params_sif.sbatch
# =============================================================================
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/lc_rebase
cd "$REPO" || { echo "cannot cd $REPO"; exit 2; }
mkdir -p results/carbon_calibration_sif

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
export PYTHONUNBUFFERED=1

CLM5="${CLM5:-/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
CLIM="${CLIM:-results/land_forcing/era5_land_forcing_clim_2deg.nc}"
SIF_OBS="${SIF_OBS:-results/land_forcing/tropomi_sif740_growingseason_2deg.nc}"
: "${STEPS:=60}"
: "${N_SPINUP:=40}"
: "${N_VERIFY:=4}"
: "${DT:=7200}"
: "${N_LAYERS:=6}"
: "${SOIL_DEPTH:=3.0}"
: "${MAX_ARCH:=40}"
: "${LR:=0.03}"
: "${WARMUP:=10}"
: "${OM_TO_OC:=0.58}"
: "${SIF_WEIGHT:=1.0}"
: "${OUTDIR:=results/carbon_calibration_sif}"

if [ ! -f "$SIF_OBS" ]; then
    echo "SIF_OBS not found: $SIF_OBS -- run build_sif_observations.sbatch first."; exit 3
fi

echo "=== node $(hostname) $(date) ==="
$PY -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())" 2>&1 | tail -2

echo "=== fast carbon+SIF calibration: steps=$STEPS max_arch=$MAX_ARCH sif_obs=$SIF_OBS ==="
$PY scripts/run/train_carbon_params.py \
    --fast-analytic \
    --with-sif \
    --sif-obs "$SIF_OBS" \
    --rebuild \
    --surf-path "$CLM5" \
    --climatology "$CLIM" \
    --steps "$STEPS" \
    --n-spinup "$N_SPINUP" \
    --n-verify "$N_VERIFY" \
    --dt "$DT" \
    --n-layers "$N_LAYERS" \
    --soil-depth "$SOIL_DEPTH" \
    --max-archetypes "$MAX_ARCH" \
    --om-to-oc "$OM_TO_OC" \
    --sif-weight "$SIF_WEIGHT" \
    --lr "$LR" \
    --warmup-steps "$WARMUP" \
    --optimizer muon \
    --output "$OUTDIR"
RC=$?
echo "=== train_carbon_params (fast+SIF) exit=$RC ==="
if [ "$RC" -eq 0 ]; then
    echo "--- tuned_carbon_parameters.json ---"; cat "$OUTDIR/tuned_carbon_parameters.json"
    echo "--- scorecard.json ---"; cat "$OUTDIR/scorecard.json"
fi
