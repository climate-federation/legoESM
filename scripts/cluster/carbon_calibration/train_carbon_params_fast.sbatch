#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_fast_calib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/carbon_calibration/train_fast_%j.out
# =============================================================================
# Stage-B (B2-fast) DifferLand SOM carbon calibration, CONVERGING variant.
# The forward is the fast CLOSED-FORM SOM cascade equilibrium
# (legoesm.land.carbon.fast_analytic.analytic_som_soc): a SINGLE no-grad
# precompute spin-up (n_spinup/n_verify/dt) records the SOM-parameter-independent
# inputs, then MANY cheap MUON steps differentiate the analytic cascade vs the
# observed per-archetype SOC.  This lets --steps run to hundreds (the
# grad-through-spin-up B2 was ~tens of minutes PER step).
#
# Override any knob via env, e.g.:
#   STEPS=300 N_SPINUP=60 MAX_ARCH=60 \
#     sbatch scripts/cluster/carbon_calibration/train_carbon_params_fast.sbatch
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
export PYTHONUNBUFFERED=1        # stream [target]/[fast-analytic]/[preflight]/[train]

CLM5="${CLM5:-/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
CLIM="${CLIM:-results/land_forcing/era5_land_forcing_clim_2deg.nc}"
: "${STEPS:=300}"                # MANY steps now feasible (cheap closed form)
# ONE-TIME precompute spin-up geometry == the geometry the analytic-vs-spin-up
# MATCH test validated (n_spinup=40, dt=7200, n_layers=6 -> mean|rel| ~7.5%): keeps
# the precompute a few minutes (many soil-texture batches each compile a coupled
# land graph) so the run completes promptly.
: "${N_SPINUP:=40}"
: "${N_VERIFY:=4}"
: "${DT:=7200}"                  # 2-h precompute timestep (still resolves the modifier)
: "${N_LAYERS:=6}"
: "${SOIL_DEPTH:=3.0}"
: "${MAX_ARCH:=48}"             # representative archetype subset (tuned params global)
: "${LR:=0.03}"
: "${WARMUP:=10}"
# CLM5 surfdata ORGANIC is organic MATTER density [kg/m3]; convert to CARBON
# (van Bemmelen 0.58 gC/gOM) so the observed target is carbon-consistent.
: "${OM_TO_OC:=0.58}"
: "${OUTDIR:=results/carbon_calibration}"

echo "=== node $(hostname) ==="
$PY -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())" 2>&1 | tail -2

echo "=== fast carbon calibration: steps=$STEPS precompute(n_spinup=$N_SPINUP dt=$DT) max_arch=$MAX_ARCH ==="
$PY scripts/run/train_carbon_params.py \
    --fast-analytic \
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
    --lr "$LR" \
    --warmup-steps "$WARMUP" \
    --optimizer muon \
    --output "$OUTDIR"
RC=$?
echo "=== train_carbon_params (fast) exit=$RC ==="
if [ "$RC" -eq 0 ]; then
    echo "--- tuned_carbon_parameters.json ---"; cat "$OUTDIR/tuned_carbon_parameters.json"
    echo "--- scorecard.json ---"; cat "$OUTDIR/scorecard.json"
fi
