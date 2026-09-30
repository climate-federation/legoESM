#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ledger1354
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1354/ledger_%j.out
# REPOINTED 2026-09-23: config/amip/amip_production.yaml became the CAM6
# suite on that date.  This script exists to reproduce a SPECIFIC recorded
# analysis, so it now names config/amip/amip_sundqvist_l36.yaml -- the same
# deck under its new name, byte-unchanged -- rather than silently running a
# different model.
#
# #1354 ATTRIBUTION RUN -- which physics process creates the +18.2 W/m2?
#
# MEASURED (job 9647745, interval-mean fluxes, PR #1713): the column MSE
# budget fails to close by +18.2 W/m2, steady from day 6, state consistent
# with dE/dt, no energy fixer on the lane, frozen-precip gap bounded at
# 6.5 W/m2 all-frozen.  The number is real; what is missing is the NAME.
#
# INSTRUMENT: the existing --budget-ledger (process_ledger.py), serial MPAS
# wiring.  Per process x (total-water, dry-enthalpy) column tendencies:
# turbulence / convection / microphysics / radiation / other_physics /
# clips / dynamics.  Writes budget_ledger_columns.npz at the diag cadence.
#
# PRE-REGISTERED READING RULE, corrected by dual review BEFORE data.
# Original prediction (recorded, then demoted): convection row, bechtold
# heating unpaid by moisture.  BOTH reviewers refuted the planned metric
# (energy_row + L_v*water_row is NOT an MSE attribution: vapour->liquid
# conversion adds latent heating at unchanged TOTAL water, so a CONSERVING
# scheme would read as leaky).  Corrected frame -- read the DRY-ENTHALPY rows
# against per-process expectations (GLM's healthy table):
#   turbulence   energy ~ +hfss (~20.5); ~+40 = the Louis surface-flux
#                DOUBLE-COUNT, GLM's top-ranked suspect (size ~ one hfss ~ 18,
#                which matches the leak);
#   radiation    ~ toa_net - sfc_net ~ -97;
#   convection   energy ~ -L_v * (its water row): heating paid by drying.
#                Excess beyond that = unpaid heating (my original suspect);
#   microphysics same pairing; conv+micro+clips summed |MSE| <~ 6.5 (frozen);
#   dynamics     ~ 0 (1-3 ok, GLM).
# CODEX limits, recorded: post-step q_v smoothing + hard saturation drain run
# AFTER the ledger snapshot (absent from rows; wrong SIGN for this leak and
# ~0.1 g/kg at scattered points); other_physics is an explicit closure
# RESIDUAL on this lane, so an unledgered creation lands there UNNAMED -- if
# all four named rows are healthy, the next arm splits other_physics (GWD)
# rather than declaring victory.
#
# 3 days: the leak is steady from day 3 in both 12-day runs, and the rows are
# per-step accumulations, not spin-up statistics.  Same recipe as the energy
# card otherwise (res5 serial GPU -- the ledger REFUSES MPI/partitioned).
set -uo pipefail

ROOT=/burg-archive/glab/users/pg2328/legoESM
cd "$ROOT" || exit 2
OUT="${OUT:-$ROOT/results/dycore_1354/ledger_3d}"
mkdir -p "$OUT"

export AMIP_SURFDATA="${AMIP_SURFDATA:-$ROOT/data/legoesm_surfdata_c260716.nc}"
export CLM_SURFDATA="${CLM_SURFDATA:-$ROOT/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
for f in "$AMIP_SURFDATA" "$CLM_SURFDATA"; do
  [ -e "$f" ] || { echo "MISSING INPUT: $f"; exit 3; }
done
source "$ROOT/config/amip/amip_production.ginsburg.sh"

export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export JAX_COMPILATION_CACHE_DIR="$OUT/jax_cache"; mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "sha=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo -dirty)  host=$(hostname)  $(date -u)"
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"

JAX_ENABLE_X64=1 "$PY" -u scripts/run/run_amip.py \
    --config config/amip/amip_sundqvist_l36.yaml \
    "${AMIP_PATH_FLAGS[@]}" \
    --resolution 5 \
    --budget-ledger \
    --days 3 --diag-days 1 --output "$OUT"
rc=$?

echo "=== ledger artifact ==="
ls -la "$OUT"/budget_ledger*.npz 2>/dev/null || echo "  (none -- ledger did not write)"
echo "DONE rc=$rc $(date -u)"
