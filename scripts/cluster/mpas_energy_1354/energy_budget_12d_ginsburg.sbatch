#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=en1354gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1354/en1354_%j.out
# REPOINTED 2026-09-23: config/amip/amip_production.yaml became the CAM6
# suite on that date.  This script exists to reproduce a SPECIFIC recorded
# analysis, so it now names config/amip/amip_sundqvist_l36.yaml -- the same
# deck under its new name, byte-unchanged -- rather than silently running a
# different model.
#
# #1354 / #1515 -- column energy-budget closure test, GINSBURG twin of
# energy_budget_15d.sbatch (which pins Levante paths).  Physics/grid/dycore
# come from config/amip/amip_sundqvist_l36.yaml -- NOT forked here.
#
# PRE-REGISTERED READING (unchanged from the Levante card):
#   SIGN CONVENTION, everything INTO the atmospheric column, positive:
#     toa_net is positive DOWN, so it enters as +toa_net;
#     sw_net_sfc/lw_net_sfc are documented "+into surface", so radiation the
#       SURFACE absorbs LEAVES the column: -(sw_net_sfc + lw_net_sfc);
#     hfss/hfls are positive UP into the column: +hfss +hfls.
#   dE/dt = toa_net - (sw_net_sfc + lw_net_sfc) + hfss + hfls + LEAK, and the
#   tracker reports residual = toa_net - dE/dt, so
#     LEAK = (sw_net_sfc + lw_net_sfc) - hfss - hfls - residual.
#   An earlier revision of this card wrote "LEAK = residual - sfc_net_rad_up
#   - SH - LH", which has the radiation term the wrong way round.
#   ALL SEVEN channels come from the run's own timeseries.npz: the MPAS lane
#   short-circuits the unified collector and writes its series at the END of
#   the run, so there are NO incremental energy_chunk_*.npz on this path and
#   their absence is not evidence about whether the tracker fired.
#     LEAK ~ 0 W/m2      => the budget CLOSES => the day-130+ warming is a real
#                           flux imbalance; go find the flux.
#     LEAK ~ 20 W/m2     => energy is created internally => physics bookkeeping
#                           bug (the GLM triage hypothesis for #1354, and very
#                           likely #1515 for free).
#   A persistent global leak is nonzero FROM DAY 1, so 12 days reads it -- there
#   is no need to integrate to the day-165 detonation.  Cheaper offline test:
#   none -- the tracker only fills from a running MPAS diagnostic step.
#
# WHY THIS CARD EXISTS: the #1354 thread records that the EnergyBudgetTracker
# was wired only into the full DiagnosticCollector path while the MPAS lane runs
# collect_lightweight, so an MPAS AMIP run emitted no energy chunks and this
# test had never actually been runnable.  The MPAS lane now runs the tracker per
# diag step (model_driver `_ts` carries energy_toa_net / energy_dE_dt /
# energy_residual / sw_net_sfc / lw_net_sfc / hfss / hfls), so the FIRST thing
# this job proves is that those channels are non-empty.
# RESOLUTION: level 5, NOT the production default level 6.  Two reasons, both
# recorded rather than assumed: (a) level 5 / L30 / dt 75 is the recipe #1354
# itself documents and the one the detonating runs used; (b) level 6 at fp64
# OOMs on a 48 GB card (measured: job 9631889 died asking for a single 26.4 GiB
# allocation).  The production config names `--resolution 5` as its sanctioned
# fast-iteration override.  The question here -- does the column energy budget
# close -- is a per-column bookkeeping question, so it should not depend on how
# many columns there are; if the answer turns out to be resolution-sensitive
# that is itself a finding and this assumption is what to revisit first.
set -uo pipefail

ROOT=/burg-archive/glab/users/pg2328/legoESM
cd "$ROOT" || exit 2
# Resolution in the directory name: the run-manifest guard refuses to mix two
# configurations' provenance in one directory (correctly), and a res6 attempt
# already wrote one here.
OUT="${OUT:-$ROOT/results/dycore_1354/energy_12d_res5_intervalmean}"
mkdir -p "$OUT"

# The two surfdata files are staged in THIS tree, not in the default repo the
# machine-path script points at; both use `: ${VAR:=default}` so presetting
# them here wins.  Everything else (ERA5 IC, ETOPO, observed SST/SIC, CMIP6
# forcing deck) is only staged in the default location and is read from there.
export AMIP_SURFDATA="${AMIP_SURFDATA:-$ROOT/data/legoesm_surfdata_c260716.nc}"
export CLM_SURFDATA="${CLM_SURFDATA:-$ROOT/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
for f in "$AMIP_SURFDATA" "$CLM_SURFDATA"; do
  [ -e "$f" ] || { echo "MISSING INPUT: $f"; exit 3; }
done

# Ginsburg machine paths -> exports PY + AMIP_PATH_FLAGS (IC/ETOPO/SST/SIC/forcing).
source "$ROOT/config/amip/amip_production.ginsburg.sh"

# Run OUR tree's science packages against that GPU interpreter.
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export JAX_COMPILATION_CACHE_DIR="$OUT/jax_cache"; mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "sha=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo -dirty)  host=$(hostname)  $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"

echo "=== run_amip 12 d, MPAS production recipe, --diag-days 1 ==="
JAX_ENABLE_X64=1 "$PY" -u scripts/run/run_amip.py \
    --config config/amip/amip_sundqvist_l36.yaml \
    "${AMIP_PATH_FLAGS[@]}" \
    --resolution 5 \
    --days 12 --diag-days 1 --output "$OUT"
rc=$?

echo "=== the deliverable: column energy LEAK (#1354/#1515) ==="
if [ -f "$OUT/timeseries.npz" ]; then
  JAX_PLATFORMS=cpu "$PY" scripts/validate/mpas_energy_budget_leak.py \
      "$OUT/timeseries.npz" --skip-days 1 || true
else
  echo "  no timeseries.npz -- the run did not reach its writer (rc=$rc)"
fi
echo "DONE rc=$rc $(date -u)"
