#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --job-name=en1354
# REPOINTED 2026-09-23: config/amip/amip_production.yaml became the CAM6
# suite on that date.  This script exists to reproduce a SPECIFIC recorded
# analysis, so it now names config/amip/amip_sundqvist_l36.yaml -- the same
# deck under its new name, byte-unchanged -- rather than silently running a
# different model.
#
# #1354 / #1515 — column energy-budget closure test.
#
# HYPOTHESIS (GLM triage): the day-130+ thermal runaway is a physics
# energy-bookkeeping LEAK, ~20 W/m2 global (0.17 K/day column warming), not a
# flux bias.  A persistent global leak is nonzero FROM DAY 1, so a short run
# reads it -- no need to integrate to the day-165 detonation.
#
# INSTRUMENT (validated before quoting): EnergyBudgetTracker records, per diag
# step, the area-weighted global column MOIST static energy E, toa_net =
# sw_down_toa - sw_up_toa - lw_up_toa, sfc net radiation, and
# residual = toa_net - dE/dt.  For the atmosphere column,
# dE/dt = toa_net - (sfc_net_rad_up + SH + LH) + LEAK, so
#   LEAK = residual - sfc_net_rad_up - SH - LH.
# residual and sfc_net_rad come from timeseries_incremental/energy_chunk_*.npz;
# SH (hfss) and LH (hfls) come from the CMOR output -- both needed to CLOSE it.
# A LEAK ~ 0 => budget closes => the warming is a real flux imbalance (find the
# flux).  A LEAK ~ 20 W/m2 => energy created internally => bookkeeping bug.
#
# EXACT #1354 recipe: config/amip/amip_sundqvist_l36.yaml (MPAS Voronoi L5 / L30 /
# sigma / dt 75, bechtold + morrison + multilayer land, conservative tracer
# clamp default-on, real CMIP6 forcing, ERA5 IC).  Machine paths from
# amip_production.sh.  15 days is enough for the residual; --diag-days 1.
set -euo pipefail

WT=/work/bd1083/b309178/diffESM/prmerge/fix
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
OUT="${OUT:-${WT}/results/dycore_1354/energy_15d}"
export AMIP_SURFDATA=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/data/legoesm_surfdata_c260716.nc

cd "${WT}"
PYTHONPATH="$(ls -d "${WT}"/packages/*/ | paste -sd:)":"${WT}/src":"${WT}"
export PYTHONPATH
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

# machine forcing/IC paths (defines AMIP_PATH_FLAGS)
source config/amip/amip_production.sh

echo "sha=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo -dirty)  host=$(hostname)  $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"${PY}" -c "import jax; print('jax', jax.__version__, jax.devices())"

echo "=== run_amip 15 d, energy tracker on, $(date -u) ==="
"${PY}" -u scripts/run/run_amip.py \
    --config config/amip/amip_sundqvist_l36.yaml \
    "${AMIP_PATH_FLAGS[@]}" \
    --days 12 --diag-days 1 --output "${OUT}"

echo "=== energy_chunk files ==="
ls -la "${OUT}/timeseries_incremental/"energy_chunk_*.npz 2>/dev/null || echo "  (none — check the diag path ran)"
echo "DONE $(date -u)"
