#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=cmip6_5deg
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cmip6_5deg_%j.out

# Very coarse (~5 deg) FULLY COUPLED CMIP6 attempt — FULL atmosphere physics:
#   radiation=rrtmgp, convection=sbm, turbulence=holtslag_boville (PBL),
#   gravity_wave_drag=hines (non-orographic), clouds=sundqvist.
#   Coupling: slab ocean + sea ice + SIMPLE SLAB LAND (preset slab_simple).
#   Forcing: historical (1850) transient GHG, CMOR output @5deg.
# stdout flows DIRECTLY to the .out (no `| tail` that froze the e025 log).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=$(ls -d packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null; date

OUT=results/cmip6_coupled/coarse5deg_hist_fullphys
JAX_ENABLE_X64=1 $PY -u scripts/run/run_coupled.py \
  --preset slab_simple \
  --resolution 18 --nlev 20 --dt 450 \
  --days 90 --diag-days 30 \
  --radiation rrtmgp \
  --convection sbm \
  --turbulence holtslag_boville \
  --gravity-wave-drag hines \
  --clouds sundqvist --microphysics kessler \
  --experiment historical --start-year 1850 \
  --cmip-output --cmip-resolution-deg 5.0 \
  --output "$OUT"
echo "run_rc=$?"; date
echo "=== CMOR output ==="; find "$OUT" -name "*.nc" 2>/dev/null | head -30
echo "=== DONE ==="; date
