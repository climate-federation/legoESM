#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=cmip6_L10
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cmip6_L10_%j.out
# 5deg fully-coupled INTEGRATION CHECK at coarse res: rrtmgp (true CMIP6 GHG
# radiation) + FULL atm physics + coupling, with L10 (half levels) + rad_update_steps=4
# to shrink the rrtmgp coupled-segment compile to a tractable size.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=$(ls -d packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
# real input4MIPs ozone + solar, the same set every AMIP grid runs (2026-10-06)
source config/amip/amip_production.ginsburg.sh
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null; date
OUT=results/cmip6_coupled/coarse5deg_rrtmgp_L10
JAX_ENABLE_X64=1 $PY -u scripts/run/run_coupled.py \
  --preset slab_simple --resolution 18 --nlev 10 --dt 450 \
  --days 90 --diag-days 30 --radiation rrtmgp --rad-update-steps 4 \
  --convection sbm --turbulence holtslag_boville \
  --gravity-wave-drag hines --clouds sundqvist --microphysics kessler \
  --experiment historical --start-year 1850 \
  "${CMIP6_FORCING_FLAGS[@]}" \
  --cmip-output --cmip-resolution-deg 5.0 --output "$OUT"
echo "run_rc=$?"; date
echo "=== CMOR ==="; find "$OUT" -name "*.nc" 2>/dev/null | sed 's#.*/cmor/##' | sort | head -25
echo "=== diag ==="; cat "$OUT"/diag_timeseries.csv 2>/dev/null
echo "=== DONE ==="; date
