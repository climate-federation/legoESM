#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=cmip6_5r
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cmip6_5r_%j.out
# 5deg fully-coupled, FULL atm physics, RRTMGP (true CMIP6 GHG radiation),
# rad_update_steps=4 (Issue #316: cut the rrtmgp compiled-segment compile from
# O(hours) to O(minutes)).  Historical-1850 transient GHG via #2.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=$(ls -d packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null; date
OUT=results/cmip6_coupled/coarse5deg_rrtmgp_fullphys
JAX_ENABLE_X64=1 $PY -u scripts/run/run_coupled.py \
  --preset slab_simple --resolution 18 --nlev 20 --dt 450 \
  --days 90 --diag-days 30 --radiation rrtmgp --rad-update-steps 4 \
  --convection sbm --turbulence holtslag_boville \
  --gravity-wave-drag hines --clouds sundqvist --microphysics kessler \
  --experiment historical --start-year 1850 \
  --cmip-output --cmip-resolution-deg 5.0 --output "$OUT"
echo "run_rc=$?"; date
echo "=== CMOR ==="; find "$OUT" -name "*.nc" 2>/dev/null | sed 's#.*/cmor/##' | head -20
echo "=== DONE ==="; date
