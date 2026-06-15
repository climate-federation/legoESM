#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=cmip6_5g
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cmip6_5g_%j.out
# 5deg fully-coupled, FULL atm physics, GRAY radiation (fast compile) to confirm
# the full-physics coupled stack RUNS + is stable + produces CMOR.  rrtmgp's
# giant-graph compile is a separate problem (see report).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=$(ls -d packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
OUT=results/cmip6_coupled/coarse5deg_gray_fullphys
JAX_ENABLE_X64=1 $PY -u scripts/run/run_coupled.py \
  --preset slab_simple --resolution 18 --nlev 20 --dt 450 \
  --days 90 --diag-days 30 --radiation gray \
  --convection sbm --turbulence holtslag_boville \
  --gravity-wave-drag hines --clouds sundqvist --microphysics kessler \
  --experiment historical --start-year 1850 \
  --cmip-output --cmip-resolution-deg 5.0 --output "$OUT"
echo "run_rc=$?"; date
echo "=== CMOR output ==="; find "$OUT" -name "*.nc" 2>/dev/null | head -20
echo "=== diag ==="; cat "$OUT"/diag_timeseries.csv 2>/dev/null | head
echo "=== DONE ==="; date
