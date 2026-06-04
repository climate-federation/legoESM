#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_ll
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ll_%j.out

# PIVOT TEST: latlon C-grid + NEMO bathymetry + smc03 production config (the
# documented-stable 50+yr setup) with WOA18 IC + CORE-II forcing. Does it
# survive the realistic cold-start where the tripole (adcroft) blew up?
# $1 = resolution (default 180x360), $2 = dt (default 300).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RES="${1:-180x360}"; DT="${2:-300}"
echo "=== HOST ==="; hostname; nvidia-smi --query-gpu=name --format=csv,noheader 2>&1|head -1; date
echo "res=$RES dt=$DT"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py \
  --grid latlon_bathy --latlon-res "$RES" --smoke --dt "$DT" --woa-init \
  --diag-every-days 1 --output results/omip_nemo/legoesm_latlon_woa_smoke 2>&1 | tail -45
echo "ll_rc=${PIPESTATUS[0]}"
echo "=== DONE ==="
