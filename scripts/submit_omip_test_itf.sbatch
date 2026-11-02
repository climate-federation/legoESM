#!/bin/bash
#SBATCH --job-name=omip_phys_1deg
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/slurm-omip_itf-%j.out
#
# Physics-only legoESM OMIP run: 1 deg, JRA55-do RYF 1990-91 (cycled), ETOPO bathymetry,
# WOA13v2 init, no BGC, 5 years, snapshot every 30 days.
# Based on /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/submit_marbl_1deg_test.sbatch
# To continue after 12 h: add  --restart <last restart_dayXXXXXX.npz>  and resubmit.

set -euo pipefail
cd /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy
export JAX_ENABLE_X64=1
export LEGOESM_HEATFLUX=colleague
export LEGOESM_KV=0.0
export LEGOESM_KPP='{"K_bg": 1e-6}'
export LEGOESM_WINDSTRESS=1
export LEGOESM_ITF_STRAIT_DEPTH=700
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding"

D=/work/bb1596/b384036/diffESM/data
WOA=/work/bd0854/DATA/ESMValTool2/OBS/Tier2/WOA
PY=/work/bb1596/b384036/b384036/envs/legoesm_cuda/bin/python

echo "host $(hostname)  start $(date)"
$PY -c "import jax; print('JAX devices:', jax.devices())"

$PY /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/run_omip.py \
  --grid latlon --resolution 180x360 --nlev 35 --H-max 5500 --dt 300 \
  --bathymetry $D/bathy/bathy_1deg_etopo_itf.nc --implicit-vertical-mixing \
  --forcing-mode jra55_do_tropical --jra55-cache $D/jra55_ryf9091_1deg_noleap.zarr --jra55-cycle \
  --woa-init --woa-t $D/woa13v2_thetao_annual_degC.nc \
  --woa-s $WOA/OBS_WOA_clim_2013v2_Omon_so_200001-200012.nc \
  --biogeo none --r-factor-max 0.999 --smoothing-passes 2 --physics full_bulk --no-conservation-fixer \
  --days 1095 --checkpoint-days 30 \
  --output results/omip_test_itf
echo "end $(date)"
