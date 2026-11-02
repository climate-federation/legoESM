#!/bin/bash
#SBATCH --job-name=qabg1e-2
#SBATCH --account=bb1596_gpu
#SBATCH --constraint=a100_80
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/slurm-omip_qabg1e-2-%j.out
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
export LEGOESM_KPP='{"K_bg": 1e-6, "A_bg": 1e-2}'
export LEGOESM_WINDSTRESS=1
export LEGOESM_ITF_STRAIT_DEPTH=700
export LEGOESM_BH_BT=4e11
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding"

D=/work/bb1596/b384036/diffESM/data
WOA=/work/bd0854/DATA/ESMValTool2/OBS/Tier2/WOA
PY=/work/bb1596/b384036/b384036/envs/legoesm_cuda/bin/python

echo "host $(hostname)  start $(date)"
$PY -c "import jax; print('JAX devices:', jax.devices())"

$PY /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/run_omip.py \
  --grid latlon --resolution 720x1440 --nlev 35 --H-max 5500 --dt 300 --A-h 5e4 --K-h 250 \
  --bathymetry $D/bathy/bathy_025deg_etopo.nc --implicit-vertical-mixing \
  --forcing-mode jra55_do_tropical --jra55-cache $D/jra55_ryf9091_0p25deg_noleap.zarr --jra55-cycle \
  --woa-init --woa-t $D/woa13v2_thetao_annual_025deg.nc \
  --woa-s $D/woa13v2_so_annual_025deg.nc \
  --biogeo none --r-factor-max 0.999 --smoothing-passes 2 --physics full_bulk --no-conservation-fixer \
  --days 10 --checkpoint-days 5 \
  --output results/omip_qdeg_abg1e-2
echo "end $(date)"
