#!/bin/bash
#SBATCH --job-name=omip_qdeg
#SBATCH --account=bb1596_gpu
#SBATCH --constraint=a100_80
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/slurm-omip_qdeg_prod-%j.out
#
# legoESM 1/4 deg OMIP physics-only run, JRA55-do RYF 1990/91, self-chaining.
# 30 model years: years 1-10 spin-up, years 11-30 analysis.
# Each job runs CHUNK_DAYS more model days from the latest restart and submits its
# follow-up (afterok: the chain stops on a blowup or crash). To continue after a stop,
# resubmit this file.

set -euo pipefail
TARGET_DAYS=10950         # 30 model years: 1-10 spin-up, 11-30 analysis
CHUNK_DAYS=210            # ~10 h per job at ~170 s per model day
SELF=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/submit_omip_qdeg_prod.sbatch
RDIR=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/results/omip_qdeg_prod/latlon/720x1440
D=/work/bb1596/b384036/diffESM/data
PY=/work/bb1596/b384036/b384036/envs/legoesm_cuda/bin/python

cd /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy
export JAX_ENABLE_X64=1
export LEGOESM_HEATFLUX=colleague
export LEGOESM_KV=0.0
export LEGOESM_KPP='{"K_bg": 1e-6}'
export LEGOESM_WINDSTRESS=1
export LEGOESM_ITF_STRAIT_DEPTH=700
export LEGOESM_BH_BT=4e11
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding"

LATEST=$(ls $RDIR/restart_day*.npz 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST" ]; then
  DONE=$((10#$(basename $LATEST .npz | sed 's/restart_day//')))
  RST="--restart $LATEST"
else
  DONE=0; RST=""
fi
if [ $DONE -ge $TARGET_DAYS ]; then echo "already at day $DONE >= $TARGET_DAYS, nothing to do"; exit 0; fi
NEXT=$(( DONE + CHUNK_DAYS )); [ $NEXT -gt $TARGET_DAYS ] && NEXT=$TARGET_DAYS
echo "host $(hostname)  start $(date)  day $DONE -> $NEXT  (target $TARGET_DAYS)  restart: ${LATEST:-none}"

if [ $NEXT -lt $TARGET_DAYS ]; then
  sbatch --dependency=afterok:$SLURM_JOB_ID $SELF
fi

$PY /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/run_omip.py \
  --grid latlon --resolution 720x1440 --nlev 35 --H-max 5500 --dt 300 --A-h 2e4 --K-h 250 \
  --bathymetry $D/bathy/bathy_025deg_etopo.nc --implicit-vertical-mixing \
  --forcing-mode jra55_do_tropical --jra55-cache $D/jra55_ryf9091_0p25deg_noleap.zarr --jra55-cycle \
  --woa-init --woa-t $D/woa13v2_thetao_annual_025deg.nc --woa-s $D/woa13v2_so_annual_025deg.nc \
  --biogeo none --r-factor-max 0.999 --smoothing-passes 2 --eq-smoothing-passes 0 \
  --physics full_bulk --no-conservation-fixer \
  --days $NEXT --checkpoint-days 30 $RST \
  --output results/omip_qdeg_prod
echo "end $(date)"
