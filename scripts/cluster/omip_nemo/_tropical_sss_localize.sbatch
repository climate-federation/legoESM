#!/bin/bash
#SBATCH --job-name=trop_sss_loc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trop_sss_loc_%j.out
#
# WHERE did tropical SSS rmse grow (0.575 -> 0.803) between the previous MPAS
# 180d arm and the unified one at day 60?  Band bias is unchanged, so it is
# structure.  Four terms went on at once (bbl-adv, qlead, isf, iwm); the
# location names the suspect before any A/B run is spent.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD $(git rev-parse --short HEAD)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
GW_T=$N/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
R=$ROOT/results/omip_nemo
$PY -u scripts/tmp/_where_tropical_sss_grew.py \
  --old $R/mpas_fullharm_180d/snapshot_day0060.npz \
  --new $R/mpas_unified_180d/snapshot_day0060.npz \
  --nemo-gridt $GW_T --nemo-time-idx 11 --top 25
echo "rc=$?"
echo DONE
