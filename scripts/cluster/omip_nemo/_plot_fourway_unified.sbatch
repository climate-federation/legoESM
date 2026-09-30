#!/bin/bash
#SBATCH --job-name=fourway_uni
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fourway_uni_%j.out
# All three grids + NEMO on ONE figure per field, unified arms, GATEWAY recs
# 5/11/17 = days 30/60/90 (user 2026-09-04: "show all grids together on one png").
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_omip_wt
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
# PAIRS="day:rec ..." (default all three GATEWAY scoring days)
for pair in ${PAIRS:-30:5 60:11 90:17}; do
  set -- ${pair/:/ }; printf -v day %04d $1
  $PY -u scripts/plot/plot_fourway_grids_nemo.py \
    --tripole $R/${TRP:-trp_unified180}/snapshot_day$day.npz \
    --mpas $R/${MPAS:-mpas_unified_180d}/snapshot_day$day.npz \
    --fesom $R/${FESOM:-fesom_b5_d90}/snapshot_day$day.npz \
    --nemo-gridt "$GT" --nemo-time-idx $2 --res-deg 1.0 \
    --out-dir $R/fourway_${TAG:-uni}_d$day
  echo "=== d$day rc=$? ==="
done
