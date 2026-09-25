#!/bin/bash
#SBATCH --job-name=plot4way
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/plot4way_%j.out
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc

$PY -u scripts/plot/plot_fourway_grids_nemo.py \
  --tripole $R/xgrid_trp_d30/snapshot_final.npz \
  --mpas $R/xgrid_mpas_d30/snapshot_final.npz \
  --fesom $R/fesom_core2_d30/snapshot_day0030.npz \
  --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
  --out-dir $R/fourway_d30
rc=$?
echo "=== rc=$rc ==="
exit $rc
