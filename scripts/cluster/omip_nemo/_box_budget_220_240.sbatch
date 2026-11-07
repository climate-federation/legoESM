#!/bin/bash
#SBATCH --job-name=boxbud
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/boxbud_%j.out
# Which wall carries the 220-240E lower-thermocline convergence (isopycnal
# probe 9649116)?  Volume budget of the 50-150 m box from the STORED tracer
# fluxes, tripole day 15 and 30 vs NEMO GATEWAY recs 2 and 5.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=$ROOT/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
ARM=${ARM:-trp_unified180}
for pair in "0015 2:3" "0030 5:6"; do
  set -- $pair
  for LAYER in 0,50 50,150 150,300; do
    echo "########## $ARM day $1 recs $2 layer $LAYER"
    $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
      --legoesm-snapshot $R/$ARM/snapshot_day$1.npz --nemo-gridt $GW/ORCA1_5d_20000101_20000331_grid_T.nc \
      --nemo-t-recs $2 --label ${ARM}_d$1 --nemo-w-recs $2 \
      --nemo-ufile $GW/ORCA1_5d_20000101_20000331_grid_U.nc --nemo-vfile $GW/ORCA1_5d_20000101_20000331_grid_V.nc \
      --nemo-wfile $GW/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo --mesh-mask data/grids/eORCA1.2_mesh_mask.nc \
      --box-budget 220,240 --budget-layer $LAYER 2>&1 | sed -n '/=== BOX/,$p' | grep -vE "RuntimeWarning|^\s|^READ"
  done
done
echo DONE
