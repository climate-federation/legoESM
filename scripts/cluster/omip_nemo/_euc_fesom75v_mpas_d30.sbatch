#!/bin/bash
#SBATCH --job-name=euc_fm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/euc_fm_%j.out
# Undercurrent/thermocline instrument (same as job 9648737 on the tripole) on
# the two unstructured grids at day 30 vs GATEWAY recs 5:6.  FESOM snapshots
# carry no vertical mass flux, so the w block is dropped there (u_east only).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=$ROOT/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
GT=$GW/ORCA1_5d_20000101_20000331_grid_T.nc
for ARM in fesom_nemo75v_d30; do   # MPAS unified snapshot predates u_east; needs a rewrite
  SNAP=$R/$ARM/snapshot_day0030.npz
  echo; echo "############ $ARM"
  [ -f "$SNAP" ] || { echo "FATAL: missing $SNAP"; continue; }
  WARGS="--nemo-w-recs 5:6"
  $PY -c "import numpy as np,sys; sys.exit(0 if 'mass_flux_w' in np.load('$SNAP').files else 1)" \
    && WARGS="--nemo-w-recs 5:6 --nemo-wfile $GW/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo --meridional-lon 240,260 --nemo-vfile $GW/ORCA1_5d_20000101_20000331_grid_V.nc"
  echo "w-block args: '${WARGS}'"
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-t-recs 5:6 \
    --label ${ARM}_d30 --nemo-ufile $GW/ORCA1_5d_20000101_20000331_grid_U.nc $WARGS \
    2>&1 | grep -vE "Warning|warn|^\s+[a-zA-Z_]+ = np\." | tail -150
  echo "=== $ARM rc=$?"
done
echo DONE
