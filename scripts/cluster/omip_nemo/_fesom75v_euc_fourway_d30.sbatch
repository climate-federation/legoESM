#!/bin/bash
#SBATCH --job-name=f75v_euc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/f75v_euc_%j.out
# FESOM on NEMO's 75 levels, day 30, with node velocities: (1) the same
# undercurrent/thermocline instrument as jobs 9648737 (tripole) so the three
# grids' EUC can be compared on one day against the same NEMO records;
# (2) the four-way figure (tripole/MPAS/FESOM-75/NEMO) at day 30.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=$ROOT/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
GT=$GW/ORCA1_5d_20000101_20000331_grid_T.nc
SNAP=$R/fesom_nemo75v_d30/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: missing $SNAP"; exit 1; }
$PY -c "import numpy as np; z=np.load('$SNAP'); print('keys', [k for k in z.files if k in ('u_east','v_north','lat','lon','lat_T','T','S','cell_area')]); print('u_east', z['u_east'].shape if 'u_east' in z.files else None)"

echo "=== EUC / thermocline FESOM-75v d30 vs GATEWAY recs 5:6 ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-t-recs 5:6 \
  --label fesom_nemo75v_d30 --nemo-wfile $GW/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile $GW/ORCA1_5d_20000101_20000331_grid_U.nc --meridional-lon 240,260 \
  --nemo-vfile $GW/ORCA1_5d_20000101_20000331_grid_V.nc 2>&1 | grep -vE "Warning|warn" | tail -160
echo "=== euc rc=$? ==="

echo "=== fourway d30 with FESOM-75v ==="
$PY -u scripts/plot/plot_fourway_grids_nemo.py \
  --tripole $R/trp_unified180/snapshot_day0030.npz \
  --mpas $R/mpas_unified_180d/snapshot_day0030.npz \
  --fesom $SNAP \
  --nemo-gridt "$GT" --nemo-time-idx 5 --res-deg 1.0 \
  --out-dir $R/fourway_f75v_d0030
echo "=== fourway rc=$? ==="
ls $R/fourway_f75v_d0030
echo DONE
