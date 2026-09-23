#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score_ab_emin_%j.out
# Score both A/B arms' final-day snapshots (day ~29.98, written as snapshot_final)
# vs NEMO GATEWAY rec 5 (days 26-30): nino3 bias + SST/SSS/MLD maps. Also prints
# the EUC core + cold-tongue tendency (d10->final) for fix vs base.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate:$ROOT/scripts/run"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

for a in fix base; do
  SNAP=$ROOT/results/omip_nemo/ab_emin_$a/snapshot_final.npz
  [ -f "$SNAP" ] || { echo "MISSING $SNAP"; continue; }
  echo "############ arm $a (final day) vs GATEWAY rec 5 ############"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label ab_emin_${a}_final \
    --output-dir $ROOT/results/omip_nemo/compare_ab_emin_${a}_final 2>&1 \
    | grep -E "nino3_5S5N|\[SST\]|\[SSS\]|MLD" | head -6
done

echo "############ EUC core + cold-tongue tendency (d10 -> final), fix vs base ############"
$PY - <<'PY'
import numpy as np, glob, os
def m(f):
    z=np.load(f); T=np.asarray(z['T'])[...,0]; u=np.asarray(z['u'])
    uc=0.5*(u[:,:-1]+u[:,1:]) if u.shape[1]==T.shape[1]+1 else u
    lat=np.asarray(z['lat_T']); lon=np.asarray(z['lon_T'])
    if np.nanmax(np.abs(lat))<=np.pi+1e-6: lat=np.degrees(lat); lon=np.degrees(lon)%360
    else: lon=lon%360
    zc=np.abs(np.asarray(z['z_center_ref'])); wet=np.asarray(z['land_mask'])>0.5
    eq=(np.abs(lat)<=1)&(lon>=215)&(lon<=225); kmax=int(np.searchsorted(zc,300))+1
    ueq=uc[...,:kmax][eq]; euc=float(np.nanmax(ueq)) if ueq.size else float('nan')
    ct=(np.abs(lat)<=2)&(lon>=220)&(lon<=240)&wet
    return euc, float(np.nanmean(T[ct]))
for a in ['base','fix']:
    d10=f'results/omip_nemo/ab_emin_{a}/snapshot_day0010.npz'
    fin=f'results/omip_nemo/ab_emin_{a}/snapshot_final.npz'
    e10,c10=m(d10); ef,cf=m(fin)
    print(f'{a}: EUC {e10:.4f}->{ef:.4f} m/s ; coldtongue {c10:.4f}->{cf:.4f} C (tend {cf-c10:+.4f})')
PY
echo DONE
