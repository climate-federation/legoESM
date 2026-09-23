#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eq_tsec_%j.out
# GLM's grid-vs-config discriminator: equatorial 2S-2N zonal T-section (0-300m)
# + surface T + EUC, tripole vs MPAS vs NEMO, from EXISTING d30 snapshots.
# Localized differences (coastal/Galapagos/thermocline sharpness) = grid;
# smooth basin-scale offset = config (e.g. --iwm).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
$PY - <<'PY'
import numpy as np, glob, os
def load(f):
    z=np.load(f); T=np.asarray(z['T']); u=np.asarray(z['u'])
    lat=np.asarray(z['lat_T']); lon=np.asarray(z['lon_T'])
    if np.nanmax(np.abs(lat))<=np.pi+1e-6: lat=np.degrees(lat); lon=np.degrees(lon)%360
    else: lon=lon%360
    zc=np.abs(np.asarray(z['z_center_ref'])); wet=np.asarray(z['land_mask'])>0.5
    return T,u,lat,lon,zc,wet
def eqsec(f, lonbins=np.arange(150,290,10)):
    T,u,lat,lon,zc,wet=load(f)
    uc=0.5*(u[:,:-1]+u[:,1:]) if u.shape[1]==T.shape[1]+1 else u
    eq=(np.abs(lat)<=2)&wet
    sst=[]; euc=[]
    for lo in lonbins:
        m=eq&(lon>=lo)&(lon<lo+10)
        if m.any():
            sst.append(np.nanmean(T[...,0][m]))
            kmax=int(np.searchsorted(zc,300))+1
            euc.append(np.nanmax(uc[...,:kmax][m]) if uc[...,:kmax][m].size else np.nan)
        else: sst.append(np.nan); euc.append(np.nan)
    return lonbins, np.array(sst), np.array(euc)
runs={'tripole':'results/omip_nemo/nemolev_trp_long180/snapshot_day0030.npz',
      'MPAS':'results/omip_nemo/mpas_modfrc_d30/snapshot_day0030.npz'}
for k,f in runs.items():
    if not os.path.exists(f):
        alt=sorted(glob.glob(f'results/omip_nemo/*{k.lower()}*/snapshot_day0030.npz'))+sorted(glob.glob(f'results/omip_nemo/*{k.lower()}*/snapshot_final.npz'))
        if alt: runs[k]=alt[0]
print("Equatorial 2S-2N surface T [C] by longitude (grid-localized vs smooth):")
print("lonE   " + " ".join("%4d"%l for l in np.arange(150,290,10)))
res={}
for k,f in runs.items():
    if os.path.exists(f):
        lb,sst,euc=eqsec(f); res[k]=(sst,euc)
        print("%-7s"%k + " ".join("%4.1f"%v for v in sst))
if 'tripole' in res and 'MPAS' in res:
    d=res['tripole'][0]-res['MPAS'][0]
    print("trp-MPAS " + " ".join("%+4.1f"%v for v in d))
    print("\nEUC core [m/s] by longitude:")
    for k in ['tripole','MPAS']:
        if k in res: print("%-7s"%k + " ".join("%4.2f"%v for v in res[k][1]))
    print("\n=> smooth ~constant trp-MPAS offset = CONFIG (e.g. iwm); localized spikes (coast/Galapagos ~270-280E) = GRID")
PY
echo DONE
