#!/bin/bash
#SBATCH --job-name=evd_delta
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:15:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/evd_delta_%j.out
# Did the EVD lever ENGAGE? arm (trp_phase4evd_d5) minus control (trp_phase4_d5)
# at day 5: where and how much T changed. A null nino3 score with a null delta
# field is a control that proves nothing; a null score with a nonzero delta
# elsewhere is a refutation of EVD as the nino3 lever.
set -u
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python - <<PYEOF
import numpy as np
a=np.load("$R/trp_phase4evd_d5/snapshot_final.npz"); c=np.load("$R/trp_phase4_d5/snapshot_final.npz")
Ta=np.nan_to_num(a["T"]); Tc=np.nan_to_num(c["T"]); d=Ta-Tc
_lk=[k for k in ("lat_T","lat","lat2d","nav_lat","latitude") if k in a.files][0]; _ok=[k for k in ("lon_T","lon","lon2d","nav_lon","longitude") if k in a.files][0]
lat=np.asarray(a[_lk]); lon=np.asarray(a[_ok])%360
if lat.ndim==1: lon,lat=np.meshgrid(lon,lat)
print("shapes", d.shape, lat.shape)
ad=np.abs(d)
print("GLOBAL max|dT| = %.4f K; cells |dT|>0.01: %d, >0.1: %d (of %d wet-ish)"%(ad.max(), (ad>0.01).sum(), (ad>0.1).sum(), (Ta!=0).sum()))
k=np.unravel_index(ad.argmax(), ad.shape); print("argmax at lat %.1f lon %.1f level %d"%(lat[k[0],k[1]], lon[k[0],k[1]], k[2]))
for name,(la0,la1,lo0,lo1) in {"nino3":(-5,5,210,270),"eq_pac_2S2N":(-2,2,140,280),"SO_50-70S":(-70,-50,0,360),"NH_50-80N":(50,80,0,360)}.items():
    m=(lat>=la0)&(lat<=la1)&(lon>=lo0)&(lon<=lo1)
    top=ad[m][:, :3]; print("%-12s max|dT| top3 = %.4f, cols with sfc |dT|>0.01: %d/%d, mean dT sfc = %+.4f, mean dT 0-10 lev = %+.4f"%(name, top.max(), (ad[m][:,0]>0.01).sum(), m.sum(), d[m][:,0].mean(), d[m][:,:10].mean()))
# zonal-mean of the column-max |dT| by 10-deg latitude band
for b in range(-80,80,10):
    m=(lat>=b)&(lat<b+10); 
    if m.any(): print("lat %+3d..%+3d: max|dT| %.4f  cols>0.01: %d"%(b,b+10, ad[m].max(), (ad[m].max(axis=1)>0.01).sum()))
PYEOF
echo "=== card exit: $? ==="
