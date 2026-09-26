#!/bin/bash
#SBATCH --job-name=avtspread
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/avtspread_%j.out
# GLM's objection (2026-09-08): our 20-60 m diffusivity is ONE instantaneous
# field while the oracle's is a daily mean, and for an intermittent lognormal
# field that ordering alone could manufacture the 100x gap. The oracle's own
# hour-to-hour spread settles it: if every hour of its day sits near its daily
# mean, averaging order cannot be the explanation.
set -u
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY - <<'PY2'
import numpy as np, netCDF4 as nc
F=("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/"
   "RUN_TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc")
d=nc.Dataset(F)
lat=np.asarray(d.variables["nav_lat_grid_T"][:]); lon=np.asarray(d.variables["nav_lon_grid_T"][:])%360.
box=(np.abs(lat)<=2.)&(lon>=220.)&(lon<240.)
depw=d.variables["depthw"][:]
for lo,hi in ((0.,20.),(20.,60.),(60.,160.)):
    sel=(depw>=lo)&(depw<hi)
    med=[]
    for t in range(d.variables["avt"].shape[0]):
        a=np.ma.filled(d.variables["avt"][t][sel],np.nan)[:,box]
        med.append(np.nanmedian(a))
    med=np.array(med)
    print(f"{lo:5.0f}-{hi:5.0f} m : hourly medians min {med.min():.3e} max {med.max():.3e} "
          f"daily-mean-of-medians {med.mean():.3e}  max/min {med.max()/med.min():6.2f}")
print("\nREAD: if max/min is small, the oracle's mixing in that band is NOT intermittent,")
print("so a median of one instant against a median of a daily mean cannot differ by 100x.")
PY2
echo DONE
