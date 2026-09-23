#!/bin/bash
#SBATCH --job-name=d852
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/d852_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
$PY - <<'PY2'
import numpy as np, netCDF4 as nc
GW="/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/"
w=nc.Dataset(GW+"ORCA1_5d_20000101_20000331_grid_W.nc")
t=nc.Dataset(GW+"ORCA1_5d_20000101_20000331_grid_T.nc")
zw=w.variables["depthw"][:]; zt=t.variables["deptht"][:]
lat=np.asarray(t.variables["nav_lat"][:]); lon=np.asarray(t.variables["nav_lon"][:])%360.
box=(np.abs(lat)<=2.0)&(lon>=220.)&(lon<240.)
avt=np.ma.filled(np.ma.masked_invalid(w.variables["avt"][5]),np.nan)
to =np.ma.filled(np.ma.masked_invalid(t.variables["to"][5]),np.nan)
for z0 in (15.,50.,150.,155.):
    k=int(np.argmin(np.abs(zw-z0)))
    a=avt[k][box]
    print(f"interface {z0:6.1f} m -> index {k:2d} depthw {zw[k]:7.2f}  "
          f"avt median {np.nanmedian(a):.3e} max {np.nanmax(a):.3e}  "
          f"n_finite {np.isfinite(a).sum()}/{box.sum()}")
print("deptht spacing near the layer:", [f"{zt[k]:.1f}" for k in range(20,32)])
kt=int(np.argmin(np.abs(zw-50.))); kb=int(np.argmin(np.abs(zw-155.)))
for k in (kt,kb):
    dT=(to[k]-to[k-1])[box]; dz=zt[k]-zt[k-1]
    print(f"k={k} depthw {zw[k]:.2f}  dz(deptht) {dz:.4f} m  "
          f"dT median {np.nanmedian(dT):+.4f}  dT/dz median {np.nanmedian(dT)/dz:+.4e} K/m")
PY2
echo DONE
