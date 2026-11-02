"""Full-water-column T/S bias of legoESM vs WOA13.
Usage: python eval_column_woa.py --runs <dir> [<dir> ...] --years 1 3 5 --tag omip_v3
Last year listed is used for sections/maps; all years for the profile drift."""
import argparse, glob, re, os, numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
E="/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/eval_bias_woa"
p=argparse.ArgumentParser(); p.add_argument("--runs",nargs="+",required=True)
p.add_argument("--years",nargs="+",type=int,required=True); p.add_argument("--tag",default="run")
a=p.parse_args(); os.makedirs(E,exist_ok=True)
zz=np.load(E+"/z_levels.npz"); zc, zh = zz["zc"], zz["zh"]; dz=np.diff(zh); nz=len(zc)
WOA_T="/work/bb1596/b384036/diffESM/data/woa13v2_thetao_annual_degC.nc"
WOA_S="/work/bd0854/DATA/ESMValTool2/OBS/Tier2/WOA/OBS_WOA_clim_2013v2_Omon_so_200001-200012.nc"

files=[f for r in a.runs for f in glob.glob(r.rstrip("/")+"/restart_day*.npz")]
day=lambda f:int(re.search(r"day(\d+)",f).group(1))
def annual(y):
    fs=[f for f in files if 365*(y-1)<day(f)<=365*y]
    if not fs: raise SystemExit(f"no snapshots for year {y}")
    T=np.mean([np.load(f)["T"] for f in fs],axis=0); S=np.mean([np.load(f)["S"] for f in fs],axis=0)
    print(f"year {y}: {len(fs)} snapshots"); return T,S
z0=np.load(files[0]); H=z0["H_bathy"]; ocean=z0["land_mask"]>0.5
wet=ocean[...,None]&(zc[None,None,:]<H[...,None])

def woa3d(path,var):
    da=xr.open_dataset(path,decode_times=False)[var]
    if "time" in da.dims: da=da.mean("time")
    latn=[d for d in da.dims if "lat" in d][0]; lonn=[d for d in da.dims if "lon" in d][0]
    dn=[d for d in da.dims if d not in (latn,lonn)][0]
    da=da.assign_coords({lonn:(da[lonn]%360), dn:np.abs(da[dn])}).sortby(lonn).sortby(latn)
    return np.asarray(da.interp({dn:zc}).transpose(latn,lonn,dn).values,float)
wT=woa3d(WOA_T,"thetao"); wS=woa3d(WOA_S,"so")
ok=wet&np.isfinite(wT)&np.isfinite(wS)

lat=np.linspace(-89.5,89.5,180); lon=np.linspace(0.5,359.5,360); LON,LAT=np.meshgrid(lon,lat)
w2=np.cos(np.deg2rad(LAT))
atl=(((LON>=290)|(LON<20))|((LON>=260)&(LAT>=18))|((LON>=275)&(LAT>=9)))&(LAT<70)
atl&=~((LAT<-35)&~((LON>=290)|(LON<20)))
ind=(LON>=20)&(LON<147)&(LAT<30)&~atl&~((LON>=100)&(LAT>-8))
pac=~atl&~ind&(LAT<66)
basins={"Global":np.ones_like(atl),"Atlantic":atl,"Pacific":pac,"Indian":ind}

ys=a.years; data={y:annual(y) for y in ys}; T,S=data[ys[-1]]
bT=np.where(ok,T-wT,np.nan); bS=np.where(ok,S-wS,np.nan)
def setz(ax):
    ax.margins(y=0); ax.set_yscale("symlog",linthresh=500,linscale=2); ax.set_ylim(zh[-1],0)
    ax.set_yticks([50,200,500,1000,2000,3000,5000]); ax.set_ylabel("depth (m)")

# --- 1. zonal-mean bias sections per basin
fig,ax=plt.subplots(4,2,figsize=(14,16),constrained_layout=True)
for i,(name,bm) in enumerate(basins.items()):
    m3=ok&bm[...,None]
    for j,(b,w,lim,lab,cm) in enumerate([(bT,wT,3,"T bias (°C)","RdBu_r"),(bS,wS,0.5,"S bias (psu)","BrBG")]):
        zm=np.nanmean(np.where(m3,b,np.nan),axis=1); zw=np.nanmean(np.where(m3,w,np.nan),axis=1)
        im=ax[i,j].pcolormesh(lat,zc,zm.T,cmap=cm,vmin=-lim,vmax=lim,shading="auto")
        cs=ax[i,j].contour(lat,zc,zw.T,levels=[4,8,12,16,20,24] if j==0 else [34.2,34.5,34.8,35.1,35.5],colors="k",linewidths=0.6)
        ax[i,j].clabel(cs,fontsize=7); setz(ax[i,j]); ax[i,j].set_title(f"{name}: {lab}, year {ys[-1]}  (contours: WOA)")
        plt.colorbar(im,ax=ax[i,j])
fig.savefig(f"{E}/{a.tag}_sections_year{ys[-1]}.png",dpi=110); plt.close(fig)

# --- 2. bias maps at fixed depths
targets=[100,500,1000,2000]
fig,ax=plt.subplots(4,2,figsize=(15,16),constrained_layout=True)
for i,d in enumerate(targets):
    k=int(np.argmin(np.abs(zc-d)))
    for j,(b,lim,cm,u) in enumerate([(bT,3,"RdBu_r","°C"),(bS,0.5,"BrBG","psu")]):
        im=ax[i,j].imshow(b[...,k],origin="lower",extent=[0,360,-90,90],cmap=cm,vmin=-lim,vmax=lim,aspect="auto")
        ax[i,j].set_title(f"{'T' if j==0 else 'S'} bias at {zc[k]:.0f} m, year {ys[-1]}"); plt.colorbar(im,ax=ax[i,j],label=u)
fig.savefig(f"{E}/{a.tag}_depthmaps_year{ys[-1]}.png",dpi=110); plt.close(fig)

# --- 3. global-mean profiles and their drift
def gprof(X):
    return np.array([np.nansum(np.where(ok[...,k],X[...,k],0)*w2)/np.sum(np.where(ok[...,k],w2,0)) for k in range(nz)])
fig,ax=plt.subplots(1,4,figsize=(18,7),constrained_layout=True)
pw=(gprof(wT),gprof(wS))
for y in ys:
    pt,ps=gprof(data[y][0]),gprof(data[y][1])
    ax[0].plot(pt,zc,label=f"yr {y}"); ax[1].plot(ps,zc,label=f"yr {y}")
    ax[2].plot(pt-pw[0],zc,label=f"yr {y}"); ax[3].plot(ps-pw[1],zc,label=f"yr {y}")
ax[0].plot(pw[0],zc,"k--",label="WOA13"); ax[1].plot(pw[1],zc,"k--",label="WOA13")
for x,t in zip(ax,["global-mean T (°C)","global-mean S (psu)","T bias (°C)","S bias (psu)"]):
    setz(x); x.set_xlabel(t); x.legend(fontsize=8); x.grid(alpha=0.3)
ax[2].axvline(0,c="k",lw=0.8); ax[3].axvline(0,c="k",lw=0.8)
fig.savefig(f"{E}/{a.tag}_profiles.png",dpi=110); plt.close(fig)

# --- table: volume-weighted bias by depth band, last year
vw=w2[...,None]*dz[None,None,:]
print(f"\nYear {ys[-1]} volume-weighted bias vs WOA13 (global)")
print("  depth band        T mean   T RMSE    S mean   S RMSE")
for lo,hi in [(0,200),(200,1000),(1000,2000),(2000,6000)]:
    km=(zc>=lo)&(zc<hi); m=ok&km[None,None,:]
    f=lambda b: (np.sum(b[m]*vw[m])/np.sum(vw[m]), np.sqrt(np.sum(b[m]**2*vw[m])/np.sum(vw[m])))
    (tm,tr),(sm,sr)=f(np.nan_to_num(bT)),f(np.nan_to_num(bS))
    print(f"  {lo:4d}-{hi:4d} m    {tm:+7.2f}  {tr:7.2f}   {sm:+7.3f}  {sr:7.3f}")
print("\nFigures in", E)
