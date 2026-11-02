"""SST/SSS bias of a model-year mean vs WOA13 surface.
Usage: python plot_bias_woa.py <run_dir>[:<run_dir2>...] <year | y0-y1> [tag]"""
import sys, glob, re, os, numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RUNS = [r.rstrip("/") for r in sys.argv[1].split(":")]
_y = sys.argv[2].split("-"); Y0, Y1 = int(_y[0]), int(_y[-1])
TAG = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.dirname(os.path.dirname(RUNS[-1])))
WOA_T="/work/bb1596/b384036/diffESM/data/woa13v2_thetao_annual_degC.nc"
WOA_S="/work/bd0854/DATA/ESMValTool2/OBS/Tier2/WOA/OBS_WOA_clim_2013v2_Omon_so_200001-200012.nc"
ylab = f"year {Y1}" if Y0 == Y1 else f"years {Y0}-{Y1}"
OUT=f"/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/eval_bias_woa/{TAG}_{ylab.replace(' ','')}.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
day = lambda f: int(re.search(r"day(\d+)", f).group(1))
files = {}
for r in RUNS:
    for f in glob.glob(r + "/restart_day*.npz"):
        if 365*(Y0-1) < day(f) <= 365*Y1: files[day(f)] = f
files = [files[d] for d in sorted(files)]
if not files: sys.exit(f"no snapshots for {ylab}")
print(len(files), f"snapshots in {ylab} (days {day(files[0])}-{day(files[-1])})")
sst = np.zeros((180,360)); sss = np.zeros((180,360))
for f in files:
    z = np.load(f); sst += z["T"][...,0]; sss += z["S"][...,0]
sst /= len(files); sss /= len(files)
ocean = np.load(files[0])["land_mask"] > 0.5

def woa_surface(path, var):
    da = xr.open_dataset(path, decode_times=False)[var]
    if "time" in da.dims: da = da.mean("time")
    latn=[d for d in da.dims if "lat" in d][0]; lonn=[d for d in da.dims if "lon" in d][0]
    other=[d for d in da.dims if d not in (latn,lonn)]
    if other: da = da.isel({other[0]:0})
    da = da.assign_coords({lonn:(da[lonn]%360)}).sortby(lonn).sortby(latn)
    return np.asarray(da.values, float)
woa_t = woa_surface(WOA_T,"thetao"); woa_s = woa_surface(WOA_S,"so")

lat=np.linspace(-89.5,89.5,180); w=np.cos(np.deg2rad(lat))[:,None]*np.ones((1,360))
def stats(b):
    m=ocean&np.isfinite(b); return np.sum(b[m]*w[m])/np.sum(w[m]), np.sqrt(np.sum(b[m]**2*w[m])/np.sum(w[m]))
bt=np.where(ocean,sst-woa_t,np.nan); bs=np.where(ocean,sss-woa_s,np.nan)
(mt,rt),(ms,rs)=stats(bt),stats(bs)
print(f"SST bias: mean {mt:+.2f} C, RMSE {rt:.2f} C")
print(f"SSS bias: mean {ms:+.2f} psu, RMSE {rs:.2f} psu")

ext=[0,360,-90,90]; zm=lambda a: np.nanmean(np.where(ocean&np.isfinite(a),a,np.nan),axis=1)
fig,ax=plt.subplots(2,2,figsize=(15,8),constrained_layout=True)
im=ax[0,0].imshow(bt,origin="lower",extent=ext,cmap="RdBu_r",vmin=-5,vmax=5,aspect="auto")
ax[0,0].set_title(f"SST bias, {ylab} mean - WOA13   (mean {mt:+.2f}, RMSE {rt:.2f} °C)"); plt.colorbar(im,ax=ax[0,0],label="°C")
im=ax[0,1].imshow(bs,origin="lower",extent=ext,cmap="BrBG",vmin=-2,vmax=2,aspect="auto")
ax[0,1].set_title(f"SSS bias, {ylab} mean - WOA13   (mean {ms:+.2f}, RMSE {rs:.2f} psu)"); plt.colorbar(im,ax=ax[0,1],label="psu")
ax[1,0].plot(lat,zm(sst),label="model"); ax[1,0].plot(lat,zm(woa_t),"k--",label="WOA13")
ax[1,0].set_xlabel("lat"); ax[1,0].set_ylabel("SST (°C)"); ax[1,0].set_title("Zonal-mean SST"); ax[1,0].legend()
ax[1,1].plot(lat,zm(sss),label="model"); ax[1,1].plot(lat,zm(woa_s),"k--",label="WOA13")
ax[1,1].set_xlabel("lat"); ax[1,1].set_ylabel("SSS (psu)"); ax[1,1].set_title("Zonal-mean SSS"); ax[1,1].legend()
fig.suptitle(f"legoESM 1° JRA55-do RYF, physics only — {ylab} mean")
fig.savefig(OUT,dpi=120); print("saved",OUT)
