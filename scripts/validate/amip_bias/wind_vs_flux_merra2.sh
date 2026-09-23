#!/bin/bash
cd /work/bd1083/b309178/diffESM/legoesm_pg/wt_watervapor
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
export PYTHONPATH=$(ls -d $PWD/packages/*/ | tr '\n' ':')$PWD/src
CPUS=$($PY -c "import os;print(','.join(map(str,sorted(os.sched_getaffinity(0))[:8])))")
taskset -c "$CPUS" $PY - <<'PYEOF' 2>&1 | grep -v "CUDA\|cuInit\|jax_plugins\|^  \|^    \|Traceback\|_check_cuda"
import sys, glob; sys.path.insert(0,"scripts/validate/amip_bias")
import numpy as np, xarray as xr, regional_bias as rb
MERRA="/work/bd1179/b309141/climateeval_input/reanalysis_MERRA2/mon"

def merra(var, months, mlat, mlon):
    """MERRA2 monthly climatology on the model grid, without dask.

    The shared loader uses open_mfdataset, which needs a chunk manager this
    environment does not have.  MERRA2 stores one file per year, so the
    per-file monthly means are accumulated directly -- same arithmetic, no
    dependency.  Years are equally weighted, as the shared loader does.
    """
    fs = sorted(glob.glob(f"{MERRA}/{var}/*.nc"))
    if not fs:
        raise SystemExit(f"FATAL: no MERRA2 {var}")
    acc, n = None, 0
    for f in fs:
        d = xr.open_dataset(f)
        v = d[var]
        yr = int(str(np.asarray(v.time)[0])[:4])
        if yr < rb.REF_MIN_YEAR:
            continue
        sel = v.groupby("time.month").mean("time")
        got = [m for m in months if m in set(np.asarray(sel["month"]).tolist())]
        if not got:
            continue
        a = np.asarray(sel.sel(month=got).mean("month"), dtype=np.float64)
        acc = a if acc is None else acc + a
        n += 1
    if n == 0:
        raise SystemExit(f"FATAL: no MERRA2 {var} years >= {rb.REF_MIN_YEAR}")
    arr = acc / n
    d0 = xr.open_dataset(fs[0])
    rlat = np.asarray(d0["lat"], dtype=np.float64)
    rlon = np.asarray(d0["lon"], dtype=np.float64) % 360.0
    if arr.shape != (rlat.size, rlon.size):
        arr = arr.T
    return np.asarray(rb.bin_to_model(arr, rlat, rlon, mlat, mlon, label=var))

B={"tropical ocean 20S-20N":(-20,20,0,360),"trades 10-30N":(10,30,0,360),
   "trades 10-30S":(-30,-10,0,360),"global":(-90,90,0,360)}
run="wv_sfcoff30"
du,dv=rb._load_model(run,"ua"),rb._load_model(run,"va")
lat,lon=np.asarray(du.lat),np.asarray(du.lon)
plev=np.asarray(du["plev"],dtype=np.float64); k=int(np.argmin(np.abs(plev-100000.0)))
months=rb._month_labels(du)
msp=np.hypot(np.asarray(du["ua"]).mean(0)[k], np.asarray(dv["va"]).mean(0)[k])
osp=np.hypot(merra("uas",months,lat,lon), merra("vas",months,lat,lon))
mh=np.asarray(rb._load_model(run,"hfls")["hfls"]).mean(0)
oh=merra("hfls",months,lat,lon)
ps=np.asarray(rb._load_model(run,"ps")["ps"]).mean(0)
_fs=sorted(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_fx_*.nc"))
if not _fs: raise SystemExit("FATAL: no sftlf")
_d=xr.open_dataset(_fs[0])
fl=np.asarray(rb.bin_to_model(np.asarray(_d["sftlf"],dtype=np.float64)/100.0,
    np.asarray(_d.lat), np.asarray(_d.lon)%360.0, lat, lon, label="sftlf"))
ocean=(fl<0.5)&(ps>100500.0)
print(f"{'band':<24}{'mdl|U|135m':>12}{'M2|U|10m':>10}{'U ratio':>9}{'mdl hfls':>10}{'M2 hfls':>9}{'LH ratio':>10}")
for n,b in B.items():
    a=rb.region_mean(msp,lat,lon,b,valid=ocean); o=rb.region_mean(osp,lat,lon,b,valid=ocean)
    h=rb.region_mean(mh,lat,lon,b,valid=ocean); ho=rb.region_mean(oh,lat,lon,b,valid=ocean)
    print(f"{n:<24}{a:12.2f}{o:10.2f}{a/o:9.2f}{h:10.1f}{ho:9.1f}{h/ho:10.2f}")
print("\nHEIGHT MISMATCH, stated not hidden: the model column is its LOWEST FULL")
print("LEVEL (~135 m); MERRA2's is 10 m. A log profile makes 135 m read ~20%")
print("HIGHER than 10 m, so a U ratio near 1.0 already means the model is slow.")
print("Latent heat is like-for-like and needs no such correction.")
PYEOF
echo WIND3_DONE
