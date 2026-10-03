#!/usr/bin/env bash
# ClimateEval data root whose ERA5 monthly tas for one month is replaced by the
# mean over a DAY WINDOW of that month (from the DKRZ ERA5 daily-mean pool), so
# a model window (cmor_window.py) is scored against ERA5 on the SAME days.
# Every other dataset/variable is a symlink to the real root (unchanged).
#
#   climateeval_era5_window_ref.sh <yyyy-mm-dd first> <yyyy-mm-dd last> <out_root>
#
# Output: <out_root> usable as --data-root-dir; the ERA5 tas file holds ONE
# time step (mid-month stamp of that month) on the original 0.25-degree grid.
set -euo pipefail
D0=$1; D1=$2; OUT=$3
SRC=/work/bd1179/b309141/climateeval_input
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
YM=${D0:0:7}
source /usr/share/Modules/init/bash
module load cdo/2.6.0-gcc-11.2.0
mkdir -p "$OUT/reanalysis_ERA5/mon/tas"
for d in "$SRC"/*; do
  [[ $(basename "$d") == reanalysis_ERA5 ]] && continue
  ln -sfn "$d" "$OUT/$(basename "$d")"
done
for d in "$SRC"/reanalysis_ERA5/*; do
  [[ $(basename "$d") == mon ]] && continue
  ln -sfn "$d" "$OUT/reanalysis_ERA5/$(basename "$d")"
done
for d in "$SRC"/reanalysis_ERA5/mon/*; do
  [[ $(basename "$d") == tas ]] && continue
  ln -sfn "$d" "$OUT/reanalysis_ERA5/mon/$(basename "$d")"
done
TMP="$OUT/_t2m_window.nc"
cdo -s -O -f nc -timmean -seldate,"$D0","$D1" -setgridtype,regular \
  "/pool/data/ERA5/E5/sf/an/1D/167/E5sf00_1D_${YM}_167.grb" "$TMP"
"$PY" - "$SRC" "$TMP" "$OUT" "$YM" "$D0" "$D1" <<'EOF'
import glob, sys
import numpy as np, xarray as xr
src, tmp, out, ym, d0, d1 = sys.argv[1:]
f = sorted(glob.glob(f"{src}/reanalysis_ERA5/mon/tas/tas_*.nc"))[0]
ref = xr.open_dataset(f)
y, m = map(int, ym.split("-"))
one = ref.sel(time=(ref.time.dt.year == y) & (ref.time.dt.month == m)).load()
assert one.sizes["time"] == 1, one.time.values
w = xr.open_dataset(tmp)
v = next(w[k] for k in w.data_vars if w[k].ndim >= 3).isel(time=0).squeeze()
v = v.rename({v.dims[0]: "lat", v.dims[1]: "lon"}).sortby("lat")
a = v.values
# Gaussian rows stop at +-89.78: pole rows = zonal mean of the nearest ring;
# longitude made periodic for the 359.75 column.
a = np.vstack([np.full((1, a.shape[1]), a[0].mean()), a, np.full((1, a.shape[1]), a[-1].mean())])
a = np.hstack([a, a[:, :1]])
lat = np.concatenate([[-90.0], v.lat.values, [90.0]])
lon = np.concatenate([v.lon.values, [v.lon.values[0] + 360.0]])
v = xr.DataArray(a, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
new = v.interp(lat=one.lat, lon=one.lon).values.astype(np.float32)
assert np.isfinite(new).all()
old = one["tas"].values[0]
print(f"window {d0}..{d1}: global mean {float(new.mean()):.2f} K vs month "
      f"{float(old.mean()):.2f} K; corr {np.corrcoef(new.ravel(), old.ravel())[0, 1]:.4f}")
one["tas"].values[0] = new
one.attrs["comment"] = f"tas replaced by the ERA5 daily-mean average over {d0}..{d1}"
stamp = str(one.time.values[0])[:19].replace("-", "").replace(":", "").replace("T", "")
one.to_netcdf(f"{out}/reanalysis_ERA5/mon/tas/tas_native6_ERA5_Amon_mon_{stamp}-{stamp}.nc")
EOF
rm -f "$TMP"
