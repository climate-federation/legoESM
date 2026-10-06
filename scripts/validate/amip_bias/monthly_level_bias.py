#!/usr/bin/env python3
"""Monthly temperature bias at given pressure levels, model minus ERA5, by region.

For each month the model's CMOR monthly ta carries, ERA5's monthly ta of the
same month is box-averaged into the model cells (t2m_window_metrics.bin_to_model)
and the cos-lat weighted bias is printed for 45-70N land / ocean and the
tropics (land = sftlf >= 50 %).  Cells below ground on either side (non-finite)
are left out and counted.

Usage: monthly_level_bias.py ERA5_TA.nc MODEL_TA.nc MODEL_SFTLF.nc PLEV_PA [PLEV_PA ...]
"""
import sys

import numpy as np
import xarray as xr

from t2m_window_metrics import bin_to_model


def main(e5f, mf, lff, plevs):
    m = xr.open_dataset(mf)["ta"]
    lf = xr.open_dataset(lff)["sftlf"].squeeze(drop=True).values
    lat, lon = m["lat"].values, m["lon"].values
    e5 = xr.open_dataset(e5f)["ta"]
    LA = lat[:, None] * np.ones((1, lon.size))
    land = lf >= 50.0
    regions = {"45-70N land": land & (LA >= 45) & (LA <= 70),
               "45-70N ocean": ~land & (LA >= 45) & (LA <= 70),
               "tropics": np.abs(LA) <= 30}
    wcos = np.cos(np.deg2rad(LA))
    print("month  plev  " + "  ".join(f"{r:>14s}" for r in regions))
    for t in m["time"].values:
        ym = str(t)[:7]
        e5m = e5.sel(time=ym)
        if e5m.sizes.get("time", 1) != 1:
            raise SystemExit(f"ERA5 has {e5m.sizes['time']} records for {ym}")
        e5m = e5m.squeeze("time", drop=True)
        for p in plevs:
            mv = m.sel(time=t).sel(plev=p, method="nearest")
            ev = e5m.sel(plev=p, method="nearest")
            if abs(float(mv.plev) - p) > 1 or abs(float(ev.plev) - p) > 1:
                raise SystemExit(f"level {p} missing")
            la, lo = ev["lat"].values, ev["lon"].values
            v = ev.transpose("lat", "lon").values
            ok = np.isfinite(v)
            ref = bin_to_model(np.where(ok, v, 0.0), la, lo, lat, lon)
            cov = bin_to_model(ok.astype(float), la, lo, lat, lon)
            ref = np.where(cov > 0.999, ref, np.nan)      # drop cells with any ERA5 point below ground
            d = mv.values - ref
            out = []
            for name, msk in regions.items():
                g = msk & np.isfinite(d)
                out.append(f"{(d[g] * wcos[g]).sum() / wcos[g].sum():+7.2f} n{int(g.sum()):4d}")
            print(f"{ym}  {p / 100:4.0f}  " + "  ".join(f"{x:>14s}" for x in out))


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], [float(x) for x in sys.argv[4:]])
