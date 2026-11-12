"""Daily climatologies for the ocean-WB forcing (user decisions 2026-11-11).

* ``siconc_oras5_1993_2017_daily.npy`` (365, 180, 360): ORAS5 sea-ice fraction averaged by
  noleap day of year over the TRAINING years only (no leakage from validation/test); Feb 29 is
  pooled with Feb 28; land / no-data -> 0.
* ``runoff_jra55do_daily.npy`` (365, 180, 360) [kg m-2 s-1]: the JRA55-do ``friver`` the OMIP
  runs use (the 1-degree ``jra55_do_fesom_1958_1958_1deg_noleap.zarr`` cache, 3-hourly records)
  averaged to daily means.  The cache's cells are centred at lon 0.5..359.5 while the model's are
  at 0..359, so the runoff is remapped conservatively onto the model centres.

    python build_ocean_wb_climatologies.py --oras5 DIR --jra55 ZARR --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np

SICONC_CHANNEL = 302          # ORAS5 store: 4 x 75 levels, then zos, mlotst, siconc, sithick


def noleap_doy(date):
    d = np.datetime64(date, "D")
    y = d.astype("datetime64[Y]")
    doy = int((d - y.astype("datetime64[D]")) / np.timedelta64(1, "D"))
    leap = int(str(y)) % 4 == 0 and (int(str(y)) % 100 != 0 or int(str(y)) % 400 == 0)
    if leap and doy >= 59:                      # Feb 29 -> Feb 28, later days shift back one
        doy -= 1
    return doy


def siconc_climatology(store, first=1993, last=2017):
    acc = np.zeros((365, 180, 360))
    n = np.zeros(365)
    d, end = np.datetime64(f"{first}-01-01"), np.datetime64(f"{last}-12-31")
    while d <= end:
        a = np.load(Path(store) / f"{d}.npy", mmap_mode="r")
        k = noleap_doy(d)
        acc[k] += np.nan_to_num(np.asarray(a[SICONC_CHANNEL], np.float64))
        n[k] += 1
        d += np.timedelta64(1, "D")
    if (n == 0).any():
        raise SystemExit(f"day-of-year bins without data: {np.flatnonzero(n == 0)}")
    return np.clip(acc / n[:, None, None], 0.0, 1.0)


def runoff_climatology(zarr_path):
    import xarray as xr
    ds = xr.open_zarr(zarr_path)
    fr = np.asarray(ds["friver"].values, np.float64)
    if fr.shape[0] != 365 * 8:
        raise SystemExit(f"expected 2920 3-hourly records, got {fr.shape[0]}")
    lat = np.asarray(ds["lat"].values)
    if not np.allclose(np.sort(lat), np.arange(-89.5, 90.0)):
        raise SystemExit("JRA55 cache latitudes are not the model's 1-degree centres")
    fr = np.nan_to_num(fr.reshape(365, 8, *fr.shape[1:]).mean(axis=1))
    if lat[0] > lat[-1]:
        fr = fr[:, ::-1]
    lon = np.asarray(ds["lon"].values)
    if not np.allclose(lon, np.arange(360.0)):
        # the 1-degree caches are centred at lon 0.5..359.5, the model at 0..359: conservative
        # remap onto the model centres (shared core regridder; total discharge conserved)
        from legoesm.grids.regridding import conservative_regrid_latlon
        lat_s = np.sort(lat)
        fr = conservative_regrid_latlon(np.moveaxis(fr, 0, -1), lat_s, lon, lat_s,
                                        np.arange(360.0))
        fr = np.moveaxis(fr, -1, 0)
    return fr


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oras5", required=True, type=Path)
    p.add_argument("--jra55", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "siconc_oras5_1993_2017_daily.npy", siconc_climatology(a.oras5).astype(np.float32))
    np.save(a.out / "runoff_jra55do_daily.npy", runoff_climatology(a.jra55).astype(np.float32))
    (a.out / "meta.json").write_text(json.dumps({
        "siconc": f"ORAS5 store {a.oras5}, channel {SICONC_CHANNEL}, 1993-2017, noleap doy, "
                  "Feb 29 pooled with Feb 28, NaN->0",
        "runoff": f"friver daily mean of {a.jra55} [kg m-2 s-1]",
        "grid": "lat -89.5..89.5 S->N, lon 0..359"}, indent=1))
    print("CLIMATOLOGIES_DONE", flush=True)


if __name__ == "__main__":
    main()
