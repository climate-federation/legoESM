"""Monthly sea-ice concentration for run_omip.py --sea-ice-clim, from HadISST.
Output: siconc (month, lat, lon), fraction 0-1, lat ascending, lon 0-360.
Default: the months of the JRA55-do repeat year 1990/91 (May-Dec 1990, Jan-Apr 1991);
--years Y0-Y1 gives a multi-year monthly climatology instead."""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hadisst", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", default=None)
    a = ap.parse_args()
    import xarray as xr
    ds = xr.open_dataset(a.hadisst)
    var = "siconc" if "siconc" in ds else "sic"
    da = ds[var]
    lat = next(d for d in da.dims if d.lower().startswith("lat"))
    lon = next(d for d in da.dims if d.lower().startswith("lon"))
    t = da["time"].dt
    if a.years:
        y0, y1 = (int(x) for x in a.years.split("-"))
        clim = da.where((t.year >= y0) & (t.year <= y1), drop=True).groupby("time.month").mean("time")
        label = f"monthly climatology {y0}-{y1}"
    else:
        months = [da.sel(time=(t.year == (1991 if m <= 4 else 1990)) & (t.month == m)).squeeze("time", drop=True)
                  for m in range(1, 13)]
        clim = xr.concat(months, dim="month").assign_coords(month=np.arange(1, 13))
        label = "JRA55-do RYF 1990/91 months (May-Dec 1990, Jan-Apr 1991)"
    v = np.asarray(clim.transpose("month", lat, lon).values, float)
    units = str(da.attrs.get("units", ""))
    if units.strip() == "%" or np.nanmax(v) > 1.5:
        v = v / 100.0
    v = np.where(np.isfinite(v), np.clip(v, 0.0, 1.0), np.nan)
    la = np.asarray(clim[lat].values, float)
    lo = np.mod(np.asarray(clim[lon].values, float), 360.0)
    oi, oj = np.argsort(la), np.argsort(lo)
    la, lo, v = la[oi], lo[oj], v[:, oi][:, :, oj]
    out = xr.Dataset({"siconc": (("month", "lat", "lon"), v.astype(np.float32))},
                     coords={"month": np.arange(1, 13), "lat": la, "lon": lo})
    out["siconc"].attrs = {"units": "1", "long_name": "sea-ice area fraction"}
    out.attrs.update(source=a.hadisst, description=f"HadISST sea-ice concentration, {label} (build_seaice_clim.py)")
    out.to_netcdf(a.out)
    R = 6.371e6
    dla, dlo = np.deg2rad(np.median(np.diff(la))), np.deg2rad(np.median(np.diff(lo)))
    area = (R ** 2 * dla * dlo * np.cos(np.deg2rad(la)))[:, None] * np.ones((1, lo.size))
    ext = lambda f, m: np.nansum(area * (f > 0.15) * m) / 1e12  # noqa: E731
    S, N = (la < 0)[:, None], (la > 0)[:, None]
    print(f"wrote {a.out}: {label}; units in file {units!r}; shape {v.shape}")
    print("  sea-ice extent (>15%) [10^6 km2]   Feb   Mar   Sep")
    print(f"    Arctic                         {ext(v[1], N):5.1f} {ext(v[2], N):5.1f} {ext(v[8], N):5.1f}   (obs ~15 / ~15.5 / ~6-7)")
    print(f"    Antarctic                      {ext(v[1], S):5.1f} {ext(v[2], S):5.1f} {ext(v[8], S):5.1f}   (obs ~3 / ~4 / ~18-19)")


if __name__ == "__main__":
    main()
