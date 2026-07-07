"""Resumable, per-month chunked ERA5 surface-climatology fetch from ARCO-ERA5.

Companion to ``fetch_era5_hourly_climatology.py``: that driver streams the whole
climatology in one ``.load()``, which is fine locally but overruns short foreground
time limits (e.g. a 10-min sandbox cap) on the larger 1deg / 24-hour grids — and a
backgrounded job may not have network.  This driver fetches **one month per partial
file**, skipping months already on disk, so the full climatology comes down across
several short, restartable calls; ``--merge`` then assembles the 12 monthly partials
into the standard ``fetch_era5_hourly_climatology`` output format WITHOUT any network
(the static geopotential/land-sea-mask + lat/lon/hours are stored in each partial).

Fetch (repeat until all 12 months on disk; --stride 4 ~= 1deg, 8 ~= 2deg):
    scripts/data/fetch_era5_chunked.py --stride 4 --years 2019 2020 --days 6 16 26 \
        --months 1 2 3 4 5 6 --dir /tmp/ck_1deg
    scripts/data/fetch_era5_chunked.py --stride 4 ... --months 7 8 9 10 11 12 --dir /tmp/ck_1deg
Merge (network-free):
    scripts/data/fetch_era5_chunked.py --merge --dir /tmp/ck_1deg --out /tmp/era5_1deg_24h.npz
"""
import argparse
import os

import numpy as np

_ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_FVARS = ["2m_temperature", "2m_dewpoint_temperature",
          "surface_solar_radiation_downwards", "surface_thermal_radiation_downwards",
          "total_precipitation", "surface_pressure",
          "10m_u_component_of_wind", "10m_v_component_of_wind",
          "skin_temperature", "forecast_albedo",
          "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"]
# renamed / unit-converted output var names.  NOTE: the soil-moisture layers use the
# SHORT names ``swvl1``/``swvl2`` that the model loader (`train_multilayer_land_era5._pack`,
# `g("swvl1")`) reads — the older ``fetch_era5_hourly_climatology`` emits the full ARCO
# names, which `_pack` silently NaN-drops, so its soil-moisture target never loaded; this
# driver renames them so the ``lam_sm`` term actually trains.
_OUT_VARS = ["2m_temperature", "2m_dewpoint_temperature", "ssrd_wm2", "strd_wm2",
             "precip_kgms", "surface_pressure", "10m_u_component_of_wind",
             "10m_v_component_of_wind", "skin_temperature", "forecast_albedo",
             "swvl1", "swvl2"]
_G0 = 9.80665     # standard gravity for geopotential -> geometric height [m/s2]


def _rename_units(out: dict) -> dict:
    """Match fetch_era5_hourly_climatology unit conversions (J/m2 accum -> W/m2,
    m -> kg/m2/s) + rename the soil-moisture layers to the ``swvl1``/``swvl2`` keys the
    model loader reads."""
    out["ssrd_wm2"] = out.pop("surface_solar_radiation_downwards") / 3600.0
    out["strd_wm2"] = out.pop("surface_thermal_radiation_downwards") / 3600.0
    out["precip_kgms"] = out.pop("total_precipitation") * 1000.0 / 3600.0
    out["swvl1"] = out.pop("volumetric_soil_water_layer_1")
    out["swvl2"] = out.pop("volumetric_soil_water_layer_2")
    return out


def merge_partials(dir_: str) -> dict:
    """Assemble 12 monthly partials (m01.npz..m12.npz) into the final climatology dict.

    Pure-local (no network): each partial carries its own lat/lon/hours + the static
    elev_m/lsm, so the merge just stacks the monthly (nh, nlat, nlon) fields on a new
    leading month axis and copies the static fields from month 01."""
    months = [np.load(f"{dir_}/m{m:02d}.npz") for m in range(1, 13)]
    out = {v: np.stack([m[v] for m in months]) for v in _OUT_VARS}   # (12, nh, nlat, nlon)
    first = months[0]
    for k in ("elev_m", "lsm", "lat", "lon", "hours"):
        out[k] = first[k]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stride", type=int, default=4, help="0.25deg stride (4~=1deg, 8~=2deg)")
    p.add_argument("--years", type=int, nargs="+", default=[2019, 2020])
    p.add_argument("--days", type=int, nargs="+", default=[6, 16, 26])
    p.add_argument("--hours", type=int, nargs="+", default=list(range(24)))
    p.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    p.add_argument("--dir", required=True, help="partial-file directory (per-month m##.npz)")
    p.add_argument("--merge", action="store_true", help="assemble partials -> --out (no network)")
    p.add_argument("--out", default="/tmp/era5_chunked.npz")
    a = p.parse_args()
    os.makedirs(a.dir, exist_ok=True)

    if a.merge:
        out = merge_partials(a.dir)
        np.savez(a.out, **out)
        print(f"# MERGED -> {a.out} ({out['hours'].size} h, land {(out['lsm'] > 0.5).sum()})",
              flush=True)
        return

    import gcsfs
    import xarray as xr
    import pandas as pd
    nh = len(a.hours)
    ny, nd = len(a.years), len(a.days)
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(_ARCO), chunks=None)
    sub = ds.isel(latitude=slice(0, 721, a.stride), longitude=slice(0, 1440, a.stride))
    nlat, nlon = sub.sizes["latitude"], sub.sizes["longitude"]
    st = sub[["geopotential_at_surface", "land_sea_mask"]].sel(
        time=pd.Timestamp(a.years[-1], 1, 1, 0)).load()
    static = dict(
        elev_m=np.asarray(st["geopotential_at_surface"].values) / _G0,
        lsm=np.asarray(st["land_sea_mask"].values),
        lat=np.asarray(sub.latitude.values), lon=np.asarray(sub.longitude.values),
        hours=np.asarray(a.hours, dtype=float))
    for m in a.months:
        fp = f"{a.dir}/m{m:02d}.npz"
        if os.path.exists(fp):
            print(f"# month {m:02d} already done, skip", flush=True)
            continue
        times = [pd.Timestamp(y, m, d, h)
                 for y in a.years for d in a.days for h in a.hours]
        sel = sub[_FVARS].sel(time=times).load()
        out = {}
        for v in _FVARS:
            arr = np.asarray(sel[v].values).reshape(ny, nd, nh, nlat, nlon)
            out[v] = arr.mean(axis=(0, 1))                 # (nh, nlat, nlon)
        out = _rename_units(out)
        out.update(static)
        np.savez(fp, **out)
        print(f"# saved month {m:02d} -> {fp}", flush=True)


if __name__ == "__main__":
    main()
