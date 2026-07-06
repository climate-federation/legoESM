"""Fetch an ERA5 monthly-mean HOURLY-resolved surface climatology (12 x 24 h).

Extends the 4-synoptic-hour climatology to the FULL 24-hour diurnal cycle so an
offline land run resolves the diurnal cycle finely enough to remove the Jensen
sigma*T^4 warm bias (4 hours over-weight the noon SW peak; 24 hours do not).
Streams coarse (~2deg) ARCO-ERA5 surface fields + forecast_albedo + skin_temperature
(validation target).  Heavier than the 4-hour fetch (one-time data-prep job).

Run: PYTHONPATH=. .venv/bin/python scripts/data/fetch_era5_hourly_climatology.py \
        --out /tmp/era5_hourly.npz --years 2019 2020 --days 6 16 26
"""
import argparse
import numpy as np
import xarray as xr
import gcsfs
import pandas as pd

_FVARS = ["2m_temperature", "2m_dewpoint_temperature",
          "surface_solar_radiation_downwards", "surface_thermal_radiation_downwards",
          "total_precipitation", "surface_pressure",
          "10m_u_component_of_wind", "10m_v_component_of_wind",
          "skin_temperature", "forecast_albedo",
          # soil-moisture validation targets [m3/m3]: layer 1 (0-7cm) + layer 2
          # (7-28cm) span the offline-evolving root zone (the deep layers are pinned).
          "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"]
_ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/tmp/era5_hourly.npz")
    p.add_argument("--stride", type=int, default=8, help="0.25deg stride (~2deg=8)")
    p.add_argument("--years", type=int, nargs="+", default=[2019, 2020])
    p.add_argument("--days", type=int, nargs="+", default=[6, 16, 26])
    p.add_argument("--hours", type=int, nargs="+", default=list(range(24)))
    args = p.parse_args()

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(_ARCO), chunks=None)
    sub = ds.isel(latitude=slice(0, 721, args.stride), longitude=slice(0, 1440, args.stride))
    nlat, nlon = sub.sizes["latitude"], sub.sizes["longitude"]
    nh = len(args.hours)
    times = [pd.Timestamp(y, m, d, h) for y in args.years for m in range(1, 13)
             for d in args.days for h in args.hours]
    print(f"# {len(times)} timesteps -> (12,{nh},{nlat},{nlon}) hourly climatology")
    sel = sub[_FVARS].sel(time=times).load()

    ny, nd = len(args.years), len(args.days)
    out = {}
    for v in _FVARS:
        a = np.asarray(sel[v].values).reshape(ny, 12, nd, nh, nlat, nlon)
        out[v] = a.mean(axis=(0, 2))                # (12, nh, nlat, nlon)
    out["ssrd_wm2"] = out.pop("surface_solar_radiation_downwards") / 3600.0
    out["strd_wm2"] = out.pop("surface_thermal_radiation_downwards") / 3600.0
    out["precip_kgms"] = out.pop("total_precipitation") * 1000.0 / 3600.0
    st = sub[["geopotential_at_surface", "land_sea_mask"]].sel(
        time=pd.Timestamp(args.years[-1], 1, 1, 0)).load()
    out["elev_m"] = np.asarray(st["geopotential_at_surface"].values) / 9.80665
    out["lsm"] = np.asarray(st["land_sea_mask"].values)
    out["lat"] = np.asarray(sub.latitude.values); out["lon"] = np.asarray(sub.longitude.values)
    out["hours"] = np.asarray(args.hours, dtype=float)
    np.savez(args.out, **out)
    print(f"# saved {args.out}  ({nh} h, land {(out['lsm']>0.5).sum()})")


if __name__ == "__main__":
    main()
