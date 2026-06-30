"""Augment an existing ERA5 climatology npz with soil-moisture targets.

The training npz (fetch_era5_hourly_climatology.py / the legacy 4-synoptic-hour
fetch) carries skin_temperature + forecast_albedo but not soil moisture.  This adds
the ERA5 volumetric soil water of layer 1 (0-7cm) and layer 2 (7-28cm) on the SAME
grid + month/hour sampling, so the multilayer-land calibrator can score root-zone
soil moisture alongside temperature and albedo.  Idempotent: skips if already present.

Run: PYTHONPATH=. .venv/bin/python scripts/data/augment_era5_soil_moisture.py \
        --npz /tmp/era5_diurnal.npz --years 2019 2020 --days 6 16 26
"""
import argparse
import numpy as np
import xarray as xr
import gcsfs
import pandas as pd

_SM = ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"]
_ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default="/tmp/era5_diurnal.npz")
    p.add_argument("--out", default=None, help="default: overwrite --npz in place")
    p.add_argument("--years", type=int, nargs="+", default=[2019, 2020])
    p.add_argument("--days", type=int, nargs="+", default=[6, 16, 26])
    args = p.parse_args()
    out_path = args.out or args.npz

    d = dict(np.load(args.npz))
    if all(k in d for k in ("swvl1", "swvl2")):
        print(f"# {args.npz} already has swvl1/swvl2 — nothing to do")
        return
    lat = d["lat"]; lon = d["lon"]; hours = [int(h) for h in d["hours"]]
    nlat, nlon, nh = lat.size, lon.size, len(hours)

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(_ARCO), chunks=None)
    # Match the npz grid by VALUE (nearest 0.25deg cell) so any stride/order works.
    sub = ds[_SM].sel(latitude=lat, longitude=(lon % 360), method="nearest")
    # Guard: the nearest-cell snap must be within half an ERA5 cell (0.125deg) of the
    # requested grid, else the npz grid is NOT the ERA5 0.25deg grid and the targets
    # would be silently mis-located.
    dlat = float(np.max(np.abs(np.asarray(sub.latitude.values) - lat)))
    dlon = float(np.max(np.abs(((np.asarray(sub.longitude.values) - (lon % 360) + 180) % 360) - 180)))
    if max(dlat, dlon) > 0.13:
        raise SystemExit(f"npz grid is not the ERA5 0.25deg grid (max snap "
                         f"dlat={dlat:.3f} dlon={dlon:.3f} deg) — refusing to mis-locate SM")
    times = [pd.Timestamp(y, m, dd, h) for y in args.years for m in range(1, 13)
             for dd in args.days for h in hours]
    print(f"# fetching {len(times)} steps of soil moisture -> (12,{nh},{nlat},{nlon})")
    sel = sub.sel(time=times).load()

    ny, nd = len(args.years), len(args.days)
    for short, v in zip(("swvl1", "swvl2"), _SM):
        a = np.asarray(sel[v].values).reshape(ny, 12, nd, nh, nlat, nlon)
        d[short] = a.mean(axis=(0, 2)).astype(np.float32)        # (12, nh, nlat, nlon)
    np.savez(out_path, **d)
    land = (d["lsm"] > 0.5)
    sm = d["swvl1"].mean((0, 1))[land]
    print(f"# saved {out_path}: swvl1 land-mean={float(sm.mean()):.3f} m3/m3 "
          f"[{float(sm.min()):.2f}, {float(sm.max()):.2f}]")


if __name__ == "__main__":
    main()
