"""Add the stratospheric ERA5 levels (1..30 hPa) to a local 13-level IC snapshot.

The WB2 13-level IC (``prep_era5_ic_from_zarr.py``) stops at 50 hPa, and the
vertical interpolation holds the 50 hPa value above it, so a model whose top
layers sit at 3-8 hPa (CAM L32) starts ~30 K too cold there.  This takes the
SAME ERA5 analysis (same timestamp) on its 37 levels from the public anonymous
ARCO-ERA5 store, keeps every level above the base's top, regrids it in
latitude (linear) onto the base grid (e.g. the 720-row Gaussian proxy) and
writes base + upper levels as a new local Zarr.  The base levels are copied
unchanged.

    python scripts/data/add_era5_upper_levels.py BASE.zarr OUT.zarr
"""

from __future__ import annotations

import argparse

import numpy as np

ARCO_ERA5_37 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LEVEL_VARS = ("temperature", "u_component_of_wind", "v_component_of_wind",
              "specific_humidity")


def main(argv=None) -> int:
    import xarray as xr
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("out")
    p.add_argument("--src", default=ARCO_ERA5_37)
    a = p.parse_args(argv)

    base = xr.open_zarr(a.base, chunks=None)
    src = xr.open_zarr(a.src, storage_options={"token": "anon"}, chunks=None)
    t = base["time"].values
    lev = src["level"].values
    up = src[list(LEVEL_VARS)].sel(time=t, level=lev[lev < base["level"].values.min()])
    up = up.rename({"latitude": "lat", "longitude": "lon"}).load()
    assert np.array_equal(up["time"].values, t), "source has no exact IC timestamp"
    assert np.allclose(up["lon"].values, base["lon"].values), "longitudes differ"
    up = up.sortby("lat").interp(lat=base["lat"].values).assign_coords(lon=base["lon"])
    assert all(np.isfinite(up[v].values).all() for v in LEVEL_VARS)

    base = base.load()
    out = base.drop_vars(list(LEVEL_VARS) + ["level"])
    lvl = np.concatenate([up["level"].values, base["level"].values]).astype(
        base["level"].dtype)
    for v in LEVEL_VARS:
        out[v] = xr.concat([up[v].astype(base[v].dtype), base[v]], "level").assign_coords(
            level=lvl).transpose(*base[v].dims)
        out[v].attrs = base[v].attrs
    out.attrs["upper_levels_source"] = (
        f"{a.src} levels {up['level'].values.tolist()} hPa, same time, "
        f"lat-linear onto {a.base}")
    for v in out.variables:
        out[v].encoding = {}
    out.to_zarr(a.out, mode="w", consolidated=True)
    print(f"[upper-ic] levels {lvl.tolist()} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
