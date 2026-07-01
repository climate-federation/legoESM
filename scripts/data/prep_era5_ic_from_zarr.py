"""Cache a single ERA5 initial-condition snapshot from a (cloud) ERA5 Zarr LOCALLY.

``run_amip.py --ic era5 --ic-path <store>`` reads the IC via
:func:`legoesm.training.era5_to_state.load_era5_ic`, which accepts a ``gs://`` URI
and streams the snapshot at run time.  For repeated / offline / compute-node runs
it is better to DOWNLOAD the one snapshot once: this subsets a source ERA5 Zarr
(default the public anonymous WeatherBench2 ARCO-ERA5 — no account) to the nearest
time + the variables the AMIP IC needs, and writes a small LOCAL Zarr that
``load_era5_ic`` reads identically.

Unlike ``scripts/data/prep_era5_ic.py`` (Levante GRIB + CDO), this needs no local
archive and no CDO — only xarray/gcsfs against a Zarr store.

    python scripts/data/prep_era5_ic_from_zarr.py \\
        --year 1979 --month 1 --day 1 --hour 0 --out data/amip/era5_ic_1979-01-01.zarr
    # then:  run_amip.py ... --ic era5 --ic-path data/amip/era5_ic_1979-01-01.zarr

Required IC variables (canonical names; short ECMWF aliases resolved):
  pressure-level  temperature, u_component_of_wind, v_component_of_wind, specific_humidity
  surface         surface_pressure  (+ optional skin_temperature, geopotential_at_surface)
"""

from __future__ import annotations

import argparse

# Public WeatherBench2 ARCO-ERA5 (matches legoesm.ml.data.era5_loader.WB2_ERA5_ZARR);
# duplicated so this stays import-light (xarray/gcsfs only).
WB2_ERA5_ZARR_DEFAULT = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

# Canonical long name -> short ECMWF/GRIB alias (mirrors era5_to_state._ERA5_VAR_ALIASES);
# resolve_var tries both directions so either naming convention in the store works.
_ERA5_VAR_ALIASES = {
    "temperature": "t", "u_component_of_wind": "u", "v_component_of_wind": "v",
    "specific_humidity": "q", "surface_pressure": "sp", "skin_temperature": "skt",
    "geopotential_at_surface": "z_sfc",
}
_REQUIRED_VARS = ("temperature", "u_component_of_wind",
                  "v_component_of_wind", "specific_humidity", "surface_pressure")
_OPTIONAL_VARS = ("skin_temperature", "geopotential_at_surface")


def _resolve_var(ds, name: str):
    """Store key matching ``name`` or an equivalent alias, or ``None`` (mirrors resolve_var)."""
    if name in ds:
        return name
    short = _ERA5_VAR_ALIASES.get(name)
    cands = [short] if short else []
    cands += [long for long, s in _ERA5_VAR_ALIASES.items() if s == name]
    for c in cands:
        if c in ds:
            return c
    return None


def _nearest_time_index(times, year: int, month: int, day: int, hour: int) -> int:
    import numpy as np
    import pandas as pd
    try:
        target = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        return int(np.argmin(np.abs(pd.DatetimeIndex(times) - target)))
    except Exception:
        import cftime
        tgt = cftime.datetime(year, month, day, hour)
        return int(np.argmin([abs((t - tgt).total_seconds()) for t in times]))


def subset_era5_ic_snapshot(ds, *, year: int, month: int, day: int, hour: int = 0,
                            require_all: bool = True):
    """Select the nearest-time single IC snapshot + the AMIP IC variables.

    PURE (no I/O): returns a Dataset with a length-1 ``time`` dim and the required
    (+ available optional) variables resolved by alias.  Raises if a REQUIRED
    variable is absent (``require_all``) so a wrong store fails loud, not silently.
    """
    if "time" not in ds.dims and "time" not in ds.coords:
        raise KeyError("source store has no 'time' coordinate")
    idx = _nearest_time_index(ds["time"].values, year, month, day, hour)

    keep, missing = [], []
    for name in _REQUIRED_VARS:
        key = _resolve_var(ds, name)
        (keep.append(key) if key else missing.append(name))
    if missing and require_all:
        raise KeyError(
            f"source store missing required IC variables {missing}; "
            f"available: {sorted(ds.data_vars)}")
    for name in _OPTIONAL_VARS:
        key = _resolve_var(ds, name)
        if key:
            keep.append(key)

    # keepdims on time (isel with a list) so load_era5_ic's nearest-time select still works.
    return ds[keep].isel(time=[idx])


def open_era5_store(zarr_path: str):
    """Open a (gs:// anon | local) ERA5 Zarr."""
    import xarray as xr
    if zarr_path.startswith("gs://"):
        return xr.open_zarr(zarr_path, storage_options={"token": "anon"},
                            chunks=None, decode_timedelta=False)
    return xr.open_zarr(zarr_path, chunks=None, decode_timedelta=False)


def build_era5_ic(zarr_path: str, out_path: str, *, year: int, month: int,
                  day: int, hour: int = 0) -> str:
    """Open ``zarr_path``, subset the IC snapshot, write a LOCAL Zarr.  Returns ``out_path``."""
    ds = open_era5_store(zarr_path)
    try:
        snap = subset_era5_ic_snapshot(ds, year=year, month=month, day=day, hour=hour)
        snap = snap.load()                       # materialise the (small) snapshot before writing
    finally:
        ds.close()
    # Drop the SOURCE encoding (the cloud store's zarr-v2 Blosc compressor/filters)
    # so the write uses the local zarr writer's own defaults — a source codec the
    # installed zarr cannot re-emit otherwise raises "Expected a BytesBytesCodec".
    for var in snap.variables:
        snap[var].encoding = {}
    snap.to_zarr(out_path, mode="w", consolidated=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output LOCAL Zarr path for the IC snapshot")
    p.add_argument("--zarr", default=WB2_ERA5_ZARR_DEFAULT,
                   help="source ERA5 Zarr (default: public WeatherBench2 ARCO-ERA5)")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, default=1)
    p.add_argument("--day", type=int, default=1)
    p.add_argument("--hour", type=int, default=0)
    args = p.parse_args(argv)

    out_path = build_era5_ic(args.zarr, args.out, year=args.year, month=args.month,
                             day=args.day, hour=args.hour)

    import xarray as xr
    with xr.open_zarr(out_path) as chk:
        nvar = len(chk.data_vars)
        t = chk["time"].values
        print(f"[era5-ic] wrote {nvar}-variable IC snapshot @ {t[0]} "
              f"({dict(chk.sizes)}) to {out_path} from {args.zarr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
