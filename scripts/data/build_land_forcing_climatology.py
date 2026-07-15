#!/usr/bin/env python
"""Build a real 2-D ERA5 monthly land-forcing climatology NetCDF.

Science-grade replacement for the ZONAL ``--climate-from-latitude`` DEMONSTRATION
of ``scripts/data/build_global_carbon_ic.py``: that path fabricates the monthly
T / precip / SW / net-radiation from each cell's LATITUDE only.  This script
assembles the FOUR fields from a real ARCO-ERA5 monthly-mean climatology so the
global-carbon-IC driver's ``--climatology`` path reads an observed 2-D climate.

Output contract (what ``build_global_carbon_ic.py --climatology`` reads via
``_load_monthly_climatology``)
------------------------------------------------------------------------------
A NetCDF with dims ``(time=12, lat, lon)``, 1-D degree coords ``lat`` / ``lon``,
and the four monthly-mean variables the driver's DEFAULT ``--clim-*-var`` flags
name -- so NO extra flags are needed:

    tas    [K]          near-surface (2 m) air temperature   (--clim-t-var)
    pr     [kg/m^2/s]    precipitation rate                    (--clim-precip-var)
    rsds   [W/m^2]       surface downward shortwave            (--clim-sw-var)
    netrad [W/m^2]       surface NET radiation (down-positive) (--clim-netrad-var)

``netrad`` is ERA5's own surface net SW + net LW (``ssr + str``), NOT an
albedo approximation -- the driver's Task-6 review flagged the old
``--clim-netrad-var netrad`` default as having no source; this file IS that
source.  The driver conservatively regrids this file onto the cover grid
(``conservative_regrid_latlon``), so any regular lat/lon grid (ERA5's native
descending latitude included) is accepted.

Data source + access pattern
----------------------------
Streams the public ARCO-ERA5 store the sibling
``scripts/data/fetch_era5_hourly_climatology.py`` uses
(``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`` via
``gcsfs`` anon + ``xr.open_zarr`` + ``.isel`` stride + month-of-year averaging),
mirrored here with the SAME ERA5 accumulation -> flux/rate conversions
(J/m^2 over the hour -> /3600 = W/m^2; total_precipitation m -> *rho_water/3600 =
kg/m^2/s).  The 3-line ARCO open is mirrored (not factored) because the three
existing ``scripts/data`` ARCO fetchers each inline it and their per-script
reductions differ; see that sibling.

The pure ``reduce_arco_to_climatology`` (unit conversions + ``netrad = ssr+str``
+ Dataset assembly) is separated from the networked ``_fetch_arco`` GCS wrapper
so the science logic is unit-tested offline
(``tests/land/unit/test_build_land_forcing_climatology.py``).

Login-node policy: the ARCO fetch is a networked, multi-GB compute-node job --
run it via ``sbatch``/``srun``, never on the login node.

Run (compute node):
    PYTHONPATH=$(ls -d packages/*/ | sed 's:/$::' | tr '\n' ':') \
    python scripts/data/build_land_forcing_climatology.py \
        --out results/land_forcing/era5_land_forcing_clim_2deg.nc \
        --stride 8 --years 2015 2016 2017 2018 2019 2020
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import xarray as xr

from legoesm import constants

# --- ARCO-ERA5 public store + native 0.25deg grid geometry (same store the
#     sibling scripts/data/fetch_era5_hourly_climatology.py streams) ---
_ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_ERA5_NLAT = 721   # native latitude count  (90 .. -90 by 0.25deg, descending)
_ERA5_NLON = 1440  # native longitude count (0 .. 359.75 by 0.25deg)

# --- ERA5 accumulation -> flux/rate conversion (exact time unit) ---
# ERA5 surface radiation / precip fields are HOURLY ACCUMULATIONS (radiation in
# J/m^2, total_precipitation in m of water over the hour); dividing by the hour
# gives an instantaneous-equivalent flux/rate.  Same convention the sibling
# fetch_era5_hourly_climatology.py uses (`/3600`, `*1000/3600`).
_SECONDS_PER_HOUR = 3600.0

# --- pole-row guard (numerical) ---
# ERA5's native grid has cells CENTERED exactly on the poles (lat[0]=+90,
# lat[-1]=-90).  The driver's conservative_regrid_latlon measures latitude
# overlap in sin(lat); a pole-centered cell's edges straddle +-90 symmetrically
# (sin(90+d)=sin(90-d)=cos(d)), so its sin-lat extent is ~0 -> the regrid returns
# NaN there and the driver silently fills it with the field mean.  Dropping the
# two pole half-rows (no vegetated land at either pole) yields a clean regular
# grid every retained cell of which has positive area.  Degrees, numerical guard.
_POLE_EPS_DEG = 1e-6

# --- ARCO-ERA5 variable names (VERIFIED against the zarr schema 2026-07-07) ---
# Net-radiation uses ERA5's OWN net components (positive DOWNWARD, into the
# surface): ssr = net SW absorbed (>=0), str = net LW (usually <0 as the surface
# emits more than it receives); their sum is the down-positive surface net
# radiation Rn -- exactly what the downstream Priestley-Taylor PET in
# legoesm.land.carbon.climate_features.reduce_climatology_to_features consumes.
_VAR_T2M = "2m_temperature"                      # [K]      2 m air temperature
_VAR_TP = "total_precipitation"                  # [m]      accumulated / hour
_VAR_SSR = "surface_net_solar_radiation"         # [J/m^2]  net SW  (down +)
_VAR_STR = "surface_net_thermal_radiation"       # [J/m^2]  net LW  (down +, <0)
_VAR_SSRD = "surface_solar_radiation_downwards"  # [J/m^2]  down SW (down +)
_FETCH_VARS = (_VAR_T2M, _VAR_TP, _VAR_SSR, _VAR_STR, _VAR_SSRD)

# --- output variable names (== build_global_carbon_ic.py --clim-*-var defaults) ---
_OUT_TAS = "tas"
_OUT_PR = "pr"
_OUT_RSDS = "rsds"
_OUT_NETRAD = "netrad"
_OUT_DIMS = ("time", "lat", "lon")


def reduce_arco_to_climatology(raw, lat, lon) -> xr.Dataset:
    """Pure: monthly-mean ARCO-ERA5 raw arrays -> the land-forcing climatology.

    ``raw`` maps each fetched ARCO variable NAME (:data:`_FETCH_VARS`) to its
    ``(12, nlat, nlon)`` monthly-mean array in NATIVE ERA5 units:
    ``2m_temperature`` [K]; ``total_precipitation`` [m] per hourly accumulation;
    the three radiation fields [J/m^2] per hourly accumulation.  Applies the
    ERA5 accumulation -> flux/rate conversions and the net-radiation sum, and
    assembles the ``(time=12, lat, lon)`` :class:`xarray.Dataset` the
    global-carbon-IC driver's ``--climatology`` path reads (vars
    ``tas``/``pr``/``rsds``/``netrad``; coords ``lat``/``lon`` [deg]).

    No network / no I/O -> fully unit-testable on synthetic arrays.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(f"lat/lon must be 1-D; got lat.ndim={lat.ndim}, "
                         f"lon.ndim={lon.ndim}.")
    n_lat, n_lon = lat.size, lon.size
    expected = (12, n_lat, n_lon)

    def _get(name):
        if name not in raw:
            raise KeyError(
                f"raw is missing ARCO variable {name!r}; need all of {_FETCH_VARS}.")
        arr = np.asarray(raw[name], dtype=np.float64)
        if arr.shape != expected:
            raise ValueError(
                f"raw[{name!r}] shape {arr.shape} != expected {expected} "
                f"(12 months, {n_lat} lat, {n_lon} lon).")
        return arr

    t2m = _get(_VAR_T2M)
    tp = _get(_VAR_TP)
    ssr = _get(_VAR_SSR)
    str_ = _get(_VAR_STR)
    ssrd = _get(_VAR_SSRD)

    # --- unit conversions (ERA5 hourly accumulation -> flux/rate) ---
    # positive-downward convention throughout (ERA5 surface flux sign):
    tas = t2m                                              # [K] passthrough
    pr = tp * constants.rho_water / _SECONDS_PER_HOUR      # m/hr -> kg/m^2/s
    rsds = ssrd / _SECONDS_PER_HOUR                        # J/m^2/hr -> W/m^2 (down)
    netrad = (ssr + str_) / _SECONDS_PER_HOUR             # net SW + net LW -> W/m^2 (Rn, down)

    months = np.arange(1, 13, dtype=np.int32)
    ds = xr.Dataset(
        data_vars={
            _OUT_TAS: (_OUT_DIMS, tas),
            _OUT_PR: (_OUT_DIMS, pr),
            _OUT_RSDS: (_OUT_DIMS, rsds),
            _OUT_NETRAD: (_OUT_DIMS, netrad),
        },
        coords={
            "time": ("time", months),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    ds[_OUT_TAS].attrs.update(units="K", long_name="near-surface (2 m) air temperature")
    ds[_OUT_PR].attrs.update(units="kg m-2 s-1", long_name="precipitation rate")
    ds[_OUT_RSDS].attrs.update(units="W m-2",
                               long_name="surface downwelling shortwave flux")
    ds[_OUT_NETRAD].attrs.update(
        units="W m-2",
        long_name="surface net radiation (down-positive: net SW + net LW)")
    ds["lat"].attrs.update(units="degrees_north", long_name="latitude")
    ds["lon"].attrs.update(units="degrees_east", long_name="longitude")
    ds["time"].attrs.update(long_name="month of year (1-12 climatology)")
    return ds


def _reduce_samples_to_months(arr, ny, nd, nh, nlat, nlon) -> np.ndarray:
    """Pure: sample stack -> monthly mean over years/days/hours.

    ``arr`` has a leading time axis of length ``ny*12*nd*nh`` in the fetch loop
    order ``(year, month, day, hour)`` and trailing ``(nlat, nlon)``.  Reshapes
    to ``(ny, 12, nd, nh, nlat, nlon)`` and averages over the year/day/hour axes,
    leaving the ``(12, nlat, nlon)`` monthly-mean climatology (the month axis is
    preserved).
    """
    a = np.asarray(arr, dtype=np.float64).reshape(ny, 12, nd, nh, nlat, nlon)
    return a.mean(axis=(0, 2, 3))                          # (12, nlat, nlon)


def _drop_pole_rows(raw, lat):
    """Pure: drop latitude rows centred at ``|lat| >= 90 - eps``.

    Those pole-centered ERA5 cells have ~zero sin-lat area under the driver's
    ``conservative_regrid_latlon`` (see :data:`_POLE_EPS_DEG`), so keeping them
    would seed un-regriddable (NaN-then-mean-filled) cells in the output.  Trims
    the ``{name: (12, nlat, nlon)}`` dict and the ``lat`` vector consistently;
    a no-op when no row sits on a pole.
    """
    lat = np.asarray(lat, dtype=np.float64)
    keep = np.abs(lat) < (90.0 - _POLE_EPS_DEG)
    if keep.all():
        return raw, lat
    return ({name: np.asarray(v)[:, keep, :] for name, v in raw.items()},
            lat[keep])


def _fetch_arco(years, days, hours, stride):
    """Fetch the ARCO-ERA5 monthly-mean climatology of :data:`_FETCH_VARS`.

    Thin networked GCS wrapper (mirrors the sibling
    ``fetch_era5_hourly_climatology.py`` open + stride + month-of-year average).
    Averages over ``years`` x ``days`` x ``hours`` so the monthly mean resolves
    the full diurnal cycle of the radiation accumulations, then drops the exact
    pole rows (:func:`_drop_pole_rows`).  NOT exercised by the offline unit tests;
    its two non-trivial reductions are the tested pure helpers above.

    Returns ``(raw, lat, lon)`` where ``raw[name]`` is ``(12, nlat, nlon)`` in
    NATIVE ERA5 units and ``lat``/``lon`` are the (pole-trimmed) strided native
    degree coords.
    """
    import gcsfs
    import pandas as pd

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(_ARCO), chunks=None)
    sub = ds.isel(latitude=slice(0, _ERA5_NLAT, stride),
                  longitude=slice(0, _ERA5_NLON, stride))
    nlat, nlon = sub.sizes["latitude"], sub.sizes["longitude"]
    ny, nd, nh = len(years), len(days), len(hours)

    times = [pd.Timestamp(y, m, d, h)
             for y in years for m in range(1, 13) for d in days for h in hours]
    print(f"# fetching {len(times)} ARCO-ERA5 steps "
          f"({ny}y x 12m x {nd}d x {nh}h) -> (12,{nlat},{nlon}) monthly climatology")
    sel = sub[list(_FETCH_VARS)].sel(time=times).load()

    raw = {v: _reduce_samples_to_months(sel[v].values, ny, nd, nh, nlat, nlon)
           for v in _FETCH_VARS}
    lat = np.asarray(sub.latitude.values, dtype=np.float64)
    lon = np.asarray(sub.longitude.values, dtype=np.float64)
    raw, lat = _drop_pole_rows(raw, lat)
    if lat.size != nlat:
        print(f"# dropped {nlat - lat.size} pole row(s) -> "
              f"(12,{lat.size},{lon.size}) clean regular grid")
    return raw, lat, lon


def _print_sanity(ds: xr.Dataset) -> None:
    """Print min/mean/max of each output variable (finite cells)."""
    for v, unit in ((_OUT_TAS, "K"), (_OUT_PR, "kg/m2/s"),
                    (_OUT_RSDS, "W/m2"), (_OUT_NETRAD, "W/m2")):
        a = np.asarray(ds[v].values, dtype=np.float64)
        print(f"# {v:7s} [{unit:8s}] "
              f"min {np.nanmin(a):12.4g}  mean {np.nanmean(a):12.4g}  "
              f"max {np.nanmax(a):12.4g}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=str,
                   default="results/land_forcing/era5_land_forcing_clim_2deg.nc",
                   help="output NetCDF path (dims time=12,lat,lon; vars "
                        "tas/pr/rsds/netrad) directly loadable by "
                        "build_global_carbon_ic.py --climatology")
    p.add_argument("--stride", type=int, default=8,
                   help="0.25deg subsampling stride (8 ~= 2deg)")
    p.add_argument("--years", type=int, nargs="+",
                   default=[2015, 2016, 2017, 2018, 2019, 2020],
                   help="years to average into the monthly climatology")
    p.add_argument("--days", type=int, nargs="+", default=[5, 15, 25],
                   help="days-of-month sampled per month (all months have these)")
    p.add_argument("--hours", type=int, nargs="+", default=list(range(24)),
                   help="hours-of-day sampled (24 h resolves the diurnal cycle)")
    return p


def main(argv=None):
    """Fetch the ARCO-ERA5 climatology, assemble, sanity-print, and write NetCDF."""
    args = build_arg_parser().parse_args(argv)

    raw, lat, lon = _fetch_arco(args.years, args.days, args.hours, args.stride)
    ds = reduce_arco_to_climatology(raw, lat, lon)
    ds.attrs.update(
        title="ERA5 monthly land-forcing climatology (tas/pr/rsds/netrad)",
        source=_ARCO,
        history=(f"built by scripts/data/build_land_forcing_climatology.py; "
                 f"years={list(args.years)}, days={list(args.days)}, "
                 f"nhours={len(args.hours)}, stride={args.stride}"),
    )
    _print_sanity(ds)

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ds.to_netcdf(out)
    print(f"# wrote {out}  (dims {dict(ds.sizes)})")
    return out


if __name__ == "__main__":
    main()
