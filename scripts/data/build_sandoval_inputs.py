"""Regular lat-lon Sandoval et al. (2026) phi0 inputs from a JULES 1-D land file.

The Sandoval quantum-yield form needs two climatological fields per column:
the aridity index (AI = PET/P, dimensionless; sets the phi0 peak) and mGDD0
(mean growing-degree temperature above 0 degC [degC]; sets the thermal
breadth).  JULES reads them from a 1-D land-point ancillary
(``land``-dimensioned ``latitude``, ``longitude``, ``aridity_index``,
``mgdd0``).  This script scatters those points onto the regular lat-lon grid
they were sampled from (NaN off land), the format
``run_lmip_biophys.sandoval_inputs_on_grid`` regrids to the model grid.

The source grid spacing is given explicitly (``--res-deg``); every point must
sit on a cell centre of that global grid, or the script refuses.

Usage::

    python scripts/data/build_sandoval_inputs.py SRC_1D.nc OUT_2D.nc --res-deg 0.5
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

_VARS = ("aridity_index", "mgdd0")


def scatter_land_points(lat_pts, lon_pts, values: dict, *, res_deg: float,
                        tol: float = 1e-4):
    """Scatter 1-D land-point ``values`` onto the global ``res_deg`` regular
    grid whose cell centres they sit on.  Returns ``(lat, lon, {name: (nlat, nlon)})``, NaN off land.
    Raises if the points are not on one regular grid or a cell is repeated."""
    lat_pts = np.asarray(lat_pts, np.float64).ravel()
    lon_pts = np.asarray(lon_pts, np.float64).ravel()
    lon_pts = np.where(lon_pts >= 180.0, lon_pts - 360.0, lon_pts)
    dlat = dlon = float(res_deg)
    lat = np.arange(-90.0 + dlat / 2, 90.0, dlat)
    lon = np.arange(-180.0 + dlon / 2, 180.0, dlon)
    i = np.rint((lat_pts - lat[0]) / dlat).astype(int)
    j = np.rint((lon_pts - lon[0]) / dlon).astype(int)
    if (np.any(np.abs(lat[i] - lat_pts) > tol)
            or np.any(np.abs(lon[j] - lon_pts) > tol)):
        raise ValueError("land points are not on cell centres of one regular grid")
    flat = i * lon.size + j
    if np.unique(flat).size != flat.size:
        raise ValueError("two land points fall in the same grid cell")
    out = {}
    for name, v in values.items():
        g = np.full(lat.size * lon.size, np.nan)
        g[flat] = np.asarray(v, np.float64).ravel()
        out[name] = g.reshape(lat.size, lon.size)
    return lat, lon, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="JULES 1-D land file (latitude, longitude, "
                                "aridity_index, mgdd0 on 'land')")
    ap.add_argument("out", help="output regular lat-lon NetCDF")
    ap.add_argument("--res-deg", type=float, required=True,
                    help="source grid spacing [deg] (WFDEI: 0.5)")
    a = ap.parse_args(argv)
    with xr.open_dataset(a.src) as ds:
        lat, lon, grids = scatter_land_points(
            ds["latitude"].values, ds["longitude"].values,
            {v: ds[v].values for v in _VARS}, res_deg=a.res_deg)
        src_attrs = dict(ds.attrs)
    out = xr.Dataset(
        {"aridity_index": (("lat", "lon"), grids["aridity_index"],
                           {"units": "1", "long_name": "aridity index PET/P"}),
         "mgdd0": (("lat", "lon"), grids["mgdd0"],
                   {"units": "degC",
                    "long_name": "mean growing-degree temperature above 0 degC"})},
        coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})},
        attrs={"title": "Sandoval et al. (2026) phi0 inputs, regular lat-lon",
               "source": a.src,
               "source_title": src_attrs.get("title", ""),
               "history": "scripts/data/build_sandoval_inputs.py"})
    out.to_netcdf(a.out)
    n = int(np.isfinite(grids["aridity_index"]).sum())
    print(f"wrote {a.out}: {lat.size}x{lon.size} grid, {n} land cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
