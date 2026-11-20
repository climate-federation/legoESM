"""Write a geography file (legoesm.grids.geography) for an idealized or the
present-day Earth.

Idealized shapes (Enderton & Marshall 2009, JAS 66, 1593: ocean flat 5.2 km,
barriers one model cell wide, pole to pole, land not protruding):
  aquaplanet  all ocean
  ridge       one barrier along --barrier-lon
  eqpas       ridge with a gap 20S-20N
  drake       ridge with a gap south of 40S
  box         a flat (elevation 0) land box --box LAT0 LAT1 LON0 LON1
present_day   --elevation-file (ETOPO) + --land-mask-file (sftlf) [+ --clm-surfdata]
              copied unchanged (land fraction on its own lat_lf/lon_lf), so the
              legacy regrid reproduces --topography + --land-mask-file bitwise.

    python scripts/data/make_geography.py ridge --barrier-lon 0 -o ridge.nc
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# --- Enderton & Marshall (2009) benchmark geometry ---
_OCEAN_DEPTH_M = 5200.0
_EQPAS_GAP_LAT = 20.0
_DRAKE_GAP_LAT = -40.0
# --- idealized source grid (VOTED V8) ---
_SRC_DEG = 0.25
_BARRIER_VERTEX_DEG = 1.0
# --- present-day elevation units accepted as metres (VOTED V15) ---
_METRE_UNITS = ("", "m", "meter", "meters", "metre", "metres")


def _source_grid():
    lat = np.arange(-90.0 + _SRC_DEG / 2, 90.0, _SRC_DEG)
    lon = np.arange(_SRC_DEG / 2, 360.0, _SRC_DEG)
    return lat, lon


def _meridian(lon_deg, lat0, lat1):
    n = int(round(abs(lat1 - lat0) / _BARRIER_VERTEX_DEG)) + 1
    return np.linspace(lat0, lat1, n), np.full(n, float(lon_deg))


def idealized(shape, *, barrier_lon=None, box=None):
    import xarray as xr
    from legoesm.grids.geography import GEOGRAPHY_VERSION
    lat, lon = _source_grid()
    elev = np.full((lat.size, lon.size), -_OCEAN_DEPTH_M)
    data = {}
    lines = {"aquaplanet": [], "box": [],
             "ridge": [(-90.0, 90.0)],
             "eqpas": [(-90.0, -_EQPAS_GAP_LAT), (_EQPAS_GAP_LAT, 90.0)],
             "drake": [(_DRAKE_GAP_LAT, 90.0)]}[shape]
    if lines:
        if barrier_lon is None:
            raise ValueError(f"{shape} needs --barrier-lon")
        segs = [_meridian(barrier_lon, a, b) for a, b in lines]
        nv = max(s[0].size for s in segs)
        bl = np.full((len(segs), nv), np.nan)
        bo = np.full((len(segs), nv), np.nan)
        for k, (la, lo) in enumerate(segs):
            bl[k, :la.size], bo[k, :lo.size] = la, lo
        data["barrier_lat"] = (("barrier", "vertex"), bl, {"units": "degrees_north"})
        data["barrier_lon"] = (("barrier", "vertex"), bo, {"units": "degrees_east"})
    if shape == "box":
        if box is None:
            raise ValueError("box needs --box LAT0 LAT1 LON0 LON1")
        la0, la1, lo0, lo1 = box
        if not (la0 < la1 and lo0 % 360.0 < lo1 % 360.0):
            raise ValueError("box needs LAT0 < LAT1 and LON0 < LON1 (mod 360, no seam wrap)")
        inside = ((lat[:, None] >= la0) & (lat[:, None] <= la1)
                  & (lon[None, :] >= lo0 % 360.0) & (lon[None, :] <= lo1 % 360.0))
        elev = np.where(inside, 0.0, elev)
        data["land_fraction"] = (("lat", "lon"), inside.astype(np.float64))
    data["elevation"] = (("lat", "lon"), elev, {"units": "m"})
    return xr.Dataset(data, coords={"lat": lat, "lon": lon},
                      attrs={"legoesm_geography_version": GEOGRAPHY_VERSION,
                             "shape": shape})


def present_day(elevation_file, land_mask_file, clm_surfdata=""):
    import xarray as xr
    from legoesm.grids.geography import GEOGRAPHY_VERSION
    from legoesm.grids.topography import detect_elevation_variables
    e = xr.open_dataset(elevation_file)
    ev, la, lo = detect_elevation_variables(e)
    elev = e[ev].rename({la: "lat", lo: "lon"}) if (la, lo) != ("lat", "lon") else e[ev]
    elev = elev.reset_coords(drop=True)
    units = str(elev.attrs.get("units", ""))
    if units not in _METRE_UNITS:
        raise ValueError(f"elevation {ev!r} in {elevation_file} has units {units!r}, "
                         f"not metres {_METRE_UNITS}")
    elev.attrs = {"units": "m"}
    m = xr.open_dataset(land_mask_file)
    names = set(m.data_vars) | set(m.coords)
    mv = next((c for c in ("sftlf", "lsm", "land_sea_mask", "land",
                           "land_area_fraction", "LSM") if c in names), list(m.data_vars)[0])
    mla = next(c for c in ("lat", "latitude", "y", "Y") if c in names)
    mlo = next(c for c in ("lon", "longitude", "x", "X") if c in names)
    f = m[mv]
    while f.ndim > 2:
        f = f.isel({f.dims[0]: 0})
    # the land-mask loader's own steps, in its order (NaN -> 0, then % -> fraction)
    x = np.where(np.isnan(f.values.astype(np.float64)), 0.0, f.values.astype(np.float64))
    if np.nanmax(x) > 1.5:
        x = x / 100.0
    lf = xr.DataArray(x, dims=("lat_lf", "lon_lf"),
                      coords={"lat_lf": m[mla].values, "lon_lf": m[mlo].values})
    ds = xr.Dataset({"elevation": elev, "land_fraction": lf},
                    attrs={"legoesm_geography_version": GEOGRAPHY_VERSION,
                           "shape": "present_day", "elevation_source": str(elevation_file),
                           "land_mask_source": str(land_mask_file)})
    if clm_surfdata:
        c = xr.open_dataset(clm_surfdata)
        clash = (set(c.variables) | set(c.dims)) & (set(ds.variables) | set(ds.dims))
        if clash:
            raise ValueError(f"CLM surfdata names clash with the geography: {sorted(clash)}")
        ds = ds.merge(c)
        ds.attrs["clm_surfdata_source"] = str(clm_surfdata)
    return ds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shape", choices=("aquaplanet", "ridge", "eqpas", "drake", "box",
                                      "present_day"))
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--barrier-lon", type=float, default=None)
    ap.add_argument("--box", type=float, nargs=4, default=None,
                    metavar=("LAT0", "LAT1", "LON0", "LON1"))
    ap.add_argument("--elevation-file", default="")
    ap.add_argument("--land-mask-file", default="")
    ap.add_argument("--clm-surfdata", default="")
    a = ap.parse_args(argv)
    if a.shape == "present_day":
        if not (a.elevation_file and a.land_mask_file):
            ap.error("present_day needs --elevation-file and --land-mask-file")
        ds = present_day(a.elevation_file, a.land_mask_file, a.clm_surfdata)
    else:
        ds = idealized(a.shape, barrier_lon=a.barrier_lon,
                       box=None if a.box is None else tuple(a.box))
    ds.to_netcdf(a.output)
    from legoesm.grids.geography import open_geography
    open_geography(a.output).close()
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
