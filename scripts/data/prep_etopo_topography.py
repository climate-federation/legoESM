"""Prepare a model-ready lat-lon TOPOGRAPHY NetCDF from an elevation dataset.

``run_amip.py --topography <file>`` (and the authoritative
``config/amip/amip_production.yaml``) needs a regular lat-lon NetCDF with an
elevation field; :func:`legoesm.grids.topography.load_real_topography` bilinearly
interpolates it to the model grid for ``phis`` AND derives the land fraction from
sub-grid sampling of the elevation (fraction of sub-grid points with elevation>0).
So the topography file is ALSO the land-sea source for a ``--topography`` run.

This regrids ANY regular lat-lon elevation dataset (NOAA ETOPO, GEBCO, GMTED2010,
ETOPO5, ...) onto a regular ``target_res_deg`` lat-lon grid and writes
``elevation(lat, lon)`` in the layout ``load_real_topography`` auto-detects
(``_detect_variables``: elevation name ``elevation``/``z``/``topo``..., coords
``lat``/``lon``).  The DOWNLOAD of the (large) source relief file is left to the
operator — point ``--input`` at it; NOAA ETOPO 2022 is public (no account):
https://www.ncei.noaa.gov/products/etopo-global-relief-model.  A coarser source
(ETOPO5 / a 1-deg relief grid) is plenty for a C48-class AMIP.

    python scripts/data/prep_etopo_topography.py --input ETOPO.nc \\
        --out data/amip/etopo_0p25deg.nc --resolution-deg 0.25
    # then point run_amip.py at it:  --topography data/amip/etopo_0p25deg.nc

Keep the target FINER than the model grid (default 0.25 deg) so the sub-grid land
fraction is smooth; ~1 deg matches Levante's ``etopo_1deg_clean.nc`` for parity.
"""

from __future__ import annotations

import argparse

# Elevation/lat/lon variable names load_real_topography._detect_variables accepts
# (write the output with names from these so it round-trips without --elev-var).
_ELEV_CANDIDATES = ("elevation", "altitude", "z", "topo", "Band1", "ROSE",
                    "bedrock_topography", "surface_elevation")
_LAT_CANDIDATES = ("lat", "latitude", "y", "Y")
_LON_CANDIDATES = ("lon", "longitude", "x", "X")


def _resolve(ds, candidates, *, kind: str, explicit: str = ""):
    """First of ``candidates`` (or ``explicit``) present in ``ds``; raise on a miss."""
    names = set(ds.variables)
    if explicit:
        if explicit not in names:
            raise KeyError(f"requested {kind} {explicit!r} not in dataset; have {sorted(names)}")
        return explicit
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"no {kind} variable among {candidates}; have {sorted(names)}")


def regrid_elevation_to_latlon(ds, *, var_name: str = "", target_res_deg: float = 0.25):
    """Regrid an elevation :class:`xarray.Dataset` to a regular lat-lon grid.

    PURE (no I/O): resolves the elevation + lat/lon variables, squeezes any extra
    dims, normalises longitude to [0, 360) ascending and latitude ascending, then
    bilinearly interpolates onto a regular ``target_res_deg`` grid.  Returns a
    Dataset with ``elevation(lat, lon)`` [m] + 1-D ``lat``/``lon`` coords [deg] —
    the layout ``load_real_topography`` auto-detects.
    """
    import numpy as np
    import xarray as xr

    if not (0.0 < float(target_res_deg) <= 10.0):
        raise ValueError(f"target_res_deg must be in (0, 10], got {target_res_deg!r}")

    ev = _resolve(ds, _ELEV_CANDIDATES, kind="elevation", explicit=var_name)
    la = _resolve(ds, _LAT_CANDIDATES, kind="latitude")
    lo = _resolve(ds, _LON_CANDIDATES, kind="longitude")

    da = ds[ev]
    drop = [d for d in da.dims if d not in (la, lo)]
    if drop:
        da = da.isel({d: 0 for d in drop})
    da = da.rename({la: "lat", lo: "lon"}).transpose("lat", "lon")

    # Normalise source coords: lon in [0, 360) ascending, lat ascending.  A global
    # grid that includes BOTH 0 and 360 (e.g. the 361-lon ETOPO) collides to a
    # duplicate 0 after the mod, which breaks interp's unique-index requirement —
    # drop the duplicate endpoint(s) on each axis.
    lon = np.asarray(da["lon"].values, dtype=np.float64) % 360.0
    da = da.assign_coords(lon=lon).sortby("lon").sortby("lat")
    da = da.drop_duplicates("lon", keep="first").drop_duplicates("lat", keep="first")
    # A GLOBAL source is periodic in lon: wrap one column onto each side so
    # targets past the last source lon interpolate across the seam.  Without
    # it they fall outside the hull (a 1e-14 overshoot is enough) and get the
    # neighbouring column copied in by the nearest-fill below (#1712).
    src_lon = np.asarray(da["lon"].values)
    if src_lon.size > 1:
        dlon = float(np.median(np.diff(src_lon)))
        if abs(src_lon[-1] - src_lon[0] + dlon - 360.0) < 0.5 * dlon:
            da = xr.concat(
                [da.isel(lon=[-1]).assign_coords(lon=[src_lon[-1] - 360.0]),
                 da,
                 da.isel(lon=[0]).assign_coords(lon=[src_lon[0] + 360.0])],
                dim="lon")

    res = float(target_res_deg)
    tgt_lat = np.arange(-90.0 + res / 2.0, 90.0, res)
    tgt_lon = np.arange(res / 2.0, 360.0, res)
    out = da.interp(lat=tgt_lat, lon=tgt_lon, method="linear")
    # Edge targets outside the source hull (poles / lon seam) -> nearest-fill, then 0.
    out = out.interpolate_na(dim="lon", method="nearest").interpolate_na(
        dim="lat", method="nearest").fillna(0.0)

    elev = np.asarray(out.values, dtype=np.float64)
    return xr.Dataset(
        {"elevation": (("lat", "lon"), elev)},
        coords={"lat": tgt_lat, "lon": tgt_lon},
        attrs={
            "title": f"Model-ready topography ({res} deg lat-lon)",
            "source_variable": ev,
            "units": "m",
            "note": "elevation regridded for legoesm.grids.topography.load_real_topography; "
                    "land fraction = sub-grid fraction with elevation>0",
        },
    )


def build_topography(input_path: str, out_path: str, *, var_name: str = "",
                     target_res_deg: float = 0.25) -> str:
    """Open ``input_path``, regrid, write ``out_path`` (NetCDF).  Returns ``out_path``."""
    import xarray as xr
    ds = xr.open_dataset(input_path)
    try:
        out = regrid_elevation_to_latlon(ds, var_name=var_name, target_res_deg=target_res_deg)
    finally:
        ds.close()
    out.to_netcdf(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="source elevation NetCDF (NOAA ETOPO / GEBCO / ETOPO5 / ...)")
    p.add_argument("--out", required=True, help="output model-ready topography NetCDF")
    p.add_argument("--var-name", default="",
                   help=f"elevation variable; empty -> auto-detect ({', '.join(_ELEV_CANDIDATES)})")
    p.add_argument("--resolution-deg", type=float, default=0.25,
                   help="target lat-lon resolution [deg] (default 0.25; ~1 matches Levante)")
    args = p.parse_args(argv)

    out_path = build_topography(args.input, args.out, var_name=args.var_name,
                                target_res_deg=args.resolution_deg)

    import numpy as np
    import xarray as xr
    with xr.open_dataset(out_path) as chk:
        elev = np.asarray(chk["elevation"].values)
        land_frac = float(np.mean(elev > 0.0))
        print(
            f"[etopo] wrote {elev.shape[0]}x{elev.shape[1]} topography "
            f"(elev [{float(elev.min()):.0f}, {float(elev.max()):.0f}] m, "
            f"land fraction {land_frac:.3f}) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
