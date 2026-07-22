#!/usr/bin/env python3
"""Build a CMIP-style ``sftlf`` land-fraction NetCDF from CLM surfdata.

Why: the AMIP production chain derives ``f_land`` from ETOPO elevation
(``elevation > 0``) when no ``--land-mask-file`` is given.  ETOPO carries
real BATHYMETRY for below-sea-level inland seas (Caspian −28 m, Aral,
Great Lakes), so those cells classify as OCEAN, receive a nearest-
neighbour prescribed SST, and evaporate at the uncapped potential rate —
the confirmed source of the 2000 W/m² hfls / 177 mm prw central-Asia
hotspots in the 2-yr AMIP pilot (2026-07-22).  The CLM surfdata already
staged for the multilayer land model carries its OWN land fraction
(``LANDFRAC_PFT``, Caspian/Aral = 1.0), so the land tile and the surface
mask stay mutually consistent.

The output is a minimal regular-lat-lon NetCDF with 1-D ``lat``/``lon``
coordinates and an ``sftlf`` variable in [0, 1] — exactly what
``legoesm.grids.topography._load_land_fraction_file`` auto-detects for
``run_amip.py --land-mask-file``.

Usage::

    python3 scripts/data/build_sftlf_from_surfdata.py \
        data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc \
        data/clm/sftlf_clm_1.9x2.5.nc
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def build_sftlf(surfdata_path: str, out_path: str,
                var_name: str = "LANDFRAC_PFT") -> dict:
    """Extract the CLM land fraction onto 1-D lat/lon and write ``sftlf``.

    Returns a small summary dict (shapes + a few probe values) for
    logging/tests.
    """
    import netCDF4 as nc

    with nc.Dataset(surfdata_path) as ds:
        if var_name not in ds.variables:
            raise KeyError(
                f"{surfdata_path}: no variable {var_name!r}; "
                f"available: {sorted(ds.variables)[:20]}..."
            )
        frac = np.asarray(ds.variables[var_name][:], dtype=np.float64)
        lat2d = np.asarray(ds.variables["LATIXY"][:], dtype=np.float64)
        lon2d = np.asarray(ds.variables["LONGXY"][:], dtype=np.float64)

    if frac.ndim != 2:
        raise ValueError(f"{var_name} must be 2-D (lsmlat, lsmlon); "
                         f"got shape {frac.shape}")
    # The CLM surfdata grid must be regular (LATIXY constant along lon,
    # LONGXY constant along lat) for a 1-D coordinate extraction to be
    # loss-free.  Fail loudly otherwise rather than silently regridding.
    if not np.allclose(lat2d, lat2d[:, :1]):
        raise ValueError("LATIXY varies along longitude — not a regular "
                         "lat-lon grid; refusing 1-D extraction")
    if not np.allclose(lon2d, lon2d[:1, :]):
        raise ValueError("LONGXY varies along latitude — not a regular "
                         "lat-lon grid; refusing 1-D extraction")
    lat = lat2d[:, 0]
    lon = lon2d[0, :]

    # CLM stores fraction in [0, 1] with tiny float overshoot (max seen
    # 1.0000000193); clip, do not rescale.
    frac = np.clip(frac, 0.0, 1.0)

    with nc.Dataset(out_path, "w") as out:
        out.createDimension("lat", lat.size)
        out.createDimension("lon", lon.size)
        vlat = out.createVariable("lat", "f8", ("lat",))
        vlat.units = "degrees_north"
        vlat.standard_name = "latitude"
        vlat[:] = lat
        vlon = out.createVariable("lon", "f8", ("lon",))
        vlon.units = "degrees_east"
        vlon.standard_name = "longitude"
        vlon[:] = lon
        v = out.createVariable("sftlf", "f8", ("lat", "lon"))
        v.units = "1"
        v.long_name = "land_area_fraction"
        v.comment = ("fraction in [0,1] (not percent); clipped from "
                     f"{var_name}")
        v[:] = frac
        out.source = surfdata_path
        out.history = (f"build_sftlf_from_surfdata.py: extracted {var_name} "
                       "as sftlf (fraction); inland seas (Caspian/Aral) are "
                       "LAND per the CLM land model's own mask")

    def probe(lat_deg, lon_deg):
        j = int(np.argmin(np.abs(lat - lat_deg)))
        i = int(np.argmin(np.abs(lon - (lon_deg % 360.0))))
        return float(frac[j, i])

    return {
        "n_lat": int(lat.size), "n_lon": int(lon.size),
        "caspian_42N_51E": probe(42.0, 51.0),
        "aral_45N_60E": probe(45.0, 60.0),
        "mid_atlantic_30N_40W": probe(30.0, -40.0),
        "out_path": out_path,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("surfdata", help="CLM surfdata NetCDF (LANDFRAC_PFT)")
    p.add_argument("out", help="output sftlf NetCDF path")
    p.add_argument("--var", default="LANDFRAC_PFT",
                   help="land-fraction variable in the surfdata file")
    args = p.parse_args(argv)
    info = build_sftlf(args.surfdata, args.out, args.var)
    print("sftlf written:", info)
    if info["caspian_42N_51E"] < 0.5:
        print("WARNING: Caspian probe < 0.5 — mask does not fix the "
              "inland-sea misclassification this script exists for",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
