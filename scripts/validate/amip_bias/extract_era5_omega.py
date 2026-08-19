#!/usr/bin/env python
"""Extract ERA5 monthly pressure-vertical-velocity onto the model grid, once.

`omega` is the field that says where the atmosphere ascends and subsides, and
it is the one field the campaign's ClimateEval reference tree does NOT carry —
that tree has ERA5 `ua` but not `va`, so omega cannot even be derived from it
by continuity. It IS in the DKRZ ERA5 pool as GRIB parameter 135, which needs
`cfgrib`/eccodes, which the JAX environment does not have.

So this runs ONCE under the ESMValTool/benchmarking environment and writes a
small `.npz` the ordinary probes can read:

    /work/bd1083/b309178/mambaforge/envs/benchmarking/bin/python \\
        scripts/validate/amip_bias/extract_era5_omega.py --year 1979

The source is on a REDUCED GAUSSIAN grid (an unstructured point list, not a
lat-lon array), so the points are binned onto the target grid with cos(lat)
weights — the same area-weighted reduction `regional_bias.bin_to_model` applies
to the rectangular references, written out here for a point list because that
function takes grid axes.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

DEFAULT_OUT = pathlib.Path(__file__).resolve().parent / "era5_omega_clim.npz"
GRIB = "/pool/data/ERA5/E5/pl/an/1M/{code}/E5pl00_1M_{year}_{code}.grb"
# 135 = pressure vertical velocity, 131 = u, 132 = v.  The winds are pulled too
# so the SAME continuity estimator the model side must use can be applied to
# ERA5 and checked against ERA5's own omega -- without that control, a
# model-derived omega compared to a reference TRUE omega mixes an estimator
# error into every difference.
CODES = {"omega": "135", "u": "131", "v": "132"}


def bin_points(values, plat, plon, mlat, mlon):
    """Area-weighted bin of scattered (plat, plon) points onto a lat-lon grid.

    A cell that receives no point is FATAL rather than filled: a silently
    empty cell would enter every downstream mean as whatever the fill was.
    """
    dlat = float(mlat[1] - mlat[0])
    dlon = float(mlon[1] - mlon[0])
    iy = np.clip(((plat - (mlat[0] - dlat / 2)) // dlat).astype(int),
                 0, mlat.size - 1)
    ix = np.clip(((plon % 360.0 - (mlon[0] - dlon / 2)) % 360.0 // dlon
                  ).astype(int), 0, mlon.size - 1)
    w = np.cos(np.deg2rad(plat))
    flat = iy * mlon.size + ix
    n = mlat.size * mlon.size
    good = np.isfinite(values)
    num = np.bincount(flat[good], weights=(values * w)[good], minlength=n)
    den = np.bincount(flat[good], weights=w[good], minlength=n)
    if (den <= 0).any():
        raise SystemExit(f"FATAL: {int((den <= 0).sum())} target cells got no "
                         "ERA5 point")
    return (num / den).reshape(mlat.size, mlon.size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=1979,
                    help="Epoch-match this to the run being scored.")
    ap.add_argument("--nlat", type=int, default=36)
    ap.add_argument("--nlon", type=int, default=72)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    import xarray as xr
    mlat = -90.0 + 180.0 / args.nlat * (np.arange(args.nlat) + 0.5)
    mlon = 360.0 / args.nlon * (np.arange(args.nlon) + 0.5)
    fields, plev = {}, None
    for name, code in CODES.items():
        d = xr.open_dataset(GRIB.format(year=args.year, code=code),
                            engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = [k for k in d.data_vars][0]
        plev = np.asarray(d["isobaricInhPa"], dtype=np.float64) * 100.0
        plat = np.asarray(d["latitude"], dtype=np.float64)
        plon = np.asarray(d["longitude"], dtype=np.float64)
        a = np.asarray(d[var], dtype=np.float64)      # (time, plev, values)
        out = np.empty((a.shape[0], plev.size, mlat.size, mlon.size))
        for t in range(a.shape[0]):
            for k in range(plev.size):
                out[t, k] = bin_points(a[t, k], plat, plon, mlat, mlon)
        fields[name] = out
        print(f"  {name} ({code}) done", flush=True)

    out = fields["omega"]
    np.savez_compressed(args.out, omega=out, ua=fields["u"], va=fields["v"],
                        plev=plev, lat=mlat, lon=mlon,
                        year=args.year, units="Pa s-1",
                        source=GRIB.format(year=args.year, code="135/131/132"))
    print(f"wrote {args.out}  shape {out.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
