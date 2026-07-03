"""Write a SYNTHETIC idealized land-sea mask NetCDF for SMOKE-TESTING ``--ocean-only``.

The realistic ocean-only campaign (``--ocean-only``) needs a land mask (``--land-mask-path``)
to EXCLUDE land columns — over land the model's own land-surface biases dominate, so the
closure (the LES correction's lever) is the wrong tool there.  WITHOUT a mask the model is
flat (all ocean) and ``--ocean-only`` is a safe NO-OP: it cannot test that ocean-only
actually excludes land columns.  This writes a tiny IDEALIZED lat-lon mask (an ``lsm`` field
:func:`legoesm.grids.topography.load_land_fraction` auto-detects + regrids to ANY model grid)
with a clear land/ocean split, so the FULL ocean-only path can be smoked WITHOUT a real
CMIP6 ``sftlf`` / ERA5 ``lsm`` download::

    python scripts/data/make_synthetic_land_mask.py /tmp/land.nc
    python scripts/experiment/smoke_compare_reanalysis.py --ocean-only --land-mask-path /tmp/land.nc

This is NOT a real land-sea distribution — for a PRODUCTION comparison supply a real mask
(CMIP6 ``sftlf`` percent, ERA5 ``lsm`` fraction; the local NCAR-RDA ERA5 archive carries the
invariant ``128_172_lsm``).
"""

from __future__ import annotations

import argparse


def build_synthetic_land_mask_dataset(
    *, nlat: int = 24, nlon: int = 48,
    land_lat_min: float = 20.0, land_lat_max: float = 60.0,
):
    """An :class:`xarray.Dataset` with an ``lsm(lat, lon)`` field: land (1.0) in a zonal
    band ``[land_lat_min, land_lat_max]`` (degrees), ocean (0.0) elsewhere — a clear
    land/ocean split so ``--ocean-only`` has columns to exclude.  ``lat`` runs 90->-90
    descending (the ERA5 convention; the loader re-orders regardless).  The band default
    (20-60N, a NH mid-latitude continent analog) leaves both hemispheres' tropics + the SH
    as ocean, so any model grid regrids to a mix of land + ocean columns."""
    import numpy as np
    import xarray as xr

    if not (-90.0 <= land_lat_min < land_lat_max <= 90.0):
        raise ValueError(
            f"need -90 <= land_lat_min ({land_lat_min}) < land_lat_max ({land_lat_max}) "
            "<= 90 for a non-empty land band.")
    lat = np.linspace(90.0, -90.0, int(nlat))
    lon = np.linspace(0.0, 360.0, int(nlon), endpoint=False)
    land = (lat[:, None] >= land_lat_min) & (lat[:, None] <= land_lat_max)
    lsm = np.broadcast_to(land, (int(nlat), int(nlon))).astype(np.float64)
    return xr.Dataset({"lsm": (("lat", "lon"), lsm)}, coords={"lat": lat, "lon": lon})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", help="output NetCDF path for the land-sea mask")
    p.add_argument("--nlat", type=int, default=24, help="mask latitudes (default 24)")
    p.add_argument("--nlon", type=int, default=48, help="mask longitudes (default 48)")
    p.add_argument("--land-lat-min", type=float, default=20.0,
                   help="southern edge of the idealized land band [deg] (default 20)")
    p.add_argument("--land-lat-max", type=float, default=60.0,
                   help="northern edge of the idealized land band [deg] (default 60)")
    args = p.parse_args(argv)
    ds = build_synthetic_land_mask_dataset(
        nlat=args.nlat, nlon=args.nlon,
        land_lat_min=args.land_lat_min, land_lat_max=args.land_lat_max)
    ds.to_netcdf(args.out)
    n_land = int((ds["lsm"].values > 0.5).sum())
    print(f"[land-mask] wrote SYNTHETIC idealized land-sea mask ({args.nlat}x{args.nlon}, "
          f"land band [{args.land_lat_min}, {args.land_lat_max}] deg, {n_land} land cells) "
          f"to {args.out} — NOT a real land distribution; for production supply a real "
          "CMIP6 sftlf / ERA5 lsm.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
