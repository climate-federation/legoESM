"""Build an AMIP prescribed-SST/SIC forcing from WOA18 surface temperature.

The synthetic ``cos^2(lat)`` SST used for the AMIP smoke test is zonally
symmetric and lacks real spatial structure (warm pool, cold tongue, western
boundary currents, coastal upwelling).  This builder replaces it with the World
Ocean Atlas 2018 annual decadal-average surface temperature (``t_an`` at
depth=0) — a citable observed SST climatology — regridded onto the AMIP forcing
grid, with land flood-filled (nearest ocean) and a freezing-point sea-ice
concentration derived from the SST.

Grid / format are mirrored from a template AMIP forcing file (the synthetic
deck), so the output is consumed unchanged by ``load_amip_forcing`` with
``--dataset hadisst`` (``sst`` in degC + ``sic`` fraction).

Limitations (state these in any publication that uses the output):
  * WOA ``t00`` is the ANNUAL mean — this forcing has NO seasonal cycle.  For a
    seasonal AMIP, build from the monthly WOA products (``t01``..``t12``).
  * SIC is a freezing-ramp proxy (``sic`` ramps 1->0 as annual-mean SST rises
    through ``[T_freeze_ocean, T_freeze_ocean + ramp]``), so it captures
    perennial pack ice but not the seasonal marginal ice zone.

Source: WOA18 decav, ``woa18_decav_t00_01.nc`` (1-degree in-situ temperature).

Run (on a compute node):
    python scripts/data/build_amip_sst_from_woa.py \
        --woa-t data/woa18/woa18_decav_t00_01.nc \
        --template forcing_amip_regen/sst_sic_amip_1979-2014.nc \
        --out forcing_amip_woa/sst_sic_amip_1979-2014.nc
"""
from __future__ import annotations

import argparse

import numpy as np

from legoesm import constants

# Marginal-ice-zone ramp width [degC]: the annual-mean SST band over which the
# freezing-ramp sea-ice fraction relaxes from 1 (pack ice) to 0 (open water).
# A modelling choice for the proxy, not a physical constant.
_SIC_RAMP_C_DEFAULT = 0.5


def fill_land_nearest(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill ``~valid`` cells with the nearest valid value (lon-periodic).

    The Euclidean distance transform runs on a 3x lon-tiled copy so the dateline
    seam fills from across the wrap, then the centre tile is returned.
    """
    from scipy.ndimage import distance_transform_edt

    nlat, nlon = field.shape
    tiled = np.concatenate([field, field, field], axis=1)
    vtiled = np.concatenate([valid, valid, valid], axis=1)
    # indices of the nearest valid cell for every cell
    idx = distance_transform_edt(
        ~vtiled, return_distances=False, return_indices=True)
    filled = tiled[tuple(idx)]
    return filled[:, nlon:2 * nlon]


def regrid_to_amip(woa_sst: np.ndarray, woa_lat: np.ndarray,
                   woa_lon: np.ndarray, tgt_lat: np.ndarray,
                   tgt_lon: np.ndarray) -> np.ndarray:
    """Linear-regrid a (land-filled) WOA surface field to the AMIP grid.

    WOA lon is ``-179.5..179.5``; it is rolled to ``0..360`` ascending and
    periodic-padded one wrap each side, and lat is edge-padded to the poles, so
    the AMIP grid (lat to +-90, lon 0..357.5) needs NO extrapolation —
    ``bounds_error=True`` then guarantees every target point was interpolated.
    """
    from scipy.interpolate import RegularGridInterpolator

    # roll lon into [0, 360) ascending, carrying the data columns
    lon360 = np.mod(woa_lon, 360.0)
    order = np.argsort(lon360)
    lon360 = lon360[order]
    sst = woa_sst[:, order]
    # periodic pad in lon (one wrapped column each side)
    lon_pad = np.concatenate(
        [[lon360[-1] - 360.0], lon360, [lon360[0] + 360.0]])
    sst = np.concatenate([sst[:, -1:], sst, sst[:, :1]], axis=1)
    # edge pad in lat out to the poles
    lat_pad = np.concatenate([[-90.0], woa_lat, [90.0]])
    sst = np.concatenate([sst[:1, :], sst, sst[-1:, :]], axis=0)

    interp = RegularGridInterpolator(
        (lat_pad, lon_pad), sst, method="linear", bounds_error=True)
    lon_mesh, lat_mesh = np.meshgrid(tgt_lon, tgt_lat)
    return interp((lat_mesh, lon_mesh))


def sst_to_sic(sst_C: np.ndarray, freeze_C: float | None = None,
               ramp_C: float = _SIC_RAMP_C_DEFAULT) -> np.ndarray:
    """Freezing-ramp sea-ice fraction in [0, 1].

    ``sic = 1`` where ``SST <= freeze_C`` and ramps linearly to 0 as SST rises
    to ``freeze_C + ramp_C``.  ``freeze_C`` defaults to the seawater freezing
    point ``constants.T_freeze_ocean`` (in degC).
    """
    if ramp_C <= 0.0:
        raise ValueError(f"ramp_C must be positive, got {ramp_C}")
    if freeze_C is None:
        freeze_C = float(constants.T_freeze_ocean - constants.T_freeze)
    return np.clip((freeze_C + ramp_C - sst_C) / ramp_C, 0.0, 1.0)


def build(woa_t_path: str, template_path: str, ramp_C: float):
    """Read WOA + template, return (sst_amip_2d_degC, sic_amip_2d, template_ds)."""
    import xarray as xr

    woa = xr.open_dataset(woa_t_path, decode_times=False)
    tvar = next(v for v in woa.data_vars if "t_an" in v)
    da = woa[tvar]
    # surface level of the annual file: time=0, depth=0 -> (lat, lon)
    da = da.isel(time=0, depth=0) if "depth" in da.dims else da.isel(time=0)
    woa_sst = np.asarray(da.values, dtype=np.float64)
    woa_lat = np.asarray(woa["lat"].values, dtype=np.float64)
    woa_lon = np.asarray(woa["lon"].values, dtype=np.float64)

    valid = np.isfinite(woa_sst)
    if valid.mean() < 0.3:
        raise SystemExit(
            f"WOA surface SST is mostly NaN ({valid.mean():.2f} valid) — "
            "wrong variable or vertical level?")

    filled = fill_land_nearest(woa_sst, valid)

    tmpl = xr.open_dataset(template_path, decode_times=False)
    tgt_lat = np.asarray(tmpl["lat"].values, dtype=np.float64)
    tgt_lon = np.asarray(tmpl["lon"].values, dtype=np.float64)
    sst_amip = regrid_to_amip(filled, woa_lat, woa_lon, tgt_lat, tgt_lon)
    sic_amip = sst_to_sic(sst_amip, ramp_C=ramp_C)
    return sst_amip, sic_amip, tmpl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build an AMIP SST/SIC forcing from WOA18 surface T.")
    p.add_argument("--woa-t", default="data/woa18/woa18_decav_t00_01.nc",
                   help="WOA18 annual temperature NetCDF (t_an, degC).")
    p.add_argument("--template",
                   default="forcing_amip_regen/sst_sic_amip_1979-2014.nc",
                   help="AMIP forcing file whose grid/time/attrs are mirrored.")
    p.add_argument("--out", required=True, help="Output NetCDF path.")
    p.add_argument("--ramp-c", type=float, default=_SIC_RAMP_C_DEFAULT,
                   help="Freezing-ramp SIC width [degC].")
    args = p.parse_args(argv)

    sst_amip, sic_amip, tmpl = build(args.woa_t, args.template, args.ramp_c)

    nt = int(tmpl.sizes["time"])
    # annual climatology -> constant in time (broadcast over the template months)
    sst_t = np.broadcast_to(sst_amip[None], (nt, *sst_amip.shape)).copy()
    sic_t = np.broadcast_to(sic_amip[None], (nt, *sic_amip.shape)).copy()

    out = tmpl.copy()
    out["sst"] = (("time", "lat", "lon"), sst_t)
    out["sst"].attrs["units"] = "degC"
    out["sst"].attrs["source"] = f"WOA18 decav t_an surface ({args.woa_t})"
    out["sic"] = (("time", "lat", "lon"), sic_t)
    out["sic"].attrs["units"] = "1"
    out["sic"].attrs["note"] = (
        "freezing-ramp proxy from annual-mean SST (no seasonal MIZ)")
    out.attrs["history"] = (
        "build_amip_sst_from_woa.py: WOA18 annual surface SST, land "
        "flood-filled (nearest ocean), freezing-ramp SIC")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_netcdf(args.out)
    print(f"wrote {args.out}: SST degC min={float(sst_amip.min()):.2f} "
          f"max={float(sst_amip.max()):.2f} mean={float(sst_amip.mean()):.2f}; "
          f"SIC ice-frac mean={float(sic_amip.mean()):.3f}, "
          f"ice-cells={float((sic_amip > 0.5).mean()) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
