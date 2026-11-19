"""Prescribed land surface boundary (atmosphere-only AMIP on the MPAS lane).

A monthly climatology of the land skin temperature, the land turbulent heat
fluxes and (optionally) the land albedo stands in for the interactive land
model: the atmosphere receives them through the SAME hand-off channels the
multilayer land uses on that lane (``T_sfc`` and ``sfc_albedo`` blended over
the land fraction, ``forcing["shflx_land"]`` / ``["lhflx_land"]`` /
``["evap_land"]``), read from a file instead of solved.  Which of them are
handed over is ``ExperimentConfig.land_prescribed_channels``.

File contract (written by ``scripts/data/build_era5_prescribed_land.py``):
regular lat-lon ``ts`` [K], ``hfss`` and ``hfls`` [W m-2, ``positive='up'``:
POSITIVE UPWARD, out of the surface into the atmosphere -- the model's land
flux convention], optional ``albedo`` [1, in [0, 1]], on a ``time`` axis of at
most 12 mid-month days of a 365-day climatological year (0 = 1 January 00:00).

Regridding reuses the AMIP SST/SIC owner (``regrid_monthly_latlon_to_grid``,
bilinear); time interpolation reuses its cyclic mid-month bracket
(``climatology_interp_indices``, linear, closing the December-January seam).
Both run on the host once per refresh (the land channels are host-side).
"""
from __future__ import annotations

import os
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.forcing.amip import (
    climatology_interp_indices, regrid_monthly_latlon_to_grid,
)
from legoesm.forcing.time_utils import NOLEAP_DAYS_PER_YEAR

REQUIRED_VARS = ("ts", "hfss", "hfls")
OPTIONAL_VARS = ("albedo",)
# Units the file must declare (the builder writes exactly these).
_UNITS = {"ts": "K", "hfss": "W m-2", "hfls": "W m-2", "albedo": "1"}
_UPWARD_FLUXES = ("hfss", "hfls")
# climatology_interp_indices wraps on the 365-day year only for <= 12 records.
_MAX_RECORDS = 12


class PrescribedLand(NamedTuple):
    times_doy: np.ndarray   # (ntime,) mid-month anchors, days of a 365-day year
    ts: np.ndarray          # (ntime, ncol) skin temperature [K]
    hfss: np.ndarray        # (ntime, ncol) sensible heat flux [W m-2], positive up
    hfls: np.ndarray        # (ntime, ncol) latent heat flux [W m-2], positive up
    path: str
    albedo: np.ndarray | None = None   # (ntime, ncol) broadband albedo [1], or None


def load_prescribed_land(path: str, grid) -> PrescribedLand:
    """Read the climatology file and regrid every record onto ``grid``'s
    columns.  Raises on a missing file, a missing variable, a wrong unit or
    flux sign convention, a time axis that is not a <= 12-record climatology
    inside one 365-day year, non-finite values, or an albedo outside [0, 1]."""
    import xarray as xr

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"land_prescribed_flux_file {path!r} does not exist (build it with "
            "scripts/data/build_era5_prescribed_land.py)")
    ds = xr.open_dataset(path, decode_times=False)
    try:
        missing = [v for v in REQUIRED_VARS if v not in ds]
        if missing:
            raise KeyError(
                f"{path}: missing variables {missing}; expected {REQUIRED_VARS}")
        present = REQUIRED_VARS + tuple(v for v in OPTIONAL_VARS if v in ds)
        for v in present:
            units = str(ds[v].attrs.get("units", ""))
            if units != _UNITS[v]:
                raise ValueError(
                    f"{path}: {v} must carry units={_UNITS[v]!r}; got {units!r}")
        for v in _UPWARD_FLUXES:
            if str(ds[v].attrs.get("positive", "")).lower() != "up":
                raise ValueError(
                    f"{path}: {v} must carry positive='up' (model land-flux "
                    f"convention); got {ds[v].attrs.get('positive')!r}")
        times = np.asarray(ds["time"].values, dtype=np.float64)
        if (times.ndim != 1 or not 2 <= times.size <= _MAX_RECORDS
                or np.any(np.diff(times) <= 0)):
            raise ValueError(
                f"{path}: time axis must be 2..{_MAX_RECORDS} strictly "
                f"increasing climatological records, got {times}")
        if times[0] < 0.0 or times[-1] >= NOLEAP_DAYS_PER_YEAR:
            raise ValueError(
                f"{path}: time anchors must lie in [0, {NOLEAP_DAYS_PER_YEAR}) "
                f"days of the climatological year, got [{times[0]}, {times[-1]}]")
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        if np.any(np.diff(lon) <= 0) or lon[0] < 0.0 or lon[-1] >= 360.0:
            # the shared regridder maps targets into [0, 360): a -180..180
            # axis would extrapolate every column east of 180 E silently
            raise ValueError(
                f"{path}: lon must be strictly increasing within [0, 360), got "
                f"[{lon[0]}, {lon[-1]}]")
        fields = {}
        for v in present:
            arr = np.asarray(ds[v].values, dtype=np.float64)
            if arr.shape != (times.size, lat.size, lon.size):
                raise ValueError(
                    f"{path}: {v} has shape {arr.shape}, expected "
                    f"(time, lat, lon) = {(times.size, lat.size, lon.size)}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{path}: {v} contains non-finite values")
            if lat[0] > lat[-1]:          # the regridder wants ascending latitude
                arr = arr[:, ::-1, :]
            fields[v] = arr
        if lat[0] > lat[-1]:
            lat = lat[::-1]
    finally:
        ds.close()
    alb = fields.get("albedo")
    if alb is not None and (alb.min() < 0.0 or alb.max() > 1.0):
        raise ValueError(
            f"{path}: albedo outside [0, 1]: [{alb.min()}, {alb.max()}]")
    cols = {v: regrid_monthly_latlon_to_grid(arr, lat, lon, grid).reshape(
        times.size, -1) for v, arr in fields.items()}
    return PrescribedLand(times_doy=times, ts=cols["ts"], hfss=cols["hfss"],
                          hfls=cols["hfls"], path=path, albedo=cols.get("albedo"))


def prescribed_land_at_day(pl: PrescribedLand, day: float):
    """``(ts, hfss, hfls, albedo)`` at simulated ``day`` (days since 1 January
    of the start year on the 365-day clock): linear between the mid-month
    anchors, periodic over the year.  ``albedo`` is None when the file has
    none."""
    i0, i1, w = climatology_interp_indices(
        jnp.asarray(pl.times_doy), float(day))
    i0, i1, w = int(i0), int(i1), float(w)
    return tuple(None if a is None else (1.0 - w) * a[i0] + w * a[i1]
                 for a in (pl.ts, pl.hfss, pl.hfls, pl.albedo))
