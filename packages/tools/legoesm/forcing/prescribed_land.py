"""Prescribed land surface boundary (atmosphere-only AMIP on the MPAS lane).

A monthly climatology of the land skin temperature and the land turbulent
heat fluxes replaces the interactive land tile: the atmosphere receives the
same three channels the multilayer land hands over each step (``T_sfc`` over
the land fraction, ``forcing["shflx_land"]`` / ``["lhflx_land"]``), read from a
file instead of solved.  The file is built by
``scripts/data/build_era5_prescribed_land.py``: regular lat-lon ``ts`` [K],
``hfss`` and ``hfls`` [W m-2, POSITIVE UPWARD] with a ``time`` axis of
mid-month days of a 365-day climatological year (0 = 1 January 00:00).

Regridding reuses the AMIP SST/SIC owner (``regrid_monthly_latlon_to_grid``);
time interpolation is linear between the mid-month anchors, periodic over the
365-day year, evaluated on the host (the land channel is host-side already).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from legoesm.forcing.amip import regrid_monthly_latlon_to_grid

DAYS_PER_YEAR = 365.0
REQUIRED_VARS = ("ts", "hfss", "hfls")
OPTIONAL_VARS = ("albedo",)


class PrescribedLand(NamedTuple):
    times_doy: np.ndarray   # (ntime,) mid-month anchors, days of a 365-day year
    ts: np.ndarray          # (ntime, ncol) skin temperature [K]
    hfss: np.ndarray        # (ntime, ncol) sensible heat flux [W m-2], positive up
    hfls: np.ndarray        # (ntime, ncol) latent heat flux [W m-2], positive up
    path: str
    albedo: np.ndarray | None = None   # (ntime, ncol) broadband surface albedo [-], or None


def load_prescribed_land(path: str, grid) -> PrescribedLand:
    """Load the climatology file and regrid every month onto ``grid``'s columns."""
    import xarray as xr

    ds = xr.open_dataset(path, decode_times=False)
    try:
        missing = [v for v in REQUIRED_VARS if v not in ds]
        if missing:
            raise KeyError(f"{path}: missing variables {missing}; expected {REQUIRED_VARS}")
        for v in ("hfss", "hfls"):
            if str(ds[v].attrs.get("positive", "")).lower() != "up":
                raise ValueError(
                    f"{path}: {v} must carry positive='up' (model land-flux convention); "
                    f"got {ds[v].attrs.get('positive')!r}")
        times = np.asarray(ds["time"].values, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(f"{path}: time axis must be strictly increasing, got {times}")
        if times[0] < 0.0 or times[-1] >= DAYS_PER_YEAR:
            raise ValueError(
                f"{path}: time anchors must lie in [0, {DAYS_PER_YEAR}) days of the "
                f"climatological year, got [{times[0]}, {times[-1]}]")
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        fields = {}
        for v in REQUIRED_VARS + tuple(v for v in OPTIONAL_VARS if v in ds):
            arr = np.asarray(ds[v].values, dtype=np.float64)
            if arr.shape[0] != times.size:
                raise ValueError(f"{path}: {v} has {arr.shape[0]} records, time has {times.size}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{path}: {v} contains non-finite values")
            if lat[0] > lat[-1]:          # the regridder wants ascending latitude
                arr = arr[:, ::-1, :]
            fields[v] = arr
        if lat[0] > lat[-1]:
            lat = lat[::-1]
    finally:
        ds.close()
    ncol_fields = {}
    for v, arr in fields.items():
        reg = regrid_monthly_latlon_to_grid(arr, lat, lon, grid)
        ncol_fields[v] = reg.reshape(times.size, -1)
    alb = ncol_fields.get("albedo")
    if alb is not None and (alb.min() < 0.0 or alb.max() > 1.0):
        raise ValueError(f"{path}: albedo outside [0, 1]: [{alb.min()}, {alb.max()}]")
    return PrescribedLand(times_doy=times, ts=ncol_fields["ts"],
                          hfss=ncol_fields["hfss"], hfls=ncol_fields["hfls"], path=path,
                          albedo=alb)


def prescribed_land_at_day(pl: PrescribedLand, day: float):
    """``(ts, hfss, hfls, albedo)`` at simulated ``day`` (days since 1 January
    of the start year on the 365-day clock), linear between anchors, periodic;
    ``albedo`` is None when the file carries none."""
    doy = float(day) % DAYS_PER_YEAR
    t = pl.times_doy
    n = t.size
    # periodic bracket: anchors extended by one on each side
    if doy < t[0]:
        t0, t1, i0, i1 = t[-1] - DAYS_PER_YEAR, t[0], n - 1, 0
    elif doy >= t[-1]:
        t0, t1, i0, i1 = t[-1], t[0] + DAYS_PER_YEAR, n - 1, 0
    else:
        i1 = int(np.searchsorted(t, doy, side="right"))
        i0 = i1 - 1
        t0, t1 = t[i0], t[i1]
    w = (doy - t0) / (t1 - t0)
    out = []
    for arr in (pl.ts, pl.hfss, pl.hfls, pl.albedo):
        out.append(None if arr is None else (1.0 - w) * arr[i0] + w * arr[i1])
    return tuple(out)
