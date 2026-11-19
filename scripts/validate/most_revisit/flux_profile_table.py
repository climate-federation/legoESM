#!/usr/bin/env python
"""Common flux-profile table: one NetCDF per site, read by every flux-gradient analysis.

Time is the END of each averaging interval, seconds since 1970 UTC. Heights are metres
above the local surface, per time because snow and ice move them; displacement is an
attribute and is never subtracted here. Temperature is air temperature in K, H and LE are
positive upward in W m-2, pressure in Pa. Loaders apply the source's own flags only;
physical filters live in the analysis.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from legoesm import constants

REQUIRED_ATTRS = ("site", "latitude", "longitude", "surface", "canopy_height_m", "displacement_m",
                  "source", "licence", "averaging_minutes", "flux_kind", "flux_method")
FLUX_KINDS = ("native", "mean_of_shorter_periods")
RANGES = {"wind": (0.0, 40.0), "temp": (180.0, 330.0), "pressure": (50000.0, 110000.0),
          "ustar": (0.0, 5.0), "t_flux": (180.0, 330.0)}


def vector_speed(u, v):
    """Speed of the vector mean of already block-averaged components."""
    return np.hypot(u, v)


def _check_units(name, x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        raise ValueError(f"{name} has no finite value")
    lo, hi = RANGES[name]
    med = np.nanmedian(x)
    if not lo <= med <= hi:
        raise ValueError(f"{name} median {med} outside {lo}..{hi}: wrong units?")


def write_table(path, time_end, z_u, wind, z_t, temp, z_f, ustar, h, le, pressure, t_flux, attrs):
    """time_end (time); z_u, wind (time, lev_u); z_t, temp (time, lev_t); z_f, ustar, h, le, t_flux (time, lev_f)."""
    missing = [k for k in REQUIRED_ATTRS if k not in attrs]
    if missing:
        raise ValueError(f"missing attributes {missing}")
    if attrs["flux_kind"] not in FLUX_KINDS:
        raise ValueError(f"unknown flux_kind {attrs['flux_kind']}")
    time_end = np.asarray(time_end, dtype=np.int64)
    if np.any(np.diff(time_end) <= 0):
        raise ValueError("time must increase strictly")
    for name, x in (("wind", wind), ("temp", temp), ("pressure", pressure), ("ustar", ustar), ("t_flux", t_flux)):
        _check_units(name, x)
    # Profile pairs are (i, i+1) kept only where z[i+1] > z[i]: a top-down table silently scores nothing.
    # Equal heights (two masts at one level) only drop that pair; flux levels are picked by height, not order.
    for name, z in (("z_u", z_u), ("z_t", z_t)):
        if np.any(np.diff(np.nanmedian(np.asarray(z, dtype=float), axis=0)) < 0):
            raise ValueError(f"{name} levels must ascend: the analysis pairs level i with i+1")
    dims = {"z_u": ("time", "lev_u"), "wind": ("time", "lev_u"), "z_t": ("time", "lev_t"),
            "temp": ("time", "lev_t"), "z_f": ("time", "lev_f"), "ustar": ("time", "lev_f"),
            "h": ("time", "lev_f"), "le": ("time", "lev_f"), "t_flux": ("time", "lev_f"),
            "pressure": ("time",)}
    data = dict(z_u=z_u, wind=wind, z_t=z_t, temp=temp, z_f=z_f, ustar=ustar, h=h, le=le,
                t_flux=t_flux, pressure=pressure)
    ds = xr.Dataset({k: (dims[k], np.asarray(v, dtype=np.float32)) for k, v in data.items()},
                    coords={"time": ("time", time_end)}, attrs=dict(attrs))
    ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00 UTC, end of interval"
    ds.to_netcdf(path, encoding={k: {"zlib": True, "complevel": 4} for k in data})


def matched(ds, flux_level=0):
    """Times with u*, H, and at least two wind and two temperature levels, all at once."""
    f = ds.isel(lev_f=flux_level)
    return (np.isfinite(f.ustar) & np.isfinite(f.h)
            & (np.isfinite(ds.wind).sum("lev_u") >= 2) & (np.isfinite(ds.temp).sum("lev_t") >= 2))


def report(path, neutral_zeta=0.02):
    """Counts and end-to-end checks a loader must pass before its table is used."""
    kappa = constants.kappa_von_karman
    ds = xr.open_dataset(path, decode_times=False)
    out = {"site": ds.attrs["site"], "times": int(ds.time.size)}
    hour_local = (ds.time.values / 3600.0 + ds.attrs["longitude"] / 15.0) % 24
    for k in range(ds.sizes["lev_f"]):
        f = ds.isel(lev_f=k)
        ok = matched(ds, k).values
        diurnal = [float(np.nanmedian(f.h.values[(hour_local >= a) & (hour_local < a + 1)])) for a in range(24)]
        row = {"z_f": float(np.nanmedian(f.z_f)), "matched": int(ok.sum()),
               "h_peak_local_hour": int(np.nanargmax(diurnal)), "h_median_at_peak": float(np.nanmax(diurnal))}
        # Near-neutral phi_m for adjacent wind levels: a wrong height, unit, or u* shows here.
        pairs = {}
        rho_cp = 1200.0  # ponytail: rough, only sets zeta for the neutral window
        zeta_per_z = -kappa * constants.g * (f.h.values / rho_cp) / (f.t_flux.values * f.ustar.values**3)
        for i in range(ds.sizes["lev_u"] - 1):
            z1, z2 = ds.z_u.values[:, i], ds.z_u.values[:, i + 1]
            zg = np.sqrt(z1 * z2)
            phi = kappa * (ds.wind.values[:, i + 1] - ds.wind.values[:, i]) / (np.log(z2 / z1) * f.ustar.values)
            sel = ok & (np.abs(zeta_per_z * zg) < neutral_zeta) & (f.ustar.values > 0.2) & np.isfinite(phi)
            key = f"{np.nanmedian(z1):.4g}-{np.nanmedian(z2):.4g}"
            pairs[key] = [int(sel.sum()), float(np.median(phi[sel])) if sel.any() else None]
        row["near_neutral_phi_m"] = pairs
        out[f"flux_{k}"] = row
    return out
