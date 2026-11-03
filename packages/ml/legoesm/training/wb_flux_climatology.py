"""ERA5 surface-flux climatology for the WeatherBench CAM6 arm.

Owner decisions 2026-11-03: the CAM6 arm's seven prescribed surface fluxes
(sensible / latent heat, wind stresses, upward and downward shortwave, upward
longwave) are a 2015-2019 ERA5 climatology instead of the sample's
initial-time values.  ERA5 hourly means (the training loader's own signs,
``era5_turbulent_flux_planes`` / ``era5_radiation_flux_planes``; upward
fluxes are down minus net before averaging) are regridded to the model grid with
the loader's regridder, averaged per (day of year, UTC hour) over the five
years with 29 February dropped, then smoothed by a centred 61-day running
mean that wraps at the year end.  A forecast reads the climatology at each
step's valid time: linear in hour, the slot's own day; a 29 February valid
time reads 28 February.  The same climatology serves training, the 2014
validation and the 2020 evaluation.

Store: ``<state_cache_dir>/era5_flux_clim_<tag>.zarr``, array ``clim`` of
shape (365, 24, 7, n_columns) float32, one chunk per day.  Slot ``(d, h)``
holds the ERA5 means timestamped hour ``h`` (each the mean over the hour
ENDING at ``h``) of noleap day ``d``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FLUX_KEYS = ("sfc_shf", "sfc_lhf", "sfc_tau_x", "sfc_tau_y",
             "sfc_sw_up", "sfc_sw_down", "sfc_lw_up")
N_TURBULENT = 4      # the first four are the physics-state flux anchors
YEARS = (2015, 2019)
WINDOW_DAYS = 61
N_DAYS, N_HOURS = 365, 24
STORE_FORMAT = 1
_FEB28 = 58          # 0-based noleap day of 28 February


def _leap(y) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0


def noleap_day(t) -> int:
    """0-based day of a 365-day year; 29 February maps to 28 February."""
    t = np.datetime64(t, "D")
    y = int(str(np.datetime64(t, "Y")))
    d = int((t - np.datetime64(f"{y}-01-01", "D")) / np.timedelta64(1, "D"))
    if _leap(y) and d > _FEB28:
        d = _FEB28 if d == _FEB28 + 1 else d - 1
    return d


def climatology_hours(day, years=YEARS):
    """Hourly timestamps averaged into noleap ``day`` (29 Feb never included),
    one list per UTC hour."""
    out = [[] for _ in range(N_HOURS)]
    for y in range(years[0], years[1] + 1):
        date = (np.datetime64(f"{y}-01-01", "D")
                + np.timedelta64(day + (1 if _leap(y) and day > _FEB28 else 0), "D"))
        for h in range(N_HOURS):
            out[h].append(np.datetime64(date, "h") + np.timedelta64(h, "h"))
    return out


def smooth_days(raw, window=WINDOW_DAYS):
    """Centred running mean of odd ``window`` days along axis 0, wrapping."""
    if window % 2 != 1 or window > raw.shape[0]:
        raise ValueError(f"window {window} must be odd and <= {raw.shape[0]}")
    h = window // 2
    ext = np.concatenate([raw[-h:], raw, raw[:h]], axis=0).astype(np.float64)
    c = np.cumsum(ext, axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c], axis=0)
    return (c[window:] - c[:-window]) / window


def store_tag(grid) -> str:
    return hashlib.sha256(json.dumps([
        STORE_FORMAT, list(FLUX_KEYS), list(YEARS), WINDOW_DAYS,
        int(grid.n_lat), int(grid.n_lon),
        np.asarray(grid.lat).tolist(), np.asarray(grid.lon).tolist()]).encode()
    ).hexdigest()[:16]


def store_path(root, grid) -> Path:
    return Path(root) / f"era5_flux_clim_{store_tag(grid)}.zarr"


def window_frames(clim, t0, n_hours):
    """``(n_hours + 1, 7, n_columns)`` hourly climatology from ``t0`` (a whole
    UTC hour): frame k is slot (noleap day, hour) of ``t0 + k`` hours."""
    t0 = np.datetime64(t0, "h")
    days, rows = {}, []
    for k in range(int(n_hours) + 1):
        t = t0 + np.timedelta64(k, "h")
        d = noleap_day(t)
        if d not in days:
            days[d] = np.asarray(clim[d])
        rows.append(days[d][int(str(t)[11:13])])
    return np.stack(rows)


def frames_at(frames, elapsed_s):
    """Linear interpolation of hourly ``frames`` at ``elapsed_s`` seconds
    after the first frame (traced; clamped to the frames)."""
    import jax.numpy as jnp

    x = jnp.asarray(elapsed_s, jnp.float64) / 3600.0
    n = frames.shape[0]
    i0 = jnp.clip(jnp.floor(x), 0, n - 2).astype(jnp.int32)
    w = jnp.clip(x - i0, 0.0, 1.0).astype(frames.dtype)
    return frames[i0] * (1 - w) + frames[i0 + 1] * w


class FluxClimatology:
    """Read side; ``frames(t0, n_hours)`` for one sample."""

    def __init__(self, path):
        import zarr

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"flux climatology {self.path} does not exist; build it with "
                "scripts/data/build_wb_flux_climatology.py (SLURM).")
        g = zarr.open_group(str(self.path), mode="r")
        if not bool(g.attrs.get("complete", False)):
            raise RuntimeError(f"flux climatology {self.path} is not finalised.")
        self._clim = g["clim"]

    def frames(self, t0, n_hours):
        return window_frames(self._clim, t0, n_hours)
