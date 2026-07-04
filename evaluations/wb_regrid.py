"""Offline regridding of model-grid fields onto the WeatherBench-2 common grid.

Eval-only (NOT in the JIT/AD path): uses numpy + scipy ``RegularGridInterpolator``,
mirroring the linear-interpolation core of
``training.era5_to_state.regrid_latlon_to_gaussian`` (which goes the OTHER
direction, lat-lon -> Gaussian). Bilinear, periodic in longitude, and robust to
either latitude ordering (Gaussian S->N or ERA5 N->S).

The scorer regrids BOTH the model forecast and the ERA5 verification through
this function onto the same WB2 grid, so absolute grid orientation only has to
be self-consistent.
"""
from __future__ import annotations

import numpy as np

__all__ = ["WB2_RESOLUTION_DEG", "wb2_grid", "regrid_to_wb2"]

WB2_RESOLUTION_DEG = 1.5   # WB2 headline common grid resolution [deg]


def wb2_grid(resolution_deg: float = WB2_RESOLUTION_DEG):
    """WB2 target grid coordinates in DEGREES.

    Returns ``(lat, lon)`` with ``lat`` ascending -90..90 inclusive and ``lon``
    0..360 exclusive of the wrap point. At 1.5 deg this is 121 x 240.
    """
    n_lat = int(round(180.0 / resolution_deg)) + 1
    n_lon = int(round(360.0 / resolution_deg))
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    return lat, lon


def regrid_to_wb2(field, src_lat_deg, src_lon_deg, *,
                  resolution_deg: float = WB2_RESOLUTION_DEG, mask: bool = False):
    """Bilinearly regrid a ``(n_lat_src, n_lon_src)`` field onto the WB2 grid.

    Parameters
    ----------
    field : array (n_lat_src, n_lon_src)
        Source field on a lat-lon (or Gaussian lat) grid.
    src_lat_deg, src_lon_deg : 1-D arrays [deg]
        Source coordinates. Latitude may be ascending or descending (handled);
        longitude in [0, 360). Longitude is treated periodically so the 360/0
        seam interpolates without a discontinuity.
    resolution_deg : float
        WB2 grid resolution (default 1.5).
    mask : bool
        If True, treat ``field`` as a boolean mask: regrid as float and return a
        bool array thresholded at 0.5 (a target cell is valid iff it interpolates
        from a majority of valid source cells).

    Returns
    -------
    (field_wb2, wb2_lat, wb2_lon)
        ``field_wb2`` shape (n_lat_tgt, n_lon_tgt); coordinates in degrees.
    """
    from scipy.interpolate import RegularGridInterpolator

    field = np.asarray(field, dtype=np.float64)
    src_lat = np.asarray(src_lat_deg, dtype=np.float64)
    src_lon = np.asarray(src_lon_deg, dtype=np.float64)

    # RegularGridInterpolator needs strictly ascending axes: flip a descending
    # latitude (ERA5 is N->S) along with the field.
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        field = field[::-1, :]

    # Pad longitude periodically (append the first column at lon+360) so the
    # 360/0 seam interpolates.
    lon_p = np.concatenate([src_lon, src_lon[:1] + 360.0])
    field_p = np.concatenate([field, field[:, :1]], axis=1)

    interp = RegularGridInterpolator(
        (src_lat, lon_p), field_p, method="linear",
        bounds_error=False, fill_value=None,   # extrapolate poleward of the last lat
    )

    tgt_lat, tgt_lon = wb2_grid(resolution_deg)
    lon2d, lat2d = np.meshgrid(tgt_lon, tgt_lat)          # (n_lat_tgt, n_lon_tgt)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    out = interp(pts).reshape(lat2d.shape)

    if mask:
        out = out >= 0.5
    return out, tgt_lat, tgt_lon
