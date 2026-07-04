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
    if not (np.isfinite(resolution_deg) and resolution_deg > 0.0):
        raise ValueError(f"resolution_deg must be finite and positive, got {resolution_deg}")
    n_lat_intervals = 180.0 / resolution_deg
    n_lon_cells = 360.0 / resolution_deg
    if not (np.isclose(n_lat_intervals, round(n_lat_intervals))
            and np.isclose(n_lon_cells, round(n_lon_cells))):
        raise ValueError(
            f"resolution_deg={resolution_deg} must divide both 180 and 360 "
            "(e.g. 0.25, 0.5, 1.0, 1.5, 2.0)"
        )
    n_lat = int(round(n_lat_intervals)) + 1
    n_lon = int(round(n_lon_cells))
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    return lat, lon


def regrid_to_wb2(field, src_lat_deg, src_lon_deg, *,
                  resolution_deg: float = WB2_RESOLUTION_DEG, mask: bool = False,
                  mask_threshold: float = 0.5):
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
        bool array thresholded at ``mask_threshold``.
    mask_threshold : float
        Threshold for ``mask=True``. 0.5 = majority (a target cell is valid iff
        the bilinear interpolation of the source mask exceeds 0.5). Use ~1.0 for
        a CONSERVATIVE mask: a target is valid only if ALL contributing source
        cells are valid — required for a validity mask on a bilinearly
        interpolated field, so a "valid" target's field value cannot have been
        contaminated by an invalid (e.g. below-ground/clamped) source cell.

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

    # Pad longitude periodically on BOTH sides so any target longitude in
    # [0, 360) interpolates across the 360/0 seam regardless of where src_lon
    # starts (offset grids with src_lon[0] != 0 included). Prepend the last
    # column at lon-360 and append the first column at lon+360; the result is
    # strictly increasing because src_lon[-1] < src_lon[0] + 360.
    lon_p = np.concatenate([src_lon[-1:] - 360.0, src_lon, src_lon[:1] + 360.0])
    field_p = np.concatenate([field[:, -1:], field, field[:, :1]], axis=1)

    interp = RegularGridInterpolator(
        (src_lat, lon_p), field_p, method="linear",
        bounds_error=False, fill_value=None,   # extrapolate poleward of the last lat
    )

    tgt_lat, tgt_lon = wb2_grid(resolution_deg)
    lon2d, lat2d = np.meshgrid(tgt_lon, tgt_lat)          # (n_lat_tgt, n_lon_tgt)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    out = interp(pts).reshape(lat2d.shape)

    if mask:
        out = out >= mask_threshold
    return out, tgt_lat, tgt_lon
