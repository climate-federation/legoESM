"""CAM ``lininterp`` with boundary extrapolation, and the cloud-optics path guard.

Shared by the CAM6 cloud-optics ports (Mitchell ice/snow, gammadist liquid).
Port of CESM2.1 CAM ``utils/interpolate_data.F90`` ``lininterp_init`` with
``extrap_method_bndry`` (:178-212) and its interior bracket search (:263-286),
for monotone grids in either direction:

* increasing grid: y <= x[0] -> x[0]'s value; y > x[-1] -> x[-1]'s value;
  interior bracket x[j] < y <= x[j+1];
* decreasing grid: y > x[0] -> x[0]'s value; y <= x[-1] -> x[-1]'s value;
  interior bracket x[j] >= y > x[j+1];
* weights wgts = (x[j+1]-y)/(x[j+1]-x[j]) on j, wgtn = (y-x[j])/(x[j+1]-x[j]) on
  j+1; a boundary point takes weight 1 on the end node.

The bracket is CAM's, not ``jnp.interp``'s (which takes the right interval at an
exact knot): same values, but the AD slope at a knot follows CAM's interval.

``lininterp_init`` is the host (numpy, float64) start-up form used by the CAM6
prescribed-aerosol mapping: ``extrap_method_bndry`` and ``extrap_method_cycle``
(:213-247, increasing input grid only) plus every ``endrun`` check of
``lininterp_init`` (:93-122, :286-301), which raise ``ValueError`` here.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# CAM6 cloud_rad_props.F90:491/:641 (ice), :677/:720 (liquid): a cell whose
# in-cloud water path is below this gets zero optics.
_PATH_MIN = 1.0e-80   # [kg/m2]


def bndry_weights(x, y, increasing: bool, xp=jnp):
    """Indices (jm, jp) and weights (wm, wp) of CAM's boundary-extrapolating
    linear interpolation of ``y`` (...) on grid ``x``: (n,) shared, or (..., n)
    one grid per point.  ``increasing`` is the grid's direction (static).
    ``xp`` is the array module (``numpy`` for host float64 start-up use)."""
    x = xp.asarray(x)
    y = xp.asarray(y)
    n = x.shape[-1]
    below = (x < y[..., None]) if increasing else (x >= y[..., None])
    j = xp.clip(xp.sum(below, axis=-1) - 1, 0, n - 2)
    if x.ndim == 1:
        x0, x1, first, last = x[j], x[j + 1], x[0], x[n - 1]
    else:
        x0 = xp.take_along_axis(x, j[..., None], axis=-1)[..., 0]
        x1 = xp.take_along_axis(x, j[..., None] + 1, axis=-1)[..., 0]
        first, last = x[..., 0], x[..., n - 1]
    if increasing:
        lo, hi = y <= first, y > last
    else:
        lo, hi = y > first, y <= last
    out = lo | hi
    dx = x1 - x0
    wm = xp.where(out, 1.0, (x1 - y) / dx)
    wp = xp.where(out, 0.0, (y - x0) / dx)
    jm = xp.where(lo, 0, xp.where(hi, n - 1, j))
    jp = xp.where(lo, 0, xp.where(hi, n - 1, j + 1))
    return jm, jp, wm, wp


def lininterp_init(yin, yout, extrap_method: int, cyclicmin=None, cyclicmax=None):
    """Host port of CAM ``lininterp_init`` for an increasing ``yin`` (n,) and
    targets ``yout`` (m,): returns numpy (jjm, jjp, wgts, wgtn), 0-based.
    ``extrap_method`` 1 = bndry, 2 = cycle on [cyclicmin, cyclicmax) (CAM's
    defaults 0 and 360 when absent)."""
    yin = np.asarray(yin, dtype=np.float64)
    yout = np.asarray(yout, dtype=np.float64)
    nin = yin.shape[0]
    if nin < 2:
        raise ValueError("LININTERP: Must have at least 2 input points for interpolation")
    cmin = 0.0 if cyclicmin is None else float(cyclicmin)
    cmax = 360.0 if cyclicmax is None else float(cyclicmax)  # coeff-ok: CAM's default cyclic max (degrees)
    if cmax <= cmin:
        raise ValueError("LININTERP: cyclic min value must be < max value")
    icount = int(np.sum(yin[:-1] > yin[1:]))
    if icount == nin - 1:
        raise ValueError("LININTERP: decreasing input grid is refused (user decision 2026-10-16). "
                         "Reverse the coordinate and the data along that axis so the coordinate "
                         "increases, then call again.")
    if icount > 0:
        raise ValueError("LININTERP: Non-monotonic input coordinate array found")
    if extrap_method not in (1, 2):
        raise ValueError(f"LININTERP: extrap_method {extrap_method} not ported (1 = bndry, 2 = cycle)")
    jjm, jjp, wgts, wgtn = bndry_weights(yin, yout, True, xp=np)
    lo = yout <= yin[0]
    hi = yout > yin[-1]
    interior = (yout > yin[jjm]) & (yout <= yin[jjp]) & (jjp == jjm + 1)
    if extrap_method == 2:
        dyinwrap = yin[0] + (cmax - cmin) - yin[-1]
        avgdyin = abs(yin[-1] - yin[0]) / (nin - 1.0)
        ratio = dyinwrap / avgdyin
        if ratio < 0.9 or ratio > 1.1:  # coeff-ok: CAM lininterp_init dyinwrap check
            raise ValueError(f"Lininterp: Bad dyinwrap value = {dyinwrap} avg= {avgdyin}")
        wrap = lo | hi
        jjm = np.where(wrap, nin - 1, jjm)
        jjp = np.where(wrap, 0, jjp)
        wgts = np.where(lo, (yin[0] - yout) / dyinwrap,
                        np.where(hi, (yin[0] + (cmax - cmin) - yout) / dyinwrap, wgts))
        wgtn = np.where(lo, (yout + (cmax - cmin) - yin[-1]) / dyinwrap,
                        np.where(hi, (yout - yin[-1]) / dyinwrap, wgtn))
    found = lo | hi | interior
    ratio = wgts + wgtn
    bad = (ratio < 0.9) | (ratio > 1.1)  # coeff-ok: CAM lininterp_init weight-sum check
    if np.any(bad):
        raise ValueError("Bad weight computed in LININTERP_init")
    if not np.all(found):
        raise ValueError("LININTERP: Point found without interp indices")
    return jjm, jjp, wgts, wgtn


def lininterp_2d(arr, lon_wgts, lat_wgts):
    """CAM ``lininterp3d2d`` / ``lininterp2d1d`` (:471-472, :432-433), same
    product and sum order: ``arr`` (..., nlat, nlon) -> (..., m)."""
    iim, iip, wgtw, wgte = lon_wgts
    jjm, jjp, wgts, wgtn = lat_wgts
    arr = np.asarray(arr)
    return (arr[..., jjm, iim] * wgtw * wgts + arr[..., jjm, iip] * wgte * wgts
            + arr[..., jjp, iim] * wgtw * wgtn + arr[..., jjp, iip] * wgte * wgtn)


def interp1(x, f, y, increasing: bool = True):
    """``lininterp1d`` of table row ``f`` (n,) on grid ``x`` at ``y``."""
    jm, jp, wm, wp = bndry_weights(x, y, increasing)
    f = jnp.asarray(f)
    return f[jm] * wm + f[jp] * wp


def path_on(path):
    """CAM's ``path >= 1e-80``.  1e-80 underflows to 0 in fp32, where every
    positive path exceeds it, so the predicate there is ``path > 0``."""
    path = jnp.asarray(path)
    thr = jnp.asarray(_PATH_MIN, dtype=path.dtype)
    return jnp.where(thr > 0, path >= thr, path > 0)
