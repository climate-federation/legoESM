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
"""
from __future__ import annotations

import jax.numpy as jnp

# CAM6 cloud_rad_props.F90:491/:641 (ice), :677/:720 (liquid): a cell whose
# in-cloud water path is below this gets zero optics.
_PATH_MIN = 1.0e-80   # [kg/m2]


def bndry_weights(x, y, increasing: bool):
    """Indices (jm, jp) and weights (wm, wp) of CAM's boundary-extrapolating
    linear interpolation of ``y`` (...) on grid ``x``: (n,) shared, or (..., n)
    one grid per point.  ``increasing`` is the grid's direction (static)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n = x.shape[-1]
    below = (x < y[..., None]) if increasing else (x >= y[..., None])
    j = jnp.clip(jnp.sum(below, axis=-1) - 1, 0, n - 2)
    if x.ndim == 1:
        x0, x1, first, last = x[j], x[j + 1], x[0], x[n - 1]
    else:
        x0 = jnp.take_along_axis(x, j[..., None], axis=-1)[..., 0]
        x1 = jnp.take_along_axis(x, j[..., None] + 1, axis=-1)[..., 0]
        first, last = x[..., 0], x[..., n - 1]
    if increasing:
        lo, hi = y <= first, y > last
    else:
        lo, hi = y > first, y <= last
    out = lo | hi
    dx = x1 - x0
    wm = jnp.where(out, 1.0, (x1 - y) / dx)
    wp = jnp.where(out, 0.0, (y - x0) / dx)
    jm = jnp.where(lo, 0, jnp.where(hi, n - 1, j))
    jp = jnp.where(lo, 0, jnp.where(hi, n - 1, j + 1))
    return jm, jp, wm, wp


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
