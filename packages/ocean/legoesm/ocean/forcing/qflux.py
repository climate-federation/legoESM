"""Slab-ocean q-flux (ocean-heat-transport convergence) climatology loader.

The standard CMIP slab-ocean boundary condition: a spatially+seasonally
varying prescribed heat-flux convergence ``q_flux(x, month)`` [W/m2, +INTO the
mixed layer] so a mixed-layer slab reproduces the observed SST climatology.
Derived (offline) as ``q_flux = -(monthly-mean net downward surface heat flux
from an AMIP-SST-forced run)`` — see ``scripts/data/generate_qflux_climatology.py``.

This module loads a ``(12, nlat, nlon)`` climatology NetCDF, bilinearly regrids
it to the ocean grid (reusing the AMIP regrid), and interpolates it to a model
day with the SAME mid-month cyclic Taylor (2000) scheme the AMIP SST path uses
(``climatology_interp_indices``) — minus the SST-specific freezing clamp.  The
coupled driver threads the per-step map into the slab step (no recompile — the
slab runs eagerly).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.forcing.amip import (
    climatology_interp_indices,
    regrid_monthly_latlon_to_grid,
)


class QFluxForcing(NamedTuple):
    """Loaded q-flux climatology, regridded to the ocean grid.

    ``times`` — mid-month day anchors [days], same axis convention as the AMIP
    forcing (first-record-relative for a ≤12-month climatology, so the annual
    cycle wraps).  ``qflux`` — ``(ntime, *ocean_grid_shape_2d)`` [W/m2],
    +INTO the mixed layer.
    """
    times: jnp.ndarray
    qflux: jnp.ndarray


def _detect_qflux_varname(ds) -> str:
    """First matching q-flux variable name in the dataset."""
    for name in ("q_flux", "qflux", "qdp", "oht_convergence", "hfcorr"):
        if name in ds.variables:
            return name
    raise ValueError(
        "q-flux file has no recognized variable "
        "(tried q_flux/qflux/qdp/oht_convergence/hfcorr); "
        f"has {list(ds.variables)}"
    )


def load_qflux_climatology(path: str, grid) -> QFluxForcing:
    """Load a ``(ntime, nlat, nlon)`` q-flux NetCDF, regridded to ``grid``.

    ``path`` — NetCDF with a q-flux variable [W/m2] over ``(time, lat, lon)``.
    ``grid`` — the OCEAN grid (exposes ``grid_lat``/``grid_lon`` in radians).
    Returns a :class:`QFluxForcing` with a first-record-relative day axis so the
    ≤12-month climatology wraps annually (``climatology_interp_indices``).
    """
    import xarray as xr

    ds = xr.open_dataset(path, decode_times=False)
    try:
        var = _detect_qflux_varname(ds)
        data = np.asarray(ds[var].values, dtype=np.float64)  # (ntime, nlat, nlon)
        if data.ndim != 3:
            raise ValueError(
                f"q-flux variable {var!r} must be (time, lat, lon); "
                f"got shape {data.shape}")
        lat_src = np.asarray(ds["lat"].values, dtype=np.float64)
        lon_src = np.asarray(ds["lon"].values, dtype=np.float64)
        # Ascending source lat for RegularGridInterpolator (files are usually
        # +90..-90); flip lat axis + data together if descending.
        if lat_src[0] > lat_src[-1]:
            lat_src = lat_src[::-1]
            data = data[:, ::-1, :]
        time_days = np.asarray(ds["time"].values, dtype=np.float64)
    finally:
        ds.close()

    qflux = regrid_monthly_latlon_to_grid(data, lat_src, lon_src, grid)
    # Follow the active precision policy (float32 by default; float64 under
    # JAX_ENABLE_X64).  A 12-month climatology's day axis (15..349) and the
    # modulo wrap in climatology_interp_indices are well within float32.
    return QFluxForcing(
        times=jnp.asarray(time_days),
        qflux=jnp.asarray(qflux),
    )


def qflux_at_time(forcing: QFluxForcing, day: float) -> jnp.ndarray:
    """Interpolate the q-flux climatology to ``day`` [W/m2, +into mixed layer].

    Same cyclic mid-month scheme as the AMIP SST path
    (``climatology_interp_indices``), WITHOUT the SST-specific freezing clamp —
    q-flux is signed and physically unbounded, so no clamp is applied.
    """
    times = forcing.times
    if times.shape[0] == 1:
        return forcing.qflux[0]
    idx, idx_next, weight = climatology_interp_indices(times, day)
    return (1.0 - weight) * forcing.qflux[idx] + weight * forcing.qflux[idx_next]
