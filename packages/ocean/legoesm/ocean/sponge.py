"""Sponge layer relaxation for ocean models.

Provides a ``SpongeForcing`` container and utilities for computing
spatially varying relaxation coefficients.  Sponge layers nudge the
model state toward a reference profile near domain boundaries, damping
wave reflections and preventing wall instabilities in channel
experiments (standard practice in MITgcm RBCS, MOM6 ALE_sponge).

The relaxation is applied as a tendency::

    dT/dt += gamma(x,y) * (T_ref(x,y,z) - T)

where gamma [1/s] ramps from zero in the interior to 1/tau at the
boundary.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class SpongeForcing(NamedTuple):
    """Sponge layer relaxation fields.

    Parameters
    ----------
    gamma : array
        Relaxation rate [1/s].  Shape ``(n_lat, n_lon)`` for lat-lon
        or ``(nCells,)`` for MPAS.  Zero in the interior, ramping to
        ``1/tau`` near boundaries.
    T_ref : array
        Reference temperature, same shape as the model T field.
    S_ref : array
        Reference salinity, same shape as the model S field.
    u_ref : array or None
        Reference u velocity (optional).
    v_ref : array or None
        Reference v velocity (optional, not used for MPAS edge velocity).
    """
    gamma: jnp.ndarray
    T_ref: jnp.ndarray
    S_ref: jnp.ndarray
    u_ref: jnp.ndarray | None = None
    v_ref: jnp.ndarray | None = None


def compute_sponge_gamma(
    lat_deg: np.ndarray,
    lat_south: float,
    lat_north: float,
    width_deg: float = 2.0,
    timescale_days: float = 1.0,
) -> np.ndarray:
    """Quadratic sponge ramp as a function of latitude — the shared kernel.

    Ramp from 0 in the interior to ``1/tau`` at the south/north walls, with
    south taking precedence in the (degenerate) overlap. Works on any-shaped
    ``lat_deg`` array; the grid-specific wrappers below supply the latitudes.

    Parameters
    ----------
    lat_deg : ndarray
        Latitudes [degrees], any shape.
    lat_south, lat_north : float
        Domain boundaries [degrees].
    width_deg : float
        Sponge zone width [degrees].
    timescale_days : float
        Relaxation e-folding timescale [days].

    Returns
    -------
    gamma : ndarray, same shape as ``lat_deg``
        Relaxation coefficient [1/s].
    """
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    tau = timescale_days * 86400.0
    dist_south = lat_deg - lat_south
    dist_north = lat_north - lat_deg
    # Nested where reproduces the original ``if south elif north`` precedence.
    return np.where(
        dist_south < width_deg,
        (1.0 - dist_south / width_deg) ** 2 / tau,
        np.where(
            dist_north < width_deg,
            (1.0 - dist_north / width_deg) ** 2 / tau,
            0.0,
        ),
    )


def compute_sponge_gamma_latlon(
    grid,
    lat_south: float,
    lat_north: float,
    width_deg: float = 2.0,
    timescale_days: float = 1.0,
) -> np.ndarray:
    """Sponge relaxation coefficient on a lat-lon grid, shape ``(n_lat, n_lon)``
    (constant in longitude). Thin wrapper over :func:`compute_sponge_gamma`."""
    lat_deg = np.degrees(np.asarray(grid.lat))
    gamma_lat = compute_sponge_gamma(
        lat_deg, lat_south, lat_north, width_deg, timescale_days,
    )
    return np.broadcast_to(
        gamma_lat[:, None], (grid.n_lat, grid.n_lon),
    ).copy()


def compute_sponge_gamma_mpas(
    mesh,
    lat_south: float,
    lat_north: float,
    width_deg: float = 2.0,
    timescale_days: float = 1.0,
) -> np.ndarray:
    """Sponge relaxation coefficient on an MPAS Voronoi mesh, shape
    ``(nCells,)``. Thin wrapper over :func:`compute_sponge_gamma`."""
    lat_deg = np.degrees(np.asarray(mesh.latCell))
    return compute_sponge_gamma(
        lat_deg, lat_south, lat_north, width_deg, timescale_days,
    )
