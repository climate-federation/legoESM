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


def compute_sponge_gamma_latlon(
    grid,
    lat_south: float,
    lat_north: float,
    width_deg: float = 2.0,
    timescale_days: float = 1.0,
) -> np.ndarray:
    """Compute sponge relaxation coefficient on a lat-lon grid.

    Quadratic ramp from 0 in the interior to ``1/tau`` at the walls.

    Parameters
    ----------
    grid : LatLonGrid
    lat_south, lat_north : float
        Domain boundaries [degrees].
    width_deg : float
        Sponge zone width [degrees].
    timescale_days : float
        Relaxation e-folding timescale [days].

    Returns
    -------
    gamma : ndarray, shape (n_lat, n_lon)
        Relaxation coefficient [1/s].
    """
    lat_deg = np.degrees(np.asarray(grid.lat))
    tau = timescale_days * 86400.0
    gamma = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float64)

    for i, lat in enumerate(lat_deg):
        dist_south = lat - lat_south
        dist_north = lat_north - lat
        if dist_south < width_deg:
            gamma[i, :] = (1.0 - dist_south / width_deg) ** 2 / tau
        elif dist_north < width_deg:
            gamma[i, :] = (1.0 - dist_north / width_deg) ** 2 / tau

    return gamma


def compute_sponge_gamma_mpas(
    mesh,
    lat_south: float,
    lat_north: float,
    width_deg: float = 2.0,
    timescale_days: float = 1.0,
) -> np.ndarray:
    """Compute sponge relaxation coefficient on an MPAS Voronoi mesh.

    Same quadratic ramp as the lat-lon version, applied per cell.

    Parameters
    ----------
    mesh : VoronoiMesh
    lat_south, lat_north : float
        Domain boundaries [degrees].
    width_deg : float
        Sponge zone width [degrees].
    timescale_days : float
        Relaxation e-folding timescale [days].

    Returns
    -------
    gamma : ndarray, shape (nCells,)
        Relaxation coefficient [1/s].
    """
    lat_deg = np.degrees(np.asarray(mesh.latCell))
    tau = timescale_days * 86400.0
    n_cells = lat_deg.shape[0]
    gamma = np.zeros(n_cells, dtype=np.float64)

    for i, lat in enumerate(lat_deg):
        dist_south = lat - lat_south
        dist_north = lat_north - lat
        if dist_south < width_deg:
            gamma[i] = (1.0 - dist_south / width_deg) ** 2 / tau
        elif dist_north < width_deg:
            gamma[i] = (1.0 - dist_north / width_deg) ** 2 / tau

    return gamma
