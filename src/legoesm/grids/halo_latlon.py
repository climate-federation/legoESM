"""Halo exchange for the latitude-longitude grid.

Much simpler than cubed-sphere halo exchange:
- Longitude: periodic wrap (no face connectivity)
- Latitude: zero-gradient (Neumann) boundary condition at poles
- No angle rotation needed for vectors
"""

from __future__ import annotations

import jax.numpy as jnp


def pad_halo_latlon(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a scalar field with halo cells on each side.

    Parameters
    ----------
    data : jax.Array
        Scalar field, shape (n_lat, n_lon).
    halo : int
        Number of halo cells on each side (default 1).

    Returns
    -------
    jax.Array : Padded field, shape (n_lat+2*halo, n_lon+2*halo).
    """
    # Longitude: periodic wrap
    data_lon = jnp.concatenate(
        [data[:, -halo:], data, data[:, :halo]], axis=1,
    )

    # Latitude: zero-gradient (repeat edge rows)
    south = jnp.tile(data_lon[:1, :], (halo, 1))
    north = jnp.tile(data_lon[-1:, :], (halo, 1))
    padded = jnp.concatenate([south, data_lon, north], axis=0)
    return padded


def pad_halo_vector_latlon(
    u: jnp.ndarray,
    v: jnp.ndarray,
    halo: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad vector components with halo cells.

    On a lat-lon grid no angle rotation is needed.
    Longitude: periodic. Latitude: zero-gradient.

    Parameters
    ----------
    u, v : jax.Array
        Vector components, shape (n_lat, n_lon).
    halo : int
        Number of halo cells on each side (default 1).

    Returns
    -------
    (u_padded, v_padded) : each shape (n_lat+2*halo, n_lon+2*halo).
    """
    return pad_halo_latlon(u, halo), pad_halo_latlon(v, halo)
