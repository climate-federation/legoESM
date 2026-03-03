"""Halo exchange for the latitude-longitude grid.

Much simpler than cubed-sphere halo exchange:
- Longitude: periodic wrap (no face connectivity)
- Latitude: zero-gradient (Neumann) boundary condition at poles
- No angle rotation needed for vectors
"""

from __future__ import annotations

import jax.numpy as jnp


def pad_halo_latlon(data: jnp.ndarray) -> jnp.ndarray:
    """Pad a scalar field with one halo cell on each side.

    Parameters
    ----------
    data : jax.Array
        Scalar field, shape (n_lat, n_lon).

    Returns
    -------
    jax.Array : Padded field, shape (n_lat+2, n_lon+2).
    """
    # Longitude: periodic wrap
    data_lon = jnp.concatenate([data[:, -1:], data, data[:, :1]], axis=1)

    # Latitude: zero-gradient (repeat edge rows)
    padded = jnp.concatenate(
        [data_lon[:1, :], data_lon, data_lon[-1:, :]],
        axis=0,
    )
    return padded


def pad_halo_vector_latlon(
    u: jnp.ndarray,
    v: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad vector components with halo cells.

    On a lat-lon grid no angle rotation is needed.
    Longitude: periodic. Latitude: zero-gradient.

    Parameters
    ----------
    u, v : jax.Array
        Vector components, shape (n_lat, n_lon).

    Returns
    -------
    (u_padded, v_padded) : each shape (n_lat+2, n_lon+2).
    """
    return pad_halo_latlon(u), pad_halo_latlon(v)
