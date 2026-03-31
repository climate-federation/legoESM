"""Halo exchange for the latitude-longitude grid.

- Longitude: periodic wrap
- Latitude: pole-folding boundary condition
  - Scalars: fold across pole with 180 deg longitude shift (no sign change)
  - Vectors: fold across pole with 180 deg longitude shift AND sign reversal
    (east/north directions flip when crossing the pole)
"""

from __future__ import annotations

import jax.numpy as jnp


def _fold_pole_rows(
    data: jnp.ndarray,
    halo: int,
    negate: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute pole-folded halo rows for south and north poles.

    When crossing a pole, a point at latitude (90+delta) maps to
    latitude (90-delta) at longitude+180 deg.  For vector components
    the sign also reverses.

    Parameters
    ----------
    data : shape (n_lat, n_lon_padded)
        Field (already padded in longitude).
    halo : int
    negate : bool
        If True, negate the folded values (for vector components).

    Returns
    -------
    south, north : halo rows ready for concatenation.
    """
    half = data.shape[1] // 2
    sign = -1.0 if negate else 1.0

    # South pole: ghost rows mirror the first `halo` rows, reversed in
    # latitude order and shifted by 180 deg in longitude.
    south = sign * jnp.roll(data[:halo][::-1], half, axis=1)

    # North pole: mirror the last `halo` rows.
    north = sign * jnp.roll(data[-halo:][::-1], half, axis=1)

    return south, north


def pad_halo_latlon(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a scalar field with halo cells using pole-folding.

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

    # Latitude: pole-folding (scalar — no sign change)
    south, north = _fold_pole_rows(data_lon, halo, negate=False)
    padded = jnp.concatenate([south, data_lon, north], axis=0)
    return padded


def pad_halo_latlon_vector(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a single vector component with pole-folding and sign reversal.

    Use this for metric-weighted velocity fields (u*dx, v*dy) or any
    quantity that changes sign when crossing a pole.

    Parameters
    ----------
    data : jax.Array
        Vector component field, shape (n_lat, n_lon).
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

    # Latitude: pole-folding (vector — negate)
    south, north = _fold_pole_rows(data_lon, halo, negate=True)
    padded = jnp.concatenate([south, data_lon, north], axis=0)
    return padded


def pad_halo_vector_latlon(
    u: jnp.ndarray,
    v: jnp.ndarray,
    halo: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad vector components with pole-folding and sign reversal.

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
    return pad_halo_latlon_vector(u, halo), pad_halo_latlon_vector(v, halo)


# ==============================================================================
# Native 3D halo padding (batch across levels, no vmap)
# ==============================================================================


def _fold_pole_rows_3d(
    data: jnp.ndarray,
    halo: int,
    negate: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pole-fold for 3D arrays, shape (n_lat, n_lon_padded, nlev).

    Same logic as _fold_pole_rows but keeps the level axis intact.
    """
    half = data.shape[1] // 2
    sign = -1.0 if negate else 1.0

    south = sign * jnp.roll(data[:halo][::-1], half, axis=1)
    north = sign * jnp.roll(data[-halo:][::-1], half, axis=1)
    return south, north


def pad_halo_latlon_3d(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a scalar 3D field with halo cells using pole-folding.

    Parameters
    ----------
    data : jax.Array
        Scalar field, shape (n_lat, n_lon, nlev).
    halo : int
        Number of halo cells on each side (default 1).

    Returns
    -------
    jax.Array : Padded field, shape (n_lat+2*halo, n_lon+2*halo, nlev).
    """
    # Longitude: periodic wrap (axis 1)
    data_lon = jnp.concatenate(
        [data[:, -halo:], data, data[:, :halo]], axis=1,
    )
    # Latitude: pole-folding (scalar — no sign change)
    south, north = _fold_pole_rows_3d(data_lon, halo, negate=False)
    return jnp.concatenate([south, data_lon, north], axis=0)


def pad_halo_latlon_vector_3d(data: jnp.ndarray, halo: int = 1) -> jnp.ndarray:
    """Pad a single 3D vector component with pole-folding and sign reversal.

    Parameters
    ----------
    data : jax.Array
        Vector component field, shape (n_lat, n_lon, nlev).
    halo : int
        Number of halo cells on each side (default 1).

    Returns
    -------
    jax.Array : Padded field, shape (n_lat+2*halo, n_lon+2*halo, nlev).
    """
    data_lon = jnp.concatenate(
        [data[:, -halo:], data, data[:, :halo]], axis=1,
    )
    south, north = _fold_pole_rows_3d(data_lon, halo, negate=True)
    return jnp.concatenate([south, data_lon, north], axis=0)


def pad_halo_vector_latlon_3d(
    u: jnp.ndarray,
    v: jnp.ndarray,
    halo: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad 3D vector components with pole-folding and sign reversal.

    Parameters
    ----------
    u, v : jax.Array
        Vector components, shape (n_lat, n_lon, nlev).
    halo : int
        Number of halo cells on each side (default 1).

    Returns
    -------
    (u_padded, v_padded) : each shape (n_lat+2*halo, n_lon+2*halo, nlev).
    """
    return pad_halo_latlon_vector_3d(u, halo), pad_halo_latlon_vector_3d(v, halo)
