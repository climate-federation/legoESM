"""Fourier polar filter for lat-lon grid CFL stability.

The polar CFL problem: dx = R*dlon*cos(lat) -> 0 at poles, making
explicit time-stepping unstable unless high-wavenumber modes are damped.

The filter truncates Fourier modes in longitude at each latitude,
keeping only wavenumbers that satisfy the CFL condition for the fastest
wave (external gravity wave) at the given time step.

The maximum stable wavenumber at each latitude is:
    k_max = CFL_limit * R * cos(lat) / (c_max * dt)

where c_max is the maximum wave speed (external gravity wave) and
CFL_limit is the RK3 stability limit (sqrt(3) ~ 1.73).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid


def compute_polar_filter_mask(
    grid: LatLonGrid,
    dt: float = 600.0,
    max_wave_speed: float = 300.0,
    cutoff_lat_deg: float = 60.0,
    safety_factor: float = 0.85,
    *,
    is_v_face: bool = False,
) -> jnp.ndarray:
    """Precompute the Fourier polar filter mask.

    Returns a 2D mask (n_lat, n_freq) for cell-centered fields or
    (n_lat+1, n_freq) for v-face fields, where n_freq = n_lon//2 + 1,
    suitable for multiplying the output of jnp.fft.rfft.

    Parameters
    ----------
    grid : LatLonGrid
        The lat-lon grid.
    dt : float
        Time step [seconds].
    max_wave_speed : float
        Maximum wave speed [m/s] (external gravity wave speed).
    cutoff_lat_deg : float
        Latitude (degrees) beyond which filtering is applied.
    safety_factor : float
        Fraction of the theoretical CFL limit to use (< 1 for margin).
    is_v_face : bool, optional
        When True, build a mask for v-face fields (lat-interface
        values, shape ``(n_lat+1, ...)``) using ``grid.cos_lat_v`` /
        the half-cell-offset lat-interface coordinates.  Codex review
        Stage 3-E round 2 caught that applying the cell-centered mask
        to v-face indices introduces a half-cell lat offset that
        admits ``k`` modes the actual v-face CFL forbids.

    Returns
    -------
    jax.Array : Float mask, shape (n_lat, n_freq) if ``is_v_face``
                is False, else (n_lat+1, n_freq).
    """
    # RK3 stability limit for centered differences: sqrt(3)
    cfl_limit = jnp.sqrt(3.0) * safety_factor

    if is_v_face:
        # v-face: shape (n_lat+1,).  ``grid.cos_lat_v`` is zero at the
        # global poles by construction (wall BC in regular lat-lon).
        # lat_v = (lat[:-1] + lat[1:]) / 2 in the interior, ±π/2 at
        # the global poles.  Use ``arcsin(sin(lat_v))`` would be
        # exact, but here we mirror the construction in
        # ``create_latlon_grid``.
        cos_lat_face = grid.cos_lat_v
        # Interior lat-face values, then pad with ±π/2 at the poles.
        lat_v_interior = 0.5 * (grid.lat[:-1] + grid.lat[1:])
        lat_face = jnp.concatenate([
            jnp.asarray([-jnp.pi / 2.0]),
            lat_v_interior,
            jnp.asarray([jnp.pi / 2.0]),
        ])
    else:
        cos_lat_face = grid.cos_lat
        lat_face = grid.lat

    # CFL-based max wavenumber at each latitude
    max_k_cfl = cfl_limit * grid.radius * cos_lat_face / (max_wave_speed * dt)
    max_k_cfl = jnp.clip(max_k_cfl, 1.0, float(grid.n_lon // 2))

    # Only apply filtering poleward of cutoff
    cutoff_rad = jnp.deg2rad(cutoff_lat_deg)
    is_poleward = jnp.abs(lat_face) > cutoff_rad
    max_k = jnp.where(is_poleward, max_k_cfl, float(grid.n_lon // 2))

    # Build mask
    n_freq = grid.n_lon // 2 + 1
    wavenumbers = jnp.arange(n_freq)
    mask = (wavenumbers[None, :] <= max_k[:, None]).astype(jnp.float32)

    return mask


def fourier_filter(
    field: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Apply Fourier polar filter to a 2D field using a precomputed mask.

    Parameters
    ----------
    field : jax.Array
        2D field, shape (n_lat, n_lon).
    grid : LatLonGrid
        The lat-lon grid.
    mask : jax.Array
        Precomputed filter mask, shape (n_lat, n_freq).

    Returns
    -------
    jax.Array : Filtered field, shape (n_lat, n_lon).
    """
    field_hat = jnp.fft.rfft(field, axis=-1)
    return jnp.fft.irfft(field_hat * mask, n=grid.n_lon, axis=-1)


def fourier_filter_3d(
    field_3d: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Apply Fourier polar filter to a 3D field using a precomputed mask.

    Parameters
    ----------
    field_3d : jax.Array
        3D field, shape (n_lat, n_lon, nlev).
    grid : LatLonGrid
        The lat-lon grid.
    mask : jax.Array
        Precomputed filter mask, shape (n_lat, n_freq).

    Returns
    -------
    jax.Array : Filtered field, shape (n_lat, n_lon, nlev).
    """
    field_hat = jnp.fft.rfft(field_3d, axis=1)
    return jnp.fft.irfft(field_hat * mask[:, :, None], n=grid.n_lon, axis=1)
