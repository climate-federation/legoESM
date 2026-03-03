"""Post-hoc conservation correctors for SFNO predictions.

Neural operators do not enforce conservation laws by construction.
These correctors adjust SFNO predictions to preserve:

Atmosphere:
- Global dry air mass (via uniform surface pressure correction)
- Global moisture budget (via proportional humidity correction)
- Non-negative humidity (clipping)

Ocean:
- Global ocean volume (via uniform eta correction)
- Global heat content (via uniform T correction)
- Global salt content (via uniform S correction)

References
----------
- Watt-Meyer et al. (2023). ACE: A fast, skillful learned global
  atmospheric model for climate prediction. arXiv:2310.02074.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.gaussian import GaussianGrid


def correct_dry_air_mass(
    p_s_new: jnp.ndarray,
    p_s_old: jnp.ndarray,
    grid: GaussianGrid,
) -> jnp.ndarray:
    """Correct surface pressure to conserve global dry air mass.

    Applies a spatially uniform additive correction so that the
    global mean surface pressure is preserved.

    Parameters
    ----------
    p_s_new : array, shape (n_lat, n_lon)
        Predicted surface pressure.
    p_s_old : array, shape (n_lat, n_lon)
        Original surface pressure (before SFNO step).
    grid : GaussianGrid
        Grid with Gaussian quadrature weights for area averaging.

    Returns
    -------
    array, shape (n_lat, n_lon)
        Corrected surface pressure.
    """
    # Area weights: w(lat) for Gaussian quadrature, uniform in longitude
    w = grid.weights[:, None]  # (n_lat, 1)
    w_total = jnp.sum(w) * grid.n_lon

    # Global mean difference
    dp = jnp.sum((p_s_new - p_s_old) * w) * grid.n_lon / w_total

    return p_s_new - dp


def correct_moisture(
    q_new: jnp.ndarray,
    q_old: jnp.ndarray,
    p_s: jnp.ndarray,
    dsigma: jnp.ndarray,
    grid: GaussianGrid,
) -> jnp.ndarray:
    """Correct humidity to conserve global moisture budget.

    Applies a proportional scaling so that the column-integrated
    moisture (weighted by surface pressure and layer thickness)
    is preserved globally.

    Parameters
    ----------
    q_new : array, shape (n_lat, n_lon, nlev)
        Predicted specific humidity.
    q_old : array, shape (n_lat, n_lon, nlev)
        Original specific humidity.
    p_s : array, shape (n_lat, n_lon)
        Surface pressure [Pa].
    dsigma : array, shape (nlev,)
        Sigma layer thicknesses.
    grid : GaussianGrid
        Grid for area weighting.

    Returns
    -------
    array, shape (n_lat, n_lon, nlev)
        Corrected humidity.
    """
    w = grid.weights[:, None]  # (n_lat, 1)

    # Column-integrated moisture: integral(q * dp) = p_s * sum(q * dsigma)
    col_old = jnp.sum(q_old * dsigma[None, None, :], axis=-1) * p_s
    col_new = jnp.sum(q_new * dsigma[None, None, :], axis=-1) * p_s

    # Global integrals
    global_old = jnp.sum(col_old * w)
    global_new = jnp.sum(col_new * w)

    # Proportional correction factor
    ratio = global_old / jnp.maximum(global_new, 1e-30)

    return q_new * ratio


def clip_humidity(q: jnp.ndarray) -> jnp.ndarray:
    """Enforce non-negative specific humidity.

    Parameters
    ----------
    q : array
        Specific humidity (any shape).

    Returns
    -------
    array
        Clipped humidity with q >= 0.
    """
    return jnp.maximum(q, 0.0)


# ============================================================================
# Ocean conservation correctors
# ============================================================================

def correct_ocean_volume(
    eta_new: jnp.ndarray,
    eta_old: jnp.ndarray,
    grid: GaussianGrid,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Correct sea surface height to conserve global ocean volume.

    Applies a spatially uniform additive correction to eta so that
    the global integral of eta (over ocean cells) is preserved.

    Parameters
    ----------
    eta_new : array, shape (n_lat, n_lon)
        Predicted sea surface height [m].
    eta_old : array, shape (n_lat, n_lon)
        Original sea surface height [m].
    grid : GaussianGrid
        Grid with Gaussian quadrature weights.
    mask : array, shape (n_lat, n_lon)
        Ocean mask (1=ocean, 0=land).

    Returns
    -------
    array, shape (n_lat, n_lon)
        Corrected eta.
    """
    w = grid.weights[:, None]  # (n_lat, 1)
    dlon = 2.0 * jnp.pi / grid.n_lon
    area = (grid.radius ** 2) * w * dlon  # (n_lat, 1)
    weighted_area = mask * area

    ocean_area = jnp.sum(weighted_area)
    vol_old = jnp.sum(eta_old * weighted_area)
    vol_new = jnp.sum(eta_new * weighted_area)

    correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
    return eta_new + correction * mask


def correct_ocean_heat(
    T_new: jnp.ndarray,
    T_old: jnp.ndarray,
    h_k_new: jnp.ndarray,
    h_k_old: jnp.ndarray,
    grid: GaussianGrid,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Correct ocean temperature to conserve global heat content.

    Applies a spatially uniform additive correction to T so that
    the volume-integrated temperature is preserved.

    Parameters
    ----------
    T_new : array, shape (n_lat, n_lon, nlev)
        Predicted temperature [degC].
    T_old : array, shape (n_lat, n_lon, nlev)
        Original temperature [degC].
    h_k_new : array, shape (n_lat, n_lon, nlev)
        New layer thicknesses [m].
    h_k_old : array, shape (n_lat, n_lon, nlev)
        Old layer thicknesses [m].
    grid : GaussianGrid
        Grid for area weighting.
    mask : array, shape (n_lat, n_lon)
        Ocean mask (1=ocean, 0=land).

    Returns
    -------
    array, shape (n_lat, n_lon, nlev)
        Corrected temperature.
    """
    w = grid.weights[:, None]
    dlon = 2.0 * jnp.pi / grid.n_lon
    area = (grid.radius ** 2) * w * dlon
    weighted_area = mask * area
    mask_3d = mask[..., None]

    # Volume-integrated heat
    heat_old = jnp.sum(jnp.sum(T_old * h_k_old, axis=-1) * weighted_area)
    heat_new = jnp.sum(jnp.sum(T_new * h_k_new, axis=-1) * weighted_area)
    ocean_volume = jnp.sum(jnp.sum(h_k_new, axis=-1) * weighted_area)

    correction = (heat_old - heat_new) / jnp.maximum(ocean_volume, 1.0)
    return T_new + correction * mask_3d


def correct_ocean_salt(
    S_new: jnp.ndarray,
    S_old: jnp.ndarray,
    h_k_new: jnp.ndarray,
    h_k_old: jnp.ndarray,
    grid: GaussianGrid,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Correct ocean salinity to conserve global salt content.

    Same approach as heat correction but for salinity.

    Parameters
    ----------
    S_new : array, shape (n_lat, n_lon, nlev)
        Predicted salinity [PSU].
    S_old : array, shape (n_lat, n_lon, nlev)
        Original salinity [PSU].
    h_k_new : array, shape (n_lat, n_lon, nlev)
        New layer thicknesses [m].
    h_k_old : array, shape (n_lat, n_lon, nlev)
        Old layer thicknesses [m].
    grid : GaussianGrid
        Grid for area weighting.
    mask : array, shape (n_lat, n_lon)
        Ocean mask (1=ocean, 0=land).

    Returns
    -------
    array, shape (n_lat, n_lon, nlev)
        Corrected salinity.
    """
    w = grid.weights[:, None]
    dlon = 2.0 * jnp.pi / grid.n_lon
    area = (grid.radius ** 2) * w * dlon
    weighted_area = mask * area
    mask_3d = mask[..., None]

    salt_old = jnp.sum(jnp.sum(S_old * h_k_old, axis=-1) * weighted_area)
    salt_new = jnp.sum(jnp.sum(S_new * h_k_new, axis=-1) * weighted_area)
    ocean_volume = jnp.sum(jnp.sum(h_k_new, axis=-1) * weighted_area)

    correction = (salt_old - salt_new) / jnp.maximum(ocean_volume, 1.0)
    return S_new + correction * mask_3d
