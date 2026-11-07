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

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)

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
    # Total spherical area weight: latitude weights summed, replicated over
    # all n_lon longitudes.
    w_total = jnp.sum(w) * grid.n_lon

    # Area-weighted global-mean surface-pressure difference.  Summing
    # ``(p_s_new - p_s_old) * w`` over (lat, lon) already accumulates all
    # n_lon longitudes, so divide by the full ``w_total`` directly.  (The
    # previous form multiplied the numerator by an extra ``grid.n_lon``,
    # which double-counted longitude and made ``dp`` n_lon-times too large —
    # the correction then only no-op'd near zero imbalance.)
    dp = jnp.sum((p_s_new - p_s_old) * w) / w_total

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

    # Column-integrated moisture: integral(q * dp) = p_s * sum(q * dsigma).
    # Both ``col_old`` and ``col_new`` reduce ``q * dsigma`` over the
    # level axis with the same weight; stack and reduce once.  Then the
    # subsequent area-weighted reduction collapses to one ``sum`` call.
    _col_pair = jnp.sum(
        jnp.stack([q_old, q_new], axis=-1) * dsigma[None, None, :, None],
        axis=-2,
    ) * p_s[..., None]
    _global_pair = jnp.sum(_col_pair * w[..., None], axis=(0, 1))
    global_old = _global_pair[..., 0]
    global_new = _global_pair[..., 1]

    # Proportional correction factor
    ratio = global_old / jnp.maximum(global_new, _TINY)

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
    weighted_area = mask * grid.grid_area  # canonical Gaussian cell area

    ocean_area = jnp.sum(weighted_area)
    vol_old = jnp.sum(eta_old * weighted_area)
    vol_new = jnp.sum(eta_new * weighted_area)

    correction = (vol_old - vol_new) / jnp.maximum(ocean_area, 1.0)
    return eta_new + correction * mask


def correct_ocean_tracer(
    x_new: jnp.ndarray,
    x_old: jnp.ndarray,
    h_k_new: jnp.ndarray,
    h_k_old: jnp.ndarray,
    grid: GaussianGrid,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Correct an ocean tracer to conserve its global volume integral.

    Applies a spatially uniform additive correction to the tracer (T for
    heat, S for salt) so that its volume integral is preserved.

    Parameters
    ----------
    x_new : array, shape (n_lat, n_lon, nlev)
        Predicted tracer (T [degC] or S [PSU]).
    x_old : array, shape (n_lat, n_lon, nlev)
        Original tracer, same units as x_new.
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
        Corrected tracer.
    """
    weighted_area = mask * grid.grid_area  # canonical Gaussian cell area
    mask_3d = mask[..., None]

    # Volume-integrated tracer — fuse the 3 column reductions into one
    # stack and the 3 area reductions into one ``axis=(0, 1)`` collapse.
    _inner = jnp.sum(
        jnp.stack([x_old * h_k_old, x_new * h_k_new, h_k_new], axis=-1),
        axis=-2,
    )
    _global = jnp.sum(_inner * weighted_area[..., None], axis=(0, 1))
    total_old = _global[..., 0]
    total_new = _global[..., 1]
    ocean_volume = _global[..., 2]

    correction = (total_old - total_new) / jnp.maximum(ocean_volume, 1.0)
    return x_new + correction * mask_3d
