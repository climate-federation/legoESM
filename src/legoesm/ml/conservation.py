"""Post-hoc conservation correctors for SFNO predictions.

Neural operators do not enforce conservation laws by construction.
These correctors adjust SFNO predictions to preserve:
- Global dry air mass (via uniform surface pressure correction)
- Global moisture budget (via proportional humidity correction)
- Non-negative humidity (clipping)

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
