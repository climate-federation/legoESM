"""Vertical interpolation from pressure levels to sigma/hybrid coordinates.

Pure JAX implementation — fully differentiable and JIT-compatible.

ERA5 / WeatherBench2 data lives on fixed pressure levels (e.g., 13 levels
from 1000 to 50 hPa).  legoESM uses terrain-following sigma or hybrid
sigma-pressure coordinates where the actual pressure depends on surface
pressure: p_k = sigma_k * p_s (sigma) or p_k = A_k * p_ref + B_k * p_s
(hybrid).

This module interpolates 3D fields from pressure levels to model levels
using log-pressure linear interpolation, which is standard practice in
atmospheric science (temperature varies approximately linearly in log(p)).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp



def interp_pressure_to_sigma(
    field_plev: jax.Array,
    plev_Pa: jax.Array,
    p_s: jax.Array,
    sigma_full: jax.Array,
) -> jax.Array:
    """Interpolate a 3D field from pressure levels to sigma levels.

    Parameters
    ----------
    field_plev : array, shape (..., n_plev)
        Field on pressure levels (top-to-bottom or bottom-to-top;
        must be consistent with plev_Pa ordering).
    plev_Pa : array, shape (n_plev,)
        Pressure levels in Pa, **ascending** (e.g., 5000, 10000, ..., 100000).
    p_s : array, shape (...)
        Surface pressure in Pa.
    sigma_full : array, shape (n_model_lev,)
        Model sigma values at full levels (top-to-bottom: small→large).

    Returns
    -------
    array, shape (..., n_model_lev)
        Field interpolated to model sigma levels.
    """
    # Target pressures: p_target[k] = sigma[k] * p_s
    p_target = p_s[..., None] * sigma_full  # (..., n_model_lev)
    return _interp_in_logp(field_plev, plev_Pa, p_target)


def interp_pressure_to_hybrid(
    field_plev: jax.Array,
    plev_Pa: jax.Array,
    p_s: jax.Array,
    A_full: jax.Array,
    B_full: jax.Array,
    p_ref: float,
) -> jax.Array:
    """Interpolate a 3D field from pressure levels to hybrid sigma-pressure levels.

    Parameters
    ----------
    field_plev : array, shape (..., n_plev)
        Field on pressure levels.
    plev_Pa : array, shape (n_plev,)
        Source pressure levels in Pa, **ascending**.
    p_s : array, shape (...)
        Surface pressure in Pa.
    A_full, B_full : array, shape (n_model_lev,)
        Hybrid coordinate coefficients: p(k) = A(k)*p_ref + B(k)*p_s.
    p_ref : float
        Reference pressure in Pa.

    Returns
    -------
    array, shape (..., n_model_lev)
    """
    p_target = A_full * p_ref + p_s[..., None] * B_full
    return _interp_in_logp(field_plev, plev_Pa, p_target)


def _interp_in_logp(
    field_plev: jax.Array,
    plev_Pa: jax.Array,
    p_target: jax.Array,
) -> jax.Array:
    """Log-pressure linear interpolation (differentiable).

    For each target pressure p_t, find the two bracketing source levels
    p_lo, p_hi and interpolate linearly in log(p):

        f(p_t) = f_lo + (f_hi - f_lo) * (ln(p_t) - ln(p_lo)) / (ln(p_hi) - ln(p_lo))

    Below-surface extrapolation: hold constant from lowest level.
    Above-model-top extrapolation: hold constant from highest level.

    Parameters
    ----------
    field_plev : (..., n_plev) — source field on pressure levels
    plev_Pa : (n_plev,) — source pressure levels, ascending
    p_target : (..., n_model_lev) — target pressures

    Returns
    -------
    (..., n_model_lev) — interpolated field
    """
    n_plev = plev_Pa.shape[0]
    log_plev = jnp.log(plev_Pa)
    log_p_target = jnp.log(p_target)

    # searchsorted: find index i such that plev_Pa[i-1] <= p_target < plev_Pa[i]
    # Result shape: same as p_target
    idx_hi = jnp.searchsorted(plev_Pa, p_target)

    # Clamp to valid range [1, n_plev-1] for bracket computation
    idx_hi = jnp.clip(idx_hi, 1, n_plev - 1)
    idx_lo = idx_hi - 1

    # Gather source values at bracket levels
    # field_plev has shape (..., n_plev), we need to index the last axis
    f_lo = jnp.take_along_axis(field_plev, idx_lo, axis=-1)
    f_hi = jnp.take_along_axis(field_plev, idx_hi, axis=-1)

    # Log-pressure coordinates of brackets
    lp_lo = log_plev[idx_lo]
    lp_hi = log_plev[idx_hi]

    # Interpolation weight
    denom = lp_hi - lp_lo
    # Avoid division by zero (happens when idx_lo == idx_hi after clamping)
    denom = jnp.where(denom == 0.0, 1.0, denom)
    alpha = (log_p_target - lp_lo) / denom

    # Clamp alpha to [0, 1] for extrapolation (hold constant)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    return f_lo + alpha * (f_hi - f_lo)
