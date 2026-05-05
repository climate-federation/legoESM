"""Training loss functions for differentiable dycore rollouts.

Builds on ``ml.loss`` (area_weighted_mse, spectral_loss) with
dycore-specific additions: per-level pressure weighting, multi-day
rollout loss, and SegmentCarry comparison.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.grids.gaussian import sh_analysis_3d
from legoesm.ml.loss import spectral_loss


class LossConfig(NamedTuple):
    """Configuration for dycore training loss."""
    # Variable weights (relative importance)
    w_T: float = 1.0          # temperature
    w_u: float = 0.5          # zonal wind
    w_v: float = 0.5          # meridional wind
    w_q: float = 0.2          # specific humidity
    w_ps: float = 0.3         # surface pressure
    # Loss components
    spectral_weight: float = 0.0   # weight for spectral loss term
    level_weighting: str = "pressure"  # "uniform", "pressure", or "boundary_layer"
    # Pressure weighting parameters
    p_ref_Pa: float = 50000.0      # reference pressure for level weighting [Pa]
    p_scale_Pa: float = 30000.0    # width of weighting function [Pa]


def level_weights(
    sigma_full: jax.Array,
    p_s_mean: float = 101325.0,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Compute per-level weights emphasizing mid-troposphere.

    Parameters
    ----------
    sigma_full : (nlev,) — model sigma levels
    p_s_mean : float — approximate mean surface pressure [Pa]
    config : LossConfig

    Returns
    -------
    (nlev,) — normalized weights summing to nlev
    """
    if config.level_weighting == "uniform":
        return jnp.ones_like(sigma_full)

    p_levels = sigma_full * p_s_mean  # approximate pressure at each level

    if config.level_weighting == "pressure":
        # Gaussian-like weighting centered at p_ref, width p_scale
        w = jnp.exp(-0.5 * ((p_levels - config.p_ref_Pa) / config.p_scale_Pa) ** 2)
        # Add a floor so stratosphere isn't zero
        w = w + 0.1
    elif config.level_weighting == "boundary_layer":
        # Emphasize lower troposphere (p > 700 hPa)
        w = jnp.where(p_levels > 70000.0, 2.0, 1.0)
    else:
        w = jnp.ones_like(sigma_full)

    # Normalize so mean weight = 1
    return w * (sigma_full.shape[0] / jnp.sum(w))


def carry_mse(
    pred_carry,
    target_carry,
    sigma_full: jax.Array,
    lat_weights: jax.Array | None = None,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Compute weighted MSE between two SegmentCarry states.

    Works directly on SegmentCarry fields (raw arrays on the model grid),
    avoiding regridding in the loss computation.

    Parameters
    ----------
    pred_carry, target_carry : SegmentCarry
        Predicted and target states on the model grid.
    sigma_full : (nlev,) — sigma levels for pressure weighting
    lat_weights : (n_lat,) or None — latitude weights for area weighting.
        If None, uniform weighting is used (appropriate for cubed-sphere
        where cells have approximately equal area).
    config : LossConfig

    Returns
    -------
    scalar — weighted MSE loss
    """
    lev_w = level_weights(sigma_full, config=config)

    from legoesm.core.precision import _resolve_dtype
    loss = jnp.array(0.0, dtype=_resolve_dtype(None, "accumulate"))

    def _lat_weighted_mean(sq_err: jax.Array) -> jax.Array:
        """Mean over all dims, with optional latitude weighting.

        When ``lat_weights`` is provided and ``sq_err`` has an axis
        of length ``n_lat = len(lat_weights)``, the mean is
        replaced by the area-weighted mean
        ``mean(sq · lat_w) · n_lat / Σ(lat_w)`` (resolution-
        independent, identical correction as iter-63 ml/loss.py).
        Without lat_weights or on non-Gaussian shapes (e.g.
        cubed-sphere with leading face dim) this falls back to
        a uniform mean.
        """
        if lat_weights is None:
            return jnp.mean(sq_err)
        n_lat_w = lat_weights.shape[0]
        # Apply weight on the FIRST axis of length n_lat.
        for axis, dim in enumerate(sq_err.shape):
            if dim == n_lat_w:
                shape = [1] * sq_err.ndim
                shape[axis] = n_lat_w
                w = lat_weights.reshape(shape)
                return jnp.mean(sq_err * w) * n_lat_w / jnp.sum(lat_weights)
        # No matching axis — fall back to uniform mean.
        return jnp.mean(sq_err)

    # Temperature: (..., nlev)
    dT = pred_carry.T - target_carry.T
    loss = loss + config.w_T * _lat_weighted_mean(dT ** 2 * lev_w)

    # Winds: (..., nlev)
    du = pred_carry.u - target_carry.u
    dv = pred_carry.v - target_carry.v
    loss = loss + config.w_u * _lat_weighted_mean(du ** 2 * lev_w)
    loss = loss + config.w_v * _lat_weighted_mean(dv ** 2 * lev_w)

    # Moisture: (..., nlev)
    dq = pred_carry.q_v - target_carry.q_v
    loss = loss + config.w_q * _lat_weighted_mean(dq ** 2 * lev_w)

    # Surface pressure: (...)
    dp = pred_carry.p_s - target_carry.p_s
    loss = loss + config.w_ps * _lat_weighted_mean(dp ** 2)

    return loss


def multi_day_loss(
    pred_carries,
    target_carries,
    sigma_full: jax.Array,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Multi-day rollout loss: average carry_mse over multiple lead times.

    Parameters
    ----------
    pred_carries : pytree with leading (n_days,) dimension
        Predicted states at each day from differentiable_rollout.
    target_carries : pytree with leading (n_days,) dimension
        Target states from ERA5 at corresponding days.
    sigma_full : (nlev,)
    config : LossConfig

    Returns
    -------
    scalar — averaged loss across all lead times
    """
    # Use ``jax.vmap`` directly over the leading day axis of the carry
    # pytrees instead of a closure-over-arange + ``tree.map(lambda x: x[i])``.
    # The closure pattern forced JAX to retrace every call (the closure
    # captured ``pred_carries`` / ``target_carries`` by identity); vmap
    # with ``in_axes=0`` lets the batching machinery slice the leading
    # axis without any Python tree walk inside the inner loop.
    def _day_loss(pred_i, target_i):
        return carry_mse(pred_i, target_i, sigma_full, config=config)

    day_losses = jax.vmap(_day_loss)(pred_carries, target_carries)
    return jnp.mean(day_losses)


def carry_spectral_loss(
    pred_carry,
    target_carry,
    grid,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Spectral loss on temperature field (for spectral grids).

    Penalizes errors in the large-scale spectral pattern of T.
    Only meaningful for spectral (Gaussian) grids where SH analysis
    is available.

    Parameters
    ----------
    pred_carry, target_carry : SegmentCarry
    grid : GaussianGrid with sh_analysis_3d
    config : LossConfig

    Returns
    -------
    scalar — spectral L2 loss on T
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "carry_spectral_loss requires JAX_ENABLE_X64=True for "
            "float64 spectral transforms."
        )

    # Reshape for SH analysis: (..., nlev) → (n_lat, n_lon, nlev)
    pred_T = pred_carry.T.astype(jnp.float64)
    target_T = target_carry.T.astype(jnp.float64)

    pred_hat = sh_analysis_3d(grid, pred_T)
    target_hat = sh_analysis_3d(grid, target_T)

    return spectral_loss(pred_hat, target_hat)


def combined_loss(
    pred_carry,
    target_carry,
    sigma_full: jax.Array,
    grid=None,
    config: LossConfig = LossConfig(),
) -> jax.Array:
    """Combined loss: weighted MSE + optional spectral penalty.

    Parameters
    ----------
    pred_carry, target_carry : SegmentCarry
    sigma_full : (nlev,)
    grid : GaussianGrid or None (spectral loss requires grid)
    config : LossConfig

    Returns
    -------
    scalar — total loss
    """
    loss = carry_mse(pred_carry, target_carry, sigma_full, config=config)

    if config.spectral_weight > 0.0 and grid is not None:
        loss = loss + config.spectral_weight * carry_spectral_loss(
            pred_carry, target_carry, grid, config=config,
        )

    return loss
