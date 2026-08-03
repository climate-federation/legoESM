"""Loss functions for SFNO training.

Provides area-weighted MSE losses suitable for training on the sphere,
where grid cells at lower latitudes cover more area. Supports
per-variable monitoring and autoregressive rollout losses.
"""

from __future__ import annotations

import jax.numpy as jnp



def area_weighted_mse(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Area-weighted mean squared error.

    Uses Gaussian quadrature weights to account for the varying
    grid cell area with latitude.

    Parameters
    ----------
    pred : array, shape (..., n_lat, n_lon, n_channels)
        Predicted fields.
    target : array, shape (..., n_lat, n_lon, n_channels)
        Target fields.
    weights : array, shape (n_lat,)
        Gaussian quadrature weights for latitude.
    mask : array, optional, shape (n_lat, n_lon)
        Optional spatial mask (e.g. land/sea). Values in [0, 1].

    Returns
    -------
    scalar
        Area-weighted MSE averaged over all dimensions.
    """
    sq_err = (pred - target) ** 2  # (..., n_lat, n_lon, n_channels)

    # Weight by latitude.  ``weights`` are Gauss-Legendre weights on
    # μ = sin(lat) summing to 2.0 — they encode the cos(lat) area
    # element directly.
    w = weights[:, None, None]  # (n_lat, 1, 1)

    if mask is None:
        # NO-MASK PATH — kept byte-identical to the original.  The proper
        # area-weighted mean is Σ(sq·w) / (B · Σw · n_lon · n_channels);
        # jnp.mean(sq·w) divides by the full array size (B · n_lat · n_lon ·
        # n_channels), so the ratio n_lat / Σw recovers it.
        weighted = sq_err * w
        n_lat = weights.shape[0]
        return jnp.mean(weighted) * n_lat / jnp.sum(weights)

    # MASKED PATH — normalise by the ACTUAL summed mask·weight, not the full
    # array size.  Zeroing the numerator with mask[..., None] but still dividing
    # by jnp.mean (full size) biases the MSE LOW by the unmasked fraction.  The
    # weighted mean over the kept cells is Σ(sq·mask·w) / Σ(mask·w); with mask≡1
    # over a region this equals the unmasked area_weighted_mse of that region.
    mask_b = jnp.broadcast_to(mask[..., None], sq_err.shape)
    weight_b = jnp.broadcast_to(w, sq_err.shape) * mask_b
    denom = jnp.sum(weight_b)
    # Guard a fully-masked input (denom == 0 -> 0/0 NaN, inf gradient).
    denom = jnp.maximum(denom, jnp.asarray(jnp.finfo(sq_err.dtype).tiny, sq_err.dtype))
    return jnp.sum(sq_err * weight_b) / denom


def latitude_weighted_rmse(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    lat_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Zonal-mean square-error, then latitude-weighted RMSE.

    Used by the AIMIP scorecard and per-variant evaluation scripts. Takes
    a 2D ``(n_lat, n_lon)`` field (a single mid-level slice or surface
    field), computes the zonal-mean square error per latitude row, then
    forms the latitude-weighted mean and returns its square root.

    Equivalent to ``sqrt(area_weighted_mse(...))`` for a 2D field, but
    keeps the explicit zonal-then-meridional formula used in WeatherBench
    scorecards.
    """
    sq_zonal = jnp.mean((pred - target) ** 2, axis=-1)
    return jnp.sqrt(jnp.sum(sq_zonal * lat_weights) / jnp.sum(lat_weights))


def latitude_weighted_bias(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    lat_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Zonal-mean bias, then latitude-weighted mean.

    Companion to :func:`latitude_weighted_rmse` for the AIMIP scorecard.
    Returns the signed area-weighted mean error of a 2D
    ``(n_lat, n_lon)`` field.
    """
    diff_zonal = jnp.mean(pred - target, axis=-1)
    return jnp.sum(diff_zonal * lat_weights) / jnp.sum(lat_weights)


def per_variable_mse(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray,
    channel_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-channel area-weighted MSE for monitoring.

    Parameters
    ----------
    pred : array, shape (..., n_lat, n_lon, n_channels)
        Predicted fields.
    target : array, shape (..., n_lat, n_lon, n_channels)
        Target fields.
    weights : array, shape (n_lat,)
        Gaussian quadrature weights.
    channel_weights : array, optional, shape (n_channels,)
        Per-channel importance weights — multiply each channel's
        area-weighted MSE before returning.  When None, all
        channels are reported with weight 1.

    Returns
    -------
    per_channel : array, shape (n_channels,)
        MSE for each channel (multiplied by channel_weights when
        provided).  Each entry is a true area-weighted mean
        independent of grid resolution.
    """
    sq_err = (pred - target) ** 2

    # Average over batch, longitude, leaving (n_lat, n_channels)
    w = weights[:, None, None]
    weighted = sq_err * w

    axes = tuple(range(weighted.ndim - 1))
    # Use the same correction factor as ``area_weighted_mse``
    # (iter-63 fix) so the per-channel MSE is a proper area-
    # weighted mean: Σ(sq·w)/(B·Σw·n_lon) instead of mean(sq·w).
    n_lat = weights.shape[0]
    per_channel = jnp.mean(weighted, axis=axes) * n_lat / jnp.sum(weights)

    # Apply optional per-channel weights.  Previously the argument
    # was ``_channel_weights`` (leading underscore) and never read,
    # silently dropping any caller-supplied weighting.  Iter-64 fix.
    if channel_weights is not None:
        per_channel = per_channel * channel_weights

    return per_channel


def weighted_mae(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Area-weighted mean absolute error.

    Parameters
    ----------
    pred : array, shape (..., n_lat, n_lon, n_channels)
        Predicted fields.
    target : array, shape (..., n_lat, n_lon, n_channels)
        Target fields.
    weights : array, shape (n_lat,)
        Gaussian quadrature weights for latitude.

    Returns
    -------
    scalar
        Area-weighted MAE averaged over all dimensions.
    """
    abs_err = jnp.abs(pred - target)
    w = weights[:, None, None]
    # Same resolution-independence correction as ``area_weighted_mse`` /
    # ``per_variable_mse``: ``jnp.mean(abs·w)`` divides by the full array size
    # (B·n_lat·n_lon·n_ch); multiplying by ``n_lat/Σw`` recovers the proper
    # area-weighted mean Σ(abs·w)/(B·Σw·n_lon·n_ch). Without it the MAE was off
    # by ~n_lat/Σw and lived on a different scale from the corrected MSE.
    n_lat = weights.shape[0]
    return jnp.mean(abs_err * w) * n_lat / jnp.sum(weights)


def almost_fair_crps(
    ensemble: jnp.ndarray,
    target: jnp.ndarray,
    *,
    alpha: float = 0.95,
) -> jnp.ndarray:
    """Almost-fair CRPS field for a finite ensemble.

    This follows the finite-ensemble adjustment used in the referenced
    stochastic S2S literature: it reduces the small-ensemble bias in the
    pairwise spread term while remaining simple to evaluate from samples.

    Parameters
    ----------
    ensemble : array, shape (n_members, ..., n_lat, n_lon, n_channels)
        Ensemble predictions.
    target : array, shape (..., n_lat, n_lon, n_channels)
        Deterministic target field.
    alpha : float, default=0.95
        Finite-ensemble adjustment factor. Values near 1.0 recover a
        near-fair ensemble score while remaining numerically stable for
        small ensemble sizes.

    Returns
    -------
    array
        Pointwise almost-fair CRPS.
    """
    n_members = ensemble.shape[0]
    if n_members <= 1:
        return jnp.abs(ensemble[0] - target)

    obs_term = jnp.mean(jnp.abs(ensemble - target[None, ...]), axis=0)
    pairwise_sum = jnp.sum(
        jnp.abs(ensemble[:, None, ...] - ensemble[None, :, ...]),
        axis=(0, 1),
    )
    coeff = (n_members - 1 + alpha) / (2.0 * n_members * n_members * (n_members - 1))
    return obs_term - coeff * pairwise_sum


def area_weighted_afcrps(
    ensemble: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    alpha: float = 0.95,
) -> jnp.ndarray:
    """Area-weighted almost-fair CRPS averaged over all non-ensemble dimensions.

    Carries the SAME resolution-independence correction as its siblings
    ``area_weighted_mse`` / ``per_variable_mse`` / ``weighted_mae``: a bare
    ``jnp.mean(crps * w)`` divides by the full array size rather than by the
    weight sum, so the result is low by ``n_lat / Σw`` — exactly 5x at T5, ~32x
    at T42, i.e. resolution-DEPENDENT (#1413). This is the S2S training
    objective, so an uncorrected value put the CRPS term on a different scale
    from the MSE/MAE terms it is mixed and compared with.
    """
    crps = almost_fair_crps(ensemble, target, alpha=alpha)
    n_lat = weights.shape[0]
    w = weights[:, None, None]
    return jnp.mean(crps * w) * n_lat / jnp.sum(weights)


def spectral_loss(
    pred_hat: jnp.ndarray,
    target_hat: jnp.ndarray,
) -> jnp.ndarray:
    """L2 loss in spectral space.

    Optional regularizer that penalizes spectral coefficient mismatches,
    encouraging accurate representation of large-scale patterns.

    Parameters
    ----------
    pred_hat : complex array, shape (n_sh,) or (n_sh, nlev)
        Predicted spectral coefficients.
    target_hat : complex array, shape (n_sh,) or (n_sh, nlev)
        Target spectral coefficients.

    Returns
    -------
    scalar
        Mean squared difference of spectral coefficients.
    """
    diff = pred_hat - target_hat
    return jnp.mean(jnp.abs(diff) ** 2)
