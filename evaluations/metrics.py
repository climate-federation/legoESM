"""Core evaluation metrics for WeatherBench2.

All spatial metrics are area-weighted using Gaussian quadrature weights
to account for varying grid cell area with latitude.
"""

from __future__ import annotations

import jax.numpy as jnp


def rmse(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Latitude-weighted root mean squared error.

    Parameters
    ----------
    pred : array, shape (..., n_lat, n_lon)
        Predicted field.
    target : array, shape (..., n_lat, n_lon)
        Target field.
    weights : array, shape (n_lat,)
        Gaussian quadrature weights for latitude.
    mask : array, optional, shape (n_lat, n_lon)
        Spatial mask in [0, 1].

    Returns
    -------
    scalar
        Area-weighted RMSE.
    """
    sq_err = (pred - target) ** 2
    w = weights[:, None]
    if mask is not None:
        w = w * mask
    # numerator and denominator both broadcast over any leading dims of sq_err
    wmse = jnp.sum(sq_err * w) / jnp.sum(w * jnp.ones_like(sq_err))
    return jnp.sqrt(wmse)


def acc(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    climatology: jnp.ndarray,
    weights: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Anomaly Correlation Coefficient.

    ACC = sum(w * pred' * target') / sqrt(sum(w * pred'^2) * sum(w * target'^2))

    where primes denote anomalies from climatology.

    Parameters
    ----------
    pred : array, shape (..., n_lat, n_lon)
        Predicted field.
    target : array, shape (..., n_lat, n_lon)
        Target field.
    climatology : array, shape (n_lat, n_lon)
        Climatological mean field.
    weights : array, shape (n_lat,)
        Gaussian quadrature weights.
    mask : array, optional, shape (n_lat, n_lon)
        Spatial mask in [0, 1]; masked-out cells are excluded from the ACC
        (used for below-ground pressure-level cells).

    Returns
    -------
    scalar
        ACC in [-1, 1].
    """
    pred_anom = pred - climatology
    target_anom = target - climatology

    w = weights[:, None]
    if mask is not None:
        w = w * mask               # exclude below-ground / invalid cells from the ACC
    numerator = jnp.sum(w * pred_anom * target_anom)
    denom_pred = jnp.sum(w * pred_anom**2)
    denom_target = jnp.sum(w * target_anom**2)
    denominator = jnp.sqrt(denom_pred * denom_target)

    return numerator / jnp.maximum(denominator, 1e-12)


def bias(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Area-weighted mean bias (pred - target).

    Parameters
    ----------
    pred : array, shape (..., n_lat, n_lon)
        Predicted field.
    target : array, shape (..., n_lat, n_lon)
        Target field.
    weights : array, shape (n_lat,)
        Gaussian quadrature weights.
    mask : array, optional, shape (n_lat, n_lon)
        Spatial mask in [0, 1]; masked-out cells are excluded from the mean
        (used for below-ground pressure-level cells).

    Returns
    -------
    scalar
        Area-weighted mean bias.
    """
    diff = pred - target
    if mask is not None:
        w = weights[:, None] * mask
        # denominator broadcasts over any leading dims exactly like the numerator
        return jnp.sum(w * diff) / jnp.sum(w * jnp.ones_like(diff))
    w = weights[:, None]
    return jnp.sum(w * diff) / jnp.sum(w * jnp.ones_like(diff))


def spread_skill_ratio(
    ensemble_preds: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Spread-skill ratio for ensemble evaluation.

    Ratio of ensemble spread to RMSE of the ensemble mean.
    Ideally close to 1 for a well-calibrated ensemble.

    Parameters
    ----------
    ensemble_preds : array, shape (n_members, ..., n_lat, n_lon)
        Ensemble member predictions.
    target : array, shape (..., n_lat, n_lon)
        Target field.
    weights : array, shape (n_lat,)
        Gaussian quadrature weights.

    Returns
    -------
    scalar
        Spread / skill ratio.
    """
    ens_mean = jnp.mean(ensemble_preds, axis=0)
    skill = rmse(ens_mean, target, weights)

    # Spread: RMS of deviations from ensemble mean
    deviations = ensemble_preds - ens_mean[None, ...]
    w = weights[:, None]
    n_members = ensemble_preds.shape[0]
    spread_sq = jnp.sum(w * deviations**2) / (n_members * jnp.sum(w))
    spread = jnp.sqrt(spread_sq)

    return spread / jnp.maximum(skill, 1e-12)


def compute_scorecard(
    pred_ds,
    target_ds,
    clim_ds,
    weights: jnp.ndarray,
    variables: list[str],
    levels: list[int],
    lead_times: list[int] | None = None,
) -> dict:
    """Compute a scorecard of metrics per (variable, level, lead_time).

    Parameters
    ----------
    pred_ds : dict
        Predictions keyed by ``(variable, level, lead_time)`` with values
        as arrays of shape (n_lat, n_lon).
    target_ds : dict
        Targets with the same keys.
    clim_ds : dict
        Climatology keyed by ``(variable, level)`` with values (n_lat, n_lon).
    weights : array, shape (n_lat,)
        Gaussian quadrature weights.
    variables : list of str
        Variable names.
    levels : list of int
        Pressure levels.
    lead_times : list of int, optional
        Lead times in hours. If None, inferred from pred_ds keys.

    Returns
    -------
    dict
        Nested dict: ``{variable: {level: {lead_time: {rmse, acc, bias}}}}``.
    """
    if lead_times is None:
        lead_times = sorted({k[2] for k in pred_ds.keys()})

    scorecard = {}
    for var in variables:
        scorecard[var] = {}
        for lev in levels:
            scorecard[var][lev] = {}
            clim = clim_ds.get((var, lev))
            for lt in lead_times:
                key = (var, lev, lt)
                if key not in pred_ds or key not in target_ds:
                    continue
                p = pred_ds[key]
                t = target_ds[key]
                entry = {
                    "rmse": float(rmse(p, t, weights)),
                    "bias": float(bias(p, t, weights)),
                }
                if clim is not None:
                    entry["acc"] = float(acc(p, t, clim, weights))
                scorecard[var][lev][lt] = entry

    return scorecard
