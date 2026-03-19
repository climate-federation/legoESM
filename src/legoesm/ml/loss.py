"""Loss functions for SFNO training.

Provides area-weighted MSE losses suitable for training on the sphere,
where grid cells at lower latitudes cover more area. Supports
per-variable monitoring and autoregressive rollout losses.
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx

from legoesm.grids.gaussian import GaussianGrid


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

    # Apply spatial mask
    if mask is not None:
        sq_err = sq_err * mask[..., None]

    # Weight by latitude: weights has shape (n_lat,)
    w = weights[:, None, None]  # (n_lat, 1, 1)
    weighted = sq_err * w

    return jnp.mean(weighted)


def per_variable_mse(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray,
    _channel_weights: jnp.ndarray | None = None,
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
        Per-channel importance weights for the total loss.

    Returns
    -------
    per_channel : array, shape (n_channels,)
        MSE for each channel.
    """
    sq_err = (pred - target) ** 2

    # Average over batch, longitude (and optionally batch dims)
    # keeping channel dimension
    w = weights[:, None, None]
    weighted = sq_err * w

    # Average over all spatial dims, keep channels
    axes = tuple(range(weighted.ndim - 1))
    per_channel = jnp.mean(weighted, axis=axes)

    return per_channel


def autoregressive_loss(
    model: eqx.Module,
    initial: jnp.ndarray,
    targets: jnp.ndarray,
    grid: GaussianGrid,
    n_steps: int = 2,
) -> jnp.ndarray:
    """Multi-step autoregressive rollout loss.

    Rolls out the model for n_steps from the initial condition and
    computes area-weighted MSE at each step. This encourages the
    model to produce stable multi-step predictions.

    Parameters
    ----------
    model : eqx.Module (SFNO)
        The SFNO model (callable: (x, grid) → y).
    initial : array, shape (n_lat, n_lon, n_channels)
        Initial state.
    targets : array, shape (n_steps, n_lat, n_lon, n_channels)
        Target states at each rollout step.
    grid : GaussianGrid
        Grid for area weights.
    n_steps : int
        Number of autoregressive steps.

    Returns
    -------
    scalar
        Mean area-weighted MSE across all rollout steps.
    """
    total_loss = jnp.float32(0.0)
    state = initial

    for step in range(n_steps):
        pred = model(state, grid)
        step_loss = area_weighted_mse(
            pred, targets[step], grid.weights.astype(jnp.float32)
        )
        total_loss = total_loss + step_loss
        state = pred  # Autoregressive: use prediction as next input

    return total_loss / n_steps


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
    return jnp.mean(abs_err * w)


def empirical_crps(
    ensemble: jnp.ndarray,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """Empirical CRPS field for a finite ensemble.

    Parameters
    ----------
    ensemble : array, shape (n_members, ..., n_lat, n_lon, n_channels)
        Ensemble predictions.
    target : array, shape (..., n_lat, n_lon, n_channels)
        Deterministic target field.

    Returns
    -------
    array
        CRPS evaluated pointwise, retaining all non-ensemble dimensions.
    """
    obs_term = jnp.mean(jnp.abs(ensemble - target[None, ...]), axis=0)
    pairwise = jnp.abs(ensemble[:, None, ...] - ensemble[None, :, ...])
    return obs_term - 0.5 * jnp.mean(pairwise, axis=(0, 1))


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
    """Area-weighted almost-fair CRPS averaged over all non-ensemble dimensions."""
    crps = almost_fair_crps(ensemble, target, alpha=alpha)
    w = weights[:, None, None]
    return jnp.mean(crps * w)


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
