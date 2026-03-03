"""Loss functions for SFNO training.

Provides area-weighted MSE losses suitable for training on the sphere,
where grid cells at lower latitudes cover more area. Supports
per-variable monitoring and autoregressive rollout losses.
"""

from __future__ import annotations

from typing import Callable

import jax
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
