"""Z-score normalization for neural operator inputs/outputs.

Provides per-channel normalization to zero mean and unit variance,
essential for stable training of neural operators like SFNO.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class NormalizationStats(NamedTuple):
    """Per-channel mean and standard deviation for Z-score normalization.

    Attributes
    ----------
    mean : jax.Array, shape (..., n_channels)
        Channel means computed from training data.
    std : jax.Array, shape (..., n_channels)
        Channel standard deviations (clamped away from zero).
    """
    mean: jnp.ndarray
    std: jnp.ndarray


def normalize(x: jnp.ndarray, stats: NormalizationStats) -> jnp.ndarray:
    """Apply Z-score normalization: (x - mean) / std."""
    return (x - stats.mean) / stats.std


def denormalize(x: jnp.ndarray, stats: NormalizationStats) -> jnp.ndarray:
    """Invert Z-score normalization: x * std + mean."""
    return x * stats.std + stats.mean


def compute_normalization_stats(
    data: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    eps: float = 1e-6,
) -> NormalizationStats:
    """Compute per-channel normalization statistics from training data.

    Parameters
    ----------
    data : array, shape (n_samples, ..., n_channels)
        Training data. The last axis is the channel dimension;
        statistics are computed over all other axes.
    weights : array, optional
        Area weights for latitude-weighted averaging. If provided,
        must broadcast with data over the spatial dimensions.
    eps : float
        Minimum standard deviation to avoid division by zero.

    Returns
    -------
    NormalizationStats
        Per-channel mean and std arrays of shape (n_channels,).
    """
    axes = tuple(range(data.ndim - 1))

    if weights is None:
        mean = jnp.mean(data, axis=axes)
        std = jnp.std(data, axis=axes)
    else:
        # Weighted mean and std over spatial and sample dimensions
        w = weights / jnp.sum(weights)
        # Expand weights to broadcast: (1, ..., 1)
        for _ in range(data.ndim - weights.ndim):
            w = jnp.expand_dims(w, axis=0)
        mean = jnp.sum(data * w, axis=axes)
        var = jnp.sum((data - mean) ** 2 * w, axis=axes)
        std = jnp.sqrt(var)

    std = jnp.maximum(std, eps)
    return NormalizationStats(mean=mean, std=std)
