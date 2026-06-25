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
        # Weighted mean and std over the sample + spatial dimensions.
        # Align the weight axes to the SPATIAL dims (immediately after the
        # leading sample axis), with trailing singletons for the remaining
        # spatial + channel axes. The previous leading ``expand_dims`` placed a
        # ``(n_lat,)`` weight as ``(1,1,1,n_lat)`` — area-weighting the CHANNEL
        # axis instead of latitude (wrong, or a shape error when n_ch != n_lat).
        w = weights.reshape(
            (1,) + weights.shape + (1,) * (data.ndim - 1 - weights.ndim))
        # Normalise by the ACTUAL summed weight over the averaged axes, so a
        # latitude-only weight still averages (not sums) the uniform sample /
        # longitude axes — Σ(x·w)/Σw is the proper weighted mean for any weight
        # shape (the old ``w/Σw`` only normalised the latitude sum).
        w_b = jnp.broadcast_to(w, data.shape)
        w_sum = jnp.sum(w_b, axis=axes)
        mean = jnp.sum(data * w_b, axis=axes) / w_sum
        var = jnp.sum((data - mean) ** 2 * w_b, axis=axes) / w_sum
        std = jnp.sqrt(var)

    std = jnp.maximum(std, eps)
    return NormalizationStats(mean=mean, std=std)
