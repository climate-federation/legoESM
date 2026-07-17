"""Z-score normalization for neural operator inputs/outputs.

Provides per-channel normalization to zero mean and unit variance,
essential for stable training of neural operators like SFNO.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
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
    # Floor the VARIANCE before the sqrt (not the std after it).  For a constant
    # channel var == 0, and d(sqrt(var))/dvar = 1/(2 sqrt(var)) -> inf there, so
    # the backward pass is 0 * inf = NaN even though jnp.maximum(std, eps) makes
    # the forward value finite.  Clamping var to eps**2 first gives std >= eps
    # AND a finite gradient.
    var_floor = eps * eps

    if weights is None:
        mean = jnp.mean(data, axis=axes)
        var = jnp.mean((data - mean) ** 2, axis=axes)
        std = jnp.sqrt(jnp.maximum(var, var_floor))
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
        # Clamp variance before sqrt (see the var_floor note above) so a constant
        # channel yields std == eps with a finite gradient instead of 0 * inf.
        std = jnp.sqrt(jnp.maximum(var, var_floor))

    return NormalizationStats(mean=mean, std=std)


def save_normalization_stats(stats: NormalizationStats, path) -> None:
    """Persist per-channel stats to an ``.npz`` sidecar.

    The SFNO PE checkpoint (``eqx.tree_serialise_leaves``) serialises ONLY the
    ``SFNO`` network leaves, not the ``SFNOPrimitiveEquationModel`` wrapper that
    carries ``norm_stats`` — so the stat VALUES are not in the checkpoint.  A
    normalised emulator therefore needs its stats saved alongside the checkpoint
    and reloaded at eval time so training and evaluation apply the SAME Z-score
    transform.  Values are stored as float64 numpy arrays (device-agnostic).

    Parameters
    ----------
    stats : NormalizationStats
        Per-channel mean/std (jax or numpy arrays).
    path : str | os.PathLike
        Destination ``.npz`` file.  Parent directories must exist.
    """
    mean = np.asarray(stats.mean, dtype=np.float64)
    std = np.asarray(stats.std, dtype=np.float64)
    if mean.shape != std.shape:
        raise ValueError(
            f"mean/std shape mismatch: mean {mean.shape} vs std {std.shape}."
        )
    np.savez(path, mean=mean, std=std)


def load_normalization_stats(path) -> NormalizationStats:
    """Load per-channel stats from an ``.npz`` sidecar written by
    :func:`save_normalization_stats`.

    Returns arrays as ``jnp`` so the result plugs directly into
    :func:`normalize` / :func:`denormalize` (and the SFNO PE bridge).
    """
    with np.load(path) as data:
        if "mean" not in data or "std" not in data:
            raise KeyError(
                f"normalization sidecar {path!r} missing 'mean'/'std' arrays; "
                f"found {list(data.keys())!r}."
            )
        mean = jnp.asarray(data["mean"])
        std = jnp.asarray(data["std"])
    return NormalizationStats(mean=mean, std=std)
