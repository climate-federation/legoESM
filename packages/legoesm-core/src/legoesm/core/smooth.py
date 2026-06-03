"""Smooth differentiable approximations to discontinuous operations.

In a differentiable model, all operations must have well-defined gradients.
These functions replace common discontinuities (Heaviside steps, max, min,
clamping) with smooth approximations parameterized by a sharpness factor.

Higher sharpness -> closer to the true discontinuous function but with
steeper gradients. Lower sharpness -> smoother gradients but less accurate.
"""

import jax
import jax.numpy as jnp


def sigmoid_switch(x: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Smooth approximation to the Heaviside step function.

    H(x) ~ sigmoid(sharpness * x)

    Parameters
    ----------
    x : array
        Input values. Output ~ 0 where x < 0, ~ 1 where x > 0.
    sharpness : float
        Controls transition steepness. Default 100.

    Returns
    -------
    array : Values in (0, 1).
    """
    return jax.nn.sigmoid(sharpness * x)


def smooth_max(a: jax.Array, b: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Smooth differentiable approximation to max(a, b).

    Uses the LogSumExp trick: max(a, b) ~ log(exp(s*a) + exp(s*b)) / s

    Parameters
    ----------
    a, b : arrays
        Input values (will be broadcast to the same shape).
    sharpness : float
        Controls approximation quality. Default 100.

    Returns
    -------
    array : Smooth approximation to element-wise max(a, b).
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    # Broadcast to same shape before stacking
    a_s, b_s = jnp.broadcast_arrays(a * sharpness, b * sharpness)
    stacked = jnp.stack([a_s, b_s], axis=0)
    return jax.nn.logsumexp(stacked, axis=0) / sharpness


def smooth_min(a: jax.Array, b: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Smooth differentiable approximation to min(a, b).

    min(a, b) = -max(-a, -b)
    """
    return -smooth_max(-a, -b, sharpness)


def smooth_clamp(
    x: jax.Array, lo: float, hi: float, sharpness: float = 100.0
) -> jax.Array:
    """Smooth differentiable clamp to [lo, hi].

    Parameters
    ----------
    x : array
        Input values.
    lo, hi : float
        Lower and upper bounds.
    sharpness : float
        Controls transition steepness.

    Returns
    -------
    array : Values smoothly clamped to approximately [lo, hi].
    """
    return smooth_min(smooth_max(x, jnp.asarray(lo), sharpness), jnp.asarray(hi), sharpness)


def smooth_relu(x: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Smooth approximation to ReLU: max(x, 0).

    Uses softplus: log(1 + exp(s*x)) / s
    """
    return jax.nn.softplus(x * sharpness) / sharpness


def smooth_abs(x: jax.Array, epsilon: float = 1e-7) -> jax.Array:
    """Smooth approximation to |x|.

    Uses sqrt(x^2 + epsilon) which is differentiable everywhere.
    """
    return jnp.sqrt(x * x + epsilon)
