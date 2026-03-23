"""Change-of-variable preconditioning for 4D-Var.

Minimizing J~(v) = J(x_b + B^{1/2} v) is equivalent to minimizing J(x)
but with a much better-conditioned Hessian:
  I + B^{1/2} H^T R^{-1} H B^{1/2}
instead of:
  B^{-1} + H^T R^{-1} H
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp


def preconditioned_cost_fn(
    cost_fn: Callable,
    B,
    x_b: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build preconditioned cost function J~(v) = J(x_b + B^{1/2} v).

    Parameters
    ----------
    cost_fn : callable
        J(x) in x-space.
    B : background error covariance
        Must have .sqrt_multiply(v) method.
    x_b : jax.Array
        Background in control space.

    Returns
    -------
    callable
        J~: v -> scalar. Minimizing J~ over v gives x* = x_b + B^{1/2} v*.
    """
    def J_tilde(v):
        x = x_b + B.sqrt_multiply(v)
        return cost_fn(x)
    return J_tilde
