"""Shared vertical-profile comparison primitives (mass-weighted, AD-safe).

Both the SCM-RCE metrics (``training/scm_rce_metrics.py``) and the LES-truth suite
score assembly (``atmosphere/les_suite/score.py``) compare column profiles to a
reference with the same vertically mass-weighted, standard-deviation-normalized
RMSE. This is the single low-level home for that arithmetic (``legoesm.core`` is
below both ``training`` and ``atmosphere.les_suite``, so neither import is
circular). ``scm_rce_metrics`` re-exports :func:`safe_sqrt`, :func:`weighted_std`,
and :func:`weighted_rmse` from here for backward compatibility — do not fork them.

All functions are pure and JAX-safe (finite gradients at a perfect fit).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def safe_sqrt(x: Array) -> Array:
    """``sqrt`` with a finite gradient at ``x == 0``.

    ``sqrt(0)`` is finite (0) but its derivative ``1/(2 sqrt(0))`` is ``inf``, so a
    PERFECT fit (residual 0 — the exact minimiser a trainer targets) would produce
    a NaN gradient. The double-``where`` masks the zero out of the differentiated
    branch, giving the exact value ``sqrt(0) == 0`` on the forward pass AND a
    finite (zero) gradient on the backward pass.
    """
    x = jnp.asarray(x)
    safe_x = jnp.where(x > 0, x, jnp.ones_like(x))
    return jnp.where(x > 0, jnp.sqrt(safe_x), jnp.zeros_like(x))


def weighted_std(profile: Array, weights: Array) -> Array:
    """Mass-weighted vertical standard deviation (weights should sum to 1)."""
    profile = jnp.asarray(profile)
    weights = jnp.asarray(weights, dtype=profile.dtype)
    mean = jnp.sum(weights * profile)
    var = jnp.sum(weights * (profile - mean) ** 2)
    return safe_sqrt(var)


def weighted_rmse(diff: Array, weights: Array) -> Array:
    """Mass-weighted vertical RMSE of ``diff`` (weights should sum to 1).

    Uses :func:`safe_sqrt` so the gradient stays finite at a perfect fit.
    """
    diff = jnp.asarray(diff)
    weights = jnp.asarray(weights, dtype=diff.dtype)
    return safe_sqrt(jnp.sum(weights * diff ** 2))


def layer_weights_from_heights(heights_m: Array) -> Array:
    """Normalized layer-thickness weights for a surface-first height grid.

    Returns ``(nz,)`` non-negative weights summing to 1, proportional to each
    level's control-volume thickness (midpoint rule: half the distance to each
    neighbour; endpoints get their one-sided half-layer). Used as the mass-weight
    proxy when a density profile is not carried; multiply by ``ρ`` externally for a
    true mass weighting. Heights must be strictly increasing with index.
    """
    z = jnp.asarray(heights_m)
    if z.ndim != 1 or z.shape[0] < 2:
        raise ValueError("heights_m must be 1-D with >=2 levels")
    dz = jnp.diff(z)
    # control-volume thickness at each level (midpoint rule)
    thick = jnp.concatenate(
        [
            dz[:1] * 0.5,                      # surface half-layer
            0.5 * (dz[:-1] + dz[1:]),          # interior
            dz[-1:] * 0.5,                     # top half-layer
        ]
    )
    total = jnp.sum(thick)
    return thick / jnp.where(total > 0, total, jnp.ones_like(total))
