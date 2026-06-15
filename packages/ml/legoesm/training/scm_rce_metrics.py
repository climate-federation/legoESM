"""Shared SCM-RCE profile metrics.

The SCM campaign and the gradient trainer both compare single-column RCE
profiles to CRM truth with the same vertically mass-weighted,
standard-deviation-normalized RMSE.  Keep the arithmetic here so the
derivative-free and AD paths cannot drift.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def weighted_std(profile: jax.Array, weights: jax.Array) -> jax.Array:
    """Mass-weighted vertical standard deviation."""
    profile = jnp.asarray(profile)
    weights = jnp.asarray(weights, dtype=profile.dtype)
    mean = jnp.sum(weights * profile)
    var = jnp.sum(weights * (profile - mean) ** 2)
    return jnp.sqrt(jnp.maximum(var, jnp.asarray(0.0, dtype=profile.dtype)))


def weighted_rmse(diff: jax.Array, weights: jax.Array) -> jax.Array:
    """Mass-weighted vertical RMSE."""
    diff = jnp.asarray(diff)
    weights = jnp.asarray(weights, dtype=diff.dtype)
    return jnp.sqrt(jnp.sum(weights * diff ** 2))


def score_profiles_jax(
    ref: Any,
    T_profile: jax.Array,
    qv_profile: jax.Array,
    qcond_profile: jax.Array,
    *,
    profile_floor: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return component and combined normalized SCM-vs-CRM profile scores."""
    T_profile = jnp.asarray(T_profile)
    qv_profile = jnp.asarray(qv_profile, dtype=T_profile.dtype)
    qcond_profile = jnp.asarray(qcond_profile, dtype=T_profile.dtype)
    dtype = T_profile.dtype
    weights = jnp.asarray(ref.mass_weights, dtype=dtype)
    T_ref = jnp.asarray(ref.T_ref, dtype=dtype)
    qv_ref = jnp.asarray(ref.qv_ref, dtype=dtype)
    qcond_ref = jnp.asarray(ref.qcond_ref, dtype=dtype)
    floor = jnp.asarray(profile_floor, dtype=dtype)

    T_std = jnp.maximum(weighted_std(T_ref, weights), floor)
    qv_std = jnp.maximum(weighted_std(qv_ref, weights), floor)
    qcond_std = jnp.maximum(weighted_std(qcond_ref, weights), floor)
    T_rmse = weighted_rmse((T_profile - T_ref) / T_std, weights)
    qv_rmse = weighted_rmse((qv_profile - qv_ref) / qv_std, weights)
    cloud_rmse = weighted_rmse((qcond_profile - qcond_ref) / qcond_std, weights)
    combined = jnp.sqrt((T_rmse**2 + qv_rmse**2 + cloud_rmse**2) / 3.0)
    return T_rmse, qv_rmse, cloud_rmse, combined
