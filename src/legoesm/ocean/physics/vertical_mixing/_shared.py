"""Shared vertical-mixing kernels.

Factored to remove the identical stack -> vmap -> unstack block that the
constant- and Richardson-coefficient schemes duplicated (they differ only in
the per-field diffusion function and the viscosity/diffusivity coefficients).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def vmap_vertical_diffusion(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    vel_coeff,
    tracer_coeff,
    diffuse_fn: Callable,
    apply_diffusion: bool,
):
    """Apply a per-field vertical-diffusion function to the [u, v] and [T, S]
    pairs via a single vmap each, returning stacked tendencies.

    The velocity pair uses ``vel_coeff`` (viscosity), the tracer pair uses
    ``tracer_coeff`` (diffusivity); ``diffuse_fn(q, coeff)`` returns the
    per-field tendency (it captures z_coord / jacobian / dt). Shared by the
    constant and Richardson schemes, which differ only in ``diffuse_fn`` and the
    coefficients.

    When ``apply_diffusion`` is False, returns zero-tendency stacks (the
    diffusion is deferred to the implicit backward-Euler solve in the dynamics
    step).

    Returns
    -------
    vel_tend, tr_tend : array ``(2, ...)`` each — ``[du_dt, dv_dt]`` and
        ``[dT_dt, dS_dt]``.
    """
    if apply_diffusion:
        vel_tend = jax.vmap(
            lambda q: diffuse_fn(q, vel_coeff), in_axes=0, out_axes=0,
        )(jnp.stack([u, v], axis=0))
        tr_tend = jax.vmap(
            lambda q: diffuse_fn(q, tracer_coeff), in_axes=0, out_axes=0,
        )(jnp.stack([T, S], axis=0))
    else:
        zero_uv = jnp.zeros_like(u)
        vel_tend = jnp.stack([zero_uv, zero_uv], axis=0)
        zero_T = jnp.zeros_like(T)
        tr_tend = jnp.stack([zero_T, zero_T], axis=0)
    return vel_tend, tr_tend
