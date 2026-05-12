"""Constant-coefficient vertical mixing.

Wraps the existing vertical_diffusion from mixing.py with constant
viscosity A_v and diffusivity K_v.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.physics.mixing import vertical_diffusion
from legoesm.ocean.physics.vertical_mixing.config import ConstantVerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def constant_vertical_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: ConstantVerticalMixingConfig,
    apply_diffusion: bool = True,
) -> VerticalMixingOutput:
    """Apply constant-coefficient vertical diffusion to u, v, T, S.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
        Velocity components.
    T, S : array (6, n, n, nlev)
        Temperature and salinity.
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : ConstantVerticalMixingConfig

    Returns
    -------
    VerticalMixingOutput
    """
    nlev = u.shape[-1]

    # Velocities with viscosity A_v / tracers with diffusivity K_v.
    # When ``apply_diffusion`` is False, the diffusion is deferred to a
    # backward-Euler implicit solve in the dynamics step.
    if apply_diffusion:
        vel = jnp.stack([u, v], axis=0)
        vel_tend = jax.vmap(
            lambda q: vertical_diffusion(q, z_coord, jacobian, cfg.A_v),
            in_axes=0, out_axes=0,
        )(vel)

        tracers = jnp.stack([T, S], axis=0)
        tr_tend = jax.vmap(
            lambda q: vertical_diffusion(q, z_coord, jacobian, cfg.K_v),
            in_axes=0, out_axes=0,
        )(tracers)
    else:
        zero_uv = jnp.zeros_like(u)
        vel_tend = jnp.stack([zero_uv, zero_uv], axis=0)
        zero_T = jnp.zeros_like(T)
        tr_tend = jnp.stack([zero_T, zero_T], axis=0)

    # Constant K/A diagnostics at interfaces
    K_diag = jnp.full((*u.shape[:-1], nlev - 1), cfg.K_v, dtype=u.dtype)
    A_diag = jnp.full((*u.shape[:-1], nlev - 1), cfg.A_v, dtype=u.dtype)

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0],
        dS_dt=tr_tend[1],
        K_v=K_diag,
        A_v=A_diag,
    )
