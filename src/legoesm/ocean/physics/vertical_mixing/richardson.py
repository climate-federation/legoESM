"""Pacanowski & Philander (1981) Richardson-number dependent mixing.

K = K_0 / (1 + alpha * Ri)^n + K_bg
A = K * Pr_t + A_bg
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import compute_buoyancy_frequency
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing.config import RichardsonVerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def richardson_vertical_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: RichardsonVerticalMixingConfig,
) -> VerticalMixingOutput:
    """Apply Richardson-number dependent vertical mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
        In-situ density.
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : RichardsonVerticalMixingConfig

    Returns
    -------
    VerticalMixingOutput
    """
    eps = _EPS

    # N^2 at interfaces
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)

    # Shear^2 at interfaces
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    du = u[..., :-1] - u[..., 1:]
    dv = v[..., :-1] - v[..., 1:]
    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)

    # Richardson number
    Ri = N2 / jnp.maximum(S2, eps)
    Ri = jnp.maximum(Ri, 0.0)  # Clip negative Ri (unstable → max mixing)

    # Diffusivity and viscosity at interfaces
    K_v = cfg.K_0 / (1.0 + cfg.alpha * Ri) ** cfg.n + cfg.K_bg
    A_v = K_v * cfg.Pr_t + cfg.A_bg

    # Apply variable-K vertical diffusion.  ``vertical_diffusion_variable_K``
    # natively handles arbitrary leading batch axes (its arithmetic uses
    # ``[..., :-1]`` and ``[..., 1:]`` indexing), so a single direct call
    # on the leading-stacked tensor is equivalent to a per-component
    # ``jax.vmap`` — but without the vmap wrapper.  ``A_v`` /
    # ``K_v`` (shape ``(..., nlev-1)``) broadcast cleanly across the
    # extra leading axis.
    vel = jnp.stack([u, v], axis=0)
    vel_tend = vertical_diffusion_variable_K(vel, z_coord, jacobian, A_v)

    tracers = jnp.stack([T, S], axis=0)
    tr_tend = vertical_diffusion_variable_K(tracers, z_coord, jacobian, K_v)

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0],
        dS_dt=tr_tend[1],
        K_v=K_v,
        A_v=A_v,
    )
