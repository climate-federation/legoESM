"""Biharmonic lateral mixing.

Wraps hyperdiffusion_3d from core/operators_3d.py.
Velocity is masked (no-slip), tracers unmasked.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.operators_3d import hyperdiffusion_3d
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.physics.lateral_mixing.config import BiharmonicConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput


def biharmonic_lateral_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: BiharmonicConfig,
) -> LateralMixingOutput:
    """Apply biharmonic lateral mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    mask : array (6, n, n)
    grid : CubedSphereGrid
    cfg : BiharmonicConfig

    Returns
    -------
    LateralMixingOutput
    """
    mask_3d = mask[..., jnp.newaxis]
    z = jnp.zeros_like(u)

    du_dt = z
    dv_dt = z
    if cfg.B_h_momentum > 0:
        du_dt = hyperdiffusion_3d(u * mask_3d, grid, cfg.B_h_momentum)
        dv_dt = hyperdiffusion_3d(v * mask_3d, grid, cfg.B_h_momentum)

    dT_dt = z
    dS_dt = z
    if cfg.B_h_tracer > 0:
        dT_dt = hyperdiffusion_3d(T, grid, cfg.B_h_tracer)
        dS_dt = hyperdiffusion_3d(S, grid, cfg.B_h_tracer)

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
