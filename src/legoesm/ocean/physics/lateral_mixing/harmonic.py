"""Harmonic (Laplacian) lateral mixing.

Wraps the existing laplacian_viscosity_3d from mixing.py.
Velocity is masked (no-slip BC), tracers unmasked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.physics.mixing import laplacian_viscosity_3d
from legoesm.ocean.physics.lateral_mixing.config import HarmonicConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput


def harmonic_lateral_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: HarmonicConfig,
) -> LateralMixingOutput:
    """Apply Laplacian lateral mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    mask : array (6, n, n) — land mask (1=ocean)
    grid : CubedSphereGrid
    cfg : HarmonicConfig

    Returns
    -------
    LateralMixingOutput
    """
    mask_3d = mask[..., jnp.newaxis]
    z = jnp.zeros_like(u)

    # Velocity: masked before Laplacian (no-slip BC)
    du_dt = z
    dv_dt = z
    if cfg.A_h > 0:
        vel_masked = jnp.stack([u * mask_3d, v * mask_3d], axis=0)
        vel_lap = jax.vmap(
            lambda q: laplacian_viscosity_3d(q, grid, cfg.A_h),
            in_axes=0, out_axes=0,
        )(vel_masked)
        du_dt = vel_lap[0]
        dv_dt = vel_lap[1]

    # Tracers: unmasked (smooth gradients at coastlines)
    dT_dt = z
    dS_dt = z
    if cfg.K_h > 0:
        tracers = jnp.stack([T, S], axis=0)
        tr_lap = jax.vmap(
            lambda q: laplacian_viscosity_3d(q, grid, cfg.K_h),
            in_axes=0, out_axes=0,
        )(tracers)
        dT_dt = tr_lap[0]
        dS_dt = tr_lap[1]

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
