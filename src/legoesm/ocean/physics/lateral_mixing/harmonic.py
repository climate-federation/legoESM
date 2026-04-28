"""Harmonic (Laplacian) lateral mixing.

Wraps the existing laplacian_viscosity_3d from mixing.py.
Velocity is masked (no-slip BC), tracers unmasked.
"""

from __future__ import annotations

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

    # Stack the (u, v) and (T, S) pairs along a trailing axis and fold
    # that into the level dim so ``laplacian_viscosity_3d`` (which uses
    # ``pad_halo_4d`` + ``divergence_3d``) runs ONCE on the thicker
    # ``(6, n, n, nlev*2)`` field.  The previous ``vmap``-over-the-pair
    # entered the operator twice and emitted two separate halo MPI
    # exchanges per call under multi-GPU sharding.

    # Velocity: masked before Laplacian (no-slip BC)
    du_dt = z
    dv_dt = z
    if cfg.A_h > 0:
        vel_stack = jnp.stack(
            [u * mask_3d, v * mask_3d], axis=-1,
        )  # (6, n, n, nlev, 2)
        n_face, n_i, n_j, nlev_t, n_pair = vel_stack.shape
        vel_flat = vel_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
        vel_lap_flat = laplacian_viscosity_3d(vel_flat, grid, cfg.A_h)
        vel_lap = vel_lap_flat.reshape(n_face, n_i, n_j, nlev_t, n_pair)
        du_dt = vel_lap[..., 0]
        dv_dt = vel_lap[..., 1]

    # Tracers: unmasked (smooth gradients at coastlines)
    dT_dt = z
    dS_dt = z
    if cfg.K_h > 0:
        tr_stack = jnp.stack([T, S], axis=-1)  # (6, n, n, nlev, 2)
        n_face, n_i, n_j, nlev_t, n_pair = tr_stack.shape
        tr_flat = tr_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
        tr_lap_flat = laplacian_viscosity_3d(tr_flat, grid, cfg.K_h)
        tr_lap = tr_lap_flat.reshape(n_face, n_i, n_j, nlev_t, n_pair)
        dT_dt = tr_lap[..., 0]
        dS_dt = tr_lap[..., 1]

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
