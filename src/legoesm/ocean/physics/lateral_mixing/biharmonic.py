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

    # Stack the (u, v) and (T, S) pairs along a trailing axis and fold
    # that into the level dim so ``hyperdiffusion_3d`` (∇⁴ = ∇²∇², two
    # ``pad_halo_4d`` halos per call) runs ONCE on the thicker
    # ``(6, n, n, nlev*2)`` field instead of being called twice in a
    # row.  Halves the halo MPI cost of biharmonic mixing under
    # multi-GPU sharding.
    du_dt = z
    dv_dt = z
    if cfg.B_h_momentum > 0:
        vel_stack = jnp.stack(
            [u * mask_3d, v * mask_3d], axis=-1,
        )  # (6, n, n, nlev, 2)
        n_face, n_i, n_j, nlev_t, n_pair = vel_stack.shape
        vel_flat = vel_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
        vel_hyper = hyperdiffusion_3d(vel_flat, grid, cfg.B_h_momentum)
        vel_hyper = vel_hyper.reshape(n_face, n_i, n_j, nlev_t, n_pair)
        du_dt = vel_hyper[..., 0]
        dv_dt = vel_hyper[..., 1]

    dT_dt = z
    dS_dt = z
    if cfg.B_h_tracer > 0:
        tr_stack = jnp.stack([T, S], axis=-1)  # (6, n, n, nlev, 2)
        n_face, n_i, n_j, nlev_t, n_pair = tr_stack.shape
        tr_flat = tr_stack.reshape(n_face, n_i, n_j, nlev_t * n_pair)
        tr_hyper = hyperdiffusion_3d(tr_flat, grid, cfg.B_h_tracer)
        tr_hyper = tr_hyper.reshape(n_face, n_i, n_j, nlev_t, n_pair)
        dT_dt = tr_hyper[..., 0]
        dS_dt = tr_hyper[..., 1]

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
