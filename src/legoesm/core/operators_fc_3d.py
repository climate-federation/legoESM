"""3D FC-Gram operator wrappers for cubed-sphere grids.

vmap of all 2D FC operators over vertical levels, following the
same pattern as operators_3d.py (moveaxis + vmap).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.operators_fc import (
    FCOperatorConfig,
    fc_gradient_x,
    fc_gradient_y,
    fc_divergence,
    fc_curl_z,
    fc_laplacian,
    fc_hyperdiffusion,
    fc_flux_divergence,
    fc_scalar_advection,
    fc_divergence_damping,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid


def _vmap_2d(fn_2d, *args_3d, grid, fc_config, extra_kwargs=None):
    """Generic vmap helper: moveaxis(-1,0) → vmap → moveaxis(0,-1)."""
    args_t = [jnp.moveaxis(a, -1, 0) for a in args_3d]
    kwargs = extra_kwargs or {}

    def single_level(*a_k):
        return fn_2d(*a_k, grid=grid, fc_config=fc_config, **kwargs)

    result = jax.vmap(single_level)(*args_t)
    return jnp.moveaxis(result, 0, -1)


def fc_gradient_x_3d(field_3d: jax.Array, grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig) -> jax.Array:
    """FC x-gradient at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return _vmap_2d(fc_gradient_x, field_3d, grid=grid, fc_config=fc_config)


def fc_gradient_y_3d(field_3d: jax.Array, grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig) -> jax.Array:
    """FC y-gradient at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return _vmap_2d(fc_gradient_y, field_3d, grid=grid, fc_config=fc_config)


def fc_divergence_3d(u_3d: jax.Array, v_3d: jax.Array,
                     grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig) -> jax.Array:
    """FC divergence at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return _vmap_2d(fc_divergence, u_3d, v_3d, grid=grid, fc_config=fc_config)


def fc_curl_z_3d(u_3d: jax.Array, v_3d: jax.Array,
                 grid: CubedSphereGrid,
                 fc_config: FCOperatorConfig) -> jax.Array:
    """FC vorticity at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return _vmap_2d(fc_curl_z, u_3d, v_3d, grid=grid, fc_config=fc_config)


def fc_laplacian_3d(field_3d: jax.Array, grid: CubedSphereGrid,
                    fc_config: FCOperatorConfig) -> jax.Array:
    """FC Laplacian at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return _vmap_2d(fc_laplacian, field_3d, grid=grid, fc_config=fc_config)


def fc_hyperdiffusion_3d(field_3d: jax.Array, grid: CubedSphereGrid,
                         fc_config: FCOperatorConfig,
                         coeff: float) -> jax.Array:
    """FC hyperdiffusion at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return _vmap_2d(fc_hyperdiffusion, field_3d, grid=grid,
                    fc_config=fc_config, extra_kwargs={"coeff": coeff})


def fc_flux_divergence_3d(q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
                          grid: CubedSphereGrid,
                          fc_config: FCOperatorConfig) -> jax.Array:
    """FC conservative flux divergence at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return _vmap_2d(fc_flux_divergence, q_3d, u_3d, v_3d,
                    grid=grid, fc_config=fc_config)


def fc_scalar_advection_3d(q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
                           grid: CubedSphereGrid,
                           fc_config: FCOperatorConfig) -> jax.Array:
    """FC scalar advection at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return _vmap_2d(fc_scalar_advection, q_3d, u_3d, v_3d,
                    grid=grid, fc_config=fc_config)


def fc_divergence_damping_3d(u_3d: jax.Array, v_3d: jax.Array,
                             grid: CubedSphereGrid,
                             fc_config: FCOperatorConfig,
                             ) -> tuple[jax.Array, jax.Array]:
    """FC divergence damping at all levels.

    Parameters
    ----------
    u_3d, v_3d : jax.Array, shape (6, n, n, nlev)
    grid : CubedSphereGrid
    fc_config : FCOperatorConfig

    Returns
    -------
    du_damp, dv_damp : jax.Array, shape (6, n, n, nlev)
    """
    u_t = jnp.moveaxis(u_3d, -1, 0)  # (nlev, 6, n, n)
    v_t = jnp.moveaxis(v_3d, -1, 0)

    def single_level(u_k, v_k):
        return fc_divergence_damping(u_k, v_k, grid, fc_config)

    du_t, dv_t = jax.vmap(single_level)(u_t, v_t)
    return jnp.moveaxis(du_t, 0, -1), jnp.moveaxis(dv_t, 0, -1)
