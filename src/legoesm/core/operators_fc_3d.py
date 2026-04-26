"""3D FC-Gram operator wrappers for cubed-sphere grids.

The underlying 2D FC operators in :mod:`legoesm.core.operators_fc` are
ndim-aware: they accept either ``(6, n, n)`` or ``(6, n, n, nlev)`` and
do a single 4D ``pad_halo`` MPI exchange in the latter case.  The 3D
wrappers are therefore thin pass-throughs to the 2D functions, replacing
the previous ``vmap_over_levels`` pattern that forced ``nlev`` separate
``pad_halo`` MPI exchanges per fc_*_3d call.
"""

from __future__ import annotations

import jax

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


def fc_gradient_x_3d(field_3d: jax.Array, grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig,
                     padded: jax.Array | None = None) -> jax.Array:
    """FC x-gradient at all levels. (6,n,n,nlev) -> (6,n,n,nlev).

    Optional ``padded=`` skips the internal halo exchange — pre-pad
    ``field_3d`` once and pass it to both ``fc_gradient_x_3d`` and
    ``fc_gradient_y_3d`` to halve the halo cost of paired calls.
    """
    return fc_gradient_x(field_3d, grid, fc_config, padded=padded)


def fc_gradient_y_3d(field_3d: jax.Array, grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig,
                     padded: jax.Array | None = None) -> jax.Array:
    """FC y-gradient at all levels. (6,n,n,nlev) -> (6,n,n,nlev).

    See :func:`fc_gradient_x_3d` for ``padded=`` usage.
    """
    return fc_gradient_y(field_3d, grid, fc_config, padded=padded)


def fc_divergence_3d(u_3d: jax.Array, v_3d: jax.Array,
                     grid: CubedSphereGrid,
                     fc_config: FCOperatorConfig,
                     padded: tuple[jax.Array, jax.Array] | None = None) -> jax.Array:
    """FC divergence at all levels. (6,n,n,nlev) -> (6,n,n,nlev).

    Optional ``padded=(u_pad, v_pad)`` shares the vector halo with a
    co-located ``fc_curl_z_3d`` on the same input.
    """
    return fc_divergence(u_3d, v_3d, grid, fc_config, padded=padded)


def fc_curl_z_3d(u_3d: jax.Array, v_3d: jax.Array,
                 grid: CubedSphereGrid,
                 fc_config: FCOperatorConfig,
                 padded: tuple[jax.Array, jax.Array] | None = None) -> jax.Array:
    """FC vorticity at all levels. (6,n,n,nlev) -> (6,n,n,nlev).

    See :func:`fc_divergence_3d` for ``padded=`` usage.
    """
    return fc_curl_z(u_3d, v_3d, grid, fc_config, padded=padded)


def fc_laplacian_3d(field_3d: jax.Array, grid: CubedSphereGrid,
                    fc_config: FCOperatorConfig,
                    padded: jax.Array | None = None) -> jax.Array:
    """FC Laplacian at all levels. (6,n,n,nlev) -> (6,n,n,nlev).

    Optional ``padded=`` skips the internal halo exchange — share with
    a co-located ``fc_hyperdiffusion_3d`` on the same input to halve
    the halo cost.
    """
    return fc_laplacian(field_3d, grid, fc_config, padded=padded)


def fc_hyperdiffusion_3d(field_3d: jax.Array, grid: CubedSphereGrid,
                         fc_config: FCOperatorConfig,
                         coeff: float,
                         padded: jax.Array | None = None) -> jax.Array:
    """FC hyperdiffusion at all levels. (6,n,n,nlev) -> (6,n,n,nlev).

    See :func:`fc_laplacian_3d` for ``padded=`` usage.
    """
    return fc_hyperdiffusion(field_3d, grid, fc_config, coeff, padded=padded)


def fc_flux_divergence_3d(q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
                          grid: CubedSphereGrid,
                          fc_config: FCOperatorConfig) -> jax.Array:
    """FC conservative flux divergence at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return fc_flux_divergence(q_3d, u_3d, v_3d, grid, fc_config)


def fc_scalar_advection_3d(q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
                           grid: CubedSphereGrid,
                           fc_config: FCOperatorConfig) -> jax.Array:
    """FC scalar advection at all levels. (6,n,n,nlev) -> (6,n,n,nlev)."""
    return fc_scalar_advection(q_3d, u_3d, v_3d, grid, fc_config)


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
    return fc_divergence_damping(u_3d, v_3d, grid, fc_config)
