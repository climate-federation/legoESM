"""Quadratic bottom drag: tau = -C_d * |u| * u."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.bottom_drag.config import QuadraticDragConfig
from legoesm.ocean.physics.bottom_drag.output import BottomDragOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def quadratic_bottom_drag(
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: QuadraticDragConfig,
) -> BottomDragOutput:
    """Apply quadratic bottom drag at the deepest level.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : QuadraticDragConfig

    Returns
    -------
    BottomDragOutput
    """
    eps = _EPS

    # Bottom layer thickness
    dz_bottom = z_coord.dz_ref[-1] * jacobian  # (6, n, n)

    # Speed at bottom level
    u_bot = u[..., -1]
    v_bot = v[..., -1]
    speed = jnp.sqrt(u_bot**2 + v_bot**2 + eps)

    # Drag: -C_d * |u| * u / dz_bottom
    inv_dz = 1.0 / jnp.maximum(dz_bottom, eps)
    drag_u = -cfg.C_d * speed * u_bot * inv_dz
    drag_v = -cfg.C_d * speed * v_bot * inv_dz

    du_dt = jnp.zeros_like(u)
    dv_dt = jnp.zeros_like(v)
    du_dt = du_dt.at[..., -1].set(drag_u)
    dv_dt = dv_dt.at[..., -1].set(drag_v)

    return BottomDragOutput(du_dt=du_dt, dv_dt=dv_dt)
