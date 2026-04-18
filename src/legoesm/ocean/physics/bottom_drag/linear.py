"""Linear bottom drag: du/dt = -r * u / dz_bottom.

The coefficient r has units [m/s] so that the bottom stress
tau = rho_0 * r * u is independent of vertical resolution.
This matches MITgcm's ``bottomDragLinear`` convention and is
consistent with the quadratic drag implementation.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.bottom_drag.config import LinearDragConfig
from legoesm.ocean.physics.bottom_drag.output import BottomDragOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)


def linear_bottom_drag(
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: LinearDragConfig,
) -> BottomDragOutput:
    """Apply linear bottom drag at the deepest level.

    du/dt = -r * u / dz_bottom   [m/s^2]

    where dz_bottom = dz_ref[-1] * jacobian is the physical thickness
    of the bottom layer.

    Parameters
    ----------
    u, v : array (..., nlev)
        Velocity components.
    z_coord : OceanZStarCoordinate
        Vertical coordinate (provides reference layer thicknesses).
    jacobian : array (...)
        z-star Jacobian (eta + H_bathy) / H_max at cell centers.
    cfg : LinearDragConfig
        Configuration with r in [m/s].

    Returns
    -------
    BottomDragOutput
    """
    # Bottom layer thickness
    dz_bottom = z_coord.dz_ref[-1] * jacobian  # (...)
    inv_dz = 1.0 / jnp.maximum(dz_bottom, _EPS)

    du_dt = jnp.zeros_like(u)
    dv_dt = jnp.zeros_like(v)
    du_dt = du_dt.at[..., -1].set(-cfg.r * u[..., -1] * inv_dz)
    dv_dt = dv_dt.at[..., -1].set(-cfg.r * v[..., -1] * inv_dz)
    return BottomDragOutput(du_dt=du_dt, dv_dt=dv_dt)
