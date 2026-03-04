"""Linear bottom drag: tau = -r * u."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.bottom_drag.config import LinearDragConfig
from legoesm.ocean.physics.bottom_drag.output import BottomDragOutput


def linear_bottom_drag(
    u: jnp.ndarray,
    v: jnp.ndarray,
    cfg: LinearDragConfig,
) -> BottomDragOutput:
    """Apply linear bottom drag at the deepest level.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    cfg : LinearDragConfig

    Returns
    -------
    BottomDragOutput
    """
    du_dt = jnp.zeros_like(u)
    dv_dt = jnp.zeros_like(v)
    du_dt = du_dt.at[..., -1].set(-cfg.r * u[..., -1])
    dv_dt = dv_dt.at[..., -1].set(-cfg.r * v[..., -1])
    return BottomDragOutput(du_dt=du_dt, dv_dt=dv_dt)
