"""SST/SSS restoring to target profiles."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.physics.surface_forcing.config import RestoringConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput


def restoring_surface_forcing(
    T: jnp.ndarray,
    S: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: RestoringConfig,
) -> SurfaceForcingOutput:
    """Apply SST/SSS restoring to target profiles.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    grid : CubedSphereGrid
    cfg : RestoringConfig

    Returns
    -------
    SurfaceForcingOutput
    """
    shape_3d = T.shape
    dtype = T.dtype
    lat = grid.grid_lat  # protocol: (6, n, n) or (n_lat, n_lon) etc.

    # Target SST profile
    if cfg.T_profile == "cosine":
        # T* = T_eq - (T_eq - T_pole) * sin^2(lat)
        T_star = cfg.T_star_eq - (cfg.T_star_eq - cfg.T_star_pole) * jnp.sin(lat) ** 2
    else:
        T_star = jnp.full_like(lat, cfg.T_star_eq, dtype=dtype)

    S_star = jnp.full_like(lat, cfg.S_star, dtype=dtype)

    # Restoring tendency in surface layer only
    dT_dt = jnp.zeros(shape_3d, dtype=dtype)
    dS_dt = jnp.zeros(shape_3d, dtype=dtype)
    dT_dt = dT_dt.at[..., 0].set(-(T[..., 0] - T_star) / cfg.tau_T)
    dS_dt = dS_dt.at[..., 0].set(-(S[..., 0] - S_star) / cfg.tau_S)

    du_dt = jnp.zeros(shape_3d, dtype=dtype)
    dv_dt = jnp.zeros(shape_3d, dtype=dtype)
    Q_net = jnp.zeros_like(lat)
    tau_x = jnp.zeros_like(lat)
    tau_y = jnp.zeros_like(lat)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )
