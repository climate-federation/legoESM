"""Prescribed surface forcing: fixed wind stress and heat/freshwater fluxes."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def prescribed_surface_forcing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: PrescribedForcingConfig,
) -> SurfaceForcingOutput:
    """Apply prescribed surface forcing to the top ocean layer.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    grid : CubedSphereGrid
    cfg : PrescribedForcingConfig

    Returns
    -------
    SurfaceForcingOutput
    """
    nlev = u.shape[-1]
    shape_3d = u.shape
    dtype = u.dtype

    # Top layer thickness
    dz_0 = z_coord.dz_ref[0] * jacobian  # (6, n, n)
    inv_rho_dz = 1.0 / (rho_0_ref * jnp.maximum(dz_0, 1e-10))

    # Wind stress
    if cfg.wind_profile == "cosine_latitude":
        lat = grid.lat  # (6, n, n)
        # Cosine latitude profile: tau_x = -tau_max * cos(pi * lat / max_lat)
        lat_range = jnp.pi / 2.0  # 90 degrees
        tau_x = -cfg.tau_max * jnp.cos(jnp.pi * lat / lat_range)
        tau_y = jnp.zeros_like(tau_x)
    else:
        tau_x = jnp.full_like(dz_0, cfg.tau_x, dtype=dtype)
        tau_y = jnp.full_like(dz_0, cfg.tau_y, dtype=dtype)

    du_dt = jnp.zeros(shape_3d, dtype=dtype)
    dv_dt = jnp.zeros(shape_3d, dtype=dtype)
    du_dt = du_dt.at[..., 0].set(tau_x * inv_rho_dz)
    dv_dt = dv_dt.at[..., 0].set(tau_y * inv_rho_dz)

    # Heat flux: dT/dt = Q_net / (rho_0 * c_sw * dz_0)
    Q_net = jnp.full_like(dz_0, cfg.Q_net, dtype=dtype)
    inv_rho_csw_dz = 1.0 / (rho_0_ref * c_sw * jnp.maximum(dz_0, 1e-10))
    dT_dt = jnp.zeros(shape_3d, dtype=dtype)
    dT_dt = dT_dt.at[..., 0].set(Q_net * inv_rho_csw_dz)

    # Freshwater (virtual salt flux): dS/dt = -S * E_minus_P / dz_0
    dS_dt = jnp.zeros(shape_3d, dtype=dtype)
    if cfg.E_minus_P != 0.0:
        inv_dz = 1.0 / jnp.maximum(dz_0, 1e-10)
        dS_dt = dS_dt.at[..., 0].set(-S[..., 0] * cfg.E_minus_P * inv_dz)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )
