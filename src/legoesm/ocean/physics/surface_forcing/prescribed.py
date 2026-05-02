"""Prescribed surface forcing: fixed wind stress and heat/freshwater fluxes."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.physics.surface_forcing.wind_profiles import compute_wind_stress
from legoesm.ocean.vertical import OceanZStarCoordinate


def prescribed_surface_forcing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid,  # Any grid with grid_lat property (GridProtocol)
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

    # Wind stress from shared grid-agnostic computation
    tau_x, tau_y = compute_wind_stress(grid.grid_lat, cfg)

    # Build top-layer-only tendencies via ``jnp.pad`` along the
    # trailing axis instead of allocating a fresh full ``(*, nlev)``
    # zero buffer per field and scattering the surface row.  Single
    # Pad HLO op each — this physics fires every ocean step in
    # configurations that use the prescribed surface forcing.
    nlev = shape_3d[-1]
    pad_axes = ((0, 0),) * (len(shape_3d) - 1)
    du_dt = jnp.pad(
        (tau_x * inv_rho_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )
    dv_dt = jnp.pad(
        (tau_y * inv_rho_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )

    # Heat flux: dT/dt = Q_net / (rho_0 * c_sw * dz_0)
    Q_net = jnp.full_like(dz_0, cfg.Q_net, dtype=dtype)
    inv_rho_csw_dz = 1.0 / (rho_0_ref * c_sw * jnp.maximum(dz_0, 1e-10))
    dT_dt = jnp.pad(
        (Q_net * inv_rho_csw_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )

    # Freshwater (virtual salt flux): dS/dt = +S * E_minus_P / dz_0
    # Positive E-P means net evaporation → water leaves → salt concentrates → dS/dt > 0
    if cfg.E_minus_P != 0.0:
        inv_dz = 1.0 / jnp.maximum(dz_0, 1e-10)
        dS_dt = jnp.pad(
            (S[..., 0] * cfg.E_minus_P * inv_dz)[..., None],
            (*pad_axes, (0, nlev - 1)),
        )
    else:
        dS_dt = jnp.zeros(shape_3d, dtype=dtype)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )
