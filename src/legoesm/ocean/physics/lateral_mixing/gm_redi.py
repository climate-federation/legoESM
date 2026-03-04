"""Gent-McWilliams (1990) / Redi isopycnal diffusion.

GM bolus transport + Redi along-isopycnal diffusion with DM95 tapering.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.operators_3d import gradient_x_3d, gradient_y_3d
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import compute_buoyancy_frequency
from legoesm.ocean.physics.mixing import (
    laplacian_viscosity_3d,
    vertical_diffusion_variable_K,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def gm_redi_lateral_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: GMRediConfig,
) -> LateralMixingOutput:
    """Apply GM-Redi lateral mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    grid : CubedSphereGrid
    cfg : GMRediConfig

    Returns
    -------
    LateralMixingOutput
    """
    eps = 1e-12
    nlev = u.shape[-1]
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # --- Isopycnal slopes at interfaces ---
    # Horizontal density gradients at full levels
    drho_dx = gradient_x_3d(rho, grid)
    drho_dy = gradient_y_3d(rho, grid)

    # Average to interfaces
    drho_dx_half = 0.5 * (drho_dx[..., :-1] + drho_dx[..., 1:])
    drho_dy_half = 0.5 * (drho_dy[..., :-1] + drho_dy[..., 1:])

    # Vertical density gradient at interfaces
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    drho_dz = (rho[..., :-1] - rho[..., 1:]) / jnp.maximum(dz_half, eps)
    # drho_dz should be negative for stable stratification
    drho_dz_safe = jnp.minimum(drho_dz, -eps)

    # Slopes: S_x = -(drho/dx) / (drho/dz), S_y = -(drho/dy) / (drho/dz)
    S_x = -drho_dx_half / drho_dz_safe
    S_y = -drho_dy_half / drho_dz_safe

    # --- DM95 tapering ---
    S_mag = jnp.sqrt(S_x**2 + S_y**2 + eps)
    taper = 0.5 * (1.0 + jnp.tanh((cfg.S_max - S_mag) / (0.1 * cfg.S_max + eps)))

    S_x = S_x * taper
    S_y = S_y * taper

    # --- GM bolus velocity ---
    # Streamfunction at interfaces: psi_x = kappa_GM * S_x
    psi_x = cfg.kappa_GM * S_x  # (6, n, n, nlev-1)
    psi_y = cfg.kappa_GM * S_y

    # Bolus velocity at full levels: u_bolus[k] = -(psi[k-1/2] - psi[k+1/2]) / dz[k]
    # Pad with zeros at surface and bottom (psi = 0 at boundaries)
    z_pad = jnp.zeros((*psi_x.shape[:-1], 1), dtype=psi_x.dtype)
    psi_x_ext = jnp.concatenate([z_pad, psi_x, z_pad], axis=-1)
    psi_y_ext = jnp.concatenate([z_pad, psi_y, z_pad], axis=-1)

    u_bolus = -(psi_x_ext[..., :-1] - psi_x_ext[..., 1:]) / jnp.maximum(dz_actual, eps)
    v_bolus = -(psi_y_ext[..., :-1] - psi_y_ext[..., 1:]) / jnp.maximum(dz_actual, eps)

    # GM tracer tendency: bolus advection of T, S
    dT_dx = gradient_x_3d(T, grid)
    dT_dy = gradient_y_3d(T, grid)
    dS_dx = gradient_x_3d(S, grid)
    dS_dy = gradient_y_3d(S, grid)

    dT_gm = -(u_bolus * dT_dx + v_bolus * dT_dy)
    dS_gm = -(u_bolus * dS_dx + v_bolus * dS_dy)

    # --- Redi diffusion ---
    # Horizontal part: kappa_Redi * nabla^2(tracer)
    dT_redi_h = laplacian_viscosity_3d(T, grid, cfg.kappa_Redi)
    dS_redi_h = laplacian_viscosity_3d(S, grid, cfg.kappa_Redi)

    # Vertical correction: d/dz(kappa_Redi * (S_x^2 + S_y^2) * dT/dz)
    S2_half = S_x**2 + S_y**2
    K_redi_vert = cfg.kappa_Redi * S2_half  # At interfaces

    tracers = jnp.stack([T, S], axis=0)
    redi_vert = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K_redi_vert),
        in_axes=0, out_axes=0,
    )(tracers)

    dT_dt = dT_gm + dT_redi_h + redi_vert[0]
    dS_dt = dS_gm + dS_redi_h + redi_vert[1]

    # GM does not directly produce momentum tendencies
    du_dt = jnp.zeros_like(u)
    dv_dt = jnp.zeros_like(v)

    return LateralMixingOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt)
