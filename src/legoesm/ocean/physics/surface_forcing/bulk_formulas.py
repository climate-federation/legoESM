"""COARE-like bulk air-sea flux formulation."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def _saturation_specific_humidity(T_K: jnp.ndarray) -> jnp.ndarray:
    """Clausius-Clapeyron saturation specific humidity over water.

    Uses the August-Roche-Magnus formula.

    Parameters
    ----------
    T_K : array
        Temperature [K].

    Returns
    -------
    array : Saturation specific humidity [kg/kg].
    """
    T_C = T_K - 273.15
    e_s = 611.2 * jnp.exp(17.67 * T_C / (T_C + 243.5))
    p_atm = 101325.0  # Standard atmosphere [Pa]
    return 0.622 * e_s / (p_atm - 0.378 * e_s)


def bulk_formula_surface_forcing(
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: BulkFormulaConfig,
) -> SurfaceForcingOutput:
    """Apply COARE-like bulk formulas for air-sea fluxes.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : BulkFormulaConfig

    Returns
    -------
    SurfaceForcingOutput
    """
    shape_3d = T.shape
    dtype = T.dtype

    # SST in Kelvin
    T_s = T[..., 0] + 273.15  # (6, n, n)

    # Sensible heat flux: Q_sh = rho_a * c_pa * C_H * U_a * (T_s - T_a)
    Q_sh = cfg.rho_a * cfg.c_pa * cfg.C_H * cfg.U_a * (T_s - cfg.T_a)

    # Latent heat flux: Q_lh = rho_a * L_v * C_E * U_a * (q_sat - q_a)
    q_sat = _saturation_specific_humidity(T_s)
    Q_lh = cfg.rho_a * cfg.L_v * cfg.C_E * cfg.U_a * (q_sat - cfg.q_a)

    # Upward longwave: Q_lw_up = epsilon * sigma * T_s^4
    sigma_sb = 5.67e-8  # Stefan-Boltzmann constant
    emissivity = 0.97
    Q_lw_up = emissivity * sigma_sb * T_s**4

    # Net heat flux (positive into ocean)
    Q_net = cfg.SW_down - Q_lw_up + cfg.LW_down - Q_sh - Q_lh

    # Wind stress: tau = rho_a * C_D * U_a^2
    tau = cfg.rho_a * cfg.C_D * cfg.U_a**2
    tau_x = jnp.full_like(T[..., 0], tau, dtype=dtype)
    tau_y = jnp.zeros_like(T[..., 0])

    # Convert to top-layer tendencies
    dz_0 = z_coord.dz_ref[0] * jacobian
    inv_rho_dz = 1.0 / (rho_0_ref * jnp.maximum(dz_0, 1e-10))
    inv_rho_csw_dz = 1.0 / (rho_0_ref * c_sw * jnp.maximum(dz_0, 1e-10))

    du_dt = jnp.zeros(shape_3d, dtype=dtype)
    dv_dt = jnp.zeros(shape_3d, dtype=dtype)
    du_dt = du_dt.at[..., 0].set(tau_x * inv_rho_dz)

    dT_dt = jnp.zeros(shape_3d, dtype=dtype)
    dT_dt = dT_dt.at[..., 0].set(Q_net * inv_rho_csw_dz)

    # No freshwater forcing in basic bulk formulation
    dS_dt = jnp.zeros(shape_3d, dtype=dtype)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )
