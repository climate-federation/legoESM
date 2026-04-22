"""COARE-like bulk air-sea flux formulation."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_P_ATM = 101325.0  # Standard atmosphere [Pa]


def _saturation_specific_humidity(T_K: jnp.ndarray) -> jnp.ndarray:
    """Saturation specific humidity at standard atmosphere pressure."""
    return saturation_specific_humidity(T_K, jnp.full_like(T_K, _P_ATM))


def bulk_formula_surface_forcing(
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: BulkFormulaConfig,
) -> SurfaceForcingOutput:
    """Apply bulk formulas for air-sea fluxes.

    Supports constant coefficients or stability-dependent MOST algorithms
    (COARE 3.0 or Large & Yeager 2004) selected via ``cfg.bulk_scheme``.

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
    q_sat = _saturation_specific_humidity(T_s)

    # Upward longwave: Q_lw_up = epsilon * sigma * T_s^4
    emissivity = 0.97
    Q_lw_up = emissivity * constants.sigma_sb * T_s ** 4

    if cfg.bulk_scheme in ("coare3", "large_yeager"):
        from legoesm.coupler.bulk_flux import compute_most_fluxes
        # Wind is zonal only (prescribed), zero meridional
        u_a = jnp.full_like(T_s, cfg.U_a, dtype=dtype)
        v_a = jnp.zeros_like(T_s)
        T_a = jnp.full_like(T_s, cfg.T_a, dtype=dtype)
        q_a = jnp.full_like(T_s, cfg.q_a, dtype=dtype)
        rho_a = jnp.full_like(T_s, cfg.rho_a, dtype=dtype)

        tau_x, tau_y, Q_sh, Q_lh, _ = compute_most_fluxes(
            u_a, v_a, T_a, q_a, T_s, q_sat, rho_a,
            z_ref=cfg.z_ref,
            z0_init=cfg.z0,
            scheme=cfg.bulk_scheme,
            n_iter=cfg.bulk_n_iter,
        )
        # Fluxes are positive upward; stress opposes wind
        tau_x = -tau_x  # flip to positive eastward
    else:
        # Constant coefficients: wind is zonal-only (u_a = U_a, v_a = 0)
        # to match the directional convention used in the MOST path.
        Q_sh = cfg.rho_a * cfg.c_pa * cfg.C_H * cfg.U_a * (T_s - cfg.T_a)
        Q_lh = cfg.rho_a * cfg.L_v * cfg.C_E * cfg.U_a * (q_sat - cfg.q_a)

        # tau = rho_a * C_D * |U_a| * (u_a, v_a)  — directional stress
        U_a_speed = jnp.abs(cfg.U_a)
        tau_x = jnp.full_like(T_s, cfg.rho_a * cfg.C_D * U_a_speed * cfg.U_a, dtype=dtype)
        tau_y = jnp.zeros_like(T_s)

    # Net heat flux (positive into ocean)
    Q_net = cfg.SW_down - Q_lw_up + cfg.LW_down - Q_sh - Q_lh

    # Convert to top-layer tendencies
    dz_0 = z_coord.dz_ref[0] * jacobian
    inv_rho_dz = 1.0 / (rho_0_ref * jnp.maximum(dz_0, 1e-10))
    inv_rho_csw_dz = 1.0 / (rho_0_ref * c_sw * jnp.maximum(dz_0, 1e-10))

    du_dt = jnp.zeros(shape_3d, dtype=dtype)
    dv_dt = jnp.zeros(shape_3d, dtype=dtype)
    du_dt = du_dt.at[..., 0].set(tau_x * inv_rho_dz)
    dv_dt = dv_dt.at[..., 0].set(tau_y * inv_rho_dz)

    dT_dt = jnp.zeros(shape_3d, dtype=dtype)
    dT_dt = dT_dt.at[..., 0].set(Q_net * inv_rho_csw_dz)

    # No freshwater forcing in basic bulk formulation
    dS_dt = jnp.zeros(shape_3d, dtype=dtype)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )
