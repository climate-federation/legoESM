"""Prescribed surface forcing: fixed wind stress and heat/freshwater fluxes."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing._shared import (
    WindStressConvention,
    surface_tendency_factors,
    wind_stress_sign,
)
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.physics.surface_forcing.wind_profiles import compute_wind_stress
from legoesm.ocean.vertical import OceanZStarCoordinate

__physics_contract__ = {
    "summary": (
        "Prescribed surface forcing: apply a fixed/analytic wind-stress profile "
        "and prescribed heat and freshwater fluxes to the top ocean layer "
        "(idealized spin-up forcing)."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "degC", "S": "psu",
        "jacobian": "1 (z-star dimensionless)", "grid.grid_lat": "rad",
        "cfg.tau_max": "N/m^2", "cfg.Q_net": "W/m^2", "cfg.E_minus_P": "m/s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
        "Q_net": "W/m^2", "tau_x": "N/m^2", "tau_y": "N/m^2",
    },
    "sign_convention": (
        "Top-layer boundary sources (NOT interior-conservative); applies +tau "
        "as an on-ocean stress DIRECTLY (accelerates the surface layer in the "
        "stress direction; note the OPPOSITE sign to the external / "
        "atmosphere-convention scheme); prescribed Q_net > 0 warms; "
        "cfg.E_minus_P > 0 (net evaporation) SALINIFIES via a virtual salt flux "
        "(dS/dt = S*(E-P)/dz_0 > 0), < 0 (net precipitation) freshens; z "
        "positive up."
    ),
    # Prescribed surface boundary source/sink; not interior-conservative.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Idealized prescribed air-sea forcing; wind-stress profiles after "
        "Bryan (1987) JPO 17 / Munk (1950) double-gyre & ACC configurations"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_surface_forcing_sign_convention.py — a "
        "prescribed zonal wind stress accelerates the surface layer eastward; "
        "a positive Q_net warms the top layer; the rest of the column is "
        "untouched."
    ),
}


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
    dtype = u.dtype

    # Top layer thickness — zero on land cells (where jacobian = 0).
    # Use jnp.where to set inv_rho_dz to 0 on land, preventing huge values
    # (~1e7) that would contaminate ocean cells via interpolation to u/v
    # faces.  The previous jnp.maximum(dz_0, 1e-10) clamp produced large
    # but finite values on land — fine when du_dt is masked at point of
    # use, but catastrophic when du_dt is interpolated to neighbouring
    # u/v faces (T->u, T->v) where the ocean side gets contaminated.
    dz_0 = z_coord.dz_ref[0] * jacobian  # T-point shape; 0 on land
    is_ocean = dz_0 > cfg.min_wet_cell_thickness_m
    # Wet-cell flux→tendency reciprocals (#518: shared helper; eos rho_0/c_sw).
    inv_rho_dz, inv_rho_csw_dz = surface_tendency_factors(
        is_ocean, dz_0, rho_0_ref, c_sw)

    # Wind stress from shared grid-agnostic computation (T-point shape)
    tau_x, tau_y = compute_wind_stress(grid.grid_lat, cfg)

    # T-point tendencies (cell-center stagger).  Zero on land via inv_rho_dz.
    # OCEAN_DIRECT convention (+tau on-ocean), named for self-documentation
    # (#518 item 11); sign is a static +1.0 — behaviour unchanged.
    _tau_sign = wind_stress_sign(WindStressConvention.OCEAN_DIRECT)
    du_dt_T = _tau_sign * tau_x * inv_rho_dz   # T-point shape
    dv_dt_T = _tau_sign * tau_y * inv_rho_dz   # T-point shape

    # Detect C-grid staggering: lat-lon C-grid has u at (n_lat, n_lon+1)
    # and v at (n_lat+1, n_lon), while T is at (n_lat, n_lon).  On A-grid
    # or cubed-sphere, u and v share T's shape — no interpolation needed.
    T_2d_shape = inv_rho_dz.shape
    u_2d_shape = u.shape[:-1]
    v_2d_shape = v.shape[:-1]
    is_cgrid_u = (
        len(u_2d_shape) == 2 and len(T_2d_shape) == 2
        and u_2d_shape[0] == T_2d_shape[0]
        and u_2d_shape[1] == T_2d_shape[1] + 1
    )
    is_cgrid_v = (
        len(v_2d_shape) == 2 and len(T_2d_shape) == 2
        and v_2d_shape[0] == T_2d_shape[0] + 1
        and v_2d_shape[1] == T_2d_shape[1]
    )

    # Interpolate du_dt to u-faces (lon-stagger, periodic wrap)
    if is_cgrid_u:
        du_dt_uf = 0.5 * (du_dt_T + jnp.roll(du_dt_T, 1, axis=1))
        du_dt_uf = jnp.concatenate([du_dt_uf, du_dt_uf[:, 0:1]], axis=1)
    else:
        du_dt_uf = du_dt_T

    # Interpolate dv_dt to v-faces (lat-stagger, zero at poles)
    if is_cgrid_v:
        dv_dt_int = 0.5 * (dv_dt_T[:-1] + dv_dt_T[1:])  # (n_lat-1, n_lon)
        dv_dt_vf = jnp.pad(dv_dt_int, ((1, 1), (0, 0)))
    else:
        dv_dt_vf = dv_dt_T

    # Build top-layer-only tendencies via jnp.pad along the trailing axis
    pad_axes_u = ((0, 0),) * (len(u.shape) - 1)
    pad_axes_v = ((0, 0),) * (len(v.shape) - 1)
    pad_axes_T = ((0, 0),) * (len(T.shape) - 1)

    du_dt = jnp.pad(du_dt_uf[..., None], (*pad_axes_u, (0, nlev - 1)))
    dv_dt = jnp.pad(dv_dt_vf[..., None], (*pad_axes_v, (0, nlev - 1)))

    # Heat flux: dT/dt = Q_net / (rho_0 * c_sw * dz_0)  — T-point
    # (inv_rho_csw_dz from the shared helper above).
    Q_net = jnp.full_like(dz_0, cfg.Q_net, dtype=dtype)
    dT_dt = jnp.pad(
        (Q_net * inv_rho_csw_dz)[..., None], (*pad_axes_T, (0, nlev - 1)),
    )

    # Freshwater (virtual salt flux): T-point
    if cfg.E_minus_P != 0.0:
        inv_dz = jnp.where(is_ocean, 1.0 / jnp.maximum(dz_0, 1.0e-10), 0.0)
        dS_dt = jnp.pad(
            (S[..., 0] * cfg.E_minus_P * inv_dz)[..., None],
            (*pad_axes_T, (0, nlev - 1)),
        )
    else:
        dS_dt = jnp.zeros(T.shape, dtype=dtype)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )
