"""COARE-like bulk air-sea flux formulation."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.core.bulk_flux import compute_most_fluxes
from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing._shared import surface_tendency_factors
from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

__physics_contract__ = {
    "summary": (
        "COARE-like bulk air-sea flux surface forcing: turbulent (sensible + "
        "latent) heat and wind stress from a prescribed near-surface "
        "atmospheric state, applied as sources to the top ocean layer "
        "(constant-Cd or MOST/COARE-3.0/Large-Yeager stability-dependent "
        "exchange coefficients). Salinity tendency is identically 0 here: the "
        "evaporative virtual-salt flux is applied via the dedicated freshwater "
        "channel (not this module) to avoid double-counting."
    ),
    "inputs": {
        "T": "degC (SST)", "S": "psu", "jacobian": "1 (z-star dimensionless)",
        "cfg.U_a": "m/s", "cfg.T_a": "K", "cfg.q_a": "kg/kg",
        "cfg.C_H": "1 (dimensionless exchange coeff)",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s",
        "dS_dt": "psu/s (identically 0; salinity via the freshwater channel)",
        "Q_net": "W/m^2", "tau_x": "N/m^2", "tau_y": "N/m^2",
    },
    "sign_convention": (
        "Surface boundary fluxes deposited in the TOP layer only (sources/sinks, "
        "NOT interior-conservative); Q_net > 0 warms the ocean "
        "(dT/dt = Q_net/(rho_0*c_sw*dz_0)); wind stress accelerates the surface "
        "layer in the stress direction; dS_dt = 0 in this module (the "
        "evaporative virtual-salt flux is handled by the freshwater channel); "
        "exchange coefficients C_D,C_H,C_E >= 0; z positive up. An unknown "
        "bulk_scheme raises ValueError."
    ),
    # Air-sea boundary source/sink; not a conservative interior operator.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Fairall et al. (2003) COARE 3.0, J. Climate 16, 571-591; Large & "
        "Yeager (2004/2009) CORE-II bulk formulae"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_bulk_flux_ly09.py + "
        "tests/ocean/unit/test_ncar_bulk.py + "
        "tests/ocean/unit/test_surface_forcing_sign_convention.py — warm/humid "
        "air over cool water gives downward (warming) Q_net; the wind-stress "
        "sign matches convention; an unknown bulk_scheme raises."
    ),
}


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

    T_s = T[..., 0] + constants.T_freeze  # (6, n, n)
    # Surface saturation humidity over SALINE ocean water: Large & Yeager (2004)
    # / OMIP prescribe q_s = 0.98 * q_sat(SST, p) (~2% reduction of saturation
    # vapour pressure over seawater).  Omitting it biases the latent heat flux
    # (and evaporative freshwater) high.
    q_sat = cfg.q_sat_salinity_factor * saturation_specific_humidity(
        T_s, jnp.full_like(T_s, constants.p_atm_std)
    )

    # Upward longwave from a grey surface: surface emission PLUS the
    # reflected component of the incident longwave.  An earlier form
    # used only ``ε σ T_s^4`` and absorbed the spurious ``(1-ε)·LW_down``
    # into Q_net, biasing the ocean heat flux when emissivity < 1.
    # Codex iter-41 #2 (mirrors the iter-13 sea-ice fix).
    Q_lw_up = (
        cfg.emissivity * constants.sigma_sb * T_s ** 4
        + (1.0 - cfg.emissivity) * cfg.LW_down
    )

    _valid_bulk = ("constant", "coare3", "large_yeager")
    if cfg.bulk_scheme not in _valid_bulk:
        raise ValueError(
            f"Unknown bulk_scheme {cfg.bulk_scheme!r}; expected one of {_valid_bulk}."
        )
    if cfg.bulk_scheme in ("coare3", "large_yeager"):
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
        # ``compute_most_fluxes`` returns stress in the ATMOSPHERIC convention
        # (tau = -rho*u*^2 * u/|U|, i.e. it OPPOSES the wind — drag on the air).
        # The ocean is forced by the stress in the WIND direction, so flip the
        # sign of BOTH components consistently: positive = stress into the ocean
        # along the wind (eastward for u_a>0, northward for v_a>0).  Flipping
        # only tau_x (leaving tau_y in the atmospheric convention) would drive
        # the ocean the WRONG way meridionally once v_a != 0.
        tau_x, tau_y = -tau_x, -tau_y
    else:
        # Constant coefficients: wind is zonal-only (u_a = U_a, v_a = 0)
        # to match the directional convention used in the MOST path.
        # Heat fluxes scale with WIND SPEED |U_a| (the air-sea exchange
        # rate is set by how vigorously the air is moving, not by the
        # signed zonal component).  The previous code used signed
        # ``cfg.U_a``, which would invert the sign of Q_sh and Q_lh
        # under easterlies (cfg.U_a < 0) — i.e. a strong easterly would
        # falsely *warm* a cool ocean.  Stress, on the other hand, IS
        # directional: tau_x = rho_a · C_D · |U| · u.
        U_a_speed = jnp.abs(cfg.U_a)
        Q_sh = cfg.rho_a * cfg.c_pa * cfg.C_H * U_a_speed * (T_s - cfg.T_a)
        Q_lh = cfg.rho_a * cfg.L_v * cfg.C_E * U_a_speed * (q_sat - cfg.q_a)

        # tau = rho_a * C_D * |U_a| * (u_a, v_a)  — directional stress
        tau_x = jnp.full_like(T_s, cfg.rho_a * cfg.C_D * U_a_speed * cfg.U_a, dtype=dtype)
        tau_y = jnp.zeros_like(T_s)

    # Net heat flux (positive into ocean)
    Q_net = cfg.SW_down - Q_lw_up + cfg.LW_down - Q_sh - Q_lh

    # Convert to top-layer tendencies.  Mask land columns (jacobian = 0):
    # without it ``1/max(dz_0, 1e-10)`` yields ~1e7-scale heat/momentum
    # tendencies on dry cells that contaminate neighbouring ocean faces when
    # interpolated.  Mirrors the ``is_ocean`` guard in ``prescribed.py`` /
    # ``external.py`` (codex review, finding #5).
    dz_0 = z_coord.dz_ref[0] * jacobian
    is_ocean = dz_0 > cfg.min_wet_cell_thickness_m
    # Wet-cell flux→tendency reciprocals (#518: shared helper; eos rho_0/c_sw).
    inv_rho_dz, inv_rho_csw_dz = surface_tendency_factors(
        is_ocean, dz_0, rho_0_ref, c_sw)

    # Pad with zero on trailing axis instead of alloc-zeros +
    # scatter — single Pad HLO op per field.  Same pattern as the
    # ``prescribed.py`` and ``restoring.py`` rewrites.
    nlev = shape_3d[-1]
    pad_axes = ((0, 0),) * (len(shape_3d) - 1)
    du_dt = jnp.pad(
        (tau_x * inv_rho_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )
    dv_dt = jnp.pad(
        (tau_y * inv_rho_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )
    dT_dt = jnp.pad(
        (Q_net * inv_rho_csw_dz)[..., None], (*pad_axes, (0, nlev - 1)),
    )
    # Salinity forcing is intentionally delegated to the model's dedicated
    # freshwater channel (P - E + runoff via ``model.step(freshwater=...)``),
    # NOT emitted here.  Adding a virtual-salt flux from this scheme's
    # evaporation (E = Q_lh / L_v) would DOUBLE-COUNT the evaporative salt
    # concentration already carried by the freshwater channel.  This "basic"
    # bulk scheme therefore reports dS_dt = 0 by design; a run that needs
    # salinity forcing must route E - P - R through the freshwater channel.
    dS_dt = jnp.zeros(shape_3d, dtype=dtype)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )
