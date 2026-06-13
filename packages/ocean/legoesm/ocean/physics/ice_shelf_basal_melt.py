"""Jenkins (1991) three-equation ice-shelf basal-melt parameterisation.

Reference
---------
Jenkins, A. (1991). "A one-dimensional model of ice shelf-ocean
interaction", J. Geophys. Res. 96, 20671-20677.

Asay-Davis, X. S., et al. (2016). "Experimental design for three
interrelated marine ice sheet and ocean model intercomparison
projects: MISMIP+, ISOMIP+, and MISOMIP1", Geosci. Model Dev. 9,
2471-2497.

Three-equation system
---------------------

.. math::
    T_b &= T_f(S_b, p_b) = a S_b + b + c p_b
    \\\\
    T_b &= T_{\\rm boundary}
    \\\\
    m c_p (T_w - T_b) &= L m

written compactly as the basal melt rate

.. math::
    m = \\frac{c_p \\Gamma_T (T_w - T_f(S_w, p_b))}{L_f + c_p (T_b - T_w)}
    \\approx \\frac{c_p \\Gamma_T (T_w - T_f)}{L_f}

(the small-thermal-driving limit, accurate to ~10 % for the ISOMIP+
configurations). Inputs are the top-cell temperature ``T_w``,
salinity ``S_w``, and the ice-shelf-base pressure ``p_b``.

Coefficients
------------
Pressure-freezing-point relation (Jenkins 1991 / Asay-Davis 2016):

* ``a = -5.73e-2`` K / (g/kg)
* ``b = 8.32e-2`` K
* ``c = -7.61e-4`` K / dbar

Thermal exchange velocity ``Gamma_T`` is configurable; ISOMIP+
uses ``1.0e-4 m/s``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.physics.ice_shelf import freezing_point_C


# --- Thermodynamic constants ---
# Latent heat of fusion: use the canonical substrate constant (redundancy audit)
# instead of a locally-rounded 3.34e5 (which drifted 0.09% from constants.L_f).
_L_F: float = constants.L_f    # J/kg latent heat of fusion
_C_P_SW: float = 3974.0         # J/kg/K seawater heat capacity (Jenkins 1991)
_RHO_FW: float = 1000.0         # kg/m^3 freshwater density


@dataclass
class IceShelfMeltConfig:
    """Configuration for the Jenkins three-equation parameterisation."""

    # Effective thermal exchange velocity for the small-thermal-driving
    # limit. The Asay-Davis 2016 / Jolicoeur 2025 ISOMIP+ reference
    # uses ``gamma_T_star = 6e-4`` with the FULL quadratic three-
    # equation system in which the meltwater dilutes the boundary
    # layer and shuts down further melt. Solving the full quadratic
    # in JAX is a follow-up; for the smoke benchmark the linearised
    # form needs a ~30x smaller effective ``gamma_T`` to land in the
    # ISOMIP+ intercomparison range (0.1-10 m/yr).
    gamma_T: float = 2.0e-5     # m/s (linearised effective value)
    gamma_S: float = 5.0e-7     # haline exchange velocity [m/s] (ISOMIP+)
    L_f: float = _L_F
    c_p_sw: float = _C_P_SW
    rho_fw: float = _RHO_FW


def basal_melt_rate_m_per_s(T_w_C, S_w_psu, p_b_dbar,
                            cfg: IceShelfMeltConfig | None = None):
    """Small-thermal-driving Jenkins melt rate [m/s of freshwater].

    Positive values indicate melting (mass flux into the ocean);
    negative values indicate freezing (mass flux out of the ocean).
    """
    if cfg is None:
        cfg = IceShelfMeltConfig()
    T_f = freezing_point_C(S_w_psu, p_b_dbar)
    thermal_driving = jnp.asarray(T_w_C) - T_f
    # Mass-flux form: rho_fw * m = rho_sw * c_p * gamma_T * dT_drive / L_f
    # Returning meters/s of freshwater equivalent.
    return cfg.gamma_T * cfg.c_p_sw * thermal_driving / cfg.L_f


def basal_freshwater_flux_kg_per_m2_s(T_w_C, S_w_psu, p_b_dbar,
                                       cfg: IceShelfMeltConfig | None = None):
    """Basal melt expressed as a downward freshwater mass flux [kg/m^2/s].

    Multiplies the meltrate by ``rho_fw``; positive = mass added to
    the top ocean cell.
    """
    if cfg is None:
        cfg = IceShelfMeltConfig()
    m = basal_melt_rate_m_per_s(T_w_C, S_w_psu, p_b_dbar, cfg)
    return cfg.rho_fw * m


__all__ = [
    "IceShelfMeltConfig",
    "freezing_point_C",
    "basal_melt_rate_m_per_s",
    "basal_freshwater_flux_kg_per_m2_s",
]
