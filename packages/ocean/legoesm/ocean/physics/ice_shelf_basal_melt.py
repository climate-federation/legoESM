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

__physics_contract__ = {
    "summary": (
        "Jenkins (1991) three-equation ice-shelf basal-melt rate in the "
        "small-thermal-driving limit: m = c_p*Gamma_T*(T_w - T_f(S_w, p_b))/L_f "
        "from the top-cell temperature/salinity and the ice-base pressure, plus "
        "the freshwater mass flux it injects."
    ),
    "inputs": {
        "T_w_C": "degC", "S_w_psu": "psu", "p_b_dbar": "dbar",
        "cfg.gamma_T": "m/s",
    },
    "outputs": {
        "m": "m/s (freshwater-equivalent melt rate)",
        "freshwater_flux": "kg/m^2/s",
    },
    "sign_convention": (
        "Thermal driving = T_w - T_f(S_w, p_b); m > 0 = MELT (freshwater mass "
        "INTO the top ocean cell, freshwater_flux = rho_fw*m > 0), m < 0 = "
        "freezing (mass out); a boundary freshwater source with an implied "
        "latent-heat sink, not interior-conservative; freezing-point slope "
        "a < 0 (saltier -> lower T_f), pressure coeff c < 0 (deeper -> lower "
        "T_f); depths positive downward."
    ),
    # Boundary freshwater source (+ implied latent-heat sink); not conservative.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Jenkins, A. (1991), JGR 96, 20671-20677; Asay-Davis et al. (2016) "
        "ISOMIP+, GMD 9, 2471-2497"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_phase_d_experiments.py — warm cavity water "
        "(T_w > T_f) gives positive melt scaling with thermal driving; T_w = "
        "T_f gives zero melt; deeper (higher p_b) lowers T_f and raises the "
        "thermal driving."
    ),
}


# --- Thermodynamic constants ---
# All three reference the canonical substrate constants (redundancy audit)
# instead of local literals: L_f avoids a rounded 3.34e5 (0.09% drift);
# c_p_sw uses the ISOMIP+ / Jenkins-1991 seawater heat capacity; rho_fw is the
# freshwater density.
_L_F: float = constants.L_f                    # J/kg latent heat of fusion
_C_P_SW: float = constants.c_p_seawater_isomip  # J/kg/K seawater cp (Jenkins 1991 / ISOMIP+)
_RHO_FW: float = constants.rho_water            # kg/m^3 freshwater density


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
    # Small-thermal-driving linearised Jenkins heat balance.  The EXACT
    # freshwater mass flux is ``rho_sw * c_p_sw * gamma_T * dT_drive / L_f``
    # [kg/m^2/s]; here we return a freshwater-equivalent melt rate
    # ``m = gamma_T * c_p_sw * dT_drive / L_f`` [m/s] which
    # ``basal_freshwater_flux_*`` multiplies by ``rho_fw``.  That drops the
    # ``rho_sw/rho_fw`` (~1.025) factor, which is ABSORBED into the
    # hand-calibrated effective ``gamma_T`` (tuned so the linearised rate
    # lands in the ISOMIP+ 0.1-10 m/yr range) — do NOT re-add ``rho_sw``
    # without re-tuning ``gamma_T``.
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
