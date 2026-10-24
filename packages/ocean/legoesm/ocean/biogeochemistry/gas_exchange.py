"""Air-sea CO2 gas exchange.

Implements the Wanninkhof (2014) parameterization for the gas transfer
velocity and computes the net air-sea CO2 flux from the pCO2 difference.

All functions are JAX-compatible.

Faithfulness
------------
``tests/ocean/unit/test_bgc_carbonate_faithful.py`` pins these forms to
round-off (rel 1e-12) against an INDEPENDENT reimplementation with the
published Wanninkhof (2014) coefficients.

FAITHFUL:
- ``schmidt_number_co2`` — Wanninkhof (2014) Table 1 quartic fit for CO2 in
  seawater (2116.8, -136.25, 4.7353, -0.092307, 0.0007555).
- ``gas_transfer_velocity`` — Wanninkhof (2014) k_w = 0.251*U10^2*(Sc/660)^-0.5
  [cm/hr], with the conventional 660 Schmidt normalization (the fit itself
  evaluates to Sc ~= 668 at 20 degC, S=35 — 660 is the reference, not that value).

DEPARTURES:
- The Schmidt fit is valid 0-40 degC; ``schmidt_number_co2`` CLIPS T to that
  range (a silent clamp — Sc is held flat and its dT gradient is zero outside).
- ``air_sea_co2_flux`` sign convention: F = k_w*K0*rho_sw*(pCO2_atm - pCO2_ocean),
  POSITIVE = into the ocean (ocean uptake).  The cm/hr -> m/s conversion is exact
  (divide by 3600 s/hr and 100 cm/m).

References
----------
- Wanninkhof, R. (2014). Relationship between wind speed and gas exchange
  over the ocean revisited. Limnol. Oceanogr.: Methods, 12, 351-362.
"""

from __future__ import annotations

import jax.numpy as jnp
from legoesm.ocean.biogeochemistry.carbonate import (
    co2_solubility,
    solve_carbonate_system,
)
from legoesm.ocean.biogeochemistry.config import AirSeaCO2Diagnostics

from legoesm import constants


def schmidt_number_co2(T_degC: jnp.ndarray) -> jnp.ndarray:
    """Schmidt number for CO2 in seawater.

    Wanninkhof (2014), Table 1, polynomial fit for 0-40 degC.

    Parameters
    ----------
    T_degC : array
        Sea surface temperature [degC].

    Returns
    -------
    Sc : array
        Schmidt number (dimensionless).
    """
    T = jnp.clip(T_degC, 0.0, 40.0)
    return 2116.8 - 136.25 * T + 4.7353 * T ** 2 - 0.092307 * T ** 3 + 0.0007555 * T ** 4


def gas_transfer_velocity(
    U10: jnp.ndarray,
    T_degC: jnp.ndarray,
) -> jnp.ndarray:
    """Gas transfer velocity for CO2 (piston velocity).

    Wanninkhof (2014): k_w = 0.251 * U10^2 * (Sc/660)^(-0.5)

    Parameters
    ----------
    U10 : array
        10-meter wind speed [m/s].
    T_degC : array
        SST [degC].

    Returns
    -------
    k_w : array
        Piston velocity [m/s].
    """
    Sc = schmidt_number_co2(T_degC)
    k_w_cm_hr = 0.251 * U10 ** 2 * (Sc / 660.0) ** (-0.5)
    # cm/hr -> m/s: exact, divide by 3600 s/hr and 100 cm/m (1 cm/hr = 2.7778e-6 m/s)
    return k_w_cm_hr / 3600.0 / 100.0


def air_sea_co2_flux(
    DIC_surf: jnp.ndarray,
    ALK_surf: jnp.ndarray,
    T_surf: jnp.ndarray,
    S_surf: jnp.ndarray,
    U10: jnp.ndarray,
    pCO2_atm: float = 400.0,
    rho_sw: float = constants.rho_ocean,
) -> AirSeaCO2Diagnostics:
    """Compute air-sea CO2 flux.

    F_CO2 = k_w * K0 * rho_sw * (pCO2_atm - pCO2_ocean) [mol C/m^2/s]

    Positive flux = into the ocean (ocean uptake).

    Parameters
    ----------
    DIC_surf : array
        Surface DIC [mol C/m^3].
    ALK_surf : array
        Surface alkalinity [mol eq/m^3].
    T_surf : array
        SST [degC].
    S_surf : array
        SSS [PSU].
    U10 : array
        10-m wind speed [m/s].
    pCO2_atm : float
        Atmospheric pCO2 [uatm].
    rho_sw : float
        Seawater density [kg/m^3].

    Returns
    -------
    AirSeaCO2Diagnostics
        Contains pCO2_ocean, pH, flux_co2, k_w.
    """
    # Solve for ocean pCO2
    pCO2_ocean, pH = solve_carbonate_system(DIC_surf, ALK_surf, T_surf, S_surf, rho_sw)

    # Gas transfer
    k_w = gas_transfer_velocity(U10, T_surf)
    K0 = co2_solubility(T_surf, S_surf)

    # Net flux: mol/(kg*atm) * kg/m^3 * m/s * uatm * 1e-6 atm/uatm
    # = mol/m^2/s
    flux_co2 = k_w * K0 * rho_sw * (pCO2_atm - pCO2_ocean) * 1.0e-6

    return AirSeaCO2Diagnostics(
        pCO2_ocean=pCO2_ocean,
        pH=pH,
        flux_co2=flux_co2,
        k_w=k_w,
    )


# --------------------------------------------------------------------------- #
# Oxygen (npzd_v2).  Schmidt number: Wanninkhof (2014) Table 1 for O2;        #
# saturation: Garcia & Gordon (1992) combined fit (eq. 8, without the         #
# erroneous A3*Ts^2 term).  Check value: T=10 degC, S=35 -> 0.282015 mol/m^3. #
# --------------------------------------------------------------------------- #
_V_MOLAR_O2_ML_PER_MOL = 22391.6   # ml of O2 per mol at STP (Garcia & Gordon use 22.3916 L/mol)


def schmidt_number_o2(T_degC: jnp.ndarray) -> jnp.ndarray:
    """Schmidt number of O2 in seawater (Wanninkhof 2014, fit range -2..40 degC)."""
    T = jnp.clip(T_degC, -2.0, 40.0)
    return 1920.4 + T * (-135.6 + T * (5.2122 + T * (-0.10939 + T * 0.00093777)))


def o2_saturation(T_degC: jnp.ndarray, S_psu: jnp.ndarray) -> jnp.ndarray:
    """O2 saturation concentration [mol/m^3] at 1 atm (Garcia & Gordon 1992)."""
    T_K = T_degC + constants.T_freeze
    Ts = jnp.log(((constants.T_freeze + 25.0) - T_degC) / T_K)
    ln_c = (2.00907 + Ts * (3.22014 + Ts * (4.05010 + Ts * (4.94457
            + Ts * (-0.256847 + Ts * 3.88767))))
            + S_psu * ((-6.24523e-3 + Ts * (-7.37614e-3 + Ts * (-1.03410e-2
                        + Ts * (-8.17083e-3)))) + S_psu * (-4.88682e-7)))
    return jnp.exp(ln_c) / _V_MOLAR_O2_ML_PER_MOL * 1.0e3   # ml/L -> mol/m^3


def air_sea_o2_flux(
    O2_surf: jnp.ndarray,
    T_surf: jnp.ndarray,
    S_surf: jnp.ndarray,
    U10: jnp.ndarray,
    P_atm: jnp.ndarray | float = 1.0,
) -> jnp.ndarray:
    """Air-sea O2 flux [mol O2/m^2/s], positive into the ocean.

    F = k_O2 (P_atm * O2_sat(T, S) - O2_surf), k from Wanninkhof (2014) with
    the O2 Schmidt number; same cm/hr -> m/s conversion as the CO2 path.
    """
    Sc = schmidt_number_o2(T_surf)
    k_cm_hr = 0.251 * U10 ** 2 * (Sc / 660.0) ** (-0.5)
    k = k_cm_hr / 3600.0 / 100.0
    return k * (P_atm * o2_saturation(T_surf, S_surf) - O2_surf)


def piston_velocity(
    U10_sqr: jnp.ndarray,
    schmidt: jnp.ndarray,
    ice_frac: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Gas transfer velocity [m/s] for any gas: (1 - f_ice) * 0.251 <U10^2> (Sc/660)^-1/2.

    ``U10_sqr`` is the mean of the squared 10-m wind speed [m^2/s^2].  Same
    Wanninkhof (2014) coefficient as ``gas_transfer_velocity`` (0.251 cm/h per
    (m/s)^2, converted to s/m); used by the MARBL surface-flux port.
    """
    return (1.0 - ice_frac) * (0.251 * 1.0e-2 / 3600.0) * U10_sqr * jnp.sqrt(660.0 / schmidt)
