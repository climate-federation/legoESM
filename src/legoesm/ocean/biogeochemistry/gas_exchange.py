"""Air-sea gas exchange for CO2 and O2.

Gas transfer velocity: Wanninkhof (2014), k = a <U10^2> (Sc/660)^-1/2 with
a = 0.251 cm/h per (m/s)^2, Schmidt numbers from the same paper (valid
-2..40 degC).  O2 saturation: Garcia & Gordon (1992) combined fit.  These
match MARBL (marbl_surface_flux_mod, marbl_oxygen, marbl_schmidt_number_mod).

All functions are JAX-compatible.

References
----------
- Wanninkhof, R. (2014). Relationship between wind speed and gas exchange
  over the ocean revisited. Limnol. Oceanogr.: Methods, 12, 351-362.
- Garcia, H. E. & Gordon, L. I. (1992). Oxygen solubility in seawater:
  better fitting equations. Limnol. Oceanogr., 37, 1307-1312.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.biogeochemistry.carbonate import (
    co2_solubility,
    solve_carbonate_system,
)
from legoesm.ocean.biogeochemistry.config import AirSeaCO2Diagnostics

# Wanninkhof (2014) coefficient 0.251 cm/h / (m/s)^2 expressed in s/m.
_KW_W14 = 0.251 * 1.0e-2 / 3600.0


def schmidt_number_co2(T_degC: jnp.ndarray) -> jnp.ndarray:
    """Schmidt number of CO2 in seawater (Wanninkhof 2014, fit range -2..40 degC)."""
    T = jnp.clip(T_degC, -2.0, 40.0)
    return 2116.8 + T * (-136.25 + T * (4.7353 + T * (-0.092307 + T * 0.0007555)))


def schmidt_number_o2(T_degC: jnp.ndarray) -> jnp.ndarray:
    """Schmidt number of O2 in seawater (Wanninkhof 2014, fit range -2..40 degC)."""
    T = jnp.clip(T_degC, -2.0, 40.0)
    return 1920.4 + T * (-135.6 + T * (5.2122 + T * (-0.10939 + T * 0.00093777)))


def piston_velocity(
    U10_sqr: jnp.ndarray,
    schmidt: jnp.ndarray,
    ice_frac: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Gas transfer velocity [m/s]: (1 - f_ice) 0.251 <U10^2> (Sc/660)^-1/2.

    ``U10_sqr`` is the mean of the squared 10-m wind speed [m^2/s^2]; use it
    rather than (mean U10)^2 when averaging over variable winds.
    """
    return (1.0 - ice_frac) * _KW_W14 * U10_sqr * jnp.sqrt(660.0 / schmidt)


def gas_transfer_velocity(
    U10: jnp.ndarray,
    T_degC: jnp.ndarray,
) -> jnp.ndarray:
    """CO2 piston velocity [m/s] for a 10-m wind speed U10 [m/s]."""
    return piston_velocity(U10 ** 2, schmidt_number_co2(T_degC))


def air_sea_co2_flux(
    DIC_surf: jnp.ndarray,
    ALK_surf: jnp.ndarray,
    T_surf: jnp.ndarray,
    S_surf: jnp.ndarray,
    U10: jnp.ndarray,
    pCO2_atm: float = 400.0,
    rho_sw: float = constants.rho_ocean,
) -> AirSeaCO2Diagnostics:
    """Air-sea CO2 flux [mol C/m^2/s], positive into the ocean.

    F = k_w K0 rho_sw (pCO2_atm - pCO2_ocean) 1e-6, DIC/ALK in mol/m^3.
    """
    pCO2_ocean, pH = solve_carbonate_system(DIC_surf, ALK_surf, T_surf, S_surf, rho_sw)
    k_w = gas_transfer_velocity(U10, T_surf)
    K0 = co2_solubility(T_surf, S_surf)
    flux_co2 = k_w * K0 * rho_sw * (pCO2_atm - pCO2_ocean) * 1.0e-6
    return AirSeaCO2Diagnostics(
        pCO2_ocean=pCO2_ocean,
        pH=pH,
        flux_co2=flux_co2,
        k_w=k_w,
    )


def o2_saturation(T_degC: jnp.ndarray, S_psu: jnp.ndarray) -> jnp.ndarray:
    """O2 saturation [mol/m^3] at 1 atm total pressure (Garcia & Gordon 1992, eq. 8).

    Combined-fit coefficients without the erroneous A3*Ts^2 term.
    Check value: T = 10 degC, S = 35 -> 0.282015 mol/m^3.
    """
    T_K = T_degC + constants.T_freeze
    Ts = jnp.log(((constants.T_freeze + 25.0) - T_degC) / T_K)
    ln_c = (2.00907 + Ts * (3.22014 + Ts * (4.05010 + Ts * (4.94457
            + Ts * (-0.256847 + Ts * 3.88767))))
            + S_psu * ((-6.24523e-3 + Ts * (-7.37614e-3 + Ts * (-1.03410e-2
                        + Ts * (-8.17083e-3)))) + S_psu * (-4.88682e-7)))
    return jnp.exp(ln_c) / constants.V_molar_O2  # ml/L -> mol/m^3


def air_sea_o2_flux(
    O2_surf: jnp.ndarray,
    T_surf: jnp.ndarray,
    S_surf: jnp.ndarray,
    U10_sqr: jnp.ndarray,
    P_atm: jnp.ndarray | float = 1.0,
    ice_frac: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Air-sea O2 flux [same units as O2_surf * m/s], positive into the ocean.

    F = k_O2 (P_atm O2_sat(T, S) - O2), with P_atm in atm and O2_sat in
    the units of ``O2_surf`` taken as mol/m^3.
    """
    k = piston_velocity(U10_sqr, schmidt_number_o2(T_surf), ice_frac)
    return k * (P_atm * o2_saturation(T_surf, S_surf) - O2_surf)
