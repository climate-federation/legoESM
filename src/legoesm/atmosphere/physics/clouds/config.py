"""Configuration for diagnostic cloud fraction schemes.

Provides CloudConfig for controlling cloud fraction diagnosis and
cloud optical property computation for radiation coupling.
"""

from __future__ import annotations

from typing import NamedTuple


class CloudConfig(NamedTuple):
    """Configuration for diagnostic cloud fraction and cloud-radiation coupling.

    Fields
    ------
    scheme : str
        Cloud fraction scheme:
        - ``"sundqvist"``: RH-based (Sundqvist 1988). Simple, no condensate needed.
        - ``"xu_randall"``: RH + condensate-based (Xu & Randall 1996).
          Requires explicit q_cloud/q_ice from microphysics.
        - ``"none"``: No clouds (clear-sky radiation).
    rh_crit : float
        Critical relative humidity for cloud onset (default 0.7).
        Used by Sundqvist scheme and as lower bound in Xu-Randall.
    alpha_xr : float
        Condensate scaling in Xu-Randall formula (default 100.0).
    p_xr : float
        RH exponent in Xu-Randall formula (default 0.25).
    r_eff_liq : float
        Effective radius for liquid cloud droplets [m] (default 10e-6 = 10 um).
    r_eff_ice : float
        Effective radius for ice cloud particles [m] (default 30e-6 = 30 um).
    q_c_diagnostic : float
        Typical in-cloud liquid water content [kg/kg] used when explicit
        cloud condensate is not available (default 0.2e-3 = 0.2 g/kg).
    T_freeze : float
        Temperature [K] at which condensate begins transitioning to ice
        (default 273.15).
    T_ice_only : float
        Temperature [K] below which all condensate is ice (default 233.15).
    """
    scheme: str = "none"
    rh_crit: float = 0.7
    alpha_xr: float = 100.0
    p_xr: float = 0.25
    r_eff_liq: float = 10.0e-6
    r_eff_ice: float = 30.0e-6
    q_c_diagnostic: float = 0.2e-3
    T_freeze: float = 273.15  # = constants.T_freeze
    T_ice_only: float = 233.15
