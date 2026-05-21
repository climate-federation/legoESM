"""Configuration for diagnostic cloud fraction schemes.

Provides CloudConfig for controlling cloud fraction diagnosis and
cloud optical property computation for radiation coupling.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants


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
    rh_crit_bl : float
        Critical relative humidity in the boundary layer (default 0.7 = same
        as rh_crit, i.e. no BL adjustment). Recommended ~0.55 for AMIP runs
        with Sundqvist to allow marine BL cloud at lower RH.  Only active
        when ``sigma_bl < 1.0``.
    sigma_bl : float
        Sigma level (p/p_s) above which rh_crit_bl replaces rh_crit
        (default 1.0 = disabled). Use 0.85 to cover the lowest ~1.5 km.
    """
    scheme: str = "none"
    rh_crit: float = 0.7
    alpha_xr: float = 100.0
    p_xr: float = 0.25
    r_eff_liq: float = 10.0e-6
    r_eff_ice: float = 30.0e-6
    q_c_diagnostic: float = 0.2e-3
    T_freeze: float = constants.T_freeze
    T_ice_only: float = 233.15
    rh_crit_bl: float = 0.7
    sigma_bl: float = 1.0
