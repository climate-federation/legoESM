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
        - ``"resolved"``: cloud-resolving (CRM) cloud fraction. A grid
          cell is fully cloudy where it holds condensate — SAM's
          convention at CRM resolution. SAM uses a HARD binary
          ``cf = (qn > 0)``; this is a smooth surrogate
          ``cf = q_cond/(q_cond + q_cloud_resolved_ref)`` → 1 for
          ``q_cond ≫ ref``.  ``ref`` is a tunable cloud-PRESENCE scale,
          not a SAM constant.  ``cf`` is smooth in condensate, but (a)
          the gradient ``∂cf/∂q_cond → 1/ref`` is large as ``q_cond→0``
          (a caveat for differentiable training through radiation, not
          for forward RCE), and (b) full radiation-operator AD-smoothness
          also depends on the RRTMGP cloud-overlap/McICA treatment.
          Requires explicit q_cloud/q_ice.
        - ``"none"``: No clouds (clear-sky radiation).
    rh_crit : float
        Critical relative humidity for cloud onset (default 0.7).
        Used by Sundqvist scheme and as lower bound in Xu-Randall.
    alpha_xr : float
        Condensate scaling in Xu-Randall formula (default 100.0).
    p_xr : float
        RH exponent in Xu-Randall formula (default 0.25).
    gamma_xr : float
        Saturation-deficit exponent in the Xu-Randall denominator
        ``((1−RH)·q_sat)^γ`` (default 0.49, the Xu & Randall 1996
        best-fit value; the earlier code omitted it ⇒ γ=1).
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
    q_cloud_resolved_ref : float
        Condensate scale [kg/kg] for the ``"resolved"`` (CRM) cloud
        fraction ``cf = q_cond/(q_cond + ref)`` (default 1e-6 = 1 mg/kg,
        so any resolved cloud with ``q_cond ≳ 0.1 g/kg`` gives cf ≈ 1).
    """
    scheme: str = "none"
    rh_crit: float = 0.7
    alpha_xr: float = 100.0
    p_xr: float = 0.25
    gamma_xr: float = 0.49
    r_eff_liq: float = 10.0e-6
    r_eff_ice: float = 30.0e-6
    Nc_default: float = 1.0e8        # fallback cloud-droplet number [1/m³] for the
                                     # gamma-PSD liquid effective radius when the
                                     # passed n_cloud is 0/garbage — e.g. SAM
                                     # specified-Nc Morrison (dopredictNc=.false.,
                                     # MorrisonConfig.predict_Nc=False) where the
                                     # prognostic Nc slot stays 0. Matches Morrison
                                     # Nc_0=1e8 so RRTMGP r_eff_liq is SAM-faithful.
    q_c_diagnostic: float = 0.2e-3
    T_freeze: float = constants.T_freeze
    T_ice_only: float = 233.15
    q_cloud_resolved_ref: float = 1.0e-6
    # M2005 cloud-ice bulk density [kg/m³] (RHOI) for the PSD ice effective
    # radius EFFI=1.5/LAMI, LAMI=(ρ_ci·π·N_i/q_i)^(1/3) (RAD-1-ice). Only
    # used when ``compute_cloud_properties`` is given explicit ``n_ice``.
    rho_cloud_ice: float = 500.0
