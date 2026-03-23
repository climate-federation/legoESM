"""Configuration for atmospheric radiation schemes.

Provides configuration NamedTuples for:
1. Gray two-stream radiation (Frierson et al. 2006 / O'Gorman & Schneider 2008)
2. RRTMGP correlated-k radiation (via jax-rrtmgp)
3. Top-level RadiationConfig that selects the active scheme.

References
----------
- Frierson, D. M. W., Held, I. M., & Zurita-Gotor, P. (2006).
  A Gray-Radiation Aquaplanet Moist GCM. Part I: Static Stability
  and Eddy Scale. J. Atmos. Sci., 63, 2548-2566.
- O'Gorman, P. A. & Schneider, T. (2008). The Hydrological Cycle
  over a Wide Range of Climates Simulated with an Idealized GCM.
  J. Climate, 21, 3815-3832.
"""

from __future__ import annotations

from typing import NamedTuple


class GrayRadiationConfig(NamedTuple):
    """Configuration for two-stream gray radiation.

    LW parameters follow Frierson et al. (2006) with moisture feedback.
    SW uses Beer-Lambert absorption (no scattering).

    Fields
    ------
    tau_equator : float
        Equatorial LW optical depth (default 7.2).
    tau_pole : float
        Polar LW optical depth (default 1.8).
    linear_frac : float
        Fraction f_l of linear sigma weighting vs sigma^4 (default 0.2).
    tau_moist_coeff : float
        Moisture optical depth coefficient: tau += coeff * q_v (default 1150.0).
    lw_diff_factor : float
        Diffusivity factor D for hemispheric-mean (default 1.66, ~5/3).
    sfc_emissivity : float
        Surface emissivity for LW (default 1.0).
    sw_tau_0 : float
        SW optical depth scale (default 0.22). Set to 0.0 for the
        strict surface-absorbing SW limit often used in Frierson-style
        gray setups. The SW optical depth profile is:
        tau_sw(sigma) = sw_tau_0 * sigma^sw_exponent.
    sw_exponent : float
        Exponent for the SW optical-depth profile (default 2.0).
        Controls how SW absorption is distributed vertically.
        Only the downward SW beam is absorbed; reflected upward
        SW escapes directly to TOA (Frierson/Isca convention).
    S_0 : float
        Total solar irradiance [W/m^2] (default 1360.0).
    sfc_albedo : float
        Surface albedo for SW (default 0.31).
    perpetual_equinox : bool
        If True, use annual+daily mean insolation at equinox (default True).
    obliquity : float
        Obliquity of the ecliptic [degrees] (default 23.45).
    """
    tau_equator: float = 7.2
    tau_pole: float = 1.8
    linear_frac: float = 0.2
    tau_moist_coeff: float = 0.0115  # [m²/kg] moisture LW optical depth (Frierson 2006)
    lw_diff_factor: float = 1.66
    sfc_emissivity: float = 1.0
    sw_tau_0: float = 0.22
    sw_exponent: float = 2.0
    S_0: float = 1360.0
    sfc_albedo: float = 0.31
    perpetual_equinox: bool = True
    obliquity: float = 23.45


class RRTMGPConfig(NamedTuple):
    """Configuration for RRTMGP correlated-k radiation.

    Gas files default to empty strings, which tells the backend to use
    the jax-rrtmgp built-in defaults.

    Fields
    ------
    lw_gas_file : str
        Path to LW gas optics file (empty = jax-rrtmgp default).
    sw_gas_file : str
        Path to SW gas optics file (empty = jax-rrtmgp default).
    lw_cloud_file : str
        Path to LW cloud optics file (empty = jax-rrtmgp default).
    sw_cloud_file : str
        Path to SW cloud optics file (empty = jax-rrtmgp default).
    co2_ppmv : float
        CO2 concentration [ppmv] (default 415.0).
    ch4_ppbv : float
        CH4 concentration [ppbv] (default 1900.0).
    n2o_ppbv : float
        N2O concentration [ppbv] (default 332.0).
    sfc_emissivity : float
        Surface emissivity for LW (default 0.98).
    sfc_albedo : float
        Surface albedo for SW (default 0.06).
    S_0 : float
        Total solar irradiance [W/m^2] (default 1360.86).
    aerosol_ssa : float
        Bulk aerosol single-scattering albedo used when aerosol optical depth
        is externally prescribed (default 0.93).
    aerosol_g : float
        Bulk aerosol asymmetry factor used when aerosol optical depth is
        externally prescribed (default 0.70).
    use_scan : bool
        If True, use lax.scan (differentiable); else fori_loop (default False).
    include_clouds : bool
        If True, include cloud optics (default False).
    """
    lw_gas_file: str = ""
    sw_gas_file: str = ""
    lw_cloud_file: str = ""
    sw_cloud_file: str = ""
    co2_ppmv: float = 415.0
    ch4_ppbv: float = 1900.0
    n2o_ppbv: float = 332.0
    sfc_emissivity: float = 0.98
    sfc_albedo: float = 0.06
    S_0: float = 1360.86
    aerosol_ssa: float = 0.93
    aerosol_g: float = 0.70
    use_scan: bool = False
    include_clouds: bool = False


class OzoneProfileConfig(NamedTuple):
    """Configuration for the ozone profile used in RRTMGP radiation.

    Controls how ozone volume mixing ratio is computed per column per level.
    Only affects the RRTMGP backend; gray radiation does not use ozone.

    Fields
    ------
    source : str
        Ozone profile source:

        - ``"standard"``: built-in US Standard Atmosphere 1976 profile.
          Gaussian in log-pressure, peak at 10 hPa, 8 ppmv, no latitude
          dependence.  This is the original default.
        - ``"analytical"``: latitude-dependent Gaussian profile with
          configurable parameters.  Peak scaled by
          ``1 + 0.5 * sin²(lat)`` when ``lat_dependence`` is True.
        - ``"none"``: zero ozone (disables ozone absorption entirely).
    p_peak_hPa : float
        Peak pressure [hPa] for the analytical profile (default 30.0).
    o3_max_vmr : float
        Maximum ozone VMR at the profile peak (default 8.0e-6 = 8 ppmv).
    sigma_logp : float
        Width of the Gaussian in log-pressure space (default 1.5).
    lat_dependence : bool
        If True, scale analytical ozone by ``1 + 0.5 * sin²(lat)``
        (default True).  Only used when ``source="analytical"``.
    """
    source: str = "standard"
    p_peak_hPa: float = 30.0
    o3_max_vmr: float = 8.0e-6
    sigma_logp: float = 1.5
    lat_dependence: bool = True


class RadiationConfig(NamedTuple):
    """Top-level radiation configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active radiation scheme: "gray" or "rrtmgp".
    gray : GrayRadiationConfig
        Configuration for gray radiation.
    rrtmgp : RRTMGPConfig
        Configuration for RRTMGP radiation.
    update_interval_steps : int
        Recompute radiation every N time steps (1 = every step).
    diurnal_cycle : bool
        If True, compute instantaneous insolation from the solar zenith
        angle (per column, per time step) instead of daily-mean insolation.
        Requires ``set_time(day_of_year, seconds_of_day)`` to be called on
        the physics function before each radiation step.  Default False
        for backward compatibility with idealized experiments.
    ozone : OzoneProfileConfig
        Ozone profile configuration for RRTMGP.  Ignored by gray radiation.
    cloud_scheme : str
        Cloud fraction scheme for cloud-radiation coupling:
        ``"none"`` (clear-sky, default), ``"sundqvist"``, or ``"xu_randall"``.
        Only affects RRTMGP; gray radiation ignores clouds.
    """
    scheme: str = "gray"
    gray: GrayRadiationConfig = GrayRadiationConfig()
    rrtmgp: RRTMGPConfig = RRTMGPConfig()
    update_interval_steps: int = 1
    diurnal_cycle: bool = False
    ozone: OzoneProfileConfig = OzoneProfileConfig()
    cloud_scheme: str = "none"
