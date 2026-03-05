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
        SW optical depth scale (default 0.22).
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
    tau_moist_coeff: float = 1150.0
    lw_diff_factor: float = 1.66
    sfc_emissivity: float = 1.0
    sw_tau_0: float = 0.22
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
    use_scan: bool = False
    include_clouds: bool = False


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
    """
    scheme: str = "gray"
    gray: GrayRadiationConfig = GrayRadiationConfig()
    rrtmgp: RRTMGPConfig = RRTMGPConfig()
    update_interval_steps: int = 1
