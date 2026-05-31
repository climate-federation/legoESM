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

from typing import TYPE_CHECKING, NamedTuple

from legoesm import constants

if TYPE_CHECKING:
    from legoesm.atmosphere.physics.clouds.config import CloudConfig


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
        Moisture optical depth coefficient [m^2/kg]: dtau_k = coeff * q_v * dp_k / g
        (default 0.0115, Frierson 2006).
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
    S_0: float = constants.S_0
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
    use_scan : bool | None
        Column recurrence implementation inside the two-stream solver.

        * ``None`` (default, issue #273 tuning): defer the choice to
          ``rte_utils.recurrent_op_with_halos`` which auto-picks
          ``True`` on GPU/TPU (``lax.scan`` lowers to one fused
          kernel + drastically reduces the XLA graph size — directly
          shrinks the 2600s cold / 600s warm AMIP JIT cost called
          out in issue #273) and ``False`` on CPU/Metal (where the
          unrolled path benchmarked faster historically).
        * ``True``: force ``jax.lax.scan`` — smaller graph,
          preferred for large nlev and reverse-mode AD.
        * ``False``: force Python for-loop unroll — larger graph,
          sometimes faster on CPU for typical atmospheric nlev.
    include_clouds : bool
        If True, include cloud optics (default False).
    use_optimal_angle : bool
        If True, replace the fixed Fu-Liou ``1.66`` longwave diffusivity
        secant with the per-band, per-column optimal angle computed from
        the ``optimal_angle_fit`` polynomial in the gas-optics file (see
        upstream ``compute_optimal_angles``).  Default False to preserve
        bit-reproducibility with the historical legoESM output; enable
        for upper-troposphere/stratosphere fidelity matching upstream
        rte-rrtmgp.
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
    # Optional DIRECT-beam surface albedo (RAD-3). None ⇒ use sfc_albedo for
    # both beams (legacy). SAM splits direct (Briegleb zenith-dependent ocean
    # albedo) from diffuse (sfc_albedo=0.07 RCEMIP); set this to the direct
    # value so the two-stream solver reflects the direct beam faithfully.
    sfc_albedo_direct: float | None = None
    S_0: float = constants.S_0
    aerosol_ssa: float = 0.93
    aerosol_g: float = 0.70
    use_scan: bool | None = None
    include_clouds: bool = False
    use_optimal_angle: bool = False


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
        - ``"mls"``: the SAM mid-latitude-summer (MLS) standard O3 profile,
          bundled from gSAM's ``rrtmg_lw.nc`` and interpolated (log-log) to
          the model levels.  Use this for SAM-faithful RCEMIP runs — it is
          the same ozone the gSAM oracle uses (vs the Gaussian "standard",
          which over-estimates lower-stratospheric O3 ~3x).  See
          :mod:`legoesm.atmosphere.physics.radiation.ozone_mls`.
        - ``"none"``: zero ozone (disables ozone absorption entirely).
        - ``"ml"``: machine-learning ridge regression predictor of Ma et al.
          (UKESM-trained, per-gridpoint T -> O3 column).  Requires
          ``ml_weights_path`` to point at a directory of NetCDF weights;
          see :mod:`legoesm.atmosphere.physics.radiation.ozone_ml`.
    p_peak_hPa : float
        Peak pressure [hPa] for the analytical profile (default 30.0).
    o3_max_vmr : float
        Maximum ozone VMR at the profile peak (default 8.0e-6 = 8 ppmv).
    sigma_logp : float
        Width of the Gaussian in log-pressure space (default 1.5).
    lat_dependence : bool
        If True, scale analytical ozone by ``1 + 0.5 * sin²(lat)``
        (default True).  Only used when ``source="analytical"``.
    ml_weights_path : str or None
        Directory of NetCDF coefficient files for ``source="ml"``.  Must
        contain ``coefs*.nc``, ``Scaler_x*.nc``, ``Scaler_y*.nc`` and a
        pressure-coordinate sidecar (``plev.npy`` / ``plev.nc``).
    ml_mmr_to_vmr : bool
        If True, multiply ridge output by ``M_dry / M_o3`` to convert mass
        mixing ratio to volume mixing ratio.  Set False if upstream
        weights are already in VMR.  Default True.
    """
    source: str = "standard"
    p_peak_hPa: float = 30.0
    o3_max_vmr: float = 8.0e-6
    sigma_logp: float = 1.5
    lat_dependence: bool = True
    ml_weights_path: str | None = None
    ml_mmr_to_vmr: bool = True


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
    rce_fixed_cos_zenith : float or None
        Perpetual fixed-zenith RCE insolation (SAM ``doperpetual``).  When
        set (e.g. 0.620 = 51.7° SAM RCE, or 0.7425 = 42.05° RCEMIP), the
        TOA insolation is ``S_0·cosθ`` UNIFORMLY (no latitude/daily-mean
        dependence) and ``cosθ`` is used directly as the SW optical-path
        cosine — matching SAM's fixed-sun RCE rather than the daily-mean
        daytime-effective cos(SZA).  Pair with a reduced ``S_0`` (SAM RCE
        uses 685 W/m² ⇒ 685·0.620 ≈ 425 W/m²).  ``None`` (default) keeps
        the latitude-based daily-mean / perpetual-equinox path.
    """
    scheme: str = "gray"
    gray: GrayRadiationConfig = GrayRadiationConfig()
    rrtmgp: RRTMGPConfig = RRTMGPConfig()
    update_interval_steps: int = 1
    diurnal_cycle: bool = False
    ozone: OzoneProfileConfig = OzoneProfileConfig()
    cloud_scheme: str = "none"
    rce_fixed_cos_zenith: float | None = None
    # Optional full ``CloudConfig`` (rh_crit, xu_p, alpha_xr, q_c_diagnostic, ...).
    # When ``None`` the integration bridge builds a default
    # ``CloudConfig(scheme=cloud_scheme)`` — backward-compatible.
    # When supplied, its scalar fields flow through the AD graph so
    # cloud-fraction knobs become trainable end-to-end via the
    # cloud-radiation coupling (AIMIP).  Forward-reference avoids a
    # circular import (clouds.config is a downstream consumer that
    # already imports from this module via the integration bridge).
    cloud_config: "CloudConfig | None" = None
