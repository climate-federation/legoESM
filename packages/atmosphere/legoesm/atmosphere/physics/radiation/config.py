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
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.simple_lw import (
    SimpleLWConfig,
)

if TYPE_CHECKING:
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.radiation.solar import OrbitalParameters


# Machine-readable tunable/fixed split for the radiation scheme configs.
# Reviewed 2026-06-13 (placeholders replaced with physically-correct
# classifications):
#   * gray tau_equator/tau_pole are PRIMARY trained knobs (the legacy-8 set
#     in training/trainable_params.py DEFAULT_TRAINABLE) -> tier 1 with their
#     flat build_segment_fn aliases preserved as legacy_name;
#   * surface albedo (the AIMIP-trained blended-surface knob that reaches the
#     heating through both solvers) -> tier 1;
#   * other closure knobs (LW/SW optical-depth shape, ozone profile,
#     emissivity) -> tier 2;
#   * RRTMGP gas concentrations and bulk aerosol optics are read into the
#     optics-cache key (any flip rebuilds the solver instance, see
#     RRTMGP._instance_cache_key) and are NOT meant to be trained -> tier 3.
__param_spec__ = {
    "GrayRadiationConfig": {
        "scheme_key": "atm.rad.GrayRadiationConfig",
        "excluded": {
            "sfc_emissivity": "physics: surface boundary emissivity is a domain boundary condition, not a sigmoid-tunable closure (fix via config)",
        },
        "params": {
            "tau_equator": {"units": "1", "bounds": (2.0, 15.0), "tunable_tier": 1, "transform": "sigmoid", "category": "optical_depth", "reference": "Frierson et al. (2006)", "shape": None, "legacy_name": "tau_equator"},
            "tau_pole": {"units": "1", "bounds": (0.5, 5.0), "tunable_tier": 1, "transform": "sigmoid", "category": "optical_depth", "reference": "Frierson et al. (2006)", "shape": None, "legacy_name": "tau_pole"},
            "linear_frac": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "longwave", "reference": "O'Gorman & Schneider (2008)", "shape": None},
            "tau_moist_coeff": {"units": "m2/kg", "bounds": (0.0, 0.05), "tunable_tier": 2, "transform": "sigmoid", "category": "longwave", "reference": "Frierson et al. (2006)", "shape": None},
            "lw_diff_factor": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "longwave", "reference": "Fu & Liou (1992) diffusivity factor", "shape": None},
            "sw_tau_0": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "shortwave", "reference": "Frierson et al. (2006)", "shape": None},
            "sw_exponent": {"units": "1", "bounds": (0.5, 6.0), "tunable_tier": 2, "transform": "sigmoid", "category": "shortwave", "reference": "gray radiation scheme default", "shape": None},
            "sfc_albedo": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "albedo", "reference": "gray radiation scheme default", "shape": None},
            "obliquity": {"units": "degree", "bounds": (0.0, 90.0), "tunable_tier": 2, "transform": "sigmoid", "category": "solar", "reference": "gray radiation scheme default", "shape": None},
        },
    },
    "OzoneProfileConfig": {
        "scheme_key": "atm.rad.OzoneProfileConfig",
        "excluded": {
        },
        "params": {
            "p_peak_hPa": {"units": "hPa", "bounds": (1.0, 100.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ozone_profile", "reference": "US Standard Atmosphere 1976 / analytical ozone profile default", "shape": None},
            "o3_max_vmr": {"units": "mol/mol", "bounds": (1.0e-06, 2.0e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "ozone_profile", "reference": "US Standard Atmosphere 1976 / analytical ozone profile default", "shape": None},
            "sigma_logp": {"units": "1", "bounds": (0.3, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ozone_profile", "reference": "analytical ozone profile default", "shape": None},
        },
    },
    "RRTMGPConfig": {
        "scheme_key": "atm.rad.RRTMGPConfig",
        "excluded": {
            # RRTMGP optics/solver instances are cached keyed on these Python
            # config floats (rrtmgp.py _optics_cache_key / _instance_cache_key).
            # A traced trainable leaf here would break the Python hash or rebuild
            # the solver per parameter value -- NOT trainable until RRTMGP
            # consumes them as per-call traced inputs. Fixed (tier 0).
            "co2_ppmv": "RRTMGP optics-cache key (Python-hashed); not trainable until consumed as a traced input",
            "ch4_ppbv": "RRTMGP optics-cache key (Python-hashed); not trainable until consumed as a traced input",
            "n2o_ppbv": "RRTMGP optics-cache key (Python-hashed); not trainable until consumed as a traced input",
            "aerosol_ssa": "RRTMGP instance-cache key (Python-hashed); not trainable until consumed as a traced input",
            "aerosol_g": "RRTMGP instance-cache key (Python-hashed); not trainable until consumed as a traced input",
            # sfc_albedo/sfc_emissivity are also folded into _instance_cache_key
            # via _hashable() (which does float(x)); a traced leaf would fail the
            # hash. The TRAINABLE surface-albedo path is the coupler's legacy-8
            # albedo_ice/albedo_ocean -> albedo_col (a per-call traced input), not
            # these config fields. Fixed (tier 0) until RRTMGP drops them from the
            # Python cache key for the trainable path.
            "sfc_albedo": "RRTMGP instance-cache key (_hashable/float); train via coupler albedo_col, not this config field",
            "sfc_emissivity": "RRTMGP instance-cache key (_hashable/float); not trainable until removed from the Python cache key",
        },
        "params": {
        },
    },
}


class GrayRadiationConfig(NamedTuple):
    """Configuration for two-stream gray radiation.

    LW parameters follow Frierson et al. (2006) with moisture feedback.
    SW uses Beer-Lambert absorption (no scattering).

    Fields
    ------
    tau_equator : float
        Equatorial LW optical depth (default 7.2; Isca B_FRIERSON default 6.0).
    tau_pole : float
        Polar LW optical depth (default 1.8; Isca B_FRIERSON default 1.5).
    linear_frac : float
        Fraction f_l of linear sigma weighting vs sigma^4 (default 0.2;
        Isca B_FRIERSON ``linear_tau`` default 0.1).
    tau_moist_coeff : float
        Moisture optical depth coefficient [m^2/kg]: dtau_k = coeff * q_v * dp_k / g
        (default 0.0115). NOTE: interactive-vapor LW is a Byrne & O'Gorman (2013)-
        style add-on; Frierson (2006) / Isca B_FRIERSON have NO moisture LW term
        (their tau is a prescribed dry function of lat & sigma). See the module
        docstring's Faithfulness section. Set ``q_v=None`` to drop this term
        (recovers the prescribed-dry LW optical depth, not full Frierson).
    lw_diff_factor : float
        Diffusivity factor D applied as exp(-D*dtau) (default 1.66, ~5/3).
        DEPARTURE: Isca/FHZ06 apply NO separate D (their prescribed tau is already
        diffusive); set D=1.0 to reproduce the oracle for a given tau.
    sfc_emissivity : float
        Surface emissivity for LW (default 1.0 = Isca's black ``b_surf``).
    sw_tau_0 : float
        SW optical depth scale (default 0.22). Set to 0.0 for the strict
        surface-absorbing SW limit; Isca B_FRIERSON's default ``atm_abs=0.0``
        is a fully transparent SW atmosphere. The SW optical depth profile is:
        tau_sw(sigma) = sw_tau_0 * sigma^sw_exponent.
    sw_exponent : float
        Exponent for the SW optical-depth profile (default 2.0; Isca
        ``solar_exponent`` default 4.0). Controls how SW absorption is
        distributed vertically. Only the downward SW beam is absorbed;
        reflected upward SW escapes directly to TOA (Frierson/Isca convention).
    S_0 : float
        Total solar irradiance [W/m^2] (default constants.S_0 = 1361.0).
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
    tau_moist_coeff: float = 0.0115  # [m²/kg] Byrne&O'Gorman(2013)-style moisture LW; NOT in Frierson dry LW
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
        Total solar irradiance [W/m^2] (default constants.S_0 = 1361.0).
    aerosol_ssa : float
        Bulk aerosol single-scattering albedo used when aerosol optical depth
        is externally prescribed (default 0.93).
    aerosol_g : float
        Bulk aerosol asymmetry factor used when aerosol optical depth is
        externally prescribed (default 0.70).
    aerosol_ssa_bands : tuple[float, ...] | None
        Optional PER-SHORTWAVE-BAND aerosol single-scattering albedo, length =
        the SW gas-optics band count (14 for the shipped RRTMGP-SW table). When
        set, the two-stream solver uses the value of the band each g-point
        belongs to instead of the scalar ``aerosol_ssa`` — real aerosols scatter
        very differently in the UV/visible vs the near-IR. ``None`` (default) =>
        grey aerosol at the scalar ``aerosol_ssa`` (byte-identical).
    aerosol_g_bands : tuple[float, ...] | None
        Optional PER-SHORTWAVE-BAND aerosol asymmetry factor (same length /
        semantics as ``aerosol_ssa_bands``). ``None`` (default) => grey aerosol
        at the scalar ``aerosol_g``.
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
    # G-point accumulation strategy in the two-stream RTE solve.
    #   0  -> memory-frugal checkpointed ``lax.scan`` over g-points (REQUIRED
    #         for reverse-mode AD / training at high resolution).
    #   >0 -> process g-points in parallel blocks of this size via ``vmap``
    #         (the g-point axis is embarrassingly parallel; the sequential
    #         scan launches one tiny kernel per g-point and starves the GPU,
    #         ~26x slower in a microbench).  It holds this many g-points'
    #         activations for the backward pass.  A block of ~16-32 recovers
    #         most of the parallelism while bounding peak memory at high
    #         resolution.
    #
    # DEFAULT 16 (was 0): the scan path (0) with gpoint_checkpoint=True emits a
    # distinct prevent_cse body per g-point (~Ng-fold code) => multi-HOUR GPU
    # compile for reverse-mode AD training (rrtmgp+rollout adjoint took ~9 h,
    # never reaching epoch-0). The 16-wide vmap block compiles ONE reused body
    # (minutes) and is ~26x faster at runtime, with peak memory bounded to 16
    # g-points. This is the right default everywhere RRTMGP is used; set 0 only
    # to reproduce the exact legacy g-point accumulation order.
    gpoint_batch_size: int = 16
    # Wrap the per-g-point scan step in jax.checkpoint(prevent_cse=True) for
    # reverse-mode AD memory (recompute one g-point per backward step).  True =
    # byte-for-byte legacy (required for high-res rrtmgp training).  Set False
    # for FORWARD/inference: prevent_cse=True disables CSE and forces XLA to
    # emit a distinct compiled body per g-point, inflating the executable code
    # ~Ng-fold — that overflows the XLA-CPU LLVM-JIT contiguous executable
    # region (rrtmgp CPU "Failed to materialize symbols") and bloats GPU/TPU
    # compile.  A plain scan (False) compiles ONE reused body; answer-identical
    # (no AD-memory benefit, which forward runs do not need).  Only applies to
    # the scan path (gpoint_batch_size<=0).
    gpoint_checkpoint: bool = True
    # Run the optics tables + RTE solve in float32 even when JAX x64 is on.
    # The dycore needs fp64, but radiation (a flux calculation) does not —
    # fp32 is ~2x faster on fp64-limited GPUs (e.g. RTX 8000, fp64 ≈ 1/32 of
    # fp32) with negligible heating change (benchmark: heating identical to
    # <0.01 K/day vs fp64).  Default off; the MPAS driver enables it for the
    # long-run rrtmgp path.
    compute_fp32: bool = False
    # Column-chunking for the rrtmgp XLA compile wall at higher horizontal
    # resolution.  0 (default) = disabled, byte-identical single-shot solve.
    #   >0 -> jax.lax.map ``solve_columns`` over fixed-size blocks of this many
    #         columns.  Radiation columns are INDEPENDENT, so the result is
    #         numerically EXACT; the per-block body compiles ONCE at this size,
    #         capping the highly super-linear rrtmgp JIT cost independent of the
    #         total column count (C24/C48 at L20 compile instead of stalling).
    #         Must divide ncol.  A pure compile-time NUMERICS knob — NOT a
    #         tunable/trainable parameter (levels stay coupled, never chunked).
    column_chunk_size: int = 0
    # Optional PER-SHORTWAVE-BAND aerosol optics (length = SW band count, 14 for
    # the shipped table).  None => grey aerosol at the scalar aerosol_ssa/g
    # (byte-identical).  Tuples so the value stays a hashable solver-cache key.
    # APPENDED at the end of the field list (not next to aerosol_ssa/g) so no
    # existing positional RRTMGPConfig(...) argument binding shifts.
    aerosol_ssa_bands: tuple[float, ...] | None = None
    aerosol_g_bands: tuple[float, ...] | None = None


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
        Active radiation scheme: "gray", "rrtmgp", "mc3d" or "simple_lw".
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
    # Stevens (2005) DYCOMS-II simple longwave -- the cloud-top radiative
    # cooling the marine-stratocumulus decks (DYCOMS_RF01, ASTEX209) are
    # driven by. Longwave only; see radiation/simple_lw.py.
    simple_lw: SimpleLWConfig = SimpleLWConfig()
    rrtmgp: RRTMGPConfig = RRTMGPConfig()
    # "mc3d": 3D Monte-Carlo ray-traced shortwave (plane LES/CRM only) + gray
    # longwave. See docs/specs/mc3d_raytracer.md. mc3d holds MC numerics.
    mc3d: MC3DRadiationConfig = MC3DRadiationConfig()
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
    # Realistic Earth orbit (Berger 1978) for AMIP-II / CMIP insolation.
    # When set, ``_compute_insolation`` uses the orbital declination and
    # scales the TOA flux by the Earth-Sun distance factor (a/r)^2 (the
    # eccentricity-driven perihelion/aphelion asymmetry).  ``None`` (default)
    # ⇒ circular orbit, so idealized/aquaplanet experiments are unchanged.
    orbit: "OrbitalParameters | None" = None
    # Route a moist higher-order turbulence closure's sub-grid PDF cloud fraction
    # (CLUBB) into the cloud optics instead of the RH-diagnosed grid-scale one.
    # A moist closure is physically LESS overcast over a saturated marine
    # boundary layer, so the ``cf * q_c_diagnostic`` condensate floor — which
    # sets the marine-Sc liquid water path and hence the planetary albedo — drops
    # toward the observed value (the marine-Sc over-bright bias lever).  Requires
    # a cf-producing closure (turbulence.scheme='clubb', diagnostic
    # CLUBBConfig.prognostic=False); combined.py raises if set without one, and
    # make_radiation_physics raises on non-hydrostatic dycores (READ side wired
    # for hydrostatic only).  ``False`` (default) keeps the RH grid-scale cloud
    # fraction (byte-identical).
    use_clubb_cloud_fraction: bool = False
    # Clear-sky TOA diagnostic (#843, lean-lane port): run a SECOND clouds-off
    # radiation pass per radiation step and attach ``sw_up_toa_clr`` /
    # ``lw_up_toa_clr`` to the tendency bundle for the CMOR rsutcs/rlutcs feed
    # (SW_CRE = rsut - rsutcs, LW_CRE = rlutcs - rlut).  Aerosols/ozone/GHG are
    # KEPT, only the cloud optics are dropped (CMIP "assuming clear sky").
    # Consumed by the LEAN hydrostatic/MPAS radiation factory
    # (``_make_hydrostatic_radiation``); the compiled cube/lat-lon lane keeps
    # its own gate (``PhysicsPipeline._clear_sky_diag``).  Drivers set it from
    # ``OutputConfig.clear_sky_diag`` (the ``--clear-sky-diag`` CLI flag);
    # ``make_radiation_physics`` raises on model types without the second-pass
    # wiring rather than silently ignoring it.  Static Python bool (never
    # traced); ``False`` (default) adds no ops — byte-identical.
    clear_sky_diag: bool = False
