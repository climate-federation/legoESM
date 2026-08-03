"""Model integration bridge for radiation.

Provides `make_radiation_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

Supported model types:
- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
- "mpas"         : MPASPrimitiveEquationModel (Voronoi mesh, sigma coords)
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticTendencies,
    MPASNonHydrostaticState,
    MPASNonHydrostaticTendencies,
    NonHydrostaticState,
    NonHydrostaticTendencies,
    PlaneNonHydrostaticState,
    PlaneNonHydrostaticTendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    HeightCoordinate,
    TerrainMetric,
)
from legoesm import constants

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
from legoesm.atmosphere.physics.radiation.config import (
    OzoneProfileConfig,
    RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.output import RadiationOutput
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.grids.gaussian import sh_analysis_3d
from legoesm.core.precision import get_policy
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle,
    daily_mean_insolation,
    daylight_fraction,
    earth_sun_distance_factor,
    perpetual_equinox_insolation,
)
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


# Briegleb (1992) clear-sky ocean-albedo formula constants + ice fallback (fixed).
_OCEAN_ALB_A = 0.026
_OCEAN_ALB_B = 0.065
_OCEAN_ALB_MU_EXP = 1.7
_OCEAN_ALB_POLY = 0.15
_OCEAN_ALB_ROOT1 = 0.1
_ICE_ALBEDO_FALLBACK = 0.75

def _apply_T_sfc_override(T_sfc, override):
    """Apply a per-column ``T_sfc`` override over an arbitrary-shape T_sfc.

    The driver-supplied ``override`` is always a flat ``(ncol,)`` array
    or ``None``.  ``T_sfc`` may be ``(face, x, y)`` (cubed sphere),
    ``(ny, nx)`` (plane), ``(nCells,)`` (MPAS), ``(n_lat, n_lon)``
    (spectral PE Gaussian grid), or any other shape whose flattened
    size matches ``ncol``.  Sentinel entries (the finite
    ``NO_SFC_T_OVERRIDE`` value, or a legacy ``NaN``) keep the
    per-column fallback ``T_sfc.reshape(-1)``; physical entries (above
    ``SFC_T_OVERRIDE_VALID_MIN``) win — the SAME validity predicate the
    turbulence resolver uses, so both consumers agree (#911).  Output is
    reshaped back to ``T_sfc.shape`` so downstream code sees the same
    layout it always saw — no broadcasting surprises.

    A wrong-sized ``override`` (scalar, shape-``(1,)``, etc.) raises
    ``ValueError`` rather than silently broadcasting across every
    surface column (Phase B v2 codex iter-4 medium finding).  Callers
    must supply exactly one value per surface column.
    """
    if override is None:
        return T_sfc
    orig_shape = T_sfc.shape
    flat = T_sfc.reshape(-1)
    ov_arr = jnp.asarray(override)
    if ov_arr.shape != flat.shape:
        raise ValueError(
            f"T_sfc override shape {tuple(ov_arr.shape)} does not match "
            f"the flattened surface-column shape {tuple(flat.shape)}.  "
            "The driver must supply exactly one override value per "
            "column; broadcasting from a scalar or shape-(1,) override "
            "would silently corrupt every column with a single value."
        )
    from legoesm.atmosphere.physics.physics_state import (
        SFC_T_OVERRIDE_VALID_MIN,
    )
    # Physical override wins; the finite sentinel (and any legacy NaN) keeps
    # the fallback.  Matches the turbulence resolver's predicate exactly.
    out_flat = jnp.where(ov_arr > SFC_T_OVERRIDE_VALID_MIN, ov_arr, flat)
    return out_flat.reshape(orig_shape)


def _make_T_sfc_override_cell():
    """Per-factory closure cell carrying an optional ``T_sfc`` override.

    Each radiation factory creates one of these and attaches the
    ``set_T_sfc_override`` setter to its returned ``physics_fn``.  The
    SCM driver calls it before each physics evaluation when
    ``SCMForcing(prescribe="T_s")`` so the radiative surface boundary
    matches turbulence's bulk-flux boundary (Phase B v2 codex iter-2
    high finding — without this, longwave emission used
    ``T[..., -1]`` while turbulence saw the prescribed skin
    temperature, producing a silent split surface boundary).

    Returns ``(cell, set_fn)`` where ``cell`` is a length-1 list
    (mutable closure), and ``set_fn(value)`` writes ``value`` (a
    ``(ncol,)`` jax.Array or ``None`` to clear).
    """
    cell = [None]

    def set_T_sfc_override(value):
        cell[0] = value

    return cell, set_T_sfc_override


def _make_time_state():
    """Create a mutable time-state dict and its ``set_time`` mutator.

    Returns ``(_time, set_time)`` where *_time* is the mutable dict and
    *set_time* is a function that updates it in-place.
    """
    _time = {"day_of_year": 80.0, "seconds_of_day": 43200.0}  # coeff-ok: idealized time (spring equinox, noon)

    def set_time(day_of_year: float, seconds_of_day: float):
        _time["day_of_year"] = day_of_year
        _time["seconds_of_day"] = seconds_of_day

    return _time, set_time


def _get_radiation_fn(config: RadiationConfig):
    """Select the radiation backend based on config.scheme."""
    if config.scheme == "gray":
        return gray_radiation, config.gray
    elif config.scheme == "rrtmgp":
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        return rrtmgp_radiation, config.rrtmgp
    else:
        raise ValueError(f"Unknown radiation scheme: {config.scheme!r}")


def _compute_insolation(
    lat: jnp.ndarray,
    config: RadiationConfig,
    lon: jnp.ndarray | None = None,
    day_of_year: float = 80.0,  # coeff-ok: idealized default (spring equinox)
    seconds_of_day: float = 43200.0,  # coeff-ok: idealized default (local noon)
) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
    """Compute TOA insolation and (optionally) cosine zenith angle.

    Returns
    -------
    insolation : jnp.ndarray
        TOA downward SW flux per column [W/m^2].
    cos_sza : jnp.ndarray or None
        Cosine of solar zenith angle (clipped >=0) per column.
        Only returned when ``config.diurnal_cycle`` is True.
    f_day : jnp.ndarray or None
        Daylight fraction per column.  Returned for non-diurnal daily-mean
        insolation so that RRTMGP can use a daytime-effective cos(SZA)
        rather than a day+night average.
    eccf : float or jnp.ndarray
        Earth-Sun distance factor ``(a/r)^2`` (1.0 on the circular orbit).
        The returned ``insolation`` already includes it (correct for the
        gray solver and the rsdt diagnostic); it is returned SEPARATELY so
        the RRTMGP consumer can apply it as a SW-flux scale while keeping
        ``cos_sza`` purely geometric (the optical-path cosine must stay
        <= 1 and unscaled by distance).
    """
    S_0 = config.rrtmgp.S_0 if config.scheme == "rrtmgp" else config.gray.S_0
    obliquity = config.gray.obliquity
    # Realistic orbit (Berger 1978) when enabled — None ⇒ circular orbit, so
    # the idealized/aquaplanet paths below are bit-for-bit unchanged.
    orbit = getattr(config, "orbit", None)
    # Eccentricity (a/r)^2 flux factor; 1.0 on the circular-orbit path.  It
    # scales the incoming SW *flux*, NOT the optical-path cosine.
    eccf = (earth_sun_distance_factor(day_of_year, orbit)
            if orbit is not None else 1.0)

    # SAM perpetual fixed-zenith RCE (doperpetual): uniform TOA insolation
    # S_0·cosθ with cosθ used directly as the SW optical-path cosine — no
    # latitude / daily-mean / daytime-effective rescaling. (RAD-2.)
    # Perpetual RCE is a fixed-geometry idealization → orbit does not apply.
    if config.rce_fixed_cos_zenith is not None:
        cos_zen = jnp.full_like(lat, config.rce_fixed_cos_zenith)
        return S_0 * cos_zen, cos_zen, None, 1.0

    if config.diurnal_cycle and lon is not None:
        hour = seconds_of_day / 3600.0
        cos_sza = cos_zenith_angle(lat, lon, day_of_year, hour, obliquity,
                                   orbit=orbit)
        cos_sza_pos = jnp.maximum(cos_sza, 0.0)
        # insolation carries (a/r)^2 (for gray + the rsdt diagnostic); cos_sza
        # stays geometric for the RRTMGP optical path.
        return S_0 * eccf * cos_sza_pos, cos_sza_pos, None, eccf

    # No diurnal cycle — daily-mean or perpetual-equinox insolation.
    gray_config = config.gray
    if gray_config.perpetual_equinox:
        # Equinox: f_day = 0.5 everywhere (idealized — orbit not applied).
        f_day = jnp.full_like(lat, 0.5)
        return perpetual_equinox_insolation(lat, S_0), None, f_day, 1.0
    f_day = daylight_fraction(lat, day_of_year, obliquity, orbit=orbit)
    # daily_mean_insolation applies the (a/r)^2 eccentricity factor internally
    # when ``orbit`` is set; eccf is returned so the RRTMGP consumer can keep
    # cos_zenith geometric and apply the distance factor as a flux scale.
    return (daily_mean_insolation(lat, day_of_year, S_0, obliquity,
                                  orbit=orbit), None, f_day, eccf)


def sam_ocean_albedo(
    cos_zenith: jnp.ndarray | float,
    T_sfc: jnp.ndarray | float = 300.0,  # coeff-ok: idealized default surface T [K]
    sea_ice_T: float = 271.0,  # coeff-ok: sea-ice albedo threshold T [K]
) -> jnp.ndarray:
    """SAM RAD_RRTM surface albedo over ocean (Briegleb 1986 direct beam).

    Faithful to gSAM ``cam_rad_parameterizations.f90:albedo`` (the ``ocean``
    branch). For ice-free ocean (``T_sfc > 271 K``) the DIRECT-beam albedo is
    zenith-dependent::

        a_dir = 0.026/(μ^1.7 + 0.065) + 0.15·(μ−0.1)·(μ−0.5)·(μ−1.0)

    with μ = cos(zenith). SAM pairs this with a fixed DIFFUSE albedo
    ``adif = 0.07`` (the RCEMIP value, AAW 2017). legoESM's RRTMGP applies a
    single surface albedo to BOTH beams, so for the fixed-zenith DIRECT-beam
    RCE this returns the direct value — the dominant reflected-SW term. Using
    it for the (small) diffuse fraction too, rather than 0.07, is the
    documented single-albedo approximation (RAD-3); a full direct/diffuse
    split would thread ``sfc_albedo_direct`` separately through the solver.
    Sea ice / snow (``T_sfc ≤ sea_ice_T``, SAM's literal 271 K ≈
    ``constants.T_freeze_ocean``): a_dir = 0.75. Night (μ ≤ 0): 0.

    Exponent 1.7 > 0 so ``μ^1.7`` is AD-finite at μ = 0 (no safe_pow needed).
    """
    mu = jnp.clip(cos_zenith, 0.0, 1.0)
    a_ocean = (
        _OCEAN_ALB_A / (mu ** _OCEAN_ALB_MU_EXP + _OCEAN_ALB_B)
        + _OCEAN_ALB_POLY * (mu - _OCEAN_ALB_ROOT1) * (mu - 0.5) * (mu - 1.0)
    )
    a = jnp.where(jnp.asarray(T_sfc) > sea_ice_T, a_ocean, _ICE_ALBEDO_FALLBACK)
    return jnp.where(jnp.asarray(cos_zenith) > 0.0, a, 0.0)


def _compute_ozone_vmr(
    p_full: jnp.ndarray,
    lat: jnp.ndarray,
    ozone_config: OzoneProfileConfig,
    *,
    T: jnp.ndarray | None = None,
    lon: jnp.ndarray | None = None,
    ml_ozone_coefs=None,
) -> jnp.ndarray | None:
    """Compute ozone VMR for RRTMGP.

    Parameters
    ----------
    p_full : jnp.ndarray
        Pressure at full levels (ncol, nlev) [Pa].
    lat : jnp.ndarray
        Latitude (ncol,) [rad].
    ozone_config : OzoneProfileConfig
    T : jnp.ndarray, optional
        Temperature (ncol, nlev) [K].  Required for ``source="ml"``.
    lon : jnp.ndarray, optional
        Longitude (ncol,) [rad].  Required for ``source="ml"``.
    ml_ozone_coefs : MLOzoneCoefficients, optional
        Pre-loaded ridge weights.  Required for ``source="ml"``.

    Returns
    -------
    jnp.ndarray or None
        Ozone VMR (ncol, nlev), or None to use the built-in profile.
    """
    if ozone_config.source == "standard":
        return None  # rrtmgp_radiation uses its built-in _standard_o3_profile

    if ozone_config.source == "none":
        return jnp.full_like(p_full, 1.0e-10)

    if ozone_config.source == "mls":
        # SAM RCEMIP ozone: the MLS standard profile read from gSAM's
        # rrtmg_lw.nc, interpolated (log-log) to the model levels. Faithful
        # to the oracle vs the built-in skewed-Gaussian _standard_o3_profile.
        from legoesm.atmosphere.physics.radiation.ozone_mls import (
            mls_ozone_vmr,
        )
        return mls_ozone_vmr(p_full)

    if ozone_config.source == "analytical":
        p_hPa = p_full / 100.0
        p_peak = ozone_config.p_peak_hPa
        sigma = ozone_config.sigma_logp
        o3 = ozone_config.o3_max_vmr * jnp.exp(
            -0.5 * ((jnp.log(p_hPa) - jnp.log(p_peak)) / sigma) ** 2
        )
        if ozone_config.lat_dependence:
            # Ozone is ~2x higher at poles than equator in the lower strat.
            lat_factor = 1.0 + 0.5 * jnp.sin(lat) ** 2  # (ncol,)
            o3 = o3 * lat_factor[:, None]
        return jnp.clip(o3, 1.0e-10, None)

    if ozone_config.source == "ml":
        if T is None or lon is None or ml_ozone_coefs is None:
            raise ValueError(
                "OzoneProfileConfig.source='ml' requires T, lon, and "
                "ml_ozone_coefs to be passed through the radiation "
                "backend.  Did make_radiation_physics() succeed in "
                "loading ml_weights_path?"
            )
        from legoesm.atmosphere.physics.radiation.ozone_ml import predict_ozone_ml
        return predict_ozone_ml(
            T=T,
            lat=lat,
            lon=lon,
            p_full=p_full,
            coefs=ml_ozone_coefs,
            mmr_to_vmr=ozone_config.ml_mmr_to_vmr,
        )

    raise ValueError(
        f"Unknown OzoneProfileConfig.source: {ozone_config.source!r}. "
        f"Choose from 'standard', 'analytical', 'mls', 'none', 'ml'."
    )


# ===========================================================================
# Shared column extraction for all hydrostatic grids
# ===========================================================================

def _extract_tracer_columns(state, ncol, nlev, dtype=None):
    """Extract water vapor and cloud condensate columns from state tracers.

    Works for any state type (HydrostaticState, SpectralHydrostaticState, etc.)
    Returns (q_v_col, q_cloud_col, q_ice_col, n_cloud_col, n_ice_col), each
    shaped (ncol, nlev) (number columns ``None`` when absent).

    ``n_cloud_col`` / ``n_ice_col`` are the double-moment cloud-droplet / ice
    NUMBER columns (Morrison / Seifert-Beheng), fed to the M2005 PSD effective
    radii in ``compute_cloud_properties`` so RRTMGP gets droplet-number-aware
    r_eff (not a fixed constant). They are ``None`` unless the state's tracer
    dict carries ``"N_c"`` / ``"N_i"`` — single-moment schemes and
    specified-Nc Morrison (``dopredictNc=.false.``) leave them absent, so those
    fall back to the constant ``config.r_eff_liq`` / ``r_eff_ice`` (unchanged
    behaviour). UNIT NOTE: by legoESM convention ``N_c`` is per-VOLUME [#/m³]
    and ``N_i`` per-MASS [#/kg] (see ``_warm_rain.effective_Nc`` and
    ``morrison``); both match what ``compute_cloud_properties`` expects, so they
    are passed RAW (no ρ rescale) — mirroring the cubed-sphere NH path.
    """
    if dtype is None:
        # Try to infer dtype from state.T or state.T_hat.  Fall back to
        # the configured compute precision rather than hardcoding fp64
        # — the latter would force radiation to allocate fp64 zeros on
        # Metal/fp32 backends, which then upcast the whole RRTMGP
        # column to fp64 via dtype promotion.
        if hasattr(state, "T"):
            dtype = state.T.data.dtype
        elif hasattr(state, "T_hat"):
            # Spectral state stores complex T_hat; the corresponding
            # real-valued T column should match the spectral grid's
            # real dtype.
            dtype = state.T_hat.data.real.dtype
        else:
            dtype = get_policy().compute
    T_col_shape = (ncol, nlev)
    q_v_col = jnp.zeros(T_col_shape, dtype=dtype)
    q_cloud_col = None
    q_ice_col = None
    n_cloud_col = None
    n_ice_col = None

    tracers = getattr(state, "tracers", None)
    if tracers is not None:
        if "q_v" in tracers:
            _qv_raw = tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        if "q_c" in tracers:
            _qc_raw = tracers["q_c"]
            _qc_data = _qc_raw.data if hasattr(_qc_raw, "data") else _qc_raw
            q_cloud_col = jnp.maximum(_qc_data.reshape(ncol, nlev), 0.0)
        if "q_i" in tracers:
            _qi_raw = tracers["q_i"]
            _qi_data = _qi_raw.data if hasattr(_qi_raw, "data") else _qi_raw
            q_ice_col = jnp.maximum(_qi_data.reshape(ncol, nlev), 0.0)
        # Double-moment NUMBER columns (Morrison / Seifert-Beheng) → M2005 PSD
        # r_eff. N_c per-VOLUME [#/m³], N_i per-MASS [#/kg]; passed raw (see
        # docstring UNIT NOTE). Absent ⇒ None ⇒ constant-r_eff fallback.
        if "N_c" in tracers:
            _nc_raw = tracers["N_c"]
            _nc_data = _nc_raw.data if hasattr(_nc_raw, "data") else _nc_raw
            n_cloud_col = jnp.maximum(_nc_data.reshape(ncol, nlev), 0.0)
        if "N_i" in tracers:
            _ni_raw = tracers["N_i"]
            _ni_data = _ni_raw.data if hasattr(_ni_raw, "data") else _ni_raw
            n_ice_col = jnp.maximum(_ni_data.reshape(ncol, nlev), 0.0)

    return q_v_col, q_cloud_col, q_ice_col, n_cloud_col, n_ice_col


def _get_grid_lat_lon(grid_or_mesh, shape_2d):
    """Get latitude/longitude arrays from any grid type.

    Handles cubed-sphere, lat-lon, and MPAS Voronoi grids uniformly.
    Returns (lat, lon) broadcast to shape_2d.
    """
    if hasattr(grid_or_mesh, 'latCell'):
        # MPAS Voronoi mesh: lat/lon already (nCells,) = shape_2d
        return jnp.asarray(grid_or_mesh.latCell), jnp.asarray(grid_or_mesh.lonCell)

    lat = jnp.asarray(grid_or_mesh.grid_lat)
    lon = jnp.asarray(grid_or_mesh.grid_lon)
    if lat.ndim < len(shape_2d):
        lat = jnp.broadcast_to(
            lat.reshape((*lat.shape, *([1] * (len(shape_2d) - lat.ndim)))),
            shape_2d,
        )
    if lon.ndim < len(shape_2d):
        if lon.ndim == 1 and len(shape_2d) == 2:
            lon = jnp.broadcast_to(lon[None, :], shape_2d)
        else:
            lon = jnp.broadcast_to(
                lon.reshape((*([1] * (len(shape_2d) - lon.ndim)), *lon.shape)),
                shape_2d,
            )
    return lat, lon


def _pack_hydrostatic_tendencies(dT_dt, state, shape_3d, shape_2d,
                                 sw_net_sfc=None, lw_net_sfc=None,
                                 sw_up_toa=None, lw_up_toa=None,
                                 sw_down_toa=None,
                                 sw_down_sfc=None, lw_down_sfc=None,
                                 sw_up_toa_clearsky=None,
                                 lw_up_toa_clearsky=None,
                                 sw_down_sfc_clearsky=None,
                                 lw_down_sfc_clearsky=None,
                                 sw_up_sfc_clearsky=None):
    """Pack column heating rate into a HydrostaticTendencies.

    Returns a HydrostaticTendencies with only dT_dt non-zero.
    Works for cubed-sphere, lat-lon, and MPAS (v fields are zero or None
    depending on whether state.v is present).

    ``sw_net_sfc`` / ``lw_net_sfc`` (both [W/m^2, +into surface], native 2D
    layout) are optional surface radiative net fluxes attached as diagnostics
    so the lean MPAS coupled loop can export them to the coupler; ``None`` (the
    default, e.g. every non-radiation tendency) leaves the fields unset —
    behaviourally identical to the pre-export packer.
    """
    dims_3d = state.T.dims
    dims_2d = state.p_s.dims
    # Pin all zero-tendency placeholders to the upstream state precision
    # so we never silently promote a f32 column path to f64 just to
    # carry an unused-momentum tendency placeholder on the diff result.
    _u_dtype = state.u.data.dtype
    _ps_dtype = state.p_s.data.dtype

    has_v = state.v is not None
    du_shape = state.u.data.shape  # (6,n,n,nlev) or (nEdges,nlev)
    du_dims = state.u.dims

    dv_dt = None
    if has_v:
        dv_dt = Field(
            data=jnp.zeros(state.v.data.shape, dtype=state.v.data.dtype), name="dv_dt_rad",
            dims=state.v.dims, units="m/s^2",
        )

    sw_field = None if sw_net_sfc is None else Field(
        data=sw_net_sfc.reshape(shape_2d).astype(_ps_dtype),
        name="sw_net_sfc_rad", dims=dims_2d, units="W/m^2")
    lw_field = None if lw_net_sfc is None else Field(
        data=lw_net_sfc.reshape(shape_2d).astype(_ps_dtype),
        name="lw_net_sfc_rad", dims=dims_2d, units="W/m^2")

    # TOA fluxes for the CMOR rlut/rsut/rsdt feed (CMOR signs: *_up positive
    # upward/outgoing, sw_down positive downward/incoming — exactly the
    # radiation solver's own flux orientation, no sign flip here).
    def _toa_field(arr, name):
        if arr is None:
            return None
        return Field(data=arr.reshape(shape_2d).astype(_ps_dtype),
                     name=name, dims=dims_2d, units="W/m^2")

    # Downwelling counterparts (+down): forcing for an interactive land tile on
    # the lean MPAS loop (AtmToSurface.sw_down/lw_down); NOT derivable from the
    # net fields at the consumer without re-assuming sfc albedo/emissivity.
    swd_field = None if sw_down_sfc is None else Field(
        data=sw_down_sfc.reshape(shape_2d).astype(_ps_dtype),
        name="sw_down_sfc_rad", dims=dims_2d, units="W/m^2")
    lwd_field = None if lw_down_sfc is None else Field(
        data=lw_down_sfc.reshape(shape_2d).astype(_ps_dtype),
        name="lw_down_sfc_rad", dims=dims_2d, units="W/m^2")
    return HydrostaticTendencies(
        du_dt=Field(
            data=jnp.zeros(du_shape, dtype=_u_dtype), name="du_dt_rad",
            dims=du_dims, units="m/s^2",
        ),
        dT_dt=Field(data=dT_dt, name="dT_dt_rad", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(
            data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dp_s_dt_rad",
            dims=dims_2d, units="Pa/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dphis_dt_rad",
            dims=dims_2d, units="m^2/s^3",
        ),
        dv_dt=dv_dt,
        sw_net_sfc=sw_field,
        lw_net_sfc=lw_field,
        sw_up_toa=_toa_field(sw_up_toa, "sw_up_toa_rad"),
        lw_up_toa=_toa_field(lw_up_toa, "lw_up_toa_rad"),
        sw_down_toa=_toa_field(sw_down_toa, "sw_down_toa_rad"),
        sw_down_sfc=swd_field,
        lw_down_sfc=lwd_field,
        # Clear-sky TOA outgoing (CMOR rsutcs/rlutcs).  SAME positive-UPWARD
        # orientation as the all-sky pair above — both come from the solver's
        # ``*_flux_up[:, 0]``, so no sign flip here or downstream.
        sw_up_toa_clearsky=_toa_field(sw_up_toa_clearsky,
                                      "sw_up_toa_clearsky_rad"),
        lw_up_toa_clearsky=_toa_field(lw_up_toa_clearsky,
                                      "lw_up_toa_clearsky_rad"),
        # Clear-sky SURFACE downwelling (CMOR rsdscs/rldscs).  SAME
        # positive-DOWN orientation as sw_down_sfc / lw_down_sfc above — both
        # come from the solver's ``*_flux_down[:, -1]`` (the surface is the
        # LAST half level), so again no sign flip anywhere.  ``_toa_field`` is
        # a plain (ncol,)->2D Field packer despite the name; reused so the
        # dtype/dims/units handling is defined in exactly one place.
        sw_down_sfc_clearsky=_toa_field(sw_down_sfc_clearsky,
                                        "sw_down_sfc_clearsky_rad"),
        lw_down_sfc_clearsky=_toa_field(lw_down_sfc_clearsky,
                                        "lw_down_sfc_clearsky_rad"),
        # Clear-sky SURFACE UPWELLING shortwave (CMOR rsuscs, which the
        # table declares positive="up").  Orientation matches the all-sky
        # rsus the collector derives (down - net): both are radiation
        # LEAVING the surface, and this one comes straight from the
        # solver's ``sw_flux_up[:, -1]``, so no sign flip here or
        # downstream.
        sw_up_sfc_clearsky=_toa_field(sw_up_sfc_clearsky,
                                      "sw_up_sfc_clearsky_rad"),
    )


def _call_radiation_backend(
    radiation_config: RadiationConfig,
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    p_half: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    lat: jnp.ndarray,
    q_v: jnp.ndarray | None,
    insolation: jnp.ndarray,
    cos_sza: jnp.ndarray | None = None,
    sfc_albedo_override: jnp.ndarray | float | None = None,
    sfc_emissivity_override: jnp.ndarray | float | None = None,
    q_cloud: jnp.ndarray | None = None,
    q_ice: jnp.ndarray | None = None,
    n_ice: jnp.ndarray | None = None,
    n_cloud: jnp.ndarray | None = None,
    ghg_vmr_override: dict | None = None,
    f_day: jnp.ndarray | None = None,
    rrtmgp_solver=None,
    lon: jnp.ndarray | None = None,
    ml_ozone_coefs=None,
    o3_vmr_override: jnp.ndarray | None = None,
    aerosol_od: jnp.ndarray | None = None,
    aerosol_lw_od: jnp.ndarray | None = None,
    solar_spectral_fraction: jnp.ndarray | None = None,
    eccf: float | jnp.ndarray = 1.0,
    cloud_fraction_override: jnp.ndarray | None = None,
    conv_precip: jnp.ndarray | None = None,
):
    """Call configured radiation backend with a unified integration interface.

    Parameters
    ----------
    cos_sza : jnp.ndarray or None
        If provided (diurnal cycle), used directly as RRTMGP cos(zenith).
        Otherwise derived from ``insolation / S_0``.
    q_cloud : jnp.ndarray or None
        Cloud liquid water mixing ratio (ncol, nlev) [kg/kg].
    q_ice : jnp.ndarray or None
        Cloud ice mixing ratio (ncol, nlev) [kg/kg].
    ghg_vmr_override : dict or None
        Runtime GHG VMR overrides passed to RRTMGP (e.g. transient CO2).
    f_day : jnp.ndarray or None
        Daylight fraction per column for non-diurnal RRTMGP.  When provided,
        the two-stream solver uses the daytime-effective cos(SZA) instead of
        the day+night average, and SW fluxes/heating are rescaled by f_day
        to recover daily-mean energy balance.
    o3_vmr_override : jnp.ndarray or None
        Pre-computed external ozone VMR (ncol, nlev) — e.g. the CMIP6
        input4MIPs ozone file interpolated by
        ``ModelDriver._precompute_external_forcing``.  When supplied it
        takes precedence over the config-driven ``_compute_ozone_vmr``
        (standard / analytical / ML profiles), matching the precedence
        the coupled ``physics_pipeline`` path applies.
    aerosol_od : jnp.ndarray or None
        Per-layer aerosol optical depth (ncol, nlev) from the external
        forcing pipeline (Kinne climatology + volcanic), passed to the
        RRTMGP solver as ``aerosol_optical_depth``.  Ignored by gray
        radiation.
    aerosol_lw_od : jnp.ndarray or None
        Per-layer LONGWAVE aerosol absorption optical depth (ncol, nlev)
        from the external forcing pipeline (volcanic stratospheric,
        gap #9), passed to the RRTMGP solver as
        ``aerosol_absorption_optical_depth_lw``.  ``None`` (default) is a
        no-op in the solver — byte-identical to no volcanic LW aerosol.
        Ignored by gray radiation.
    solar_spectral_fraction : jnp.ndarray or None
        Per-g-point solar weights for spectral solar-cycle forcing,
        passed through to ``solve_columns``.
    """
    if radiation_config.scheme == "gray":
        radiation_fn, scheme_config = _get_radiation_fn(radiation_config)
        return radiation_fn(
            T=T,
            p_full=p_full,
            p_half=p_half,
            sfc_temperature=sfc_temperature,
            lat=lat,
            q_v=q_v,
            insolation=insolation,
            config=scheme_config,
        )

    # RRTMGP: use actual cos_sza if available (diurnal cycle), else derive
    # from daily-mean insolation using the daytime-effective zenith angle.
    # The eccentricity (a/r)^2 factor scales the SW *flux* (folded into
    # _sw_scale), NOT the optical-path cosine — so divide ``eccf`` out of any
    # cos_sza derived from the (eccf-folded) ``insolation``.  Gated on a static
    # orbit flag so the circular-orbit path is bit-for-bit unchanged.
    _orbit_on = getattr(radiation_config, "orbit", None) is not None
    _sw_scale = None
    if cos_sza is None:
        S_0 = radiation_config.rrtmgp.S_0
        if f_day is not None:
            # Use daytime-effective cos(SZA): insol = (a/r)^2·S_0·f_day·<cos>_day
            # so <cos>_day = insol / (eccf·S_0·f_day).  The solver sees the
            # geometric daytime optical path; SW output is rescaled by
            # f_day·eccf afterward to recover daily-mean energy + distance.
            f_day_safe = jnp.maximum(f_day, 1.0e-6)
            cos_sza = jnp.clip(
                insolation / (eccf * S_0 * f_day_safe), 0.0, 1.0,
            )
            _sw_scale = f_day * eccf if _orbit_on else f_day
        else:
            cos_sza = jnp.clip(
                insolation / (eccf * jnp.clip(S_0, 1.0e-6, None)),
                0.0,
                1.0,
            )
            if _orbit_on:
                _sw_scale = jnp.full((cos_sza.shape[0],), eccf,
                                     dtype=cos_sza.dtype)
    elif _orbit_on:
        # Diurnal path: cos_sza is already geometric; apply the distance factor
        # to the SW flux (the solver runs with S_0, not S_0·eccf).
        _sw_scale = jnp.full((cos_sza.shape[0],), eccf, dtype=cos_sza.dtype)
    q_v_safe = q_v if q_v is not None else jnp.zeros_like(T)

    # Compute ozone VMR based on config — unless the caller supplied a
    # pre-computed external (CMIP6 file) ozone column, which wins.
    if o3_vmr_override is not None:
        o3_vmr = o3_vmr_override
    else:
        o3_vmr = _compute_ozone_vmr(
            p_full, lat, radiation_config.ozone,
            T=T, lon=lon, ml_ozone_coefs=ml_ozone_coefs,
        )

    # Compute cloud properties if cloud scheme is active.
    cloud_kwargs = {}
    # Bound unconditionally: the subcolumn-overlap block below reads both, and
    # with cloud_scheme="none" this branch never runs.
    cloud_props = None
    cloud_config = None
    if radiation_config.cloud_scheme != "none":
        # Honor a caller-supplied ``cloud_config`` (e.g., AIMIP's
        # trainable Xu-Randall knobs) when present; otherwise build a
        # default cloud config matching the scheme string.
        if radiation_config.cloud_config is not None:
            cloud_config = radiation_config.cloud_config
        else:
            cloud_config = CloudConfig(scheme=radiation_config.cloud_scheme)
        dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)
        cloud_props = compute_cloud_properties(
            T=T,
            p_full=p_full,
            q_v=q_v_safe,
            dp=dp,
            config=cloud_config,
            q_cloud=q_cloud,
            q_ice=q_ice,
            n_ice=n_ice,
            n_cloud=n_cloud,
            conv_precip=conv_precip,
            cloud_fraction_override=cloud_fraction_override,
        )
        # ``to_rrtmg_kwargs`` builds the kwargs without ``cloud_fraction``
        # (commit 4c9591bb, lost in AIMIP-#312 merge, restored iter-15
        # in ``physics_pipeline.py`` and iter-16 here) — see docstring
        # for why.  iter-17 centralised the helper so the bug can't
        # resurface at a third call site.
        cloud_kwargs = cloud_props.to_rrtmg_kwargs()

    # RRTMGP path: use solver directly (config is baked in).
    if rrtmgp_solver is None:
        # Fallback: construct a solver on the fly (e.g. called without
        # the pre-built solver from make_radiation_physics).
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
        rrtmgp_solver = RRTMGP.from_legoesm_config(radiation_config.rrtmgp)

    _rad_kwargs = dict(
        T=T,
        p_full=p_full,
        p_half=p_half,
        sfc_temperature=sfc_temperature,
        q_v=q_v_safe,
        cos_zenith=cos_sza,
        sfc_albedo=sfc_albedo_override,
        sfc_emissivity=sfc_emissivity_override,
        o3_vmr=o3_vmr,
        ghg_vmr_override=ghg_vmr_override,
        aerosol_optical_depth=aerosol_od,
        aerosol_absorption_optical_depth_lw=aerosol_lw_od,
        solar_spectral_fraction=solar_spectral_fraction,
        **cloud_kwargs,
    )
    # Maximum-random-overlap SUBCOLUMNS (opt-in).  The default path hands
    # every layer's GRID-MEAN water path to ONE homogeneous column, so cloud
    # spread thinly over many partly cloudy layers is solved as one deep
    # uniform cloud.  Measured against a Monte-Carlo independent-column
    # reference on a real model state, that costs ~30% of the cloud albedo
    # (0.3995 -> 0.2786) and ~18 W/m2 of OLR (219.2 -> 237.8, reference 239.0).
    # See clouds/subcolumns.py for the validation and for the closed-form
    # alternative that was measured and rejected.
    #
    # The kwargs are chosen BEFORE the solve so exactly ONE solve runs: an
    # earlier version solved the base column and then the subcolumns and threw
    # the base away, doubling the cost and compiling an extra shape.
    _ov = ("none" if cloud_props is None else
           getattr(cloud_config, "cloud_vertical_overlap_optics", "none"))
    _n_sub = 1
    if _ov == "max_random":
        from legoesm.atmosphere.physics.clouds import subcolumns as _sub
        _n_sub = int(getattr(cloud_config, "cloud_n_subcolumns",
                             _sub.N_SUBCOLUMNS_DEFAULT))
        _ncol = T.shape[0]
        _mask = _sub.generate_subcolumns(cloud_props.cloud_fraction, _n_sub)
        _lwp, _iwp = _sub.subcolumn_paths(
            _mask, cloud_props.cloud_fraction, cloud_props.lwp, cloud_props.iwp)
        _rad_kwargs = _sub.expand_kwargs(_rad_kwargs, _n_sub, _ncol)
        _rad_kwargs["cloud_path_liq"] = _lwp
        _rad_kwargs["cloud_path_ice"] = _iwp
    elif _ov != "none":
        # Dispatch hardening: a typo must never silently run the legacy
        # single-column path (validated on the static config value).
        raise ValueError(
            f"unknown cloud_vertical_overlap_optics {_ov!r}; "
            "expected 'none' or 'max_random'"
        )

    # Column-chunk the rrtmgp solve when configured: the per-block body
    # compiles ONCE at ``column_chunk_size`` columns, capping the highly
    # super-linear rrtmgp XLA compile time at higher horizontal resolution.
    # Columns are physically independent, so this is numerically EXACT.
    _col_chunk = getattr(radiation_config.rrtmgp, "column_chunk_size", 0)
    if _col_chunk and _col_chunk > 0:
        result = rrtmgp_solver.solve_columns_chunked(
            column_chunk_size=_col_chunk, **_rad_kwargs,
        )
    else:
        result = rrtmgp_solver.solve_columns(**_rad_kwargs)

    if _n_sub > 1:
        from legoesm.atmosphere.physics.clouds import subcolumns as _sub
        result = _sub.average_output(result, _n_sub, T.shape[0])

    # When using daytime-effective cos(SZA), the solver computes SW fluxes at
    # the daytime level (1/f_day times too large).  Rescale to daily-mean.
    if _sw_scale is not None:
        s = _sw_scale[:, None]  # (ncol, 1) for broadcasting against (ncol, nlev)
        result = RadiationOutput(
            lw_flux_up=result.lw_flux_up,
            lw_flux_down=result.lw_flux_down,
            sw_flux_up=result.sw_flux_up * s,
            sw_flux_down=result.sw_flux_down * s,
            heating_rate=result.lw_heating_rate + result.sw_heating_rate * s,
            lw_heating_rate=result.lw_heating_rate,
            sw_heating_rate=result.sw_heating_rate * s,
        )

    # Carry the prescribed TOA insolation (this scope's per-column
    # ``insolation`` argument, [0, S_0]) so downstream diagnostics read the
    # true TOA incident SW rather than the clamped top-halo flux (#620).
    # Set AFTER the daytime->daily-mean rescale rebuild so the field survives.
    result = result._replace(toa_insolation=insolation)
    return result


def _validate_cloud_gate(radiation_config: RadiationConfig) -> None:
    """Fail loudly on an inconsistent cloud-radiation gate.

    ``RadiationConfig.cloud_scheme`` (which cloud-fraction/optics scheme runs)
    and ``RRTMGPConfig.include_clouds`` (whether the RRTMGP solver honours the
    cloud optics) are independent knobs with no cross-sync — ``RadiationConfig``
    is a plain ``NamedTuple`` and ``ExperimentConfig.validate_strict`` only
    checks membership, never consistency.  When a cloud scheme is active but
    ``include_clouds`` is False the cloud optics are still computed and passed
    to ``solve_columns`` (the ``cloud_scheme != "none"`` branch of
    ``_call_radiation_backend``), then SILENTLY NULLED by the
    ``has_clouds = include_clouds and ...`` gate in ``rrtmgp.py`` → clear-sky
    radiation with no error/warning and a dead gradient through any trained
    cloud knobs (e.g. AIMIP's Xu-Randall ``cloud_config``, whose whole purpose
    is end-to-end trainable cloud-radiation coupling).

    Rather than silently coerce one knob (which would blanket-enable RRTMGP
    cloud optics on every standalone dycore path, including ones whose tracer
    extractor would then pass partial/bare condensate and run optically inert),
    this RAISES so the caller must make the two consistent at the config
    source.  The coupled pipeline (``physics_pipeline._build_rrtmgp_radiation_fn``)
    and the matrix runner already derive ``include_clouds`` from
    ``cloud_scheme`` before constructing the config, so they never trip this;
    direct ``RadiationConfig`` builders (AIMIP, ``combined.py``) must do the
    same.  Only RRTMGP has the gate; gray radiation is never inconsistent.
    """
    # mc3d shares the RRTMGP optics path (Phase 2b: it calls solve_columns for
    # the per-g-point shortwave optical field), so it has the SAME silent-cloud-
    # drop hazard when include_clouds=False.
    if (
        radiation_config.scheme in ("rrtmgp", "mc3d")
        and radiation_config.cloud_scheme != "none"
        and not radiation_config.rrtmgp.include_clouds
    ):
        raise ValueError(
            "Inconsistent cloud-radiation gate: cloud_scheme="
            f"{radiation_config.cloud_scheme!r} is active but "
            "RRTMGPConfig.include_clouds is False, so the computed cloud "
            "optics would be SILENTLY discarded by the RRTMGP solver "
            "(clear-sky radiation, no error, dead cloud gradient). Set "
            "rrtmgp=RRTMGPConfig(include_clouds=True) to honour the clouds, "
            "or cloud_scheme='none' for a genuine clear-sky run."
        )


def make_radiation_physics(
    radiation_config: RadiationConfig,
    model_type: str = "hydrostatic",
    column_mesh=None,
    sfc_albedo_override: jnp.ndarray | float | None = None,
    sfc_emissivity_override: jnp.ndarray | float | None = None,
    nc_from_aerosol: bool = False,
    activation_config=None,
    use_clubb_cloud_fraction: bool = False,
) -> Callable:
    """Create a physics function for radiation matching a model's signature.

    Parameters
    ----------
    radiation_config : RadiationConfig
        Radiation configuration (selects gray or RRTMGP).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "plane", "mpas_nh",
        "spectral_pe", "mpas".
    column_mesh : jax.sharding.Mesh or None, optional
        Issue #273 follow-up.  When supplied, the per-column radiation
        kernel (which is naturally embarrassingly parallel across
        columns) is sharded across the mesh's ``'col'`` axis.  Use
        ``legoesm.parallel.column_shard.create_column_mesh`` to build
        one.  Required invariant: ``ncol`` (the flattened horizontal
        column count = ``6 · n · n`` on the cubed sphere) must be
        divisible by the mesh's device count.  Default ``None``
        preserves single-mesh behavior bit-exact.

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    # Cloud-radiation gate consistency — single chokepoint every standalone
    # dycore radiation factory (incl. combined.py / AIMIP spectral_pe) passes
    # through.  See ``_validate_cloud_gate`` for why this is required.
    _validate_cloud_gate(radiation_config)

    # ``sfc_albedo_override`` / ``sfc_emissivity_override`` are build-time
    # surface fields (e.g. AIMIP's trained spatial ``(ncol,)`` arrays) routed to
    # the radiation solve as PER-CALL overrides via ``_resolve_surface_field`` —
    # so a trained/traced value reaches the heating WITHOUT being written into
    # ``RRTMGPConfig.sfc_*`` (which RRTMGP folds into its Python solver-cache key
    # and would then key by tracer identity). Only the spectral_pe builder
    # consumes them today; reject them loudly elsewhere rather than silently
    # dropping a trained surface field.
    if (sfc_albedo_override is not None or sfc_emissivity_override is not None) \
            and model_type != "spectral_pe":
        raise ValueError(
            "sfc_albedo_override / sfc_emissivity_override are only wired for "
            f"model_type='spectral_pe', got {model_type!r}. Extend the relevant "
            "_make_*_radiation builder before passing surface overrides there."
        )

    # 3D Monte-Carlo ray tracing is plane-LES/CRM only (periodic horizontal BC).
    if radiation_config.scheme == "mc3d" and model_type != "plane":
        raise ValueError(
            "scheme='mc3d' (3D Monte-Carlo ray tracing) is only wired for "
            f"model_type='plane' (LES/CRM); got model_type={model_type!r}."
        )

    # CLUBB-cloud-fraction -> radiation routing (moist-closure sub-grid cf feeds
    # the cloud optics via PhysicsState).  Only the hydrostatic builder threads
    # ``phys_state`` into the radiation physics_fn today (it covers both AMIP
    # grids: cubed-sphere and lat-lon hydrostatic).  Refuse LOUDLY on the other
    # dycores rather than silently ignoring the request (dispatch-hardening) —
    # the turbulence WRITE side is wired on all grids, so extend the matching
    # _make_*_radiation READ side before enabling it there.
    if use_clubb_cloud_fraction and model_type != "hydrostatic":
        raise NotImplementedError(
            "RadiationConfig.use_clubb_cloud_fraction is only wired for "
            f"model_type='hydrostatic', got {model_type!r}.  Extend the "
            "corresponding _make_*_radiation builder (thread phys_state ->"
            " cloud_fraction_override) before enabling CLUBB-cf routing there."
        )

    # Load heavy/static RRTMGP optics once outside model JIT traces. mc3d also
    # needs the RRTMGP optics tables (Phase 2b: 3D-MC shortwave uses RRTMGP
    # per-g-point optics; falls back to gray optics if the tables are absent).
    rrtmgp_solver = None
    if radiation_config.scheme in ("rrtmgp", "mc3d"):
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
        try:
            RRTMGP.preload(radiation_config.rrtmgp)
            rrtmgp_solver = RRTMGP.from_legoesm_config(radiation_config.rrtmgp)
        except Exception:
            if radiation_config.scheme == "rrtmgp":
                raise
            # mc3d: gray-optics fallback when RRTMGP data is unavailable.
            rrtmgp_solver = None

    # Load ML ozone ridge weights once (outside JIT).  Gray radiation
    # ignores ozone, so skip the (potentially large) NetCDF load when
    # scheme="gray" even if ozone.source="ml" is set.
    ml_ozone_coefs = None
    if radiation_config.ozone.source == "ml" and radiation_config.scheme == "rrtmgp":
        if not radiation_config.ozone.ml_weights_path:
            raise ValueError(
                "OzoneProfileConfig.source='ml' requires ml_weights_path to "
                "point at a directory of NetCDF ridge weights."
            )
        from legoesm.atmosphere.physics.radiation.ozone_ml import (
            load_ml_ozone_coefficients,
        )
        ml_ozone_coefs = load_ml_ozone_coefficients(
            radiation_config.ozone.ml_weights_path
        )

    if model_type == "hydrostatic":
        return _make_hydrostatic_radiation(radiation_config, rrtmgp_solver,
                                            ml_ozone_coefs=ml_ozone_coefs,
                                            column_mesh=column_mesh,
                                            nc_from_aerosol=nc_from_aerosol,
                                            activation_config=activation_config,
                                            use_clubb_cloud_fraction=use_clubb_cloud_fraction)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_radiation(radiation_config, rrtmgp_solver,
                                               ml_ozone_coefs=ml_ozone_coefs)
    elif model_type == "plane":
        return _make_plane_radiation(radiation_config, rrtmgp_solver,
                                     ml_ozone_coefs=ml_ozone_coefs)
    elif model_type == "mpas_nh":
        return _make_mpas_nh_radiation(radiation_config, rrtmgp_solver,
                                       ml_ozone_coefs=ml_ozone_coefs)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_radiation(
            radiation_config, rrtmgp_solver,
            ml_ozone_coefs=ml_ozone_coefs,
            sfc_albedo_override=sfc_albedo_override,
            sfc_emissivity_override=sfc_emissivity_override,
        )
    elif model_type == "mpas":
        return _make_mpas_radiation(radiation_config, rrtmgp_solver,
                                     ml_ozone_coefs=ml_ozone_coefs,
                                     column_mesh=column_mesh,
                                     nc_from_aerosol=nc_from_aerosol)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'plane', "
            f"'mpas_nh', 'spectral_pe', 'mpas'."
        )


# ===========================================================================
# Hydrostatic PE (cubed-sphere, lat-lon, and MPAS Voronoi)
# ===========================================================================

def _make_hydrostatic_radiation(
    radiation_config: RadiationConfig,
    rrtmgp_solver=None,
    ml_ozone_coefs=None,
    column_mesh=None,
    nc_from_aerosol: bool = False,
    activation_config=None,
    use_clubb_cloud_fraction: bool = False,
) -> Callable:
    """Create radiation physics_fn for any hydrostatic model.

    Handles cubed-sphere, lat-lon FV, and MPAS Voronoi grids via the
    shared ``_get_grid_lat_lon`` / ``_extract_tracer_columns`` /
    ``_pack_hydrostatic_tendencies`` helpers.

    Signature: (state, grid_or_mesh, sigma_coord) -> HydrostaticTendencies

    When ``column_mesh`` is provided (issue #273 follow-up), the
    per-column radiation kernel runs sharded across the mesh's
    ``'col'`` axis.  Caller is responsible for ensuring the flattened
    column count ``ncol = ∏ shape_2d`` divides the mesh's device
    count.

    When the attached cloud config enables ``convective_cloud``, the fn
    reads the LAGGED ``phys_state.conv_precip`` carry (published by the
    convection module the previous step) and threads it into the Slingo
    cumulus cloud fraction — the standalone-path analogue of the FV
    pipeline's ``conv_precip`` threading.

    When ``nc_from_aerosol`` is True AND the scheme is ``rrtmgp`` the
    cloud-optics droplet number is overridden with the per-column Andreae
    (2009) AOD->CCN diagnostic (``forcing["aerosol_od"]``) so the radiation
    effective radius responds to the prescribed aerosol — the Twomey first
    indirect effect, kept consistent with the microphysics specified-Nc
    fill.  Gray radiation ignores ``n_cloud`` entirely, so the override is
    skipped there (the microphysics second-indirect fill still runs).
    Default False is byte-identical (the double-moment N_c carry / constant
    r_eff is used).
    """
    _time, set_time = _make_time_state()
    _T_sfc_override_cell, set_T_sfc_override = _make_T_sfc_override_cell()
    # Static build-time gate for the Slingo convective-cloud carry read:
    # only a cloud config that ENABLES convective_cloud makes the fn a
    # phys_state consumer (byte-identical otherwise).
    _conv_cloud_active = (
        radiation_config.cloud_scheme != "none"
        and radiation_config.cloud_config is not None
        and bool(getattr(radiation_config.cloud_config,
                         "convective_cloud", False))
    )

    # --- Clear-sky TOA diagnostic (CMOR rsutcs/rlutcs) build-time gate ---
    # STATIC Python bools resolved once here, never a traced ``jnp.where``:
    # ``clear_sky_diag=False`` (the default) compiles no extra radiation HLO
    # and leaves the tendency's clear-sky slots None -> byte-identical.
    #
    # Whether a SECOND solve is needed is decided by whether any cloud is
    # radiatively active:
    #   * gray radiation returns before the cloud block in
    #     ``_call_radiation_backend`` (it ignores clouds entirely), and
    #   * ``cloud_scheme == "none"`` skips ``compute_cloud_properties``, so no
    #     cloud path/fraction ever reaches the solver
    #     (``has_clouds = include_clouds and cloud_path_* is not None`` in
    #     rrtmgp.solve_columns is then False).
    # In both cases the all-sky solve IS the clear-sky solve, so the clear-sky
    # slots ALIAS the all-sky TOA arrays — exact, and with zero extra cost.
    _clear_sky_diag = bool(radiation_config.clear_sky_diag)
    _clear_sky_clouds_active = (radiation_config.scheme != "gray"
                                and radiation_config.cloud_scheme != "none")
    _clear_sky_second_pass = _clear_sky_diag and _clear_sky_clouds_active
    # Cloud-free twin of the radiation config, built ONCE (not per step).
    # ``rrtmgp`` is left UNTOUCHED so the second call reuses the SAME prebuilt
    # solver / optics cache; dropping the cloud scheme alone already removes
    # every cloud input from the solve.  ``clear_sky_diag`` is cleared so the
    # twin can never be mistaken for a recursion trigger.
    _rad_cfg_clearsky = (
        radiation_config._replace(cloud_scheme="none", cloud_config=None,
                                  clear_sky_diag=False)
        if _clear_sky_second_pass else None)

    def physics_fn(state, grid_or_mesh, sigma_coord,
                   forcing=None, phys_state=None) -> HydrostaticTendencies:
        T = state.T.data
        p_s = state.p_s.data

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape
        ncol = int(math.prod(int(s) for s in shape_2d))

        lat, lon = _get_grid_lat_lon(grid_or_mesh, shape_2d)

        # Pressure at full and half levels
        p_full = sigma_coord.pressure_at_full(p_s)
        p_half = sigma_coord.pressure_at_half(p_s)

        # Surface temperature = lowest-level temperature (default), overridable
        # by EITHER a per-step TRACED ``forcing["T_sfc"]`` (the AMIP path —
        # passes a time-varying prescribed SST through the JIT'd dycore step
        # without retracing) OR the static ``set_T_sfc_override`` closure (the
        # SCM/fixed-anchor path).  Traced forcing wins when supplied; the two
        # never both apply per call.  Both go through ``_apply_T_sfc_override``
        # so the (ncol,) shape contract + NaN-sentinel semantics are shared.
        _ovr = None
        if forcing is not None and forcing.get("T_sfc") is not None:
            _ovr = forcing["T_sfc"]
        else:
            _ovr = _T_sfc_override_cell[0]
        T_sfc = _apply_T_sfc_override(T[..., -1], _ovr)

        # External CMIP6 forcing (MPAS / standalone-driver paths): the
        # coupled cube/lat-lon pipeline threads these through
        # ``SegmentForcing``; here they arrive via the same per-step
        # TRACED ``forcing`` dict as ``T_sfc`` so the JIT'd step never
        # retraces when the monthly forcing values change.
        #   o3_vmr      : (ncol, nlev) external ozone VMR
        #   aerosol_od  : (ncol, nlev) per-layer aerosol optical depth
        #   ghg_vmr     : dict[str, scalar] transient GHG VMRs
        _o3_ext = forcing.get("o3_vmr") if forcing is not None else None
        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
        _aer_lw_ext = (
            forcing.get("aerosol_lw_od") if forcing is not None else None
        )
        _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None

        # Calendar time: prefer per-step TRACED forcing values (the MPAS
        # AMIP loop) over the static ``set_time`` closure.  The closure
        # cell is read at TRACE time inside the JIT'd dycore step, so a
        # multi-year MPAS run would otherwise integrate with the
        # insolation frozen at the initial day — no seasonal or diurnal
        # cycle (found 2026-06-10 while wiring CMIP6 forcing into the
        # standalone grid paths).
        _doy = _time["day_of_year"]
        _sod = _time["seconds_of_day"]
        if forcing is not None and forcing.get("day_of_year") is not None:
            _doy = forcing["day_of_year"]
        if forcing is not None and forcing.get("seconds_of_day") is not None:
            _sod = forcing["seconds_of_day"]
        # Transient solar (CMIP6 TSI + optional per-g-point spectral weights):
        # traced per-step forcing, same channel as T_sfc/o3/ghg above, so a
        # multi-year MPAS/standalone run follows the solar file WITHOUT
        # retracing (the coupled cube/lat-lon pipeline threads the equivalent
        # via SegmentForcing/current_s_0).  Absent keys -> the configured
        # static S_0 and the solver's default spectrum, byte-identical.
        _tsi_ext = forcing.get("tsi") if forcing is not None else None
        _ssf_ext = (forcing.get("solar_spectral_fraction")
                    if forcing is not None else None)

        insol, cos_sza, f_day, eccf = _compute_insolation(
            lat, radiation_config,
            lon=lon,
            day_of_year=_doy,
            seconds_of_day=_sod,
        )
        if _tsi_ext is not None:
            # insolation is EXACTLY linear in S_0 in both the diurnal and
            # daily-mean branches of _compute_insolation, so a post-scale by
            # tsi/S_0_config is the transient-TSI application with no second
            # orbital computation.
            _S0_cfg = (radiation_config.rrtmgp.S_0
                       if radiation_config.scheme == "rrtmgp"
                       else radiation_config.gray.S_0)
            insol = insol * (_tsi_ext / _S0_cfg)

        # Flatten to column-major (ncol, nlev)
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        lon_col = lon.reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = cos_sza.reshape(ncol) if cos_sza is not None else None

        q_v_col, q_cloud_col, q_ice_col, n_cloud_col, n_ice_col = (
            _extract_tracer_columns(state, ncol, nlev)
        )

        # Aerosol-CCN droplet number for the cloud-optics PSD (Twomey first
        # indirect effect): override the (dead-zeros / constant-r_eff) N_c
        # with the per-column Andreae (2009) AOD->CCN diagnostic so the
        # radiation effective radius is consistent with the microphysics
        # specified-Nc fill.  Mirrors the coupled physics_pipeline radiation
        # fill; fail fast if the coupling is configured but no aerosol field
        # was threaded.  Done BEFORE the optional column-shard below so the
        # overridden field shards with the rest.  Only ``rrtmgp`` consumes a
        # droplet number (gray ignores ``n_cloud`` entirely), so the override
        # is gated on the scheme — gray runs the microphysics fill (second
        # indirect effect) but skips this radiation-only first-indirect path
        # instead of computing a droplet field the gray optics would discard.
        if nc_from_aerosol and radiation_config.scheme == "rrtmgp":
            # Route through the SAME activation dispatch as the microphysics
            # N_c fill (proxy | arg) so the first (radiation r_eff) and second
            # (autoconversion) indirect effects see ONE droplet number per
            # step.  activation_config=None -> default proxy, byte-identical
            # to the previous direct specified_nc_field call.  The
            # missing-aerosol_od raise is proxy-only, mirroring the
            # microphysics guard (ARG needs no AOD).
            from legoesm.atmosphere.physics.microphysics.arg_activation import (  # noqa: E501
                ActivationConfig,
                activated_nc_field,
            )
            _act_cfg = (activation_config if activation_config is not None
                        else ActivationConfig())
            if _act_cfg.scheme == "proxy" and _aer_ext is None:
                raise ValueError(
                    "nc_from_aerosol=True but no 'aerosol_od' was passed to "
                    "the radiation physics_fn via forcing — enable external "
                    "aerosol forcing (--aerosol-forcing external) or disable "
                    "--aerosol-ccn."
                )
            _aer_num = (forcing.get("aerosol_number")
                        if forcing is not None else None)
            n_cloud_col = activated_nc_field(
                _act_cfg, (ncol, nlev),
                aerosol_od=(None if _aer_ext is None
                            else jnp.asarray(_aer_ext)),
                T=T_col, p=p_full_col,
                aerosol_number=(None if _aer_num is None
                                else jnp.asarray(_aer_num)),
            )

        f_day_col = f_day.reshape(ncol) if f_day is not None else None

        # CLUBB cloud-fraction READ: a moist higher-order turbulence closure
        # writes its PDF sub-grid cloud fraction into ``phys_state.cloud_fraction``
        # (the turbulence physics_fn -> PhysicsState carry); route it to the cloud
        # optics so radiation reflects the moist closure's less-overcast marine BL
        # instead of the RH grid-scale fraction.  Gated to the clubb-active path by
        # the caller (``use_clubb_cloud_fraction`` is only True when turbulence is a
        # cf-producing scheme), so on non-clubb runs this stays None (byte-
        # identical).  PhysicsState carries the column form (ncol, nlev); reshape
        # defensively to the local column layout.
        _cf_ovr = None
        if use_clubb_cloud_fraction and phys_state is not None:
            _cf_ovr = getattr(phys_state, "cloud_fraction", None)
            if _cf_ovr is not None:
                _cf_ovr = _cf_ovr.reshape(T_col.shape)

        # Convective-precip READ (standalone-path Slingo cumulus fraction):
        # the convection module published its column-integrated in-updraft
        # rain-production rate [kg/m^2/s] into ``phys_state.conv_precip``
        # LAST step (radiation runs first in the module chain — one-step
        # lag, the FV pipeline's ``conv_precip`` convention).  Gated by the
        # cloud config's ``convective_cloud`` so every other run keeps
        # ``None`` (byte-identical; compute_cloud_properties' misconfig
        # guard still fires if convective_cloud is on with no carry).
        _conv_precip_col = None
        if _conv_cloud_active and phys_state is not None:
            _conv_precip_col = getattr(phys_state, "conv_precip", None)
            if _conv_precip_col is not None:
                _conv_precip_col = _conv_precip_col.reshape(ncol)

        # Issue #273 follow-up: optionally shard the per-column radiation
        # workload across ``column_mesh`` so a 4×A100 (or any device
        # count that fails cubed-sphere face-divisibility) keeps every
        # device busy on the radiation hot path.  Sharding propagates
        # through ``_call_radiation_backend`` automatically because the
        # backend kernels are purely functional over the column axis.
        if column_mesh is not None:
            from legoesm.parallel.column_shard import shard_columns
            n_dev = column_mesh.shape["col"]
            if ncol % n_dev != 0:
                raise ValueError(
                    f"column_mesh requires ncol={ncol} divisible by "
                    f"n_devices={n_dev}.  Pick an n_devices that divides "
                    f"6·n·n for the cubed-sphere grid, or pre-pad upstream."
                )
            T_col = shard_columns(T_col, column_mesh)
            p_full_col = shard_columns(p_full_col, column_mesh)
            p_half_col = shard_columns(p_half_col, column_mesh)
            T_sfc_col = shard_columns(T_sfc_col, column_mesh)
            lat_col = shard_columns(lat_col, column_mesh)
            lon_col = shard_columns(lon_col, column_mesh)
            insol_col = shard_columns(insol_col, column_mesh)
            if cos_sza_col is not None:
                cos_sza_col = shard_columns(cos_sza_col, column_mesh)
            if q_v_col is not None:
                q_v_col = shard_columns(q_v_col, column_mesh)
            if q_cloud_col is not None:
                q_cloud_col = shard_columns(q_cloud_col, column_mesh)
            if q_ice_col is not None:
                q_ice_col = shard_columns(q_ice_col, column_mesh)
            if n_cloud_col is not None:
                n_cloud_col = shard_columns(n_cloud_col, column_mesh)
            if n_ice_col is not None:
                n_ice_col = shard_columns(n_ice_col, column_mesh)
            if f_day_col is not None:
                f_day_col = shard_columns(f_day_col, column_mesh)
            if _o3_ext is not None:
                _o3_ext = shard_columns(_o3_ext, column_mesh)
            if _aer_ext is not None:
                _aer_ext = shard_columns(_aer_ext, column_mesh)
            if _aer_lw_ext is not None:
                _aer_lw_ext = shard_columns(_aer_lw_ext, column_mesh)
            if _cf_ovr is not None:
                _cf_ovr = shard_columns(_cf_ovr, column_mesh)
            if _conv_precip_col is not None:
                _conv_precip_col = shard_columns(_conv_precip_col, column_mesh)

        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            eccf=eccf,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
            cos_sza=cos_sza_col,
            q_cloud=q_cloud_col,
            q_ice=q_ice_col,
            n_cloud=n_cloud_col,
            n_ice=n_ice_col,
            f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver,
            lon=lon_col,
            ml_ozone_coefs=ml_ozone_coefs,
            o3_vmr_override=_o3_ext,
            aerosol_od=_aer_ext,
            aerosol_lw_od=_aer_lw_ext,
            ghg_vmr_override=_ghg_ext,
            cloud_fraction_override=_cf_ovr,
            conv_precip=_conv_precip_col,
            solar_spectral_fraction=_ssf_ext,
        )

        # --- Clear-sky TOA second pass (CMOR rsutcs/rlutcs) ---
        # Identical inputs to the all-sky solve above (same columns, same
        # zenith/insolation, same gases + ozone + AEROSOL — CMIP6 clear-sky
        # removes CLOUDS only) with every cloud input dropped and the
        # cloud-free config twin selected.  Python-``if`` on a build-time
        # static bool: the disabled path emits no HLO at all.
        _sw_up_toa_clr = None
        _lw_up_toa_clr = None
        _sw_down_sfc_clr = None
        _lw_down_sfc_clr = None
        _sw_up_sfc_clr = None
        if _clear_sky_second_pass:
            _rad_out_clr = _call_radiation_backend(
                radiation_config=_rad_cfg_clearsky,
                eccf=eccf,
                T=T_col,
                p_full=p_full_col,
                p_half=p_half_col,
                sfc_temperature=T_sfc_col,
                lat=lat_col,
                q_v=q_v_col,
                insolation=insol_col,
                cos_sza=cos_sza_col,
                q_cloud=None,
                q_ice=None,
                n_cloud=None,
                n_ice=None,
                f_day=f_day_col,
                rrtmgp_solver=rrtmgp_solver,
                lon=lon_col,
                ml_ozone_coefs=ml_ozone_coefs,
                o3_vmr_override=_o3_ext,
                aerosol_od=_aer_ext,
                aerosol_lw_od=_aer_lw_ext,
                ghg_vmr_override=_ghg_ext,
                cloud_fraction_override=None,
                conv_precip=None,
                solar_spectral_fraction=_ssf_ext,
            )
            # TOA is the FIRST half level, the surface the LAST — the same
            # indexing the all-sky block below uses, so the clear-sky quartet
            # is read off exactly where its all-sky partners are.
            _sw_up_toa_clr = _rad_out_clr.sw_flux_up[:, 0]
            _lw_up_toa_clr = _rad_out_clr.lw_flux_up[:, 0]
            _sw_down_sfc_clr = _rad_out_clr.sw_flux_down[:, -1]
            _lw_down_sfc_clr = _rad_out_clr.lw_flux_down[:, -1]
            # Surface UPWELLING SW from the SAME cloud-free solve (CMOR
            # rsuscs, positive UP): the solver's own ``sw_flux_up`` at the
            # LAST half level, i.e. the albedo-reflected clear-sky
            # downwelling.  Free -- the second pass is already paid for.
            _sw_up_sfc_clr = _rad_out_clr.sw_flux_up[:, -1]
        elif _clear_sky_diag:
            # No cloud is radiatively active (gray, or cloud_scheme='none'),
            # so the all-sky solve already IS the clear-sky solve: alias it
            # rather than paying for a second, provably identical, call.
            _sw_up_toa_clr = rad_out.sw_flux_up[:, 0]
            _lw_up_toa_clr = rad_out.lw_flux_up[:, 0]
            _sw_down_sfc_clr = rad_out.sw_flux_down[:, -1]
            _lw_down_sfc_clr = rad_out.lw_flux_down[:, -1]
            _sw_up_sfc_clr = rad_out.sw_flux_up[:, -1]

        dT_dt = rad_out.heating_rate.reshape(shape_3d)
        # Surface net radiative fluxes [W/m^2, +into surface], carried so the
        # lean MPAS coupled loop can export them to the coupler. Surface is the
        # LAST half-level (T_sfc uses T[..., -1]); at the surface the upward SW
        # is the albedo-reflected downward, so (down - up) equals the compiled
        # path's sw_down*(1-albedo) (physics_pipeline.py) with no albedo term.
        # lw net (down - up) matches that path's lw_net_sfc convention exactly.
        _swn = rad_out.sw_flux_down[:, -1] - rad_out.sw_flux_up[:, -1]
        _lwn = rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]
        # TOA is the FIRST half-level (surface is the last, see above):
        # rlut = lw_flux_up[:, 0], rsut = sw_flux_up[:, 0] — the range-limited
        # top-halo up-faces, exactly what the compiled path reads
        # (physics_pipeline.py:2219-2220), already in CMOR sign conventions.
        # rsdt = PRESCRIBED toa_insolation (#620), NOT the quadratically clamped
        # top-halo down-flux rad_out.sw_flux_down[:, 0] (~15% low; historically
        # ~2x high before the range-limit) — matches the compiled path
        # (physics_pipeline.py:2225-2228). Halo fallback keeps a value for any
        # path that leaves toa_insolation=None (e.g. the zero-radiation stub).
        return _pack_hydrostatic_tendencies(
            dT_dt, state, shape_3d, shape_2d,
            sw_net_sfc=_swn, lw_net_sfc=_lwn,
            sw_up_toa=rad_out.sw_flux_up[:, 0],
            lw_up_toa=rad_out.lw_flux_up[:, 0],
            sw_down_toa=(rad_out.toa_insolation
                         if rad_out.toa_insolation is not None
                         else rad_out.sw_flux_down[:, 0]),
            sw_down_sfc=rad_out.sw_flux_down[:, -1],
            lw_down_sfc=rad_out.lw_flux_down[:, -1],
            sw_up_toa_clearsky=_sw_up_toa_clr,
            lw_up_toa_clearsky=_lw_up_toa_clr,
            sw_down_sfc_clearsky=_sw_down_sfc_clr,
            lw_down_sfc_clearsky=_lw_down_sfc_clr,
            sw_up_sfc_clearsky=_sw_up_sfc_clr)

    physics_fn.set_time = set_time
    physics_fn.set_T_sfc_override = set_T_sfc_override
    # Marker: this physics_fn consumes a per-step traced ``forcing`` dict
    # (currently ``forcing["T_sfc"]``).  The combined-physics dispatcher
    # (_make_hydrostatic_combined) checks this attribute and forwards
    # ``forcing`` only to fns that advertise it — so unmarked sub-physics
    # keep their 3-arg signature unchanged.
    physics_fn._wants_forcing = True
    # Marker: this physics_fn READS ``phys_state`` (the CLUBB sub-grid cloud
    # fraction carry) but writes no PhysicsState carry of its own, so it keeps
    # its single-return contract.  combined.py's accumulator forwards
    # ``phys_state`` to accepts_ps=False fns that advertise this flag.  Only set
    # when the feature is active (byte-identical otherwise: unmarked => not
    # forwarded => the RH grid-scale cloud path is unchanged).
    if use_clubb_cloud_fraction or _conv_cloud_active:
        physics_fn._wants_phys_state_ro = True
    return physics_fn

# MPAS uses the same unified hydrostatic radiation function.
_make_mpas_radiation = _make_hydrostatic_radiation


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_radiation(
    radiation_config: RadiationConfig,
    rrtmgp_solver=None,
    ml_ozone_coefs=None,
) -> Callable:
    """Create radiation physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric) -> NonHydrostaticTendencies
    """
    _time, set_time = _make_time_state()
    _T_sfc_override_cell, set_T_sfc_override = _make_T_sfc_override_cell()

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
    ) -> NonHydrostaticTendencies:
        theta_p = state.theta_prime.data   # (6, n, n, nlev)
        rho_p = state.rho_prime.data       # (6, n, n, nlev)
        lat = grid.lat                     # (6, n, n)
        lon = grid.lon                     # (6, n, n)

        # Reference profiles (1D -> broadcast)
        theta_0 = height_coord.theta_ref   # (nlev,)
        rho_0 = height_coord.rho_ref       # (nlev,)

        # Total fields
        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p,
            rho_0 + rho_p,
        )

        # Temperature: T = theta * exner
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape  # (6, n, n, nlev)
        shape_w = state.w.data.shape  # (6, n, n, nlev+1)
        shape_2d = state.phis.data.shape  # (6, n, n)

        # Interface pressure from evolving column state (not fixed reference).
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )

        # Surface temperature = lowest-level temperature (default)
        # with optional SCM-driver override via set_T_sfc_override hook.
        T_sfc = _apply_T_sfc_override(T[..., -1], _T_sfc_override_cell[0])

        # Insolation (and optionally cos_sza for diurnal cycle).
        insol, cos_sza, f_day, eccf = _compute_insolation(
            lat, radiation_config,
            lon=lon,
            day_of_year=_time["day_of_year"],
            seconds_of_day=_time["seconds_of_day"],
        )

        # Reshape to columns
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        lon_col = lon.reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = cos_sza.reshape(ncol) if cos_sza is not None else None

        # NH state stores water vapor in tracer slot 0 when moist tracers exist.
        n_tracers = state.tracers.data.shape[-1]
        if n_tracers > 0:
            q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
        else:
            q_v = jnp.zeros_like(T)
        q_v_col = q_v.reshape(ncol, nlev)

        # Cloud condensate from tracer slots 1 (q_c) and 3 (q_i).
        q_cloud_col = None
        q_ice_col = None
        if n_tracers > 1:
            q_cloud_col = jnp.clip(
                state.tracers.data[..., 1], 0.0, None
            ).reshape(ncol, nlev)
        if n_tracers > 3:
            q_ice_col = jnp.clip(
                state.tracers.data[..., 3], 0.0, None
            ).reshape(ncol, nlev)
        # Cloud-ice NUMBER (per-mass [1/kg]) from tracer slot 8 = N_i in the
        # canonical 9-slot double-moment layout (q_v,q_c,q_r,q_i,q_s,q_g,
        # N_c,N_r,N_i). ALL double-moment schemes share N_c=6 / N_i=8:
        # morrison, seifert_beheng, thompson, ml_emulator, and p3 (p3 reuses
        # slots 4/5 for q_rim/B_rim but keeps N_c=6/N_r=7/N_i=8, with N_i
        # per-mass via N_i_target/ρ). Feeds the M2005 PSD ice effective radius
        # EFFI=1.5/LAMI (RAD-1-ice). The ``> 8`` guard keeps single-moment
        # layouts (kessler/sundqvist, 3 slots) on the constant r_eff. CAVEAT
        # (codex iter-12): a hypothetical non-standard ≥9-slot layout that did
        # NOT place N_i at slot 8 would mis-read it — a tracer-metadata/scheme
        # key would be more robust but is deferred; every current 9-slot scheme
        # honours this layout.
        n_ice_col = None
        if n_tracers > 8:
            n_ice_col = jnp.clip(
                state.tracers.data[..., 8], 0.0, None
            ).reshape(ncol, nlev)
        # Cloud-droplet NUMBER (per-VOLUME [#/m³]) from slot 6 (N_c in the
        # Morrison double-moment layout) ⇒ the M2005 PSD liquid effective radius
        # reffc=(PGAM+3)/(2·LAMC). Same ``> 8`` (Morrison) guard as N_i.
        n_cloud_col = None
        if n_tracers > 8:
            n_cloud_col = jnp.clip(
                state.tracers.data[..., 6], 0.0, None
            ).reshape(ncol, nlev)

        f_day_col = f_day.reshape(ncol) if f_day is not None else None
        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            eccf=eccf,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
            cos_sza=cos_sza_col,
            q_cloud=q_cloud_col,
            q_ice=q_ice_col,
            n_ice=n_ice_col,
            n_cloud=n_cloud_col,
            f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver,
            lon=lon_col,
            ml_ozone_coefs=ml_ozone_coefs,
        )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
        dT_dt = rad_out.heating_rate.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")
        # Pin zero-tendency placeholders to the upstream state precision
        # so x64 default zeros do not leak into the f32 column path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype

        return NonHydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dv_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dw_dt=Field(
                data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_rad",
                dims=dims_w, units="m/s^2",
            ),
            dtheta_prime_dt=Field(
                data=dtheta_prime_dt, name="dtheta_prime_dt_rad",
                dims=dims_3d, units="K/s",
            ),
            drho_prime_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_rad",
                dims=dims_3d, units="kg/m^3/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_rad",
                dims=dims_2d, units="m^2/s^3",
            ),
            dtracers_dt=Field(
                data=jnp.zeros_like(state.tracers.data),
                name="dtracers_dt_rad",
                dims=dims_tr, units="1/s",
            ),
        )

    physics_fn.set_time = set_time
    physics_fn.set_T_sfc_override = set_T_sfc_override
    return physics_fn


# ===========================================================================
# Plane (doubly-periodic Cartesian CRM)
# ===========================================================================

def _mc3d_plane_heating(
    radiation_config: RadiationConfig,
    grid,
    terrain_metric,
    rrtmgp_solver,
    p_full_col,
    p_half_col,
    T_col,
    T_sfc_col,
    lat_col,
    q_v_col,
    q_cloud_col,
    q_ice_col,
    n_cloud_col,
    n_ice_col,
    insol_col,
    cos_sza_col,
    rho_total,
    ny: int,
    nx: int,
    nlev: int,
    day_of_year,
    seconds_of_day,
):
    """3D Monte-Carlo shortwave + gray longwave heating for the plane dycore.

    Returns ``dT/dt`` ``(ny, nx, nlev)`` [K/s] (top-down). Shortwave uses the 3D
    MC ray tracer fed by RRTMGP per-g-point optics (Phase 2b) when the solver is
    available, else a gray-shortwave optical field (Phase 2 fallback). Longwave
    reuses the gray two-stream column kernel so the scheme is physically complete
    (Phase 3 replaces LW with MC). Assumes flat plane terrain (1D z interfaces)
    and horizontally-uniform solar forcing, valid for idealized LES/CRM at
    constant lat0.
    """
    import jax
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.radiation import gray as _gray
    from legoesm.atmosphere.physics.radiation.mc3d import plane_adapter

    gray_cfg = radiation_config.gray
    mc_cfg = radiation_config.mc3d

    # 3D ray tracing needs an INSTANTANEOUS solar zenith for the beam slant.
    # The daily-mean / perpetual-equinox path returns cos_sza=None (no single
    # sun angle); defaulting to an overhead beam (mu0=1) would silently put SW
    # absorption too deep. Require diurnal_cycle=True or rce_fixed_cos_zenith.
    if cos_sza_col is None:
        raise ValueError(
            "scheme='mc3d' requires an instantaneous solar zenith angle: set "
            "RadiationConfig.diurnal_cycle=True (with set_time) or "
            "rce_fixed_cos_zenith. Daily-mean/perpetual-equinox insolation has "
            "no single beam direction for the 3D ray tracer."
        )
    mu0 = jnp.mean(cos_sza_col)

    z_half_td = terrain_metric.z_half_3d[0, 0, :]   # flat terrain -> 1D
    q_v_safe = jnp.clip(q_v_col, 0.0, None)

    # RRTMGP cloud kwargs (shared by SW + LW spectral optics; clear-sky when the
    # cloud scheme is 'none'). Only built when the RRTMGP solver is present.
    cloud_kwargs = {}
    if rrtmgp_solver is not None and radiation_config.cloud_scheme != "none":
        from legoesm.atmosphere.physics.clouds.cloud_fraction import (
            compute_cloud_properties,
        )
        from legoesm.atmosphere.physics.clouds.config import CloudConfig
        cloud_config = (
            radiation_config.cloud_config
            if radiation_config.cloud_config is not None
            else CloudConfig(scheme=radiation_config.cloud_scheme)
        )
        dp = p_half_col[:, 1:] - p_half_col[:, :-1]
        cloud_kwargs = compute_cloud_properties(
            T=T_col, p_full=p_full_col, q_v=q_v_safe, dp=dp,
            config=cloud_config, q_cloud=q_cloud_col, q_ice=q_ice_col,
            n_ice=n_ice_col, n_cloud=n_cloud_col,
        ).to_rrtmg_kwargs()

    # --- shortwave optical field ---
    # NOTE: the plane mc3d path is AEROSOL-FREE (no aerosol_optical_depth is
    # threaded into solve_columns here), targeting the LES/CRM regime. So the
    # scatter split is exactly gas-Rayleigh + cloud-Mie (no aerosol), matching
    # the oracle's 3-way reduced to 2-way. Aerosol-laden SW would need the
    # aerosol optical depth threaded AND a 3rd (aerosol-HG) scatter branch.
    if rrtmgp_solver is not None:
        # Phase 2b: RRTMGP per-g-point spectral optics (gas + cloud + Rayleigh).
        (tau, ssa, g, rayleigh_frac, r_eff_um,
         solar_normal) = rrtmgp_solver.solve_columns(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, q_v=q_v_safe, cos_zenith=cos_sza_col,
            sw_optical_field_only=True, **cloud_kwargs,
        )
        ngpt = tau.shape[0]
        tau_td = tau.reshape(ngpt, ny, nx, nlev)
        ssa_td = ssa.reshape(ngpt, ny, nx, nlev)
        g_td = g.reshape(ngpt, ny, nx, nlev)
        rayleigh_td = rayleigh_frac.reshape(ngpt, ny, nx, nlev)
        # Per-band vertical TOA flux = beam-normal * mu0.
        incident_flux = solar_normal * mu0
        # Mie cloud phase (microhh LUT) when enabled: select each g-point's band
        # slice via the RRTMGP g-point->band map, thread cloud r_eff [um].
        mie_lut_cdf = mie_lut_ang = band_of_gpt = r_eff_td = None
        if mc_cfg.use_mie:
            from legoesm.atmosphere.physics.radiation.mc3d import mie as _mie
            lut = _mie.load_mie_sampling_lut()
            # Pass the SMALL per-band LUT + the g-point->band map; the band slice
            # is gathered inside the g-point scan (no (ngpt, n_r, n_mie) copy).
            mie_lut_cdf = lut.phase_cdf                        # (n_band, n_mie)
            mie_lut_ang = lut.phase_cdf_angle                 # (n_band, n_r, n_mie)
            band_of_gpt = jnp.asarray(
                rrtmgp_solver.optics_lib.gas_optics_sw.g_point_to_bnd)
            r_eff_td = r_eff_um.reshape(ny, nx, nlev)
    else:
        # Phase 2 fallback: single-band gray shortwave optics (pure absorption).
        tau_td, ssa_td, g_td = plane_adapter.gray_sw_optical_field(
            p_half_col, gray_cfg, ny, nx)
        rayleigh_td = mie_lut_cdf = mie_lut_ang = band_of_gpt = r_eff_td = None
        incident_flux = jnp.mean(insol_col).reshape(1)

    # Fold the time (day-of-year + integer second-of-day) into the MC seed so
    # each timestep draws an INDEPENDENT photon realization. With a fixed seed
    # the same per-column speckle pattern repeats every step, so its Monte-Carlo
    # error never averages out over a time integration (it is identical, not
    # independent, each step). Folding day then second decorrelates across both
    # days and within a day (assumes dt >= 1 s; finer steps in the same integer
    # second share a realization).
    _doy_key = jnp.asarray(day_of_year).astype(jnp.int32)
    _sod_key = jnp.asarray(seconds_of_day).astype(jnp.int32)
    _t_base = jax.random.fold_in(
        jax.random.fold_in(jax.random.PRNGKey(int(mc_cfg.seed)), _doy_key),
        _sod_key)
    key = _t_base
    dT_dt_sw, _sfc_sw, _tod = plane_adapter.compute_plane_sw_heating(
        tau_td, ssa_td, g_td, incident_flux, grid, z_half_td, rho_total,
        mu0=mu0, albedo=gray_cfg.sfc_albedo, config=mc_cfg, key=key,
        rayleigh_frac_td=rayleigh_td, mie_lut_cdf=mie_lut_cdf,
        mie_lut_ang=mie_lut_ang, band_of_gpt=band_of_gpt, r_eff_td=r_eff_td)

    # --- longwave: 3D-MC thermal emission ---
    key_lw = jax.random.fold_in(
        jax.random.fold_in(
            jax.random.PRNGKey(int(mc_cfg.seed) + 1), _doy_key), _sod_key)
    if rrtmgp_solver is not None:
        # Phase 3b: RRTMGP per-g-point spectral LW optics + Planck (cloud-aware).
        abs_od, planck, planck_bot, planck_top, planck_sfc = (
            rrtmgp_solver.solve_columns(
                T=T_col, p_full=p_full_col, p_half=p_half_col,
                sfc_temperature=T_sfc_col, q_v=q_v_safe, cos_zenith=cos_sza_col,
                lw_optical_field_only=True, **cloud_kwargs,
            ))
        nglw = abs_od.shape[0]
        dT_dt_lw, _sfc_lw, _olr = plane_adapter.compute_plane_lw_heating_spectral(
            abs_od.reshape(nglw, ny, nx, nlev),
            planck.reshape(nglw, ny, nx, nlev),
            planck_sfc.reshape(nglw, ny, nx),
            grid, z_half_td, rho_total,
            emissivity=radiation_config.rrtmgp.sfc_emissivity,
            config=mc_cfg, key=key_lw,
            planck_bottom_td=planck_bot.reshape(nglw, ny, nx, nlev),
            planck_top_td=planck_top.reshape(nglw, ny, nx, nlev))
    else:
        # Phase 2 fallback: gray broadband LW (k_abs=dtau_lw/dz, B=sigma T^4/pi,
        # blackbody surface). Reduces to the column gray result for uniform
        # columns; captures 3D emission/absorption for heterogeneous fields.
        p_s_col = p_half_col[:, -1]
        dtau_lw = _gray._compute_lw_optical_depth(
            p_half_col, p_s_col, lat_col, q_v_col, gray_cfg)
        dz_layer = z_half_td[:-1] - z_half_td[1:]              # top-down: >0
        k_abs_col = dtau_lw / jnp.clip(dz_layer, 1.0, None)[None, :]
        T_td = T_col.reshape(ny, nx, nlev)
        dT_dt_lw, _sfc_lw, _olr = plane_adapter.compute_plane_lw_heating(
            k_abs_col.reshape(ny, nx, nlev),
            constants.sigma_sb * T_td ** 4 / jnp.pi,
            (constants.sigma_sb * T_sfc_col ** 4 / jnp.pi).reshape(ny, nx),
            grid, z_half_td, rho_total,
            emissivity=1.0, config=mc_cfg, key=key_lw)

    return dT_dt_sw + dT_dt_lw


def _make_plane_radiation(
    radiation_config: RadiationConfig,
    rrtmgp_solver=None,
    ml_ozone_coefs=None,
) -> Callable:
    """Create radiation physics_fn for the plane CompressibleEulerPlaneModel.

    Mirrors :func:`_make_nonhydrostatic_radiation` but for
    ``PlaneNonHydrostaticState`` (shape ``(ny, nx, nlev)``) and
    ``PlaneNonHydrostaticGrid`` (``grid_lat``/``grid_lon`` return a
    constant tropical lat/lon over the plane).

    Signature: ``(state, grid, height_coord, terrain_metric) ->
    PlaneNonHydrostaticTendencies``.
    """
    _time, set_time = _make_time_state()
    _T_sfc_override_cell, set_T_sfc_override = _make_T_sfc_override_cell()

    def physics_fn(
        state: PlaneNonHydrostaticState,
        grid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
    ) -> PlaneNonHydrostaticTendencies:
        theta_p = state.theta_prime.data   # (ny, nx, nlev)
        rho_p = state.rho_prime.data       # (ny, nx, nlev)
        theta_0 = height_coord.theta_ref   # (nlev,)
        rho_0 = height_coord.rho_ref       # (nlev,)

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p, rho_0 + rho_p,
        )
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape          # (ny, nx, nlev)
        shape_w = state.w.data.shape      # (ny, nx, nlev+1)
        shape_2d = state.phis.data.shape  # (ny, nx)
        ny, nx = shape_2d
        ncol = ny * nx

        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p, rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )
        T_sfc = _apply_T_sfc_override(T[..., -1], _T_sfc_override_cell[0])

        # Plane lat/lon: PlaneGrid.grid_lat returns constant lat0 over
        # (ny, nx), already in radians (deg2rad applied in property).
        lat, lon = _get_grid_lat_lon(grid, shape_2d)
        insol, cos_sza, f_day, eccf = _compute_insolation(
            lat, radiation_config, lon=lon,
            day_of_year=_time["day_of_year"],
            seconds_of_day=_time["seconds_of_day"],
        )

        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        lon_col = lon.reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = (
            cos_sza.reshape(ncol) if cos_sza is not None else None
        )
        f_day_col = f_day.reshape(ncol) if f_day is not None else None

        # NH plane state stores water vapor in tracer slot 0; q_c at 1,
        # q_i at 3 — mirroring the cubed-sphere NH tracer layout used
        # by ``_make_nonhydrostatic_radiation``.
        n_tracers = state.tracers.data.shape[-1]
        if n_tracers > 0:
            q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
        else:
            q_v = jnp.zeros_like(T)
        q_v_col = q_v.reshape(ncol, nlev)

        q_cloud_col = None
        q_ice_col = None
        if n_tracers > 1:
            q_cloud_col = jnp.clip(
                state.tracers.data[..., 1], 0.0, None,
            ).reshape(ncol, nlev)
        if n_tracers > 3:
            q_ice_col = jnp.clip(
                state.tracers.data[..., 3], 0.0, None,
            ).reshape(ncol, nlev)
        # Double-moment NUMBER columns for the M2005 PSD effective radii, same
        # layout/units/guard as the cubed-sphere NH path: N_c per-VOLUME [#/m³]
        # at slot 6, N_i per-MASS [#/kg] at slot 8 — the canonical 9-slot layout
        # (q_v,q_c,q_r,q_i,q_s,q_g,N_c,N_r,N_i) shared by ALL double-moment
        # schemes (morrison, seifert_beheng, thompson, p3, ml_emulator). The
        # ``> 8`` guard keeps single-moment layouts (kessler/sundqvist, 3 slots)
        # on the constant r_eff. Without this, plane-CRM RRTMGP used a fixed
        # r_eff regardless of droplet number — unfaithful to SAM's PSD reffc/EFFI.
        n_cloud_col = None
        n_ice_col = None
        if n_tracers > 8:
            n_cloud_col = jnp.clip(
                state.tracers.data[..., 6], 0.0, None,
            ).reshape(ncol, nlev)
            n_ice_col = jnp.clip(
                state.tracers.data[..., 8], 0.0, None,
            ).reshape(ncol, nlev)

        if radiation_config.scheme == "mc3d":
            # 3D Monte-Carlo shortwave + gray longwave (plane LES/CRM only).
            dT_dt = _mc3d_plane_heating(
                radiation_config, grid, terrain_metric, rrtmgp_solver,
                p_full_col, p_half_col, T_col, T_sfc_col, lat_col, q_v_col,
                q_cloud_col, q_ice_col, n_cloud_col, n_ice_col,
                insol_col, cos_sza_col, rho_total, ny, nx, nlev,
                _time["day_of_year"], _time["seconds_of_day"],
            )
        else:
            rad_out = _call_radiation_backend(
                radiation_config=radiation_config,
                eccf=eccf,
                T=T_col, p_full=p_full_col, p_half=p_half_col,
                sfc_temperature=T_sfc_col, lat=lat_col, q_v=q_v_col,
                insolation=insol_col, cos_sza=cos_sza_col,
                q_cloud=q_cloud_col, q_ice=q_ice_col,
                n_cloud=n_cloud_col, n_ice=n_ice_col, f_day=f_day_col,
                rrtmgp_solver=rrtmgp_solver, lon=lon_col,
                ml_ozone_coefs=ml_ozone_coefs,
            )
            dT_dt = rad_out.heating_rate.reshape(shape_3d)

        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        # Match the PlaneNonHydrostaticState field convention
        # (("y","x","z"), ("y","x","z_half"), ...) so summed tendencies
        # across surface_flux + radiation + microphysics carry
        # consistent metadata for the dycore composer.
        dims_3d = ("y", "x", "z")
        dims_w = ("y", "x", "z_half")
        dims_2d = ("y", "x")
        dims_tr = ("y", "x", "z", "tracer")
        _sd = T.dtype
        _pd = state.phis.data.dtype

        return PlaneNonHydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_sd), name="du_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dv_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_sd), name="dv_dt_rad",
                dims=dims_3d, units="m/s^2",
            ),
            dw_dt=Field(
                data=jnp.zeros(shape_w, dtype=_sd), name="dw_dt_rad",
                dims=dims_w, units="m/s^2",
            ),
            dtheta_prime_dt=Field(
                data=dtheta_prime_dt, name="dtheta_prime_dt_rad",
                dims=dims_3d, units="K/s",
            ),
            drho_prime_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_sd),
                name="drho_prime_dt_rad", dims=dims_3d, units="kg/m^3/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_pd), name="dphis_dt_rad",
                dims=dims_2d, units="m^2/s^3",
            ),
            dtracers_dt=Field(
                data=jnp.zeros_like(state.tracers.data),
                name="dtracers_dt_rad", dims=dims_tr, units="1/s",
            ),
        )

    physics_fn.set_time = set_time
    physics_fn.set_T_sfc_override = set_T_sfc_override
    return physics_fn


# ===========================================================================
# MPAS Voronoi non-hydrostatic compressible Euler
# ===========================================================================

def _make_mpas_nh_radiation(
    radiation_config: RadiationConfig,
    rrtmgp_solver=None,
    ml_ozone_coefs=None,
) -> Callable:
    """Create radiation physics_fn for the MPAS NH dycore.

    Mirrors :func:`_make_nonhydrostatic_radiation` but for
    ``MPASNonHydrostaticState`` (cell-centred quantities on Voronoi
    cells, shape ``(nCells, nlev)``; ``u`` on TRiSK edges shape
    ``(nEdges, nlev)``; ``w`` at half levels shape
    ``(nCells, nlev+1)``).

    Signature: ``(state, mesh, height_coord, terrain_metric) ->
    MPASNonHydrostaticTendencies``.
    """
    _time, set_time = _make_time_state()
    _T_sfc_override_cell, set_T_sfc_override = _make_T_sfc_override_cell()

    def physics_fn(
        state: MPASNonHydrostaticState,
        mesh,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
    ) -> MPASNonHydrostaticTendencies:
        theta_p = state.theta_prime.data   # (nCells, nlev)
        rho_p = state.rho_prime.data       # (nCells, nlev)
        theta_0 = height_coord.theta_ref   # (nlev,)
        rho_0 = height_coord.rho_ref       # (nlev,)

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p, rho_0 + rho_p,
        )
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_2d = (mesh.nCells,)
        shape_cell_3d = (mesh.nCells, nlev)
        shape_edge_3d = state.u.data.shape          # (nEdges, nlev)
        shape_w = state.w.data.shape                # (nCells, nlev+1)

        # Half-level pressure from evolving column state (matches the
        # cubed-sphere NH path).
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p, rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )
        T_sfc = _apply_T_sfc_override(T[..., -1], _T_sfc_override_cell[0])

        # MPAS lat/lon at cells handled by `_get_grid_lat_lon` via the
        # `hasattr(grid_or_mesh, 'latCell')` branch.
        lat, lon = _get_grid_lat_lon(mesh, shape_2d)
        insol, cos_sza, f_day, eccf = _compute_insolation(
            lat, radiation_config, lon=lon,
            day_of_year=_time["day_of_year"],
            seconds_of_day=_time["seconds_of_day"],
        )

        # Columns are already (ncol, nlev) — no reshape needed for the
        # cell-centred quantities.
        T_col = T
        p_full_col = p
        p_half_col = p_half
        T_sfc_col = T_sfc
        lat_col = lat
        lon_col = lon
        insol_col = insol
        cos_sza_col = cos_sza
        f_day_col = f_day

        # Tracer slot layout (matches the cubed-sphere NH variant):
        #   [0] q_v   [1] q_c   [2] q_r   [3] q_i
        # Codex review 2026-05-24 iter-3: cloud-aware backends MUST
        # have access to q_c (slot 1) AND q_i (slot 3); otherwise a
        # caller wiring ``RRTMGPConfig(include_clouds=True)`` with
        # fewer slots would get a silent CLEAR-SKY run instead of an
        # error. Hard-fail at call time before we lose the
        # information.
        n_tracers = state.tracers.data.shape[-1]
        _wants_clouds = (
            radiation_config.scheme == "rrtmgp"
            and getattr(radiation_config.rrtmgp, "include_clouds", False)
        )
        if _wants_clouds and n_tracers < 4:
            raise ValueError(
                f"Cloud-aware radiation (scheme='rrtmgp', "
                f"include_clouds=True) requires state.tracers with "
                f"at least 4 slots (q_v, q_c, q_r, q_i); MPAS NH "
                f"state carries {n_tracers}. Allocate the state with "
                f">= 4 tracers or disable cloud optics."
            )
        if n_tracers > 0:
            q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
        else:
            q_v = jnp.zeros_like(T)
        q_v_col = q_v

        q_cloud_col = None
        q_ice_col = None
        if n_tracers > 1:
            q_cloud_col = jnp.clip(
                state.tracers.data[..., 1], 0.0, None,
            )
        if n_tracers > 3:
            q_ice_col = jnp.clip(
                state.tracers.data[..., 3], 0.0, None,
            )
        # Double-moment NUMBER columns for the M2005 PSD effective radii (same
        # slot layout/units/guard as the cubed-sphere NH + plane paths): N_c
        # per-VOLUME [#/m³] slot 6, N_i per-MASS [#/kg] slot 8 — the canonical
        # 9-slot layout shared by all double-moment schemes (morrison,
        # seifert_beheng, thompson, p3, ml_emulator); ``> 8`` guard keeps
        # single-moment schemes (kessler/sundqvist) on the constant r_eff.
        # MPAS columns are already (nCells, nlev) — no reshape needed.
        n_cloud_col = None
        n_ice_col = None
        if n_tracers > 8:
            n_cloud_col = jnp.clip(state.tracers.data[..., 6], 0.0, None)
            n_ice_col = jnp.clip(state.tracers.data[..., 8], 0.0, None)

        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            eccf=eccf,
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col, q_v=q_v_col,
            insolation=insol_col, cos_sza=cos_sza_col,
            q_cloud=q_cloud_col, q_ice=q_ice_col,
            n_cloud=n_cloud_col, n_ice=n_ice_col, f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver, lon=lon_col,
            ml_ozone_coefs=ml_ozone_coefs,
        )

        dT_dt = rad_out.heating_rate          # (nCells, nlev)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dims_cell = ("nCells", "nlev")
        dims_edge = ("nEdges", "nlev")
        dims_w = ("nCells", "nlev_half")
        dims_2d = ("nCells",)
        dims_tr = ("nCells", "nlev", "tracer")
        _sd = T.dtype
        _pd = state.phis.data.dtype

        return MPASNonHydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(shape_edge_3d, dtype=_sd),
                name="du_dt_rad", dims=dims_edge, units="m/s^2",
            ),
            dw_dt=Field(
                data=jnp.zeros(shape_w, dtype=_sd),
                name="dw_dt_rad", dims=dims_w, units="m/s^2",
            ),
            dtheta_prime_dt=Field(
                data=dtheta_prime_dt, name="dtheta_prime_dt_rad",
                dims=dims_cell, units="K/s",
            ),
            drho_prime_dt=Field(
                data=jnp.zeros(shape_cell_3d, dtype=_sd),
                name="drho_prime_dt_rad", dims=dims_cell, units="kg/m^3/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_pd),
                name="dphis_dt_rad", dims=dims_2d, units="m^2/s^3",
            ),
            dtracers_dt=Field(
                data=jnp.zeros_like(state.tracers.data),
                name="dtracers_dt_rad", dims=dims_tr, units="1/s",
            ),
        )

    physics_fn.set_time = set_time
    physics_fn.set_T_sfc_override = set_T_sfc_override
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_radiation(
    radiation_config: RadiationConfig,
    rrtmgp_solver=None,
    ml_ozone_coefs=None,
    sfc_albedo_override: jnp.ndarray | float | None = None,
    sfc_emissivity_override: jnp.ndarray | float | None = None,
) -> Callable:
    """Create radiation physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord) -> SpectralHydrostaticState

    Transforms spectral state to Gaussian grid, computes radiation,
    then transforms temperature tendency back to spectral space.

    Memory note (RRTMGP path)
    -------------------------
    ``two_stream.solve_lw`` / ``solve_sw`` iterate over 16 LW + 14 SW
    bands, each accumulating per-layer flux intermediates that would
    otherwise be stored for the autodiff backward pass.  Under
    ``eqx.filter_value_and_grad`` over a 48-step daily rollout this
    materializes hundreds of GiB at T21 L8 -- well past a single
    consumer GPU (RTX 5090, 24 GiB).  We wrap the inner
    ``_physics_fn_core`` in :func:`jax.checkpoint` with
    ``nothing_saveable`` so the entire RRTMGP solve is recomputed
    from scratch during backward.  The outer
    ``jax.checkpoint(step_fn, prevent_cse=True)`` in
    ``spectral_rollout`` already recomputes per-step activations; the
    nested radiation checkpoint trades ~2x extra forward compute for
    the activation-storage relief that lets the production AIMIP
    run finish on a single GPU.
    """
    _time, set_time = _make_time_state()
    _T_sfc_override_cell, set_T_sfc_override = _make_T_sfc_override_cell()

    def _physics_fn_core(
        state, grid, sigma_coord, grid_fields=None,
        sim_time_seconds=0.0, forcing=None,
    ):
        # 1. Transform spectral state to grid space
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        T = fields['T']         # (n_lat, n_lon, nlev)
        p_s = fields['p_s']     # (n_lat, n_lon)
        lat = grid.lat          # (n_lat,)

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        # Pressure at full and half levels
        p_full = sigma_coord.pressure_at_full(p_s)
        p_half = sigma_coord.pressure_at_half(p_s)

        # Surface temperature = lowest level (default), overridable by a
        # per-step TRACED ``forcing["T_sfc"]`` (the AMIP path — prescribed
        # SST/SIC blend, time-varying without retrace) or the static
        # ``set_T_sfc_override`` closure.  Same precedence as the
        # hydrostatic/MPAS radiation factory.
        _ovr = None
        if forcing is not None and forcing.get("T_sfc") is not None:
            _ovr = forcing["T_sfc"]
        else:
            _ovr = _T_sfc_override_cell[0]
        T_sfc = _apply_T_sfc_override(T[..., -1], _ovr)  # (n_lat, n_lon)

        # External CMIP6 forcing via the same traced ``forcing`` dict
        # (see _make_hydrostatic_radiation): o3_vmr / aerosol_od are
        # (ncol, nlev) columns, ghg_vmr is a dict of traced scalars.
        _o3_ext = forcing.get("o3_vmr") if forcing is not None else None
        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
        _aer_lw_ext = (
            forcing.get("aerosol_lw_od") if forcing is not None else None
        )
        _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None

        # Surface albedo / emissivity overrides: a per-step TRACED
        # ``forcing["sfc_albedo"]`` wins over the static build-time override
        # (``sfc_albedo_override`` closed over from ``make_radiation_physics`` —
        # e.g. AIMIP's trained spatial field). Routing them here (NOT into
        # ``RRTMGPConfig.sfc_*``) keeps the trained value off RRTMGP's Python
        # solver-cache key. Same precedence pattern as T_sfc / o3 / aerosol.
        _alb_ovr = forcing.get("sfc_albedo") if forcing is not None else None
        if _alb_ovr is None:
            _alb_ovr = sfc_albedo_override
        _emis_ovr = forcing.get("sfc_emissivity") if forcing is not None else None
        if _emis_ovr is None:
            _emis_ovr = sfc_emissivity_override

        # Effective time-of-day for the diurnal cycle.  ``_time`` holds
        # the *initial* day_of_year + seconds_of_day captured at module
        # import (or set via ``set_time`` between epochs); the scan
        # body in :func:`spectral_rollout` passes the current
        # in-rollout elapsed time as ``sim_time_seconds`` so each
        # radiation evaluation sees the correct cos(SZA) at its hour
        # of day.  Without this thread, all rad calls within a
        # rollout would share the static initial-IC time and the
        # diurnal pattern would be frozen (verified 2026-05-22 --
        # the v10 production setup had this bug).
        secs_init = _time["seconds_of_day"]
        day_init = _time["day_of_year"]
        total_secs = secs_init + sim_time_seconds
        secs_eff = jnp.mod(total_secs, 86400.0)
        day_eff = day_init + jnp.floor_divide(total_secs, 86400.0)
        # Per-step TRACED calendar time from the forcing dict wins over
        # the static ``set_time`` closure + ``sim_time_seconds`` thread —
        # the closure cell is baked at trace time inside the JIT'd
        # spectral step, so a production AMIP loop must pass time as a
        # traced value (same rationale as the hydrostatic/MPAS factory).
        if forcing is not None and forcing.get("day_of_year") is not None:
            day_eff = forcing["day_of_year"]
        if forcing is not None and forcing.get("seconds_of_day") is not None:
            secs_eff = forcing["seconds_of_day"]

        # Insolation (with diurnal cycle support).
        # For diurnal cycle we need 2-D lat/lon; otherwise lat is 1-D and
        # the result is broadcast to (n_lat, n_lon).
        if radiation_config.diurnal_cycle:
            lat_2d = jnp.broadcast_to(lat[:, None], (n_lat, n_lon))
            lon_2d = jnp.broadcast_to(grid.lon[None, :], (n_lat, n_lon))
            insol, cos_sza, f_day, eccf = _compute_insolation(
                lat_2d, radiation_config,
                lon=lon_2d,
                day_of_year=day_eff,
                seconds_of_day=secs_eff,
            )
        else:
            insol_1d, _, f_day_1d, eccf = _compute_insolation(
                lat, radiation_config,
                day_of_year=day_eff,
                seconds_of_day=secs_eff,
            )
            insol = jnp.broadcast_to(insol_1d[:, None], (n_lat, n_lon))
            cos_sza = None
            f_day = jnp.broadcast_to(f_day_1d[:, None], (n_lat, n_lon)) if f_day_1d is not None else None

        # Reshape to columns: (n_lat, n_lon, ...) -> (ncol, ...)
        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = jnp.broadcast_to(lat[:, None], (n_lat, n_lon)).reshape(ncol)
        lon_col = jnp.broadcast_to(grid.lon[None, :], (n_lat, n_lon)).reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = cos_sza.reshape(ncol) if cos_sza is not None else None

        q_v_col, q_cloud_col, q_ice_col, n_cloud_col, n_ice_col = (
            _extract_tracer_columns(state, ncol, nlev)
        )

        f_day_col = f_day.reshape(ncol) if f_day is not None else None

        # Flatten any 2-D (n_lat, n_lon) surface override to (ncol,); scalars
        # and (ncol,) arrays pass through (the radiation backend / RRTMGP's
        # _resolve_surface_field broadcasts over the column axis).
        def _to_col(x):
            if x is not None and hasattr(x, "ndim") and x.ndim >= 2:
                return x.reshape(ncol)
            return x
        _alb_col = _to_col(_alb_ovr)
        _emis_col = _to_col(_emis_ovr)

        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            eccf=eccf,
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            lat=lat_col,
            q_v=q_v_col,
            insolation=insol_col,
            cos_sza=cos_sza_col,
            sfc_albedo_override=_alb_col,
            sfc_emissivity_override=_emis_col,
            q_cloud=q_cloud_col,
            q_ice=q_ice_col,
            n_cloud=n_cloud_col,
            n_ice=n_ice_col,
            f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver,
            lon=lon_col,
            ml_ozone_coefs=ml_ozone_coefs,
            o3_vmr_override=_o3_ext,
            aerosol_od=_aer_ext,
            aerosol_lw_od=_aer_lw_ext,
            ghg_vmr_override=_ghg_ext,
        )

        # Reshape heating rate back to (n_lat, n_lon, nlev)
        dT_dt = rad_out.heating_rate.reshape(n_lat, n_lon, nlev)

        # 3. Transform T tendency to spectral space
        dT_hat = sh_analysis_3d(grid, dT_dt)

        # No wind or surface pressure tendencies from radiation
        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        return SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=zero_3d),
            div_hat=state.div_hat.replace(data=zero_3d),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        )

    # Wrap the inner radiation compute in ``jax.checkpoint`` only for
    # the RRTMGP path -- gray two-stream is cheap enough that the
    # extra recompute on backward is wasted.  ``static_argnums`` skips
    # ``grid`` and ``sigma_coord`` which are static (non-array)
    # NamedTuples; ``grid_fields`` may be a pytree of grid-space
    # arrays and flows through normally.
    if radiation_config.scheme == "rrtmgp":
        _physics_fn_ckpt = jax.checkpoint(
            _physics_fn_core,
            static_argnums=(1, 2),
            prevent_cse=True,
        )

        def physics_fn(
            state, grid, sigma_coord, grid_fields=None,
            sim_time_seconds=0.0, forcing=None,
        ):
            return _physics_fn_ckpt(
                state, grid, sigma_coord, grid_fields, sim_time_seconds,
                forcing,
            )
    else:
        physics_fn = _physics_fn_core

    physics_fn.set_time = set_time
    physics_fn.set_T_sfc_override = set_T_sfc_override
    # Marker: consumes the per-step traced ``forcing`` dict (T_sfc /
    # o3_vmr / aerosol_od / ghg_vmr) — the spectral combined dispatcher
    # forwards ``forcing`` only to fns that advertise it.
    physics_fn._wants_forcing = True
    return physics_fn


# _make_mpas_radiation is defined as an alias above (= _make_hydrostatic_radiation)


# Public promotions (CLAUDE.md cross-module private-import ratchet):
# these symbols are imported by sibling modules; expose a public alias
# so importers use the sanctioned public name (definitions keep the
# original underscore name for in-module callers).
compute_ozone_vmr = _compute_ozone_vmr
