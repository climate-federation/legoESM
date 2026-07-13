"""Main surface coupler: factory and step function.

The coupler steps all surface tiles, blends their responses, and
accumulates fluxes for asynchronous coupling. It never accesses
full atmospheric state — only AtmToSurface.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.precision import get_policy
from legoesm.core.bulk_flux import (
    simple_bulk_fluxes, compute_most_fluxes, apply_gustiness,
    ocean_surface_q_sat,
)
from legoesm.land.multilayer_land import init_multilayer_land_state
from legoesm.land.surface_params import reshape_params
from legoesm.surface_albedo import ocean_albedo as compute_ocean_albedo
from legoesm.core.field import Field
from legoesm.coupler.accumulator import (
    FluxAccumulator,
    accumulate,
    accumulator_from_flux,
    mean_accumulator,
    reset_accumulator,
)
from legoesm.coupler.config import CouplerConfig, TileConfig
from legoesm.core.coupling_fields import (
    AtmToSurface,
    SurfaceToAtm,
    TileResponse,
)

# LY09 sea-surface saturation reduction for typical seawater salinity (~35 PSU).
# The saturation vapor pressure over saline water is ~2 % lower than over
# fresh water; q_sat at the air-sea interface is correspondingly reduced.
# Required by OMIP-2 protocol (Griffies 2016 §2.2 → Large & Yeager 2009 §3).
_Q_SAT_SALINE_FACTOR = 0.98
from legoesm.coupler.lake import LakeConfig, LakeState, step_lake
from legoesm.coupler.tile_fractions import (
    blend_tiles,
    compute_tile_fractions,
)
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.state import SeaIceState
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.slab_land import step_land
from legoesm.land.multilayer_land import step_multilayer_land
from legoesm.land.state import LandState


class SurfaceState(NamedTuple):
    """Combined surface state for all tiles."""
    land: LandState
    ice: SeaIceState
    lake: LakeState
    accumulator: FluxAccumulator
    carbon: CarbonState | None = None


class _TileContext(NamedTuple):
    """Per-step inputs shared by every surface-tile step (see :func:`make_coupler`).

    The coupler iterates a uniform catalog of tile-step closures; each closure
    receives this context and returns ``(TileResponse, state_updates)``, so the
    surface tiles are a data-driven set rather than hardcoded inline pairs.
    """

    sfc_state: SurfaceState
    atm_forcing: AtmToSurface
    doy: float
    dt: float
    ocean_sst: jnp.ndarray
    ocean_u: jnp.ndarray
    ocean_v: jnp.ndarray
    land_params: Any  # materialized LandSurfaceParams, or None


def _validate_coupler_config(config: CouplerConfig) -> None:
    """Fail fast on clearly invalid coupler parameters."""
    if config.coupling_dt <= 0.0:
        raise ValueError(f"coupling_dt must be > 0, got {config.coupling_dt!r}")
    if config.U_min < 0.0:
        raise ValueError(f"U_min must be >= 0, got {config.U_min!r}")
    if not 0.0 <= config.ocean_albedo <= 1.0:
        raise ValueError(
            "ocean_albedo must be in [0, 1], got "
            f"{config.ocean_albedo!r}",
        )
    if not 0.0 <= config.ocean_emissivity <= 1.0:
        raise ValueError(
            "ocean_emissivity must be in [0, 1], got "
            f"{config.ocean_emissivity!r}",
        )
    if config.ocean_z0 <= 0.0:
        raise ValueError(f"ocean_z0 must be > 0, got {config.ocean_z0!r}")
    if config.Cd_ocean < 0.0:
        raise ValueError(f"Cd_ocean must be >= 0, got {config.Cd_ocean!r}")
    if config.Ch_ocean < 0.0:
        raise ValueError(f"Ch_ocean must be >= 0, got {config.Ch_ocean!r}")


def _apply_carbon_override(cold_start: CarbonState,
                           override: CarbonState,
                           carbon_shape: tuple[int, ...]) -> CarbonState:
    """Replace a cold-start :class:`CarbonState`'s pools with a seeded IC.

    Validates each of the eight pools' shape against the cold-start reference and
    casts to its dtype, so the seeded carry pytree is byte-for-byte structurally
    identical to the cold-start one (same shapes + dtypes) whether or not a carbon
    IC is supplied -- only the values differ.  Any per-pool shape / type mismatch
    RAISES (never a silent broadcast).
    """
    if type(override) is not type(cold_start):
        raise ValueError(
            f"carbon_override must be a {type(cold_start).__name__}; got "
            f"{type(override).__name__}.")
    fields = {}
    for f in cold_start._fields:
        o = jnp.asarray(getattr(override, f))
        ref = getattr(cold_start, f)
        if o.shape != ref.shape:
            raise ValueError(
                f"carbon_override.{f} has shape {o.shape}; expected {ref.shape} "
                f"(the run's carbon column shape {carbon_shape}).")
        fields[f] = o.astype(ref.dtype)
    return cold_start._replace(**fields)


def init_surface_state(
    shape: tuple[int, ...],
    T_soil_init: float = 280.0,  # coeff-ok: initial condition [K]
    W_bucket_init: float = 75.0,  # coeff-ok: initial condition [kg/m^2]
    T_epi_init: float = 285.0,  # coeff-ok: initial condition [K]
    T_hypo_init: float = 278.0,  # coeff-ok: initial condition [K]
    T_ice_init: float = 260.0,  # coeff-ok: initial condition [K]
    land_config: LandConfig | None = None,
    carbon_override: CarbonState | None = None,
) -> SurfaceState:
    """Initialize all surface tile states.

    Parameters
    ----------
    shape : tuple
        Spatial shape, typically (6, n, n).
    land_config : LandConfig, optional
        If provided and ``land_config.carbon.scheme == "differland"``,
        initialises prognostic carbon pools.
    carbon_override : CarbonState, optional
        A spun-up per-cell carbon IC (e.g. the finidat ``global_carbon_ic.npz``
        loaded by
        :func:`legoesm.land.carbon.global_init.load_finidat_carbon_ic`) that SEEDS
        the prognostic pools INSTEAD of the cold-start :func:`init_carbon_state`
        defaults, so a coupled run starts carbon at its mapped equilibrium.  Its
        per-pool shape is validated against the run's carbon-column shape and cast
        to the cold-start dtype (a mismatch raises).  ``None`` (default) -> the
        cold-start is byte-identical.  Only honoured when the carbon cycle is
        active (``differland``); supplying it with carbon off raises.
    """
    dims_2d = ("face", "x", "y")
    _sd = get_policy().storage

    # ``T_soil_init`` may be a scalar (uniform; legacy default) OR a spatial
    # array of shape ``shape`` (a warm start, e.g. the lat-varying near-surface
    # air temperature) — removes the artificial tropical cold-soil spin-up of
    # the uniform 280 K default.
    _Tsi = jnp.asarray(T_soil_init)
    if isinstance(land_config, MultiLayerLandConfig):
        # For multi-layer land, ncol = product of spatial dims
        ncol = math.prod(shape)
        T_init_col = T_soil_init if _Tsi.ndim == 0 else _Tsi.reshape(ncol)
        land = init_multilayer_land_state(
            ncol, land_config, T_init=T_init_col,
        )
    else:
        T_soil_data = (jnp.full(shape, T_soil_init, dtype=_sd) if _Tsi.ndim == 0
                       else jnp.broadcast_to(_Tsi.astype(_sd), shape))
        land = LandState(
            T_soil=Field(data=T_soil_data,
                         name="T_soil", dims=dims_2d, units="K"),
            W_bucket=Field(data=jnp.full(shape, W_bucket_init, dtype=_sd),
                           name="W_bucket", dims=dims_2d, units="kg/m2"),
            snow_depth=Field(data=jnp.zeros(shape, dtype=_sd),
                             name="snow_depth", dims=dims_2d, units="kg/m2"),
            snow_age=Field(data=jnp.zeros(shape, dtype=_sd),
                           name="snow_age", dims=dims_2d, units="s"),
            # Initialise runoff to zeros so the pytree shape is
            # invariant across timesteps (slab_land sets it to a
            # populated array after every step; matches LakeState.Q_freeze
            # convention).  Audit F13.
            runoff=jnp.zeros(shape, dtype=_sd),
        )

    ice = SeaIceState(
        h_ice=Field(data=jnp.zeros(shape, dtype=_sd),
                    name="h_ice", dims=dims_2d, units="m"),
        T_ice=Field(data=jnp.full(shape, T_ice_init, dtype=_sd),
                    name="T_ice", dims=dims_2d, units="K"),
        concentration=Field(data=jnp.zeros(shape, dtype=_sd),
                           name="ice_concentration", dims=dims_2d, units="1"),
    )

    # Pin lake temperatures to the same storage precision as the rest
    # of the coupler state (sea-ice / land use ``_sd`` above) so the
    # lake fields don't inadvertently default to f64 under x64 mode.
    lake = LakeState(
        T_epi=Field(data=jnp.full(shape, T_epi_init, dtype=_sd),
                    name="T_epi", dims=dims_2d, units="K"),
        T_hypo=Field(data=jnp.full(shape, T_hypo_init, dtype=_sd),
                     name="T_hypo", dims=dims_2d, units="K"),
        # Initialise Q_freeze to zeros so the pytree shape is
        # invariant across timesteps (two_layer_lake populates this
        # at every step).  Audit F14.
        Q_freeze=Field(data=jnp.zeros(shape, dtype=_sd),
                       name="Q_freeze", dims=dims_2d, units="W/m2"),
    )

    # Pin the accumulator leaves to the SAME storage precision as the rest of
    # SurfaceState (``_sd`` above); without an explicit dtype the zeros follow
    # the global x64 default, so under JAX_ENABLE_X64 the accumulator widens to
    # f64 while land/ice/lake/carbon stay at the storage policy dtype -- a mixed
    # pytree that triggers lax.scan carry-dtype warnings / promotion.
    acc = reset_accumulator(shape, dtype=_sd)

    # Carbon pools (only for differland scheme)
    carbon = None
    if land_config is not None and land_config.carbon.scheme == "differland":
        # Multi-layer land uses columnar (ncol,) shape; slab uses spatial shape
        if isinstance(land_config, MultiLayerLandConfig):
            carbon_shape = (math.prod(shape),)
        else:
            carbon_shape = shape
        carbon = init_carbon_state(carbon_shape, land_config.carbon)
        if carbon_override is not None:
            # Seed the prognostic pools from a spun-up carbon IC (the finidat)
            # instead of the cold-start defaults, so a coupled run starts carbon
            # at its mapped equilibrium.  Shape-validated + dtype-matched to the
            # cold-start reference (None => byte-identical cold-start).
            carbon = _apply_carbon_override(carbon, carbon_override, carbon_shape)
    elif carbon_override is not None:
        raise ValueError(
            "carbon_override was supplied but the land carbon cycle is inactive "
            "(land_config.carbon.scheme != 'differland'); a seeded carbon IC "
            "needs the differland scheme on the land config.")

    return SurfaceState(land=land, ice=ice, lake=lake, accumulator=acc,
                        carbon=carbon)


def ocean_tile_response(
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: CouplerConfig,
) -> TileResponse:
    """Compute surface response for the ocean tile.

    Ocean provides SST with fixed albedo/emissivity. Bulk fluxes
    are computed using either constant coefficients or stability-dependent
    MOST algorithms (COARE 3.0 or Large & Yeager 2004).
    """
    shape = ocean_sst.shape
    # Resolve the thermodynamic convention ONCE (getattr-safe for any
    # config lacking the field) and validate it, so the q_sat curve and
    # the MOST call below never disagree (#762, codex round-20).
    _thermo_conv = getattr(config, "thermo_convention", "legoesm")
    if _thermo_conv not in ("legoesm", "aerobulk"):
        raise ValueError(
            f"Unknown thermo_convention {_thermo_conv!r}; expected "
            "'legoesm' or 'aerobulk'."
        )
    rho = forcing.rho_lowest

    valid_schemes = ("constant", "most", "coare3", "large_yeager")
    if config.bulk_scheme not in valid_schemes:
        raise ValueError(
            f"Unknown coupler bulk_scheme {config.bulk_scheme!r}; "
            f"expected one of {valid_schemes}."
        )
    # The aerobulk convention (SST-dependent L_vap, moist cp_air, Goff
    # q_sat) is the NEMO/AeroBulk MOST set — it engages ONLY on the MOST
    # solver schemes ("most"/"coare3"/"large_yeager").  The 'constant'
    # fixed-coefficient closure is a different closure entirely (constant
    # C_H/C_E, constant L_v/c_pd in simple_bulk_fluxes), so applying Goff
    # q_sat there alone would be a HALF-convention; keep it on Tetens
    # (codex round-20).  ``most`` = iterative MOST at the fixed ``ocean_z0``
    # roughness (parity with the sea-ice/land/lake dispatchers).
    _is_most = config.bulk_scheme in ("most", "coare3", "large_yeager")
    q_sfc = ocean_surface_q_sat(
        ocean_sst, forcing.p_surface,
        thermo_convention=_thermo_conv, bulk_scheme=config.bulk_scheme,
        saline_factor=_Q_SAT_SALINE_FACTOR)

    if _is_most:
        # Use wind relative to ocean surface current
        u_rel = forcing.u_lowest - ocean_u
        v_rel = forcing.v_lowest - ocean_v
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            u_rel, v_rel,
            forcing.T_lowest, forcing.q_lowest,
            ocean_sst, q_sfc,
            rho,
            z_ref=config.z_ref,
            z_t=config.z_t_atm,
            z_q=config.z_q_atm,
            z0_init=config.ocean_z0,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
            # COARE free-convection gustiness (w*) on the tile flux: keeps the
            # air-sea interface energy-consistent with the atmosphere surface
            # layer (which already carries gustiness_w_zi) and lets a calm warm
            # ocean evaporate realistically.  0.0 => off => byte-identical.
            gustiness_w_zi=config.gustiness_w_zi,
            thermo_convention=_thermo_conv,
            stability_scheme=config.stability_scheme,
        )
    else:
        # Constant neutral coefficients (original behavior).  Sub-grid
        # convective gustiness floor (Wing 2018) on the effective wind so
        # light-wind/convective columns evaporate realistically (the dry-column
        # / weak-hydrological-cycle fix); subsumes the U_min numerical floor.
        wind_speed = apply_gustiness(
            forcing.u_lowest, forcing.v_lowest, config.gustiness,
        )
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            ocean_sst, q_sfc, rho, wind_speed,
            config.Cd_ocean, config.Ch_ocean,
        )

    # Ocean albedo: constant or zenith-dependent
    # Honour CouplerConfig.ocean_albedo for the constant-albedo path by
    # overriding alpha_ocean_const in the OceanAlbedoConfig.
    oac = config.ocean_albedo_config
    if oac.method == "constant":
        oac = oac._replace(alpha_ocean_const=config.ocean_albedo)
    alpha_ocean = compute_ocean_albedo(
        forcing.cos_zenith, oac,
    )
    # Ensure correct shape
    if not hasattr(alpha_ocean, 'shape') or alpha_ocean.shape != shape:
        alpha_ocean = jnp.broadcast_to(jnp.asarray(alpha_ocean), shape)

    # Surface upward longwave: ε σ T⁴ + (1-ε)·lw_down.  Same direct
    # expression as the loop-11 ``two_layer_lake.py`` fix — avoids the
    # full ``surface_radiation_fluxes`` call which recomputes
    # ``sw_net`` and the LW balance only to discard them.
    lw_up = (
        config.ocean_emissivity * constants.sigma_sb * ocean_sst ** 4
        + (1.0 - config.ocean_emissivity) * forcing.lw_down
    )

    # ``jnp.full`` is one ``Broadcast`` HLO op vs the
    # ``broadcast_to(jnp.array(scalar), shape)`` form which adds a
    # ``ConvertElementType`` for the implicit Python-float promotion
    # — same per-coupler-step micro-optimisation as the loop-18 lake
    # rewrite.
    _ssh_dtype = ocean_sst.dtype
    # Ocean tile freshwater: P − E, where evap is back-derived from
    # lhflx using L_v (ocean is liquid, never sublimes).  Positive =
    # freshwater INTO ocean.
    evap_rate = lhflx / constants.L_v   # kg/m²/s, positive = up (ocean → atm)
    freshwater_flux = forcing.precip_total - evap_rate
    return TileResponse(
        T_sfc=ocean_sst,
        albedo=alpha_ocean,
        emissivity=jnp.full(shape, config.ocean_emissivity, dtype=_ssh_dtype),
        z0=jnp.full(shape, config.ocean_z0, dtype=_ssh_dtype),
        q_surface=q_sfc,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up,
        u_ocean_sfc=ocean_u,
        v_ocean_sfc=ocean_v,
        co2_flux=jnp.zeros(shape, dtype=_ssh_dtype),
        freshwater_flux=freshwater_flux,
        # Ocean tile is itself the source of ocean heat — does not
        # extract from the ocean.  Sea-ice tiles report their
        # extraction; the ocean column treats the sum across tiles
        # (after blending) as a heat-budget sink.
        ocean_heat_extraction=jnp.zeros(shape, dtype=_ssh_dtype),
        # Ocean tile contributes its own wind stress (already in
        # tau_x/tau_y) — back-reaction is the ice tile's job.
        ocean_stress_x=jnp.zeros(shape, dtype=_ssh_dtype),
        ocean_stress_y=jnp.zeros(shape, dtype=_ssh_dtype),
        # Ocean evaporation: lhflx already used L_v, so evap_rate
        # is the correct mass flux.
        surface_mass_flux=evap_rate,
        # Ocean tile is the salt-budget sink, not a source of salt
        # back to itself — zero flux on this channel.
        salt_flux=jnp.zeros(shape, dtype=_ssh_dtype),
    )


def make_coupler(
    coupler_config: CouplerConfig,
    land_config: LandConfig,
    ice_config: SeaIceConfig,
    lake_config: LakeConfig,
    lat: jnp.ndarray | None = None,
    grid=None,
    land_param_provider=None,
    land_features: jnp.ndarray | None = None,
    land_soil_frozen_fraction: jnp.ndarray | None = None,
):
    """Factory that returns step_surface function.

    Parameters
    ----------
    lat : jnp.ndarray, optional
        Latitude [radians], same spatial shape as forcing fields.
        Required when the land carbon cycle is enabled.
    grid : CubedSphereGrid, optional
        Required when ``ice_config.dynamics != "none"`` or
        ``ice_config.transport != "none"``.
    land_param_provider : eqx.Module, optional
        Provider that produces spatially-varying ``LandSurfaceParams``.
        If None, step functions use scalar config values (backward compat).
    land_soil_frozen_fraction : jnp.ndarray, optional
        STATIC per-cell annual frozen fraction ``phi`` in ``[0, 1]`` (same
        spatial shape as ``lat`` / the forcing fields), the perennial-frost
        index that drives the permafrost/anaerobic SOM protection ``f_perma``
        in the multilayer land carbon step (see
        :func:`legoesm.land.carbon.carbon_cycle.perennial_frost_protection`).
        ``phi`` is CLIMATOLOGICAL (permafrost extent is slowly-varying), so a
        static field -- built once from the run's climate, or carried in the
        seeded carbon IC (``global_carbon_ic.npz`` ``soil_frozen_fraction``,
        the SAME cover-weighted ``phi`` that scaled the seeded equilibrium) --
        is the correct source and avoids a prognostic annual diagnostic.
        Captured as a compile-time constant (not a per-step traced arg) since
        it never changes over the run.  ``None`` (default) -> the carbon step
        receives ``soil_frozen_fraction=None`` and is BYTE-IDENTICAL to the
        no-protection path (a static feature gate, not a data-dependent
        branch): a coupled run without a permafrost IC is unchanged.  Only
        meaningful for the multilayer (differland-carbon) land tile; supplying
        it with a slab land config raises (the slab tile has no SOM column).
    land_features : jnp.ndarray, optional
        Static feature matrix ``(ncol, n_input)`` for neural provider.
        Required when ``land_param_provider`` is a ``NeuralParamProvider``.

    Returns
    -------
    step_surface : callable
        (SurfaceState, AtmToSurface, TileConfig, ocean_sst, ocean_u,
         ocean_v, dt, doy) -> (SurfaceState, SurfaceToAtm)
    """
    _validate_coupler_config(coupler_config)
    U_min = coupler_config.U_min
    coupling_dt = float(coupler_config.coupling_dt)
    _lat = lat
    _grid = grid
    _use_multilayer = isinstance(land_config, MultiLayerLandConfig)
    _land_param_provider = land_param_provider
    _land_features = land_features
    # Static per-cell permafrost index phi (climatological, never changes over
    # the run) -> closure constant, threaded into the multilayer carbon step so
    # a coupled run maintains the seeded permafrost SOC.
    _land_soil_frozen_fraction = land_soil_frozen_fraction
    if _land_soil_frozen_fraction is not None:
        # phi only drives the SOM protection in the multilayer (differland) carbon
        # column -- which itself requires lat -- so a slab config or a missing lat
        # is a caller error, not a silent no-op.
        if not _use_multilayer:
            raise ValueError(
                "land_soil_frozen_fraction (permafrost phi) is only supported for "
                "the multilayer land carbon path; got a non-MultiLayerLandConfig "
                f"land_config ({type(land_config).__name__}). Omit it for slab land.")
        if _lat is None:
            raise ValueError(
                "land_soil_frozen_fraction (phi) requires lat (the per-cell carbon "
                "grid the land tile flattens phi against, so its size can be "
                "validated); pass lat.")
        # Fail fast on a malformed STATIC phi so a bad finidat field cannot silently
        # alter protection or scalar-broadcast: finite, in [0, 1] (an annual frozen
        # FRACTION), and EXACTLY the lat/forcing grid size.  phi is a setup-time
        # constant (concrete array), so these host-side checks add no per-step cost
        # and never run under trace.
        _phi0 = jnp.asarray(_land_soil_frozen_fraction)
        if not bool(jnp.all(jnp.isfinite(_phi0))):
            raise ValueError("land_soil_frozen_fraction (phi) must be finite.")
        _phi_lo = float(jnp.min(_phi0)); _phi_hi = float(jnp.max(_phi0))
        if _phi_lo < 0.0 or _phi_hi > 1.0:
            raise ValueError(
                "land_soil_frozen_fraction (phi) must be in [0, 1] (an annual "
                f"frozen fraction); got [{_phi_lo}, {_phi_hi}].")
        if _phi0.size != jnp.asarray(_lat).size:
            raise ValueError(
                "land_soil_frozen_fraction size must match the lat/forcing grid; "
                f"got {_phi0.size} vs {jnp.asarray(_lat).size}.")

    # --- Surface-tile catalog -------------------------------------------------
    # Each tile is a step closure ``(ctx) -> (TileResponse, state_updates)``.  The
    # four Earth-system surface tiles step INDEPENDENTLY of one another (no tile
    # reads another's freshly-stepped state; ice concentration is consumed only
    # afterwards, by tile-fraction blending), so the coupler can iterate this
    # catalog in any order and assemble the result uniformly — replacing the
    # former hardcoded inline land/ice/lake/ocean stepping with a data-driven set.

    def _step_land_tile(ctx: _TileContext):
        if _use_multilayer:
            # Multi-layer land operates on columnar (ncol,) arrays.
            # Flatten (6,n,n) forcing to (ncol,) and unflatten response.
            _spatial_shape = ctx.atm_forcing.sw_down.shape
            _flat_forcing = jax.tree.map(
                lambda x: x.reshape(-1) if hasattr(x, 'reshape') else x,
                ctx.atm_forcing,
            )
            _flat_lat = (_lat.reshape(-1)
                         if _lat is not None and hasattr(_lat, 'reshape')
                         else _lat)
            # Flatten the static (6,n,n) permafrost phi to (ncol,) exactly like
            # lat; None stays None so the carbon step takes its byte-identical
            # no-protection branch (static gate, not jnp.where).
            _flat_frozen = (
                _land_soil_frozen_fraction.reshape(-1)
                if (_land_soil_frozen_fraction is not None
                    and hasattr(_land_soil_frozen_fraction, 'reshape'))
                else _land_soil_frozen_fraction)
            land_new, land_resp_flat, carbon_new = step_multilayer_land(
                ctx.sfc_state.land, _flat_forcing, land_config, U_min, ctx.dt,
                lat=_flat_lat, carbon_state=ctx.sfc_state.carbon, doy=ctx.doy,
                land_params=ctx.land_params,
                soil_frozen_fraction=_flat_frozen,
            )
            # Unflatten TileResponse fields back to spatial shape
            land_resp = jax.tree.map(
                lambda x: (x.reshape(_spatial_shape)
                           if hasattr(x, 'reshape') and x.ndim == 1
                              and x.shape[0] == math.prod(_spatial_shape)
                           else x),
                land_resp_flat,
            )
        else:
            land_new, land_resp, carbon_new = step_land(
                ctx.sfc_state.land, ctx.atm_forcing, land_config, U_min, ctx.dt,
                lat=_lat, carbon_state=ctx.sfc_state.carbon, doy=ctx.doy,
                land_params=ctx.land_params,
            )
        return land_resp, {"land": land_new, "carbon": carbon_new}

    def _step_ice_tile(ctx: _TileContext):
        ice_new, ice_resp = step_sea_ice(
            ctx.sfc_state.ice, ctx.atm_forcing, ctx.ocean_sst, ctx.ocean_u,
            ctx.ocean_v, ice_config, U_min, ctx.dt, grid=_grid)
        return ice_resp, {"ice": ice_new}

    def _step_lake_tile(ctx: _TileContext):
        lake_new, lake_resp = step_lake(
            ctx.sfc_state.lake, ctx.atm_forcing, lake_config, U_min, ctx.dt)
        return lake_resp, {"lake": lake_new}

    def _step_ocean_tile(ctx: _TileContext):
        # Diagnostic tile — the ocean model handles its own state, so no update.
        ocean_resp = ocean_tile_response(
            ctx.atm_forcing, ctx.ocean_sst, ctx.ocean_u, ctx.ocean_v,
            coupler_config)
        return ocean_resp, {}

    _tile_catalog = (
        ("ocean", _step_ocean_tile),
        ("ice", _step_ice_tile),
        ("land", _step_land_tile),
        ("lake", _step_lake_tile),
    )

    def step_surface(
        sfc_state: SurfaceState,
        atm_forcing: AtmToSurface,
        tile_config: TileConfig,
        ocean_sst: jnp.ndarray,
        ocean_u_sfc: jnp.ndarray,
        ocean_v_sfc: jnp.ndarray,
        dt: float,
        doy: float = 0.0,
    ) -> tuple[SurfaceState, SurfaceToAtm]:
        """Step all surface tiles and return blended response."""
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")
        if tile_config.f_land.shape != atm_forcing.sw_down.shape:
            raise ValueError(
                "tile_config.f_land shape must match forcing shape, got "
                f"{tile_config.f_land.shape!r} vs {atm_forcing.sw_down.shape!r}",
            )
        if tile_config.f_lake.shape != atm_forcing.sw_down.shape:
            raise ValueError(
                "tile_config.f_lake shape must match forcing shape, got "
                f"{tile_config.f_lake.shape!r} vs {atm_forcing.sw_down.shape!r}",
            )

        # 1. Materialize spatial land params (once per coupler step)
        if _land_param_provider is not None:
            if _land_features is not None:
                _lp = _land_param_provider(_land_features)
            else:
                _lp = _land_param_provider()
            # For slab land: reshape (ncol,) -> spatial shape (e.g. (6,n,n))
            if not _use_multilayer:
                _lp = reshape_params(_lp, atm_forcing.sw_down.shape)
        else:
            _lp = None

        # 2. Step every surface tile via the uniform catalog.  Each closure
        # returns its TileResponse and its state updates; the tiles step
        # independently so iteration order does not affect the result.
        ctx = _TileContext(
            sfc_state=sfc_state, atm_forcing=atm_forcing, doy=doy, dt=dt,
            ocean_sst=ocean_sst, ocean_u=ocean_u_sfc, ocean_v=ocean_v_sfc,
            land_params=_lp,
        )
        responses: dict = {}
        updates: dict = {}
        for _name, _tile_step in _tile_catalog:
            _resp, _upd = _tile_step(ctx)
            responses[_name] = _resp
            updates.update(_upd)

        # 3. Tile fractions (ice concentration from the updated ice state).
        # The ice tile's ice->ocean exchange fluxes are returned per-grid-cell
        # and blended by f_water in blend_tiles (F11), so no pre-step
        # concentration is needed here.
        ice_conc = updates["ice"].concentration.data
        # Multi-category: sum across categories for total concentration
        if ice_conc.ndim > len(atm_forcing.sw_down.shape):
            ice_conc = jnp.sum(ice_conc, axis=-1)
        fracs = compute_tile_fractions(tile_config, ice_conc)

        # 4. Blend the tile responses (area-weighted; conservation unchanged).
        blended = blend_tiles(
            responses["ocean"], responses["ice"], responses["land"],
            responses["lake"], fracs)

        # 7. Accumulate
        acc_new = accumulate(sfc_state.accumulator, blended, dt)
        dt_arr = jnp.asarray(dt, dtype=acc_new.total_dt.dtype)
        coupling_dt_arr = jnp.asarray(coupling_dt, dtype=acc_new.total_dt.dtype)

        # If we crossed the coupling window, emit the window mean and carry any
        # residual dt from this step into the next window.
        def _on_flush(_):
            dt_prev = sfc_state.accumulator.total_dt
            dt_to_close = jnp.clip(coupling_dt_arr - dt_prev, 0.0, dt_arr)
            acc_closed = accumulate(sfc_state.accumulator, blended, dt_to_close)
            blended_out = mean_accumulator(acc_closed)

            # Cast blended_out to match blended's leaf dtypes so both
            # jax.lax.cond branches return the same types.
            blended_out = jax.tree.map(
                lambda a, b: a.astype(b.dtype) if hasattr(b, "dtype") else a,
                blended_out, blended,
            )

            dt_excess = jnp.maximum(dt_arr - dt_to_close, 0.0)
            acc_next = accumulator_from_flux(
                blended,
                dt_excess,
                dtype=acc_new.total_dt.dtype,
            )
            return acc_next, blended_out

        def _no_flush(_):
            return acc_new, blended

        acc_next, blended_out = jax.lax.cond(
            acc_new.total_dt >= coupling_dt_arr,
            _on_flush,
            _no_flush,
            operand=None,
        )

        new_state = SurfaceState(
            land=updates["land"], ice=updates["ice"], lake=updates["lake"],
            accumulator=acc_next, carbon=updates["carbon"])

        return new_state, blended_out

    return step_surface
