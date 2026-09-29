"""Bulk aerodynamic surface layer fluxes.

Simple bulk formulas for surface momentum, sensible heat, and latent
heat fluxes using neutral drag and transfer coefficients.

    |V| = sqrt(u^2 + v^2 + eps)
    u* = sqrt(Cd) * |V|
    tau_x = -rho * Cd * |V| * u
    tau_y = -rho * Cd * |V| * v
    SH = rho * c_pd * Ch * |V| * (T_sfc - T)
    LH = rho * L_v * Ch * |V| * (q_sfc - q_v)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.core.bulk_flux import (
    surface_reference_state,
    compute_most_fluxes,
    compute_sam_oceflx_fluxes,
    simple_bulk_fluxes,
    validate_bulk_scheme,
)

# Bulk schemes that solve a stability-dependent surface layer (and therefore
# honour the ``z_ref_model_level`` height correction); the constant-coefficient
# path ignores z_ref.
_STABILITY_SCHEMES = ("most", "coare3", "large_yeager", "large_yeager_cesm")


__physics_contract__ = {
    "summary": (
        "Bulk-aerodynamic surface-layer fluxes of momentum, sensible heat, "
        "and latent heat (constant neutral coefficients or iterative "
        "Monin-Obukhov / COARE-3.0 / Large-Yeager), plus a tiled-surface "
        "area-weighted aggregator."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3",
    },
    "outputs": {
        "tau_x": "Pa", "tau_y": "Pa", "shflx": "W/m^2",
        "lhflx": "W/m^2", "ustar": "m/s",
    },
    "sign_convention": (
        "Surface (bottom-boundary) fluxes are a SOURCE for the column, so "
        "nothing is conserved here. Stress opposes the wind: "
        "tau_x = -rho Cd |V| u. shflx = rho c_pd Ch |V| (T_sfc - T) and "
        "lhflx = rho L_v Ch |V| (q_sfc - q_v) are POSITIVE UPWARD (out of the "
        "surface into the atmosphere) when the surface is warmer / moister. "
        "ustar >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Large & Yeager (2004), NCAR/TN-460+STR; "
        "Fairall et al. (2003) COARE 3.0, J. Climate 16, 571-591"
    ),
    "idealized_test": (
        "tests/unit/test_tiled_surface_fluxes.py and "
        "tests/unit/test_les_surface_coupling_conservation.py: equilibrium "
        "(u=v=0, T=T_sfc, q_v=q_sfc) -> zero fluxes; tiled fluxes aggregate "
        "linearly by area fraction and ustar = sqrt(|tau|/rho)."
    ),
}


def surface_fluxes_at_lowest_level(u, v, T, q_v, T_sfc, q_sfc, rho, config,
                                   z_low):
    """``compute_surface_fluxes`` told how high its own input actually is.

    Every turbulence kernel feeds the surface solver the LOWEST FULL LEVEL's
    wind, temperature and humidity.  Whether it also tells the solver the
    height of that level is ``SurfaceLayerConfig.z_ref_model_level``; with it
    off the solver divides by ``config.z_ref`` (10 m) for values that came
    from over a hundred metres up, which inflates the fluxes.

    Two things have to move together and that is why this is one function
    rather than a keyword at seven call sites: the height AND the temperature.
    The bulk formula wants the air-sea temperature difference at the reference
    height, so the input is brought down dry-adiabatically
    (``T + g/c_p * z``, ~1.5 K at 150 m, the COARE convention).  Passing the
    height without that adjustment reads the stability off a dry adiabat and
    is a different bug from the one being fixed.

    ``z_low`` is the height of the lowest full level ABOVE THE LOCAL SURFACE
    [m], shape (ncol,).  Callers pass ``z_full[:, -1] - z_half[:, -1]``: on the
    sigma lanes the surface interface is already zero so the subtraction is an
    identity, but the nonhydrostatic lane's heights are absolute
    terrain-following altitudes, and without it a column over a 2 km mountain
    would be handed a 2 km reference height and warmed ~20 K (codex).

    The caller is responsible for only enabling this where ``T_sfc`` is a REAL
    surface temperature.  Where it is a stand-in for the lowest air
    temperature, warming the air input alone manufactures a permanent
    air-surface contrast of -g/c_p*z (~1.3 K) and with it a spurious downward
    sensible heat flux that no test would catch (GLM).  That is why the switch
    is off in this config by default and turned on by the driver, which knows
    whether the run has a surface.
    """
    # BOTH halves, or NEITHER.  ``compute_surface_fluxes`` honours ``z_ref``
    # on the MOST schemes only -- the constant-coefficient path ignores it --
    # so on a constant-Cd config the height would be dropped while the
    # temperature adjustment survived, leaving a one-sided ~1.3 K cooling of
    # the surface relative to the air and a sensible heat flux made of
    # nothing (GLM).  That is the same defect this helper's precondition
    # warns about, arriving through a different door.
    if (not getattr(config, "z_ref_model_level", False)
            or z_low is None
            or config.bulk_scheme not in _STABILITY_SCHEMES):
        return compute_surface_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, config)
    T_ref, z_ref = surface_reference_state(T, config.z_ref, z_low)
    return compute_surface_fluxes(
        u, v, T_ref, q_v, T_sfc, q_sfc, rho, config, z_ref=z_ref)


def compute_surface_fluxes(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    config: SurfaceLayerConfig,
    z_ref=None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute bulk aerodynamic surface fluxes.

    Supports constant neutral coefficients or stability-dependent
    MOST algorithms (COARE 3.0, Large & Yeager 2004) selected via
    ``config.bulk_scheme``.

    Parameters
    ----------
    u : jax.Array
        Lowest-level zonal wind [m/s], shape (ncol,).
    v : jax.Array
        Lowest-level meridional wind [m/s], shape (ncol,).
    T : jax.Array
        Lowest-level temperature [K], shape (ncol,).
    q_v : jax.Array
        Lowest-level water vapor mixing ratio [kg/kg], shape (ncol,).
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Lowest-level air density [kg/m^3], shape (ncol,).
    config : SurfaceLayerConfig
        Surface layer parameters.
    z_ref : array or None
        Per-column reference height [m] overriding ``config.z_ref`` (MOST
        schemes only; the constant-coefficient path ignores it).

    Returns
    -------
    tau_x : jax.Array
        Surface zonal stress [Pa], shape (ncol,).
    tau_y : jax.Array
        Surface meridional stress [Pa], shape (ncol,).
    shflx : jax.Array
        Surface sensible heat flux [W/m^2], shape (ncol,). Positive upward.
    lhflx : jax.Array
        Surface latent heat flux [W/m^2], shape (ncol,). Positive upward.
    ustar : jax.Array
        Friction velocity [m/s], shape (ncol,).
    """
    validate_bulk_scheme(config.bulk_scheme)
    # Route every stability-dependent bulk scheme — the fixed-roughness
    # Monin-Obukhov land scheme ("most") as well as the ocean air-sea schemes
    # ("coare3"/"large_yeager") — through the iterative MOST solver.  "most"
    # uses the local roughness ``config.z0`` via the neutral log law and the
    # selectable ``stability_scheme`` stable branch; without this it would fall
    # through to the constant-Cd path and silently ignore z0 and stability.
    if config.bulk_scheme == "large_yeager_cesm":
        # CESM/CIME ``shr_flux_atmOcn`` (CAM6 coupler air-sea law): its own
        # fixed-iteration solver, not the generic MOST loop.  ``T`` is the
        # lane's lowest-level temperature (potential-temperature corrected by
        # the caller when ``z_ref_model_level`` is on), as for the other schemes.
        tau_x, tau_y, shflx, lhflx, ustar = compute_sam_oceflx_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho,
            z_bot=(config.z_ref if z_ref is None else z_ref),
            variant="cesm",
        )
        return _apply_prescribed_scalar_fluxes(
            config, tau_x, tau_y, shflx, lhflx, ustar, rho)
    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        tau_x, tau_y, shflx, lhflx, ustar = compute_most_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho,
            z_ref=(config.z_ref if z_ref is None else z_ref),
            z0_init=config.z0,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
            gustiness_w_zi=getattr(config, "gustiness_w_zi", None),
            thermo_convention=getattr(config, "thermo_convention", "legoesm"),
            stability_scheme=getattr(config, "stability_scheme", "dyer1974"),
            unstable_gamma=config.most_unstable_gamma,
            stable_beta=config.most_stable_beta,
            z0h_z0_ratio=config.z0h_z0_ratio,
        )
        # The prescribed-flux override applies on THIS branch too. Returning
        # early without it would let a config that sets both a MOST scheme and
        # a prescribed flux silently ignore the prescription -- the closure
        # would run on MOST-derived fluxes while the caller believed it had
        # pinned them to the deck.
        return _apply_prescribed_scalar_fluxes(
            config, tau_x, tau_y, shflx, lhflx, ustar, rho)

    # Constant neutral coefficients (default)
    Cd = config.Cd_neutral
    Ch = config.Ch_neutral

    # Wind speed with minimum to avoid division by zero
    wind_speed = jnp.sqrt(u ** 2 + v ** 2 + 1e-4)  # coeff-ok: wind-speed floor [m^2/s^2]

    # Friction velocity
    ustar = jnp.sqrt(Cd) * wind_speed

    # Momentum + sensible/latent heat fluxes — the shared constant-coefficient
    # bulk formula (redundancy audit): tau = -rho*Cd*|U|*u, shflx =
    # rho*c_pd*Ch*|U|*(T_sfc-T), lhflx = rho*L_v*Ch*|U|*(q_sfc-q_v).  Identical to
    # legoesm.core.bulk_flux.simple_bulk_fluxes, so use it instead of re-deriving.
    from legoesm.core.bulk_flux import simple_bulk_fluxes

    tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, wind_speed, Cd, Ch,
    )

    return _apply_prescribed_scalar_fluxes(
        config, tau_x, tau_y, shflx, lhflx, ustar, rho)


# Floor on the prescribed stress magnitude inside ustar's nested roots; far
# below any measurable stress, it only bounds the derivative at zero.
_PRESCRIBED_TAU_FLOOR_PA = 1.0e-12  # coeff-ok: numerical floor for a 0/0 gradient


def surface_moisture_flux(config, lhflx, T_sfc):
    """Surface water flux [kg/m2/s, positive up] for a kernel's moisture BC.

    The coupler's prescribed mass flux (``config.prescribed_evap_kg_m2_s``)
    when one was folded in -- the tiles' water leaves the surface exactly as
    it enters the atmosphere, whatever latent heat each tile charged for it.
    Otherwise the latent heat flux is converted back with the SAME
    temperature-dependent ``L_v(T_sfc)`` the bulk laws charged, so the two
    are exact inverses (a constant ``L_v`` here against an ``L_v(T)`` there
    lost ~2 % of warm-ocean evaporation).  Standalone lanes without a coupler
    have no phase information here: sublimation over their prescribed ice is
    charged at ``L_v`` on both sides, consistently.
    """
    evap = getattr(config, "prescribed_evap_kg_m2_s", None)
    if evap is not None:
        return jnp.broadcast_to(jnp.asarray(evap, dtype=jnp.asarray(lhflx).dtype),
                                jnp.shape(lhflx))
    from legoesm.thermo import latent_heat_vaporization
    return lhflx / latent_heat_vaporization(T_sfc)


def latent_enthalpy_correction(lhflx, evap):
    """Heat [W/m2, positive up] the atmosphere must ADD to its sensible flux so
    that its energy intake equals the physical ``shflx + lhflx``.

    The atmosphere's moist enthalpy credits every kilogram of vapour with the
    reference ``constants.L_v`` (its constant-L convention, see
    core.conservation), while the surface charged ``L(T_sfc, phase) * E`` --
    at 30 degC about 3 % less.  ``lhflx - L_v * E`` is that difference, the
    enthalpy carried by the water itself (user decision 2026-09-28, 4a):
    booked into the diffusion's heat lower BC, never into the reported
    ``shflx``/``lhflx`` (those stay the physical fluxes).  Exactly zero when
    the surface charged the constant.
    """
    return lhflx - constants.L_v * evap   # latent-ok: moist-enthalpy reference L_v of the atmosphere


def prescribed_into_surface_flux(surface_flux, rho_sfc, *, shflx=None,
                                 lhflx=None, tau_x=None, tau_y=None):
    """Replace the prescribed components of a ``(tau_x, tau_y, shflx, lhflx,
    ustar)`` surface-flux tuple, keeping the others.

    The tiled (mosaic) surface path hands a kernel this tuple instead of
    letting it call :func:`compute_surface_fluxes`; a prescribed coupler /
    ERA5 flux must therefore land IN the tuple, component by component, so
    the unprescribed components (the tiled stress when only heat is given,
    the tiled heat when only the stress is) survive.  ``ustar`` is rebuilt
    from the stress whenever a stress component is replaced, with the same
    floor as :func:`_apply_prescribed_scalar_fluxes`.
    """
    tx, ty, sh, lh, us = surface_flux
    if shflx is not None:
        sh = jnp.broadcast_to(jnp.asarray(shflx), sh.shape)
    if lhflx is not None:
        lh = jnp.broadcast_to(jnp.asarray(lhflx), lh.shape)
    if tau_x is not None or tau_y is not None:
        if tau_x is not None:
            tx = jnp.broadcast_to(jnp.asarray(tau_x), tx.shape)
        if tau_y is not None:
            ty = jnp.broadcast_to(jnp.asarray(tau_y), ty.shape)
        us = jnp.sqrt(
            jnp.sqrt(tx ** 2 + ty ** 2 + _PRESCRIBED_TAU_FLOOR_PA ** 2)
            / rho_sfc)
    return tx, ty, sh, lh, us


def _apply_prescribed_scalar_fluxes(config, tau_x, tau_y, shflx, lhflx, ustar, rho):
    """Override the turbulent surface fluxes with prescribed values.

    Scalars: when ``config.prescribed_shflx_w_m2`` /
    ``config.prescribed_lhflx_w_m2`` is set, the bulk sensible/latent heat
    fluxes are replaced by it (``jnp.full_like`` broadcasts a scalar or an
    (ncol,) array). Momentum: when ``config.prescribed_tau_x_pa`` /
    ``config.prescribed_tau_y_pa`` is set, the bulk stress components are
    replaced and ``ustar`` is rebuilt from the (possibly mixed
    prescribed/interactive) stress magnitude,
    ``ustar = sqrt(sqrt(tau_x**2 + tau_y**2) / rho)``, so a scheme that
    builds its mixing from the friction velocity sees the prescribed stress.
    Components left unset keep their bulk (interactive) values; when none of
    the four fields is set the behaviour is exactly the previous one.

    Parameters
    ----------
    config : SurfaceLayerConfig
        Closure configuration; only its ``prescribed_*`` fields are read.
    tau_x, tau_y, shflx, lhflx, ustar : jnp.ndarray
        Bulk surface stresses, heat fluxes and friction velocity, (ncol,).
    rho : jnp.ndarray
        Lowest-level density, (ncol,); used only to rebuild ``ustar`` from a
        prescribed stress.

    Returns
    -------
    tuple of jnp.ndarray
        ``(tau_x, tau_y, shflx, lhflx, ustar)`` with prescribed components
        substituted.
    """
    if config.prescribed_shflx_w_m2 is not None:
        shflx = jnp.full_like(shflx, config.prescribed_shflx_w_m2)
    if config.prescribed_lhflx_w_m2 is not None:
        lhflx = jnp.full_like(lhflx, config.prescribed_lhflx_w_m2)
    if (config.prescribed_tau_x_pa is not None
            or config.prescribed_tau_y_pa is not None):
        if config.prescribed_tau_x_pa is not None:
            tau_x = jnp.full_like(tau_x, config.prescribed_tau_x_pa)
        if config.prescribed_tau_y_pa is not None:
            tau_y = jnp.full_like(tau_y, config.prescribed_tau_y_pa)
        # ustar^2 = |tau| / rho  =>  ustar = sqrt(sqrt(tx^2 + ty^2) / rho).
        # The floor keeps the nested roots differentiable at exactly zero
        # stress (a calm ERA5 column): without it the cotangent is 0/0.
        ustar = jnp.sqrt(
            jnp.sqrt(tau_x ** 2 + tau_y ** 2 + _PRESCRIBED_TAU_FLOOR_PA ** 2)
            / rho)
    return tau_x, tau_y, shflx, lhflx, ustar


def beta_limited_surface_humidity(
    q_sat_sfc: jax.Array,
    q_air: jax.Array,
    f_land: jax.Array,
    beta_land: float | jax.Array,
) -> jax.Array:
    """Soil-moisture-limited effective surface humidity for a BLENDED surface.

    ``beta_land`` may be a scalar (the static ``mpas_land_beta`` knob) or a
    per-column array of shape ``(ncol,)`` (the traced root-zone ``beta_soil``
    from the interactive multilayer land, #1312 phase 2b) — the formula below
    is elementwise either way.

    On a non-tiled surface the bulk latent flux is
    ``LH ∝ (q_sfc - q_air)``.  Using the saturated ``q_sat_sfc`` everywhere
    makes every land cell an infinite swamp (beta = 1).  This throttles the
    LAND fraction's humidity gradient by ``beta_land`` while leaving the
    ocean/ice fraction saturated::

        q_sfc = q_air + (1 - f_land * (1 - beta_land)) * (q_sat_sfc - q_air)

    so the land-fraction latent flux is ``beta_land`` times its wet-surface
    potential — a first-order analogue of the tiled pipeline's per-tile
    alpha method (``physics_pipeline._tiled_surface_flux``).  The
    decomposition is exact only for a fixed shared transfer coefficient; a
    stability-dependent MOST/COARE bulk scheme recomputes ``C_E`` from the
    throttled ``q_sfc``, so this is approximate, not a true mosaic.
    ``beta_land = 1``
    returns ``q_sat_sfc`` exactly (byte-identical wet surface);
    ``beta_land = 0`` zeroes the land-fraction humidity gradient in both
    directions (no land evaporation and no land dew — a closed surface).

    Parameters
    ----------
    q_sat_sfc : array
        Saturation mixing ratio at the surface anchor [kg/kg], shape (ncol,).
    q_air : array
        Lowest-level vapor mixing ratio [kg/kg], shape (ncol,).
    f_land : array
        Land fraction in [0, 1], shape (ncol,).
    beta_land : float
        Land evaporation efficiency in [0, 1].

    Returns
    -------
    q_sfc : array
        Effective surface humidity for the bulk latent flux [kg/kg].
    """
    return q_air + (1.0 - f_land * (1.0 - beta_land)) * (q_sat_sfc - q_air)


class SurfaceTileSpec(NamedTuple):
    """Per-tile, per-column surface STATE for a mosaic (tiled) surface.

    Each grid column is treated as an area-weighted mosaic of open ocean,
    sea ice, and land.  Turbulent fluxes are computed SEPARATELY on each
    tile — each with its own surface temperature, saturation humidity, and
    (separately supplied) bulk scheme/roughness — then AREA-WEIGHTED (flux
    aggregation).  This is the physically-correct tiled-surface approach:
    it lets the ocean air-sea scheme (COARE 3.0 — Charnock wave roughness +
    convective gustiness) run on the ocean tile while the FIXED-ROUGHNESS
    land Monin-Obukhov scheme (``"most"``) runs on the land tile — instead
    of blending the surface TEMPERATURES first and applying a single
    (wrong-over-land) scheme to the blend.

    This struct holds ONLY traced arrays so it is a clean JAX pytree; the
    per-tile :class:`SurfaceLayerConfig` objects (which carry the static
    ``bulk_scheme`` string) are passed SEPARATELY to
    :func:`compute_tiled_surface_fluxes` as static arguments — mirroring
    how :func:`compute_surface_fluxes` takes its ``config`` apart from the
    arrays, and keeping the string out of the trace.

    Fractions are per column and must partition unity
    (``frac_ocean + frac_ice + frac_land = 1``); a tile with zero fraction
    contributes nothing to the blend but is still evaluated, so shapes stay
    static and there is no data-dependent control flow (JIT/AD safe).

    Fields
    ------
    frac_ocean, frac_ice, frac_land : jax.Array
        Open-ocean / sea-ice / land area fractions, shape ``(ncol,)``.
    T_ocean, T_ice, T_land : jax.Array
        Per-tile surface temperatures [K], shape ``(ncol,)``.
    q_sfc_ocean, q_sfc_ice, q_sfc_land : jax.Array
        Per-tile surface saturation specific humidities [kg/kg]
        (the ocean value already carries any saline reduction factor),
        shape ``(ncol,)``.
    """

    frac_ocean: jax.Array
    frac_ice: jax.Array
    frac_land: jax.Array
    T_ocean: jax.Array
    T_ice: jax.Array
    T_land: jax.Array
    q_sfc_ocean: jax.Array
    q_sfc_ice: jax.Array
    q_sfc_land: jax.Array


def _single_tile_flux(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    config: SurfaceLayerConfig,
    z_ref=None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Single-tile surface flux with correct fixed-roughness MOST routing.

    Like :func:`compute_surface_fluxes`, but routes the fixed-roughness
    land Monin-Obukhov scheme (``"most"``) to the iterative
    :func:`compute_most_fluxes` as well — :func:`compute_surface_fluxes`
    only reaches that path for the ocean schemes (``coare3`` /
    ``large_yeager``) and silently treats ``"most"`` as constant, which
    would ignore the land roughness ``z0``.  Reuses the same core bulk
    routines (no re-derived flux numerics).
    """
    validate_bulk_scheme(config.bulk_scheme)
    if config.bulk_scheme == "large_yeager_cesm":
        return compute_sam_oceflx_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho,
            z_bot=(config.z_ref if z_ref is None else z_ref),
            variant="cesm",
        )
    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        return compute_most_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho,
            z_ref=(config.z_ref if z_ref is None else z_ref),
            z0_init=config.z0,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
            gustiness_w_zi=getattr(config, "gustiness_w_zi", None),
            thermo_convention=getattr(config, "thermo_convention", "legoesm"),
            stability_scheme=getattr(config, "stability_scheme", "dyer1974"),
            unstable_gamma=config.most_unstable_gamma,
            stable_beta=config.most_stable_beta,
            z0h_z0_ratio=config.z0h_z0_ratio,
        )

    # Constant neutral coefficients.
    wind_speed = jnp.sqrt(u ** 2 + v ** 2 + 1e-4)  # coeff-ok: wind-speed floor [m^2/s^2]
    ustar = jnp.sqrt(config.Cd_neutral) * wind_speed
    tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, wind_speed,
        config.Cd_neutral, config.Ch_neutral,
    )
    return tau_x, tau_y, shflx, lhflx, ustar


def compute_tiled_surface_fluxes(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    rho: jax.Array,
    tiles: SurfaceTileSpec,
    config_ocean: SurfaceLayerConfig,
    config_ice: SurfaceLayerConfig,
    config_land: SurfaceLayerConfig,
    z_low=None,
    return_water: bool = False,
):
    """Area-weighted (mosaic) surface fluxes over ocean / ice / land tiles.

    Computes the turbulent surface fluxes SEPARATELY on each tile (each
    with its own surface temperature and saturation humidity from
    ``tiles`` and its own bulk scheme/roughness from the ``config_*``
    arguments) and AREA-WEIGHTS them.  The lowest-level atmospheric state
    (``u, v, T, q_v, rho``) is shared across tiles — only the surface
    boundary differs.

    Stress and sensible/latent heat fluxes aggregate LINEARLY by area
    fraction (flux conservation over a heterogeneous cell).  The blended
    friction velocity is recovered from the blended stress magnitude
    (``u* = sqrt(|tau| / rho)``) so it is consistent with the aggregated
    momentum flux rather than an ad-hoc average of the per-tile ``u*``.

    Parameters
    ----------
    u, v, T, q_v, rho : jax.Array
        Lowest-level zonal/meridional wind [m/s], temperature [K], water
        vapour specific humidity [kg/kg], and air density [kg/m^3], each
        shape ``(ncol,)``.
    tiles : SurfaceTileSpec
        Per-tile fractions, surface temperatures, and saturation
        humidities (traced arrays).
    config_ocean, config_ice, config_land : SurfaceLayerConfig
        Per-tile bulk scheme + roughness — STATIC (the ``bulk_scheme``
        string is resolved at trace time).  Typically ``coare3`` for the
        ocean, ``constant`` for ice, and ``most`` (with a land roughness
        ``z0``) for land.

    Returns
    -------
    tau_x, tau_y, shflx, lhflx, ustar : jax.Array
        Blended surface zonal/meridional stress [Pa], sensible and latent
        heat fluxes [W/m^2] (positive upward), and friction velocity
        [m/s], each shape ``(ncol,)``.
    """
    # Each tile HAS a real surface temperature, so the reference-height
    # correction applies here exactly as on the single-surface path: when the
    # caller supplies the lowest level's height above ground, the air is
    # brought down to it dry-adiabatically and the solver is told the height.
    # Without this the mosaic path kept the 10 m mislabel after the
    # single-surface path was fixed (codex).
    def _tile(T_sfc_tile, q_sfc_tile, cfg):
        """One tile, with the height correction decided by ITS OWN config.

        Gating all three tiles on the ocean tile's config was wrong: the land
        and ice tiles carry their own bulk scheme, and a tile on the
        constant-coefficient default would have ignored the height while
        still receiving the warmed air temperature -- a fabricated flux on
        exactly the tiles the ocean switch was not about (GLM).
        """
        if (z_low is None
                or not getattr(cfg, "z_ref_model_level", False)
                or cfg.bulk_scheme not in _STABILITY_SCHEMES):
            return _single_tile_flux(u, v, T, q_v, T_sfc_tile, q_sfc_tile,
                                     rho, cfg)
        return _single_tile_flux(
            u, v, T + (constants.g / constants.c_pd) * z_low, q_v,
            T_sfc_tile, q_sfc_tile, rho, cfg, z_ref=z_low)

    f_ocean = _tile(tiles.T_ocean, tiles.q_sfc_ocean, config_ocean)
    f_ice = _tile(tiles.T_ice, tiles.q_sfc_ice, config_ice)
    f_land = _tile(tiles.T_land, tiles.q_sfc_land, config_land)

    def _blend(idx: int) -> jax.Array:
        return (
            tiles.frac_ocean * f_ocean[idx]
            + tiles.frac_ice * f_ice[idx]
            + tiles.frac_land * f_land[idx]
        )

    tau_x = _blend(0)
    tau_y = _blend(1)
    shflx = _blend(2)
    lhflx = _blend(3)

    # Friction velocity consistent with the AGGREGATED momentum flux
    # (|tau| = rho * u*^2), not a raw average of per-tile u*.
    tau_mag = jnp.sqrt(tau_x ** 2 + tau_y ** 2)
    ustar = jnp.sqrt(tau_mag / jnp.maximum(rho, 1e-6))  # coeff-ok: density floor [kg/m^3]

    if not return_water:
        return tau_x, tau_y, shflx, lhflx, ustar
    # The blended WATER flux: each tile's latent heat inverted with the L_v of
    # ITS OWN surface temperature (what its bulk law charged), then area
    # weighted.  Dividing the blended heat by one L_v(T_blend) is not the same
    # number on a mixed cell; the kernel takes this as its moisture BC.
    # Each tile is inverted with the L its law CHARGED.  All three tile laws
    # run through _single_tile_flux, which never passes ``L_latent``, so the
    # ice tile (``constant`` scheme, tiled_surface_tile_configs) charged
    # L_v(T_ice) like the others -- the atmosphere-side mosaic has no phase
    # information; the coupled lane's sea-ice model charges L_s itself and
    # hands its water down the coupler's evaporation channel instead.
    from legoesm.thermo import latent_heat_vaporization
    water = (tiles.frac_ocean * f_ocean[3] / latent_heat_vaporization(tiles.T_ocean)
             + tiles.frac_ice * f_ice[3] / latent_heat_vaporization(tiles.T_ice)
             + tiles.frac_land * f_land[3] / latent_heat_vaporization(tiles.T_land))
    return (tau_x, tau_y, shflx, lhflx, ustar), water
