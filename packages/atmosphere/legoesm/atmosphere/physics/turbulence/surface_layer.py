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

from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.core.bulk_flux import (
    compute_most_fluxes,
    simple_bulk_fluxes,
    validate_bulk_scheme,
)


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


def compute_surface_fluxes(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    config: SurfaceLayerConfig,
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
    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        tau_x, tau_y, shflx, lhflx, ustar = compute_most_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho,
            z_ref=config.z_ref,
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
        return tau_x, tau_y, shflx, lhflx, ustar

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
    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        return compute_most_fluxes(
            u, v, T, q_v, T_sfc, q_sfc, rho,
            z_ref=config.z_ref,
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
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
    f_ocean = _single_tile_flux(
        u, v, T, q_v, tiles.T_ocean, tiles.q_sfc_ocean, rho, config_ocean,
    )
    f_ice = _single_tile_flux(
        u, v, T, q_v, tiles.T_ice, tiles.q_sfc_ice, rho, config_ice,
    )
    f_land = _single_tile_flux(
        u, v, T, q_v, tiles.T_land, tiles.q_sfc_land, rho, config_land,
    )

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

    return tau_x, tau_y, shflx, lhflx, ustar
