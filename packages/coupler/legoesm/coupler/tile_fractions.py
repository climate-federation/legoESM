"""Tile fraction computation and smooth blending utilities."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.coupler.config import TileConfig
from legoesm.core.coupling_fields import SurfaceToAtm, TileResponse


class TileFractions(NamedTuple):
    """Resolved tile fractions. All shape (6, n, n), sum to 1."""
    f_ocean: jax.Array
    f_ice: jax.Array
    f_land: jax.Array
    f_lake: jax.Array


def compute_tile_fractions(
    tile_config: TileConfig,
    ice_concentration: jax.Array,
) -> TileFractions:
    """Compute the 4 tile fractions from static config and prognostic ice.

    f_water = 1 - f_land - f_lake
    f_ice = f_water * ice_concentration
    f_ocean = f_water - f_ice
    """
    # Keep static fractions physically valid and conservative even if input
    # masks are slightly out of bounds due to interpolation/regridding noise.
    f_land = jnp.clip(tile_config.f_land, 0.0, 1.0)
    f_lake = jnp.clip(tile_config.f_lake, 0.0, 1.0)
    total_static = f_land + f_lake
    # Floor the reciprocal denominator at 1.0 so the *dead* branch of the where
    # never forms 1/0 at total_static = 0 (a pure-ocean cell, f_land = f_lake =
    # 0 — the most common cell).  The reciprocal is only *selected* when
    # total_static > 1, where ``maximum(total_static, 1.0) == total_static`` keeps
    # it bit-identical; without the floor, reverse-mode AD differentiates
    # ``1/total_static`` at 0 -> inf and the where injects ``0*inf = NaN`` into
    # d/d(f_land), d/d(f_lake) for every ocean cell (tile-mask sensitivity /
    # end-to-end adjoint).
    static_scale = jnp.where(
        total_static > 1.0, 1.0 / jnp.maximum(total_static, 1.0), 1.0
    )
    f_land = f_land * static_scale
    f_lake = f_lake * static_scale

    f_water = 1.0 - f_land - f_lake
    f_ice = f_water * jnp.clip(ice_concentration, 0.0, 1.0)
    f_ocean = f_water - f_ice
    return TileFractions(f_ocean=f_ocean, f_ice=f_ice,
                         f_land=f_land, f_lake=f_lake)


def blend_tiles(
    ocean_resp: TileResponse,
    ice_resp: TileResponse,
    land_resp: TileResponse,
    lake_resp: TileResponse,
    fracs: TileFractions,
) -> SurfaceToAtm:
    """Area-weighted blending of tile responses into a single SurfaceToAtm.

    result_field = f_ocean * ocean + f_ice * ice + f_land * land + f_lake * lake

    Fractions are continuous floats — no hard conditionals.
    """
    fo = fracs.f_ocean
    fi = fracs.f_ice
    fl = fracs.f_land
    fk = fracs.f_lake
    # Water fraction (ocean + ice).  The sea-ice tile returns its ice->ocean
    # mass/energy EXCHANGE fluxes (freshwater / heat / salt) as PER-GRID-CELL
    # (per-water-area) budgets — it has already summed the per-process,
    # per-category area weights internally — so they are blended with the full
    # water fraction, NOT ``f_ice``.  This delivers the exact per-cell budget
    # regardless of how the ice fraction evolved within the step (melt retreat,
    # terminal melt-out, new-ice formation, ridging, ITD remap), and avoids any
    # 1/conc normalisation that could blow up as a cell melts out.  F11.
    f_water = fo + fi

    def _blend(o, i, l, k):
        return fo * o + fi * i + fl * l + fk * k

    def _blend_exchange(o, i, l, k):
        # Ice term is per-grid-cell (per-water-area) -> weight by f_water; the
        # other tiles' contributions (ocean P-E, land runoff, lake P-E) are
        # per-tile-area and keep their own fractions.
        return fo * o + f_water * i + fl * l + fk * k

    # Radiative-equivalent skin temperature for the LW boundary.  Blend the
    # per-tile EMISSION FLUX eps_i*sigma*T_rad_i^4 (NOT T_sfc_i), then invert with
    # eps_grid = sum_i f_i*eps_i so the atmosphere boundary
    # eps_grid*sigma*T_rad^4 + (1-eps_grid)*La equals sum_i f_i*lw_up_i EXACTLY.
    # Area-averaging T and eps independently is not LW-flux-conserving for mixed
    # cells (T^4 and eps*T^4 are nonlinear); T_rad == T_sfc for single-tile cells.
    # ``T_rad_i`` is each tile's emission-equivalent temperature: for most tiles
    # the skin temperature, but for the two-leaf canopy the canopy column
    # temperature (T_sfc there is the SOIL temperature, used only for sensible
    # heat) — so we MUST use the tile's ``T_rad`` when present, not ``T_sfc``.
    _sb = constants.sigma_sb

    def _T_rad(r):
        tr = getattr(r, "T_rad", None)
        return r.T_sfc if tr is None else tr

    eps_grid = _blend(ocean_resp.emissivity, ice_resp.emissivity,
                      land_resp.emissivity, lake_resp.emissivity)
    lw_emit_grid = _blend(
        ocean_resp.emissivity * _sb * _T_rad(ocean_resp) ** 4,
        ice_resp.emissivity * _sb * _T_rad(ice_resp) ** 4,
        land_resp.emissivity * _sb * _T_rad(land_resp) ** 4,
        lake_resp.emissivity * _sb * _T_rad(lake_resp) ** 4,
    )
    T_rad = (lw_emit_grid / jnp.maximum(eps_grid * _sb, 1.0e-12)) ** 0.25

    _ice_lhflx_cell = ice_resp.lhflx_exchange
    if _ice_lhflx_cell is None:
        raise ValueError(
            "blend_tiles: the sea-ice tile response carries no lhflx_exchange "
            "(the realized per-cell latent paired with surface_mass_flux); "
            "every sea_ice response constructor sets it -- a hand-built ice "
            "tile must too, or the atmosphere gets water without its heat.")
    return SurfaceToAtm(
        T_sfc=_blend(ocean_resp.T_sfc, ice_resp.T_sfc,
                         land_resp.T_sfc, lake_resp.T_sfc),
        T_rad=T_rad,
        albedo=_blend(ocean_resp.albedo, ice_resp.albedo,
                      land_resp.albedo, lake_resp.albedo),
        emissivity=eps_grid,
        z0=_blend(ocean_resp.z0, ice_resp.z0,
                  land_resp.z0, lake_resp.z0),
        q_surface=_blend(ocean_resp.q_surface, ice_resp.q_surface,
                         land_resp.q_surface, lake_resp.q_surface),
        shflx=_blend(ocean_resp.shflx, ice_resp.shflx,
                     land_resp.shflx, lake_resp.shflx),
        # Latent heat flux.  The SEA-ICE latent term is delivered on the SAME
        # realized per-cell / f_water basis as its moisture (surface_mass_flux):
        # L_s * (per-cell sublimation mass), so the atmosphere's ice latent
        # ENERGY and water MASS agree exactly and both survive terminal melt-out
        # (post-step f_ice -> 0 would otherwise drop the energy while the mass,
        # blended by f_water, is delivered).  Equals f_ice*ice_lhflx in the
        # steady no-clamp limit (L_s*surface_mass_flux = lhflx*conc_pre).  Ocean
        # evap / land ET / lake evap keep their own instantaneous fractions
        # (#28, codex; mirrors the F11 exchange convention).
        #
        # NOTE (bounded surface-energy residual, tracked with #28): the sea-ice
        # surface energy balance inside step_sea_ice forms Q_sfc with the
        # UNCAPPED bulk lhflx, while this delivers the REALIZED (capped) latent
        # energy to the atmosphere.  When the over-ablation cap fires the two
        # differ by L_s*(uncapped - realized) sublimation -- bounded by the thin
        # column's latent capacity (L_s*rho_ice*h/dt) and only on sub-cm clamped
        # ice -- so the atmosphere energy & water now pair exactly but the ice
        # thermo solve was cooled by slightly more latent than the atmosphere
        # received.  Closing this exactly needs an implicit latent-cap in the
        # T_new solve (a thermo rework); the mass budget IS closed here.
        # The ice tile publishes its REALIZED per-cell latent heat (sum over
        # categories of L_s(T_k) * m_k, ``lhflx_exchange``) beside its mass flux,
        # so the blend takes it on the same exchange basis as the mass.  Its
        # ``lhflx`` is PER ICE AREA and cannot be used here (off by the
        # concentration basis).
        lhflx=(fo * ocean_resp.lhflx
               + f_water * _ice_lhflx_cell
               + fl * land_resp.lhflx + fk * lake_resp.lhflx),
        tau_x=_blend(ocean_resp.tau_x, ice_resp.tau_x,
                     land_resp.tau_x, lake_resp.tau_x),
        tau_y=_blend(ocean_resp.tau_y, ice_resp.tau_y,
                     land_resp.tau_y, lake_resp.tau_y),
        lw_up=_blend(ocean_resp.lw_up, ice_resp.lw_up,
                     land_resp.lw_up, lake_resp.lw_up),
        u_ocean_sfc=_blend(ocean_resp.u_ocean_sfc, ice_resp.u_ocean_sfc,
                           land_resp.u_ocean_sfc, lake_resp.u_ocean_sfc),
        v_ocean_sfc=_blend(ocean_resp.v_ocean_sfc, ice_resp.v_ocean_sfc,
                           land_resp.v_ocean_sfc, lake_resp.v_ocean_sfc),
        co2_flux=_blend(ocean_resp.co2_flux, ice_resp.co2_flux,
                        land_resp.co2_flux, lake_resp.co2_flux),
        # Tile-blended freshwater flux to the ocean.  Each tile populates
        # freshwater_flux (zeros for tiles that don't deliver freshwater).
        # The ice tile's melt/freeze freshwater is a mass TRANSFER from the
        # ice present during the step, so it carries the exchange-area weight
        # (F11); the ocean P-E / land runoff / lake P-E terms keep their normal
        # fractions.
        freshwater_flux=_blend_exchange(
            ocean_resp.freshwater_flux, ice_resp.freshwater_flux,
            land_resp.freshwater_flux, lake_resp.freshwater_flux,
        ),
        # Tile-blended heat extracted from the ocean by ice (ice-only channel;
        # an energy TRANSFER -> exchange-area weight).  Audit F8 / F11.
        ocean_heat_extraction=_blend_exchange(
            ocean_resp.ocean_heat_extraction,
            ice_resp.ocean_heat_extraction,
            land_resp.ocean_heat_extraction,
            lake_resp.ocean_heat_extraction,
        ),
        # Tile-blended back-reaction stress on the ocean.  Audit F9.
        ocean_stress_x=_blend(
            ocean_resp.ocean_stress_x, ice_resp.ocean_stress_x,
            land_resp.ocean_stress_x, lake_resp.ocean_stress_x,
        ),
        ocean_stress_y=_blend(
            ocean_resp.ocean_stress_y, ice_resp.ocean_stress_y,
            land_resp.ocean_stress_y, lake_resp.ocean_stress_y,
        ),
        # Phase-aware blended surface mass flux (Audit F3).  The sea-ice tile
        # returns its REALIZED sublimation/deposition mass as a PER-GRID-CELL
        # flux (already weighted by the pre-step ice fraction inside the kernel),
        # so it carries the exchange-area weight f_water like the other ice
        # transfer channels -- this delivers the ice's atmosphere water loss in
        # full even at terminal melt-out (post-step f_ice -> 0 would otherwise
        # drop it), keeping the ice + ocean + atmosphere water budget closed
        # (#28).  Ocean P-E / land ET / lake P-E keep their own fractions.
        surface_mass_flux=_blend_exchange(
            ocean_resp.surface_mass_flux, ice_resp.surface_mass_flux,
            land_resp.surface_mass_flux, lake_resp.surface_mass_flux,
        ),
        # Salt-flux blend (only sea-ice tile is non-zero; brine rejection /
        # melt is a salt-mass TRANSFER -> exchange-area weight).  F11.
        salt_flux=_blend_exchange(
            ocean_resp.salt_flux, ice_resp.salt_flux,
            land_resp.salt_flux, lake_resp.salt_flux,
        ),
        # River runoff sub-component (LAND tile only) — the depth-spreadable
        # part of ``freshwater_flux``.  Ocean P-E, ice melt and lake P-E are
        # surface (top-cell) fluxes and are NOT included here.  f_land weight
        # matches the land term inside ``freshwater_flux`` above.
        river_runoff_flux=fl * land_resp.freshwater_flux,
        # Surface non-ocean freshwater = ice melt (exchange-area weight f_water,
        # F11) + lake P-E (f_lake).  Same weights as inside ``freshwater_flux``;
        # excludes ocean P-E (kept current by the consumer) and land runoff
        # (depth-spread channel above).
        ice_lake_freshwater_flux=(
            f_water * ice_resp.freshwater_flux + fk * lake_resp.freshwater_flux
        ),
    )
