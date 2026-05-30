"""Tile fraction computation and smooth blending utilities."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.coupler.config import TileConfig
from legoesm.coupler.coupling_fields import SurfaceToAtm, TileResponse


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

    def _blend(o, i, l, k):
        return fo * o + fi * i + fl * l + fk * k

    return SurfaceToAtm(
        T_surface=_blend(ocean_resp.T_surface, ice_resp.T_surface,
                         land_resp.T_surface, lake_resp.T_surface),
        albedo=_blend(ocean_resp.albedo, ice_resp.albedo,
                      land_resp.albedo, lake_resp.albedo),
        emissivity=_blend(ocean_resp.emissivity, ice_resp.emissivity,
                          land_resp.emissivity, lake_resp.emissivity),
        z0=_blend(ocean_resp.z0, ice_resp.z0,
                  land_resp.z0, lake_resp.z0),
        q_surface=_blend(ocean_resp.q_surface, ice_resp.q_surface,
                         land_resp.q_surface, lake_resp.q_surface),
        shflx=_blend(ocean_resp.shflx, ice_resp.shflx,
                     land_resp.shflx, lake_resp.shflx),
        lhflx=_blend(ocean_resp.lhflx, ice_resp.lhflx,
                     land_resp.lhflx, lake_resp.lhflx),
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
        # Tile-blended freshwater flux to the ocean.  Each tile
        # populates freshwater_flux as a populated array (zeros for
        # tiles that don't deliver freshwater), so this blend is a
        # straight area-weighted sum.
        freshwater_flux=_blend(
            ocean_resp.freshwater_flux, ice_resp.freshwater_flux,
            land_resp.freshwater_flux, lake_resp.freshwater_flux,
        ),
        # Tile-blended heat extracted from the ocean by ice.  Audit F8.
        ocean_heat_extraction=_blend(
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
        # Phase-aware blended surface mass flux.  Audit F3.
        surface_mass_flux=_blend(
            ocean_resp.surface_mass_flux, ice_resp.surface_mass_flux,
            land_resp.surface_mass_flux, lake_resp.surface_mass_flux,
        ),
        # Salt-flux blend (only sea-ice tile is non-zero in current
        # implementation; other tiles return zeros).
        salt_flux=_blend(
            ocean_resp.salt_flux, ice_resp.salt_flux,
            land_resp.salt_flux, lake_resp.salt_flux,
        ),
    )
