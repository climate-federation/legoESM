"""Main surface coupler: factory and step function.

The coupler steps all surface tiles, blends their responses, and
accumulates fluxes for asynchronous coupling. It never accesses
full atmospheric state — only AtmToSurface.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.core.field import Field
from legoesm.coupler.accumulator import (
    FluxAccumulator,
    accumulate,
    reset_accumulator,
)
from legoesm.coupler.config import CouplerConfig, TileConfig
from legoesm.coupler.coupling_fields import (
    AtmToSurface,
    SurfaceToAtm,
    TileResponse,
)
from legoesm.coupler.lake import LakeConfig, LakeState, step_lake
from legoesm.coupler.tile_fractions import (
    TileFractions,
    blend_tiles,
    compute_tile_fractions,
)
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.state import SeaIceState
from legoesm.land.config import LandConfig
from legoesm.land.slab_land import step_land
from legoesm.land.state import LandState


class SurfaceState(NamedTuple):
    """Combined surface state for all tiles."""
    land: LandState
    ice: SeaIceState
    lake: LakeState
    accumulator: FluxAccumulator


def init_surface_state(
    shape: tuple[int, ...],
    T_soil_init: float = 280.0,
    W_bucket_init: float = 75.0,
    T_epi_init: float = 285.0,
    T_hypo_init: float = 278.0,
    T_ice_init: float = 260.0,
) -> SurfaceState:
    """Initialize all surface tile states.

    Parameters
    ----------
    shape : tuple
        Spatial shape, typically (6, n, n).
    """
    dims_2d = ("face", "x", "y")

    land = LandState(
        T_soil=Field(data=jnp.full(shape, T_soil_init),
                     name="T_soil", dims=dims_2d, units="K"),
        W_bucket=Field(data=jnp.full(shape, W_bucket_init),
                       name="W_bucket", dims=dims_2d, units="kg/m2"),
    )

    ice = SeaIceState(
        h_ice=Field(data=jnp.zeros(shape),
                    name="h_ice", dims=dims_2d, units="m"),
        T_ice=Field(data=jnp.full(shape, T_ice_init),
                    name="T_ice", dims=dims_2d, units="K"),
        concentration=Field(data=jnp.zeros(shape),
                           name="ice_concentration", dims=dims_2d, units="1"),
    )

    lake = LakeState(
        T_epi=Field(data=jnp.full(shape, T_epi_init),
                    name="T_epi", dims=dims_2d, units="K"),
        T_hypo=Field(data=jnp.full(shape, T_hypo_init),
                     name="T_hypo", dims=dims_2d, units="K"),
    )

    acc = reset_accumulator(shape)

    return SurfaceState(land=land, ice=ice, lake=lake, accumulator=acc)


def ocean_tile_response(
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: CouplerConfig,
) -> TileResponse:
    """Compute surface response for the ocean tile.

    Ocean provides SST with fixed albedo/emissivity. Bulk fluxes
    are computed here using the coupler's ocean exchange coefficients.
    """
    shape = ocean_sst.shape
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + config.U_min ** 2
    )

    q_sfc = saturation_mixing_ratio(ocean_sst, forcing.p_surface)
    rho = forcing.rho_lowest

    tau_x = -rho * config.Cd_ocean * wind_speed * forcing.u_lowest
    tau_y = -rho * config.Cd_ocean * wind_speed * forcing.v_lowest
    shflx = rho * constants.c_pd * config.Ch_ocean * wind_speed * (ocean_sst - forcing.T_lowest)
    lhflx = rho * constants.L_v * config.Ch_ocean * wind_speed * (q_sfc - forcing.q_lowest)
    lw_up = config.ocean_emissivity * constants.sigma_sb * ocean_sst ** 4

    return TileResponse(
        T_surface=ocean_sst,
        albedo=jnp.broadcast_to(jnp.array(config.ocean_albedo), shape),
        emissivity=jnp.broadcast_to(jnp.array(config.ocean_emissivity), shape),
        z0=jnp.broadcast_to(jnp.array(config.ocean_z0), shape),
        q_surface=q_sfc,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up,
        u_ocean_sfc=ocean_u,
        v_ocean_sfc=ocean_v,
        co2_flux=jnp.zeros(shape),
    )


def make_coupler(
    coupler_config: CouplerConfig,
    land_config: LandConfig,
    ice_config: SeaIceConfig,
    lake_config: LakeConfig,
):
    """Factory that returns step_surface function.

    Returns
    -------
    step_surface : callable
        (SurfaceState, AtmToSurface, TileConfig, ocean_sst, ocean_u,
         ocean_v, dt) -> (SurfaceState, SurfaceToAtm)
    """
    U_min = coupler_config.U_min

    def step_surface(
        sfc_state: SurfaceState,
        atm_forcing: AtmToSurface,
        tile_config: TileConfig,
        ocean_sst: jnp.ndarray,
        ocean_u_sfc: jnp.ndarray,
        ocean_v_sfc: jnp.ndarray,
        dt: float,
    ) -> tuple[SurfaceState, SurfaceToAtm]:
        """Step all surface tiles and return blended response."""

        # 1. Step land
        land_new, land_resp = step_land(
            sfc_state.land, atm_forcing, land_config, U_min, dt)

        # 2. Step sea ice
        ice_new, ice_resp = step_sea_ice(
            sfc_state.ice, atm_forcing, ocean_sst, ocean_u_sfc,
            ocean_v_sfc, ice_config, U_min, dt)

        # 3. Step lake
        lake_new, lake_resp = step_lake(
            sfc_state.lake, atm_forcing, lake_config, U_min, dt)

        # 4. Ocean tile (diagnostic — ocean model handles its own state)
        ocean_resp = ocean_tile_response(
            atm_forcing, ocean_sst, ocean_u_sfc, ocean_v_sfc,
            coupler_config)

        # 5. Tile fractions (ice concentration from updated ice state)
        fracs = compute_tile_fractions(tile_config, ice_new.concentration.data)

        # 6. Blend
        blended = blend_tiles(ocean_resp, ice_resp, land_resp, lake_resp, fracs)

        # 7. Accumulate
        acc_new = accumulate(sfc_state.accumulator, blended, dt)

        new_state = SurfaceState(
            land=land_new, ice=ice_new, lake=lake_new, accumulator=acc_new)

        return new_state, blended

    return step_surface
