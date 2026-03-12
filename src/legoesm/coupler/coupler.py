"""Main surface coupler: factory and step function.

The coupler steps all surface tiles, blends their responses, and
accumulates fluxes for asynchronous coupling. It never accesses
full atmospheric state — only AtmToSurface.
"""

from __future__ import annotations

from typing import NamedTuple
import warnings

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
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
from legoesm.coupler.coupling_fields import (
    AtmToSurface,
    SurfaceToAtm,
    TileResponse,
)
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
from legoesm.land.config import LandConfig
from legoesm.land.slab_land import step_land
from legoesm.land.state import LandState


class SurfaceState(NamedTuple):
    """Combined surface state for all tiles."""
    land: LandState
    ice: SeaIceState
    lake: LakeState
    accumulator: FluxAccumulator
    carbon: CarbonState | None = None


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
    if config.blend_sharpness != 20.0:
        warnings.warn(
            "CouplerConfig.blend_sharpness is currently unused in blend_tiles().",
            RuntimeWarning,
            stacklevel=2,
        )


def init_surface_state(
    shape: tuple[int, ...],
    T_soil_init: float = 280.0,
    W_bucket_init: float = 75.0,
    T_epi_init: float = 285.0,
    T_hypo_init: float = 278.0,
    T_ice_init: float = 260.0,
    land_config: LandConfig | None = None,
) -> SurfaceState:
    """Initialize all surface tile states.

    Parameters
    ----------
    shape : tuple
        Spatial shape, typically (6, n, n).
    land_config : LandConfig, optional
        If provided and ``land_config.carbon.scheme == "differland"``,
        initialises prognostic carbon pools.
    """
    dims_2d = ("face", "x", "y")

    land = LandState(
        T_soil=Field(data=jnp.full(shape, T_soil_init),
                     name="T_soil", dims=dims_2d, units="K"),
        W_bucket=Field(data=jnp.full(shape, W_bucket_init),
                       name="W_bucket", dims=dims_2d, units="kg/m2"),
        snow_depth=Field(data=jnp.zeros(shape),
                         name="snow_depth", dims=dims_2d, units="kg/m2"),
        snow_age=Field(data=jnp.zeros(shape),
                       name="snow_age", dims=dims_2d, units="s"),
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

    # Carbon pools (only for differland scheme)
    carbon = None
    if land_config is not None and land_config.carbon.scheme == "differland":
        carbon = init_carbon_state(shape, land_config.carbon)

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
    q_sfc = saturation_mixing_ratio(ocean_sst, forcing.p_surface)
    rho = forcing.rho_lowest
    lw_up = config.ocean_emissivity * constants.sigma_sb * ocean_sst ** 4

    if config.bulk_scheme in ("coare3", "large_yeager"):
        from legoesm.coupler.bulk_flux import compute_most_fluxes
        # Use wind relative to ocean surface current
        u_rel = forcing.u_lowest - ocean_u
        v_rel = forcing.v_lowest - ocean_v
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            u_rel, v_rel,
            forcing.T_lowest, forcing.q_lowest,
            ocean_sst, q_sfc,
            rho,
            z_ref=config.z_ref,
            z0_init=config.ocean_z0,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
        )
    else:
        # Constant neutral coefficients (original behavior)
        wind_speed = jnp.sqrt(
            forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + config.U_min ** 2
        )
        tau_x = -rho * config.Cd_ocean * wind_speed * forcing.u_lowest
        tau_y = -rho * config.Cd_ocean * wind_speed * forcing.v_lowest
        shflx = rho * constants.c_pd * config.Ch_ocean * wind_speed * (ocean_sst - forcing.T_lowest)
        lhflx = rho * constants.L_v * config.Ch_ocean * wind_speed * (q_sfc - forcing.q_lowest)

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

    return TileResponse(
        T_surface=ocean_sst,
        albedo=alpha_ocean,
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
    lat: jnp.ndarray | None = None,
    grid=None,
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

        # 1. Step land (with optional carbon cycle)
        land_new, land_resp, carbon_new = step_land(
            sfc_state.land, atm_forcing, land_config, U_min, dt,
            lat=_lat, carbon_state=sfc_state.carbon, doy=doy,
        )

        # 2. Step sea ice
        ice_new, ice_resp = step_sea_ice(
            sfc_state.ice, atm_forcing, ocean_sst, ocean_u_sfc,
            ocean_v_sfc, ice_config, U_min, dt, grid=_grid)

        # 3. Step lake
        lake_new, lake_resp = step_lake(
            sfc_state.lake, atm_forcing, lake_config, U_min, dt)

        # 4. Ocean tile (diagnostic — ocean model handles its own state)
        ocean_resp = ocean_tile_response(
            atm_forcing, ocean_sst, ocean_u_sfc, ocean_v_sfc,
            coupler_config)

        # 5. Tile fractions (ice concentration from updated ice state)
        ice_conc = ice_new.concentration.data
        # Multi-category: sum across categories for total concentration
        if ice_conc.ndim > len(atm_forcing.sw_down.shape):
            ice_conc = jnp.sum(ice_conc, axis=-1)
        fracs = compute_tile_fractions(tile_config, ice_conc)

        # 6. Blend
        blended = blend_tiles(ocean_resp, ice_resp, land_resp, lake_resp, fracs)

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
            land=land_new, ice=ice_new, lake=lake_new, accumulator=acc_next,
            carbon=carbon_new)

        return new_state, blended_out

    return step_surface
