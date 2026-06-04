"""Coupled Earth System Driver.

Orchestrates atmosphere + coupler (land, sea ice, lake) + optional
ocean model into a single time integration. Builds on the existing
``ModelDriver`` for atmosphere and ``make_coupler`` for surface exchange.

The coupler step runs at segment boundaries (host-side), via a callback
from ``ModelDriver.run()``.  This is a split-step approach where
atmosphere and surface exchange alternate each diagnostic interval.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax.numpy as jnp

from legoesm import constants
from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.config import ExperimentConfig

logger = logging.getLogger("legoesm.driver.earth_system")


class EarthSystemDriver:
    """Coupled atmosphere + surface driver.

    Uses a ``ModelDriver`` for the atmosphere component and
    ``make_coupler`` for land/ice/lake surface exchange.  The coupler
    is invoked at segment boundaries after each atmosphere segment.

    Parameters
    ----------
    config : ExperimentConfig
        Atmosphere experiment configuration.
    coupler_config : CouplerConfig, optional
    output_dir : str or Path, optional
    """

    def __init__(
        self,
        config: ExperimentConfig,
        coupler_config=None,
        land_config=None,
        ice_config=None,
        lake_config=None,
        output_dir=None,
    ):
        self.config = config
        self._atm = ModelDriver(config, output_dir=output_dir)
        self._coupler_config = coupler_config
        self._land_config = land_config
        self._ice_config = ice_config
        self._lake_config = lake_config
        self._step_surface = None
        self._sfc_state = None
        self._tile_config = None
        self._coupler_cfg = None

    @property
    def output_dir(self) -> Path:
        return self._atm.output_dir

    def setup(self) -> None:
        """Initialize all components: atmosphere, coupler, surface state."""
        # 1. Atmosphere setup (grid, dycore, physics, state, diagnostics)
        self._atm.setup()

        # 2. Coupler setup
        from legoesm.coupler.coupler import make_coupler, init_surface_state
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.land.config import LandConfig
        from legoesm.ice.config import SeaIceConfig
        from legoesm.coupler.lake.config import LakeConfig

        self._coupler_cfg = self._coupler_config or CouplerConfig()
        land_cfg = self._land_config or LandConfig()
        ice_cfg = self._ice_config or SeaIceConfig()
        lake_cfg = self._lake_config or LakeConfig()

        # Build coupler step function
        self._step_surface = make_coupler(
            self._coupler_cfg, land_cfg, ice_cfg, lake_cfg,
            lat=self._atm._grid_lat,
            grid=self._atm.grid,
        )

        # Initialize surface state
        shape_2d = self._atm.grid.grid_shape_2d
        self._sfc_state = init_surface_state(
            shape_2d, land_config=land_cfg,
        )

        # Tile fractions
        from legoesm.core.precision import get_policy
        _sd = get_policy().storage
        f_land = self._atm._f_land if self._atm._f_land is not None else jnp.zeros(shape_2d, dtype=_sd)
        self._tile_config = TileConfig(
            f_land=f_land,
            f_lake=jnp.zeros(shape_2d, dtype=_sd),
        )

        logger.info("  EarthSystem: atmosphere + coupler initialized")

    def _build_atm_forcing(self, day: float):
        """Build AtmToSurface coupling fields from atmosphere state and physics."""
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.forcing.surface_utils import blend_surface_temperature

        state = self._atm.state
        q_v = self._atm.q_v
        p_s = state.p_s.data
        T_low = state.T.data[..., -1]
        u_low = state.u.data[..., -1]
        v_low = state.v.data[..., -1]
        q_low = q_v[..., -1] if q_v is not None else jnp.zeros_like(T_low)
        sigma_full = jnp.asarray(self._atm.sigma.sigma_full)
        p_low = p_s * sigma_full[-1]
        rho_low = p_low / (constants.R_d * T_low)

        # Extract real radiation and precipitation from last atmosphere physics
        aux = getattr(self._atm, '_carry_aux', {})
        sw_net_sfc = aux.get("held_sw_net_sfc", jnp.zeros_like(p_s))
        lw_net_sfc = aux.get("held_lw_net_sfc", jnp.zeros_like(p_s))
        seg_precip = aux.get("seg_precip", jnp.zeros_like(p_s))

        # Reconstruct gross downward fluxes from net fluxes.
        # sw_net = sw_down * (1 - albedo) → sw_down = sw_net / (1 - albedo)
        # Use the atmosphere's effective surface albedo for reconstruction.
        cfg = self.config
        sst, sic = self._atm.get_sst_sic(day)
        from legoesm.forcing.surface_utils import blend_surface_property
        albedo_eff = blend_surface_property(
            sic,
            getattr(cfg, 'albedo_ice', 0.6),
            getattr(cfg, 'albedo_ocean', 0.06),
        )
        sw_down = sw_net_sfc / jnp.maximum(1.0 - albedo_eff, 0.01)

        # lw_net = eps * lw_down - eps * sigma * T_sfc^4
        # lw_down = (lw_net + eps * sigma * T_sfc^4) / eps
        T_sfc = blend_surface_temperature(sst, sic, cfg.T_ice)
        eps_sfc = 0.96  # typical surface emissivity
        lw_up_sfc = eps_sfc * constants.sigma_sb * T_sfc ** 4
        lw_down = (lw_net_sfc + lw_up_sfc) / jnp.maximum(eps_sfc, 0.01)

        # Snow fraction: approximate from T_lowest < freezing
        precip_total = jnp.maximum(seg_precip, 0.0)
        snow_frac = jnp.where(T_low < constants.T_freeze, 1.0, 0.0)
        precip_snow = precip_total * snow_frac

        # Cosine zenith: daily-mean approximation cos_zen = Q / S_0
        from legoesm.forcing.time_utils import day_to_calendar
        doy, _ = day_to_calendar(day)
        lat = self._atm._grid_lat
        if lat is not None:
            from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
            S_0 = getattr(cfg, 'S_0', 1360.0)
            Q_daily = daily_mean_insolation(lat, float(doy), S_0=S_0)
            cos_zen = jnp.clip(Q_daily / S_0, 0.0, 1.0)
        else:
            cos_zen = jnp.full_like(p_s, 0.5)

        return AtmToSurface(
            sw_down=sw_down,
            lw_down=lw_down,
            precip_total=precip_total,
            precip_snow=precip_snow,
            T_lowest=T_low,
            q_lowest=q_low,
            u_lowest=u_low,
            v_lowest=v_low,
            p_lowest=p_low,
            p_surface=p_s,
            rho_lowest=rho_low,
            cos_zenith=cos_zen,
            co2_ppmv=jnp.full_like(p_s, self.config.co2_ppmv),
            has_radiation=jnp.ones_like(p_s),
            has_precipitation=jnp.where(precip_total > 0, 1.0, 0.0),
        )

    def _step_coupler(self, day: float, dt: float):
        """Execute one coupler step: land + ice + lake surface exchange."""
        atm_forcing = self._build_atm_forcing(day)

        # Get ocean SST for ice coupling
        sst, _ = self._atm.get_sst_sic(day)

        from legoesm.forcing.time_utils import day_to_calendar
        doy, _ = day_to_calendar(day)

        self._sfc_state, sfc_response = self._step_surface(
            self._sfc_state,
            atm_forcing,
            self._tile_config,
            ocean_sst=sst,
            ocean_u_sfc=jnp.zeros_like(sst),
            ocean_v_sfc=jnp.zeros_like(sst),
            dt=dt,
            doy=float(doy),
        )

        return sfc_response

    def _segment_hook(self, driver, day, dt_segment):
        """Callback invoked at each segment boundary by ModelDriver.

        Runs the coupler and feeds surface temperature back to the
        atmosphere for the next segment.
        """
        if self._step_surface is None:
            return

        sfc_response = self._step_coupler(day, dt_segment)

        # Feed surface response back: update atmosphere's land surface
        # temperature override if available.  This is the primary feedback
        # mechanism — the coupler's blended T_sfc influences the next
        # atmosphere segment's boundary layer computation.
        if hasattr(driver, '_sfc_T_override'):
            driver._sfc_T_override = sfc_response.T_sfc
        # Store last surface response for diagnostics
        self._last_sfc_response = sfc_response

    def run(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run the coupled integration.

        Runs the atmosphere via ModelDriver with a segment callback
        that invokes the coupler at each diagnostic interval boundary.
        """
        logger.info("Starting coupled Earth System run")

        # Run atmosphere with coupler callback at each segment boundary
        status = self._atm.run(
            start_step=start_step,
            start_day=start_day,
            segment_callback=self._segment_hook,
        )

        logger.info(f"Earth System run: {status}")
        return status

    @property
    def state(self):
        return self._atm.state

    @property
    def surface_state(self):
        return self._sfc_state

    @property
    def diagnostics(self):
        return self._atm.diagnostics

    def save_checkpoint(self, step: int, day: float) -> None:
        self._atm.save_checkpoint(step, day)
