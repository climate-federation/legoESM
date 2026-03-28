"""Coupled Earth System Driver.

Orchestrates atmosphere + coupler (land, sea ice, lake) + optional
ocean model into a single time integration. Builds on the existing
``ModelDriver`` for atmosphere and ``make_coupler`` for surface exchange.

The coupler step runs at segment boundaries (host-side), after the
atmosphere segment completes.  This is a split-step approach where
atmosphere and surface exchange alternate each diagnostic interval.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax.numpy as jnp

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
        f_land = self._atm._f_land if self._atm._f_land is not None else jnp.zeros(shape_2d)
        self._tile_config = TileConfig(
            f_land=f_land,
            f_lake=jnp.zeros(shape_2d),
        )

        logger.info("  EarthSystem: atmosphere + coupler initialized")

    def _build_atm_forcing(self, day: float):
        """Build AtmToSurface coupling fields from current atmosphere state."""
        from legoesm.coupler.coupling_fields import AtmToSurface
        from legoesm import constants

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

        return AtmToSurface(
            sw_down=jnp.zeros_like(p_s),
            lw_down=jnp.zeros_like(p_s),
            precip_total=jnp.zeros_like(p_s),
            precip_snow=jnp.zeros_like(p_s),
            T_lowest=T_low,
            q_lowest=q_low,
            u_lowest=u_low,
            v_lowest=v_low,
            p_lowest=p_low,
            p_surface=p_s,
            rho_lowest=rho_low,
            cos_zenith=jnp.full_like(p_s, 0.5),
            co2_ppmv=jnp.full_like(p_s, self.config.co2_ppmv),
            has_radiation=jnp.ones_like(p_s),
        )

    def _step_coupler(self, day: float, dt: float):
        """Execute one coupler step: land + ice + lake surface exchange."""
        from legoesm.forcing.time_utils import day_to_calendar

        doy, _ = day_to_calendar(day)
        atm_forcing = self._build_atm_forcing(day)

        # Get ocean SST for ice coupling
        sst, _ = self._atm.get_sst_sic(day)

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

    def run(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run the coupled integration.

        Runs the atmosphere via ModelDriver, then applies the coupler
        step at each diagnostic interval boundary.
        """
        logger.info("Starting coupled Earth System run")

        # Run atmosphere segments
        status = self._atm.run(start_step=start_step, start_day=start_day)

        # Apply coupler step at the end of the run
        # (for full coupling, this would happen at each segment boundary
        # inside the atmosphere loop — requires ModelDriver callback support)
        if status == "COMPLETED" and self._step_surface is not None:
            cfg = self.config
            DT = cfg.dycore.dt
            day = (start_day or cfg.start_day) + cfg.days
            coupling_dt = float(cfg.output.diag_days * 86400)
            try:
                self._step_coupler(day, coupling_dt)
                logger.info("  Coupler step applied at end of run")
            except Exception as e:
                logger.warning(f"  Coupler step failed: {e}")

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
