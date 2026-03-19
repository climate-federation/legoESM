"""Composable model driver for legoESM.

Orchestrates grid creation, vertical coordinate, dycore, physics,
forcing, state initialization, time-stepping, diagnostics, and
checkpointing into a single reusable class.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.forcing.time_utils import day_to_calendar

from legoesm.core.conservation import compute_global_moisture, fix_moisture_hydrostatic
from legoesm.core.tracers import TracerRegistry, make_moisture_registry, init_tracers, clip_positive_definite
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import PhysicsPipeline, build_physics_pipeline
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.io.restart import save_restart, load_restart

logger = logging.getLogger("legoesm.driver")


class ModelDriver:
    """Top-level simulation driver.

    Encapsulates the full AMIP integration workflow: grid creation,
    physics setup, state initialization, time loop, diagnostics,
    and checkpoint/restart.

    Parameters
    ----------
    config : ExperimentConfig
        Complete experiment configuration.
    output_dir : str or Path, optional
        Override output directory. If None, auto-generates.
    """

    def __init__(self, config: ExperimentConfig, output_dir: str | Path | None = None):
        self.config = config
        self.grid = None
        self.sigma = None
        self.model = None
        self.physics = None
        self.state = None
        self.tracers: dict[str, jax.Array] = {}
        self.tracer_registry: TracerRegistry = make_moisture_registry()
        self.get_sst_sic = None
        self.diagnostics = None
        self._phis_data = None
        self._f_land = None
        self._fric_decay = None
        self._qv_smooth_coeff = None

        if output_dir is not None:
            self._output_dir = Path(output_dir)
        elif config.output.output_dir:
            self._output_dir = Path(config.output.output_dir)
        else:
            N = config.grid.resolution
            NLEV = config.grid.nlev
            N_DAYS = config.days
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = Path(f"results/amip/C{N}_L{NLEV}_{N_DAYS}d_{run_id}")

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    # Backward-compatible accessors for individual tracers.
    @property
    def q_v(self) -> jax.Array:
        return self.tracers.get("q_v")

    @q_v.setter
    def q_v(self, value):
        self.tracers["q_v"] = value

    @property
    def q_c(self) -> jax.Array:
        return self.tracers.get("q_c")

    @q_c.setter
    def q_c(self, value):
        self.tracers["q_c"] = value

    @property
    def q_r(self) -> jax.Array:
        return self.tracers.get("q_r")

    @q_r.setter
    def q_r(self, value):
        self.tracers["q_r"] = value

    def setup(self) -> None:
        """Initialize grid, dycore, physics, forcing, and state."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._create_grid()
        self._create_topography()
        self._create_dycore()
        self._create_forcing()
        self._init_state()
        self._create_physics()
        self._setup_external_forcing()
        self._create_diagnostics()
        self._create_friction()
        self._save_config()

    def _create_grid(self) -> None:
        """Create horizontal grid and vertical coordinate."""
        gc = self.config.grid

        if gc.grid_type == "cubed_sphere":
            from legoesm.grids.cubed_sphere import create_cubed_sphere
            self.grid = create_cubed_sphere(gc.resolution)
        elif gc.grid_type == "gaussian":
            from legoesm.grids.gaussian import create_gaussian_grid
            self.grid = create_gaussian_grid(gc.resolution)
        elif gc.grid_type == "latlon":
            from legoesm.grids.latlon import create_latlon_grid
            self.grid = create_latlon_grid(gc.resolution)
        else:
            raise ValueError(f"Unknown grid type: {gc.grid_type}")

        if gc.vertical_coord == "hybrid":
            from legoesm.grids.vertical import make_hybrid_levels
            self.sigma = make_hybrid_levels(
                gc.nlev, p_top_Pa=gc.p_top_Pa, stretching=gc.stretching,
            )
        else:
            from legoesm.grids.vertical import create_sigma_coordinate
            self.sigma = create_sigma_coordinate(gc.nlev)

        logger.info(f"  Grid: {gc.grid_type} {gc.resolution}, "
              f"{gc.nlev} levels ({gc.vertical_coord})")

    def _create_topography(self) -> None:
        """Load or generate topography and land-sea mask."""
        from legoesm.grids.topography import (
            TopographyConfig, load_real_topography,
            gaussian_mountain, phis_from_topography,
            land_mask_from_topography,
        )

        N = self.config.grid.resolution
        topo = self.config.topography

        if self.config.grid.grid_type == "cubed_sphere":
            shape_2d = (6, N, N)
        else:
            shape_2d = (self.grid.n_lat, self.grid.n_lon)

        if topo == "flat":
            self._phis_data = jnp.zeros(shape_2d)
            self._f_land = jnp.zeros(shape_2d)
        elif topo == "gaussian":
            z_s = gaussian_mountain(self.grid)
            self._phis_data = phis_from_topography(z_s)
            self._f_land = land_mask_from_topography(z_s)
        else:
            topo_config = TopographyConfig(
                source="file", path=topo,
                smoothing_passes=self.config.topo_smoothing,
                edge_blend_strength=self.config.topo_edge_blend,
            )
            self._phis_data, self._f_land = load_real_topography(
                self.grid, config=topo_config
            )

    def _create_dycore(self) -> None:
        """Create the dynamical core model."""
        dc = self.config.dycore
        N = self.config.grid.resolution
        dx_min = float(self.grid.dx.min()) / 2.0 if hasattr(self.grid, 'dx') else 1e5
        DT = dc.dt
        HYPERDIFF = dc.hyperdiff_scale * (48 / N) ** 4

        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
        )
        dycore_config = CDGridPrimitiveEquationConfig(
            A_h=0.05 * dx_min ** 2 / DT,
            hyperdiff_coeff=HYPERDIFF, hyperdiff_ps_coeff=0.0,
            use_conservation_fixer=dc.conservation_fixer,
            fix_mass=dc.fix_mass,
        )
        self.model = CDGridPrimitiveEquationModel(self.grid, self.sigma, dycore_config)

        self._hyperdiff = HYPERDIFF
        logger.info(f"  Dycore: {dc.discretization}, dt={DT}s")

    def _create_forcing(self) -> None:
        """Load SST/SIC forcing data."""
        cfg = self.config

        if cfg.dataset == "analytical":
            from legoesm.forcing.analytical import analytical_sst_sic
            lat_deg = np.degrees(np.asarray(self.grid.lat))
            T_ice = cfg.T_ice

            def get_sst_sic(day):
                return analytical_sst_sic(lat_deg, day, T_ice=T_ice)

            self.get_sst_sic = get_sst_sic
        else:
            from legoesm.forcing.amip import (
                AMIPForcingConfig, get_amip_preset,
                load_amip_forcing, get_forcing_at_time,
            )
            if cfg.dataset == "custom":
                forcing_config = AMIPForcingConfig(
                    dataset="custom", path=cfg.forcing_path,
                    sst_var=cfg.sst_var or "sst",
                    sic_var=cfg.sic_var or "sic",
                    sst_offset=cfg.sst_offset, sic_scale=cfg.sic_scale,
                )
            else:
                forcing_config = get_amip_preset(cfg.dataset)._replace(
                    path=cfg.forcing_path
                )

            forcing = load_amip_forcing(forcing_config, self.grid)
            self._forcing = forcing

            def get_sst_sic(day):
                return get_forcing_at_time(forcing, day)

            self.get_sst_sic = get_sst_sic

    def _init_state(self) -> None:
        """Initialize atmospheric state and moisture."""
        from legoesm.atmosphere.physics.held_suarez import held_suarez_init
        from legoesm.diagnostics.column_integrals import column_water_vapor

        cfg = self.config
        N = cfg.grid.resolution
        NLEV = cfg.grid.nlev

        if cfg.grid.grid_type == "cubed_sphere":
            shape_3d = (6, N, N, NLEV)
        else:
            shape_3d = (self.grid.n_lat, self.grid.n_lon, NLEV)

        self.state = held_suarez_init(
            self.grid, self.sigma, T_init=cfg.T_init, phis=self._phis_data
        )

        # Initialize all tracers via registry
        self.tracers = init_tracers(self.tracer_registry, shape_3d)

        # Moisture initialization
        p_full_init = self.state.p_s.data[..., None] * self.sigma.sigma_full
        q_sat_init = saturation_mixing_ratio(self.state.T.data, p_full_init)
        self.tracers["q_v"] = cfg.RH_init * q_sat_init * self.sigma.sigma_full ** 2
        self.tracers["q_v"] = jnp.minimum(self.tracers["q_v"], q_sat_init)

        mean_qv = float(jnp.mean(self.tracers["q_v"])) * 1000.0
        cwv = float(jnp.mean(
            column_water_vapor(self.tracers["q_v"], self.state.p_s.data, self.sigma.dsigma)
        ))
        logger.info(f"  State init: T={cfg.T_init}K, q_v={mean_qv:.2f} g/kg, CWV={cwv:.1f} kg/m2")

    def _create_physics(self) -> None:
        """Build the physics pipeline."""
        self.physics = build_physics_pipeline(self.grid, self.sigma, self.config)
        logger.info(f"  Physics: {self.config.radiation} + SBM convection")

    def _setup_external_forcing(self) -> None:
        """Configure external forcing: solar, ozone, aerosol, CMIP GHG."""
        from legoesm.forcing.external import (
            SolarConfig, OzoneConfig, AerosolConfig,
            get_solar_forcing_at_time, get_ozone_at_time, get_aerosol_at_time,
        )

        cfg = self.config
        self._solar_config = SolarConfig(
            S_0=cfg.S_0, source=cfg.solar_source,
            path=cfg.solar_file, spectral_var=cfg.solar_spectral_var,
        )
        self._use_solar_spectral = (cfg.solar_source == "spectral_file")

        # Ozone external forcing
        self._ozone_ext_active = (cfg.radiation in ("rrtmg", "rrtmgp")
                                  and cfg.ozone_forcing == "external")
        self._ozone_ext_config = OzoneConfig(
            enabled=self._ozone_ext_active,
            source="climatology", path=cfg.ozone_file,
            use_reference_if_missing=True,
        )

        # Aerosol external forcing
        self._aerosol_active = (cfg.radiation in ("rrtmg", "rrtmgp")
                                and cfg.aerosol_forcing == "external")
        self._aerosol_config = AerosolConfig(
            enabled=self._aerosol_active,
            source="climatology", path=cfg.aerosol_file,
            use_reference_if_missing=True,
            reference_aod_550=cfg.aerosol_reference_aod,
            volcanic_enabled=bool(cfg.volcanic_aerosol_file),
            volcanic_path=cfg.volcanic_aerosol_file,
            volcanic_scale=cfg.volcanic_aerosol_scale,
        )

        # Solar init
        solar_init = get_solar_forcing_at_time(self._solar_config, cfg.start_day)
        if self._use_solar_spectral:
            self._solar_weights_template = jnp.asarray(
                solar_init["solar_fraction_by_gpt"]
            )
        else:
            self._solar_weights_template = jnp.array([], dtype=jnp.float64)

        # CMIP experiment GHG override
        self._experiment = cfg.experiment
        self._start_year = cfg.start_year
        if self._experiment:
            from legoesm.forcing.experiments import ghg_at_year
            co2, ch4, n2o = ghg_at_year(self._experiment, self._start_year)
            self.config = cfg._replace(co2_ppmv=co2, ch4_ppbv=ch4, n2o_ppbv=n2o)
            logger.info(f"  CMIP: {self._experiment} (year {self._start_year}), "
                  f"CO2={co2:.1f} ppmv")

    def _precompute_external_forcing(self, day, p_s, lat):
        """Pre-compute ozone/aerosol fields outside JIT boundary."""
        from legoesm.forcing.external import get_ozone_at_time, get_aerosol_at_time
        from legoesm.forcing.surface_utils import distribute_column_aod_to_layers

        nlev = self.sigma.sigma_full.shape[0]
        shape_2d = p_s.shape
        ncol = int(np.prod(np.array(shape_2d)))

        p_full = p_s[..., None] * self.sigma.sigma_full
        p_half = p_s[..., None] * self.sigma.sigma_half
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        lat_col = lat.reshape(ncol)

        o3_vmr = jnp.zeros((ncol, nlev))
        if self._ozone_ext_active:
            o3_vmr = jnp.asarray(get_ozone_at_time(
                self._ozone_ext_config, day,
                lat_grid=lat_col, p_grid=p_full_col,
            ))

        aerosol_od = jnp.zeros((ncol, nlev))
        if self._aerosol_active:
            aerosol_col = get_aerosol_at_time(
                self._aerosol_config, day, lat_grid=lat_col,
            )
            aerosol_od = distribute_column_aod_to_layers(
                jnp.asarray(aerosol_col), p_half_col,
            )

        return o3_vmr, aerosol_od

    def _create_diagnostics(self) -> None:
        """Set up diagnostic collection."""
        self.diagnostics = DiagnosticCollector(
            nlev=self.config.grid.nlev,
            sigma_full=self.sigma.sigma_full,
            dsigma=self.sigma.dsigma,
            monthly_means=self.config.output.monthly_means,
            cmip_output=self.config.output.cmip_output,
            clear_sky_diag=self.config.output.clear_sky_diag,
            n_days=self.config.days,
        )

    def _create_friction(self) -> None:
        """Precompute Rayleigh friction decay factors."""
        cfg = self.config
        DT = cfg.dycore.dt
        sigma_full = self.sigma.sigma_full

        k_f_max = cfg.k_BL_max_per_day / 86400.0
        k_free = cfg.k_free_per_day / 86400.0
        k_f = k_free + k_f_max * jnp.maximum(
            0.0, (sigma_full - cfg.sigma_b) / (1.0 - cfg.sigma_b)
        )
        self._fric_decay = jnp.exp(-k_f * DT)
        self._qv_smooth_coeff = self._hyperdiff * 0.5

    def _save_config(self) -> None:
        """Save experiment config to output directory."""
        amip_cfg = self.config.to_amip_config()
        from legoesm.forcing.amip_config import save_config
        save_config(amip_cfg, self._output_dir / "experiment_config.json")

    def save_checkpoint(self, step: int, day: float) -> None:
        """Save checkpoint to output directory.

        TODO: Migrate to io/restart.py for unified restart API
        """
        from legoesm.forcing.amip_config import save_checkpoint
        elapsed_day = day - self.config.start_day
        ckpt_path = self._output_dir / f"checkpoint_day_{int(elapsed_day):04d}.npz"
        amip_cfg = self.config.to_amip_config()
        save_checkpoint(
            path=ckpt_path, state=self.state, q_v=self.q_v,
            step=step, day=day, config=amip_cfg,
            q_c=self.q_c, q_r=self.q_r,
        )
        logger.info(f"  Checkpoint: {ckpt_path.name}")

    def load_checkpoint(self, path: str | Path) -> tuple[int, float]:
        """Load state from a checkpoint. Returns (step, day).

        TODO: Migrate to io/restart.py for unified restart API
        """
        from legoesm.forcing.amip_config import load_checkpoint
        state, q_v, step, day, _, _, q_c, q_r = load_checkpoint(
            Path(path), self.grid, self.sigma,
        )
        self.state = state
        self.q_v = q_v
        if q_c is not None:
            self.q_c = q_c
        if q_r is not None:
            self.q_r = q_r
        return step, day

    def run(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run the time integration.

        Uses a JIT-compiled unified physics step with ``jax.lax.cond``
        for radiation sub-cycling (held tendencies reused between
        radiation update steps).

        Parameters
        ----------
        start_step : int
            Starting time step (for restart).
        start_day : float, optional
            Starting day (for restart). Defaults to config.start_day.

        Returns
        -------
        str
            Run status ("COMPLETED" or "BLOWUP at day ...").
        """
        from legoesm.core.operators_3d import hyperdiffusion_3d
        from legoesm.forcing.external import get_solar_forcing_at_time

        cfg = self.config
        DT = cfg.dycore.dt
        N_DAYS = cfg.days
        START_DAY = start_day if start_day is not None else cfg.start_day
        MICROPHYSICS = cfg.microphysics
        RAD_UPDATE_STEPS = cfg.rad_update_steps

        n_steps_total = int(N_DAYS * 86400 / DT)
        diag_interval = int(cfg.output.diag_days * 86400 / DT)
        checkpoint_interval = (
            int(cfg.output.checkpoint_days * 86400 / DT)
            if cfg.output.checkpoint_days > 0 else 0
        )

        sigma_full = self.sigma.sigma_full
        dsigma = self.sigma.dsigma

        if cfg.grid.grid_type == "cubed_sphere":
            N = cfg.grid.resolution
            shape_2d = (6, N, N)
            shape_3d = (6, N, N, cfg.grid.nlev)
        else:
            shape_2d = (self.grid.n_lat, self.grid.n_lon)
            shape_3d = (*shape_2d, cfg.grid.nlev)

        ncol = int(np.prod(np.array(shape_2d)))
        nlev = cfg.grid.nlev

        # Solar forcing (initial)
        solar_init = get_solar_forcing_at_time(self._solar_config, START_DAY)
        current_s_0 = float(solar_init["tsi"])
        solar_weights = (
            jnp.asarray(solar_init["solar_fraction_by_gpt"])
            if self._use_solar_spectral
            else self._solar_weights_template
        )

        # Build JIT-compiled unified physics step
        step_unified = self.physics.build_step_unified()

        # Held radiation tendencies
        held_dT_rad = jnp.zeros(shape_3d)
        held_sw_net_sfc = jnp.zeros(shape_2d)
        held_lw_net_sfc = jnp.zeros(shape_2d)
        held_sw_up_toa = jnp.zeros(shape_2d)
        held_lw_up_toa = jnp.zeros(shape_2d)
        held_sw_down_toa = jnp.zeros(shape_2d)

        # External forcing (pre-compute outside JIT)
        o3_vmr, aerosol_od = self._precompute_external_forcing(
            START_DAY, self.state.p_s.data, self.grid.lat,
        )

        # Moisture conservation fixer
        FIX_MOISTURE = cfg.fix_moisture
        if FIX_MOISTURE:
            target_moisture = compute_global_moisture(
                self.q_v, self.state.p_s.data, dsigma, self.grid,
            )

        run_status = "COMPLETED"
        lat_deg_grid = np.degrees(np.asarray(self.grid.lat))

        logger.info(f"Starting: {n_steps_total - start_step} steps, {N_DAYS} days")

        # --- JIT warmup ---
        t_jit_start = time.time()
        day = START_DAY + (start_step + 1) * DT / 86400.0
        day_of_year, seconds_of_day = day_to_calendar(day)
        sst, sic = self.get_sst_sic(day)

        self.state = self.model.step_with_physics(self.state, DT)

        phys_out, (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa) = \
            step_unified(
                jnp.bool_(True),
                self.state.T.data, self.state.p_s.data,
                self.q_v, self.q_c, self.q_r,
                self.state.u.data, self.state.v.data,
                sst, sic, self.grid.lat, self.grid.lon,
                day_of_year, seconds_of_day, DT,
                solar_weights, current_s_0,
                o3_vmr, aerosol_od,
                held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
            )

        # Apply warmup tendencies
        new_T = self.state.T.data + DT * phys_out.dT_dt
        self.q_v = jnp.maximum(self.q_v + DT * phys_out.dq_v_dt, 0.0)
        self.q_c = jnp.maximum(self.q_c + DT * phys_out.dq_c_dt, 0.0)
        self.q_r = jnp.maximum(self.q_r + DT * phys_out.dq_r_dt, 0.0)

        if MICROPHYSICS == "none":
            p_full = self.state.p_s.data[..., None] * sigma_full
            q_sat = saturation_mixing_ratio(new_T, p_full)
            excess = jnp.maximum(self.q_v - q_sat, 0.0)
            self.q_v = self.q_v - excess
            new_T = new_T + constants.L_v * excess / constants.c_pd

        self.state = self.state._replace(T=self.state.T.replace(data=new_T))
        self.q_v = jnp.maximum(
            self.q_v + DT * hyperdiffusion_3d(self.q_v, self.grid, self._qv_smooth_coeff), 0.0
        )
        self.state = self.state._replace(
            u=self.state.u.replace(data=self.state.u.data * self._fric_decay),
            v=self.state.v.replace(data=self.state.v.data * self._fric_decay),
        )

        jax.block_until_ready(self.state.u.data)
        t_jit = time.time() - t_jit_start
        logger.info(f"  JIT compiled in {t_jit:.1f}s")

        # --- Main time loop ---
        t_start = time.time()

        for step in range(start_step + 1, n_steps_total):
            day = START_DAY + (step + 1) * DT / 86400.0
            day_of_year, seconds_of_day = day_to_calendar(day)

            sst, sic = self.get_sst_sic(day)

            # (a) Dynamics
            self.state = self.model.step_with_physics(self.state, DT)

            # (b) Physics with radiation sub-cycling
            need_rad_py = (RAD_UPDATE_STEPS <= 1) or ((step + 1) % RAD_UPDATE_STEPS == 0)
            need_rad_jax = jnp.bool_(need_rad_py)

            # Update external forcing on radiation steps
            if need_rad_py:
                # Solar
                solar_now = get_solar_forcing_at_time(self._solar_config, day)
                current_s_0 = float(solar_now["tsi"])
                if self._use_solar_spectral:
                    solar_weights = jnp.asarray(solar_now["solar_fraction_by_gpt"])

                # Ozone + aerosol
                o3_vmr, aerosol_od = self._precompute_external_forcing(
                    day, self.state.p_s.data, self.grid.lat,
                )

                # CMIP GHG trajectory
                if self._experiment and cfg.radiation in ("rrtmg", "rrtmgp"):
                    from legoesm.forcing.experiments import ghg_at_year
                    current_year = self._start_year + day / 365.0
                    ghg = ghg_at_year(self._experiment, current_year)
                    # GHG override passed via config to radiation_fn at build time;
                    # for transient experiments the pipeline already uses config defaults.

            phys_out, (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                        held_sw_up_toa, held_lw_up_toa, held_sw_down_toa) = \
                step_unified(
                    need_rad_jax,
                    self.state.T.data, self.state.p_s.data,
                    self.q_v, self.q_c, self.q_r,
                    self.state.u.data, self.state.v.data,
                    sst, sic, self.grid.lat, self.grid.lon,
                    day_of_year, seconds_of_day, DT,
                    solar_weights, current_s_0,
                    o3_vmr, aerosol_od,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                )

            # (c) Update state
            new_T = self.state.T.data + DT * phys_out.dT_dt
            self.q_v = jnp.maximum(self.q_v + DT * phys_out.dq_v_dt, 0.0)
            self.q_c = jnp.maximum(self.q_c + DT * phys_out.dq_c_dt, 0.0)
            self.q_r = jnp.maximum(self.q_r + DT * phys_out.dq_r_dt, 0.0)

            # Saturation adjustment
            if MICROPHYSICS == "none":
                p_full = self.state.p_s.data[..., None] * sigma_full
                q_sat = saturation_mixing_ratio(new_T, p_full)
                excess = jnp.maximum(self.q_v - q_sat, 0.0)
                self.q_v = self.q_v - excess
                new_T = new_T + constants.L_v * excess / constants.c_pd
                precip_ls = jnp.sum(
                    excess * self.state.p_s.data[..., None] * dsigma, axis=-1
                ) / (constants.g * DT)
            else:
                precip_ls = jnp.zeros(shape_2d)

            self.state = self.state._replace(
                T=self.state.T.replace(data=new_T)
            )

            # Moisture conservation fixer
            if FIX_MOISTURE:
                self.q_v = fix_moisture_hydrostatic(
                    self.q_v, target_moisture,
                    self.state.p_s.data, dsigma, self.grid,
                )

            # Moisture smoothing
            self.q_v = jnp.maximum(
                self.q_v + DT * hyperdiffusion_3d(self.q_v, self.grid, self._qv_smooth_coeff),
                0.0,
            )

            # Rayleigh friction
            self.state = self.state._replace(
                u=self.state.u.replace(data=self.state.u.data * self._fric_decay),
                v=self.state.v.replace(data=self.state.v.data * self._fric_decay),
            )

            # Diagnostics
            elapsed_day = day - START_DAY
            if (step + 1) % diag_interval == 0:
                jax.block_until_ready(self.state.u.data)
                diag_info = self.diagnostics.collect(
                    elapsed_day=elapsed_day,
                    day=day,
                    state=self.state,
                    q_v=self.q_v,
                    q_c=self.q_c,
                    q_r=self.q_r,
                    sst=sst,
                    sic=sic,
                    precip_total=phys_out.precip + precip_ls,
                    sw_up_toa=phys_out.sw_up_toa,
                    lw_up_toa=phys_out.lw_up_toa,
                    sw_net_sfc=phys_out.sw_net_sfc,
                    lw_net_sfc=phys_out.lw_net_sfc,
                    sw_down_toa=phys_out.sw_down_toa,
                    T_ice=cfg.T_ice,
                    lat_deg_grid=lat_deg_grid,
                )

                logger.info(f"  Day {elapsed_day:6.0f}: T={diag_info['mean_T']:.1f}K, "
                      f"precip={diag_info['mean_precip']:.1f}mm/d, "
                      f"max_v={diag_info['max_v']:.1f}m/s")

                # Stability check
                error = self.diagnostics.check_stability(self.state, elapsed_day)
                if error:
                    logger.warning(f"  {error}")
                    run_status = error
                    break

            # Checkpoint
            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                self.save_checkpoint(step + 1, day)

        # Finalize
        jax.block_until_ready(self.state.u.data)
        total_wall = time.time() - t_start
        logger.info(f"Done: {total_wall:.1f}s wall time, status={run_status}")

        self.diagnostics.save(self._output_dir)
        self.save_results(run_status, t_jit, total_wall)
        logger.info(self.diagnostics.print_summary())

        # Final checkpoint
        if checkpoint_interval > 0:
            self.save_checkpoint(n_steps_total, START_DAY + N_DAYS)

        return run_status

    def save_results(self, run_status: str, jit_time: float, wall_time: float) -> None:
        """Write results.txt summary file."""
        cfg = self.config
        d = self.diagnostics
        with open(self._output_dir / "results.txt", "w") as f:
            f.write(f"legoESM AMIP run\n")
            f.write(f"Grid: {cfg.grid.grid_type} {cfg.grid.resolution} / "
                    f"L{cfg.grid.nlev}, dt={cfg.dycore.dt}s, {cfg.days} days\n")
            f.write(f"Radiation: {cfg.radiation}\n")
            f.write(f"Status: {run_status}\n\n")
            f.write(f"JIT compilation: {jit_time:.1f}s\n")
            f.write(f"Wall time: {wall_time:.1f}s\n\n")
            if d.times:
                f.write(f"Final <T_atm>: {d.T_atm[-1]:.3f} K\n")
                f.write(f"Final <Precip>: {d.precip[-1]:.2f} mm/day\n")
                f.write(f"Final <CWV>: {d.CWV[-1]:.1f} kg/m2\n")
            f.write(f"\n{d.energy_tracker.summary()}\n")
        logger.info(f"  Results saved to {self._output_dir}")
