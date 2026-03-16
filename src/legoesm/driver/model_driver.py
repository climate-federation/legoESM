"""Composable model driver for legoESM.

Orchestrates grid creation, vertical coordinate, dycore, physics,
forcing, state initialization, time-stepping, diagnostics, and
checkpointing into a single reusable class.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.forcing.time_utils import day_to_calendar

from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import PhysicsPipeline, build_physics_pipeline
from legoesm.driver.diagnostics import DiagnosticCollector


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
        self.q_v = None
        self.q_c = None
        self.q_r = None
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

    def setup(self) -> None:
        """Initialize grid, dycore, physics, forcing, and state."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._create_grid()
        self._create_topography()
        self._create_dycore()
        self._create_forcing()
        self._init_state()
        self._create_physics()
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

        print(f"  Grid: {gc.grid_type} {gc.resolution}, "
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

        if dc.discretization == "finite_volume":
            from legoesm.atmosphere.dynamics.primitive_eq_fv import (
                FVPrimitiveEquationModel, FVPrimitiveEquationConfig,
            )
            div_damp_2 = 0.05 * dx_min ** 2 / DT
            dycore_config = FVPrimitiveEquationConfig(
                div_damp_2=div_damp_2, div_damp_4=0.0,
                hyperdiff_coeff=HYPERDIFF, hyperdiff_ps_coeff=0.0,
                use_conservation_fixer=dc.conservation_fixer,
                fix_mass=dc.fix_mass, use_limiter=True,
            )
            self.model = FVPrimitiveEquationModel(self.grid, self.sigma, dycore_config)
        elif dc.discretization == "cgrid":
            from legoesm.atmosphere.dynamics.primitive_eq_cgrid import (
                CGPrimitiveEquationModel, CGPrimitiveEquationConfig,
            )
            div_damp_2 = 0.05 * dx_min ** 2 / DT
            div_damp_4 = 0.01 * dx_min ** 4 / DT
            dycore_config = CGPrimitiveEquationConfig(
                hyperdiff_coeff=HYPERDIFF, hyperdiff_ps_coeff=0.0,
                div_damp_2=div_damp_2, div_damp_4=div_damp_4,
                use_conservation_fixer=dc.conservation_fixer,
                fix_mass=dc.fix_mass,
            )
            self.model = CGPrimitiveEquationModel(self.grid, self.sigma, dycore_config)
        else:
            from legoesm.atmosphere.dynamics.primitive_eq import (
                PrimitiveEquationModel, PrimitiveEquationConfig,
            )
            div_damp = 0.12 * dx_min ** 2 / DT
            dycore_config = PrimitiveEquationConfig(
                hyperdiff_coeff=HYPERDIFF, hyperdiff_ps_coeff=HYPERDIFF,
                div_damp_coeff=div_damp,
                use_conservation_fixer=dc.conservation_fixer,
                fix_mass=dc.fix_mass,
            )
            self.model = PrimitiveEquationModel(self.grid, self.sigma, dycore_config)

        self._hyperdiff = HYPERDIFF
        print(f"  Dycore: {dc.discretization}, dt={DT}s")

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

        # Moisture initialization
        p_full_init = self.state.p_s.data[..., None] * self.sigma.sigma_full
        q_sat_init = saturation_mixing_ratio(self.state.T.data, p_full_init)
        self.q_v = cfg.RH_init * q_sat_init * self.sigma.sigma_full ** 2
        self.q_v = jnp.minimum(self.q_v, q_sat_init)

        self.q_c = jnp.zeros(shape_3d)
        self.q_r = jnp.zeros(shape_3d)

        mean_qv = float(jnp.mean(self.q_v)) * 1000.0
        cwv = float(jnp.mean(
            column_water_vapor(self.q_v, self.state.p_s.data, self.sigma.dsigma)
        ))
        print(f"  State init: T={cfg.T_init}K, q_v={mean_qv:.2f} g/kg, CWV={cwv:.1f} kg/m2")

    def _create_physics(self) -> None:
        """Build the physics pipeline."""
        self.physics = build_physics_pipeline(self.grid, self.sigma, self.config)
        print(f"  Physics: {self.config.radiation} + SBM convection")

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
        """Save checkpoint to output directory."""
        from legoesm.forcing.amip_config import save_checkpoint
        elapsed_day = day - self.config.start_day
        ckpt_path = self._output_dir / f"checkpoint_day_{int(elapsed_day):04d}.npz"
        amip_cfg = self.config.to_amip_config()
        save_checkpoint(
            path=ckpt_path, state=self.state, q_v=self.q_v,
            step=step, day=day, config=amip_cfg,
            q_c=self.q_c, q_r=self.q_r,
        )
        print(f"  Checkpoint: {ckpt_path.name}")

    def load_checkpoint(self, path: str | Path) -> tuple[int, float]:
        """Load state from a checkpoint. Returns (step, day)."""
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

        cfg = self.config
        DT = cfg.dycore.dt
        N_DAYS = cfg.days
        START_DAY = start_day if start_day is not None else cfg.start_day
        MICROPHYSICS = cfg.microphysics

        n_steps_total = int(N_DAYS * 86400 / DT)
        diag_interval = int(cfg.output.diag_days * 86400 / DT)
        checkpoint_interval = (
            int(cfg.output.checkpoint_days * 86400 / DT)
            if cfg.output.checkpoint_days > 0 else 0
        )

        sigma_full = self.sigma.sigma_full
        sigma_half = self.sigma.sigma_half
        dsigma = self.sigma.dsigma

        if cfg.grid.grid_type == "cubed_sphere":
            N = cfg.grid.resolution
            shape_2d = (6, N, N)
            shape_3d = (6, N, N, cfg.grid.nlev)
        else:
            shape_2d = (self.grid.n_lat, self.grid.n_lon)
            shape_3d = (*shape_2d, cfg.grid.nlev)

        # Solar forcing
        from legoesm.forcing.external import SolarConfig, get_solar_forcing_at_time
        solar_config = SolarConfig(S_0=cfg.S_0, source="constant")
        solar_init = get_solar_forcing_at_time(solar_config, START_DAY)
        solar_weights = jnp.array([], dtype=jnp.float64)

        # Held radiation tendencies
        held_dT_rad = jnp.zeros(shape_3d)
        held_sw_net_sfc = jnp.zeros(shape_2d)
        held_lw_net_sfc = jnp.zeros(shape_2d)
        held_sw_up_toa = jnp.zeros(shape_2d)
        held_lw_up_toa = jnp.zeros(shape_2d)
        held_sw_down_toa = jnp.zeros(shape_2d)

        run_status = "COMPLETED"
        lat_deg_grid = np.degrees(np.asarray(self.grid.lat))

        print(f"\n  Starting: {n_steps_total - start_step} steps, {N_DAYS} days")

        t_start = time.time()

        for step in range(start_step, n_steps_total):
            day = START_DAY + (step + 1) * DT / 86400.0
            day_of_year, seconds_of_day = day_to_calendar(day)

            sst, sic = self.get_sst_sic(day)

            # (a) Dynamics
            self.state = self.model.step_with_physics(self.state, DT)

            # (b) Radiation (always compute for now — sub-cycling TODO)
            from legoesm.forcing.surface_utils import blend_surface_temperature, blend_surface_property
            ncol = np.prod(np.array(shape_2d))
            nlev = cfg.grid.nlev

            T_sfc = blend_surface_temperature(sst, sic, cfg.T_ice)
            albedo = blend_surface_property(sic, cfg.albedo_ice, cfg.albedo_ocean)
            emissivity = blend_surface_property(sic, cfg.emissivity_ice, cfg.sfc_emissivity)

            p_full = self.state.p_s.data[..., None] * sigma_full
            p_half = self.state.p_s.data[..., None] * sigma_half

            T_col = self.state.T.data.reshape(ncol, nlev)
            p_full_col = p_full.reshape(ncol, nlev)
            p_half_col = p_half.reshape(ncol, nlev + 1)
            q_v_col = self.q_v.reshape(ncol, nlev)
            T_sfc_col = T_sfc.reshape(ncol)
            lat_col = jnp.asarray(self.grid.lat).reshape(-1)[:ncol]
            lon_col = jnp.asarray(self.grid.lon).reshape(-1)[:ncol]
            albedo_col = albedo.reshape(ncol)
            emis_col = emissivity.reshape(ncol)

            current_s_0 = float(solar_init["tsi"])

            rad_out = self.physics.radiation_fn(
                T_col, p_full_col, p_half_col, q_v_col,
                T_sfc_col, lat_col, lon_col,
                day_of_year, seconds_of_day,
                albedo_col, emis_col,
                jnp.zeros((ncol, nlev)),  # o3
                jnp.zeros((ncol, nlev)),  # aerosol
                solar_weights, current_s_0,
            )

            dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)
            sw_net_sfc = (rad_out.sw_flux_down[:, -1].reshape(shape_2d)
                          * (1.0 - albedo))
            lw_net_sfc = (rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]).reshape(shape_2d)
            sw_up_toa = rad_out.sw_flux_up[:, 0].reshape(shape_2d)
            lw_up_toa = rad_out.lw_flux_up[:, 0].reshape(shape_2d)
            sw_down_toa = rad_out.sw_flux_down[:, 0].reshape(shape_2d)

            # (c) Physics (convection + microphysics + BL)
            phys_out = self.physics.physics_step_no_rad(
                self.state.T.data, self.state.p_s.data,
                self.q_v, self.q_c, self.q_r,
                self.state.u.data, self.state.v.data,
                sst, sic, lat_col.reshape(shape_2d) if lat_col.size == np.prod(np.array(shape_2d)) else jnp.asarray(self.grid.lat),
                DT, dT_dt_rad, sw_net_sfc, lw_net_sfc,
                sw_up_toa, lw_up_toa, sw_down_toa,
            )

            # (d) Update state
            new_T = self.state.T.data + DT * phys_out.dT_dt
            self.q_v = jnp.maximum(self.q_v + DT * phys_out.dq_v_dt, 0.0)
            self.q_c = jnp.maximum(self.q_c + DT * phys_out.dq_c_dt, 0.0)
            self.q_r = jnp.maximum(self.q_r + DT * phys_out.dq_r_dt, 0.0)

            # Saturation adjustment
            if MICROPHYSICS == "none":
                q_sat = saturation_mixing_ratio(new_T, p_full)
                excess = jnp.maximum(self.q_v - q_sat, 0.0)
                self.q_v = self.q_v - excess
                new_T = new_T + constants.L_v * excess / constants.c_pd
                precip_ls = jnp.sum(
                    excess * self.state.p_s.data[..., None] * dsigma,
                    axis=-1
                ) / (constants.g * DT)
            else:
                precip_ls = jnp.zeros(shape_2d)

            self.state = self.state._replace(
                T=self.state.T.replace(data=new_T)
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
                    sw_up_toa=sw_up_toa,
                    lw_up_toa=lw_up_toa,
                    sw_net_sfc=sw_net_sfc,
                    lw_net_sfc=lw_net_sfc,
                    sw_down_toa=sw_down_toa,
                    T_ice=cfg.T_ice,
                    lat_deg_grid=lat_deg_grid,
                )

                print(f"  Day {elapsed_day:6.0f}: T={diag_info['mean_T']:.1f}K, "
                      f"precip={diag_info['mean_precip']:.1f}mm/d, "
                      f"max_v={diag_info['max_v']:.1f}m/s")

                # Stability check
                error = self.diagnostics.check_stability(self.state, elapsed_day)
                if error:
                    print(f"  {error}")
                    run_status = error
                    break

            # Checkpoint
            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                self.save_checkpoint(step + 1, day)

        # Finalize
        jax.block_until_ready(self.state.u.data)
        total_wall = time.time() - t_start
        print(f"\n  Done: {total_wall:.1f}s wall time, status={run_status}")

        self.diagnostics.save(self._output_dir)
        print(f"  {self.diagnostics.print_summary()}")

        return run_status
