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
from legoesm.core.tracers import TracerRegistry, make_moisture_registry, init_tracers
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
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
        self._hyperdiffusion_3d_fn = None
        self._ensemble_size = 1
        self._device_config = None
        self._carry_aux: dict = {}  # held radiation + carry metadata for checkpoint

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
        # Strict validation — abort early on invalid parameters
        self.config.validate_strict()

        # Activate precision policy before any JAX array creation.
        from legoesm.runtime.precision import apply_precision
        apply_precision(self.config.precision)
        logger.info(f"  Precision: {self.config.precision}")

        # Config cross-validation
        config_warnings = self.config.validate()
        for w in config_warnings:
            logger.warning(f"  Config: {w}")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._create_grid()
        self._create_topography()
        self._create_dycore()
        self._create_forcing()
        self._init_state()
        self._create_ensemble()
        self._create_physics()
        self._setup_external_forcing()
        self._create_diagnostics()
        self._create_friction()
        self._save_config()
        self._setup_parallel()

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
        elif gc.grid_type == "voronoi":
            from legoesm.grids.voronoi import create_voronoi_mesh
            self.grid = create_voronoi_mesh(gc.resolution, lloyd_iterations=50)
        else:
            raise ValueError(
                f"Unknown grid_type={gc.grid_type!r}. "
                f"Supported: cubed_sphere, gaussian, latlon, voronoi"
            )

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

        # Cache lat/lon accessors via GridProtocol for grid-agnostic use
        self._grid_lat = self.grid.grid_lat
        self._grid_lon = self.grid.grid_lon

    def _create_topography(self) -> None:
        """Load or generate topography and land-sea mask."""
        from legoesm.grids.topography import (
            TopographyConfig, load_real_topography,
            gaussian_mountain, phis_from_topography,
            land_mask_from_topography,
        )

        topo = self.config.topography
        shape_2d = self.grid.grid_shape_2d

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
        """Create the dynamical core model via the component factory.

        The factory resolves ``(model_type, discretization, grid_type)``
        from :attr:`config` and instantiates the correct solver with
        physically derived diffusion coefficients.
        """
        from legoesm.driver.component_factory import (
            create_atmosphere_dycore, compute_diffusion,
        )

        self.model = create_atmosphere_dycore(self.config, self.grid, self.sigma)

        # Keep hyperdiffusion coefficient for moisture smoothing later.
        diff = compute_diffusion(self.grid, self.config.dycore)
        self._hyperdiff = diff.hyperdiff

        dc = self.config.dycore

        # --- CFL validation (uses cfl module, warns and adjusts if unsafe) ---
        from legoesm.core.cfl import cfl_check_and_adjust
        gc = self.config.grid
        model_type_map = {
            "shallow_water": "shallow_water",
            "hydrostatic": "primitive_eq",
            "nonhydrostatic": "compressible",
        }
        cfl_model = model_type_map.get(dc.model_type, "primitive_eq")
        dt_safe = cfl_check_and_adjust(
            dc.dt, gc.resolution, model_type=cfl_model,
            radius=getattr(self.grid, 'radius', 6.371229e6),
            grid_type=gc.grid_type,
        )
        if dt_safe < dc.dt:
            logger.warning(
                f"  CFL: reducing dt from {dc.dt:.0f}s to {dt_safe:.0f}s "
                f"for {gc.grid_type} C{gc.resolution}"
            )
            self.config = self.config._replace(
                dycore=dc._replace(dt=dt_safe),
            )
            dc = self.config.dycore

        logger.info(
            f"  Dycore: {dc.model_type}/{dc.discretization} on "
            f"{self.config.grid.grid_type}, dt={dc.dt}s"
        )

    def _create_forcing(self) -> None:
        """Load SST/SIC forcing data."""
        cfg = self.config

        if cfg.dataset == "analytical":
            from legoesm.forcing.analytical import analytical_sst_sic
            lat_deg = np.degrees(np.asarray(self._grid_lat))
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
        from legoesm.diagnostics.column_integrals import column_water_vapor

        cfg = self.config
        N = cfg.grid.resolution
        NLEV = cfg.grid.nlev

        if cfg.grid.grid_type == "voronoi":
            from legoesm.atmosphere.physics.held_suarez_mpas import held_suarez_init_mpas
            shape_3d = (self.grid.nCells, NLEV)
            self.state = held_suarez_init_mpas(
                self.grid, self.sigma, T_init=cfg.T_init,
            )
            if jnp.any(self._phis_data != 0):
                self.state = self.state._replace(
                    phis=self.state.phis.replace(data=self._phis_data),
                )
        elif cfg.dycore.discretization == "spectral":
            from legoesm.atmosphere.dynamics.spectral_pe import isothermal_rest_state_spectral
            shape_3d = (self.grid.n_lat, self.grid.n_lon, NLEV)
            phis_arg = self._phis_data if jnp.any(self._phis_data != 0) else None
            self.state = isothermal_rest_state_spectral(
                self.grid, self.sigma, T_init=cfg.T_init, phis=phis_arg,
            )
        else:
            if cfg.grid.grid_type == "cubed_sphere":
                from legoesm.atmosphere.physics.held_suarez import held_suarez_init
                shape_3d = (6, N, N, NLEV)
                self.state = held_suarez_init(
                    self.grid, self.sigma, T_init=cfg.T_init, phis=self._phis_data
                )
            else:
                # Lat-lon and Gaussian grids use (n_lat, n_lon, nlev) layout
                from legoesm.atmosphere.physics.held_suarez_latlon import held_suarez_init_latlon
                shape_3d = (self.grid.n_lat, self.grid.n_lon, NLEV)
                self.state = held_suarez_init_latlon(
                    self.grid, self.sigma, T_init=cfg.T_init,
                )
                if jnp.any(self._phis_data != 0):
                    self.state = self.state._replace(
                        phis=self.state.phis.replace(data=self._phis_data),
                    )

        # Initialize all tracers via registry
        self.tracers = init_tracers(self.tracer_registry, shape_3d)

        # Moisture initialization (spectral and MPAS use dry physics)
        if hasattr(self.state, 'p_s') and hasattr(self.state.p_s, 'data'):
            p_full_init = self.state.p_s.data[..., None] * self.sigma.sigma_full
            q_sat_init = saturation_mixing_ratio(self.state.T.data, p_full_init)
            self.tracers["q_v"] = cfg.RH_init * q_sat_init * self.sigma.sigma_full ** 2
            self.tracers["q_v"] = jnp.minimum(self.tracers["q_v"], q_sat_init)

            mean_qv = float(jnp.mean(self.tracers["q_v"])) * 1000.0
            cwv = float(jnp.mean(
                column_water_vapor(self.tracers["q_v"], self.state.p_s.data, self.sigma.dsigma)
            ))
            logger.info(f"  State init: T={cfg.T_init}K, q_v={mean_qv:.2f} g/kg, CWV={cwv:.1f} kg/m2")
        else:
            logger.info(f"  State init: T={cfg.T_init}K (dry spectral)")

    def _create_ensemble(self) -> None:
        """Create ensemble members if ensemble_size > 1.

        Perturbs the initial state and tracers to create *ensemble_size*
        members.  Each leaf array gains a leading ensemble dimension:
        ``(n_members, ...)``.  Diagnostics later use ``ensemble_mean``
        before collecting.
        """
        self._ensemble_size = self.config.ensemble_size
        # Save a single-member template for unpack_carry during ensemble runs
        self._state_template = self.state
        if self._ensemble_size <= 1:
            return

        from legoesm.parallel.ensemble import perturb_initial_conditions

        key = jax.random.PRNGKey(42)
        self.state = perturb_initial_conditions(
            self.state, key, self._ensemble_size, scale=0.01,
        )
        # Tile tracers: each tracer (S,...) -> (n_members, S,...)
        for name, arr in self.tracers.items():
            if arr is not None:
                self.tracers[name] = jnp.broadcast_to(
                    arr[None], (self._ensemble_size,) + arr.shape
                ).copy()  # copy so each member can diverge

        logger.info(f"  Ensemble: {self._ensemble_size} members (IC perturbation scale=0.01)")

    def _create_physics(self) -> None:
        """Build the physics pipeline."""
        self.physics = build_physics_pipeline(self.grid, self.sigma, self.config)
        logger.info(f"  Physics: {self.config.radiation} + SBM convection")

    def _setup_external_forcing(self) -> None:
        """Configure external forcing: solar, ozone, aerosol, GHG."""
        from legoesm.forcing.external import (
            SolarConfig, OzoneConfig, AerosolConfig, GHGConfig,
            get_solar_forcing_at_time, get_ozone_at_time, get_aerosol_at_time,
            get_ghg_at_time, ghg_concentrations_to_vmr,
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
            self._solar_weights_template = jnp.array([], dtype=jnp.float32)

        # CMIP experiment GHG override
        self._experiment = cfg.experiment
        self._start_year = cfg.start_year
        if self._experiment:
            from legoesm.forcing.experiments import ghg_at_year
            co2, ch4, n2o = ghg_at_year(self._experiment, self._start_year)
            self.config = cfg._replace(co2_ppmv=co2, ch4_ppbv=ch4, n2o_ppbv=n2o)
            cfg = self.config
            logger.info(f"  CMIP: {self._experiment} (year {self._start_year}), "
                  f"CO2={co2:.1f} ppmv")

        # GHG forcing config
        self._ghg_active = (cfg.radiation in ("rrtmg", "rrtmgp")
                            and cfg.ghg_forcing == "external")
        if self._ghg_active:
            self._ghg_config = GHGConfig(
                co2_ppmv=cfg.co2_ppmv,
                ch4_ppbv=cfg.ch4_ppbv,
                n2o_ppbv=cfg.n2o_ppbv,
                source="annual_file",
                path=cfg.ghg_file,
                start_year=cfg.start_year,
            )
            # Log initial GHG values
            ghg_init = get_ghg_at_time(self._ghg_config, cfg.start_day)
            logger.info(
                f"  GHG external: CO2={ghg_init['co2_ppmv']:.1f}ppmv, "
                f"CH4={ghg_init['ch4_ppbv']:.0f}ppbv, "
                f"N2O={ghg_init['n2o_ppbv']:.1f}ppbv, "
                f"CFC11={ghg_init['cfc11_pptv']:.0f}pptv, "
                f"CFC12={ghg_init['cfc12_pptv']:.0f}pptv"
            )
        else:
            self._ghg_config = GHGConfig(
                co2_ppmv=cfg.co2_ppmv,
                ch4_ppbv=cfg.ch4_ppbv,
                n2o_ppbv=cfg.n2o_ppbv,
                source="constant",
            )

    def _precompute_external_forcing(self, day, p_s, lat):
        """Pre-compute ozone/aerosol/GHG fields outside JIT boundary."""
        from legoesm.forcing.external import (
            get_ozone_at_time, get_aerosol_at_time,
            get_ghg_at_time, ghg_concentrations_to_vmr,
        )
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

        # GHG VMR override (None for gray radiation / constant forcing)
        ghg_vmr = None
        if self._ghg_active:
            ghg_conc = get_ghg_at_time(self._ghg_config, day)
            ghg_vmr = ghg_concentrations_to_vmr(ghg_conc)

        return o3_vmr, aerosol_od, ghg_vmr

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
            output_dir=self._output_dir,
        )

    def _sync_and_collect_diagnostics(self, **kwargs) -> dict:
        """Synchronize device computation and collect diagnostics.

        When running in multi-device mode, gathers the state from all
        devices before diagnostic collection.  Consolidates the
        ``jax.block_until_ready`` + ``diagnostics.collect`` pattern
        into a single method to avoid scattered sync points.

        In ``perf_mode``, uses ``collect_lightweight()`` which computes
        only scalar reductions (``jnp.mean``, ``jnp.max``) directly on
        the sharded arrays — no full-state gather, no host materialization
        via ``np.asarray()``.  JAX handles the cross-device reductions
        internally for SPMD-sharded arrays, so global means and max-wind
        are still correct.  Snapshots, profiles, and monthly means are
        skipped.
        """
        perf_mode = kwargs.pop("perf_mode", False)

        if perf_mode:
            # Lightweight path: scalar reductions only, no gather.
            state = kwargs.get('state', self.state)
            jax.block_until_ready(state.u.data)
            return self.diagnostics.collect_lightweight(
                elapsed_day=kwargs.get('elapsed_day', 0.0),
                state=state,
                q_v=kwargs.get('q_v', None),
                sst=kwargs.get('sst', None),
                sic=kwargs.get('sic', None),
                precip_total=kwargs.get('precip_total', None),
                sw_up_toa=kwargs.get('sw_up_toa', None),
                lw_up_toa=kwargs.get('lw_up_toa', None),
                sw_net_sfc=kwargs.get('sw_net_sfc', None),
                lw_net_sfc=kwargs.get('lw_net_sfc', None),
            )

        if self._device_config is not None and self._device_config.mesh is not None:
            from legoesm.parallel.sharded_dynamics import gather_state
            gathered = gather_state(kwargs.get('state', self.state), self._device_config)
            kwargs['state'] = gathered
        jax.block_until_ready(kwargs.get('state', self.state).u.data)
        return self.diagnostics.collect(**kwargs)

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

        # Select grid-appropriate hyperdiffusion operator
        if cfg.grid.grid_type == "cubed_sphere":
            from legoesm.core.operators_3d import hyperdiffusion_3d
        else:
            from legoesm.core.operators_latlon_3d import hyperdiffusion_3d
        self._hyperdiffusion_3d_fn = hyperdiffusion_3d

    def _save_config(self) -> None:
        """Save experiment config to output directory.

        Writes the canonical ExperimentConfig JSON.  Also writes a
        legacy ``amip_config.json`` for backward-compatible tooling.
        """
        from legoesm.driver.config import save_experiment_config
        save_experiment_config(self.config, self._output_dir / "experiment_config.json")

        # Legacy AMIP-format sidecar (serialization boundary adapter)
        from legoesm.forcing.amip_config import save_config as _save_amip
        _save_amip(self.config.to_amip_config(), self._output_dir / "amip_config.json")

    def _setup_parallel(self) -> None:
        """Set up multi-device parallelism if available.

        Tries to use the canonical ``ParallelRuntime`` for device mesh
        creation and state sharding.  Falls back to local device
        detection for backward compatibility when ``ParallelRuntime``
        is not configured (e.g., non-MPI single-node runs).
        """
        self._device_config = None

        # Try canonical ParallelRuntime path first (handles distributed/MPI).
        try:
            from legoesm.parallel.mesh import get_active_config
            active = get_active_config()
            if active is not None and active.n_devices > 1:
                self._device_config = active
                logger.info(
                    f"  Parallel: using active config — "
                    f"{active.n_devices} devices, "
                    f"distributed={active.is_distributed}"
                )
        except Exception:
            pass

        # Fall back to local device detection.
        if self._device_config is None:
            from legoesm.parallel.device_config import detect_devices, configure_jax_for_device
            hw = detect_devices()
            configure_jax_for_device(hw)

            if hw.device_count <= 1:
                return
            if self.config.grid.grid_type != "cubed_sphere":
                return

            from legoesm.parallel.mesh import create_device_mesh
            try:
                self._device_config = create_device_mesh(n_devices=hw.device_count)
            except ValueError:
                logger.warning(
                    f"Cannot create cubed-sphere mesh for {hw.device_count} "
                    f"devices; using single-device mode"
                )
                return

        if self._device_config is None:
            return

        # Shard state across devices
        from legoesm.parallel.sharded_dynamics import shard_state
        self.state = shard_state(self.state, self._device_config)

        from legoesm.parallel.mesh import shard_pytree
        self.tracers = shard_pytree(self.tracers, self._device_config)

        logger.info(
            f"  Parallel: {self._device_config.n_devices} devices, "
            f"tiling={self._device_config.tiling}"
        )

    def save_checkpoint(self, step: int, day: float) -> None:
        """Save checkpoint to output directory using unified restart API.

        Accepts ``ExperimentConfig`` directly — the restart layer handles
        the AMIP wire-format conversion internally.
        """
        elapsed_day = day - self.config.start_day
        ckpt_path = self._output_dir / f"checkpoint_day_{int(elapsed_day):04d}.npz"

        backend = self.config.output.checkpoint_format if hasattr(self.config.output, 'checkpoint_format') else "npz"

        save_restart(
            path=ckpt_path,
            state=self.state,
            q_v=self.q_v,
            step=step,
            day=day,
            config=self.config,
            q_c=self.q_c,
            q_r=self.q_r,
            carry_aux=self._carry_aux if self._carry_aux else None,
            backend=backend,
        )
        logger.info(f"  Checkpoint: {ckpt_path.name}")

    def load_checkpoint(self, path: str | Path) -> tuple[int, float]:
        """Load state from a checkpoint using unified restart API.

        Returns (step, day).
        """
        result = load_restart(
            Path(path), self.grid, self.sigma, strict=True,
        )
        state, q_v, step, day, _, _, q_c, q_r, metadata, carry_aux = result
        self.state = state
        self.q_v = q_v
        if q_c is not None:
            self.q_c = q_c
        if q_r is not None:
            self.q_r = q_r
        self._carry_aux = carry_aux if carry_aux else {}
        if metadata:
            logger.info(f"  Loaded restart: step={step}, day={day}, "
                       f"digest={metadata.state_digest[:16]}...")
        return step, day

    def run(self, start_step: int = 0, start_day: float | None = None,
            compiled: bool = True) -> str:
        """Run the time integration.

        Parameters
        ----------
        start_step : int
            Starting time step (for restart).
        start_day : float, optional
            Starting day (for restart). Defaults to config.start_day.
        compiled : bool
            If ``True`` (default), use compiled segment execution via
            ``jax.lax.scan``.  If ``False``, use the legacy per-step
            Python loop (useful for debugging or when the compiled path
            is not applicable).

        Returns
        -------
        str
            Run status ("COMPLETED" or "BLOWUP at day ...").
        """
        # MPAS and spectral states use different pytree layouts;
        # use dedicated simple run loops.
        if self.config.grid.grid_type == "voronoi":
            return self._run_mpas(start_step, start_day)
        if self.config.dycore.discretization == "spectral":
            return self._run_spectral(start_step, start_day)
        if compiled:
            return self._run_compiled(start_step, start_day)
        return self._run_per_step(start_step, start_day)

    # ==================================================================
    # MPAS execution path (Held-Suarez forcing, no separate physics pipeline)
    # ==================================================================

    def _run_mpas(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run MPAS model with Held-Suarez forcing.

        Uses the MPAS PE model's built-in physics_fn interface instead
        of the general physics pipeline (which assumes A-grid u/v).
        """
        import time
        from legoesm.atmosphere.physics.held_suarez_mpas import held_suarez_forcing_mpas

        cfg = self.config
        DT = cfg.dycore.dt
        N_DAYS = cfg.days
        n_steps_total = int(N_DAYS * 86400.0 / DT)
        DIAG_INTERVAL = int(cfg.output.diag_days * 86400.0 / DT) if cfg.output.diag_days > 0 else n_steps_total
        START_DAY = start_day if start_day is not None else cfg.start_day

        physics_fn = held_suarez_forcing_mpas

        run_status = "COMPLETED"
        logger.info(f"Starting MPAS: {n_steps_total - start_step} steps, {N_DAYS} days")

        t_start = time.time()

        for step in range(start_step, n_steps_total):
            self.state = self.model.step(self.state, DT, physics_fn=physics_fn)

            # Diagnostics at intervals
            if DIAG_INTERVAL > 0 and (step + 1) % DIAG_INTERVAL == 0:
                elapsed_day = (step + 1) * DT / 86400.0
                day = START_DAY + elapsed_day
                T_data = self.state.T.data
                p_s_data = self.state.p_s.data
                u_data = self.state.u.data

                mean_T = float(jnp.mean(T_data))
                mean_ps = float(jnp.mean(p_s_data))
                max_u = float(jnp.max(jnp.abs(u_data)))
                T_min = float(jnp.min(T_data))
                T_max = float(jnp.max(T_data))

                elapsed = time.time() - t_start
                rate = elapsed_day / (elapsed + 1e-10)
                logger.info(
                    f"  Day {elapsed_day:6.1f}: T=[{T_min:.1f},{T_max:.1f}]K "
                    f"mean={mean_T:.1f}K  p_s={mean_ps/100:.1f}hPa  "
                    f"|u|_max={max_u:.1f}m/s  ({rate:.1f} sim-days/s)"
                )

                # Blowup detection
                if not jnp.all(jnp.isfinite(T_data)):
                    run_status = f"BLOWUP at day {elapsed_day:.1f}"
                    logger.error(run_status)
                    break

        elapsed = time.time() - t_start
        logger.info(f"MPAS run {run_status} in {elapsed:.1f}s")
        return run_status

    # ==================================================================
    # Spectral execution path (Held-Suarez + radiation on Gaussian grid)
    # ==================================================================

    def _run_spectral(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run spectral PE model with physics coupling.

        Physics tendencies are computed on the Gaussian grid and converted
        back to spectral space via SH analysis.
        """
        import time
        from legoesm.atmosphere.dynamics.spectral_pe import (
            spectral_pe_to_grid,
            SpectralHydrostaticState,
        )
        from legoesm.grids.gaussian import (
            sh_analysis_3d,
            sh_analysis_oc2_3d,
            sh_analysis_dmu_3d,
        )
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
        from legoesm.forcing.surface_utils import blend_surface_temperature

        cfg = self.config
        DT = cfg.dycore.dt
        N_DAYS = cfg.days
        n_steps_total = int(N_DAYS * 86400.0 / DT)
        DIAG_INTERVAL = int(cfg.output.diag_days * 86400.0 / DT) if cfg.output.diag_days > 0 else n_steps_total
        START_DAY = start_day if start_day is not None else cfg.start_day
        a = self.grid.radius

        # Rayleigh friction profile
        sigma_full = self.sigma.sigma_full
        K_F = 1.0 / 86400.0
        SIGMA_B = 0.7
        k_f = K_F * jnp.maximum(0.0, (sigma_full - SIGMA_B) / (1.0 - SIGMA_B))

        gray_config = GrayRadiationConfig()
        shape_2d = (self.grid.n_lat, self.grid.n_lon)
        shape_3d = (*shape_2d, cfg.grid.nlev)
        S_0 = 1361.0
        T_ice = cfg.T_ice

        # Precompute spectral transform constants
        _im_over_a = 1j * self.grid.ms.astype(jnp.float64) / a
        _one_over_a = 1.0 / a
        cos_lat_3d = self.grid.cos_lat[:, None, None]

        def _spectral_physics_fn(state, grid, sigma_coord):
            """Compute physics tendencies and return spectral tendencies."""
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
            T_g = fields['T']
            u_g = fields['u']
            v_g = fields['v']
            p_s_g = fields['p_s']

            sst, sic = self.get_sst_sic(self._current_day)
            # Broadcast from (n_lat,) to (n_lat, n_lon) if needed
            if sst.ndim == 1 and len(shape_2d) == 2:
                sst = jnp.broadcast_to(sst[:, None], shape_2d)
                sic = jnp.broadcast_to(sic[:, None], shape_2d)
            T_sfc = blend_surface_temperature(sst, sic, T_ice)

            p_full = p_s_g[..., None] * sigma_full
            p_half = p_s_g[..., None] * self.sigma.sigma_half
            T_col = T_g.reshape(-1, cfg.grid.nlev)
            p_full_col = p_full.reshape(-1, cfg.grid.nlev)
            p_half_col = p_half.reshape(-1, cfg.grid.nlev + 1)
            q_v_col = jnp.zeros_like(T_col)  # dry physics
            T_sfc_col = T_sfc.reshape(-1)
            # Broadcast lat from (n_lat,) to (n_lat, n_lon) then flatten
            lat_2d = jnp.broadcast_to(
                self._grid_lat[:, None], shape_2d
            )
            lat_col = lat_2d.reshape(-1)

            insol = daily_mean_insolation(lat_col, self._current_day, S_0)
            rad_out = gray_radiation(
                T=T_col, p_full=p_full_col, p_half=p_half_col,
                sfc_temperature=T_sfc_col, lat=lat_col,
                q_v=q_v_col, insolation=insol, config=gray_config,
            )
            dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)

            # Rayleigh friction
            du_dt = -k_f * u_g
            dv_dt = -k_f * v_g

            # Convert (du, dv) to spectral (dvor, ddiv)
            du_cos = du_dt * cos_lat_3d
            dv_cos = dv_dt * cos_lat_3d
            dvor_hat = (
                _im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
                + _one_over_a * sh_analysis_dmu_3d(grid, du_cos)
            )
            ddiv_hat = (
                _im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
                - _one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
            )
            dT_hat = sh_analysis_3d(grid, dT_dt_rad)
            dlnps_hat = jnp.zeros_like(state.lnps_hat.data)

            return SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=dvor_hat),
                div_hat=state.div_hat.replace(data=ddiv_hat),
                T_hat=state.T_hat.replace(data=dT_hat),
                lnps_hat=state.lnps_hat.replace(data=dlnps_hat),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            )

        self._current_day = START_DAY
        run_status = "COMPLETED"
        logger.info(f"Starting spectral: {n_steps_total - start_step} steps, {N_DAYS} days")

        t_start = time.time()
        for step in range(start_step, n_steps_total):
            self._current_day = START_DAY + (step + 1) * DT / 86400.0

            self.state = self.model.step(
                self.state, DT, physics_fn=_spectral_physics_fn,
            )

            if DIAG_INTERVAL > 0 and (step + 1) % DIAG_INTERVAL == 0:
                elapsed_day = (step + 1) * DT / 86400.0
                fields = spectral_pe_to_grid(self.state, self.grid, self.sigma)
                T_g = fields['T']
                p_s_g = fields['p_s']
                u_g, v_g = fields['u'], fields['v']

                mean_T = float(jnp.mean(T_g))
                T_min = float(jnp.min(T_g))
                T_max = float(jnp.max(T_g))
                mean_ps = float(jnp.mean(p_s_g))
                max_wind = float(jnp.max(jnp.sqrt(u_g**2 + v_g**2)))

                elapsed = time.time() - t_start
                rate = elapsed_day / (elapsed + 1e-10)
                logger.info(
                    f"  Day {elapsed_day:6.1f}: T=[{T_min:.1f},{T_max:.1f}]K "
                    f"mean={mean_T:.1f}K  p_s={mean_ps/100:.1f}hPa  "
                    f"|v|_max={max_wind:.1f}m/s  ({rate:.1f} sim-days/s)"
                )

                if not jnp.all(jnp.isfinite(T_g)):
                    run_status = f"BLOWUP at day {elapsed_day:.1f}"
                    logger.error(run_status)
                    break

        elapsed = time.time() - t_start
        logger.info(f"Spectral run {run_status} in {elapsed:.1f}s")
        return run_status

    # ==================================================================
    # Compiled segment execution path
    # ==================================================================

    def _run_compiled(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run using compiled segments (jax.lax.scan over N steps).

        The hot integration loop is compiled into segments of
        ``segment_length`` steps.  Host Python only runs between
        segments for diagnostics, checkpoints, and forcing updates.
        """
        from legoesm.forcing.external import get_solar_forcing_at_time
        from legoesm.driver.compiled_segments import (
            SegmentCarry, pack_carry, unpack_carry,
            compute_segment_length, build_segment_fn, pack_forcing,
        )

        cfg = self.config
        DT = cfg.dycore.dt
        N_DAYS = cfg.days
        START_DAY = start_day if start_day is not None else cfg.start_day
        RAD_UPDATE_STEPS = cfg.rad_update_steps

        n_steps_total = int(N_DAYS * 86400 / DT)
        diag_interval = int(cfg.output.diag_days * 86400 / DT)
        checkpoint_interval = (
            int(cfg.output.checkpoint_days * 86400 / DT)
            if cfg.output.checkpoint_days > 0 else 0
        )

        # Compute segment length = GCD of all cadence intervals
        segment_length = compute_segment_length(
            diag_interval, checkpoint_interval, RAD_UPDATE_STEPS,
        )
        n_steps_remaining = n_steps_total - start_step
        n_segments = (n_steps_remaining + segment_length - 1) // segment_length

        sigma_full = self.sigma.sigma_full
        dsigma = self.sigma.dsigma

        shape_2d = self.grid.grid_shape_2d
        shape_3d = (*shape_2d, cfg.grid.nlev)

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

        # Initial held radiation tendencies — restore from checkpoint if available
        _ens = self._ensemble_size
        _ens_3d = (_ens, *shape_3d) if _ens > 1 else shape_3d
        _ens_2d = (_ens, *shape_2d) if _ens > 1 else shape_2d
        _aux = self._carry_aux
        held_dT_rad = _aux.get("held_dT_rad", jnp.zeros(_ens_3d))
        held_sw_net_sfc = _aux.get("held_sw_net_sfc", jnp.zeros(_ens_2d))
        held_lw_net_sfc = _aux.get("held_lw_net_sfc", jnp.zeros(_ens_2d))
        held_sw_up_toa = _aux.get("held_sw_up_toa", jnp.zeros(_ens_2d))
        held_lw_up_toa = _aux.get("held_lw_up_toa", jnp.zeros(_ens_2d))
        held_sw_down_toa = _aux.get("held_sw_down_toa", jnp.zeros(_ens_2d))

        # External forcing
        o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
            START_DAY, self.state.p_s.data, self._grid_lat,
        )

        # Compute fixed moisture target for conservation fixer —
        # restore from checkpoint if available, else compute from IC.
        from legoesm.core.conservation import compute_global_moisture
        _target_moisture = _aux.get("target_moisture", jnp.asarray(0.0))
        if cfg.fix_moisture and float(_target_moisture) == 0.0:
            _target_moisture = compute_global_moisture(
                self.q_v, self.state.p_s.data, dsigma, self.grid,
            )
            logger.info(f"  Moisture target: {float(_target_moisture):.6e} kg")

        # Compute fixed dry mass target for target-anchored conservation
        from legoesm.core.operators import global_integral
        from legoesm.core.field import Field
        _p_s_field = Field(self.state.p_s.data, name="p_s", dims=("face", "x", "y"), units="Pa")
        _target_mass = _aux.get("target_mass", jnp.asarray(0.0))
        if cfg.dycore.fix_mass and float(_target_mass) == 0.0:
            _target_mass = global_integral(_p_s_field, self.grid)
            logger.info(f"  Mass target: {float(_target_mass):.6e} Pa·m²")

        run_status = "COMPLETED"
        lat_deg_grid = np.degrees(np.asarray(self._grid_lat))

        # Build the compiled segment function ONCE (outside the loop).
        # Per-segment forcing (SST, SIC, solar, ozone, aerosol) is now
        # passed as an explicit SegmentForcing argument to run_segment,
        # so changing forcing values does NOT trigger JIT recompilation.
        run_segment = build_segment_fn(
            model=self.model,
            step_unified=step_unified,
            grid=self.grid,
            sigma_full=sigma_full,
            dsigma=dsigma,
            dt=DT,
            rad_update_steps=RAD_UPDATE_STEPS,
            microphysics=cfg.microphysics,
            fix_moisture=cfg.fix_moisture,
            fix_mass=cfg.dycore.fix_mass,
            fric_decay=self._fric_decay,
            qv_smooth_coeff=self._qv_smooth_coeff,
            lat=self._grid_lat,
            lon=self._grid_lon,
            start_day=START_DAY,
            gradient_checkpoint=(
                cfg.gradient_checkpoint
                if cfg.gradient_checkpoint
                else segment_length > 50
            ),
            hyperdiffusion_3d_fn=self._hyperdiffusion_3d_fn,
            tau_equator=cfg.tau_equator,
            tau_pole=cfg.tau_pole,
            sbm_tau_c=cfg.sbm_tau_c,
            sbm_RH_ref=cfg.sbm_RH_ref,
            C_H=cfg.C_H,
            C_E=cfg.C_E,
            albedo_ice=cfg.albedo_ice,
            albedo_ocean=cfg.albedo_ocean,
            ghg_vmr_override=ghg_vmr,
        )

        logger.info(
            f"Starting compiled run: {n_steps_remaining} steps, "
            f"{n_segments} segments of {segment_length} steps"
        )

        current_step = start_step
        t_jit = 0.0
        t_start = time.time()

        for seg_idx in range(n_segments):
            seg_steps = min(segment_length, n_steps_total - current_step)
            seg_end_step = current_step + seg_steps
            day = START_DAY + seg_end_step * DT / 86400.0
            day_of_year, seconds_of_day = day_to_calendar(day)
            sst, sic = self.get_sst_sic(day)

            # Update external forcing at segment boundary (if radiation-aligned)
            if RAD_UPDATE_STEPS > 1 and seg_idx > 0:
                solar_now = get_solar_forcing_at_time(self._solar_config, day)
                current_s_0 = float(solar_now["tsi"])
                if self._use_solar_spectral:
                    solar_weights = jnp.asarray(solar_now["solar_fraction_by_gpt"])
                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
                    day, self.state.p_s.data, self._grid_lat,
                )

            # Pack per-segment forcing into a SegmentForcing pytree.
            forcing = pack_forcing(
                sst=sst, sic=sic,
                day_of_year=day_of_year, seconds_of_day=seconds_of_day,
                solar_weights=solar_weights, s_0=current_s_0,
                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
            )

            # Pack state into carry
            carry = pack_carry(
                self.state, self.q_v, self.q_c, self.q_r,
                held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                current_step,
                target_moisture=_target_moisture,
                target_mass=_target_mass,
                precip_accum=jnp.zeros(_ens_2d),
            )

            # Shard carry across devices for SPMD execution
            if self._device_config is not None and self._device_config.mesh is not None:
                from legoesm.parallel.mesh import shard_pytree
                carry = shard_pytree(carry, self._device_config)

            # Time first segment for JIT measurement
            if seg_idx == 0:
                t_jit_start = time.time()

            # Execute compiled segment (vmap over ensemble if needed)
            if self._ensemble_size > 1:
                carry = jax.vmap(run_segment, in_axes=(0, None, None))(carry, seg_steps, forcing)
            else:
                carry = run_segment(carry, seg_steps, forcing)

            if seg_idx == 0:
                jax.block_until_ready(carry.u)
                t_jit = time.time() - t_jit_start
                logger.info(f"  Segment 0 (incl. JIT) in {t_jit:.1f}s")

            # Unpack carry back to driver state.
            # For ensemble runs, unpack the ensemble-mean for diagnostics;
            # keep full ensemble in carry for the next segment.
            _target_moisture = carry.target_moisture
            _target_mass = carry.target_mass
            if self._ensemble_size > 1:
                from legoesm.parallel.ensemble import ensemble_mean
                mean_carry = ensemble_mean(carry)
                (self.state, self.q_v, self.q_c, self.q_r,
                 held_tuple, _, seg_precip) = unpack_carry(mean_carry, self._state_template)
            else:
                (self.state, self.q_v, self.q_c, self.q_r,
                 held_tuple, _, seg_precip) = unpack_carry(carry, self.state)
            (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
             held_sw_up_toa, held_lw_up_toa, held_sw_down_toa) = held_tuple

            # Keep carry auxiliary fields for checkpoint persistence
            self._carry_aux = {
                "held_dT_rad": held_dT_rad,
                "held_sw_net_sfc": held_sw_net_sfc,
                "held_lw_net_sfc": held_lw_net_sfc,
                "held_sw_up_toa": held_sw_up_toa,
                "held_lw_up_toa": held_lw_up_toa,
                "held_sw_down_toa": held_sw_down_toa,
                "target_moisture": _target_moisture,
                "target_mass": _target_mass,
            }

            current_step = seg_end_step

            # --- Host-side actions at segment boundaries ---
            elapsed_day = day - START_DAY

            # Diagnostics
            if diag_interval > 0 and current_step % diag_interval == 0:
                diag_info = self._sync_and_collect_diagnostics(
                    elapsed_day=elapsed_day,
                    day=day,
                    state=self.state,
                    q_v=self.q_v,
                    q_c=self.q_c,
                    q_r=self.q_r,
                    sst=sst,
                    sic=sic,
                    precip_total=seg_precip,
                    sw_up_toa=held_sw_up_toa,
                    lw_up_toa=held_lw_up_toa,
                    sw_net_sfc=held_sw_net_sfc,
                    lw_net_sfc=held_lw_net_sfc,
                    sw_down_toa=held_sw_down_toa,
                    T_ice=cfg.T_ice,
                    lat_deg_grid=lat_deg_grid,
                )

                # CFL computed host-side from final segment state (not in hot loop)
                from legoesm.core.cfl import cfl_number_from_state, estimate_min_dx_cubed_sphere
                _dx_min = estimate_min_dx_cubed_sphere(cfg.grid.resolution) if hasattr(self.grid, 'n') else 1e6
                _seg_max_cfl = float(cfl_number_from_state(
                    self.state.u.data, self.state.v.data, _dx_min, DT,
                ))

                elapsed_wall = time.time() - t_start
                days_done = elapsed_day
                eta_str = ""
                if days_done > 0:
                    rate = elapsed_wall / days_done
                    remaining = (N_DAYS - days_done) * rate
                    eta_str = f", ETA {remaining/3600:.1f}h"
                logger.info(
                    f"  Day {elapsed_day:6.0f}: T={diag_info['mean_T']:.1f}K, "
                    f"max_v={diag_info['max_v']:.1f}m/s"
                    f"{eta_str}"
                )

                # CFL monitoring
                if _seg_max_cfl > 0:
                    logger.info(f"    CFL max: {_seg_max_cfl:.2f}")

                # Stability check
                error = self.diagnostics.check_stability(self.state, elapsed_day)
                if error:
                    logger.warning(f"  {error}")
                    run_status = error
                    break

                # Adaptive dt: if CFL exceeds threshold, halve dt and rebuild
                if _seg_max_cfl > 1.0:
                    DT = DT / 2.0
                    logger.warning(
                        f"  CFL={_seg_max_cfl:.2f} > 1.0 at day {elapsed_day:.0f}. "
                        f"Halving dt to {DT:.0f}s."
                    )
                    n_steps_total = int(cfg.days * 86400 / DT)
                    diag_interval = int(cfg.output.diag_days * 86400 / DT)
                    checkpoint_interval = (
                        int(cfg.output.checkpoint_days * 86400 / DT)
                        if cfg.output.checkpoint_days > 0 else 0
                    )
                    segment_length = compute_segment_length(
                        diag_interval, checkpoint_interval, RAD_UPDATE_STEPS,
                    )
                    run_segment = build_segment_fn(
                        model=self.model, step_unified=step_unified,
                        grid=self.grid, sigma_full=sigma_full, dsigma=dsigma,
                        dt=DT, rad_update_steps=RAD_UPDATE_STEPS,
                        microphysics=cfg.microphysics, fix_moisture=cfg.fix_moisture,
                        fric_decay=self._fric_decay, qv_smooth_coeff=self._qv_smooth_coeff,
                        lat=self._grid_lat, lon=self._grid_lon, start_day=START_DAY,
                        gradient_checkpoint=cfg.gradient_checkpoint or segment_length > 50,
                        hyperdiffusion_3d_fn=self._hyperdiffusion_3d_fn,
                        tau_equator=cfg.tau_equator, tau_pole=cfg.tau_pole,
                        sbm_tau_c=cfg.sbm_tau_c, sbm_RH_ref=cfg.sbm_RH_ref,
                        C_H=cfg.C_H, C_E=cfg.C_E,
                        albedo_ice=cfg.albedo_ice, albedo_ocean=cfg.albedo_ocean,
                        ghg_vmr_override=ghg_vmr,
                    )

            # Checkpoint
            if checkpoint_interval > 0 and current_step % checkpoint_interval == 0:
                self.save_checkpoint(current_step, day)

            # Periodic diagnostic flush (every ~365 days) to cap memory
            if elapsed_day > 0 and int(elapsed_day) % 365 == 0 and diag_interval > 0 and current_step % diag_interval == 0:
                self.diagnostics.flush_to_disk(self._output_dir)

        # Finalize
        jax.block_until_ready(self.state.u.data)
        total_wall = time.time() - t_start
        logger.info(f"Done: {total_wall:.1f}s wall time, status={run_status}")

        self.diagnostics.save(self._output_dir)
        self.save_results(run_status, t_jit, total_wall)
        logger.info(self.diagnostics.print_summary())

        if checkpoint_interval > 0:
            self.save_checkpoint(n_steps_total, START_DAY + N_DAYS)

        return run_status

    # ==================================================================
    # Legacy per-step execution path
    # ==================================================================

    def _run_per_step(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run using per-step Python orchestration (legacy path).

        Uses a JIT-compiled unified physics step with ``jax.lax.cond``
        for radiation sub-cycling (held tendencies reused between
        radiation update steps).

        This path is retained for debugging and as a reference
        implementation.  For production use, prefer ``run(compiled=True)``.
        """
        hyperdiffusion_3d = self._hyperdiffusion_3d_fn
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

        shape_2d = self.grid.grid_shape_2d
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
        o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
            START_DAY, self.state.p_s.data, self._grid_lat,
        )

        # Moisture conservation fixer
        FIX_MOISTURE = cfg.fix_moisture
        if FIX_MOISTURE:
            target_moisture = compute_global_moisture(
                self.q_v, self.state.p_s.data, dsigma, self.grid,
            )

        run_status = "COMPLETED"
        lat_deg_grid = np.degrees(np.asarray(self._grid_lat))

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
                sst, sic, self._grid_lat, self._grid_lon,
                day_of_year, seconds_of_day, DT,
                solar_weights, current_s_0,
                o3_vmr, aerosol_od,
                held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                ghg_vmr_override=ghg_vmr,
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

                # Ozone + aerosol + GHG
                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
                    day, self.state.p_s.data, self._grid_lat,
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
                    sst, sic, self._grid_lat, self._grid_lon,
                    day_of_year, seconds_of_day, DT,
                    solar_weights, current_s_0,
                    o3_vmr, aerosol_od,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                    ghg_vmr_override=ghg_vmr,
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
                diag_info = self._sync_and_collect_diagnostics(
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
