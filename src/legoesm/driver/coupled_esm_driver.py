"""Fully coupled Earth System Model driver.

Orchestrates atmosphere + slab/dynamic ocean + land (slab/Richards')
+ sea ice + lake + optional carbon cycle into a single integration.

Builds on ``ModelDriver`` for atmosphere and ``make_coupler`` for
surface exchange.  The coupler + ocean step runs at segment boundaries
via a callback from ``ModelDriver.run()``.

Configuration is via ``CoupledConfig`` (see ``coupled_config.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.coupled_config import CoupledConfig

logger = logging.getLogger("legoesm.driver.coupled_esm")


class CoupledESMDriver:
    """Coupled atmosphere + ocean + land + carbon driver.

    Parameters
    ----------
    atm_config : ExperimentConfig
        Atmosphere experiment configuration.
    coupled_config : CoupledConfig
        Ocean/land/carbon mode selection.
    output_dir : str or Path, optional
    """

    def __init__(
        self,
        atm_config: ExperimentConfig,
        coupled_config: CoupledConfig | None = None,
        coupler_config=None,
        ice_config=None,
        lake_config=None,
        output_dir=None,
    ):
        self.atm_config = atm_config
        self.coupled_cfg = coupled_config or CoupledConfig()
        self._atm = ModelDriver(atm_config, output_dir=output_dir)
        self._coupler_config = coupler_config
        self._ice_config = ice_config
        self._lake_config = lake_config

        # Populated during setup()
        self._step_surface = None
        self._sfc_state = None
        self._ocean_state = None
        self._ocean_step = None
        self._last_sfc_response = None
        self._coupled_diag = []

    @property
    def output_dir(self) -> Path:
        return self._atm.output_dir

    # ==================================================================
    # Setup
    # ==================================================================

    def setup(self) -> None:
        """Initialize all components."""
        # 1. Atmosphere
        self._atm.setup()

        # 2. Ocean (slab / two-layer)
        self._init_ocean()

        # 3. Coupler (land + ice + lake + ocean tile blending)
        self._init_coupler()

        # 4. Carbon / CO2 tracer (if active)
        self._init_carbon()

        # 5. Override SST source: slab ocean instead of file
        self._override_sst()

        logger.info("CoupledESM: all components initialized")
        logger.info(f"  ocean_mode={self.coupled_cfg.ocean_mode}, "
                    f"land_mode={self.coupled_cfg.land_mode}, "
                    f"carbon_active={self.coupled_cfg.carbon_active}")

    def _init_ocean(self):
        """Initialize the slab/two-layer ocean."""
        from legoesm.ocean.simple_ocean import make_ocean, init_slab_state

        cfg = self.coupled_cfg
        shape_2d = self._atm.grid.grid_shape_2d

        # Get initial SST from the atmosphere's SST source (day 0)
        sst_init, _ = self._atm.get_sst_sic(0.0)
        T_sfc_mean = float(jnp.mean(sst_init))

        self._ocean_state = init_slab_state(
            shape_2d, T_sfc_init=T_sfc_mean,
        )
        # Override with the spatially varying AMIP SST
        from legoesm.core.field import Field
        self._ocean_state = self._ocean_state._replace(
            T_sfc=Field(
                data=jnp.array(sst_init, dtype=self._ocean_state.T_sfc.data.dtype),
                name="T_sfc", dims=self._ocean_state.T_sfc.dims,
                units="K",
            ),
        )

        self._ocean_step = make_ocean(cfg.ocean_config)
        logger.info(f"  Ocean: mode={cfg.ocean_mode}, "
                    f"h_mix={cfg.ocean_config.h_mix}m, "
                    f"T_sfc_init={T_sfc_mean:.1f}K")

    def _init_coupler(self):
        """Initialize coupler, land, ice, lake surface states."""
        from legoesm.coupler.coupler import make_coupler, init_surface_state
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.land.config import resolve_land_config
        from legoesm.ice.config import SeaIceConfig
        from legoesm.coupler.lake.config import LakeConfig
        from legoesm.core.precision import get_policy

        cfg = self.coupled_cfg
        shape_2d = self._atm.grid.grid_shape_2d
        _sd = get_policy().storage

        coupler_cfg = self._coupler_config or CouplerConfig()
        ice_cfg = self._ice_config or SeaIceConfig()
        lake_cfg = self._lake_config or LakeConfig()

        # Select land config based on mode (shared with run_lmip).
        land_cfg = resolve_land_config(cfg.land_mode, cfg.land_config)

        # Enable carbon in land config if carbon_active + differland
        if cfg.carbon_active and cfg.carbon_land == "differland":
            from legoesm.land.carbon.config import CarbonConfig
            carbon_cfg = CarbonConfig(scheme="differland")
            land_cfg = land_cfg._replace(carbon=carbon_cfg)

        self._land_cfg = land_cfg  # store for diagnostics

        # PFT parameter provider (if requested and land is active)
        land_param_provider = None
        if cfg.use_pft and cfg.land_mode != "none":
            land_param_provider = self._build_pft_provider(shape_2d)

        # Build coupler step function
        self._step_surface = make_coupler(
            coupler_cfg, land_cfg, ice_cfg, lake_cfg,
            lat=self._atm._grid_lat,
            grid=self._atm.grid,
            land_param_provider=land_param_provider,
        )

        # Initialize surface state
        self._sfc_state = init_surface_state(
            shape_2d, land_config=land_cfg,
        )

        # Tile fractions
        if cfg.f_land_mode == "zero":
            f_land = jnp.zeros(shape_2d, dtype=_sd)
        elif cfg.f_land_mode == "analytical":
            # Use driver's land mask if non-trivial; otherwise generate one
            # based on latitude (simple continents approximation).
            if (self._atm._f_land is not None
                    and float(jnp.max(self._atm._f_land)) > 0):
                f_land = self._atm._f_land
            elif cfg.land_mode != "none":
                # Generate analytical land mask: ~30% land by area
                # Land at |lat| > 20 in two longitude sectors
                lat = self._atm._grid_lat
                if lat is not None:
                    if lat.ndim < len(shape_2d):
                        lat_2d = jnp.broadcast_to(
                            lat.reshape(lat.shape + (1,) * (len(shape_2d) - lat.ndim)),
                            shape_2d,
                        )
                    else:
                        lat_2d = lat
                    abs_lat = jnp.abs(lat_2d) * 180.0 / jnp.pi
                    # Land where |lat| > 25 degrees (crude polar/midlat continents)
                    f_land = jnp.where(abs_lat > 25.0, 0.5, 0.0).astype(_sd)
                else:
                    f_land = jnp.zeros(shape_2d, dtype=_sd)
            else:
                f_land = jnp.zeros(shape_2d, dtype=_sd)
        else:
            f_land = jnp.zeros(shape_2d, dtype=_sd)

        self._tile_config = TileConfig(
            f_land=f_land,
            f_lake=jnp.zeros(shape_2d, dtype=_sd),
        )

        land_frac = float(jnp.mean(f_land))
        pft_str = " (PFT)" if land_param_provider is not None else ""
        logger.info(f"  Land: mode={cfg.land_mode}{pft_str}, "
                    f"f_land_mean={land_frac:.2f}")

    def _build_pft_provider(self, shape_2d):
        """Create a PFTParamProvider with analytical PFT fractions."""
        import math
        from legoesm.land.param_providers import PFTParamProvider

        lat = self._atm._grid_lat
        if lat is None:
            logger.warning("  PFT requested but no latitude available; "
                           "falling back to scalar params")
            return None

        # Flatten to (ncol,)
        lat_flat = jnp.ravel(lat) if lat.ndim > 1 else lat
        ncol = lat_flat.shape[0]
        if lat.ndim > 1:
            ncol = math.prod(shape_2d)
            lat_flat = jnp.broadcast_to(lat, shape_2d).ravel()

        abs_lat_deg = jnp.abs(lat_flat) * 180.0 / jnp.pi

        # 17 CLM5 PFTs — assign analytical fractions by latitude band
        # 0=bare_soil, 1=NET_temperate, 2=NET_boreal, 3=NDT_boreal,
        # 4=BET_tropical, 5=BET_temperate, 6=BDT_tropical, 7=BDT_temperate,
        # 8=BDT_boreal, 9=BES, 10=BDS_temperate, 11=BDS_boreal,
        # 12=C3_arctic, 13=C3_non_arctic, 14=C4, 15=crop, 16=bare_soil_2
        n_pft = 17
        fracs = jnp.zeros((ncol, n_pft))

        # Tropical broadleaf (|lat| < 15)
        tropical = (abs_lat_deg < 15.0).astype(jnp.float32)
        fracs = fracs.at[:, 4].set(0.7 * tropical)   # BET_tropical
        fracs = fracs.at[:, 14].set(0.3 * tropical)   # C4 grass

        # Temperate (15-45)
        temperate = ((abs_lat_deg >= 15.0) & (abs_lat_deg < 45.0)).astype(jnp.float32)
        fracs = fracs.at[:, 7].set(0.3 * temperate)   # BDT_temperate
        fracs = fracs.at[:, 13].set(0.4 * temperate)  # C3_non_arctic
        fracs = fracs.at[:, 15].set(0.3 * temperate)  # crop

        # Boreal (45-65)
        boreal = ((abs_lat_deg >= 45.0) & (abs_lat_deg < 65.0)).astype(jnp.float32)
        fracs = fracs.at[:, 2].set(0.5 * boreal)    # NET_boreal
        fracs = fracs.at[:, 12].set(0.3 * boreal)   # C3_arctic
        fracs = fracs.at[:, 0].set(0.2 * boreal)    # bare_soil

        # Polar (>65)
        polar = (abs_lat_deg >= 65.0).astype(jnp.float32)
        fracs = fracs.at[:, 0].set(0.7 * polar)     # bare_soil
        fracs = fracs.at[:, 12].set(0.3 * polar)    # C3_arctic

        provider = PFTParamProvider.from_defaults(fracs)
        logger.info(f"  PFT: {n_pft} types, {ncol} columns, "
                    f"analytical latitude-band fractions")
        return provider

    def _init_carbon(self):
        """Set up CO2 tracer in the atmosphere if carbon is active."""
        cfg = self.coupled_cfg
        if not cfg.co2_tracer:
            return

        # CO2 as a prognostic atmospheric tracer.
        # Mixing ratio: co2_ppmv * 1e-6 * (M_CO2 / M_air).
        # Molar masses come from ``legoesm.constants`` per CLAUDE.md
        # (no hardcoded physical constants in production code).
        co2_init_kgkg = (
            cfg.co2_ppmv_init * 1.0e-6
            * (constants.M_CO2 / constants.M_air)
        )

        # Get 3D shape from atmosphere state
        T_data = self._atm.state.T.data
        shape_3d = T_data.shape

        self._co2_field = jnp.full(shape_3d, co2_init_kgkg,
                                   dtype=T_data.dtype)
        logger.info(f"  CO2 tracer: init={cfg.co2_ppmv_init:.1f} ppmv "
                    f"({co2_init_kgkg:.6e} kg/kg), shape={shape_3d}")

    def _override_sst(self):
        """Replace the atmosphere's file-based SST with slab ocean SST."""
        # Store the original for fallback SIC
        self._original_get_sst_sic = self._atm.get_sst_sic

        def _coupled_get_sst_sic(day):
            # SST from slab ocean
            sst = self._ocean_state.T_sfc.data
            # SIC from prognostic sea ice state (if available) or file
            if (self._sfc_state is not None
                    and hasattr(self._sfc_state, 'ice')
                    and self._sfc_state.ice is not None):
                sic = self._sfc_state.ice.concentration.data
            else:
                _, sic = self._original_get_sst_sic(day)
            return sst, sic

        self._atm.get_sst_sic = _coupled_get_sst_sic
        logger.info("  SST override: atmosphere reads SST from slab ocean")

    # ==================================================================
    # Coupling step
    # ==================================================================

    def _build_atm_forcing(self, day: float):
        """Build AtmToSurface from atmosphere state and physics."""
        from legoesm.coupler.coupling_fields import AtmToSurface
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

        # Radiation and precipitation from last atmosphere physics
        aux = getattr(self._atm, '_carry_aux', {})
        sw_net_sfc = aux.get("held_sw_net_sfc", jnp.zeros_like(p_s))
        lw_net_sfc = aux.get("held_lw_net_sfc", jnp.zeros_like(p_s))
        seg_precip = aux.get("seg_precip", jnp.zeros_like(p_s))

        # Reconstruct gross downward fluxes from net
        acfg = self.atm_config
        sst, sic = self._atm.get_sst_sic(day)
        from legoesm.forcing.surface_utils import blend_surface_property
        albedo_eff = blend_surface_property(
            sic,
            getattr(acfg, 'albedo_ice', 0.6),
            getattr(acfg, 'albedo_ocean', 0.06),
        )
        sw_down = sw_net_sfc / jnp.maximum(1.0 - albedo_eff, 0.01)

        T_sfc = blend_surface_temperature(sst, sic, acfg.T_ice)
        # Surface emissivity comes from the coupler config (per-tile
        # ocean/ice/land emissivity is blended via tile fractions
        # downstream).  The 0.96 broad-spectrum default lives in the
        # ``CoupledDriverConfig.surface_emissivity`` field, falling
        # back to the canonical ocean emissivity from
        # ``constants.emissivity_ocean`` if not set.
        eps_sfc = getattr(
            self.coupled_cfg, "surface_emissivity",
            constants.emissivity_ocean,
        )
        lw_up_sfc = eps_sfc * constants.sigma_sb * T_sfc ** 4
        lw_down = (lw_net_sfc + lw_up_sfc) / jnp.maximum(eps_sfc, 0.01)

        precip_total = jnp.maximum(seg_precip, 0.0)
        # Smooth snow fraction (Wigmosta 1994 / Dai 2008): ramp from 0
        # at T_low = T_freeze + 2 K to 1 at T_low = T_freeze - 2 K.
        # The prior hard step ``where(T_low < T_freeze, 1, 0)`` killed
        # gradients (training/DA paths) and miscounted mixed-phase
        # precipitation in the 0–4 °C band.
        snow_frac = jnp.clip(
            (constants.T_freeze + 2.0 - T_low) / 4.0, 0.0, 1.0,
        )
        precip_snow = precip_total * snow_frac

        # Cosine zenith
        from legoesm.forcing.time_utils import day_to_calendar
        doy, _ = day_to_calendar(day)
        lat = self._atm._grid_lat
        if lat is not None:
            from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
            # Solar constant from legoesm.constants per CLAUDE.md.
            # ``acfg.S_0`` allows override for sensitivity studies.
            S_0 = getattr(acfg, 'S_0', constants.S_0)
            Q_daily = daily_mean_insolation(lat, float(doy), S_0=S_0)
            cos_zen = jnp.clip(Q_daily / S_0, 0.0, 1.0)
        else:
            cos_zen = jnp.full_like(p_s, 0.5)

        # CO2: prognostic or constant
        if self.coupled_cfg.co2_tracer and hasattr(self, '_co2_field'):
            co2_lowest = self._co2_field[..., -1]
            co2_ppmv = (
                co2_lowest / (constants.M_CO2 / constants.M_air) * 1.0e6
            )
        else:
            co2_ppmv = jnp.full_like(p_s, self.atm_config.co2_ppmv)

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
            co2_ppmv=co2_ppmv,
            has_radiation=jnp.ones_like(p_s),
            has_precipitation=jnp.where(precip_total > 0, 1.0, 0.0),
        )

    def _step_ocean(self, atm_forcing, dt):
        """Advance the slab ocean one coupling step.

        **One-way ice -> ocean coupling (intentional for the slab ocean).**
        ``step_sea_ice`` populates ice -> ocean back-reaction channels on
        the surface response (``freshwater_flux``, ``ocean_heat_extraction``,
        ``salt_flux``, ``ocean_stress_x``/``ocean_stress_y``), but this
        driver advances the :class:`SimpleOcean` slab, a thermodynamic
        mixed-layer model with no prognostic salinity and no prognostic
        momentum.  It therefore cannot consume those feedbacks:

        * ``salt_flux`` / ``freshwater_flux`` -> no salinity prognostic;
        * ``ocean_stress_x``/``ocean_stress_y`` -> no momentum prognostic
          (the slab returns ``u_sfc = v_sfc = 0``);
        * ``ocean_heat_extraction`` -> the slab already diagnoses its own
          ``Q_freeze`` (the heat removed by its freezing clamp); adding the
          ice's basal heat extraction on top would double-count against it.

        The channels remain available on ``self._last_sfc_response`` for a
        full prognostic ocean (salinity + momentum + a two-way
        ``Q_freeze`` <-> ice-seeding contract), which is tracked as separate
        feature work; they are deliberately NOT applied to the slab here.
        """
        if self._ocean_step is None:
            return
        self._ocean_state, sst_new, u_sfc, v_sfc = self._ocean_step(
            self._ocean_state, atm_forcing, dt,
        )
        self._ocean_u_sfc = u_sfc
        self._ocean_v_sfc = v_sfc

    def _step_co2_tracer(self, dt):
        """Apply blended surface CO2 flux to lowest atmospheric level."""
        if not self.coupled_cfg.co2_tracer or not hasattr(self, '_co2_field'):
            return
        if self._last_sfc_response is None:
            return

        co2_flux = self._last_sfc_response.co2_flux  # kgCO2/m2/s, +up
        p_s = self._atm.state.p_s.data
        dsigma = jnp.asarray(self._atm.sigma.dsigma)
        # Layer mass of lowest level: dp / g [kg/m2]
        dp = p_s * dsigma[-1]
        mass_air = dp / constants.g
        dco2 = co2_flux / jnp.maximum(mass_air, 1.0) * dt
        self._co2_field = self._co2_field.at[..., -1].add(dco2)

    def _segment_hook(self, driver, day, dt_segment):
        """Callback at each segment boundary: step ocean + coupler.

        The segment may span many atmosphere time steps (e.g. an entire
        diagnostic interval).  Surface coupling — ocean, land, ice, and
        carbon — must be sub-cycled at ``coupling_dt`` (default 3600 s)
        so that the carbon cycle's forward-Euler integration remains
        stable and fluxes are physically consistent.
        """
        from legoesm.forcing.time_utils import day_to_calendar

        coupling_dt = self.coupled_cfg.coupling_dt  # default 3600 s
        n_sub = max(1, int(round(dt_segment / coupling_dt)))
        sub_dt = dt_segment / n_sub

        atm_forcing = self._build_atm_forcing(day)

        for _ in range(n_sub):
            # Step slab ocean
            self._step_ocean(atm_forcing, sub_dt)

            # Step coupler (land, ice, lake, ocean tile blending)
            sst = self._ocean_state.T_sfc.data
            u_sfc = getattr(self, '_ocean_u_sfc', jnp.zeros_like(sst))
            v_sfc = getattr(self, '_ocean_v_sfc', jnp.zeros_like(sst))

            doy, _ = day_to_calendar(day)

            self._sfc_state, sfc_response = self._step_surface(
                self._sfc_state,
                atm_forcing,
                self._tile_config,
                ocean_sst=sst,
                ocean_u_sfc=u_sfc,
                ocean_v_sfc=v_sfc,
                dt=sub_dt,
                doy=float(doy),
            )
            self._last_sfc_response = sfc_response

            # CO2 tracer update
            self._step_co2_tracer(sub_dt)

        # Diagnostics (once per segment, not per sub-step)
        self._log_coupled_diag(day)

    def _log_coupled_diag(self, day):
        """Record coupled diagnostics for this segment.

        Stacks all reductions into one ``jnp.stack`` and pulls them in
        a single ``np.asarray`` transfer.  Each ``float(jnp.X(...))``
        was previously its own device→host sync, serialising 3-5
        GPU stalls per coupling segment.
        """
        sst = self._ocean_state.T_sfc.data
        has_co2 = self.coupled_cfg.co2_tracer and hasattr(self, '_co2_field')
        has_T_sfc = self._last_sfc_response is not None

        terms = [jnp.mean(sst), jnp.min(sst), jnp.max(sst)]
        if has_co2:
            terms.append(jnp.mean(self._co2_field))
        if has_T_sfc:
            terms.append(jnp.mean(self._last_sfc_response.T_surface))
        host = np.asarray(jnp.stack(terms))

        diag = {
            "day": float(day),
            "sst_mean": float(host[0]),
            "sst_min": float(host[1]),
            "sst_max": float(host[2]),
        }
        idx = 3
        if has_co2:
            diag["co2_ppmv_mean"] = (
                float(host[idx])
                / (constants.M_CO2 / constants.M_air) * 1e6
            )
            idx += 1
        if has_T_sfc:
            diag["T_sfc_mean"] = float(host[idx])

        self._coupled_diag.append(diag)

    # ==================================================================
    # Run
    # ==================================================================

    def run(self, start_step: int = 0, start_day: float | None = None) -> str:
        """Run the coupled integration."""
        logger.info("Starting coupled ESM run")
        status = self._atm.run(
            start_step=start_step,
            start_day=start_day,
            segment_callback=self._segment_hook,
        )
        logger.info(f"Coupled ESM run: {status}")
        return status

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def state(self):
        return self._atm.state

    @property
    def ocean_state(self):
        return self._ocean_state

    @property
    def surface_state(self):
        return self._sfc_state

    @property
    def diagnostics(self):
        return self._atm.diagnostics

    @property
    def coupled_diagnostics(self):
        return self._coupled_diag

    def save_checkpoint(self, step: int, day: float) -> None:
        """Save atmosphere + ocean + surface + CO2 state."""
        self._atm.save_checkpoint(step, day)

        # Save coupled state alongside the atmosphere checkpoint.
        elapsed_day = day - self.atm_config.start_day
        coupled_path = self.output_dir / f"coupled_day_{int(elapsed_day):04d}.npz"
        arrays = {}

        # Ocean state (SlabOceanState is a NamedTuple of Fields)
        if self._ocean_state is not None:
            arrays["ocean_T_sfc"] = np.asarray(self._ocean_state.T_sfc.data)
            arrays["ocean_T_deep"] = np.asarray(self._ocean_state.T_deep.data)

        # CO2 tracer field
        if hasattr(self, '_co2_field') and self._co2_field is not None:
            arrays["co2_field"] = np.asarray(self._co2_field)

        # Surface state (land, ice, lake, accumulator, carbon) — flatten
        # the pytree into a dict of named arrays for serialization.
        if self._sfc_state is not None:
            leaves_with_path = jax.tree_util.tree_leaves_with_path(
                self._sfc_state,
            )
            for path_parts, leaf in leaves_with_path:
                key = "sfc_" + ".".join(str(p) for p in path_parts)
                arrays[key] = np.asarray(leaf)

        if arrays:
            np.savez(coupled_path, **arrays)
            logger.info(f"  Coupled checkpoint: {coupled_path.name}")

    def load_coupled_checkpoint(self, step_day: float,
                               checkpoint_dir: str | Path | None = None) -> None:
        """Load coupled state saved alongside an atmosphere checkpoint.

        Parameters
        ----------
        step_day : float
            Elapsed day used in the filename (same as atmosphere checkpoint).
        checkpoint_dir : str or Path, optional
            Directory containing the coupled checkpoint.  Defaults to
            ``self.output_dir``.
        """
        from legoesm.core.field import Field

        base = Path(checkpoint_dir) if checkpoint_dir is not None else self.output_dir
        coupled_path = base / f"coupled_day_{int(step_day):04d}.npz"
        if not coupled_path.exists():
            logger.warning(f"No coupled checkpoint at {coupled_path}")
            return

        data = np.load(coupled_path)

        if "ocean_T_sfc" in data and self._ocean_state is not None:
            self._ocean_state = self._ocean_state._replace(
                T_sfc=Field(
                    data=jnp.asarray(data["ocean_T_sfc"]),
                    name="T_sfc",
                    dims=self._ocean_state.T_sfc.dims,
                    units="K",
                ),
                T_deep=Field(
                    data=jnp.asarray(data["ocean_T_deep"]),
                    name="T_deep",
                    dims=self._ocean_state.T_deep.dims,
                    units="K",
                ),
            )

        if "co2_field" in data:
            self._co2_field = jnp.asarray(data["co2_field"])

        # Restore surface state from flattened pytree leaves.
        sfc_keys = [k for k in data.files if k.startswith("sfc_")]
        if sfc_keys and self._sfc_state is not None:
            leaves_with_path = jax.tree_util.tree_leaves_with_path(
                self._sfc_state,
            )
            # Build a lookup from stringified path -> saved array
            saved = {}
            for k in sfc_keys:
                saved[k] = data[k]

            # Replace leaves in-order (same traversal as save)
            new_leaves = []
            for path_parts, leaf in leaves_with_path:
                key = "sfc_" + ".".join(str(p) for p in path_parts)
                if key in saved:
                    new_leaves.append(jnp.asarray(saved[key]))
                else:
                    new_leaves.append(leaf)

            self._sfc_state = jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(self._sfc_state),
                new_leaves,
            )

        logger.info(f"  Loaded coupled checkpoint: {coupled_path.name}")
