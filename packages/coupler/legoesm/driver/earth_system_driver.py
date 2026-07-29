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
from legoesm.driver.air_sea_consistency import validate_air_sea_consistency
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.model_driver import ModelDriver

from legoesm import constants

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
        # setup() materializes `self._coupler_config or CouplerConfig()`, whose
        # bulk_scheme is "constant" REGARDLESS of the atmosphere -- the same
        # split-interface bug guarded in CoupledESMDriver.  This driver builds
        # the same make_coupler ocean tile, so it needs the same guard (codex).
        validate_air_sea_consistency(config, coupler_config)
        self._atm = ModelDriver(config, output_dir=output_dir)
        # Same contract as CoupledESMDriver: this driver's _build_atm_forcing
        # reads held_sw_net_sfc / held_lw_net_sfc / seg_precip out of the
        # atmosphere's _carry_aux (see below), so a lane that never writes
        # them must refuse rather than force the surface with zeros.
        self._atm._requires_surface_flux_export = True
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
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.coupler.coupler import init_surface_state, make_coupler
        from legoesm.coupler.lake.config import LakeConfig
        from legoesm.ice.config import SeaIceConfig
        from legoesm.land.config import LandConfig

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
        # Moist-air density: rho = p / (R_d * T_v), T_v = T*(1 + (1/eps - 1)*q).
        # The dry form rho = p/(R_d*T) underestimates density by ~0.6% in the
        # tropics and biases every downstream bulk-flux surface stress /
        # turbulent flux that reads forcing.rho_lowest -- the SAME T_v
        # correction the canonical extract_atm_to_surface uses.
        T_v_low = T_low * (1.0 + (1.0 / constants.epsilon - 1.0) * q_low)
        rho_low = p_low / (constants.R_d * T_v_low)

        # Extract real radiation and precipitation from last atmosphere
        # physics.  STRICT, via the SAME shared checker CoupledESMDriver uses:
        # a lane that advanced a segment with an active radiation /
        # precipitation source must have stashed these, or the surface is
        # silently forced with sw_down=0 / precip=0.
        from legoesm.core.coupling_fields import require_surface_radiation_aux
        aux = getattr(self._atm, '_carry_aux', {})
        _acfg = self.config
        require_surface_radiation_aux(
            aux,
            radiation_active=(
                getattr(_acfg, "radiation", "none") not in (None, "none")),
            precip_active=(
                getattr(_acfg, "microphysics", "none") not in (None, "none")
                or getattr(_acfg, "convection", "none") not in (None, "none")),
            lane=type(self._atm).__name__,
        )
        sw_net_sfc = aux.get("held_sw_net_sfc", jnp.zeros_like(p_s))
        lw_net_sfc = aux.get("held_lw_net_sfc", jnp.zeros_like(p_s))
        seg_precip = aux.get("seg_precip", jnp.zeros_like(p_s))

        # Reconstruct gross downward fluxes from net fluxes.
        # sw_net = sw_down * (1 - albedo) → sw_down = sw_net / (1 - albedo)
        # lw_net = eps * lw_down - eps * sigma * T_sfc^4
        #        → lw_down = (lw_net + eps * sigma * T_sfc^4) / eps
        # The surface feedback (_segment_hook) feeds the coupler's blended
        # dynamic albedo / skin temperature back to radiation, so radiation
        # produced these held net fluxes with THOSE values (the response held
        # when the next segment's forcing was packed — still current here,
        # since _step_coupler updates _last_sfc_response only after this call).
        # Invert with the same albedo / T_sfc for a consistent gross flux;
        # fall back to the static blend before the first coupler step.
        cfg = self.config
        sst, sic = self._atm.get_sst_sic(day)
        from legoesm.forcing.surface_utils import (
            blend_surface_property,
            surface_emissivity_for_lw_inversion,
            surface_temperature_for_lw_boundary,
        )
        # _last_sfc_response is only set after the first coupler step; this
        # reconstruction runs before it on segment 0.
        _resp = getattr(self, "_last_sfc_response", None)
        _dyn_sfc = (
            _resp is not None and getattr(_resp, "albedo", None) is not None
        )
        # ALL radiation forms the LW boundary with the LW-derived T_rad: RRTMGP/
        # RRTMG with the paired (T_rad, eps_grid); gray/none as a black surface at
        # a brightness temperature.  Invert lw_net with the SAME (eps, T) radiation
        # emitted with — NOT the aerodynamic/sensible-heat T_sfc (the canopy
        # air-space temp Tc over vegetated cells).
        _radiation = getattr(self.config, "radiation", "gray")
        if _dyn_sfc:
            albedo_eff = _resp.albedo
            T_sfc = surface_temperature_for_lw_boundary(
                _radiation, T_rad=getattr(_resp, "T_rad", _resp.T_sfc),
                lw_up=_resp.lw_up)
        else:
            albedo_eff = blend_surface_property(
                sic,
                cfg.albedo_ice,
                cfg.albedo_ocean,
            )
            T_sfc = blend_surface_temperature(sst, sic, cfg.T_ice)
        sw_down = sw_net_sfc / jnp.maximum(1.0 - albedo_eff, 0.01)
        # Emissivity matching the emission: RRTMGP/RRTMG + feedback -> the
        # tile-blended eps_col; RRTMGP/RRTMG static -> the EXACT ocean/ice/land
        # emissivity blend the radiation pipeline emitted with (configured
        # emissivity_* values, not a constant ocean/ice approximation); gray/none
        # -> an idealized black surface (eps = 1.0).
        _phys = self._atm.physics
        eps_sfc = surface_emissivity_for_lw_inversion(
            _radiation,
            dynamic_emissivity=(
                getattr(_resp, "emissivity", None) if _dyn_sfc else None),
            static_sfc_emissivity=_phys.static_surface_emissivity(
                sic, land_active=_phys.f_land is not None),
        )
        lw_up_sfc = eps_sfc * constants.sigma_sb * T_sfc ** 4
        lw_down = (lw_net_sfc + lw_up_sfc) / jnp.maximum(eps_sfc, 0.01)

        # Snow fraction: smooth Wigmosta 1994 / Dai 2008 ramp (shared helper).
        # The prior hard step ``where(T_low < T_freeze, 1, 0)`` killed
        # d(snow)/d(T_low) on training/DA paths and miscounted mixed-phase
        # precip in the 0–4 °C band; the helper matches the coupled driver so
        # the two cannot diverge.
        from legoesm.forcing.surface_utils import snow_fraction
        precip_total = jnp.maximum(seg_precip, 0.0)
        snow_frac = snow_fraction(T_low, constants.T_freeze)
        precip_snow = precip_total * snow_frac

        # Cosine zenith: daily-mean approximation cos_zen = Q / S_0.  Route the day-of-year
        # through the atmosphere's seasonal insolation seam (iter 449/462) so the surface
        # insolation runs the SAME season as the atmosphere (config.insolation_start_doy) —
        # offset 0 (default) == day_to_calendar(day), byte-identical. Mirrors CoupledESMDriver
        # (iter 461); else the surface saw JANUARY insolation while the atmosphere did not.
        doy, _ = self._atm._calendar_for_radiation(day)
        lat = self._atm._grid_lat
        if lat is not None:
            from legoesm.atmosphere.physics.radiation.solar import (
                daily_mean_insolation, earth_orbit, earth_sun_distance_factor,
            )
            S_0 = cfg.S_0
            # Realistic orbit (Berger 1978) when enabled; None ⇒ circular.
            _orbit = (earth_orbit()
                      if getattr(cfg, "orbital_insolation", False) else None)
            _eccf = (earth_sun_distance_factor(float(doy), _orbit)
                     if _orbit is not None else 1.0)
            Q_daily = daily_mean_insolation(lat, float(doy), S_0=S_0,
                                            orbit=_orbit)
            # cos_zen is a geometric optical-path cosine: use the orbital
            # declination but divide out the (a/r)^2 flux factor so it stays
            # <= 1 (eccf scales flux, not the sun angle).  _eccf == 1.0 on the
            # circular orbit ⇒ bit-identical to the legacy path.
            cos_zen = jnp.clip(Q_daily / (_eccf * S_0), 0.0, 1.0)
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

        # Same seasonal insolation seam as the atmosphere (iter 449/462) for the surface
        # step's day-of-year; offset 0 (default) => identical.
        doy, _ = self._atm._calendar_for_radiation(day)

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

        NOTE (issue F4): unlike ``CoupledESMDriver._segment_hook`` this hook
        does NOT sub-cycle -- ``_step_coupler`` integrates the land/ice/lake
        surface with a SINGLE forward-Euler step of the full ``dt_segment``.
        That is acceptable only for the small idealized configs this
        REFERENCE/TEST-ONLY driver is exercised with (it is instantiated
        nowhere in packages/, src/ or scripts/ -- only in tests).  A real
        coupled run MUST use ``CoupledESMDriver``, which sub-cycles the
        surface + ocean + carbon at ``coupling_dt``; do NOT wire this driver
        into a production entry point without first adding the same
        n_sub/sub_dt loop, or a large ``dt_segment`` (now correctly reported
        per segment) becomes an unstable forward-Euler surface step.
        """
        if self._step_surface is None:
            return

        sfc_response = self._step_coupler(day, dt_segment)

        # Store the latest surface response and feed the coupler's blended
        # dynamic surface albedo + skin temperature back to the atmosphere's
        # radiation for the next segment.
        #
        # This replaces the long-dead ``driver._sfc_T_override`` write: that
        # attribute was never defined or read by ModelDriver (``hasattr`` was
        # permanently False), so the surface feedback silently never happened —
        # radiation kept using the frozen config albedo / skin temperature.
        # The real channel is ``ModelDriver.get_sfc_override``, which threads
        # the (albedo, T_sfc, emissivity) triple into radiation as a traced
        # SegmentForcing (no recompile, AD-safe — mirrors the SST/SIC feedback).
        # The emissivity carries the land tile's LAI-dependent eps_eff so the
        # atmospheric LW boundary matches the emissivity the land used to form
        # its conservative LW_out.
        self._last_sfc_response = sfc_response

        from legoesm.forcing.surface_utils import (
            surface_temperature_for_lw_boundary,
        )

        def _get_sfc_override(day, _self=self):
            r = _self._last_sfc_response
            if r is None or getattr(r, "albedo", None) is None:
                # Before the first coupler response, return NO override so the
                # radiation keeps its OWN internal land/ocean/ice blend — a
                # static ocean/ice seed here would run segment-zero land cells
                # with ocean/ice radiative properties.  The resulting one-time
                # seg0->seg1 None->array recompile is benign (one extra compile,
                # not a per-segment retrace).
                return None, None, None
            _cons = getattr(
                _self.config, "radiation", "gray") in ("rrtmgp", "rrtmg")
            if not _cons:
                # Gray/none emit as an idealized BLACK surface (eps = 1) and
                # cannot honour the canopy's eps_col, so feed them a black-surface
                # BRIGHTNESS temperature from the full upward flux
                # (sigma*T_bb^4 = LW_out).  Feeding T_rad would emit
                # sigma*T_rad^4 = LW_emit/eps_col and overstate canopy emission by
                # ~1/eps_col.  No emissivity override (gray keeps eps = 1).  NOT
                # the aerodynamic/sensible-heat T_sfc (the canopy air-space Tc).
                T_bb = surface_temperature_for_lw_boundary(
                    "gray", T_rad=getattr(r, "T_rad", r.T_sfc), lw_up=r.lw_up)
                return r.albedo, T_bb, None
            # RRTMGP/RRTMG: the tile-blended (albedo, T_rad, emissivity) flow in.
            # T_rad (radiative-equivalent skin T) + dynamic eps_grid make the LW
            # boundary eps*sigma*T^4 + (1-eps)*La reproduce the area-weighted sum
            # of tile lw_up exactly for mixed land/ocean/ice cells.
            return r.albedo, getattr(r, "T_rad", r.T_sfc), getattr(r, "emissivity", None)

        driver.get_sfc_override = _get_sfc_override

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
