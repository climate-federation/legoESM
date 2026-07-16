"""Physics pipeline for the composable model driver.

Wraps radiation, convection, microphysics, and boundary-layer exchange
into a single callable that replaces the inline physics step in run_amip.py.

The pipeline is **grid-agnostic**: all flattening/unflattening between
native grid layout and ``(ncol, nlev)`` column format is handled by a
``ColumnAdapter`` (see ``grid_adapters.py``).  Scheme selection is
**registry-driven**: see ``kernel_registry.py``.
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.forcing.surface_utils import (
    blend_surface_property,
    blend_surface_temperature,
)
from legoesm.core.grid_adapters import make_adapter
from legoesm.core.physics_output import PhysicsOutput  # shared tendency pytree (moved to core)


def _pin_carry_dtype(updated, carry_in):
    """Pin an updated stateful-physics carry to its input dtype.

    Keeps the per-step carry dtype-stable (lax.scan requirement and the
    issue-#413 feed-back contract).  ``carry_in is None`` (fresh seed by
    the pipeline's warm-start fallback) leaves the update unpinned — the
    fallback seeds at the column dtype, which is then the stable dtype.
    """
    if updated is None or carry_in is None:
        return updated
    if updated.dtype != carry_in.dtype:
        return updated.astype(carry_in.dtype)
    return updated


class PhysicsPipeline:
    """Encapsulates the full physics pipeline for operator-split stepping.

    Combines radiation, convection, microphysics, and boundary-layer
    exchange into a single JIT-compiled function.  Manages radiation
    sub-cycling (held tendencies) internally.

    All grid-specific flattening is delegated to a ``ColumnAdapter``,
    and all scheme selection is resolved at build time via the kernel
    registries — the hot path contains no ``if/elif`` dispatch.

    Parameters
    ----------
    adapter : ColumnAdapter
        Grid-agnostic column adapter for reshape operations.
    sigma_full : jax.Array
        Sigma at full levels, shape (nlev,).
    sigma_half : jax.Array
        Sigma at half levels, shape (nlev+1,).
    dsigma : jax.Array
        Layer thickness in sigma, shape (nlev,).
    convection_fn : callable
        Resolved convection kernel (column-format in/out).
    convection_config : object
        Configuration NamedTuple for the convection kernel.
    radiation_fn : callable
        JIT-compiled radiation function (column-format in/out).
    T_ice : float
        Sea-ice temperature [K].
    C_H : float
        Sensible heat exchange coefficient.
    C_E : float
        Latent heat exchange coefficient.
    albedo_ice : float
        Sea-ice albedo.
    albedo_ocean : float
        Ocean albedo.
    emissivity_ice : float
        Sea-ice emissivity.
    emissivity_ocean : float
        Ocean emissivity.
    micro_fn : callable or None
        Microphysics function.
    micro_config : object or None
        Microphysics backend config.
    dynamic_albedo : bool
        Use temperature/zenith-dependent albedo.
    """

    def __init__(
        self,
        adapter,
        sigma_full,
        sigma_half,
        dsigma,
        convection_fn,
        convection_config,
        radiation_fn,
        T_ice=constants.T_freeze_ocean,
        C_H=None,
        C_E=None,
        albedo_ice=0.65,
        albedo_ocean=0.06,
        emissivity_ice=0.95,
        emissivity_ocean=0.97,
        emissivity_land=0.96,
        C_land=2.0e5,
        micro_fn=None,
        micro_config=None,
        dynamic_albedo=False,
        diurnal_cycle=False,
        turbulence_fn=None,
        turbulence_config=None,
        gwd_fn=None,
        gwd_config=None,
        physics_parameterization=None,
        column_mesh=None,
    ):
        self.adapter = adapter
        self.sigma_full = sigma_full
        self.sigma_half = sigma_half
        self.dsigma = dsigma
        self.convection_fn = convection_fn
        self.convection_config = convection_config
        self.radiation_fn = radiation_fn
        self.T_ice = T_ice
        # Resolve the surface exchange coefficients to the canonical
        # ``ExperimentConfig`` defaults when not supplied, so the single
        # source of truth lives in the config schema (the sole caller
        # ``build_physics_pipeline`` always passes explicit values).
        if C_H is None or C_E is None:
            from legoesm.driver.config import ExperimentConfig
            _defaults = ExperimentConfig._field_defaults
            if C_H is None:
                C_H = _defaults["C_H"]
            if C_E is None:
                C_E = _defaults["C_E"]
        self.C_H = C_H
        self.C_E = C_E
        self.albedo_ice = albedo_ice
        self.albedo_ocean = albedo_ocean
        self.emissivity_ice = emissivity_ice
        self.emissivity_ocean = emissivity_ocean
        self.emissivity_land = emissivity_land
        # Slab-land heat capacity [J/m^2/K] — effective for a ~0.15 m
        # active soil layer (rho*c ~ 1.4e6 J/m^3/K).  Updated once per
        # radiation call via a semi-implicit surface energy balance.
        self.C_land = C_land
        # Land surface fields — None disables the land tile entirely
        # (pure ocean/ice surface).  Set post-construction by the driver:
        #   f_land      : (..., ) land fraction in [0, 1]
        #   albedo_land : (..., ) land surface albedo
        # rad_update_steps is the radiation sub-cycle cadence; the slab
        # land steps by ``rad_update_steps * dt`` each radiation call.
        self.f_land = None
        self.albedo_land = None
        self.rad_update_steps = 1
        # Optional MULTILAYER (Richards) land tile.  When ``land_ml_cfg`` is set the
        # differentiable forward advances a MultiLayerLandState (carried in
        # SegmentCarry.land_ml) in place of the scalar-T_land slab, supplying the land
        # surface temperature and albedo for the blend.  All None ⇒ slab path.
        self.land_ml_cfg = None        # MultiLayerLandConfig
        self.land_ml_params = None     # LandSurfaceParams (per land column)
        self.land_ml_lat = None        # (ncol,) latitude [rad], column order
        self.land_ml_doy = 0.0
        self.land_ml_u_min = 1.0
        # Optional PRESCRIBED carbon state (fixed leaf carbon -> fixed LAI) for the
        # multilayer tile.  None (default) ⇒ no carbon coupling (Jarvis stomata /
        # byte-identical).  When set (+ land_ml_cfg.stomata.enabled +
        # carbon="differland") the Farquhar photosynthesis-stomata path activates,
        # making Vc_max25 / g1 / LCMA affect the surface flux — i.e. TRAINABLE in the
        # coupled calibration — without paying a multi-decade carbon-pool spin-up.
        self.land_ml_carbon = None     # CarbonState (prescribed) or None
        # When True, T_land is stepped each radiation call (full slab-land
        # tile, --land-mask-file path).  When False, T_land is carried but
        # NOT updated — the land albedo/T_sfc blend still applies (passive
        # mode, --topography path with no explicit land IC).
        self.slab_land_active = False
        # Tiled (mosaic) surface fluxes: when True, the turbulent surface
        # fluxes are computed SEPARATELY per surface tile (ocean / sea-ice /
        # land) — the ocean bulk scheme (e.g. COARE3) on the ocean tile and
        # the fixed-roughness land Monin-Obukhov scheme ("most") on the land
        # tile — then area-weighted, instead of applying one scheme to the
        # blended surface temperature (which runs the ocean scheme over
        # land, the bm_v3 blowup).  Set post-construction by the driver from
        # ExperimentConfig.surface_tiled.  ``surface_z0_land`` is the land
        # roughness length [m] used by the land tile's MOST scheme.
        self.surface_tiled = False
        self.surface_z0_land = 0.1
        # Prognostic soil-water bucket (Manabe 1969) for the slab-land tile.
        # When True, the land evaporation efficiency beta = beta_min +
        # (1-beta_min)*clip(W/W_max, 0, 1) (the SAME formula as
        # legoesm.land.slab_land) limits land latent heat by soil wetness,
        # instead of evaporating at the saturated (swamp) rate everywhere.
        # ``w_land`` (soil water [kg/m^2]) is a prognostic carry threaded like
        # ``T_land``.  Set post-construction by the driver from
        # ExperimentConfig.  All no-ops (beta=1, byte-identical to the
        # wet-surface path) when ``land_soil_bucket`` is False.
        self.land_soil_bucket = False
        self.land_bucket_w_max = 150.0     # bucket capacity [kg/m^2]
        self.land_beta_min = 0.1           # min moisture availability (dry soil)
        self.land_bucket_w_init_frac = 0.5  # initial fill fraction of W_max
        # Bucket runoff partition (Green-Ampt infiltration excess + saturation
        # excess), shared with legoesm.land.slab_land via partition_bucket_runoff.
        # Set post-construction by the driver from ExperimentConfig.
        self.land_K_infiltration = 1.0e-5    # saturated infiltration capacity [m/s]
        self.land_infil_suction_boost = 2.0  # Green-Ampt suction enhancement [-]
        self.land_infiltration_excess = True  # enable Hortonian infiltration excess
        # ``land_stomatal_beta``: route the soil-water availability through the
        # SHARED land stomatal model (Jarvis 1976) instead of the bare bucket
        # ramp — the soil beta becomes ``min(beta_soil, beta_canopy)`` where
        # the canopy term closes stomata in low light / high VPD (Pierre's
        # steer: use the existing stomatal soil-water limitation, not a
        # parallel bucket).  ``stomata_config`` (a land ``StomataConfig``) is
        # set by the driver when enabled.  Off → soil-only beta (the prior
        # bucket path, byte-identical).  Requires ``land_soil_bucket`` (the
        # bucket supplies ``beta_soil``).
        self.land_stomatal_beta = False
        self.stomata_config = None
        # Prognostic snow + snow-albedo feedback on the slab-land tile.  ``snow``
        # (SWE [kg/m^2]) + ``snow_age`` [s] are prognostic carries threaded like
        # ``w_land``; when ``snow_albedo_feedback`` is on the land albedo is
        # brightened by the snow cover (``legoesm.surface_albedo.land_albedo``).
        # Set by the driver from ExperimentConfig.  Off → static vegetation
        # albedo (byte-identical).
        self.snow_albedo_feedback = False
        from legoesm.surface_albedo import LandAlbedoConfig
        self.land_albedo_config = LandAlbedoConfig()
        self.micro_fn = micro_fn
        self.micro_config = micro_config
        # ``dynamic_albedo``: zenith-angle-dependent ocean albedo
        # (Briegleb 1992 via ``legoesm.surface_albedo.ocean_albedo``)
        # applied in ``compute_radiation_core`` before the sea-ice/land
        # blends.  ``diurnal_cycle`` selects instantaneous cos(SZA) vs
        # the daytime-effective daily-mean cosine — matching the zenith
        # convention the radiation solver itself uses.
        self.dynamic_albedo = dynamic_albedo
        # Realistic Earth orbit (Berger 1978) for AMIP-II/CMIP insolation;
        # None ⇒ circular orbit.  Set by ``build_physics_pipeline`` from
        # ``ExperimentConfig.orbital_insolation``; used by the radiation
        # builders (closed over ``config``) and the diagnostics below.
        self.orbit = None
        self.diurnal_cycle = diurnal_cycle
        self.turbulence_fn = turbulence_fn
        self.turbulence_config = turbulence_config
        self.gwd_fn = gwd_fn
        self.gwd_config = gwd_config
        self.physics_parameterization = physics_parameterization
        # Issue #273 follow-up: optional column-shard mesh for the
        # per-column radiation kernel.  When supplied, the column-format
        # arrays passed into ``self.radiation_fn`` are placed on the
        # mesh's ``'col'`` axis so radiation executes distributed
        # without changing the JIT'd radiation kernel itself.
        # ``None`` (default) preserves bit-exact single-mesh behavior.
        self.column_mesh = column_mesh
        self._cloud_scheme = "none"  # set by build_physics_pipeline
        # Clear-sky diagnostic (#843): static Python bool set by
        # build_physics_pipeline from config.output.clear_sky_diag.  When True,
        # the compiled segment runs a SECOND clouds-off compute_radiation_core
        # pass to produce CMOR rsutcs/rlutcs; when False (default) every
        # clear-sky code path is a byte-identical no-op.
        self._clear_sky_diag = False  # set by build_physics_pipeline (#843)
        # Opt-in convective cumulus cloud-fraction source (set by
        # build_physics_pipeline from ExperimentConfig.convective_cloud).
        # When True, compute_radiation_core feeds the lagged convective precip
        # to the cloud diagnosis so the convecting tropics get radiative cloud.
        self._cloud_convective = False
        # Optional cloud-tuning overrides (None => CloudConfig default =>
        # byte-identical); set by build_physics_pipeline from ExperimentConfig.
        self._cloud_rh_crit = None
        self._cloud_q_c_diagnostic = None
        self._cloud_conv_cloud_max = None
        self._cloud_conv_cloud_condensate = None
        self._cloud_inhomogeneity_factor = None
        self._cloud_optics_inhomogeneity = None
        self._cloud_fsd = None
        self._cloud_p_xr = None
        self._cloud_alpha_xr = None
        self._cloud_diagnostic_condensate_scheme = None
        self._cloud_adiabatic_lwc_rate = None
        # Convection scheme name + grid/vertical-coordinate objects for
        # grid-operator-backed convection inputs (moisture convergence,
        # resolved w, CMT winds).  Set by build_physics_pipeline; with
        # the defaults the profile-prognostic input plumbing is inert
        # (consuming schemes engage their built-in proxies).
        self._conv_scheme = "none"
        self._grid = None
        self._sigma_coord = None
        # Stateful-physics carry plumbing (issue #413).  Set by
        # build_physics_pipeline from the experiment config; the
        # defaults keep every carry slot inert (diagnostic schemes).
        # ``_turb_energy_field`` is the kernel keyword AND PhysicsState
        # slot ("tke" / "qke") from the shared turbulence_scheme_traits;
        # ``_gwd_prognostic`` marks the spectral GWD scheme whose wave-
        # action spectrum threads through ``gwd_spectrum``.
        self._turb_energy_field = None
        self._gwd_prognostic = False
        # ``_gwd_orographic`` marks schemes (single or '+'-composite) whose
        # launch stress accepts the per-column ``h_topo_col``;
        # ``subgrid_topo_stddev`` is the grid-shaped SSO field the driver
        # attaches from ``subgrid_orography_path`` (and re-scatters under
        # MPI, like ``f_land``). None -> kernels use scalar config.h_topo.
        self._gwd_orographic = False
        self.subgrid_topo_stddev = None
        # Set for a stateless '+'-composite GWD (issue #834): the combined
        # executor returns a (GWDOutput, spectrum) tuple even with no stateful
        # part, so the pipeline must unpack it.
        self._gwd_composite = False

    def _blend_land(self, ocean_field, land_field):
        """Blend an ocean/ice surface field with a land field by ``f_land``.

        ``f_land`` broadcasts against the 2-D surface fields.  Only called
        when ``self.f_land is not None`` (the land tile is active).
        """
        return self.f_land * land_field + (1.0 - self.f_land) * ocean_field

    def static_surface_emissivity(self, sic, *, land_active):
        """Surface LW emissivity blend radiation emits with absent an override.

        Mirrors the ocean/ice (+ optional land) blend formed in
        ``compute_radiation_core`` so the coupled drivers can invert the held
        ``lw_net_sfc`` back to gross ``lw_down`` with the SAME emissivity field
        radiation actually used — not a constant ocean/ice approximation (which
        ignores the configured ``emissivity_*`` values and the land tile, biasing
        the reconstructed surface forcing).

        Parameters
        ----------
        sic : array
            Sea-ice concentration [0, 1].
        land_active : bool
            Whether the land tile contributes (``f_land`` set AND a land skin
            temperature present); matches ``compute_radiation_core``'s gate.
        """
        emissivity = blend_surface_property(
            sic, self.emissivity_ice, self.emissivity_ocean)
        if land_active and self.f_land is not None:
            emissivity = self._blend_land(emissivity, self.emissivity_land)
        return emissivity
    def _land_surface_bulk(self, T_low, u_low, v_low, p_s):
        """Lowest-level air density [kg/m^3] and wind speed [m/s] for the
        land surface bulk fluxes.

        Shared by the slab-land SEB step and the soil-water bucket so the
        two use ONE consistent transfer estimate (no re-derived bulk
        formula).  ``1.0`` is the [m^2/s^2] wind-speed floor.
        """
        rho_low = (p_s * self.sigma_full[-1]) / (constants.R_d * T_low)
        wind_speed = jnp.sqrt(u_low ** 2 + v_low ** 2 + 1.0)
        return rho_low, wind_speed

    def _land_beta(self, w_land, T_land=None, sw_down_sfc=None,
                   q_air=None, p_s=None):
        """Soil-moisture evaporation efficiency ``beta`` in ``[beta_min, 1]``.

        ``beta_soil = beta_min + (1-beta_min)*clip(W/W_max, 0, 1)`` — the SAME
        Manabe bucket availability ramp as ``legoesm.land.slab_land`` (no
        re-derived hydrology).  Returns ``None`` (the caller treats it as
        ``beta = 1``, a wet/swamp surface) when the bucket is inactive or
        ``w_land`` is ``None``, so the slab-land path is byte-identical to
        the pre-bucket behaviour.

        When ``land_stomatal_beta`` is set AND the stomatal forcing
        (``T_land``, ``sw_down_sfc``, ``q_air``, ``p_s``) is supplied, the
        soil availability is routed through the SHARED land Jarvis (1976)
        stomatal model: ``beta = compute_stomatal_beta(jarvis_gs(...),
        beta_soil)`` = ``min(beta_soil, clip(gs/gs_ref))`` — REUSING
        ``legoesm.land.stomata`` (no re-derived conductance numerics,
        the same fallback path ``compute_effective_beta`` takes when the
        carbon state is unavailable).  ``gs`` closes the canopy term in low
        light (so ``beta -> 0`` at night) and high VPD.  Without the forcing
        (or when disabled) the soil-only ramp is returned unchanged.
        """
        if not self.land_soil_bucket or w_land is None:
            return None
        w_frac = jnp.clip(w_land / self.land_bucket_w_max, 0.0, 1.0)
        beta_soil = self.land_beta_min + (1.0 - self.land_beta_min) * w_frac
        if (not self.land_stomatal_beta or self.stomata_config is None
                or T_land is None or sw_down_sfc is None
                or q_air is None or p_s is None):
            return beta_soil
        # Shared Jarvis stomatal limitation (carbon state unavailable on the
        # AMIP slab path -> LAI=None -> min(beta_soil, beta_canopy)).
        from legoesm.land.stomata import (
            compute_stomatal_beta,
            jarvis_gs,
        )
        gs = jarvis_gs(T_land, sw_down_sfc, q_air, p_s, beta_soil,
                       self.stomata_config)
        return compute_stomatal_beta(gs, None, beta_soil, self.stomata_config)

    def _bucket_update(self, w_land, precip_total, beta_land,
                       T_land, T_low, q_air, u_low, v_low, p_s, dt):
        """Advance the Manabe (1969) soil-water bucket one physics step, adding the
        Green-Ampt-style infiltration-excess + saturation-excess runoff partition.

            infiltration = min(P, K_s*rho*(1 + B*(1 - W/W_max)))   (Hortonian cap)
            dW/dt = infiltration - E,   W in [0, W_max]
            runoff = (P - infiltration)  +  max(W - W_max, 0)/dt   (Hortonian + Dunne)

        ``P`` (``precip_total``) is total surface precip over land [kg/m^2/s]; ``E``
        is the ``beta``-limited land evaporation — the SAME ``beta``-scaled bulk
        latent flux /L_v that cools ``T_land`` in :meth:`_step_slab_land` and
        moistens the BL upstream.  The bucket consumes that SAME ``E``
        (``limit_evaporation=False``) so it stays consistent with the already-
        applied energy/moisture flux rather than re-limiting in isolation (which
        would desync the bucket water from the BL — only the standalone
        ``slab_land.step_land`` tile, which recomputes its own latent flux from the
        water-limited evaporation, may water-limit).  This adds the runoff
        diagnostic (previously the overflow was silently clipped away and lost); the
        bucket can still clip at 0 under the beta-floor over-evaporation exactly as
        the prior code did.  Partition shared with ``legoesm.land.slab_land`` via
        ``partition_bucket_runoff`` (no re-derived bucket numerics).  Returns
        ``(w_land, None)`` unchanged when the bucket is inactive.
        """
        if not self.land_soil_bucket or w_land is None:
            return w_land, None
        beta = 1.0 if beta_land is None else beta_land
        rho_low, wind_speed = self._land_surface_bulk(T_low, u_low, v_low, p_s)
        q_sat_land = saturation_specific_humidity(T_land, p_s)
        # beta-limited land evaporation mass flux [kg/m^2/s] (negative = dew
        # onto soil, a source); the SAME flux applied to the SEB / BL upstream.
        evap = beta * rho_low * self.C_E * wind_speed * (q_sat_land - q_air)
        from legoesm.land.bucket_hydrology import partition_bucket_runoff
        w_new, _evap_act, runoff, _ri, _rs = partition_bucket_runoff(
            w_land, precip_total, evap, dt, self.land_bucket_w_max,
            self.land_K_infiltration, self.land_infil_suction_boost,
            infiltration_excess=self.land_infiltration_excess,
            limit_evaporation=False,
        )
        return w_new, runoff

    def _land_albedo_eff(self, lat, snow):
        """Snow-brightened land albedo via ``legoesm.surface_albedo.land_albedo``
        when snow-albedo feedback is active; else the static vegetation albedo.

        ``snow_age`` is fixed at 0 (fresh-snow albedo) — SWE-only snow.  The
        gate is a compile-time Python ``if`` (off ⇒ returns ``self.albedo_land``
        unchanged, byte-identical); the snow-cover blend inside ``land_albedo``
        is ``jnp.where`` (traced, per cell).
        """
        if not self.snow_albedo_feedback or snow is None or lat is None:
            return self.albedo_land
        from legoesm.surface_albedo import land_albedo
        base = (jnp.broadcast_to(self.albedo_land, snow.shape)
                if self.albedo_land is not None else None)
        return land_albedo(lat, snow, jnp.zeros_like(snow),
                           self.land_albedo_config, base_albedo=base)

    def _step_slab_land(self, T_land, sw_down_sfc, lw_down_sfc,
                        T, p_s, q_v, u, v, dt, beta_land=None,
                        albedo_land=None):
        """Advance the slab-land skin temperature by one radiation step.

        Semi-implicit surface energy balance::

            C_land dT/dt = SW_net + eps*LW_down - eps*sigma*T^4 - SH - LH

        linearized about the current ``T_land``.  Every flux term damps
        (``dF/dT < 0``), so the denominator ``C_land - dt*dF/dT`` is
        always larger than ``C_land`` and the update is unconditionally
        stable for any radiation cadence.

        ``beta_land`` is the soil-moisture evaporation efficiency
        (:meth:`_land_beta`): it scales the land latent heat flux (and its
        temperature derivative) so a dry bucket evaporates less and warms
        (the desert-heating effect).  ``None`` ⇒ ``beta = 1`` (the legacy
        saturated wet-surface flux, byte-identical to the pre-bucket path).
        """
        T_air = T[..., -1]
        q_air = q_v[..., -1]
        rho_low, wind_speed = self._land_surface_bulk(
            T_air, u[..., -1], v[..., -1], p_s,
        )
        sh_coef = rho_low * constants.c_pd * self.C_H * wind_speed
        lh_coef = rho_low * constants.L_v * self.C_E * wind_speed
        beta = 1.0 if beta_land is None else beta_land
        eps = self.emissivity_land
        sb = constants.sigma_sb

        # Snow-brightened albedo when supplied by the caller (snow-albedo
        # feedback); else the static vegetation albedo (byte-identical).
        _alb = self.albedo_land if albedo_land is None else albedo_land
        q_sat_land = saturation_specific_humidity(T_land, p_s)
        sw_net = sw_down_sfc * (1.0 - _alb)
        lw_net = eps * lw_down_sfc - eps * sb * T_land ** 4
        shflx = sh_coef * (T_land - T_air)
        lhflx = beta * lh_coef * (q_sat_land - q_air)
        flux = sw_net + lw_net - shflx - lhflx

        # Clausius-Clapeyron derivative of saturation specific humidity
        # (the latent term carries the same beta factor as ``lhflx``).
        dqsat_dT = q_sat_land * constants.L_v / (constants.R_v * T_land ** 2)
        dflux_dT = (-4.0 * eps * sb * T_land ** 3
                    - sh_coef - beta * lh_coef * dqsat_dT)

        dt_rad = dt * self.rad_update_steps
        return T_land + dt_rad * flux / (self.C_land - dt_rad * dflux_dT)

    def _step_multilayer_land_tile(self, land_ml, sw_down_col, lw_down_col,
                                   T, p_s, q_v, u, v, precip_col, dt,
                                   land_ml_params=None):
        """Advance the MULTILAYER (Richards) land tile one radiation step and return
        ``(land_ml_new, T_sfc_col, albedo_col)`` — all in flattened COLUMN space.

        Builds the ``AtmToSurface`` forcing from the lowest atmospheric level (T, q,
        wind, p) plus the surface down-welling SW/LW and the lagged precip, then steps
        ``step_multilayer_land`` with the pipeline's land config / per-column params.
        Pure + differentiable w.r.t. the land params (the whole point of the refactor).
        Deferred land imports avoid a core->land top-level cross-package cycle."""
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.land.multilayer_land import step_multilayer_land
        from legoesm.thermo import saturation_mixing_ratio
        ad = self.adapter
        f2 = lambda g: ad.flatten_2d(g)
        T_air = f2(T[..., -1]); q_air = f2(q_v[..., -1])
        u_low = f2(u[..., -1]); v_low = f2(v[..., -1]); p_s_col = f2(p_s)
        rho = p_s_col / (constants.R_d * T_air)
        precip = precip_col if precip_col is not None else jnp.zeros_like(p_s_col)
        ones = jnp.ones_like(p_s_col)
        forcing = AtmToSurface(
            sw_down=sw_down_col, lw_down=lw_down_col, precip_total=precip,
            precip_snow=jnp.where(T_air < constants.T_freeze, precip, 0.0),
            T_lowest=T_air, q_lowest=q_air, u_lowest=u_low, v_lowest=v_low,
            p_lowest=0.99 * p_s_col, p_surface=p_s_col, rho_lowest=rho,
            cos_zenith=0.5 * ones, co2_ppmv=412.0 * ones,
            has_radiation=ones, has_precipitation=ones)
        dt_rad = dt * self.rad_update_steps
        # carbon_state is PRESCRIBED (fixed LAI) when set — the returned, evolved
        # carbon pools are discarded so the prescribed leaf carbon is reused every
        # step (no carbon spin-up), activating the Farquhar Vc_max25/g1/LCMA path.
        # Transient land-use cover threads the per-segment params as a traced arg
        # (SegmentForcing doctrine); None -> the baked self.land_ml_params, so the
        # static path is byte-identical.
        _lmp = land_ml_params if land_ml_params is not None else self.land_ml_params
        land_new, resp, _ = step_multilayer_land(
            land_ml, forcing, self.land_ml_cfg, self.land_ml_u_min, dt_rad,
            lat=self.land_ml_lat, doy=self.land_ml_doy,
            land_params=_lmp, carbon_state=self.land_ml_carbon)
        return land_new, resp.T_sfc, resp.albedo

    def _land_qsfc_multilayer(self, land_ml, T_land, p_s, land_ml_params=None):
        """Effective land-tile surface humidity for the multilayer land tile,
        ``q_sfc = beta_soil * q_sat(T_land)`` (the alpha-method, matching
        ``simple_seb``'s ``q_sfc = beta_effective * q_sat_sfc``), in GRID format.

        ``beta_soil`` is the root-zone soil-moisture stress the land SEB itself
        applies (``legoesm.land.multilayer_land.land_tile_beta_soil``, resolved
        from the SAME config / land_params thresholds).  Threaded into
        :meth:`_tiled_surface_flux` so the atmospheric land latent flux is
        throttled by soil moisture instead of running at the ``beta = 1``
        saturated-surface potential rate — the fix for the multilayer land
        over-evaporation (land hfls ~775 W/m^2, Bowen ~0.04 -> physical), where
        the throttled land-model flux was discarded and the atmosphere
        re-derived a saturated land surface.  ``T_land`` is the (grid) skin
        temperature already coupled from the Richards column.
        """
        from legoesm.land.multilayer_land import land_tile_beta_soil
        ad = self.adapter
        _lmp = land_ml_params if land_ml_params is not None else self.land_ml_params
        beta_col = land_tile_beta_soil(
            land_ml.theta_soil, self.land_ml_cfg, _lmp)
        q_sat_land_col = ad.flatten_2d(
            saturation_specific_humidity(T_land, p_s))
        return ad.unflatten_2d(beta_col * q_sat_land_col)

    def _tiled_surface_flux(self, u_low, v_low, T_low, q_low, rho_low,
                           sst, sic, T_land, p_s, beta_land=None,
                           q_sfc_land_override=None):
        """Area-weighted (mosaic) surface turbulent flux over ocean/ice/land.

        Used when ``self.surface_tiled`` is True (the active land tile).  The
        ocean tile uses the experiment's ocean bulk scheme (e.g. COARE3 +
        convective gustiness, from ``turbulence_config.surface``); the land
        tile uses the FIXED-ROUGHNESS land Monin-Obukhov scheme (``"most"``,
        roughness ``surface_z0_land``, no Charnock/gustiness); sea ice uses
        constant neutral coefficients.  This keeps the ocean air-sea scheme
        off the land columns — the bm_v3 blowup was COARE3 run over land.

        Ocean/ice surface humidities are wet-surface (saturated).  The LAND
        tile uses the soil-moisture-limited effective humidity
        ``q_eff = q_air + beta*(q_sat(T_land) - q_air)`` so the land latent
        flux is ``beta`` times its potential (wet-surface) value — the same
        ``beta`` (:meth:`_land_beta`) that limits the SEB in
        :meth:`_step_slab_land`.  ``beta_land=None`` ⇒ ``beta = 1`` (the
        legacy saturated land surface, byte-identical to the pre-bucket
        path).  Returns ``(tau_x, tau_y, shflx, lhflx, ustar)`` in column
        format.
        """
        from legoesm.atmosphere.physics.turbulence.surface_layer import (
            SurfaceTileSpec,
            compute_tiled_surface_fluxes,
        )

        ad = self.adapter
        f_land = ad.flatten_2d(self.f_land)
        sic_col = ad.flatten_2d(sic)
        frac_land = f_land
        frac_ice = (1.0 - f_land) * sic_col
        frac_ocean = (1.0 - f_land) * (1.0 - sic_col)

        T_ocean = ad.flatten_2d(sst)
        T_ice_grid = jnp.broadcast_to(
            jnp.asarray(self.T_ice, dtype=sst.dtype), sst.shape,
        )

        # Land tile: soil-moisture-limited effective surface humidity.
        # Three cases, in precedence order:
        #  1. q_sfc_land_override (MULTILAYER land): the land model's own
        #     alpha-method humidity q_sfc = beta_soil*q_sat(T_land) computed by
        #     _land_qsfc_multilayer — used directly so the atmospheric land
        #     latent flux matches the throttled land SEB (fix for the beta=1
        #     over-evaporation, hfls ~775 -> physical).
        #  2. beta_land (SLAB bucket): q_eff = q_air + beta*(q_sat - q_air) makes
        #     the land latent flux beta times its wet-surface potential.
        #  3. neither (beta_land None): beta=1, the saturated land surface.
        q_sat_land_col = ad.flatten_2d(saturation_specific_humidity(T_land, p_s))
        if q_sfc_land_override is not None:
            q_sfc_land_col = ad.flatten_2d(q_sfc_land_override)
        elif beta_land is None:
            q_sfc_land_col = q_sat_land_col
        else:
            beta_col = ad.flatten_2d(beta_land)
            q_sfc_land_col = q_low + beta_col * (q_sat_land_col - q_low)

        tiles = SurfaceTileSpec(
            frac_ocean=frac_ocean,
            frac_ice=frac_ice,
            frac_land=frac_land,
            T_ocean=T_ocean,
            T_ice=ad.flatten_2d(T_ice_grid),
            T_land=ad.flatten_2d(T_land),
            q_sfc_ocean=ad.flatten_2d(saturation_specific_humidity(sst, p_s)),
            q_sfc_ice=ad.flatten_2d(
                saturation_specific_humidity(T_ice_grid, p_s)
            ),
            q_sfc_land=q_sfc_land_col,
        )

        ocean_cfg = self.turbulence_config.surface
        land_cfg = ocean_cfg._replace(
            bulk_scheme="most", z0=self.surface_z0_land, gustiness_w_zi=0.0,
            # AIR-SEA-only option (#762): the land tile keeps the default
            # thermodynamic convention even when the ocean tile runs aerobulk.
            thermo_convention="legoesm",
        )
        ice_cfg = ocean_cfg._replace(bulk_scheme="constant")

        return compute_tiled_surface_fluxes(
            u_low, v_low, T_low, q_low, rho_low,
            tiles, ocean_cfg, ice_cfg, land_cfg,
        )

    def physics_step_no_rad(self, T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic,
                            lat, dt, dT_dt_rad, sw_net_sfc, lw_net_sfc,
                            sw_up_toa, lw_up_toa, sw_down_toa,
                            sbm_tau_c=None, sbm_RH_ref=None,
                            C_H=None, C_E=None,
                            q_i=None, q_s=None, q_g=None,
                            N_c=None, N_r=None, N_i=None,
                            T_land=None, aerosol_od=None,
                            sfc_shflx_override=None, sfc_lhflx_override=None,
                            tke=None, qke=None, gwd_spectrum=None,
                            w_land=None, snow=None, land_ml=None,
                            land_ml_params=None):
        """Convection + microphysics + BL exchange with held radiation.

        ``T_land`` is the slab-land skin temperature.  When the land tile
        is active (``self.f_land is not None``) the surface temperature
        used for bulk turbulent fluxes is the land/ocean blend, so land
        columns exchange heat and moisture against the land surface.

        ``aerosol_od`` is the per-layer aerosol optical depth in COLUMN
        format (ncol, nlev) from the external forcing pipeline.  Consumed
        only when the microphysics config sets ``nc_from_aerosol``: the
        column AOD is inverted to a specified droplet number
        (Andreae 2009, ``aerosol_activation.ccn_from_aod``) that fills
        ``hydrometeors.N_c`` for specified-Nc double-moment schemes.

        ``tke`` / ``qke`` / ``gwd_spectrum`` are the stateful-physics
        carries (issue #413), flattened-column layout like
        ``conv_prog``: prognostic turbulent energy ``(ncol, nlev)`` for
        the TKE-family / MYNN-2.5 schemes and the wave-action spectrum
        ``(ncol, n_azimuths, n_wavenumbers)`` for the prognostic
        spectral GWD.  The updated values ride the returned
        ``PhysicsOutput`` (like ``conv_prog``); inactive slots pass
        through unchanged.  A ``None`` or wrong-shape carry for an
        ACTIVE scheme raises at trace time — a silent reseed here is
        exactly the issue-#405 bug class.  Seed with
        ``init_physics_state`` and feed the updated value back.
        """
        _C_H = self.C_H if C_H is None else C_H
        _C_E = self.C_E if C_E is None else C_E

        ad = self.adapter
        nlev = self.sigma_full.shape[0]
        shape_3d = T.shape
        shape_2d = p_s.shape

        T_sfc = blend_surface_temperature(sst, sic, self.T_ice)
        if self.f_land is not None and T_land is not None:
            T_sfc = self._blend_land(T_sfc, T_land)

        # Soil-moisture evaporation efficiency for the land tile + bucket
        # (None ⇒ beta=1, wet surface; byte-identical to the pre-bucket path).
        # Recover the surface downwelling SW from the held net SW for the
        # Jarvis PAR term (sw_net = sw_down*(1-albedo); over a land tile the
        # blend albedo ≈ albedo_land, so this is accurate where stomata
        # apply).  Only computed when a land tile is present — ``albedo_land``
        # is ``None`` on ocean-only runs (``1.0 - None`` is an eager Python
        # subtraction, NOT dead-code-eliminated), so guard it; beta stays None
        # (=1, wet surface) with no land albedo.
        if self.albedo_land is not None:
            # Snow brightens the land albedo when the feedback is active (else
            # the static vegetation albedo, byte-identical).
            _alb_par = (self._land_albedo_eff(lat, snow)
                        if (self.snow_albedo_feedback and snow is not None)
                        else self.albedo_land)
            _sw_down_sfc = sw_net_sfc / jnp.maximum(1.0 - _alb_par, 1e-3)
            beta_land = self._land_beta(
                w_land, T_land=T_land, sw_down_sfc=_sw_down_sfc,
                q_air=q_v[..., -1], p_s=p_s,
            )
        else:
            beta_land = None

        p_full = p_s[..., None] * self.sigma_full
        p_half = p_s[..., None] * self.sigma_half

        # Flatten to columns via adapter
        T_col = ad.flatten_3d(T)
        p_full_col = ad.flatten_3d(p_full)
        p_half_col = p_half.reshape(ad.ncol, nlev + 1)
        q_v_col = ad.flatten_3d(q_v)
        p_s_col = ad.flatten_2d(p_s)
        lat_col = ad.flatten_2d(lat)

        z_full_col = None
        z_half_col = None
        if (self.physics_parameterization is not None
                or self.turbulence_fn is not None
                or self.gwd_fn is not None):
            from legoesm.atmosphere.physics._shared import compute_heights_from_sigma

            z_full_col, z_half_col = compute_heights_from_sigma(T_col, p_half_col)

        _conv_cfg = self.convection_config

        if sbm_tau_c is not None and _conv_cfg is not None and hasattr(_conv_cfg, 'tau_c'):
            _conv_cfg = _conv_cfg._replace(tau_c=sbm_tau_c)
        if sbm_RH_ref is not None and _conv_cfg is not None and hasattr(_conv_cfg, 'rh_ref'):
            _conv_cfg = _conv_cfg._replace(rh_ref=sbm_RH_ref)

        # Static per-scheme plumbing traits (shared with the bridge
        # factories so the two call paths cannot drift).
        from legoesm.atmosphere.physics.convection.integration import (
            convection_scheme_traits,
            diagnose_w_grid_columns_hydrostatic,
        )
        from legoesm.atmosphere.physics._shared import (
            compute_moisture_convergence,
            moisture_convergence_supported,
        )

        _ctr = convection_scheme_traits(self._conv_scheme)

        # --- Convective prognostic carry --------------------------------
        # Scalar-prognostic schemes (mass_flux M_c / edmf a_u) carry
        # (ncol,); profile-prognostic schemes carry the full (ncol, nlev)
        # conv_prog_profile.  A None or wrong-shape seed (warm start /
        # scheme switch / legacy (ncol,) zeros) re-initializes to the
        # scheme default, mirroring the bridge.  Inside lax.scan callers
        # must seed the correct shape up front — a mismatch there
        # surfaces as a loud carry-structure trace error, never silent.
        _prog_shape = (
            (ad.ncol, nlev) if _ctr.is_profile_prognostic else (ad.ncol,)
        )
        if conv_prog is None or tuple(conv_prog.shape) != _prog_shape:
            if _conv_cfg is not None and hasattr(_conv_cfg, 'M_c_init'):
                conv_prog = jnp.full(_prog_shape, _conv_cfg.M_c_init, dtype=T_col.dtype)
            elif _conv_cfg is not None and hasattr(_conv_cfg, 'a_u_init'):
                conv_prog = jnp.full(_prog_shape, _conv_cfg.a_u_init, dtype=T_col.dtype)
            else:
                conv_prog = jnp.zeros(_prog_shape, dtype=T_col.dtype)
        conv_prog_out = conv_prog

        # Stateful turbulence / GWD carries (issue #413): pass through
        # unchanged unless the active scheme advances them below.
        tke_out = tke
        qke_out = qke
        gwd_spectrum_out = gwd_spectrum

        # --- Grid-operator-backed convection inputs ----------------------
        # Winds for CMT (ZM/Tiedtke/Bechtold), resolved w for the KF
        # trigger / Kuo's w_lcl gate, and large-scale moisture
        # convergence for Tiedtke/Bechtold/Kuo.  ``u``/``v`` must live on
        # the T grid (cell centres); staggered layouts (e.g. MPAS edge
        # winds) degrade exactly like the bridge: zero CMT, zero/None w,
        # None MC (consumers then engage their built-in proxies; Kuo is
        # correctly quiescent).
        _winds_on_t_grid = (
            u is not None and v is not None
            and u.shape == shape_3d and v.shape == shape_3d
        )
        u_conv_col = v_conv_col = None
        if _ctr.is_cmt_capable:
            if _winds_on_t_grid:
                u_conv_col = ad.flatten_3d(u)
                v_conv_col = ad.flatten_3d(v)
            else:
                u_conv_col = jnp.zeros((ad.ncol, nlev), dtype=T_col.dtype)
                v_conv_col = jnp.zeros((ad.ncol, nlev), dtype=T_col.dtype)

        w_grid_col = None
        if _ctr.is_w_grid_consumer or _ctr.is_simple_mc_consumer:
            if (self._grid is not None and self._sigma_coord is not None
                    and _winds_on_t_grid):
                w_grid_col = diagnose_w_grid_columns_hydrostatic(
                    u, v, p_s, self._grid, self._sigma_coord,
                    T_col, p_full_col, q_v_col,
                    need_concrete=_ctr.is_w_grid_consumer,
                    dtype=T_col.dtype,
                )
            elif _ctr.is_w_grid_consumer:
                # KF reads w_grid unconditionally — concrete zeros.
                w_grid_col = jnp.zeros((ad.ncol, nlev), dtype=T_col.dtype)

        mc_col = None
        if _ctr.is_mc_consumer or _ctr.is_simple_mc_consumer:
            if (self._grid is not None and _winds_on_t_grid
                    and moisture_convergence_supported(self._grid)):
                mc_col = compute_moisture_convergence(q_v, u, v, self._grid)

        turb_out = None
        micro_out_ml = None
        predicted_micro = {}
        if self.physics_parameterization is not None:
            if _conv_cfg is None or not hasattr(_conv_cfg, 'M_c_init'):
                raise ValueError(
                    "physics_parameterization requires physical mass_flux convection",
                )
            if self.turbulence_config is None:
                raise ValueError(
                    "physics_parameterization requires physical louis turbulence",
                )
            from legoesm.atmosphere.physics.convection.mass_flux import (
                diagnose_mass_flux_closure,
            )
            from legoesm.atmosphere.physics.ml_parameterization import (
                apply_physics_parameterization,
            )

            T_sfc_col = ad.flatten_2d(T_sfc)
            q_sat_sfc_col = ad.flatten_2d(saturation_specific_humidity(T_sfc, p_s))
            rho_col_phys = p_full_col / (constants.R_d * T_col)
            closure = diagnose_mass_flux_closure(
                T=T_col,
                q_v=q_v_col,
                p_full=p_full_col,
                p_half=p_half_col,
                M_c=conv_prog,
                dt=dt,
                config=_conv_cfg,
            )
            conv_out, conv_prog_out, turb_out, micro_out_ml, predicted_micro = apply_physics_parameterization(
                self.physics_parameterization,
                closure=closure,
                T=T_col,
                u=ad.flatten_3d(u),
                v=ad.flatten_3d(v),
                q_v=q_v_col,
                q_c=ad.flatten_3d(q_c) if q_c is not None else jnp.zeros_like(T_col),
                q_r=ad.flatten_3d(q_r) if q_r is not None else jnp.zeros_like(T_col),
                p_full=p_full_col,
                p_half=p_half_col,
                p_s=p_s_col,
                z_full=z_full_col,
                z_half=z_half_col,
                T_sfc=T_sfc_col,
                q_sfc=q_sat_sfc_col,
                lat=lat_col,
                rho=rho_col_phys,
                M_c=conv_prog,
                dt=dt,
                mass_flux_config=_conv_cfg,
                louis_config=self.turbulence_config,
            )
        # Convection (resolved kernel; the branches below are STATIC
        # Python on the build-time scheme traits — no traced dispatch).
        elif _ctr.is_profile_prognostic:
            if _ctr.is_stochastic:
                # Bechtold, deterministic only (enable_stochastic=True is
                # rejected at build in _resolve_convection).  With
                # prng_key=None the AR1 state passes through untouched and
                # the stochastic multiplier is exactly 1, so a zero stoch
                # input is bit-identical and needs no carry slot.
                _stoch_zero = jnp.zeros((ad.ncol,), dtype=T_col.dtype)
                conv_out, conv_prog_out, _ = self.convection_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_conv_col, v=v_conv_col,
                    conv_prog_profile=conv_prog,
                    conv_stoch_state=_stoch_zero,
                    prng_key=None,
                    dt=dt, config=_conv_cfg,
                    moisture_convergence=mc_col,
                )
            elif _ctr.is_cmt_capable:
                if _ctr.is_mc_consumer:
                    # Tiedtke: CMT winds + moisture convergence.
                    conv_out, conv_prog_out = self.convection_fn(
                        T=T_col, q_v=q_v_col,
                        p_full=p_full_col, p_half=p_half_col,
                        u=u_conv_col, v=v_conv_col,
                        conv_prog_profile=conv_prog,
                        dt=dt, config=_conv_cfg,
                        moisture_convergence=mc_col,
                    )
                else:
                    # Zhang-McFarlane: CMT winds, no MC kwarg.
                    conv_out, conv_prog_out = self.convection_fn(
                        T=T_col, q_v=q_v_col,
                        p_full=p_full_col, p_half=p_half_col,
                        u=u_conv_col, v=v_conv_col,
                        conv_prog_profile=conv_prog,
                        dt=dt, config=_conv_cfg,
                    )
            elif _ctr.is_w_grid_consumer:
                # Kain-Fritsch: resolved-w trigger.
                conv_out, conv_prog_out = self.convection_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    w_grid=w_grid_col,
                    conv_prog_profile=conv_prog,
                    dt=dt, config=_conv_cfg,
                )
            else:
                # Emanuel: plain profile carry.
                conv_out, conv_prog_out = self.convection_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    conv_prog_profile=conv_prog,
                    dt=dt, config=_conv_cfg,
                )
        elif _conv_cfg is not None and hasattr(_conv_cfg, 'M_c_init'):
            conv_out, conv_prog_out = self.convection_fn(
                T=T_col, q_v=q_v_col, p_full=p_full_col, p_half=p_half_col,
                M_c=conv_prog, dt=dt, config=_conv_cfg,
            )
        elif _conv_cfg is not None and hasattr(_conv_cfg, 'a_u_init'):
            conv_out, conv_prog_out = self.convection_fn(
                T=T_col, q_v=q_v_col, p_full=p_full_col, p_half=p_half_col,
                a_u=conv_prog, dt=dt, config=_conv_cfg,
            )
        elif _ctr.is_simple_mc_consumer:
            # Kuo: stateless leaf driven by the large-scale moisture
            # convergence, plus resolved w for the oracle's ``w_lcl>0``
            # gate (None → convergence-sign proxy).  Previously the
            # pipeline called Kuo without MC, leaving it permanently
            # quiescent on resolved grids (latent bug, fixed 2026-06-10).
            conv_out = self.convection_fn(
                T=T_col, q_v=q_v_col, p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=_conv_cfg,
                moisture_convergence=mc_col,
                w_grid=w_grid_col,
            )
        else:
            conv_out = self.convection_fn(
                T=T_col, q_v=q_v_col, p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=_conv_cfg,
            )
        dT_dt_conv = ad.unflatten_3d(conv_out.dT_dt)
        dq_v_dt_conv = ad.unflatten_3d(conv_out.dq_v_dt)
        # Convection no longer surfaces its own precip diagnostic. Its
        # detrained condensate joins the cloud-water bucket and is
        # routed through microphysics for proper sedimentation /
        # melting / evaporation; surface precipitation is owned by
        # ``micro_out.precipitation`` (read into ``precip_micro`` below).
        dq_c_dt_conv = ad.unflatten_3d(conv_out.dq_c_conv_dt)
        precip = jnp.zeros(shape_2d, dtype=T.dtype)

        # Microphysics (resolved kernel — no dispatch here)
        _sd = T.dtype  # inherit storage dtype from state arrays
        dT_dt_micro = jnp.zeros(shape_3d, dtype=_sd)
        dq_v_dt_micro = jnp.zeros(shape_3d, dtype=_sd)
        dq_c_dt = jnp.zeros(shape_3d, dtype=_sd)
        dq_r_dt = jnp.zeros(shape_3d, dtype=_sd)
        dq_i_dt = jnp.zeros(shape_3d, dtype=_sd)
        dq_s_dt = jnp.zeros(shape_3d, dtype=_sd)
        dq_g_dt = jnp.zeros(shape_3d, dtype=_sd)
        dN_c_dt = jnp.zeros(shape_3d, dtype=_sd)
        dN_r_dt = jnp.zeros(shape_3d, dtype=_sd)
        dN_i_dt = jnp.zeros(shape_3d, dtype=_sd)
        precip_micro = jnp.zeros(shape_2d, dtype=_sd)
        # Isolated saturation-adjustment condensation (q_v->q_c) for the joint
        # vapour donor clamp below; None unless the micro scheme exposes it.
        _micro_dq_v_to_qc = None

        if micro_out_ml is not None:
            dT_dt_micro = ad.unflatten_3d(micro_out_ml.dT_dt)
            dq_v_dt_micro = ad.unflatten_3d(micro_out_ml.dq_v_dt)
            dq_c_dt = ad.unflatten_3d(micro_out_ml.dq_c_dt)
            dq_r_dt = ad.unflatten_3d(micro_out_ml.dq_r_dt)
            precip_micro = ad.unflatten_2d(micro_out_ml.precipitation)
            dq_i_dt = ad.unflatten_3d(micro_out_ml.dq_i_dt)
            dq_s_dt = ad.unflatten_3d(micro_out_ml.dq_s_dt)
            dq_g_dt = ad.unflatten_3d(micro_out_ml.dq_g_dt)
            dN_c_dt = ad.unflatten_3d(micro_out_ml.dN_c_dt)
            dN_r_dt = ad.unflatten_3d(micro_out_ml.dN_r_dt)
            dN_i_dt = ad.unflatten_3d(micro_out_ml.dN_i_dt)
        elif self.micro_fn is not None:
            from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
            q_c_col = ad.flatten_3d(q_c)
            q_r_col = ad.flatten_3d(q_r)
            rho_col = p_full_col / (constants.R_d * T_col)
            dp_col = p_half_col[:, 1:] - p_half_col[:, :-1]
            dz_col = dp_col / (rho_col * constants.g)
            _z = jnp.zeros_like(q_c_col)
            # Aerosol-CCN specified droplet number: under specified-Nc
            # (``predict_Nc=False`` — the N_c carry slot exists for
            # double-moment schemes but is dead zeros, never evolved),
            # fill the N_c input with the Andreae (2009) AOD->CCN
            # diagnostic so ``effective_Nc(..., nc_specified_field=True)``
            # sees the aerosol-driven per-column value instead of the
            # constant Nc_0 — the aerosol -> microphysics link.  The
            # override is unconditional on the carry VALUE: gating on
            # ``N_c is None`` would silently skip every Morrison run
            # (the full moisture registry always allocates the N_c
            # tracer; codex review 2026-06-10 hypothesis confirmed).
            # Prognostic-Nc runs (predict_Nc=True) keep their carry.
            _nc_aer_wanted = (
                getattr(self.micro_config, "nc_from_aerosol", False)
                and not getattr(self.micro_config, "predict_Nc", False)
            )
            if _nc_aer_wanted and aerosol_od is None:
                # Fail fast at trace time: a configured aerosol-CCN
                # coupling with no aerosol field would silently feed
                # zero N_c (→ Nc_0 fallback) into every column —
                # exactly the silent no-op class the codex review
                # flagged.  ``aerosol_od`` is a static-None only when
                # the forcing pipeline was never wired.
                raise ValueError(
                    "nc_from_aerosol=True but no aerosol_od was passed "
                    "to physics_step_no_rad — enable external aerosol "
                    "forcing (--aerosol-forcing external) or disable "
                    "--aerosol-ccn."
                )
            _nc_aer_specified = _nc_aer_wanted and aerosol_od is not None
            if _nc_aer_specified:
                from legoesm.atmosphere.physics.microphysics.aerosol_activation import (  # noqa: E501
                    ccn_from_aod,
                )
                # Column AOD = sum of the per-layer ODs the forcing
                # pipeline distributed from the Kinne climatology
                # (~550 nm).  The Andreae fit uses AOT500; the
                # 500-vs-550 nm difference (~5-10 % for Angstrom
                # exponents 0.7-1.7) is well inside the fit's factor-2
                # scatter, so no spectral correction is applied.
                _aod_col = jnp.sum(aerosol_od, axis=-1)        # (ncol,)
                _n_ccn = ccn_from_aod(_aod_col)                # (ncol,)
                _n_c_col = jnp.broadcast_to(
                    _n_ccn[:, None], q_c_col.shape,
                )
            else:
                _n_c_col = ad.flatten_3d(N_c) if N_c is not None else _z
            hydrometeors = HydrometeorState(
                q_c=q_c_col, q_r=q_r_col,
                q_i=ad.flatten_3d(q_i) if q_i is not None else _z,
                q_s=ad.flatten_3d(q_s) if q_s is not None else _z,
                q_g=ad.flatten_3d(q_g) if q_g is not None else _z,
                N_c=_n_c_col,
                N_r=ad.flatten_3d(N_r) if N_r is not None else _z,
                N_i=ad.flatten_3d(N_i) if N_i is not None else _z,
            )
            if (
                self.physics_parameterization is not None
                and getattr(self.physics_parameterization.model, "microphysics_scheme", "none")
                == "sundqvist"
                and "rain_survival_fraction" in predicted_micro
            ):
                from legoesm.atmosphere.physics.ml_parameterization import (
                    apply_predicted_sundqvist_rain_survival_fraction,
                )
                micro_out = apply_predicted_sundqvist_rain_survival_fraction(
                    predicted_micro["rain_survival_fraction"],
                    T=T_col,
                    q_v=q_v_col,
                    hydrometeors=hydrometeors,
                    p_full=p_full_col,
                    p_half=p_half_col,
                    rho=rho_col,
                    dz=dz_col,
                    dt=dt,
                    config=self.micro_config,
                )
            else:
                micro_out = self.micro_fn(
                    T=T_col, q_v=q_v_col, hydrometeors=hydrometeors,
                    p_full=p_full_col, p_half=p_half_col,
                    rho=rho_col, dz=dz_col, dt=dt,
                    config=self.micro_config,
                )
            dT_dt_micro = ad.unflatten_3d(micro_out.dT_dt)
            dq_v_dt_micro = ad.unflatten_3d(micro_out.dq_v_dt)
            dq_c_dt = ad.unflatten_3d(micro_out.dq_c_dt)
            dq_r_dt = ad.unflatten_3d(micro_out.dq_r_dt)
            precip_micro = ad.unflatten_2d(micro_out.precipitation)
            dq_i_dt = ad.unflatten_3d(micro_out.dq_i_dt)
            dq_s_dt = ad.unflatten_3d(micro_out.dq_s_dt)
            dq_g_dt = ad.unflatten_3d(micro_out.dq_g_dt)
            dN_c_dt = ad.unflatten_3d(micro_out.dN_c_dt)
            dN_r_dt = ad.unflatten_3d(micro_out.dN_r_dt)
            dN_i_dt = ad.unflatten_3d(micro_out.dN_i_dt)
            _c = micro_out.dq_v_to_qc_dt
            _micro_dq_v_to_qc = (
                ad.unflatten_3d(_c) if _c is not None else None)

        # NOTE: the JOINT vapour donor clamp (codex cycle-3) is applied later,
        # AFTER convection + turbulence + GWD are summed into the total
        # tendencies (see "JOINT vapour donor clamp" just before the
        # PhysicsOutput return).  It must see EVERY same-step vapour sink —
        # convection AND turbulence drying (codex#4 round-2 HIGH) — not just
        # convection, so it can only be applied on the assembled totals.

        # Convection→microphysics coupling: TRUE detrainment (plume / mass-flux
        # schemes, ``detrains_to_cloud``) adds convective condensate to the
        # cloud-water tendency; microphysics processes the augmented bucket on
        # the next step (operator splitting), giving proper autoconversion /
        # sedimentation / evaporation for convective rain.
        #
        # An ADJUSTMENT scheme (Betts-Miller sbm / dca / Kuo) instead produces a
        # column-net DRYING that is convective PRECIPITATION, not lingering
        # grid-scale cloud water.  Routing it into q_c let q_c accumulate ~100x
        # (in-cloud LWP -> tens of kg/m2, planetary albedo ~0.85, net TOA loss
        # ~-190 W/m2, runaway cold drift / OLR collapse) because Kessler
        # autoconversion cannot rain out a convective-precip-rate source.  So
        # precipitate the column-integrated convective condensate DIRECTLY: the
        # latent heat is already in ``dT_dt_conv`` (energy-neutral) and the
        # column water removed equals the added precip (mass-conserving).
        if _ctr.detrains_to_cloud:
            dq_c_dt = dq_c_dt + dq_c_dt_conv
            # #832: mass-flux schemes (bechtold/tiedtke) SPLIT the detrained
            # condensate by ``precip_efficiency``: ``dq_c_conv_dt`` = anvil cloud
            # (added to q_c above -> microphysics next step) and ``dq_r_conv_dt``
            # = in-updraft rain that falls out THIS step.  The pipeline consumed
            # ONLY ``dq_c_conv_dt``, so the rain fraction (default 70%) vanished
            # from the water budget while its condensation latent heat stayed in
            # ``dT_dt_conv`` -> near-zero convective surface precip and an
            # upper-tropospheric warm drift (detrainment-level +30 K).
            # Precipitate the in-updraft rain DIRECTLY (it is already a falling
            # species, not lingering grid cloud water).
            #   SIGN: ``dq_r_conv_dt >= 0`` is a rain SOURCE [kg/kg/s] (tiedtke/
            #     bechtold docstring), so the column-integrated rain formed
            #     ``sum(dq_r_conv_dt*dp/g) >= 0`` leaves as surface precip
            #     (positive-down); the ``maximum(.,0)`` is a defensive floor.
            #   ENERGY: neutral — the condensation latent heat of ALL condensate
            #     (cloud + rain) is already in ``dT_dt_conv`` (the scheme heated
            #     on condensation); precipitating the liquid adds no heat.
            #   MASS: NO ADDITIONAL leak — ``dq_r_conv_dt`` is EXACTLY the fraction
            #     split off ``dq_c_conv_dt`` by ``precip_efficiency`` in the scheme
            #     (tiedtke.py), and we precipitate exactly that field, so the water
            #     the scheme diverted to rain now reaches the surface instead of
            #     vanishing.  (The absolute column budget is only as tight as
            #     Tiedtke's underlying mass-flux solve — the default "advective"
            #     path conserves to truncation order, not machine-exact — but that
            #     residual pre-dates and is independent of this routing fix.)
            # None for schemes/efficiencies that emit no separate rain species
            # (precip_efficiency=0) -> no-op, byte-identical.
            if conv_out.dq_r_conv_dt is not None:
                dq_r_dt_conv = ad.unflatten_3d(conv_out.dq_r_conv_dt)
                _dp_r = p_s[..., None] * (
                    self.sigma_half[1:] - self.sigma_half[:-1])
                precip_conv_rain = jnp.maximum(
                    jnp.sum(dq_r_dt_conv * _dp_r / constants.g, axis=-1),
                    0.0)  # (..., n, n) kg/m2/s in-updraft rain to the surface
                precip = precip + precip_conv_rain
        else:
            # Convective precip = the column-net VAPOUR sink of the convective
            # tendency (mass-EXACT for every adjustment scheme: water removed
            # from q_v == surface precip, independent of how a scheme defines
            # its dq_c_conv_dt — sbm/dca rescale it to this, but Kuo's
            # heating-derived condensate does not equal it exactly).
            _dp = p_s[..., None] * (self.sigma_half[1:] - self.sigma_half[:-1])
            precip_conv = jnp.maximum(
                -jnp.sum(dq_v_dt_conv * _dp / constants.g, axis=-1),
                0.0)  # (..., n, n) kg/m2/s
            precip = precip + precip_conv

        # Boundary layer surface exchange (grid-agnostic: uses [..., -1] indexing).
        #
        # Surface scalar exchange must be applied EXACTLY ONCE per step.
        # When a turbulence scheme (Smagorinsky/Louis/TKE/YSU/HB/CLUBB-lite/
        # EDMF) or the joint physics_parameterization is active, surface
        # heat/moisture flux is the bottom BC of the implicit vertical-
        # diffusion solve inside turbulence — applying the bulk BL kick
        # here on top of that double-counts the flux and roughly doubles
        # the effective surface drag/heating (audit 2026-05-12 HIGH #2).
        #
        # Strategy: always compute bulk shflx/lhflx for diagnostics so the
        # output struct has well-defined values, but only inject the
        # bottom-layer T/q kick when no turbulence scheme owns surface
        # exchange.  If turbulence is active, its TurbulenceOutput.shflx /
        # lhflx overrides the bulk values further below.
        rho_low = (p_s * self.sigma_full[-1]) / (constants.R_d * T[..., -1])
        wind_speed = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1.0)
        dp_low = p_s * (self.sigma_half[-1] - self.sigma_half[-2])

        shflx = rho_low * constants.c_pd * _C_H * wind_speed * (T_sfc - T[..., -1])
        q_sat_sfc = saturation_specific_humidity(T_sfc, p_s)
        lhflx = rho_low * constants.L_v * _C_E * wind_speed * (q_sat_sfc - q_v[..., -1])

        turb_owns_surface = (
            self.turbulence_fn is not None
            or self.physics_parameterization is not None
        )

        # --- SHARED air-sea surface fluxes (coupler-authoritative) -----------
        # When the coupled driver supplies the tile-blended surface SH/LH (its
        # bulk scheme, q_sfc = 0.98*q_sat mixing ratio, ocean-tile C_H/C_E),
        # the atmosphere DISCARDS its own bulk estimate and uses the coupler's
        # numbers so the heat + water leaving the atmosphere EQUALS what the
        # coupler feeds the ocean (the air-sea budget closes).  ``None`` vs
        # array is a STATIC structural choice (set once by the driver closure
        # for the whole run), so a Python ``if`` is correct here -- the JAX
        # feature-gating exception (NOT a data-dependent jnp.where, which would
        # trace both branches).  Sign convention: both override fields are
        # [W/m2, positive UP = surface->atmosphere], identical to the bulk
        # ``shflx``/``lhflx`` they replace, so the downstream bottom-level T/q
        # kick (positive shflx warms the surface air; positive lhflx moistens
        # it) and the returned PhysicsOutput diagnostics are sign-consistent
        # with the ocean side (which applies q_net = ... - shflx - lhflx, i.e.
        # the SAME positive-up fluxes as a heat SINK on the ocean).
        _flux_override = (
            sfc_shflx_override is not None and sfc_lhflx_override is not None
        )
        if _flux_override:
            if turb_owns_surface:
                # A turbulence / unified-physics scheme applies the surface
                # flux as the IMPLICIT bottom BC of its vertical-diffusion
                # solve; overlaying the coupler flux on top would double-count
                # (or silently disagree with) the surface exchange.  The
                # shared-flux air-sea coupling is only well-posed against the
                # explicit bulk-BL surface path -- fail LOUDLY rather than
                # corrupt the budget (CLAUDE.md: no silent degradation).
                raise ValueError(
                    "Coupler shared surface-flux override (couple_surface_"
                    "fluxes) is incompatible with a turbulence / unified-"
                    "physics scheme that owns surface exchange: the turbulence "
                    "scheme already applies the surface flux as its implicit "
                    "bottom boundary condition, so the override would double-"
                    "count it.  Use the bulk-BL surface path (no turbulence "
                    "scheme) when enabling shared air-sea fluxes, or extend the "
                    "turbulence surface BC to ingest the coupler flux first."
                )
            shflx = sfc_shflx_override
            lhflx = sfc_lhflx_override

        dT_dt = dT_dt_rad + dT_dt_conv + dT_dt_micro
        dq_v_dt = dq_v_dt_conv + dq_v_dt_micro

        # Apply the explicit bottom-level surface kick from the bulk path OR
        # the coupler override (``_flux_override`` implies ``not
        # turb_owns_surface`` here -- the turbulence case raised above).
        if not turb_owns_surface:
            evap_rate = lhflx / constants.L_v
            dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
            dq_BL = constants.g * evap_rate / dp_low
            dT_dt = dT_dt.at[..., -1].add(dT_BL)
            dq_v_dt = dq_v_dt.at[..., -1].add(dq_BL)

        # Momentum tendencies from turbulence and GWD
        du_dt = jnp.zeros(shape_3d, dtype=_sd)
        dv_dt = jnp.zeros(shape_3d, dtype=_sd)

        if (self.turbulence_fn is not None and turb_out is None) or self.gwd_fn is not None:
            u_col = ad.flatten_3d(u)
            v_col = ad.flatten_3d(v)
            rho_col_phys = p_full_col / (constants.R_d * T_col)
            if z_full_col is None or z_half_col is None:
                from legoesm.atmosphere.physics._shared import compute_heights_from_sigma
                z_full_col, z_half_col = compute_heights_from_sigma(T_col, p_half_col)

        if turb_out is not None:
            du_dt = du_dt + ad.unflatten_3d(turb_out.du_dt)
            dv_dt = dv_dt + ad.unflatten_3d(turb_out.dv_dt)
            dT_dt = dT_dt + ad.unflatten_3d(turb_out.dT_dt)
            dq_v_dt = dq_v_dt + ad.unflatten_3d(turb_out.dq_v_dt)
            # Surface flux diagnostic comes from the stability-dependent
            # surface layer inside the turbulence scheme rather than the
            # bulk-formula placeholder.
            if getattr(turb_out, 'shflx', None) is not None:
                shflx = ad.unflatten_2d(turb_out.shflx)
            if getattr(turb_out, 'lhflx', None) is not None:
                lhflx = ad.unflatten_2d(turb_out.lhflx)
        elif self.turbulence_fn is not None:
            T_sfc_col = ad.flatten_2d(T_sfc)
            q_sat_sfc_col = ad.flatten_2d(
                saturation_specific_humidity(T_sfc, p_s)
            )
            _turb_kwargs = dict(
                u=u_col, v=v_col, T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                z_full=z_full_col, z_half=z_half_col,
                T_sfc=T_sfc_col, q_sfc=q_sat_sfc_col,
                rho=rho_col_phys, dt=dt, config=self.turbulence_config,
            )
            # Tiled (mosaic) surface flux: compute the surface turbulent flux
            # PER TILE (ocean bulk scheme on ocean, land Monin-Obukhov on
            # land) and inject it as the BL bottom boundary condition, rather
            # than running one scheme on the blended surface temperature.
            # Restricted to the kernels that accept the injected ``surface_flux``
            # tuple (louis / clubb_lite / clubb) by ExperimentConfig.validate_strict.
            # ``beta_land`` (None unless the soil-water bucket is active)
            # soil-moisture-limits the land tile's latent flux.
            if (self.surface_tiled and self.f_land is not None
                    and T_land is not None):
                # MULTILAYER land: override the land-tile surface humidity with
                # the land model's soil-moisture-throttled q_sfc so the BL
                # latent flux is not the beta=1 saturated potential rate.
                _q_sfc_land_ml = (
                    self._land_qsfc_multilayer(land_ml, T_land, p_s,
                                               land_ml_params=land_ml_params)
                    if land_ml is not None else None
                )
                _turb_kwargs["surface_flux"] = self._tiled_surface_flux(
                    u_col[:, -1], v_col[:, -1], T_col[:, -1], q_v_col[:, -1],
                    rho_col_phys[:, -1], sst, sic, T_land, p_s,
                    beta_land=beta_land, q_sfc_land_override=_q_sfc_land_ml,
                )
            if self._turb_energy_field is not None:
                # Stateful scheme (issue #413): kernel takes the
                # prognostic energy under its trait-named keyword
                # ("tke" for the MY-2.5 family, "qke" for MYNN-2.5)
                # and returns (TurbulenceOutput, energy_new).  A None
                # or wrong-shape carry for the ACTIVE scheme means the
                # caller dropped it — fail loudly at trace time rather
                # than silently reseed every step (the #405 bug class;
                # codex review).  Seed via ``init_physics_state`` and
                # feed the ``PhysicsOutput`` value back each step.
                _energy_in = tke if self._turb_energy_field == "tke" else qke
                if (_energy_in is None
                        or tuple(_energy_in.shape) != (ad.ncol, nlev)):
                    raise ValueError(
                        f"turbulence scheme carries prognostic "
                        f"{self._turb_energy_field!r} but the caller "
                        "passed "
                        f"{None if _energy_in is None else tuple(_energy_in.shape)} "
                        f"(expected {(ad.ncol, nlev)}) — the carry would "
                        "silently reseed every step (issue #405/#413). "
                        "Seed it with init_physics_state and thread the "
                        "updated PhysicsOutput value back."
                    )
                _turb_kwargs[self._turb_energy_field] = _energy_in
                turb_out, _energy_new = self.turbulence_fn(**_turb_kwargs)
                if self._turb_energy_field == "tke":
                    tke_out = _energy_new
                else:
                    qke_out = _energy_new
            else:
                turb_out = self.turbulence_fn(**_turb_kwargs)
            du_dt = du_dt + ad.unflatten_3d(turb_out.du_dt)
            dv_dt = dv_dt + ad.unflatten_3d(turb_out.dv_dt)
            dT_dt = dT_dt + ad.unflatten_3d(turb_out.dT_dt)
            dq_v_dt = dq_v_dt + ad.unflatten_3d(turb_out.dq_v_dt)
            if getattr(turb_out, 'shflx', None) is not None:
                shflx = ad.unflatten_2d(turb_out.shflx)
            if getattr(turb_out, 'lhflx', None) is not None:
                lhflx = ad.unflatten_2d(turb_out.lhflx)

        if self.gwd_fn is not None:
            lat_col = ad.flatten_2d(lat)
            _gwd_kwargs = dict(
                u=u_col, v=v_col, T=T_col,
                p_full=p_full_col, p_half=p_half_col,
                z_full=z_full_col, z_half=z_half_col,
                rho=rho_col_phys, lat=lat_col,
                dt=dt, config=self.gwd_config,
            )
            if self._gwd_prognostic:
                # Prognostic spectral GWD (issue #413): the wave-action
                # spectrum is the carry; kernel returns
                # (GWDOutput, spectrum_new).  None / wrong shape for the
                # ACTIVE scheme = dropped carry — fail loudly rather
                # than silently reseed every step (the #405 bug class;
                # codex review).
                # For a '+'-composite (issue #834) the resolved gwd_config is
                # the full GravityWaveDragConfig; the spectrum params live on
                # its ``prognostic_spectral`` sub-config.  For pure
                # ``prognostic_spectral`` the resolved config IS that
                # sub-config (get_gwd_fn returns config.prognostic_spectral).
                _sc = getattr(self.gwd_config, "prognostic_spectral",
                              self.gwd_config)
                _spec_shape = (ad.ncol, _sc.n_azimuths, _sc.n_wavenumbers)
                _spec_in = gwd_spectrum
                if (_spec_in is None
                        or tuple(_spec_in.shape) != _spec_shape):
                    raise ValueError(
                        "prognostic_spectral GWD carries a wave-action "
                        "spectrum but the caller passed "
                        f"{None if _spec_in is None else tuple(_spec_in.shape)} "
                        f"(expected {_spec_shape}) — the carry would "
                        "silently reseed every step (issue #405/#413). "
                        "Seed it with init_physics_state and thread the "
                        "updated PhysicsOutput value back."
                    )
                gwd_out, gwd_spectrum_out = self.gwd_fn(
                    spectrum_in=_spec_in, **_gwd_kwargs,
                )
            elif self._gwd_composite:
                # Stateless '+'-composite (e.g. ``hines+mcfarlane``, issue #834):
                # the combined executor mirrors the prognostic signature and
                # returns ``(GWDOutput, spectrum_out)`` even with no stateful
                # part, so pass ``spectrum_in=None`` and discard the (None)
                # spectrum — there is no wave-action carry to thread.  A
                # composite with an orographic member (mcfarlane/lindzen) still
                # needs the per-column SSO h_topo_col (else that member falls
                # back to scalar config.h_topo — a 500 m mountain over ocean).
                if (self._gwd_orographic
                        and self.subgrid_topo_stddev is not None):
                    _gwd_kwargs["h_topo_col"] = ad.flatten_2d(
                        self.subgrid_topo_stddev
                    )
                gwd_out, _ = self.gwd_fn(spectrum_in=None, **_gwd_kwargs)
            else:
                if (self._gwd_orographic
                        and self.subgrid_topo_stddev is not None):
                    _gwd_kwargs["h_topo_col"] = ad.flatten_2d(
                        self.subgrid_topo_stddev
                    )
                gwd_out = self.gwd_fn(**_gwd_kwargs)
            du_dt = du_dt + ad.unflatten_3d(gwd_out.du_dt)
            dv_dt = dv_dt + ad.unflatten_3d(gwd_out.dv_dt)
            dT_dt = dT_dt + ad.unflatten_3d(gwd_out.dT_dt)

        # JOINT vapour donor clamp (codex cycle-3, applied on ASSEMBLED totals).
        # Kessler reports the saturation condensation it performed
        # (_micro_dq_v_to_qc), computed from the PRE-physics q_v.  But the same
        # summed step also removes vapour via convection AND turbulence drying
        # (TurbulenceOutput.dq_v_dt is signed and CAN dry a level).  If the
        # combined sink drives q_v below 0 the state update floors q_v to 0 but
        # KEEPS the q_c increment -> q_c created from vapour that was floored
        # away (the ~1 kg/kg impossible cloud water -> planetary-albedo runaway
        # / OLR collapse, the coupled cold drift).
        #
        # Applying it HERE (after convection+turbulence+GWD are summed) is what
        # lets it see every same-step vapour sink, closing the turbulence-drying
        # hole that an earlier convection-only placement left (codex#4 round-2
        # HIGH).  ``dq_v_dt`` already CONTAINS ``-_sink_cond``, so
        # ``dq_v_dt + _sink_cond`` is the vapour tendency from all OTHER
        # processes; the condensation may consume at most the vapour that
        # survives them.  Reverting un-suppliable condensation is a mass/energy-
        # exact triple for Kessler: vapour kept (dq_v_dt += cond_lost), cloud not
        # formed (dq_c_dt -= cond_lost), latent heat not released
        # (dT_dt -= L_v*cond_lost/c_pd).  Convective + turbulent tendencies are
        # left UNTOUCHED (codex#4 round-1 HIGH x2): scaling convective drying
        # creates water for detraining schemes and breaks SBM/Kuo column-MSE
        # closure.  No-op (scale=1) whenever vapour is sufficient.
        if _micro_dq_v_to_qc is not None:
            _sink_cond = jnp.maximum(_micro_dq_v_to_qc, 0.0)  # [kg/kg/s] >= 0
            # Vapour available to the saturation condensation after every OTHER
            # same-step vapour process (dq_v_dt holds -_sink_cond; add it back).
            _q_v_for_cond = jnp.clip(q_v + dt * (dq_v_dt + _sink_cond), 0.0)
            # AD-safe donor scale min(1, q/(sink·dt)): the floored divisor bounds
            # the VJP under fp32 exactly as _warm_rain.donor_clamp_scale does
            # (inlined to avoid a cross-package private-module import).
            _sink_dt = jnp.maximum(dt * _sink_cond, 1.0e-15)
            _scale = jnp.minimum(1.0, _q_v_for_cond / _sink_dt)
            _cond_lost = _sink_cond * (1.0 - _scale)  # vapour couldn't supply
            dq_v_dt = dq_v_dt + _cond_lost            # keep the vapour
            dq_c_dt = dq_c_dt - _cond_lost            # do not form the cloud
            dT_dt = dT_dt - constants.L_v * _cond_lost / constants.c_pd

        # Advance the prognostic soil-water bucket (no-op / w_land unchanged
        # when the bucket is inactive).  Total surface precip is the source;
        # the beta-limited land evaporation is the sink.  ``land_runoff`` is the
        # diagnosed Hortonian + saturation-excess runoff leaving the column.
        w_land_new, land_runoff = self._bucket_update(
            w_land, precip + precip_micro, beta_land,
            T_land, T[..., -1], q_v[..., -1],
            u[..., -1], v[..., -1], p_s, dt,
        )

        # Advance the prognostic snow water equivalent (snow-albedo feedback):
        # snowfall (precip when the lowest-level air is below freezing) minus
        # degree-day melt.  No-op / snow unchanged when the feedback is off.
        if self.snow_albedo_feedback and snow is not None:
            from legoesm.land.snow_budget import update_snow
            precip_snow_diag = jnp.where(
                T[..., -1] < constants.T_freeze, precip + precip_micro, 0.0,
            )
            snow_new, _, _ = update_snow(
                snow, jnp.zeros_like(snow), T_land, precip_snow_diag, dt,
                Q_net=None,
            )
        else:
            snow_new = snow

        return PhysicsOutput(
            dT_dt=dT_dt,
            dq_v_dt=dq_v_dt,
            dq_c_dt=dq_c_dt,
            dq_r_dt=dq_r_dt,
            precip=precip + precip_micro,
            sw_net_sfc=sw_net_sfc,
            lw_net_sfc=lw_net_sfc,
            sw_up_toa=sw_up_toa,
            lw_up_toa=lw_up_toa,
            sw_down_toa=sw_down_toa,
            du_dt=du_dt,
            dv_dt=dv_dt,
            dq_i_dt=dq_i_dt,
            dq_s_dt=dq_s_dt,
            dq_g_dt=dq_g_dt,
            dN_c_dt=dN_c_dt,
            dN_r_dt=dN_r_dt,
            dN_i_dt=dN_i_dt,
            conv_prog=conv_prog_out,
            shflx=shflx,
            lhflx=lhflx,
            # Carry dtype stability: pin each updated carry to its INPUT
            # dtype so the value fed back next step (and the lax.scan
            # carry) never changes dtype.  The prognostic-spectral GWD
            # carry in particular must stay at the seed's default float
            # dtype — its internal level scan promotes via the config-
            # derived wavelength grid, so an f32 spectrum breaks the
            # kernel under x64 (same dtype rule as the MPAS seed).
            tke=_pin_carry_dtype(tke_out, tke),
            qke=_pin_carry_dtype(qke_out, qke),
            gwd_spectrum=_pin_carry_dtype(gwd_spectrum_out, gwd_spectrum),
            w_land=_pin_carry_dtype(w_land_new, w_land),
            snow=_pin_carry_dtype(snow_new, snow),
            land_runoff=land_runoff,
        )

    def _toa_insolation(self, lat, lon, day_of_year, seconds_of_day, s_0):
        """Prescribed TOA incident shortwave [W/m^2] — the incoming solar the
        radiation solver is GIVEN, used for the ``rsdt`` diagnostic.

        ``rsdt`` previously read ``sw_flux_down`` at the top halo, whose value
        comes from the quadratic top-boundary extrapolation in
        ``rte/two_stream._replace_top_flux``.  For downwelling SW the true TOA
        value exceeds every interior level (the column only attenuates
        downward), so the extrapolation's range-limit (kept deliberately to
        bound the BUG-B drifted-state overshoot, ``sw_down`` 1121 W/m^2) caps
        the diagnostic ~15 % below ``S_0 cos(SZA)`` (≈330 vs ≈340 W/m^2,
        C48).  The physical TOA incident flux is not an extrapolation at all —
        it is the prescribed insolation boundary condition.  This returns that
        insolation with the EXACT convention the solver uses (the column
        ``insol`` in the radiation builders): instantaneous ``S_0 cos(SZA)``
        under a diurnal cycle, else the daily-mean insolation.  Computed on the
        native grid (``lat``/``lon``) so it is ``sw_down_toa`` directly, and
        consistent with ``rsut`` (same ``S_0``/zenith), keeping the TOA budget
        ``R = rsdt - rsut - rlut`` correct.  Heating rates are unaffected (the
        halo is stripped before use); the BUG-B clamp on the halo is untouched.
        """
        from legoesm.atmosphere.physics.radiation.solar import (
            cos_zenith_angle,
            daily_mean_insolation,
            earth_sun_distance_factor,
        )
        orbit = getattr(self, "orbit", None)
        eccf = (earth_sun_distance_factor(day_of_year, orbit)
                if orbit is not None else 1.0)
        if self.diurnal_cycle:
            hour = seconds_of_day / 3600.0
            cos_sza = cos_zenith_angle(lat, lon, day_of_year, hour, orbit=orbit)
            return s_0 * eccf * jnp.maximum(cos_sza, 0.0)
        return daily_mean_insolation(lat, day_of_year, s_0, orbit=orbit)

    def compute_radiation_core(self, T, p_s, q_v, sst, sic, lat, lon,
                               day_of_year, seconds_of_day,
                               solar_weights, s_0,
                               o3_vmr_precomputed, aerosol_od_precomputed,
                               aerosol_lw_od_precomputed=None,
                               tau_equator=None, tau_pole=None,
                               albedo_ice=None, albedo_ocean=None,
                               ghg_vmr_override=None,
                               q_c=None, q_r=None,
                               q_i=None, N_c=None, N_i=None,
                               cloud_scheme="none",
                               u=None, v=None, dt=None, T_land=None,
                               sfc_albedo_override=None,
                               sfc_T_override=None,
                               sfc_emissivity_override=None,
                               conv_precip=None, land_ml=None, w_land=None,
                               snow=None, land_ml_params=None):
        """Compute radiation tendencies and fluxes (pure JAX, no I/O).

        Returns ``(dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa,
        lw_up_toa, sw_down_toa, T_land_new, land_ml_new)`` as an 8-tuple
        (``land_ml_new`` is the advanced multilayer land state, or the
        unchanged ``land_ml`` / ``None`` on the slab path).

        When the land tile is active (``self.f_land is not None``) the
        surface temperature/albedo/emissivity passed to the radiation
        solver are land/ocean blends, and the slab-land skin temperature
        ``T_land`` is advanced one radiation step by a semi-implicit
        surface energy balance.  Otherwise ``T_land`` is returned
        unchanged and the surface is pure ocean/ice.
        """
        _albedo_ice = self.albedo_ice if albedo_ice is None else albedo_ice
        _albedo_ocean = self.albedo_ocean if albedo_ocean is None else albedo_ocean

        ad = self.adapter
        nlev = self.sigma_full.shape[0]

        T_sfc = blend_surface_temperature(sst, sic, self.T_ice)
        if self.dynamic_albedo:
            # Zenith-dependent ocean albedo (Briegleb 1992).  Use the
            # SAME zenith convention as the radiation solver: the
            # instantaneous cos(SZA) under a diurnal cycle, else the
            # daytime-effective daily-mean cosine
            # mu = Q_day / (S_0 · f_day) (what RRTMGP sees as
            # cos_zenith on the non-diurnal path).  Ice/land albedo
            # blends below are unchanged.
            from legoesm.surface_albedo import (
                ocean_albedo, OceanAlbedoConfig,
            )
            from legoesm.atmosphere.physics.radiation.solar import (
                cos_zenith_angle, daily_mean_insolation, daylight_fraction,
                earth_sun_distance_factor,
            )
            _orbit = getattr(self, "orbit", None)
            if self.diurnal_cycle:
                _hour = seconds_of_day / 3600.0
                _mu = jnp.maximum(
                    cos_zenith_angle(lat, lon, day_of_year, _hour,
                                     orbit=_orbit), 0.0,
                )
            else:
                # mu is the optical-path cosine (geometry): use the orbital
                # declination but NOT the (a/r)^2 flux factor (divide it out).
                _eccf = (earth_sun_distance_factor(day_of_year, _orbit)
                         if _orbit is not None else 1.0)
                _q_day = daily_mean_insolation(lat, day_of_year, s_0,
                                               orbit=_orbit) / _eccf
                _f_day = daylight_fraction(lat, day_of_year, orbit=_orbit)
                _mu = jnp.clip(
                    _q_day / (s_0 * jnp.maximum(_f_day, 1.0e-6)), 0.0, 1.0,
                )
            _albedo_ocean_dyn = ocean_albedo(
                _mu, OceanAlbedoConfig(method="zenith"),
            )
            albedo = blend_surface_property(sic, _albedo_ice,
                                            _albedo_ocean_dyn)
        else:
            albedo = blend_surface_property(sic, _albedo_ice, _albedo_ocean)
        emissivity = blend_surface_property(sic, self.emissivity_ice, self.emissivity_ocean)

        # --- Land tile: blend land surface into T_sfc / albedo / emissivity.
        # MULTILAYER tile (land_ml present) supplies the surface T (top soil layer) +
        # per-column albedo/emissivity; else the scalar-T_land slab.
        _ml_active = self.land_ml_cfg is not None and land_ml is not None
        _land_active = self.f_land is not None and (T_land is not None or _ml_active)
        # Transient land-use cover: per-segment traced multilayer params (None ->
        # the baked self.land_ml_params, byte-identical static path).  Resolved
        # once here so it is in scope for BOTH the radiation-albedo blend and the
        # land skin-T update below whenever the multilayer tile is active.
        _lmp_rad = (land_ml_params if land_ml_params is not None
                    else self.land_ml_params) if _ml_active else None
        if _land_active:
            if _ml_active:
                T_land_grid = ad.unflatten_2d(land_ml.T_soil[:, 0])
                alb_land = ad.unflatten_2d(_lmp_rad.albedo_veg)
                emis_land = ad.unflatten_2d(_lmp_rad.emissivity)
            else:
                # Snow-brightened land albedo (snow-albedo feedback); the
                # static vegetation albedo when off (byte-identical).
                _alb_land_eff = (
                    self._land_albedo_eff(lat, snow)
                    if (self.snow_albedo_feedback and snow is not None)
                    else self.albedo_land)
                T_land_grid, alb_land, emis_land = (
                    T_land, _alb_land_eff, self.emissivity_land)
            T_sfc = self._blend_land(T_sfc, T_land_grid)
            albedo = self._blend_land(albedo, alb_land)
            emissivity = self._blend_land(emissivity, emis_land)

        # --- Coupler-provided dynamic surface overrides ---
        # In a coupled run the ocean/sea-ice/land tile models compute dynamic
        # surface albedo (temperature/zenith/snow-dependent) and skin
        # temperature and the coupler tile-blends them into a single field.
        # When threaded back as a per-segment traced forcing (NOT a closure
        # const → no recompile; mirrors the SST/SIC feedback), these REPLACE
        # the static internal blend above so the radiation actually sees the
        # ice-albedo feedback / zenith ocean albedo / snow brightening and the
        # ice/land prognostic skin temperature.  ``None`` (AMIP / standalone /
        # uncoupled) leaves the static blend untouched ⇒ byte-identical.
        if sfc_albedo_override is not None:
            albedo = sfc_albedo_override
        if sfc_T_override is not None:
            T_sfc = sfc_T_override
        # Dynamic surface emissivity (tile-blended, incl. the canopy's LAI-
        # dependent eps_eff) replaces the static blend so the LW boundary
        # ``eps·σ·T_sfc⁴ + (1−eps)·La`` uses the SAME emissivity the land tile
        # used to form its conservative ``LW_out`` / ``T_surface`` — closing the
        # land→atmosphere LW consistency gap.  ``None`` ⇒ static blend (AMIP /
        # uncoupled), byte-identical.
        if sfc_emissivity_override is not None:
            emissivity = sfc_emissivity_override

        p_full = p_s[..., None] * self.sigma_full
        p_half = p_s[..., None] * self.sigma_half

        # Flatten to columns via adapter.  ``q_v`` is kept in the
        # repo's mixing-ratio convention here; each ``radiation_fn``
        # wrapper is responsible for converting to the unit its solver
        # expects.  Gray radiation consumes mixing ratio directly (its
        # optical depth uses ``q_v · dp / g`` as column water).  The
        # RRTMGP wrapper converts to specific humidity inside the
        # builder before invoking the solver (audit 2026-05-12 #6 fix,
        # narrowed to RRTMGP per Codex review).
        T_col = ad.flatten_3d(T)
        p_full_col = ad.flatten_3d(p_full)
        p_half_col = p_half.reshape(ad.ncol, nlev + 1)
        q_v_col = ad.flatten_3d(q_v)
        T_sfc_col = ad.flatten_2d(T_sfc)
        lat_col = ad.flatten_2d(lat)
        lon_col = ad.flatten_2d(lon)
        albedo_col = ad.flatten_2d(albedo)
        emis_col = ad.flatten_2d(emissivity)

        # Compute cloud properties for cloud-radiation coupling
        cloud_kwargs = {}
        if cloud_scheme != "none":
            from legoesm.atmosphere.physics.clouds.cloud_fraction import (
                compute_cloud_properties,
            )
            dp_col = p_half_col[:, 1:] - p_half_col[:, :-1]
            # ``convective_cloud`` (opt-in) adds a bounded cumulus cloud cover
            # from the lagged convective precip so the convecting tropics get
            # radiative cloud the RH-based stratiform scheme misses.  Default
            # False => CloudConfig defaults (no convective term, no guard).
            # Activate the convective cloud term only where the convective
            # precip is actually plumbed (the compiled segment threads it via
            # the lagged carry).  Auxiliary callers that don't pass conv_precip
            # — the single warm-up step, any non-compiled per-step path —
            # degrade to no convective cloud rather than tripping the loud
            # compute_cloud_properties guard.  The guard still fires for a
            # direct convective_cloud=True + conv_precip=None misconfiguration.
            # Optional cloud-tuning overrides (None => CloudConfig default =>
            # byte-identical).  The SW/LW knob for e.g. the coare3 moisture-
            # driven albedo overshoot (raise rh_crit / lower q_c_diagnostic).
            # Built via the shared ``build_cloud_config`` so the clt diagnostic
            # (DiagnosticCollector) selects the SAME cloud fraction (#689).
            from legoesm.atmosphere.physics.clouds.config import (
                build_cloud_config,
            )
            cloud_config = build_cloud_config(
                cloud_scheme,
                convective_cloud=(getattr(self, "_cloud_convective", False)
                                  and conv_precip is not None),
                rh_crit=getattr(self, "_cloud_rh_crit", None),
                q_c_diagnostic=getattr(self, "_cloud_q_c_diagnostic", None),
                conv_cloud_max=getattr(self, "_cloud_conv_cloud_max", None),
                conv_cloud_condensate=getattr(
                    self, "_cloud_conv_cloud_condensate", None),
                cloud_inhomogeneity_factor=getattr(
                    self, "_cloud_inhomogeneity_factor", None),
                cloud_optics_inhomogeneity=getattr(
                    self, "_cloud_optics_inhomogeneity", None),
                cloud_fsd=getattr(self, "_cloud_fsd", None),
                p_xr=getattr(self, "_cloud_p_xr", None),
                alpha_xr=getattr(self, "_cloud_alpha_xr", None),
                diagnostic_condensate_scheme=getattr(
                    self, "_cloud_diagnostic_condensate_scheme", None),
                adiabatic_lwc_rate=getattr(
                    self, "_cloud_adiabatic_lwc_rate", None),
            )
            # Column convective precip [kg/m²/s] for the convective cloud cover;
            # flattened to the (ncol,) column layout like the other inputs.
            conv_precip_col = (
                None if conv_precip is None else ad.flatten_2d(conv_precip)
            )
            # When no microphysics is wired (``self.micro_fn is None``)
            # the prognostic ``q_c`` is a zero tracer and feeding it to
            # ``compute_cloud_properties`` short-circuits the diagnostic
            # condensate path: explicit-condensate overrides the
            # ``cf · q_c_diagnostic`` fallback, producing nonzero
            # cloud fraction but zero LWP/IWP — i.e. an optically inert
            # cloud (audit 2026-05-12 MEDIUM-HIGH #7).  Pass
            # ``q_cloud=None`` in that case so the diagnostic scheme
            # builds in-cloud condensate from the fraction and a
            # typical value.
            if self.micro_fn is None or q_c is None:
                q_c_col = None
            else:
                q_c_col = ad.flatten_3d(q_c)
            # Cloud ice + double-moment NUMBER columns (None for warm-rain /
            # diagnostic-cloud runs ⇒ constant r_eff, legacy behaviour). When a
            # double-moment scheme supplies them, they drive the M2005 PSD
            # liquid/ice effective radii — N_c per-VOLUME [#/m³], N_i per-MASS
            # [#/kg], passed raw (same convention as the dynamical-core paths).
            q_i_col = None if q_i is None else ad.flatten_3d(q_i)
            n_cloud_col = None if N_c is None else ad.flatten_3d(N_c)
            n_ice_col = None if N_i is None else ad.flatten_3d(N_i)
            # Aerosol-CCN droplet number for the radiation PSD: under
            # specified-Nc with aerosol coupling, feed the SAME
            # Andreae (2009) AOD->CCN diagnostic into the cloud-optics
            # effective radius so the Twomey (first indirect) effect is
            # consistent between the microphysics and the radiation.
            # Overrides the (dead-zeros) N_c carry — same rationale as
            # the microphysics fill in ``physics_step_no_rad``;
            # prognostic-Nc runs keep their carry.
            if (aerosol_od_precomputed is not None
                    and getattr(self.micro_config, "nc_from_aerosol",
                                False)
                    and not getattr(self.micro_config, "predict_Nc",
                                    False)):
                from legoesm.atmosphere.physics.microphysics.aerosol_activation import (  # noqa: E501
                    ccn_from_aod,
                )
                _aod_col = jnp.sum(aerosol_od_precomputed, axis=-1)
                n_cloud_col = jnp.broadcast_to(
                    ccn_from_aod(_aod_col)[:, None], T_col.shape,
                )
            # ``compute_cloud_properties`` is parameterised on mixing
            # ratio (RH from q_v vs q_sat_mixing_ratio); leave the
            # mixing-ratio q_v here and only feed the converted
            # specific humidity to the radiation solver.
            cloud_props = compute_cloud_properties(
                T=T_col, p_full=p_full_col, q_v=q_v_col, dp=dp_col,
                config=cloud_config, q_cloud=q_c_col, q_ice=q_i_col,
                n_cloud=n_cloud_col, n_ice=n_ice_col,
                conv_precip=conv_precip_col,
            )
            # ``to_rrtmg_kwargs`` builds the kwargs without
            # ``cloud_fraction`` (commit 4c9591bb, lost in AIMIP-#312
            # merge, restored iter-15) — see docstring for why.  iter-17
            # centralised the helper so the bug can't resurface at a
            # third call site.
            cloud_kwargs = cloud_props.to_rrtmg_kwargs()

        # Issue #273 follow-up: when a column mesh is configured, place
        # every column-format input on the mesh's 'col' axis before
        # invoking the JIT'd radiation kernel.  Sharding propagates
        # through the kernel automatically because the column ops are
        # purely functional; the kernel itself is unchanged.
        if self.column_mesh is not None:
            from legoesm.parallel.column_shard import shard_columns
            n_dev = self.column_mesh.shape["col"]
            if ad.ncol % n_dev != 0:
                raise ValueError(
                    f"column_mesh requires ncol={ad.ncol} divisible by "
                    f"n_devices={n_dev}.  Pick an n_devices that divides "
                    f"the flattened column count, or disable "
                    f"shard_radiation_columns."
                )
            _shard = lambda x: (
                None if x is None else shard_columns(x, self.column_mesh)
            )
            T_col = _shard(T_col)
            p_full_col = _shard(p_full_col)
            p_half_col = _shard(p_half_col)
            q_v_col = _shard(q_v_col)
            T_sfc_col = _shard(T_sfc_col)
            lat_col = _shard(lat_col)
            lon_col = _shard(lon_col)
            albedo_col = _shard(albedo_col)
            emis_col = _shard(emis_col)
            o3_vmr_precomputed = _shard(o3_vmr_precomputed)
            aerosol_od_precomputed = _shard(aerosol_od_precomputed)
            aerosol_lw_od_precomputed = _shard(aerosol_lw_od_precomputed)
            if cloud_kwargs:
                cloud_kwargs = {k: _shard(v) for k, v in cloud_kwargs.items()}

        rad_out = self.radiation_fn(
            T_col, p_full_col, p_half_col, q_v_col,
            T_sfc_col, lat_col, lon_col,
            day_of_year, seconds_of_day,
            albedo_col, emis_col,
            o3_vmr_precomputed, aerosol_od_precomputed,
            solar_weights, s_0,
            tau_equator=tau_equator, tau_pole=tau_pole,
            ghg_vmr_override=ghg_vmr_override,
            aerosol_lw_od_col=aerosol_lw_od_precomputed,
            **cloud_kwargs,
        )

        # Unflatten back to native grid shape via adapter
        dT_dt_rad = ad.unflatten_3d(rad_out.heating_rate)
        sw_down_sfc = ad.unflatten_2d(rad_out.sw_flux_down[:, -1])
        sw_net_sfc = sw_down_sfc * (1.0 - albedo)
        lw_net_sfc = ad.unflatten_2d(
            rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]
        )
        sw_up_toa = ad.unflatten_2d(rad_out.sw_flux_up[:, 0])
        lw_up_toa = ad.unflatten_2d(rad_out.lw_flux_up[:, 0])
        # rsdt = prescribed TOA incident SW the solver was given (#620), not the
        # quadratically clamped top-halo SW flux (rad_out.sw_flux_down[:, 0],
        # ~15% low).  Halo fallback keeps a value for any path (e.g. the
        # zero-radiation stub) that leaves toa_insolation=None.
        sw_down_toa = ad.unflatten_2d(
            rad_out.toa_insolation if rad_out.toa_insolation is not None
            else rad_out.sw_flux_down[:, 0]
        )

        # --- Land skin-temperature / soil-state update ---
        if _ml_active:
            # MULTILAYER: advance the Richards soil column from the surface SW/LW
            # (column space) + the lagged precip; the slab T_land rides through.
            precip_col = (ad.flatten_2d(conv_precip)
                          if conv_precip is not None else None)
            land_ml_new, T_sfc_ml_col, _ = self._step_multilayer_land_tile(
                land_ml, rad_out.sw_flux_down[:, -1], rad_out.lw_flux_down[:, -1],
                T, p_s, q_v, u, v, precip_col, dt, land_ml_params=_lmp_rad)
            # Couple the multilayer land SKIN TEMPERATURE back to T_land so the
            # atmospheric BL surface fluxes (tiled _tiled_surface_flux / the non-
            # tiled T_sfc blend) see the EVOLVING Richards soil column.  Previously
            # T_land rode through frozen -> the multilayer land drove ONLY radiation
            # (albedo/LW), never the turbulent-flux boundary, so use_multilayer_land
            # was a passive passenger in AMIP.  Land-fraction blending downstream
            # masks this to land cells (ocean uses SST), matching the slab path.
            # FOLLOW-UP (tracked): the tile also returns q_surface + per-cell z0
            # (resp.q_surface/resp.z0) which the atmospheric land flux does NOT yet
            # consume (it recomputes q_sfc from slab beta_land + uses the scalar
            # surface_z0_land).  So the multilayer HYDROLOGY/stomata drive T_sfc but
            # not yet the BL latent flux directly.  Full coupling = thread
            # q_sfc_land_col + z0_land_col into _tiled_surface_flux; deferred to a
            # validated follow-up (needs the CLM surfdata staged to run end-to-end).
            # Cast to the carried T_land dtype: the multilayer response T_sfc is at
            # working precision (float64 under x64) but T_land rides the SegmentCarry
            # at storage dtype (float32) — a lax.scan carry needs matching dtypes.
            T_land_new = ad.unflatten_2d(T_sfc_ml_col).astype(T_land.dtype)
        # --- Slab-land skin temperature update (semi-implicit SEB) ---
        # ``beta_land`` (None unless the soil-water bucket is active)
        # soil-moisture-limits the land latent flux, so a dry bucket warms
        # the land skin (desert-heating).  w_land itself is advanced in
        # physics_step_no_rad (where total precip is available).
        elif _land_active and self.slab_land_active:
            lw_down_sfc = ad.unflatten_2d(rad_out.lw_flux_down[:, -1])
            # Snow-brightened land albedo for the SEB net SW (feedback on);
            # static vegetation albedo when off (byte-identical).
            _alb_seb = (self._land_albedo_eff(lat, snow)
                        if (self.snow_albedo_feedback and snow is not None)
                        else None)
            T_land_new = self._step_slab_land(
                T_land, sw_down_sfc, lw_down_sfc, T, p_s, q_v, u, v, dt,
                beta_land=self._land_beta(
                    w_land, T_land=T_land, sw_down_sfc=sw_down_sfc,
                    q_air=q_v[..., -1], p_s=p_s,
                ),
                albedo_land=_alb_seb,
            )
            land_ml_new = land_ml
        else:
            T_land_new = T_land
            land_ml_new = land_ml

        return (dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa,
                sw_down_toa, T_land_new, land_ml_new)

    def build_step_unified(self, static_need_rad: bool | None = None,
                           rad_stop_gradient: bool = False,
                           jit: bool = True):
        """Build a JIT-compiled unified physics step with radiation sub-cycling.

        ``rad_stop_gradient`` (radiation-as-forcing): wrap the radiation core's
        outputs (heating + TOA/surface fluxes) in ``jax.lax.stop_gradient`` so
        radiation is applied FORWARD but carries no reverse-mode gradient.
        rrtmgp's adjoint is the dominant XLA compile cost of the differentiable
        rollout (it grows with grid size — minutes at T21, >10 h at T106), yet
        the only rrtmgp-tunable param is surface albedo (2 scalars; ``tau`` is
        gray-only).  Treating radiation as a slowly-varying forcing collapses
        that compile so high-res training becomes feasible; the state loss
        still trains convection/turbulence/surface, and TOA/surface fluxes
        follow once the state matches.  Albedo, if needed, is tuned via a cheap
        separate path (forward-mode / finite-diff on the 2 scalars), NOT this
        rollout adjoint.  Default False (full adjoint, unchanged behavior).

        SCOPE: this also makes the slab-land skin temperature ``T_land_new``
        forward-only — it is a ``compute_radiation_core`` output, so leaving it
        differentiable would drag the rrtmgp adjoint back in.  Intended (the
        radiation-driven land skin update is forcing too); moot for ocean-only
        AIMIP (``T_land`` inert).  A land run needing differentiable skin-T
        must use the full adjoint (False).

        ``jit`` (default True) wraps the step in ``jax.jit`` — the production path.
        Pass ``jit=False`` for differentiable parameter calibration that feeds a
        TRACED value into the pipeline via an attribute the step reads (e.g.
        ``land_ml_params`` in the coupled land calibrator): a jitted step would
        capture that tracer as a closure constant and leak it across
        ``value_and_grad`` calls (``UnexpectedTracerError``).  The un-jitted step
        is inlined into the caller's ``lax.scan`` trace, so there is no separate
        compiled artefact to capture the tracer.

        Returns a function ``step_unified(need_rad, T, p_s, q_v, q_c, q_r,
        conv_prog, u, v, sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
        solar_weights, s_0, o3_vmr, aerosol_od, held, ..., T_land, land_ml) ->
        (PhysicsOutput, held tuple, T_land_new, land_ml_new)``.

        ``T_land`` is the slab-land skin temperature carried through the
        radiation sub-cycle; it is advanced on radiation steps and held
        constant otherwise.  Pass ``None`` (the default) for ocean-only
        runs — the land tile is then inert.  ``land_ml`` is the prognostic
        multilayer (Richards) land state pytree carried alongside the slab
        ``T_land`` (``None`` for the slab/ocean-only path) and returned as
        the 4th value ``land_ml_new``.  Every consumer MUST unpack all four
        returns — see ``compiled_segments`` (length-aware) and the per-step
        loop in ``model_driver``.

        Parameters
        ----------
        static_need_rad : bool or None, optional
            Issue #316: ``jax.lax.cond`` inside a ``lax.scan`` body
            materialises both branches in the HLO graph; with a large
            radiation branch (RRTMGP: ~30 g-point band solves) the
            Conditional inflates the WhileLoop body, and XLA
            optimization passes (algebraic_simplifier, CSE) scale
            poorly — XLA JIT time grew from ~50 s at scan length 1 to
            > 2 h at scan length 4 320 in production AMIP runs.
            When the caller knows at build time whether radiation
            fires every step (``True``) or never (``False``) — the
            normal case under :func:`build_segment_fn` subcycling —
            the cond is elided here and only one branch is traced.
            ``None`` (default) preserves the original data-dependent
            cond for callers that still gate radiation inline.
        """
        pipeline = self

        def step_unified(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                         sst, sic, lat, lon,
                         day_of_year, seconds_of_day, dt,
                         solar_weights, s_0,
                         o3_vmr, aerosol_od,
                         held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                         held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                         tau_equator=None, tau_pole=None,
                         sbm_tau_c=None, sbm_RH_ref=None,
                         C_H=pipeline.C_H, C_E=pipeline.C_E,
                         albedo_ice=pipeline.albedo_ice,
                         albedo_ocean=pipeline.albedo_ocean,
                         ghg_vmr_override=None,
                         aerosol_lw_od=None,
                         T_land=None,
                         q_i=None, q_s=None, q_g=None,
                         N_c=None, N_r=None, N_i=None,
                         sfc_albedo_override=None,
                         sfc_T_override=None,
                         sfc_emissivity_override=None,
                         sfc_shflx_override=None,
                         sfc_lhflx_override=None,
                         tke=None, qke=None, gwd_spectrum=None,
                         conv_precip=None, land_ml=None, w_land=None,
                         snow=None, land_ml_params=None):

            def _rad_branch(args):
                (T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, lon,
                 day_of_year, seconds_of_day, dt,
                 solar_weights, s_0, o3_vmr, aerosol_od, aerosol_lw_od,
                 held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                 held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                 tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref,
                 C_H, C_E, albedo_ice, albedo_ocean,
                 ghg_vmr_override, T_land,
                 q_i, q_s, q_g, N_c, N_r, N_i,
                 sfc_albedo_override, sfc_T_override, sfc_emissivity_override,
                 sfc_shflx_override, sfc_lhflx_override,
                 tke, qke, gwd_spectrum,
                 conv_precip, land_ml, w_land, snow, land_ml_params) = args

                (dT_dt_rad, sw_net_sfc, lw_net_sfc,
                 sw_up_toa, lw_up_toa, sw_down_toa, T_land_new, land_ml_new) = \
                    pipeline.compute_radiation_core(
                        T, p_s, q_v, sst, sic, lat, lon,
                        day_of_year, seconds_of_day,
                        solar_weights, s_0, o3_vmr, aerosol_od,
                        aerosol_lw_od_precomputed=aerosol_lw_od,
                        tau_equator=tau_equator, tau_pole=tau_pole,
                        albedo_ice=albedo_ice, albedo_ocean=albedo_ocean,
                        ghg_vmr_override=ghg_vmr_override,
                        q_c=q_c, q_i=q_i, N_c=N_c, N_i=N_i,
                        cloud_scheme=pipeline._cloud_scheme,
                        u=u, v=v, dt=dt, T_land=T_land,
                        sfc_albedo_override=sfc_albedo_override,
                        sfc_T_override=sfc_T_override,
                        sfc_emissivity_override=sfc_emissivity_override,
                        conv_precip=conv_precip, land_ml=land_ml, w_land=w_land,
                        snow=snow, land_ml_params=land_ml_params,
                    )

                # Radiation-as-forcing: cut radiation's reverse-mode so the
                # expensive rrtmgp adjoint never enters the rollout backward
                # graph (the dominant, grid-size-scaling compile cost).
                # T_land_new is included (it is a radiation-core output;
                # leaving it differentiable re-introduces the rrtmgp adjoint)
                # -> the slab-land skin update is forward-only too. Moot for
                # ocean-only AIMIP; see build_step_unified docstring SCOPE.
                if rad_stop_gradient:
                    (dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa,
                     lw_up_toa, sw_down_toa, T_land_new) = jax.lax.stop_gradient(
                        (dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa,
                         lw_up_toa, sw_down_toa, T_land_new))

                physics_out = pipeline.physics_step_no_rad(
                    T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, dt,
                    dT_dt_rad, sw_net_sfc, lw_net_sfc,
                    sw_up_toa, lw_up_toa, sw_down_toa,
                    sbm_tau_c=sbm_tau_c, sbm_RH_ref=sbm_RH_ref,
                    C_H=C_H, C_E=C_E, T_land=T_land,
                    q_i=q_i, q_s=q_s, q_g=q_g, N_c=N_c, N_r=N_r, N_i=N_i,
                    aerosol_od=aerosol_od,
                    sfc_shflx_override=sfc_shflx_override,
                    sfc_lhflx_override=sfc_lhflx_override,
                    tke=tke, qke=qke, gwd_spectrum=gwd_spectrum,
                    w_land=w_land, snow=snow, land_ml=land_ml,
                    land_ml_params=land_ml_params,
                )

                # Cast to storage dtype so both lax.cond branches match.
                # The stateful-physics carries are EXEMPT: their dtype is
                # pinned to the carry-in dtype by physics_step_no_rad
                # (storage-downcasting the GWD spectrum here fed an f32
                # carry back into the f64-internal kernel scan next step).
                from legoesm.core.precision import get_policy
                _dt = get_policy().storage
                _cast = lambda x: x.astype(_dt) if hasattr(x, 'astype') else x
                new_held = tuple(_cast(h) for h in (
                    dT_dt_rad, sw_net_sfc, lw_net_sfc,
                    sw_up_toa, lw_up_toa, sw_down_toa,
                ))
                _carries = (physics_out.tke, physics_out.qke,
                            physics_out.gwd_spectrum, physics_out.w_land,
                            physics_out.snow)
                physics_out = jax.tree.map(_cast, physics_out)
                physics_out = physics_out._replace(
                    tke=_carries[0], qke=_carries[1],
                    gwd_spectrum=_carries[2], w_land=_carries[3],
                    snow=_carries[4],
                )
                return physics_out, new_held, _cast(T_land_new), land_ml_new

            def _no_rad_branch(args):
                (T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, lon,
                 day_of_year, seconds_of_day, dt,
                 solar_weights, s_0, o3_vmr, aerosol_od, aerosol_lw_od,
                 held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                 held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                 tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref,
                 C_H, C_E, albedo_ice, albedo_ocean,
                 ghg_vmr_override, T_land,
                 q_i, q_s, q_g, N_c, N_r, N_i,
                 sfc_albedo_override, sfc_T_override, sfc_emissivity_override,
                 sfc_shflx_override, sfc_lhflx_override,
                 tke, qke, gwd_spectrum,
                 conv_precip, land_ml, w_land, snow, land_ml_params) = args
                del conv_precip  # radiation-only input; unused on the no-rad path

                physics_out = pipeline.physics_step_no_rad(
                    T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, dt,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                    sbm_tau_c=sbm_tau_c, sbm_RH_ref=sbm_RH_ref,
                    C_H=C_H, C_E=C_E, T_land=T_land,
                    q_i=q_i, q_s=q_s, q_g=q_g, N_c=N_c, N_r=N_r, N_i=N_i,
                    aerosol_od=aerosol_od,
                    sfc_shflx_override=sfc_shflx_override,
                    sfc_lhflx_override=sfc_lhflx_override,
                    tke=tke, qke=qke, gwd_spectrum=gwd_spectrum,
                    w_land=w_land, snow=snow, land_ml=land_ml,
                    land_ml_params=land_ml_params,
                )

                # Cast to storage dtype — must match _rad_branch
                # (including the stateful-carry exemption).
                from legoesm.core.precision import get_policy
                _dt = get_policy().storage
                _cast = lambda x: x.astype(_dt) if hasattr(x, 'astype') else x
                new_held = tuple(_cast(h) for h in (
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                ))
                _carries = (physics_out.tke, physics_out.qke,
                            physics_out.gwd_spectrum, physics_out.w_land,
                            physics_out.snow)
                physics_out = jax.tree.map(_cast, physics_out)
                physics_out = physics_out._replace(
                    tke=_carries[0], qke=_carries[1],
                    gwd_spectrum=_carries[2], w_land=_carries[3],
                    snow=_carries[4],
                )
                # multilayer land state (if any) rides through the no-rad sub-steps
                # unchanged — it advances only on radiation steps (like the slab).
                return physics_out, new_held, _cast(T_land), land_ml

            args = (T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, lon,
                    day_of_year, seconds_of_day, dt,
                    solar_weights, s_0, o3_vmr, aerosol_od, aerosol_lw_od,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                    tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref,
                    C_H, C_E, albedo_ice, albedo_ocean,
                    ghg_vmr_override, T_land,
                    q_i, q_s, q_g, N_c, N_r, N_i,
                    sfc_albedo_override, sfc_T_override, sfc_emissivity_override,
                    sfc_shflx_override, sfc_lhflx_override,
                    tke, qke, gwd_spectrum,
                    conv_precip, land_ml, w_land, snow, land_ml_params)

            # Issue #316 fix: when the caller knows at build time which
            # branch to take, skip the cond — keeps only the live branch
            # in the HLO graph and bounds XLA compile time when this
            # function is called inside a long ``lax.scan``.
            if static_need_rad is True:
                del need_rad
                return _rad_branch(args)
            if static_need_rad is False:
                del need_rad
                return _no_rad_branch(args)
            return jax.lax.cond(need_rad, _rad_branch, _no_rad_branch, args)

        return jax.jit(step_unified) if jit else step_unified


# ---------------------------------------------------------------------------
# Radiation wrapper builders
# ---------------------------------------------------------------------------

def _build_none_radiation_fn(config):
    """Build a zero-tendency radiation_fn for ``radiation='none'``.

    Returns a :class:`RadiationOutput` with all-zero heating rates and SW/LW
    fluxes, so the pipeline EXPLICITLY disables radiation. Previously
    ``radiation='none'`` (a documented disable value) fell through the dispatch
    and silently built the full RRTMGP scheme. Matches the gray/rrtmgp
    radiation_fn call signature (all inputs ignored).
    """
    del config
    from legoesm.atmosphere.physics.radiation.output import RadiationOutput

    @jax.jit
    def radiation_fn(T_col, p_full_col, p_half_col, q_v_col, T_sfc_col,
                     lat_col, lon_col, day_of_year, seconds_of_day,
                     albedo_col, emis_col, o3_vmr_col, aerosol_od_col,
                     solar_weights, s_0=0.0,
                     tau_equator=None, tau_pole=None,
                     ghg_vmr_override=None,
                     aerosol_lw_od_col=None,
                     cloud_path_liq=None, cloud_path_ice=None,
                     cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                     cloud_fraction=None):
        del aerosol_lw_od_col  # zero-radiation: LW aerosol is a no-op
        ncol, nlev = T_col.shape
        z_full = jnp.zeros((ncol, nlev), dtype=T_col.dtype)
        z_half = jnp.zeros((ncol, nlev + 1), dtype=T_col.dtype)
        return RadiationOutput(
            lw_flux_up=z_half, lw_flux_down=z_half,
            sw_flux_up=z_half, sw_flux_down=z_half,
            heating_rate=z_full, lw_heating_rate=z_full,
            sw_heating_rate=z_full,
        )

    return radiation_fn


def _build_gray_radiation_fn(config):
    """Build a JIT-compiled gray radiation wrapper from config."""
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    from legoesm.atmosphere.physics.radiation.solar import (
        cos_zenith_angle, daily_mean_insolation,
        earth_orbit, earth_sun_distance_factor,
    )
    from legoesm.driver.kernel_registry import (
        RADIATION_REGISTRY, resolve_kernel,
    )

    gray_radiation = resolve_kernel(RADIATION_REGISTRY, "gray")
    diurnal = config.diurnal_cycle
    S_0 = config.S_0
    # Realistic (Berger 1978) orbit for AMIP-II/CMIP; None ⇒ circular orbit,
    # so idealized runs are bit-for-bit unchanged.
    orbit = earth_orbit() if getattr(config, "orbital_insolation", False) else None

    gray_config = GrayRadiationConfig(
        tau_equator=config.tau_equator,
        tau_pole=config.tau_pole,
        S_0=S_0,
        sfc_albedo=config.albedo_ocean,
        perpetual_equinox=False,
    )

    @jax.jit
    def radiation_fn(T_col, p_full_col, p_half_col, q_v_col, T_sfc_col,
                     lat_col, lon_col, day_of_year, seconds_of_day,
                     albedo_col, emis_col, o3_vmr_col, aerosol_od_col,
                     solar_weights, s_0=S_0,
                     tau_equator=None, tau_pole=None,
                     ghg_vmr_override=None,
                     aerosol_lw_od_col=None,
                     cloud_path_liq=None, cloud_path_ice=None,
                     cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                     cloud_fraction=None):
        del ghg_vmr_override  # gray radiation does not use GHG concentrations
        del aerosol_lw_od_col  # gray radiation does not use aerosol LW od
        del cloud_path_liq, cloud_path_ice, cloud_r_eff_liq, cloud_r_eff_ice, cloud_fraction
        # Rebuild config with traced tau values when provided
        _cfg = gray_config
        if tau_equator is not None:
            _cfg = _cfg._replace(tau_equator=tau_equator)
        if tau_pole is not None:
            _cfg = _cfg._replace(tau_pole=tau_pole)

        if diurnal:
            hour = seconds_of_day / 3600.0
            cos_sza = cos_zenith_angle(lat_col, lon_col, day_of_year, hour,
                                       orbit=orbit)
            # Eccentricity scales the incident flux by (a/r)^2 (1.0 when
            # circular); the cosine carries geometry only.
            eccf = (earth_sun_distance_factor(day_of_year, orbit)
                    if orbit is not None else 1.0)
            insol = s_0 * eccf * jnp.maximum(cos_sza, 0.0)
        else:
            insol = daily_mean_insolation(lat_col, day_of_year, s_0,
                                          orbit=orbit)
        # Thread the pipeline's blended (ice/ocean/land, plus coupler
        # overrides) surface albedo into the gray SW reflection so the
        # solver sees the same surface as the energy budget — previously
        # gray used only the static ``config.sfc_albedo`` and the
        # blended albedo was silently dropped (audit 2026-06-10).
        #
        # NOTE — ``emis_col`` (the blended / coupler-dynamic surface emissivity,
        # incl. the canopy eps_eff) is INTENTIONALLY NOT forwarded here.  Gray
        # radiation keeps its idealized black-surface convention
        # (``GrayRadiationConfig.sfc_emissivity = 1.0``, the Held-Suarez /
        # Frierson default).  Only RRTMGP honours the dynamic surface emissivity
        # (``solve_columns(sfc_emissivity=emis_col)``); threading it into gray
        # would shift every idealized gray run's surface LW by ~3-5 %.  This is a
        # deliberate scheme divergence from the albedo handling above, not the
        # same silently-dropped bug (user decision 2026-06-21).
        del emis_col
        return gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=_cfg,
            sfc_albedo=albedo_col,
        )

    return radiation_fn


def _build_rrtmgp_radiation_fn(config):
    """Build a JIT-compiled RRTMGP radiation wrapper from config."""
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    from legoesm.atmosphere.physics.radiation.solar import (
        cos_zenith_angle, daily_mean_insolation, daylight_fraction,
        earth_orbit, earth_sun_distance_factor,
    )
    from legoesm.atmosphere.physics.radiation.output import RadiationOutput

    diurnal = config.diurnal_cycle
    S_0 = config.S_0
    # Realistic (Berger 1978) orbit for AMIP-II/CMIP; None ⇒ circular orbit,
    # so idealized runs are bit-for-bit unchanged.
    orbit = earth_orbit() if getattr(config, "orbital_insolation", False) else None

    # Issue #273 GPU tuning: defer the scan-vs-unroll choice to
    # ``rte_utils.recurrent_op_with_halos`` when the experiment
    # config leaves ``rrtmgp_use_scan`` at its ``None`` default —
    # auto-picks ``True`` on GPU/TPU (one fused scan kernel) and
    # ``False`` on CPU/Metal (unrolled).  Explicit ``True``/``False``
    # in the experiment config still overrides for benchmarking and
    # AD workflows.
    _exp_use_scan = getattr(config, 'rrtmgp_use_scan', None)
    rrtmg_config = RRTMGPConfig(
        co2_ppmv=config.co2_ppmv,
        ch4_ppbv=config.ch4_ppbv,
        n2o_ppbv=config.n2o_ppbv,
        sfc_emissivity=config.sfc_emissivity,
        sfc_albedo=config.albedo_ocean,
        S_0=S_0,
        use_scan=_exp_use_scan,
        gpoint_batch_size=getattr(config, 'rrtmgp_gpoint_batch_size', 0),
        gpoint_checkpoint=getattr(config, 'rrtmgp_gpoint_checkpoint', True),
        column_chunk_size=getattr(config, 'rrtmgp_column_chunk_size', 0),
        include_clouds=(getattr(config, 'cloud_scheme', 'none') != 'none'),
    )

    solver = RRTMGP.from_legoesm_config(rrtmg_config)

    @jax.jit
    def radiation_fn(T_col, p_full_col, p_half_col, q_v_col, T_sfc_col,
                     lat_col, lon_col, day_of_year, seconds_of_day,
                     albedo_col, emis_col, o3_vmr_col, aerosol_od_col,
                     solar_weights, s_0=S_0,
                     tau_equator=None, tau_pole=None,
                     ghg_vmr_override=None,
                     aerosol_lw_od_col=None,
                     cloud_path_liq=None, cloud_path_ice=None,
                     cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                     cloud_fraction=None):
        del tau_equator, tau_pole  # RRTMGP does not use gray optical depth
        _sw_scale = None
        # Eccentricity scales the incident SW *flux* by (a/r)^2; the orbital
        # declination enters the geometry (cos_zenith).  1.0 ⇒ circular orbit.
        eccf = (earth_sun_distance_factor(day_of_year, orbit)
                if orbit is not None else 1.0)
        # Per-step incident irradiance relative to the solver's baked-in S_0.
        # The RRTMGP solver is built ONCE with the static config S_0, so the
        # actual irradiance reaching the SW fluxes must be applied as an output
        # scale: this captures BOTH time-varying TSI (solar_source=file passes
        # s_0 = TSI(t)) AND the (a/r)^2 distance factor.  Without it, the rsdt
        # diagnostic (built from s_0 below) would move while the solved SW
        # fluxes/heating silently stayed at the static S_0 — a hidden TOA
        # energy-budget inconsistency.  s_0 == S_0 and a circular orbit ⇒ 1.0
        # (bit-identical output to the legacy path).
        _irr_scale = (s_0 / S_0) * eccf
        if diurnal:
            hour = seconds_of_day / 3600.0
            cos_sza = cos_zenith_angle(lat_col, lon_col, day_of_year, hour,
                                       orbit=orbit)
            cos_zenith = jnp.maximum(cos_sza, 0.0)
            # Prescribed TOA incident SW = s_0·(a/r)^2·max(cosθ,0) for the rsdt
            # diagnostic (#620); matches _compute_insolation's diurnal return.
            insol = s_0 * eccf * cos_zenith
            # cos_zenith stays geometric; the irradiance (TSI x distance vs the
            # solver's S_0) scales the SW flux.
            _sw_scale = jnp.full((cos_zenith.shape[0],), _irr_scale,
                                 dtype=cos_zenith.dtype)
        else:
            # Daytime-effective cos(SZA): use daylight fraction so the solver
            # sees the correct optical path during sunlit hours.  SW fluxes
            # are then rescaled by f_day to recover daily-mean energy.
            # daily_mean_insolation already includes the (a/r)^2 factor when
            # ``orbit`` is set; divide it out for the geometric cosine.
            insol = daily_mean_insolation(lat_col, day_of_year, s_0,
                                          orbit=orbit)
            f_day = daylight_fraction(lat_col, day_of_year, orbit=orbit)
            f_day_safe = jnp.maximum(f_day, 1.0e-6)
            # insol ∝ s_0·eccf; dividing by (s_0·f_day) leaves a geometric
            # cos_zenith (both s_0 and eccf cancel) — the optical path stays
            # <= 1 and irradiance-independent.
            insol_geom = insol / eccf
            cos_zenith = jnp.clip(
                insol_geom / (s_0 * f_day_safe), 0.0, 1.0,
            )
            _sw_scale = f_day * _irr_scale

        # Water-vapor unit convention: the upstream pipeline passes
        # ``q_v`` as **mixing ratio** r = m_v / m_d.  RRTMGP's internal
        # H2O VMR formula ``mol_ratio * q / (1 - q)`` expects **specific
        # humidity** q = m_v / (m_v + m_d).  Convert at the solver
        # boundary so that gray radiation and other consumers of
        # ``q_v_col`` (e.g. cloud-fraction diagnostics) keep their
        # mixing-ratio inputs while RRTMGP sees the right unit.
        # Audit 2026-05-12 #6, narrowed to RRTMGP per Codex review.
        q_v_specific = q_v_col / (1.0 + jnp.clip(q_v_col, 0.0, None))
        _rad_kwargs = dict(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, q_v=q_v_specific,
            cos_zenith=cos_zenith,
            sfc_albedo=albedo_col,
            sfc_emissivity=emis_col,
            o3_vmr=o3_vmr_col,
            aerosol_optical_depth=aerosol_od_col,
            aerosol_absorption_optical_depth_lw=aerosol_lw_od_col,
            solar_spectral_fraction=solar_weights if solar_weights.size > 0 else None,
            ghg_vmr_override=ghg_vmr_override,
            cloud_path_liq=cloud_path_liq,
            cloud_path_ice=cloud_path_ice,
            cloud_r_eff_liq=cloud_r_eff_liq,
            cloud_r_eff_ice=cloud_r_eff_ice,
            cloud_fraction=cloud_fraction,
        )
        # Column-chunk the rrtmgp solve when configured: the per-block body
        # compiles ONCE at column_chunk_size, capping the super-linear rrtmgp
        # XLA compile time at higher horizontal resolution.  Columns are
        # independent → numerically exact.  ``column_chunk_size`` is a static
        # closure int, so this is a compile-time feature gate (plain ``if``).
        if rrtmg_config.column_chunk_size and rrtmg_config.column_chunk_size > 0:
            result = solver.solve_columns_chunked(
                column_chunk_size=rrtmg_config.column_chunk_size, **_rad_kwargs,
            )
        else:
            result = solver.solve_columns(**_rad_kwargs)

        # Rescale SW fluxes/heating to daily-mean when using daytime-effective SZA
        if _sw_scale is not None:
            s = _sw_scale[:, None]
            result = RadiationOutput(
                lw_flux_up=result.lw_flux_up,
                lw_flux_down=result.lw_flux_down,
                sw_flux_up=result.sw_flux_up * s,
                sw_flux_down=result.sw_flux_down * s,
                heating_rate=result.lw_heating_rate + result.sw_heating_rate * s,
                lw_heating_rate=result.lw_heating_rate,
                sw_heating_rate=result.sw_heating_rate * s,
            )

        # Carry the prescribed TOA insolation so the CMOR rsdt diagnostic
        # reads true TOA incident SW, not the clamped top-halo flux (#620).
        # AFTER the rescale rebuild (which drops the field) so it survives.
        result = result._replace(toa_insolation=insol)
        return result

    return radiation_fn


# Map radiation scheme names to builder functions.
_RADIATION_BUILDERS: dict[str, callable] = {
    "none": _build_none_radiation_fn,  # explicit zero-radiation (was silently rrtmgp)
    "gray": _build_gray_radiation_fn,
    "rrtmgp": _build_rrtmgp_radiation_fn,
    "rrtmg": _build_rrtmgp_radiation_fn,  # common alias
}


# ---------------------------------------------------------------------------
# Convection resolver
# ---------------------------------------------------------------------------

# Schemes accepted by ``ExperimentConfig.validate_strict`` that the
# unified pipeline can NOT build.  Empty since audit 2026-06-10 — every
# registered convection scheme is wired through
# ``PhysicsPipeline.physics_step_no_rad`` (the carry is a full
# ``(ncol, nlev)`` ``conv_prog_profile`` for the profile-prognostic
# schemes, plus CMT winds / w_grid / moisture-convergence plumbing
# mirroring the bridge factory).  ``tests/unit/test_advertised_buildability.py``
# keeps this shrink-only: adding an entry is a reviewed decision.
_PIPELINE_UNSUPPORTED_CONVECTION = frozenset()


def _resolve_convection(config):
    """Resolve convection kernel and config from ExperimentConfig.

    Returns (kernel_fn, kernel_config).

    All registered schemes are supported.  Scalar-prognostic schemes
    (mass_flux ``M_c``, edmf ``a_u``) thread a ``(ncol,)`` carry;
    profile-prognostic schemes (zhang_mcfarlane, kain_fritsch, emanuel,
    tiedtke, bechtold) thread the full ``(ncol, nlev)``
    ``conv_prog_profile`` — see the trait-driven dispatch in
    ``physics_step_no_rad``.

    One explicit exclusion: Bechtold with ``enable_stochastic=True``
    needs a per-segment PRNG-key carry that the unified driver does not
    thread (the AR1 state would also need a checkpoint slot).  The
    deterministic default (``enable_stochastic=False``) is bit-identical
    to the bridge path; stochastic runs use
    :func:`legoesm.atmosphere.physics.convection.integration.make_convection_physics`.
    """
    from legoesm.atmosphere.physics.convection.config import (
        ConvectionConfig,
        SBMConfig,
    )
    from legoesm.driver.kernel_registry import (
        CONVECTION_REGISTRY, resolve_kernel,
    )

    scheme = config.convection
    if scheme == "none":
        return _noop_convection, None

    conv_fn = resolve_kernel(CONVECTION_REGISTRY, scheme)

    # Build the per-scheme config
    if scheme == "sbm":
        conv_config = SBMConfig(
            tau_c=config.sbm_tau_c,
            rh_ref=config.sbm_RH_ref,
            cape_threshold=getattr(config, 'sbm_cape_threshold', 70.0),
        )
    elif scheme == "bechtold":
        # Expose the Bechtold CAPE trigger threshold so it is tunable for the
        # coarse-resolution convective-precip deficit (default matches
        # BechtoldConfig.cape_threshold ⇒ byte-identical when unset).  Also
        # thread the shared convective rain-split knob (#832 follow-up):
        # Bechtold otherwise detrains 100% of its condensate to cloud (no
        # dq_r_conv_dt), the over-bright-anvil / dry-column-runaway failure
        # mode; precip_efficiency > 0 drains it as rain like Tiedtke.
        from legoesm.atmosphere.physics.convection.config import BechtoldConfig
        # #929 None-sentinel precip_efficiency + campaign split/downdraft
        # threading, combined.  ``None`` keeps BechtoldConfig's 0.7 default (ON,
        # the anvil-drain fix); an EXPLICIT value overrides (0.0 = legacy).
        _pe = getattr(config, "convective_precip_efficiency", None)
        _bechtold_kwargs = dict(
            cape_threshold=getattr(config, 'bechtold_cape_threshold', 70.0),
            p_conv_top_pa=getattr(config, 'bechtold_conv_top_pa', 15000.0),
            # Bechtold takes this dedicated branch (never the shared _split
            # block below), so thread the precip-split selector + autoconv
            # params HERE or "--convective-precip-split autoconversion" silently
            # runs the constant split (codex HIGH).
            precip_split_scheme=getattr(config, 'convective_precip_split', 'constant'),
            autoconv_q_c_crit=getattr(config, 'autoconv_q_c_crit', 5.0e-4),
            autoconv_pe_max=getattr(config, 'autoconv_pe_max', 0.9),
            # Convective-downdraft strength (#847) + penetrative transport
            # (opt-in): re-evaporation moistens the sub-cloud layer; the
            # transport advects low-MSE dry air down (downdraft_alpha·M_b),
            # drying the BL.  Defaults reproduce BechtoldConfig (byte-identical).
            downdraft_evap_efficiency=getattr(config, 'bechtold_downdraft_evap', 0.05),
            downdraft_alpha=getattr(config, 'bechtold_downdraft_alpha', 0.3),
            downdraft_RH_min=getattr(config, 'bechtold_downdraft_rh_min', 0.2),
            downdraft_transport=getattr(config, 'bechtold_downdraft_transport', False),
            downdraft_entrain_rate=getattr(config, 'bechtold_downdraft_entrain_rate', 5.0e-4),
            downdraft_detrain_scale_m=getattr(config, 'bechtold_downdraft_detrain_scale_m', 700.0),
            # Full IFS deep CAPE closure (PR #1095) — threaded so the flag is
            # REACHABLE from the AMIP driver.  Missing-field fallback = True,
            # matching both config defaults (a config-like caller without the
            # field gets the scheme default, not the legacy closure).
            use_ifs_cape_closure=getattr(
                config, 'bechtold_use_ifs_cape_closure', True),
            # IFS Kessler sub-cloud rain evaporation (default False = legacy,
            # matching the scheme default).
            use_ifs_subcloud_evap=getattr(
                config, 'bechtold_use_ifs_subcloud_evap', False),
        )
        if _pe is not None:
            _bechtold_kwargs["precip_efficiency"] = _pe
        conv_config = BechtoldConfig(**_bechtold_kwargs)
    else:
        cc = ConvectionConfig(scheme=scheme)
        conv_config = getattr(cc, scheme)
        # #832/#929: thread the shared ExperimentConfig convective rain-split
        # knob into EVERY mass-flux scheme whose config exposes
        # ``precip_efficiency`` (tiedtke, zhang_mcfarlane, kain_fritsch,
        # mass_flux, edmf — the shared ``split_convective_rain``; Bechtold is
        # threaded in its own branch above).  ``None`` (the default sentinel)
        # keeps each scheme's OWN default (Tiedtke 0.0 = legacy no-split,
        # byte-identical); an EXPLICIT >0 value overrides it.
        _pe = getattr(config, "convective_precip_efficiency", None)
        if (_pe is not None and _pe > 0.0
                and hasattr(conv_config, "precip_efficiency")):
            conv_config = conv_config._replace(precip_efficiency=_pe)

        # Convective precip-split SCHEME (Bechtold / Tiedtke expose
        # ``precip_split_scheme`` + the autoconv params).  "autoconversion"
        # replaces the constant ``precip_efficiency`` with the PHYSICAL
        # Sundqvist-1978 split on the plume updraft cloud water.  hasattr-guarded
        # so a scheme without the field keeps its default "constant"; the scheme
        # body raises on an unknown value (dispatch-hardening).
        _split = getattr(config, "convective_precip_split", "constant")
        if _split != "constant":
            if not hasattr(conv_config, "precip_split_scheme"):
                # A requested non-constant split on a scheme that cannot honour
                # it must fail loudly, not silently run constant physics (codex
                # MED). Only bechtold/tiedtke expose the plume q_c_u it needs.
                raise ValueError(
                    f"convective_precip_split={_split!r} requires a convection "
                    "scheme with the physical autoconversion split (bechtold or "
                    f"tiedtke); scheme {scheme!r} does not support it")
            conv_config = conv_config._replace(
                precip_split_scheme=_split,
                autoconv_q_c_crit=getattr(config, "autoconv_q_c_crit", 5.0e-4),
                autoconv_pe_max=getattr(config, "autoconv_pe_max", 0.9))

    _check_pipeline_convection_supported(scheme, conv_config)

    return conv_fn, conv_config


def _check_pipeline_convection_supported(scheme, conv_config):
    """Build-time guard: reject convection configs whose extra state the
    unified driver cannot thread.  Currently only Bechtold's stochastic
    mode (today's ``ExperimentConfig`` cannot reach it — Bechtold
    tunables aren't exposed there yet — but the guard keeps any future
    tunable wiring from silently running the deterministic path)."""
    if scheme == "bechtold" and getattr(conv_config, "enable_stochastic", False):
        raise NotImplementedError(
            "bechtold with enable_stochastic=True is not supported by the "
            "unified driver pipeline: the stochastic AR1 multiplier needs "
            "a per-segment PRNG-key carry (and a conv_stoch_state "
            "checkpoint slot) that the driver does not thread.  Run the "
            "deterministic default (enable_stochastic=False, bit-identical "
            "plumbing), or drive the scheme through `legoesm.atmosphere."
            "physics.convection.integration.make_convection_physics`."
        )


def _noop_convection(T, q_v, p_full, p_half, dt, config):
    """No-op convection kernel returning zeros."""
    from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    ncol, nlev = T.shape
    z2 = jnp.zeros_like(T)
    z1 = jnp.zeros((ncol,), dtype=T.dtype)
    return ConvectionOutput(
        dT_dt=z2, dq_v_dt=z2, dq_c_conv_dt=z2, cape=z1, convective_mask=z1,
    )


# ---------------------------------------------------------------------------
# Microphysics resolver
# ---------------------------------------------------------------------------

# Schemes accepted by ``ExperimentConfig.validate_strict`` that the
# unified pipeline can NOT build.  Empty since audit 2026-06-10 (p3 and
# ml_emulator are wired below); shrink-only, guarded by
# ``tests/unit/test_advertised_buildability.py``.
_PIPELINE_UNSUPPORTED_MICROPHYSICS = frozenset()


def required_microphysics_tracer_slots(
    scheme_name: str,
    scheme_config=None,
) -> int:
    """Return the canonical minimum global tracer slots for a scheme."""
    from legoesm.atmosphere.physics.microphysics.integration import (
        min_tracer_slots,
    )

    try:
        return int(min_tracer_slots(scheme_name, scheme_config))
    except KeyError as exc:
        raise ValueError(
            f"Unknown microphysics scheme {scheme_name!r}; cannot determine "
            "required tracer slots."
        ) from exc


def validate_microphysics_tracer_slots(
    scheme_name: str,
    have_slots: int,
    *,
    context: str,
    scheme_config=None,
) -> int:
    """Fail loudly if a global tracer state cannot hold scheme tendencies."""
    need_slots = required_microphysics_tracer_slots(scheme_name, scheme_config)
    if have_slots < need_slots:
        raise ValueError(
            f"{context} has too few tracer slots for microphysics scheme "
            f"{scheme_name!r}: have={have_slots}, need={need_slots}. "
            "Slot layout is [0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, "
            "[4]=q_s, [5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i."
        )
    return need_slots


def _resolve_microphysics(config):
    """Resolve microphysics kernel and config from ExperimentConfig.

    Returns (kernel_fn, kernel_config) or (None, None) if disabled.

    ``p3`` matches the standard kernel contract directly — its prognostic
    ice properties reuse the 9-slot hydrometeor layout (``q_s`` → rime
    mass ``q_rim``, ``q_g`` → rime volume ``B_rim``; the driver's full
    moisture registry already allocates those tracers for p3).

    ``ml_emulator`` takes the Equinox network as an extra argument, so —
    mirroring the bridge factory — the emulator is built once here from
    the scheme config (random-init from ``config.seed``; training code
    swaps trained weights in) and bound into a standard-contract wrapper.
    """
    if config.microphysics == "none":
        return None, None

    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.driver.kernel_registry import (
        MICROPHYSICS_REGISTRY, resolve_kernel,
    )

    scheme = config.microphysics
    micro_fn = resolve_kernel(MICROPHYSICS_REGISTRY, scheme)
    mc = MicrophysicsConfig(scheme=scheme)
    micro_config = getattr(mc, scheme)
    required_microphysics_tracer_slots(scheme, micro_config)

    # Aerosol-CCN coupling (Andreae 2009 AOD->CCN): only meaningful for
    # schemes whose warm rain consumes a droplet number through
    # ``effective_Nc`` with a specified-Nc mode (currently Morrison).
    # Fail loudly on a scheme that would silently ignore the flag.
    if getattr(config, "nc_from_aerosol", False):
        if "nc_from_aerosol" not in getattr(micro_config, "_fields", ()):
            raise ValueError(
                f"nc_from_aerosol=True is not supported by the "
                f"{scheme!r} microphysics scheme (no specified-Nc "
                "aerosol mode); use --microphysics morrison or drop "
                "--aerosol-ccn."
            )
        micro_config = micro_config._replace(nc_from_aerosol=True)

    # Sub-grid in-cloud warm-rain closure (#613): evaluate the non-linear
    # KK2000 warm-rain rates on in-cloud condensate (q_c / cloud fraction).
    # Fail loudly on a scheme that has no such field rather than silently
    # ignoring the flag.
    if getattr(config, "subgrid_autoconversion", False):
        if "subgrid_autoconversion" not in getattr(micro_config, "_fields", ()):
            raise ValueError(
                f"subgrid_autoconversion=True is not supported by the "
                f"{scheme!r} microphysics scheme (no sub-grid warm-rain "
                "closure); use --microphysics morrison or drop "
                "--subgrid-autoconv."
            )
        micro_config = micro_config._replace(subgrid_autoconversion=True)

    if scheme == "ml_emulator":
        from legoesm.atmosphere.physics.microphysics.ml_emulator import (
            MicrophysicsEmulator,
        )
        _model = MicrophysicsEmulator(
            micro_config.n_input, micro_config.n_hidden,
            micro_config.n_layers, micro_config.n_output,
            key=jax.random.PRNGKey(micro_config.seed),
        )
        _ml_kernel = micro_fn

        def micro_fn(*, T, q_v, hydrometeors, p_full, p_half, rho, dz, dt,
                     config):
            return _ml_kernel(
                T, q_v, hydrometeors, p_full, p_half, rho, dz, dt,
                config, _model,
            )

    return micro_fn, micro_config


# ---------------------------------------------------------------------------
# Turbulence resolver
# ---------------------------------------------------------------------------

def apply_surface_flux_config(tc, config):
    """Propagate the experiment-level surface bulk-flux settings into the
    active scheme's ``SurfaceLayerConfig``.

    Single source of truth for EVERY dycore backend (#870): the FV pipeline,
    the MPAS standalone physics, and the spectral path all consume the
    ``TurbulenceConfig`` this returns, so ``surface_bulk_scheme=coare3`` /
    ``surface_gustiness_zi`` / ``surface_thermo_convention`` /
    ``surface_stability_scheme`` reach the surface fluxes identically on all
    grids.  (Previously this injection lived only in the FV
    ``_resolve_turbulence`` — the MPAS AMIP lane silently ran the default
    constant-coefficient surface layer.)

    Default experiment settings => ``tc`` returned UNCHANGED (same object,
    byte-identical; the override identity contract in
    ``test_turbulence_config_for_default_vs_override`` relies on this).
    """
    sbs = getattr(config, "surface_bulk_scheme", "constant")
    gzi = getattr(config, "surface_gustiness_zi", None)
    stc = getattr(config, "surface_thermo_convention", "legoesm")
    sss = getattr(config, "surface_stability_scheme", "dyer1974")
    if (sbs == "constant" and gzi is None and stc == "legoesm"
            and sss == "dyer1974"):
        return tc
    sub = getattr(tc, tc.scheme, None)  # e.g. tc.louis; "none"/clubb=None safe
    if sub is None or getattr(sub, "surface", None) is None:
        return tc
    surf = sub.surface
    if sbs != "constant":
        surf = surf._replace(bulk_scheme=sbs)
    if gzi is not None:
        # COARE convective-gustiness BL depth (only effective with a MOST
        # bulk_scheme); the diagnosed fix for the calm-warm-ocean low hfls.
        surf = surf._replace(gustiness_w_zi=gzi)
    if stc != "legoesm":
        # AeroBulk thermodynamic-constants parity (#762; only effective
        # with a MOST bulk_scheme).
        surf = surf._replace(thermo_convention=stc)
    if sss != "dyer1974":
        # Stable-regime MOST functions: keep the atmosphere surface layer
        # on the SAME stable functions as the coupler ocean tile (both
        # driven by the one --surface-stability-scheme flag) so the
        # interface cannot split Dyer-vs-SHEBA across its two sides.
        surf = surf._replace(stability_scheme=sss)
    return tc._replace(**{tc.scheme: sub._replace(surface=surf)})


def turbulence_config_for(config):
    """The ``TurbulenceConfig`` to build the turbulence kernel from.

    The explicit ``config.turbulence_override`` if set (it must share
    ``config.turbulence``'s scheme — enforced by
    ``ExperimentConfig.validate_strict``), else the default
    ``TurbulenceConfig(scheme=config.turbulence)``.  Single source of truth so
    every dycore backend (FV ``_resolve_turbulence``, MPAS, spectral) honours an
    injected override consistently (e.g. a corrected per-column
    ``clubb_lite.C_K`` from the LES-informed correction loop).

    The experiment-level surface bulk-flux settings are applied here too
    (``apply_surface_flux_config``) so every backend gets the same surface
    layer (#870).
    """
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    override = getattr(config, "turbulence_override", None)
    if override is None:
        tc = TurbulenceConfig(scheme=getattr(config, "turbulence", "none"))
        # Thread the experiment-level marine-Sc cloud-top entrainment flag into
        # the ACTIVE scheme's nested config HERE — the single source of truth all
        # dycores consume (FV via _resolve_turbulence, MPAS + spectral directly),
        # so the knob is not silently inert on MPAS/spectral (the l_mix lesson).
        # Only for a scheme that carries the field (louis).  Default off (flag
        # False) => byte-identical (no _replace).  An explicit turbulence_override
        # (below) is authoritative and is never touched here.
        eff = getattr(config, "louis_cloudtop_entrainment_efficiency", 0.0)
        if eff > 0.0:
            scheme = tc.scheme
            nested = getattr(tc, scheme, None)
            if (nested is not None
                    and "cloudtop_entrainment_efficiency" in getattr(nested, "_fields", ())):
                tc = tc._replace(**{scheme: nested._replace(
                    cloudtop_entrainment_efficiency=eff)})
        return apply_surface_flux_config(tc, config)
    # Under MPI a GLOBAL per-column override must be sliced to the rank's columns
    # (else broadcast_column_param mismatches the rank-local l_mix). Deferred so the
    # parallel layout machinery is only touched when an override is actually set;
    # active_column_layout() is None in serial → a strict no-op (override verbatim),
    # and resolves the layout across lat-lon / cubed-sphere / MPAS grid families.
    from legoesm.atmosphere.physics.turbulence.override_sharding import (
        active_column_layout,
        localize_turbulence_override,
    )

    layout = active_column_layout()
    tc = override if layout is None else localize_turbulence_override(
        override, layout)
    return apply_surface_flux_config(tc, config)


def _resolve_turbulence(config):
    """Resolve turbulence kernel and config from ExperimentConfig.

    Returns (kernel_fn, kernel_config) or (None, None) if disabled.
    """
    scheme = getattr(config, 'turbulence', 'none')
    if scheme == "none":
        return None, None

    from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn

    tc = turbulence_config_for(config)
    # The experiment-level surface bulk-flux settings (coare3 / gustiness /
    # thermo convention / stability scheme) are already applied by
    # ``turbulence_config_for`` -> ``apply_surface_flux_config`` (#870), so
    # the sub-config returned here carries the patched SurfaceLayerConfig.
    _name, turb_fn, turb_config = get_turbulence_fn(tc)
    return turb_fn, turb_config


# ---------------------------------------------------------------------------
# Gravity wave drag resolver
# ---------------------------------------------------------------------------

def _resolve_gwd(config):
    """Resolve gravity wave drag kernel and config from ExperimentConfig.

    Returns (kernel_fn, kernel_config) or (None, None) if disabled.
    """
    scheme = getattr(config, 'gravity_wave_drag', 'none')
    if scheme == "none":
        return None, None

    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        get_gwd_fn,
    )

    gc = GravityWaveDragConfig(scheme=scheme)
    _name, gwd_fn, gwd_config = get_gwd_fn(gc)
    return gwd_fn, gwd_config


def _resolve_physics_parameterization(config, nlev: int):
    """Resolve the optional joint ML physics parameterization."""
    scheme = getattr(config, 'physics_parameterization', 'none')
    if scheme == "none":
        return None
    if scheme != "ml":
        raise ValueError(
            f"Unknown physics_parameterization={scheme!r}. Supported: 'none', 'ml'.",
        )
    if getattr(config, 'convection', 'none') != "mass_flux":
        raise ValueError(
            "physics_parameterization='ml' requires convection='mass_flux'",
        )
    if getattr(config, 'turbulence', 'none') != "louis":
        raise ValueError(
            "physics_parameterization='ml' requires turbulence='louis'",
        )
    microphysics_scheme = getattr(config, 'microphysics', 'none')
    if microphysics_scheme not in ("none", "kessler", "sundqvist"):
        raise ValueError(
            "physics_parameterization='ml' currently supports "
            "microphysics='none', 'kessler', or 'sundqvist'",
        )
    from legoesm.atmosphere.physics.ml_parameterization import (
        load_physics_parameterization_assets,
    )
    return load_physics_parameterization_assets(
        nlev=nlev,
        hidden_dim=getattr(config, 'physics_parameterization_hidden_dim', 128),
        n_layers=getattr(config, 'physics_parameterization_layers', 3),
        seed=getattr(config, 'physics_parameterization_seed', 0),
        checkpoint_path=getattr(config, 'physics_parameterization_checkpoint', ''),
        stats_path=getattr(config, 'physics_parameterization_stats', ''),
        microphysics_scheme=microphysics_scheme,
    )


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_physics_pipeline(grid, sigma, config):
    """Build a PhysicsPipeline from an ExperimentConfig.

    All scheme selection happens here via registries; the resulting
    ``PhysicsPipeline`` contains only resolved callables and performs
    no ``if/elif`` dispatch at runtime.

    Parameters
    ----------
    grid : CubedSphereGrid, LatLonGrid, SingleColumnGrid, or similar
        Any grid satisfying ``GridProtocol``.
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : ExperimentConfig

    Returns
    -------
    PhysicsPipeline
    """
    # Build grid-agnostic column adapter
    adapter = make_adapter(grid)

    # Resolve radiation via registry-driven builder
    rad_scheme = config.radiation
    if rad_scheme not in _RADIATION_BUILDERS:
        raise ValueError(
            f"Unknown radiation scheme {rad_scheme!r}; expected one of "
            f"{sorted(_RADIATION_BUILDERS)}. (Previously this silently defaulted "
            "to rrtmgp, masking typos and running full RRTMGP for 'none'.)"
        )
    radiation_fn = _RADIATION_BUILDERS[rad_scheme](config)

    # Resolve convection via registry
    convection_fn, convection_config = _resolve_convection(config)

    # Resolve microphysics via registry
    micro_fn, micro_config = _resolve_microphysics(config)

    # Water-budget closure guard (root cause of the coarse-CMIP6 pr=0 +
    # corrupted-TOA-flux bug, 2026-06-15).  Convection no longer surfaces its
    # own precipitation: it detrains condensate into the cloud-water bucket
    # (``dq_c_conv_dt``) and surface precip is owned by
    # ``micro_out.precipitation`` (see ``physics_step_no_rad``).  With
    # ``microphysics='none'`` that convective condensate has NO sink, so:
    #   (a) surface precipitation is identically zero (CMOR ``pr`` = 0), and
    #   (b) ``q_c`` accumulates without bound — and if a cloud scheme is
    #       active, the unbounded ``q_c`` drives the cloud optics to
    #       optically-thick/garbage values, corrupting the radiation
    #       (TOA SW/LW fluxes diverged: rsut->470, rlut->8 W/m^2).
    # Idealized dry/moist-adjustment tests legitimately run convection with no
    # microphysics, so this is a loud WARNING (not a hard error); a realistic
    # coupled run must enable a microphysics scheme (e.g. 'kessler') to close
    # the water budget.  ADJUSTMENT schemes (sbm/dca/kuo, ``detrains_to_cloud=
    # False``) are EXEMPT — they precipitate their convective drying DIRECTLY
    # (the TOA-drift fix), so they close the budget without microphysics and
    # never trap q_c; only TRUE-detrainment schemes (which feed q_c, whose only
    # sink is microphysics) hit this trap.
    from legoesm.atmosphere.physics.convection.integration import (
        convection_scheme_traits as _cst,
    )
    if (config.convection != "none" and config.microphysics == "none"
            and _cst(config.convection).detrains_to_cloud):
        _extra = (
            " AND cloud_scheme=%r is active, so the unbounded cloud water "
            "will also corrupt the cloud-radiation optics" % config.cloud_scheme
            if getattr(config, "cloud_scheme", "none") != "none" else ""
        )
        logger.warning(
            "convection=%r with microphysics='none': the detrained ANVIL "
            "cloud water (dq_c_conv_dt) has no precipitation sink and "
            "accumulates unbounded (water trap)%s. Mass-flux schemes with an "
            "in-updraft rain split (precip_efficiency>0 — e.g. Bechtold's 0.7 "
            "default) DO precipitate their rain fraction (dq_r_conv_dt) to the "
            "surface each step, so surface precipitation is NOT necessarily "
            "zero, but the suspended anvil fraction still needs a microphysics "
            "sink. Enable a microphysics scheme (e.g. --microphysics kessler) "
            "to close the water budget.",
            config.convection, _extra,
        )

    # Resolve turbulence
    turb_fn, turb_config = _resolve_turbulence(config)

    # Resolve gravity wave drag
    gwd_fn, gwd_config = _resolve_gwd(config)

    # Resolve optional joint ML physics parameterization
    physics_parameterization = _resolve_physics_parameterization(
        config,
        nlev=int(sigma.sigma_full.shape[0]),
    )

    # Issue #273 follow-up: build a column-shard mesh when the
    # ExperimentConfig opts in.  The mesh shards the flattened column
    # axis across the *runtime-selected* device set so the per-column
    # radiation kernel parallelizes on device counts that fail
    # cubed-sphere face-divisibility (e.g. 4-GPU node) while still
    # honoring whatever subset of visible devices the active
    # ``ParallelRuntime`` / ``DeviceConfig`` owns.
    #
    # Codex adversarial review 019e544b (2026-05-23): using raw
    # ``jax.devices()`` here would silently override a runtime that
    # had been bootstrapped onto a subset of devices (e.g. an
    # ensemble member that explicitly took 2-of-4) and create
    # hard-to-debug cross-mesh resharding.  Always prefer the active
    # ``DeviceConfig.mesh.devices``; fall back to ``jax.devices()``
    # only when no runtime is active (e.g. unit tests that build the
    # pipeline directly without a bootstrap step).
    column_mesh = None
    if getattr(config, "shard_radiation_columns", False):
        from legoesm.parallel.column_shard import create_column_mesh
        from legoesm.parallel.mesh import get_active_config
        import jax

        active = get_active_config()
        if active is not None and active.mesh is not None:
            runtime_devices = list(active.mesh.devices.reshape(-1))
        else:
            runtime_devices = list(jax.devices())

        n_runtime_devices = len(runtime_devices)
        if n_runtime_devices > 1:
            # Validate up-front so a misconfiguration fails at build
            # time, not deep inside the JIT'd hot path on the first
            # radiation call.
            if adapter.ncol % n_runtime_devices != 0:
                raise ValueError(
                    f"shard_radiation_columns=True requires "
                    f"adapter.ncol={adapter.ncol} divisible by the "
                    f"runtime-active device count "
                    f"({n_runtime_devices}).  Pick a device count "
                    f"that divides 6·n·n on the cubed sphere, "
                    f"reduce ExperimentConfig.n_devices to a value "
                    f"that divides ncol, or disable "
                    f"shard_radiation_columns."
                )
            column_mesh = create_column_mesh(
                n_devices=n_runtime_devices,
                devices=runtime_devices,
            )

    pipeline = PhysicsPipeline(
        adapter=adapter,
        sigma_full=sigma.sigma_full,
        sigma_half=sigma.sigma_half,
        dsigma=sigma.dsigma,
        convection_fn=convection_fn,
        convection_config=convection_config,
        radiation_fn=radiation_fn,
        T_ice=config.T_ice,
        C_H=config.C_H,
        C_E=config.C_E,
        albedo_ice=config.albedo_ice,
        albedo_ocean=config.albedo_ocean,
        emissivity_ice=config.emissivity_ice,
        emissivity_ocean=config.sfc_emissivity,
        micro_fn=micro_fn,
        micro_config=micro_config,
        dynamic_albedo=config.dynamic_albedo,
        diurnal_cycle=config.diurnal_cycle,
        turbulence_fn=turb_fn,
        turbulence_config=turb_config,
        gwd_fn=gwd_fn,
        gwd_config=gwd_config,
        physics_parameterization=physics_parameterization,
        column_mesh=column_mesh,
    )
    from legoesm.atmosphere.physics.radiation.solar import earth_orbit
    pipeline.orbit = (earth_orbit()
                      if getattr(config, 'orbital_insolation', False) else None)
    pipeline._cloud_scheme = getattr(config, 'cloud_scheme', 'none')
    # Clear-sky diagnostic (#843): enable the 2nd clouds-off radiation pass
    # only when config.output.clear_sky_diag is set (default off).
    pipeline._clear_sky_diag = bool(
        getattr(getattr(config, 'output', None), 'clear_sky_diag', False))
    pipeline._cloud_convective = getattr(config, 'convective_cloud', False)
    pipeline._cloud_rh_crit = getattr(config, 'cloud_rh_crit', None)
    pipeline._cloud_q_c_diagnostic = getattr(config, 'cloud_q_c_diagnostic', None)
    pipeline._cloud_conv_cloud_max = getattr(config, 'cloud_conv_cloud_max', None)
    pipeline._cloud_conv_cloud_condensate = getattr(
        config, 'cloud_conv_cloud_condensate', None)
    pipeline._cloud_inhomogeneity_factor = getattr(
        config, 'cloud_inhomogeneity_factor', None)
    pipeline._cloud_optics_inhomogeneity = getattr(
        config, 'cloud_optics_inhomogeneity', None)
    pipeline._cloud_fsd = getattr(config, 'cloud_fsd', None)
    pipeline._cloud_p_xr = getattr(config, 'cloud_p_xr', None)
    pipeline._cloud_alpha_xr = getattr(config, 'cloud_alpha_xr', None)
    pipeline._cloud_diagnostic_condensate_scheme = getattr(
        config, 'cloud_diagnostic_condensate_scheme', None)
    pipeline._cloud_adiabatic_lwc_rate = getattr(
        config, 'cloud_adiabatic_lwc_rate', None)
    pipeline._conv_scheme = getattr(config, 'convection', 'none')
    pipeline._grid = grid
    pipeline._sigma_coord = sigma
    # Stateful-physics carry plumbing (issue #413): energy slot from the
    # shared traits (cannot drift from seeding/guard), prognostic GWD flag.
    from legoesm.atmosphere.physics.turbulence.integration import (
        turbulence_scheme_traits,
    )
    pipeline._turb_energy_field = turbulence_scheme_traits(
        getattr(config, 'turbulence', 'none'),
    ).energy_field
    # A GWD scheme threads the prognostic wave-action spectrum when it is
    # ``prognostic_spectral`` OR a '+'-composite that contains it (issue #834).
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        gwd_carries_spectrum,
    )
    _gwd_scheme = getattr(config, 'gravity_wave_drag', 'none')
    pipeline._gwd_prognostic = gwd_carries_spectrum(_gwd_scheme)
    # A stateless '+'-composite (no prognostic_spectral part) still returns the
    # (GWDOutput, spectrum_out) tuple from the combined executor, so the
    # pipeline must unpack it via the composite branch rather than the plain
    # single-return path.
    pipeline._gwd_composite = (
        "+" in _gwd_scheme and not pipeline._gwd_prognostic
    )
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        gwd_scheme_is_orographic,
    )
    pipeline._gwd_orographic = gwd_scheme_is_orographic(
        getattr(config, 'gravity_wave_drag', 'none'),
    )
    return pipeline
