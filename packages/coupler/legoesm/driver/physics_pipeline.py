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
from legoesm.forcing.surface_utils import blend_surface_temperature
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
        self.micro_fn = micro_fn
        self.micro_config = micro_config
        # ``dynamic_albedo``: zenith-angle-dependent ocean albedo
        # (Briegleb 1992 via ``legoesm.surface_albedo.ocean_albedo``)
        # applied in ``compute_radiation_core`` before the sea-ice/land
        # blends.  ``diurnal_cycle`` selects instantaneous cos(SZA) vs
        # the daytime-effective daily-mean cosine — matching the zenith
        # convention the radiation solver itself uses.
        self.dynamic_albedo = dynamic_albedo
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

    def _blend_land(self, ocean_field, land_field):
        """Blend an ocean/ice surface field with a land field by ``f_land``.

        ``f_land`` broadcasts against the 2-D surface fields.  Only called
        when ``self.f_land is not None`` (the land tile is active).
        """
        return self.f_land * land_field + (1.0 - self.f_land) * ocean_field

    def _step_slab_land(self, T_land, sw_down_sfc, lw_down_sfc,
                        T, p_s, q_v, u, v, dt):
        """Advance the slab-land skin temperature by one radiation step.

        Semi-implicit surface energy balance::

            C_land dT/dt = SW_net + eps*LW_down - eps*sigma*T^4 - SH - LH

        linearized about the current ``T_land``.  Every flux term damps
        (``dF/dT < 0``), so the denominator ``C_land - dt*dF/dT`` is
        always larger than ``C_land`` and the update is unconditionally
        stable for any radiation cadence.  Land evaporation uses the
        saturated (wet-surface) bulk flux — no soil-moisture limit.
        """
        T_air = T[..., -1]
        q_air = q_v[..., -1]
        rho_low = (p_s * self.sigma_full[-1]) / (constants.R_d * T_air)
        wind_speed = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1.0)
        sh_coef = rho_low * constants.c_pd * self.C_H * wind_speed
        lh_coef = rho_low * constants.L_v * self.C_E * wind_speed
        eps = self.emissivity_land
        sb = constants.sigma_sb

        q_sat_land = saturation_specific_humidity(T_land, p_s)
        sw_net = sw_down_sfc * (1.0 - self.albedo_land)
        lw_net = eps * lw_down_sfc - eps * sb * T_land ** 4
        shflx = sh_coef * (T_land - T_air)
        lhflx = lh_coef * (q_sat_land - q_air)
        flux = sw_net + lw_net - shflx - lhflx

        # Clausius-Clapeyron derivative of saturation specific humidity.
        dqsat_dT = q_sat_land * constants.L_v / (constants.R_v * T_land ** 2)
        dflux_dT = (-4.0 * eps * sb * T_land ** 3
                    - sh_coef - lh_coef * dqsat_dT)

        dt_rad = dt * self.rad_update_steps
        return T_land + dt_rad * flux / (self.C_land - dt_rad * dflux_dT)

    def physics_step_no_rad(self, T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic,
                            lat, dt, dT_dt_rad, sw_net_sfc, lw_net_sfc,
                            sw_up_toa, lw_up_toa, sw_down_toa,
                            sbm_tau_c=None, sbm_RH_ref=None,
                            C_H=None, C_E=None,
                            q_i=None, q_s=None, q_g=None,
                            N_c=None, N_r=None, N_i=None,
                            T_land=None, aerosol_od=None,
                            sfc_shflx_override=None, sfc_lhflx_override=None,
                            tke=None, qke=None, gwd_spectrum=None):
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
                _sc = self.gwd_config
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
            else:
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
        )
        if self.diurnal_cycle:
            hour = seconds_of_day / 3600.0
            cos_sza = cos_zenith_angle(lat, lon, day_of_year, hour)
            return s_0 * jnp.maximum(cos_sza, 0.0)
        return daily_mean_insolation(lat, day_of_year, s_0)

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
                               conv_precip=None):
        """Compute radiation tendencies and fluxes (pure JAX, no I/O).

        Returns ``(dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa,
        lw_up_toa, sw_down_toa, T_land_new)`` as a 7-tuple.

        When the land tile is active (``self.f_land is not None``) the
        surface temperature/albedo/emissivity passed to the radiation
        solver are land/ocean blends, and the slab-land skin temperature
        ``T_land`` is advanced one radiation step by a semi-implicit
        surface energy balance.  Otherwise ``T_land`` is returned
        unchanged and the surface is pure ocean/ice.
        """
        from legoesm.forcing.surface_utils import blend_surface_property

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
            )
            if self.diurnal_cycle:
                _hour = seconds_of_day / 3600.0
                _mu = jnp.maximum(
                    cos_zenith_angle(lat, lon, day_of_year, _hour), 0.0,
                )
            else:
                _q_day = daily_mean_insolation(lat, day_of_year, s_0)
                _f_day = daylight_fraction(lat, day_of_year)
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

        # --- Land tile: blend land surface into T_sfc / albedo / emissivity
        _land_active = self.f_land is not None and T_land is not None
        if _land_active:
            T_sfc = self._blend_land(T_sfc, T_land)
            albedo = self._blend_land(albedo, self.albedo_land)
            emissivity = self._blend_land(emissivity, self.emissivity_land)

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
            from legoesm.atmosphere.physics.clouds.config import CloudConfig
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
            _cc_over = {}
            if getattr(self, "_cloud_rh_crit", None) is not None:
                _cc_over["rh_crit"] = self._cloud_rh_crit
            if getattr(self, "_cloud_q_c_diagnostic", None) is not None:
                _cc_over["q_c_diagnostic"] = self._cloud_q_c_diagnostic
            if getattr(self, "_cloud_conv_cloud_max", None) is not None:
                _cc_over["conv_cloud_max"] = self._cloud_conv_cloud_max
            cloud_config = CloudConfig(
                scheme=cloud_scheme,
                convective_cloud=(getattr(self, "_cloud_convective", False)
                                  and conv_precip is not None),
                **_cc_over,
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

        # --- Slab-land skin temperature update (semi-implicit SEB) ---
        if _land_active:
            lw_down_sfc = ad.unflatten_2d(rad_out.lw_flux_down[:, -1])
            T_land_new = self._step_slab_land(
                T_land, sw_down_sfc, lw_down_sfc, T, p_s, q_v, u, v, dt,
            )
        else:
            T_land_new = T_land

        return (dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa,
                sw_down_toa, T_land_new)

    def build_step_unified(self, static_need_rad: bool | None = None):
        """Build a JIT-compiled unified physics step with radiation sub-cycling.

        Returns a function ``step_unified(need_rad, T, p_s, q_v, q_c, q_r,
        conv_prog, u, v, sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
        solar_weights, s_0, o3_vmr, aerosol_od, held, ..., T_land) ->
        (PhysicsOutput, held tuple, T_land_new)``.

        ``T_land`` is the slab-land skin temperature carried through the
        radiation sub-cycle; it is advanced on radiation steps and held
        constant otherwise.  Pass ``None`` (the default) for ocean-only
        runs — the land tile is then inert.

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

        @jax.jit
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
                         sfc_shflx_override=None,
                         sfc_lhflx_override=None,
                         tke=None, qke=None, gwd_spectrum=None,
                         conv_precip=None):

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
                 sfc_albedo_override, sfc_T_override,
                 sfc_shflx_override, sfc_lhflx_override,
                 tke, qke, gwd_spectrum, conv_precip) = args

                (dT_dt_rad, sw_net_sfc, lw_net_sfc,
                 sw_up_toa, lw_up_toa, sw_down_toa, T_land_new) = \
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
                        conv_precip=conv_precip,
                    )

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
                            physics_out.gwd_spectrum)
                physics_out = jax.tree.map(_cast, physics_out)
                physics_out = physics_out._replace(
                    tke=_carries[0], qke=_carries[1],
                    gwd_spectrum=_carries[2],
                )
                return physics_out, new_held, _cast(T_land_new)

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
                 sfc_albedo_override, sfc_T_override,
                 sfc_shflx_override, sfc_lhflx_override,
                 tke, qke, gwd_spectrum, conv_precip) = args
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
                            physics_out.gwd_spectrum)
                physics_out = jax.tree.map(_cast, physics_out)
                physics_out = physics_out._replace(
                    tke=_carries[0], qke=_carries[1],
                    gwd_spectrum=_carries[2],
                )
                return physics_out, new_held, _cast(T_land)

            args = (T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, lon,
                    day_of_year, seconds_of_day, dt,
                    solar_weights, s_0, o3_vmr, aerosol_od, aerosol_lw_od,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                    tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref,
                    C_H, C_E, albedo_ice, albedo_ocean,
                    ghg_vmr_override, T_land,
                    q_i, q_s, q_g, N_c, N_r, N_i,
                    sfc_albedo_override, sfc_T_override,
                    sfc_shflx_override, sfc_lhflx_override,
                    tke, qke, gwd_spectrum, conv_precip)

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

        return step_unified


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
    )
    from legoesm.driver.kernel_registry import (
        RADIATION_REGISTRY, resolve_kernel,
    )

    gray_radiation = resolve_kernel(RADIATION_REGISTRY, "gray")
    diurnal = config.diurnal_cycle
    S_0 = config.S_0

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
            cos_sza = cos_zenith_angle(lat_col, lon_col, day_of_year, hour)
            insol = s_0 * jnp.maximum(cos_sza, 0.0)
        else:
            insol = daily_mean_insolation(lat_col, day_of_year, s_0)
        # Thread the pipeline's blended (ice/ocean/land, plus coupler
        # overrides) surface albedo into the gray SW reflection so the
        # solver sees the same surface as the energy budget — previously
        # gray used only the static ``config.sfc_albedo`` and the
        # blended albedo was silently dropped (audit 2026-06-10).
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
    )
    from legoesm.atmosphere.physics.radiation.output import RadiationOutput

    diurnal = config.diurnal_cycle
    S_0 = config.S_0

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
        if diurnal:
            hour = seconds_of_day / 3600.0
            cos_sza = cos_zenith_angle(lat_col, lon_col, day_of_year, hour)
            cos_zenith = jnp.maximum(cos_sza, 0.0)
            # Prescribed TOA incident SW = S_0·max(cosθ,0) for the rsdt
            # diagnostic (#620); matches _compute_insolation's diurnal return.
            insol = s_0 * cos_zenith
        else:
            # Daytime-effective cos(SZA): use daylight fraction so the solver
            # sees the correct optical path during sunlit hours.  SW fluxes
            # are then rescaled by f_day to recover daily-mean energy.
            insol = daily_mean_insolation(lat_col, day_of_year, s_0)
            f_day = daylight_fraction(lat_col, day_of_year)
            f_day_safe = jnp.maximum(f_day, 1.0e-6)
            cos_zenith = jnp.clip(
                insol / (s_0 * f_day_safe), 0.0, 1.0,
            )
            _sw_scale = f_day

        # Water-vapor unit convention: the upstream pipeline passes
        # ``q_v`` as **mixing ratio** r = m_v / m_d.  RRTMGP's internal
        # H2O VMR formula ``mol_ratio * q / (1 - q)`` expects **specific
        # humidity** q = m_v / (m_v + m_d).  Convert at the solver
        # boundary so that gray radiation and other consumers of
        # ``q_v_col`` (e.g. cloud-fraction diagnostics) keep their
        # mixing-ratio inputs while RRTMGP sees the right unit.
        # Audit 2026-05-12 #6, narrowed to RRTMGP per Codex review.
        q_v_specific = q_v_col / (1.0 + jnp.clip(q_v_col, 0.0, None))
        result = solver.solve_columns(
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
    else:
        cc = ConvectionConfig(scheme=scheme)
        conv_config = getattr(cc, scheme)

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

def _resolve_turbulence(config):
    """Resolve turbulence kernel and config from ExperimentConfig.

    Returns (kernel_fn, kernel_config) or (None, None) if disabled.
    """
    scheme = getattr(config, 'turbulence', 'none')
    if scheme == "none":
        return None, None

    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn

    tc = TurbulenceConfig(scheme=scheme)
    _name, turb_fn, turb_config = get_turbulence_fn(tc)
    # Propagate the experiment-level surface bulk-flux algorithm into the
    # scheme's SurfaceLayerConfig.  Default "constant" => unchanged (byte-
    # identical).  The stability-dependent MOST schemes (coare3/large_yeager)
    # add the convective-gustiness w* term absent from the constant neutral
    # coefficients — the fix for anemic evaporation over a calm warm ocean.
    sbs = getattr(config, "surface_bulk_scheme", "constant")
    gzi = getattr(config, "surface_gustiness_zi", None)
    if (turb_config is not None
            and getattr(turb_config, "surface", None) is not None
            and (sbs != "constant" or gzi is not None)):
        surf = turb_config.surface
        if sbs != "constant":
            surf = surf._replace(bulk_scheme=sbs)
        if gzi is not None:
            # COARE convective-gustiness BL depth (only effective with a MOST
            # bulk_scheme); the diagnosed fix for the calm-warm-ocean low hfls.
            surf = surf._replace(gustiness_w_zi=gzi)
        turb_config = turb_config._replace(surface=surf)
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
            "convection=%r with microphysics='none': convective condensate "
            "detrains into q_c with no precipitation sink, so surface "
            "precipitation is identically ZERO and cloud water accumulates "
            "unbounded (water trap)%s. Enable a microphysics scheme "
            "(e.g. --microphysics kessler) to close the water budget.",
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
    pipeline._cloud_scheme = getattr(config, 'cloud_scheme', 'none')
    pipeline._cloud_convective = getattr(config, 'convective_cloud', False)
    pipeline._cloud_rh_crit = getattr(config, 'cloud_rh_crit', None)
    pipeline._cloud_q_c_diagnostic = getattr(config, 'cloud_q_c_diagnostic', None)
    pipeline._cloud_conv_cloud_max = getattr(config, 'cloud_conv_cloud_max', None)
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
    pipeline._gwd_prognostic = (
        getattr(config, 'gravity_wave_drag', 'none') == "prognostic_spectral"
    )
    return pipeline
