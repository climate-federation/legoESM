"""Physics pipeline for the composable model driver.

Wraps radiation, convection, microphysics, and boundary-layer exchange
into a single callable that replaces the inline physics step in run_amip.py.

The pipeline is **grid-agnostic**: all flattening/unflattening between
native grid layout and ``(ncol, nlev)`` column format is handled by a
``ColumnAdapter`` (see ``grid_adapters.py``).  Scheme selection is
**registry-driven**: see ``kernel_registry.py``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.forcing.surface_utils import blend_surface_temperature
from legoesm.driver.grid_adapters import ColumnAdapter, make_adapter


class PhysicsOutput(NamedTuple):
    """Output from a single physics step."""
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    dq_c_dt: jax.Array
    dq_r_dt: jax.Array
    precip: jax.Array
    sw_net_sfc: jax.Array
    lw_net_sfc: jax.Array
    sw_up_toa: jax.Array
    lw_up_toa: jax.Array
    sw_down_toa: jax.Array
    du_dt: jax.Array
    dv_dt: jax.Array
    dq_i_dt: jax.Array
    dq_s_dt: jax.Array
    dq_g_dt: jax.Array
    dN_c_dt: jax.Array
    dN_r_dt: jax.Array
    dN_i_dt: jax.Array
    conv_prog: jax.Array
    shflx: jax.Array | None = None   # surface sensible heat flux [W/m2]
    lhflx: jax.Array | None = None   # surface latent heat flux [W/m2]


class HeldRadiation(NamedTuple):
    """Held radiation tendencies for sub-cycling."""
    dT_dt_rad: jax.Array
    sw_net_sfc: jax.Array
    lw_net_sfc: jax.Array
    sw_up_toa: jax.Array
    lw_up_toa: jax.Array
    sw_down_toa: jax.Array


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
        T_ice=271.35,
        C_H=0.0044,
        C_E=0.0044,
        albedo_ice=0.65,
        albedo_ocean=0.06,
        emissivity_ice=0.95,
        emissivity_ocean=0.97,
        micro_fn=None,
        micro_config=None,
        dynamic_albedo=False,
        turbulence_fn=None,
        turbulence_config=None,
        gwd_fn=None,
        gwd_config=None,
        physics_parameterization=None,
    ):
        self.adapter = adapter
        self.sigma_full = sigma_full
        self.sigma_half = sigma_half
        self.dsigma = dsigma
        self.convection_fn = convection_fn
        self.convection_config = convection_config
        self.radiation_fn = radiation_fn
        self.T_ice = T_ice
        self.C_H = C_H
        self.C_E = C_E
        self.albedo_ice = albedo_ice
        self.albedo_ocean = albedo_ocean
        self.emissivity_ice = emissivity_ice
        self.emissivity_ocean = emissivity_ocean
        self.micro_fn = micro_fn
        self.micro_config = micro_config
        self.dynamic_albedo = dynamic_albedo
        self.turbulence_fn = turbulence_fn
        self.turbulence_config = turbulence_config
        self.gwd_fn = gwd_fn
        self.gwd_config = gwd_config
        self.physics_parameterization = physics_parameterization
        self._cloud_scheme = "none"  # set by build_physics_pipeline

    def physics_step_no_rad(self, T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic,
                            lat, dt, dT_dt_rad, sw_net_sfc, lw_net_sfc,
                            sw_up_toa, lw_up_toa, sw_down_toa,
                            sbm_tau_c=None, sbm_RH_ref=None,
                            C_H=None, C_E=None,
                            q_i=None, q_s=None, q_g=None,
                            N_c=None, N_r=None, N_i=None):
        """Convection + microphysics + BL exchange with held radiation."""
        _C_H = self.C_H if C_H is None else C_H
        _C_E = self.C_E if C_E is None else C_E

        ad = self.adapter
        nlev = self.sigma_full.shape[0]
        shape_3d = T.shape
        shape_2d = p_s.shape

        T_sfc = blend_surface_temperature(sst, sic, self.T_ice)

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
        if sbm_RH_ref is not None and _conv_cfg is not None and hasattr(_conv_cfg, 'RH_ref'):
            _conv_cfg = _conv_cfg._replace(RH_ref=sbm_RH_ref)

        if conv_prog is None:
            if _conv_cfg is not None and hasattr(_conv_cfg, 'M_c_init'):
                conv_prog = jnp.full((ad.ncol,), _conv_cfg.M_c_init, dtype=T_col.dtype)
            elif _conv_cfg is not None and hasattr(_conv_cfg, 'a_u_init'):
                conv_prog = jnp.full((ad.ncol,), _conv_cfg.a_u_init, dtype=T_col.dtype)
            else:
                conv_prog = jnp.zeros((ad.ncol,), dtype=T_col.dtype)
        conv_prog_out = conv_prog

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
        # Convection (resolved kernel — no dispatch here)
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
        else:
            conv_out = self.convection_fn(
                T=T_col, q_v=q_v_col, p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=_conv_cfg,
            )
        dT_dt_conv = ad.unflatten_3d(conv_out.dT_dt)
        dq_v_dt_conv = ad.unflatten_3d(conv_out.dq_v_dt)
        precip = ad.unflatten_2d(conv_out.precipitation)

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
            hydrometeors = HydrometeorState(
                q_c=q_c_col, q_r=q_r_col,
                q_i=ad.flatten_3d(q_i) if q_i is not None else _z,
                q_s=ad.flatten_3d(q_s) if q_s is not None else _z,
                q_g=ad.flatten_3d(q_g) if q_g is not None else _z,
                N_c=ad.flatten_3d(N_c) if N_c is not None else _z,
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

        # Boundary layer exchange (grid-agnostic: uses [..., -1] indexing)
        rho_low = (p_s * self.sigma_full[-1]) / (constants.R_d * T[..., -1])
        wind_speed = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1.0)
        dp_low = p_s * (self.sigma_half[-1] - self.sigma_half[-2])

        shflx = rho_low * constants.c_pd * _C_H * wind_speed * (T_sfc - T[..., -1])
        q_sat_sfc = saturation_specific_humidity(T_sfc, p_s)
        lhflx = rho_low * constants.L_v * _C_E * wind_speed * (q_sat_sfc - q_v[..., -1])
        evap_rate = lhflx / constants.L_v

        dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
        dq_BL = constants.g * evap_rate / dp_low

        dT_dt = dT_dt_rad + dT_dt_conv + dT_dt_micro
        dT_dt = dT_dt.at[..., -1].add(dT_BL)

        dq_v_dt = dq_v_dt_conv + dq_v_dt_micro
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
        elif self.turbulence_fn is not None:
            T_sfc_col = ad.flatten_2d(T_sfc)
            q_sat_sfc_col = ad.flatten_2d(
                saturation_specific_humidity(T_sfc, p_s)
            )
            turb_out = self.turbulence_fn(
                u=u_col, v=v_col, T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                z_full=z_full_col, z_half=z_half_col,
                T_sfc=T_sfc_col, q_sfc=q_sat_sfc_col,
                rho=rho_col_phys, dt=dt, config=self.turbulence_config,
            )
            du_dt = du_dt + ad.unflatten_3d(turb_out.du_dt)
            dv_dt = dv_dt + ad.unflatten_3d(turb_out.dv_dt)
            dT_dt = dT_dt + ad.unflatten_3d(turb_out.dT_dt)
            dq_v_dt = dq_v_dt + ad.unflatten_3d(turb_out.dq_v_dt)

        if self.gwd_fn is not None:
            lat_col = ad.flatten_2d(lat)
            gwd_out = self.gwd_fn(
                u=u_col, v=v_col, T=T_col,
                p_full=p_full_col, p_half=p_half_col,
                z_full=z_full_col, z_half=z_half_col,
                rho=rho_col_phys, lat=lat_col,
                dt=dt, config=self.gwd_config,
            )
            du_dt = du_dt + ad.unflatten_3d(gwd_out.du_dt)
            dv_dt = dv_dt + ad.unflatten_3d(gwd_out.dv_dt)
            dT_dt = dT_dt + ad.unflatten_3d(gwd_out.dT_dt)

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
        )

    def compute_radiation_core(self, T, p_s, q_v, sst, sic, lat, lon,
                               day_of_year, seconds_of_day,
                               solar_weights, s_0,
                               o3_vmr_precomputed, aerosol_od_precomputed,
                               tau_equator=None, tau_pole=None,
                               albedo_ice=None, albedo_ocean=None,
                               ghg_vmr_override=None,
                               q_c=None, q_r=None,
                               cloud_scheme="none"):
        """Compute radiation tendencies and fluxes (pure JAX, no I/O).

        Returns (dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa,
                 sw_down_toa) as a 6-tuple.
        """
        from legoesm.forcing.surface_utils import blend_surface_property

        _albedo_ice = self.albedo_ice if albedo_ice is None else albedo_ice
        _albedo_ocean = self.albedo_ocean if albedo_ocean is None else albedo_ocean

        ad = self.adapter
        nlev = self.sigma_full.shape[0]

        T_sfc = blend_surface_temperature(sst, sic, self.T_ice)
        albedo = blend_surface_property(sic, _albedo_ice, _albedo_ocean)
        emissivity = blend_surface_property(sic, self.emissivity_ice, self.emissivity_ocean)

        p_full = p_s[..., None] * self.sigma_full
        p_half = p_s[..., None] * self.sigma_half

        # Flatten to columns via adapter
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
        if cloud_scheme != "none" and q_c is not None:
            from legoesm.atmosphere.physics.clouds.config import CloudConfig
            from legoesm.atmosphere.physics.clouds.cloud_fraction import (
                compute_cloud_properties,
            )
            dp_col = p_half_col[:, 1:] - p_half_col[:, :-1]
            q_c_col = ad.flatten_3d(q_c)
            q_i_col = None
            cloud_config = CloudConfig(scheme=cloud_scheme)
            cloud_props = compute_cloud_properties(
                T=T_col, p_full=p_full_col, q_v=q_v_col, dp=dp_col,
                config=cloud_config, q_cloud=q_c_col, q_ice=q_i_col,
            )
            cloud_kwargs = {
                "cloud_path_liq": cloud_props.lwp,
                "cloud_path_ice": cloud_props.iwp,
                "cloud_r_eff_liq": cloud_props.r_eff_liq,
                "cloud_r_eff_ice": cloud_props.r_eff_ice,
                "cloud_fraction": cloud_props.cloud_fraction,
            }

        rad_out = self.radiation_fn(
            T_col, p_full_col, p_half_col, q_v_col,
            T_sfc_col, lat_col, lon_col,
            day_of_year, seconds_of_day,
            albedo_col, emis_col,
            o3_vmr_precomputed, aerosol_od_precomputed,
            solar_weights, s_0,
            tau_equator=tau_equator, tau_pole=tau_pole,
            ghg_vmr_override=ghg_vmr_override,
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
        sw_down_toa = ad.unflatten_2d(rad_out.sw_flux_down[:, 0])

        return (dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa,
                sw_down_toa)

    def build_step_unified(self):
        """Build a JIT-compiled unified physics step with radiation sub-cycling.

        Returns a function ``step_unified(need_rad, T, p_s, q_v, q_c, q_r,
        conv_prog, u, v, sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
        solar_weights, s_0, o3_vmr, aerosol_od, held) -> (PhysicsOutput,
        HeldRadiation)``.
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
                         ghg_vmr_override=None):

            def _rad_branch(args):
                (T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, lon,
                 day_of_year, seconds_of_day, dt,
                 solar_weights, s_0, o3_vmr, aerosol_od,
                 held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                 held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                 tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref,
                 C_H, C_E, albedo_ice, albedo_ocean,
                 ghg_vmr_override) = args

                (dT_dt_rad, sw_net_sfc, lw_net_sfc,
                 sw_up_toa, lw_up_toa, sw_down_toa) = \
                    pipeline.compute_radiation_core(
                        T, p_s, q_v, sst, sic, lat, lon,
                        day_of_year, seconds_of_day,
                        solar_weights, s_0, o3_vmr, aerosol_od,
                        tau_equator=tau_equator, tau_pole=tau_pole,
                        albedo_ice=albedo_ice, albedo_ocean=albedo_ocean,
                        ghg_vmr_override=ghg_vmr_override,
                        q_c=q_c,
                        cloud_scheme=pipeline._cloud_scheme,
                    )

                physics_out = pipeline.physics_step_no_rad(
                    T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, dt,
                    dT_dt_rad, sw_net_sfc, lw_net_sfc,
                    sw_up_toa, lw_up_toa, sw_down_toa,
                    sbm_tau_c=sbm_tau_c, sbm_RH_ref=sbm_RH_ref,
                    C_H=C_H, C_E=C_E,
                )

                # Cast to storage dtype so both lax.cond branches match
                from legoesm.core.precision import get_policy
                _dt = get_policy().storage
                _cast = lambda x: x.astype(_dt) if hasattr(x, 'astype') else x
                new_held = tuple(_cast(h) for h in (
                    dT_dt_rad, sw_net_sfc, lw_net_sfc,
                    sw_up_toa, lw_up_toa, sw_down_toa,
                ))
                physics_out = jax.tree.map(_cast, physics_out)
                return physics_out, new_held

            def _no_rad_branch(args):
                (T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, lon,
                 day_of_year, seconds_of_day, dt,
                 solar_weights, s_0, o3_vmr, aerosol_od,
                 held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                 held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                 tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref,
                 C_H, C_E, albedo_ice, albedo_ocean,
                 ghg_vmr_override) = args

                physics_out = pipeline.physics_step_no_rad(
                    T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, dt,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                    sbm_tau_c=sbm_tau_c, sbm_RH_ref=sbm_RH_ref,
                    C_H=C_H, C_E=C_E,
                )

                # Cast to storage dtype — must match _rad_branch
                from legoesm.core.precision import get_policy
                _dt = get_policy().storage
                _cast = lambda x: x.astype(_dt) if hasattr(x, 'astype') else x
                new_held = tuple(_cast(h) for h in (
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                ))
                physics_out = jax.tree.map(_cast, physics_out)
                return physics_out, new_held

            args = (T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, lon,
                    day_of_year, seconds_of_day, dt,
                    solar_weights, s_0, o3_vmr, aerosol_od,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                    tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref,
                    C_H, C_E, albedo_ice, albedo_ocean,
                    ghg_vmr_override)

            return jax.lax.cond(need_rad, _rad_branch, _no_rad_branch, args)

        return step_unified


# ---------------------------------------------------------------------------
# Radiation wrapper builders
# ---------------------------------------------------------------------------

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
                     ghg_vmr_override=None):
        del ghg_vmr_override  # gray radiation does not use GHG concentrations
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
        return gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=_cfg,
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

    rrtmg_config = RRTMGPConfig(
        co2_ppmv=config.co2_ppmv,
        ch4_ppbv=config.ch4_ppbv,
        n2o_ppbv=config.n2o_ppbv,
        sfc_emissivity=config.sfc_emissivity,
        sfc_albedo=config.albedo_ocean,
        S_0=S_0,
        use_scan=True,
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
                     cloud_path_liq=None, cloud_path_ice=None,
                     cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                     cloud_fraction=None):
        del tau_equator, tau_pole  # RRTMGP does not use gray optical depth
        _sw_scale = None
        if diurnal:
            hour = seconds_of_day / 3600.0
            cos_sza = cos_zenith_angle(lat_col, lon_col, day_of_year, hour)
            cos_zenith = jnp.maximum(cos_sza, 0.0)
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

        result = solver.solve_columns(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, q_v=q_v_col,
            cos_zenith=cos_zenith,
            sfc_albedo=albedo_col,
            sfc_emissivity=emis_col,
            o3_vmr=o3_vmr_col,
            aerosol_optical_depth=aerosol_od_col,
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

        return result

    return radiation_fn


# Map radiation scheme names to builder functions.
_RADIATION_BUILDERS: dict[str, callable] = {
    "gray": _build_gray_radiation_fn,
    "rrtmgp": _build_rrtmgp_radiation_fn,
    "rrtmg": _build_rrtmgp_radiation_fn,  # common alias
}


# ---------------------------------------------------------------------------
# Convection resolver
# ---------------------------------------------------------------------------

def _resolve_convection(config):
    """Resolve convection kernel and config from ExperimentConfig.

    Returns (kernel_fn, kernel_config).

    Prognostic schemes (mass_flux, edmf) are returned directly and their
    prognostic variable is threaded explicitly through the unified driver.
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
            RH_ref=config.sbm_RH_ref,
            CAPE_threshold=getattr(config, 'sbm_cape_threshold', 70.0),
        )
    else:
        cc = ConvectionConfig(scheme=scheme)
        conv_config = getattr(cc, scheme)

    return conv_fn, conv_config


def _noop_convection(T, q_v, p_full, p_half, dt, config):
    """No-op convection kernel returning zeros."""
    from legoesm.atmosphere.physics.convection.output import ConvectionOutput
    ncol, nlev = T.shape
    z2 = jnp.zeros_like(T)
    z1 = jnp.zeros((ncol,), dtype=T.dtype)
    return ConvectionOutput(
        dT_dt=z2, dq_v_dt=z2, precipitation=z1, cape=z1, convective_mask=z1,
    )


# ---------------------------------------------------------------------------
# Microphysics resolver
# ---------------------------------------------------------------------------

def _resolve_microphysics(config):
    """Resolve microphysics kernel and config from ExperimentConfig.

    Returns (kernel_fn, kernel_config) or (None, None) if disabled.
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
    from legoesm.atmosphere.physics.turbulence.integration import _get_turbulence_fn

    tc = TurbulenceConfig(scheme=scheme)
    _name, turb_fn, turb_config = _get_turbulence_fn(tc)
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
        _get_gwd_fn,
    )

    gc = GravityWaveDragConfig(scheme=scheme)
    _name, gwd_fn, gwd_config = _get_gwd_fn(gc)
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
        # Default to rrtmgp for any non-gray scheme (preserves old behaviour)
        rad_scheme = "rrtmgp"
    radiation_fn = _RADIATION_BUILDERS[rad_scheme](config)

    # Resolve convection via registry
    convection_fn, convection_config = _resolve_convection(config)

    # Resolve microphysics via registry
    micro_fn, micro_config = _resolve_microphysics(config)

    # Resolve turbulence
    turb_fn, turb_config = _resolve_turbulence(config)

    # Resolve gravity wave drag
    gwd_fn, gwd_config = _resolve_gwd(config)

    # Resolve optional joint ML physics parameterization
    physics_parameterization = _resolve_physics_parameterization(
        config,
        nlev=int(sigma.sigma_full.shape[0]),
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
        turbulence_fn=turb_fn,
        turbulence_config=turb_config,
        gwd_fn=gwd_fn,
        gwd_config=gwd_config,
        physics_parameterization=physics_parameterization,
    )
    pipeline._cloud_scheme = getattr(config, 'cloud_scheme', 'none')
    return pipeline
