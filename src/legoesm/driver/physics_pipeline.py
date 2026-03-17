"""Physics pipeline for the composable model driver.

Wraps radiation, convection, microphysics, and boundary-layer exchange
into a single callable that replaces the inline physics step in run_amip.py.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.forcing.surface_utils import blend_surface_temperature


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
    exchange into a single JIT-compiled function. Manages radiation
    sub-cycling (held tendencies) internally.

    Parameters
    ----------
    sigma_full : jax.Array
        Sigma at full levels, shape (nlev,).
    sigma_half : jax.Array
        Sigma at half levels, shape (nlev+1,).
    dsigma : jax.Array
        Layer thickness in sigma, shape (nlev,).
    sbm_config : SBMConfig
        Convection configuration.
    radiation_fn : callable
        JIT-compiled radiation function.
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
        sigma_full,
        sigma_half,
        dsigma,
        sbm_config,
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
    ):
        self.sigma_full = sigma_full
        self.sigma_half = sigma_half
        self.dsigma = dsigma
        self.sbm_config = sbm_config
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

    def physics_step_no_rad(self, T, p_s, q_v, q_c, q_r, u, v, sst, sic,
                            lat, dt, dT_dt_rad, sw_net_sfc, lw_net_sfc,
                            sw_up_toa, lw_up_toa, sw_down_toa):
        """Convection + microphysics + BL exchange with held radiation."""
        nlev = self.sigma_full.shape[0]
        shape_3d = T.shape
        shape_2d = p_s.shape
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]

        T_sfc = blend_surface_temperature(sst, sic, self.T_ice)

        p_full = p_s[..., None] * self.sigma_full
        p_half = p_s[..., None] * self.sigma_half

        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        q_v_col = q_v.reshape(ncol, nlev)

        conv_out = sbm_convection(
            T=T_col, q_v=q_v_col, p_full=p_full_col, p_half=p_half_col,
            dt=dt, config=self.sbm_config,
        )
        dT_dt_conv = conv_out.dT_dt.reshape(shape_3d)
        dq_v_dt_conv = conv_out.dq_v_dt.reshape(shape_3d)
        precip = conv_out.precipitation.reshape(shape_2d)

        # Microphysics
        dT_dt_micro = jnp.zeros(shape_3d)
        dq_v_dt_micro = jnp.zeros(shape_3d)
        dq_c_dt = jnp.zeros(shape_3d)
        dq_r_dt = jnp.zeros(shape_3d)
        precip_micro = jnp.zeros(shape_2d)

        if self.micro_fn is not None:
            from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
            q_c_col = q_c.reshape(ncol, nlev)
            q_r_col = q_r.reshape(ncol, nlev)
            rho_col = p_full_col / (constants.R_d * T_col)
            dp_col = p_half_col[:, 1:] - p_half_col[:, :-1]
            dz_col = dp_col / (rho_col * constants.g)
            hydrometeors = HydrometeorState(
                q_c=q_c_col, q_r=q_r_col,
                q_i=jnp.zeros_like(q_c_col), q_s=jnp.zeros_like(q_c_col),
                q_g=jnp.zeros_like(q_c_col), N_c=jnp.zeros_like(q_c_col),
                N_r=jnp.zeros_like(q_c_col), N_i=jnp.zeros_like(q_c_col),
            )
            micro_out = self.micro_fn(
                T=T_col, q_v=q_v_col, hydrometeors=hydrometeors,
                p_full=p_full_col, p_half=p_half_col,
                rho=rho_col, dz=dz_col, dt=dt,
                config=self.micro_config,
            )
            dT_dt_micro = micro_out.dT_dt.reshape(shape_3d)
            dq_v_dt_micro = micro_out.dq_v_dt.reshape(shape_3d)
            dq_c_dt = micro_out.dq_c_dt.reshape(shape_3d)
            dq_r_dt = micro_out.dq_r_dt.reshape(shape_3d)
            precip_micro = micro_out.precipitation.reshape(shape_2d)

        # Boundary layer exchange
        rho_low = (p_s * self.sigma_full[-1]) / (constants.R_d * T[..., -1])
        wind_speed = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1.0)
        dp_low = p_s * (self.sigma_half[-1] - self.sigma_half[-2])

        shflx = rho_low * constants.c_pd * self.C_H * wind_speed * (T_sfc - T[..., -1])
        q_sat_sfc = saturation_mixing_ratio(T_sfc, p_s)
        lhflx = rho_low * constants.L_v * self.C_E * wind_speed * (q_sat_sfc - q_v[..., -1])
        evap_rate = lhflx / constants.L_v

        dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
        dq_BL = constants.g * evap_rate / dp_low

        dT_dt = dT_dt_rad + dT_dt_conv + dT_dt_micro
        dT_dt = dT_dt.at[..., -1].add(dT_BL)

        dq_v_dt = dq_v_dt_conv + dq_v_dt_micro
        dq_v_dt = dq_v_dt.at[..., -1].add(dq_BL)

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
        )

    def compute_radiation_core(self, T, p_s, q_v, sst, sic, lat, lon,
                               day_of_year, seconds_of_day,
                               solar_weights, s_0,
                               o3_vmr_precomputed, aerosol_od_precomputed):
        """Compute radiation tendencies and fluxes (pure JAX, no I/O).

        Returns (dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa,
                 sw_down_toa) as a 6-tuple.
        """
        from legoesm.forcing.surface_utils import blend_surface_property

        nlev = self.sigma_full.shape[0]
        shape_3d = T.shape
        shape_2d = p_s.shape
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]

        T_sfc = blend_surface_temperature(sst, sic, self.T_ice)
        albedo = blend_surface_property(sic, self.albedo_ice, self.albedo_ocean)
        emissivity = blend_surface_property(sic, self.emissivity_ice, self.emissivity_ocean)

        p_full = p_s[..., None] * self.sigma_full
        p_half = p_s[..., None] * self.sigma_half

        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        q_v_col = q_v.reshape(ncol, nlev)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        lon_col = lon.reshape(ncol)
        albedo_col = albedo.reshape(ncol)
        emis_col = emissivity.reshape(ncol)

        rad_out = self.radiation_fn(
            T_col, p_full_col, p_half_col, q_v_col,
            T_sfc_col, lat_col, lon_col,
            day_of_year, seconds_of_day,
            albedo_col, emis_col,
            o3_vmr_precomputed, aerosol_od_precomputed,
            solar_weights, s_0,
        )

        dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)
        sw_down_sfc = rad_out.sw_flux_down[:, -1].reshape(shape_2d)
        sw_net_sfc = sw_down_sfc * (1.0 - albedo)
        lw_net_sfc = (rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]).reshape(shape_2d)
        sw_up_toa = rad_out.sw_flux_up[:, 0].reshape(shape_2d)
        lw_up_toa = rad_out.lw_flux_up[:, 0].reshape(shape_2d)
        sw_down_toa = rad_out.sw_flux_down[:, 0].reshape(shape_2d)

        return (dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa,
                sw_down_toa)

    def build_step_unified(self):
        """Build a JIT-compiled unified physics step with radiation sub-cycling.

        Returns a function ``step_unified(need_rad, T, p_s, q_v, q_c, q_r,
        u, v, sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
        solar_weights, s_0, o3_vmr, aerosol_od, held) -> (PhysicsOutput,
        HeldRadiation)``.
        """
        pipeline = self

        @jax.jit
        def step_unified(need_rad, T, p_s, q_v, q_c, q_r, u, v,
                         sst, sic, lat, lon,
                         day_of_year, seconds_of_day, dt,
                         solar_weights, s_0,
                         o3_vmr, aerosol_od,
                         held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                         held_sw_up_toa, held_lw_up_toa, held_sw_down_toa):

            def _rad_branch(args):
                (T, p_s, q_v, q_c, q_r, u, v, sst, sic, lat, lon,
                 day_of_year, seconds_of_day, dt,
                 solar_weights, s_0, o3_vmr, aerosol_od,
                 held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                 held_sw_up_toa, held_lw_up_toa, held_sw_down_toa) = args

                (dT_dt_rad, sw_net_sfc, lw_net_sfc,
                 sw_up_toa, lw_up_toa, sw_down_toa) = \
                    pipeline.compute_radiation_core(
                        T, p_s, q_v, sst, sic, lat, lon,
                        day_of_year, seconds_of_day,
                        solar_weights, s_0, o3_vmr, aerosol_od,
                    )

                physics_out = pipeline.physics_step_no_rad(
                    T, p_s, q_v, q_c, q_r, u, v, sst, sic, lat, dt,
                    dT_dt_rad, sw_net_sfc, lw_net_sfc,
                    sw_up_toa, lw_up_toa, sw_down_toa,
                )

                new_held = (dT_dt_rad, sw_net_sfc, lw_net_sfc,
                            sw_up_toa, lw_up_toa, sw_down_toa)
                return physics_out, new_held

            def _no_rad_branch(args):
                (T, p_s, q_v, q_c, q_r, u, v, sst, sic, lat, lon,
                 day_of_year, seconds_of_day, dt,
                 solar_weights, s_0, o3_vmr, aerosol_od,
                 held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                 held_sw_up_toa, held_lw_up_toa, held_sw_down_toa) = args

                physics_out = pipeline.physics_step_no_rad(
                    T, p_s, q_v, q_c, q_r, u, v, sst, sic, lat, dt,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                )

                new_held = (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                            held_sw_up_toa, held_lw_up_toa, held_sw_down_toa)
                return physics_out, new_held

            args = (T, p_s, q_v, q_c, q_r, u, v, sst, sic, lat, lon,
                    day_of_year, seconds_of_day, dt,
                    solar_weights, s_0, o3_vmr, aerosol_od,
                    held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                    held_sw_up_toa, held_lw_up_toa, held_sw_down_toa)

            return jax.lax.cond(need_rad, _rad_branch, _no_rad_branch, args)

        return step_unified


def build_physics_pipeline(grid, sigma, config):
    """Build a PhysicsPipeline from an ExperimentConfig.

    Parameters
    ----------
    grid : CubedSphereGrid or similar
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : ExperimentConfig

    Returns
    -------
    PhysicsPipeline
    """
    from legoesm.atmosphere.physics.convection.config import SBMConfig
    from legoesm.atmosphere.physics.radiation.config import (
        GrayRadiationConfig, RRTMGPConfig,
    )
    from legoesm.atmosphere.physics.radiation.gray import gray_radiation
    from legoesm.atmosphere.physics.radiation.solar import (
        cos_zenith_angle, daily_mean_insolation,
    )

    sbm_config = SBMConfig(tau_c=config.sbm_tau_c, RH_ref=config.sbm_RH_ref)

    diurnal = config.diurnal_cycle
    S_0 = config.S_0

    if config.radiation == "gray":
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
                         solar_weights, s_0=S_0):
            if diurnal:
                hour = seconds_of_day / 3600.0
                cos_sza = cos_zenith_angle(lat_col, lon_col, day_of_year, hour)
                insol = s_0 * jnp.maximum(cos_sza, 0.0)
            else:
                insol = daily_mean_insolation(lat_col, day_of_year, s_0)
            return gray_radiation(
                T=T_col, p_full=p_full_col, p_half=p_half_col,
                sfc_temperature=T_sfc_col, lat=lat_col,
                q_v=q_v_col, insolation=insol, config=gray_config,
            )
    else:
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
        rrtmg_config = RRTMGPConfig(
            co2_ppmv=config.co2_ppmv,
            ch4_ppbv=config.ch4_ppbv,
            n2o_ppbv=config.n2o_ppbv,
            sfc_emissivity=config.sfc_emissivity,
            sfc_albedo=config.albedo_ocean,
            S_0=S_0,
            use_scan=False,
            include_clouds=False,
        )

        @jax.jit
        def radiation_fn(T_col, p_full_col, p_half_col, q_v_col, T_sfc_col,
                         lat_col, lon_col, day_of_year, seconds_of_day,
                         albedo_col, emis_col, o3_vmr_col, aerosol_od_col,
                         solar_weights, s_0=S_0,
                         co2_vmr=None, ch4_vmr=None, n2o_vmr=None):
            if diurnal:
                hour = seconds_of_day / 3600.0
                cos_sza = cos_zenith_angle(lat_col, lon_col, day_of_year, hour)
                cos_zenith = jnp.maximum(cos_sza, 0.0)
            else:
                insol = daily_mean_insolation(lat_col, day_of_year, s_0)
                cos_zenith = jnp.clip(insol / jnp.clip(s_0, 1e-6, None), 0.0, 1.0)
            return rrtmgp_radiation(
                T=T_col, p_full=p_full_col, p_half=p_half_col,
                sfc_temperature=T_sfc_col, q_v=q_v_col,
                cos_zenith=cos_zenith, config=rrtmg_config,
                sfc_albedo_override=albedo_col,
                sfc_emissivity_override=emis_col,
                o3_vmr=o3_vmr_col,
                aerosol_optical_depth=aerosol_od_col,
                solar_spectral_fraction=solar_weights if solar_weights.size > 0 else None,
            )

    # Microphysics backend
    micro_fn = None
    micro_config = None
    if config.microphysics != "none":
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
        mc = MicrophysicsConfig(scheme=config.microphysics)
        _backends = {}
        if config.microphysics == "kessler":
            from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
            _backends["kessler"] = (kessler_microphysics, mc.kessler)
        elif config.microphysics == "sundqvist":
            from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
            _backends["sundqvist"] = (sundqvist_microphysics, mc.sundqvist)
        elif config.microphysics == "seifert_beheng":
            from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
            _backends["seifert_beheng"] = (seifert_beheng_microphysics, mc.seifert_beheng)
        elif config.microphysics == "morrison":
            from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
            _backends["morrison"] = (morrison_microphysics, mc.morrison)
        elif config.microphysics == "thompson":
            from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
            _backends["thompson"] = (thompson_microphysics, mc.thompson)
        if config.microphysics in _backends:
            micro_fn, micro_config = _backends[config.microphysics]

    return PhysicsPipeline(
        sigma_full=sigma.sigma_full,
        sigma_half=sigma.sigma_half,
        dsigma=sigma.dsigma,
        sbm_config=sbm_config,
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
    )
