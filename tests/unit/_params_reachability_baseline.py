"""Shrink-only baseline for the --params reachability audit (issue #691).

Every tier-1/2 ``__param_spec__`` parameter here is NOT settable via the
``--params`` qualified-name loader in any of its component's run drivers.  The
audit test asserts the computed uncovered set EQUALS this baseline; shrink it as
parameters are wired through (never grow silently).  The gray-radiation entries
left on 2026-08-11: gray is no longer trained at all, so its params moved to
tunable_tier 0 and the audit (which only sees tier 1-2) stopped listing them.

2026-08-04: re-shrunk 415 -> 106 after the main sync reverted the atm
post-setup class router (``apply_params_to_pipeline``). The route was
re-applied onto main's version of run_config_yaml.py; the numbers below are
recomputed, not restored from the pre-sync file.
"""

UNREACHABLE_PARAMS = frozenset({
    # atm.aerosol: CCNFromAODConfig (2)
    'atm.aerosol.CCNFromAODConfig.aot_coeff',
    'atm.aerosol.CCNFromAODConfig.aot_exponent',
    # atm.aerosol: PrognosticAerosolConfig (4)
    'atm.aerosol.PrognosticAerosolConfig.dry_dep_velocity_m_s',
    'atm.aerosol.PrognosticAerosolConfig.emission_number_flux_m2_s',
    'atm.aerosol.PrognosticAerosolConfig.so2_oxidation_timescale_s',
    'atm.aerosol.PrognosticAerosolConfig.wet_scavenging_coeff_m2_kg',
    # atm.clouds: CloudConfig (8)
    'atm.clouds.CloudConfig.Nc_default',
    'atm.clouds.CloudConfig.T_ice_only',
    'atm.clouds.CloudConfig.conv_cloud_coeff',
    'atm.clouds.CloudConfig.conv_precip_scale',
    'atm.clouds.CloudConfig.gamma_xr',
    'atm.clouds.CloudConfig.q_cloud_resolved_ref',
    'atm.clouds.CloudConfig.r_eff_ice',
    'atm.clouds.CloudConfig.r_eff_liq',
    # atm.pblh: PBLHeightConfig (1)
    'atm.pblh.PBLHeightConfig.Ri_crit',
    # atm.rad: GrayRadiationConfig — none: every gray param is tier 0
    # (never trainable) since 2026-08-11.
    # atm.rad: OzoneProfileConfig (3)
    'atm.rad.OzoneProfileConfig.o3_max_vmr',
    'atm.rad.OzoneProfileConfig.p_peak_hPa',
    'atm.rad.OzoneProfileConfig.sigma_logp',
    # land.canopy: interception (3)
    'land.canopy.interception.dewmx',
    'land.canopy.interception.fwet_exponent',
    'land.canopy.interception.maximum_leaf_wetted_fraction',
    # land.canopy: sif (8)
    'land.canopy.sif.escape_probability',
    'land.canopy.sif.kd',
    'land.canopy.sif.kf',
    'land.canopy.sif.kn0',
    'land.canopy.sif.kn_beta',
    'land.canopy.sif.kn_gamma',
    'land.canopy.sif.kp',
    'land.canopy.sif.max_electron_yield',
    # land.d13c: phi_c4_leakiness (1)
    'land.d13c.phi_c4_leakiness',
    # land.land_use_change: clear_burn_frac (1)
    'land.land_use_change.clear_burn_frac',
    # land.land_use_change: clear_slash_frac (1)
    'land.land_use_change.clear_slash_frac',
    # land.land_use_change: prod_frac_100yr (1)
    'land.land_use_change.prod_frac_100yr',
    # land.land_use_change: prod_frac_10yr (1)
    'land.land_use_change.prod_frac_10yr',
    # land.land_use_change: prod_frac_1yr (1)
    'land.land_use_change.prod_frac_1yr',
    # land.land_use_change: tau_100yr_years (1)
    'land.land_use_change.tau_100yr_years',
    # land.land_use_change: tau_10yr_years (1)
    'land.land_use_change.tau_10yr_years',
    # land.land_use_change: tau_1yr_years (1)
    'land.land_use_change.tau_1yr_years',
    # land.land_use_change: tau_regrow_years (1)
    'land.land_use_change.tau_regrow_years',
    # land.snow_bands: alpha_glacier_ice (1)
    'land.snow_bands.alpha_glacier_ice',
    # land.snow_bands: blow_snow_wind_thresh_ms (1)
    'land.snow_bands.blow_snow_wind_thresh_ms',
    # land.snow_bands: lapse_rate_K_m (1)
    'land.snow_bands.lapse_rate_K_m',
    # land.snow_bands: lw_elev_lapse_W_m2_per_m (1)
    'land.snow_bands.lw_elev_lapse_W_m2_per_m',
    # land.snow_bands: sw_elev_grad_per_m (1)
    'land.snow_bands.sw_elev_grad_per_m',
    # ocean.backscatter: E_max (1)
    'ocean.backscatter.E_max',
    # ocean.backscatter: c_bs (1)
    'ocean.backscatter.c_bs',
    # ocean.backscatter: efficiency (1)
    'ocean.backscatter.efficiency',
    # ocean.backscatter: tau_relax_days (1)
    'ocean.backscatter.tau_relax_days',
    # ocean.conv: enhanced_diffusion (2)
    'ocean.conv.enhanced_diffusion.K_bg',
    'ocean.conv.enhanced_diffusion.K_conv',
    # ocean.conv: plume (3)
    'ocean.conv.plume.T_excess',
    'ocean.conv.plume.alpha_plume',
    'ocean.conv.plume.epsilon',
    # ocean.eke: eke (7)
    'ocean.eke.eke.alpha_eke',
    'ocean.eke.eke.c_eps',
    'ocean.eke.eke.c_k',
    'ocean.eke.eke.eke_crhin',
    'ocean.eke.eke.eke_cross',
    'ocean.eke.eke.k_iso',
    'ocean.eke.eke.kappa_gm_max',
    # ocean.eke: geometric (13)
    'ocean.eke.geometric.alpha',
    'ocean.eke.geometric.c_eps_geometric',
    'ocean.eke.geometric.gamma_n',
    'ocean.eke.geometric.kappa_e',
    'ocean.eke.geometric.kappa_gm_max',
    'ocean.eke.geometric.kappa_gm_min',
    'ocean.eke.geometric.kappa_n_max',
    'ocean.eke.geometric.kappa_n_min',
    'ocean.eke.geometric.kappa_u',
    'ocean.eke.geometric.l_mix_max',
    'ocean.eke.geometric.r_d_max',
    'ocean.eke.geometric.r_d_min',
    'ocean.eke.geometric.rossby_factor',
    # ocean.ice_shelf: gamma_S (1)
    'ocean.ice_shelf.gamma_S',
    # ocean.ice_shelf: gamma_T (1)
    'ocean.ice_shelf.gamma_T',
    # ocean.ice_shelf: melt_factor_linear (1)
    'ocean.ice_shelf.melt_factor_linear',
    # ocean.lat: harmonic (2)
    'ocean.lat.harmonic.A_h',
    'ocean.lat.harmonic.K_h',
    # ocean.lat: mle (1)
    'ocean.lat.mle.ce',
    # ocean.sf: bulk (6)
    'ocean.sf.bulk.C_D',
    'ocean.sf.bulk.C_E',
    'ocean.sf.bulk.C_H',
    'ocean.sf.bulk.emissivity',
    'ocean.sf.bulk.q_sat_salinity_factor',
    'ocean.sf.bulk.z0',
    # ocean.sf: flux_feedback (1)
    'ocean.sf.flux_feedback.tau_restore_s',
    # ocean.sf: prescribed (2)
    'ocean.sf.prescribed.tau_max',
    'ocean.sf.prescribed.tropical_wind_scale',
    # ocean.sf: restoring (2)
    'ocean.sf.restoring.tau_S',
    'ocean.sf.restoring.tau_T',
    # ocean.sw_penetration: rgb_ir_extinction_m (1)
    'ocean.sw_penetration.rgb_ir_extinction_m',
    # ocean.sw_penetration: rgb_ir_fraction (1)
    'ocean.sw_penetration.rgb_ir_fraction',
    # ocean.tidal_forcing: amplitude_scale (1)
    'ocean.tidal_forcing.amplitude_scale',
    # ocean.tidal_forcing: beta_sal (1)
    'ocean.tidal_forcing.beta_sal',
    # ocean.tidal_forcing: love_factor (1)
    'ocean.tidal_forcing.love_factor',
})
