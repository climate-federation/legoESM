"""Shrink-only baseline for the --params reachability audit (issue #691).

Every tier-1/2 ``__param_spec__`` parameter here is NOT settable via the
``--params`` qualified-name loader in any of its component's run drivers.  The
audit test asserts the computed uncovered set EQUALS this baseline; shrink it as
parameters are wired through (never grow silently).  Idealized gray-radiation
params are a documented conscious exclusion (--config-only).

2026-08-04: re-shrunk 415 -> 106 after the main sync reverted the atm
post-setup class router (``apply_params_to_pipeline``). The route was
re-applied onto main's version of run_config_yaml.py; the numbers below are
recomputed, not restored from the pre-sync file.

2026-08-10 (origin/main merge): RECOMPUTED against the merged tree, 104
entries.  The atm post-setup class router survives the merge, so every
atm.conv / atm.gwd / atm.micro / atm.turb entry main still carried is
reachable here and was dropped; main's new PrognosticAerosolConfig quartet is
genuinely unreachable and was added.  Not a hand edit — the set below equals
``test_params_reachability_audit._compute_uncovered()`` on the merge result.
"""

UNREACHABLE_PARAMS = frozenset({
    # atm.aerosol: PrognosticAerosolConfig (4) — new on main at the 2026-08-10
    # sync; no ExperimentConfig scalar routes them, so they are --config-only.
    'atm.aerosol.PrognosticAerosolConfig.dry_dep_velocity_m_s',
    'atm.aerosol.PrognosticAerosolConfig.emission_number_flux_m2_s',
    'atm.aerosol.PrognosticAerosolConfig.so2_oxidation_timescale_s',
    'atm.aerosol.PrognosticAerosolConfig.wet_scavenging_coeff_m2_kg',
    # atm: AhmedNeelinDCAConfig (7)
    # atm: BechtoldConfig — M_b_max wired 2026-07-10 (#869 campaign lever)
    # RCAPQADV blend weight (2026-07-17 ZDQCV closure work): same conscious
    # exclusion as the rest of the BechtoldConfig family (CLI/ExperimentConfig
    # scalars, not the --params qualified-name loader).
    # downdraft_alpha / downdraft_entrain_rate were REMOVED 2026-07-23: their
    # convention-named ExperimentConfig scalars (bechtold_downdraft_alpha /
    # bechtold_downdraft_entrain_rate) are threaded unconditionally by
    # _resolve_convection, so they are now in _ATM_SCALAR_PARAM_MAP
    # (--params-reachable).  downdraft_evap_efficiency stays CLI-only: its
    # scalar (bechtold_downdraft_evap) is NOT convention-named, so the map's
    # verified-threading contract does not cover it.
    # atm: CCNFromAODConfig (2)
    'atm.aerosol.CCNFromAODConfig.aot_coeff',
    'atm.aerosol.CCNFromAODConfig.aot_exponent',
    # atm: CLUBBLiteConfig (4)
    # atm: CLUBBParams (48) — like every other turbulence scheme, full CLUBB is
    # calibrated via the SCM-RCE campaign / LES tuning collector, not the atm
    # run_amip --params scalar map (build_atm_scalar_param_map exposes 0
    # atm.turb.* params). Driver-level reachability (nested CLUBBConfig.params
    # descent in build_atm_scalar_param_map) is tracked as a follow-up.
    # atm: CloudConfig (6)
    # (Nc_default + conv_cloud_coeff wired 2026-08-02: flat ExperimentConfig
    #  scalars cloud_Nc_default / cloud_conv_cloud_coeff, threaded by
    #  build_cloud_config on both the FV pipeline and the MPAS standalone lane.)
    'atm.clouds.CloudConfig.T_ice_only',
    'atm.clouds.CloudConfig.conv_precip_scale',
    'atm.clouds.CloudConfig.gamma_xr',
    'atm.clouds.CloudConfig.q_cloud_resolved_ref',
    'atm.clouds.CloudConfig.r_eff_ice',
    'atm.clouds.CloudConfig.r_eff_liq',
    # atm.pblh: PBLHeightConfig (1)
    'atm.pblh.PBLHeightConfig.Ri_crit',
    # atm.rad: GrayRadiationConfig (9)
    'atm.rad.GrayRadiationConfig.linear_frac',
    'atm.rad.GrayRadiationConfig.lw_diff_factor',
    'atm.rad.GrayRadiationConfig.obliquity',
    'atm.rad.GrayRadiationConfig.sfc_albedo',
    'atm.rad.GrayRadiationConfig.sw_exponent',
    'atm.rad.GrayRadiationConfig.sw_tau_0',
    'atm.rad.GrayRadiationConfig.tau_equator',
    'atm.rad.GrayRadiationConfig.tau_moist_coeff',
    'atm.rad.GrayRadiationConfig.tau_pole',
    # atm: HinesConfig (0) — Fmax + total_rms_wind became reachable when
    # gwd_config_for started threading the hines_* ExperimentConfig scalars
    # into the kernel leaf on every lane (2026-07-24).
    # atm: HoltslagBovilleConfig (13)
    # atm: KainFritschConfig (12)
    # atm: KesslerConfig (5)
    # The hard-saturation-adjustment trigger + heating cap (all warm-rain
    # micro configs) were REMOVED from this baseline 2026-07-23: they are now
    # routed onto ExperimentConfig flat scalars (hard_sat_adjust_threshold /
    # hard_sat_max_heating_K) and reachable via --params through
    # _ATM_SCALAR_PARAM_MAP (the day-137 summer-regime tuning need).  The
    # remaining sibling micro params stay SCM-RCE-only (pipeline-internal).
    # atm: KuoConfig (2)
    # atm: LindzenConfig (2)
    # atm: LouisConfig (1) — l_mix_max / Ri_crit / b_louis / c_louis / d_louis
    # became reachable 2026-08-02: d8268e6fa made turbulence_config_for thread
    # the flat louis_* scalars into the active louis sub-config, and they now
    # carry --louis-* CLI flags plus _ATM_SCALAR_PARAM_MAP entries.
    # b_heat_ratio stays unreachable: it has no flat ExperimentConfig scalar.
    # atm: MYNN25Config (9)
    # atm: MassFluxConfig (6)
    # atm: McFarlaneConfig (6) — directional_spread became reachable via the
    # mcfarlane_directional_spread scalar + gwd_config_for (2026-07-24); the
    # rest still have no ExperimentConfig scalar to route through.
    # atm: MorrisonConfig (15; 5 wired 2026-07-26 + ice_snow_d_auto wired
    # 2026-07-27 via thread_morrison_scalars — anvil-ice tuning)
    # atm: OzoneProfileConfig (3)
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
    # ocean.bottom_drag: linear (1)
    'ocean.bottom_drag.linear.r',
    # ocean.bottom_drag: quadratic (1)
    'ocean.bottom_drag.quadratic.C_d',
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
