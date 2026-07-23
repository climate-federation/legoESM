"""Shrink-only baseline for the --params reachability audit (issue #691).

Every tier-1/2 ``__param_spec__`` parameter here is NOT settable via the
``--params`` qualified-name loader in any of its component's run drivers.  The
audit test asserts the computed uncovered set EQUALS this baseline; shrink it as
parameters are wired through (never grow silently).  Idealized gray-radiation
params are a documented conscious exclusion (--config-only).
"""

UNREACHABLE_PARAMS = frozenset({
    # atm: AhmedNeelinDCAConfig (7)
    'atm.conv.AhmedNeelinDCAConfig.a_mm_per_hr',
    'atm.conv.AhmedNeelinDCAConfig.b_c',
    'atm.conv.AhmedNeelinDCAConfig.p_bl_top_pa',
    'atm.conv.AhmedNeelinDCAConfig.p_lft_top_pa',
    'atm.conv.AhmedNeelinDCAConfig.tau_adjust_s',
    'atm.conv.AhmedNeelinDCAConfig.w_b',
    'atm.conv.AhmedNeelinDCAConfig.w_l',
    # atm: BechtoldConfig (14)
    'atm.conv.BechtoldConfig.M_b_max',
    'atm.conv.BechtoldConfig.cape_pbl_depth',
    # RCAPQADV blend weight (2026-07-17 ZDQCV closure work): same conscious
    # exclusion as the rest of the BechtoldConfig family (CLI/ExperimentConfig
    # scalars, not the --params qualified-name loader).
    'atm.conv.BechtoldConfig.cape_qadv_weight',
    'atm.conv.BechtoldConfig.cloud_depth_deep',
    'atm.conv.BechtoldConfig.cloud_depth_shallow_max',
    'atm.conv.BechtoldConfig.delta_deep',
    'atm.conv.BechtoldConfig.delta_midlevel',
    'atm.conv.BechtoldConfig.delta_shallow',
    'atm.conv.BechtoldConfig.downdraft_alpha',
    # CLI-only (--bechtold-downdraft-entrain-rate), like its siblings
    # downdraft_alpha / downdraft_evap_efficiency: the whole penetrative-
    # downdraft family is exposed as ExperimentConfig scalars + CLI flags, not
    # via the --params qualified-name loader.  Conscious exclusion.
    'atm.conv.BechtoldConfig.downdraft_entrain_rate',
    'atm.conv.BechtoldConfig.downdraft_evap_efficiency',
    'atm.conv.BechtoldConfig.stochastic_amplitude',
    'atm.conv.BechtoldConfig.stochastic_decorrelation',
    'atm.conv.BechtoldConfig.tau_M_u_relax',
    'atm.conv.BechtoldConfig.tau_bl',
    # atm: CCNFromAODConfig (2)
    'atm.aerosol.CCNFromAODConfig.aot_coeff',
    'atm.aerosol.CCNFromAODConfig.aot_exponent',
    # atm: CLUBBLiteConfig (4)
    'atm.turb.CLUBBLiteConfig.C_K',
    'atm.turb.CLUBBLiteConfig.C_eps',
    'atm.turb.CLUBBLiteConfig.Pr_t',
    'atm.turb.CLUBBLiteConfig.l_mix_max',
    # atm: CLUBBParams (48) — like every other turbulence scheme, full CLUBB is
    # calibrated via the SCM-RCE campaign / LES tuning collector, not the atm
    # run_amip --params scalar map (build_atm_scalar_param_map exposes 0
    # atm.turb.* params). Driver-level reachability (nested CLUBBConfig.params
    # descent in build_atm_scalar_param_map) is tracked as a follow-up.
    'atm.turb.CLUBBParams.C1',
    'atm.turb.CLUBBParams.C10',
    'atm.turb.CLUBBParams.C11',
    'atm.turb.CLUBBParams.C12',
    'atm.turb.CLUBBParams.C14',
    'atm.turb.CLUBBParams.C2rt',
    'atm.turb.CLUBBParams.C2rtthl',
    'atm.turb.CLUBBParams.C2thl',
    'atm.turb.CLUBBParams.C4',
    'atm.turb.CLUBBParams.C6rt',
    'atm.turb.CLUBBParams.C6rtb',
    'atm.turb.CLUBBParams.C6thl',
    'atm.turb.CLUBBParams.C6thlb',
    'atm.turb.CLUBBParams.C7',
    'atm.turb.CLUBBParams.C8',
    'atm.turb.CLUBBParams.C_invrs_tau_N2',
    'atm.turb.CLUBBParams.C_invrs_tau_N2_wp2',
    'atm.turb.CLUBBParams.C_invrs_tau_N2_xp2',
    'atm.turb.CLUBBParams.C_invrs_tau_bkgnd',
    'atm.turb.CLUBBParams.C_invrs_tau_sfc',
    'atm.turb.CLUBBParams.C_invrs_tau_shear',
    'atm.turb.CLUBBParams.C_uu_buoy',
    'atm.turb.CLUBBParams.C_uu_shr',
    'atm.turb.CLUBBParams.C_wp3_pr_turb',
    'atm.turb.CLUBBParams.Lscale_mu_coef',
    'atm.turb.CLUBBParams.beta',
    'atm.turb.CLUBBParams.c_K',
    'atm.turb.CLUBBParams.c_K1',
    'atm.turb.CLUBBParams.c_K10',
    'atm.turb.CLUBBParams.c_K10h',
    'atm.turb.CLUBBParams.c_K2',
    'atm.turb.CLUBBParams.c_K6',
    'atm.turb.CLUBBParams.c_K8',
    'atm.turb.CLUBBParams.c_K9',
    'atm.turb.CLUBBParams.coef_spread_DG_means_rt',
    'atm.turb.CLUBBParams.coef_spread_DG_means_thl',
    'atm.turb.CLUBBParams.gamma_coef',
    'atm.turb.CLUBBParams.gamma_coefb',
    'atm.turb.CLUBBParams.lambda0_stability_coef',
    'atm.turb.CLUBBParams.lmin_coef',
    'atm.turb.CLUBBParams.mu',
    'atm.turb.CLUBBParams.mult_coef',
    'atm.turb.CLUBBParams.nu1',
    'atm.turb.CLUBBParams.nu2',
    'atm.turb.CLUBBParams.nu6',
    'atm.turb.CLUBBParams.nu8',
    'atm.turb.CLUBBParams.nu9',
    'atm.turb.CLUBBParams.slope_coef_spread_DG_means_w',
    # atm: CloudConfig (8)
    'atm.clouds.CloudConfig.Nc_default',
    'atm.clouds.CloudConfig.T_ice_only',
    'atm.clouds.CloudConfig.conv_cloud_coeff',
    'atm.clouds.CloudConfig.conv_precip_scale',
    'atm.clouds.CloudConfig.gamma_xr',
    'atm.clouds.CloudConfig.q_cloud_resolved_ref',
    'atm.clouds.CloudConfig.r_eff_ice',
    'atm.clouds.CloudConfig.r_eff_liq',
    # atm: ConvectiveEDMFConfig (6)
    'atm.conv.ConvectiveEDMFConfig.M_b_max',
    'atm.conv.ConvectiveEDMFConfig.a_u_init',
    'atm.conv.ConvectiveEDMFConfig.cape_activation_scale',
    'atm.conv.ConvectiveEDMFConfig.cape_threshold',
    'atm.conv.ConvectiveEDMFConfig.delta_0',
    'atm.conv.ConvectiveEDMFConfig.tau_a',
    # atm: DCAConfig (1)
    'atm.conv.DCAConfig.cape_threshold',
    # atm: E3SMBeresConfig (6)
    'atm.gwd.E3SMBeresConfig.al',
    'atm.gwd.E3SMBeresConfig.cf',
    'atm.gwd.E3SMBeresConfig.hdepth_min_km',
    'atm.gwd.E3SMBeresConfig.hdepth_scaling_factor',
    'atm.gwd.E3SMBeresConfig.mfcc_c0',
    'atm.gwd.E3SMBeresConfig.mfcc_peak',
    # atm: E3SMCAMConfig (5)
    'atm.gwd.E3SMCAMConfig.dback',
    'atm.gwd.E3SMCAMConfig.effgw',
    'atm.gwd.E3SMCAMConfig.fcrit2',
    'atm.gwd.E3SMCAMConfig.prndl',
    'atm.gwd.E3SMCAMConfig.umcfac',
    # atm: E3SMFrontalConfig (3)
    'atm.gwd.E3SMFrontalConfig.c0',
    'atm.gwd.E3SMFrontalConfig.frontgfc',
    'atm.gwd.E3SMFrontalConfig.taubgnd',
    # atm: E3SMOrographicConfig (2)
    'atm.gwd.E3SMOrographicConfig.oro_min_h',
    'atm.gwd.E3SMOrographicConfig.oro_min_wind',
    # atm: EmanuelConfig (12)
    'atm.conv.EmanuelConfig.M_b_max',
    'atm.conv.EmanuelConfig.alpha_closure',
    'atm.conv.EmanuelConfig.cape_threshold',
    'atm.conv.EmanuelConfig.cu_coefficient',
    'atm.conv.EmanuelConfig.damp_coefficient',
    'atm.conv.EmanuelConfig.delta_0',
    'atm.conv.EmanuelConfig.downdraft_efficiency',
    'atm.conv.EmanuelConfig.elcrit',
    'atm.conv.EmanuelConfig.entp',
    'atm.conv.EmanuelConfig.parcel_perturb_T',
    'atm.conv.EmanuelConfig.precip_efficiency_max',
    'atm.conv.EmanuelConfig.tlcrit',
    # atm: FastSBMConfig (18)
    'atm.fastsbm.FastSBMConfig.aerosol_dry_median',
    'atm.fastsbm.FastSBMConfig.aerosol_geom_std',
    'atm.fastsbm.FastSBMConfig.bigg_a',
    'atm.fastsbm.FastSBMConfig.bigg_b0',
    'atm.fastsbm.FastSBMConfig.ccn_number',
    'atm.fastsbm.FastSBMConfig.cdnc',
    'atm.fastsbm.FastSBMConfig.cloud_geom_std',
    'atm.fastsbm.FastSBMConfig.fall_a_graupel',
    'atm.fastsbm.FastSBMConfig.fall_a_snow',
    'atm.fastsbm.FastSBMConfig.fall_b_graupel',
    'atm.fastsbm.FastSBMConfig.fall_b_snow',
    'atm.fastsbm.FastSBMConfig.golovin_b',
    'atm.fastsbm.FastSBMConfig.ice_aggregation_efficiency',
    'atm.fastsbm.FastSBMConfig.melt_rate_high',
    'atm.fastsbm.FastSBMConfig.melt_rate_mid',
    'atm.fastsbm.FastSBMConfig.rho_graupel',
    'atm.fastsbm.FastSBMConfig.rho_snow',
    'atm.fastsbm.FastSBMConfig.ventilation_max',
    # atm: GrayRadiationConfig (9)
    'atm.rad.GrayRadiationConfig.linear_frac',
    'atm.rad.GrayRadiationConfig.lw_diff_factor',
    'atm.rad.GrayRadiationConfig.obliquity',
    'atm.rad.GrayRadiationConfig.sfc_albedo',
    'atm.rad.GrayRadiationConfig.sw_exponent',
    'atm.rad.GrayRadiationConfig.sw_tau_0',
    'atm.rad.GrayRadiationConfig.tau_equator',
    'atm.rad.GrayRadiationConfig.tau_moist_coeff',
    'atm.rad.GrayRadiationConfig.tau_pole',
    # atm: HinesConfig (2)
    'atm.gwd.HinesConfig.Fmax',
    'atm.gwd.HinesConfig.total_rms_wind',
    # atm: HoltslagBovilleConfig (13)
    'atm.turb.HoltslagBovilleConfig.Ri_crit',
    'atm.turb.HoltslagBovilleConfig.betah',
    'atm.turb.HoltslagBovilleConfig.betam',
    'atm.turb.HoltslagBovilleConfig.betas',
    'atm.turb.HoltslagBovilleConfig.fak',
    'atm.turb.HoltslagBovilleConfig.fakn',
    'atm.turb.HoltslagBovilleConfig.free_ri_stable_c1',
    'atm.turb.HoltslagBovilleConfig.free_ri_stable_c2',
    'atm.turb.HoltslagBovilleConfig.free_ri_unstable_coeff',
    'atm.turb.HoltslagBovilleConfig.ml_free',
    'atm.turb.HoltslagBovilleConfig.pblh_mech_coeff',
    'atm.turb.HoltslagBovilleConfig.pblh_ustar_fac',
    'atm.turb.HoltslagBovilleConfig.sffrac',
    # atm: KainFritschConfig (12)
    'atm.conv.KainFritschConfig.M_b_max',
    'atm.conv.KainFritschConfig.cape_consumption_time',
    'atm.conv.KainFritschConfig.cape_removal_fraction',
    'atm.conv.KainFritschConfig.delta_0',
    'atm.conv.KainFritschConfig.dtlcl_coeff',
    'atm.conv.KainFritschConfig.entrain_const',
    'atm.conv.KainFritschConfig.parcel_perturb_T',
    'atm.conv.KainFritschConfig.rad_max_m',
    'atm.conv.KainFritschConfig.rad_min_m',
    'atm.conv.KainFritschConfig.timec_max_s',
    'atm.conv.KainFritschConfig.timec_min_s',
    'atm.conv.KainFritschConfig.usl_depth_pa',
    # atm: KesslerConfig (7)
    # ``hard_sat_adjust_threshold`` (all warm-rain micro configs below) is the
    # opt-in hard-saturation-adjustment RH trigger: a per-scheme __param_spec__
    # tier-2 float, calibratable via the SCM-RCE training path
    # (build_trainable_params), NOT via the run_amip --params qualified-name
    # loader — exactly like the sibling atm-micro scheme params in this file
    # (the atm micro *Config is pipeline-internal, not routed onto
    # ExperimentConfig).  Conscious addition (2026-07-23).
    'atm.micro.KesslerConfig.accretion_coeff',
    'atm.micro.KesslerConfig.autoconversion_rate',
    'atm.micro.KesslerConfig.autoconversion_threshold',
    'atm.micro.KesslerConfig.evaporation_coeff',
    'atm.micro.KesslerConfig.hard_sat_adjust_threshold',
    'atm.micro.KesslerConfig.hard_sat_max_heating_K',
    'atm.micro.KesslerConfig.rain_fall_speed',
    # atm: KuoConfig (2)
    'atm.conv.KuoConfig.anthes_rh_offset',
    'atm.conv.KuoConfig.entrainment',
    # atm: LindzenConfig (2)
    'atm.gwd.LindzenConfig.fcrit2',
    'atm.gwd.LindzenConfig.h_topo',
    # atm: LouisConfig (6)
    'atm.turb.LouisConfig.Ri_crit',
    'atm.turb.LouisConfig.b_heat_ratio',
    'atm.turb.LouisConfig.b_louis',
    'atm.turb.LouisConfig.c_louis',
    'atm.turb.LouisConfig.d_louis',
    'atm.turb.LouisConfig.l_mix_max',
    # atm: MYNN25Config (9)
    'atm.turb.MYNN25Config.A1',
    'atm.turb.MYNN25Config.A2',
    'atm.turb.MYNN25Config.B1',
    'atm.turb.MYNN25Config.B2',
    'atm.turb.MYNN25Config.C1',
    'atm.turb.MYNN25Config.C2',
    'atm.turb.MYNN25Config.C3',
    'atm.turb.MYNN25Config.C5',
    'atm.turb.MYNN25Config.gamma1',
    # atm: MassFluxConfig (6)
    'atm.conv.MassFluxConfig.M_b_max',
    'atm.conv.MassFluxConfig.M_scale',
    'atm.conv.MassFluxConfig.cape_activation_scale',
    'atm.conv.MassFluxConfig.cape_threshold',
    'atm.conv.MassFluxConfig.delta_0',
    'atm.conv.MassFluxConfig.tau_adj',
    # atm: McFarlaneConfig (7)
    'atm.gwd.McFarlaneConfig.G_0',
    'atm.gwd.McFarlaneConfig.directional_spread',
    'atm.gwd.McFarlaneConfig.efficiency',
    'atm.gwd.McFarlaneConfig.envelope_scale',
    'atm.gwd.McFarlaneConfig.fcrit2',
    'atm.gwd.McFarlaneConfig.h_topo',
    'atm.gwd.McFarlaneConfig.min_wind',
    # atm: MorrisonConfig (23)
    'atm.micro.MorrisonConfig.N_i0',
    'atm.micro.MorrisonConfig.Nc_0',
    'atm.micro.MorrisonConfig.a_v_i',
    'atm.micro.MorrisonConfig.a_v_r',
    'atm.micro.MorrisonConfig.a_v_s',
    'atm.micro.MorrisonConfig.agg_coeff',
    'atm.micro.MorrisonConfig.bergeron_rate',
    'atm.micro.MorrisonConfig.cooper_T_act',
    'atm.micro.MorrisonConfig.cooper_a',
    'atm.micro.MorrisonConfig.dep_coeff',
    'atm.micro.MorrisonConfig.evap_coeff',
    'atm.micro.MorrisonConfig.hard_sat_adjust_threshold',
    'atm.micro.MorrisonConfig.hard_sat_max_heating_K',
    'atm.micro.MorrisonConfig.ice_snow_d_auto',
    'atm.micro.MorrisonConfig.k_ac',
    'atm.micro.MorrisonConfig.k_au',
    'atm.micro.MorrisonConfig.k_sc',
    'atm.micro.MorrisonConfig.melt_rate',
    'atm.micro.MorrisonConfig.rain_selfcoll_k',
    'atm.micro.MorrisonConfig.rain_vent_f1',
    'atm.micro.MorrisonConfig.rain_vent_f2',
    'atm.micro.MorrisonConfig.rime_coeff',
    'atm.micro.MorrisonConfig.x_star',
    # atm: OzoneProfileConfig (3)
    'atm.rad.OzoneProfileConfig.o3_max_vmr',
    'atm.rad.OzoneProfileConfig.p_peak_hPa',
    'atm.rad.OzoneProfileConfig.sigma_logp',
    # atm: P3Config (18)
    'atm.micro.P3Config.N_i0',
    'atm.micro.P3Config.Nc_0',
    'atm.micro.P3Config.a_v_i',
    'atm.micro.P3Config.a_v_r',
    'atm.micro.P3Config.agg_coeff',
    'atm.micro.P3Config.cooper_T_act',
    # gSAM scheme-1 nucleation-gate params (2026-07-17 Cooper-faithfulness
    # closure): same conscious exclusion as the rest of the P3Config family —
    # the whole class is not routed through any driver's --params loader yet.
    'atm.micro.P3Config.cooper_T_nuc',
    'atm.micro.P3Config.cooper_a',
    'atm.micro.P3Config.cooper_supi_min',
    'atm.micro.P3Config.dep_coeff',
    'atm.micro.P3Config.evap_coeff',
    'atm.micro.P3Config.hard_sat_adjust_threshold',
    'atm.micro.P3Config.hard_sat_max_heating_K',
    'atm.micro.P3Config.k_ac',
    'atm.micro.P3Config.k_au',
    'atm.micro.P3Config.k_sc',
    'atm.micro.P3Config.melt_rate',
    'atm.micro.P3Config.rain_rime_coeff',
    'atm.micro.P3Config.rime_coeff',
    'atm.micro.P3Config.x_star',
    # atm: PBLHeightConfig (1)
    'atm.pblh.PBLHeightConfig.Ri_crit',
    # atm: PrognosticSpectralConfig (3)
    'atm.gwd.PrognosticSpectralConfig.breaking_threshold',
    'atm.gwd.PrognosticSpectralConfig.launch_flux',
    'atm.gwd.PrognosticSpectralConfig.tau_decay',
    # atm: RayleighConfig (4)
    'atm.gwd.RayleighConfig.k_max',
    'atm.gwd.RayleighConfig.sigma_b',
    'atm.gwd.RayleighConfig.sponge_k',
    'atm.gwd.RayleighConfig.sponge_top',
    # atm: SDMConfig (3)
    'atm.sdm.SDMConfig.cdnc',
    'atm.sdm.SDMConfig.golovin_b',
    'atm.sdm.SDMConfig.r_rain',
    # atm: SeifertBehengConfig (9)
    'atm.micro.SeifertBehengConfig.Nc_0',
    'atm.micro.SeifertBehengConfig.a_v_r',
    'atm.micro.SeifertBehengConfig.evap_coeff',
    'atm.micro.SeifertBehengConfig.hard_sat_adjust_threshold',
    'atm.micro.SeifertBehengConfig.hard_sat_max_heating_K',
    'atm.micro.SeifertBehengConfig.k_ac',
    'atm.micro.SeifertBehengConfig.k_au',
    'atm.micro.SeifertBehengConfig.k_sc',
    'atm.micro.SeifertBehengConfig.x_star',
    # atm: SmagorinskyConfig (3)
    'atm.turb.SmagorinskyConfig.C_s',
    'atm.turb.SmagorinskyConfig.Pr_t',
    'atm.turb.SmagorinskyConfig.l_mix_max',
    # atm: SundqvistConfig (6) — like its siblings above, the microphysics
    # scheme *Configs are not threaded through ExperimentConfig scalars; the
    # SBK89 enhancement coefficients join the existing conscious entries.
    'atm.micro.SundqvistConfig.auto_rate',
    'atm.micro.SundqvistConfig.bergeron_enh_coeff',
    'atm.micro.SundqvistConfig.coalescence_enh_coeff',
    'atm.micro.SundqvistConfig.evap_coeff',
    'atm.micro.SundqvistConfig.qc_crit',
    'atm.micro.SundqvistConfig.rh_crit',
    # atm: SurfaceLayerConfig (5) — conscious exclusion: like Cd_neutral /
    # Ch_neutral / z0, the two MOST stability-function coefficients are trained
    # via the AIMIP classical bundle (aimip_params.surface_most_*), not the
    # ExperimentConfig --params scalar map. Same reachability status as their
    # SurfaceLayerConfig siblings.
    'atm.turb.SurfaceLayerConfig.Cd_neutral',
    'atm.turb.SurfaceLayerConfig.Ch_neutral',
    'atm.turb.SurfaceLayerConfig.most_stable_beta',
    'atm.turb.SurfaceLayerConfig.most_unstable_gamma',
    'atm.turb.SurfaceLayerConfig.z0',
    # atm: TKEConfig (4)
    'atm.turb.TKEConfig.Ce',
    'atm.turb.TKEConfig.Ck',
    'atm.turb.TKEConfig.Pr_t',
    'atm.turb.TKEConfig.l_mix_max',
    # atm: ThompsonConfig (23)
    'atm.micro.ThompsonConfig.N_i0',
    'atm.micro.ThompsonConfig.Nc_0',
    'atm.micro.ThompsonConfig.a_v_g',
    'atm.micro.ThompsonConfig.a_v_i',
    'atm.micro.ThompsonConfig.a_v_r',
    'atm.micro.ThompsonConfig.a_v_s',
    'atm.micro.ThompsonConfig.agg_coeff',
    'atm.micro.ThompsonConfig.bergeron_rate',
    'atm.micro.ThompsonConfig.cooper_T_act',
    'atm.micro.ThompsonConfig.cooper_a',
    'atm.micro.ThompsonConfig.dep_coeff',
    'atm.micro.ThompsonConfig.evap_coeff',
    'atm.micro.ThompsonConfig.hard_sat_adjust_threshold',
    'atm.micro.ThompsonConfig.hard_sat_max_heating_K',
    'atm.micro.ThompsonConfig.ice_snow_d_auto',
    'atm.micro.ThompsonConfig.k_ac',
    'atm.micro.ThompsonConfig.k_au',
    'atm.micro.ThompsonConfig.k_sc',
    'atm.micro.ThompsonConfig.melt_rate',
    'atm.micro.ThompsonConfig.rime_coeff',
    'atm.micro.ThompsonConfig.rime_to_graupel_rate',
    'atm.micro.ThompsonConfig.rime_to_graupel_threshold',
    'atm.micro.ThompsonConfig.x_star',
    # atm: TiedtkeConfig (15)
    'atm.conv.TiedtkeConfig.M_b_max',
    'atm.conv.TiedtkeConfig.cape_threshold',
    'atm.conv.TiedtkeConfig.cloud_depth_deep',
    'atm.conv.TiedtkeConfig.cloud_depth_shallow_max',
    'atm.conv.TiedtkeConfig.delta_deep',
    'atm.conv.TiedtkeConfig.delta_midlevel',
    'atm.conv.TiedtkeConfig.delta_shallow',
    'atm.conv.TiedtkeConfig.downdraft_alpha',
    'atm.conv.TiedtkeConfig.downdraft_evap_efficiency',
    'atm.conv.TiedtkeConfig.mc_proxy_RH_crit',
    'atm.conv.TiedtkeConfig.midlevel_M_b_fraction',
    'atm.conv.TiedtkeConfig.moisture_convergence_threshold',
    'atm.conv.TiedtkeConfig.tau_MC_proxy',
    'atm.conv.TiedtkeConfig.tau_M_u_relax',
    'atm.conv.TiedtkeConfig.tau_shallow_M_b',
    # atm: TurbulentEDMFConfig (7)
    'atm.turb.TurbulentEDMFConfig.Ce',
    'atm.turb.TurbulentEDMFConfig.Ck',
    'atm.turb.TurbulentEDMFConfig.Pr_t',
    'atm.turb.TurbulentEDMFConfig.a_updraft',
    'atm.turb.TurbulentEDMFConfig.entrainment_rate',
    'atm.turb.TurbulentEDMFConfig.l_mix_max',
    'atm.turb.TurbulentEDMFConfig.parcel_dT',
    # atm: YSUConfig (10)
    'atm.turb.YSUConfig.Pr_t',
    'atm.turb.YSUConfig.Ri_crit',
    'atm.turb.YSUConfig.countergrad_coeff',
    # entrainment_coeff renamed -> entrainment_ratio (Hong06 prescribed
    # entrainment-flux ratio; spec carries legacy_name) — a RENAME in place,
    # not baseline growth (net count unchanged at 10).
    'atm.turb.YSUConfig.entrainment_ratio',
    'atm.turb.YSUConfig.entrainment_width_frac',
    'atm.turb.YSUConfig.l_mix_max',
    'atm.turb.YSUConfig.louis_b',
    'atm.turb.YSUConfig.louis_c',
    'atm.turb.YSUConfig.louis_d',
    'atm.turb.YSUConfig.ws_conv_coeff',
    # atm: ZhangMcFarlaneConfig (6)
    'atm.conv.ZhangMcFarlaneConfig.M_b_max',
    'atm.conv.ZhangMcFarlaneConfig.cape_threshold',
    'atm.conv.ZhangMcFarlaneConfig.delta_0',
    'atm.conv.ZhangMcFarlaneConfig.dmpdz',
    'atm.conv.ZhangMcFarlaneConfig.tau_cape',
    'atm.conv.ZhangMcFarlaneConfig.tiedke_add',
    # ocean: BackscatterConfig (4)
    'ocean.backscatter.E_max',
    'ocean.backscatter.c_bs',
    'ocean.backscatter.efficiency',
    'ocean.backscatter.tau_relax_days',
    # ocean: BulkFormulaConfig (6)
    'ocean.sf.bulk.C_D',
    'ocean.sf.bulk.C_E',
    'ocean.sf.bulk.C_H',
    'ocean.sf.bulk.emissivity',
    'ocean.sf.bulk.q_sat_salinity_factor',
    'ocean.sf.bulk.z0',
    # ocean: EKEConfig (7)
    'ocean.eke.eke.alpha_eke',
    'ocean.eke.eke.c_eps',
    'ocean.eke.eke.c_k',
    'ocean.eke.eke.eke_crhin',
    'ocean.eke.eke.eke_cross',
    'ocean.eke.eke.k_iso',
    'ocean.eke.eke.kappa_gm_max',
    # ocean: EnhancedDiffusionConfig (2)
    'ocean.conv.enhanced_diffusion.K_bg',
    'ocean.conv.enhanced_diffusion.K_conv',
    # ocean: FluxFeedbackConfig (1)
    'ocean.sf.flux_feedback.tau_restore_s',
    # ocean: GeometricConfig (13)
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
    # ocean: HarmonicConfig (2)
    'ocean.lat.harmonic.A_h',
    'ocean.lat.harmonic.K_h',
    # ocean: IceShelfConfig (3)
    'ocean.ice_shelf.gamma_S',
    'ocean.ice_shelf.gamma_T',
    'ocean.ice_shelf.melt_factor_linear',
    # ocean: LinearDragConfig (1)
    'ocean.bottom_drag.linear.r',
    # ocean: MLEConfig (1)
    'ocean.lat.mle.ce',
    # ocean: PlumeConfig (3)
    'ocean.conv.plume.T_excess',
    'ocean.conv.plume.alpha_plume',
    'ocean.conv.plume.epsilon',
    # ocean: PrescribedForcingConfig (2)
    'ocean.sf.prescribed.tau_max',
    'ocean.sf.prescribed.tropical_wind_scale',
    # ocean: QuadraticDragConfig (1)
    'ocean.bottom_drag.quadratic.C_d',
    # ocean: RestoringConfig (2)
    'ocean.sf.restoring.tau_S',
    'ocean.sf.restoring.tau_T',
    # ocean: ShortwavePenetrationConfig (2)
    'ocean.sw_penetration.rgb_ir_extinction_m',
    'ocean.sw_penetration.rgb_ir_fraction',
    # ocean: TidalForcingConfig (3) — CONSCIOUS entry (equilibrium-tide forcing).
    # These ARE settable via YAML (ocean.tidal_forcing.{love_factor,beta_sal,
    # amplitude_scale}, built into a TidalForcingConfig by config.py). They are
    # not yet on run_omip's curated OMIPRunConfig --params surface; wiring
    # tidal_forcing onto OMIPRunConfig (+ a --tidal-forcing flag) is the follow-up
    # that shrinks these three away.
    'ocean.tidal_forcing.amplitude_scale',
    'ocean.tidal_forcing.beta_sal',
    'ocean.tidal_forcing.love_factor',
    # atm: aerosol activation / prognostic aerosol (5) — CONSCIOUS entry.
    # ActivationConfig hangs off MorrisonConfig.activation (reachable
    # programmatically, wired into microphysics/integration.py) but is not yet
    # on the AMIP flattened-scalar --params surface; PrognosticAerosolConfig is
    # a tested driver hook not yet stepped by any production driver.  Wiring
    # ActivationConfig onto the AMIP scalar map (+ the prognostic-aerosol
    # driver hook) is the follow-up that shrinks these five away.
    'atm.aerosol.ActivationConfig.w_char_m_s',
    'atm.aerosol.PrognosticAerosolConfig.dry_dep_velocity_m_s',
    'atm.aerosol.PrognosticAerosolConfig.emission_number_flux_m2_s',
    'atm.aerosol.PrognosticAerosolConfig.so2_oxidation_timescale_s',
    'atm.aerosol.PrognosticAerosolConfig.wet_scavenging_coeff_m2_kg',
    # land: SIF (8) + elevation snow bands (8) — CONSCIOUS entry (#933).
    # Both configs landed with the 2026-07 land campaign (FvCB/SIF
    # consolidation #897, high-elevation snow/ice bands) fully spec'd but
    # off run_lmip's --params surface: SIFConfig is a fluorescence
    # DIAGNOSTIC of the big-leaf photosynthesis path (its params calibrate
    # against satellite SIF, not the water/energy budget); the elevation
    # snow bands are opt-in (--elev-bands) and run_lmip builds
    # ElevationSnowBandConfig with defaults + the per-cell band_dz only.
    # Wiring both onto run_lmip's curated --params surface when their
    # calibration campaigns need it is the follow-up that shrinks these
    # sixteen away.
    'land.canopy.sif.escape_probability',
    'land.canopy.sif.kd',
    'land.canopy.sif.kf',
    'land.canopy.sif.kn0',
    'land.canopy.sif.kn_beta',
    'land.canopy.sif.kn_gamma',
    'land.canopy.sif.kp',
    'land.canopy.sif.max_electron_yield',
    'land.snow_bands.alpha_glacier_ice',
    'land.snow_bands.blow_snow_wind_thresh_ms',
    'land.snow_bands.lapse_rate_K_m',
    'land.snow_bands.lw_elev_lapse_W_m2_per_m',
    'land.snow_bands.sw_elev_grad_per_m',
    # --- Land closures spec'd but not routed by any run_lmip land-surface
    # scheme's config tree (#691). apply_params_to_config(driver="run_lmip")
    # does not carry these nested *Config NamedTuples under simple_seb/two_leaf/
    # clm_ml, so they are not settable from a calibration file yet. Conscious
    # baseline entries (shrink as each config is wired through the router). The
    # land_use_change family surfaced when the module was registered in
    # param_collector.SPEC_MODULES (its spec is in-scope, so the drift guard
    # requires registration); interception / d13c are pre-existing #691 debt.
    # (clm_ml / two_leaf_canopy params ARE reachable under their
    # --land-surface-scheme and are deliberately NOT baselined.)
    'land.canopy.interception.dewmx',
    'land.canopy.interception.fwet_exponent',
    'land.canopy.interception.maximum_leaf_wetted_fraction',
    'land.d13c.phi_c4_leakiness',
    'land.land_use_change.clear_burn_frac',
    'land.land_use_change.clear_slash_frac',
    'land.land_use_change.prod_frac_100yr',
    'land.land_use_change.prod_frac_10yr',
    'land.land_use_change.prod_frac_1yr',
    'land.land_use_change.tau_100yr_years',
    'land.land_use_change.tau_10yr_years',
    'land.land_use_change.tau_1yr_years',
    'land.land_use_change.tau_regrow_years',
})
