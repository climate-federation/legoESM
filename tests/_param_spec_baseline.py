"""Field-level seed baseline for the param-spec gate's TODO modules.

``PARAM_SPEC_TODO`` (in ``tests/test_param_specs.py``) allowlists whole *modules*
that do not yet carry a ``__param_spec__``. Without a field-level floor, a NEW
``: float`` default added to one of those pre-existing modules would slip through
unclassified (the module is already allowlisted) — the param-hygiene invariant
could regress silently in exactly the ~30 modules where most physics configs
live. This baseline closes that gap: it records the set of spec-required
``Class.field`` names present in each TODO module at seed time, and the gate
asserts a TODO module's CURRENT float-default fields are a SUBSET (removals fine,
additions red — a new tunable must be specced/excluded or the whole module
graduated out of TODO).

SHRINK-ONLY, in lockstep with ``PARAM_SPEC_TODO``: when a module graduates (gains
a complete ``__param_spec__`` and leaves ``PARAM_SPEC_TODO``), delete its entry
here too. Regenerate with ``scripts/tmp/_seed_param_spec_baseline.py``.
"""

from __future__ import annotations

# Seeded 2026-06-12 from the measured tree.
# PARALLEL-MERGE BACKLOG (2026-06-13): seeded for the concurrent CLUBB
# port (CLUBBConfig + CLUBBParams, ~109 float fields) that landed on main
# via a separate PR. Pins its CURRENT float fields so a NEW unspecced float
# cannot slip into clubb.py while it sits in PARAM_SPEC_TODO. Delete when
# clubb.py graduates (ships a __param_spec__). Re-seed with
# scripts/tmp/_seed_param_spec_baseline.py.
PARAM_SPEC_FIELD_BASELINE: dict[str, frozenset[str]] = {
    'packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py': frozenset({
        'CLUBBConfig.T0',
        'CLUBBConfig.clubb_dt',
        'CLUBBConfig.rt_tol',
        'CLUBBConfig.thl_tol',
        'CLUBBConfig.tke_min',
        'CLUBBConfig.w_tol',
        'CLUBBConfig.wp2_max',
        'CLUBBParams.C1',
        'CLUBBParams.C10',
        'CLUBBParams.C11',
        'CLUBBParams.C11b',
        'CLUBBParams.C11c',
        'CLUBBParams.C12',
        'CLUBBParams.C13',
        'CLUBBParams.C14',
        'CLUBBParams.C1b',
        'CLUBBParams.C1c',
        'CLUBBParams.C2rt',
        'CLUBBParams.C2rtthl',
        'CLUBBParams.C2thl',
        'CLUBBParams.C4',
        'CLUBBParams.C6rt',
        'CLUBBParams.C6rt_Lscale0',
        'CLUBBParams.C6rtb',
        'CLUBBParams.C6rtc',
        'CLUBBParams.C6thl',
        'CLUBBParams.C6thl_Lscale0',
        'CLUBBParams.C6thlb',
        'CLUBBParams.C6thlc',
        'CLUBBParams.C7',
        'CLUBBParams.C7_Lscale0',
        'CLUBBParams.C7b',
        'CLUBBParams.C7c',
        'CLUBBParams.C8',
        'CLUBBParams.C8b',
        'CLUBBParams.C_invrs_tau_N2',
        'CLUBBParams.C_invrs_tau_N2_clear_wp3',
        'CLUBBParams.C_invrs_tau_N2_wp2',
        'CLUBBParams.C_invrs_tau_N2_wpxp',
        'CLUBBParams.C_invrs_tau_N2_xp2',
        'CLUBBParams.C_invrs_tau_bkgnd',
        'CLUBBParams.C_invrs_tau_sfc',
        'CLUBBParams.C_invrs_tau_shear',
        'CLUBBParams.C_invrs_tau_wpxp_N2_thresh',
        'CLUBBParams.C_invrs_tau_wpxp_Ri',
        'CLUBBParams.C_uu_buoy',
        'CLUBBParams.C_uu_shr',
        'CLUBBParams.C_wp2_pr_dfsn',
        'CLUBBParams.C_wp2_splat',
        'CLUBBParams.C_wp3_pr_dfsn',
        'CLUBBParams.C_wp3_pr_tp',
        'CLUBBParams.C_wp3_pr_turb',
        'CLUBBParams.Cx_max',
        'CLUBBParams.Cx_min',
        'CLUBBParams.K_hm_min_coef',
        'CLUBBParams.Lscale_mu_coef',
        'CLUBBParams.Lscale_pert_coef',
        'CLUBBParams.Richardson_num_max',
        'CLUBBParams.Richardson_num_min',
        'CLUBBParams.Skw_denom_coef',
        'CLUBBParams.Skw_max_mag',
        'CLUBBParams.a3_coef_min',
        'CLUBBParams.a_const',
        'CLUBBParams.alpha_corr',
        'CLUBBParams.altitude_threshold',
        'CLUBBParams.beta',
        'CLUBBParams.bv_efold',
        'CLUBBParams.c_K',
        'CLUBBParams.c_K1',
        'CLUBBParams.c_K10',
        'CLUBBParams.c_K10h',
        'CLUBBParams.c_K2',
        'CLUBBParams.c_K6',
        'CLUBBParams.c_K8',
        'CLUBBParams.c_K9',
        'CLUBBParams.c_K_hm',
        'CLUBBParams.c_K_hmb',
        'CLUBBParams.coef_spread_DG_means_rt',
        'CLUBBParams.coef_spread_DG_means_thl',
        'CLUBBParams.gamma_coef',
        'CLUBBParams.gamma_coefb',
        'CLUBBParams.gamma_coefc',
        'CLUBBParams.lambda0_stability_coef',
        'CLUBBParams.lmin_coef',
        'CLUBBParams.mu',
        'CLUBBParams.mult_coef',
        'CLUBBParams.nu1',
        'CLUBBParams.nu10',
        'CLUBBParams.nu2',
        'CLUBBParams.nu6',
        'CLUBBParams.nu8',
        'CLUBBParams.nu9',
        'CLUBBParams.nu_hm',
        'CLUBBParams.omicron',
        'CLUBBParams.pdf_component_stdev_factor_w',
        'CLUBBParams.rtp2_clip_coef',
        'CLUBBParams.slope_coef_spread_DG_means_w',
        'CLUBBParams.taumax',
        'CLUBBParams.taumin',
        'CLUBBParams.thlp2_rad_cloud_frac_thresh',
        'CLUBBParams.thlp2_rad_coef',
        'CLUBBParams.up2_sfc_coef',
        'CLUBBParams.upsilon_precip_frac_rat',
        'CLUBBParams.wpxp_L_thresh',
        'CLUBBParams.wpxp_Ri_exp',
        'CLUBBParams.xp3_coef_base',
        'CLUBBParams.xp3_coef_slope',
        'CLUBBParams.z_displace',
        'CLUBBParams.zeta_vrnce_rat',
    }),
}
