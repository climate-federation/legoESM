"""Configuration for atmospheric convection schemes.

Provides configuration NamedTuples for:
1. Simplified Betts-Miller (SBM) — relaxation-based convection (Frierson 2007)
2. Deep Convective Adjustment (DCA) — simplest baseline adjustment.  Two
   variants via ``DCAConfig.variant``: the Manabe-style pairwise
   moist-adiabatic adjustment (default) and the Ahmed-Neelin-Adames (2020)
   lower-tropospheric-buoyancy (B_L) closure (``AhmedNeelinDCAConfig``).
3. Kuo — column moisture-excess (Kuo 1965/1974)
4. Prognostic Mass-Flux — Arakawa-Wu type (1 prognostic var: M_c)
5. Simplified EDMF — eddy-diffusivity mass-flux (1 prognostic var: a_u)
6. Top-level ConvectionConfig that selects the active scheme.

References
----------
- Frierson, D. M. W. (2007). The Dynamics of Idealized Convection
  Schemes and Their Effect on the Zonally Averaged Tropical Circulation.
  J. Atmos. Sci., 64, 1959-1976.
- Ahmed, F., Adames, A. F., & Neelin, J. D. (2020). Deep convective
  adjustment of temperature and moisture. J. Atmos. Sci., 77, 2163-2186.
- Kuo, H. L. (1974). Further studies of the parameterization of the
  influence of cumulus convection on large-scale flow. J. Atmos. Sci.,
  31, 1232-1240.
- Arakawa, A., & Wu, C.-M. (2013). A unified representation of deep
  moist convection in numerical modeling of the atmosphere. Part I.
  J. Atmos. Sci., 70, 1977-1992.
"""

from __future__ import annotations

from typing import NamedTuple


__param_spec__ = {
    "AhmedNeelinDCAConfig": {
        "scheme_key": "atm.conv.AhmedNeelinDCAConfig",
        "excluded": {
            "precip_heaviside_sharpness": "numerics: softplus sharpness smoothing the eq-8 Heaviside",
        },
        "params": {
            "a_mm_per_hr": {"units": "mm/h/(m/s^2)", "bounds": (0.2, 1.8), "tunable_tier": 1, "transform": "sigmoid", "category": "cape_closure", "reference": "Ahmed, Adames & Neelin (2020) Table 1", "shape": None},
            "b_c": {"units": "m/s^2", "bounds": (-0.045, -0.005), "tunable_tier": 1, "transform": "sigmoid", "category": "trigger", "reference": "Ahmed, Adames & Neelin (2020) Table 1", "shape": None},
            "layer_edge_width_pa": {"units": "Pa", "bounds": (500.0, 7500.0), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "Ahmed, Adames & Neelin (2020) scheme default", "shape": None},
            "layer_min_depth_pa": {"units": "Pa", "bounds": (1650.0, 15000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "Ahmed, Adames & Neelin (2020) scheme default", "shape": None},
            "p_bl_top_pa": {"units": "Pa", "bounds": (75000.0, 92000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cloud_base", "reference": "Ahmed, Adames & Neelin (2020) Table 1", "shape": None},
            "p_lft_top_pa": {"units": "Pa", "bounds": (40000.0, 60000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cloud_base", "reference": "Ahmed, Adames & Neelin (2020) Table 1", "shape": None},
            "tau_adjust_s": {"units": "s", "bounds": (1800.0, 21600.0), "tunable_tier": 2, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Ahmed, Adames & Neelin (2020) eq 42", "shape": None},
            "w_b": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cape_closure", "reference": "Ahmed, Adames & Neelin (2020) Table 1", "shape": None},
            "w_l": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cape_closure", "reference": "Ahmed, Adames & Neelin (2020) Table 1", "shape": None},
        },
    },
    "BechtoldConfig": {
        "scheme_key": "atm.conv.BechtoldConfig",
        "excluded": {
            "cape_sharpness": "numerics: sigmoid sharpness on the CAPE trigger gate",
            "cape_sink_heating_ratio": "inert: the quasi-equilibrium heating ceiling it scales has no consumer in bechtold.py, so the leaf carries no loss gradient; re-tier to 2 in the same PR that implements the sink",
            "depth_split_sharpness": "numerics: sigmoid sharpness on the deep/shallow depth blend",
            "downdraft_RH_min": "trigger: column-mean RH threshold below which the downdraft fires (not sigmoid-tunable, fix via config)",
            "downdraft_rh_sharpness": "numerics: sigmoid sharpness on the downdraft RH trigger [1/RH-fraction]",
            "downdraft_detrain_scale_m": "numerics: near-surface height scale [m] over which the penetrative-downdraft mass flux tapers to zero (structural deposit depth, not a trained closure)",
            "dx_m": "grid property: horizontal grid spacing [m] for the IFS ZTAURES resolution factor (cumastrn.F90:762-768); set by the driver from the grid, never trained; 0 = resolution-agnostic legacy",
            "epsilon_midlevel": "entrainment: TUNED transition-blend base rate (1e-4, intentionally below IFS ENTSHALP*ENTRORG=3.5e-3; see audit F6), scaled in-scheme",
            "epsilon_shallow": "entrainment: IFS shallow base rate scaled in-scheme",
            "lcl_membership_sharpness": "numerics: sigmoid sharpness on the below-LCL level membership [1/level index]",
            "parcel_dT": "trigger: fixed sub-cloud parcel temperature perturbation",
            "precip_efficiency": "off/on precip-efficiency discontinuity gated by a static Python branch (`if config.precip_efficiency > 0.0` in bechtold.py); not a differentiable trainable leaf (a traced leaf breaks the JIT gate). Retune via config, not sigmoid-trained across the off/on discontinuity.",
            "p_conv_top_pa": "numerics: convective-top pressure [Pa] terminating the plume/subsidence gate (stability, not a trained closure); 150 hPa deep-convection top",
            "theta_implicit": "numerics: off-centering of the implicit_flux backward-Euler subsidence solve (stability, iteration-coupled; clamped to [0.5,1.0], not trainable)",
        },
        "params": {
            "autoconv_q_c_crit": {"units": "kg/kg", "bounds": (1.0e-4, 2.0e-3), "tunable_tier": 2, "transform": "sigmoid", "category": "precipitation_efficiency", "reference": "Sundqvist (1978) autoconversion critical cloud water", "shape": None},
            "autoconv_pe_max": {"units": "1", "bounds": (0.5, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "precipitation_efficiency", "reference": "convective precip-efficiency ceiling (Sundqvist 1978 form)", "shape": None},
            "M_b_max": {"units": "kg/m^2/s", "bounds": (0.02, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "Bechtold et al. (2008) stability cap", "shape": None},
            "cape_pbl_depth": {"units": "m", "bounds": (200.0, 1500.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cape_closure", "reference": "Bechtold et al. (2008)", "shape": None},
            "cape_qadv_weight": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cape_closure", "reference": "IFS RCAPQADV=0.8 (sucumf.F90:219)", "shape": None},
            "cape_threshold": {"units": "J/kg", "bounds": (23.1, 210.0), "tunable_tier": 0, "transform": "sigmoid", "category": "trigger", "reference": "tier 0 / AD-unreachable, see _CAPE_TRIGGER_AD_NOTE (#1417). Original: Bechtold et al. (2008)", "shape": None},
            "cloud_depth_deep": {"units": "m", "bounds": (1500.0, 5000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "updraft", "reference": "Tiedtke (1989) depth split", "shape": None},
            "cloud_depth_shallow_max": {"units": "m", "bounds": (800.0, 2500.0), "tunable_tier": 2, "transform": "sigmoid", "category": "updraft", "reference": "Tiedtke (1989) depth split", "shape": None},
            "cmt_c_d": {"units": "1", "bounds": (0.0, 2.0), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing", "reference": "Gregory et al. (1997) CMT", "shape": None},
            "cmt_c_u": {"units": "1", "bounds": (0.0, 2.0), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing", "reference": "Gregory et al. (1997) CMT", "shape": None},
            "dnoprc": {"units": "kg/kg", "bounds": (7.5e-5, 1.2e-3), "tunable_tier": 2, "transform": "sigmoid", "category": "conversion", "reference": "IFS ZDNOPRC=3e-4 (cuascn.F90:277)", "shape": None},
            "rprcon": {"units": "1/m", "bounds": (3.5e-4, 5.6e-3), "tunable_tier": 2, "transform": "sigmoid", "category": "conversion", "reference": "IFS RPRCON=1.4e-3 (sucumf.F90:164)", "shape": None},
            "delta_deep": {"units": "1/m", "bounds": (2.475e-05, 0.000225), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Bechtold et al. (2008) IFS Cy49r1", "shape": None},
            "delta_midlevel": {"units": "1/m", "bounds": (6.6e-05, 0.0006), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Tiedtke (1989) mid-level", "shape": None},
            "delta_shallow": {"units": "1/m", "bounds": (2.475e-05, 0.000225), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Bechtold et al. (2008) IFS Cy49r1", "shape": None},
            "downdraft_alpha": {"units": "1", "bounds": (0.0, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "downdraft", "reference": "Tiedtke (1989) downdraft", "shape": None},
            "downdraft_evap_efficiency": {"units": "1", "bounds": (0.0, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "downdraft", "reference": "Tiedtke (1989) downdraft", "shape": None},
            "downdraft_entrain_rate": {"units": "1/m", "bounds": (1.0e-4, 2.0e-3), "tunable_tier": 2, "transform": "sigmoid", "category": "downdraft", "reference": "Tiedtke (1989) penetrative-downdraft entrainment", "shape": None},
            # --- deep-plume entrainment / detrainment base rates (IFS cuascn) ---
            # Exposed 2026-08-14: these set the ITCZ width and tropical rain
            # concentration.  Raising epsilon_deep dilutes the deep plume faster
            # in dry air, so convection survives only where the column is
            # already moist -> narrower, wetter rain band.  Both are scaled
            # in-scheme by the IFS height/RH factors; these are the BASE rates.
            # Bounds bracket the published IFS deep value (1.75e-3 / 0.75e-4)
            # by a factor ~2.4 either way.
            "epsilon_deep": {"units": "1/m", "bounds": (7.0e-4, 4.2e-3), "tunable_tier": 1, "transform": "sigmoid", "category": "entrainment", "reference": "IFS cuascn ENTRORG deep base rate", "shape": None},
            "delta_deep": {"units": "1/m", "bounds": (3.0e-5, 1.8e-4), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "IFS cuascn deep detrainment base rate", "shape": None},
            "mc_normalize_scale": {"units": "kg/m^2/s", "bounds": (0.005, 0.2), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "Bechtold et al. (2008) Fig. 2", "shape": None},
            "parcel_dq": {"units": "kg/kg", "bounds": (0.0, 0.003), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Bechtold et al. (2008) scheme default", "shape": None},
            "stochastic_amplitude": {"units": "1", "bounds": (0.0, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "Bechtold et al. (2014) AR1 perturbation", "shape": None},
            "stochastic_decorrelation": {"units": "s", "bounds": (1800.0, 21600.0), "tunable_tier": 2, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Bechtold et al. (2014) AR1 perturbation", "shape": None},
            "tau_M_u_relax": {"units": "s", "bounds": (600.0, 5400.0), "tunable_tier": 2, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Tiedtke (1989) profile relaxation", "shape": None},
            "tau_bl": {"units": "s", "bounds": (1188.0, 10800.0), "tunable_tier": 1, "transform": "sigmoid", "category": "cape_closure", "reference": "Bechtold et al. (2008) PBL closure", "shape": None},
        },
    },
    "ConvectiveEDMFConfig": {
        "scheme_key": "atm.conv.ConvectiveEDMFConfig",
        "excluded": {
            "precip_efficiency": "default 0 = disabled (legacy no rain-split, shared split_convective_rain gated `if > 0.0`); enable + retune via config, not sigmoid-trained from the off state",
            "epsilon_0": "entrainment: bulk-plume base rate held fixed in the EDMF mass-flux core",
            "w_u_min": "numerics: minimum updraft velocity floor",
            "w_u_max": "numerics: physical cap on the updraft velocity sqrt(2*CAPE) (stability, not a tunable closure)",
            "theta_implicit": "numerics: off-centering of the implicit_flux backward-Euler subsidence solve (stability, iteration-coupled; not trainable)",
        },
        "params": {
            "M_b_max": {"units": "kg/m^2/s", "bounds": (0.02, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "EDMF mass-flux stability cap", "shape": None},
            "a_u_init": {"units": "1", "bounds": (0.0, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "updraft", "reference": "EDMF scheme default", "shape": None},
            "cape_activation_scale": {"units": "J/kg", "bounds": (3.3, 30.0), "tunable_tier": 2, "transform": "sigmoid", "category": "trigger", "reference": "EDMF scheme default", "shape": None},
            "cape_threshold": {"units": "J/kg", "bounds": (23.1, 210.0), "tunable_tier": 0, "transform": "sigmoid", "category": "trigger", "reference": "tier 0 / AD-unreachable, see _CAPE_TRIGGER_AD_NOTE (#1417). Original: EDMF scheme default", "shape": None},
            "delta_0": {"units": "1/m", "bounds": (0.00066, 0.006), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "EDMF scheme default", "shape": None},
            "tau_a": {"units": "s", "bounds": (600.0, 5400.0), "tunable_tier": 1, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "EDMF scheme default", "shape": None},
        },
    },
    "DCAConfig": {
        "scheme_key": "atm.conv.DCAConfig",
        "excluded": {
            "cape_sharpness": "numerics: sigmoid sharpness on the CAPE gate",
            "instability_blend_sharpness": "numerics: sigmoid sharpness on the per-pair superadiabatic blend",
            "mixing_fraction": "numerics: per-iteration adjustment fraction (default 1.0 at domain boundary, not sigmoid-tunable)",
        },
        "params": {
            "cape_threshold": {"units": "J/kg", "bounds": (33.0, 300.0), "tunable_tier": 0, "transform": "sigmoid", "category": "trigger", "reference": "tier 0 / AD-unreachable, see _CAPE_TRIGGER_AD_NOTE (#1417). Original: Manabe et al. (1965) moist adjustment", "shape": None},
        },
    },
    "EmanuelConfig": {
        "scheme_key": "atm.conv.EmanuelConfig",
        "excluded": {
            "cape_sharpness": "numerics: sigmoid sharpness on the CAPE gate",
            "cbmf_positive_sharpness": "numerics: softplus sharpness on the relaxed CBMF positive-part",
            "denom_floor": "numerics: SIJ denominator magnitude floor (oracle ABS(DENOM)<0.01)",
            "epsilon_0": "entrainment: near-undilute bulk-plume rate held fixed (mixing handled by the ensemble)",
            "level_window_sharpness": "numerics: sigmoid sharpness on the ICB/INB cloud-layer windows",
            "lcl_pressure_sharpness": "numerics: sigmoid sharpness on the pressure-bounded sub-cloud layer",
            "below_lcl_index_sharpness": "numerics: sigmoid sharpness on the below-LCL index indicator (downdraft re-evap)",
            "precip_efficiency_lcl": "default 0 = disabled/off (enable via config, not training)",
            "precip_efficiency_water": "precipitation_efficiency: default 1.0 at domain boundary (not sigmoid-tunable, fix via config)",
            "sat_branch_sharpness": "numerics: sigmoid sharpness on the saturated-mixture re-solve switch",
            "sij_gate_sharpness": "numerics: sigmoid sharpness on the 0<SIJ<0.9 entrainment band gate",
            "smooth_trigger_sharpness": "numerics: sigmoid sharpness on the buoyancy-sort weighting",
            "strict_index_sharpness": "numerics: sigmoid sharpness on the strict integer-index inequalities",
        },
        "params": {
            "M_b_max": {"units": "kg/m^2/s", "bounds": (0.02, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "Emanuel (1991) stability cap", "shape": None},
            "alpha_closure": {"units": "1", "bounds": (0.05, 0.6), "tunable_tier": 1, "transform": "sigmoid", "category": "cape_closure", "reference": "Emanuel (1991) CONVECT v4.3c ALPHA", "shape": None},
            "c_l_emanuel": {"units": "J/kg/K", "bounds": (2000.0, 4500.0), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing", "reference": "Emanuel (1991) CONVECT v4.3c CL", "shape": None},
            "cape_threshold": {"units": "J/kg", "bounds": (23.1, 210.0), "tunable_tier": 0, "transform": "sigmoid", "category": "trigger", "reference": "tier 0 / AD-unreachable, see _CAPE_TRIGGER_AD_NOTE (#1417). Original: Emanuel (1991)", "shape": None},
            "cbmf_carry_max": {"units": "kg/m^2/s", "bounds": (0.099, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "mass_flux", "reference": "Emanuel (1991) anti-runaway guard", "shape": None},
            "cu_coefficient": {"units": "1", "bounds": (0.231, 2.1), "tunable_tier": 1, "transform": "sigmoid", "category": "entrainment", "reference": "Emanuel (1991) alpha entrainment scale", "shape": None},
            "damp_coefficient": {"units": "1", "bounds": (0.0, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "cape_closure", "reference": "Emanuel (1991) CONVECT v4.3c DAMP", "shape": None},
            "delta_0": {"units": "1/m", "bounds": (6.6e-05, 0.0006), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Emanuel (1991) bulk plume", "shape": None},
            "downdraft_efficiency": {"units": "1", "bounds": (0.0, 0.6), "tunable_tier": 2, "transform": "sigmoid", "category": "downdraft", "reference": "Emanuel (1991) downdraft re-evaporation", "shape": None},
            "dtmax": {"units": "K", "bounds": (0.297, 2.7), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Emanuel (1991) CONVECT v4.3c DTMAX", "shape": None},
            "elcrit": {"units": "kg/kg", "bounds": (0.000363, 0.0033), "tunable_tier": 1, "transform": "sigmoid", "category": "precipitation_efficiency", "reference": "Emanuel (1991) CONVECT v4.3c ELCRIT", "shape": None},
            "entp": {"units": "1", "bounds": (0.495, 4.5), "tunable_tier": 2, "transform": "sigmoid", "category": "entrainment", "reference": "Emanuel (1991) CONVECT v4.3c ENTP", "shape": None},
            "mse_min_search_offset": {"units": "Pa", "bounds": (16500.0, 150000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "numerics", "reference": "Emanuel (1991) source-level search offset", "shape": None},
            "parcel_perturb_T": {"units": "K", "bounds": (0.0, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "trigger", "reference": "Emanuel (1991) sub-cloud perturbation", "shape": None},
            "parcel_perturb_q": {"units": "kg/kg", "bounds": (0.0, 0.003), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Emanuel (1991) sub-cloud perturbation", "shape": None},
            "precip_efficiency_max": {"units": "1", "bounds": (0.5, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "precipitation_efficiency", "reference": "Emanuel (1991) CONVECT v4.3c EPMAX", "shape": None},
            "sij_upper_gate": {"units": "1", "bounds": (0.297, 2.7), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing", "reference": "Emanuel (1991) CONVECT v4.3c SIJ upper bound", "shape": None},
            "tlcrit": {"units": "degC", "bounds": (-165.0, -18.15), "tunable_tier": 2, "transform": "sigmoid", "category": "precipitation_efficiency", "reference": "Emanuel (1991) CONVECT v4.3c TLCRIT", "shape": None},
        },
    },
    "KainFritschConfig": {
        "scheme_key": "atm.conv.KainFritschConfig",
        "excluded": {
            "precip_efficiency": "default 0 = disabled (legacy no rain-split, shared split_convective_rain gated `if > 0.0`); enable + retune via config, not sigmoid-trained from the off state",
            "cape_or_sharpness": "numerics: sigmoid sharpness on the CAPE-OR fallback trigger",
            "cape_sharpness": "numerics: sigmoid sharpness on the CAPE gate",
            "cape_threshold": "default 0 = disabled/off (KF gates on the trigger function, not CAPE)",
            "cloud_depth_min": "trigger: fixed shallow/deep cloud-depth split (not sigmoid-tunable, fix via config)",
            "cloud_depth_sharpness": "numerics: sigmoid sharpness on the deep/shallow blend",
            "cloud_top_detrainment_fraction": "fixed KF-Eta cloud-top detrainment-level fraction (structural detrainment profile, not a tunable closure rate)",
            "cloud_top_detrainment_width_levels": "numerics: smoothing width [levels] of the cloud-top detrainment taper",
            "condload_fresh_retention_fraction": "fixed KF CONDLOAD retained fresh-condensate fraction",
            "dtlcl_pos_sharpness": "numerics: sharpness of the outer positive-part on the DTLCL base",
            "epsilon_0": "entrainment: legacy constant rate, superseded by the faithful radius-based profile",
            "rh_trigger_rhmax": "fixed KF-Eta trigger-3 upper RH branch threshold",
            "rh_trigger_slope": "fixed KF-Eta trigger-3 mid-RH branch coefficient",
            "rh_trigger_u00": "fixed KF-Eta trigger-3 RH threshold coefficient",
            "trigger_sharpness": "numerics: sigmoid sharpness on the trigger threshold",
            "wkl_floor": "numerics: cube-root base floor keeping the DTLCL gradient finite at WKL->0",
            "wkl_softplus_sharpness": "numerics: softplus sharpness inside the DTLCL surrogate",
            "theta_implicit": "numerics: off-centering of the implicit_flux backward-Euler subsidence solve (stability, iteration-coupled; clamped to [0.5,1.0], not trainable)",
        },
        "params": {
            "M_b_max": {"units": "kg/m^2/s", "bounds": (0.02, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "Kain & Fritsch (1990) stability cap", "shape": None},
            "cape_consumption_time": {"units": "s", "bounds": (600.0, 5400.0), "tunable_tier": 1, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Kain (2004) CAPE-removal timescale", "shape": None},
            "cape_or_threshold": {"units": "J/kg", "bounds": (660.0, 6000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kain & Fritsch (1990) CAPE-OR fallback", "shape": None},
            "cape_or_w_ref": {"units": "m/s", "bounds": (0.0066, 0.06), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kain & Fritsch (1990) CAPE-OR fallback", "shape": None},
            "cape_removal_fraction": {"units": "1", "bounds": (0.297, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cape_closure", "reference": "Kain (2004) FABE residual-CAPE fraction", "shape": None},
            "delta_0": {"units": "1/m", "bounds": (6.6e-05, 0.0006), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Kain & Fritsch (1990) bulk plume", "shape": None},
            "dtlcl_coeff": {"units": "K", "bounds": (1.5312, 13.92), "tunable_tier": 2, "transform": "sigmoid", "category": "trigger", "reference": "Kain (2004) Eq. 1 DTLCL coefficient", "shape": None},
            "dtlcl_dx_scale": {"units": "1", "bounds": (0.33, 3.0), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kain (2004) grid-length scaling", "shape": None},
            "dtlcl_exponent": {"units": "1", "bounds": (0.1089, 0.99), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kain (2004) Eq. 1 DTLCL exponent", "shape": None},
            "entrain_const": {"units": "1", "bounds": (0.0099, 0.09), "tunable_tier": 1, "transform": "sigmoid", "category": "entrainment", "reference": "Kain (2004) Eq. 5-6 radius entrainment", "shape": None},
            "parcel_perturb_T": {"units": "K", "bounds": (0.0, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "trigger", "reference": "Kain & Fritsch (1990) sub-cloud perturbation", "shape": None},
            "parcel_perturb_q": {"units": "kg/kg", "bounds": (0.0, 0.003), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kain & Fritsch (1990) sub-cloud perturbation", "shape": None},
            "rad_max_m": {"units": "m", "bounds": (660.0, 6000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "entrainment", "reference": "Kain (2004) updraft-radius upper bound", "shape": None},
            "rad_min_m": {"units": "m", "bounds": (330.0, 3000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "entrainment", "reference": "Kain (2004) updraft-radius lower bound", "shape": None},
            "rad_wkl_ref": {"units": "m/s", "bounds": (0.033, 0.3), "tunable_tier": 3, "transform": "sigmoid", "category": "entrainment", "reference": "Kain (2004) updraft-radius WKL ramp", "shape": None},
            "timec_max_s": {"units": "s", "bounds": (1188.0, 10800.0), "tunable_tier": 2, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Kain (2004) TIMEC clamp upper bound", "shape": None},
            "timec_min_s": {"units": "s", "bounds": (594.0, 5400.0), "tunable_tier": 2, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Kain (2004) TIMEC clamp lower bound", "shape": None},
            "usl_depth_pa": {"units": "Pa", "bounds": (1650.0, 15000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "cloud_base", "reference": "Kain (2004) DPMIN updraft-source-layer depth", "shape": None},
            "w_thresh_offset": {"units": "K", "bounds": (0.0, 6.0), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kain & Fritsch (1990) legacy linear trigger", "shape": None},
            "w_thresh_scale": {"units": "K/(m/s)", "bounds": (0.33, 3.0), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kain & Fritsch (1990) legacy linear trigger", "shape": None},
            "wklcl_ref": {"units": "m/s", "bounds": (0.0066, 0.06), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kain (2004) Eq. 2 LCL-height velocity threshold", "shape": None},
            "wklcl_zref": {"units": "m", "bounds": (660.0, 6000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kain (2004) Eq. 2 LCL-height reference", "shape": None},
        },
    },
    "KuoConfig": {
        "scheme_key": "atm.conv.KuoConfig",
        "excluded": {
            "cvgu_activation_scale": "numerics: tanh activation scale of the 0-1 convective_mask DIAGNOSTIC (no tendency effect; cvgu ~1e-4 kg/m^2/s when firing)",
            "icond_sharpness": "numerics: sigmoid sharpness on the smooth icond activation gates",
            "ptenq_sign_floor": "numerics: division-by-zero guard in the scale-free ptenq sign",
            "qv_min": "numerics: in-cloud vapor / LCL-detection floor",
            "zint_floor": "numerics: safety floor on the |zint| normalisation denominator",
        },
        "params": {
            "anthes_rh_offset": {"units": "1", "bounds": (0.0, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "cape_closure", "reference": "Kuo-Anthes (1977) moistening split offset", "shape": None},
            "buoyancy_scale_K": {"units": "K", "bounds": (0.033, 0.3), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kuo (1974) buoyancy-gate normalisation", "shape": None},
            "entrainment": {"units": "1/m", "bounds": (1.65e-05, 0.00015), "tunable_tier": 1, "transform": "sigmoid", "category": "entrainment", "reference": "Kuo (1974) Mahfouf reference Eps", "shape": None},
            "supersat_scale": {"units": "kg/kg", "bounds": (3.3e-06, 3e-05), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Kuo (1974) condensation-gate normalisation", "shape": None},
        },
    },
    "MassFluxConfig": {
        "scheme_key": "atm.conv.MassFluxConfig",
        "excluded": {
            "precip_efficiency": "default 0 = disabled (legacy no rain-split, shared split_convective_rain gated `if > 0.0`); enable + retune via config, not sigmoid-trained from the off state",
            "M_c_init": "default 0 = disabled/off (initial base mass flux, enable via config, not training)",
            "epsilon_0": "entrainment: bulk-plume base rate held fixed in the prognostic mass-flux core",
            "theta_implicit": "numerics: off-centering of the implicit_flux backward-Euler subsidence solve (stability, iteration-coupled; clamped to [0.5,1.0], not trainable)",
        },
        "params": {
            "M_b_max": {"units": "kg/m^2/s", "bounds": (0.02, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "Arakawa & Wu (2013) stability cap", "shape": None},
            "M_scale": {"units": "kg/m^2/s", "bounds": (0.0033, 0.03), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "Arakawa & Wu (2013) equilibrium mass-flux scale", "shape": None},
            "cape_activation_scale": {"units": "J/kg", "bounds": (3.3, 30.0), "tunable_tier": 2, "transform": "sigmoid", "category": "trigger", "reference": "Arakawa & Wu (2013) scheme default", "shape": None},
            "cape_threshold": {"units": "J/kg", "bounds": (23.1, 210.0), "tunable_tier": 0, "transform": "sigmoid", "category": "trigger", "reference": "tier 0 / AD-unreachable, see _CAPE_TRIGGER_AD_NOTE (#1417). Original: Arakawa & Wu (2013) scheme default", "shape": None},
            "delta_0": {"units": "1/m", "bounds": (0.00033, 0.003), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Arakawa & Wu (2013) bulk plume", "shape": None},
            "tau_adj": {"units": "s", "bounds": (1188.0, 10800.0), "tunable_tier": 1, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Arakawa & Wu (2013) mass-flux relaxation", "shape": None},
        },
    },
    "SBMConfig": {
        "scheme_key": "atm.conv.SBMConfig",
        "excluded": {
            "cloud_mask_sharpness": "numerics: sigmoid sharpness on the cloud-layer (T_moist - T) mask",
            "smooth_trigger_sharpness": "numerics: sigmoid sharpness on the smooth CAPE trigger",
        },
        "params": {
            "cape_threshold": {"units": "J/kg", "bounds": (23.1, 210.0), "tunable_tier": 0, "transform": "sigmoid", "category": "trigger", "reference": "tier 0 / AD-unreachable, see _CAPE_TRIGGER_AD_NOTE (#1417). Original: Frierson (2007)", "shape": None},
            "rh_ref": {"units": "1", "bounds": (0.4, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "cape_closure", "reference": "Frierson (2007) reference RH", "shape": None, "legacy_name": "sbm_RH_ref"},
            "tau_c": {"units": "s", "bounds": (1800.0, 21600.0), "tunable_tier": 1, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Frierson (2007) relaxation timescale", "shape": None, "legacy_name": "sbm_tau_c"},
        },
    },
    "TiedtkeConfig": {
        "scheme_key": "atm.conv.TiedtkeConfig",
        "excluded": {
            "cape_sharpness": "numerics: sigmoid sharpness on the CAPE gate",
            "depth_split_sharpness": "numerics: sigmoid sharpness on the deep/shallow depth blend",
            "downdraft_RH_min": "trigger: column-mean RH threshold below which the downdraft fires (not sigmoid-tunable, fix via config)",
            "downdraft_rh_sharpness": "numerics: sigmoid sharpness on the downdraft RH trigger [1/RH-fraction]",
            "epsilon_deep": "entrainment: deep-branch base rate held fixed in-scheme",
            "epsilon_midlevel": "entrainment: mid-level base rate held fixed in-scheme",
            "epsilon_shallow": "entrainment: shallow-branch base rate held fixed in-scheme",
            "lcl_membership_sharpness": "numerics: sigmoid sharpness on the below-LCL level membership [1/level index]",
            "moisture_convergence_sharpness": "numerics: sigmoid sharpness on the MC-proxy threshold",
            "parcel_dT": "trigger: fixed sub-cloud parcel temperature perturbation",
            "precip_efficiency": "default 0 = disabled (legacy no rain-split, gated `if > 0.0` in tiedtke.py); enable + retune via config, not sigmoid-trained from the off state",
            "theta_implicit": "numerics: off-centering of the implicit_flux backward-Euler subsidence solve (stability, iteration-coupled; clamped to [0.5,1.0], not trainable)",
        },
        "params": {
            "autoconv_q_c_crit": {"units": "kg/kg", "bounds": (1.0e-4, 2.0e-3), "tunable_tier": 2, "transform": "sigmoid", "category": "precipitation_efficiency", "reference": "Sundqvist (1978) autoconversion critical cloud water", "shape": None},
            "autoconv_pe_max": {"units": "1", "bounds": (0.5, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "precipitation_efficiency", "reference": "convective precip-efficiency ceiling (Sundqvist 1978 form)", "shape": None},
            "M_b_max": {"units": "kg/m^2/s", "bounds": (0.02, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "Tiedtke (1989) stability cap", "shape": None},
            "cape_threshold": {"units": "J/kg", "bounds": (23.1, 210.0), "tunable_tier": 0, "transform": "sigmoid", "category": "trigger", "reference": "tier 0 / AD-unreachable, see _CAPE_TRIGGER_AD_NOTE (#1417). Original: Tiedtke (1989)", "shape": None},
            "cloud_depth_deep": {"units": "m", "bounds": (1500.0, 5000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "updraft", "reference": "Tiedtke (1989) depth split", "shape": None},
            "cloud_depth_shallow_max": {"units": "m", "bounds": (800.0, 2500.0), "tunable_tier": 2, "transform": "sigmoid", "category": "updraft", "reference": "Tiedtke (1989) depth split", "shape": None},
            "cmt_c_d": {"units": "1", "bounds": (0.0, 2.0), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing", "reference": "Gregory et al. (1997) CMT", "shape": None},
            "cmt_c_u": {"units": "1", "bounds": (0.0, 2.0), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing", "reference": "Gregory et al. (1997) CMT", "shape": None},
            "delta_deep": {"units": "1/m", "bounds": (3.3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Tiedtke (1989) deep branch", "shape": None},
            "delta_midlevel": {"units": "1/m", "bounds": (6.6e-05, 0.0006), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Tiedtke (1989) mid-level branch", "shape": None},
            "delta_shallow": {"units": "1/m", "bounds": (9.9e-05, 0.0009), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Tiedtke (1989) shallow branch", "shape": None},
            "downdraft_alpha": {"units": "1", "bounds": (0.0, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "downdraft", "reference": "Tiedtke (1989) downdraft", "shape": None},
            "downdraft_evap_efficiency": {"units": "1", "bounds": (0.0, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "downdraft", "reference": "Tiedtke (1989) downdraft", "shape": None},
            "mc_proxy_RH_crit": {"units": "1", "bounds": (0.3, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "trigger", "reference": "Tiedtke (1989) moisture-convergence proxy", "shape": None},
            "midlevel_M_b_fraction": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "Tiedtke (1989) mid-level mass-flux fraction", "shape": None},
            "moisture_convergence_threshold": {"units": "kg/kg/s", "bounds": (3.3e-09, 3e-08), "tunable_tier": 2, "transform": "sigmoid", "category": "trigger", "reference": "Tiedtke (1989) MC-proxy threshold", "shape": None},
            "parcel_dq": {"units": "kg/kg", "bounds": (0.0, 0.003), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Tiedtke (1989) sub-cloud perturbation", "shape": None},
            "tau_MC_proxy": {"units": "s", "bounds": (1188.0, 10800.0), "tunable_tier": 2, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Tiedtke (1989) MC-proxy timescale", "shape": None},
            "tau_M_u_relax": {"units": "s", "bounds": (600.0, 5400.0), "tunable_tier": 2, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Tiedtke (1989) profile relaxation", "shape": None},
            "tau_shallow_M_b": {"units": "s", "bounds": (1188.0, 10800.0), "tunable_tier": 1, "transform": "sigmoid", "category": "cape_closure", "reference": "Tiedtke (1989) shallow cloud-base mass-flux timescale", "shape": None},
        },
    },
    "ZhangMcFarlaneConfig": {
        "scheme_key": "atm.conv.ZhangMcFarlaneConfig",
        "excluded": {
            "precip_efficiency": "default 0 = disabled (legacy no rain-split, shared split_convective_rain gated `if > 0.0`); enable + retune via config, not sigmoid-trained from the off state",
            "cape_sharpness": "numerics: sigmoid sharpness on the CAPE trigger",
            "epsilon_0": "entrainment: bulk-plume base rate held fixed (dilute CAPE uses dmpdz)",
            "parcel_dT": "trigger: fixed sub-cloud parcel temperature perturbation",
            "parcel_tpert": "default 0 = disabled/off (optional PBL temperature perturbation)",
            "tp_fac": "default 0 = disabled/off (PBL-perturbation multiplier)",
            "theta_implicit": "numerics: off-centering of the implicit_flux backward-Euler subsidence solve (stability, iteration-coupled; clamped to [0.5,1.0], not trainable)",
        },
        "params": {
            "M_b_max": {"units": "kg/m^2/s", "bounds": (0.02, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "mass_flux", "reference": "Zhang & McFarlane (1995) stability cap", "shape": None},
            "cape_threshold": {"units": "J/kg", "bounds": (23.1, 210.0), "tunable_tier": 0, "transform": "sigmoid", "category": "trigger", "reference": "tier 0 / AD-unreachable, see _CAPE_TRIGGER_AD_NOTE (#1417). Original: Zhang & McFarlane (1995)", "shape": None},
            "cmt_c_d": {"units": "1", "bounds": (0.0, 1.65), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing", "reference": "Gregory et al. (1997) CMT", "shape": None},
            "cmt_c_u": {"units": "1", "bounds": (0.0, 1.65), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing", "reference": "Gregory et al. (1997) CMT", "shape": None},
            "delta_0": {"units": "1/m", "bounds": (0.00033, 0.003), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Zhang & McFarlane (1995) bulk plume", "shape": None},
            "dmpdz": {"units": "1/m", "bounds": (-0.003, -0.00033), "tunable_tier": 1, "transform": "sigmoid", "category": "entrainment", "reference": "Zhang & McFarlane (1995) dilute-parcel entrainment (Raymond-Blyth 1992)", "shape": None},
            "parcel_dq": {"units": "kg/kg", "bounds": (0.0, 0.003), "tunable_tier": 3, "transform": "sigmoid", "category": "trigger", "reference": "Zhang & McFarlane (1995) sub-cloud perturbation", "shape": None},
            "pbl_top_pa": {"units": "Pa", "bounds": (50000.0, 90000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "cloud_base", "reference": "Zhang & McFarlane (1995) launch-level search bound", "shape": None},
            "tau_cape": {"units": "s", "bounds": (1188.0, 10800.0), "tunable_tier": 1, "transform": "sigmoid", "category": "relaxation_timescale", "reference": "Zhang & McFarlane (1995) CAPE relaxation", "shape": None},
            "tiedke_add": {"units": "K", "bounds": (0.0, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "trigger", "reference": "Zhang & McFarlane (1995) cloud-level buoyancy offset", "shape": None},
        },
    },
}


class SBMConfig(NamedTuple):
    """Configuration for Simplified Betts-Miller convection.

    Fields
    ------
    tau_c : float
        Relaxation timescale [s] (default 7200 = 2 hours).
    rh_ref : float
        Reference relative humidity for moisture profile (default 0.7).
    cape_threshold : float
        Minimum CAPE [J/kg] to trigger convection (default 70.0).
    smooth_trigger_sharpness : float
        Sigmoid sharpness for smooth trigger [1/(J/kg)] (default 0.01).
    """
    tau_c: float = 7200.0
    rh_ref: float = 0.7
    cape_threshold: float = 70.0
    # Default 0.1 (was 0.01).  At CAPE=0 the looser 0.01 gives
    # ``sigmoid(0.01·-70) ≈ 0.33`` (33 % activation when CAPE is zero
    # — gating leaks).  0.1 gives ``sigmoid(-7) ≈ 9e-4`` (effectively 0)
    # while preserving smoothness near the threshold.
    smooth_trigger_sharpness: float = 0.1
    # Sigmoid sharpness for the cloud-layer mask (T_moist - T) [1/K].
    # Hard ``T_moist >= T`` boolean masks kill ``jax.grad`` through layer
    # top/bottom transitions; the smoothed sigmoid keeps gradients alive.
    # 5/K gives ~0.5 at T_moist == T and ~0.01 at T_moist - T = -1 K, which
    # is sharp enough to behave like a hard mask in forward integration but
    # differentiable for training.
    cloud_mask_sharpness: float = 5.0


class AhmedNeelinDCAConfig(NamedTuple):
    """Configuration for the Ahmed-Neelin-Adames (2020) DCA closure.

    Lower-tropospheric-buoyancy (B_L) precipitation-buoyancy closure of
    Ahmed, Adames & Neelin (2020), *Deep Convective Adjustment of
    Temperature and Moisture*, J. Atmos. Sci. 77, 2163-2186 (hereafter
    ANA20).  Distinct from the Manabe-style pairwise moist-adiabatic
    adjustment (the default ``DCAConfig`` path): convection here is driven
    by the empirical eq-(8) precipitation–buoyancy closure
    ``P = a·(B_L − B_c)+``, with the implied column latent heating
    partitioned heating-up / drying-down along the eq-(41) direction
    (slope −1 in the q̂–T̂ plane) so column-integrated moist static energy
    is conserved and ``B_L`` is driven toward the QE line ``B_L = B_c``.
    The column relaxation time scale is emergent; ``tau_adjust_s`` is a
    reference value (the paper's ≈2 h, ANA20 eq 42) and is NOT the
    operative rate (see its field doc).

    Two-layer construction (ANA20 §2, Table 1):

    * Boundary layer (BL): surface to ``p_bl_top_pa`` (Δp_B = 150 hPa).
    * Lower free troposphere (LFT): ``p_bl_top_pa`` to ``p_lft_top_pa``
      (Δp_L = 350 hPa).

    Moist enthalpy ``e = T + (L_v/c_p)·q`` [K] (ANA20 after eq 6); the
    saturation version ``e* = T + (L_v/c_p)·q*`` uses the model's own
    saturation specific humidity (``legoesm.thermo``).  The buoyancy (eq 7):

        B_L = (g·Π_L/e_L*)·[ w_b·(e_B/Π_B) + w_L·(e_L/Π_L) − e_L*/Π_L ]

    where Π(p) = (p/p0)^κ is the Exner function (p0 = ``constants.p_ref``).
    The partial derivatives of this form reproduce ANA20 eqs (16)-(17)
    exactly (validated).

    Precipitation (eq 8):  P = a·(B_L − B_c)·H(B_L − B_c), with the
    Heaviside smoothed to a softplus for differentiability.

    Fields
    ------
    w_b : float
        Boundary-layer weight w_B in B_L (ANA20 Table 1, 0.52).
    w_l : float
        Lower-free-troposphere weight w_L = 1 − w_B (0.48).
    a_mm_per_hr : float
        Slope ``a`` of the P–B_L line [mm h⁻¹ (m s⁻²)⁻¹] (Table 1, 0.6).
        Converted internally to SI mass flux [kg m⁻² s⁻¹ per m s⁻²].
    b_c : float
        Critical buoyancy B_c [m s⁻²] (Table 1, −1.5e-2).
    tau_adjust_s : float
        Reference convective adjustment time scale τ_c [s] (ANA20 eq 42,
        ≈2 h → 7200 s).  NOTE: the operative precipitation rate is the
        empirical eq-(8) closure ``a·(B_L − B_c)``, NOT this τ; the paper's
        τ_c is itself derived from ``a`` and the observational EOF vertical
        structures (eqs 25-27) that are unavailable in-model, so the
        column relaxation time emerges from eq (8) and is generally longer
        than this nominal value.  Retained for reference / diagnostics.
    p_bl_top_pa : float
        Pressure at the top of the boundary layer [Pa].  The BL spans
        ``p_s`` → ``p_bl_top_pa``; Δp_B ≈ 150 hPa for ``p_s ≈ 1000 hPa``.
        Default 8.5e4 (850 hPa).
    p_lft_top_pa : float
        Pressure at the top of the lower free troposphere [Pa]; the LFT
        spans ``p_bl_top_pa`` → ``p_lft_top_pa`` (≈500 hPa).  Default 5.0e4.
    layer_edge_width_pa : float
        Half-width [Pa] of the smooth (sigmoid) layer-membership
        transition at each layer edge.  A hard pressure mask kills
        ``jax.grad`` through the BL/LFT boundaries; the sigmoid keeps the
        layer averages differentiable.  Default 2.5e3 (25 hPa).
    layer_min_depth_pa : float
        Minimum depth [Pa] each of the BL and LFT layers is guaranteed to
        retain.  Over high topography / low surface pressure a fixed
        850/500-hPa layer top can sit above the surface, leaving the layer
        empty and ``e_B/Π_B`` meaningless; the BL top is clamped to
        ``p_s − layer_min_depth_pa`` and the LFT top to
        ``p_bl_top − layer_min_depth_pa`` so both layers always have mass.
        Inactive on a standard ``p_s ≈ 1000 hPa`` column.  Default 5.0e3
        (50 hPa).
    precip_heaviside_sharpness : float
        Sharpness [(m s⁻²)⁻¹] of the softplus that smooths the eq-8
        Heaviside ``H(B_L − B_c)``.  Large so the forward precip tracks
        the ramp closely while keeping the gradient alive near B_c.
        Default 5.0e2.
    """
    w_b: float = 0.52
    w_l: float = 0.48
    a_mm_per_hr: float = 0.6
    b_c: float = -1.5e-2
    tau_adjust_s: float = 7200.0
    p_bl_top_pa: float = 8.5e4
    p_lft_top_pa: float = 5.0e4
    layer_edge_width_pa: float = 2.5e3
    layer_min_depth_pa: float = 5.0e3
    precip_heaviside_sharpness: float = 5.0e2


class DCAConfig(NamedTuple):
    """Configuration for Deep Convective Adjustment.

    Two variants are routed by the ``variant`` field:

    * ``variant="manabe"`` (default): the Manabe-Smagorinsky-Strickler
      (1965) pairwise moist-adiabatic adjustment (the historical legoESM
      ``dca`` scheme).  Behaviour of all existing configs is unchanged.
    * ``variant="ahmed_neelin"``: the Ahmed-Neelin-Adames (2020)
      lower-tropospheric-buoyancy (B_L) precipitation-buoyancy closure
      (see :class:`AhmedNeelinDCAConfig`).

    Fields
    ------
    variant : str
        ``"manabe"`` (default) or ``"ahmed_neelin"``.
    n_iterations : int
        (Manabe only) Number of bottom-to-top adjustment sweeps per call
        (default 3).  A single sweep only partially relaxes a deep column
        toward the moist adiabat, so one call per physics step leaves the
        free troposphere several K too cold under steady radiative
        cooling; free-tropospheric ``mean|T - T_moist|`` falls
        monotonically with sweeps (≈9.0/8.4/7.3/5.5 K at 1/3/5/10) with
        the column maximum temperature unchanged (the simultaneous pair
        solve keeps every sweep enthalpy-conserving and bounded).
    mixing_fraction : float
        (Manabe only) Fraction of adjustment applied per iteration
        (default 1.0).
    cape_threshold : float
        (Manabe only) Minimum CAPE [J/kg] to trigger convection (default
        100.0).  Columns with CAPE below this are not adjusted.
    cape_sharpness : float
        (Manabe only) Sigmoid sharpness [1/(J/kg)] for smooth CAPE gating
        (default 0.1).
    instability_blend_sharpness : float
        (Manabe only) Dimensionless sigmoid sharpness on the
        superadiabatic-instability metric controlling per-pair adjustment
        blending inside the ``lax.scan`` sweep (default 10.0).  Lifted
        from a hardcoded literal so the trigger transition width is
        tunable.
    ahmed_neelin : AhmedNeelinDCAConfig
        Sub-configuration for the ``variant="ahmed_neelin"`` closure.
    """
    variant: str = "manabe"
    n_iterations: int = 3
    mixing_fraction: float = 1.0
    cape_threshold: float = 100.0
    cape_sharpness: float = 0.1   # sigmoid(-10)≈5e-5 at CAPE=0; 0.5 at threshold
    instability_blend_sharpness: float = 10.0
    ahmed_neelin: AhmedNeelinDCAConfig = AhmedNeelinDCAConfig()


class KuoConfig(NamedTuple):
    """Configuration for the canonical Kuo (1965) convection scheme.

    Faithful to J.-F. Mahfouf's reference Kuo implementation
    (AJFMAHFOUF/MOIST_CONVECTION_KUO, ``src/kuo_schemes.f90``).  The
    convective moisture/heat source is the **large-scale moisture
    convergence** ``cvgu = Σ max(0, ∂q/∂t|dyn) dp/g`` (a bounded
    external dynamical tendency), threaded in as ``moisture_convergence``
    — NOT the column supersaturation.  When no large-scale convergence
    is supplied (e.g. a pure single-column RCE), Kuo is correctly
    quiescent: it has no source to redistribute.

    Closure (Kuo-1965, the faithful default ``partition="kuo1965"``):

        dt/dt = cvgu/zint · (tc − t)
        dq/dt = −ptenq + cvgu/zint · (qvc − qv)
        zint  = Σ (qvc − qv + (tc − t)/alpha) dp/g

    over levels that are buoyant AND have positive vertical velocity at
    the LCL (``icond==2``) AND positive local convergence
    (``ptenq > 0``).  ``tc, qvc`` come from an entraining moist-adiabat
    ascent (entrainment ``Eps``) with condensation removed.

    The optional ``partition="anthes"`` Kuo-Anthes (1977) closure splits
    the source between heating ``(1 − bkuo)`` and moistening ``bkuo``
    with ``bkuo = clip(1 − RH_mean − rh_offset, 0, 1)`` — b is a
    moistening FRACTION, so it is clamped to [0, 1] (a near-saturated
    layer with ``RH_mean + rh_offset > 1`` moistens nothing rather than
    flipping the term into spurious extra drying).

    Fields
    ------
    entrainment : float
        Fractional entrainment rate ``Eps`` for the cloud parcel
        ascent [1/m].  Oracle value ``5.0e-5``.
    newton_iters : int
        Number of Newton iterations for the implicit moist-adiabat
        temperature solve per level (oracle uses 5).
    partition : str
        ``"kuo1965"`` (default, faithful) or ``"anthes"`` (Kuo-Anthes
        1977 RH-dependent heating/moistening split).
    anthes_rh_offset : float
        Offset in the Kuo-Anthes moistening parameter
        ``bkuo = (1 − RH_mean − anthes_rh_offset)`` (oracle uses 0.1).
    qv_min : float
        Floor on in-cloud vapor / LCL detection [kg/kg] (oracle uses
        ``1e-9``).
    icond_sharpness : float
        Dimensionless sigmoid sharpness for the smooth ``icond``
        activation gates (supersaturation, buoyancy, w_lcl>0, ptenq>0).
        Each gate argument is normalised to O(1) by the scales below
        before the sigmoid, so a single large dimensionless sharpness
        (default 50) makes every gate a crisp Heaviside approaching the
        oracle's hard ``if`` switches while staying differentiable.
    buoyancy_scale_K : float
        Normalisation scale [K] for the buoyancy gate ``tvc − tve`` —
        the sigmoid argument is ``(tvc − tve)/buoyancy_scale_K``.  A
        small value (0.1 K) keeps the buoyancy threshold sharp.
    supersat_scale : float
        Normalisation scale [kg/kg] for the condensation gate
        ``qv − qsat`` — argument ``(qv − qsat)/supersat_scale``.
    ptenq_sign_floor : float
        Division-by-zero guard [kg/kg/s] in the scale-free sign
        ``ptenq / (|ptenq| + floor)`` feeding the ``ptenq > 0``
        activation gate.  The oracle gate is a Heaviside on the SIGN of
        the local convergence (on for ANY positive value, regardless of
        magnitude); the scale-free sign makes the smooth gate ~1 across
        the whole convergent column down to its exponential tail and ~0
        only for clear subsidence.  1e-30 = pure numerical guard.
    zint_floor : float
        Safety floor [kg/m²] on the ``|zint|`` normalisation denominator
        so the closure is finite when the convective layer is empty.
    cvgu_activation_scale : float
        Activation scale [kg/m²/s] of the 0-1 ``convective_mask``
        DIAGNOSTIC ``tanh(cvgu / scale)``.  Typical firing columns have
        ``cvgu ~ 1e-4`` kg/m²/s, so the default 1e-8 saturates the mask
        to ~1 whenever the scheme fires while keeping ``tanh(0) = 0``
        on the quiescent path.  Diagnostic-only smoothing (no effect on
        tendencies) — matches the sibling ``buoyancy_scale_K`` /
        ``supersat_scale`` trigger-normaliser pattern.
    """
    entrainment: float = 5.0e-5
    newton_iters: int = 5
    partition: str = "kuo1965"
    anthes_rh_offset: float = 0.1
    qv_min: float = 1.0e-9
    icond_sharpness: float = 50.0
    buoyancy_scale_K: float = 0.1
    supersat_scale: float = 1.0e-5
    ptenq_sign_floor: float = 1.0e-30
    zint_floor: float = 1.0e-12
    cvgu_activation_scale: float = 1.0e-8


class MassFluxConfig(NamedTuple):
    """Configuration for Prognostic Mass-Flux convection (Arakawa-Wu type).

    Fields
    ------
    tau_adj : float
        Mass flux relaxation timescale [s].
    epsilon_0 : float
        Entrainment rate [1/m].
    delta_0 : float
        Detrainment rate [1/m].
    M_scale : float
        Equilibrium mass flux scale [kg/m^2/s].
    cape_activation_scale : float
        Sigmoid scale for CAPE trigger [J/kg].
    cape_threshold : float
        CAPE threshold [J/kg].
    M_c_init : float
        Initial base mass flux [kg/m^2/s].
    """
    tau_adj: float = 3600.0
    epsilon_0: float = 1e-3
    delta_0: float = 1e-3
    M_scale: float = 0.01
    # Trigger: ``convective_mask = sigmoid((cape − cape_threshold)
    # / cape_activation_scale)``.  Defaults below give ``sigmoid(-7)
    # ≈ 9e-4`` at CAPE=0 (effectively no firing) and full activation
    # 100 J/kg above the 70 J/kg threshold.  The earlier defaults
    # (``cape_threshold=0``, ``cape_activation_scale=100``) gave 50 %
    # activation at CAPE=0 — the gating-leak that the validation
    # script flags.
    cape_activation_scale: float = 10.0
    cape_threshold: float = 70.0
    precip_efficiency: float = 0.0  # shared split_convective_rain rain-split (default off = legacy)
    M_c_init: float = 0.0
    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
    # --- mass-flux kernel vertical solve (Tiedtke 1989 flux form / #824) ---
    # Selects the compensating-subsidence + detrainment solve in the SHARED
    # kernel ``mass_flux.apply_mass_flux_kernel``; see
    # ``BechtoldConfig.subsidence_solve`` for the full rationale and the
    # measured leak numbers.  ``"advective"`` (DEFAULT HERE — preserves this
    # scheme's shipped behaviour byte-for-byte) is the legacy donor-cell
    # advective form, which leaves a non-telescoping ``(phi/rho) dM/dz``
    # residual so column MSE / total water close only to truncation order.
    # ``"implicit_flux"`` selects the IMPLICIT (backward-Euler, theta-blended)
    # CONSERVATIVE flux-form solve: the flux divergence telescopes to the
    # vanishing top/base boundary flux (machine-precision column closure) and
    # the detrained condensate is paired with its vapor sink ``-dq_c``.
    # NOTE the two solves are NOT merely different discretisations of one
    # operator: implicit_flux additionally books that vapor sink, so switching
    # changes the column water budget ON PURPOSE.  Unknown values raise
    # ValueError at kernel entry (static Python str; dispatch-hardening).
    subsidence_solve: str = "advective"
    # Off-centering for the implicit_flux solve.  1.0 = fully implicit
    # (backward Euler); the kernel clamps to [0.5, 1.0].  Unused when
    # ``subsidence_solve == "advective"``.
    theta_implicit: float = 1.0


class ZhangMcFarlaneConfig(NamedTuple):
    """Configuration for the Zhang & McFarlane (1995) deep-convection scheme.

    Single-plume mass-flux scheme with CAPE-relaxation closure and
    optional Gregory et al. 1997 convective momentum transport.

    Fields
    ------
    tau_cape : float
        CAPE relaxation timescale [s] (default 3600 = 1 hour).
    cape_threshold : float
        CAPE threshold [J/kg] below which convection is suppressed
        (default 70.0, the classic ZM 1995 value).
    cape_sharpness : float
        Sigmoid sharpness on the CAPE trigger [1/(J/kg)].  Default
        ``0.1`` — ≈0.5 activation at threshold and ~95% activation
        ~30 J/kg above it.
    parcel_dT : float
        Sub-cloud parcel temperature perturbation [K] (default 0.5).
    parcel_dq : float
        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
    epsilon_0 : float
        Bulk-plume entrainment rate [1/m] (default 1e-3).
    delta_0 : float
        Bulk-plume detrainment rate [1/m] (default 1e-3).
    enable_cmt : bool
        Whether to compute convective momentum transport tendencies
        (default ``True``).  When ``False``, the bridge zero-fills
        ``du_dt_conv`` / ``dv_dt_conv``.
    cmt_c_u, cmt_c_d : float
        Pressure-gradient correction coefficients in the Gregory et
        al. 1997 closure (default 0.55 each — the canonical value).
    M_b_max : float
        Hard upper bound on the cloud-base mass flux ``M_b`` [kg/m²/s]
        (default 0.05 — about half the literature peak tropical value
        ~0.1; tighter than peak because the unbounded CAPE/tau closure
        can spike to ~2 kg/m²/s in a high-CAPE column and the per-layer
        heating ~M·(T_u−T)·δ scales linearly).  The
        CAPE/τ_cape closure is unbounded above; without this cap a
        column with CAPE >> 5 kJ/kg yields M_b that drives
        column-integrated heating > 10⁴ W/m² and blows up the
        integration on the next dynamics step.
    """
    tau_cape: float = 3600.0
    cape_threshold: float = 70.0
    cape_sharpness: float = 0.1
    parcel_dT: float = 0.5
    parcel_dq: float = 1.0e-3
    epsilon_0: float = 1.0e-3
    delta_0: float = 1.0e-3
    # --- FAITHFUL ZM dilute-parcel CAPE (Raymond-Blyth 1992) -----------
    # The ZM trigger and closure use the CAPE of a DILUTE entraining
    # plume (``buoyan_dilute``/``parcel_dilute`` in zm_conv.F90), not an
    # undilute moist adiabat.  ``dmpdz`` is the fractional entrainment
    # rate [1/m] (E3SM/CAM default ``−1.0e-3``; NEGATIVE by the oracle's
    # ``mp`` sign convention).  ``tiedke_add`` is the buoyancy offset [K]
    # added at every cloud level (oracle default 0.5 K).  ``tp_fac`` ×
    # ``parcel_tpert`` is the optional PBL temperature perturbation (both
    # default 0).  ``pbl_top_pa`` bounds the launch-level (max-MSE)
    # search to the PBL.  Set ``use_dilute_cape = False`` to recover the
    # legacy undilute moist-adiabat CAPE.
    use_dilute_cape: bool = True
    dmpdz: float = -1.0e-3
    tiedke_add: float = 0.5
    tp_fac: float = 0.0
    parcel_tpert: float = 0.0
    pbl_top_pa: float = 7.0e4
    enable_cmt: bool = True
    cmt_c_u: float = 0.55
    cmt_c_d: float = 0.55
    M_b_max: float = 0.05
    # Buoyancy-death memory is OFF by default.  The audit's cycle-2 P2
    # concern (a plume terminated by negative buoyancy can revive
    # above an inversion) is real, but several attempted detector
    # designs each introduced their own edge cases (sub-LCL warm-
    # bubble leak, miss of weak positive CAPE, miss of cloud-base
    # launch, growth of the launched gate during revival).  The
    # ``_plume.entraining_detraining_plume`` ``buoyancy_death_memory``
    # kwarg is wired through and the option is fully tested in
    # isolation, but enabling it by default would require a more
    # robust state-machine design plus a dedicated validation
    # campaign.  Schemes that *want* the single-plume monotone
    # termination can opt in by setting this to True; the default
    # preserves the legacy local-only filter behaviour that all
    # existing scheme test fixtures were calibrated against.
    buoyancy_death_memory: bool = False
    precip_efficiency: float = 0.0  # shared split_convective_rain rain-split (default off = legacy)
    # --- mass-flux kernel vertical solve (Tiedtke 1989 flux form / #824) ---
    # See ``MassFluxConfig.subsidence_solve`` (identical semantics; the shared
    # kernel is ``mass_flux.apply_mass_flux_kernel``).  ``"advective"`` is the
    # DEFAULT HERE and preserves this scheme's shipped behaviour byte-for-byte;
    # ``"implicit_flux"`` is the conservative flux-form solve that also books
    # the ``-dq_c`` vapor sink.  Unknown values raise at kernel entry.
    subsidence_solve: str = "advective"
    theta_implicit: float = 1.0


class KainFritschConfig(NamedTuple):
    """Configuration for the Kain & Fritsch (1990, 2004 update) scheme.

    Single-plume bulk mass-flux scheme distinguished by its
    boundary-layer trigger function: convection fires when the
    perturbed parcel temperature at the LCL exceeds the environmental
    temperature at the LCL.  By default (``faithful_trigger=True``) the
    perturbation is the Fritsch-Chappell w-dependent ``DTLCL`` of Kain
    (2004) — see the "Faithful KF-Eta" fields below; the legacy linear
    ``w_thresh_offset/w_thresh_scale`` trigger is used only when
    ``faithful_trigger=False``.  The trigger is smoothed via a sigmoid
    (``trigger_sharpness``) to preserve gradients.  Deep-vs-shallow
    cloud branches are blended on cloud depth.  No convective
    momentum transport — KF emits ``du_dt_conv = dv_dt_conv = None``.

    Fields
    ------
    w_thresh_offset : float
        LEGACY trigger offset [K] (default 2.0).  Used ONLY when
        ``faithful_trigger=False``; the faithful trigger uses ``DTLCL``.
    w_thresh_scale : float
        LEGACY conversion factor from ``w_grid`` [m/s] to a temperature
        perturbation [K] in the trigger function (default 1.0 K per
        m/s).  Used ONLY when ``faithful_trigger=False``.
    trigger_sharpness : float
        Sigmoid sharpness on the trigger threshold [1/K].  Larger
        values approach a hard ``> 0`` step; smaller values broaden
        the transition.  Default 5.0 → ~95% activation 0.6 K above
        threshold.
    cape_consumption_time : float
        CAPE-removal timescale [s] (default 1800.0).
    parcel_perturb_T : float
        Sub-cloud parcel temperature perturbation [K] (default 0.5).
    parcel_perturb_q : float
        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
    epsilon_0 : float
        Bulk-plume entrainment rate [1/m] (default 2e-3).
    delta_0 : float
        Bulk-plume detrainment rate [1/m] (default 2e-3).
    cloud_depth_min : float
        Cloud-depth threshold [m] separating shallow and deep
        branches (default 4000.0).
    cloud_depth_sharpness : float
        Sigmoid sharpness on the deep/shallow blend [1/m] (default
        1e-3 — ~95% activation 1500 m above threshold).
    enable_shallow : bool
        Whether to include the shallow-cloud branch.  ``False``
        disables shallow tendencies regardless of cloud depth
        (default ``True``).
    cape_threshold : float
        CAPE threshold [J/kg] below which convection is gated off.
        Default 0.0 — KF gates primarily on the trigger function,
        not on CAPE.
    cape_sharpness : float
        Sigmoid sharpness on the CAPE gate [1/(J/kg)] (default 0.02).
    M_b_max : float
        Hard upper bound on the *applied* cloud-base mass flux ``M_b``
        [kg/m²/s] (default 0.05 — about 1/7 of the oracle deep-tropical
        ``M_b≈0.33``).  The cap is a stability bound on the unbounded
        CAPE/TIMEC closure, which can spike to ~1 kg/m²/s in a high-CAPE
        column where the per-layer heating ~M·(T_u−T)·δ scales linearly.
        The diagnostic carry packs the UNCAPPED closure ``M_b`` so it stays
        responsive to the trigger above the cap.
    """
    w_thresh_offset: float = 2.0
    w_thresh_scale: float = 1.0
    trigger_sharpness: float = 5.0
    cape_consumption_time: float = 1800.0
    parcel_perturb_T: float = 0.5
    parcel_perturb_q: float = 1.0e-3
    # Deep-convection entrainment/detrainment (~2e-4 /m).  The earlier
    # ``2e-3`` (shallow-cumulus range) over-diluted the single plume so it
    # lost buoyancy in the lower troposphere and could not warm the free
    # troposphere — see ZhangMcFarlaneConfig / EmanuelConfig.
    epsilon_0: float = 2.0e-4
    delta_0: float = 2.0e-4
    cloud_depth_min: float = 4000.0
    cloud_depth_sharpness: float = 1.0e-3
    enable_shallow: bool = True
    # Single-plume scheme — see ZhangMcFarlaneConfig.buoyancy_death_memory.
    buoyancy_death_memory: bool = False
    cape_threshold: float = 0.0
    cape_sharpness: float = 0.1
    # CAPE-based OR fallback for the dynamical trigger.  Kain-Fritsch fires
    # where resolved grid-scale ascent lifts a parcel past its LCL
    # (``T_lcl + w_thresh_scale·w_grid − w_thresh_offset > T_env``).  In a
    # single-column model (and any dycore that does not expose a divergence
    # operator) ``w_grid`` is zero, so the 2 K ``w_thresh_offset`` becomes a
    # permanent suppression and the scheme never fires — leaving the column
    # in near-radiative equilibrium.  When the undilute CAPE exceeds
    # ``cape_or_threshold`` the trigger fires regardless of ``w_grid``.  The
    # threshold is set to a deliberately EXTREME value (2000 J/kg, deep
    # maritime-tropical CAPE) with a tight sigmoid so the fallback is a
    # near-no-op for the moderate-CAPE columns of a 3-D run (≈0.007 at
    # 1000 J/kg, 0.5 at 2000) — those columns are handled by the resolved
    # w-trigger, and any 3-D column carrying ≳2000 J/kg essentially always
    # has the resolved ascent to satisfy it anyway.  The branch therefore
    # only rescues the ``w = 0`` SCM/idealised case (RCE CAPE ~ 10⁴ J/kg).
    # Set ``cape_or_threshold = inf`` to disable and recover the pure
    # w-trigger behaviour exactly.
    cape_or_threshold: float = 2000.0
    cape_or_sharpness: float = 0.005
    # The CAPE-OR fallback is itself gated by the ABSENCE of resolved
    # grid-scale ascent, ``exp(-(w_grid_at_lcl / cape_or_w_ref)^2)``, so it
    # only engages where ``w_grid ≈ 0`` (SCM, or a dycore with no
    # divergence operator).  Wherever the bridge supplies a non-negligible
    # ``w_grid`` (any 3-D run with resolved ascent) the gate →0 and the
    # pure w-trigger is used unchanged — so KF's documented response to
    # resolved divergence is preserved exactly.  ``cape_or_w_ref`` is the
    # vertical-velocity scale [m/s] at which the fallback is suppressed.
    cape_or_w_ref: float = 0.02
    M_b_max: float = 0.05
    # --- Faithful KF-Eta (Kain 2004) trigger / closure parameters -------
    # The default trigger is now the Fritsch-Chappell w-dependent temperature
    # perturbation DTLCL of Kain (2004), transcribed from WRF
    # ``module_cu_kfeta.F`` (oracle).  ``faithful_trigger=True`` selects it;
    # ``False`` recovers the legacy linear ``T_lcl + w_thresh_scale*w -
    # w_thresh_offset`` trigger for back-compat / existing tuning.
    faithful_trigger: bool = True
    # Updraft-source-layer (USL) depth [Pa] mass-weighted for the trigger
    # parcel.  Oracle DPMIN = 5e3 Pa (~50 hPa, the canonical KF source layer).
    usl_depth_pa: float = 5.0e3
    # DTLCL coefficient and exponent (Kain 2004 Eq. 1: DTLCL = c * WKL^p, K).
    dtlcl_coeff: float = 4.64
    dtlcl_exponent: float = 0.33
    # KF-Eta trigger-3 relative-humidity perturbation DTRH (WRF
    # module_cu_kfeta.F lines 996-1017).  The ETA branch uses U00=0.75:
    # humid LCL environments get an additional temperature perturbation
    # derived from the local saturation derivative.  This stays faithful to
    # the reference trigger path without imposing external SCM forcing.
    enable_rh_trigger_perturb: bool = True
    rh_trigger_u00: float = 0.75
    rh_trigger_rhmax: float = 0.95
    rh_trigger_slope: float = 0.25
    # KF CONDLOAD precipitation fallout (KF90 Eq. 9; WRF module_cu_kfeta.F
    # lines 2869-2923) lets only 60% of fresh condensate participate in
    # immediate conversion and retains the remaining 40% as cloud condensate.
    # The full WTW/load recursion is not in the shared plume helper, but this
    # reference fraction prevents treating all fresh updraft condensate as
    # retained grid cloud.
    condload_fresh_retention_fraction: float = 0.4
    # Reference vertical velocity for the LCL-height threshold (Kain 2004
    # Eq. 2: WKLCL = wklcl_ref * min(ZLCL, z_ref)/z_ref) [m/s] and [m].
    wklcl_ref: float = 0.02
    wklcl_zref: float = 2.0e3
    # Grid length the DTLCL formula is calibrated for [m] (Kain 2004: 25 km;
    # WKL scales w by DX/dtlcl_ref_dx).  The SCM/idealised bridge passes its
    # own ``w_grid`` already at-resolution, so dtlcl_dx_scale defaults to 1.
    dtlcl_dx_scale: float = 1.0
    # Tiny floor [m/s^(1/3) scale] inside the cube-root power so the base of
    # ``x^p`` (p<1, infinite slope at 0) never hits exactly 0, keeping the
    # gradient finite at WKL->0.  The DTLCL surrogate is
    # ``coeff*softplus_pos(g(WKL)^p - g(0)^p)`` (zero at the cutoff,
    # ~coeff*WKL^p above).  ``wkl_floor`` = 1e-8 makes the base-point term
    # ``g(0)^p`` ~0.04 (so DTLCL within ~7% of the oracle 4.64*WKL^0.33 for
    # tropical WKL~0.1-0.5) while staying C^1 everywhere.
    wkl_floor: float = 1.0e-8
    wkl_softplus_sharpness: float = 1.0e4
    # Sharpness [1/K^p] of the outer smooth positive-part on the DTLCL base
    # (with the ln2/k residual subtracted so DTLCL is ~0 at the WKL=0 cutoff).
    dtlcl_pos_sharpness: float = 1.0e3
    # Updraft-radius entrainment (Kain 2004 Eq. 5-6).  The oracle's
    # environmental inflow MASS rate is REI = VMFLCL * DP * entrain_const/RAD
    # with DP = rho*g*dz (Pa).  The bulk-plume needs the *fractional* rate
    # per unit HEIGHT, epsilon = (1/M) dM/dz = rho(z)*g*entrain_const/RAD
    # [1/m] — i.e. the rho*g factor converts the oracle's per-pressure inflow
    # into a per-height fractional rate (see _faithful_entrainment_profile).
    # RAD ramps 1000 m (WKL<=0) -> 2000 m (WKL>=0.1).
    faithful_entrainment: bool = True
    entrain_const: float = 0.03
    rad_min_m: float = 1.0e3
    rad_max_m: float = 2.0e3
    rad_wkl_ref: float = 0.1
    # Smooth surrogate for the oracle's cloud-top total-detrainment step
    # (WRF KF-Eta ``UDR(LTOP)=UMF(LTOP)+UDR(LTOP)-UER(LTOP)``): detrain this
    # fraction of the remaining updraft over the LNB-centred transition
    # layer.  0.99 is the differentiable finite-rate stand-in for total
    # detrainment; the width is in model levels.
    cloud_top_detrainment_fraction: float = 0.99
    cloud_top_detrainment_width_levels: float = 1.0
    # Convective (CAPE-removal) timescale bounds [s] (oracle TIMEC clamp
    # [1800, 3600]).  The SCM/idealised bridge does not expose the LCL/
    # mid-trop wind that sets TIMEC=DX/VCONV, so ``cape_consumption_time``
    # (above) is used as the operative TIMEC, clamped into these bounds.
    timec_min_s: float = 1800.0
    timec_max_s: float = 3600.0
    # Target residual-CAPE fraction of the closure (oracle FABE lands near
    # 1.05-STAB .. 0.95-STAB with STAB=0.95, i.e. ~5-10% residual; the
    # bulk one-pass closure removes CAPE over TIMEC so this is the nominal
    # fraction removed per call, used only for diagnostics/documentation).
    cape_removal_fraction: float = 0.90
    precip_efficiency: float = 0.0  # shared split_convective_rain rain-split (default off = legacy)
    # NOTE: a cloud-base-height precipitation-efficiency retention scaling was
    # removed (validator codex review-2 #1): under the ConvectionOutput
    # contract microphysics owns precipitation, so scaling the cloud-water
    # source down here leaked column water with no output channel to receive
    # the precipitating fraction.  The full detrained condensate is now handed
    # to microphysics, which applies precip efficiency via autoconversion.
    # --- mass-flux kernel vertical solve (Tiedtke 1989 flux form / #824) ---
    # See ``MassFluxConfig.subsidence_solve``.  Kain-Fritsch DEFAULTS to
    # ``"implicit_flux"`` (PR #988 hardcoded it at the call site; this promotes
    # the hardcoded value to a config field with the SAME default, so shipped
    # behaviour is unchanged and the matched-kernel campaign can address every
    # member of the family uniformly).  Selecting ``"advective"`` also disables
    # the paired detrained-condensate latent release, because the advective
    # solve does not debit vapor for that condensate.
    subsidence_solve: str = "implicit_flux"
    theta_implicit: float = 1.0


class EmanuelConfig(NamedTuple):
    """Configuration for the Emanuel (1991) buoyancy-sorting scheme.

    Single-plume mass-flux scheme with a buoyancy-sorted ensemble of
    mixed parcels: at every cloud level the parcel may mix with
    environmental air in a discrete spectrum of mixing fractions; the
    fractions with positive buoyancy continue to ascend while the
    fractions with negative buoyancy descend.  The smooth-everywhere
    formulation replaces the hard ascend/descend switch with a
    sigmoid weighting on buoyancy.

    Distinct from Zhang-McFarlane (single bulk plume) and Kain-Fritsch
    (single plume with deep/shallow blend) by the per-level
    distribution of detrainment that the buoyancy-sorted ensemble
    produces.

    Fields
    ------
    n_mixing_fractions : int
        Number of discrete mixing fractions in the buoyancy-sort
        ensemble.  Default 8 — Emanuel 1991 uses 50; the smaller
        value here is a cost / accuracy compromise.
    cu_coefficient : float
        Entrainment scale factor (Emanuel's α).  Default 0.7.
    precip_efficiency_water : float
        Precipitation efficiency above LCL (default 1.0).
    precip_efficiency_lcl : float
        Precipitation efficiency below LCL (default 0.0).
    cape_threshold : float
        CAPE gate [J/kg] (default 70.0).
    cape_sharpness : float
        Sigmoid sharpness on the CAPE gate [1/(J/kg)] (default 0.02).
    parcel_perturb_T : float
        Sub-cloud parcel temperature perturbation [K] (default 0.5).
    parcel_perturb_q : float
        Sub-cloud parcel humidity perturbation [kg/kg] (default 1e-3).
    enable_unsaturated_downdraft : bool
        Whether to include the unsaturated downdraft branch (rain
        evaporation cooling) (default ``True``).
    downdraft_efficiency : float
        Fraction of precipitation that re-evaporates below cloud base
        in the downdraft (default 0.2).
    smooth_trigger_sharpness : float
        Sigmoid sharpness on the buoyancy-sort weighting [1/K]
        (default 0.5).
    epsilon_0 : float
        Bulk-plume entrainment rate [1/m] (default 1.5e-3).
    delta_0 : float
        Bulk-plume detrainment rate [1/m] (default 1.5e-3).
    M_b_max : float
        Hard upper bound on the cloud-base mass flux ``M_b`` [kg/m²/s]
        (default 0.005 — about 1/20 of the literature peak tropical value 0.1; tighter than peak because the unbounded CAPE/tau closure can spike to ~2 kg/m²/s in a high-CAPE column and the per-layer heating ~M·(T_u−T)·δ scales linearly).
    """
    n_mixing_fractions: int = 8
    cu_coefficient: float = 0.7
    precip_efficiency_water: float = 1.0
    precip_efficiency_lcl: float = 0.0
    cape_threshold: float = 70.0
    cape_sharpness: float = 0.1
    parcel_perturb_T: float = 0.5
    parcel_perturb_q: float = 1.0e-3
    # --- Prognostic cloud-base mass-flux (CBMF) closure ----------------
    # FAITHFUL to oracle convect43c.f (CONVECT v4.3c).  CBMF is a
    # prognostic quantity relaxed each call toward the sub-cloud
    # quasi-equilibrium:
    #     CBMF = (1 - DAMP·dt/300)·CBMF_old + 0.1·ALPHA·DTMA   (≥0)
    # ``alpha_closure`` (ALPHA) and ``damp_coefficient`` (DAMP) are the
    # oracle's standard values 0.2 and 0.1 (DAMP < 1).  ``dtmax`` (DTMAX)
    # is the maximum negative temperature perturbation [K] a lifted
    # parcel is allowed below its LFC (oracle 0.9 K).
    alpha_closure: float = 0.2
    damp_coefficient: float = 0.1
    dtmax: float = 0.9
    # Pressure sharpness [1/Pa] for the differentiable mask that bounds
    # the DTPBL average to levels between the launch level NK and cloud
    # base ICB (CONVECT v4.3c lines 553-557).  1e-3 gives an O(1 kPa)
    # transition, matching the shared LCL crossing sharpness.
    lcl_pressure_sharpness: float = 1.0e-3
    # Sharpness [1/level] of the below-LCL smooth index indicator used by the
    # optional unsaturated-downdraft re-evaporation (surface-last: level index
    # > k_lcl_smooth ⇒ below LCL).  Default 2.0 gives an ~1-level transition,
    # matching the historical inline value.  Numerics (not a tunable closure).
    below_lcl_index_sharpness: float = 2.0
    # Sharpness [1/(kg/m²/s)] of the softplus positive-part applied to the
    # relaxed CBMF so it is ~0 when the relaxation target goes negative
    # (stable column) without a hard ``max`` that would kill the gradient.
    # Large because CBMF magnitudes are O(0.01-0.1) kg/m²/s.
    cbmf_positive_sharpness: float = 1.0e3
    # Upper bound [kg/m²/s] on the *carried* prognostic CBMF — looser than
    # the per-step transport cap ``M_b_max`` so the closure's memory can
    # ramp to the oracle's deep-tropical CBMF (~0.12 kg/m²/s) instead of
    # being frozen at ``M_b_max``.  Set well above the oracle peak; the
    # plume transport is still bounded at ``M_b_max``.  Acts only as an
    # anti-runaway guard for a persistently violently-unstable column.
    cbmf_carry_max: float = 0.3
    # Default-OFF.  Emanuel 1991's downdraft re-evaporates a fraction
    # of *precipitation* (rain) back to vapor in the BL.  In a model
    # without an explicit q_r tracer the implementation can only draw
    # from ``dq_c_conv_dt`` (the cloud-water source), so enabling it
    # produces a column-net moistening on CAPE-positive soundings —
    # the wrong sign of ``Q_v`` that the validation script flags.
    # Production runs with a full microphysics chain that owns q_r
    # should override this to ``True``.
    enable_unsaturated_downdraft: bool = False
    downdraft_efficiency: float = 0.2
    smooth_trigger_sharpness: float = 0.5
    # Bulk-plume entrainment/detrainment for the cloud-base updraft.
    # Deep-convection value (~2e-4 /m): in Emanuel's scheme the
    # cloud-environment mixing is represented explicitly by the
    # buoyancy-sorted ensemble (``n_mixing_fractions``), so the bulk
    # ascent should be near-undilute.  The earlier ``1.5e-3`` (shallow-
    # cumulus range) double-counted dilution — it over-entrained the
    # bulk plume on top of the ensemble mixing, collapsing the updraft
    # buoyancy in the lower troposphere so deep convection could not
    # warm the free troposphere (anti-convective, super-adiabatic,
    # ~30 K-too-cold RCE).
    epsilon_0: float = 2.0e-4
    delta_0: float = 2.0e-4
    M_b_max: float = 0.05
    # --- GENUINE (i,j) episodic-mixing buoyancy-sort spectrum ----------
    # Faithful port of the Fortran CONVECT v4.3c SIJ/ELIJ/MENT mixing
    # matrix (convect43c.f lines 588-712).  When ``use_genuine_mixing``
    # is True (default) the scheme builds the full ``(nlev, nlev)``
    # mixing matrix — every origin level i mixes with environment air in
    # the neutral-buoyancy fraction spectrum, each mixture's buoyancy
    # sets its detrainment level j, and the environmental tendencies are
    # assembled from MENT(i,j).  When False it falls back to the legacy
    # single-sigmoid ``_mixture_buoyancy`` surrogate (kept for back-compat
    # / ablation).  See ``_emanuel_mixing.py``.
    use_genuine_mixing: bool = True
    # Emanuel's effective liquid-water heat capacity CL [J/kg/K] (oracle
    # value 2500).  This is a scheme-internal thermodynamic coefficient
    # in CONVECT's liquid-water-static-energy formulation, distinct from
    # the canonical ``constants.c_pw`` (4218 J/kg/K at standard
    # conditions); kept here so the SIJ/ELIJ algebra matches the oracle
    # term-for-term rather than monkey-patching a global constant.
    c_l_emanuel: float = 2500.0
    # Autoconversion threshold ELCRIT [kg/kg] and critical temperature
    # TLCRIT [degC] for the precipitation efficiency EP (oracle .0011 /
    # -55.0).
    elcrit: float = 1.1e-3
    tlcrit: float = -55.0
    # Mixing-rate coefficient ENTP in M(i) (oracle 1.5).
    entp: float = 1.5
    # --- Smoothing sharpnesses for the discrete sort (AD-safety) -------
    # Each replaces a hard Fortran switch with a smooth surrogate; the
    # forward result tracks the discrete sort to a stated tolerance (see
    # the oracle-vs-ours mixing-matrix comparison in
    # ``.physics-validator/emanuel``).
    # Sigmoid sharpness [1/level] for the ICB/INB cloud-layer windows.
    level_window_sharpness: float = 6.0
    # Sigmoid sharpness [dimensionless] for the ``0 < SIJ < 0.9``
    # entrainment gate (oracle counts a mixture only inside this band).
    sij_gate_sharpness: float = 40.0
    # Upper SIJ gate (oracle 0.9).
    sij_upper_gate: float = 0.9
    # Magnitude floor for the SIJ denominator (oracle ``ABS(DENOM)<0.01``).
    denom_floor: float = 0.01
    # Offset [Pa] for the smooth max-MSE (NK) source-level selection.
    mse_min_search_offset: float = 5.0e4
    # Sigmoid sharpness for the saturated-mixture re-solve switch
    # (oracle ``SIJ<0 .or. SIJ>1 .or. ALTEM>CWAT``).
    sat_branch_sharpness: float = 100.0
    # Sharpness [1/level] for the STRICT integer-index inequalities
    # (``j>i``, ``k<i``, AMP1/AD ``j>t``/``k<t``).  These compare integer
    # level indices, so the 0.5-shifted sigmoid is evaluated at half-integer
    # arguments; a high sharpness makes it ≈binary (σ(±10)≈4.5e-5 at the
    # diagonal) so the strict ``J.GT.I`` / ``K=1,I-1`` Fortran bounds do not
    # leak onto the diagonal (codex iter-5 #2).  Distinct from the FRACTIONAL
    # ``level_window_sharpness`` (ICB/INB cloud edges), which must stay
    # moderate to keep the cloud-top/base transition differentiable.
    strict_index_sharpness: float = 20.0
    # Maximum precipitation efficiency EPMAX in the precip-efficiency EP =
    # EPMAX·(1 − ELACRIT/CLW), clipped to [0, EPMAX] (oracle convect43c.f
    # ``EPMAX = 0.999``).
    precip_efficiency_max: float = 0.999
    # NOTE: Emanuel deliberately has NO ``subsidence_solve`` selector, unlike
    # the rest of the mass-flux family.  Its SHIPPED path
    # (``use_genuine_mixing=True``) is a buoyancy-sorting MIXING MATRIX that
    # computes its own tendencies and never calls
    # ``mass_flux.apply_mass_flux_kernel``; only the legacy surrogate branch
    # does, and that branch's ``sort_multiplier`` rescaling would leave an
    # unpaired vapor debit under a vapor-debiting solve.  See the comment at
    # the kernel call in emanuel.py.  Emanuel is therefore reported as
    # OUTSIDE the matched-kernel family, not silently kernel-matched.


class TiedtkeConfig(NamedTuple):
    """Configuration for the Tiedtke (1989) bulk mass-flux scheme.

    Three-class scheme with deep, mid-level, and shallow branches
    blended on cloud depth.  Downdraft is included with an RH-based
    trigger; convective momentum transport via Gregory et al. 1997.
    First scheme that actually exercises the new ``(ncol, nlev)``
    profile carry for ``M_u(k)``.

    The full Tiedtke 1989 closure uses column moisture convergence for
    the deep branch.  Production passes the real per-level
    ``compute_moisture_convergence`` field (column-integrated inside the
    scheme); when it is ``None`` the scheme falls back to a
    saturation-EXCESS proxy ``MC_proxy = ∫ max(q_v - RH_crit·q_sat, 0)
    dp / (g·tau_MC_proxy)`` with the same qualitative behavior (positive
    in moist columns, zero in dry).

    Fields
    ------
    epsilon_deep, delta_deep : float
        Entrainment / detrainment rates for deep branch [1/m].
    epsilon_shallow, delta_shallow : float
        Same for shallow branch.
    epsilon_midlevel, delta_midlevel : float
        Same for mid-level branch.
    enable_downdraft : bool
        Whether to include the downdraft branch (default ``True``).
    downdraft_alpha : float
        Downdraft / updraft mass flux ratio at LFS (default 0.3).
    downdraft_RH_min : float
        Below this column-mean RH the downdraft fires (default 0.2).
    moisture_convergence_threshold : float
        Column moisture-convergence (or saturation-excess proxy)
        threshold [kg/kg/s].  Below this the deep branch is suppressed
        (default 1e-8).
    moisture_convergence_sharpness : float
        Sigmoid sharpness on the MC threshold [s/(kg/kg)] (default 1e8).
    cape_threshold : float
        Secondary CAPE gate [J/kg] (default 70.0).
    cape_sharpness : float
        Sigmoid sharpness on CAPE gate (default 0.02).
    cloud_depth_deep : float
        Depth threshold separating mid-level from deep branches [m]
        (default 3000.0).
    cloud_depth_shallow_max : float
        Depth threshold separating shallow from mid-level branches
        [m] (default 1500.0).
    depth_split_sharpness : float
        Sigmoid sharpness on the depth thresholds [1/m] (default 1e-3).
    enable_cmt : bool
        Whether to compute CMT (default ``True``).
    cmt_c_u, cmt_c_d : float
        Gregory et al. 1997 closure coefficients (default 0.7).
    downdraft_rh_sharpness : float
        Sigmoid sharpness of the downdraft RH trigger
        ``sigmoid(k · (downdraft_RH_min − rh_below))`` in
        [1/RH-fraction].  The RH argument is O(0.1), so the default 10
        gates crisply around ``downdraft_RH_min``.  NOT interchangeable
        with a level-index sharpness (different argument units) —
        replaces the former dead ``smooth_trigger_sharpness`` field,
        whose 0.02 default was mis-scaled for this argument anyway.
    lcl_membership_sharpness : float
        Sigmoid sharpness of the below-LCL level membership
        ``sigmoid(k · (level − k_lcl_smooth))`` in [1/level index]
        (default 2.0 — a ~1-level transition).
    tau_M_u_relax : float
        Implicit-Euler relaxation timescale [s] for the profile carry
        ``M_u`` toward its diagnosed equilibrium (default 1800.0).
    parcel_dT, parcel_dq : float
        Sub-cloud parcel perturbations (defaults 0.5 K, 1e-3 kg/kg).
    """
    epsilon_deep: float = 1.0e-4
    delta_deep: float = 1.0e-4
    epsilon_shallow: float = 3.0e-4
    delta_shallow: float = 3.0e-4
    epsilon_midlevel: float = 1.0e-4
    delta_midlevel: float = 2.0e-4
    enable_downdraft: bool = True
    downdraft_alpha: float = 0.3
    downdraft_RH_min: float = 0.2
    # Fraction of the downdraft mass flux that re-evaporates as rain
    # falling through the subcloud layer (default 0.05 — matches a
    # historical hardcoded literal that was previously dimensionally
    # wrong; the current implementation distributes the resulting
    # evaporation rate over below-LCL layers by mass weight, with a
    # matching dq_v source so the column water budget closes).
    downdraft_evap_efficiency: float = 0.05
    moisture_convergence_threshold: float = 1.0e-8
    moisture_convergence_sharpness: float = 1.0e8
    cape_threshold: float = 70.0
    cape_sharpness: float = 0.1
    cloud_depth_deep: float = 3000.0
    cloud_depth_shallow_max: float = 1500.0
    depth_split_sharpness: float = 1.0e-3
    enable_cmt: bool = True
    cmt_c_u: float = 0.7
    cmt_c_d: float = 0.7
    downdraft_rh_sharpness: float = 10.0
    lcl_membership_sharpness: float = 2.0
    tau_M_u_relax: float = 1800.0
    parcel_dT: float = 0.5
    parcel_dq: float = 1.0e-3
    tau_MC_proxy: float = 3600.0   # for saturation-excess MC proxy
    # Critical column-mean RH above which the MC proxy starts firing.
    # The proxy approximates moisture convergence as the column-integrated
    # vapor in excess of ``RH_crit * q_sat``: positive in moist columns,
    # vanishing in dry ones.
    mc_proxy_RH_crit: float = 0.6
    tau_shallow_M_b: float = 3600.0  # Shallow-cloud-base mass-flux timescale [s]
    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
    midlevel_M_b_fraction: float = 0.5  # M_b_midlevel = M_b_shallow * this
    # Single-plume scheme — see ZhangMcFarlaneConfig.buoyancy_death_memory.
    buoyancy_death_memory: bool = False
    # Convective precipitation efficiency [0,1]: the fraction of the detrained
    # plume condensate that has precipitated in the updraft and is therefore
    # emitted as RAIN (ConvectionOutput.dq_r_conv_dt — a precipitating species
    # that sediments via microphysics and is invisible to radiation) rather
    # than detrained as suspended cloud water.  The plume docstring delegates
    # precipitation to the calling scheme; this implements it as a bulk
    # efficiency (same convention as SBMConfig.sbm_precip_efficiency /
    # EmanuelConfig.precip_efficiency_max).  Observed deep-convective CPE
    # ~0.5-0.9.  Default 0.0 = OFF (detrain all condensate as cloud, preserving
    # existing behaviour); without it 100% of convective condensate loads the
    # grid-scale cloud + radiation, which microphysics cannot drain.
    precip_efficiency: float = 0.0
    # Precipitation split scheme (see BechtoldConfig): "constant" (fixed
    # precip_efficiency, legacy) or "autoconversion" (physical Sundqvist-1978
    # split on the plume updraft cloud water q_c_u). Unknown => raise at entry.
    precip_split_scheme: str = "constant"
    autoconv_q_c_crit: float = 5.0e-4   # [kg/kg] Sundqvist critical updraft cloud water
    autoconv_pe_max: float = 0.9        # [1] ceiling on the precipitating fraction
    # --- mass-flux kernel vertical solve (Tiedtke 1989 flux form / #824) ---
    # See ``MassFluxConfig.subsidence_solve`` (identical semantics; the shared
    # kernel is ``mass_flux.apply_mass_flux_kernel``).  ``"advective"`` is the
    # DEFAULT HERE and preserves this scheme's shipped behaviour byte-for-byte;
    # ``"implicit_flux"`` is the conservative flux-form solve that also books
    # the ``-dq_c`` vapor sink.  Unknown values raise at kernel entry.
    subsidence_solve: str = "advective"
    theta_implicit: float = 1.0


# --- CAPE trigger trainability (#1417) --------------------------------------
# `cape_threshold` (and the `cape_sharpness` beside it) are declared tier 0 =
# EXCLUDED in every spec below, though they read like textbook tunables.
#
# The trigger is `sigmoid(cape_sharpness * (CAPE - cape_threshold))`. Above an
# argument of ~36.7 the sigmoid returns EXACTLY 1.0 in float64, so its VJP
# `s*(1-s)` is EXACTLY 0 -- measured: x=36.0 -> 2.220e-16, x=37.0 -> 0.0. A
# deep-tropical column sits far inside that dead zone (measured CAPE
# 4679/6992/5042/5042/1641 J/kg for bechtold/emanuel/mass_flux/tiedtke/
# zhang_mcfarlane), and a stable column clamps CAPE to exactly 0 through the
# positive-part integrand, which is a second dead zone. Gradient survives only
# in a narrow marginal band, so in a training run dominated by vigorous or
# stable columns -- most of the globe, most of the time -- these parameters are
# collected by `build_trainable_params` and then silently do not move.
#
# Excluding them keeps the spec TRUTHFUL. Making them genuinely trainable needs
# a reformulated trigger (normalise by a CAPE scale, or soften the positive-part
# clamp); both change trigger behaviour and must be validated as physics, not
# slipped in as a tier edit. `kain_fritsch` is unaffected (cape_threshold = 0,
# it gates on the trigger function instead).
# ----------------------------------------------------------------------------

class BechtoldConfig(NamedTuple):
    """Configuration for the Bechtold/IFS convection scheme.

    Builds on :class:`TiedtkeConfig` (three-class blend, downdraft,
    CMT) with two distinguishing features:

    * **PBL-CAPE / departure-CAPE closure** (Bechtold 2008):
      ``M_b ∝ (CAPE_pbl - CAPE_eq)+ / tau_bl`` where ``CAPE_pbl`` is
      diagnosed from a mass-weighted parcel within the boundary layer
      rather than the surface parcel.
    * **AR1 stochastic perturbation** (Bechtold 2014): ``M_b *= (1 +
      amplitude * ε)`` where ``ε`` is an AR1-process realization with
      decorrelation timescale ``stochastic_decorrelation``.  When
      ``enable_stochastic`` is ``False`` the multiplier is 1.

    Stochasticity defaults to OFF for reproducibility.  When enabled,
    the leaf consumes a ``prng_key`` argument; the convection bridge
    splits ``PhysicsState.prng_key`` into a Bechtold sub-key (folded
    with module id ``0xBEC4``) and an advanced master key, returning
    the latter as part of the multi-field carry update so subsequent
    steps see independent random streams.

    Inherits sensible defaults from Tiedtke 1989 with the entrainment
    revision from Bechtold et al. 2008.

    Fields
    ------
    epsilon_deep, delta_deep : float
        Deep-branch base entrainment / detrainment [1/m].  Default
        1.75e-3 / 0.75e-4 (IFS Cy49r1 Part IV Ch.6).  ``bechtold.py``
        multiplies these by the IFS height factors
        ``(1.3 − RH)·(q_sat/q_sat_base)³`` (entrainment) and
        ``(1.6 − RH)`` (detrainment); the f_scale decay — not a small
        constant ε — is what makes the deep plume penetrate.
    epsilon_shallow, delta_shallow : float
        Shallow-branch base rates [1/m] (default 3.5e-3 = 2× deep ε via
        the IFS f_ε factor, 0.75e-4).  Note: IFS ties shallow detrainment
        to the shallow entrainment (``D_shallow = E_shallow·(1.6 − RH)``);
        ``bechtold.py`` keeps the simpler prescribed-δ₀ shallow form for
        conservation, so ``delta_shallow`` is the shallow detrainment
        base directly.
    epsilon_midlevel, delta_midlevel : float
        Mid-level branch [1/m] (default 1e-4, 2e-4).  IFS applies the ENTSHALP=2
        factor to KTYPE>=2 — BOTH shallow (KTYPE=2) and mid (KTYPE=3) — so its
        elevated-source mid-level entrainment is ENTSHALP*ENTRORG=3.5e-3
        (cuascn.F90:500-507).  Our ``midlevel_weight`` is NOT that source/type
        trigger, though: it is the cloud-depth transition blend between shallow
        and deep (see ``bechtold.py``), so attaching 3.5e-3 to it over-entrains
        deepening surface plumes (audit F6, +19 K SCM-RCE regression).  Held at
        the tuned 1e-4 pending a proper elevated-source classifier — a documented
        fidelity gap, not a claim the coefficient value is wrong.
    cape_pbl_depth : float
        PBL depth [m] for the parcel-source mass weighting (default
        500.0).
    tau_bl : float
        PBL closure timescale [s] (default 3600.0).
    enable_stochastic : bool
        Whether to apply the AR1 stochastic perturbation (default
        ``False`` — reproducibility).
    stochastic_amplitude : float
        Multiplicative perturbation amplitude (default 0.5).
    stochastic_decorrelation : float
        AR1 decorrelation timescale [s] (default 7200.0).
    enable_downdraft, downdraft_alpha, downdraft_RH_min : as Tiedtke.
    enable_cmt, cmt_c_u, cmt_c_d : as Tiedtke.
    downdraft_rh_sharpness, lcl_membership_sharpness : as Tiedtke.
    cape_threshold, cape_sharpness,
    parcel_dT, parcel_dq, tau_M_u_relax,
    cloud_depth_deep, cloud_depth_shallow_max, depth_split_sharpness :
        as Tiedtke.
    """
    # Tiedtke-inherited / revised.
    # IFS Cy49r1 base entrainment/detrainment rates [1/m] (Part IV Ch.6
    # eqs 6.7/6.8; ecmwf-ifs/openifs).  These are the *base* fractional
    # rates ε₀/δ₀; ``bechtold.py`` multiplies them by the IFS height
    # factors ``(1.3 − RH)·(q_sat/q_sat_base)³`` (entrainment) and
    # ``(1.6 − RH)`` (detrainment).  ε₀_deep = 1.75e-3, δ₀_deep = 0.75e-4
    # are the published IFS deep values; the f_scale decay (not a smaller
    # constant ε) is what makes the deep plume penetrate — a constant ε of
    # 1.75e-3 over-dilutes and collapses the updraught (the cold-RCE bug),
    # which is why this is now applied with the IFS vertical scaling rather
    # than as a constant.  Shallow ε₀ carries the IFS f_ε = 2 factor
    # (3.5e-3 = 2 × deep).
    epsilon_deep: float = 1.75e-3
    delta_deep: float = 0.75e-4
    epsilon_shallow: float = 3.5e-3
    delta_shallow: float = 0.75e-4
    # IFS-faithfulness audit F6 (documented fidelity gap — kept at the tuned
    # 1.0e-4): IFS gates the ENTSHALP=2 entrainment factor on KTYPE>=2, i.e. BOTH
    # shallow (KTYPE=2) and elevated mid-level (KTYPE=3) convection, giving
    # epsilon = ENTSHALP*ENTRORG = 3.5e-3 (cuascn.F90:500-507).  That is already
    # reflected in ``epsilon_shallow`` (=3.5e-3).  Our ``midlevel_weight`` is NOT
    # an IFS KTYPE trigger, though: it is the smooth cloud-depth transition blend
    # (1 - deep - shallow) between the shallow and deep classes, so it also tags
    # deepening SURFACE-based plumes as they grow through intermediate depth.
    # Attaching 3.5e-3 to that blend over-entrains (2x) those growing deep plumes
    # and regressed equilibrium SCM-RCE by +19 K mean moist-adiabat deviation
    # (isolated controlled comparison, 100-day Wing-2018 gray-RCEMIP).  The
    # coefficient value is IFS-correct; it is structurally MIS-ATTACHED to our
    # depth-blend object.  A faithful mid-level treatment needs an elevated-source
    # (KTYPE=3) classifier and branch, which this depth-blend scheme does not have
    # — so the blend keeps its tuned 1.0e-4 (below the deep rate that surface
    # plumes actually carry) as a documented fidelity gap.
    epsilon_midlevel: float = 1.0e-4
    # IFS ties mid-level detrainment to entrainment (D=E*(1.6-RH) for KTYPE>=2,
    # cuascn.F90:507,510).  We keep a PRESCRIBED delta_midlevel here for the SAME
    # documented conservation reason as delta_shallow (tying D=E on coarse grids
    # regressed the column MSE budget); note delta_midlevel > delta_shallow, so
    # the mid-level net entrainment (eps-delta) is LESS aggressive than the
    # already-shipping shallow branch.  Residual faithfulness gap, conservation
    # takes precedence (CLAUDE.md).
    delta_midlevel: float = 2.0e-4
    enable_downdraft: bool = True
    downdraft_alpha: float = 0.3
    downdraft_RH_min: float = 0.2
    # See TiedtkeConfig.downdraft_evap_efficiency for definition.
    downdraft_evap_efficiency: float = 0.05
    # -- Penetrative downdraft thermodynamic transport (opt-in, default OFF) --
    # The downdraft branch above drives ONLY rain re-evaporation (locally
    # MOISTENS + cools the sub-cloud layer) and CMT momentum — it has NO
    # mass-flux transport of air, so it can only WET the marine boundary
    # layer (strengthening it makes the BL more humid, not less).
    # ``downdraft_transport`` adds the Tiedtke-1989 penetrative-downdraft
    # thermodynamic transport: a downdraft initiated at the level of minimum
    # moist static energy (the level of free sinking) advects low-MSE (dry,
    # low-q_v) mid-tropospheric air DOWN into the sub-cloud layer, DRYING it.
    # Physically this is the missing marine-BL ventilation: a drier sub-cloud
    # layer -> a larger sea-air humidity gradient (stronger surface
    # evaporation) AND less BL liquid cloud (lower planetary albedo).
    # Conservative flux form: transports s = c_p T + g z and q_v with the
    # downdraft mass flux vanishing at BOTH the origin and the surface, so the
    # column integrals of s and q_v are conserved to machine precision (the
    # rain re-evaporation phase source above is the separate, already
    # budget-closed term).  Default False => BYTE-IDENTICAL to the
    # re-evaporation-only downdraft.
    downdraft_transport: bool = False
    # Fractional entrainment rate [1/m] of the descending downdraft plume
    # (mixes it toward the environment as it sinks).  Default = the IFS
    # ENTRDD = 3.0e-4 (sucumf.F90:144, "average entrainment rate for
    # downdrafts"); the earlier 5.0e-4 was an unsourced mid-range pick from
    # the Tiedtke O(1e-4 - 1e-3) band.  Larger => the downdraft arrives
    # less dry => weaker BL drying.  Only active with the opt-in
    # downdraft_transport (default OFF => no production change).
    downdraft_entrain_rate: float = 3.0e-4
    # Near-surface height scale [m] over which the downdraft mass flux tapers
    # to zero as it detrains its air into the sub-cloud layer (the depth of
    # the drying deposit ~ a marine sub-cloud-layer depth).
    downdraft_detrain_scale_m: float = 700.0
    enable_cmt: bool = True
    cmt_c_u: float = 0.7
    cmt_c_d: float = 0.7
    # The earlier ``cape_threshold = 0.0`` with ``cape_sharpness = 0.005``
    # left the closure essentially always-on (``softplus(0)/0.005 ≈ 138
    # J/kg`` of phantom CAPE even when CAPE = 0).  Match ZM/Tiedtke with a
    # meaningful 70 J/kg trigger threshold; ``cape_sharpness = 0.1``
    # [1/(J/kg)] gives a tight CAPE sigmoid around it.
    cape_threshold: float = 70.0
    cape_sharpness: float = 0.1
    # See TiedtkeConfig.downdraft_rh_sharpness / lcl_membership_sharpness.
    downdraft_rh_sharpness: float = 10.0
    lcl_membership_sharpness: float = 2.0
    parcel_dT: float = 0.5
    parcel_dq: float = 1.0e-3
    tau_M_u_relax: float = 1800.0
    cloud_depth_deep: float = 3000.0
    cloud_depth_shallow_max: float = 1500.0
    depth_split_sharpness: float = 1.0e-3
    # Bechtold-specific
    use_pbl_cape: bool = True
    cape_pbl_depth: float = 500.0
    tau_bl: float = 3600.0
    # CAPE quasi-equilibrium heating ceiling (the C12/RCE warm-runaway
    # harden).  Bechtold's M_b closure is a CAPE-relaxation SURROGATE with
    # no quasi-equilibrium constraint on the APPLIED heating: at pinned
    # M_b_max the scheme sustains large column heating for months while the
    # PBL-parcel CAPE never drains (measured: the scheme's own tendencies
    # GENERATE CAPE on a convecting fixture — downdraft below-LCL moistening
    # feeds the parcel), so nothing bounds the warming (C12 pilot: mean T
    # 267->312 K over days 90-170; SCM-RCE moist-adiabat bias ~50 K).  The
    # sink caps the column-integrated positive convective heating by the
    # quasi-equilibrium energy flux (Arakawa & Schubert 1974 lineage):
    #     H = (c_p/g)·∫ max(dT_dt,0) dp  ≤  ratio · M_b · CAPE   [W/m²]
    # scaling ALL tendencies (and the M_u carry) by
    #     f = clip(ratio·M_b·CAPE / H, 0, 1).
    # M_b·CAPE is the closure's own available-energy flux; `heating_ratio` absorbs the
    # heating-to-KE-generation ratio (tunable, SCM-RCE-calibrated).  A
    # vigorous tower (large CAPE) keeps its full heating; the runaway mode
    # (heating at pinned M_b with modest CAPE) is throttled.  False =
    # bit-exact legacy path.
    # DEFAULT OFF since the 2026-07-17 merge: use_ifs_cape_closure (default
    # True, IFS cumastrn ZMFUB1 closure) consumes CAPE at the SOURCE, fixing
    # the same runaway faithfully; stacking the surrogate ceiling on top
    # double-throttles.  Opt-in lever for legacy/no-IFS-closure configs.
    cape_relaxation_sink: bool = False
    cape_sink_heating_ratio: float = 5.0
    # IFS convective-turnover CAPE-closure timescale (audit F1).  When True
    # (default) the deep closure divides PBL-CAPE by the state-dependent
    # tau_conv = cloud_depth/(2+w_mean), clamped [720,10800] s (cumastrn.F90:773),
    # rather than the fixed tau_bl; False restores the byte-identical fixed-tau_bl
    # closure.  The closure STRUCTURE follows IFS; the resolution-dependent
    # ZTAURES magnitude factor is held at 1.0 (no grid spacing in a column scheme),
    # a documented approximation (see bechtold.py), so this is IFS-structured, not
    # byte-level IFS at a given resolution.  tau_bl stays the reference/fallback
    # timescale (used when False and as the numerator of the tau_bl/tau_conv
    # rescale).  Static Python bool feature-gate in bechtold.py (not a traced leaf).
    use_convective_turnover_tau: bool = True
    # Full IFS deep CAPE closure ``ZMFUB1 = ZCAPE*ZMFUB/(ZHEAT*ZXTAU)``
    # (cumastrn.F90:704-833; see bechtold._ifs_cape_closure_target): the deep
    # cloud-base mass flux is the one that consumes the plume-diagnosed CAPE
    # (ZCAPE, incl. condensate loading) over the convective-turnover time
    # (ZXTAU, the same F1 machinery above) at the column's per-unit-mass-flux
    # stabilization rate (ZHEAT).  Supersedes the tau-only turnover rescale
    # when True (the turnover time is one factor of this closure); False keeps
    # the F1 behavior.  Deep-weighted, cap-after-rescale, quiescence-gated
    # floor — same smoothing doctrine as F1.  The RCAPQADV advection
    # correction, RCAPDCYCL diurnal subtraction and CFL-form ZMFMAX are
    # documented gaps (unplumbed inputs).  Default True since 2026-07-16 after
    # the validation stack: codex adversarial review CLEAN (11 rounds), exact
    # hand-computed oracle mirror, controlled 100-day gray-RCE SCM A/B
    # (neutral-to-slightly-better; stable) and a 15-day C24 analytical-AMIP
    # A/B (stable, near-neutral deltas).  False restores the legacy surrogate
    # deep closure byte-identically.
    use_ifs_cape_closure: bool = True
    # IFS Kessler sub-cloud evaporation of convective rain (cuflxn.F90:436-475;
    # see bechtold._ifs_subcloud_rain_evaporation): the post-split rain flux
    # accumulates downward and evaporates below cloud base at the oracle
    # RCPECONS Kessler rate, limited by the ZRHEBC RH break (deep-ocean 0.85 /
    # non-deep 0.92, deep-weight blended; land values need an absent land
    # mask).  Supersedes the crude ``downdraft_evap_efficiency``-bounded
    # re-evaporation (which fires only below downdraft_RH_min=0.2 — IFS
    # evaporates routinely up to the break).  Default True since 2026-07-16:
    # codex CLEAN x3, machine-exact water/enthalpy pair tests, 100-day
    # gray-RCE A/B (stable, near-neutral) and 15-day C24 AMIP A/B (stable;
    # precip 1.21->1.19 mm/day, CWV +0.05 — the physically-expected evap
    # signature).  False restores the legacy re-evaporation byte-identically.
    use_ifs_subcloud_evap: bool = True
    # IFS in-updraft precipitation formation (cuascn.F90:718-773; see
    # bechtold._ifs_inplume_precip_conversion): the plume condensate converts
    # to rain DURING the ascent via the oracle analytic L-integration
    # (RPRCON=1.4e-3 Kessler-Sundqvist rate / (0.75*w_u), Bergeron-Findeisen
    # enhancement, ZDNOPRC=3e-4 threshold, 5e-3 condensate cap), replacing
    # the post-hoc precip split of the DETRAINED condensate (both
    # precip_split_scheme variants are bypassed when on; the sub-cloud evap
    # then acts on the formed rain — the full cuascn->cuflxn chain).
    # One-pass replay: plume buoyancy loading keeps the unconverted (heavier)
    # condensate — a documented O(0.1-0.3 K) approximation.  Default True
    # since 2026-07-16: codex CLEAN x3, 100-day gray-RCE A/B (stable; moist-
    # adiabat realism IMPROVES 10.27 -> 9.82 K) and 15-day C24 AMIP A/B
    # (stable, near-neutral: energy residual +1.9 W/m^2 on the pre-existing
    # 297 baseline with the residual std improved, moisture residual
    # improved).  False restores the legacy split path byte-identically.
    use_ifs_inplume_precip: bool = True
    # Horizontal grid spacing [m] for the IFS ZTAURES resolution factor on the
    # convective turnover time (ZDX = sqrt(cell area), cumastrn.F90:713,
    # 762-768).  0 (default) = legacy resolution-agnostic ZTAURES = 1.0; the
    # driver sets it from the grid (a static Python float — evaluated at trace
    # time, no traced ops).  At ESM resolutions (dx > 125 km) the factor caps
    # at 3, i.e. a 3x LONGER turnover / weaker deep flux than the legacy
    # floor — validate before enabling by default.
    dx_m: float = 0.0
    # IFS convective downdraft (cudlfsn.F90 LFS + cuddrafn.F90 saturated
    # entraining descent; see bechtold._ifs_downdraft): downdraft mass-flux
    # transport of heat/moisture (perturbation-flux divergence), rain-flux
    # debit with a conserving env vapor/enthalpy ledger, faithful M_d into
    # the CAPE closure's ZHEAT and into CMT.  Default True since 2026-07-17
    # (RCE A/B: equilibrium T-drift 0.052 -> 0.018 K/day; AMIP stable);
    # False is byte-identical legacy.
    use_ifs_downdraft: bool = True
    # IFS shallow PBL-equilibrium closure (cumastrn.F90:468-484, 551-567; see
    # bechtold._ifs_shallow_pbl_target): the shallow class realizes
    # ZMFUB = ZDHPBL/ZDH capped by the true CFL ZMFMAX, with the sub-cloud
    # moist-energy supply in flux form (same-step bulk SHF+LHF + sub-cloud
    # radiative convergence — cloud-base turbulent flux and dynamics
    # advection are documented departures).  Needs the pipeline-supplied
    # shf/lhf/dT_dt_rad kwargs (None => inert).  Default STILL False — the
    # 2026-07-17 flip campaign HELD this one: AMIP A/B stable but the
    # largest mean-state reshape of the family (rms dT 0.66 K / max 13 K at
    # 10 days, r16); RCE-inert (no trigger data).  Flip needs a skill-gated
    # longer run.  False is byte-identical legacy.
    use_ifs_shallow_closure: bool = False
    # IFS RCAPDCYCL=2 diurnal-cycle CAPE correction (cumastrn.F90:780-833;
    # see bechtold._ifs_capdcycl): subtracts the sub-cloud CAPE production
    # over a BL timescale so land deep convection peaks late afternoon.
    # Requires use_ifs_cape_closure + pipeline shf/lhf/land_frac kwargs
    # (None => inert).  Default True since 2026-07-17 (AMIP-with-diurnal A/B
    # stable, modest deltas; RCE-inert as expected — fixed zenith).
    use_ifs_capdcycl: bool = True
    # IFS RCAPQADV=0.8 moisture/temperature-advection CAPE correction
    # (cumastrn.F90:734-760, :801, :819-823; see bechtold._ifs_cape_qadv_terms):
    # the closure adjusts ZCAPE with the plume buoyancy against the
    # dynamics-advected environment (ZCAPE2), the column moisture-advection
    # supply (ZDQCV), and the near-saturated/resolved-ascent branch gate
    # (ZSATFR / omega at ~500 hPa).  Requires use_ifs_cape_closure + the leaf
    # dT_dt_dyn/dq_dt_dyn kwargs — the DYNAMICS tendencies (PTENTA/PTENQA;
    # process-split (post_dyn - pre_dyn)/dt, or an SCM's prescribed
    # large-scale advective forcing).  omega optional (gates only the extreme
    # resolved-ascent branch).  None => inert.  Driver supply of the dynamics
    # tendencies is NOT yet wired in the production pipeline (owed follow-up:
    # the bridge/pipeline REJECT the flag until then); default False,
    # byte-identical legacy.
    use_ifs_cape_qadv: bool = False
    # RCAPQADV blend weight (sucumf.F90:219).
    cape_qadv_weight: float = 0.8
    # IFS land RH break for the sub-cloud rain evaporation (cuflxn.F90:
    # 222-223: 0.70 deep / 0.75 non-deep over land vs 0.85/0.92 ocean).
    # Needs the pipeline land_frac kwarg (None => ocean values, legacy).
    # Default True since 2026-07-17 (AMIP A/B stable, modest deltas; oracle
    # land/ocean split).
    use_ifs_land_rhebc: bool = True
    # IFS convective snow: FOEALFCU wet-bulb rain/snow partition + RTAUMEL
    # melt inside the sub-cloud precip march, FOLD variant (surface snow
    # forcibly melted into the lowest layer; formation-side freezing heat
    # paired so column enthalpy closes exactly — see
    # _ifs_subcloud_rain_evaporation).  Requires use_ifs_subcloud_evap.
    # Default True since 2026-07-17 (RCE A/B: T-drift 0.052 -> 0.026 K/day;
    # AMIP stable).
    use_ifs_snow_melt: bool = True
    enable_stochastic: bool = False
    stochastic_amplitude: float = 0.5
    stochastic_decorrelation: float = 7200.0
    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
    # Strong-convergence normaliser used to make the moisture-convergence
    # enhancement an O(1) multiplier of M_b (Bechtold 2008 Fig. 2 — typical
    # tropical strong-convergence is ≈ 0.05 kg/m²/s).
    mc_normalize_scale: float = 0.05
    # Single-plume scheme — see ZhangMcFarlaneConfig.buoyancy_death_memory.
    buoyancy_death_memory: bool = False
    # Vertical solve for the compensating subsidence + detrainment in the
    # shared mass-flux kernel.  ``"advective"`` (default) is the legacy
    # donor-cell advective form — BYTE-IDENTICAL to the historical scheme
    # but conserves column dry-static-energy only to truncation order
    # (the advective form leaves a non-telescoping ``(φ/ρ)dM/dz`` residual
    # ⇒ a resolution-dependent MSE leak).  ``"implicit_flux"`` selects the
    # IMPLICIT (backward-Euler / θ-blended) CONSERVATIVE flux-form solve
    # (mass_flux.apply_mass_flux_kernel_implicit_flux): flux-form
    # CONSERVATIVE unconditionally (transports s = c_pT+gz and q_v so the
    # column MSE budget telescopes to machine precision for any M/θ/dt),
    # and donor-cell backward-Euler STABLE in the production M/dt/Δp regime
    # (diagonally dominant for θ·dt·g·M/Δp < 1; removes the 2Δz checkerboard
    # the explicit flux form NaN'd on — decoupled from the M_b_max clip).
    # The dispatch raises ValueError on any other value (fn-entry, static).
    # DEFAULT FLIPPED "advective" -> "implicit_flux" (2026-07-22): the leaf
    # column-budget probe measured the advective residual at production
    # L20/dt300 as a -9.5 mm/day column-WATER leak and ~-107 W/m² enthalpy
    # leak on an active tropical fixture — the dominant term of the AMIP
    # E-P non-closure (1.4 mm/day global) and heating/moisture mispairing.
    # The implicit_flux solve zeroes the water residual to machine
    # precision on the same fixture (and EDMF made the identical default
    # flip for the same reason, #824).  "advective" stays selectable for
    # byte-exact legacy reproduction.
    subsidence_solve: str = "implicit_flux"
    # Off-centering for the implicit_flux solve.  1.0 = fully implicit
    # (backward Euler, most damping, default); the kernel clamps to
    # [0.5, 1.0] (θ ≥ 0.5 removes the explicit-side amplification).  Unused
    # when subsidence_solve == "advective".
    theta_implicit: float = 1.0
    # In-updraft convective precipitation efficiency [dimensionless]: the
    # fraction of detrained plume condensate diverted to RAIN (dq_r_conv_dt,
    # sediments via microphysics, radiatively invisible) instead of suspended
    # anvil cloud (dq_c_conv_dt).  Default 0.7 = ON (unlike Tiedtke's 0.0):
    # without the split, undrained anvil cloud radiatively loads the
    # polar-night column and runs the equilibrium away (#929).  0.0 restores
    # the legacy no-split behaviour (dq_r_conv_dt=None) byte-for-byte.  Used by
    # the "constant" precip_split_scheme below; mirrors TiedtkeConfig.
    precip_efficiency: float = 0.7
    # Precipitation split scheme for the detrained condensate:
    #   "constant"       — fixed `precip_efficiency` fraction (above).
    #   "autoconversion" — PHYSICAL Sundqvist (1978) autoconversion on the
    #                      plume updraft cloud water q_c_u (convective_
    #                      autoconversion_split): the precip efficiency EMERGES
    #                      from the updraft loading (thin updraft -> anvil,
    #                      loaded updraft -> rain) instead of a tuned constant.
    # Unknown values raise at scheme entry (dispatch-hardening).
    precip_split_scheme: str = "constant"
    # Autoconversion params (used only when precip_split_scheme="autoconversion"):
    autoconv_q_c_crit: float = 5.0e-4   # [kg/kg] Sundqvist critical updraft cloud water
    autoconv_pe_max: float = 0.9        # [1] ceiling on the precipitating fraction
    # Convective-top pressure [Pa]: the plume mass-flux carry AND the shared
    # kernel's compensating-subsidence gate vanish above this cutoff, so the
    # (non-self-detraining) Bechtold plume terminates here instead of
    # plateauing at M_b_max to the model top. 150 hPa is a physical deep-
    # convection top, TIGHTER than the kernel's 100 hPa default — required
    # because Bechtold's plume does not decay at its LNB; the top-heavy
    # subsidence otherwise bakes the lower stratosphere into a slow blow-up.
    p_conv_top_pa: float = 15000.0
    # --- in-plume conversion (IFS cuascn.F90:718-773 / sucumf.F90:164) ---
    # ECMWF tuning constants of the in-plume Sundqvist conversion, promoted
    # from module constants (2026-07-27 anvil-ice campaign): rprcon scales
    # the conversion RATE, dnoprc the precip-onset condensate threshold.
    # More conversion => drier detrained outflow => thinner anvil.
    # APPENDED AT THE TUPLE END to preserve the positional ABI.
    rprcon: float = 1.4e-3    # IFS RPRCON conversion rate [1/m]
    dnoprc: float = 3.0e-4    # IFS ZDNOPRC precip-onset condensate [kg/kg]


class ConvectiveEDMFConfig(NamedTuple):
    """Configuration for simplified EDMF convection (mass-flux part only).

    Fields
    ------
    epsilon_0 : float
        Entrainment rate [1/m].
    delta_0 : float
        Detrainment rate [1/m].
    a_u_init : float
        Initial updraft area fraction.
    tau_a : float
        Relaxation timescale for a_u [s].
    w_u_min : float
        Minimum updraft velocity [m/s].
    w_u_max : float
        Maximum updraft velocity [m/s] — a physical cap on ``sqrt(2·CAPE)``.
        A real cumulus updraft tops out at ~10–50 m/s; an uncapped high-CAPE
        column can yield >80 m/s and, times ``rho``, an unphysically strong
        compensating subsidence aloft (#824).
    cape_activation_scale : float
        Sigmoid scale for CAPE trigger [J/kg].
    cape_threshold : float
        CAPE threshold [J/kg].
    subsidence_solve : str
        Compensating-subsidence vertical solve passed to
        ``apply_mass_flux_kernel``: ``"implicit_flux"`` (default) is the
        CONSERVATIVE, backward-Euler damping tridiagonal solve (MSE-conserving,
        damps the 2Δz checkerboard — #824); ``"advective"`` is the legacy
        explicit donor-cell form (conserves only to truncation order).
    theta_implicit : float
        Off-centering of the ``implicit_flux`` backward-Euler solve in [0.5, 1.0]
        (1.0 = fully implicit).  Numerics/stability, not a tunable closure.
    """
    epsilon_0: float = 2e-3
    delta_0: float = 2e-3
    a_u_init: float = 0.1
    tau_a: float = 1800.0
    w_u_min: float = 0.1
    w_u_max: float = 50.0
    # Trigger gating — see MassFluxConfig for rationale (sharper
    # ``cape_activation_scale`` and a 70 J/kg threshold close the
    # CAPE=0 leak from the earlier 50 % activation).
    cape_activation_scale: float = 10.0
    cape_threshold: float = 70.0
    M_b_max: float = 0.05   # see ZhangMcFarlaneConfig.M_b_max
    # Conservative, damping subsidence solve by default (#824): the legacy
    # explicit "advective" solve leaked column static energy and, with the
    # non-detraining updraft profile, drove a day-5 full-physics AMIP blowup.
    subsidence_solve: str = "implicit_flux"
    theta_implicit: float = 1.0
    precip_efficiency: float = 0.0  # shared split_convective_rain rain-split (default off = legacy)


class ConvectionConfig(NamedTuple):
    """Top-level convection configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active convection scheme: ``"sbm"``, ``"dca"``, ``"kuo"``,
        ``"mass_flux"``, ``"edmf"``, ``"zhang_mcfarlane"``, or
        ``"none"``.  Future PRs (KF, Emanuel, Tiedtke, Bechtold) add
        their literal here.
    sbm, dca, kuo, mass_flux, edmf, zhang_mcfarlane :
        Per-scheme configuration NamedTuples.
    update_interval_steps : int
        NOT YET IMPLEMENTED in the production physics pipeline — convection
        is recomputed EVERY step regardless of this value.  Only the SCM
        enforces it (rejecting values != 1 for stateful/non-autonomous
        integrators, see ``scm.py``).  Retained as a forward-looking config
        knob; setting it != 1 in a production driver is a silent no-op.
    """
    scheme: str = "sbm"
    sbm: SBMConfig = SBMConfig()
    dca: DCAConfig = DCAConfig()
    kuo: KuoConfig = KuoConfig()
    mass_flux: MassFluxConfig = MassFluxConfig()
    edmf: ConvectiveEDMFConfig = ConvectiveEDMFConfig()
    zhang_mcfarlane: ZhangMcFarlaneConfig = ZhangMcFarlaneConfig()
    kain_fritsch: KainFritschConfig = KainFritschConfig()
    emanuel: EmanuelConfig = EmanuelConfig()
    tiedtke: TiedtkeConfig = TiedtkeConfig()
    bechtold: BechtoldConfig = BechtoldConfig()
    # NOT YET IMPLEMENTED in the production pipeline (see docstring above):
    # convection runs every step; only the SCM reads this (rejection guard).
    update_interval_steps: int = 1
