"""Configuration for atmospheric microphysics schemes.

Provides configuration NamedTuples for:
1. Kessler — warm-rain one-moment (refactored from physics/kessler.py)
2. Sundqvist — large-scale diagnostic condensation
3. Seifert-Beheng — two-moment warm rain
4. Morrison — double-moment ice+liquid
5. Thompson — hybrid moment with graupel
6. ML Emulator — Equinox MLP surrogate
7. P3 — Predicted Particle Properties single-category ice
8. Top-level MicrophysicsConfig that selects the active scheme.

References
----------
- Kessler (1969): On the Distribution and Continuity of Water Substance.
- Sundqvist et al. (1989): Condensation and cloud parameterization studies.
- Seifert & Beheng (2001): A two-moment cloud microphysics scheme.
- Morrison et al. (2005): A new double-moment microphysics scheme.
- Thompson et al. (2008): Explicit forecasts of winter precipitation.
- Morrison & Milbrandt (2015): Parameterization of cloud microphysics
  based on the prediction of bulk ice particle properties. Part I.
  J. Atmos. Sci., 72, 287-311.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.arg_activation import ActivationConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig


__param_spec__ = {
    "KesslerConfig": {
        "scheme_key": "atm.micro.KesslerConfig",
        "excluded": {
            "saturation_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
        },
        "params": {
            # PRIMARY warm-rain knobs (Kessler 1969).
            "autoconversion_threshold": {"units": "kg/kg", "bounds": (0.0001, 0.003), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Kessler (1969)", "shape": None},
            "autoconversion_rate": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Kessler (1969)", "shape": None},
            "accretion_coeff": {"units": "1", "bounds": (0.5, 6.6), "tunable_tier": 1, "transform": "sigmoid", "category": "accretion", "reference": "Kessler (1969)", "shape": None},
            "evaporation_coeff": {"units": "1", "bounds": (0.1, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "evaporation", "reference": "Kessler (1969)", "shape": None},
            "rain_fall_speed": {"units": "m/s", "bounds": (1.0, 15.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Kessler (1969)", "shape": None},
            "hard_sat_adjust_threshold": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
            "hard_sat_max_heating_K": {"units": "K", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
        },
    },
    "MicrophysicsMLEmulatorConfig": {
        "scheme_key": "atm.micro.MicrophysicsMLEmulatorConfig",
        "excluded": {
            "norm_dt": "numerics: input normalisation scale (time-step), fixed feature-engineering constant",
            "norm_T": "numerics: input normalisation scale (temperature), fixed feature-engineering constant",
            "norm_dz": "numerics: input normalisation scale (layer thickness), fixed feature-engineering constant",
            "norm_q_factor": "numerics: input normalisation scale (mixing ratio), fixed feature-engineering constant",
            "norm_rho": "numerics: input normalisation scale (air density), fixed feature-engineering constant",
        },
        # The MLP weights are trained via the Equinox module, not via these
        # NamedTuple float fields. The ``norm_*`` fields are fixed input-
        # normalisation scales (feature engineering), not physical closures,
        # so none are spec-eligible trainable params.
        "params": {},
    },
    "MorrisonConfig": {
        "scheme_key": "atm.micro.MorrisonConfig",
        "excluded": {
            # Fall-speed power-law EXPONENTS feed math.gamma(4 + b) / math.gamma(1 + b)
            # (morrison.py, _warm_rain.py) to precompute the cons* moment factors.
            # math.gamma is a Python C function requiring a float, so a traced
            # trainable leaf breaks JIT/grad. Fixed until those gamma calls use a
            # JAX-traceable gamma (jax.scipy.special.gamma / exp(lgamma)); the
            # fall-speed PREFACTORS fall_a_* stay trainable (plain multipliers).
            "fall_b_r": "fall-speed exponent inside math.gamma(4+b) (non-traceable); fix via JAX gamma to train",
            "fall_b_i": "fall-speed exponent inside math.gamma(4+b) (non-traceable); fix via JAX gamma to train",
            "fall_b_s": "fall-speed exponent inside math.gamma(4+b) (non-traceable); fix via JAX gamma to train",
            "fall_b_g": "fall-speed exponent inside math.gamma(4+b) (non-traceable); fix via JAX gamma to train",
            "autoconversion_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "breakup_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "hom_freeze_T_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "hom_ice_nuc_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "homogeneous_freeze_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "ice_deposition_efficiency": "physics: default at domain boundary / not sigmoid-tunable (fix via config)",
            "ice_sigmoid_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "koop_s_hom_min": "numerics: solver/smoothing/tolerance/iteration parameter",
            "lamg_min": "numerics: solver/smoothing/tolerance/iteration parameter",
            "lami_min": "numerics: solver/smoothing/tolerance/iteration parameter",
            "lamr_min": "numerics: solver/smoothing/tolerance/iteration parameter",
            "lams_min": "numerics: solver/smoothing/tolerance/iteration parameter",
            "melt_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "nuc_T_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "nuc_rh_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "saturation_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "subgrid_rh_crit": "physics-fidelity sub-grid closure: in-cloud cf critical RH, mirrors the cloud scheme (no tunable knob)",
            "subgrid_cf_min": "numerics: cloud-fraction floor capping the in-cloud enhancement (AD/numeric safety)",
        },
        "params": {
            # --- Warm rain (Seifert-Beheng + KK2000) ---
            "k_au": {"units": "m^3 kg^-1 s^-1", "bounds": (50.0, 5000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "k_ac": {"units": "m^3 kg^-1 s^-1", "bounds": (1.0, 20.0), "tunable_tier": 1, "transform": "sigmoid", "category": "accretion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "x_star": {"units": "kg", "bounds": (5e-11, 1e-9), "tunable_tier": 2, "transform": "sigmoid", "category": "autoconversion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "Nc_0": {"units": "1/m^3", "bounds": (1e7, 1e9), "tunable_tier": 2, "transform": "sigmoid", "category": "number_concentration", "reference": "Seifert & Beheng (2001)", "shape": None},
            "k_sc": {"units": "m^3 kg^-1 s^-1", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "D_eq": {"units": "m", "bounds": (0.0003, 0.0033), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "rain_selfcoll_k": {"units": "1", "bounds": (1.0, 20.0), "tunable_tier": 2, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "rain_breakup_d0": {"units": "m", "bounds": (0.0001, 0.0009), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "rain_breakup_steepness": {"units": "1/m", "bounds": (500.0, 7000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "evap_coeff": {"units": "1", "bounds": (0.1, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "evaporation", "reference": "M2005 scheme default", "shape": None},
            "rain_vent_f1": {"units": "1", "bounds": (0.3, 2.5), "tunable_tier": 2, "transform": "sigmoid", "category": "evaporation", "reference": "Morrison et al. (2005)", "shape": None},
            "rain_vent_f2": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "evaporation", "reference": "Morrison et al. (2005)", "shape": None},
            # --- Ice nucleation (Cooper 1986) ---
            "N_i0": {"units": "1/m^3", "bounds": (1.0, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "N_i_nuc_max": {"units": "1/m^3", "bounds": (1e5, 5e6), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Morrison et al. (2005)", "shape": None},
            "cooper_a": {"units": "1/K", "bounds": (0.1, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "cooper_T_act": {"units": "K", "bounds": (255.0, 273.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "ice_nuc_radius": {"units": "m", "bounds": (3e-06, 3e-05), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Morrison et al. (2005)", "shape": None},
            # --- Homogeneous ice nucleation (Ren & MacKenzie 2005 / Koop 2000) ---
            "koop_s_hom_a": {"units": "1", "bounds": (1.5, 7.0), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Ren & MacKenzie (2005)", "shape": None},
            "koop_s_hom_b": {"units": "1/K", "bounds": (0.001, 0.012), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Ren & MacKenzie (2005)", "shape": None},
            "koop_s_hom_max": {"units": "1", "bounds": (1.0, 5.1), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Ren & MacKenzie (2005)", "shape": None},
            "hom_freeze_T_max": {"units": "K", "bounds": (225.0, 240.0), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Koop et al. (2000)", "shape": None},
            "hom_ice_nuc_N": {"units": "1/m^3", "bounds": (1e4, 1e7), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Karcher & Lohmann (2002)", "shape": None},
            # --- Rain freezing (Bigg 1953) ---
            "bigg_aimm": {"units": "1/K", "bounds": (0.2, 2.0), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Bigg (1953)", "shape": None},
            "bigg_bimm": {"units": "1", "bounds": (33.0, 300.0), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Bigg (1953)", "shape": None},
            "homogeneous_freeze_T": {"units": "K", "bounds": (228.0, 240.0), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Morrison et al. (2005)", "shape": None},
            # --- Ice depositional growth / WBF ---
            "dep_coeff": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "Morrison et al. (2005)", "shape": None},
            "q_i_min_growth": {"units": "kg/kg", "bounds": (3e-10, 3e-09), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Morrison et al. (2005)", "shape": None},
            "rho_cloud_ice": {"units": "kg/m^3", "bounds": (100.0, 917.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison et al. (2005)", "shape": None},
            "bergeron_rate": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "Morrison et al. (2005)", "shape": None},
            "T_center": {"units": "K", "bounds": (248.0, 268.0), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Morrison et al. (2005)", "shape": None},
            "T_width": {"units": "K", "bounds": (3.0, 30.0), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Morrison et al. (2005)", "shape": None},
            # --- Riming ---
            "rime_coeff": {"units": "1", "bounds": (0.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "riming", "reference": "Morrison et al. (2005)", "shape": None},  # unclamped riming-rate multiplier (default 1.0 nominal), not a [0,1] probability
            # --- Ice -> snow autoconversion / aggregation ---
            "ice_snow_d_auto": {"units": "m", "bounds": (8e-05, 0.0008), "tunable_tier": 2, "transform": "sigmoid", "category": "aggregation", "reference": "Morrison et al. (2005)", "shape": None},
            "agg_coeff": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "aggregation", "reference": "Morrison et al. (2005)", "shape": None},
            "snow_aggregation_eii": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "aggregation", "reference": "Morrison et al. (2005)", "shape": None},
            # --- Melting ---
            "melt_rate": {"units": "1/s", "bounds": (0.0005, 0.05), "tunable_tier": 2, "transform": "sigmoid", "category": "melting", "reference": "Morrison et al. (2005)", "shape": None},
            # --- Fall speeds (PSD mass-weighted moments) ---
            "a_v_r": {"units": "m^(1-b)/s", "bounds": (40.0, 400.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Seifert & Beheng (2001)", "shape": None},
            "b_v_r": {"units": "1", "bounds": (0.15, 1.5), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Seifert & Beheng (2001)", "shape": None},
            "a_v_i": {"units": "m^(1-b)/s", "bounds": (15.0, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison et al. (2005)", "shape": None},
            "b_v_i": {"units": "1", "bounds": (0.08, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison et al. (2005)", "shape": None},
            "a_v_s": {"units": "m^(1-b)/s", "bounds": (9.0, 90.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison et al. (2005)", "shape": None},
            "b_v_s": {"units": "1", "bounds": (0.09, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison et al. (2005)", "shape": None},
            "fall_a_r": {"units": "m^(1-b)/s", "bounds": (280.0, 2500.0), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison et al. (2005)", "shape": None},
            "fall_a_i": {"units": "m^(1-b)/s", "bounds": (230.0, 2100.0), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "gSAM M2005 default", "shape": None},
            "fall_a_s": {"units": "m^(1-b)/s", "bounds": (3.8, 35.0), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison et al. (2005)", "shape": None},
            "fall_a_g": {"units": "m^(1-b)/s", "bounds": (6.3, 58.0), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison et al. (2005)", "shape": None},
            "lamr_max": {"units": "1/m", "bounds": (16500.0, 150000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison et al. (2005)", "shape": None},
            "lami_max": {"units": "1/m", "bounds": (330000.0, 3000000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison et al. (2005)", "shape": None},
            "lams_max": {"units": "1/m", "bounds": (33000.0, 300000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison et al. (2005)", "shape": None},
            "lamg_max": {"units": "1/m", "bounds": (16500.0, 150000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison et al. (2005)", "shape": None},
            # --- Snow / graupel bulk properties + processes ---
            "rho_snow": {"units": "kg/m^3", "bounds": (33.0, 300.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison et al. (2005)", "shape": None},
            "snow_vent_f1": {"units": "1", "bounds": (0.3, 2.6), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Morrison et al. (2005)", "shape": None},
            "snow_vent_f2": {"units": "1", "bounds": (0.09, 0.84), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Morrison et al. (2005)", "shape": None},
            "snow_collect_eff": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "riming", "reference": "Morrison et al. (2005)", "shape": None},
            "rho_graupel": {"units": "kg/m^3", "bounds": (132.0, 1200.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison et al. (2005)", "shape": None},
            "n0_graupel": {"units": "1/m^4", "bounds": (1.3e6, 1.2e7), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison et al. (2005)", "shape": None},
            "graupel_vent_f1": {"units": "1", "bounds": (0.3, 2.6), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Morrison et al. (2005)", "shape": None},
            "graupel_vent_f2": {"units": "1", "bounds": (0.09, 0.84), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Morrison et al. (2005)", "shape": None},
            "graupel_collect_eff": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "riming", "reference": "Morrison et al. (2005)", "shape": None},
            "graupel_rain_collect_eff": {"units": "1", "bounds": (0.0, 2.0), "tunable_tier": 3, "transform": "sigmoid", "category": "riming", "reference": "Morrison et al. (2005)", "shape": None},  # unclamped collection-rate multiplier (default 1.0 nominal), not a [0,1] probability
            "graupel_embryo_mass": {"units": "kg", "bounds": (5e-11, 5e-10), "tunable_tier": 3, "transform": "sigmoid", "category": "riming", "reference": "Morrison et al. (2005)", "shape": None},
            "hard_sat_adjust_threshold": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
            "hard_sat_max_heating_K": {"units": "K", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
        },
    },
    "P3Config": {
        "scheme_key": "atm.micro.P3Config",
        "excluded": {
            "autoconversion_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "breakup_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "cooper_supi_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "ice_sigmoid_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "melt_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "rho_rim_min": "numerics: solver/smoothing/tolerance/iteration parameter",
            "saturation_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
        },
        "params": {
            # --- Warm rain (Seifert-Beheng liquid phase) ---
            "k_au": {"units": "m^3 kg^-1 s^-1", "bounds": (50.0, 5000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "k_ac": {"units": "m^3 kg^-1 s^-1", "bounds": (1.0, 20.0), "tunable_tier": 1, "transform": "sigmoid", "category": "accretion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "x_star": {"units": "kg", "bounds": (5e-11, 1e-9), "tunable_tier": 2, "transform": "sigmoid", "category": "autoconversion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "Nc_0": {"units": "1/m^3", "bounds": (1e7, 1e9), "tunable_tier": 2, "transform": "sigmoid", "category": "number_concentration", "reference": "Seifert & Beheng (2001)", "shape": None},
            "k_sc": {"units": "m^3 kg^-1 s^-1", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "D_eq": {"units": "m", "bounds": (0.0003, 0.0033), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "evap_coeff": {"units": "1", "bounds": (0.1, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "evaporation", "reference": "Seifert & Beheng (2001)", "shape": None},
            "a_v_r": {"units": "m^(1-b)/s", "bounds": (40.0, 400.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Seifert & Beheng (2001)", "shape": None},
            "b_v_r": {"units": "1", "bounds": (0.15, 1.5), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Seifert & Beheng (2001)", "shape": None},
            # --- Ice nucleation (Cooper 1986) ---
            "N_i0": {"units": "1/m^3", "bounds": (1.0, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "N_i_nuc_max": {"units": "1/m^3", "bounds": (2e4, 5e6), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "gSAM P3 scheme-1 cap 100/L (module_mp_p3.f90:3090)", "shape": None},
            "cooper_a": {"units": "1/K", "bounds": (0.1, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "cooper_T_act": {"units": "K", "bounds": (255.0, 273.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "cooper_T_nuc": {"units": "K", "bounds": (248.0, 266.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "gSAM P3 scheme-1 gate T<-15C (module_mp_p3.f90:3084)", "shape": None},
            "cooper_supi_min": {"units": "1", "bounds": (0.0, 0.2), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "gSAM P3 scheme-1 gate supi>=0.05 (module_mp_p3.f90:3084)", "shape": None},
            # --- Ice depositional growth ---
            "dep_coeff": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            "q_i_min_growth": {"units": "kg/kg", "bounds": (3e-10, 3e-09), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            # --- Riming (collection efficiencies) ---
            "rime_coeff": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "riming", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            "rain_rime_coeff": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "riming", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            # --- Aggregation / melting ---
            "agg_coeff": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "aggregation", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            "melt_rate": {"units": "1/s", "bounds": (0.0005, 0.05), "tunable_tier": 2, "transform": "sigmoid", "category": "melting", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            # --- Ice fall speed (predicted-property power law) ---
            "a_v_i": {"units": "m^(1-b)/s", "bounds": (13.0, 120.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            "b_v_i": {"units": "1", "bounds": (0.1, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            "c_rim_fallspeed": {"units": "1", "bounds": (0.1, 1.2), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            # --- Rime density properties ---
            "rho_rim_max": {"units": "kg/m^3", "bounds": (300.0, 917.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            "rho_rim_accrete": {"units": "kg/m^3", "bounds": (132.0, 900.0), "tunable_tier": 3, "transform": "sigmoid", "category": "riming", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            "rho_ice_ref": {"units": "kg/m^3", "bounds": (165.0, 917.0), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            "hard_sat_adjust_threshold": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
            "hard_sat_max_heating_K": {"units": "K", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
        },
    },
    "SeifertBehengConfig": {
        "scheme_key": "atm.micro.SeifertBehengConfig",
        "excluded": {
            "autoconversion_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "breakup_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "saturation_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
        },
        "params": {
            "k_au": {"units": "m^3 kg^-1 s^-1", "bounds": (50.0, 5000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "k_ac": {"units": "m^3 kg^-1 s^-1", "bounds": (1.0, 20.0), "tunable_tier": 1, "transform": "sigmoid", "category": "accretion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "x_star": {"units": "kg", "bounds": (5e-11, 1e-9), "tunable_tier": 2, "transform": "sigmoid", "category": "autoconversion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "Nc_0": {"units": "1/m^3", "bounds": (1e7, 1e9), "tunable_tier": 2, "transform": "sigmoid", "category": "number_concentration", "reference": "Seifert & Beheng (2001)", "shape": None},
            "k_sc": {"units": "m^3 kg^-1 s^-1", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "D_eq": {"units": "m", "bounds": (0.0003, 0.0033), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "evap_coeff": {"units": "1", "bounds": (0.1, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "evaporation", "reference": "Seifert & Beheng (2001)", "shape": None},
            "a_v_r": {"units": "m^(1-b)/s", "bounds": (40.0, 400.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Seifert & Beheng (2001)", "shape": None},
            "b_v_r": {"units": "1", "bounds": (0.15, 1.5), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Seifert & Beheng (2001)", "shape": None},
            "hard_sat_adjust_threshold": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
            "hard_sat_max_heating_K": {"units": "K", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
        },
    },
    "SundqvistConfig": {
        "scheme_key": "atm.micro.SundqvistConfig",
        "excluded": {
            "sigmoid_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
        },
        "params": {
            # PRIMARY large-scale-condensation knobs (Sundqvist et al. 1989).
            "rh_crit": {"units": "1", "bounds": (0.5, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "critical_rh", "reference": "Sundqvist et al. (1989)", "shape": None},
            "qc_crit": {"units": "kg/kg", "bounds": (0.0001, 0.0015), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist et al. (1989)", "shape": None},
            "auto_rate": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist et al. (1989)", "shape": None},
            "evap_coeff": {"units": "1", "bounds": (0.0001, 0.0015), "tunable_tier": 2, "transform": "sigmoid", "category": "evaporation", "reference": "Sundqvist et al. (1989)", "shape": None},
            # SBK89 Sec. 5 precipitation-release enhancements: coalescence F1
            # (function of the precipitation flux from above) and Bergeron F2
            # (mixed-phase temperature window); c_0 is multiplied and q_c,crit
            # divided by F1*F2.
            "coalescence_enh_coeff": {"units": "(kg m^-2 s^-1)^-1/2", "bounds": (0.0, 1000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist, Berge & Kristjansson (1989)", "shape": None},
            "bergeron_enh_coeff": {"units": "1", "bounds": (0.0, 20.0), "tunable_tier": 2, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist, Berge & Kristjansson (1989)", "shape": None},
            "bergeron_T_peak_K": {"units": "K", "bounds": (248.0, 268.0), "tunable_tier": 3, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist, Berge & Kristjansson (1989)", "shape": None},
            "bergeron_T_width_K": {"units": "K", "bounds": (2.0, 15.0), "tunable_tier": 3, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist, Berge & Kristjansson (1989)", "shape": None},
        },
    },
    "ThompsonConfig": {
        "scheme_key": "atm.micro.ThompsonConfig",
        "excluded": {
            "autoconversion_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "breakup_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "graupel_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "ice_deposition_efficiency": "physics: default at domain boundary / not sigmoid-tunable (fix via config)",
            "ice_sigmoid_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "melt_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "rain_evap_rh_floor": "numerics: solver/smoothing/tolerance/iteration parameter",
            "saturation_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
        },
        "params": {
            # --- Warm rain (Seifert-Beheng helpers, gamma-corrected) ---
            "k_au": {"units": "m^3 kg^-1 s^-1", "bounds": (50.0, 5000.0), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "k_ac": {"units": "m^3 kg^-1 s^-1", "bounds": (1.0, 20.0), "tunable_tier": 1, "transform": "sigmoid", "category": "accretion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "x_star": {"units": "kg", "bounds": (5e-11, 1e-9), "tunable_tier": 2, "transform": "sigmoid", "category": "autoconversion", "reference": "Seifert & Beheng (2001)", "shape": None},
            "Nc_0": {"units": "1/m^3", "bounds": (1e7, 1e9), "tunable_tier": 2, "transform": "sigmoid", "category": "number_concentration", "reference": "Thompson et al. (2008)", "shape": None},
            "k_sc": {"units": "m^3 kg^-1 s^-1", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "D_eq": {"units": "m", "bounds": (0.0003, 0.0033), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Seifert & Beheng (2001)", "shape": None},
            "evap_coeff": {"units": "1", "bounds": (0.1, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "evaporation", "reference": "Thompson et al. (2008)", "shape": None},
            "a_v_r": {"units": "m^(1-b)/s", "bounds": (40.0, 400.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Seifert & Beheng (2001)", "shape": None},
            "b_v_r": {"units": "1", "bounds": (0.15, 1.5), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Seifert & Beheng (2001)", "shape": None},
            "mu_c": {"units": "1", "bounds": (0.99, 9.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Thompson et al. (2008)", "shape": None},
            "mu_r": {"units": "1", "bounds": (0.33, 3.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Thompson et al. (2008)", "shape": None},
            # --- Ice nucleation (Cooper 1986) ---
            "N_i0": {"units": "1/m^3", "bounds": (1.0, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "N_i_nuc_max": {"units": "1/m^3", "bounds": (1e5, 5e6), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Thompson et al. (2008)", "shape": None},
            "cooper_a": {"units": "1/K", "bounds": (0.1, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "cooper_T_act": {"units": "K", "bounds": (255.0, 273.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            # --- Ice depositional growth / WBF ---
            "dep_coeff": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "Thompson et al. (2008)", "shape": None},
            "q_i_min_growth": {"units": "kg/kg", "bounds": (3e-10, 3e-09), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Thompson et al. (2008)", "shape": None},
            "rho_cloud_ice": {"units": "kg/m^3", "bounds": (100.0, 917.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Thompson et al. (2008)", "shape": None},
            "bergeron_rate": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "Thompson et al. (2008)", "shape": None},
            "T_center": {"units": "K", "bounds": (248.0, 268.0), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Thompson et al. (2008)", "shape": None},
            "T_width": {"units": "K", "bounds": (3.0, 30.0), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Thompson et al. (2008)", "shape": None},
            # --- Riming ---
            "rime_coeff": {"units": "1", "bounds": (0.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "riming", "reference": "Thompson et al. (2008)", "shape": None},  # unclamped riming-rate multiplier (default 1.0 nominal), not a [0,1] probability
            # --- Ice -> snow autoconversion / aggregation ---
            "ice_snow_d_auto": {"units": "m", "bounds": (8e-05, 0.0008), "tunable_tier": 2, "transform": "sigmoid", "category": "aggregation", "reference": "Thompson et al. (2008)", "shape": None},
            "agg_coeff": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 2, "transform": "sigmoid", "category": "aggregation", "reference": "Thompson et al. (2008)", "shape": None},
            # --- Melting ---
            "melt_rate": {"units": "1/s", "bounds": (0.0005, 0.05), "tunable_tier": 2, "transform": "sigmoid", "category": "melting", "reference": "Thompson et al. (2008)", "shape": None},
            # --- Graupel formation from riming ---
            "rime_to_graupel_threshold": {"units": "kg/kg/s", "bounds": (3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "riming", "reference": "Thompson et al. (2008)", "shape": None},
            "rime_to_graupel_rate": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "riming", "reference": "Thompson et al. (2008)", "shape": None},
            # --- Frozen-species fall speeds ---
            "a_v_i": {"units": "m^(1-b)/s", "bounds": (15.0, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Thompson et al. (2008)", "shape": None},
            "b_v_i": {"units": "1", "bounds": (0.08, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Thompson et al. (2008)", "shape": None},
            "a_v_s": {"units": "m^(1-b)/s", "bounds": (9.0, 90.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Thompson et al. (2008)", "shape": None},
            "b_v_s": {"units": "1", "bounds": (0.09, 0.9), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Thompson et al. (2008)", "shape": None},
            "a_v_g": {"units": "m^(1-b)/s", "bounds": (26.0, 240.0), "tunable_tier": 2, "transform": "sigmoid", "category": "fall_speed", "reference": "Thompson et al. (2008)", "shape": None},
            "b_v_g": {"units": "1", "bounds": (0.13, 1.2), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "Thompson et al. (2008)", "shape": None},
            "hard_sat_adjust_threshold": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
            "hard_sat_max_heating_K": {"units": "K", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
        },
    },
}


class KesslerConfig(NamedTuple):
    """Configuration for Kessler warm-rain microphysics."""
    autoconversion_threshold: float = 1.0e-3   # q_c threshold [kg/kg]
    autoconversion_rate: float = 1.0e-3         # Rate [1/s]
    accretion_coeff: float = 2.2                # Collection coefficient
    evaporation_coeff: float = 1.0              # Evaporation coefficient
    rain_fall_speed: float = 5.0                # Terminal velocity [m/s]
    saturation_sharpness: float = 100.0         # Smooth switch sharpness
    # --- Hard (iterated) saturation-adjustment guard (opt-in) ---
    # Iterated (bracketed-bisection) saturation adjustment that lands q_v ON liquid
    # saturation curve where q_v > hard_sat_adjust_threshold * q_sat, draining
    # local super-saturation pools the smooth sigmoid path cannot (conserving
    # c_pd*T + L_v*q_v exactly).  Default OFF => byte-identical to the smooth
    # path.  See ``_warm_rain.saturation_adjustment``.
    hard_saturation_adjustment: bool = False
    hard_sat_adjust_threshold: float = 1.1      # RH trigger q_v > thr*q_sat [-]
    hard_sat_max_heating_K: float = 5.0         # per-step latent-heating cap [K]


class SundqvistConfig(NamedTuple):
    """Configuration for Sundqvist large-scale condensation."""
    rh_crit: float = 0.8              # Critical relative humidity
    sigmoid_sharpness: float = 20.0   # Sharpness for smooth activation
    auto_rate: float = 1e-3           # Autoconversion rate c_0 [1/s]
    # Critical cloud water for autoconversion: P_auto = c_0·q_c·
    # (1 − exp(−(q_c/q_c,crit)²)) (Sundqvist 1989).  Suppresses
    # autoconversion below q_c,crit (drizzle forms only once cloud water
    # is large enough).  0 → the linear no-threshold limit.
    qc_crit: float = 5e-4             # [kg/kg]
    evap_coeff: float = 5e-4          # Sub-cloud evaporation coefficient
    # --- SBK89 (Sec. 5) precipitation-release enhancements ---
    # Coalescence F1 = 1 + c1·sqrt(P) with P the precipitation flux falling
    # in from above [kg m^-2 s^-1]: existing precipitation collects cloud
    # water and accelerates release (F1 ≈ 6 at 1 mm/h with the default).
    # 0 → enhancement off (plain Sundqvist base autoconversion).
    coalescence_enh_coeff: float = 300.0   # c1 [(kg m^-2 s^-1)^-1/2]
    # Bergeron-Findeisen F2 = 1 + c2·exp(−((T − T_peak)/T_width)²): a smooth
    # mixed-phase window peaking near −15 °C, where the ice-liquid saturation
    # difference e_sw − e_si (the Bergeron growth driver) is largest; decays
    # to ~1 above freezing by construction.  0 → enhancement off.
    bergeron_enh_coeff: float = 3.0        # peak amplification [-]
    bergeron_T_peak_K: float = constants.T_freeze - 15.0   # [K]
    bergeron_T_width_K: float = 7.0        # Gaussian half-width [K]


class SeifertBehengConfig(NamedTuple):
    """Configuration for Seifert-Beheng two-moment warm rain."""
    k_au: float = 6e2                # Autoconversion rate [1/(kg*s)]
    x_star: float = 2.6e-10          # Separation mass [kg]
    # ``N_c`` per-volume: see ``_warm_rain.effective_Nc`` Notes.
    Nc_0: float = 1e8                # Initial cloud droplet number [1/m³]
    k_ac: float = 5.25               # Accretion rate [m^3/(kg*s)]
    k_sc: float = 1e-3               # Self-collection rate [m^3/(kg*s)]
    D_eq: float = 1.1e-3             # Equilibrium breakup diameter [m]
    breakup_sharpness: float = 1e4   # Sigmoid sharpness for breakup [1/m]
    a_v_r: float = 130.0             # Rain fall speed coefficient a [m^(1-b)/s]
    b_v_r: float = 0.5               # Rain fall speed exponent b
    evap_coeff: float = 1.0          # Evaporation coefficient
    saturation_sharpness: float = 100.0  # Sigmoid sharpness for saturation [1/(kg/kg)]
    # Autoconversion onset sharpness — dimensionless steepness on the
    # normalised argument ``x_c / x_star − 1`` (iter-97 fix).  Default
    # of 10 gives transition over ~10 % of x_star around the threshold;
    # set higher for a sharper SB-style step, lower for a smoother
    # Kessler-like onset.  Distinct from ``saturation_sharpness`` which
    # is in units of kg/kg and would be 100× too steep here.
    autoconversion_sharpness: float = 10.0
    # --- Hard (iterated) saturation-adjustment guard (opt-in) ---
    # See ``_warm_rain.saturation_adjustment`` / ``KesslerConfig`` above.
    # Default OFF => byte-identical to the smooth path.
    hard_saturation_adjustment: bool = False
    hard_sat_adjust_threshold: float = 1.1      # RH trigger q_v > thr*q_sat [-]
    hard_sat_max_heating_K: float = 5.0         # per-step latent-heating cap [K]


class MorrisonConfig(NamedTuple):
    """Configuration for Morrison double-moment (ice+liquid)."""
    # Reference flavor. "mg" (DEFAULT, for GLOBAL/GCM runs) = E3SM/CESM
    # Morrison-Gettelman parameter + process set (micro_mg_utils.F90). "sam" =
    # SAM/gSAM M2005 set (the plane-CRM-vs-gSAM validation target; the plane CRM
    # driver sets this explicitly). Resolved at scheme entry by
    # ``resolve_morrison_flavor``: the "mg" flavor overrides ``lami_max=1/10µm``,
    # ``snow_aggregation_eii=0.5``, ``rho_snow=250``, ``fall_b_i=1.0`` and
    # selects ``ice_to_snow_scheme="mg_ferrier"`` (180-s Ferrier ice→snow). Warm
    # rain (kk2000) and ice deposition (m2005) are ALREADY MG-faithful in both.
    morrison_flavor: str = "mg"      # "mg" (global default) | "sam" (CRM)
    # Warm-rain autoconversion + accretion scheme:
    #   "kk2000" (default) = Khairoutdinov-Kogan 2000, the SAM M2005
    #     DEFAULT (IRAIN=0): PRC=1350·qc^2.47·(Nc[#/cm³])^-1.79,
    #     PRA=67·(qc·qr)^1.15. Faithful to the gSAM oracle.
    #   "seifert_beheng" = the legacy simplified SB proxy (k_au·q_c^2·onset
    #     sigmoid autoconv + bilinear k_ac·rho·q_c·q_r accretion). NOT the
    #     published universal functions — only the au/x_* number closure is
    #     SB-faithful.
    #   "seifert_beheng_sb2001" = the PUBLISHED SB2001 universal functions —
    #     the gSAM IRAIN=1 MASS closures (module_mp_graupel.f90:1835-1844,
    #     :1960-1962): phi_au=600·tau^0.68·(1−tau^0.68)^3 with the q_c^4·N_c^-2
    #     rate, phi_ac=(tau/(tau+5e-4))^4 with the fixed 5.78 kernel, plus the
    #     au/x_* rain-number source. Faithful for a SUPPLIED nu; the default
    #     fixed nu (0.4315) is an approximation of gSAM's spatially-diagnosed
    #     pgam, and the SB2001-specific cloud-NUMBER sinks are not represented.
    warm_rain_scheme: str = "kk2000"
    predict_Nc: bool = False         # SAM M2005 dopredictNc. False (SAM DEFAULT) =
                                     # SPECIFIED constant droplet number Nc_0: the
                                     # size distribution uses Nc_0 and cloud number is
                                     # NOT evolved (dN_c/dt=0). True = prognostic Nc,
                                     # which needs SAM's droplet-activation source
                                     # (not yet ported) — leaving it False matches SAM
                                     # and avoids the sink-only Nc<0 drift.
    # Aerosol-CCN coupling (AMIP): with predict_Nc=False, the physics
    # pipeline fills ``hydrometeors.N_c`` with a per-column SPECIFIED
    # droplet number diagnosed from the prescribed aerosol optical depth
    # (Andreae 2009 AOT–CCN inversion, ``aerosol_activation.ccn_from_aod``)
    # and ``effective_Nc`` uses that field instead of the constant Nc_0.
    # N_c is still NOT evolved (dN_c/dt = 0).  Closes the aerosol →
    # microphysics link (Twomey r_eff + KK2000 Nc^-1.79 lifetime
    # effects).  Ignored when predict_Nc=True.
    nc_from_aerosol: bool = False
    # Sub-grid in-cloud autoconversion/accretion (Morrison & Gettelman 2008;
    # Boutle et al. 2014).  Warm-rain rates are strongly non-linear in cloud
    # water (KK2000 PRC ∝ q_c^2.47), so evaluating them on the GRID-MEAN q_c
    # systematically UNDER-produces drizzle in partly-filled boxes.  When True,
    # autoconversion + accretion are evaluated on the IN-CLOUD water q_c/cf and
    # the resulting tendency is scaled back by the cloud fraction cf — i.e. the
    # standard "all warm rain happens in the cloudy fraction" closure, giving an
    # enhancement cf^(1−2.47)=cf^−1.47 (autoconv) / cf^−1.30 (accretion).  cf is
    # the Sundqvist √-form from the local RH (subgrid_rh_crit, mirroring the
    # cloud scheme), floored at subgrid_cf_min for AD/numeric safety.  This is a
    # physics-fidelity correction (no tunable knob), default False to preserve
    # bit-reproducibility of existing runs.
    subgrid_autoconversion: bool = False
    subgrid_rh_crit: float = 0.7     # critical RH for the in-cloud cf (Sundqvist)
    subgrid_cf_min: float = 0.1      # cf floor (caps enhancement at cf_min^-1.47)
    # Warm rain (Seifert-Beheng knobs; consumed only when
    # warm_rain_scheme="seifert_beheng")
    k_au: float = 6e2
    x_star: float = 2.6e-10
    Nc_0: float = 1e8
    k_ac: float = 5.25
    k_sc: float = 1e-3
    D_eq: float = 1.1e-3
    breakup_sharpness: float = 1e4
    # Rain self-collection + breakup:
    #   "sb2001" (default) = the SAM NRAGG (Seifert-Beheng 2001,
    #     module_mp_graupel.f90:1980) functional form with a lambda clamp and an
    #     explicit-Euler non-overshoot limiter (NOT the raw Fortran rate):
    #     NRAGG=−5.78·dum·q_r·N_r·ρ with dum=1 for
    #     mean drops < 300µm (self-collection) and dum=2−exp(2300·(1/LAMR−300µm))
    #     for larger drops (→ active breakup as dum<0). Uses the prognostic N_r.
    #     (legoESM's legacy k_sc=1e-3 self-collected ~5580× too WEAKLY.)
    #   "legacy" = the sigmoid-breakup form (k_sc, breakup_sharpness, D_eq).
    rain_selfcoll_scheme: str = "sb2001"
    rain_selfcoll_k: float = 5.78        # SAM/SB2001 self-collection coeff
    rain_breakup_d0: float = 300.0e-6    # SB2001 breakup onset mean size [m]
    rain_breakup_steepness: float = 2300.0  # SB2001 breakup exponent [1/m]
    a_v_r: float = 130.0
    b_v_r: float = 0.5
    evap_coeff: float = 1.0
    # Rain evaporation:
    #   "m2005" (default, faithful) = SAM PRE diffusion+ventilation evaporation
    #     of the rain PSD (module_mp_graupel.f90:1994): EPSR=2π·N0R·DV·[F1R/LAMR
    #     + F2R·CONS9·(ARN·ρ/μ)^½·SC^⅓·LAMR^(1−CONS34)], PRE=EPSR·(q_sat−q_v)/AB.
    #     Uses the prognostic N_r (double-moment rain). Ventilation accelerates
    #     evaporation of falling drops → faithful cold-pool cooling.
    #   "bulk" = the legacy evap_coeff·subsat·q_r^0.525 (number-blind).
    rain_evap_scheme: str = "m2005"
    rain_vent_f1: float = 0.78       # SAM F1R ventilation coefficient
    rain_vent_f2: float = 0.308      # SAM F2R ventilation coefficient
    saturation_sharpness: float = 100.0
    autoconversion_sharpness: float = 10.0  # See SB config — iter-97/99
    # Ice nucleation (Cooper 1986, SAM M2005 INUC=0)
    # Canonical Cooper curve: N_i = 0.005·exp(0.304·(T_f−T)) per LITRE
    # = 5·exp(...) per m³.  The base coefficient is therefore 5 /m³ (NOT
    # 5e3 = 5 /L, which over-nucleated by 1000× before iter-9).  Divided
    # by ρ to the per-mass ``N_i`` stored in HydrometeorState, capped at
    # ``N_i_nuc_max`` (SAM "limit to 500 L⁻¹").
    N_i0: float = 5.0                # Cooper base ice crystal number [1/m³]
    N_i_nuc_max: float = 5.0e5       # Cooper cap [1/m³] (= 500 /L)
    cooper_a: float = 0.304          # Cooper exponent
    cooper_T_act: float = 265.0      # Activation temperature [K]
    ice_sigmoid_sharpness: float = 5.0  # Sharpness for ice-liquid partition
    # Smooth supersaturation gate for nucleation (AD-regularised surrogate
    # for SAM's HARD gate (RH_liq≥0.999 & T≤265.15) OR RH_ice≥1.08).
    # Sharpnesses chosen so leakage is negligible: at 1 % RH below the
    # threshold σ(1000·−0.01)=σ(−10)≈4.5e-5; at 2 K above 265.15 K
    # σ(3·−2)=σ(−6)≈2.5e-3 (Codex iter-9: 200/1 leaked ~0.1–0.27).
    nuc_rh_sharpness: float = 1000.0    # 1/(RH units) — ramp ~0.1 % RH
    nuc_T_sharpness: float = 3.0        # 1/K — ramp ~0.3 K at 265.15 K
    ice_nuc_radius: float = 10.0e-6     # Initial nucleated crystal radius [m]
    #                                     (SAM MI0 = 4/3·π·ρ_ci·r³)
    # Homogeneous ice nucleation (Koop 2000 / Kärcher-Lohmann 2002) — OPT-IN.
    # SAM M2005 Cooper (INUC=0) carries ONLY primary/heterogeneous ice (≤500/L),
    # so NOTHING caps cirrus ice-supersaturation: the diffusional-growth rate
    # EPSI ∝ N_i^⅔·q_i^⅓ bootstraps slowly when fresh convective outflow floods
    # an upper-tropospheric level with vapour, leaving RH_ice far above 100%
    # (transiently >1000% in a violent small-domain RCE spin-up — the deposition
    # sink lags the convective source). Real cirrus homogeneous freezing of
    # aqueous haze bursts a HIGH crystal number once RH_ice exceeds the
    # homogeneous threshold S_hom(T); those crystals deposit the excess vapour
    # and pin RH_ice near S_hom (~1.5-1.6). This adds that missing process as a
    # supersaturation-gated ice-NUMBER (+seed-mass) source that feeds the
    # EXISTING M2005 deposition (so the vapour/ice/number/heat budgets + donor
    # clamps stay consistent automatically). Default OFF ⇒ byte-identical to the
    # SAM-faithful Cooper-only path; enable for RCE cirrus realism. AD-safe.
    #   S_hom(T) = koop_s_hom_a − koop_s_hom_b·T  [K-units], clipped to
    #   [koop_s_hom_min, koop_s_hom_max]: ≈1.64 at 185 K, ≈1.58 at 200 K,
    #   ≈1.44 at 235 K. This is the Ren & MacKenzie (2005, QJRMS 131:1585)
    #   ANALYTICAL LINEAR FIT to the Koop et al. (2000, Nature 406) water-
    #   activity homogeneous-freezing threshold for a representative critical
    #   nucleation rate; NOT Koop's full J(Δa_w) integral. Only valid in the
    #   cold cirrus regime, so activation is additionally gated on T below
    #   ``hom_freeze_T_max`` (homogeneous freezing of aqueous haze needs
    #   T ≲ −38 °C). The ``koop_s_hom_min`` floor only bites for T the cold
    #   gate has already switched off.
    homogeneous_ice_nucleation: bool = False
    koop_s_hom_a: float = 2.349         # Ren-MacKenzie 2005 intercept [-]
    koop_s_hom_b: float = 1.0 / 259.0   # Ren-MacKenzie 2005 slope [1/K]
    koop_s_hom_min: float = 1.4         # floor on S_hom [-]
    koop_s_hom_max: float = 1.7         # cap on S_hom [-]
    hom_freeze_T_max: float = 235.0     # max T for homogeneous freezing [K]
    hom_freeze_T_sharpness: float = 1.0  # cold-gate ramp [1/K] (~few-K width)
    hom_ice_nuc_sharpness: float = 200.0  # smooth RH_ice gate ramp [1/(RH unit)]
    hom_ice_nuc_N: float = 1.0e6        # homogeneous crystal number [1/m³]
    #                                     (~1 cm⁻³; Kärcher-Lohmann 2002 cirrus
    #                                     range 1e4-1e7 /m³); stored per-mass via
    #                                     /ρ. STABILITY: the boosted diffusional-
    #                                     growth EPSI·dt/ABI stays <1 (no explicit
    #                                     overshoot) for N up to ~1e7 /m³ at
    #                                     dt≤20 s; an ON-only cap backstops it.
    # Depositional growth + sublimation.
    #   "m2005" (default) = faithful SAM M2005 diffusional growth
    #     PRD = EPSI·(q_v−q_sat_i)/ABI with EPSI ∝ ρ·DV·N_i^⅔·q_i^⅓ and the
    #     ABI psychrometric (latent-heat) correction; SUBLIMATION when
    #     q_v < q_sat_i (donor-clamped). Tuned by the DIMENSIONLESS
    #     ``ice_deposition_efficiency`` (NOT ``dep_coeff`` — they have
    #     different units; keeping them separate avoids the iter-8 codex
    #     "two-meanings" config trap).
    #   "heuristic" = the legacy form dep_coeff·max(S_i,0)·q_i·N_i^⅓
    #     (deposition only, no sublimation; dep_coeff has units 1/s-ish).
    ice_deposition_scheme: str = "m2005"
    ice_deposition_efficiency: float = 1.0   # m2005 dimensionless multiplier
    rho_cloud_ice: float = 500.0     # SAM M2005 cloud-ice bulk density
    #                                  [kg/m³] (RHOI); ≠ solid-ice 917.
    dep_coeff: float = 1e-3          # legacy "heuristic" deposition coeff
    # Floor on q_i used inside the diffusional-growth term so freshly
    # nucleated particles (N_i > 0, q_i ≈ 0) can grow.  ~10 µm-sized
    # crystals at N_i ~ 5e3 m^-3 correspond to q_i ~ 1e-9 kg/kg.
    q_i_min_growth: float = 1e-9     # Minimum effective q_i for deposition [kg/kg]
    # Wegener-Bergeron-Findeisen (cloud water → ice in mixed phase):
    #   "emergent" (default) = SAM M2005 behaviour. Verified in the gSAM
    #     source: NO explicit Bergeron rate — the vapour budget
    #     (module_mp_graupel.f90:3690) is deposition/sublimation/nucleation
    #     ONLY; line 3924 "EQUILIBRIUM SS INCLUDING BERGERON EFFECT" shows
    #     the effect is carried by the deposition's equilibrium
    #     supersaturation, not a rate. WBF here emerges from ice deposition
    #     (PRD draws q_v toward ice saturation) + saturation adjustment
    #     (evaporates cloud water as q_v falls below liquid saturation),
    #     so glaciation is DEPOSITION-RATE-LIMITED (supercooled liquid
    #     persists). DEPENDS on ice_deposition_scheme="m2005": pairing
    #     "emergent" with the "heuristic" deposition is not faithful (the
    #     heuristic does not diffuse vapour to ice saturation properly).
    #     As of iter-14 the m2005 deposition GROWTH branch is no longer
    #     f_ice-gated (EPSI∝N_i^⅔ self-gates on ice), so WBF now fires across
    #     the full mixed phase incl. the WARM tail (265–273 K), matching SAM.
    #   "bergeron_heuristic" = the legacy explicit rate; forces glaciation
    #     far faster than the deposition physics supports (~50× at 260 K in
    #     a liquid-saturated test — magnitude is regime-dependent).
    # NOTE: the default is "emergent", a SCIENTIFICALLY MEANINGFUL change
    # from the legacy explicit Bergeron (more supercooled liquid, slower
    # mixed-phase glaciation) — not a neutral refactor.
    wbf_scheme: str = "emergent"
    bergeron_rate: float = 1e-3      # heuristic Bergeron conversion rate [1/s]
    T_center: float = 258.0          # Bergeron T window center [K]
    T_width: float = 10.0            # Bergeron T window width [K]
    # Riming
    rime_coeff: float = 1.0          # Riming collection efficiency
    # Rain freezing (Bigg 1953 immersion freezing of supercooled rain):
    #   "bigg" (default, faithful) = SAM MNUCCR/NNUCCR (module_mp_graupel.f90:
    #     3251-3257): rate ∝ (exp(AIMM·(T₀−T))−1)·N_r/LAMR^{6,3}. Supercooled
    #     rain freezes to snow (legoESM has no graupel — SAM freezes to
    #     graupel), releasing the latent heat of fusion. Critical for the
    #     phase + buoyancy of cold convective updrafts.
    #   "none" = no rain freezing (legacy — rain stayed liquid below 0°C).
    rain_freeze_scheme: str = "bigg"
    bigg_aimm: float = 0.66          # SAM AIMM [1/K] (Bigg immersion freezing)
    bigg_bimm: float = 100.0         # SAM BIMM (Bigg immersion freezing)
    # Homogeneous freezing of cloud water (SAM module_mp_graupel.f90:4661):
    # below ≈−40 °C all supercooled cloud water freezes instantly to cloud ice
    # (droplet number → ice number), releasing L_f. Smooth sigmoid threshold
    # for AD; the rain analog is handled by the steep Bigg rate.
    do_homogeneous_freezing: bool = True
    homogeneous_freeze_T: float = 233.15      # −40 °C threshold [K]
    # Sharp sigmoid (~0.5 K transition) to approximate SAM's hard T≤233.15
    # switch closely while staying AD-smooth (codex iter-25: 5/K was too soft —
    # half-froze at exactly −40 °C and leaked above). The huge N_i this can
    # inject (full droplet number) is bounded downstream by the LAMI clamp.
    homogeneous_freeze_sharpness: float = 20.0  # sigmoid sharpness [1/K]
    # Ice → snow autoconversion:
    #   "m2005_autoconv" (default, faithful) = SAM PRCI: depositional growth
    #     of the cloud-ice PSD across the snow-size threshold DCS converts ice
    #     to snow (module_mp_graupel.f90:3322-3326). Deposition-driven (only
    #     when ice-supersaturated); computed in the m2005 deposition block, so
    #     it REQUIRES ice_deposition_scheme="m2005" (else it is 0). The number
    #     transfer NPRCI = PRCI/CONS22 (capped at N_i/dt) IS wired: it debits
    #     N_i always and credits N_s under double-moment snow (an earlier
    #     "dropped — single-moment snow" note here was stale).
    #   "heuristic" = the legacy constant-rate agg_coeff·q_i·f_ice.
    ice_to_snow_scheme: str = "m2005_autoconv"
    ice_snow_d_auto: float = 250.0e-6   # SAM DCS [m] (clice_snow_Dauto)
    # Aggregation (heuristic ice→snow rate)
    agg_coeff: float = 1e-3          # Ice-to-snow aggregation rate [1/s]
    # Melting
    melt_rate: float = 5e-3          # Melting rate [1/s]
    melt_sharpness: float = 2.0      # Sigmoid sharpness near T_freeze
    # Ice sedimentation
    a_v_i: float = 50.0              # Ice fall speed coefficient [m^(1-b)/s]
    b_v_i: float = 0.25              # Ice fall speed exponent
    # Snow sedimentation
    a_v_s: float = 30.0              # Snow fall speed coefficient
    b_v_s: float = 0.3               # Snow fall speed exponent
    # === M2005 PSD mass-weighted fall speeds (iter-15 M5) ===
    # "m2005_psd" (default, faithful): rain + cloud ice fall speeds are the
    #   SAM mass-weighted moments of the PSD using the PROGNOSTIC N_r, N_i
    #   (module_mp_graupel.f90:1854-1865, 4230-4248):
    #     LAMR = (π·ρ_w·N_r/(ρ·q_r))^⅓   (clamped lamr_min..lamr_max)
    #     UMR  = AR·Γ(4+BR)/6 · LAMR^−BR · (ρ_su/ρ)^0.54   (cap 9.1·dum)
    #     LAMI = (ρ_ci·π·N_i/q_i)^⅓       (clamped lami_min..lami_max)
    #     UMI  = AI·Γ(4+BI)/6 · LAMI^−BI · (ρ_su/ρ)^0.35   (cap 1.2·(ρ_su/ρ)^0.35)
    #   Snow FALL SPEED: default is single-moment bulk q-power (V_t below); a
    #   prognostic N_s selects the SAM double-moment PSD speed (LAMS, UMS/UNS).
    #   Graupel FALL SPEED: default is a single-moment fixed-N0G PSD closure (a
    #   legoESM Marshall-Palmer closure, NOT SAM); a prognostic N_g selects the
    #   SAM double-moment PSD speed (LAMG, UMG/UNG). test_m2005_fall_speed_faithful
    #   pins the SAM double-moment speeds (the fixed-N0G graupel closure is out of
    #   the SAM-oracle scope). (Number-budget/nucleation double-moment handling is
    #   documented at its own sites, not here.)
    # "bulk_qpower" = the legacy V_t = a_v·(q·ρ/ρ_sfc)^b_v for RAIN/ICE/SNOW.
    #   fall_speed_scheme selects only the rain/ice/snow speed; GRAUPEL is
    #   computed separately (always its PSD closure: fixed-N0G, or double-moment
    #   from a prognostic N_g) regardless of fall_speed_scheme.
    fall_speed_scheme: str = "m2005_psd"
    fall_a_r: float = 841.99667      # SAM AR rain fall-speed coeff [m^(1-BR)/s]
    fall_b_r: float = 0.8            # SAM BR rain fall-speed exponent
    fall_a_i: float = 700.0          # SAM AI cloud-ice fall-speed coeff
    fall_b_i: float = 0.865          # gSAM BI = clice_fall_b (MK tune,
                                     # micro_params.f90:62; the M2005-ORIGINAL
                                     # 1.0 fell ~3x too slow → anvil cloud-ice
                                     # over-accumulated. gSAM v=0.03/0.24/1.78
                                     # vs M2005-orig 0.01/0.07/0.7 m/s).
    # Standard air density at 850 mb (SAM RHOSU = 85000/(R_d·273.15)); the
    # (ρ_su/ρ)^0.54 factor accelerates fall speeds in thin upper-trop air.
    rho_su: float = 8.5e4 / (constants.R_d * constants.T_freeze)
    lamr_min: float = 1.0 / 2800.0e-6   # SAM LAMMINR (slope floor) [1/m]
    lamr_max: float = 1.0 / 20.0e-6     # SAM LAMMAXR (slope cap) [1/m]
    lami_min: float = 1.0 / 600.0e-6    # SAM LAMMINI=1/(2·DCS+100µm), DCS=250µm
    lami_max: float = 1.0 / 1.0e-6      # SAM LAMMAXI [1/m]
    # SNOW PSD (double-moment, used only when N_s is prognostic). LAMS=
    # (π·ρ_sn·N_s/q_s)^⅓ (per-mass N_s), UMS=AS·Γ(4+BS)/6/LAMS^BS·(ρ_su/ρ)^0.54,
    # UNS=AS·Γ(1+BS)/LAMS^BS·… (SAM AS=11.72, BS=0.41, RHOSN=100).
    rho_snow: float = 100.0             # SAM RHOSN snow bulk density [kg/m³]
    fall_a_s: float = 11.72             # SAM AS snow fall-speed coeff
    fall_b_s: float = 0.41              # SAM BS snow fall-speed exponent
    lams_min: float = 1.0 / 2000.0e-6   # SAM LAMMINS [1/m]
    lams_max: float = 1.0 / 10.0e-6     # SAM LAMMAXS [1/m]
    # Snow vapor deposition/sublimation PRDS (double-moment snow only): EPSS=
    # 2π·N0S·ρ·DV·[F1S/LAMS²+F2S·CONS10·(ASN·ρ/μ)^½·SC^⅓·LAMS^(1−CONS35)],
    # PRDS=EPSS·(q_v−q_sat_i)/ABI (SAM F1S=0.86, F2S=0.28). Grows snow by vapor
    # deposition (anvil) and sublimates it in dry downdrafts (cooling).
    do_snow_deposition: bool = True
    snow_vent_f1: float = 0.86          # SAM F1S snow ventilation coefficient
    snow_vent_f2: float = 0.28          # SAM F2S snow ventilation coefficient
    # Snow riming of cloud water PSACWS (double-moment snow only): PSACWS=
    # Γ(BS+3)·π/4·ECI·ASN·q_c·ρ·N0S/LAMS^(BS+3) — PSD collection of supercooled
    # droplets by falling snow, freezing onto it (riming, +L_f). Replaces the
    # crude rime_coeff·q_s·q_c·f_ice.
    do_snow_riming: bool = True
    snow_collect_eff: float = 0.7       # SAM ECI snow-droplet collection eff.
    # Snow melting PSMLT (double-moment snow only): heat-balance-limited melting
    # melt=2π·N0S·KAP·(T−T0)₊/L_f·[F1S/LAMS²+F2S·ventilation], KAP=1.414e3·μ
    # (air thermal conductivity). Replaces the crude bulk melt_rate·q_s.
    do_snow_melting: bool = True
    # Snow self-aggregation NSAGG (double-moment snow only): Passarelli-1978 /
    # Reisner-1998. Falling flakes collide+merge ⇒ snow NUMBER sink (mass
    # conserved). NSAGG=CONS15·ASN·ρ^((2+BS)/3)·q_s^((2+BS)/3)·(N_s·ρ)^((4−BS)/3)/ρ
    # with CONS15=−1108·EII·π^((1−BS)/3)·ρ_sn^((−2−BS)/3)/(4·720).
    do_snow_aggregation: bool = True
    snow_aggregation_eii: float = 0.1   # SAM EII snow-snow collection eff.
    # Graupel (M4, iter-34): SAM freezes supercooled rain to GRAUPEL (dense
    # frozen drops), not snow. Graupel in slot [5] is single-moment (fixed
    # intercept N0G, a legoESM Marshall-Palmer closure — SAM graupel is itself
    # two-moment) by default, or double-moment (PSD slope LAMG from prognostic
    # N_g) when N_g is supplied; SAM module_mp_graupel.f90: AG=19.3, BG=0.37,
    # RHOG=400, LAMMING/LAMMAXG slope limits. do_graupel=False ⇒ frozen-rain→snow.
    do_graupel: bool = True
    rho_graupel: float = 400.0          # SAM RHOG graupel bulk density [kg/m³]
    fall_a_g: float = 19.3              # SAM AG graupel fall-speed coeff
    fall_b_g: float = 0.37              # SAM BG graupel fall-speed exponent
    n0_graupel: float = 4.0e6           # fixed intercept N0G [1/m⁴] (single-mom.)
    lamg_min: float = 1.0 / 2000.0e-6   # SAM LAMMING [1/m] = 500
    lamg_max: float = 1.0 / 20.0e-6     # SAM LAMMAXG [1/m] = 50000
    graupel_vent_f1: float = 0.86       # SAM F1S (graupel shares snow vent params)
    graupel_vent_f2: float = 0.28       # SAM F2S
    do_graupel_melting: bool = True     # PGMLT graupel→rain (heat-balance)
    # Graupel riming PSACWG (iter-35): graupel collects supercooled cloud water
    # (q_c→q_g + L_f). SAM PSACWG=CONS14·AGN·q_c·ρ·N0G/LAMG^(BG+3),
    # CONS14=Γ(BG+3)·π/4·ECI. The primary graupel GROWTH mechanism in updrafts.
    do_graupel_riming: bool = True
    graupel_collect_eff: float = 0.7    # SAM ECI graupel-droplet collection eff.
    # Graupel vapor deposition/sublimation PRDG (iter-36): the graupel analog of
    # snow PRDS. PRDG=EPSG·(q_v−q_sat_i)/ABI, EPSG=2π·N0G·DV·[F1S/LAMG²+F2S·vent].
    # Deposition grows graupel (+L_s); sublimation cools dry downdrafts (−L_s).
    do_graupel_deposition: bool = True
    # Graupel rain accretion PRACG (iter-37): graupel collects rain (two-PSD
    # gravitational collection). Cold branch (T<0): rain→graupel + L_f. SAM
    # PRACG=CONS41·VDIFF·ρ·N0RR·N0G/LAMR³·[5/(LAMR³·LAMG)+2/(LAMR²·LAMG²)+
    # 0.5/(LAMR·LAMG³)], CONS41=π²·ECR·ρ_w. The warm-branch graupel→rain shedding
    # is deferred (partially captured by PGMLT melting).
    do_graupel_rain_accretion: bool = True
    graupel_rain_collect_eff: float = 1.0   # SAM ECR rain-graupel collection eff.
    # Snow→graupel conversion PGSACW (iter-38): heavily-rimed snow densifies to
    # graupel (Rutledge-Hobbs 1984 / Reisner 1998). PGSACW=min(PSACWS, CONS17·dt·
    # N0S·q_c²·ASN²/(ρ·LAMS^(2BS+2))), CONS17=3·ρ_su·π·ECI²·Γ(2BS+2)/(ρ_g−ρ_sn).
    # Gated on q_s≥0.1 g/kg AND q_c≥0.5 g/kg. Snow number sink NSCNG via MG0.
    do_snow_to_graupel: bool = True
    graupel_embryo_mass: float = 1.6e-10    # SAM MG0 graupel embryo mass [kg]
    # Aerosol -> cloud-droplet-number ACTIVATION scheme selector (opt-in).
    # Consumed only when ``nc_from_aerosol=True``: "proxy" (default) keeps the
    # Andreae (2009) AOD->CCN diagnostic byte-identical; "arg" switches to the
    # physically-based Abdul-Razzak & Ghan (2000) modal activation
    # (``arg_activation.activated_nc_field``). Appended LAST (nested config,
    # non-float) so positional construction and the param spec are unaffected.
    activation: ActivationConfig = ActivationConfig()
    # --- Hard (iterated) saturation-adjustment guard (opt-in) ---
    # Iterated (bracketed-bisection) saturation adjustment that lands q_v ON liquid
    # saturation curve where q_v > hard_sat_adjust_threshold * q_sat, draining
    # local super-saturation pools the smooth sigmoid path cannot (conserving
    # c_pd*T + L_v*q_v exactly).  Default OFF => byte-identical to the smooth
    # path.  Threaded from ExperimentConfig via
    # ``apply_microphysics_experiment_flags``.  See
    # ``_warm_rain.saturation_adjustment``.
    hard_saturation_adjustment: bool = False
    hard_sat_adjust_threshold: float = 1.1      # RH trigger q_v > thr*q_sat [-]
    hard_sat_max_heating_K: float = 5.0         # per-step latent-heating cap [K]


class ThompsonConfig(NamedTuple):
    """Configuration for Thompson hybrid-moment microphysics."""
    # All Morrison params
    k_au: float = 6e2
    x_star: float = 2.6e-10
    Nc_0: float = 1e8
    k_ac: float = 5.25
    k_sc: float = 1e-3
    D_eq: float = 1.1e-3
    breakup_sharpness: float = 1e4
    a_v_r: float = 130.0
    b_v_r: float = 0.5
    evap_coeff: float = 1.0
    saturation_sharpness: float = 100.0
    autoconversion_sharpness: float = 10.0  # See SB config — iter-97/99
    # Relative-humidity-deficit resolution floor for rain evaporation: rain
    # does not evaporate where the liquid sub-saturation is below ~RH 99.995 %.
    # Suppresses the spurious in-cloud evaporation that rectifies float32
    # round-off and inflated the cloudy-column LWP (18-24 % DYCOMS fp32-vs-fp64
    # spread). Applied to the deficit itself, so resolved deficits (WBF ~0.1,
    # sub-cloud downdrafts) lose only the ≲ floor/deficit fraction (≲0.05 %).
    # Calibrated: the measured float32 in-cloud deficit-noise band is ~9e-6
    # (max), and the DYCOMS cold-variant in-cloud evaporation sits at deficit
    # ~1e-5-5e-5; 5e-5 is the smallest floor that clears BOTH (a 1-2e-5 floor
    # bisects that band and makes the cold-case fp32-vs-fp64 spread WORSE).
    # See _warm_rain.rain_evaporation.
    rain_evap_rh_floor: float = 5.0e-5
    # Cooper(1986) base ice number [1/m³] (= 0.005/L).  Was 5e3 — 1000x too
    # high (the canonical Cooper base is 0.005/L = 5/m³, the value MorrisonConfig
    # uses); 5e3 pinned all clouds colder than ~-15 C at the N_i_nuc_max cap.
    N_i0: float = 5.0
    cooper_a: float = 0.304
    # SAM "limit to 500 L⁻¹" cap on Cooper-nucleated ice number. Without it
    # the bare ``N_i0·exp(cooper_a·(T_freeze−T))`` diverges at very cold
    # tropopause/sponge temperatures and overflows fp32 → N_i = inf/NaN.
    # Mirrors ``MorrisonConfig.N_i_nuc_max``.
    N_i_nuc_max: float = 5.0e5       # [1/m³] = 500 /L
    cooper_T_act: float = 265.0
    ice_sigmoid_sharpness: float = 5.0
    dep_coeff: float = 1e-3
    q_i_min_growth: float = 1e-9
    # --- Faithful Thompson-2008 ice depositional growth + ice→snow ---
    # Capacitance-based vapour-diffusion ice growth (Thompson et al. 2008,
    # following Reisner et al. 1998 / the M2005 lineage) replaces the legacy
    # ``dep_coeff·S_i·q_i·N_i^⅓`` heuristic when ``ice_growth_scheme="capacitance"``.
    # PRCI depositional ice→snow autoconversion converts cloud ice whose
    # depositional growth carries it ACROSS the snow-size threshold ``D_cs``
    # into snow — the actual Thompson-2008 ice→snow mechanism, far stronger in
    # supersaturated convective cores than the ``agg_coeff·q_i`` relaxation, so
    # cloud ice drains to fast-falling snow instead of piling up and driving a
    # latent-heating convective runaway.
    ice_growth_scheme: str = "capacitance"   # "capacitance" | "heuristic"
    # Snow microphysics: "thompson2008" = faithful bimodal-PSD snow (Field-2005
    # moments, mass-weighted fall speed + ventilated vapour deposition, in
    # ``_thompson_snow.py``); "bulk_qpower" = legacy capped power-law fall speed
    # with no snow deposition.
    snow_scheme: str = "thompson2008"        # "thompson2008" | "bulk_qpower"
    rho_cloud_ice: float = 500.0             # cloud-ice bulk density [kg/m³]
    ice_deposition_efficiency: float = 1.0   # EPSI tuning [-]
    ice_snow_d_auto: float = 250.0e-6        # D_cs ice→snow size threshold [m]
    bergeron_rate: float = 1e-3
    T_center: float = 258.0
    T_width: float = 10.0
    rime_coeff: float = 1.0
    agg_coeff: float = 1e-3
    melt_rate: float = 5e-3
    melt_sharpness: float = 2.0
    a_v_i: float = 50.0
    b_v_i: float = 0.25
    a_v_s: float = 30.0
    b_v_s: float = 0.3
    # Graupel
    rime_to_graupel_threshold: float = 1e-4  # Riming threshold for graupel [kg/kg/s]
    rime_to_graupel_rate: float = 0.5        # Fraction converted to graupel
    graupel_sharpness: float = 1e4           # Sigmoid sharpness
    a_v_g: float = 80.0                      # Graupel fall speed coefficient
    b_v_g: float = 0.4                       # Graupel fall speed exponent
    # Gamma distribution shape
    mu_c: float = 3.0                        # Cloud droplet shape parameter
    mu_r: float = 1.0                        # Rain drop shape parameter
    # --- Hard (iterated) saturation-adjustment guard (opt-in) ---
    # See ``_warm_rain.saturation_adjustment`` / ``KesslerConfig``.  Newton-
    # iterated saturation adjustment that lands q_v ON the liquid saturation
    # curve where q_v > hard_sat_adjust_threshold * q_sat (conserving
    # c_pd*T + L_v*q_v exactly).  Default OFF => byte-identical smooth path.
    hard_saturation_adjustment: bool = False
    hard_sat_adjust_threshold: float = 1.1   # RH trigger q_v > thr*q_sat [-]
    hard_sat_max_heating_K: float = 5.0      # per-step latent-heating cap [K]


class P3Config(NamedTuple):
    """Configuration for P3 (Predicted Particle Properties) microphysics.

    Single ice category with predicted rime mass (q_rim) and rime volume
    (B_rim). Liquid phase uses Seifert-Beheng warm-rain helpers.

    Ice particle properties (fall speed, density) are diagnosed from
    (q_i, N_i, q_rim, B_rim) rather than assumed from a fixed habit.

    Slot reuse in HydrometeorState
    --------------------------------
    q_s → q_rim  [kg/kg]        rime mass mixing ratio
    q_g → B_rim  [m³/kg_air]    rime volume per unit air mass

    The rime density rho_rim = q_rim / B_rim [kg/m³] spans the full
    range from unrimed aggregates (~50 kg/m³) to dense graupel
    (~900 kg/m³).
    """
    # --- Warm rain (Seifert-Beheng liquid phase) ---
    k_au: float = 6e2
    x_star: float = 2.6e-10
    Nc_0: float = 1e8
    k_ac: float = 5.25
    k_sc: float = 1e-3
    D_eq: float = 1.1e-3
    breakup_sharpness: float = 1e4
    a_v_r: float = 130.0
    b_v_r: float = 0.5
    evap_coeff: float = 1.0
    saturation_sharpness: float = 100.0
    autoconversion_sharpness: float = 10.0
    # --- Ice nucleation (Cooper 1986, gSAM P3 scheme-1 semantics) ---
    # Was 5e3 — 1000x the canonical Cooper base (0.005/L = 5/m³, = MorrisonConfig).
    N_i0: float = 5.0               # Cooper base ice crystal number [1/m³] (= 0.005/L)
    cooper_a: float = 0.304         # Cooper exponent
    # ORACLE cap: gSAM P3 scheme-1 100 L⁻¹·SCF with SCF=1 (module_mp_p3.f90:3090;
    # scheme 2 uses 150 L⁻¹). Also bounds the Cooper exponential so cold
    # tropopause temperatures cannot overflow fp32 (mirrors Morrison).
    # (Was 5e5 = M2005's 500/L — a documented departure, closed 2026-07-17.)
    N_i_nuc_max: float = 1.0e5      # [1/m³] = 100 /L
    cooper_T_act: float = 265.0     # Mixed-phase process gate (dep/riming/agg) [K]
    # Nucleation-specific ORACLE gate (module_mp_p3.f90:3084): T < −15 °C AND
    # supi >= 0.05, smoothed by ice_sigmoid_sharpness / cooper_supi_sharpness.
    cooper_T_nuc: float = 258.15    # Nucleation activation temperature [K] (−15 °C)
    cooper_supi_min: float = 0.05   # Min ice supersaturation for nucleation [-]
    ice_sigmoid_sharpness: float = 5.0
    cooper_supi_sharpness: float = 200.0  # Sigmoid sharpness on supi gate [-]
    # --- Ice depositional growth ---
    dep_coeff: float = 1e-3
    q_i_min_growth: float = 1e-9    # Minimum effective q_i for deposition [kg/kg]
    # --- Cloud riming (ice collects cloud droplets) ---
    rime_coeff: float = 0.5         # Collection efficiency E_ri [-]
    # --- Rain riming (ice collects rain drops, freezes) ---
    rain_rime_coeff: float = 0.1    # Collection efficiency E_rr [-]
    # --- Self-collection / aggregation (N_i reduction) ---
    agg_coeff: float = 1e-3         # Aggregation rate [1/s]
    # --- Melting ---
    melt_rate: float = 5e-3         # Melting rate [1/s]
    melt_sharpness: float = 2.0     # Sigmoid sharpness near T_freeze [1/K]
    # --- P3 fall speed: V_t = a_v_i * (q_i*rho_ratio)^b_v_i * density_factor ---
    a_v_i: float = 40.0             # Base fall speed coefficient
    b_v_i: float = 0.3              # Fall speed exponent
    # --- Rime density limits [kg/m³] ---
    rho_rim_min: float = 50.0       # Minimum rime density (unrimed aggregates)
    rho_rim_max: float = 900.0      # Maximum rime density (dense graupel)
    # --- Fall speed density enhancement ---
    # V_t *= (rho_rim / rho_ice_ref)^c_rim_fallspeed
    c_rim_fallspeed: float = 0.4    # Density enhancement exponent [-]
    rho_ice_ref: float = 500.0      # Reference rime density for scaling [kg/m³]
    # --- Accreted rime density (for dB_rim/dt from riming) ---
    rho_rim_accrete: float = 400.0  # Density of newly accreted rime [kg/m³]
    # --- Bulk ice density (for B_rim from nucleation) ---
    rho_ice: float = constants.rho_ice  # Solid ice density [kg/m³]
    # --- Hard (iterated) saturation-adjustment guard (opt-in) ---
    # See ``_warm_rain.saturation_adjustment`` / ``KesslerConfig``.  Newton-
    # iterated saturation adjustment that lands q_v ON the liquid saturation
    # curve where q_v > hard_sat_adjust_threshold * q_sat (conserving
    # c_pd*T + L_v*q_v exactly).  Default OFF => byte-identical smooth path.
    hard_saturation_adjustment: bool = False
    hard_sat_adjust_threshold: float = 1.1   # RH trigger q_v > thr*q_sat [-]
    hard_sat_max_heating_K: float = 5.0      # per-step latent-heating cap [K]


class MicrophysicsMLEmulatorConfig(NamedTuple):
    """Configuration for ML microphysics emulator (Equinox MLP)."""
    n_input: int = 9
    n_hidden: int = 128
    n_layers: int = 3
    n_output: int = 7
    seed: int = 0
    use_residual: bool = True
    norm_T: float = 300.0       # Temperature scale [K] for input normalization
    norm_q_factor: float = 1e3  # q_v / q_c / q_r / q_i scale
    norm_rho: float = 1.2       # Air density scale [kg/m^3]
    norm_dz: float = 1000.0     # Layer thickness scale [m]
    norm_dt: float = 3600.0     # Time-step scale [s]


class MicrophysicsConfig(NamedTuple):
    """Top-level microphysics configuration.

    Selects the active scheme and holds sub-configurations.

    Fields
    ------
    scheme : str
        Active scheme: "kessler", "sundqvist", "seifert_beheng",
        "morrison", "thompson", "p3", "sdm", "fast_sbm", "ml_emulator",
        or "none".
    kessler : KesslerConfig
    sundqvist : SundqvistConfig
    seifert_beheng : SeifertBehengConfig
    morrison : MorrisonConfig
    thompson : ThompsonConfig
    p3 : P3Config
    sdm : SDMConfig
        Super-Droplet Method (Shima et al. 2009). The column path is a
        diffusional-condensation adapter; the full Lagrangian model is in
        ``microphysics/sdm/box_model.py``.
    fast_sbm : FastSBMConfig
        Fast spectral-bin microphysics (WRF FSBM-2 port). The column path
        reconstructs the 33-bin liquid spectrum from bulk (q_c, q_r, N_r)
        and runs oracle condensation + Bott coalescence per step (see
        ``docs/science/specs/bin_microphysics.md``).
    ml_emulator : MicrophysicsMLEmulatorConfig
    """
    scheme: str = "none"
    kessler: KesslerConfig = KesslerConfig()
    sundqvist: SundqvistConfig = SundqvistConfig()
    seifert_beheng: SeifertBehengConfig = SeifertBehengConfig()
    morrison: MorrisonConfig = MorrisonConfig()
    thompson: ThompsonConfig = ThompsonConfig()
    p3: P3Config = P3Config()
    sdm: SDMConfig = SDMConfig()
    fast_sbm: FastSBMConfig = FastSBMConfig()
    ml_emulator: MicrophysicsMLEmulatorConfig = MicrophysicsMLEmulatorConfig()


def apply_microphysics_experiment_flags(
    scheme_config,
    scheme: str,
    *,
    nc_from_aerosol: bool = False,
    subgrid_autoconversion: bool = False,
    hard_saturation_adjustment: bool = False,
    hard_sat_adjust_threshold: float | None = None,
    hard_sat_max_heating_K: float | None = None,
    homogeneous_ice_nucleation: bool = False,
):
    """Thread ExperimentConfig-level microphysics switches onto a per-scheme
    sub-config NamedTuple, raising LOUDLY on a scheme that lacks the field.

    The switches are warm-rain closures implemented by the warm-rain-based
    schemes: Morrison's ``effective_Nc`` (specified-Nc aerosol mode) and
    in-cloud autoconversion, and the shared ``saturation_adjustment`` hard
    (iterated) saturation-adjustment guard (kessler/seifert_beheng/morrison/
    thompson/p3).  This single helper is shared by the coupled
    (``physics_pipeline._resolve_microphysics``) and the combined-physics /
    MPAS (``model_driver._run_mpas``) paths so the gating + fail-loud
    validation is written ONCE — a scheme that would silently ignore the
    flag raises instead, on either path (no duplicated dispatch).

    Parameters
    ----------
    scheme_config : NamedTuple
        The active per-scheme sub-config (e.g. ``MorrisonConfig``).
    scheme : str
        Scheme name, used only in the error message.
    nc_from_aerosol, subgrid_autoconversion, hard_saturation_adjustment : bool
        ExperimentConfig switches; when True the matching field is set on
        ``scheme_config`` (raising if the field is absent).
    hard_sat_adjust_threshold, hard_sat_max_heating_K : float or None
        Optional overrides of the hard saturation-adjustment RH trigger and
        per-step latent-heating cap [K] (ExperimentConfig flat scalars /
        ``--hard-sat-*`` CLI flags).  ``None`` (default) keeps the per-scheme
        ``__param_spec__`` defaults; a value raises if the scheme lacks the
        field (no silently-inert override).  Bounds are enforced upstream by
        ``ExperimentConfig.validate_strict`` (spec bounds (1, 2) / (0.5, 50)).

    Returns
    -------
    NamedTuple
        ``scheme_config`` with the requested flags applied (a new instance;
        unchanged when all switches are False).
    """
    fields = getattr(scheme_config, "_fields", ())
    if nc_from_aerosol:
        if "nc_from_aerosol" not in fields:
            raise ValueError(
                f"nc_from_aerosol=True is not supported by the {scheme!r} "
                "microphysics scheme (no specified-Nc aerosol mode); use "
                "--microphysics morrison or drop --aerosol-ccn."
            )
        scheme_config = scheme_config._replace(nc_from_aerosol=True)
    if subgrid_autoconversion:
        if "subgrid_autoconversion" not in fields:
            raise ValueError(
                f"subgrid_autoconversion=True is not supported by the "
                f"{scheme!r} microphysics scheme; use --microphysics morrison "
                "or drop --subgrid-autoconversion."
            )
        scheme_config = scheme_config._replace(subgrid_autoconversion=True)
    if hard_saturation_adjustment:
        if "hard_saturation_adjustment" not in fields:
            raise ValueError(
                f"hard_saturation_adjustment=True is not supported by the "
                f"{scheme!r} microphysics scheme (no warm-rain saturation "
                "adjustment); use a warm-rain scheme (kessler, seifert_beheng, "
                "morrison, thompson, p3) or drop --hard-saturation-adjustment."
            )
        scheme_config = scheme_config._replace(hard_saturation_adjustment=True)
    if homogeneous_ice_nucleation:
        if "homogeneous_ice_nucleation" not in fields:
            raise ValueError(
                f"homogeneous_ice_nucleation=True is not supported by the "
                f"{scheme!r} microphysics scheme (no Koop/Ren-MacKenzie "
                "cirrus nucleation); use --microphysics morrison or drop "
                "--homogeneous-ice-nucleation."
            )
        scheme_config = scheme_config._replace(homogeneous_ice_nucleation=True)
    for _field, _val in (("hard_sat_adjust_threshold", hard_sat_adjust_threshold),
                         ("hard_sat_max_heating_K", hard_sat_max_heating_K)):
        if _val is None:
            continue
        if _field not in fields:
            raise ValueError(
                f"{_field}={_val!r} is not supported by the {scheme!r} "
                "microphysics scheme (no warm-rain hard saturation "
                "adjustment); use a warm-rain scheme (kessler, "
                "seifert_beheng, morrison, thompson, p3) or drop the "
                "override."
            )
        scheme_config = scheme_config._replace(**{_field: float(_val)})
    return scheme_config
