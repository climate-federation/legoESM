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
from legoesm.thermo import CAM_QSAT_CAP_SHARPNESS
from legoesm.atmosphere.physics.microphysics.arg_activation import ActivationConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig


# ---------------------------------------------------------------------------
# Uniform super-saturation guard: the single source of truth
# ---------------------------------------------------------------------------
# EVERY microphysics scheme reachable through ``microphysics.integration.
# _get_microphysics_fn`` must either carry the hard (iterated) saturation-
# adjustment guard with these exact defaults, or appear in
# ``HARD_SAT_GUARD_EXEMPT`` with a real reason.  Enforced mechanically by
# ``tests/test_microphysics_supersaturation_guard.py`` (shrink-only exemption
# list + synthetic-violation self-tests).
#
# The three fields, identical on every guarded scheme:
#   hard_saturation_adjustment : bool  -- opt-in gate; False => byte-identical
#   hard_sat_adjust_threshold  : float -- RH trigger, fires where q_v > thr*q_sat
#   hard_sat_max_heating_K     : float -- per-step latent-heating cap [K]
HARD_SAT_GUARD_FIELDS = (
    "hard_saturation_adjustment",
    "hard_sat_adjust_threshold",
    "hard_sat_max_heating_K",
)
HARD_SAT_GUARD_DEFAULTS = {
    "hard_saturation_adjustment": False,
    "hard_sat_adjust_threshold": 1.1,
    "hard_sat_max_heating_K": 5.0,
}
# Schemes carrying the guard (kept sorted; used in the fail-loud messages below
# and asserted against the factory by the ratchet).
HARD_SAT_GUARD_SCHEMES = (
    "ml_emulator", "morrison", "p3", "sundqvist", "thompson",
)
# SHRINK-ONLY exemptions.  Each reason was verified in the scheme's own source,
# not inferred from its name (CLAUDE.md: every allow-list reason string is a
# claim).  Adding an entry needs the same standard of evidence.
HARD_SAT_GUARD_EXEMPT = {
    "sdm": (
        "Super-Droplet Method: WHERE DROPLETS EXIST, super-saturation is a "
        "PROGNOSTIC quantity, not a residual to be clipped -- "
        "sdm/column.py computes the saturation ratio via "
        "relative_humidity(T, p_full, q_v) and "
        "sdm/condensation.py::integrate_radius advances the Maxwell/Koehler "
        "growth law dr/dt against (S-1), so the vapour sink is the resolved "
        "condensational growth of the population. Overlaying an instantaneous "
        "on-curve adjustment would double-count that sink and destroy the "
        "activation/relaxation transient the method exists to represent. "
        "KNOWN LIMITATION THIS EXEMPTION DOES NOT COVER: the default "
        "condensation-only adapter (column_do_coalescence=False) has NO "
        "aerosol-Koehler activation, so with q_c = 0 the reconstruction gives "
        "N_eff = 0 and a super-saturated CLEAR cell is left undrained at any "
        "RH (its own comment says so: 'A supersaturated clear cell still "
        "produces nothing'). That is a missing-ACTIVATION defect in SDM, "
        "pinned by test_sdm_clear_cell_supersaturation_is_a_known_gap; the "
        "fix is activation (the column_do_coalescence=True branch already "
        "nucleates cdnc embryos where S > 1), NOT bolting a bulk saturation "
        "adjustment onto a super-droplet scheme."
    ),
    "fast_sbm": (
        "Fast spectral-bin microphysics: same reason as sdm, at bin "
        "resolution. fast_sbm/supersaturation.py::integrate_supersaturation "
        "advances the super-saturation relaxation equation explicitly and "
        "fast_sbm/diffusional_growth.py grows each of the 33 bins with "
        "dm_k/dt = B_k*s. The bin spectrum IS the adjustment; forcing q_v onto "
        "the bulk saturation curve on top of it would be inconsistent with the "
        "resolved spectrum and break the WRF FSBM-2 oracle match."
    ),
    "kessler": (
        "Kessler is the faithful port of ESCOMP atmos_phys0_31_000 "
        "kessler_run (user decision 2026-10-19: the scheme named Kessler must "
        "BE the oracle). The oracle has no hard-adjustment guard: its own "
        "Durran-Klemp (1983) adjustment max(prod, -q_c) removes the full "
        "linearised supersaturation in every CFL subcycle. KesslerConfig keeps "
        "the three guard fields only so an old deck fails loudly: "
        "kessler.py::_require_oracle_defaults raises on any non-default value."
    ),
    "seifert_beheng": (
        "The scheme is gSAM 1.8.8 M2005 (liquid only, SB2001 warm rain): "
        "seifert_beheng.py calls saturation_adjustment_gsam (gSAM's "
        "satadj_liquid, a Newton solve onto the liquid saturation curve that "
        "condenses AND evaporates) on EVERY call before any process rate, so "
        "the state never leaves the call super-saturated w.r.t. liquid. The "
        "opt-in guard has no gSAM counterpart: setting it raises (oracle port "
        "2026-10-20; the guard fields stay on SeifertBehengConfig at their "
        "standard defaults)."
    ),
    "none": (
        "microphysics='none' has NO scheme function at all -- "
        "_get_microphysics_fn returns (name, None, None) and the physics "
        "pipeline early-returns before any condensation is computed. There is "
        "no code path in which a guard could act."
    ),
}


__param_spec__ = {
    "KesslerConfig": {
        "scheme_key": "atm.micro.KesslerConfig",
        "excluded": {
            "saturation_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "evaporation_coeff": "not in the oracle (ESCOMP atmos_phys0_31_000 kessler_run): a non-default or traced value raises",
            "rain_fall_speed": "not in the oracle (ESCOMP atmos_phys0_31_000 kessler_run): a non-default or traced value raises",
            "hard_sat_adjust_threshold": "not in the oracle (ESCOMP atmos_phys0_31_000 kessler_run): a non-default or traced value raises",
            "hard_sat_max_heating_K": "not in the oracle (ESCOMP atmos_phys0_31_000 kessler_run): a non-default or traced value raises",
        },
        "params": {
            # PRIMARY warm-rain knobs (Kessler 1969).
            "autoconversion_threshold": {"units": "kg/kg", "bounds": (0.0001, 0.003), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Kessler (1969)", "shape": None},
            "autoconversion_rate": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Kessler (1969)", "shape": None},
            "accretion_coeff": {"units": "1", "bounds": (0.5, 6.6), "tunable_tier": 1, "transform": "sigmoid", "category": "accretion", "reference": "Kessler (1969)", "shape": None},
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
        # so none of THOSE are spec-eligible trainable params. The two
        # hard-saturation-guard floats ARE real closure knobs (identical
        # meaning + bounds to every other microphysics scheme's).
        "params": {
            "hard_sat_adjust_threshold": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
            "hard_sat_max_heating_K": {"units": "K", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
        },
    },
    "MorrisonConfig": {
        "scheme_key": "atm.micro.MorrisonConfig",
        "excluded": {
            "gsam_satadj_excess_sharpness": "numerics: smooth-switch sharpness (RULE 5), value approved by user 2026-10-21",
            "gsam_satadj_qc_floor_sharpness": "numerics: smooth-switch sharpness (RULE 5), value approved by user 2026-10-21",
            "gsam_satadj_p_min_width_pa": "numerics: smooth-switch width (RULE 5), value approved by user 2026-10-21",
            "sb2001_autoconv_ramp_top": "numerics: smooth-switch width (RULE 5), value decided by codex+GLM 2026-10-29 (#2232)",
            "cam_qsat_cap_sharpness": "numerics: smooth-switch sharpness (RULE 5), value approved by user 2026-10-21",
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
            "mg2_ice_slope_log_width": "numerics: solver/smoothing/tolerance/iteration parameter",
            "mg2_qi_limit_width": "numerics: solver/smoothing/tolerance/iteration parameter",
            "mg2_presence_ramp_top": "numerics: solver/smoothing/tolerance/iteration parameter",
            "mg2_frozen_rain_ice_ramp_frac": "numerics: solver/smoothing/tolerance/iteration parameter",
            "nuc_T_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "nuc_rh_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "saturation_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "subgrid_rh_crit": "physics-fidelity sub-grid closure: in-cloud cf critical RH, mirrors the cloud scheme (no tunable knob)",
            "subgrid_cf_min": "numerics: cloud-fraction floor capping the in-cloud enhancement (AD/numeric safety)",
            "kk2000_cam6_relvar": "physics-fidelity: CAM6 relvarmax constant (clubb_intr.F90:2416-2421) used by kk2000_cam6 only when no per-cell relvar is passed; the CLUBB liquid-partition lanes pass CLUBB's per-cell relvar (clubb_intr.F90:2425), so this is a fixed reference value, not a calibration knob",
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
            # Upper bound widened 2100 -> 6300 (user, 2026-09-09): the production
            # run sits EXACTLY on the old ceiling, so the ice-fallout /
            # precipitation-efficiency hypothesis for the tropical moist bias
            # could not be tested in either direction. A parameter pinned on its
            # own bound is not a tested parameter.
            "fall_a_i": {"units": "m^(1-b)/s", "bounds": (230.0, 6300.0), "tunable_tier": 3, "transform": "sigmoid", "category": "fall_speed", "reference": "gSAM M2005 default", "shape": None},
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
            # tier 1 (user decision 2026-10-14): the only warm-rain knobs the
            # kk2000 laws read, so the production core set trains warm rain.
            "autocon_fact": {"units": "1", "bounds": (0.1, 20.0), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "CAM6 MG2 kk2000 (micro_mg_utils.F90) = 0.01*1350*qc^2.47*Nc^-1.1 vs this port's 1350*qc^2.47*Nc^-1.79: ratio 0.01*Nc^0.69 = 0.10-0.51 over Nc 30-300 cm^-3; lower bound covers that CAM6-equivalent range; upper bound 20 leaves room for enhancement tests", "shape": None},
            "accre_enhan_fact": {"units": "1", "bounds": (0.1, 10.0), "tunable_tier": 1, "transform": "sigmoid", "category": "accretion", "reference": "CAM6 MG2 accre_enhan multiplier on KK2000 accretion (micro_mg_utils.F90 accrete_cloud_water_rain; micro_mg_cam.F90 sets it to 1); bounds one decade each way. MG2 pairs it with the cloud-number sink npra=pra*nc/qc, which this port lacks (N_c is not evolved when predict_Nc=False)", "shape": None},
        },
    },
    "P3Config": {
        "scheme_key": "atm.micro.P3Config",
        "excluded": {
            "cooper_supi_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "ice_sigmoid_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "melt_sharpness": "numerics: solver/smoothing/tolerance/iteration parameter",
            "rho_rim_min": "numerics: solver/smoothing/tolerance/iteration parameter",
        },
        "params": {
            # --- Specified droplet number (WRF P3 nccnst) ---
            "Nc_0": {"units": "1/m^3", "bounds": (1e7, 1e9), "tunable_tier": 2, "transform": "sigmoid", "category": "number_concentration", "reference": "WRF v4.8.0 P3 nccnst = 2e8 (module_mp_p3.F:196)", "shape": None},
            # --- Ice nucleation (Cooper 1986) ---
            "N_i0": {"units": "1/m^3", "bounds": (1.0, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "N_i_nuc_max": {"units": "1/m^3", "bounds": (2e4, 5e6), "tunable_tier": 3, "transform": "sigmoid", "category": "ice_nucleation", "reference": "WRF v4.8.0 P3 cap 100/L (module_mp_p3.F:3276)", "shape": None},
            "cooper_a": {"units": "1/K", "bounds": (0.1, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "Cooper (1986)", "shape": None},
            "cooper_T_nuc": {"units": "K", "bounds": (248.0, 266.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "WRF v4.8.0 P3 gate T<-15C (module_mp_p3.F:3272)", "shape": None},
            "cooper_supi_min": {"units": "1", "bounds": (0.0, 0.2), "tunable_tier": 2, "transform": "sigmoid", "category": "ice_nucleation", "reference": "WRF v4.8.0 P3 gate supi>=0.05 (module_mp_p3.F:3272)", "shape": None},
            # --- Rime density properties ---
            "rho_rim_max": {"units": "kg/m^3", "bounds": (300.0, 917.0), "tunable_tier": 3, "transform": "sigmoid", "category": "size_distribution", "reference": "Morrison & Milbrandt (2015) P3", "shape": None},
            "hard_sat_adjust_threshold": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
            "hard_sat_max_heating_K": {"units": "K", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
        },
    },
    "SeifertBehengConfig": {
        "scheme_key": "atm.micro.SeifertBehengConfig",
        "excluded": {
            "gsam_satadj_excess_sharpness": "numerics: smooth-switch sharpness (RULE 5), value approved by user 2026-10-21",
            "gsam_satadj_qc_floor_sharpness": "numerics: smooth-switch sharpness (RULE 5), value approved by user 2026-10-21",
            "gsam_satadj_p_min_width_pa": "numerics: smooth-switch width (RULE 5), value approved by user 2026-10-21",
            "sb2001_autoconv_ramp_top": "numerics: smooth-switch width (RULE 5), value decided by codex+GLM 2026-10-29 (#2232)",
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
            # Zhao-Carr namelist coefficients added by the oracle port; whether
            # (and within which bounds) they are trainable is a pending user decision.
            "psautco": "physics: Zhao-Carr namelist coefficient; trainability not yet decided",
            "wminco_liquid": "physics: Zhao-Carr namelist coefficient; trainability not yet decided",
            "wminco_ice": "physics: Zhao-Carr namelist coefficient; trainability not yet decided",
        },
        "params": {
            # PRIMARY large-scale-condensation knobs (Sundqvist et al. 1989).
            "rh_crit": {"units": "1", "bounds": (0.5, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "critical_rh", "reference": "Sundqvist et al. (1989)", "shape": None},
            "qc_crit": {"units": "kg/kg", "bounds": (0.0001, 0.0015), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist et al. (1989)", "shape": None},
            "auto_rate": {"units": "1/s", "bounds": (0.0001, 0.01), "tunable_tier": 1, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist et al. (1989)", "shape": None},
            "evap_coeff": {"units": "1", "bounds": (1.0e-5, 0.0015), "tunable_tier": 2, "transform": "sigmoid", "category": "evaporation", "reference": "Zhao-Carr evpco: GFS operational 2e-5 (zhaocarr_precpd.f:215); lower bound 1e-5 to hold it (user 2026-10-27). Sundqvist et al. (1989)", "shape": None},
            # SBK89 Sec. 5 precipitation-release enhancements: coalescence F1
            # (function of the precipitation flux from above) and Bergeron F2
            # (mixed-phase temperature window); c_0 is multiplied and q_c,crit
            # divided by F1*F2.
            "coalescence_enh_coeff": {"units": "(kg m^-2 s^-1)^-1/2", "bounds": (0.0, 1000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist, Berge & Kristjansson (1989)", "shape": None},
            "bergeron_enh_coeff": {"units": "1", "bounds": (0.0, 20.0), "tunable_tier": 2, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist, Berge & Kristjansson (1989)", "shape": None},
            "bergeron_T_peak_K": {"units": "K", "bounds": (248.0, 268.0), "tunable_tier": 3, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist, Berge & Kristjansson (1989)", "shape": None},
            "bergeron_T_width_K": {"units": "K", "bounds": (2.0, 15.0), "tunable_tier": 3, "transform": "sigmoid", "category": "autoconversion", "reference": "Sundqvist, Berge & Kristjansson (1989)", "shape": None},
            "hard_sat_adjust_threshold": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
            "hard_sat_max_heating_K": {"units": "K", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "hard saturation-adjustment guard", "shape": None},
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
    """Configuration for Sundqvist large-scale microphysics.

    The precipitation stage is a port of the Zhao-Carr scheme (ccpp-physics
    v7.0.1 ``zhaocarr_precpd_run``); see ``sundqvist.py`` for the field ->
    oracle-name mapping.  The oracle hard-codes ``qc_crit`` = 3e-4,
    ``coalescence_enh_coeff`` = 300 and ``bergeron_enh_coeff`` = 0.5.
    """
    rh_crit: float = 0.8              # Critical relative humidity (oracle u00k)
    # Smooth RH_crit gate of the condensation surrogate and of the precipitation
    # evaporation; the oracle is its hard-switch limit (sharpness -> inf).
    sigmoid_sharpness: float = 20.0
    auto_rate: float = 1e-3           # Liquid release rate c_0 [1/s] (oracle prautco)
    # Critical cloud water m_r of the SBK89 release
    # (1 - exp(-(m*F1*F2/(m_r*max(b, 0.01)))^2)) [kg/kg] (oracle 1/cmr).
    qc_crit: float = 3e-4             # oracle m_r (tier-2 decision R14)
    # Rain-evaporation coefficient (oracle evpco):
    # E = evap_coeff*sqrt(1/dt)*(rh_crit - RH)*(dp/g dt)*sqrt(P_r).
    evap_coeff: float = 2e-5          # GFS operational evpco (tier-2 decision R12)
    # --- SBK89 (Sec. 5) precipitation-release enhancements ---
    # Coalescence F1 = 1 + c1*sqrt(P/dt), P the precipitation [kg m^-2] per
    # step falling in from above (oracle c1).
    coalescence_enh_coeff: float = 300.0   # c1 [(kg m^-2 s^-1)^-1/2]
    # Bergeron-Findeisen F2 = 1 + c2*sqrt(min(max(0, 268 K - T), 20 K)) (oracle c2).
    bergeron_enh_coeff: float = 0.5        # c2 [K^-1/2], oracle (tier-2 decision R13)
    # Parameters of the PRE-PORT Gaussian F2; no counterpart in the oracle.
    # Setting either to a non-default value raises.
    bergeron_T_peak_K: float = constants.T_freeze - 15.0   # [K]
    bergeron_T_width_K: float = 7.0        # [K]
    # --- Hard (iterated) saturation-adjustment guard (opt-in) ---
    # The SAME guard the five bulk warm-rain schemes carry, with the SAME
    # defaults, applied to Sundqvist's OWN smooth condensation rate through the
    # shared ``_warm_rain.hard_saturation_blend`` (no second implementation).
    # Default OFF => the smooth condensation surrogate alone: a STATIC Python
    # ``if`` in ``sundqvist._sundqvist_step``, so the OFF path never even
    # traces the hard branch.
    #
    # What it changes when ON (Sundqvist is NOT the same failure mode as the
    # five bulk schemes -- documented honestly): Sundqvist's smooth branch
    # ``sigmoid(k*(RH - rh_crit)) * max(q_v - q_sat, 0)/dt`` gates on RH, an
    # O(1) argument, so the sigmoid SATURATES to ~1 at high RH and the full
    # supersaturation is removed toward ``q_sat(T_old)`` in ONE step.  It does
    # not under-drain; it OVERSHOOTS -- it ignores the latent heating that
    # raises q_sat (landing sub-saturated) and at RH 1.4 releases ~20 K of
    # latent heat in a single step.  The guard makes the landing point the
    # enthalpy-consistent equilibrium ``q_eq = q_sat(T + L_v/c_pd*(q_v-q_eq))``
    # and rate-limits the per-step heating to ``hard_sat_max_heating_K``, so a
    # large pool drains ONTO the curve over several steps instead of detonating.
    #
    # THRESHOLD SEMANTICS DIFFER ON THIS SCHEME -- read before tuning.  In
    # ``hard_saturation_blend`` the RH threshold controls only the smooth
    # activation ramp; the on-curve CAP ``min(blended, cond_hard)`` is applied
    # UNCONDITIONALLY.  For the five bulk schemes the smooth rate is divided by
    # the psychrometric factor, so it rarely exceeds ``cond_hard`` and the cap
    # seldom binds -- there the guard really does "fire above the threshold".
    # Sundqvist's smooth rate has NO psychrometric division, so it exceeds
    # ``cond_hard`` wherever the RH_crit sigmoid saturates: at T=300 K,
    # p=90 kPa, RH=1.02 the activation is only 0.083 yet the cap still cuts the
    # rate ~5x.  So on Sundqvist the guard changes the answer at essentially
    # every super-saturated cell and ``hard_sat_adjust_threshold`` is a WEAK
    # knob (it modulates the smooth/on-curve blend below the cap, not an
    # on/off trigger).  Its __param_spec__ tier-2 entry is kept for
    # cross-scheme uniformity of the sweep interface; do not expect the
    # (1.0, 2.0) sweep to behave like it does on Kessler.
    hard_saturation_adjustment: bool = False
    hard_sat_adjust_threshold: float = 1.1      # blend knob here, NOT an on/off trigger [-]
    hard_sat_max_heating_K: float = 5.0         # per-step latent-heating cap [K]
    # --- Zhao-Carr ice route and conversion thresholds ---
    # Ice -> snow autoconversion coefficient (oracle psautco) [1/s];
    # psaut = psautco*dt*exp(0.025*(T - 273.16))*max(0, cwm - wmini).
    psautco: float = 0.0
    # Condensate left unconverted, wmin = wminco*p/1e5 [kg/kg] (oracle
    # wminco(1) liquid, wminco(2) ice).
    wminco_liquid: float = 0.0
    wminco_ice: float = 0.0


class SeifertBehengConfig(NamedTuple):
    """Configuration for Seifert-Beheng two-moment warm rain.

    The scheme is gSAM 1.8.8 M2005 (liquid only, SB2001 warm rain, specified
    droplet number), which reads ONLY ``Nc_0`` (gSAM's NDCNST, per-volume).
    Every other field belonged to the former proxy rates and has no gSAM
    counterpart: ``seifert_beheng_microphysics`` raises if one is set.
    """
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
    # gSAM satadj_liquid smooth switches (RULE 5), as MorrisonConfig's; hard
    # limit = gSAM exactly: math.inf, math.inf, 0.0.  All three values approved
    # by user 2026-10-21.
    gsam_satadj_excess_sharpness: float = 1.0e4   # saturation step [-]
    gsam_satadj_qc_floor_sharpness: float = 1.0e4  # positivity, on T1 - T_l [1/K]
    gsam_satadj_p_min_width_pa: float = 100.0     # p >= 5000 Pa gate width [Pa]
    # RULE 5 smoothing of gSAM's SB2001 autoconversion gate q_c >= 1e-6 kg/kg
    # (module_mp_graupel.f90:1806/2843): the C2 step reaches 1 at this value
    # [kg/kg]; <= 1e-6 = gSAM's hard step (codex+GLM 2026-10-29, #2232; KNOWN
    # DIFFERENCE in _warm_rain.autoconversion_sb2001).
    sb2001_autoconv_ramp_top: float = 2.0e-6


# Unset Morrison DCS per flavour [m] (MorrisonConfig.ice_snow_d_auto=None,
# resolved by resolve_morrison_flavor): SAM clice_snow_Dauto, CAM6 MG2
# micro_mg_dcs (namelist_defaults_cam.xml:1244).
SAM_DCS_M = 250.0e-6
MG2_DCS_M = 500.0e-6


def resolve_unset_ice_snow_d_auto(cfg):
    """THE resolution of an unset Morrison DCS: by flavour (sam 250 um, mg
    500 um); an explicit value is returned unchanged.  Used by the scheme
    (resolve_morrison_flavor) and by the trainable collector's seed."""
    if cfg.ice_snow_d_auto is not None:
        return cfg
    if cfg.morrison_flavor == "mg":
        return cfg._replace(ice_snow_d_auto=MG2_DCS_M)
    if cfg.morrison_flavor == "sam":
        return cfg._replace(ice_snow_d_auto=SAM_DCS_M)
    return cfg      # unknown flavour: resolve_morrison_flavor raises


def mg2_rel_ice_params(micro_leaf):
    """``(dcs [m], ice-slope RULE 5 width)`` the CAM6 radiation ice radius
    reads from the run's microphysics leaf config (dcs resolved by flavour);
    ``(None, None)`` for a leaf without them, which that path refuses."""
    if micro_leaf is None or not hasattr(micro_leaf, "mg2_ice_slope_log_width"):
        return None, None
    m = resolve_unset_ice_snow_d_auto(micro_leaf)
    return m.ice_snow_d_auto, m.mg2_ice_slope_log_width


def resolve_warm_rain_scheme(cfg) -> str:
    """THE warm-rain law a Morrison config runs: an explicit
    ``warm_rain_scheme`` as set; unset (None) by flavour, ``mg`` -> the CAM6
    MG2 law ``kk2000_cam6``, ``sam`` -> gSAM M2005's ``kk2000``."""
    if cfg.warm_rain_scheme is not None:
        return cfg.warm_rain_scheme
    if cfg.morrison_flavor == "mg":
        return "kk2000_cam6"
    if cfg.morrison_flavor == "sam":
        return "kk2000"
    raise ValueError(
        f"Unknown morrison_flavor: {cfg.morrison_flavor!r}; choose 'mg' or "
        "'sam' (an unset warm_rain_scheme resolves by flavour).")


# Unset (None) spec'd fields -> the resolver that gives their value from a
# config instance.  The trainable collector seeds from it (the class default
# when no run config is passed, i.e. the default "mg" flavour), and the
# param-spec gate requires these fields to be classified.
__param_unset_defaults__ = {
    "MorrisonConfig": {"ice_snow_d_auto": resolve_unset_ice_snow_d_auto}}


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
    #   None (default, unset) = resolved by flavour
    #     (resolve_warm_rain_scheme): mg -> "kk2000_cam6", sam -> "kk2000".
    #   "kk2000" = Khairoutdinov-Kogan 2000, the SAM M2005
    #     DEFAULT (IRAIN=0): PRC=1350·qc^2.47·(Nc[#/cm³])^-1.79,
    #     PRA=67·(qc·qr)^1.15. Faithful to the gSAM oracle.
    #   "seifert_beheng" = the same law as "seifert_beheng_sb2001" (below);
    #     it was a simplified proxy (k_au·q_c^2 onset sigmoid + bilinear
    #     k_ac accretion) until the 2026-10 oracle port.  The proxy fields
    #     k_au / x_star / k_ac / autoconversion_sharpness have no gSAM
    #     counterpart and raise if set under it.
    #   "seifert_beheng_sb2001" = Seifert & Beheng (2001) as gSAM M2005 runs
    #     it (IRAIN=1, module_mp_graupel.f90:1830-1844, :1955-1962): the
    #     phi_au / phi_ac universal-function rates with gSAM's gates, the
    #     droplet shape nu diagnosed from N_c (gSAM default dofix_pgam=.false.)
    #     and the droplet-slope clamp on N_c, plus the au/x_* rain-number
    #     source. Oracle-pinned at predict_Nc=False; the SB2001-specific
    #     cloud-NUMBER sinks (predict_Nc=True only) are not represented.
    warm_rain_scheme: str | None = None
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
    # Former Seifert-Beheng PROXY knobs: k_au / x_star / k_ac are read by no
    # law (warm_rain_scheme="seifert_beheng" now runs SB2001 as gSAM does and
    # raises if they are set); Nc_0 is the specified droplet number.
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
    #   "mg2" = CAM6 MG2 Beheng (1994) self-collection, nragg = -8 nric qric
    #     rho on the in-precip rain (micro_mg_utils.F90:1343-1364), times
    #     precip_frac.  morrison_flavor="mg" (the global default) RESOLVES the
    #     "sb2001" default to it in ``resolve_morrison_flavor`` (an explicit
    #     "legacy"/"mg2" is kept); SB2001 itself is selected with
    #     morrison_flavor="sam".
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
    # None = the flavour's (resolve_morrison_flavor): sam "m2005_autoconv",
    # mg "mg_ferrier"; under mg any other explicit value raises (user
    # decision 2026-10-24, 1a: the mg preset never overwrites a choice).
    ice_to_snow_scheme: str | None = None
    # DCS [m]; None = the flavour's (resolve_morrison_flavor at scheme entry):
    # SAM clice_snow_Dauto 250 um, mg = CAM6 MG2 micro_mg_dcs 500 um
    # (namelist_defaults_cam.xml:1244).  An explicit value always wins.
    ice_snow_d_auto: float | None = None
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
    lamr_min: float = 1.0 / 2800.0e-6   # SAM LAMMINR (slope floor) [1/m]; morrison_flavor="mg" resolves 1/500 um (MG2)
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
    # path (mg flavour).  The sam flavour runs gSAM's satadj_liquid (smooth by default, RULE 5) on
    # every call and REJECTS this guard.  Threaded from ExperimentConfig via
    # ``apply_microphysics_experiment_flags``.  See
    # ``_warm_rain.saturation_adjustment``.
    hard_saturation_adjustment: bool = False
    hard_sat_adjust_threshold: float = 1.1      # RH trigger q_v > thr*q_sat [-]
    hard_sat_max_heating_K: float = 5.0         # per-step latent-heating cap [K]

    # --- IFS/SAM homogeneous-freezing ice-supersaturation allowance ---
    # gSAM cloud.f90 (Khairoutdinov 2023, after IFS): pristine air below
    # 235 K may stay ice-supersaturated up to rh_homo = 2.583 - T/207.8
    # (~1.45 at 235 K, ~1.67 at 190 K); the allowance is withdrawn the
    # moment cloud ice exists. ON by default (SAM-faithful); set False to
    # target plain ice saturation.  See thermo.homogeneous_freezing_rh_factor.
    homogeneous_ice_supersaturation: bool = True

    # --- MG2 CFL sub-stepped sedimentation -------------------------------
    # APPENDED AT THE TUPLE END so every positional construction and every
    # pickle written before 2026-09-22 keeps binding the same fields.
    # MG2 (micro_mg2_0.F90 sedimentation loop) CFL sub-stepping of rain /
    # ice / snow / graupel sedimentation: per column nstep = 1 + floor(max
    # V·dt/dz), capped at ``sed_cfl_substeps_max``.  True (DEFAULT, user
    # decision 2026-09-22 -- the one-pass form let a hydrometeor fall at most
    # one layer per call, a defect at every dt where V·dt/dz > 1: production
    # rain CFL ~3.4 at 112.5 s).  False = the legacy one-pass flux-capped
    # form, kept for reproducing pre-2026-09-22 runs.
    sed_cfl_substeps: bool = True
    # Static bound of the fixed-shape sub-step loop (a loop count, not a
    # tunable): iterations beyond a column's own nstep are masked, so the
    # COST is linear in this number -- measured on 2048 columns, CPU x64,
    # whole Morrison call: one pass 10 ms, cap 8 20 ms, 16 24 ms, 32 32 ms,
    # 96 59 ms, 128 74 ms, 256 131 ms (sigma-36 dt 112.5; L32 dt 600 is
    # within 15 %).  Required counts per lane (ISA column, the caps' density
    # factor included): production sigma-36 at 112.5 s needs 8, CAM L32 at
    # 600 s needs 90, at 1800 s 269.  Capped at 1024.  Exceeding the cap
    # clamps the fall (mass conserved, transport wrong); the required count
    # is always reported and ``sed_cfl_substeps_strict`` makes it fatal.
    # 256 stays the value in BOTH arms of CAM6 run 1; per-deck sizing
    # (production 16, CAM6 128) is deferred to the next campaign as its own
    # one-lever change (user 2026-09-22, amip_runs/_cam6/PREREGISTRATION.md).
    sed_cfl_substeps_max: int = 256
    # Fail loudly (runtime error under jit, attached to the sedimentation
    # tendencies so it cannot be eliminated as dead code) when any column
    # needs more sub-steps than the cap; off = the count is only reported
    # (``MicrophysicsOutput.sed_substeps_required``).
    sed_cfl_substeps_strict: bool = False

    # DIAGNOSTIC ONLY.  Publish the APPLIED (post-donor-clamp) cloud-water
    # budget terms on ``MicrophysicsOutput.qc_budget``.  Static Python branch,
    # so the default graph and every existing run are untouched; nothing in the
    # model reads it.  Exists because the donor clamp scales all q_c sinks by a
    # common factor, so terms re-derived outside the scheme are PRE-clamp and
    # cannot close the budget.  NOT a tunable: it selects no physics.
    publish_qc_budget: bool = False
    # CAM6 MG2-style warm-rain multipliers, read ONLY by the kk2000 branch
    # (appended at the END: positional ABI).  autocon_fact scales the KK2000
    # autoconversion rate before the SAM number caps, so the rain-number
    # source and cloud-number sink follow it; accre_enhan_fact scales KK2000
    # accretion (MG2 accre_enhan, micro_mg_utils.F90 accrete_cloud_water_rain;
    # MG2's paired cloud-number sink npra is absent here).  1.0 = unscaled.
    autocon_fact: float = 1.0
    accre_enhan_fact: float = 1.0
    # Relative variance of in-cloud liquid for warm_rain_scheme="kk2000_cam6"
    # (CAM6 ``relvar``).  10.0 = CAM6's ``relvarmax`` when deep convection is
    # not CLUBB (clubb_intr.F90:2416-2421), the value it falls back to where
    # CLUBB's qc variance is unavailable; CAM6 clips to [0.001, relvarmax].
    # Read only when the caller passes no per-cell relvar (the CLUBB
    # liquid-partition lane passes CLUBB's).
    kk2000_cam6_relvar: float = 10.0
    # CAM6 MG2 in-cloud warm rain: autoconversion + accretion on q_c/lcldm,
    # q_r/precip_frac (N_c/lcldm when prognostic), tendencies times lcldm,
    # with lcldm = max(ast, 1e-4), ast = max(CLUBB liquid fraction, CAM6
    # aist ice-stratus fraction) (micro_mg2_0.F90:878, :1224-1322;
    # micro_mg_cam.F90:1809-1810; clubb_intr.F90:2575).  Needs that cloud
    # fraction at the call (``cloud_fraction=``); raises without it.
    # It is the CAM6 MG2 warm-rain mode, so it also selects (oracle audit
    # 2026-10-19): MG2's 'in_cloud' precip_frac for the in-precip rain; rain
    # evaporation only in the precipitating clear air (precip_frac - lcldm,
    # with rain_evap_scheme="m2005"); rain mass+number sedimenting the
    # POST-source state; and MG2's two-sided rain-number re-adjustment.
    # False = rates on the grid-mean state (the pre-existing behaviour).
    warm_rain_incloud: bool = False

    # --- Who supplies the cloud liquid: this scheme, or the turbulence closure
    # APPENDED AT THE TUPLE END (it was inserted mid-tuple on 2026-09-24,
    # re-binding every later positional value; moved here so the pre-insertion
    # prefix binds as before).
    # Default False keeps this scheme's saturation adjustment as the liquid
    # SOURCE, which is what every run before this field did.
    #
    # True says an assumed-PDF closure (CLUBB) already diagnosed the layer's
    # cloud liquid and handed it to the host, so the saturation adjustment here
    # is switched OFF ENTIRELY -- both signs.  This is the CAM6 arrangement, not
    # an invention: micro_mg2_0.F90:2688-2730 carries a residual
    # "remove any excess over-saturation" block, it is gated on
    # ``allow_sed_supersat`` at :2700, and micro_mg_cam.F90:668-672 sets that
    # flag ``.false.`` whenever ``do_clubb_sgs``.  Enumerating every write to
    # ``qctend`` in MG2 (:2634 ice melt, :2680 homogeneous freezing -- a sink,
    # :2718 the gated residual) leaves MG2 with NO vapour-to-liquid
    # condensation at all in a CLUBB configuration.
    #
    # BOTH SIGNS (user decision 2026-10-19, oracle audit tier 1).  An
    # intermediate version kept the negative (cloud-water evaporation) branch
    # as the emergent WBF sink; MG2 under CLUBB has no such term (every qctend
    # write in micro_mg2_0.F90 is a process sink, droplet sedimentation, melt,
    # homogeneous freezing or the CLUBB-disabled residual), and that branch
    # existed only through the smoothing tail (0 at the hard limit), removing
    # up to 29 % of q_c per 600 s step at the production sharpness.  With
    # ``wbf_scheme="emergent"`` this scheme therefore has no in-microphysics
    # WBF liquid sink under this flag (the closure's next liquid diagnosis
    # still sees the vapour that ice deposition removed).  Every other sink (autoconversion, accretion, riming,
    # sedimentation) is untouched.
    #
    # WHY IT MATTERS: the adjustment removes the whole supersaturation on every
    # call, so running microphysics N times inside one physics step adjusts N
    # times.  A 3/6/15 sub-step sweep moved band liquid monotonically by 21%,
    # which is a step-count sensitivity a liquid SOURCE should not have.
    #
    # Turning this on without the closure actually delivering liquid removes the
    # model's only liquid source; ``ExperimentConfig.validate_strict`` refuses
    # that pairing rather than letting it run.
    liquid_from_closure: bool = False
    # morrison_flavor="sam" gSAM satadj_liquid smooth switches (RULE 5; see
    # _warm_rain.saturation_adjustment_gsam).  Hard limit = gSAM exactly:
    # math.inf, math.inf, 0.0.  All three values approved by user 2026-10-21.
    gsam_satadj_excess_sharpness: float = 1.0e4   # saturation step, on (q_t - q_s)/q_sat [-]
    gsam_satadj_qc_floor_sharpness: float = 1.0e4  # positivity, on T1 - T_l [1/K]
    gsam_satadj_p_min_width_pa: float = 100.0     # on/off, p >= 5000 Pa gate width [Pa]
    # RULE 5 smoothing of gSAM's SB2001 autoconversion gate q_c >= 1e-6 kg/kg
    # (module_mp_graupel.f90:1806/2843): the C2 step reaches 1 at this value
    # [kg/kg]; <= 1e-6 = gSAM's hard step (codex+GLM 2026-10-29, #2232; KNOWN
    # DIFFERENCE in _warm_rain.autoconversion_sb2001).
    sb2001_autoconv_ramp_top: float = 2.0e-6
    # MG2's CAM Goff-Gratch q_sat = 1 cap (RULE 5, limiter class): sharpness
    # on e_sat/p; math.inf = CAM's hard cap.  Value approved by user 2026-10-21.
    cam_qsat_cap_sharpness: float = CAM_QSAT_CAP_SHARPNESS
    # RULE 5 smoothing widths of two CAM6 MG2 limits (0 = MG2's hard form;
    # values approved by user 2026-10-21, see the KNOWN DIFFERENCE in _warm_rain.py):
    # LogSumExp width of the ice-slope clamp in ln(lami) (mg2_ice_slope) and
    # of the qi-conservation factor (mg2_qi_conservation_ratio).
    mg2_ice_slope_log_width: float = 0.01
    mg2_qi_limit_width: float = 1.0e-2

    # CAM6 prognostic cloud-droplet number (MG2's nc budget on this scheme's
    # state).  APPENDED AT THE TUPLE END (positional ABI).  Needs
    # warm_rain_incloud=True, warm_rain_scheme="kk2000_cam6", predict_Nc=False,
    # nc_from_aerosol=False and liquid_from_closure=True (start-up errors); the
    # call must pass CLUBB's
    # per-cell ``relvar``.  Under it the warm rain reads the prognostic number
    # (max(N_c, 0), divided by lcldm, no 1 m^-3 floor) and CLUBB's relvar.
    # False = main's behaviour bit for bit on the configurations without a snow
    # number (tests/atmosphere/microphysics/unit/test_morrison_switch_off_golden.py);
    # MG2-snow mode (warm_rain_incloud + snow number) carries the approved
    # smooth 0 C steps and slope rounding with the switch off too (2026-10-21).
    prognostic_droplet_number: bool = False
    # RULE 5 smoothing of the CAM6 MG2 condensate-presence test q >= qsmall
    # (1e-18 kg/kg) in the ported droplet-number and cloud-ice code: the C2 step
    # reaches 1 at this value [kg/kg]; 1e-18 = MG2's hard test (user decision
    # 2026-10-22, 1a; KNOWN DIFFERENCE in _warm_rain.mg2_presence).
    mg2_presence_ramp_top: float = 1.0e-17
    # RULE 5 smoothing of MG2's frozen-rain destination (cloud ice where the
    # mean drop 1/lamr < Dcs, else snow): ramp over this fraction of Dcs below
    # it; 0 = MG2's hard test (user decision 2026-10-22, 15a; KNOWN DIFFERENCE
    # in _warm_rain.mg2_frozen_rain_ice_fraction).
    mg2_frozen_rain_ice_ramp_frac: float = 0.1


# Hard ceiling of ``sed_cfl_substeps_max`` wherever it is set (leaf, applier,
# ExperimentConfig): the loop cost is LINEAR in the cap, so a typo ("2560")
# silently decouples a run's cost from its physics.  The largest configured
# lane needs 269 (CAM L32 at 1800 s).  ONE definition, imported by the other
# validators (GLM 2026-09-22: three independent 1024s can drift apart).
SED_CFL_SUBSTEPS_MAX_LIMIT = 1024


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

    # --- IFS/SAM homogeneous-freezing ice-supersaturation allowance ---
    # gSAM cloud.f90 (Khairoutdinov 2023, after IFS): pristine air below
    # 235 K may stay ice-supersaturated up to rh_homo = 2.583 - T/207.8
    # (~1.45 at 235 K, ~1.67 at 190 K); the allowance is withdrawn the
    # moment cloud ice exists. ON by default (SAM-faithful); set False to
    # target plain ice saturation.  See thermo.homogeneous_freezing_rh_factor.
    homogeneous_ice_supersaturation: bool = True


class P3Config(NamedTuple):
    """Configuration for P3 (Predicted Particle Properties) microphysics.

    ``p3.py`` is a faithful port of WRF v4.8.0 ``module_mp_p3.F`` (P3 v4.5.2)
    for ``mp_physics = 50``: one 2-moment ice category with predicted rime mass
    (q_rim) and rime volume (B_rim), lookup table v5.4_2momI, specified droplet
    number, KK2000 warm rain, Cooper nucleation; oracle-tested to 1e-10 at the
    hard limit.  Every field below is read by the port; the three smoothing
    fields ``ice_sigmoid_sharpness``, ``cooper_supi_sharpness``,
    ``melt_sharpness`` reach the oracle at 1e6 / 1e9 / 1e6.  The opt-in
    ``hard_saturation_adjustment`` guard (not part of WRF P3, default off) is
    applied after the P3 step as for ml_emulator.

    Slot reuse in HydrometeorState
    --------------------------------
    q_s → q_rim  [kg/kg]        rime mass mixing ratio
    q_g → B_rim  [m³/kg_air]    rime volume per unit air mass

    The rime density rho_rim = q_rim / B_rim [kg/m³] spans the full
    range from unrimed aggregates (~50 kg/m³) to dense graupel
    (~900 kg/m³).
    """
    # --- Specified droplet number (WRF P3 nccnst, module_mp_p3.F:196) ---
    Nc_0: float = 2e8               # [1/m³] (user decision 2026-10-21: WRF value)
    # --- Ice nucleation (Cooper 1986, as WRF v4.8.0 P3 runs it) ---
    N_i0: float = 5.0               # Cooper base ice crystal number [1/m³] (= 0.005/L)
    cooper_a: float = 0.304         # Cooper exponent
    # ORACLE cap: WRF P3 100 L⁻¹·SCF with SCF=1 (module_mp_p3.F:3276). Also
    # bounds the Cooper exponential so cold tropopause temperatures cannot
    # overflow fp32 (mirrors Morrison).
    N_i_nuc_max: float = 1.0e5      # [1/m³] = 100 /L
    # Nucleation-specific ORACLE gate (WRF module_mp_p3.F:3272): T < −15 °C AND
    # supi >= 0.05, smoothed by ice_sigmoid_sharpness / cooper_supi_sharpness.
    cooper_T_nuc: float = 258.15    # Nucleation activation temperature [K] (−15 °C)
    cooper_supi_min: float = 0.05   # Min ice supersaturation for nucleation [-]
    ice_sigmoid_sharpness: float = 5.0
    cooper_supi_sharpness: float = 200.0  # Sigmoid sharpness on supi gate [-]
    melt_sharpness: float = 2.0     # Sigmoid sharpness near T_freeze [1/K]
    # --- Rime density limits [kg/m³] ---
    rho_rim_min: float = 50.0       # Minimum rime density (unrimed aggregates)
    rho_rim_max: float = 900.0      # Maximum rime density (dense graupel)
    # --- Hard (iterated) saturation-adjustment guard (opt-in) ---
    # See ``_warm_rain.saturation_adjustment``.  Newton-iterated saturation
    # adjustment that lands q_v ON the liquid saturation curve where
    # q_v > hard_sat_adjust_threshold * q_sat (conserving c_pd*T + L_v*q_v
    # exactly).  Default OFF => byte-identical smooth path.
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
    # --- Hard (iterated) saturation-adjustment guard (opt-in) ---
    # An MLP surrogate emits UNCONSTRAINED tendencies: nothing in the network
    # (or in its ``__physics_contract__``, which claims ``conserves: none``)
    # stops it predicting a dq_v_dt that leaves the column super-saturated.
    # "It is an emulator" is therefore NOT a reason to skip the check -- it is
    # the scheme with the WEAKEST structural guarantee of the ten.
    #
    # Placement differs from the bulk schemes on purpose: the emulator has no
    # internal condensation rate to blend into, so the guard runs on the
    # PREDICTED POST-STEP state ``(T + dT_dt*dt, q_v + dq_v_dt*dt)`` via the
    # shared ``_warm_rain.hard_saturation_drain`` -- the same entry point and
    # the same placement as the validated MPAS driver post-step hook.  Because
    # the drain's mass rate limit is applied against the POST-step q_v, the
    # corrected q_v is >= 0 by construction.  Default OFF => byte-identical
    # (static Python ``if``).
    hard_saturation_adjustment: bool = False
    hard_sat_adjust_threshold: float = 1.1      # RH trigger q_v > thr*q_sat [-]
    hard_sat_max_heating_K: float = 5.0         # per-step latent-heating cap [K]


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
    morrison_scalars: dict | None = None,
    morrison_flavor: str | None = None,
    morrison_sed_cfl_substeps: bool | None = None,
    liquid_from_closure: bool | None = None,
    morrison_sed_cfl_substeps_max: int | None = None,
    morrison_sed_cfl_substeps_strict: bool | None = None,
    morrison_do_graupel: bool | None = None,
    morrison_warm_rain_scheme: str | None = None,
    morrison_warm_rain_incloud: bool | None = None,
    morrison_prognostic_droplet_number: bool | None = None,
):
    """Thread ExperimentConfig-level microphysics switches onto a per-scheme
    sub-config NamedTuple, raising LOUDLY on a scheme that lacks the field.

    The switches are warm-rain closures implemented by the warm-rain-based
    schemes: Morrison's ``effective_Nc`` (specified-Nc aerosol mode) and
    in-cloud autoconversion, and the shared ``saturation_adjustment`` hard
    (iterated) saturation-adjustment guard (kessler/seifert_beheng/morrison/
    thompson/p3).  This single helper is shared by the coupled
    (``physics_pipeline._resolve_microphysics``) and the combined-physics /
    MPAS (``model_driver._run_column``) paths so the gating + fail-loud
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
        if ("hard_saturation_adjustment" not in fields
                or scheme not in HARD_SAT_GUARD_SCHEMES):
            raise ValueError(
                f"hard_saturation_adjustment=True is not supported by the "
                f"{scheme!r} microphysics scheme; use one of "
                f"{HARD_SAT_GUARD_SCHEMES} or drop "
                "--hard-saturation-adjustment. (sdm / fast_sbm resolve "
                "super-saturation explicitly by droplet/bin diffusional "
                "growth and 'none' runs no microphysics at all -- see "
                "HARD_SAT_GUARD_EXEMPT.)"
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
                "microphysics scheme (no hard saturation-adjustment guard); "
                f"use one of {HARD_SAT_GUARD_SCHEMES} or drop the override."
            )
        scheme_config = scheme_config._replace(**{_field: float(_val)})
    if morrison_flavor is not None:
        # Same HARD scheme gate as the scalars below: the flavor selector is
        # a Morrison-only concept; on any other scheme it would be silently
        # inert (P3/Thompson have no morrison_flavor leaf to begin with, but
        # the gate keys on the SCHEME, not field presence, for the same
        # reason as the scalar gate).
        if scheme != "morrison":
            raise ValueError(
                f"morrison_flavor={morrison_flavor!r} is only supported by "
                f"the morrison microphysics scheme (got {scheme!r}); use "
                "--microphysics morrison or drop the override.")
        scheme_config = scheme_config._replace(
            morrison_flavor=morrison_flavor)
    if liquid_from_closure is not None and not hasattr(
            scheme_config, "liquid_from_closure"):
        # Scheme gate, same shape as the Morrison-only knobs below: a deck that
        # asks for this on a scheme with no saturation adjustment to switch off
        # must be told, not silently ignored.
        if liquid_from_closure:
            raise ValueError(
                f"liquid_from_closure=True is not supported by the {scheme!r} "
                "microphysics scheme (it has no saturation adjustment to hand "
                "over); use --microphysics morrison or drop it.")
    elif liquid_from_closure is not None:
        # TRI-STATE, and the None matters.  This knob is slaved to the closure,
        # so a driver passing False must CLEAR a directly-constructed True
        # rather than leave it standing -- otherwise a model builds with no
        # liquid source at all (codex round 1).  But the helper is also called
        # a SECOND time further down this same file to thread the Morrison
        # scalar knobs, and a plain ``False`` default made that second call
        # silently undo the first: setting morrison_flavor or
        # morrison_sed_cfl_substeps_max disabled the gate while the closure
        # stayed on (codex round 2, reproduced).  None means "not this
        # caller's business, leave it".
        scheme_config = scheme_config._replace(
            liquid_from_closure=bool(liquid_from_closure))
    if morrison_sed_cfl_substeps is not None:
        if scheme != "morrison":
            raise ValueError(
                f"morrison_sed_cfl_substeps={morrison_sed_cfl_substeps!r} is "
                "only supported by the morrison microphysics scheme (got "
                f"{scheme!r}); use --microphysics morrison or drop it.")
        if not isinstance(morrison_sed_cfl_substeps, bool):
            raise TypeError(
                "morrison_sed_cfl_substeps must be a bool, got "
                f"{morrison_sed_cfl_substeps!r}")
        scheme_config = scheme_config._replace(
            sed_cfl_substeps=morrison_sed_cfl_substeps)
    if morrison_sed_cfl_substeps_max is not None:
        # scheme gate FIRST: a non-Morrison deck that also mistyped the value
        # should be told which knob does not belong, not its range
        if scheme != "morrison":
            raise ValueError(
                "morrison_sed_cfl_substeps_max is only supported by the "
                f"morrison microphysics scheme (got {scheme!r}).")
        if (not isinstance(morrison_sed_cfl_substeps_max, int)
                or isinstance(morrison_sed_cfl_substeps_max, bool)
                or not 1 <= morrison_sed_cfl_substeps_max
                <= SED_CFL_SUBSTEPS_MAX_LIMIT):
            raise ValueError(
                "morrison_sed_cfl_substeps_max must be an int in "
                f"[1, {SED_CFL_SUBSTEPS_MAX_LIMIT}] (the sub-step loop cost "
                f"is linear in it), got {morrison_sed_cfl_substeps_max!r}")
        scheme_config = scheme_config._replace(
            sed_cfl_substeps_max=morrison_sed_cfl_substeps_max)
    if morrison_sed_cfl_substeps_strict is not None:
        if not isinstance(morrison_sed_cfl_substeps_strict, bool):
            raise TypeError(
                "morrison_sed_cfl_substeps_strict must be a bool, got "
                f"{morrison_sed_cfl_substeps_strict!r}")
        if scheme != "morrison":
            raise ValueError(
                "morrison_sed_cfl_substeps_strict is only supported by the "
                f"morrison microphysics scheme (got {scheme!r}).")
        scheme_config = scheme_config._replace(
            sed_cfl_substeps_strict=morrison_sed_cfl_substeps_strict)
    if morrison_do_graupel is not None:
        if not isinstance(morrison_do_graupel, bool):
            raise TypeError(
                f"morrison_do_graupel must be a bool, got {morrison_do_graupel!r}")
        if scheme != "morrison":
            raise ValueError(
                "morrison_do_graupel is only supported by the morrison "
                f"microphysics scheme (got {scheme!r}).")
        scheme_config = scheme_config._replace(do_graupel=morrison_do_graupel)
    if morrison_warm_rain_incloud is not None:
        if not isinstance(morrison_warm_rain_incloud, bool):
            raise TypeError("morrison_warm_rain_incloud must be a bool, got "
                            f"{morrison_warm_rain_incloud!r}")
        if scheme != "morrison":
            raise ValueError(
                "morrison_warm_rain_incloud is only read by the morrison "
                f"scheme (got {scheme!r}).")
        scheme_config = scheme_config._replace(
            warm_rain_incloud=morrison_warm_rain_incloud)
    if morrison_prognostic_droplet_number is not None:
        if not isinstance(morrison_prognostic_droplet_number, bool):
            raise TypeError("morrison_prognostic_droplet_number must be a bool, got "
                            f"{morrison_prognostic_droplet_number!r}")
        if scheme != "morrison":
            raise ValueError(
                "morrison_prognostic_droplet_number is only read by the morrison "
                f"scheme (got {scheme!r}).")
        scheme_config = scheme_config._replace(
            prognostic_droplet_number=morrison_prognostic_droplet_number)
    if morrison_warm_rain_scheme is not None:
        if scheme != "morrison":
            raise ValueError(
                f"morrison_warm_rain_scheme={morrison_warm_rain_scheme!r} is "
                f"only read by the morrison scheme (got {scheme!r}).")
        scheme_config = scheme_config._replace(
            warm_rain_scheme=morrison_warm_rain_scheme)
    if morrison_scalars:
        # Morrison ice-process tunables (``morrison_*`` ExperimentConfig flat
        # scalars).  HARD scheme gate, NOT field-presence: Thompson carries
        # all five same-named leaves and P3 four, with DIFFERENT defaults
        # (P3 rime_coeff=0.5 vs Morrison 1.0) — a presence-keyed overlay
        # silently retuned those schemes and even changed a fresh default P3
        # run (codex 2026-07-26 morrison-wiring review, Critical).
        import math
        if scheme != "morrison":
            _named = {k: v for k, v in morrison_scalars.items()
                      if v is not None}
            raise ValueError(
                f"morrison_* scalar overrides {sorted(_named)} are only "
                f"supported by the morrison microphysics scheme (the "
                f"{scheme!r} scheme shares some field NAMES but with its own "
                "calibrated defaults); use --microphysics morrison or drop "
                "the overrides."
            )
        _updates = {}
        for _leaf, _val in morrison_scalars.items():
            if _val is None:
                continue
            _cur = getattr(scheme_config, _leaf)
            if _cur is None:        # an unset leaf (DCS): any value applies
                _updates[_leaf] = float(_val)
                continue
            _cur = float(_cur)
            # rel_tol absorbs float32-host round-trip noise
            # (float32(1e-3) != 1e-3 exactly) so a stored config that merely
            # round-tripped through a narrower dtype does not force a
            # pointless _replace (config-identity / retrace churn).
            if not math.isclose(float(_val), _cur, rel_tol=1e-6, abs_tol=0.0):
                _updates[_leaf] = float(_val)
        if _updates:
            scheme_config = scheme_config._replace(**_updates)
    return scheme_config
