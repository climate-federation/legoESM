"""Configuration for the fast-SBM bin microphysics scheme.

Defaults are the WRF ``module_mp_fast_sbm.F`` oracle values (CGS → SI).
Where the oracle's reference coefficient differs from the repo-wide
``legoesm.constants`` value (different reference temperature), the oracle
value is kept here as an explicit scheme parameter so the port stays
faithful without monkey-patching shared constants.
"""

from __future__ import annotations

from typing import NamedTuple


__param_spec__ = {
    "FastSBMConfig": {
        "scheme_key": "atm.fastsbm.FastSBMConfig",
        "excluded": {
            "ice_number_floor": "numerics: solver/smoothing/tolerance/iteration parameter",
            "n_rain_floor": "numerics: solver/smoothing/tolerance/iteration parameter",
        },
        "params": {
            # --- nucleation: Köhler CCN activation (oracle JERNUCL01_KS) ---
            # Total CCN reservoir; primary aerosol knob controlling activated
            # droplet number (the maritime↔continental lever).
            "ccn_number": {"units": "m**-3", "bounds": (1.0e6, 5.0e9), "tunable_tier": 1, "transform": "softplus", "category": "nucleation", "reference": "Khain et al. (2004) JAS 61:2963", "shape": None},
            # Dry-aerosol log-normal mode: median radius + geometric std.
            "aerosol_dry_median": {"units": "m", "bounds": (1.0e-8, 5.0e-7), "tunable_tier": 2, "transform": "sigmoid", "category": "nucleation", "reference": "Khain & Pokrovsky (2002) JAS 59:2839", "shape": None},
            "aerosol_geom_std": {"units": "1", "bounds": (1.05, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "nucleation", "reference": "Khain & Pokrovsky (2002) JAS 59:2839", "shape": None},
            # Köhler solute (ammonium-sulfate) hygroscopicity inputs: van't
            # Hoff ions, molar mass, solute density (rarely retuned).
            "aerosol_ions": {"units": "1", "bounds": (1.0, 5.0), "tunable_tier": 3, "transform": "sigmoid", "category": "nucleation", "reference": "Köhler (1936); Pruppacher & Klett (1997)", "shape": None},
            "aerosol_molar_mass": {"units": "kg mol**-1", "bounds": (0.018, 0.4), "tunable_tier": 3, "transform": "sigmoid", "category": "nucleation", "reference": "Pruppacher & Klett (1997)", "shape": None},
            "aerosol_solute_density": {"units": "kg m**-3", "bounds": (1000.0, 5000.0), "tunable_tier": 3, "transform": "sigmoid", "category": "nucleation", "reference": "Pruppacher & Klett (1997) (ammonium sulfate 1770)", "shape": None},
            # --- freezing: Bigg (1953) immersion (oracle FREEZ) ---
            # Pre-factor (CGS, drop mass in grams) and slope coefficients.
            "bigg_a": {"units": "g**-1 s**-1", "bounds": (1.0e-6, 1.0e-3), "tunable_tier": 2, "transform": "sigmoid", "category": "freezing", "reference": "Bigg (1953) QJRMS 79:510; Khain et al. (2004)", "shape": None},
            "bigg_b0": {"units": "K**-1", "bounds": (0.3, 1.2), "tunable_tier": 2, "transform": "sigmoid", "category": "freezing", "reference": "Bigg (1953) QJRMS 79:510", "shape": None},
            "bigg_b_max": {"units": "K**-1", "bounds": (0.3, 1.2), "tunable_tier": 3, "transform": "sigmoid", "category": "freezing", "reference": "Bigg (1953) QJRMS 79:510 (mass-dependent slope cap)", "shape": None},
            # --- melting: Jiwen-Fan constant-timescale (oracle J_W_MELT) ---
            "melt_rate_mid": {"units": "s**-1", "bounds": (1.0e-3, 5.0e-2), "tunable_tier": 2, "transform": "softplus", "category": "melting", "reference": "Khain et al. (2004) JAS 61:2963 (J_W_MELT)", "shape": None},
            "melt_rate_high": {"units": "s**-1", "bounds": (1.0e-3, 5.0e-2), "tunable_tier": 2, "transform": "sigmoid", "category": "melting", "reference": "Khain et al. (2004) JAS 61:2963 (J_W_MELT)", "shape": None},
            # --- collision_coalescence ---
            # Golovin analytic test-kernel coefficient K = b(m_i+m_j).
            "golovin_b": {"units": "s**-1", "bounds": (1.0e2, 1.0e4), "tunable_tier": 2, "transform": "softplus", "category": "collision_coalescence", "reference": "Golovin (1963); Bott (1998) JAS 55:2284", "shape": None},
            # Ice self-collection (→ snow) sticking efficiency scaling the
            # Bott kernel; primary ice-aggregation knob (~0.1).
            "ice_aggregation_efficiency": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "collision_coalescence", "reference": "Khain et al. (2004) JAS 61:2963 (ice sticking efficiency)", "shape": None},
            # --- condensation: vapor diffusion + ventilation (oracle JERRATE_KS) ---
            "d_vapor_ref_m2s": {"units": "m**2 s**-1", "bounds": (1.0e-5, 4.0e-5), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Pruppacher & Klett (1997); oracle D_MYIN", "shape": None},
            "nu_air_ref_m2s": {"units": "m**2 s**-1", "bounds": (5.0e-6, 3.0e-5), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Pruppacher & Klett (1997); oracle COEFF_VISCOUS", "shape": None},
            "diffusivity_T_exponent": {"units": "1", "bounds": (1.0, 3.0), "tunable_tier": 3, "transform": "sigmoid", "category": "condensation", "reference": "Hall & Pruppacher (1976); oracle D_MY exponent", "shape": None},
            "ventilation_max": {"units": "1", "bounds": (2.0, 12.0), "tunable_tier": 2, "transform": "sigmoid", "category": "condensation", "reference": "Pruppacher & Klett (1997) ch. 13; oracle VENTPL_MAX", "shape": None},
            # --- sedimentation: ice terminal velocity V = a·D^b·(ρ0/ρ)^½ ---
            # Power-law prefactor a [SI: D in m, V in m/s ⇒ m^(1-b)·s^-1] and
            # exponent b per category, plus category bulk density.
            "fall_a_snow": {"units": "m**(1-b) s**-1", "bounds": (1.0, 100.0), "tunable_tier": 2, "transform": "softplus", "category": "sedimentation", "reference": "Locatelli & Hobbs (1974) JGR 79:2185 (unrimed aggregates)", "shape": None},
            "fall_b_snow": {"units": "1", "bounds": (0.2, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "sedimentation", "reference": "Locatelli & Hobbs (1974) JGR 79:2185 (unrimed aggregates)", "shape": None},
            "rho_snow": {"units": "kg m**-3", "bounds": (20.0, 300.0), "tunable_tier": 2, "transform": "softplus", "category": "sedimentation", "reference": "Pruppacher & Klett (1997) (snow bulk density)", "shape": None},
            "fall_a_graupel": {"units": "m**(1-b) s**-1", "bounds": (10.0, 500.0), "tunable_tier": 2, "transform": "softplus", "category": "sedimentation", "reference": "Locatelli & Hobbs (1974) JGR 79:2185 (lump graupel)", "shape": None},
            "fall_b_graupel": {"units": "1", "bounds": (0.3, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "sedimentation", "reference": "Locatelli & Hobbs (1974) JGR 79:2185 (lump graupel)", "shape": None},
            "rho_graupel": {"units": "kg m**-3", "bounds": (100.0, 900.0), "tunable_tier": 2, "transform": "softplus", "category": "sedimentation", "reference": "Pruppacher & Klett (1997) (graupel/hail bulk density)", "shape": None},
            # Foote-du Toit density-correction reference (level where the
            # (ρ_ref/ρ)^½ correction is unity); rarely retuned.
            "fall_rho_ref": {"units": "kg m**-3", "bounds": (0.8, 1.4), "tunable_tier": 3, "transform": "sigmoid", "category": "sedimentation", "reference": "Foote & du Toit (1969) J. Appl. Meteor. 8:249", "shape": None},
            # --- size_distribution: column-adapter spectrum reconstruction ---
            # Prescribed cloud-droplet number when the dycore carries none;
            # primary single-moment closure knob (maritime↔continental).
            "cdnc": {"units": "m**-3", "bounds": (1.0e6, 5.0e9), "tunable_tier": 1, "transform": "softplus", "category": "size_distribution", "reference": "Seifert & Beheng (2006) (maritime N_c default)", "shape": None},
            # Geometric std of the reconstructed cloud-droplet lognormal.
            "cloud_geom_std": {"units": "1", "bounds": (1.05, 2.5), "tunable_tier": 2, "transform": "sigmoid", "category": "size_distribution", "reference": "FastSBM scheme default (reconstruction)", "shape": None},
        },
    },
}


class FastSBMConfig(NamedTuple):
    """Tunable parameters of the fast-SBM scheme (oracle defaults).

    Attributes:
        d_vapor_ref_m2s: water-vapor diffusivity at the reference state
            [m²/s] (oracle ``D_MYIN = 0.211`` cm²/s; ``legoesm.constants.
            D_vapor = 2.21e-5`` uses a different reference temperature).
        nu_air_ref_m2s: kinematic viscosity of air used in the ventilation
            Reynolds number [m²/s] (oracle ``COEFF_VISCOUS = 0.13`` cm²/s).
        diffusivity_T_exponent: temperature exponent of the diffusivity law
            ``D = D_ref (p₀/p)(T/T₀)^a`` (oracle ``1.94``).
        ventilation_max: cap on the ventilation factor (oracle
            ``VENTPL_MAX = 5.0``).
    """

    d_vapor_ref_m2s: float = 2.11e-5
    nu_air_ref_m2s: float = 1.3e-5
    diffusivity_T_exponent: float = 1.94
    ventilation_max: float = 5.0
    # --- column-adapter reconstruction (stateless interface; see
    # column.py docstring) ---
    # Prescribed cloud-droplet number for spectrum reconstruction [1/m^3]
    # (same closure role and default as SDMConfig.cdnc / Seifert-Beheng
    # maritime Nc_0).
    cdnc: float = 1.0e8
    # Geometric std of the reconstructed cloud-droplet lognormal [-].
    cloud_geom_std: float = 1.4
    # Rain-drop number floor for reconstruction when the dycore carries no
    # N_r [1/m^3].
    n_rain_floor: float = 1.0e3
    # Collision kernel for the in-step Bott coalescence: "hall", "long",
    # or "golovin" (analytic test kernel, uses golovin_b).
    collision_kernel: str = "hall"
    golovin_b: float = 1.5e3
    # Static per-step count for per-bin sedimentation (oracle FALFLUXHUCM_Z
    # adapts NSUB at runtime; a traced trip count is not reverse-mode
    # differentiable, so the port fixes it).
    n_fall_substeps: int = 4
    # --- CCN activation (Köhler; oracle JERNUCL01_KS aerosol mode) ---
    # Total CCN number concentration available to activate [1/m^3].
    ccn_number: float = 1.0e8
    # Log-normal dry-aerosol mode: median radius [m] and geometric std [-]
    # (continental accumulation mode defaults).
    aerosol_dry_median: float = 5.0e-8
    aerosol_geom_std: float = 2.0
    # Solute van't Hoff ions, molar mass [kg/mol], density [kg/m^3]
    # (ammonium sulfate defaults: 3 ions, 0.132 kg/mol, 1770 kg/m^3).
    aerosol_ions: float = 3.0
    aerosol_molar_mass: float = 0.132
    aerosol_solute_density: float = 1770.0
    # --- Bigg (1953) immersion freezing (oracle FREEZ) ---
    # Pre-factor A [1/(g·s)] and slope B0, B_max [1/K] (oracle CGS DATA:
    # AFREEZMY=0.3333e-4, BFREEZMY=B_max=0.66 → constant slope by default).
    bigg_a: float = 0.3333e-4
    bigg_b0: float = 0.66
    bigg_b_max: float = 0.66
    # Habit-routing bin threshold (oracle FREEZ KRFREEZ=21, 1-based): frozen
    # drops in 0-based bins < krfreeze become pristine ice crystals, the
    # larger bins (frozen rain) become hail/graupel.
    krfreeze: int = 21
    # --- Jiwen Fan melting (oracle J_W_MELT) ---
    # Bin thresholds (0-based) + rates [1/s]: bins ≤ full melt completely,
    # ≤ mid melt at rate_mid, above at rate_high (oracle snow ladder:
    # KR≤14 full, 15–21 @ 0.5/50, ≥22 @ 0.683/120 in 1-based).
    melt_full_bin: int = 13
    melt_mid_bin: int = 20
    melt_rate_mid: float = 0.5 / 50.0
    melt_rate_high: float = 0.683 / 120.0
    # Ice-spectrum reconstruction number floor [1/m^3] (carried q_i → ice
    # spectrum for melting; single-moment ice closure).
    ice_number_floor: float = 1.0e4
    # Ice-ice aggregation (self-collection → snow) collection efficiency
    # [-], scaling the Bott kernel. ~0.1 reflects the lower sticking
    # efficiency of ice crystals vs liquid coalescence.
    ice_aggregation_efficiency: float = 0.1
    # --- Ice terminal-velocity power laws V = a·D^b (Locatelli & Hobbs
    # 1974; SI: D [m], V [m/s]) + category bulk density [kg/m^3] ---
    # Unrimed aggregates / pristine crystals (low density, slow).
    fall_a_snow: float = 11.72
    fall_b_snow: float = 0.41
    rho_snow: float = 100.0
    # Lump graupel / hail (dense, fast).
    fall_a_graupel: float = 124.0
    fall_b_graupel: float = 0.66
    rho_graupel: float = 400.0
    # Foote-du Toit (ρ_ref/ρ_air)^½ density-correction reference [kg/m^3]
    # (standard WRF near-surface value; the level where the correction = 1).
    fall_rho_ref: float = 1.2
