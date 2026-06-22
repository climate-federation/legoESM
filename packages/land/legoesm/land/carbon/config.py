"""Land carbon cycle configuration and state containers."""

from __future__ import annotations

from typing import NamedTuple

import jax


__param_spec__ = {
    "CarbonConfig": {
        "scheme_key": "land.carbon",
        "excluded": {
            "C_fol_init": "initial condition: carbon pool [gC/m2]",
            "C_lab_init": "initial condition: carbon pool [gC/m2]",
            "C_lit_init": "initial condition: carbon pool [gC/m2]",
            "C_root_init": "initial condition: carbon pool [gC/m2]",
            "C_som_init": "initial condition: carbon pool [gC/m2]",
            "C_wood_init": "initial condition: carbon pool [gC/m2]",
            "T_ref": "reference temperature [K]",
        },
        "params": {
            "Bday": {"units": "1", "bounds": (33.0, 300.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "Fday": {"units": "1", "bounds": (92.4, 840.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "K_CO2": {"units": "1", "bounds": (132.0, 1200.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "LCMA": {"units": "1", "bounds": (16.5, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "Q10_exp": {"units": "1", "bounds": (0.0132, 0.12), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "T_opt_C": {"units": "degC", "bounds": (8.25, 75.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None, "legacy_name": "T_opt"},
            "T_width_C": {"units": "degC", "bounds": (4.95, 45.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None, "legacy_name": "T_width"},
            "clab_release_period": {"units": "1", "bounds": (16.5, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "decomp_rate": {"units": "1", "bounds": (0.000165, 0.0015), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "epsilon": {"units": "1", "bounds": (0.3, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "f_auto": {"units": "1", "bounds": (0.0924, 0.84), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "f_fol": {"units": "1", "bounds": (0.0495, 0.45), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "f_lab": {"units": "1", "bounds": (0.033, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "f_root": {"units": "1", "bounds": (0.0825, 0.75), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "k_ext": {"units": "1", "bounds": (0.165, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "lab_lifespan": {"units": "1", "bounds": (0.495, 4.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "leaf_fall_period": {"units": "1", "bounds": (16.5, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "leaf_lifespan": {"units": "1", "bounds": (0.495, 4.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "moist_modifier_max": {"units": "1", "bounds": (0.99, 9.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "moist_modifier_min": {"units": "1", "bounds": (0.033, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "moisture_factor": {"units": "1", "bounds": (0.165, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "nee_amplitude": {"units": "1", "bounds": (1.65e-08, 1.5e-07), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "nee_peak_day": {"units": "1", "bounds": (66.0, 600.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "precip_ref": {"units": "1", "bounds": (9.9e-06, 9e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "r_maint_fol": {"units": "1", "bounds": (0.00165, 0.015), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "r_maint_root": {"units": "1", "bounds": (0.00066, 0.006), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "r_maint_wood": {"units": "1", "bounds": (1.65e-05, 0.00015), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "tor_litter": {"units": "1", "bounds": (0.00066, 0.006), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "tor_root": {"units": "1", "bounds": (0.00033, 0.003), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "tor_som": {"units": "1", "bounds": (1.65e-06, 1.5e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "tor_wood": {"units": "1", "bounds": (3.3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
        },
    },
}


class CarbonConfig(NamedTuple):
    """Carbon cycle configuration.

    scheme="none"       : disabled (default, backward compatible)
    scheme="differland" : DALEC-based prognostic 6-pool model (DifferLand v1.0)
    scheme="seasonal"   : prescribed repeating seasonal NEE cycle
    """
    scheme: str = "none"

    # --- GPP: Light Use Efficiency model ---
    epsilon: float = 1.0          # Light use efficiency [gC / MJ_PAR]
    k_ext: float = 0.5            # Beer's law extinction coefficient
    T_opt_C: float = 25.0         # Optimal temperature for photosynthesis [deg C]
    T_width_C: float = 15.0       # Temperature response width [deg C]
    K_CO2: float = 400.0          # CO2 half-saturation constant [ppmv]

    # --- Autotrophic respiration ---
    f_auto: float = 0.28          # Growth respiration fraction of net assimilation
    r_maint_fol: float = 0.005    # Foliage maintenance respiration rate [day^-1]
    r_maint_root: float = 0.002   # Root maintenance respiration rate [day^-1]
    r_maint_wood: float = 5e-5    # Wood maintenance respiration rate [day^-1]

    # --- NPP allocation (sequential partition) ---
    f_fol: float = 0.15           # Fraction NPP -> foliage
    f_lab: float = 0.10           # Fraction (NPP-fol) -> labile
    f_root: float = 0.25          # Fraction (NPP-fol-lab) -> roots
    # remainder -> wood

    # --- Turnover rates [day^-1] ---
    tor_wood: float = 1e-4
    tor_root: float = 1e-3
    tor_litter: float = 2e-3
    tor_som: float = 5e-6
    decomp_rate: float = 5e-4     # Litter -> SOM transfer [day^-1]

    # --- Decomposition sensitivity ---
    Q10_exp: float = 0.04         # exp(Q10_exp * (T - T_ref))
    moisture_factor: float = 0.5  # Moisture scaling strength
    T_ref: float = 283.15         # Reference temperature [K]
    precip_ref: float = 3e-5      # Reference precipitation rate [kg/m2/s]
    moist_modifier_min: float = 0.1  # Lower clip on moisture modifier
    moist_modifier_max: float = 3.0  # Upper clip on moisture modifier

    # --- Leaf properties ---
    LCMA: float = 50.0            # Leaf carbon mass per area [gC/m2]

    # --- Phenology (DALEC990 Gaussian forcing) ---
    Bday: float = 100.0           # Bud-burst day of year (NH)
    Fday: float = 280.0           # Leaf fall day of year (NH)
    leaf_lifespan: float = 1.5    # [years]
    lab_lifespan: float = 1.5     # [years]
    clab_release_period: float = 50.0   # Labile release width [days]
    leaf_fall_period: float = 50.0      # Leaf fall width [days]
    hemisphere_aware: bool = True       # Flip phenology for SH

    # --- Initial pool sizes [gC/m2] ---
    C_lab_init: float = 100.0
    C_fol_init: float = 200.0
    C_root_init: float = 300.0
    C_wood_init: float = 10000.0
    C_lit_init: float = 100.0
    C_som_init: float = 10000.0

    # --- Seasonal cycle (scheme="seasonal") ---
    nee_amplitude: float = 5e-8   # Peak NEE amplitude [kgCO2/m2/s]
    nee_peak_day: float = 200.0   # Day of peak uptake (NH)


class CarbonState(NamedTuple):
    """Prognostic carbon pool state.

    All fields share the land model's spatial shape (e.g. (6,n,n)
    for cubed-sphere slab land or (ncol,) for columnar).
    Units: gC/m2.
    """
    C_lab: jax.Array     # Labile carbon
    C_fol: jax.Array     # Foliar carbon
    C_root: jax.Array    # Root carbon
    C_wood: jax.Array    # Wood carbon
    C_lit: jax.Array     # Litter (dead foliage + roots)
    C_som: jax.Array     # Soil organic matter
