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
            "Q10_het_exp": {"units": "1", "bounds": (0.023, 0.14), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "soil-respiration Q10 2-3 (Bond-Lamberty & Thomson 2010)", "shape": None},
            "T_opt_C": {"units": "degC", "bounds": (8.25, 75.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None, "legacy_name": "T_opt"},
            "T_width_C": {"units": "degC", "bounds": (4.95, 45.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None, "legacy_name": "T_width"},
            "clab_release_period": {"units": "1", "bounds": (16.5, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "cwd_humification_eff": {"units": "1", "bounds": (0.05, 0.6), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Harmon et al. 1986 (CWD humification)", "shape": None},
            "decomp_rate": {"units": "1", "bounds": (0.000165, 0.0015), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "epsilon": {"units": "1", "bounds": (0.3, 2.0), "tunable_tier": 1, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "f_auto": {"units": "1", "bounds": (0.0924, 0.84), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "f_fol": {"units": "1", "bounds": (0.0495, 0.45), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "f_lab": {"units": "1", "bounds": (0.033, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "f_root": {"units": "1", "bounds": (0.0825, 0.75), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "k_ext": {"units": "1", "bounds": (0.165, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "lab_lifespan": {"units": "1", "bounds": (1.05, 4.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990; lo>1 required (log(L-1) phenology undefined at L<=1)", "shape": None},
            "leaf_fall_period": {"units": "1", "bounds": (16.5, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "leaf_lifespan": {"units": "1", "bounds": (1.05, 4.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990; lo>1 required (log(L-1) phenology undefined at L<=1)", "shape": None},
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
            "tor_som": {"units": "1", "bounds": (1.65e-06, 8e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "bulk-soil MRT decades-century (Jobbagy & Jackson 2000; He et al. 2016)", "shape": None},
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
    # remainder -> wood (woody PFTs) or roots (herbaceous PFTs)
    woody: bool = True            # False for herbaceous PFTs (grass/crop):
    # no wood pool — the structural fraction that would form wood is invested
    # belowground (roots) instead, so a grassland does not grow a phantom tree.

    # --- Turnover rates [day^-1] ---
    tor_wood: float = 1e-4
    tor_root: float = 1e-3
    tor_litter: float = 2e-3
    tor_som: float = 4e-5         # Bulk SOM turnover [day^-1] (~68 yr at
    # T_ref).  Raised from the old 5e-6 (~550 yr): a single effective soil-C
    # pool at millennial turnover, driven by realistic litter+CWD inputs from
    # a productive canopy, equilibrated to an unphysically large stock
    # (tropical SOM ~46 kgC/m2 at true equilibrium vs observed ~10-15).  A
    # ~68 yr reference residence time -> ~17 yr in warm tropical soil via the
    # Q10 below, matching the fast bulk (active+slow) turnover of warm-wet
    # topsoil; the passive millennial fraction is not resolved by a single
    # pool (see the multi-pool follow-up in docs/land/carbon_equilibrium_audit).
    decomp_rate: float = 5e-4     # Litter -> SOM transfer [day^-1]
    # Coarse-woody-debris humification efficiency: the fraction of wood
    # turnover that becomes stable SOM.  The remainder respires to the
    # atmosphere (CWD heterotrophic respiration).  Without this split wood
    # turnover humified 100 % into the millennial SOM pool, giving an
    # unphysically large soil-carbon stock; ~0.3 matches the litter pathway's
    # ~0.2 SOM yield and typical CWD humification (Harmon et al. 1986).
    cwd_humification_eff: float = 0.3

    # --- Decomposition sensitivity ---
    Q10_exp: float = 0.04         # AUTOTROPHIC maint. resp.: exp(Q10_exp*(T-T_ref))
    # Heterotrophic (soil) decomposition temperature sensitivity, DECOUPLED
    # from the autotrophic Q10 above.  0.09 -> Q10 ~= 2.5, the upper-realistic
    # soil-respiration range (Q10 2-3).  A strong soil Q10 makes WARM tropical
    # soils turn SOM over fast (low equilibrium SOM despite high productivity)
    # while COLD high-latitude soils retain carbon (high SOM) — the observed
    # SOC gradient the old weak, shared Q10=0.04 (Q10~1.5) could not produce.
    Q10_het_exp: float = 0.09
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


class CarbonDiagnostics(NamedTuple):
    """Instantaneous carbon-flux breakdown for one DifferLand step.

    Optional output of :func:`step_carbon_differland` (``return_diagnostics=
    True``).  Purely diagnostic — does NOT feed the state update.  Every flux
    is a per-day rate [gC/m2/day] (the internal working units of the
    DifferLand step); ``lai`` is [m2/m2].  Sign convention for ``nee``:
    positive = source to the atmosphere (same as the returned ``co2_flux``).

    The breakdown satisfies, by construction (see ``step_carbon_differland``):
        nee == r_auto - unmet_npp_deficit + r_het - gpp
        r_het == r_het_lit + r_het_som + r_het_cwd
        a_fol + a_lab + a_root + a_wood == max(npp, 0)   (allocation closes)
        npp == gpp - r_auto
    """
    gpp: jax.Array          # Gross primary production
    npp: jax.Array          # Net primary production (gpp - r_auto; may be <0)
    r_maint: jax.Array      # Maintenance (autotrophic) respiration
    r_growth: jax.Array     # Growth (autotrophic) respiration
    r_auto: jax.Array       # Total autotrophic respiration
    r_het_lit: jax.Array    # Heterotrophic respiration from litter
    r_het_som: jax.Array    # Heterotrophic respiration from SOM
    r_het_cwd: jax.Array    # Heterotrophic respiration from coarse woody debris
    r_het: jax.Array        # Total heterotrophic respiration
    nee: jax.Array          # Net ecosystem exchange (positive up)
    unmet_npp_deficit: jax.Array  # Respiration demand no pool could supply
    a_fol: jax.Array        # NPP allocated to foliage
    a_lab: jax.Array        # NPP allocated to labile
    a_root: jax.Array       # NPP allocated to roots
    a_wood: jax.Array       # NPP allocated to wood
    lab_release: jax.Array  # Labile -> foliage (phenology)
    leaf_litter: jax.Array  # Foliage -> litter (phenology)
    root_litter: jax.Array  # Root -> litter (turnover)
    wood_litter: jax.Array  # Wood turnover (total leaving the wood pool)
    wood_to_som: jax.Array  # Humified fraction of wood turnover -> SOM
    lit_to_som: jax.Array   # Litter -> SOM (decomposition)
    lai: jax.Array          # Leaf area index (C_fol / LCMA)
