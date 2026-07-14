"""Land carbon cycle configuration and state containers."""

from __future__ import annotations

from typing import NamedTuple

import jax

from legoesm import constants


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
            "T_ref": "heterotrophic (soil-decomposition) reference temperature [K]",
            "T_ref_ra": "autotrophic maintenance-respiration reference temperature [K]; fixed normalization confounded with the bulk r_maint_* amplitudes (25 degC; not trained)",
            "som_freeze_width_K": "numerics: SOM freeze-suppression curve half-width [K]",
            "dormancy_transition_width_K": "numerics: cold-deciduous dormancy sigmoid half-width [K]",
            "permafrost_frozen_fraction_width": "numerics: perennial-frost protection sigmoid half-width in frozen-fraction space [-]",
        },
        "params": {
            "Bday": {"units": "1", "bounds": (33.0, 300.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "Fday": {"units": "1", "bounds": (92.4, 840.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "K_CO2": {"units": "1", "bounds": (132.0, 1200.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "LCMA": {"units": "1", "bounds": (16.5, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "Q10_exp": {"units": "K^-1", "bounds": (0.0132, 0.12), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
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
            "lab_lifespan": {"units": "1", "bounds": (0.495, 4.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "leaf_fall_period": {"units": "1", "bounds": (16.5, 150.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "leaf_lifespan": {"units": "1", "bounds": (0.495, 4.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "moist_modifier_max": {"units": "1", "bounds": (0.99, 9.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "moist_modifier_min": {"units": "1", "bounds": (0.033, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "moisture_factor": {"units": "1", "bounds": (0.165, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "nee_amplitude": {"units": "1", "bounds": (1.65e-08, 1.5e-07), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "nee_peak_day": {"units": "1", "bounds": (66.0, 600.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "precip_ref": {"units": "1", "bounds": (9.9e-06, 9e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "r_maint_fol": {"units": "day^-1", "bounds": (0.00066, 0.006), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "recalibrated to observed cross-biome CUE ~0.45 (He et al. 2018; Collalti & Prentice 2019); DifferLand bulk maintenance closure", "shape": None},
            "r_maint_root": {"units": "day^-1", "bounds": (0.000264, 0.0024), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "recalibrated to observed cross-biome CUE ~0.45 (He et al. 2018; Collalti & Prentice 2019); DifferLand bulk maintenance closure", "shape": None},
            "r_maint_wood": {"units": "day^-1", "bounds": (6.6e-06, 6e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "recalibrated to observed cross-biome CUE ~0.45 (He et al. 2018; Collalti & Prentice 2019); DifferLand bulk maintenance closure", "shape": None},
            "nsc_ref_labile_frac": {"units": "1", "bounds": (0.005, 0.1), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "labile/NSC as a fraction of live biomass at which maintenance respiration is unthrottled; respiratory downregulation under substrate limitation (Atkin & Tjoelker 2003)", "shape": None},
            "r_maint_floor_frac": {"units": "1", "bounds": (0.0, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "basal maintenance-respiration floor retained under full NSC depletion (Atkin & Tjoelker 2003)", "shape": None},
            "freeze_dormancy_threshold_K": {"units": "K", "bounds": (263.0, 278.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "cold-deciduous winter-dormancy onset temperature (~0 degC); larch/tundra phenology", "shape": None},
            "leaf_bootstrap_lai": {"units": "1", "bounds": (0.1, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "minimum leaf area a cold-deciduous plant regrows from labile in the growing season to escape the 0-leaf GPP lock (larch/tundra leaf-out)", "shape": None},
            "leaf_bootstrap_frac": {"units": "1", "bounds": (0.0, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "max fraction of the labile reserve drawn per step to bootstrap the leaf area (rate-limits the C_lab->C_fol regrowth)", "shape": None},
            "tor_litter": {"units": "1", "bounds": (0.00066, 0.006), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "tor_root": {"units": "1", "bounds": (0.00033, 0.003), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
            "tor_som_active": {"units": "1/day", "bounds": (3e-04, 3e-03), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CENTURY/CLM4.5 active-SOM MRT ~1-5 yr (Parton et al. 1987; Koven et al. 2013)", "shape": None},
            "tor_som_slow": {"units": "1/day", "bounds": (4e-05, 2e-04), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CENTURY/CLM4.5 slow-SOM MRT ~20-50 yr (Parton et al. 1987; Koven et al. 2013)", "shape": None},
            "tor_som_passive": {"units": "1/day", "bounds": (1.5e-06, 1e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CENTURY/CLM4.5 passive-SOM MRT ~500-1000 yr (Parton et al. 1987; Koven et al. 2013)", "shape": None},
            "f_active_to_slow": {"units": "1", "bounds": (0.1, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CENTURY inter-pool humification fraction ~0.1-0.5 (Parton et al. 1987)", "shape": None},
            "f_slow_to_passive": {"units": "1", "bounds": (0.1, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CENTURY inter-pool humification fraction ~0.1-0.5 (Parton et al. 1987)", "shape": None},
            "som_freeze_floor": {"units": "1", "bounds": (0.0, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "CLM4.5/CENTURY nonzero cold-soil decomposition floor (microbial activity in unfrozen liquid films + cryoturbation; Koven et al. 2013, Parton et al. 1987)", "shape": None},
            "permafrost_protection_min": {"units": "1", "bounds": (0.1, 0.6), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "perennial-frost/anaerobic SOM protection floor: anaerobic C mineralisation ~2-10x slower than aerobic + cryoturbation burial (Schadel et al. 2016 Nature Clim. Change; Koven et al. 2013; Hugelius et al. 2014)", "shape": None},
            "permafrost_frozen_fraction_threshold": {"units": "1", "bounds": (0.4, 0.75), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "seasonal->perennial-frost (permafrost) climate boundary annual frozen fraction ~0.55-0.7 at MAAT -2 to -8 degC (Gruber 2012 permafrost zonation index; Brown et al. 1997 IPA map)", "shape": None},
            "tor_wood": {"units": "1", "bounds": (3.3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "DifferLand/DALEC990", "shape": None},
        },
    },
    "D13CConfig": {
        "scheme_key": "land.d13c",
        "excluded": {
            "ci_ca_c4": (
                "C4-characteristic intercellular:ambient CO2 setpoint [-]; FIXED (not "
                "trained). The model's Farquhar is C3-only, so the C4 branch regulates to a "
                "prescribed C4 Ci/Ca (~0.4) rather than a solved leaf state; in the C4 form "
                "the trained part enters ONLY through the product (b4 + (b3-s)*phi - a)*Ci/Ca, "
                "so freeing BOTH phi and Ci/Ca is a non-identifiable degeneracy -- phi is the "
                "sole exposed C4 lever."
            ),
        },
        "params": {
            "phi_c4_leakiness": {"units": "1", "bounds": (0.1, 0.4), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Henderson, von Caemmerer & Farquhar (1992), C4 bundle-sheath leakiness phi ~ 0.2-0.3", "shape": None},
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
    # BULK EFFECTIVE maintenance-respiration coefficients [day^-1] referenced to the
    # ``T_ref_ra`` normalization, RECALIBRATED (not tissue-measured) so the coupled
    # equilibrium carbon-use efficiency CUE = NPP/GPP TARGETS the observed
    # cross-biome ~0.40-0.50 (He et al. 2018; Collalti & Prentice 2019) instead of
    # the former unphysical ~0.11.  In the SimpleSEB / archetype path GPP is GROSS
    # (leaf dark respiration NOT subtracted), so ``r_maint_fol`` represents the
    # WHOLE foliar respiration.  The reference (T_ref_ra) and these magnitudes are
    # NON-IDENTIFIABLE together, so the reference is FIXED (excluded) and these
    # magnitudes carry the calibration (tunable, tier 2).  The prior values (0.005
    # / 0.002 / 5e-5) left maintenance ~76% of GPP even in healthy forests, driving
    # CUE to ~0.11 (with dozens of archetypes at negative NPP).
    f_auto: float = 0.28          # Growth respiration fraction of net assimilation
    r_maint_fol: float = 0.002    # Foliage maintenance respiration rate [day^-1]
    r_maint_root: float = 0.0008  # Root maintenance respiration rate [day^-1]
    r_maint_wood: float = 2e-5    # Wood maintenance respiration rate [day^-1]

    # --- High-latitude productivity rescue (opt-in, default-off, PFT-scoped) ---
    # Two mechanisms that fix the boreal/tundra carbon death spiral (annual GPP <
    # annual R_maint -> biomass -> 0).  Both default OFF -> byte-identical.
    # Mechanism 1 -- NSC/substrate-gated maintenance respiration (Atkin &
    # Tjoelker 2003): as the labile reserve C_lab depletes relative to LIVE
    # biomass, R_maint throttles toward a floor instead of cannibalising
    # structural pools to death (carbon_cycle._nsc_respiration_factor).
    nsc_gated_respiration: bool = False
    nsc_ref_labile_frac: float = 0.02   # C_lab_ref = this*(C_fol+C_root+C_wood) [-]
    r_maint_floor_frac: float = 0.10    # f_nsc floor as C_lab -> 0 [-]
    # Mechanism 2 -- cold-deciduous freeze dormancy: for cold-deciduous PFTs
    # (surface_params.is_cold_deciduous: larch / arctic grass / boreal shrub),
    # zero foliar GPP and foliar R_maint below a freeze threshold -- the larch
    # leaf-drop strategy (carbon_cycle._cold_deciduous_dormancy_factor).
    # ``cold_deciduous`` is the per-PFT trait; ``cold_deciduous_dormancy`` the
    # master enable (both must be True for the gate to engage).
    cold_deciduous_dormancy: bool = False
    cold_deciduous: bool = False
    freeze_dormancy_threshold_K: float = constants.T_freeze  # dormancy below this T [K]
    dormancy_transition_width_K: float = 2.0                 # sigmoid half-width [K]
    # Leaf bootstrap (part of the cold-deciduous mechanism): a cold-deciduous
    # plant that lost its canopy (C_fol -> 0) but kept a labile reserve (the NSC
    # gate preserved it) must regrow a minimum leaf area from labile in the
    # growing season -- else GPP (computed from the ENTRY C_fol) is 0 forever ->
    # no NPP allocation -> permanent death (the leaf-out lock).  A pure
    # C_lab -> C_fol transfer (carbon_cycle: added to lab_release), conserving.
    leaf_bootstrap_lai: float = 0.5      # target min LAI regrown from labile [m2/m2]
    leaf_bootstrap_frac: float = 0.1     # max fraction of C_lab drawn per step [-]

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
    # --- Multi-pool SOM turnover [day^-1] (CENTURY / CLM4.5 topology) ---
    # Three soil-organic-matter pools of increasing residence time in a FORWARD
    # cascade active -> slow -> passive (Parton et al. 1987; Koven et al. 2013):
    # each pool's decomposition partly humifies to the next-slower pool and
    # partly respires to the atmosphere.  These are the BASE (reference-T,
    # unfrozen) turnover rates; the realised rate is scaled by the shared
    # modifier f_temp * f_moist * f_freeze (see carbon_cycle._som_decomp_modifier)
    # so cold/frozen soils turn SOM over slowly and RETAIN carbon.
    #   active  : MRT ~3 yr   (1-5 yr)     -- fast, unprotected topsoil carbon
    #   slow    : MRT ~25 yr  (20-50 yr)   -- physically protected carbon
    #   passive : MRT ~600 yr (500-1000 yr)-- mineral-stabilised carbon
    # Replaces the former single bulk ``tor_som``: a single ~68-yr pool with no
    # freeze control could not hold the deep, freeze-protected high-latitude /
    # grassland SOC that observations show (docs/land/multipool_som_phenology_
    # plan.md).  NOTE: no ``legacy_name: "tor_som"`` alias is carried on
    # ``tor_som_active`` in the param spec -- the mapping is NOT a faithful
    # rename.  The old ``tor_som`` was the BULK pool (~4e-5/day, ~68-yr MRT);
    # ``tor_som_active`` is the FAST active pool (~9e-4/day, ~3-yr MRT, ~22x
    # faster).  ``legacy_name`` is a FUNCTIONAL alias (``param_collector`` remaps
    # it), so aliasing them would let an externally-saved tuned JSON with the old
    # bulk ``tor_som`` silently set the active pool ~22x too slow.
    tor_som_active: float = 9e-4
    tor_som_slow: float = 1.1e-4
    tor_som_passive: float = 4.5e-6
    # Inter-pool humification (transfer) fractions: of a pool's decomposition,
    # this fraction moves to the next-slower pool; the remainder respires.
    # CENTURY microbial/stabilisation efficiencies span ~0.1-0.5.
    f_active_to_slow: float = 0.30
    f_slow_to_passive: float = 0.30
    # Half-width [K] of the smooth SOM freeze-suppression curve
    # f_freeze = som_freeze_floor + (1 - som_freeze_floor)
    #           * sigmoid((T - T_freeze) / som_freeze_width_K): warm soil -> 1
    # (no suppression), frozen soil -> som_freeze_floor (a FLOOR of the unfrozen
    # rate, NOT 0), so cold soils turn SOM over slowly and RETAIN carbon.  A
    # fixed NUMERICS smoothing width (never trained), mirroring
    # soil_thermal.freeze_curve_width_K.
    som_freeze_width_K: float = 2.0
    # Cold-soil decomposition FLOOR [dimensionless fraction in [0, 1)]: the
    # smallest fraction of the unfrozen decomposition rate a frozen soil retains,
    # so ``f_freeze`` ranges ``[som_freeze_floor, 1]`` instead of ``(0, 1]``.
    # Frozen soils still decompose SOME organic matter -- microbial activity in
    # unfrozen liquid films plus cryoturbation -- so decomposition never collapses
    # to 0 (CLM4.5 / CENTURY cold-soil floor; Koven et al. 2013, Parton et al.
    # 1987).  Without it ``f_freeze -> 0`` drives the millennial slow/passive SOM
    # pools' equilibrium (``I / (m * k)``) to a runaway in the coldest archetypes
    # (a real ERA5 build overshot to ~175 kgC/m2); the floor CAPS that while
    # KEEPING the cold-retains-more-SOC direction.  A tunable closure knob (tier
    # 2), not a numerics-only constant.
    som_freeze_floor: float = 0.05
    # --- Perennial-frost / anaerobic SOM protection (permafrost carbon) ---
    # ``som_freeze_floor`` above is the INSTANTANEOUS per-timestep aerobic
    # decomposition floor (microbial activity in unfrozen liquid films) that a
    # soil retains whenever it is frozen -- but the ANNUAL turnover of a cold
    # column is dominated by its brief unfrozen (thaw-season) window, so that
    # floor alone caps high-latitude SOC at ~74 kgC/m2 (tundra), far below the
    # observed 100-300 kgC/m2 permafrost/peat stocks (Hugelius et al. 2014
    # NCSCD).  A PERENNIALLY-frozen, waterlogged column additionally protects
    # its SOM year-round -- anaerobic (O2-limited) decomposition in the
    # meltwater-saturated active layer plus cryoturbation burying carbon into
    # the perennially-frozen, decomposition-shielded permafrost (Koven et al.
    # 2013; CLM4.5 cold-soil biogeochemistry).  This is a SEPARATE, whole-column
    # suppression keyed on the PERENNIAL-frost state (the annual frozen
    # fraction, an annual statistic), NOT the instantaneous temperature: it
    # multiplies the SOM decomposition modifier by ``f_perma`` in
    # ``[permafrost_protection_min, 1]`` so the effective rate of a truly
    # perennially-frozen column can drop BELOW the aerobic ``som_freeze_floor``
    # (the seasonally-frozen-but-thawing soils are left essentially unchanged;
    # temperate/tropical soils, annual frozen fraction ~0, get ``f_perma``~1).
    # Smallest fraction of the aerobic decomposition rate a fully perennially-
    # frozen / anaerobic column retains (the deep-frost analogue of
    # ``som_freeze_floor``, applied to the WHOLE-year rate).  <1 so it can only
    # RETAIN carbon (never amplify); >0 so the millennial slow/passive
    # equilibrium ``I/(m*k)`` stays finite.  Anaerobic C mineralisation runs
    # ~2-10x slower than aerobic (Schadel et al. 2016, Nature Clim. Change) and
    # cryoturbation removes carbon from the active decomposition zone, so the
    # effective whole-column rate is ~0.1-0.6 of the aerobic-equivalent.
    # Tunable closure knob (tier 2).
    permafrost_protection_min: float = 0.3
    # Annual frozen fraction at which the perennial-frost protection is
    # HALF-engaged -- the seasonal-frost -> perennial-frost (permafrost) climate
    # boundary.  Permafrost onset is ~MAAT -2 to -8 degC, whose climatological
    # annual frozen fraction is ~0.55-0.7 (Gruber 2012 permafrost zonation
    # index; Brown et al. 1997 IPA map).  Kept comfortably >0 so temperate /
    # tropical columns (frozen fraction ~0) sit deep in the unprotected tail
    # (``f_perma``~1).  Tunable closure knob (tier 2).
    permafrost_frozen_fraction_threshold: float = 0.6
    # Smoothing half-width [frozen-fraction units] of the perennial-frost
    # protection sigmoid in frozen-fraction space (spans the
    # sporadic->discontinuous->continuous permafrost transition).  A fixed
    # NUMERICS smoothing width (never trained; sibling of ``som_freeze_width_K``).
    permafrost_frozen_fraction_width: float = 0.06
    decomp_rate: float = 5e-4     # Litter -> SOM transfer [day^-1]
    # Coarse-woody-debris humification efficiency: the fraction of wood
    # turnover that becomes stable SOM.  The remainder respires to the
    # atmosphere (CWD heterotrophic respiration).  Without this split wood
    # turnover humified 100 % into the millennial SOM pool, giving an
    # unphysically large soil-carbon stock; ~0.3 matches the litter pathway's
    # ~0.2 SOM yield and typical CWD humification (Harmon et al. 1986).
    cwd_humification_eff: float = 0.3

    # --- Decomposition sensitivity ---
    Q10_exp: float = 0.04         # AUTOTROPHIC maint. resp.: exp(Q10_exp*(T-T_ref_ra))
    # Heterotrophic (soil) decomposition temperature sensitivity, DECOUPLED
    # from the autotrophic Q10 above.  0.09 -> Q10 ~= 2.5, the upper-realistic
    # soil-respiration range (Q10 2-3).  A strong soil Q10 makes WARM tropical
    # soils turn SOM over fast (low equilibrium SOM despite high productivity)
    # while COLD high-latitude soils retain carbon (high SOM) — the observed
    # SOC gradient the old weak, shared Q10=0.04 (Q10~1.5) could not produce.
    Q10_het_exp: float = 0.09
    moisture_factor: float = 0.5  # Moisture scaling strength
    T_ref: float = 283.15         # HETEROTROPHIC (soil-decomposition) reference T [K]
    # Autotrophic maintenance-respiration reference temperature [K].  This is a
    # FIXED NORMALIZATION of the temperature response exp(Q10_exp*(T-T_ref_ra)): it
    # sets ONLY the overall magnitude (a uniform exp(-Q10_exp*dref) factor at every
    # T) and is EXACTLY CONFOUNDED with the bulk ``r_maint_*`` amplitudes, so it does
    # NOT change the temperature-response SHAPE or the warm/cold spatial pattern
    # (that pattern is set by ``Q10_exp``, which is unchanged).  It is given its OWN
    # constant -- separate from the heterotrophic ``T_ref`` -- so the autotrophic
    # calibration stays INDEPENDENT of the heterotrophic configuration (retuning soil
    # decomposition must not perturb ``R_maint``); the previous code reused the
    # 10 degC (283.15 K) heterotrophic CENTURY reference here, coupling the two.
    # FIXED (not trained; see __param_spec__ excluded) precisely BECAUSE it is
    # non-identifiable with the rate amplitudes -- the amplitudes carry the
    # calibration.  25 degC is the standard leaf/tissue dark-respiration
    # normalization (leaf R_d; Ryan 1991; Atkin & Tjoelker 2003; maintenance-resp
    # DALEC variants use a 20-25 degC leaf reference, e.g. Famiglietti et al. 2021 at
    # 20 degC), a physically recognizable choice for the dominant foliar term;
    # root/wood are driven by top-soil T (multilayer_land) and are bulk effective
    # coefficients at this reference, not literal 25 degC tissue measurements.
    # Net effect (this fixed normalization together with the recalibrated
    # ``r_maint_*``): a uniform ~0.22x reduction of the maintenance-respiration
    # MAGNITUDE, so the coupled-equilibrium CUE TARGETS the observed ~0.45 (it was
    # ~0.11 with maintenance ~76% of GPP).  Soil carbon rises via the higher litter
    # input while the heterotrophic decomposition RATES stay unchanged (the coupled
    # equilibrium SOC still shifts, cold > warm preserved).
    T_ref_ra: float = 298.15
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
    # Leaf-habit / phenology TYPE. A categorical STRUCTURAL flag (like ``woody``
    # / ``hemisphere_aware``), NOT a tunable float -- only ``: float`` fields are
    # ``__param_spec__``-eligible, so it needs no spec entry.  False (default)
    # keeps every current numeric: the DALEC990 DECIDUOUS Gaussian Bday/Fday
    # leaf-fall pulse.  True selects near-CONTINUOUS leaf turnover
    # (``leaf_lifespan``-based, see ``carbon_cycle.compute_phenology``), correct
    # for broadleaf/needleleaf EVERGREEN PFTs whose canopy never drops to ~0 LAI
    # in a dormant season.  OFFLINE archetype-IC path only -- production coupled
    # runs keep the default (docs/land/multipool_som_phenology_plan.md, Phase B).
    evergreen: bool = False

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


class D13CConfig(NamedTuple):
    """Leaf carbon-isotope (delta13C) discrimination forward configuration.

    Threads the two C4-pathway leaf-delta13C knobs into
    :mod:`legoesm.land.carbon.d13c_forward`.  The C3 branch carries NO free
    config -- its diffusion ``a``, Rubisco ``b`` and boundary-condition
    ``delta13C_air`` are FIXED published Farquhar-1989 constants living in the
    forward's provenance block.  The C4 branch (Farquhar 1983 / Henderson 1992 /
    Cerling 1997) reads the bundle-sheath leakiness ``phi_c4_leakiness`` -- the
    natural C4 water-use-efficiency / delta13C lever -- and a C4-characteristic
    intercellular:ambient CO2 ratio ``ci_ca_c4``.

    C4 Ci/Ca treatment (option (a) -- fixed setpoint).  The model's Farquhar
    biochemistry is C3-only, so it supplies no FAITHFUL C4 leaf Ci (a C3-kinetics
    Ci run for a C4 PFT is not the true CO2-concentrated bundle-sheath state).
    The C4 branch therefore uses a prescribed C4-characteristic Ci/Ca (~0.4; C4
    leaves regulate Ci/Ca relatively tightly) rather than the solved C3 Ci, so
    the C4 delta13C depends MAINLY on ``phi``.  ``ci_ca_c4`` is FIXED (excluded
    from training): in the C4 discrimination form the trained part enters ONLY
    through the product ``(b4 + (b3 - s) * phi - a) * (Ci/Ca)``, so exposing BOTH
    ``phi`` and ``Ci/Ca`` as free parameters would be a non-identifiable
    degeneracy (many ``(phi, Ci/Ca)`` pairs give the same delta13C).  ``phi``
    (leakiness, Henderson 1992) is the single, physically-primary C4 lever.
    """
    phi_c4_leakiness: float = 0.21  # C4 bundle-sheath leakiness [-] (Henderson 1992)
    ci_ca_c4: float = 0.4           # C4-characteristic intercellular:ambient CO2 ratio [-]


class CarbonState(NamedTuple):
    """Prognostic carbon pool state.

    All fields share the land model's spatial shape (e.g. (6,n,n)
    for cubed-sphere slab land or (ncol,) for columnar).
    Units: gC/m2.

    Soil organic matter is resolved as three pools of increasing turnover time
    (CENTURY / CLM4.5 topology) in a FORWARD cascade active -> slow -> passive
    (phase A2, live): litter decomposition + coarse-woody-debris humification
    feed the active pool; each pool's decomposition partly humifies to the next
    pool and partly respires.  Use :func:`som_total` for any total-SOM context.
    """
    C_lab: jax.Array     # Labile carbon
    C_fol: jax.Array     # Foliar carbon
    C_root: jax.Array    # Root carbon
    C_wood: jax.Array    # Wood carbon
    C_lit: jax.Array     # Litter (dead foliage + roots)
    C_som_active: jax.Array   # Active soil organic matter (fast turnover, ~1-5 yr)
    C_som_slow: jax.Array     # Slow soil organic matter (~20-50 yr)
    C_som_passive: jax.Array  # Passive soil organic matter (~500-1000 yr)


def som_total(state: CarbonState) -> jax.Array:
    """Total soil organic matter = active + slow + passive pools [gC/m2].

    Single source of truth for the many total-SOM call sites (diagnostics,
    conservation sums, IC mapping, plotting).  Lives here in ``config`` — the
    leaf module that defines :class:`CarbonState` and imports nothing else from
    the package — so every caller (carbon_cycle, spinup, global_init, run
    scripts, validators, tests) can import it without an import cycle.  In phase
    A1 the slow/passive pools are inert (== 0), so ``som_total == C_som_active``;
    the helper keeps every total-context site correct once the cascade is
    populated in A2.
    """
    return state.C_som_active + state.C_som_slow + state.C_som_passive


def is_concrete(*values) -> bool:
    """True iff EVERY argument can be evaluated in a Python boolean context, so
    the fail-early bound checks below -- :func:`validate_som_transfer_fractions`
    and the ``som_freeze_floor`` guard in ``carbon_cycle._freeze_modifier`` -- can
    enforce it.

    A value is concrete when a Python ``bool`` accepts it: a host scalar, a NumPy
    scalar / 0-d array, OR a materialised ``jax.Array`` outside a trace.  For all
    of these an out-of-range value still fails LOUD (production, a direct
    ``CarbonConfig(f_active_to_slow=1.5)``, AND a stray concrete
    ``jnp.asarray(1.5)`` all raise) -- the fail-early guard is NOT weakened for
    any concretely-knowable value.

    A value is NOT concrete when it is a tracer, OR a concrete ``jax`` constant
    lifted into an enclosing trace -- e.g. the ``lax.scan`` spin-up body of the
    differentiable calibration path
    (:func:`legoesm.land.carbon.global_init.equilibrate_archetypes_traced`): a
    Python ``bool`` on it raises ``TracerBoolConversionError`` (an
    ``isinstance(_, jax.core.Tracer)`` test MISSES the lifted-constant case,
    which is why boolability is probed directly).  Such abstract values are left
    unchecked here because a Python fail-early on a traced value is impossible by
    construction of automatic differentiation; their bound is instead guaranteed
    STRUCTURALLY UPSTREAM by the ``__param_spec__`` constraint transform (a
    sigmoid maps every raw input onto ``(lo, hi)``), which is how the sanctioned
    producer ``training.param_collector.build_trainable_params(...).to_overrides()``
    supplies the SOM overrides -- see
    :func:`legoesm.land.carbon.global_init.equilibrate_archetypes_traced` for the
    override-bound contract.
    """
    try:
        for v in values:
            bool(v == v)
    except jax.errors.TracerBoolConversionError:
        return False
    return True


def validate_som_transfer_fractions(
    f_active_to_slow: float, f_slow_to_passive: float,
    cwd_humification_eff: float,
) -> None:
    """Fail-early validation for the SOM cascade's humification fractions
    (dispatch-hardening discipline: a plain Python ``if ... raise ValueError``
    on the STATIC value, matching ``carbon_cycle._freeze_modifier``'s
    ``som_freeze_width_K > 0`` guard -- NOT a traced ``jnp`` check).

    Covers the two inter-pool cascade fractions AND the coarse-woody-debris
    humification efficiency ``cwd_humification_eff`` -- all three are
    "fraction of a decomposition/turnover flux routed onward, remainder
    respires" quantities that MUST lie in [0, 1].

    The ``__param_spec__`` bounds above (0.1-0.5 / 0.05-0.6) only constrain the
    TRAINING search range and are never enforced at runtime, so a direct
    ``CarbonConfig(f_active_to_slow=1.5)`` / ``CarbonConfig(cwd_humification_eff=
    1.5)`` construction (or an out-of-range float threaded straight into the
    spin-up solver) bypasses them entirely.  An out-of-[0, 1] fraction breaks
    the cascade's "carbon into a pool is positive" sign convention
    (``carbon_cycle.step_carbon_differland``): a pool's OWN heterotrophic
    respiration ``(1 - f) * D_X`` goes NEGATIVE for ``f > 1`` and a downstream
    transfer INPUT goes negative for ``f < 0``; likewise the CWD split
    ``wood_to_som = cwd_humification_eff * wood_litter`` /
    ``R_het_cwd = wood_litter - wood_to_som`` gives a negative respiration
    (creates carbon) for ``eff > 1`` and a negative SOM input (destroys carbon)
    for ``eff < 0``.

    Lives here in ``config`` (the same import-cycle-free leaf module as
    :func:`som_total`) so every public entry point that consumes these
    fractions shares ONE check instead of re-deriving it:
    ``carbon_cycle.step_carbon_differland``,
    ``spinup.run_semi_analytic_spinup`` (fails before the expensive
    transient scan) and ``spinup.analytic_slow_pool_equilibrium`` (the
    direct ``run_lmip.py`` call site that bypasses the step-level guard).
    """
    # Enforce fail-early only for concretely-knowable values (host scalars AND
    # concrete arrays both raise on an out-of-range value).  Skip for abstract
    # values -- a tracer, or a concrete constant lifted inside the lax.scan
    # spin-up body of the differentiable-calibration path -- whose bound is
    # guaranteed upstream by the __param_spec__ sigmoid transform and cannot be
    # Python-checked (see :func:`is_concrete`).
    if not is_concrete(f_active_to_slow, f_slow_to_passive, cwd_humification_eff):
        return
    if not (0.0 <= f_active_to_slow <= 1.0):
        raise ValueError(
            "f_active_to_slow must be in [0, 1] (the fraction of "
            "active-SOM decomposition humified onward to the slow pool), "
            f"got {f_active_to_slow!r}."
        )
    if not (0.0 <= f_slow_to_passive <= 1.0):
        raise ValueError(
            "f_slow_to_passive must be in [0, 1] (the fraction of "
            "slow-SOM decomposition humified onward to the passive pool), "
            f"got {f_slow_to_passive!r}."
        )
    if not (0.0 <= cwd_humification_eff <= 1.0):
        raise ValueError(
            "cwd_humification_eff must be in [0, 1] (the fraction of "
            "coarse-woody-debris turnover humified to stable SOM; the "
            "remainder respires to the atmosphere), "
            f"got {cwd_humification_eff!r}."
        )


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
        r_het_som == (1 - f_active_to_slow)*som_active_loss
                     + (1 - f_slow_to_passive)*som_slow_loss + som_passive_loss
    ``som_active_loss``/``som_slow_loss``/``som_passive_loss`` are each pool's
    TOTAL decomposition D_X [gC/m2/day] (transfer to the next pool + respiration);
    the semi-analytic slow-pool solve reads them to infer each pool's turnover.
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
    som_active_loss: jax.Array   # Active SOM total decomposition D_active
    som_slow_loss: jax.Array     # Slow SOM total decomposition D_slow
    som_passive_loss: jax.Array  # Passive SOM total decomposition D_passive
    lai: jax.Array          # Leaf area index (C_fol / LCMA)
