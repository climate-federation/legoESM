"""Land carbon cycle configuration and state containers."""

from __future__ import annotations

from typing import NamedTuple

import jax


class StomataConfig(NamedTuple):
    """Stomatal conductance + A-gs coupling configuration.

    Used by the **SimpleSEB** surface scheme (slab / multilayer land
    without the two-leaf canopy).  The ``TwoLeafCanopy`` scheme has its
    own per-column parameter set (``CanopyLandParams``); these scalars
    are the single-valued defaults for the uniform SimpleSEB path.

    Modes:

    - ``enabled=False`` (default): bucket ``beta`` only, no stomatal control.
    - ``enabled=True`` + carbon off: ``jarvis_gs`` (CO2-independent).
    - ``enabled=True`` + carbon on : coupled Leuning Farquhar + Ball-Berry
                                     (or Medlyn) via ``coupled_farquhar_stomata``.

    All photosynthesis fields refer to the **Leuning** C3/C4 model in
    ``canopy/photosynthesis.py``; the Bernacchi (CLM-style) formulation
    was removed.
    """
    enabled: bool = False
    stomata_model: str = "ball_berry"   # "ball_berry" | "medlyn"

    # --- Leuning Farquhar scalar defaults (overridden by LandSurfaceParams) ---
    Vcmax25_C3: float = 60.0        # C3 max carboxylation at 25 C [μmol/m2/s]
    Vcmax25_C4: float = 40.0        # C4 max carboxylation at 25 C [μmol/m2/s]
    alf_C3: float = 0.30            # C3 quantum yield [mol CO2 / mol photons]
    fC4: float = 0.0                # C4 area fraction [0-1] (continuous)
    TgC_default: float = 25.0       # Default 30-day mean growth T [°C]

    # --- Ball-Berry / Medlyn stomatal params ---
    g0: float = 0.01                # Residual conductance [mol/m2/s]
    g1_bb: float = 9.0              # Ball-Berry slope [-]
    g1_med: float = 4.0             # Medlyn slope [kPa^0.5]

    # --- Jarvis params (carbon off path) ---
    gs_max: float = 0.3             # Maximum conductance [mol/m2/s]
    K_PAR: float = 200.0            # PAR half-saturation [W/m2]
    T_opt_jarvis: float = 25.0      # Optimal T [°C]
    T_range_jarvis: float = 20.0    # T response half-width [°C]
    a_vpd: float = 0.05             # VPD sensitivity [1/hPa]

    # --- A-gs coupling solver ---
    n_iter_ags: int = 5             # Fixed-point iterations (Phase 2 → Newton)

    # --- Beta coupling ---
    k_ext: float = 0.5              # Beer-law canopy extinction [-]
    gs_ref: float = 0.3             # Reference gs for beta scaling [mol/m2/s]


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
    T_opt: float = 25.0           # Optimal temperature for photosynthesis [deg C]
    T_width: float = 15.0         # Temperature response width [deg C]
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
