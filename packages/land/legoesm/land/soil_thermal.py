"""Multi-layer soil heat diffusion (Task 8D).

Solves the 1D heat equation:
    C_eff(z) · ∂T/∂t = ∂/∂z [k_eff(z) · ∂T/∂z]

Thermal properties depend on soil moisture via the Johansen (1975) method.
Discretized with backward Euler and solved via Thomas algorithm.

Soil water FREEZE/THAW is modelled (opt-in, ``SoilThermalConfig.
enable_freeze_thaw``) via the APPARENT-HEAT-CAPACITY method: the soil water is
partitioned into an unfrozen liquid fraction ``theta_liq(T)`` (a smooth
freezing curve, ``freeze_curve_width_K``, with a residual film-water fraction
``theta_liq_residual_frac``) and ice ``theta_ice = theta - theta_liq``.  The
latent heat of fusion enters the effective heat capacity as
``C_latent = rho_water * L_f * d(theta_liq)/dT >= 0``, so a column cooling /
warming through ``constants.T_freeze`` releases / absorbs the fusion enthalpy
``rho_water * L_f * (theta - theta_min)`` instead of changing temperature
freely — the zero-curtain plateau.  The apparent heat capacity is evaluated at
the current temperature (linearised); the stored-enthalpy change matches the
sensible + latent enthalpy change to first order in dT, so column energy closes
to O(dt) per step (exact in the linearised metric — see
``tests/land/test_soil_freeze_thaw.py``).  When disabled (default) the module
reduces EXACTLY to sensible-heat diffusion (bit-identical to prior behaviour).

Scope of this first implementation: THERMAL freeze/thaw only (the dominant
zero-curtain physics).  Water moved by Richards at fixed temperature changes
the diagnosed ice without any temperature change; its fusion heat enters the
final thermal solve as an explicit per-layer source
(``moisture_fusion_heat_source``).  Two coupled refinements are deliberately NOT included
and are tracked follow-ups: (a) ice-aware thermal CONDUCTIVITY (frozen soil
conducts better, k_ice ~ 2.0 vs k_water ~ 0.57) — the conductivity still uses
total ``theta``; (b) hydraulic IMMOBILISATION of the ice fraction in Richards
(frozen water should not drain) — the hydraulics still see total ``theta``.
Both are secondary to the latent zero-curtain and are noted at their sites.

References
----------
- Johansen (1975): Thermal conductivity of soils. PhD thesis.
- de Vries (1963): Thermal properties of soils.
- Cox et al. (1999); Niu & Yang (2006): soil freezing curve / supercooled
  liquid water and the apparent-heat-capacity treatment of soil freeze/thaw.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.land.soil_grid import SoilGrid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.timestepping.tridiagonal import thomas_solve

__param_spec__ = {
    "SoilThermalConfig": {
        "scheme_key": "land.soil_thermal",
        "excluded": {
            "C_soil": "material: mineral soil volumetric heat capacity",
            "C_water_vol": "material: water volumetric heat capacity (= rho_water·c_pw)",
            "C_air": "material: air volumetric heat capacity",
            "k_solid": "material: mineral soil conductivity",
            "k_water": "material: water conductivity",
            "rho_bulk": "material: soil bulk density",
            "kersten_sr_floor_coarse": "numerics: log10 argument floor (sand)",
            "kersten_sr_floor_fine": "numerics: log10 argument floor (loam)",
            "sr_clip_min": "numerics: saturation-ratio floor",
            "C_ice_vol": "material: ice volumetric heat capacity (= rho_ice·c_pi)",
            "freeze_curve_width_K": "numerics: freezing-curve smoothing half-width [K]",
            "snow_bulk_density_kg_m3": "material: fixed bulk snow density of the "
                                       "one-layer snow thermal node (no density "
                                       "state; user-fixed 250 kg/m^3, 2026-10-03)",
        },
        "params": {
            "theta_liq_residual_frac": {
                "units": "1", "bounds": (0.0, 0.2), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "residual unfrozen (supercooled film) liquid-water "
                             "fraction below freezing (Niu & Yang 2006)",
                "shape": None,
            },
            "Q_geothermal": {
                "units": "W/m^2", "bounds": (0.0, 0.15), "tunable_tier": 1,
                "transform": "sigmoid", "category": "boundary",
                "reference": "Pollack et al. (1993) global mean ~0.087 W/m^2",
                "shape": None,
            },
            "k_dry_coeff_a": {
                "units": "W·m^2/(kg·K)", "bounds": (0.08, 0.20), "tunable_tier": 2,
                "transform": "sigmoid", "category": "material",
                "reference": "de Vries (1963) dry-conductivity fit", "shape": None,
            },
            "k_dry_offset_w_per_m_k": {
                "units": "W/(m·K)", "bounds": (40.0, 90.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "material",
                "reference": "de Vries (1963) dry-conductivity fit", "shape": None,
            },
            "k_dry_density_coeff": {
                "units": "1", "bounds": (0.7, 1.2), "tunable_tier": 2,
                "transform": "sigmoid", "category": "material",
                "reference": "de Vries (1963) dry-conductivity fit", "shape": None,
            },
            "kersten_slope_coarse": {
                "units": "1", "bounds": (0.4, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "Johansen (1975) Kersten number, coarse soils", "shape": None,
            },
        },
    },
}


# --- snow thermal conductivity (Jordan 1991, as in CLM5 SoilTemperatureMod) ---
# thk = tkair + (7.75e-5*rho + 1.105e-6*rho^2) * (tkice - tkair); CTSM 5.1
# SoilTemperatureMod.F90:690 with clm_varcon tkair=0.023, tkice=2.290.  The two
# end-member conductivities are part of the published fit, so they are kept with
# it rather than taken from ``constants`` (whose k_ice/k_air differ slightly).
_JORDAN_TKAIR_W_M_K = 0.023
_JORDAN_TKICE_W_M_K = 2.290
_JORDAN_LIN_M3_KG = 7.75e-5
_JORDAN_QUAD_M6_KG2 = 1.105e-6


class SoilThermalConfig(NamedTuple):
    """Configuration for soil thermal properties.

    All heat capacities are **volumetric** (J/m³/K), not specific
    (J/kg/K).  ``C_water_vol`` derives from
    ``constants.rho_water · constants.c_pw = 1000 · 4218 ≈ 4.218e6``
    (referencing the canonical constants rather than a hardcoded
    literal — the prior 4.18e6 default used a stale c_pw=4180 and was
    ~0.9 % low relative to ``constants.c_pw``).
    Storing volumetric values directly avoids per-cell multiplication
    by density inside the heat-capacity mixing formula.
    """
    # --- material heat capacities / conductivities -------------------------
    C_soil: float = 2.0e6         # mineral soil heat capacity [J/m3/K]
    C_water_vol: float = constants.rho_water * constants.c_pw  # water heat capacity [J/m3/K]
    C_air: float = 1.25e3         # air heat capacity [J/m3/K]
    k_solid: float = 2.0          # mineral soil thermal conductivity [W/m/K]
    k_water: float = 0.57         # water thermal conductivity [W/m/K]
    rho_bulk: float = 1400.0      # bulk density [kg/m3]
    soil_texture: str = "loam"    # "sand" (coarse) or "loam" (fine)
    Q_geothermal: float = 0.05   # Geothermal heat flux at bottom [W/m2]
    # --- dry conductivity (de Vries 1963): k_dry = (a·rho_b + b) / (rho_particle − c·rho_b) ---
    k_dry_coeff_a: float = 0.135           # [W·m2/(kg·K)] numerator density slope
    k_dry_offset_w_per_m_k: float = 64.7   # [W/(m·K)] numerator offset
    k_dry_density_coeff: float = 0.947     # [-] denominator density coefficient
    # --- Kersten number (Johansen 1975): K_e = slope·log10(Sr) + 1 ---------
    kersten_slope_coarse: float = 0.7      # [-] sand slope (loam slope is 1.0)
    kersten_sr_floor_coarse: float = 0.05  # [-] log10 argument floor, sand
    kersten_sr_floor_fine: float = 0.1     # [-] log10 argument floor, loam
    sr_clip_min: float = 0.01              # [-] saturation-ratio floor
    # --- soil-water freeze/thaw (apparent heat capacity) -------------------
    enable_freeze_thaw: bool = False       # opt-in; default off = sensible-only
    C_ice_vol: float = constants.rho_ice * constants.c_pi  # ice heat cap [J/m3/K]
    freeze_curve_width_K: float = 0.5      # [K] smooth freezing-curve half-width
    theta_liq_residual_frac: float = 0.05  # [-] residual unfrozen liquid fraction
    # --- one-layer snow thermal node (opt-in, ``solve_snow_soil_thermal``) --
    # False (default) = snow has no heat capacity and no insulation (the
    # legacy bucket), bit-identical to before.
    snow_insulation: bool = False
    snow_bulk_density_kg_m3: float = 250.0  # [kg/m3] fixed bulk snow density


def compute_heat_capacity(
    theta: jnp.ndarray,
    hydro_config: SoilHydraulicsConfig,
    thermal_config: SoilThermalConfig,
) -> jnp.ndarray:
    """Compute effective volumetric heat capacity [J/m3/K].

    C_eff = (1 - θ_sat)·C_soil + θ·C_water_vol + (θ_sat - θ)·C_air
    """
    theta_sat = hydro_config.theta_sat
    return ((1.0 - theta_sat) * thermal_config.C_soil
            + theta * thermal_config.C_water_vol
            + (theta_sat - theta) * thermal_config.C_air)


def liquid_water_content(
    T_soil: jnp.ndarray,
    theta: jnp.ndarray,
    thermal_config: SoilThermalConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unfrozen liquid water content ``theta_liq(T)`` and ``d(theta_liq)/dT``.

    Smooth (C-infinity) sigmoid freezing characteristic centred at
    ``constants.T_freeze`` with half-width ``freeze_curve_width_K`` [K]::

        frozen_frac(T) = sigmoid(-(T - T_freeze) / w)          in [0, 1]
        theta_min      = theta_liq_residual_frac * theta       (residual film)
        theta_liq      = theta - (theta - theta_min) * frozen_frac

    ``theta_liq`` -> ``theta`` (all liquid) for T >> T_freeze and -> ``theta_min``
    (residual supercooled film water) for T << T_freeze, monotonically
    INCREASING in T, so ``d(theta_liq)/dT >= 0``.  The total fusion enthalpy
    released between fully-thawed and fully-frozen is
    ``rho_water * L_f * (theta - theta_min)`` [J/m^3] regardless of ``w`` (which
    only sets the temperature width of the zero-curtain plateau), so energy is
    conserved by construction.

    Returns ``(theta_liq, dtheta_liq_dT)``, both the shape of ``theta``.
    """
    w = thermal_config.freeze_curve_width_K
    # ``freeze_curve_width_K`` is a fixed numerics field (never trained -> always
    # a static Python float), so validate it here: w <= 0 would divide by zero
    # and flip the sign of dtheta_liq/dT (making C_latent negative -> unstable).
    if not w > 0.0:
        raise ValueError(
            f"freeze_curve_width_K must be > 0, got {w!r}."
        )
    # ``theta_liq_residual_frac`` may be traced (tunable, tier 2) so it is not
    # Python-branched here; instead clamp the residual liquid to ``[0, theta]``
    # (traced-safe safety floor: the residual film cannot exceed the total
    # water, nor be negative).  This GUARANTEES ``theta - theta_min >= 0`` and
    # hence ``dtheta_liq/dT >= 0`` and ``C_latent >= 0`` for ANY config value.
    # It is a no-op for the spec-bounded range ``[0, 0.2]`` (r*theta < theta).
    theta_min = jnp.clip(thermal_config.theta_liq_residual_frac * theta, 0.0, theta)
    frozen = jax.nn.sigmoid(-(T_soil - constants.T_freeze) / w)  # ->1 cold, ->0 warm
    theta_liq = theta - (theta - theta_min) * frozen
    # d(frozen)/dT = sigmoid'*dx/dT = frozen*(1-frozen)*(-1/w); the minus sign
    # cancels with the -(theta - theta_min) prefactor => derivative >= 0.
    dtheta_liq_dT = (theta - theta_min) * frozen * (1.0 - frozen) / w
    return theta_liq, dtheta_liq_dT


def moisture_fusion_heat_source(
    T_soil: jnp.ndarray,
    theta_old: jnp.ndarray,
    theta_new: jnp.ndarray,
    dz: jnp.ndarray,
    thermal_config: SoilThermalConfig,
    dt: float,
) -> jnp.ndarray:
    """Fusion heat [W/m2 per layer, + = heating] of the ice change caused by a
    soil-water change at FIXED temperature.

    The apparent heat capacity only charges latent heat for ice that changes
    with T.  When water moves (infiltration, drainage, root uptake,
    evaporation) at fixed T, the diagnosed ice ``theta - theta_liq(T, theta)``
    changes too, with no fusion heat — an enthalpy leak.  This returns
    ``rho_water * L_f * [ice(T, theta_new) - ice(T, theta_old)] * dz / dt``:
    ice created releases heat, ice removed absorbs it.  ``T_soil`` MUST be the
    start-of-step temperature.  With one thermal step this closes the solver's
    linearised identity exactly; with sub-steps (capacity re-evaluated at each
    sub-step start) the closed identity is the sum over sub-steps, and the true
    nonlinear enthalpy balance holds only approximately.

    Conventions: moving water carries no sensible heat (unchanged); ice that
    leaves a layer by drainage, roots or evaporation, or moves between layers,
    is melted there at the layer's expense (new with this source).
    """
    # Same dtype for both moisture states, so unchanged water gives exactly 0.
    theta_old = jnp.asarray(theta_old).astype(jnp.result_type(theta_old, theta_new))
    theta_new = jnp.asarray(theta_new).astype(theta_old.dtype)
    liq_old, _ = liquid_water_content(T_soil, theta_old, thermal_config)
    liq_new, _ = liquid_water_content(T_soil, theta_new, thermal_config)
    d_ice = (theta_new - liq_new) - (theta_old - liq_old)
    # Sign: freezing (d_ice > 0) releases latent heat into the layer.
    return constants.rho_water * constants.L_f * d_ice * dz / dt


def compute_apparent_heat_capacity(
    T_soil: jnp.ndarray,
    theta: jnp.ndarray,
    hydro_config: SoilHydraulicsConfig,
    thermal_config: SoilThermalConfig,
) -> jnp.ndarray:
    """Effective volumetric heat capacity [J/m3/K] INCLUDING soil freeze/thaw.

    Apparent-heat-capacity method::

        C_app = C_sensible + C_latent
        C_sensible = (1 - theta_sat)*C_soil + theta_liq*C_water_vol
                     + theta_ice*C_ice_vol + (theta_sat - theta)*C_air
        C_latent   = rho_water * L_f * d(theta_liq)/dT              (>= 0)

    ``C_latent`` is the zero-curtain: near ``T_freeze`` it inflates the heat
    capacity so the fusion enthalpy is released / absorbed instead of the
    temperature moving freely.  Evaluated at the CURRENT temperature (the
    linearised apparent heat capacity used by ``solve_soil_thermal`` when
    ``enable_freeze_thaw`` is set); the sensible split uses ``C_ice_vol`` for
    the frozen fraction.
    """
    theta_sat = hydro_config.theta_sat
    theta_liq, dtheta_liq_dT = liquid_water_content(T_soil, theta, thermal_config)
    theta_ice = jnp.maximum(theta - theta_liq, 0.0)
    C_sensible = ((1.0 - theta_sat) * thermal_config.C_soil
                  + theta_liq * thermal_config.C_water_vol
                  + theta_ice * thermal_config.C_ice_vol
                  + (theta_sat - theta) * thermal_config.C_air)
    C_latent = constants.rho_water * constants.L_f * dtheta_liq_dT
    return C_sensible + C_latent


def compute_thermal_conductivity(
    theta: jnp.ndarray,
    hydro_config: SoilHydraulicsConfig,
    thermal_config: SoilThermalConfig,
) -> jnp.ndarray:
    """Compute effective thermal conductivity [W/m/K].

    Uses Johansen (1975):
        k_eff = k_dry + (k_sat - k_dry) · K_e(Sr)

    where K_e is the Kersten number.
    """
    theta_sat = hydro_config.theta_sat
    theta_r = hydro_config.theta_r

    # Saturation ratio
    Sr = jnp.clip(
        (theta - theta_r) / (theta_sat - theta_r + 1e-10),
        thermal_config.sr_clip_min,
        1.0,
    )

    # Dry conductivity (de Vries 1963)
    rho_b = thermal_config.rho_bulk
    k_dry = (
        thermal_config.k_dry_coeff_a * rho_b + thermal_config.k_dry_offset_w_per_m_k
    ) / (constants.rho_soil_particle - thermal_config.k_dry_density_coeff * rho_b)

    # Saturated conductivity (geometric mean)
    k_sat = (thermal_config.k_solid ** (1.0 - theta_sat)
             * thermal_config.k_water ** theta_sat)

    # Kersten number (Johansen 1975).  Validate the texture selector on the
    # static config value (dispatch-hardening: a typo like "Sand"/"silt" must
    # raise, not silently run the fine-soil branch).
    _valid_textures = ("sand", "loam")
    if thermal_config.soil_texture not in _valid_textures:
        raise ValueError(
            f"Unknown soil_texture {thermal_config.soil_texture!r}; "
            f"expected one of {_valid_textures}."
        )
    if thermal_config.soil_texture == "sand":
        K_e = thermal_config.kersten_slope_coarse * jnp.log10(
            jnp.clip(Sr, thermal_config.kersten_sr_floor_coarse, None)
        ) + 1.0
    else:  # "loam" (fine)
        K_e = jnp.log10(jnp.clip(Sr, thermal_config.kersten_sr_floor_fine, None)) + 1.0
    K_e = jnp.clip(K_e, 0.0, 1.0)

    return k_dry + (k_sat - k_dry) * K_e


def solve_soil_thermal(
    T_soil: jnp.ndarray,
    theta: jnp.ndarray,
    grid: SoilGrid,
    hydro_config: SoilHydraulicsConfig,
    thermal_config: SoilThermalConfig,
    G_surface: jnp.ndarray,
    dt: float,
    surface_conductance: jnp.ndarray | None = None,
    layer_source: jnp.ndarray | None = None,
    n_substeps: int = 1,
) -> jnp.ndarray:
    """Solve soil heat diffusion for one time step (backward Euler).

    Parameters
    ----------
    T_soil : jnp.ndarray
        Soil temperature [K], shape (ncol, n_layers).
    theta : jnp.ndarray
        Volumetric water content [m3/m3], shape (ncol, n_layers).
    grid : SoilGrid
        Vertical soil grid.
    hydro_config : SoilHydraulicsConfig
        For theta_sat, theta_r.
    thermal_config : SoilThermalConfig
        Thermal property parameters.
    G_surface : jnp.ndarray
        Ground heat flux into top layer [W/m2], shape (ncol,).
        Positive = into soil.
    dt : float
        Time step [s].
    surface_conductance : jnp.ndarray, optional
        Surface energy-balance conductance ``lambda = -dG_surface/dT_sfc``
        [W/m2/K, >= 0], shape (ncol,).  When supplied, the top boundary is
        treated SEMI-IMPLICITLY: the linearised T_sfc-dependence of the
        surface energy balance (sigma T^4 radiation + bulk SH/LH transfer) is
        folded into the implicit solve, so a large ``dt`` with a thin top layer
        under a stiff (high-roughness / high-insolation) surface stays stable
        instead of overshooting and diverging.  ``None`` (default) reduces
        EXACTLY to the explicit Neumann ground-heat-flux BC (bit-identical for
        every existing caller).
    layer_source : jnp.ndarray, optional
        Explicit per-layer heat source [W/m2 of column, positive = heating],
        shape (ncol, n_layers), added to each layer's RHS.  ``None`` (default)
        adds nothing (bit-identical).  See ``moisture_fusion_heat_source``.
    n_substeps : int
        Number of equal backward-Euler sub-steps (static Python int, default 1
        = one step, bit-identical).  The apparent heat capacity is re-evaluated
        at each sub-step's start temperature, which keeps a thin layer from
        overshooting the freezing curtain at a long step.

    Returns
    -------
    T_new : jnp.ndarray
        Updated soil temperature [K], shape (ncol, n_layers).
    """
    # Soil only (no snow node): the body lives in ``_solve_column``.
    return _solve_column(T_soil, theta, grid, hydro_config, thermal_config,
                         G_surface, dt, surface_conductance, layer_source,
                         n_substeps, snow=None)[0]


def snow_thermal_conductivity(rho_snow):
    """Bulk dry-snow thermal conductivity [W/m/K] from density [kg/m3].

    Jordan (1991), exactly as CLM5 (CTSM 5.1 SoilTemperatureMod.F90:690):
    ``k = tkair + (7.75e-5*rho + 1.105e-6*rho^2) * (tkice - tkair)``.
    0.223 W/m/K at 250 kg/m3.
    """
    return _JORDAN_TKAIR_W_M_K + (
        _JORDAN_LIN_M3_KG * rho_snow + _JORDAN_QUAD_M6_KG2 * rho_snow ** 2
    ) * (_JORDAN_TKICE_W_M_K - _JORDAN_TKAIR_W_M_K)


def melt_snow_node_excess(snow_swe, T_snow):
    """Melt snow with the snow node's heat above ``T_freeze`` (fixed enthalpy).

    The node holds ``E = c_ice * snow_swe * max(T_snow - T_freeze, 0)``
    [J/m2].  It melts ``m = min(snow_swe, E / L_f)`` [kg/m2]; the remaining
    pack sits at ``T_freeze`` and ``heat = E - m * L_f >= 0`` [J/m2] is what is
    left once the whole pack has melted (nonzero only then).  Exactly
    ``c_ice*S*(T-Tf) == m*L_f + heat``.  Returns
    ``(snow_swe - m, T_snow_new, m, heat)``; columns with no excess are
    unchanged.
    """
    tf = constants.T_freeze
    excess = constants.c_pi * snow_swe * jnp.maximum(T_snow - tf, 0.0)
    melt = jnp.minimum(snow_swe, excess / constants.L_f)
    # Leftover only where the whole pack melted (exact zero elsewhere, not a
    # rounding residue of excess - melt*L_f).
    heat = jnp.where(melt < snow_swe, 0.0, excess - melt * constants.L_f)
    T_new = jnp.where(excess > 0.0, tf, T_snow)
    return snow_swe - melt, T_new, melt, heat


def solve_snow_soil_thermal(
    T_snow: jnp.ndarray,
    snow_swe: jnp.ndarray,
    T_soil: jnp.ndarray,
    theta: jnp.ndarray,
    grid: SoilGrid,
    hydro_config: SoilHydraulicsConfig,
    thermal_config: SoilThermalConfig,
    G_surface: jnp.ndarray,
    dt: float,
    surface_conductance: jnp.ndarray | None = None,
    layer_source: jnp.ndarray | None = None,
    n_substeps: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Soil heat diffusion with ONE snow thermal node above the soil.

    Where a column holds snow (``snow_swe > 0``) the snowpack is an extra node
    on top of the soil column, solved in the SAME implicit tridiagonal system
    (as CLM5 solves its snow and soil layers together):

    * node temperature ``T_snow`` [K] is the SNOW-SURFACE temperature: the
      ground heat flux ``G_surface`` acts on it (explicit; a semi-implicit
      ``surface_conductance`` is refused);
    * heat capacity ``c_ice * snow_swe`` [J/m2/K] (snow is ice; the bucket holds
      no liquid);
    * the whole pack's resistance ``R_s = (snow_swe/rho_s)/k_s`` [m2 K/W] lies
      between that surface node and the top soil node, in series with the top
      soil half-layer ``dz0/(2 k0)``.  ``rho_s`` is the fixed bulk density
      ``snow_bulk_density_kg_m3``; ``k_s`` is Jordan (1991).  A single node at
      the pack MID-depth would expose the surface to only half of ``R_s``
      (CLM5 avoids that with a ~2 cm top layer); one layer of uniform cover
      over the whole cell.

    Snow-free columns (``snow_swe <= 0``) take the plain soil solve, bit-
    identical to ``solve_soil_thermal``, and return ``T_snow_new = T_soil_new
    [:, 0]`` (the node then just mirrors the skin, which is the top soil).

    Energy: the solve uses the snow heat capacity at ``snow_swe`` (the mass the
    caller hands in, i.e. after this step's accumulation, melt and
    sublimation) and starts from ``T_snow``.  Column enthalpy relative to
    ``T_freeze`` therefore changes by the boundary fluxes plus
    ``(snow_swe - swe_start) * c_ice * (T_snow - T_freeze)``: every snow-mass
    change (snowfall, melt, sublimation) carries ice enthalpy at the node's
    start-of-step temperature.  The atmosphere does not account for snowfall
    enthalpy, so the coupled budget is not closed by that term (~0.3 W/m2
    while snow falls; accepted 2026-10-03).

    Returns ``(T_snow_new (ncol,), T_soil_new (ncol, n_layers))``.
    """
    if surface_conductance is not None:
        # A semi-implicit surface (SimpleSEB) linearises the surface loss in
        # the skin temperature, which a snow surface caps at T_freeze; doing
        # that consistently needs the cap inside the solve.  Only the explicit
        # ground flux (the two-leaf canopy's Picard loop) is supported.
        raise ValueError(
            "solve_snow_soil_thermal does not support surface_conductance "
            "(semi-implicit surface); use the two-leaf canopy surface scheme.")
    T_soil_bare = solve_soil_thermal(
        T_soil, theta, grid, hydro_config, thermal_config, G_surface, dt,
        surface_conductance=surface_conductance, layer_source=layer_source,
        n_substeps=n_substeps)
    swe = jnp.maximum(snow_swe, 0.0)
    rho_s = thermal_config.snow_bulk_density_kg_m3
    R_snow = swe / rho_s / snow_thermal_conductivity(rho_s)       # [m2 K/W]
    C_snow = constants.c_pi * swe                                  # [J/m2/K]
    T_snow_cov, T_soil_cov = _solve_column(
        T_soil, theta, grid, hydro_config, thermal_config, G_surface, dt,
        surface_conductance, layer_source, n_substeps,
        snow=(T_snow, C_snow, R_snow))
    covered = swe > 0.0
    T_soil_new = jnp.where(covered[:, None], T_soil_cov, T_soil_bare)
    T_snow_new = jnp.where(covered, T_snow_cov, T_soil_bare[:, 0])
    return T_snow_new, T_soil_new


def _solve_column(T_soil, theta, grid, hydro_config, thermal_config,
                  G_surface, dt, surface_conductance, layer_source, n_substeps,
                  snow):
    """Backward-Euler soil column, optionally with one snow node on top.

    ``snow`` is ``None`` (soil only: returns ``(T_soil_new, None)``) or
    ``(T_snow, C_snow, R_snow)``: the surface flux (explicit; no semi-implicit
    surface conductance) and the heat capacity ``C_snow`` then sit on an extra top node
    coupled to soil node 0 by ``1/(R_snow + dz0/(2 k0))``; returns
    ``(T_snow_new, T_soil_new)``.
    """
    dz = grid.dz                  # (nlayers,)
    dz_if = grid.dz_interface     # (nlayers-1,)
    k_eff = compute_thermal_conductivity(theta, hydro_config, thermal_config)  # (ncol, nlayers)

    # Interface conductivity (harmonic mean for heat diffusion)
    k_half = 2.0 * k_eff[:, :-1] * k_eff[:, 1:] / (
        k_eff[:, :-1] + k_eff[:, 1:] + 1e-20
    )  # (ncol, nlayers-1)
    coeff = k_half / dz_if  # (ncol, nlayers-1)
    # Assemble the off-diagonals via ``jnp.pad`` — one Pad HLO op per
    # diagonal vs ``zeros + .at[].set`` (alloc + scatter).
    a = jnp.pad(-coeff, ((0, 0), (1, 0)))
    c = jnp.pad(-coeff, ((0, 0), (0, 1)))
    if snow is not None:
        T_snow, C_snow, R_snow = snow
        # Snow-surface node -> top soil node: whole-pack resistance in series
        # with the top soil half-layer [W/m2/K].
        g_snow = 1.0 / (R_snow + 0.5 * dz[0] / k_eff[:, 0])
        a = jnp.concatenate([jnp.zeros_like(g_snow)[:, None], -g_snow[:, None],
                             -coeff], axis=1)
        c = jnp.concatenate([-g_snow[:, None], -coeff,
                             jnp.zeros_like(g_snow)[:, None]], axis=1)
        T_s = T_snow

    # Backward Euler over ``n_substeps`` equal sub-steps; G_surface, Q_geo and
    # layer_source are rates held over the whole step.  The semi-implicit
    # surface term stays linearised about the step's ORIGINAL top temperature.
    dt_sub = dt / n_substeps
    if snow is None:
        T_top0 = T_soil[:, 0]
    T = T_soil
    for _ in range(n_substeps):
        # With freeze/thaw enabled the effective heat capacity is the apparent
        # heat capacity (sensible split + latent zero-curtain), evaluated at the
        # sub-step's start temperature; disabled (default) it is the
        # sensible-only C_eff (bit-identical for every prior caller).
        if thermal_config.enable_freeze_thaw:
            C_eff = compute_apparent_heat_capacity(
                T, theta, hydro_config, thermal_config)      # (ncol, nlayers)
        else:
            C_eff = compute_heat_capacity(theta, hydro_config, thermal_config)

        # C_eff * dz * (T_new - T_old) / dt = diffusion operator on T_new + source
        diag = C_eff * dz / dt_sub
        diag = diag.at[:, 1:].add(coeff)
        diag = diag.at[:, :-1].add(coeff)
        rhs = C_eff * dz * T / dt_sub

        if snow is None:
            # Top BC: ground heat flux
            rhs = rhs.at[:, 0].add(G_surface)

            # Semi-implicit (linearised) surface BC.  The surface flux into the top
            # layer is G(T_sfc_new) ~= G_surface - lambda*(T_new0 - T_top0) with
            # lambda = -dG/dT_sfc >= 0: lambda on the top diagonal, lambda*T_top0 on
            # the top RHS.  lambda=0 (surface_conductance is None) leaves the
            # explicit Neumann flux above untouched.
            if surface_conductance is not None:
                diag = diag.at[:, 0].add(surface_conductance)
                rhs = rhs.at[:, 0].add(surface_conductance * T_top0)

        # Bottom BC: geothermal heat flux (Neumann, positive into soil)
        rhs = rhs.at[:, -1].add(thermal_config.Q_geothermal)

        if layer_source is not None:
            rhs = rhs + layer_source

        if snow is None:
            T = thomas_solve(a, diag, c, rhs)
            continue

        # Snow node on top: the surface flux (and its linearised conductance)
        # acts on the snow surface; soil node 0 now exchanges with the snow.
        diag = diag.at[:, 0].add(g_snow)
        diag_s = C_snow / dt_sub + g_snow
        rhs_s = C_snow / dt_sub * T_s + G_surface
        sol = thomas_solve(a, jnp.concatenate([diag_s[:, None], diag], axis=1),
                           c, jnp.concatenate([rhs_s[:, None], rhs], axis=1))
        T_s, T = sol[:, 0], sol[:, 1:]
    if snow is None:
        return T, None
    return T_s, T
