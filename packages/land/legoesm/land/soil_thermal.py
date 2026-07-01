"""Multi-layer soil heat diffusion (Task 8D).

Solves the 1D heat equation:
    C_eff(z) · ∂T/∂t = ∂/∂z [k_eff(z) · ∂T/∂z]

Thermal properties depend on soil moisture via the Johansen (1975) method.
Discretized with backward Euler and solved via Thomas algorithm.

References
----------
- Johansen (1975): Thermal conductivity of soils. PhD thesis.
- de Vries (1963): Thermal properties of soils.
"""

from __future__ import annotations

from typing import NamedTuple

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
        },
        "params": {
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

    # Kersten number (Johansen 1975)
    if thermal_config.soil_texture == "sand":
        K_e = thermal_config.kersten_slope_coarse * jnp.log10(
            jnp.clip(Sr, thermal_config.kersten_sr_floor_coarse, None)
        ) + 1.0
    else:
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

    Returns
    -------
    T_new : jnp.ndarray
        Updated soil temperature [K], shape (ncol, n_layers).
    """
    dz = grid.dz                  # (nlayers,)
    dz_if = grid.dz_interface     # (nlayers-1,)

    # Compute thermal properties
    C_eff = compute_heat_capacity(theta, hydro_config, thermal_config)    # (ncol, nlayers)
    k_eff = compute_thermal_conductivity(theta, hydro_config, thermal_config)  # (ncol, nlayers)

    # Interface conductivity (harmonic mean for heat diffusion)
    k_half = 2.0 * k_eff[:, :-1] * k_eff[:, 1:] / (
        k_eff[:, :-1] + k_eff[:, 1:] + 1e-20
    )  # (ncol, nlayers-1)

    # Diffusion coefficient at interfaces
    coeff = k_half / dz_if  # (ncol, nlayers-1)

    # Build tridiagonal system for backward Euler:
    # C_eff * dz * (T_new - T_old) / dt = diffusion operator on T_new + source

    # Diagonal
    diag = C_eff * dz / dt
    diag = diag.at[:, 1:].add(coeff)
    diag = diag.at[:, :-1].add(coeff)

    # Sub-diagonal (lower)
    sub = -coeff  # (ncol, nlayers-1)

    # Super-diagonal (upper)
    sup = -coeff  # (ncol, nlayers-1)

    # RHS
    rhs = C_eff * dz * T_soil / dt

    # Top BC: ground heat flux
    rhs = rhs.at[:, 0].add(G_surface)

    # Semi-implicit (linearised) surface BC.  The surface flux into the top
    # layer is G(T_sfc_new) ~= G_surface + dG/dT_sfc * (T_new0 - T_old0)
    # = G_surface - lambda*(T_new0 - T_old0) with lambda = -dG/dT_sfc >= 0.
    # Moving the implicit -lambda*T_new0 term to the LHS adds lambda to the top
    # diagonal and lambda*T_old0 to the top RHS.  lambda=0 (surface_conductance
    # is None) leaves the explicit Neumann flux above untouched.
    if surface_conductance is not None:
        diag = diag.at[:, 0].add(surface_conductance)
        rhs = rhs.at[:, 0].add(surface_conductance * T_soil[:, 0])

    # Bottom BC: geothermal heat flux (Neumann, positive into soil)
    rhs = rhs.at[:, -1].add(thermal_config.Q_geothermal)

    # Assemble full arrays via ``jnp.pad`` — one Pad HLO op per
    # diagonal vs ``zeros + .at[].set`` (alloc + scatter).
    a = jnp.pad(sub, ((0, 0), (1, 0)))
    c = jnp.pad(sup, ((0, 0), (0, 1)))

    T_new = thomas_solve(a, diag, c, rhs)
    return T_new
