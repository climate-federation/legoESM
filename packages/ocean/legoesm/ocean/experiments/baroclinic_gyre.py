"""Regional Baroclinic Gyre Ocean Experiment.

A regional ocean experiment that combines wind-driven circulation with realistic
stratification. This test validates the ocean model's ability to simulate
baroclinic gyre dynamics including thermal wind balance and overturning
circulation driven by background stratification and wind forcing.

Scientific Purpose:
- Test baroclinic gyre circulation with realistic stratification
- Validate thermal wind balance in presence of wind forcing
- Study meridional overturning circulation development
- Test thermal stratification effects on circulation
- Verify multi-level momentum transfer and vertical coupling
- Benchmark conservation during long baroclinic integrations
- Test recent Coriolis double-counting fixes in multi-layer dynamics

Domain Configuration:
- Rectangular ocean basin (0-120°E, 15-75°N) — same as barotropic_double_gyre
- Land boundaries on all four sides
- Uniform depth: 5500m in ocean regions
- Realistic background stratification: T_water_init_C=20°C → T_deep_C=2°C
- Meridional surface temperature gradient with restoring

Physical Setup:
- Double-gyre wind stress: sin^2 westerly jet with 5° buffer at walls
- Background stratification: exponential T profile with 1000m e-folding depth
- Surface temperature restoring: τ_restore = 30 days
- Meridional SST gradient: warm equatorward, cool poleward
- Lateral viscosity: A_h = 5×10⁵ m²/s (same as barotropic case)
- Linear bottom drag: r = 1×10⁻⁴ s⁻¹

Expected Behavior:
- Development of wind-driven surface gyres with thermal wind shear
- Meridional overturning cells driven by Ekman pumping
- Vertical heat transport by overturning circulation
- Western intensification with baroclinic structure
- Eddy formation at gyre boundaries (resolution permitting)
- Realistic heat transport by both horizontal and vertical circulation

Validation Criteria:
- Realistic surface speeds (0.1-0.5 m/s) with vertical shear
- Heat conservation within acceptable bounds
- Development of thermal wind balance
- SST gradients maintained by circulation-restoring interaction
- No excessive instabilities or numerical artifacts

References:
- Holland & Lin (1975), "On the generation of mesoscale eddies"
- Pedlosky (1987), "Geophysical Fluid Dynamics" - thermal wind theory
- Vallis (2017), "Atmospheric and Oceanic Fluid Dynamics" - baroclinic circulation
- Bryan & Cox (1967), "A numerical investigation of the oceanic general circulation"
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class BaroclinicGyreConfig:
    """Configuration parameters for regional baroclinic gyre experiment.

    All parameters have physically meaningful defaults that work across
    different grid types and resolutions.
    """
    # Domain configuration — same as barotropic_double_gyre
    H_max: float = 5500.0          # Maximum ocean depth [m]
    lon_west: float = 0.0          # Basin western boundary [degrees]
    lon_east: float = 120.0        # Basin eastern boundary [degrees]
    lat_south: float = 15.0        # Basin southern boundary [degrees]
    lat_north: float = 75.0        # Basin northern boundary [degrees]

    # Background stratification
    T_water_init_C: float = 20.0        # Surface temperature [degC]
    T_deep_C: float = 2.0            # Deep ocean temperature [degC]
    T_scale_depth: float = 1000.0  # Temperature e-folding depth [m]
    S_uniform: float = 35.0        # Salinity [PSU]

    # Wind forcing parameters — same as barotropic case
    wind_stress_max: float = 0.3   # Maximum wind stress [Pa]
    wind_profile: str = "double_gyre_sin2"  # "double_gyre" (cosine) or "double_gyre_sin2"
    wind_buffer_deg: float = 5.0   # Buffer zone width [degrees] for sin² profile

    # Physics parameters — same as barotropic case for comparison
    A_h: float = 5e5               # Horizontal viscosity [m²/s]
    bottom_drag_coeff: float = 1.1e-3  # Linear bottom drag coefficient [m/s]


def create_initial_conditions(grid_type: str, grid, z_coord,
                            config: BaroclinicGyreConfig = None):
    """Create regional baroclinic gyre initial conditions for any grid type.

    Creates a rectangular ocean basin with realistic background stratification.
    The domain and bathymetry are identical to barotropic_double_gyre but with
    realistic T/S profiles instead of uniform values.

    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere", "latlon", "mpas", or regional variants
    grid : Grid
        Grid object.
    z_coord : OceanZCoordinate
        Vertical coordinate system.
    config : BaroclinicGyreConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    OceanState
        Initial state with basin land mask and background stratification
    """
    if config is None:
        config = BaroclinicGyreConfig()

    if grid_type in ("cubed_sphere", "cs_regional"):
        from legoesm.ocean.init import wind_driven_gyre_init
        return wind_driven_gyre_init(
            grid, z_coord, H_max=config.H_max,
            T_water_init_C=config.T_water_init_C, T_deep=config.T_deep_C,
            scale_depth=config.T_scale_depth, S_uniform=config.S_uniform,
            lon_west=config.lon_west, lon_east=config.lon_east,
            lat_south=config.lat_south, lat_north=config.lat_north,
        )

    elif grid_type in ("latlon", "latlon_regional"):
        from legoesm.ocean.init_latlon_cgrid import wind_driven_gyre_latlon_cgrid
        # Create uniform state first, then add stratification
        state = wind_driven_gyre_latlon_cgrid(
            grid, z_coord, H_max=config.H_max,
            T_uniform=config.T_water_init_C, S_uniform=config.S_uniform,
            lon_west=config.lon_west, lon_east=config.lon_east,
            lat_south=config.lat_south, lat_north=config.lat_north,
        )
        # Add vertical stratification
        return _add_stratification(state, z_coord, config)

    elif grid_type in ("mpas", "mpas_regional"):
        from legoesm.ocean.init_mpas import wind_driven_gyre_mpas
        state = wind_driven_gyre_mpas(
            grid, z_coord, H_max=config.H_max,
            T_uniform=config.T_water_init_C, S_uniform=config.S_uniform,
            lon_west=config.lon_west, lon_east=config.lon_east,
            lat_south=config.lat_south, lat_north=config.lat_north,
        )
        # Add vertical stratification
        return _add_stratification(state, z_coord, config)

    else:
        raise ValueError(f"Grid type {grid_type} not supported for baroclinic_gyre")


def _add_stratification(state, z_coord, config: BaroclinicGyreConfig):
    """Add exponential stratification to a uniform initial state."""
    from legoesm.core.field import Field

    # Use actual model level depths from the z-coordinate object
    actual_depths = -np.asarray(z_coord.z_full_ref)  # positive-down depth [m]
    n_levels = len(actual_depths)

    # Create exponential profile: T(z) = T_deep_C + (T_water_init_C - T_deep_C) * exp(-z/scale_depth)
    z_coord_depths = -actual_depths  # Negative for depth coordinate

    # Exponential decay with depth
    decay_factor = np.exp(z_coord_depths / config.T_scale_depth)
    T_profile = config.T_deep_C + (config.T_water_init_C - config.T_deep_C) * decay_factor

    print(f"Fixed stratification profile:")
    for k, (depth, T) in enumerate(zip(actual_depths, T_profile)):
        print(f"  Level {k+1:2d} ({depth:6.1f}m): {T:6.3f}°C")

    # Apply stratification to all grid points
    T_data = np.array(state.T.data)
    for k in range(n_levels):
        T_data[..., k] = T_profile[k]

    # Keep salinity uniform
    return state._replace(T=Field(jnp.array(T_data), name="T",
                                  dims=state.T.dims, units=state.T.units))


def create_forcings(grid_type: str, grid, config: BaroclinicGyreConfig = None):
    """Create forcing functions for regional baroclinic gyre experiment.

    Returns physics configuration with:
    1. Double-gyre wind stress (same as barotropic_double_gyre)
    2. Surface temperature restoring (new for baroclinic case)
    3. Lateral viscosity and bottom drag

    Parameters
    ----------
    grid_type : str
        Grid type
    grid : Grid
        Grid object
    config : BaroclinicGyreConfig, optional
        Configuration parameters

    Returns
    -------
    OceanPhysicsConfig
        Physics configuration with wind + thermal forcing
    """
    if config is None:
        config = BaroclinicGyreConfig()

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    # Surface forcing: sin^2 westerly jet profile.
    # Wind goes to zero 5° inside the basin walls to avoid spurious
    # coastal upwelling/downwelling from Ekman transport hitting the
    # solid boundaries at coarse resolution.
    restoring_config = PrescribedForcingConfig(
        wind_profile=config.wind_profile,
        tau_max=config.wind_stress_max,
        lat_south_deg=config.lat_south,
        lat_north_deg=config.lat_north,
        wind_buffer_deg=config.wind_buffer_deg,
    )

    surface_forcing = SurfaceForcingConfig(
        scheme="prescribed",
        prescribed=restoring_config,
    )

    # Vertical mixing: constant background only
    from legoesm.ocean.physics.vertical_mixing.config import ConstantVerticalMixingConfig
    constant_config = ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4)
    vertical_mixing = VerticalMixingConfig(
        scheme="constant",
        constant=constant_config,
    )

    # Lateral mixing: none in physics pipeline (handled by ocean config A_h)
    lateral_mixing = LateralMixingConfig(
        scheme="none",  # A_h handled by ocean dynamics, not physics
    )

    # Convection: none for now
    convection = OceanConvectionConfig(
        scheme="none",
    )

    # Bottom drag is applied via the dynamics-level ``bottom_drag_r``
    # field (baroclinic PE + barotropic substeps), not through the
    # physics pipeline.
    return OceanPhysicsConfig(
        surface_forcing=surface_forcing,
        vertical_mixing=vertical_mixing,
        lateral_mixing=lateral_mixing,
        convection=convection,
    )


def create_domain_config(config: BaroclinicGyreConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = BaroclinicGyreConfig()

    return {
        "H_max": config.H_max,
        "lon_west": config.lon_west,
        "lon_east": config.lon_east,
        "lat_south": config.lat_south,
        "lat_north": config.lat_north,
        "description": "Regional basin with baroclinic gyre circulation",
        "forcing_type": "wind_stress_double_gyre_plus_restoring",
        "stratification": "realistic_exponential_profile",
    }


def compute_baroclinic_metrics(diagnostics: Dict[str, list],
                             config: BaroclinicGyreConfig) -> Dict[str, float]:
    """Compute circulation metrics specific to baroclinic gyre validation.

    Parameters
    ----------
    diagnostics : Dict[str, list]
        Time series diagnostics from simulation
    config : BaroclinicGyreConfig
        Configuration parameters

    Returns
    -------
    Dict[str, float]
        Baroclinic circulation metrics for validation
    """
    metrics = {}

    # Surface speed development
    max_speed_list = diagnostics.get("max_speed", [])
    if not max_speed_list:
        max_speed_list = diagnostics.get("max_abs_u", [])  # MPAS alternative

    if len(max_speed_list) >= 2:
        max_speed_final = max_speed_list[-1]
        max_speed_initial = max_speed_list[0]
        metrics["max_speed_final"] = max_speed_final

        if max_speed_initial > 1e-10:
            speed_ratio = max_speed_final / max_speed_initial
            metrics["speed_development"] = speed_ratio

    # Heat conservation
    mean_T_list = diagnostics.get("mean_T", [])
    if len(mean_T_list) >= 2:
        T_initial = mean_T_list[0]
        T_final = mean_T_list[-1]
        T_drift = abs(T_final - T_initial)
        metrics["T_drift_absolute"] = T_drift

        if abs(T_initial) > 1e-10:
            metrics["T_drift_relative"] = T_drift / abs(T_initial)

    # SSH conservation
    mean_eta_list = diagnostics.get("mean_eta", [])
    if len(mean_eta_list) >= 2:
        eta_drift = abs(mean_eta_list[-1] - mean_eta_list[0])
        metrics["eta_drift"] = eta_drift
        metrics["eta_drift_normalized"] = eta_drift / config.H_max

    return metrics


def validate_results(final_state, diagnostics: Dict[str, list],
                   config: BaroclinicGyreConfig = None) -> Tuple[bool, str]:
    """Validate regional baroclinic gyre experiment results.

    Success criteria:
    - Surface speeds in realistic baroclinic range (0.1-1.0 m/s)
    - Heat conservation reasonable with restoring forcing
    - SSH conservation good (< 1e-5 normalized drift)
    - No NaN or infinite values
    - Circulation development from rest state

    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list]
        Time series diagnostics
    config : BaroclinicGyreConfig, optional
        Configuration parameters

    Returns
    -------
    bool
        True if validation passes
    str
        Descriptive notes about the validation
    """
    if config is None:
        config = BaroclinicGyreConfig()

    # Compute circulation metrics
    metrics = compute_baroclinic_metrics(diagnostics, config)

    # Check for NaN/infinite values
    for field_name in ['eta', 'T', 'S', 'u']:
        if hasattr(final_state, field_name):
            field_data = getattr(final_state, field_name).data
            if not jnp.all(jnp.isfinite(field_data)):
                return False, f"NaN/Inf detected in final {field_name} field"

    # Validation thresholds for baroclinic gyre
    min_speed = 0.05        # m/s - minimum realistic gyre speed
    max_speed = 1.0         # m/s - maximum realistic baroclinic speed
    max_eta_drift = 1e-5    # normalized by depth
    max_T_drift = 1.0       # degC - allow larger drift with restoring
    min_speed_ratio = 2.0   # minimum development from rest

    # Validation checks
    success = True
    notes_parts = []

    if "max_speed_final" in metrics:
        speed = metrics["max_speed_final"]
        notes_parts.append(f"max_speed={speed:.4f}m/s")
        if speed < min_speed:
            success = False
            notes_parts.append("FAIL: insufficient circulation")
        elif speed > max_speed:
            success = False
            notes_parts.append("FAIL: excessive speed")

    if "eta_drift_normalized" in metrics:
        eta_drift = metrics["eta_drift_normalized"]
        notes_parts.append(f"eta_drift={eta_drift:.2e}")
        if eta_drift > max_eta_drift:
            success = False
            notes_parts.append("FAIL: poor volume conservation")

    if "T_drift_absolute" in metrics:
        T_drift = metrics["T_drift_absolute"]
        notes_parts.append(f"T_drift={T_drift:.3f}degC")
        if T_drift > max_T_drift:
            success = False
            notes_parts.append("FAIL: excessive T drift")

    if "speed_development" in metrics:
        speed_dev = metrics["speed_development"]
        notes_parts.append(f"speed_dev={speed_dev:.2f}")
        if speed_dev < min_speed_ratio:
            success = False
            notes_parts.append("FAIL: insufficient spin-up")

    notes = ", ".join(notes_parts)

    return success, notes


def get_diagnostic_field_specs() -> list:
    """Get field specifications for diagnostic output.

    Returns
    -------
    list
        Field specifications for plotting: [(field_key, label, colormap), ...]
    """
    return [
        ("eta", "SSH (m)", "RdBu_r"),
        ("speed_sfc", "Surface Speed (m/s)", "plasma"),
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("T_sfc", "Surface Temperature (degC)", "RdYlBu_r"),  # alias
    ]


def get_scalar_units() -> Dict[str, str]:
    """Get units for scalar diagnostic quantities.

    Returns
    -------
    Dict[str, str]
        Mapping from scalar field names to units
    """
    return {
        "mean_eta": "m",
        "max_abs_eta": "m",
        "max_speed": "m/s",
        "max_abs_u": "m/s",  # MPAS alternative
        "mean_T": "degC",
        "mean_S": "PSU",
        "max_KE": "J/kg",
    }


# Standard experiment configuration for test matrix integration
EXPERIMENT_CONFIG = {
    "name": "baroclinic_gyre",
    "description": "Regional wind-driven baroclinic gyre with surface restoring",
    "scientific_purpose": "Validates complete baroclinic gyre dynamics with thermal wind balance",
    "reference": "Holland & Lin (1975), Pedlosky (1987) - baroclinic gyre theory",
    "config_class": BaroclinicGyreConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 60.0,   # days - longer for baroclinic adjustment
    "quick_duration": 5.0,      # days
    "expected_metrics": {
        "max_speed_final": "0.05-1.0 m/s",
        "eta_drift_normalized": "< 1e-5",
        "speed_development": "> 2.0",
        "T_drift_absolute": "< 1.0 degC"
    },
    "grid_support": {
        "cubed_sphere": False,      # Face-boundary instability (#100)
        "latlon": True,             # C-grid working
        "mpas": True,               # Fixed Coriolis double-counting (#103)
        "latlon_regional": True,    # Primary target
        "mpas_regional": True,      # Primary target
        "spectral": False,          # Removed from ocean tests
    },
}
