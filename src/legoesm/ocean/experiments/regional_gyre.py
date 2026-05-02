"""Regional Gyre Ocean Experiment.

Base experiment for wind-driven gyre circulation in a rectangular ocean basin.
This serves as the foundation for barotropic gyre, baroclinic gyre, and
tracer transport experiments. It verifies momentum transfer, circulation
patterns, and numerical properties under wind stress forcing.

Scientific Purpose:
- Validate wind stress momentum transfer to the ocean
- Test development of wind-driven circulation patterns
- Verify Sverdrup balance in the interior ocean
- Benchmark western boundary current formation
- Test numerical stability under long integrations
- Validate conservation properties during gyre circulation

Domain Configuration:
- Rectangular ocean basin (default: 0-60°E, 15-75°N)
- Land on all four sides (meridional + zonal boundaries)
- Uniform depth: 5500m in ocean regions
- Rest state stratification as background
- Basin-relative double-gyre wind stress forcing

Physical Setup:
- Double-gyre wind stress pattern: alternating cyclonic/anticyclonic
- Typical subtropical/subpolar gyre configuration
- Western intensification expected due to β-effect
- Background stratification from rest_state experiment

Expected Behavior:
- Development of coherent circulation gyres
- Western intensification of boundary currents
- Realistic maximum surface speeds (0.1-0.3 m/s)
- Balanced circulation patterns without spurious oscillations
- Conservation of momentum and energy during spin-up

Validation Criteria:
- Maximum surface speed within realistic range (0.05-0.5 m/s)
- Mean SSH drift minimal compared to ocean depth
- No excessive circulation or numerical instabilities
- Symmetric gyre patterns without grid artifacts

References:
- Munk (1950), "On the wind-driven ocean circulation"
- Stommel (1948), "The westward intensification of wind-driven ocean currents"
- Holland & Lin (1975), "On the generation of mesoscale eddies and their contribution
  to the oceanic general circulation"
- Standard ocean model validation studies
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from legoesm.constants import g


@dataclass
class RegionalGyreConfig:
    """Configuration parameters for wind-driven gyre experiment.

    All parameters have physically meaningful defaults that work across
    different grid types and resolutions.
    """
    # Background state
    T_surface: float = 20.0        # Surface temperature [degC]
    T_deep: float = 2.0            # Deep ocean temperature [degC]
    scale_depth: float = 1000.0    # Temperature e-folding depth [m]
    S_uniform: float = 35.0        # Salinity [PSU]

    # Domain configuration — rectangular basin
    H_max: float = 5500.0          # Maximum ocean depth [m]
    lon_west: float = 0.0          # Basin western boundary [degrees]
    lon_east: float = 120.0        # Basin eastern boundary [degrees]
    lat_south: float = 15.0        # Basin southern boundary [degrees]
    lat_north: float = 75.0        # Basin northern boundary [degrees]

    # Wind forcing parameters
    wind_stress_max: float = 0.1   # Maximum wind stress [Pa]
    wind_profile: str = "single_gyre"  # "single_gyre", "double_gyre", "double_gyre_sin2"
    wind_buffer_deg: float = 0.0   # Buffer zone at basin edges [degrees]

    # Physics
    A_h: float = 5e5               # Horizontal viscosity [m²/s]
    bottom_drag_coeff: float = 1.1e-3  # Linear bottom drag [m/s]


def create_initial_conditions(grid_type: str, grid, z_coord, 
                            config: RegionalGyreConfig = None):
    """Create wind-driven gyre initial conditions for any supported grid type.

    Creates a rectangular ocean basin with rest-state stratification.
    Wind forcing is applied during model integration via the prescribed
    surface forcing physics pipeline.

    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere" or "latlon".
    grid : Grid
        Grid object.
    z_coord : OceanZCoordinate
        Vertical coordinate system.
    config : RegionalGyreConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    OceanState
        Initial state with basin land mask for double-gyre experiment.
    """
    if config is None:
        config = RegionalGyreConfig()
    
    if grid_type in ("cubed_sphere", "cs_regional"):
        from legoesm.ocean.init import wind_driven_gyre_init
        return wind_driven_gyre_init(
            grid, z_coord, H_max=config.H_max,
            lon_west=config.lon_west, lon_east=config.lon_east,
            lat_south=config.lat_south, lat_north=config.lat_north,
        )

    elif grid_type in ("latlon", "latlon_regional"):
        from legoesm.ocean.init_latlon_cgrid import wind_driven_gyre_latlon_cgrid
        return wind_driven_gyre_latlon_cgrid(
            grid, z_coord, H_max=config.H_max,
            lon_west=config.lon_west, lon_east=config.lon_east,
            lat_south=config.lat_south, lat_north=config.lat_north,
        )

    elif grid_type in ("mpas", "mpas_regional"):
        from legoesm.ocean.init_mpas import wind_driven_gyre_mpas
        return wind_driven_gyre_mpas(
            grid, z_coord, H_max=config.H_max,
            lon_west=config.lon_west, lon_east=config.lon_east,
            lat_south=config.lat_south, lat_north=config.lat_north,
        )

    else:
        raise ValueError(f"Grid type {grid_type} not supported for regional_gyre")


def create_forcings(grid_type: str, grid, config: RegionalGyreConfig = None):
    """Create OceanPhysicsConfig with prescribed gyre wind forcing.

    Returns a physics config with wind stress, linear bottom drag, and
    no vertical/lateral mixing (A_h is handled by the ocean dynamics config).

    Parameters
    ----------
    grid_type : str
        Grid type
    grid : Grid
        Grid object
    config : RegionalGyreConfig, optional
        Configuration parameters

    Returns
    -------
    OceanPhysicsConfig
        Physics configuration for gyre experiment
    """
    if config is None:
        config = RegionalGyreConfig()

    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile=config.wind_profile,
                tau_max=config.wind_stress_max,
                lat_south_deg=config.lat_south,
                lat_north_deg=config.lat_north,
                wind_buffer_deg=config.wind_buffer_deg,
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )


def create_domain_config(config: RegionalGyreConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = RegionalGyreConfig()
        
    return {
        "H_max": config.H_max,
        "lon_west": config.lon_west,
        "lon_east": config.lon_east,
        "lat_south": config.lat_south,
        "lat_north": config.lat_north,
        "description": "Rectangular basin with wind-driven double-gyre circulation",
        "forcing_type": "wind_stress_gyre",
    }


def compute_circulation_metrics(diagnostics: Dict[str, list], 
                              config: RegionalGyreConfig) -> Dict[str, float]:
    """Compute circulation metrics specific to wind-driven gyre validation.

    Parameters
    ----------
    diagnostics : Dict[str, list]
        Time series diagnostics from simulation
    config : RegionalGyreConfig
        Configuration parameters

    Returns
    -------
    Dict[str, float]
        Circulation metrics for validation
    """
    metrics = {}

    # Maximum surface speed - should reach realistic gyre speeds
    max_speed_list = diagnostics.get("max_speed", [])

    if len(max_speed_list) >= 2:
        max_speed_final = max_speed_list[-1]
        metrics["max_speed_final"] = max_speed_final

        # Speed development rate
        max_speed_initial = max_speed_list[0]
        if max_speed_initial > 0:
            speed_ratio = max_speed_final / max_speed_initial
            metrics["speed_development"] = speed_ratio

    # Mean SSH drift - should be minimal for conservation
    mean_eta_list = diagnostics.get("mean_eta", [])
    if len(mean_eta_list) >= 2:
        # Use absolute drift normalized by ocean depth
        eta_drift = abs(mean_eta_list[-1] - mean_eta_list[0]) / config.H_max
        metrics["eta_drift_normalized"] = eta_drift

    return metrics


def validate_results(final_state, diagnostics: Dict[str, list], 
                   config: RegionalGyreConfig = None) -> Tuple[bool, str]:
    """Validate wind-driven gyre experiment results.

    Success criteria:
    - Maximum surface speed in realistic range (0.05 - 0.5 m/s)
    - Mean SSH drift normalized by depth < 1e-5 (good conservation)
    - No NaN or infinite values
    - Speed development shows circulation spin-up

    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list]
        Time series diagnostics
    config : RegionalGyreConfig, optional
        Configuration parameters

    Returns
    -------
    bool
        True if validation passes
    str
        Descriptive notes about the validation
    """
    if config is None:
        config = RegionalGyreConfig()
    
    # Compute circulation metrics
    metrics = compute_circulation_metrics(diagnostics, config)

    # Check for NaN/infinite values in final state
    if hasattr(final_state, 'eta') and not jnp.all(jnp.isfinite(final_state.eta.data)):
        return False, "NaN/Inf detected in final eta field"
    if hasattr(final_state, 'T') and not jnp.all(jnp.isfinite(final_state.T.data)):
        return False, "NaN/Inf detected in final temperature field"
    if hasattr(final_state, 'u') and not jnp.all(jnp.isfinite(final_state.u.data)):
        return False, "NaN/Inf detected in final u velocity field"

    # Validation thresholds
    min_speed = 0.05       # m/s - minimum realistic gyre speed
    max_speed = 0.5        # m/s - maximum realistic surface speed
    max_eta_drift = 1e-5   # normalized by depth - conservation threshold
    min_speed_ratio = 1.5  # minimum speed development for spin-up

    # Check validation criteria
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
            notes_parts.append("FAIL: poor conservation")

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
        "mean_S": "PSU"
    }


# Standard experiment configuration for test matrix integration
EXPERIMENT_CONFIG = {
    "name": "regional_gyre",
    "description": "Regional wind-driven gyre circulation in rectangular basin",
    "scientific_purpose": "Validates wind stress forcing and circulation patterns",
    "reference": "Munk (1950), Stommel (1948) - classical wind-driven circulation theory",
    "config_class": RegionalGyreConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 30.0,   # days
    "quick_duration": 2.0,      # days
    "expected_metrics": {
        "max_speed_final": "0.05-0.5 m/s",
        "eta_drift_normalized": "< 1e-5",
        "speed_development": "> 1.5"
    },
    "grid_support": {
        "cubed_sphere": True,
        "latlon": True,
        "mpas": True,
        "mpas_regional": True,
        "latlon_regional": True,
        "cs_regional": True,
        "spectral": False,
    },
}