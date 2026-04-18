"""Stommel Gyre with Passive Tracer Ocean Experiment.

A combined circulation and tracer transport test case based on Hecht et al. (2000)
that studies passive tracer advection in a wind-driven gyre circulation. This 
experiment validates both the circulation dynamics and the numerical properties
of tracer transport schemes.

Scientific Purpose:
- Validate passive tracer transport in sheared flows
- Test tracer conservation properties (integral, min/max preservation)
- Study numerical diffusion and dispersion in gyre circulation
- Benchmark tracer advection schemes in western boundary currents
- Validate long-term tracer transport and mixing
- Test tracer behavior in realistic circulation patterns

Domain Configuration:
- Global ocean with wind-driven gyre circulation
- Standard ocean depth: 5500m with 10 vertical levels
- Land at high latitudes for gyre containment
- Tracer blob initialization in gyre interior

Physical Setup:
- Wind-driven double-gyre circulation (inherited from regional_gyre experiment)
- Passive salinity tracer blob superimposed on background circulation
- Tracer blob: Gaussian shape, centered at (30°N, 30°W), 10° width
- Background salinity: 35 PSU, tracer amplitude: ±2 PSU
- Surface-intensified tracer for clear advection signal

Expected Behavior:
- Tracer blob advection by gyre circulation
- Stretching and deformation in sheared flow
- Transport through western boundary current
- Gradual mixing and filament formation
- Long-term circulation and tracer evolution

Validation Criteria:
- Tracer integral conservation within tight bounds
- Minimal spurious extrema (overshoot/undershoot)
- Realistic tracer transport patterns
- No excessive numerical diffusion or oscillations
- Conservation of tracer bounds (monotonicity)

References:
- Hecht et al. (2000), "A comparison of tracer advection in stationary and time-varying 
  ocean circulations", Ocean Modelling 2, 1-15. DOI: 10.1016/S1463-5003(00)00004-4
- Standard ocean model validation for tracer transport
- NEMO and MOM tracer transport benchmarking studies
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from legoesm.constants import g
from legoesm.core.field import Field


@dataclass
class StommelGyreTracerConfig:
    """Configuration parameters for Stommel gyre tracer experiment.
    
    Combines wind-driven gyre circulation with passive tracer transport.
    """
    # Background circulation (inherits from regional_gyre)
    T_surface: float = 20.0        # Surface temperature [°C]
    T_deep: float = 2.0            # Deep ocean temperature [°C] 
    scale_depth: float = 1000.0    # Temperature e-folding depth [m]
    H_max: float = 5500.0          # Maximum ocean depth [m]
    land_lat_threshold: float = 80.0  # Latitude threshold for land [degrees]
    
    # Tracer blob parameters
    S_background: float = 35.0     # Background salinity [PSU]
    S_amplitude: float = 2.0       # Tracer perturbation amplitude [PSU]
    blob_center_lat: float = 30.0  # Blob center latitude [degrees]
    blob_center_lon: float = -30.0 # Blob center longitude [degrees]
    blob_width_deg: float = 10.0   # Blob width [degrees]
    
    # Surface/subsurface structure
    surface_tracer_only: bool = True  # Apply tracer only at surface
    
    # Validation thresholds
    max_integral_drift: float = 1e-3  # Maximum tracer integral drift
    max_overshoot: float = 0.1        # Maximum spurious overshoot [PSU]
    max_undershoot: float = 0.1       # Maximum spurious undershoot [PSU]


def create_initial_conditions(grid_type: str, grid, z_coord, 
                            config: StommelGyreTracerConfig = None):
    """Create Stommel gyre tracer initial conditions for any grid type.
    
    Sets up a wind-driven gyre circulation with a passive salinity tracer
    blob that will be advected by the flow.
    
    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere", "latlon", "mpas" (spectral not supported)
    grid : Grid
        Grid object (type depends on grid_type)
    z_coord : OceanZCoordinate
        Vertical coordinate system
    config : StommelGyreTracerConfig, optional
        Configuration parameters. Uses defaults if None.
        
    Returns
    -------
    OceanState
        Initial state with gyre circulation + tracer blob
        
    Notes
    -----
    - Spectral grid not supported (complex tracer implementation)
    - Uses wind-gyre initialization for circulation
    - Adds Gaussian salinity blob as passive tracer
    - Tracer applied primarily at surface for clear signal
    """
    if config is None:
        config = StommelGyreTracerConfig()
    
    if grid_type == "spectral":
        raise NotImplementedError(
            "Stommel gyre tracer not implemented for spectral grid")
    
    # First create rest state background
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        state = rest_state_ocean(
            grid, z_coord, 
            T_surface=config.T_surface,
            T_deep=config.T_deep,
            S_uniform=config.S_background,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )
        
    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=config.T_surface,
            T_deep=config.T_deep,
            S_uniform=config.S_background,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )
        
    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        state = rest_state_mpas_ocean(
            grid, z_coord,
            T_surface=config.T_surface,
            T_deep=config.T_deep, 
            S_uniform=config.S_background,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )
        
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")
    
    # Add wind-driven gyre circulation
    state = _add_wind_gyre_circulation(state, grid_type, grid, z_coord, config)
    
    # Add passive tracer blob
    return _add_tracer_blob(state, grid_type, grid, z_coord, config)


def _add_wind_gyre_circulation(state, grid_type: str, grid, z_coord,
                             config: StommelGyreTracerConfig):
    """Add wind-driven gyre circulation to the state.
    
    Uses the same logic as regional_gyre experiment to establish circulation.
    """
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import wind_driven_gyre_init
        return wind_driven_gyre_init(grid, z_coord)
        
    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import wind_driven_gyre_latlon_cgrid
        return wind_driven_gyre_latlon_cgrid(grid, z_coord)
        
    elif grid_type == "mpas":
        # MPAS doesn't have dedicated gyre init; use rest state with wind forcing
        return state
    
    return state


def _add_tracer_blob(state, grid_type: str, grid, z_coord,
                   config: StommelGyreTracerConfig):
    """Add Gaussian tracer blob to the salinity field."""
    
    # Get coordinates
    if grid_type == "mpas":
        lat = np.asarray(grid.latCell, dtype=np.float64)
        lon = np.asarray(grid.lonCell, dtype=np.float64)
    elif grid_type == "latlon":
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='xy')
    else:  # cubed_sphere
        lat = np.asarray(grid.lat, dtype=np.float64)
        lon = np.asarray(grid.lon, dtype=np.float64)
    
    # Convert blob parameters to radians
    lat_c = np.radians(config.blob_center_lat)
    lon_c = np.radians(config.blob_center_lon)
    sigma = np.radians(config.blob_width_deg)
    S_bg = config.S_background
    S_amp = config.S_amplitude
    
    # Compute Gaussian blob on sphere
    # Use spherical distance for proper shape
    # Periodic longitude wrapping to [-π, π] for correct distance
    dlon = np.mod(lon - lon_c + np.pi, 2.0 * np.pi) - np.pi
    r2 = (lat - lat_c)**2 + (np.cos(lat_c) * dlon)**2
    S_blob = S_bg + S_amp * np.exp(-r2 / (2.0 * sigma**2))
    
    # Apply to salinity field
    S_data = np.array(state.S.data, dtype=np.float64, copy=True)
    if hasattr(state, 'land_mask'):
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
    else:
        mask = 1.0
    
    if config.surface_tracer_only:
        # Surface-intensified tracer
        S_data[..., 0] = S_blob * mask
    else:
        # Apply to all levels
        nlev = S_data.shape[-1]
        for k in range(nlev):
            S_data[..., k] = S_blob * mask
    
    return state._replace(S=Field(jnp.array(S_data), name="S",
                                dims=state.S.dims, units="PSU"))


def create_forcings(grid_type: str, grid, config: StommelGyreTracerConfig = None):
    """Create forcing functions for Stommel gyre tracer experiment.
    
    The experiment uses wind-driven gyre circulation with passive tracer
    transport. Wind forcing is handled by the model's surface forcing
    infrastructure.
    
    Returns
    -------
    None
        No explicit forcings (wind handled by model infrastructure)
    """
    return None


def create_domain_config(config: StommelGyreTracerConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.
    
    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = StommelGyreTracerConfig()
        
    return {
        "H_max": config.H_max,
        "land_lat_threshold": config.land_lat_threshold,
        "description": "Wind-driven gyre with passive tracer transport",
        "reference": "Hecht et al. (2000) tracer transport benchmark",
        "tracer_type": "passive_salinity_blob",
    }


def compute_tracer_conservation_metrics(state, grid_type: str, grid,
                                      initial_values: Dict[str, float],
                                      config: StommelGyreTracerConfig
                                      ) -> Dict[str, float]:
    """Compute tracer conservation metrics for current state.
    
    Parameters
    ----------
    state : OceanState
        Current ocean state
    grid_type : str
        Grid type
    grid : Grid
        Grid object
    initial_values : Dict[str, float]
        Initial tracer statistics for comparison
    config : StommelGyreTracerConfig
        Configuration parameters
        
    Returns
    -------
    Dict[str, float]
        Tracer conservation metrics
    """
    metrics = {}
    
    # Get area weights and masks
    if grid_type == "mpas":
        area = np.asarray(grid.areaCell, dtype=np.float64)
    else:
        area = np.asarray(grid.area, dtype=np.float64)
        
    if hasattr(state, 'land_mask'):
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
    else:
        mask = 1.0
    
    ocean = mask > 0.5
    
    # Current surface salinity
    S_sfc = np.asarray(state.S.data[..., 0], dtype=np.float64)
    
    # Integral conservation
    S_integral = float(np.sum(S_sfc * area * mask))
    metrics["S_integral"] = S_integral
    
    if abs(initial_values.get("S_integral", 0)) > 1e-30:
        S_integral_init = initial_values["S_integral"]
        integral_drift = abs(S_integral - S_integral_init) / abs(S_integral_init)
        metrics["S_integral_drift"] = integral_drift
    
    # Extrema preservation
    S_min = float(np.min(S_sfc[ocean]))
    S_max = float(np.max(S_sfc[ocean]))
    metrics["S_min"] = S_min
    metrics["S_max"] = S_max
    
    # Check for spurious extrema
    S_min_init = initial_values.get("S_min", S_min)
    S_max_init = initial_values.get("S_max", S_max)
    
    overshoot = max(0, S_max - S_max_init)
    undershoot = max(0, S_min_init - S_min)
    metrics["overshoot"] = overshoot
    metrics["undershoot"] = undershoot
    
    return metrics


def compute_transport_metrics(diagnostics: Dict[str, list], 
                            initial_values: Dict[str, float],
                            config: StommelGyreTracerConfig) -> Dict[str, float]:
    """Compute tracer transport metrics from time series diagnostics.
    
    Parameters
    ---------- 
    diagnostics : Dict[str, list]
        Time series diagnostics from simulation
    initial_values : Dict[str, float]
        Initial tracer statistics
    config : StommelGyreTracerConfig
        Configuration parameters
        
    Returns
    -------
    Dict[str, float]
        Transport metrics for validation
    """
    metrics = {}
    
    # Integral conservation over time
    S_integral_list = diagnostics.get("S_integral", [])
    if len(S_integral_list) >= 2:
        S_int_init = S_integral_list[0]
        S_int_final = S_integral_list[-1]
        if abs(S_int_init) > 1e-30:
            integral_drift = abs(S_int_final - S_int_init) / abs(S_int_init)
            metrics["integral_drift"] = integral_drift
    
    # Extrema evolution
    S_min_list = diagnostics.get("S_min", [])
    S_max_list = diagnostics.get("S_max", [])
    
    if len(S_min_list) >= 1 and len(S_max_list) >= 1:
        S_min_final = S_min_list[-1]
        S_max_final = S_max_list[-1]
        
        S_min_init = initial_values.get("S_min", S_min_final)
        S_max_init = initial_values.get("S_max", S_max_final)
        
        overshoot = max(0, S_max_final - S_max_init)
        undershoot = max(0, S_min_init - S_min_final)
        
        metrics["overshoot_final"] = overshoot
        metrics["undershoot_final"] = undershoot
        metrics["S_min_final"] = S_min_final
        metrics["S_max_final"] = S_max_final
        
    return metrics


def validate_results(final_state, diagnostics: Dict[str, list], 
                   config: StommelGyreTracerConfig = None,
                   **validation_kwargs) -> Tuple[bool, str]:
    """Validate Stommel gyre tracer experiment results.
    
    Success criteria:
    - Tracer integral conservation within tight bounds
    - Minimal spurious extrema (overshoot/undershoot)
    - No NaN or infinite values
    - Realistic tracer evolution and transport
    
    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list] 
        Time series diagnostics
    config : StommelGyreTracerConfig, optional
        Configuration parameters
    **validation_kwargs
        Additional validation parameters (initial_values)
        
    Returns
    -------
    bool
        True if validation passes
    str
        Descriptive notes about the validation
    """
    if config is None:
        config = StommelGyreTracerConfig()
    
    # Check for NaN/infinite values
    if hasattr(final_state, 'S') and not jnp.all(jnp.isfinite(final_state.S.data)):
        return False, "NaN/Inf detected in final salinity field"
    if hasattr(final_state, 'T') and not jnp.all(jnp.isfinite(final_state.T.data)):
        return False, "NaN/Inf detected in final temperature field"
    if hasattr(final_state, 'eta') and not jnp.all(jnp.isfinite(final_state.eta.data)):
        return False, "NaN/Inf detected in final eta field"
    
    # Compute transport metrics
    initial_values = validation_kwargs.get("initial_values", {})
    if initial_values and diagnostics:
        metrics = compute_transport_metrics(diagnostics, initial_values, config)
    else:
        metrics = {}
    
    # Validation
    success = True
    notes_parts = []
    
    # Integral conservation check
    if "integral_drift" in metrics:
        drift = metrics["integral_drift"]
        notes_parts.append(f"S_integral_drift={drift:.2e}")
        if drift > config.max_integral_drift:
            success = False
            notes_parts.append("FAIL: excessive tracer drift")
    
    # Spurious extrema check
    if "overshoot_final" in metrics:
        overshoot = metrics["overshoot_final"]
        notes_parts.append(f"overshoot={overshoot:.3f}")
        if overshoot > config.max_overshoot:
            success = False
            notes_parts.append("FAIL: excessive overshoot")
    
    if "undershoot_final" in metrics:
        undershoot = metrics["undershoot_final"]
        notes_parts.append(f"undershoot={undershoot:.3f}")
        if undershoot > config.max_undershoot:
            success = False
            notes_parts.append("FAIL: excessive undershoot")
    
    # Final extrema values
    if "S_min_final" in metrics and "S_max_final" in metrics:
        S_min = metrics["S_min_final"]
        S_max = metrics["S_max_final"]
        notes_parts.append(f"S_range=[{S_min:.2f},{S_max:.2f}]PSU")
    
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
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("SSS", "SSS (PSU)", "YlGnBu"),
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
        "mean_T": "degC", 
        "mean_S": "PSU",
        "S_min": "PSU",
        "S_max": "PSU",
        "S_integral": "PSU*m^2",
        "S_integral_rel": "",
    }


# Standard experiment configuration for test matrix integration
EXPERIMENT_CONFIG = {
    "name": "stommel_gyre_tracer",
    "description": "Wind-driven gyre circulation with passive tracer transport",
    "scientific_purpose": "Validates tracer conservation and transport in sheared flows",
    "reference": "Hecht et al. (2000), DOI: 10.1016/S1463-5003(00)00004-4",
    "config_class": StommelGyreTracerConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 60.0,   # days
    "quick_duration": 5.0,      # days  
    "expected_metrics": {
        "integral_drift": "< 1e-3",
        "overshoot_final": "< 0.1 PSU",
        "undershoot_final": "< 0.1 PSU"
    },
    "grid_support": {
        "cubed_sphere": True,
        "latlon": True, 
        "mpas": True,
        "spectral": False  # Complex tracer implementation
    },
    "special_config": {
        "passive_tracer": True,
        "conservation_critical": True,
        "long_integration": True
    }
}