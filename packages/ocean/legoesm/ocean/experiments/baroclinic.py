"""Baroclinic Adjustment Ocean Experiment.

A fundamental test case that studies the geostrophic adjustment process following
the introduction of a meridional temperature front. This experiment validates
the ocean model's ability to handle baroclinic instabilities, thermal wind
balance, and the adjustment to quasi-geostrophic equilibrium.

Scientific Purpose:
- Validate baroclinic adjustment processes
- Test thermal wind balance and geostrophic equilibrium
- Study meridional overturning circulation development
- Verify handling of temperature gradients and density fronts
- Test numerical stability during adjustment phase
- Benchmark conservation properties during front evolution

Domain Configuration:
- Global ocean with land at high latitudes (|lat| > 80°)
- Uniform depth: 5500m in ocean regions
- Rest state stratification as background
- Imposed meridional temperature front: ±5°C gradient

Physical Setup:
- Meridional temperature gradient: +5°C at equator, -5°C at poles
- Perturbation decreases exponentially with depth
- Background rest state stratification
- No initial motion - adjustment from pure thermal wind imbalance

Expected Behavior:
- Development of thermal wind circulation
- Geostrophic adjustment of velocity field
- Baroclinic instability growth (if resolution permits)
- Meridional overturning circulation establishment
- Conservation of heat and momentum during adjustment

Validation Criteria:
- Temperature drift within acceptable bounds
- No excessive circulation or numerical instabilities
- Realistic adjustment timescales (days to weeks)
- Preservation of thermal structure during evolution

References:
- Gill (1982), "Atmosphere-Ocean Dynamics" - geostrophic adjustment
- Pedlosky (1987), "Geophysical Fluid Dynamics" - baroclinic instability
- Vallis (2017), "Atmospheric and Oceanic Fluid Dynamics" - thermal wind balance
- Standard ocean model validation studies for baroclinic adjustment
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm.core.field import Field


@dataclass
class BaroclinicConfig:
    """Configuration parameters for baroclinic adjustment experiment.

    All parameters have physically meaningful defaults that work across
    different grid types and resolutions.
    """
    # Background state (inherits from rest_state)
    T_water_init_C: float = 20.0        # Surface temperature [°C]
    T_deep_C: float = 2.0            # Deep ocean temperature [°C]
    scale_depth: float = 1000.0    # Temperature e-folding depth [m]
    S_uniform: float = 35.0        # Salinity [PSU]

    # Domain configuration
    H_max: float = 5500.0          # Maximum ocean depth [m]
    land_lat_threshold: float = 80.0  # Latitude threshold for land [degrees]
    spectral_land_lat_threshold: float = 90.0  # No land for spectral grid

    # Temperature front parameters
    T_perturbation_amplitude: float = 5.0  # Temperature gradient amplitude [°C]
    vertical_decay_scale: float = 3.0      # Vertical e-folding scale (levels)


def create_initial_conditions(grid_type: str, grid, z_coord,
                            config: BaroclinicConfig = None):
    """Create baroclinic adjustment initial conditions for any grid type.

    This creates a rest state background with a superimposed meridional
    temperature front that varies as T_pert = amplitude * cos(latitude).

    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere", "latlon", "mpas", or "spectral"
    grid : Grid
        Grid object (type depends on grid_type)
    z_coord : OceanZCoordinate
        Vertical coordinate system
    config : BaroclinicConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    OceanState
        Initial state with rest-state background + temperature front

    Notes
    -----
    - Temperature perturbation: +amplitude at equator, -amplitude at poles
    - Exponential decay with depth: exp(-k / decay_scale)
    - Spectral grid: transforms perturbation to spectral coefficients
    - All other grids: applies perturbation directly in physical space
    - Creates thermal wind imbalance that drives adjustment
    """
    if config is None:
        config = BaroclinicConfig()

    # First create rest state background
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        state = rest_state_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C,
            T_deep=config.T_deep_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C,
            T_deep=config.T_deep_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        state = rest_state_mpas_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C,
            T_deep=config.T_deep_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        state = rest_state_spectral_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C,
            T_deep=config.T_deep_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.spectral_land_lat_threshold
        )

    else:
        raise ValueError(f"Unknown grid type: {grid_type}")

    # Add baroclinic temperature front
    return _add_temperature_front(state, grid_type, grid, z_coord, config)


def _add_temperature_front(state, grid_type: str, grid, z_coord,
                         config: BaroclinicConfig):
    """Add meridional temperature front to the rest state.

    Creates a meridional temperature gradient that varies as:
    T_pert(lat, z) = amplitude * cos(lat) * exp(-k / decay_scale)
    where k is the vertical level index.
    """
    T_amp = config.T_perturbation_amplitude
    decay_scale = config.vertical_decay_scale

    if grid_type == "spectral":
        # Spectral grid: work in spectral space
        from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d

        lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        T_pert = T_amp * np.cos(np.radians(lat))

        # Get current temperature in physical space
        T_grid_hat = state.T_hat.data
        nlev = T_grid_hat.shape[-1]
        T_grid = np.array(sh_synthesis_3d(grid, T_grid_hat), dtype=np.float64)

        # T_pert is (n_lat,) → broadcast to (n_lat, n_lon)
        T_pert_2d = T_pert[:, np.newaxis] * np.ones((1, grid.n_lon))

        # Add perturbation with vertical decay
        for k in range(nlev):
            decay = np.exp(-k / max(decay_scale, 1))
            T_grid[..., k] += T_pert_2d * decay

        # Transform back to spectral space
        new_T_hat = sh_analysis_3d(grid, jnp.array(T_grid))
        return state._replace(T_hat=Field(new_T_hat, name="T_hat",
                                        dims=state.T_hat.dims, units="K"))

    else:
        # Physical space grids (cubed_sphere, latlon, mpas)
        if grid_type == "mpas":
            lat = np.asarray(grid.latCell, dtype=np.float64) * 180 / np.pi
        else:
            lat = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi

        T_data = np.array(state.T.data, dtype=np.float64)  # writable copy
        nlev = T_data.shape[-1]
        T_pert = T_amp * np.cos(np.radians(lat))

        # Reshape T_pert to broadcast with T_data[..., k]
        pert_shape = T_data.shape[:-1]  # spatial dimensions
        T_pert = np.broadcast_to(T_pert.reshape(
            T_pert.shape + (1,) * (len(pert_shape) - T_pert.ndim)), pert_shape)

        # Add perturbation with vertical decay
        for k in range(nlev):
            decay = np.exp(-k / max(decay_scale, 1))
            T_data[..., k] += T_pert * decay

        return state._replace(T=Field(jnp.array(T_data), name="T",
                                    dims=state.T.dims, units="K"))


def create_forcings(grid_type: str, grid, config: BaroclinicConfig = None):
    """Create forcing functions for baroclinic adjustment experiment.

    The baroclinic adjustment experiment has no external forcings - the
    adjustment is driven by the initial temperature front imbalance.

    Returns
    -------
    None
        No forcings for this experiment
    """
    return None


def create_domain_config(config: BaroclinicConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = BaroclinicConfig()

    return {
        "H_max": config.H_max,
        "land_lat_threshold": config.land_lat_threshold,
        "spectral_land_lat_threshold": config.spectral_land_lat_threshold,
        "description": "Global ocean with meridional temperature front",
        "perturbation_type": "meridional_temperature_gradient",
    }


def compute_adjustment_metrics(diagnostics: Dict[str, list],
                             config: BaroclinicConfig) -> Dict[str, float]:
    """Compute adjustment metrics specific to baroclinic validation.

    Parameters
    ----------
    diagnostics : Dict[str, list]
        Time series diagnostics from simulation
    config : BaroclinicConfig
        Configuration parameters

    Returns
    -------
    Dict[str, float]
        Adjustment metrics for validation
    """
    metrics = {}

    # Temperature drift - should be minimal for conservation
    mean_T_list = diagnostics.get("mean_T", [])
    if len(mean_T_list) >= 2:
        T_initial = mean_T_list[0]
        T_final = mean_T_list[-1]
        if abs(T_initial) > 1e-10:  # Avoid division by very small numbers
            T_drift_relative = abs(T_final - T_initial) / abs(T_initial)
            metrics["T_drift_relative"] = T_drift_relative
        T_drift_absolute = abs(T_final - T_initial)
        metrics["T_drift_absolute"] = T_drift_absolute

    # Speed development - should show adjustment circulation
    speed_list = diagnostics.get("max_speed", [])
    if not speed_list:
        speed_list = diagnostics.get("max_abs_u", [])  # MPAS alternative

    if len(speed_list) >= 2:
        max_speed_final = speed_list[-1]
        metrics["max_speed_final"] = max_speed_final

        # Check if speeds developed from rest
        max_speed_initial = speed_list[0]
        if max_speed_initial > 1e-10:
            speed_growth = max_speed_final / max_speed_initial
            metrics["speed_growth"] = speed_growth

    return metrics


def validate_results(final_state, diagnostics: Dict[str, list],
                   config: BaroclinicConfig = None) -> Tuple[bool, str]:
    """Validate baroclinic adjustment experiment results.

    Success criteria:
    - Temperature drift within acceptable bounds (< 1e-3 relative)
    - Development of circulation from initial rest state
    - Maximum speeds within reasonable range (< 1 m/s)
    - No NaN or infinite values
    - Conservation of thermal structure

    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list]
        Time series diagnostics
    config : BaroclinicConfig, optional
        Configuration parameters

    Returns
    -------
    bool
        True if validation passes
    str
        Descriptive notes about the validation
    """
    if config is None:
        config = BaroclinicConfig()

    # Compute adjustment metrics
    metrics = compute_adjustment_metrics(diagnostics, config)

    # Check for NaN/infinite values in final state
    if hasattr(final_state, 'T') and not jnp.all(jnp.isfinite(final_state.T.data)):
        return False, "NaN/Inf detected in final temperature field"
    if hasattr(final_state, 'eta') and not jnp.all(jnp.isfinite(final_state.eta.data)):
        return False, "NaN/Inf detected in final eta field"
    if hasattr(final_state, 'u') and not jnp.all(jnp.isfinite(final_state.u.data)):
        return False, "NaN/Inf detected in final u velocity field"

    # Validation thresholds
    max_T_drift_relative = 1e-3   # Maximum relative temperature drift
    max_T_drift_absolute = 0.1    # Maximum absolute drift (°C)
    max_reasonable_speed = 1.0    # m/s - maximum reasonable adjustment speed
    min_adjustment_speed = 0.001  # m/s - minimum speed showing adjustment

    # Check validation criteria
    success = True
    notes_parts = []

    if "T_drift_relative" in metrics:
        T_drift = metrics["T_drift_relative"]
        notes_parts.append(f"T_drift={T_drift:.2e}")
        if T_drift > max_T_drift_relative:
            success = False
            notes_parts.append("FAIL: excessive T drift")
    elif "T_drift_absolute" in metrics:
        T_drift = metrics["T_drift_absolute"]
        notes_parts.append(f"T_drift={T_drift:.2e}")
        if T_drift > max_T_drift_absolute:
            success = False
            notes_parts.append("FAIL: excessive T drift")

    if "max_speed_final" in metrics:
        speed = metrics["max_speed_final"]
        notes_parts.append(f"max_speed={speed:.4f}m/s")
        if speed > max_reasonable_speed:
            success = False
            notes_parts.append("FAIL: excessive speed")
        elif speed < min_adjustment_speed:
            success = False
            notes_parts.append("FAIL: insufficient adjustment")

    if "speed_growth" in metrics:
        growth = metrics["speed_growth"]
        notes_parts.append(f"speed_growth={growth:.2f}")

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
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("eta", "SSH (m)", "RdBu_r"),
        ("speed_sfc", "Surface Speed (m/s)", "plasma"),
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
    "name": "baroclinic",
    "description": "Baroclinic adjustment from meridional temperature front",
    "scientific_purpose": "Validates thermal wind balance and geostrophic adjustment",
    "reference": "Gill (1982), Pedlosky (1987) - classical baroclinic adjustment theory",
    "config_class": BaroclinicConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 10.0,   # days
    "quick_duration": 1.0,      # days
    "expected_metrics": {
        "T_drift_relative": "< 1e-3",
        "max_speed_final": "0.001-1.0 m/s",
        "speed_growth": "> 1.0"
    },
    "grid_support": {
        "cubed_sphere": True,
        "latlon": True,
        "mpas": True,
        "spectral": True
    },
}