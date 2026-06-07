"""Barotropic Wave Propagation Ocean Experiment.

A fundamental test case that verifies the proper propagation of barotropic gravity
waves through a Gaussian SSH perturbation. This test validates the shallow water
dynamics, numerical dispersion properties, and grid-specific wave propagation
characteristics.

Scientific Purpose:
- Validate barotropic gravity wave dynamics
- Test numerical dispersion and phase speed accuracy
- Benchmark grid-specific wave propagation properties
- Verify momentum conservation during wave propagation
- Test SSH gradient handling across different grid topologies

Domain Configuration:
- Global ocean with land at high latitudes (|lat| > 80°)
- Uniform depth: 5500m in ocean regions
- Rest state stratification as background
- Gaussian SSH perturbation: 1m amplitude, 10° width, centered at (180°E, 0°N)

Expected Behavior:
- Symmetric wave propagation from initial perturbation
- Phase speed ≈ sqrt(g*H) ≈ 234 m/s for shallow water waves
- Wave front should maintain circular symmetry (on sphere)
- Minimal numerical dispersion or grid imprint
- Conservation of total energy and momentum

Validation Criteria:
- Wave front reaches expected distance: d ≈ c*t
- Maximum SSH remains close to initial amplitude (1m)
- Minimal spurious oscillations or grid-scale noise
- Symmetric propagation pattern without grid artifacts

References:
- Shallow water wave theory: Gill (1982), "Atmosphere-Ocean Dynamics"
- Numerical ocean modeling: Griffies (2004), "Fundamentals of Ocean Climate Models"
- Grid comparison studies: standard ocean model benchmarking practice
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm.core.field import Field


@dataclass
class BarotropicWaveConfig:
    """Configuration parameters for barotropic wave experiment.

    All parameters have physically meaningful defaults that work across
    different grid types and resolutions.
    """
    # Initial state (inherits from rest_state)
    T_water_init_C: float = 20.0        # Surface temperature [°C]
    T_deep: float = 2.0            # Deep ocean temperature [°C]
    scale_depth: float = 1000.0    # Temperature e-folding depth [m]
    S_uniform: float = 35.0        # Salinity [PSU]

    # Domain configuration
    H_max: float = 5500.0          # Maximum ocean depth [m]
    land_lat_threshold: float = 80.0  # Latitude threshold for land [degrees]
    spectral_land_lat_threshold: float = 90.0  # No land for spectral grid

    # Wave perturbation parameters
    eta_amplitude: float = 1.0     # SSH perturbation amplitude [m]
    sigma_degrees: float = 10.0    # Gaussian width [degrees]
    center_lon: float = 180.0      # Perturbation center longitude [degrees]
    center_lat: float = 0.0        # Perturbation center latitude [degrees]


def create_initial_conditions(grid_type: str, grid, z_coord,
                            config: BarotropicWaveConfig = None):
    """Create barotropic wave initial conditions for any grid type.

    This creates a rest state background with a superimposed Gaussian SSH
    perturbation. The wave perturbation is applied consistently across all
    grid types using great-circle distance calculations.

    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere", "latlon", "mpas", or "spectral"
    grid : Grid
        Grid object (type depends on grid_type)
    z_coord : OceanZCoordinate
        Vertical coordinate system
    config : BarotropicWaveConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    OceanState
        Initial state with rest-state background + SSH perturbation

    Notes
    -----
    - Uses same background stratification as rest_state experiment
    - Gaussian perturbation centered at (180°E, 0°N) for all grids
    - Great-circle distance ensures consistent perturbation shape
    - Spectral grid: converts perturbation to spectral coefficients
    - All other grids: applies perturbation directly in physical space
    """
    if config is None:
        config = BarotropicWaveConfig()

    # First create rest state background
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        state = rest_state_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C,
            T_deep=config.T_deep,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C,
            T_deep=config.T_deep,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        state = rest_state_mpas_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C,
            T_deep=config.T_deep,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        state = rest_state_spectral_ocean(
            grid, z_coord,
            T_water_init_C=config.T_water_init_C,
            T_deep=config.T_deep,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.spectral_land_lat_threshold
        )

    else:
        raise ValueError(f"Unknown grid type: {grid_type}")

    # Add Gaussian SSH perturbation
    return _add_gaussian_perturbation(state, grid_type, grid, config)


def _add_gaussian_perturbation(state, grid_type: str, grid,
                             config: BarotropicWaveConfig):
    """Add Gaussian SSH perturbation to the rest state.

    Uses great-circle distance centered at specified location,
    consistent across all grid types.
    """
    eta_amp = config.eta_amplitude
    sigma_rad = config.sigma_degrees * np.pi / 180.0
    lon0 = config.center_lon * np.pi / 180.0  # Convert to radians
    lat0 = config.center_lat * np.pi / 180.0

    def _great_circle_perturbation(lon_rad, lat_rad):
        """Compute Gaussian SSH perturbation using great-circle distance."""
        dlon = lon_rad - lon0
        dist = np.arccos(np.clip(
            np.sin(lat_rad) * np.sin(lat0)
            + np.cos(lat_rad) * np.cos(lat0) * np.cos(dlon),
            -1.0, 1.0,
        ))
        return eta_amp * np.exp(-0.5 * (dist / sigma_rad) ** 2)

    if grid_type == "cubed_sphere":
        lon = np.asarray(grid.lon, dtype=np.float64)
        lat = np.asarray(grid.lat, dtype=np.float64)
        perturb = _great_circle_perturbation(lon, lat)
        new_eta = state.eta.data + jnp.array(perturb)
        return state._replace(eta=Field(data=new_eta, name="eta",
                                      dims=state.eta.dims, units="m"))

    elif grid_type == "latlon":
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='xy')
        perturb = _great_circle_perturbation(lon_2d, lat_2d)
        new_eta = state.eta.data + jnp.array(perturb)
        return state._replace(eta=Field(data=new_eta, name="eta",
                                      dims=state.eta.dims, units="m"))

    elif grid_type == "mpas":
        lon = np.asarray(grid.lonCell, dtype=np.float64)
        lat = np.asarray(grid.latCell, dtype=np.float64)
        perturb = _great_circle_perturbation(lon, lat)
        new_eta = state.eta.data + jnp.array(perturb)
        return state._replace(eta=Field(data=new_eta, name="eta",
                                      dims=state.eta.dims, units="m"))

    elif grid_type == "spectral":
        from legoesm.grids.gaussian import sh_analysis
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='xy')
        perturb = _great_circle_perturbation(lon_2d, lat_2d)
        perturb_hat = sh_analysis(grid, jnp.array(perturb))
        new_eta_hat = state.eta_hat.data + perturb_hat
        return state._replace(eta_hat=Field(data=new_eta_hat, name="eta_hat",
                                          dims=state.eta_hat.dims, units="m"))

    raise ValueError(f"Unknown grid type: {grid_type}")


def create_forcings(grid_type: str, grid, config: BarotropicWaveConfig = None):
    """Create forcing functions for barotropic wave experiment.

    The barotropic wave experiment has no external forcings - wave propagation
    is driven by the initial SSH perturbation and gravity wave dynamics.

    Returns
    -------
    None
        No forcings for this experiment
    """
    return None


def create_domain_config(config: BarotropicWaveConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = BarotropicWaveConfig()

    return {
        "H_max": config.H_max,
        "land_lat_threshold": config.land_lat_threshold,
        "spectral_land_lat_threshold": config.spectral_land_lat_threshold,
        "description": "Global ocean with Gaussian SSH perturbation",
    }


def compute_wave_metrics(diagnostics: Dict[str, list], config: BarotropicWaveConfig) -> Dict[str, float]:
    """Compute wave propagation metrics specific to barotropic wave validation.

    Parameters
    ----------
    diagnostics : Dict[str, list]
        Time series diagnostics from simulation
    config : BarotropicWaveConfig
        Configuration parameters

    Returns
    -------
    Dict[str, float]
        Wave propagation metrics for validation
    """
    metrics = {}

    # Maximum SSH - should remain close to initial amplitude
    max_eta_list = diagnostics.get("max_abs_eta", [])
    if len(max_eta_list) >= 2:
        max_eta_final = max_eta_list[-1]
        max_eta_initial = max_eta_list[0]
        eta_conservation = max_eta_final / max_eta_initial if max_eta_initial != 0 else 0
        metrics["max_eta_final"] = max_eta_final
        metrics["eta_conservation"] = eta_conservation

    # Mean SSH - should remain close to zero (conservation)
    mean_eta_list = diagnostics.get("mean_eta", [])
    if len(mean_eta_list) >= 2:
        mean_eta_drift = abs(mean_eta_list[-1] - mean_eta_list[0])
        metrics["mean_eta_drift"] = mean_eta_drift

    return metrics


def validate_results(final_state, diagnostics: Dict[str, list],
                   config: BarotropicWaveConfig = None) -> Tuple[bool, str]:
    """Validate barotropic wave experiment results.

    Success criteria:
    - Maximum SSH amplitude preserved within 20% of initial value
    - Mean SSH drift < 1e-4 m (conservation)
    - No NaN or infinite values
    - Final maximum SSH > 0.1m (wave still present)

    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list]
        Time series diagnostics
    config : BarotropicWaveConfig, optional
        Configuration parameters

    Returns
    -------
    bool
        True if validation passes
    str
        Descriptive notes about the validation
    """
    if config is None:
        config = BarotropicWaveConfig()

    # Compute wave metrics
    metrics = compute_wave_metrics(diagnostics, config)

    # Check for NaN/infinite values in final state
    if hasattr(final_state, 'eta') and not jnp.all(jnp.isfinite(final_state.eta.data)):
        return False, "NaN/Inf detected in final eta field"
    if hasattr(final_state, 'T') and not jnp.all(jnp.isfinite(final_state.T.data)):
        return False, "NaN/Inf detected in final temperature field"

    # Validation thresholds
    eta_conservation_min = 0.8   # 80% of initial amplitude
    eta_conservation_max = 1.2   # 120% of initial amplitude
    mean_eta_drift_max = 1e-4    # m
    min_final_amplitude = 0.1    # m

    # Check validation criteria
    success = True
    notes_parts = []

    if "max_eta_final" in metrics:
        max_eta = metrics["max_eta_final"]
        notes_parts.append(f"max_eta={max_eta:.3f}m")
        if max_eta < min_final_amplitude:
            success = False
            notes_parts.append("FAIL: wave dissipated")

    if "eta_conservation" in metrics:
        eta_cons = metrics["eta_conservation"]
        notes_parts.append(f"eta_conservation={eta_cons:.3f}")
        if eta_cons < eta_conservation_min or eta_cons > eta_conservation_max:
            success = False
            notes_parts.append("FAIL: poor amplitude conservation")

    if "mean_eta_drift" in metrics:
        mean_drift = metrics["mean_eta_drift"]
        notes_parts.append(f"mean_drift={mean_drift:.2e}m")
        if mean_drift > mean_eta_drift_max:
            success = False
            notes_parts.append("FAIL: excessive drift")

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
        "mean_S": "PSU"
    }


# Standard experiment configuration for test matrix integration
EXPERIMENT_CONFIG = {
    "name": "barotropic_wave",
    "description": "Barotropic gravity wave propagation from Gaussian SSH perturbation",
    "scientific_purpose": "Validates shallow water dynamics and wave propagation properties",
    "reference": "Standard shallow water wave theory (Gill 1982)",
    "config_class": BarotropicWaveConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 2.0,   # days
    "quick_duration": 0.2,     # days
    "expected_metrics": {
        "max_eta_final": "> 0.1 m",
        "eta_conservation": "0.8-1.2",
        "mean_eta_drift": "< 1e-4 m"
    },
    "theoretical_wave_speed": 234.0,  # m/s for H=5500m (sqrt(g*H))
}