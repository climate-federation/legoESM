"""Rest State Ocean Experiment.

A fundamental ocean test case that verifies numerical stability and conservation
properties by starting from a stratified rest state and ensuring the model
remains close to the initial condition with minimal drift.

Scientific Purpose:
- Validate numerical stability of ocean dynamics
- Test conservation of mass, heat, and salt
- Benchmark grid-specific performance
- Verify land-ocean boundary treatment

Domain Configuration:
- Global ocean with land at high latitudes (|lat| > 80°)
- Uniform depth: 5500m in ocean regions
- Exponential temperature stratification: 20°C (surface) → 2°C (deep)
- Uniform salinity: 35 PSU
- Zero initial velocity and SSH

Expected Behavior:
- Minimal drift in mean quantities (eta, T, S)
- Typical acceptable drift: O(1e-12) for η, O(1e-6) for T
- No spurious circulation or instabilities

References:
- Standard ocean model validation practice
- Comparable to "park state" tests in atmospheric models
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from legoesm.constants import g


@dataclass
class RestStateConfig:
    """Configuration parameters for rest state experiment.

    All parameters have physically meaningful defaults that work across
    different grid types and resolutions.
    """
    # Temperature stratification
    T_surface: float = 20.0        # Surface temperature [°C]
    T_deep: float = 2.0            # Deep ocean temperature [°C]
    scale_depth: float = 1000.0    # Temperature e-folding depth [m]

    # Salinity (uniform)
    S_uniform: float = 35.0        # Salinity [PSU]

    # Domain configuration
    H_max: float = 5500.0          # Maximum ocean depth [m]
    land_lat_threshold: float = 80.0  # Latitude threshold for land [degrees]

    # Spectral grid: same land threshold as other grids; tanh taper mitigates Gibbs
    spectral_land_lat_threshold: float = 80.0


def create_initial_conditions(grid_type: str, grid, z_coord,
                            config: RestStateConfig = None):
    """Create rest state initial conditions for any grid type.

    This function serves as a unified interface that calls the appropriate
    grid-specific initialization routine while maintaining consistent
    physical parameters across all grids.

    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere", "latlon", "mpas", or "spectral"
    grid : Grid
        Grid object (type depends on grid_type)
    z_coord : OceanZCoordinate
        Vertical coordinate system
    config : RestStateConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    OceanState
        Initial state with rest-state stratification

    Notes
    -----
    - Spectral grid uses no land mask to avoid Gibbs ringing from discontinuities
    - All other grids use land at |lat| > 80° for realistic polar boundaries
    - Temperature profile: T(z) = T_deep + (T_surface - T_deep) * exp(z/scale)
    - Zero velocity and SSH everywhere
    """
    if config is None:
        config = RestStateConfig()

    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        return rest_state_ocean(
            grid, z_coord,
            T_surface=config.T_surface,
            T_deep=config.T_deep,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        return rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_surface=config.T_surface,
            T_deep=config.T_deep,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        return rest_state_mpas_ocean(
            grid, z_coord,
            T_surface=config.T_surface,
            T_deep=config.T_deep,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        # Use pure global ocean for spectral to avoid Gibbs ringing
        return rest_state_spectral_ocean(
            grid, z_coord,
            T_surface=config.T_surface,
            T_deep=config.T_deep,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.spectral_land_lat_threshold
        )

    else:
        raise ValueError(f"Unknown grid type: {grid_type}")


def create_forcings(grid_type: str, grid, config: RestStateConfig = None):
    """Create forcing functions for rest state experiment.

    The rest state experiment has no external forcings - this is the point!
    The ocean should remain in rest state under its own dynamics.

    Returns
    -------
    None
        No forcings for this experiment
    """
    return None


def create_domain_config(config: RestStateConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = RestStateConfig()

    return {
        "H_max": config.H_max,
        "land_lat_threshold": config.land_lat_threshold,
        "spectral_land_lat_threshold": config.spectral_land_lat_threshold,
        "description": "Global ocean with polar land masses",
    }


def compute_drift_metrics(diagnostics: Dict[str, list], H_max: float) -> Dict[str, float]:
    """Compute drift metrics specific to rest state validation.

    For rest state, we care about absolute drift rather than relative drift
    since initial values are often zero (e.g., mean_eta = 0).

    Parameters
    ----------
    diagnostics : Dict[str, list]
        Time series diagnostics from simulation
    H_max : float
        Ocean depth for normalization

    Returns
    -------
    Dict[str, float]
        Drift metrics for validation
    """
    metrics = {}

    # SSH drift: absolute change normalized by ocean depth
    # (avoids division by ~0 for initial eta ≈ 0)
    eta_list = diagnostics.get("mean_eta", [])
    if len(eta_list) >= 2:
        eta_drift = abs(eta_list[-1] - eta_list[0]) / H_max
        metrics["eta_drift"] = eta_drift

    # Temperature drift: relative change
    T_list = diagnostics.get("mean_T", [])
    if len(T_list) >= 2 and T_list[0] != 0:
        T_drift = abs(T_list[-1] - T_list[0]) / abs(T_list[0])
        metrics["T_drift"] = T_drift

    # Salinity drift: relative change
    S_list = diagnostics.get("mean_S", [])
    if len(S_list) >= 2 and S_list[0] != 0:
        S_drift = abs(S_list[-1] - S_list[0]) / abs(S_list[0])
        metrics["S_drift"] = S_drift

    return metrics


def validate_results(final_state, diagnostics: Dict[str, list],
                   config: RestStateConfig = None) -> Tuple[bool, str]:
    """Validate rest state experiment results.

    Success criteria:
    - SSH drift < 1e-10 (normalized by ocean depth)
    - Temperature drift < 1e-4 (relative)
    - Salinity drift < 1e-6 (relative)
    - No NaN or infinite values

    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list]
        Time series diagnostics
    config : RestStateConfig, optional
        Configuration parameters

    Returns
    -------
    bool
        True if validation passes
    str
        Descriptive notes about the validation
    """
    if config is None:
        config = RestStateConfig()

    # Compute drift metrics
    metrics = compute_drift_metrics(diagnostics, config.H_max)

    # Check for NaN/infinite values in final state
    if hasattr(final_state, 'eta') and not jnp.all(jnp.isfinite(final_state.eta.data)):
        return False, "NaN/Inf detected in final eta field"
    if hasattr(final_state, 'T') and not jnp.all(jnp.isfinite(final_state.T.data)):
        return False, "NaN/Inf detected in final temperature field"

    # Validation thresholds
    eta_threshold = 1e-10  # Normalized by H_max
    T_threshold = 1e-4     # Relative
    S_threshold = 1e-6     # Relative

    # Check drift criteria
    success = True
    notes_parts = []

    if "eta_drift" in metrics:
        eta_drift = metrics["eta_drift"]
        notes_parts.append(f"eta drift={eta_drift:.2e}")
        if eta_drift > eta_threshold:
            success = False

    if "T_drift" in metrics:
        T_drift = metrics["T_drift"]
        notes_parts.append(f"T drift={T_drift:.2e}")
        if T_drift > T_threshold:
            success = False

    if "S_drift" in metrics:
        S_drift = metrics["S_drift"]
        notes_parts.append(f"S drift={S_drift:.2e}")
        if S_drift > S_threshold:
            success = False

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
        "mean_T": "degC",
        "mean_S": "PSU"
    }


# Standard experiment configuration for test matrix integration
EXPERIMENT_CONFIG = {
    "name": "rest_state",
    "description": "Rest state adjustment test - model should remain near initial condition",
    "scientific_purpose": "Validates numerical stability and conservation properties",
    "reference": "Standard ocean model validation practice",
    "config_class": RestStateConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 1.0,   # days
    "quick_duration": 0.1,     # days
    "expected_metrics": {
        "eta_drift": "< 1e-10 (normalized)",
        "T_drift": "< 1e-4 (relative)",
        "S_drift": "< 1e-6 (relative)"
    }
}