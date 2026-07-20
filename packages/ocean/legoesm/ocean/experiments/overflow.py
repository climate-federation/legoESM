"""Overflow Ocean Experiment.

A density-driven flow test case based on Petersen et al. (2015) that studies
the descent of dense water over a bathymetric slope. This experiment validates
the representation of bottom boundary layer processes, dense water plumes,
and gravity currents in complex topography.

Scientific Purpose:
- Validate dense water overflow dynamics over topographic slopes
- Test bathymetric effects on gravity current propagation
- Study bottom boundary layer representation and mixing
- Benchmark numerical stability with steep bathymetric gradients
- Validate potential energy evolution in stratified overflow
- Test high-resolution requirements for overflow processes

Domain Configuration:
- Intermediate depth: 2000m maximum (vs 5500m in other experiments)
- High vertical resolution: 20 levels for boundary layer resolution
- Variable bathymetry: shelf (500m) to deep basin (2000m) transition
- Latitude-dependent temperature and bathymetry structure

Physical Setup:
- Cold dense water poleward of 50°N/S (T = 5°C, ρ ≈ 1027 kg/m³)
- Warm light water equatorward of 50°N/S (T = 20°C, ρ ≈ 1024 kg/m³)
- Smooth tanh transitions to avoid numerical issues
- Bathymetric shelf poleward of 40°N/S, deep basin equatorward
- Temperature decreases with depth for stable stratification

Expected Behavior:
- Dense water descent from shelf to deep basin
- Formation of bottom-trapped gravity current
- Mixing and entrainment during overflow process
- Development of hydraulic control at sill
- Potential energy decrease due to gravitational adjustment

Validation Criteria:
- Realistic overflow plume development and propagation
- Potential energy evolution within expected bounds
- No excessive numerical mixing or dissipation
- Maintenance of density stratification away from overflow
- Numerical stability with complex bathymetry

References:
- Petersen et al. (2015), "Evaluation of the arbitrary Lagrangian–Eulerian
  vertical coordinate method in the MPAS-Ocean model", Ocean Modelling 86, 93-113.
  DOI: 10.1016/j.ocemod.2014.12.004
- Ilicak et al. (2012), "Spurious dianeutral mixing and the role of momentum
  closure", Ocean Modelling 45-46, 37-49.
- Standard ocean model validation for overflow and dense water processes
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm.core.field import Field


@dataclass
class OverflowConfig:
    """Configuration parameters for overflow experiment.

    Parameters based on Petersen et al. (2015) overflow test case.
    """
    # Domain configuration (overflow-specific)
    nlev: int = 20                 # High vertical resolution for boundary layer
    H_max: float = 2000.0          # Maximum depth [m] (intermediate depth)
    land_lat_threshold: float = 80.0  # Latitude threshold for land [degrees]
    spectral_land_lat_threshold: float = 90.0  # No land for spectral grid

    # Temperature/density structure
    T_cold_C: float = 5.0            # Cold (dense) water temperature [°C]
    T_warm_C: float = 20.0           # Warm (light) water temperature [°C]
    T_deep_C: float = 2.0            # Deep water temperature [°C]
    S_uniform: float = 35.0        # Uniform salinity [PSU]

    # Transition parameters
    lat_front_deg: float = 50.0     # Temperature front latitude [degrees]
    front_width_deg: float = 5.0    # Front transition width [degrees]
    lat_shelf_deg: float = 40.0     # Bathymetric shelf latitude [degrees]
    shelf_width_deg: float = 7.0    # Shelf transition width [degrees]

    # Bathymetry parameters
    depth_shallow: float = 500.0    # Shelf depth [m]
    depth_deep: float = 2000.0      # Deep basin depth [m]

    # Vertical structure
    depth_decay_factor: float = 0.5  # Temperature decay with depth

    # Physical parameters for RPE calculation
    alpha_T: float = 2.0e-4         # Thermal expansion coefficient [1/K]
    T_reference_C: float = 12.5       # Reference temperature [°C]

    # Validation thresholds
    max_pe_drift: float = 1e-2      # Maximum PE drift (relative)
    max_T_drift: float = 1e-2       # Maximum T drift (relative)
    max_blowup_threshold: float = 200.0  # Temperature blowup threshold [°C]


def create_initial_conditions(grid_type: str, grid, z_coord,
                            config: OverflowConfig = None):
    """Create overflow initial conditions for any grid type.

    Sets up a latitude-dependent density structure with variable bathymetry
    to create an overflow scenario where dense water descends from a shelf
    into a deeper basin.

    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere", "latlon", "mpas", or "spectral"
    grid : Grid
        Grid object (type depends on grid_type)
    z_coord : OceanZCoordinate
        Vertical coordinate system (should have nlev=20)
    config : OverflowConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    OceanState
        Initial state with overflow setup (temperature + bathymetry)

    Notes
    -----
    - Intermediate depth (2000m) with high vertical resolution (20 levels)
    - Variable bathymetry: 500m shelf, 2000m deep basin
    - Temperature front creates density contrast for overflow
    - Spectral grid: limited bathymetry support, temperature only
    """
    if config is None:
        config = OverflowConfig()

    # First create rest state background with special config
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        state = rest_state_ocean(
            grid, z_coord,
            T_water_init_C=config.T_reference_C,
            T_deep=config.T_deep_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=config.T_reference_C,
            T_deep=config.T_deep_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        state = rest_state_mpas_ocean(
            grid, z_coord,
            T_water_init_C=config.T_reference_C,
            T_deep=config.T_deep_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        state = rest_state_spectral_ocean(
            grid, z_coord,
            T_water_init_C=config.T_reference_C,
            T_deep=config.T_deep_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.spectral_land_lat_threshold
        )

    else:
        raise ValueError(f"Unknown grid type: {grid_type}")

    # Add overflow structure (temperature + bathymetry)
    return _add_overflow_structure(state, grid_type, grid, z_coord, config)


def _add_overflow_structure(state, grid_type: str, grid, z_coord,
                          config: OverflowConfig):
    """Add overflow temperature structure and bathymetry."""

    # Get coordinates
    if grid_type == "mpas":
        lat = np.asarray(grid.latCell, dtype=np.float64)
        lon = np.asarray(grid.lonCell, dtype=np.float64)
    elif grid_type in ("latlon", "spectral"):
        lat_1d = np.asarray(grid.lat, dtype=np.float64)
        lon_1d = np.asarray(grid.lon, dtype=np.float64)
        lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='xy')
    else:  # cubed_sphere
        lat = np.asarray(grid.lat, dtype=np.float64)
        np.asarray(grid.lon, dtype=np.float64)

    # Convert parameters to radians
    T_cold = config.T_cold_C
    T_warm = config.T_warm_C
    T_deep = config.T_deep_C
    lat_front = np.radians(config.lat_front_deg)
    sigma_front = np.radians(config.front_width_deg)
    lat_shelf = np.radians(config.lat_shelf_deg)
    sigma_shelf = np.radians(config.shelf_width_deg)
    d_shallow = config.depth_shallow
    d_deep = config.depth_deep
    depth_decay = config.depth_decay_factor

    abs_lat = np.abs(lat)

    # Temperature profile: tanh transition at lat_front
    # Cold dense water poleward, warm light water equatorward
    T_water_init_C = T_warm + (T_cold - T_warm) * 0.5 * (
        1.0 + np.tanh((abs_lat - lat_front) / sigma_front))

    # Bathymetry: tanh transition at lat_shelf
    # Shallow shelf poleward, deep basin equatorward
    H_bathy_new = d_shallow + (d_deep - d_shallow) * 0.5 * (
        1.0 - np.tanh((abs_lat - lat_shelf) / sigma_shelf))

    if grid_type == "spectral":
        return _add_overflow_spectral(state, grid, z_coord, T_water_init_C, T_deep,
                                    depth_decay, config)
    else:
        return _add_overflow_fv(state, grid, z_coord, T_water_init_C, T_deep,
                              H_bathy_new, depth_decay, config)


def _add_overflow_spectral(state, grid, z_coord, T_water_init_C, T_deep,
                         depth_decay, config):
    """Add overflow for spectral grid (temperature only - no bathymetry)."""
    from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d

    T_hat = state.T_hat.data
    T_grid = np.array(sh_synthesis_3d(grid, T_hat), dtype=np.float64)
    nlev = T_grid.shape[-1]
    mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)

    # Apply temperature profile with depth decay
    for k in range(nlev):
        depth_frac = float(z_coord.z_full_ref[k] / z_coord.z_full_ref[-1])
        T_k = T_water_init_C * (1.0 - depth_decay * depth_frac) + T_deep * depth_frac
        T_grid[..., k] = T_k * mask

    new_T_hat = sh_analysis_3d(grid, jnp.array(T_grid))

    # Note: spectral model doesn't easily support variable bathymetry
    return state._replace(T_hat=Field(new_T_hat, name="T_hat",
                                    dims=state.T_hat.dims, units="K"))


def _add_overflow_fv(state, grid, z_coord, T_water_init_C, T_deep, H_bathy_new,
                   depth_decay, config):
    """Add overflow for finite volume grids (temperature + bathymetry)."""
    T_data = np.array(state.T.data, dtype=np.float64, copy=True)
    if hasattr(state, 'land_mask'):
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
    else:
        mask = 1.0
    nlev = T_data.shape[-1]

    # Apply temperature profile with depth decay
    for k in range(nlev):
        depth_frac = float(z_coord.z_full_ref[k] / z_coord.z_full_ref[-1])
        T_k = T_water_init_C * (1.0 - depth_decay * depth_frac) + T_deep * depth_frac
        T_data[..., k] = T_k * mask

    # Update bathymetry with land mask
    H_bathy_new_masked = H_bathy_new * mask
    # Ensure minimum depth where ocean exists
    H_bathy_new_masked = np.where(mask > 0.5,
                                  np.maximum(H_bathy_new_masked, 50.0), 0.0)

    # Build new state
    new_state = state._replace(T=Field(jnp.array(T_data), name="T",
                                     dims=state.T.dims, units="K"))

    # Add bathymetry if state supports it
    if hasattr(state, 'H_bathy'):
        new_state = new_state._replace(
            H_bathy=Field(jnp.array(H_bathy_new_masked), name="H_bathy",
                         dims=state.H_bathy.dims, units="m"))

    return new_state


def create_forcings(grid_type: str, grid, config: OverflowConfig = None):
    """Create forcing functions for overflow experiment.

    The overflow experiment has no external forcings - the dynamics
    are driven purely by the initial density and bathymetric structure.

    Returns
    -------
    None
        No forcings for this experiment
    """
    return None


def create_domain_config(config: OverflowConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = OverflowConfig()

    return {
        "nlev": config.nlev,
        "H_max": config.H_max,
        "land_lat_threshold": config.land_lat_threshold,
        "spectral_land_lat_threshold": config.spectral_land_lat_threshold,
        "description": "Dense water overflow over bathymetric slope",
        "reference": "Petersen et al. (2015) overflow test case",
        "bathymetry": "variable_shelf_to_basin",
    }


def compute_overflow_metrics(diagnostics: Dict[str, list], pe_initial: float,
                           config: OverflowConfig) -> Dict[str, float]:
    """Compute overflow and mixing metrics for validation.

    Parameters
    ----------
    diagnostics : Dict[str, list]
        Time series diagnostics from simulation
    pe_initial : float
        Initial potential energy for reference
    config : OverflowConfig
        Configuration parameters

    Returns
    -------
    Dict[str, float]
        Overflow metrics for validation
    """
    metrics = {}

    # Potential energy evolution
    pe_list = diagnostics.get("PE", [])
    if len(pe_list) >= 2 and abs(pe_initial) > 1e-30:
        pe_final = pe_list[-1]
        pe_drift = abs(pe_final - pe_initial) / abs(pe_initial)
        metrics["pe_drift_relative"] = pe_drift
        metrics["pe_final"] = pe_final
        metrics["pe_initial"] = pe_initial

    # Relative PE change
    pe_rel_list = diagnostics.get("PE_rel", [])
    if len(pe_rel_list) >= 1:
        pe_rel_final = pe_rel_list[-1]
        metrics["pe_rel_final"] = pe_rel_final

    # Temperature evolution
    mean_T_list = diagnostics.get("mean_T", [])
    if len(mean_T_list) >= 2:
        T_final = mean_T_list[-1]
        T_initial = mean_T_list[0]
        if abs(T_initial) > 1e-10:
            T_drift = abs(T_final - T_initial) / abs(T_initial)
            metrics["T_drift_relative"] = T_drift
        T_change = abs(T_final - T_initial)
        metrics["mean_T_change"] = T_change
        metrics["mean_T_final"] = T_final

    return metrics


def validate_results(final_state, diagnostics: Dict[str, list],
                   config: OverflowConfig = None,
                   **validation_kwargs) -> Tuple[bool, str]:
    """Validate overflow experiment results.

    Success criteria:
    - Potential energy and temperature evolution within bounds
    - No numerical instabilities or blowup
    - Realistic overflow behavior development
    - Conservation appropriate for gravity current process

    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list]
        Time series diagnostics
    config : OverflowConfig, optional
        Configuration parameters
    **validation_kwargs
        Additional validation parameters (pe_initial)

    Returns
    -------
    bool
        True if validation passes
    str
        Descriptive notes about the validation
    """
    if config is None:
        config = OverflowConfig()

    # Check for NaN/infinite values
    if hasattr(final_state, 'T') and not jnp.all(jnp.isfinite(final_state.T.data)):
        return False, "NaN/Inf detected in final temperature field"
    if hasattr(final_state, 'eta') and not jnp.all(jnp.isfinite(final_state.eta.data)):
        return False, "NaN/Inf detected in final eta field"

    # Check for temperature blowup — fuse min/max into a single host
    # sync rather than two ``float(jnp.X(...))`` calls.
    if hasattr(final_state, 'T'):
        _t_data = final_state.T.data
        _h = np.asarray(jnp.stack([jnp.max(_t_data), jnp.min(_t_data)]))
        max_T = float(_h[0])
        min_T = float(_h[1])
        if max_T > config.max_blowup_threshold or min_T < -config.max_blowup_threshold:
            return False, f"Temperature blowup: T_range=[{min_T:.1f}, {max_T:.1f}]°C"

    # Compute overflow metrics
    pe_initial = validation_kwargs.get("pe_initial", None)
    if pe_initial is not None:
        metrics = compute_overflow_metrics(diagnostics, pe_initial, config)
    else:
        metrics = {}

    # Validation
    success = True
    notes_parts = []

    # PE drift check
    if "pe_drift_relative" in metrics:
        pe_drift = metrics["pe_drift_relative"]
        notes_parts.append(f"PE_drift={pe_drift:.2e}")
        if pe_drift > config.max_pe_drift:
            success = False
            notes_parts.append("FAIL: excessive PE drift")

    # PE relative change
    if "pe_rel_final" in metrics:
        pe_rel = metrics["pe_rel_final"]
        notes_parts.append(f"PE_rel={pe_rel:.4e}")

    # Temperature drift
    if "T_drift_relative" in metrics:
        T_drift = metrics["T_drift_relative"]
        notes_parts.append(f"T_drift={T_drift:.2e}")
        if T_drift > config.max_T_drift:
            success = False
            notes_parts.append("FAIL: excessive T drift")

    # Temperature final
    if "mean_T_final" in metrics:
        T_final = metrics["mean_T_final"]
        notes_parts.append(f"mean_T={T_final:.2f}°C")

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
        "mean_S": "PSU",
        "PE": "J",
        "PE_rel": "",
    }


# Standard experiment configuration for test matrix integration
EXPERIMENT_CONFIG = {
    "name": "overflow",
    "description": "Dense water overflow over bathymetric slope with RPE diagnostics",
    "scientific_purpose": "Validates dense water flows and topographic effects",
    "reference": "Petersen et al. (2015), DOI: 10.1016/j.ocemod.2014.12.004",
    "config_class": OverflowConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 0.5,   # days
    "quick_duration": 0.1,     # days
    "expected_metrics": {
        "pe_drift_relative": "< 1e-2",
        "T_drift_relative": "< 1e-2",
        "pe_rel_final": "< 0"  # Energy should decrease
    },
    "grid_support": {
        "cubed_sphere": True,
        "latlon": True,
        "mpas": True,
        "spectral": False  # Limited bathymetry support
    },
    "special_config": {
        "nlev": 20,         # High vertical resolution
        "H_max": 2000.0,    # Intermediate depth
        "variable_bathymetry": True,
        "rpe_diagnostics": True
    }
}
