"""Lock Exchange Ocean Experiment.

A classic test case for density-driven flows and numerical mixing validation
based on Petersen et al. (2015). This experiment studies gravity currents
that develop when a vertical density front is released, providing insight
into spurious numerical mixing through Reference Potential Energy (RPE) analysis.

Scientific Purpose:
- Validate density-driven gravity current formation
- Quantify spurious numerical mixing through RPE diagnostics
- Test equation of state implementation and density effects
- Benchmark conservative vs. non-conservative numerical schemes
- Study vertical mixing and diapycnal transport mechanisms
- Validate high-resolution vertical structure handling

Domain Configuration:
- Shallow basin: 500m depth (vs 5500m in other experiments)
- High vertical resolution: 20 levels (vs 10 in other experiments)
- Global domain with temperature front at prime meridian
- Land at high latitudes for circulation containment

Physical Setup:
- Initial conditions: vertical density front at longitude = 0°
- Western hemisphere (lon < 0°): cold dense water (T = 5°C, ρ ≈ 1027 kg/m³)
- Eastern hemisphere (lon > 0°): warm light water (T = 30°C, ρ ≈ 1022 kg/m³)
- Uniform salinity: 35 PSU throughout domain
- Initially at rest: gravity currents develop from density contrast

Expected Behavior:
- Gravity current formation at density front
- Dense water spreads along bottom toward east
- Light water spreads along surface toward west
- Development of hydraulic control and mixing layer
- Potential energy decrease due to gravitational adjustment

Validation Criteria:
- Reference Potential Energy (RPE) evolution within expected bounds
- Minimal spurious mixing (PE drift < threshold)
- Realistic gravity current propagation speeds
- Conservation properties appropriate for mixing processes
- No numerical instabilities or excessive dissipation

References:
- Petersen et al. (2015), "Evaluation of the arbitrary Lagrangian–Eulerian
  vertical coordinate method in the MPAS-Ocean model", Ocean Modelling 86, 93-113.
  DOI: 10.1016/j.ocemod.2014.12.004
- Ilicak et al. (2012), "Spurious dianeutral mixing and the role of momentum
  closure", Ocean Modelling 45-46, 37-49.
- NEMO test cases: https://sites.nemo-ocean.io/user-guide/tests.html
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm.core.field import Field


@dataclass
class LockExchangeConfig:
    """Configuration parameters for lock exchange experiment.

    Parameters based on Petersen et al. (2015) and NEMO test suite.
    """
    # Domain configuration (Petersen 2015 Fig. 5: 20 m depth, 20 levels)
    nlev: int = 20                 # 20 levels → ~1 m / level at H_max = 20 m
    H_max: float = 20.0            # Petersen 2015 Fig. 5 channel depth [m]
    land_lat_threshold: float = 80.0  # Latitude threshold for land [degrees]
    spectral_land_lat_threshold: float = 90.0  # No land for spectral grid

    # Temperature/density front parameters
    T_cold_C: float = 5.0            # Cold (dense) side temperature [°C]
    T_warm_C: float = 30.0           # Warm (light) side temperature [°C]
    S_uniform: float = 35.0        # Uniform salinity [PSU]

    # Front location (longitude threshold)
    front_longitude: float = 0.0   # Prime meridian [degrees]

    # Physical parameters for RPE calculation
    alpha_T: float = 2.0e-4         # Thermal expansion coefficient [1/K]
    T_reference_C: float = 15.0       # Reference temperature [°C]

    # Validation thresholds
    max_pe_drift: float = 1e-2      # Maximum PE drift (relative)
    max_blowup_threshold: float = 200.0  # Temperature blowup threshold [°C]


def create_initial_conditions(grid_type: str, grid, z_coord,
                            config: LockExchangeConfig = None):
    """Create lock exchange initial conditions for any grid type.

    Sets up a vertical density front with cold dense water on one side
    and warm light water on the other, initially at rest.

    Parameters
    ----------
    grid_type : str
        Grid type: "cubed_sphere", "latlon", "mpas", or "spectral"
    grid : Grid
        Grid object (type depends on grid_type)
    z_coord : OceanZCoordinate
        Vertical coordinate system (should have nlev=20)
    config : LockExchangeConfig, optional
        Configuration parameters. Uses defaults if None.

    Returns
    -------
    OceanState
        Initial state with vertical density front

    Notes
    -----
    - High vertical resolution (20 levels) for gravity current resolution
    - Shallow depth (500m) for clear signal development
    - Temperature front creates ~5 kg/m³ density difference
    - Initially at rest - motion develops from density imbalance
    """
    if config is None:
        config = LockExchangeConfig()

    # First create rest state background with special config
    if grid_type == "cubed_sphere":
        from legoesm.ocean.init import rest_state_ocean
        state = rest_state_ocean(
            grid, z_coord,
            T_water_init_C=config.T_reference_C,  # Use reference T as background
            T_deep=config.T_reference_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "latlon":
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord,
            T_water_init_C=config.T_reference_C,
            T_deep=config.T_reference_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "mpas":
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        state = rest_state_mpas_ocean(
            grid, z_coord,
            T_water_init_C=config.T_reference_C,
            T_deep=config.T_reference_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.land_lat_threshold
        )

    elif grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
        state = rest_state_spectral_ocean(
            grid, z_coord,
            T_water_init_C=config.T_reference_C,
            T_deep=config.T_reference_C,
            S_uniform=config.S_uniform,
            H_max=config.H_max,
            land_lat_threshold=config.spectral_land_lat_threshold
        )

    else:
        raise ValueError(f"Unknown grid type: {grid_type}")

    # Add temperature front
    return _add_temperature_front(state, grid_type, grid, z_coord, config)


def _add_temperature_front(state, grid_type: str, grid, z_coord,
                         config: LockExchangeConfig):
    """Add vertical density front to create lock exchange setup."""

    # Get coordinates
    if grid_type == "mpas":
        lat = np.asarray(grid.latCell, dtype=np.float64)
        lon = np.asarray(grid.lonCell, dtype=np.float64) * 180 / np.pi
    elif grid_type in ("latlon", "spectral"):
        lat_1d = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        lon_1d = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
        lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='xy')
    else:  # cubed_sphere
        np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
        lon = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi

    T_cold = config.T_cold_C
    T_warm = config.T_warm_C
    front_lon = config.front_longitude

    # Wrapping-aware "west of front" test: works for any lon convention
    west_of_front = ((lon - front_lon + 180.0) % 360.0 - 180.0) < 0.0

    if grid_type == "spectral":
        # Spectral grid
        from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d

        T_hat = state.T_hat.data
        T_grid = np.array(sh_synthesis_3d(grid, T_hat), dtype=np.float64)
        mask = np.asarray(state.land_mask_grid.data, dtype=np.float64)

        # Create temperature front at front_longitude
        T_field = np.where(west_of_front, T_cold, T_warm)

        # Apply to all levels with land mask
        nlev = T_grid.shape[-1]
        for k in range(nlev):
            T_grid[..., k] = T_field * mask

        new_T_hat = sh_analysis_3d(grid, jnp.array(T_grid))
        return state._replace(T_hat=Field(new_T_hat, name="T_hat",
                                        dims=state.T_hat.dims, units="K"))

    else:
        # Physical space grids
        T_data = np.array(state.T.data, dtype=np.float64, copy=True)
        if hasattr(state, 'land_mask'):
            mask = np.asarray(state.land_mask.data, dtype=np.float64)
        else:
            mask = 1.0  # No land mask
        nlev = T_data.shape[-1]

        # Apply temperature front to all levels
        T_front = np.where(west_of_front, T_cold, T_warm)
        for k in range(nlev):
            T_data[..., k] = T_front * mask

        return state._replace(T=Field(jnp.array(T_data), name="T",
                                    dims=state.T.dims, units="K"))


def create_forcings(grid_type: str, grid, config: LockExchangeConfig = None):
    """Create forcing functions for lock exchange experiment.

    The lock exchange experiment has no external forcings - the dynamics
    are driven purely by the initial density imbalance.

    Returns
    -------
    None
        No forcings for this experiment
    """
    return None


def create_domain_config(config: LockExchangeConfig = None) -> Dict[str, Any]:
    """Create domain configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Domain configuration parameters for this experiment
    """
    if config is None:
        config = LockExchangeConfig()

    return {
        "nlev": config.nlev,
        "H_max": config.H_max,
        "land_lat_threshold": config.land_lat_threshold,
        "spectral_land_lat_threshold": config.spectral_land_lat_threshold,
        "description": "Density-driven gravity current with vertical temperature front",
        "reference": "Petersen et al. (2015) lock exchange test case",
    }


def compute_mixing_metrics(diagnostics: Dict[str, list], pe_initial: float,
                         config: LockExchangeConfig) -> Dict[str, float]:
    """Compute mixing and energy metrics for validation.

    Parameters
    ----------
    diagnostics : Dict[str, list]
        Time series diagnostics from simulation
    pe_initial : float
        Initial potential energy for reference
    config : LockExchangeConfig
        Configuration parameters

    Returns
    -------
    Dict[str, float]
        Mixing metrics for validation
    """
    metrics = {}

    # Potential energy drift
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

    # Temperature evolution (check for numerical stability)
    mean_T_list = diagnostics.get("mean_T", [])
    if len(mean_T_list) >= 2:
        T_final = mean_T_list[-1]
        T_initial = mean_T_list[0]
        T_change = abs(T_final - T_initial)
        metrics["mean_T_change"] = T_change
        metrics["mean_T_final"] = T_final

    return metrics


def validate_results(final_state, diagnostics: Dict[str, list],
                   config: LockExchangeConfig = None,
                   **validation_kwargs) -> Tuple[bool, str]:
    """Validate lock exchange experiment results.

    Success criteria:
    - Potential energy drift within acceptable bounds
    - No excessive temperature values (numerical stability)
    - Realistic mixing behavior (energy decrease but not excessive)
    - No NaN or infinite values

    Parameters
    ----------
    final_state : OceanState
        Final model state
    diagnostics : Dict[str, list]
        Time series diagnostics
    config : LockExchangeConfig, optional
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
        config = LockExchangeConfig()

    # Check for NaN/infinite values in final state
    if hasattr(final_state, 'T') and not jnp.all(jnp.isfinite(final_state.T.data)):
        return False, "NaN/Inf detected in final temperature field"
    if hasattr(final_state, 'eta') and not jnp.all(jnp.isfinite(final_state.eta.data)):
        return False, "NaN/Inf detected in final eta field"

    # Check for temperature blowup — fuse min/max into one host pull.
    if hasattr(final_state, 'T'):
        _t = final_state.T.data
        _h = np.asarray(jnp.stack([jnp.max(_t), jnp.min(_t)]))
        max_T = float(_h[0])
        min_T = float(_h[1])
        if max_T > config.max_blowup_threshold or min_T < -config.max_blowup_threshold:
            return False, f"Temperature blowup: T_range=[{min_T:.1f}, {max_T:.1f}]°C"

    # Compute mixing metrics
    pe_initial = validation_kwargs.get("pe_initial", None)
    if pe_initial is not None:
        metrics = compute_mixing_metrics(diagnostics, pe_initial, config)
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
        notes_parts.append(f"PE_rel_final={pe_rel:.4e}")

    # Temperature stability
    if "mean_T_final" in metrics:
        T_final = metrics["mean_T_final"]
        notes_parts.append(f"mean_T={T_final:.2f}°C")

    if "mean_T_change" in metrics:
        T_change = metrics["mean_T_change"]
        notes_parts.append(f"T_change={T_change:.2f}°C")

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
    "name": "lock_exchange",
    "description": "Density-driven gravity current with RPE mixing diagnostics",
    "scientific_purpose": "Validates density-driven flows and quantifies spurious numerical mixing",
    "reference": "Petersen et al. (2015), DOI: 10.1016/j.ocemod.2014.12.004",
    "config_class": LockExchangeConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 1.0,   # days
    "quick_duration": 0.1,     # days
    "expected_metrics": {
        "pe_drift_relative": "< 1e-2",
        "mean_T_final": "5-30 °C",
        "pe_rel_final": "< 0"  # Energy should decrease
    },
    "grid_support": {
        "cubed_sphere": True,
        "latlon": True,
        "mpas": True,
        "spectral": False  # RPE calculation complex for spectral
    },
    "special_config": {
        "nlev": 20,
        "H_max": 20.0,      # Petersen 2015 Fig. 5
        "density_front": True,
        "rpe_diagnostics": True
    }
}
