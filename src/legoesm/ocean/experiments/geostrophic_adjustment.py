"""Geostrophic Adjustment Ocean Experiment.

Tests the geostrophic adjustment process following the introduction of a
meridional temperature front. The ocean starts from rest with a thermal
wind imbalance and adjusts toward geostrophic equilibrium.

This experiment is closely related to the baroclinic adjustment experiment
but is tracked separately because the test matrix uses it for a distinct
validation scenario (shorter integration, different diagnostics).

Scientific Purpose:
- Validate geostrophic adjustment from a thermal wind imbalance
- Test the development of circulation from rest
- Verify conservation during adjustment
- Benchmark drift metrics against rest-state baseline

Domain Configuration:
- Global ocean with land at high latitudes (|lat| > 80°)
- Uniform depth: 5500m in ocean regions
- Rest state stratification as background
- Meridional temperature front: ±5°C gradient, decaying with depth

Physical Setup:
- Meridional temperature gradient: +5°C * cos(lat)
- Exponential decay with depth: exp(-k / decay_scale)
- No external forcing — adjustment is driven purely by initial imbalance
- No initial velocity or SSH perturbation

Expected Behavior:
- Development of thermal wind circulation
- Geostrophic adjustment of velocity field
- Conservation of heat and salinity
- Minimal SSH drift

References:
- Gill (1982), "Atmosphere-Ocean Dynamics"
- Pedlosky (1987), "Geophysical Fluid Dynamics"
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from legoesm.constants import g
from legoesm.core.field import Field


@dataclass
class GeostrophicAdjustmentConfig:
    """Configuration for geostrophic adjustment experiment.

    Reuses the same physics as the baroclinic experiment but is tracked
    as a separate experiment for distinct validation purposes.
    """
    # Background state
    T_water_init_C: float = 20.0        # Surface temperature [°C]
    T_deep: float = 2.0            # Deep ocean temperature [°C]
    scale_depth: float = 1000.0    # Temperature e-folding depth [m]
    S_uniform: float = 35.0        # Salinity [PSU]

    # Domain configuration
    H_max: float = 5500.0          # Maximum ocean depth [m]
    land_lat_threshold: float = 80.0
    spectral_land_lat_threshold: float = 90.0

    # Temperature front parameters
    T_perturbation_amplitude: float = 5.0  # [°C]
    vertical_decay_scale: float = 3.0      # e-folding scale in levels


def create_initial_conditions(grid_type: str, grid, z_coord,
                              config: GeostrophicAdjustmentConfig = None):
    """Create geostrophic adjustment initial conditions.

    Delegates to the baroclinic experiment's IC creation since the
    physical setup is identical (rest state + meridional temperature front).
    """
    if config is None:
        config = GeostrophicAdjustmentConfig()

    from legoesm.ocean.experiments.baroclinic import (
        BaroclinicConfig, create_initial_conditions as baroclinic_ic,
    )
    bc_config = BaroclinicConfig(
        T_water_init_C=config.T_water_init_C,
        T_deep=config.T_deep,
        scale_depth=config.scale_depth,
        S_uniform=config.S_uniform,
        H_max=config.H_max,
        land_lat_threshold=config.land_lat_threshold,
        spectral_land_lat_threshold=config.spectral_land_lat_threshold,
        T_perturbation_amplitude=config.T_perturbation_amplitude,
        vertical_decay_scale=config.vertical_decay_scale,
    )
    return baroclinic_ic(grid_type, grid, z_coord, bc_config)


def create_forcings(grid_type: str, grid,
                    config: GeostrophicAdjustmentConfig = None):
    """No external forcings — adjustment is driven by initial imbalance."""
    return None


def create_domain_config(config: GeostrophicAdjustmentConfig = None) -> Dict[str, Any]:
    if config is None:
        config = GeostrophicAdjustmentConfig()
    return {
        "H_max": config.H_max,
        "land_lat_threshold": config.land_lat_threshold,
        "description": "Global ocean with meridional temperature front for geostrophic adjustment",
    }


def validate_results(final_state, diagnostics: Dict[str, list],
                     config: GeostrophicAdjustmentConfig = None) -> Tuple[bool, str]:
    """Validate geostrophic adjustment results.

    Uses the same validation logic as the baroclinic experiment.
    """
    if config is None:
        config = GeostrophicAdjustmentConfig()

    from legoesm.ocean.experiments.baroclinic import (
        BaroclinicConfig, validate_results as baroclinic_validate,
    )
    bc_config = BaroclinicConfig(
        T_water_init_C=config.T_water_init_C,
        T_deep=config.T_deep,
        H_max=config.H_max,
    )
    return baroclinic_validate(final_state, diagnostics, bc_config)


def get_diagnostic_field_specs() -> list:
    return [
        ("SST", "SST (degC)", "RdYlBu_r"),
        ("eta", "SSH (m)", "RdBu_r"),
        ("speed_sfc", "Surface Speed (m/s)", "plasma"),
    ]


def get_scalar_units() -> Dict[str, str]:
    return {
        "mean_eta": "m",
        "max_speed": "m/s",
        "max_abs_u": "m/s",
        "mean_T": "degC",
        "mean_S": "PSU",
    }


EXPERIMENT_CONFIG = {
    "name": "geostrophic_adjustment",
    "description": "Geostrophic adjustment from meridional temperature front",
    "scientific_purpose": "Validates thermal wind adjustment and conservation from rest",
    "reference": "Gill (1982), Pedlosky (1987)",
    "config_class": GeostrophicAdjustmentConfig,
    "create_initial_conditions": create_initial_conditions,
    "create_forcings": create_forcings,
    "create_domain": create_domain_config,
    "validate": validate_results,
    "get_field_specs": get_diagnostic_field_specs,
    "get_scalar_units": get_scalar_units,
    "default_duration": 10.0,
    "quick_duration": 1.0,
    "expected_metrics": {
        "T_drift_relative": "< 1e-3",
        "max_speed_final": "0.001-1.0 m/s",
    },
    "grid_support": {
        "cubed_sphere": True,
        "latlon": True,
        "mpas": True,
        "spectral": True,
    },
}
