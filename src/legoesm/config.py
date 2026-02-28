"""Configuration system for legoESM.

Supports YAML-based configuration with sensible defaults
and programmatic override.
"""

from __future__ import annotations

import copy

import yaml
from typing import Any


# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "name": "legoESM",
        "type": "atmosphere_only",
    },
    "grid": {
        "type": "cubed_sphere",
        "resolution": 48,          # N cells per face edge (C48 ~ 200km)
        "n_levels": 1,             # 1 for shallow water
        "vertical_coord": "none",  # "none" for shallow water
    },
    "atmosphere": {
        # --- Two-axis solver selection ---
        # dynamics:       "shallow_water" | "hydrostatic" | "nonhydrostatic"
        # discretization: "finite_volume" | "spectral"
        #
        # Mapping to solver implementations:
        #   shallow_water  + finite_volume → ShallowWaterModel
        #   shallow_water  + spectral      → SpectralShallowWaterModel
        #   hydrostatic    + finite_volume → PrimitiveEquationModel
        #   nonhydrostatic + finite_volume → CompressibleEulerModel
        #
        # The legacy "equations" key is still supported for backward
        # compatibility and takes precedence when set explicitly.
        "dynamics": "shallow_water",
        "discretization": "finite_volume",
        "equations": "shallow_water",   # legacy; use dynamics+discretization
        "advection": "centered",
        "time_integrator": "ssp_rk3",
        "dt_seconds": 600,          # 10 minutes
        "hyperdiffusion_coeff": 0.0,
        "spectral": {
            "allow_unsupported": False,  # bypass Metal backend guard
        },
        "tracer_transport": {
            "n_tracers": 4,
            "hyperdiffusion_coeff": 0.0,
        },
        "nonhydrostatic": {
            "n_acoustic_substeps": 6,
            "acoustic_off_centering": 0.5,
            "sponge_width_m": 10000.0,
            "sponge_coeff": 0.05,
            "small_earth_factor": 1.0,
            "model_top_m": 40000.0,
        },
    },
    "conservation": {
        "fix_mass": True,
        "fix_energy": True,
    },
    "time": {
        "duration_hours": 120,     # 5 days
        "output_interval_hours": 6,
    },
    "output": {
        "format": "zarr",
        "path": "output/",
    },
    "hardware": {
        "precision": {
            "dynamics": "float32",
            "ml": "bfloat16",
            "conservation": "float64",
        },
        "devices": "auto",
        "parallelism": {
            "n_devices": "auto",
            "backend": None,
            "distributed": False,
        },
    },
}


class Config:
    """Configuration container with dot-access and YAML support."""

    def __init__(self, data: dict | None = None):
        self._data = data or copy.deepcopy(DEFAULT_CONFIG)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            user_config = yaml.safe_load(f)
        config = copy.deepcopy(DEFAULT_CONFIG)
        _deep_merge(config, user_config)
        return cls(config)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        """Create configuration from a dictionary."""
        config = copy.deepcopy(DEFAULT_CONFIG)
        _deep_merge(config, d)
        return cls(config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value using dot notation: config.get('grid.resolution')."""
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        """Set a value using dot notation."""
        keys = key.split(".")
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def to_dict(self) -> dict:
        return dict(self._data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"Config({self._data})"


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
