"""YAML configuration adapter for legoESM.

This module is a **serialization boundary**: it reads user-facing YAML
files and converts them into the canonical ``ExperimentConfig`` consumed
by the driver.  It is *not* a second live schema — ``ExperimentConfig``
(in ``legoesm.driver.config``) is the single source of truth for runtime
configuration.

Legacy YAML field names (e.g. ``discretization: "centered"``) are
accepted for backward compatibility and normalized to canonical names
at the boundary (see ``_LEGACY_DISCRETIZATION`` below).
"""

from __future__ import annotations

import copy
import warnings

import yaml
from typing import Any


# ======================================================================
# Legacy-name normalization tables
# ======================================================================
# Canonical discretization names are defined in
# ``legoesm.atmosphere.dynamics.DISCRETIZATION_OPTIONS``.
# The YAML layer accepts deprecated aliases for backward compatibility
# and maps them to the canonical names here, at the boundary.

_LEGACY_DISCRETIZATION: dict[str, str] = {
    "centered": "cdgrid",
    "finite_volume": "cdgrid",
    "cgrid": "cdgrid",
    "fv": "cdgrid",
}

_LEGACY_DYNAMICS: dict[str, str] = {
    # No legacy aliases at this time; table is here for future use.
}


def _normalize_discretization(raw: str) -> str:
    """Map a legacy discretization name to its canonical form."""
    canonical = _LEGACY_DISCRETIZATION.get(raw)
    if canonical is not None:
        warnings.warn(
            f"YAML discretization {raw!r} is deprecated; "
            f"use {canonical!r} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return canonical
    return raw


def _normalize_dynamics(raw: str) -> str:
    """Map a legacy dynamics name to its canonical form."""
    canonical = _LEGACY_DYNAMICS.get(raw)
    if canonical is not None:
        warnings.warn(
            f"YAML dynamics {raw!r} is deprecated; "
            f"use {canonical!r} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return canonical
    return raw


# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "name": "legoESM",
        "type": "atmosphere_only",
    },
    "mode": "atmosphere",  # "atmosphere", "coupled_climate", "research_test"
    "grid": {
        "type": "cubed_sphere",
        "resolution": 48,          # N cells per face edge (C48 ~ 200km)
        "n_levels": 1,             # 1 for shallow water
        "vertical_coord": "none",  # "none" for shallow water
    },
    "atmosphere": {
        # --- Two-axis solver selection ---
        # dynamics:       "shallow_water" | "hydrostatic" | "nonhydrostatic"
        # discretization: "cdgrid" | "spectral" | "sfno" | "latlon_fv" | "mpas"
        #
        # Mapping to solver implementations:
        #   shallow_water  + cdgrid   → CDGridShallowWaterModel
        #   shallow_water  + spectral → SpectralShallowWaterModel
        #   hydrostatic    + cdgrid   → CDGridPrimitiveEquationModel
        #   hydrostatic    + spectral → SpectralPrimitiveEquationModel
        #   nonhydrostatic + cdgrid   → CDGridCompressibleEulerModel
        #   nonhydrostatic + spectral → SpectralCompressibleEulerModel
        #
        # Legacy aliases "centered", "finite_volume", "cgrid" are accepted
        # and normalized to "cdgrid" at the boundary.
        # The legacy "equations" key is still supported for backward
        # compatibility and takes precedence when set explicitly.
        "dynamics": "shallow_water",
        "discretization": "cdgrid",
        "equations": "shallow_water",   # legacy; use dynamics+discretization
        "advection": "cdgrid",
        "time_integrator": "ssp_rk3",   # "ssp_rk3" | "ssp_rk54"
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
        if user_config:  # safe_load returns None for empty files
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
        """Return a deep copy of the config data as a plain dict."""
        return copy.deepcopy(self._data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    def to_experiment_config(self):
        """Convert this YAML-based Config to an ExperimentConfig for ModelDriver.

        This is the **serialization boundary** between user-facing YAML and
        the canonical ``ExperimentConfig`` NamedTuple consumed by the driver.
        Legacy solver / discretization names are normalized here so that
        downstream code only ever sees canonical names.
        """
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
        )

        d = self._data

        grid = GridConfig(
            grid_type=d.get("grid", {}).get("type", "cubed_sphere"),
            resolution=d.get("grid", {}).get("resolution", 48),
            nlev=d.get("grid", {}).get("n_levels", 40),
            vertical_coord=d.get("grid", {}).get("vertical_coord", "hybrid"),
            p_top_Pa=d.get("grid", {}).get("p_top_Pa", 200.0),
            stretching=d.get("grid", {}).get("stretching", 2.0),
        )

        atm = d.get("atmosphere", {})

        # Normalize legacy names at the boundary
        raw_dynamics = atm.get("dynamics", "hydrostatic")
        raw_disc = atm.get("discretization", "cdgrid")
        dynamics = _normalize_dynamics(raw_dynamics)
        discretization = _normalize_discretization(raw_disc)

        dycore = DycoreConfig(
            model_type=dynamics,
            discretization=discretization,
            dt=float(atm.get("dt_seconds", 600)),
            hyperdiff_scale=float(atm.get("hyperdiffusion_coeff", 1.0)),
            conservation_fixer=d.get("conservation", {}).get("fix_mass", True),
            fix_mass=d.get("conservation", {}).get("fix_mass", True),
        )

        time_cfg = d.get("time", {})
        output_cfg = d.get("output", {})
        output = OutputConfig(
            output_dir=output_cfg.get("path", ""),
            diag_days=max(1, int(time_cfg.get("output_interval_hours", 6) / 24)),
            checkpoint_days=int(output_cfg.get("checkpoint_days", 0)),
            monthly_means=bool(output_cfg.get("monthly_means", False)),
            cmip_output=bool(output_cfg.get("cmip_output", False)),
            clear_sky_diag=bool(output_cfg.get("clear_sky_diag", False)),
            checkpoint_format=output_cfg.get("checkpoint_format", "npz"),
        )

        # Integration time
        duration_hours = time_cfg.get("duration_hours", 120)
        days = int(duration_hours / 24)

        # Build ExperimentConfig with available overrides
        forcing = d.get("forcing", {})
        radiation = d.get("radiation", {})

        kwargs = dict(
            grid=grid,
            dycore=dycore,
            output=output,
            days=days,
            start_day=float(time_cfg.get("start_day", 0.0)),
            dataset=forcing.get("dataset", "analytical"),
            forcing_path=forcing.get("path", ""),
            radiation=radiation.get("scheme", atm.get("radiation", "gray")),
            T_init=float(d.get("surface", {}).get("T_init", 300.0)),
            RH_init=float(d.get("surface", {}).get("RH_init", 0.7)),
            distributed=bool(d.get("hardware", {}).get("parallelism", {}).get("distributed", False)),
        )

        return ExperimentConfig(**kwargs)

    def __repr__(self) -> str:
        return f"Config({self._data})"


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
