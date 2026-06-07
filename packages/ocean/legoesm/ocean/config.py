"""YAML configuration adapter for the legoESM ocean component.

This module is the ocean analogue of :mod:`legoesm.config` (the atmosphere
``Config``): it is a **serialization boundary** that reads user-facing ocean
YAML templates and converts them into the canonical runtime ocean config
NamedTuples (:class:`~legoesm.ocean.state.LatLonCGridOceanConfig`,
:class:`~legoesm.ocean.state.OceanConfig`,
:class:`~legoesm.ocean.state.SpectralOceanConfig`).  It is *not* a second live
schema — those NamedTuples remain the single source of truth for runtime
configuration; this layer only handles YAML parsing, dot-notation overrides,
typo detection, and strict validation.

Naming note
-----------
The issue (#376) proposed naming this adapter ``OceanConfig``, but that name is
already taken by the cubed-sphere runtime config NamedTuple in
:mod:`legoesm.ocean.state`.  To avoid shadowing the runtime type the adapter is
named :class:`OceanExperimentConfig` (parallel to the atmosphere ``Config``
boundary object, and consistent with ``ExperimentConfig`` being the canonical
runtime config).

Federation note
---------------
This adapter lives in ``legoesm-ocean`` (not the meta-package) to respect the
federation DAG: the meta-package ``init_experiment`` dispatches to it through the
``legoesm.experiment_registry`` table, which lazily imports this module only when
an ocean experiment is requested (meta -> ocean is the allowed direction).

YAML schema
-----------
The ocean template uses the same ``experiment:`` metadata block as atmosphere
templates plus ocean-specific sections::

    experiment: {tier, complexity, extent, maturity, description, data, ...}
    model:   {name, type: ocean_only}
    grid:    {type: latlon_cgrid|cubed_sphere|spectral, n_lat, n_lon,
              resolution, nlev}
    ocean:   {<LatLonCGridOceanConfig/OceanConfig field>: value, ...}   # flat
    time:    {dt_seconds, duration_days, output_interval_days}
    forcing: {dataset, path}
    output:  {path, format}

The ``ocean:`` section maps directly onto the runtime NamedTuple field names
(no rename table — that keeps the boundary drift-free and lets the NamedTuple
``_fields`` set drive typo detection).  ``eos: linear`` reads an optional
``eos_linear:`` sub-dict into a :class:`~legoesm.ocean.eos.LinearEOSConfig`.
"""

from __future__ import annotations

import copy
from typing import Any

import yaml

# Grid-type -> runtime config NamedTuple + the model whose _validate_config
# performs the canonical (single-source) strict validation.  Imports are
# deferred to the methods that need them so importing this boundary module stays
# cheap and does not pull the full dynamics stack at package-import time.
_GRID_TYPES = ("latlon_cgrid", "cubed_sphere", "spectral")


# Default ocean experiment config.  ``ocean: {}`` means "use every runtime
# NamedTuple default" — a bare template runs an Earth-default lat-lon C-grid
# ocean.  Mirrors the structure of ``legoesm.config.DEFAULT_CONFIG``.
DEFAULT_OCEAN_CONFIG: dict = {
    "model": {
        "name": "legoESM-ocean",
        "type": "ocean_only",
    },
    "mode": "ocean",
    "grid": {
        "type": "latlon_cgrid",
        "n_lat": 90,
        "n_lon": 180,
        "nlev": 15,
    },
    "ocean": {},              # flat overrides onto the runtime config NamedTuple
    "time": {
        "dt_seconds": 3600,
        "duration_days": 30,
        "output_interval_days": 5,
    },
    "output": {
        "format": "zarr",
        "path": "output/",
    },
}


def _resolve_target(grid_type: str):
    """Return ``(ConfigClass, ModelClass)`` for *grid_type* (deferred imports)."""
    if grid_type == "latlon_cgrid":
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        return LatLonCGridOceanConfig, LatLonCGridOceanModel
    if grid_type == "cubed_sphere":
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        return OceanConfig, OceanModel
    if grid_type == "spectral":
        from legoesm.ocean.state import SpectralOceanConfig
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
        return SpectralOceanConfig, SpectralOceanModel
    # Dispatch discipline: never silently default an unknown grid type.
    raise ValueError(
        f"ocean grid.type must be one of {_GRID_TYPES}, got {grid_type!r}"
    )


class OceanExperimentConfig:
    """Ocean YAML configuration container with dot-access and YAML support.

    Parallel to :class:`legoesm.config.Config` for the atmosphere; implements the
    same adapter protocol consumed by ``init_experiment`` (``from_yaml``,
    ``get``/``set``, ``to_yaml``, ``signature``, ``validate_strict``,
    ``get_meta``, ``run_command``).
    """

    def __init__(self, data: dict | None = None):
        self._data = data or copy.deepcopy(DEFAULT_OCEAN_CONFIG)

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_yaml(cls, path: str) -> "OceanExperimentConfig":
        """Load an ocean configuration from a YAML file (merged onto defaults)."""
        with open(path, "r") as f:
            user_config = yaml.safe_load(f)
        config = copy.deepcopy(DEFAULT_OCEAN_CONFIG)
        if user_config:  # safe_load returns None for empty files
            _deep_merge(config, user_config)
        return cls(config)

    @classmethod
    def from_dict(cls, d: dict) -> "OceanExperimentConfig":
        """Create an ocean configuration from a dictionary (merged onto defaults)."""
        config = copy.deepcopy(DEFAULT_OCEAN_CONFIG)
        _deep_merge(config, d)
        return cls(config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value using dot notation: ``cfg.get('grid.nlev')``."""
        val = self._data
        for k in key.split("."):
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        """Set a value using dot notation: ``cfg.set('ocean.A_h', 3e4)``."""
        keys = key.split(".")
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def to_dict(self) -> dict:
        """Return a deep copy of the config data as a plain dict."""
        return copy.deepcopy(self._data)

    def to_yaml(self, path: str) -> None:
        """Save the configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    def get_meta(self) -> dict:
        """Return the ``experiment:`` metadata block (or empty dict)."""
        return self.get("experiment") or {}

    # ------------------------------------------------------- runtime mapping
    def to_ocean_config(self):
        """Translate this YAML config into the canonical runtime ocean config.

        Returns one of :class:`~legoesm.ocean.state.LatLonCGridOceanConfig`,
        :class:`~legoesm.ocean.state.OceanConfig`, or
        :class:`~legoesm.ocean.state.SpectralOceanConfig` depending on
        ``grid.type``.  The ``ocean:`` YAML section maps directly onto the
        NamedTuple field names; any key not in ``ConfigClass._fields`` is a typo
        and raises (rather than being silently dropped).
        """
        grid_type = self.get("grid.type", "latlon_cgrid")
        ConfigClass, _ = _resolve_target(grid_type)

        ocean = dict(self.get("ocean") or {})
        known = set(ConfigClass._fields)
        unknown = sorted(k for k in ocean if k not in known)
        if unknown:
            raise ValueError(
                f"unknown ocean config field(s) for grid.type={grid_type!r}: "
                f"{unknown}. Valid fields: {sorted(known)}"
            )

        # Nested-config sections that the flat boundary builds explicitly.  The
        # full OceanPhysicsConfig / GMRediConfig pipelines are not yet expressible
        # in YAML (they are deep nested NamedTuples); reject them loudly rather
        # than silently ignoring, so a user is never misled into thinking a
        # physics block took effect.
        for nested in ("physics", "gm_redi"):
            if isinstance(ocean.get(nested), dict):
                raise ValueError(
                    f"ocean.{nested} (nested physics pipeline) is not yet "
                    "configurable via YAML; construct it in Python and pass the "
                    "runtime config directly. Scalar fields are supported."
                )

        eos_linear = ocean.pop("eos_linear", None)
        if eos_linear is not None:
            from legoesm.ocean.eos import LinearEOSConfig
            if not isinstance(eos_linear, dict):
                raise ValueError("ocean.eos_linear must be a mapping")
            le_known = set(LinearEOSConfig._fields)
            le_unknown = sorted(k for k in eos_linear if k not in le_known)
            if le_unknown:
                raise ValueError(
                    f"unknown ocean.eos_linear field(s): {le_unknown}. "
                    f"Valid fields: {sorted(le_known)}"
                )
            ocean["eos_linear"] = LinearEOSConfig(**eos_linear)

        return ConfigClass(**ocean)

    # ------------------------------------------------------- validation hooks
    def validate_strict(self) -> None:
        """Strict-validate the resolved ocean config (raises on invalid).

        Delegates to the runtime model's ``_validate_config`` static method —
        the single canonical validator for each grid (EOS / advection /
        barotropic-solver membership, bound checks) — so the valid sets are
        never duplicated at the YAML boundary.
        """
        grid_type = self.get("grid.type", "latlon_cgrid")
        _, ModelClass = _resolve_target(grid_type)
        cfg = self.to_ocean_config()
        # _validate_config is a @staticmethod that takes only the config (no grid
        # / state), the same check the model runs in __init__.
        ModelClass._validate_config(cfg)

        # Experiment-level (non-runtime-config) bounds on the time block.
        dt = self.get("time.dt_seconds")
        if dt is not None and float(dt) <= 0:
            raise ValueError(f"time.dt_seconds must be > 0, got {dt}")
        dur = self.get("time.duration_days")
        if dur is not None and float(dur) <= 0:
            raise ValueError(f"time.duration_days must be > 0, got {dur}")

    def signature(self) -> str:
        """Deterministic signature of the RESOLVED runtime config.

        Used by ``init_experiment`` to detect overrides that don't actually
        change the run (a misspelled or non-runtime dot-path). If the config is
        unresolvable the exception text is folded in so before/after still
        differ (a no-op is only flagged on a byte-identical resolved config).
        """
        try:
            # Fold in the time block too: it affects the run but lives outside
            # the runtime NamedTuple, so dt/duration overrides are not no-ops.
            return repr((self.to_ocean_config(), self.get("time")))
        except Exception as exc:  # noqa: BLE001
            return f"<unresolvable: {type(exc).__name__}: {exc}>"

    def run_command(self) -> str:
        """Launcher command for the generated ``run.sh`` (ocean runner)."""
        return "python scripts/run/run_omip_core2.py --config config.yaml"

    def __repr__(self) -> str:
        return f"OceanExperimentConfig({self._data})"


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* (in-place).

    Local copy (not imported from ``legoesm.config``) because that module lives
    in the meta-package and ``legoesm-ocean`` must not depend on it (federation
    DAG). Identical semantics to the atmosphere boundary's ``_deep_merge``.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
