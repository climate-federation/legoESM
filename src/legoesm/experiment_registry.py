"""Experiment config-adapter registry — mode-aware dispatch for the experiment
tooling (``init_experiment``, ``validate_templates``).

The atmosphere YAML boundary (:class:`legoesm.config.Config`) and the ocean YAML
boundary (:class:`legoesm.ocean.config.OceanExperimentConfig`) implement the same
*adapter protocol* (see :class:`ConfigAdapter` below).  This module maps an
experiment ``mode`` / ``model.type`` to the right adapter class so the
meta-package tooling can resolve, validate, and materialize any component's
experiment from one code path.

Federation
----------
This registry lives in the meta-package (which may depend on the component
packages) and imports each adapter **lazily**, only when its mode is requested —
so ``legoesm-ocean`` never has to import the meta-package to "register" itself
(that would invert the federation DAG).  Adding ``legoesm-land`` / ``legoesm-ice``
later is a one-line entry here, not a cross-package import.

Adapter protocol
----------------
Every adapter class provides::

    @classmethod from_yaml(path) -> adapter
    get(key, default=None) -> Any            # dot notation
    set(key, value) -> None                  # dot notation
    to_yaml(path) -> None
    get_meta() -> dict                        # the experiment: block
    signature() -> str                        # resolved-config signature (no-op detect)
    validate_strict() -> None                 # raise on invalid resolved config
    run_command() -> str                      # launcher line for run.sh
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ConfigAdapter(Protocol):
    """Structural type implemented by every experiment config adapter."""

    @classmethod
    def from_yaml(cls, path: str) -> "ConfigAdapter": ...
    def get(self, key: str, default=None): ...
    def set(self, key: str, value) -> None: ...
    def to_yaml(self, path: str) -> None: ...
    def get_meta(self) -> dict: ...
    def signature(self) -> str: ...
    def validate_strict(self) -> None: ...
    def run_command(self) -> str: ...


# mode / model.type  ->  (module path, class name).  Lazy import keeps the
# federation DAG intact (meta -> component, never the reverse).
_REGISTRY: dict[str, tuple[str, str]] = {
    # Atmosphere + coupled experiments use the canonical atmosphere boundary
    # (ExperimentConfig handles the coupled climate path too).
    "atmosphere": ("legoesm.config", "Config"),
    "atmosphere_only": ("legoesm.config", "Config"),
    "coupled_climate": ("legoesm.config", "Config"),
    "research_test": ("legoesm.config", "Config"),
    # Ocean-only experiments use the ocean boundary.
    "ocean": ("legoesm.ocean.config", "OceanExperimentConfig"),
    "ocean_only": ("legoesm.ocean.config", "OceanExperimentConfig"),
    # Standalone idealized sea-ice experiments (#388): setup-only adapter
    # routing to run_sea_ice_test_matrix.py.
    "sea_ice": ("legoesm.ice.experiment_config", "SeaIceExperimentConfig"),
    "sea_ice_only": ("legoesm.ice.experiment_config", "SeaIceExperimentConfig"),
}

# Modes that the ``validate_templates`` atmosphere-specific complexity check
# (declared complexity rung must equal the resolved dycore.model_type) applies
# to.  Ocean templates use a different complexity vocabulary, so they opt out.
_ATMOSPHERE_MODES = frozenset(
    {"atmosphere", "atmosphere_only", "coupled_climate", "research_test"}
)


def known_modes() -> tuple[str, ...]:
    """All registered experiment modes."""
    return tuple(_REGISTRY)


def is_atmosphere_mode(mode: str) -> bool:
    """True if *mode* resolves through the atmosphere boundary."""
    return mode in _ATMOSPHERE_MODES


def detect_mode(raw: dict) -> str:
    """Infer the experiment mode from a raw (unmerged) YAML dict.

    Dispatch precedence: explicit ``model.type`` > top-level ``mode`` > a top
    ``ocean:`` section without ``atmosphere:`` > default ``"atmosphere"``.
    Always returns a *registered* mode (raises ``ValueError`` on an explicit but
    unknown mode/type so a typo fails loudly instead of silently defaulting to
    atmosphere — dispatch discipline).
    """
    model_type = (raw.get("model") or {}).get("type")
    if model_type is not None:
        if model_type not in _REGISTRY:
            raise ValueError(
                f"model.type={model_type!r} is not a known experiment mode; "
                f"valid: {sorted(_REGISTRY)}"
            )
        return model_type
    mode = raw.get("mode")
    if mode is not None:
        if mode not in _REGISTRY:
            raise ValueError(
                f"mode={mode!r} is not a known experiment mode; "
                f"valid: {sorted(_REGISTRY)}"
            )
        return mode
    # No explicit selector: infer from section presence (ocean section but no
    # atmosphere section => ocean), else default to atmosphere (back-compat).
    if "ocean" in raw and "atmosphere" not in raw:
        return "ocean"
    return "atmosphere"


def get_adapter(mode: str):
    """Return the adapter *class* registered for *mode*."""
    try:
        module_path, class_name = _REGISTRY[mode]
    except KeyError:
        raise ValueError(
            f"no config adapter registered for mode {mode!r}; "
            f"valid: {sorted(_REGISTRY)}"
        )
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# Template ids (relative to config/templates, no .yaml) that were renamed or
# removed; resolving one raises with a pointer instead of "not found".
RETIRED_TEMPLATES: dict[str, str] = {
    "coupled/amip": (
        "template 'coupled/amip' was renamed to "
        "'3d_idealized/hydrostatic_gray_1yr': it never ran AMIP (analytical "
        "forcing, gray radiation).  For AMIP use "
        "config/amip/amip_production.yaml"
    ),
}


def retired_template_message(template: str) -> str | None:
    """Migration message if *template* (an id or a path) is retired."""
    rel = str(template).replace("\\", "/")
    rel = rel[:-5] if rel.endswith(".yaml") else rel
    rel = rel.rsplit("templates/", 1)[-1]
    return RETIRED_TEMPLATES.get(rel)


def load_adapter(path: str):
    """Read *path*'s mode, then load it through the matching adapter.

    The raw YAML is parsed once to detect the mode (``model.type`` / ``mode`` /
    section heuristics); the adapter then re-reads and merges onto its own
    defaults.  Returns a ``(mode, adapter_instance)`` tuple.
    """
    import os

    import yaml

    retired = retired_template_message(path)
    if retired is not None and not os.path.isfile(path):
        raise ValueError(retired)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    mode = detect_mode(raw)
    adapter_cls = get_adapter(mode)
    return mode, adapter_cls.from_yaml(path)
