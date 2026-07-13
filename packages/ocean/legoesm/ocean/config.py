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
import shlex
from typing import Any, NamedTuple

import yaml

# Grid-type -> runtime config NamedTuple + the model whose _validate_config
# performs the canonical (single-source) strict validation.  Imports are
# deferred to the methods that need them so importing this boundary module stays
# cheap and does not pull the full dynamics stack at package-import time.
_GRID_TYPES = ("latlon_cgrid", "cubed_sphere", "spectral")

# Map the YAML ``grid.type`` to the ``run_omip_core2.py --grid`` backend for the
# generated run.sh.  Only ``latlon_cgrid`` is wired end-to-end through the omip
# ``--config`` workflow: that path is the only one ``run_omip_core2.py --config``
# applies ``ocean.*`` overrides to (the cube ocean is parked/resolution-limited
# and spectral has no OMIP runner).  A template on any other grid therefore has
# NO runnable bundle — ``run_command`` raises so init_experiment fails fast at
# materialization rather than emitting a run.sh that dies at runtime.
_GRID_TYPE_TO_RUNNER = {
    "latlon_cgrid": "latlon_bathy",
}

# Allowed keys in the optional top-level ``setup:`` block (#388 Ask#2).  A
# template that sets ``setup:`` names one of the procedural idealized ocean
# experiments in ``legoesm.ocean.experiments`` (the "setup" axis), keeping the
# case content in its own section distinct from the ``ocean:`` physics "recipe".
# Such a template routes to the matrix runner (the existing, sole consumer of
# AVAILABLE_EXPERIMENTS) rather than the OMIP ``run_omip_core2.py`` path.
_OCEAN_MATRIX_RUNNER = "scripts/matrix/run_ocean_test_matrix.py"
# Mirror of ``run_ocean_test_matrix.py``'s ``--grid`` argparse choices (minus
# ``all``).  Kept as a small, stable local set so a ``setup.grid`` typo fails at
# the YAML boundary even for an experiment that declares no ``grid_support``
# (which would otherwise only die later in the matrix runner's argparse).  The
# per-template (name, grid) -> concrete case existence is verified against the
# live matrix catalog by tests/unit/test_ocean_setup_templates.py.
_MATRIX_GRIDS = (
    "cubed_sphere", "latlon", "mpas", "mpas_regional", "latlon_regional",
    "cs_regional", "latlon_channel", "mpas_channel", "spectral",
)


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


def _ocean_matrix_spec():
    """MatrixRunnerSpec for an ocean ``setup:`` template (#388).

    Ocean's case-name flag is ``--only`` with the ``=`` exact-match prefix; the
    runner accepts every per-run override flag.  Shared with the atmosphere /
    sea-ice adapters via :mod:`legoesm.core.setup_selector` (deferred import:
    keep this boundary module cheap).
    """
    from legoesm.core.setup_selector import MatrixRunnerSpec
    return MatrixRunnerSpec(runner_path=_OCEAN_MATRIX_RUNNER,
                            valid_grids=_MATRIX_GRIDS)


def _resolve_target(grid_type: str):
    """Return ``(ConfigClass, ModelClass)`` for *grid_type* (deferred imports)."""
    if grid_type == "latlon_cgrid":
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.state import LatLonCGridOceanConfig
        return LatLonCGridOceanConfig, LatLonCGridOceanModel
    if grid_type == "cubed_sphere":
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.state import OceanConfig
        return OceanConfig, OceanModel
    if grid_type == "spectral":
        from legoesm.ocean.dynamics.spectral_ocean_pe import SpectralOceanModel
        from legoesm.ocean.state import SpectralOceanConfig
        return SpectralOceanConfig, SpectralOceanModel
    # Dispatch discipline: never silently default an unknown grid type.
    raise ValueError(
        f"ocean grid.type must be one of {_GRID_TYPES}, got {grid_type!r}"
    )


# ---------------------------------------------------------------------------
# Nested physics-pipeline YAML mapping (#382)
# ---------------------------------------------------------------------------
# The runtime config exposes three deeply-nested NamedTuple trees as top-level
# fields annotated ``object`` (``eos_linear``/``gm_redi``/``physics`` — the
# annotation carries no type, so it cannot drive reconstruction).  These are the
# explicit YAML entry points; every level BELOW them is reconstructed generically
# from the sub-field types (concrete annotations / NamedTuple defaults), so a new
# scheme or sub-config becomes YAML-expressible with no change here.


def _nested_ocean_entry_types() -> dict:
    """``{yaml_key: NamedTuple type}`` for the object-typed nested entry points."""
    from legoesm.ocean.eos import LinearEOSConfig
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.physics.lateral_mixing.backscatter import BackscatterConfig
    from legoesm.ocean.physics.tidal_forcing import TidalForcingConfig
    return {
        "eos_linear": LinearEOSConfig,
        "gm_redi": GMRediConfig,
        "physics": OceanPhysicsConfig,
        # Astronomical (equilibrium) tidal forcing: a nested entry point (its
        # `enabled`/`love_factor`/… member names are too generic to flatten onto
        # the ocean.* namespace) so `ocean.tidal_forcing: {enabled: true, …}`
        # builds a real TidalForcingConfig instead of passing a raw dict through.
        "tidal_forcing": TidalForcingConfig,
        # Jansen–Held energy backscatter: opt-in nested entry point (default
        # None ⇒ off/bit-identical), so `ocean.backscatter: {enabled: true,
        # c_bs: 0.01, …}` builds a real BackscatterConfig.
        "backscatter": BackscatterConfig,
    }


def _is_config_namedtuple(t) -> bool:
    """True if *t* is a NamedTuple subclass (a nestable config type)."""
    return (
        isinstance(t, type)
        and issubclass(t, tuple)
        and hasattr(t, "_fields")
        and hasattr(t, "_field_defaults")
    )


def _nested_config_type(cls, field: str):
    """The NamedTuple sub-config type for ``cls.<field>`` (or ``None``).

    Prefers the runtime default's type (robust even when the field is annotated
    ``object``); falls back to the field annotation for Optional sub-configs
    whose default is ``None`` (e.g. ``eke: EKEConfig | None = None``).
    """
    import typing

    default = cls._field_defaults.get(field)
    if _is_config_namedtuple(type(default)):
        return type(default)
    try:
        ann = typing.get_type_hints(cls).get(field)
    except Exception:
        return None
    if _is_config_namedtuple(ann):
        return ann
    for arg in typing.get_args(ann):           # Optional[X] / Union[X, None]
        if _is_config_namedtuple(arg):
            return arg
    return None


def _field_allows_none(cls, field: str) -> bool:
    """True if ``cls.<field>`` is genuinely Optional (``X | None``) — so an
    explicit YAML ``null`` legitimately disables the sub-config rather than
    smuggling a ``None`` into a required slot the factory later dereferences."""
    import typing

    if cls._field_defaults.get(field) is None:   # default None ⇒ Optional in practice
        return True
    try:
        ann = typing.get_type_hints(cls).get(field)
    except Exception:
        return False
    return type(None) in typing.get_args(ann)


def _build_nested_config(cls, d: dict, *, path: str):
    """Recursively build NamedTuple *cls* from mapping *d*.

    Rejects unknown fields at EVERY level (``path`` is the dotted ``ocean.*``
    location for the error message), and recurses into nested sub-config fields
    whose YAML value is a mapping — the same typo-detection contract the flat
    ``ocean:`` boundary uses, applied all the way down the physics tree.
    """
    if not isinstance(d, dict):
        raise ValueError(
            f"ocean.{path} must be a mapping, got {type(d).__name__}")
    known = set(cls._fields)
    unknown = sorted(k for k in d if k not in known)
    if unknown:
        raise ValueError(
            f"unknown {cls.__name__} field(s) in the ocean scheme config at "
            f"ocean.{path}: {unknown}. Valid fields: {sorted(known)}")
    kwargs = {}
    for key, val in d.items():
        sub_type = _nested_config_type(cls, key)
        if sub_type is None:
            kwargs[key] = val                       # plain scalar/list/None field
        elif isinstance(val, dict):
            kwargs[key] = _build_nested_config(
                sub_type, val, path=f"{path}.{key}")
        elif val is None and _field_allows_none(cls, key):
            kwargs[key] = None                      # explicit null disables an Optional sub-config
        else:
            # A sub-config position given a scalar/list (or a null on a REQUIRED
            # sub-config) would build e.g. OceanPhysicsConfig(vertical_mixing=5
            # or None) — silently invalid, crashing later in the factory when it
            # dereferences .scheme/.enabled.  Fail fast at the boundary instead.
            _allowed = ("a mapping (a {0} sub-config) or null".format(
                sub_type.__name__) if _field_allows_none(cls, key)
                else "a mapping (a {0} sub-config)".format(sub_type.__name__))
            raise ValueError(
                f"ocean.{path}.{key} must be {_allowed}, "
                f"got {type(val).__name__}")
    return cls(**kwargs)


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
    def from_yaml(cls, path: str) -> OceanExperimentConfig:
        """Load an ocean configuration from a YAML file (merged onto defaults)."""
        with open(path) as f:
            user_config = yaml.safe_load(f)
        config = copy.deepcopy(DEFAULT_OCEAN_CONFIG)
        if user_config:  # safe_load returns None for empty files
            _deep_merge(config, user_config)
        return cls(config)

    @classmethod
    def from_dict(cls, d: dict) -> OceanExperimentConfig:
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
        # #501: a config with nested sub-configs (LatLonCGridOceanConfig) accepts
        # its members' FLAT names as ocean.* keys, kept 1:1 via flat_fields() so
        # the public flat YAML interface is unchanged by the grouping.
        known = (set(ConfigClass.flat_fields())
                 if hasattr(ConfigClass, "flat_fields")
                 else set(ConfigClass._fields))
        unknown = sorted(k for k in ocean if k not in known)
        if unknown:
            raise ValueError(
                f"unknown ocean config field(s) for grid.type={grid_type!r} "
                f"(backend {ConfigClass.__name__}): {unknown}. "
                f"Valid fields: {sorted(known)}"
            )

        # Lat-lon C-grid: the physics lateral-mixing factory is cubed-sphere-only
        # (lateral mixing is selected via the flat ocean.A_h/B_h + the top-level
        # ocean.gm_redi), so the only valid physics.lateral_mixing.scheme is
        # "none".  The runtime OceanPhysicsConfig default is "harmonic" (the cube
        # value), which would make a minimal lat-lon physics block (e.g. only
        # vertical_mixing.kpp) fail validate_strict.  Default the scheme to "none"
        # here UNLESS the user set it explicitly (an explicit "harmonic" still
        # reaches validate_strict and gets the clear lat-lon error).
        if grid_type == "latlon_cgrid" and isinstance(ocean.get("physics"), dict):
            phys = ocean["physics"]
            lm = phys.get("lateral_mixing")
            # Default the scheme to "none" only when lateral_mixing is absent/null
            # or a mapping without an explicit scheme.  A non-mapping value
            # (e.g. ``lateral_mixing: harmonic``) is left untouched so the generic
            # builder raises the proper "must be a mapping" error rather than a
            # TypeError from splatting a scalar, or silently rewriting a falsey one.
            if lm is None or (isinstance(lm, dict) and "scheme" not in lm):
                ocean["physics"] = {
                    **phys,
                    "lateral_mixing": {**(lm or {}), "scheme": "none"},
                }

        # Nested physics-pipeline sections (#382): build the deep NamedTuple
        # trees (eos_linear / gm_redi / physics) from their YAML mappings, with
        # the SAME typo-detection contract recursing to every level.  Only entry
        # points that are actually fields of this grid's config are built (the
        # unknown-field check above already rejected any that are not).  A scalar
        # where a mapping is required raises rather than silently passing through.
        for field, sub_type in _nested_ocean_entry_types().items():
            if field not in known:
                continue
            val = ocean.get(field)
            if val is None:
                continue
            if not isinstance(val, dict):
                raise ValueError(
                    f"ocean.{field} must be a mapping, got {type(val).__name__}")
            ocean[field] = _build_nested_config(sub_type, val, path=field)

        # #501: route flat ocean.* keys through from_flat so grouped members
        # (e.g. bottom_drag_r) land in their sub-config; flat configs unchanged.
        if hasattr(ConfigClass, "from_flat"):
            return ConfigClass.from_flat(**ocean)
        return ConfigClass(**ocean)

    # ------------------------------------------------------- validation hooks
    def _validate_setup(self, setup: Any) -> None:
        """Validate the optional ``setup:`` selector (#388 Ask#2).

        Checks the named experiment against the live ``AVAILABLE_EXPERIMENTS``
        registry and the experiment's own ``grid_support`` (no second list to
        drift), so a typo or an unsupported grid fails LOUDLY rather than
        dispatching to nothing — the same dispatch discipline ``_resolve_target``
        applies to ``grid.type``.
        """
        from legoesm.core.setup_selector import validate_setup
        from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS
        # Shape / keys / grid-set / run-control validation + registry membership
        # via the shared selector (single source of truth, used by every
        # component adapter).
        validate_setup(setup, _ocean_matrix_spec(),
                       known_names=AVAILABLE_EXPERIMENTS)
        # Ocean-specific: the experiment's own ``grid_support`` gate — the grid
        # must be one it actually supports, not merely a valid matrix grid.
        name, grid = setup["name"], setup["grid"]
        support = AVAILABLE_EXPERIMENTS[name].get("grid_support")
        if support is not None and not support.get(grid, False):
            supported = sorted(g for g, ok in support.items() if ok)
            raise ValueError(
                f"setup.grid={grid!r} is not supported by experiment "
                f"{name!r}; supported grids: {supported}"
            )

    def validate_strict(self) -> None:
        """Strict-validate the resolved ocean config (raises on invalid).

        Two paths:

        * OMIP-style (no ``setup:``): delegates to the runtime model's
          ``_validate_config`` static method — the single canonical validator
          for each grid (EOS / advection / barotropic-solver membership, bound
          checks) — so the valid sets are never duplicated at the YAML boundary.
        * Idealized ``setup:`` (#388 Ask#2): validates the experiment-selector
          against the registry + ``grid_support``; the procedural experiment
          owns its physics recipe, so a flat ``ocean:`` override would be
          silently ignored and is rejected loudly.
        """
        setup = self.get("setup")
        if setup is not None:
            self._validate_setup(setup)
            if self.get("ocean"):
                raise ValueError(
                    "ocean: physics overrides are not applied to a named "
                    "`setup:` experiment (the experiment defines its own "
                    "recipe; per-run controls go in the `setup:` block). "
                    "Remove the `ocean:` block, or drop `setup:` to configure "
                    "an OMIP-style run."
                )
        else:
            grid_type = self.get("grid.type", "latlon_cgrid")
            _, ModelClass = _resolve_target(grid_type)
            cfg = self.to_ocean_config()
            # _validate_config is a @staticmethod that takes only the config (no
            # grid / state), the same check the model runs in __init__.
            ModelClass._validate_config(cfg)

        # Experiment-level (non-runtime-config) bounds on the time block.
        from legoesm.core.setup_selector import require_positive_finite
        require_positive_finite("time.dt_seconds", self.get("time.dt_seconds"))
        require_positive_finite("time.duration_days",
                                self.get("time.duration_days"))

    def signature(self) -> str:
        """Deterministic signature of the RESOLVED runtime config.

        Used by ``init_experiment`` to detect overrides that don't actually
        change the run (a misspelled or non-runtime dot-path). If the config is
        unresolvable the exception text is folded in so before/after still
        differ (a no-op is only flagged on a byte-identical resolved config).
        """
        try:
            # A ``setup:`` template runs via the matrix runner, which consumes
            # only the setup block + output.path.  Signing the COMMAND-EFFECTIVE
            # invocation (which drops disabled/default controls — quick=false,
            # levels=null, empty resolution — and ignores OMIP-only sections)
            # means exactly the no-op overrides are flagged by init_experiment,
            # and the real ones register as changes.
            if self.get("setup") is not None:
                return repr(("setup", self.run_command()))
            # OMIP path: fold in EVERY field the run consumes outside the runtime
            # NamedTuple
            # (the run controls applied by resolve_ocean_run_controls), so a
            # legitimate override like ``-o output.path=...`` / ``-o grid.nlev=``
            # / ``-o grid.n_lat=`` is not mis-flagged as a no-op by
            # init_experiment.  Advisory-only fields (forcing.dataset) are
            # deliberately excluded — overriding them really is a no-op.
            # Must mirror EXACTLY the fields resolve_ocean_run_controls consumes,
            # so every documented override (-o forcing.path=, -o init.woa_init=,
            # WOA paths, grid/time/output) registers as a real change instead of
            # being rejected as a no-op by init_experiment.
            return repr((
                self.to_ocean_config(),
                self.get("time.dt_seconds"),
                self.get("time.duration_days"),
                self.get("output.path"),
                self.get("grid.nlev"),
                self.get("grid.n_lat"),
                self.get("grid.n_lon"),
                self.get("init.woa_init"),
                self.get("init.woa_t"),
                self.get("init.woa_s"),
                self.get("forcing.path"),
            ))
        except Exception as exc:  # noqa: BLE001
            return f"<unresolvable: {type(exc).__name__}: {exc}>"

    def run_command(self, config_path: str = "config.yaml") -> str:
        """Launcher command for the generated ``run.sh`` (ocean runner).

        ``run_omip_core2.py`` is a repo-root-relative script (it also resolves
        mesh/forcing paths relative to the repo), so the run.sh launches from the
        repo root and ``config_path`` should be an absolute path to the bundle's
        ``config.yaml`` (init_experiment passes that with ``workdir=repo_root``).
        Emits ``--grid`` mapped from ``grid.type`` so the template's grid backend
        actually runs (the runner otherwise defaults to ``tripole``).

        A ``setup:`` template (#388 Ask#2) instead routes to the matrix runner —
        the existing, sole consumer of ``AVAILABLE_EXPERIMENTS`` — via
        ``--only <name> --grid <grid>``, mapping any explicit per-setup
        run-controls onto its CLI and letting the experiment's own defaults
        apply for the rest.  ``config_path`` is unused on this path (the setup
        is fully specified by the selector, not the YAML body).
        """
        setup = self.get("setup")
        if setup is not None:
            from legoesm.core.setup_selector import build_matrix_command
            return build_matrix_command(_ocean_matrix_spec(), setup,
                                        output_path=self.get("output.path"))

        grid_type = self.get("grid.type", "latlon_cgrid")
        try:
            backend = _GRID_TYPE_TO_RUNNER[grid_type]
        except KeyError:
            raise ValueError(
                f"grid.type={grid_type!r} has no runnable run_omip_core2.py "
                f"--config backend (only {sorted(_GRID_TYPE_TO_RUNNER)} is "
                "wired end-to-end; the cube ocean is parked and spectral has no "
                "OMIP runner). Use grid.type=latlon_cgrid, or build the run "
                "manually."
            )
        return (
            f"python scripts/run/run_omip_core2.py --grid {backend} "
            f"--config {shlex.quote(config_path)}"
        )

    def __repr__(self) -> str:
        return f"OceanExperimentConfig({self._data})"


# ======================================================================
# Ocean run record — the FULL experiment identity for the run manifest (#376)
# ======================================================================
# ``model.config`` (the runtime NamedTuple) is necessary but NOT sufficient to
# identify an ocean experiment: the timestep, duration, grid backend, mesh,
# forcing, IC, and output destination all live OUTSIDE it (CLI / template
# ``grid``/``time``/``forcing``/``output`` sections).  Two runs can share a
# ``model.config`` yet integrate a different dt for a different number of years
# on a different grid.  ``OceanRunRecord`` bundles the runtime config WITH those
# run controls so the run manifest's ``config_hash`` identifies the experiment
# that actually ran (codex review HIGH).  It is a plain NamedTuple, so the
# recursive tagged codec below serializes + reconstructs it (and the nested
# runtime config) with no extra codec code.


class OceanRunRecord(NamedTuple):
    """Full ocean experiment identity recorded in ``run_manifest.json``.

    ``runtime_config`` is the runtime ocean config NamedTuple
    (``LatLonCGridOceanConfig`` / ``OceanConfig`` / ...); the remaining fields
    are the run controls that determine the experiment but live outside that
    config.  Everything here is hashed into ``config.config_hash``.
    """
    runtime_config: object
    grid: str = ""
    mesh: str = ""
    nlev: int = 0
    dt_seconds: float = 0.0
    total_days: float = 0.0
    output_path: str = ""
    forcing: str = ""
    forcing_path: str = ""   # resolved CORE-II NYF store dir (data identity)
    woa_init: bool = False
    woa_t: str = ""
    woa_s: str = ""
    latlon_res: str = ""
    smoke: bool = False


# Run-control fields the ocean YAML can drive, mapped to the
# ``run_omip_core2.py`` argparse ``dest`` they set.  Used by
# :func:`resolve_ocean_run_controls` so YAML ``time``/``output``/``grid``
# sections actually take effect (codex review HIGH: they were silently ignored).
_YAML_RUN_CONTROLS = ("dt", "years", "output", "nlev", "latlon_res")


def resolve_ocean_run_controls(
    adapter: OceanExperimentConfig,
    args,
    cli_given: set,
) -> dict:
    """Apply a YAML experiment's run controls onto a runner ``args`` namespace.

    Maps the template's ``time`` / ``output`` / ``grid`` sections onto the
    runner's ``dt`` / ``years`` / ``output`` / ``nlev`` / ``latlon_res`` so a
    ``--config`` run integrates the dt, duration, grid shape, and output the
    template describes — instead of the argparse defaults.  An explicitly-passed
    CLI flag always wins over the YAML (``cli_given`` = the set of argparse
    ``dest`` names the user passed on the command line).

    Mutates *args* in place; returns a dict of the values it set (for logging).
    Grid backend (``--grid``), mesh (``--mesh``) and forcing are NOT driven from
    YAML here (the mesh comes from a file, not the template); the caller warns
    about those so nothing is silently ignored.
    """
    applied: dict = {}

    def _maybe_set(dest: str, value):
        if value is None:
            return
        if dest in cli_given:
            return  # explicit CLI flag wins
        setattr(args, dest, value)
        applied[dest] = value

    dt = adapter.get("time.dt_seconds")
    _maybe_set("dt", float(dt) if dt is not None else None)

    dur = adapter.get("time.duration_days")
    _maybe_set("years", (float(dur) / 365.0) if dur is not None else None)

    _maybe_set("output", adapter.get("output.path"))

    nlev = adapter.get("grid.nlev")
    _maybe_set("nlev", int(nlev) if nlev is not None else None)

    n_lat = adapter.get("grid.n_lat")
    n_lon = adapter.get("grid.n_lon")
    if n_lat is not None and n_lon is not None:
        _maybe_set("latlon_res", f"{int(n_lat)}x{int(n_lon)}")

    # Initial condition: a template that declares a WOA cold-start IC must
    # actually enable it (else the run silently starts from rest, ignoring the
    # declared woa18 data requirement — codex review P2).
    woa = adapter.get("init.woa_init")
    if woa is not None:
        _maybe_set("woa_init", bool(woa))
    _maybe_set("woa_t", adapter.get("init.woa_t"))
    _maybe_set("woa_s", adapter.get("init.woa_s"))

    # Forcing location: thread the staged CORE-II NYF directory into the loader's
    # cache_dir so a "fetch data, then run" workflow finds it where it was staged
    # (the loader otherwise looks only under ~/.cache/.../core2_nyf — codex P2).
    _maybe_set("forcing_path", adapter.get("forcing.path") or None)

    return applied


# ======================================================================
# Run-manifest codec for ocean runtime configs (#376 Phase 4)
# ======================================================================
# The atmosphere ExperimentConfig has a bespoke (grid/dycore/output) codec in
# ``legoesm.driver.config``; the ocean runtime configs are arbitrarily-nested
# NamedTuples (``LatLonCGridOceanConfig`` -> ``constants: ConstantsConfig``,
# optional ``eos_linear``/``gm_redi``/``physics`` pipelines).  This recursive
# codec serializes any of them to a JSON-safe dict and rebuilds it exactly, so
# ``run_manifest.json`` can capture + reconstruct an ocean config and the
# ``config_hash`` round-trips.
#
# Each NamedTuple level records its own qualified type (``__type__``) so
# reconstruction never has to infer types from field annotations (the ocean
# configs annotate optional sub-configs as ``object``, which carries no type).


def _is_namedtuple(obj) -> bool:
    return (
        isinstance(obj, tuple)
        and hasattr(obj, "_fields")
        and hasattr(obj, "_asdict")
    )


def ocean_config_to_dict(cfg) -> dict:
    """Recursively serialize an ocean runtime config NamedTuple to a tagged dict.

    Every NamedTuple level is tagged with ``__type__`` = ``"module:QualName"``;
    nested NamedTuples / lists / dicts recurse; scalars and ``None`` pass through
    (NumPy scalars are normalized later by the manifest's ``_json_safe``).
    """
    return _encode_config(cfg)


def _encode_config(obj):
    if _is_namedtuple(obj):
        out = {"__type__": f"{type(obj).__module__}:{type(obj).__qualname__}"}
        for key, val in obj._asdict().items():
            out[key] = _encode_config(val)
        return out
    if isinstance(obj, dict):
        return {k: _encode_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_config(v) for v in obj]
    return obj


def ocean_config_from_dict(d: dict):
    """Reconstruct an ocean runtime config from :func:`ocean_config_to_dict`.

    Unknown fields are dropped (forward-compat: an older manifest with a since-
    removed field still loads); missing fields fall back to the NamedTuple
    default.  Only ``legoesm.*`` types may be reconstructed (a manifest cannot
    coax this into importing arbitrary modules).
    """
    return _decode_config(d)


def _resolve_config_type(tag: str):
    if ":" not in tag:
        raise ValueError(f"malformed config __type__ tag: {tag!r}")
    module_path, qualname = tag.split(":", 1)
    if not (module_path == "legoesm" or module_path.startswith("legoesm.")):
        raise ValueError(
            f"refusing to reconstruct non-legoesm config type {tag!r} "
            "(run-manifest type allowlist)"
        )
    import importlib

    obj = importlib.import_module(module_path)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _decode_config(obj):
    if isinstance(obj, dict) and "__type__" in obj:
        cls = _resolve_config_type(obj["__type__"])
        # #501: accept BOTH the nested sub-config field name (new checkpoints) AND
        # the grouped members' flat names (OLD pre-grouping checkpoints), then route
        # through from_flat so an old flat ``bottom_drag_r`` is DISTRIBUTED into
        # ``bottom_drag`` rather than silently dropped to the default.
        known = set(cls._fields)
        if hasattr(cls, "flat_fields"):
            known |= set(cls.flat_fields())
        fields = {
            k: _decode_config(v)
            for k, v in obj.items()
            if k != "__type__" and k in known
        }
        if hasattr(cls, "from_flat"):
            return cls.from_flat(**fields)
        return cls(**fields)
    if isinstance(obj, dict):
        return {k: _decode_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_config(v) for v in obj]
    return obj


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
