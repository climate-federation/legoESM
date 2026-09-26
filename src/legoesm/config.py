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
import shlex
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

# "centered" and "finite_volume" are NOT legacy — they are valid
# ambiguous names resolved grid-aware by the driver factory (cdgrid on
# cubed-sphere, latlon_cgrid on lat-lon).  Only truly defunct names
# are rewritten here at the YAML boundary.
_LEGACY_DISCRETIZATION: dict[str, str] = {
    "cgrid": "cdgrid",
    "fv": "cdgrid",
}

_LEGACY_DYNAMICS: dict[str, str] = {
    # No legacy aliases at this time; table is here for future use.
}

# Recognized children of the nested ``physics:`` block.  This is an ALLOWLIST,
# not documentation: anything outside it is rejected in
# ``to_experiment_config``.  The YAML spellings mirror run_amip's flags (so
# ``clouds`` -> ExperimentConfig.cloud_scheme), keeping the two dialects
# consistent for a user moving between them.
_PHYSICS_KEYS: frozenset[str] = frozenset({
    "forcing",
    "convection",
    "microphysics",
    "turbulence",
    "clouds",
    "gravity_wave_drag",
    "radiation",
})

# Recognized children of the nested ``atmosphere:`` block — every one is mapped
# in ``to_experiment_config``.  Same contract as ``_PHYSICS_KEYS``: an
# unrecognized key is REJECTED rather than dropped in silence.
_ATMOSPHERE_KEYS: frozenset[str] = frozenset({
    "dynamics",
    "discretization",
    "time_integrator",
    "dt_seconds",
    "hyperdiff_scale",
    # Live LOWEST-precedence fallback for the radiation scheme, behind the
    # top-level ``radiation.scheme`` block and ``physics.radiation``.  It is
    # mapped (unlike the retired keys), so it must be allowed -- omitting it
    # would reject a config this boundary still honours.
    "radiation",
})

# Keys this dialect used to DECLARE while wiring them to nothing.  They are
# rejected with migration guidance rather than a bare "unknown key", because a
# user who wrote one had every reason to think it worked: they were documented
# in DEFAULT_CONFIG and (for advection/equations) shipped in the templates.
_ATMOSPHERE_RETIRED: dict[str, str] = {
    "hyperdiffusion_coeff": (
        "renamed to 'hyperdiff_scale': it was always a dimensionless multiplier "
        "on the scheme's hyperdiffusion, not a coefficient.  Its old default "
        "0.0 switched biharmonic damping off; the new default is 1.0"
    ),
    "equations": (
        "the legacy 'equations' key never reached ExperimentConfig on this "
        "path (DEFAULT_CONFIG always supplied 'dynamics', so the axis-based "
        "branch of resolve_solver_name always won and 'equations' was "
        "ignored) -- a legacy 'equations: hydrostatic' silently ran "
        "shallow_water. Use the canonical axes: dynamics + discretization"
    ),
    "advection": (
        "'advection' mapped to no ExperimentConfig field and had no reader "
        "anywhere in the tree; the atmosphere has no tracer-advection "
        "selector. Remove it (the dycore's advection follows from "
        "'discretization')"
    ),
    "spectral": (
        "'atmosphere.spectral.allow_unsupported' is only honored when a raw "
        "Config is handed to the spectral constructors; ModelDriver builds "
        "grids from ExperimentConfig, which has no such field, so this block "
        "is dropped on the 'legoesm run' path. Remove it until a canonical "
        "field exists"
    ),
    "tracer_transport": (
        "'atmosphere.tracer_transport' has no runtime reader and reaches no "
        "ExperimentConfig field. Remove it"
    ),
    "nonhydrostatic": (
        "'atmosphere.nonhydrostatic' (n_acoustic_substeps / sponge_* / "
        "small_earth_factor / model_top_m) reaches no ExperimentConfig field: "
        "the NH factories use their own defaults and hard-code the model top. "
        "Remove it until these are wired"
    ),
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


def _require_known_keys(
    block: Any,
    name: str,
    allowed: frozenset[str],
    retired: dict[str, str] | None = None,
) -> dict:
    """Reject unrecognized children of a nested YAML block.

    This boundary's defining bug was that it *accepted* a key, ignored it, and
    ran something else -- so an unmapped key must fail LOUDLY here rather than
    reaching ``experiment_config_from_dict``, which drops unknown keys in
    silence.  ``retired`` carries per-key migration guidance for keys this
    dialect used to declare while wiring them to nothing.
    """
    if block is None:
        raise ValueError(
            f"{name}: must be a mapping, got None -- write '{name}: {{}}' or "
            "omit the block entirely"
        )
    if not isinstance(block, dict):
        raise ValueError(
            f"{name}: must be a mapping, got {type(block).__name__} "
            f"({block!r})"
        )
    retired = retired or {}
    for key in sorted(set(block) - allowed):
        if key in retired:
            raise ValueError(f"{name}.{key} is no longer accepted: {retired[key]}")
        raise ValueError(
            f"unknown {name} key {key!r}; valid keys are {sorted(allowed)}"
        )
    return block


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
        #
        # NOTE: every key declared here MUST be mapped in
        # ``to_experiment_config`` and listed in ``_ATMOSPHERE_KEYS``.  A key
        # declared but not mapped is silently discarded at the boundary -- the
        # defect class this schema is now gated against.  The removed
        # ``equations`` / ``advection`` / ``spectral`` / ``tracer_transport`` /
        # ``nonhydrostatic`` entries were exactly that: declared, documented,
        # and wired to nothing.
        "dynamics": "shallow_water",
        "discretization": "cdgrid",
        # Mirrors DycoreConfig.time_integrator's default so an unset config
        # keeps resolving the integrator grid-aware.  It was "ssp_rk3" while the
        # key was never mapped -- i.e. inert; mapping it with that stale default
        # would have silently pinned EVERY nested config to ssp_rk3.
        "time_integrator": "auto",
        "dt_seconds": 600,          # 10 minutes
        "hyperdiff_scale": 1.0,     # dimensionless multiplier on the scheme hyperdiffusion
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
            "mode": "fp32",  # fp32, fp64, mixed, mixed_fp64_storage
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


def _atm_matrix_spec():
    """Matrix-runner spec for an atmosphere ``setup:`` template (#388).

    Atmosphere's case-name filter is ``--test`` (its ``--only`` selects the
    equation-set sw/hydro/nh), and the runner has no ``--levels``/``--dt``
    flags — captured here so the shared selector emits a valid command.
    Deferred import keeps the federation DAG clean (meta -> core).
    """
    from legoesm.core.setup_selector import MatrixRunnerSpec
    return MatrixRunnerSpec(
        runner_path="scripts/matrix/run_atmosphere_test_matrix.py",
        valid_grids=("cubed_sphere", "latlon", "icosahedral", "spectral"),
        case_flag="--test",
        levels_flag=None, dt_flag=None,
        days_flag="--days", resolution_flag="--resolution",
        output_flag="--output", quick_flag="--quick",
    )


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
        """Translate this YAML-based Config into the canonical ExperimentConfig.

        This is the **YAML boundary**: it maps user-facing YAML field names onto
        canonical ``ExperimentConfig`` field names (applying legacy-name
        normalization) and builds a canonical dict, then delegates the actual
        ``dict -> NamedTuple`` reconstruction to the single canonical
        deserializer :func:`legoesm.driver.config.experiment_config_from_dict`.

        No second live schema is maintained here (D6: one serializer) — the only
        responsibility of this method is the YAML key renames / unit conversions;
        sub-config assembly and field defaulting belong to ``from_dict``.
        """
        from legoesm.driver.config import experiment_config_from_dict

        d = self._data
        atm = d.get("atmosphere", {})
        grid = d.get("grid", {})
        time_cfg = d.get("time", {})
        output_cfg = d.get("output", {})
        forcing = d.get("forcing", {})
        radiation = d.get("radiation", {})
        surface = d.get("surface", {})
        hardware = d.get("hardware", {})
        physics = d.get("physics", {})

        # conservation.fix_mass drives both the legacy ``conservation_fixer``
        # flag and ``fix_mass`` on the canonical dycore config.
        fix_mass = d.get("conservation", {}).get("fix_mass", True)

        # An UNKNOWN child of either block must fail loudly.  Mapping only the
        # recognized keys would leave the very defect these blocks fix: a typo
        # (``convectoin: bechtold``) or the canonical-but-wrong spelling
        # (``cloud_scheme:`` instead of ``clouds:``) would be dropped in silence
        # and the run would proceed on defaults.
        physics = _require_known_keys(physics, "physics", _PHYSICS_KEYS)
        atm = _require_known_keys(
            atm, "atmosphere", _ATMOSPHERE_KEYS, _ATMOSPHERE_RETIRED
        )

        # ``physics.forcing`` selects an idealized forcing.  Held-Suarez (1994)
        # Newtonian relaxation + Rayleigh drag is the only one with a canonical
        # ExperimentConfig field today, so reject any other value LOUDLY rather
        # than accepting it and running something else (the failure mode this
        # whole block exists to fix -- see the note in ``canonical`` below).
        _forcing = physics.get("forcing", "none")
        if _forcing not in ("none", "held_suarez"):
            raise ValueError(
                "physics.forcing must be one of ('none', 'held_suarez'), got "
                f"{_forcing!r}"
            )

        canonical = {
            "grid": {
                "grid_type": grid.get("type", "cubed_sphere"),
                "resolution": grid.get("resolution", 48),
                "nlev": grid.get("n_levels", 40),         # n_levels -> nlev
                "vertical_coord": grid.get("vertical_coord", "hybrid"),
                "p_top_Pa": grid.get("p_top_Pa", 200.0),
                "stretching": grid.get("stretching", 2.0),
                "tropopause_refine": float(grid.get("tropopause_refine", 1.0)),
                "sigma_top": float(grid.get("sigma_top", 0.01)),
                "sigma_refine": float(grid.get("sigma_refine", 0.12)),
                "sigma_refine_width": float(grid.get("sigma_refine_width", 0.45)),
                "sigma_layout": str(grid.get("sigma_layout", "standard")),
            },
            "dycore": {
                "model_type": _normalize_dynamics(atm.get("dynamics", "hydrostatic")),
                "discretization": _normalize_discretization(
                    atm.get("discretization", "cdgrid")
                ),
                "dt": float(atm.get("dt_seconds", 600)),  # dt_seconds -> dt
                "hyperdiff_scale": float(atm.get(
                    "hyperdiff_scale",
                    DEFAULT_CONFIG["atmosphere"]["hyperdiff_scale"])),
                "conservation_fixer": fix_mass,
                "fix_mass": fix_mass,
                # Declared in DEFAULT_CONFIG and written by the experiment
                # wizard (wizard_core: "atmosphere.time_integrator") but never
                # mapped -- so the wizard asked the user to pick an integrator
                # and then silently discarded the answer.  "auto" resolves
                # grid-aware in the driver factory.
                "time_integrator": atm.get("time_integrator", "auto"),
            },
            "output": {
                "output_dir": output_cfg.get("path", ""),
                "diag_days": max(1, int(time_cfg.get("output_interval_hours", 6) / 24)),
                "checkpoint_days": int(output_cfg.get("checkpoint_days", 0)),
                "monthly_means": bool(output_cfg.get("monthly_means", False)),
                "cmip_output": bool(output_cfg.get("cmip_output", False)),
                "clear_sky_diag": bool(output_cfg.get("clear_sky_diag", False)),
                "checkpoint_format": output_cfg.get("checkpoint_format", "npz"),
            },
            "days": int(time_cfg.get("duration_hours", 120) / 24),
            "start_day": float(time_cfg.get("start_day", 0.0)),
            "seed": int(d.get("seed", 0)),  # master RNG seed (reproducibility)
            "dataset": forcing.get("dataset", "analytical"),
            "forcing_path": forcing.get("path", ""),
            "radiation": radiation.get(
                "scheme", physics.get("radiation", atm.get("radiation", "gray"))
            ),
            # --- Column physics (the ``physics:`` block) ---
            # These axes were PARSED BUT NEVER MAPPED: every ``physics:`` key
            # was silently discarded and the run fell back to the
            # ExperimentConfig defaults.  The shipped, ``run_tested``
            # Held-Suarez template (``physics.forcing: held_suarez``) therefore
            # ran WITHOUT its Newtonian relaxation, under moist ``sbm``
            # convection and gray radiation -- not the dry dynamical-core
            # benchmark it advertises.  Every default below mirrors the
            # corresponding ExperimentConfig field default, so a config that
            # sets no ``physics:`` key resolves exactly as it did before.
            "convection": physics.get("convection", "sbm"),
            "microphysics": physics.get("microphysics", "none"),
            "turbulence": physics.get("turbulence", "none"),
            "cloud_scheme": physics.get("clouds", "none"),
            "gravity_wave_drag": physics.get("gravity_wave_drag", "none"),
            "held_suarez_forcing": _forcing == "held_suarez",
            "T_init": float(surface.get("T_init", 300.0)),
            "rh_init": float(surface.get("rh_init", surface.get("RH_init", 0.7))),
            "distributed": bool(
                hardware.get("parallelism", {}).get("distributed", False)
            ),
        }

        return experiment_config_from_dict(canonical)

    # ------------------------------------------------------------------
    # Uniform experiment-adapter protocol (shared with
    # ``legoesm.ocean.config.OceanExperimentConfig``; consumed by
    # ``init_experiment`` / ``validate_templates`` via
    # ``legoesm.experiment_registry``).
    # ------------------------------------------------------------------
    def get_meta(self) -> dict:
        """Return the ``experiment:`` metadata block (or empty dict)."""
        return self.get("experiment") or {}

    def signature(self) -> str:
        """Deterministic signature of the RESOLVED ExperimentConfig.

        Used to detect overrides that don't change the run (a typo or a
        non-runtime dot-path). If the config is unresolvable the exception text
        is folded in so before/after still differ (a no-op is only flagged when
        the resolved config is byte-identical).
        """
        try:
            # A ``setup:`` template runs via the matrix runner; sign the
            # command-effective invocation so a no-op override is flagged.
            setup = self.get("setup")
            if setup is not None:
                from legoesm.core.setup_selector import setup_signature
                return setup_signature(_atm_matrix_spec(), setup,
                                       output_path=self.get("output.path"))
            return repr(self.to_experiment_config())
        except Exception as exc:  # noqa: BLE001
            return f"<unresolvable: {type(exc).__name__}: {exc}>"

    def validate_strict(self) -> None:
        """Strict-validate the config (raises on invalid).

        A ``setup:`` template (#388) names an idealized atmosphere matrix case
        and routes to ``run_atmosphere_test_matrix.py``; it is validated through
        the shared selector (the matrix runner's exact-match zero guard is the
        runnability backstop, since the atmosphere case catalog is not an
        importable registry).  Otherwise the canonical ``ExperimentConfig`` path.
        """
        setup = self.get("setup")
        if setup is not None:
            from legoesm.core.setup_selector import validate_setup
            validate_setup(setup, _atm_matrix_spec())
            # On the matrix-runner path only ``setup:`` + ``output.path`` are
            # consumed; any other user-authored recipe / run-control section
            # (``atmosphere:``, ``grid:``, ``time:``, ``forcing:`` …) would be
            # SILENTLY IGNORED.  Reject a customised one (per-run controls
            # belong in the ``setup:`` block).  Compare to DEFAULT_CONFIG (these
            # sections are default-merged) — an untouched default is fine.
            #   * ``output``: only ``output.path`` is consumed, so a non-path
            #     ``output.*`` customisation is still flagged.
            # (init_experiment skips its machine-precision ``hardware`` injection
            # for setup: templates, so a customised ``hardware:`` here is
            # genuinely user-authored and correctly flagged as ignored.)
            _exempt = {"setup", "model", "experiment", "mode"}
            ignored = []
            for k, v in self._data.items():
                if k in _exempt:
                    continue
                if k == "output":
                    _strip = lambda d: ({kk: vv for kk, vv in d.items()
                                         if kk != "path"} if isinstance(d, dict)
                                        else d)
                    if _strip(v) != _strip(DEFAULT_CONFIG.get("output", {})):
                        ignored.append("output (only output.path is used)")
                    continue
                if v != DEFAULT_CONFIG.get(k):
                    ignored.append(k)
            if ignored:
                raise ValueError(
                    f"top-level section(s) {sorted(ignored)} are ignored by a "
                    "`setup:` template (the matrix case defines the recipe; "
                    "per-run controls go in the `setup:` block: levels/"
                    "dt_seconds/duration_days/resolution/quick). Remove them, "
                    "or drop `setup:` for a `legoesm run`."
                )
            return
        self.to_experiment_config().validate_strict()

    def run_command(self, config_path: str = "config.yaml") -> str:
        """Launcher command for the generated ``run.sh`` (atmosphere runner).

        A ``setup:`` template routes to the atmosphere matrix runner via an
        EXACT ``--test =<case> --grid <grid>`` selector (#388).  Otherwise
        ``legoesm run`` (a cwd-independent installed CLI), so the run.sh can
        ``cd`` into the bundle dir and pass the bundle-relative ``config.yaml``.
        """
        setup = self.get("setup")
        if setup is not None:
            from legoesm.core.setup_selector import build_matrix_command
            return build_matrix_command(_atm_matrix_spec(), setup,
                                        output_path=self.get("output.path"))
        return f"legoesm run {shlex.quote(config_path)}"

    def __repr__(self) -> str:
        return f"Config({self._data})"


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
