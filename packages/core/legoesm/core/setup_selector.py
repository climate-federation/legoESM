"""Shared mechanics for the recipe×setup template selector (#388).

A component's YAML experiment adapter (atmosphere ``Config``, ocean
``OceanExperimentConfig``, sea-ice ``SeaIceExperimentConfig``, …) exposes an
optional top-level ``setup:`` block that NAMES one of that component's idealized
matrix cases::

    setup:
      name: <case>          # an exact case in the component's test matrix
      grid: <grid>          # one of the matrix runner's --grid choices
      # optional per-run overrides (kept here so OMIP/recipe defaults never leak):
      quick: false
      resolution: "72x144"
      levels: 20
      dt_seconds: 900.0
      duration_days: 30.0

The adapter routes such a template to the component's matrix runner via an
EXACT-match case selector (a leading ``=`` avoids the substring filter over-
matching, e.g. ``baroclinic`` → ``baroclinic_gyre*``), mapping any explicit
overrides.

Component matrix runners differ: the case-name flag is ``--only`` for ocean but
``--test`` for atmosphere and sea-ice; ocean/atmosphere switch substring→exact
with a leading ``=`` while sea-ice ``--test`` is already exact (no prefix);
atmosphere/sea-ice have no ``--levels`` / ``--dt``.
A :class:`MatrixRunnerSpec` captures those per-component differences so the three
adapters share ONE validation + command-building implementation instead of
copy-pasting it.  Component-specific case-name validation (against a registry or
the live matrix catalog) stays in the adapter via ``known_names``.

Pure / dependency-free (``shlex`` + ``math`` + ``dataclasses``).
"""
from __future__ import annotations

import math
import shlex
from dataclasses import dataclass

# Allowed keys in a ``setup:`` block.  ``name`` + ``grid`` are required.
SETUP_KEYS: tuple[str, ...] = (
    "name", "grid", "quick", "resolution", "levels", "dt_seconds",
    "duration_days",
)


@dataclass(frozen=True)
class MatrixRunnerSpec:
    """How a component's matrix runner is driven from a ``setup:`` block.

    ``valid_grids`` is the runner's ``--grid`` choice set (excluding ``all``).
    A flag attribute set to ``None`` means the runner does NOT accept that
    override — a ``setup:`` that supplies it then fails validation rather than
    emitting a flag the runner would reject.
    """

    runner_path: str
    valid_grids: tuple[str, ...]
    case_flag: str = "--only"        # the exact-case-name filter flag
    # Prefix the case name with this to request EXACT match.  ocean / atmosphere
    # filters strip a leading ``=`` to switch from substring to exact, so they
    # use ``"="``; a runner whose case flag is ALREADY exact (sea-ice ``--test``,
    # ``tc.case == args.test``) uses ``""``.
    exact_prefix: str = "="
    grid_flag: str = "--grid"
    levels_flag: str | None = "--levels"
    dt_flag: str | None = "--dt"
    days_flag: str | None = "--days"
    resolution_flag: str | None = "--resolution"
    output_flag: str | None = "--output"
    quick_flag: str | None = "--quick"

    def _override_flag(self, setup_key: str) -> str | None:
        return {
            "levels": self.levels_flag, "dt_seconds": self.dt_flag,
            "duration_days": self.days_flag, "resolution": self.resolution_flag,
            "quick": self.quick_flag,
        }[setup_key]


def require_positive_finite(name: str, value) -> None:
    """Raise ``ValueError`` unless *value* is a finite number > 0.

    Rejects bools (``True`` coerces to 1) and ``nan``/``inf`` (``nan <= 0`` is
    False).  ``None`` is accepted ("not set / use the case default").
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")


def validate_setup(setup, spec: MatrixRunnerSpec, *, known_names=None) -> None:
    """Validate a ``setup:`` block against a runner spec (raises on invalid).

    ``known_names`` (optional): valid case names; when provided an unknown
    ``setup.name`` raises. When ``None`` the name need only be a non-empty
    string and the runner's exact-match zero-result guard is the runnability
    backstop (components with no importable case registry).
    """
    if not isinstance(setup, dict):
        raise ValueError(
            f"`setup:` must be a mapping with `name:` and `grid:` "
            f"(got {type(setup).__name__})"
        )
    unknown = sorted(k for k in setup if k not in SETUP_KEYS)
    if unknown:
        raise ValueError(
            f"unknown setup field(s): {unknown}. Valid: {list(SETUP_KEYS)}"
        )
    name, grid = setup.get("name"), setup.get("grid")
    if not isinstance(name, str) or not name:
        raise ValueError(f"setup.name must be a non-empty string, got {name!r}")
    if known_names is not None and name not in known_names:
        raise ValueError(
            f"setup.name={name!r} is not a known experiment; "
            f"valid: {sorted(known_names)}"
        )
    if not isinstance(grid, str) or not grid:
        raise ValueError(
            f"setup.grid must be a non-empty string naming a matrix grid, "
            f"got {grid!r}"
        )
    if grid not in spec.valid_grids:
        raise ValueError(
            f"setup.grid={grid!r} is not a valid matrix grid; choose one of "
            f"{sorted(spec.valid_grids)}"
        )
    # Reject overrides the runner cannot accept (its flag attribute is None).
    for key in ("levels", "dt_seconds", "duration_days", "resolution", "quick"):
        if setup.get(key) is not None and spec._override_flag(key) is None:
            raise ValueError(
                f"setup.{key} is not supported by {spec.runner_path} "
                "(this matrix runner has no matching flag)"
            )
    res = setup.get("resolution")
    # A non-empty string ("C48", "72x144") or a bare int (16) — the matrix
    # runners' --resolution accepts both (build_matrix_command str()s it).
    # Reject empty strings and other junk (bool/float/list/dict).
    if res is not None and not (
        (isinstance(res, str) and res.strip())
        or (isinstance(res, int) and not isinstance(res, bool))
    ):
        raise ValueError(
            f"setup.resolution must be a non-empty string or an int, got "
            f"{res!r}")
    lv = setup.get("levels")
    if lv is not None and (isinstance(lv, bool) or not isinstance(lv, int)
                           or lv <= 0):
        raise ValueError(f"setup.levels must be a positive int, got {lv!r}")
    for fld in ("dt_seconds", "duration_days"):
        require_positive_finite(f"setup.{fld}", setup.get(fld))
    q = setup.get("quick")
    if q is not None and not isinstance(q, bool):
        raise ValueError(f"setup.quick must be a bool, got {q!r}")


def build_matrix_command(spec: MatrixRunnerSpec, setup, *, output_path=None) -> str:
    """Build the matrix-runner launcher for a ``setup:`` template (EXACT match).

    Emits ``python <runner> <case_flag> =<name> <grid_flag> <grid>`` with any
    explicit per-run overrides the runner supports.  Case defaults apply for
    anything omitted.
    """
    parts = [
        f"python {spec.runner_path}",
        f"{spec.case_flag} {shlex.quote(spec.exact_prefix + str(setup['name']))}",
        f"{spec.grid_flag} {shlex.quote(str(setup['grid']))}",
    ]
    if setup.get("levels") is not None and spec.levels_flag:
        parts.append(f"{spec.levels_flag} {int(setup['levels'])}")
    if setup.get("resolution") and spec.resolution_flag:
        parts.append(f"{spec.resolution_flag} {shlex.quote(str(setup['resolution']))}")
    if setup.get("dt_seconds") is not None and spec.dt_flag:
        parts.append(f"{spec.dt_flag} {float(setup['dt_seconds'])}")
    if setup.get("duration_days") is not None and spec.days_flag:
        parts.append(f"{spec.days_flag} {float(setup['duration_days'])}")
    if output_path and spec.output_flag:
        parts.append(f"{spec.output_flag} {shlex.quote(str(output_path))}")
    if setup.get("quick") and spec.quick_flag:
        parts.append(spec.quick_flag)
    return " ".join(parts)


def setup_signature(spec: MatrixRunnerSpec, setup, *, output_path=None) -> str:
    """Command-effective signature: a no-op override leaves it unchanged."""
    return repr(("setup", build_matrix_command(spec, setup,
                                                output_path=output_path)))


def deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* (in-place).

    Shared by the component YAML adapters' ``from_yaml``/``from_dict`` (the
    atmosphere ``Config``, ocean ``OceanExperimentConfig`` and sea-ice
    ``SeaIceExperimentConfig``): nested dicts merge key by key, anything else
    replaces.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
