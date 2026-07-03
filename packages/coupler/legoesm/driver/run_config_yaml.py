"""Shared YAML run-config loader for the production run drivers.

A ``--config FILE`` supplies argument DEFAULTS (applied via
``parser.set_defaults``), so any explicit CLI flag still overrides the file
(precedence: CLI > config file > parser default).  An optional ``include:`` base
is merged FIRST so tuned physics can live in one shared file
(``config/cmip/cmip_tuned_physics.yaml``) and be reused across run configs.

The loader is keyed purely on ``parser._actions`` dests, so it is
driver-agnostic: ``run_coupled.py`` and ``run_amip.py`` each pass their OWN
parser and the file is validated against that driver's CLI surface.  Every
merged key MUST be a known argument dest of the calling parser — an unknown key
is a hard error (dispatch-hardening: no silent typo'd / dropped override).

This is the single source of truth for the ``--config`` mechanism; the drivers
must not re-implement it (CLAUDE.md: no duplicate utilities).
"""

from __future__ import annotations

from pathlib import Path


def read_yaml_with_includes(path, _seen=None) -> dict:
    """Read a run YAML config, recursively merging an optional ``include:`` base
    FIRST so the tuned physics can live in one shared file and be reused across
    run configs.

    Precedence: the including file's keys override the base it includes
    (base < file).  An ``include:`` path is resolved relative to the including
    file.  Cycles and missing/non-mapping files raise.  Returns the merged raw
    dict; ``load_yaml_config`` then validates + type-coerces it.
    """
    import yaml
    p = Path(path).resolve()
    _seen = set() if _seen is None else _seen
    if p in _seen:
        raise SystemExit(f"--config: 'include' cycle detected at {p}.")
    _seen.add(p)
    if not p.exists():
        raise SystemExit(f"--config: file not found: {p}.")
    doc = yaml.safe_load(p.read_text())
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise SystemExit(
            f"--config {path}: expected a YAML mapping of argument=value, "
            f"got {type(doc).__name__}.")
    base_ref = doc.pop("include", None)
    merged: dict = {}
    if base_ref is not None:
        if not isinstance(base_ref, str):
            raise SystemExit(
                f"--config {path}: 'include' must be a single path string, "
                f"got {type(base_ref).__name__}.")
        merged.update(read_yaml_with_includes(p.parent / base_ref, _seen))
    merged.update(doc)  # the including file overrides its base
    return merged


def load_yaml_config(path, parser, *, example_keys: str | None = None) -> dict:
    """Load a run YAML config file into a dict of argument defaults.

    Used by ``--config`` to make a canonical run (e.g. the tuned
    ``config/cmip/cmip_ocean_{slab,3D}.yaml`` or the AMIP production config)
    reproducible from one file.  Supports an optional ``include:`` base merged
    first (see ``read_yaml_with_includes``).  Every (merged) key MUST be a known
    argument dest of ``parser``; an unknown key raises (no silent typo'd /
    dropped override — dispatch-hardening).

    Each scalar is coerced through that argument's ``type=`` callable, because
    ``parser.set_defaults`` (how the caller applies this) BYPASSES argparse's own
    type conversion: a value written as a quoted string (e.g. ``dt: "300"``)
    would otherwise reach the run as a str.  Returns the mapping so the caller
    can feed it to ``parser.set_defaults`` (an explicit CLI flag still wins).

    ``example_keys`` is an optional human hint (driver-specific example dests)
    appended to the unknown-key error to help operators fix typos.
    """
    doc = read_yaml_with_includes(path)
    actions = {a.dest: a for a in parser._actions}
    unknown = sorted(set(doc) - set(actions))
    if unknown:
        hint = f" (e.g. {example_keys})" if example_keys else ""
        raise SystemExit(
            f"--config {path}: unknown key(s) {unknown}. Keys must be valid "
            f"argument dests of this run script{hint}.")
    out: dict = {}
    for key, value in doc.items():
        argtype = getattr(actions[key], "type", None)
        # Coerce only string scalars through the arg's type (a YAML native
        # float/int/bool is already the right Python type; type=None args are
        # str/bool flags that need no conversion).
        if argtype is not None and isinstance(value, str):
            try:
                value = argtype(value)
            except (ValueError, TypeError) as exc:
                raise SystemExit(
                    f"--config {path}: key '{key}' value {value!r} is not a "
                    f"valid {getattr(argtype, '__name__', argtype)}: {exc}")
        # ``parser.set_defaults`` BYPASSES argparse's own ``choices`` check, so
        # a scheme literal (e.g. ``vertical_mixing_scheme: garbage``) supplied
        # via --config would otherwise reach the factory unvalidated.  Enforce
        # it here so a typo'd scheme fails loudly at load (dispatch-hardening),
        # exactly as an explicit CLI flag would.
        choices = getattr(actions[key], "choices", None)
        if choices is not None and value not in choices:
            try:
                allowed = sorted(choices)
            except TypeError:
                allowed = list(choices)
            raise SystemExit(
                f"--config {path}: key '{key}' value {value!r} is not one of "
                f"the allowed choices {allowed}.")
        out[key] = value
    return out


def require_config(config_value, *, driver: str = "run") -> None:
    """Enforce ``--require-config``: a run must be driven by a committed config.

    Raises ``SystemExit`` when ``--require-config`` is set but no ``--config``
    file was supplied, so a production/test run cannot silently fall back to
    parser defaults (issue #691).  A no-op when strict mode is off.
    """
    if config_value is None:
        raise SystemExit(
            f"{driver}: --require-config was set but no --config file was "
            "given.  Pass --config <yaml> so the run is fully specified by a "
            "committed configuration (no hidden parser defaults)."
        )

