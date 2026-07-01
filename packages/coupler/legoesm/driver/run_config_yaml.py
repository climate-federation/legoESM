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


# ---------------------------------------------------------------------------
# --params : calibration (tuned parameter) loader (issue #691, format #690)
# ---------------------------------------------------------------------------
# A ``--params FILE`` supplies the CALIBRATION layer: converged scheme-parameter
# values, keyed by the ``param_collector`` qualified name ``scheme_key.field``
# (the SAME keying the registry and the SCM-RCE / AIMIP training output use, e.g.
# ``atm.clouds.CloudConfig.q_c_diagnostic: 3.0e-4``).  Applied to the built
# config AFTER ``--config``/CLI so a trained ``recommended_defaults`` drops in
# unchanged.  Every key is validated against the scheme's ``__param_spec__``
# (existence + bounds) and routed to the matching nested ``*Config`` NamedTuple
# by class name — a typo'd / out-of-bounds / absent-in-this-run parameter is a
# hard error, never a silent mis-set of physics.


def load_params_config(path) -> dict:
    """Load a calibration params YAML (``{qualified_name: value}``) → dict.

    Supports the same ``include:`` base merge as ``--config``.  Returns the raw
    ``{qualified_name: value}`` mapping; validation + routing happen in
    :func:`apply_params_to_config` (which needs the run's config object).
    """
    doc = read_yaml_with_includes(path)
    if not isinstance(doc, dict):
        raise SystemExit(f"--params {path}: top level must be a mapping of "
                         "'scheme_key.field: value' entries.")
    return dict(doc)


def _route_overrides_by_class(node, by_key: dict, *, applied: set):
    """Recursively splice ``{(module, class_name): {field: value}}`` into a
    config NamedTuple tree (depth-first; child configs updated before parent).

    Keyed on the FULL ``(defining module, class name)`` — not the bare class
    name — so same-named configs in different components (e.g. the ocean
    vertical-mixing ``TKEConfig`` vs the atmosphere turbulence ``TKEConfig``)
    never cross-route.  Records each applied key in ``applied`` so the caller can
    detect a target config that is ABSENT (never applied) or AMBIGUOUS (two
    instances of the same class in the tree)."""
    fields = getattr(node, "_fields", None)
    if fields is None or not isinstance(node, tuple):
        return node  # not a NamedTuple leaf
    replacements = {}
    for f in fields:
        child = getattr(node, f)
        new_child = _route_overrides_by_class(child, by_key, applied=applied)
        if new_child is not child:
            replacements[f] = new_child
    if replacements:
        node = node._replace(**replacements)
    key = (type(node).__module__, type(node).__name__)
    if key in by_key:
        if key in applied:
            raise SystemExit(
                f"--params: config {key[1]!r} ({key[0]}) appears more than once "
                "in the run config tree — cannot route the override unambiguously."
            )
        applied.add(key)
        # apply_param_overrides validates every field is on the NamedTuple.
        from legoesm.training.param_collector import apply_param_overrides
        node = apply_param_overrides(node, by_key[key])
    return node


def apply_params_to_config(config, params: dict, *, driver: str = "run"):
    """Return ``config`` with calibration ``params`` (qualified_name: value)
    spliced into the matching nested ``*Config`` NamedTuples.

    Each key is validated against ``param_collector.build_registry`` (must be a
    known parameter and, for scalar params, within its ``__param_spec__``
    bounds) and routed to the config of its declared ``(module, config_class)``.
    Raises ``SystemExit`` on an unknown parameter, a non-numeric or out-of-bounds
    value, or a target config that is absent from / ambiguous in this run's
    config tree — a ``--params`` file can never silently mis-set physics.  A
    no-op for empty ``params``.

    Note (soft limitation): a union config that holds ALL of a family's scheme
    sub-configs simultaneously (``VerticalMixingConfig`` carries kpp/tke/catke;
    ``MultiLayerLandConfig`` carries carbon/stomata) is always "present", so an
    override for a scheme that is not the *selected* one is applied to that
    (inert) sub-config rather than raising — it simply has no effect on the run.
    The strict absent-raise still catches wrong-component params (e.g. an
    atmosphere param in an ocean-only run).
    """
    if not params:
        return config
    from legoesm.training.param_collector import build_registry
    registry = {m.qualified_name: m for m in build_registry()}
    by_key: dict[tuple, dict[str, float]] = {}
    key_to_qname: dict[tuple, str] = {}
    for qname, value in params.items():
        meta = registry.get(qname)
        if meta is None:
            raise SystemExit(
                f"{driver} --params: unknown parameter {qname!r} (not in the "
                "parameter registry).  Keys must be a param_collector qualified "
                "name 'scheme_key.field' (see config/cmip/params_tuned.yaml)."
            )
        # Scalar params (shape_key None) are numeric — coerce (YAML may quote
        # the value) and range-check against __param_spec__ bounds.  Array params
        # keep their list value (per-element tuple bounds are not range-checked).
        if meta.shape_key is None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise SystemExit(
                    f"{driver} --params: {qname} value {value!r} is not numeric."
                )
            lo, hi = meta.bounds
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                if not (lo <= value <= hi):
                    raise SystemExit(
                        f"{driver} --params: {qname}={value} is outside its "
                        f"__param_spec__ bounds [{lo}, {hi}]."
                    )
        key = (meta.module, meta.config_class)
        by_key.setdefault(key, {})[meta.field] = value
        key_to_qname[key] = qname
    applied: set = set()
    config = _route_overrides_by_class(config, by_key, applied=applied)
    missing = set(by_key) - applied
    if missing:
        examples = ", ".join(sorted(key_to_qname[k] for k in missing))
        raise SystemExit(
            f"{driver} --params: parameter(s) {examples} target config "
            f"class(es) {sorted(k[1] for k in missing)} that are not present in "
            "this run's config (the scheme is not built into this driver's "
            "config object).  Remove them or enable the scheme."
        )
    return config

