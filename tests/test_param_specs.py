"""Spec-first guard: every physics scheme ``*Config`` declares a ``__param_spec__``.

The user's request that motivates this gate: the tunable-vs-fixed status of every
physics parameter must be *machine-readable* (so a training loop can select which
to optimize) and easy to flip. We record that next to each scheme ``*Config``
NamedTuple as a module-level ``__param_spec__`` dict literal — the parameter-level
twin of ``__physics_contract__`` (this file mirrors ``test_physics_contracts.py``).

The spec carries NO default *values* — the NamedTuple field default is the single
numeric source of truth. The spec adds only what the NamedTuple cannot express:
units, optimisation bounds, a tunability tier (the continuum: 0=fixed, 1=core,
2=extended, 3=aggressive), the constraint transform, a category, a literature
reference, an optional array ``shape`` dimension key, and an optional flat
``legacy_name`` alias. Example::

    __param_spec__ = {
        "SBMConfig": {
            "scheme_key": "sbm",          # globally unique
            "excluded": {"smooth_trigger_sharpness": "numerics: gate width"},
            "params": {
                "tau_c": {
                    "units": "s", "bounds": (1800.0, 21600.0),
                    "tunable_tier": 1, "transform": "sigmoid",
                    "category": "closure", "reference": "Frierson (2007)",
                    "shape": None, "legacy_name": "sbm_tau_c",
                },
            },
        },
    }

**Inclusion is computed, not enumerated.** A class is *spec-required* iff it
subclasses ``NamedTuple`` and has at least one ``: float``-annotated field whose
default folds to a number (a ``constants.X`` / nested-config / int / str / bool
default is auto-exempt — it cannot be a stray tunable). **Only float fields are
spec-eligible**, which is the structural guarantee that loop-iteration counts and
other discrete integers (``N_evp``, ``bulk_n_iter``, ``n_categories``, ...) can
NEVER be exposed to a tuning / backprop loop: they are non-differentiable control
flow, an ``int`` field cannot appear in a spec's ``params`` (it is not in the
float-default set, so it is flagged as an orphan), and the trainable collector
only ever materialises float params. A module with at least one
spec-required class must EITHER carry a ``__param_spec__`` covering every such
class's every float field (in ``params`` or ``excluded``) OR appear in the
SHRINK-ONLY ``PARAM_SPEC_TODO``. A NEW config module is in neither, so it FAILS
until it ships a spec — the forcing function. ``PARAM_SPEC_TODO`` burns to empty.

Validation is AST-only (no imports, no side effects): defaults are read from the
NamedTuple's AST so ``bounds ⊇ default`` is checked without importing the module.
Self-tests prove non-vacuity.
"""

from __future__ import annotations

import ast
import collections

import pytest

from tests import _ratchet_audit as ra
from tests._param_spec_baseline import PARAM_SPEC_FIELD_BASELINE

# --- scope -----------------------------------------------------------------
# Physics scheme trees across all model components (driver-level ExperimentConfig
# is the flat experiment aggregator, not a scheme config — out of scope here).
_ROSTER_PREFIXES = (
    "packages/atmosphere/legoesm/atmosphere/physics/",
    "packages/ocean/legoesm/ocean/physics/",
    "packages/land/legoesm/land/",
    "packages/ice/legoesm/ice/",
    "packages/coupler/legoesm/coupler/",
)

_VALID_TRANSFORMS = frozenset({"sigmoid", "softplus", "none"})
_VALID_TIERS = frozenset({0, 1, 2, 3})
# Array-parameter dimension keys (resolved to ints at collect time by
# param_registry.resolve_param_dims). Extend as size-dependent schemes are added.
_VALID_SHAPE_KEYS = frozenset(
    {"n_pft", "n_soil_layers", "n_bands", "n_hydrometeor_classes", "n_ocean_bands"}
)
_REQUIRED_PARAM_KEYS = frozenset(
    {"units", "bounds", "tunable_tier", "transform", "category", "reference", "shape"}
)
_OPTIONAL_PARAM_KEYS = frozenset({"legacy_name"})
_REQUIRED_CLASS_KEYS = frozenset({"scheme_key", "params"})
_OPTIONAL_CLASS_KEYS = frozenset({"excluded"})

# Scheme config modules not yet carrying a __param_spec__ (seeded 2026-06-12).
# SHRINK-ONLY: author a spec then delete the module here.
PARAM_SPEC_TODO: frozenset[str] = frozenset(
    {
        # --- atmosphere ----------------------------------------------------
        # PARALLEL-MERGE BACKLOG (2026-06-13): the concurrent CLUBB full-closure
        # port landed CLUBBParams (~40 closure constants) on main via a separate
        # PR that predates this param-spec gate. Classified here as migration
        # backlog (author __param_spec__ + delete this entry in the CLUBB owner's
        # follow-up); the field baseline below pins its current float fields so
        # no NEW unspecced float can slip into it meanwhile.
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py",
        # --- ocean ---------------------------------------------------------
        # --- land ----------------------------------------------------------
    }
)


# --- AST inclusion predicate ----------------------------------------------
def _is_namedtuple(cls: ast.ClassDef) -> bool:
    for b in cls.bases:
        if isinstance(b, ast.Name) and b.id == "NamedTuple":
            return True
        if isinstance(b, ast.Attribute) and b.attr == "NamedTuple":
            return True
    return False


def _float_default_fields(cls: ast.ClassDef) -> dict[str, float]:
    """``{field_name: folded_default}`` for every ``: float``-annotated field
    whose default folds to a number. A ``constants.X`` / nested-config / name
    default folds to ``None`` and is auto-exempt — so is any non-float annotation."""
    out: dict[str, float] = {}
    for n in cls.body:
        if not (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)):
            continue
        if not (isinstance(n.annotation, ast.Name) and n.annotation.id == "float"):
            continue
        if n.value is None:
            continue
        folded = ra.fold_numeric(n.value)
        if folded is not None:
            out[n.target.id] = folded
    return out


def spec_required_classes(src: str) -> dict[str, dict[str, float]]:
    """``{ClassName: {float_field: default}}`` for module-level NamedTuple
    subclasses that have at least one float-default (literal) field — the fields
    a spec MUST classify."""
    tree = ast.parse(src)
    out: dict[str, dict[str, float]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_namedtuple(node):
            fields = _float_default_fields(node)
            if fields:
                out[node.name] = fields
    return out


def all_float_field_names(src: str) -> dict[str, set[str]]:
    """``{ClassName: {every ``: float`` field with a default}}`` — the spec-
    ELIGIBLE set (superset of ``spec_required_classes``): also includes fields
    whose default is a ``constants.X`` reference (not AST-foldable). A genuinely
    tunable closure that defaults to a constant (``S_ice_new =
    constants.S_ice_bulk_default``) may be listed in a spec's ``params`` without
    being flagged as an orphan, but is not FORCED into the spec."""
    tree = ast.parse(src)
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_namedtuple(node):
            names = {
                n.target.id
                for n in node.body
                if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and isinstance(n.annotation, ast.Name)
                and n.annotation.id == "float"
                and n.value is not None
            }
            if names:
                out[node.name] = names
    return out


def extract_param_spec(src: str):
    """Module-level ``__param_spec__`` literal; ``None`` if absent, sentinel
    ``"__UNPARSEABLE__"`` if present but not a pure literal."""
    tree = ast.parse(src)
    for node in tree.body:
        match = (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__param_spec__" for t in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__param_spec__"
            and node.value is not None
        )
        if match:
            try:
                return ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                return "__UNPARSEABLE__"
    return None


# --- schema validation -----------------------------------------------------
def _validate_bounds(bounds, default: float, tier: int, transform: str) -> list[str]:
    errs: list[str] = []
    if isinstance(bounds, tuple) and len(bounds) == 2 and all(
        isinstance(b, (int, float)) for b in bounds
    ):
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo < hi:
            errs.append(f"bounds lo<hi violated: {bounds}")
        elif transform == "sigmoid" and default is not None and not (lo <= default <= hi):
            # default is None for a field whose default is a ``constants.X``
            # reference (not AST-resolvable); the structural lo<hi check still
            # runs, and the collector clamps the resolved value at seed time.
            errs.append(f"default {default} outside sigmoid bounds {bounds}")
    elif isinstance(bounds, tuple) and len(bounds) == 2 and all(
        isinstance(b, tuple) for b in bounds
    ):
        # per-element bounds: (lo_tuple, hi_tuple) of equal length
        lo_t, hi_t = bounds
        if len(lo_t) != len(hi_t):
            errs.append(f"per-element bounds length mismatch: {bounds}")
    elif tier >= 1:
        errs.append(f"tier>=1 requires (lo, hi) bounds, got {bounds!r}")
    return errs


def validate_param_spec(
    spec,
    required: dict[str, dict[str, float]],
    eligible: dict[str, set[str]] | None = None,
) -> list[str]:
    """Schema + completeness errors for one module's ``__param_spec__``.

    ``required`` = ``{Class: {field: literal_default}}`` — float fields with a
    *folding* (literal) default that MUST be classified. ``eligible`` =
    ``{Class: {all float field names}}`` — any ``: float`` field (incl. those
    whose default is a ``constants.X`` reference), which MAY appear in
    ``params``/``excluded`` without being an orphan. This lets a genuinely
    tunable closure that happens to default to a constant (e.g.
    ``S_ice_new = constants.S_ice_bulk_default``) be exposed as a spec param,
    while still not FORCING constants-default fields into the spec. When
    ``eligible`` is omitted it falls back to ``required`` (the strict set)."""
    if spec == "__UNPARSEABLE__":
        return ["__param_spec__ is not a pure literal (AST literal-eval failed)"]
    if not isinstance(spec, dict):
        return ["__param_spec__ must be a dict keyed by Config class name"]
    eligible = eligible or {}
    errors: list[str] = []
    # Orphan classes: a spec entry for a class with no spec-eligible float fields.
    for cls_name in spec:
        if cls_name not in required and cls_name not in eligible:
            errors.append(f"spec names class {cls_name!r} with no float fields")
    # Validate every class that is required OR eligible (the latter covers a class
    # whose float fields all default to ``constants.X`` — it is not FORCED into the
    # spec, but if it appears it must still be fully schema-validated, not skipped).
    for cls_name in sorted(set(required) | set(eligible)):
        fields = required.get(cls_name, {})
        cls_eligible = eligible.get(cls_name, set(fields))
        if cls_name not in spec:
            if cls_name in required:
                errors.append(f"class {cls_name!r} is spec-required but absent from spec")
            continue  # an eligible-only class is optional in the spec
        entry = spec[cls_name]
        if not isinstance(entry, dict):
            errors.append(f"{cls_name}: entry must be a dict")
            continue
        keys = set(entry)
        missing = _REQUIRED_CLASS_KEYS - keys
        extra = keys - _REQUIRED_CLASS_KEYS - _OPTIONAL_CLASS_KEYS
        if missing:
            errors.append(f"{cls_name}: missing keys {sorted(missing)}")
        if extra:
            errors.append(f"{cls_name}: unexpected keys {sorted(extra)}")
        sk = entry.get("scheme_key")
        if not isinstance(sk, str) or not sk.strip():
            errors.append(f"{cls_name}: scheme_key must be a non-empty string")
        params = entry.get("params", {})
        excluded = entry.get("excluded", {})
        if not isinstance(params, dict):
            errors.append(f"{cls_name}: params must be a dict")
            params = {}
        if not isinstance(excluded, dict):
            errors.append(f"{cls_name}: excluded must be a dict")
            excluded = {}
        # Completeness: every float field classified exactly once.
        classified = set(params) | set(excluded)
        unclassified = sorted(set(fields) - classified)
        if unclassified:
            errors.append(f"{cls_name}: float fields not in params/excluded: {unclassified}")
        both = sorted(set(params) & set(excluded))
        if both:
            errors.append(f"{cls_name}: fields in both params and excluded: {both}")
        orphan = sorted(classified - cls_eligible)
        if orphan:
            errors.append(f"{cls_name}: params/excluded name non-float fields: {orphan}")
        for ename, reason in excluded.items():
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{cls_name}.excluded[{ename!r}] must be a non-empty reason string")
        for pname, p in params.items():
            if pname not in cls_eligible:
                continue  # already flagged as orphan
            if not isinstance(p, dict):
                errors.append(f"{cls_name}.{pname}: param spec must be a dict")
                continue
            pk = set(p)
            pmissing = _REQUIRED_PARAM_KEYS - pk
            pextra = pk - _REQUIRED_PARAM_KEYS - _OPTIONAL_PARAM_KEYS
            if pmissing:
                errors.append(f"{cls_name}.{pname}: missing keys {sorted(pmissing)}")
            if pextra:
                errors.append(f"{cls_name}.{pname}: unexpected keys {sorted(pextra)}")
            tier = p.get("tunable_tier")
            if tier not in _VALID_TIERS:
                errors.append(f"{cls_name}.{pname}: tunable_tier must be in {sorted(_VALID_TIERS)}")
                tier = 0
            if p.get("transform") not in _VALID_TRANSFORMS:
                errors.append(f"{cls_name}.{pname}: transform must be in {sorted(_VALID_TRANSFORMS)}")
            for key in ("units", "category", "reference"):
                if not isinstance(p.get(key), str) or not p[key].strip():
                    errors.append(f"{cls_name}.{pname}: {key!r} must be a non-empty string")
            if "shape" in p and p["shape"] is not None and p["shape"] not in _VALID_SHAPE_KEYS:
                errors.append(
                    f"{cls_name}.{pname}: shape must be None or one of {sorted(_VALID_SHAPE_KEYS)}"
                )
            if "bounds" in p and isinstance(tier, int):
                # default is known only for literal-default (``required``) fields;
                # a ``constants.X``-default param passes default=None (structure
                # checked, value-in-bounds skipped — see _validate_bounds).
                errors.extend(
                    f"{cls_name}.{pname}: {e}"
                    for e in _validate_bounds(
                        p["bounds"], fields.get(pname), tier, p.get("transform")
                    )
                )
    return errors


# --- discovery / partition -------------------------------------------------
def _in_scope(rel: str) -> bool:
    # `_future/` holds parked, non-production code (CLAUDE.md: NOT wired into any
    # factory / __init__ / prod driver). Its scheme configs must never enter the
    # production trainable registry (``param_collector.SPEC_MODULES`` / the
    # ``build_trainable_params`` collector) — a scheme that no driver runs cannot
    # be a live trainable. So `_future/` modules are OUT of the param-spec roster;
    # wire the module into prod (moving it out of `_future/`) in the same PR that
    # should make it a registered trainable scheme.
    if "/_future/" in rel:
        return False
    return any(rel.startswith(p) for p in _ROSTER_PREFIXES)


def _scope_modules() -> list[str]:
    out: list[str] = []
    for f in ra.discover_py_files():
        rel = ra.rel(f)
        if not _in_scope(rel):
            continue
        if spec_required_classes(f.read_text()):
            out.append(rel)
    return sorted(out)


_SCOPE_MODULES = _scope_modules()


def test_discovery_sane() -> None:
    ra.assert_discovery_sane(ra.discover_py_files())
    assert len(_SCOPE_MODULES) >= 25, (
        f"only {len(_SCOPE_MODULES)} spec-required config modules discovered "
        f"(expected ~31) — discovery scope likely broken."
    )
    # sentinels: known config modules must be in scope
    for sentinel in (
        "packages/land/legoesm/land/soil_thermal.py",
        "packages/ice/legoesm/ice/config.py",
    ):
        assert sentinel in _SCOPE_MODULES, f"{sentinel} not in scope (predicate broke)"


def test_todo_entries_are_in_scope() -> None:
    stale = sorted(PARAM_SPEC_TODO - set(_SCOPE_MODULES))
    assert not stale, (
        f"PARAM_SPEC_TODO names modules with no spec-required Config (remove): {stale}"
    )


@pytest.mark.parametrize("rel", _SCOPE_MODULES)
def test_config_module_has_valid_param_spec(rel: str) -> None:
    src = (ra.repo_root() / rel).read_text()
    required = spec_required_classes(src)
    eligible = all_float_field_names(src)
    spec = extract_param_spec(src)
    if spec is None:
        assert rel in PARAM_SPEC_TODO, (
            f"{rel} defines a tunable-parameter Config but has no ``__param_spec__``. "
            f"Author one (see this file's docstring); a pre-existing module must be "
            f"in PARAM_SPEC_TODO."
        )
        return
    errors = validate_param_spec(spec, required, eligible)
    assert not errors, f"{rel} has an invalid __param_spec__:\n  " + "\n  ".join(errors)
    assert rel not in PARAM_SPEC_TODO, (
        f"{rel} now has a complete valid __param_spec__ — remove it from "
        f"PARAM_SPEC_TODO (shrink-only)."
    )


def test_todo_entries_still_lack_complete_specs() -> None:
    """A TODO module that has been fully specced+validated must be removed."""
    graduated: list[str] = []
    for rel in sorted(PARAM_SPEC_TODO):
        path = ra.repo_root() / rel
        if not path.is_file():
            continue
        src = path.read_text()
        spec = extract_param_spec(src)
        if spec is None:
            continue
        if not validate_param_spec(spec, spec_required_classes(src), all_float_field_names(src)):
            graduated.append(rel)
    assert not graduated, (
        "PARAM_SPEC_TODO entries now have complete specs — remove them "
        "(shrink-only): " + ", ".join(graduated)
    )


def test_scheme_keys_are_globally_unique() -> None:
    seen: dict[str, str] = {}
    dupes: list[str] = []
    for rel in _SCOPE_MODULES:
        spec = extract_param_spec((ra.repo_root() / rel).read_text())
        if not isinstance(spec, dict):
            continue
        for cls_name, entry in spec.items():
            if not isinstance(entry, dict):
                continue
            sk = entry.get("scheme_key")
            if not isinstance(sk, str):
                continue
            if sk in seen:
                dupes.append(f"{sk!r} in {rel}:{cls_name} and {seen[sk]}")
            else:
                seen[sk] = f"{rel}:{cls_name}"
    assert not dupes, "duplicate scheme_key(s): " + "; ".join(dupes)


def test_todo_files_exist() -> None:
    missing = sorted(rel for rel in PARAM_SPEC_TODO if not (ra.repo_root() / rel).is_file())
    assert not missing, f"PARAM_SPEC_TODO names non-existent files: {missing}"


# Hard physical/mathematical DOMAINS for params whose formula breaks outside them
# (the sigmoid bounds must stay inside the domain so training never enters an
# invalid region — ``lo < default < hi`` alone does not guarantee this). Keyed
# (Class, field) -> (domain_lo, domain_hi); None = unbounded on that side.
# Extend when a new param has a formula-imposed domain (codex land round).
_PARAM_DOMAINS: dict[tuple[str, str], tuple[float | None, float | None]] = {
    # van Genuchten m = 1 - 1/n requires n > 1 (else m <= 0 -> NaN in Se**(-1/m)).
    ("SoilHydraulicsConfig", "n_vg"): (1.0, None),
}


# Field-name tokens that mark a physically [0, 1]-bounded quantity (fraction,
# efficiency, emissivity, albedo, probability). Such a param's sigmoid bounds must
# stay within [0, 1] inclusive, else extended-tier training can produce
# nonphysical values (e.g. super-blackbody emissivity, >100% backscatter).
_FRACTION_NAME_RE = __import__("re").compile(
    r"(emissiv|albedo|efficiency|fraction|_frac\b|\bfrac_)", __import__("re").I
)


def test_bounds_respect_hard_param_domains() -> None:
    errors: list[str] = []
    for rel in _SCOPE_MODULES:
        spec = extract_param_spec((ra.repo_root() / rel).read_text())
        if not isinstance(spec, dict):
            continue
        for cls_name, entry in spec.items():
            if not isinstance(entry, dict):
                continue
            for pname, p in entry.get("params", {}).items():
                if not isinstance(p, dict):
                    continue
                b = p.get("bounds")
                if not (isinstance(b, tuple) and len(b) == 2 and all(isinstance(x, (int, float)) for x in b)):
                    continue
                # 1) explicit formula domains (exclusive limits, e.g. van Genuchten n > 1)
                dom = _PARAM_DOMAINS.get((cls_name, pname))
                if dom is not None:
                    lo_dom, hi_dom = dom
                    if lo_dom is not None and b[0] <= lo_dom:
                        errors.append(f"{rel}:{cls_name}.{pname} bounds lo {b[0]} violates domain > {lo_dom}")
                    if hi_dom is not None and b[1] >= hi_dom:
                        errors.append(f"{rel}:{cls_name}.{pname} bounds hi {b[1]} violates domain < {hi_dom}")
                # 2) generic fraction/efficiency/emissivity/albedo -> [0, 1] inclusive
                if _FRACTION_NAME_RE.search(pname):
                    if b[0] < 0.0:
                        errors.append(f"{rel}:{cls_name}.{pname} fraction bound lo {b[0]} < 0")
                    if b[1] > 1.0:
                        errors.append(f"{rel}:{cls_name}.{pname} fraction bound hi {b[1]} > 1")
    assert not errors, "spec bounds enter an invalid physical domain:\n  " + "\n  ".join(errors)


def _specced_fields(spec) -> dict[str, set[str]]:
    """``{ClassName: {classified_field, ...}}`` from a (possibly partial) spec."""
    out: dict[str, set[str]] = {}
    if not isinstance(spec, dict):
        return out
    for cls, entry in spec.items():
        if isinstance(entry, dict):
            params = entry.get("params", {}) if isinstance(entry.get("params"), dict) else {}
            excluded = entry.get("excluded", {}) if isinstance(entry.get("excluded"), dict) else {}
            out[cls] = set(params) | set(excluded)
    return out


def test_field_baseline_in_sync_with_todo() -> None:
    stale = sorted(set(PARAM_SPEC_FIELD_BASELINE) - PARAM_SPEC_TODO)
    assert not stale, (
        f"PARAM_SPEC_FIELD_BASELINE has entries for modules no longer in "
        f"PARAM_SPEC_TODO (a graduated module must drop its field baseline): {stale}"
    )
    missing = sorted(PARAM_SPEC_TODO - set(PARAM_SPEC_FIELD_BASELINE))
    assert not missing, (
        f"PARAM_SPEC_TODO modules missing a field baseline (additions would be "
        f"unguarded): {missing}"
    )


def unspecced_additions(required, specced, baseline) -> list[str]:
    """``Class.field`` names that are float-default + unspecced + not in the seed
    baseline (i.e. newly-added tunables that dodged classification)."""
    current = {
        f"{cls}.{fld}"
        for cls, flds in required.items()
        for fld in flds
        if fld not in specced.get(cls, set())
    }
    return sorted(current - set(baseline))


def test_todo_modules_gain_no_unspecced_float_field() -> None:
    """Field-level floor: a TODO module must not GAIN a new float-default field
    without classifying it (spec entry or exclusion). Closes the module-granular
    escape hatch — adding a tunable to a pre-existing config can no longer slip
    through unspecced just because the whole module is allowlisted."""
    errors: list[str] = []
    for rel in sorted(PARAM_SPEC_TODO):
        path = ra.repo_root() / rel
        if not path.is_file():
            continue
        src = path.read_text()
        new_unspecced = unspecced_additions(
            spec_required_classes(src),
            _specced_fields(extract_param_spec(src)),
            PARAM_SPEC_FIELD_BASELINE.get(rel, frozenset()),
        )
        if new_unspecced:
            errors.append(
                f"{rel}: new unclassified float field(s) {new_unspecced} — add a "
                f"__param_spec__ entry (units/bounds/tunable_tier) or list it in "
                f"'excluded' with a reason."
            )
    assert not errors, "param-spec field ratchet failed:\n  " + "\n  ".join(errors)


# ---------------------------------------------------------------------------
# Non-vacuity self-tests
# ---------------------------------------------------------------------------
_GOOD_REQUIRED = {"FooConfig": {"tau": 7200.0, "rh": 0.7}}
_GOOD_SPEC = {
    "FooConfig": {
        "scheme_key": "foo",
        "excluded": {"rh": "numerics: smoothing"},
        "params": {
            "tau": {
                "units": "s",
                "bounds": (1800.0, 21600.0),
                "tunable_tier": 1,
                "transform": "sigmoid",
                "category": "closure",
                "reference": "Author (2024)",
                "shape": None,
                "legacy_name": "foo_tau",
            },
        },
    },
}


def test_validator_accepts_good_spec() -> None:
    assert validate_param_spec(_GOOD_SPEC, _GOOD_REQUIRED) == []


def test_validator_flags_unclassified_float_field() -> None:
    spec = {"FooConfig": {"scheme_key": "foo", "params": {}}}
    errs = validate_param_spec(spec, _GOOD_REQUIRED)
    assert any("not in params/excluded" in e for e in errs)


def test_validator_flags_default_outside_bounds() -> None:
    bad = {
        "FooConfig": {
            "scheme_key": "foo",
            "excluded": {"rh": "x"},
            "params": {
                "tau": {**_GOOD_SPEC["FooConfig"]["params"]["tau"], "bounds": (1.0, 100.0)},
            },
        }
    }
    errs = validate_param_spec(bad, _GOOD_REQUIRED)
    assert any("outside sigmoid bounds" in e for e in errs)


def test_validator_flags_bad_tier_transform_and_orphan() -> None:
    assert validate_param_spec(
        {"FooConfig": {"scheme_key": "f", "excluded": {"rh": "x"},
                       "params": {"tau": {**_GOOD_SPEC["FooConfig"]["params"]["tau"], "tunable_tier": 9}}}},
        _GOOD_REQUIRED,
    )
    assert validate_param_spec(
        {"FooConfig": {"scheme_key": "f", "excluded": {"rh": "x"},
                       "params": {"tau": {**_GOOD_SPEC["FooConfig"]["params"]["tau"], "transform": "relu"}}}},
        _GOOD_REQUIRED,
    )
    # orphan param name (not a real float field)
    assert validate_param_spec(
        {"FooConfig": {"scheme_key": "f", "excluded": {"rh": "x", "tau": "y"},
                       "params": {"ghost": _GOOD_SPEC["FooConfig"]["params"]["tau"]}}},
        _GOOD_REQUIRED,
    )


def test_validator_flags_nonliteral_and_nondict() -> None:
    assert validate_param_spec("__UNPARSEABLE__", _GOOD_REQUIRED)
    assert validate_param_spec([1, 2], _GOOD_REQUIRED)


def test_extractor_handles_assign_annassign_and_nonliteral() -> None:
    assert extract_param_spec("__param_spec__ = {'a': 1}\n") == {"a": 1}
    assert extract_param_spec("__param_spec__: dict = {'a': 1}\n") == {"a": 1}
    assert extract_param_spec("x = 1\n") is None
    assert extract_param_spec("__param_spec__ = dict(a=1)\n") == "__UNPARSEABLE__"


def test_field_ratchet_flags_new_field_but_not_specced_or_baselined() -> None:
    req = {"C": {"a": 1.0, "b": 2.0}}
    # b is a new float field, not specced, not in baseline -> flagged
    assert unspecced_additions(req, {}, frozenset({"C.a"})) == ["C.b"]
    # b classified by a (partial) spec -> not flagged
    assert unspecced_additions(req, {"C": {"b"}}, frozenset({"C.a"})) == []
    # both in baseline (seed) -> not flagged; a removed field never flags
    assert unspecced_additions(req, {}, frozenset({"C.a", "C.b", "C.gone"})) == []


def test_spec_cannot_classify_an_int_iteration_count() -> None:
    """Guarantee: a discrete int field (loop-iteration count etc.) cannot be made
    a trainable param. It is not in the float-default set, so naming it in
    ``params`` is rejected as an orphan — keeping iteration counts out of any
    tuning/backprop path by construction."""
    required = {"C": {"tau": 7200.0}}  # only the float field is spec-eligible
    spec = {
        "C": {
            "scheme_key": "c",
            "params": {
                "n_iter": {  # an int iteration count — must NOT be classifiable
                    "units": "1", "bounds": (1.0, 100.0), "tunable_tier": 1,
                    "transform": "sigmoid", "category": "numerics",
                    "reference": "r", "shape": None,
                },
            },
        }
    }
    errs = validate_param_spec(spec, required)
    assert any("non-float" in e or "not in params/excluded" in e for e in errs)


def test_constants_default_field_is_spec_eligible_not_required() -> None:
    """A ``: float`` field whose default is a ``constants.X`` reference is NOT
    forced into the spec (not in ``required``) but MAY be classified as a tunable
    param (it is ``eligible``) — e.g. a closure that defaults to a constant."""
    src = (
        "from legoesm import constants\n"
        "class C(NamedTuple):\n"
        "    s_new: float = constants.S_ice_bulk_default\n"
        "    floor: float = 0.0\n"
    )
    assert spec_required_classes(src) == {"C": {"floor": 0.0}}        # only literal
    assert all_float_field_names(src) == {"C": {"s_new", "floor"}}     # both eligible
    required = spec_required_classes(src)
    eligible = all_float_field_names(src)
    good = {
        "C": {
            "scheme_key": "c", "excluded": {"floor": "numerics: floor"},
            "params": {
                "s_new": {
                    "units": "PSU", "bounds": (1.0, 12.0), "tunable_tier": 2,
                    "transform": "sigmoid", "category": "closure",
                    "reference": "r", "shape": None,
                },
            },
        }
    }
    assert validate_param_spec(good, required, eligible) == []          # s_new accepted
    # a truly non-existent field is still an orphan
    bad = {"C": {"scheme_key": "c", "excluded": {"floor": "x", "ghost": "y"}, "params": {}}}
    assert any("non-float fields" in e for e in validate_param_spec(bad, required, eligible))


def test_eligible_only_class_is_still_fully_validated() -> None:
    """A class whose float fields ALL default to constants.X (required={}, but
    eligible) must still be schema-validated when it appears in the spec — not
    silently skipped (else a malformed entry passes CI then fails the collector)."""
    required: dict = {}
    eligible = {"C": {"s_new"}}
    bad = {"C": {"scheme_key": "c", "params": {"s_new": {"units": "PSU"}}}}
    assert any("missing keys" in e for e in validate_param_spec(bad, required, eligible))
    good = {
        "C": {
            "scheme_key": "c",
            "params": {
                "s_new": {
                    "units": "PSU", "bounds": (1.0, 12.0), "tunable_tier": 2,
                    "transform": "sigmoid", "category": "closure",
                    "reference": "r", "shape": None,
                },
            },
        }
    }
    assert validate_param_spec(good, required, eligible) == []


def test_inclusion_predicate_excludes_constants_and_nonfloat() -> None:
    src = (
        "from legoesm import constants\n"
        "class C(NamedTuple):\n"
        "    tau: float = 7200.0\n"
        "    kappa: float = constants.kappa_vk\n"
        "    n: int = 2\n"
        "    name: str = 'x'\n"
        "    flag: bool = False\n"
    )
    assert spec_required_classes(src) == {"C": {"tau": 7200.0}}
