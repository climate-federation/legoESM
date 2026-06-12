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
default is auto-exempt — it cannot be a stray tunable). A module with at least one
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
        "packages/atmosphere/legoesm/atmosphere/physics/clouds/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/convection/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/fast_sbm/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/ml_parameterization.py",
        "packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/config.py",
        "packages/atmosphere/legoesm/atmosphere/physics/turbulence/pbl_height.py",
        # --- ocean ---------------------------------------------------------
        "packages/ocean/legoesm/ocean/physics/bottom_drag/config.py",
        "packages/ocean/legoesm/ocean/physics/convection/config.py",
        "packages/ocean/legoesm/ocean/physics/ice_shelf.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/backscatter.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py",
        "packages/ocean/legoesm/ocean/physics/lateral_mixing/eke.py",
        "packages/ocean/legoesm/ocean/physics/shortwave_penetration.py",
        "packages/ocean/legoesm/ocean/physics/surface_forcing/config.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py",
        "packages/ocean/legoesm/ocean/physics/vertical_mixing/tidal.py",
        # --- land ----------------------------------------------------------
        "packages/land/legoesm/land/carbon/config.py",
        "packages/land/legoesm/land/carbon/stomata.py",
        "packages/land/legoesm/land/config.py",
        "packages/land/legoesm/land/richards.py",
        "packages/land/legoesm/land/soil_grid.py",
        "packages/land/legoesm/land/soil_hydraulics.py",
        # --- ice -----------------------------------------------------------
        "packages/ice/legoesm/ice/config.py",
        # --- coupler -------------------------------------------------------
        "packages/coupler/legoesm/coupler/config.py",
        "packages/coupler/legoesm/coupler/lake/config.py",
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
    subclasses that have at least one float-default field."""
    tree = ast.parse(src)
    out: dict[str, dict[str, float]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_namedtuple(node):
            fields = _float_default_fields(node)
            if fields:
                out[node.name] = fields
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
        elif transform == "sigmoid" and not (lo <= default <= hi):
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


def validate_param_spec(spec, required: dict[str, dict[str, float]]) -> list[str]:
    """Schema + completeness errors for one module's ``__param_spec__``."""
    if spec == "__UNPARSEABLE__":
        return ["__param_spec__ is not a pure literal (AST literal-eval failed)"]
    if not isinstance(spec, dict):
        return ["__param_spec__ must be a dict keyed by Config class name"]
    errors: list[str] = []
    # Orphan classes: a spec entry for a class that is not spec-required.
    for cls_name in spec:
        if cls_name not in required:
            errors.append(f"spec names class {cls_name!r} with no float-default fields")
    for cls_name, fields in required.items():
        if cls_name not in spec:
            errors.append(f"class {cls_name!r} is spec-required but absent from spec")
            continue
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
        orphan = sorted(classified - set(fields))
        if orphan:
            errors.append(f"{cls_name}: params/excluded name non-float fields: {orphan}")
        for ename, reason in excluded.items():
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{cls_name}.excluded[{ename!r}] must be a non-empty reason string")
        for pname, p in params.items():
            if pname not in fields:
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
                errors.extend(
                    f"{cls_name}.{pname}: {e}"
                    for e in _validate_bounds(p["bounds"], fields[pname], tier, p.get("transform"))
                )
    return errors


# --- discovery / partition -------------------------------------------------
def _in_scope(rel: str) -> bool:
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
        "packages/atmosphere/legoesm/atmosphere/physics/convection/config.py",
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
    spec = extract_param_spec(src)
    if spec is None:
        assert rel in PARAM_SPEC_TODO, (
            f"{rel} defines a tunable-parameter Config but has no ``__param_spec__``. "
            f"Author one (see this file's docstring); a pre-existing module must be "
            f"in PARAM_SPEC_TODO."
        )
        return
    errors = validate_param_spec(spec, required)
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
        if not validate_param_spec(spec, spec_required_classes(src)):
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
