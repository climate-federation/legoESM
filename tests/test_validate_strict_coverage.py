"""Coverage ratchet: every scheme-like ``ExperimentConfig`` field is membership-
validated in ``validate_strict`` (so a *new* scheme literal cannot silently skip
early validation).

This is the config-level complement to ``test_dispatch_hardening`` (factory-level):
C3 makes the *factory* raise on an unknown scheme; this makes the *config* reject
it up front, before JIT/data-loading, with one collected error message. CLAUDE.md
flags the exact gap — *"convection/turbulence/gravity_wave_drag … typos pass early
validation, fail only at JIT inside integration.py. Add them."* (now fixed; this
test keeps them fixed and forces new fields to follow suit).

Mechanism (three layers, so no single loose heuristic carries the guarantee):
  * **Behavioural teeth (ground truth)** — for *every* ``EXPECTED_VALIDATED``
    field a bogus value is fed through ``validate_strict`` and must raise
    ``ValueError``. This is immune to AST-matching subtleties.
  * **Structural regression** — each ``EXPECTED_VALIDATED`` field must carry a
    real *membership* check (``not in``/``in``) in ``validate_strict`` source;
    ``carbon_cycle`` is the one equality-rejection (``!= "none"``). A plain
    cross-field compatibility ``==`` does NOT count (it would otherwise mask a
    deleted membership guard).
  * **Completeness** — scheme-like fields are detected two ways (an inline
    comment enumerating ≥2 alternatives, AND a scheme-like name such as
    ``*_scheme``/``*_source``/``*_forcing`` or a curated name). Every detected
    field must be either ``EXPECTED_VALIDATED`` or in the documented
    ``KNOWN_UNVALIDATED`` gap list — so a NEW scheme field, with or without an
    enum comment, cannot slip unclassified.

``KNOWN_UNVALIDATED`` is the current, documented gap (forcing-source / topography
selectors validated at their loader, not fail-early — cf. grid_type at
grids.factory). Shrink-only: add a ``validate_strict`` membership check and move
the field to ``EXPECTED_VALIDATED``.

A tripwire, not a proof: it checks an unknown value is *rejected*, not that the
accepted set is physically right. Self-tests prove non-vacuity.
"""

from __future__ import annotations

import ast
import re

import pytest

from tests import _ratchet_audit as ra

_CONFIG_REL = "packages/coupler/legoesm/driver/config.py"

# Fields that MUST be membership-checked (``not in``/``in``) in validate_strict.
EXPECTED_VALIDATED: frozenset[str] = frozenset(
    {
        "radiation",
        # grid.vertical_coord: 'sigma' | 'hybrid' | 'cam_l32' (the CAM6 L32
        # table); membership-checked in validate_strict since the cam_l32 lane.
        "vertical_coord",
        "cloud_scheme",
        "cloud_diagnostic_condensate_scheme",
        # Cloud-fraction RH saturation curve ("liquid" | "mixed_phase"), the
        # #1521 ice-saturation fix; membership-checked in validate_strict.
        "cloud_saturation_scheme",
        "microphysics",
        "convection",
        "turbulence",
        "surface_bulk_scheme",
        "surface_thermo_convention",
        "surface_stability_scheme",  # already guarded (config.py); ratchet entry
        "land_surface_scheme",
        "gravity_wave_drag",
        "e3sm_cam_source",  # membership check in validate_strict (background source)
        "physics_parameterization",
        "ic",
        "precision",
        "aimip_variant",
        "model_type",  # nested dycore.model_type
        "discretization",  # nested dycore.discretization
        "bechtold_subsidence_solve",  # Bechtold vertical solve (day-65 bisect)
        # nested dycore.mpas_vert_advection_scheme — sigma-lane vertical
        # advection ("upwind" | "van_leer"); membership + silently-inert refusal.
        "mpas_vert_advection_scheme",
        # External-forcing source selectors (2026-07-21 AMIP/CMIP audit):
        # the driver gates each channel with an equality test, so a typo
        # silently deactivated the channel before these membership checks.
        "aerosol_forcing",
        "ghg_forcing",
        "ozone_forcing",
        "ozone_source",
        "solar_source",
        "solar_spectral_band_order",
        "dataset",
        "experiment",
    }
)
# Validated by equality-rejection rather than a set membership: ``carbon_cycle``
# is rejected unless ``== "none"`` (coupled mode unsupported in this driver).
EQUALITY_VALIDATED: frozenset[str] = frozenset({"carbon_cycle"})

# Scheme/dispatch fields NOT membership-validated in validate_strict today.
# Validated at their forcing loader / builder / factory (cf. grid_type at
# grids.factory.create_grid, time_integrator at timestepping.dispatch — both C3),
# not fail-early here. Shrink-only: add a validate_strict membership check and
# move the field to EXPECTED_VALIDATED.
KNOWN_UNVALIDATED: frozenset[str] = frozenset(
    {
        # (2026-07-21 AMIP/CMIP audit: the forcing-source selectors moved to
        # EXPECTED_VALIDATED — the driver's equality-gate activation made a
        # typo a silent channel-off, not a loud loader error.)
        "topography",
        # nested GridConfig / DycoreConfig selectors (validated at factory)
        # grid_type: validated at grids.factory.create_grid, NOT fail-early.
        # validate_strict does contain ``grid.grid_type not in ("mpas",
        # "voronoi")`` but that is an incidental FEATURE gate (a continuation
        # line of a compound condition), not a legal-set check -- which is why
        # _has_membership below only counts a membership test that HEADS an
        # if/elif.
        "grid_type",        # grids.factory.create_grid (C3)
        "time_integrator",  # timestepping.dispatch.dispatch_integrator (C3)
    }
)

# Bogus values for behavioural teeth. ``carbon_cycle`` rejects any non-"none".
_BOGUS = "__nonexistent_scheme__"

# Config classes whose scheme-like str fields the completeness scan covers
# (top-level + the nested configs that carry dispatch selectors).
_SCANNED_CLASSES = ("ExperimentConfig", "GridConfig", "DycoreConfig")

# Scheme-like name signals (suffixes + curated exact names) for completeness.
_SCHEME_NAME_RE = re.compile(r"(_scheme|_source|_forcing|_parameterization|_type|integrator)$")
_SCHEME_NAME_CURATED = frozenset(
    {"radiation", "microphysics", "convection", "turbulence", "ic", "topography"}
)
# Inline comment enumerating ≥2 alternatives, with an optional trailing issue ref
# like ``(#322)`` → a scheme field (codebase idiom).
_ENUM_COMMENT = re.compile(
    r"#\s*[a-z0-9_]+(\s*[,|]\s*[a-z0-9_]+)+(\s*\(#?\d+\))?\s*$", re.I
)


def _config_ast() -> tuple[ast.ClassDef, ast.FunctionDef, str, list[str]]:
    src = ra.canonical_source_path(_CONFIG_REL).read_text()
    tree = ast.parse(src)
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "ExperimentConfig"
    )
    vstrict = next(
        n for n in ast.walk(cls)
        if isinstance(n, ast.FunctionDef) and n.name == "validate_strict"
    )
    return cls, vstrict, src, src.splitlines()


def _scheme_like_fields() -> set[str]:
    src = ra.canonical_source_path(_CONFIG_REL).read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    by_name = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name in _SCANNED_CLASSES
    }
    missing = [c for c in _SCANNED_CLASSES if c not in by_name]
    assert not missing, f"config classes not found (renamed?): {missing}"
    fields: set[str] = set()
    for cls in by_name.values():
        for n in cls.body:
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                ann = ast.get_source_segment(src, n.annotation) or ""
                if "str" not in ann:
                    continue
                name = n.target.id
                if (
                    _SCHEME_NAME_RE.search(name)
                    or name in _SCHEME_NAME_CURATED
                    or _ENUM_COMMENT.search(lines[n.lineno - 1])
                ):
                    fields.add(name)
    return fields


def _validate_strict_source() -> str:
    _cls, vstrict, src, _lines = _config_ast()
    return ast.get_source_segment(src, vstrict) or ""


def _has_membership(field: str, vsrc: str) -> bool:
    """``<accessor>.<field> (not) in ...`` — the real fail-early guard. Word
    boundary around ``in`` so it doesn't match ``index``.

    The membership test must HEAD an ``if``/``elif``: a membership that only
    appears on a continuation line of a compound condition (``... and
    self.grid.grid_type not in ("mpas", "voronoi")``) is a feature gate for
    some other switch, not a validation of this field's legal set, and
    counting it marked a real KNOWN_UNVALIDATED entry stale."""
    return any(
        re.search(rf"\.{field}\b\s*(?:not\s+in\b|in\b)", line)
        for line in vsrc.splitlines()
        if re.match(r"\s*(?:el)?if\s", line)
    )


def _has_equality_reject(field: str, vsrc: str) -> bool:
    return bool(re.search(rf"\.{field}\b\s*(?:!=|==)", vsrc))


@pytest.mark.parametrize("field", sorted(EXPECTED_VALIDATED | EQUALITY_VALIDATED))
def test_scheme_field_is_validated(field: str) -> None:
    vsrc = _validate_strict_source()
    if field in EQUALITY_VALIDATED:
        ok = _has_equality_reject(field, vsrc)
    else:
        ok = _has_membership(field, vsrc)
    assert ok, (
        f"ExperimentConfig.validate_strict no longer membership-checks {field!r}. "
        f"A typo in this scheme field would pass config validation and fail only "
        f"deep at JIT/factory time. Restore the ``if self.{field} not in (...): "
        f"errors.append(...)`` guard (CLAUDE.md dispatch rule)."
    )


def test_every_scheme_like_field_is_classified() -> None:
    """Completeness: every scheme-like field (by enum comment OR scheme-like name)
    must be validated or explicitly allow-listed — so a NEW one cannot slip."""
    detected = _scheme_like_fields()
    classified = EXPECTED_VALIDATED | EQUALITY_VALIDATED | KNOWN_UNVALIDATED
    unclassified = sorted(detected - classified)
    assert not unclassified, (
        "New scheme-like ExperimentConfig field(s) are neither validated in "
        "validate_strict nor allow-listed:\n  " + "\n  ".join(unclassified)
        + "\nAdd a validate_strict membership check (preferred) + EXPECTED_VALIDATED, "
        "or, if validated at a loader, add to KNOWN_UNVALIDATED with a reason."
    )


def test_known_unvalidated_is_shrink_only_and_real() -> None:
    """Allow-list hygiene: each KNOWN_UNVALIDATED entry must still be a real
    scheme-like field that genuinely lacks a fail-early check (else remove it).

    Asks the QUESTION DIRECTLY — does ``validate_strict`` reject a bogus value
    for this field? — instead of pattern-matching its source. That matters
    because the completeness test above already uses the source detector: if
    both tests shared one detector, a single detector blind spot would satisfy
    BOTH at once and the pair would enforce nothing for that field (GLM review).
    A behavioural probe also cannot be fooled by a compound feature gate such as
    ``... and self.grid.grid_type not in ("mpas", "voronoi")``, which rejects
    nothing about grid_type's own legal set."""
    detected = _scheme_like_fields()
    stale = sorted(
        f for f in KNOWN_UNVALIDATED
        if f not in detected or _bogus_is_rejected(f)
    )
    assert not stale, (
        f"KNOWN_UNVALIDATED entries are stale (validate_strict now REJECTS a "
        f"bogus value, or the field is no longer scheme-like) — remove them "
        f"and add the field to EXPECTED_VALIDATED: {stale}"
    )


def test_classification_sets_are_disjoint() -> None:
    sets = {
        "EXPECTED_VALIDATED": EXPECTED_VALIDATED,
        "EQUALITY_VALIDATED": EQUALITY_VALIDATED,
        "KNOWN_UNVALIDATED": KNOWN_UNVALIDATED,
    }
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sorted(sets[names[i]] & sets[names[j]])
            assert not overlap, f"{names[i]} & {names[j]} overlap: {overlap}"


# ---------------------------------------------------------------------------
# Behavioural teeth — feed a bogus value through validate_strict; must raise.
# Covers EVERY EXPECTED_VALIDATED + EQUALITY_VALIDATED field (ground truth).
# ---------------------------------------------------------------------------
def _bogus_raises(**kwargs) -> bool:
    from legoesm.driver.config import ExperimentConfig

    try:
        ExperimentConfig(**kwargs).validate_strict()
    except ValueError:
        return True
    return False


def _bogus_is_rejected(field: str) -> bool:
    """Does ``validate_strict`` reject a bogus value for *field*, with every
    other field left at its valid default?

    Nested selectors are routed into the sub-config that owns them, so a nested
    field is probed as honestly as a top-level one. A field whose owner cannot
    be determined returns False (treated as NOT validated), which is the safe
    direction: it keeps the allow-list entry rather than silently dropping a
    field out of both tests.
    """
    from legoesm.driver.config import DycoreConfig, GridConfig, OutputConfig

    _NESTED = {
        "grid": (GridConfig, ("grid_type", "vertical_coord")),
        "dycore": (DycoreConfig, ("model_type", "discretization",
                                  "time_integrator", "mpas_vert_advection_scheme")),
        "output": (OutputConfig, ("checkpoint_format",)),
    }
    for kw, (cls, fields) in _NESTED.items():
        if field in fields:
            try:
                return _bogus_raises(**{kw: cls(**{field: _BOGUS})})
            except (TypeError, ValueError):
                # The sub-config itself rejected the bogus value at construction
                # (a Literal/enum type), which is still a fail-early rejection.
                return True
    try:
        return _bogus_raises(**{field: _BOGUS})
    except TypeError:
        return False


def test_default_config_is_valid() -> None:
    """Anti-vacuity guard for the teeth: the default config must pass, so a raise
    below is attributable to the bogus field, not a broken baseline."""
    from legoesm.driver.config import ExperimentConfig

    ExperimentConfig().validate_strict()  # must not raise


@pytest.mark.parametrize(
    "field",
    sorted(
        EXPECTED_VALIDATED
        # nested DycoreConfig / GridConfig fields; tested separately below
        - {"model_type", "discretization", "mpas_vert_advection_scheme", "vertical_coord"}
        | EQUALITY_VALIDATED
    ),
)
def test_validate_strict_rejects_bogus_scheme(field: str) -> None:
    assert _bogus_raises(**{field: _BOGUS}), (
        f"ExperimentConfig.validate_strict accepted bogus {field}={_BOGUS!r}"
    )


def test_validate_strict_rejects_bogus_nested_dycore() -> None:
    from legoesm.driver.config import DycoreConfig

    assert _bogus_raises(dycore=DycoreConfig(model_type=_BOGUS))
    assert _bogus_raises(dycore=DycoreConfig(discretization=_BOGUS))
    assert _bogus_raises(dycore=DycoreConfig(mpas_vert_advection_scheme=_BOGUS))


def test_ocean_validate_strict_rejects_bogus_shortwave_selector() -> None:
    """The nested ocean qsr selector must fail before build/JIT."""
    from legoesm.ocean.config import OceanExperimentConfig

    config = OceanExperimentConfig.from_dict({"ocean": {"physics": {
        "shortwave_penetration": {"scheme": _BOGUS},
    }}})
    with pytest.raises(ValueError, match="shortwave_penetration.scheme"):
        config.validate_strict()


def test_validate_strict_rejects_bogus_nested_grid() -> None:
    from legoesm.driver.config import GridConfig

    assert _bogus_raises(grid=GridConfig(vertical_coord=_BOGUS))


# ---------------------------------------------------------------------------
# Non-vacuity self-tests
# ---------------------------------------------------------------------------
def test_detector_finds_scheme_fields() -> None:
    detected = _scheme_like_fields()
    # comment-less (radiation) AND enum-comment (convection) fields both found.
    assert {"radiation", "convection", "cloud_scheme", "ozone_source"} <= detected


def test_membership_checker_distinguishes_membership_from_equality() -> None:
    assert _has_membership("convection", "if self.convection not in _v: e()")
    assert not _has_membership("aimip_variant", "if self.aimip_variant == 'classical':")
    assert _has_equality_reject("carbon_cycle", 'if self.carbon_cycle != "none":')
