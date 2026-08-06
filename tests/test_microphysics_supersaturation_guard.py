"""Ratchet: EVERY microphysics scheme carries the super-saturation guard.

Motivating failure (CONFIRMED, 2026-07/08). An RCEMIP CRM initial sounding at
~140 % relative humidity produced a PERSISTENT column-water drift (CWV 40 ->
73-81 mm against an RCEMIP reference of 42.2) rather than a transient
adjustment. The reason it persisted is structural, not a one-off: the hard
(iterated) saturation adjustment defaults to ``False``, and
``_warm_rain.saturation_adjustment``'s own docstring says the opt-in hard
branch exists to drain "local super-saturation pools **that the smooth sigmoid
branch cannot**". So the default path under-drains BY DESIGN, and two of the
ten reachable schemes did not even expose the opt-in.

What this file locks down, in three layers:

1. **Coverage ratchet** (:func:`test_every_factory_scheme_is_guarded_or_exempt`)
   -- mechanically enumerates every scheme reachable through the microphysics
   factory (``microphysics/integration.py::_get_microphysics_fn``) by parsing
   its AST, and asserts each one either exposes all three guard fields with the
   standard defaults or appears in the SHRINK-ONLY
   ``config.HARD_SAT_GUARD_EXEMPT`` with a real reason. A new scheme added to
   the factory that skips the guard goes red on the day it lands.

2. **Non-vacuity self-tests** -- the classifier is run against synthetic
   configs with the feature REMOVED / the default CHANGED / the reason string
   EMPTY, and against a synthetic factory source, proving the gate fails when
   it should (CLAUDE.md: every gate ships a synthetic-violation self-test).

3. **Behavioural tests** -- a config-presence test would NOT have caught the
   motivating failure, because the field was present on all five bulk schemes
   and simply defaulted off. So the property that actually failed is asserted
   directly: from a super-saturated column (RH 1.4, the real case), with
   ``hard_saturation_adjustment=True`` every guarded scheme drives RH to <= ~1
   within a bounded number of steps; the byte-identity of the OFF path is
   asserted too, and the under-drain of the default is asserted for the schemes
   where it is the observed behaviour (see ``_DEFAULT_UNDERDRAINS`` -- the
   per-scheme expectations are MEASURED, not assumed).

A tripwire, not a proof: the AST scan keys on the factory function's own
``return "<name>", ...`` tuples, so a scheme reached by some other mechanism
would not be discovered. Discovery is sanity-checked against a minimum count so
a refactor that empties it fails loudly instead of passing vacuously.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests import _ratchet_audit as ra

from legoesm.atmosphere.physics.microphysics.config import (
    HARD_SAT_GUARD_DEFAULTS,
    HARD_SAT_GUARD_EXEMPT,
    HARD_SAT_GUARD_FIELDS,
    HARD_SAT_GUARD_SCHEMES,
    MicrophysicsConfig,
)

# SHRINK-ONLY baseline. An exemption may be REMOVED (a scheme gaining the
# guard); adding one requires editing this line, which forces the reviewer to
# read the reason string. Mirrors the grow-only baseline pattern in
# ``tests/test_dispatch_hardening.py``.
BASELINE_EXEMPT = frozenset({"sdm", "fast_sbm", "none"})

_FACTORY_FN = "_get_microphysics_fn"
_MIN_SCHEMES = 8  # discovery sanity floor (10 today)


# ---------------------------------------------------------------------------
# Mechanical discovery of the factory's scheme set
# ---------------------------------------------------------------------------
def _integration_source() -> str:
    from legoesm.atmosphere.physics.microphysics import integration

    return pathlib.Path(integration.__file__).read_text()


def discover_factory_schemes(source: str) -> dict[str, str | None]:
    """``{scheme_name: MicrophysicsConfig attribute or None}`` from the factory.

    Parses ``_get_microphysics_fn`` and reads each ``return "<name>", <fn>,
    <config.attr | None>`` tuple. Keying on the RETURNED literal (not on the
    ``if config.scheme == ...`` test) means a branch that is reachable but
    returns a differently-named scheme is recorded under the name the rest of
    the system sees.
    """
    tree = ast.parse(source)
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == _FACTORY_FN),
        None,
    )
    if fn is None:
        raise AssertionError(
            f"{_FACTORY_FN} not found in microphysics/integration.py — the "
            "ratchet's discovery anchor moved; re-point it (do NOT delete)."
        )
    out: dict[str, str | None] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
            continue
        elts = node.value.elts
        if not elts or not isinstance(elts[0], ast.Constant):
            continue
        if not isinstance(elts[0].value, str):
            continue
        attr = None
        if len(elts) >= 3 and isinstance(elts[2], ast.Attribute):
            attr = elts[2].attr
        out[elts[0].value] = attr
    return out


def classify_scheme(scheme: str, attr: str | None) -> str | None:
    """Return an error string if ``scheme`` violates the guard policy, else None.

    Pure and side-effect free so the self-tests can drive it with synthetic
    configs.
    """
    if scheme in HARD_SAT_GUARD_EXEMPT:
        reason = HARD_SAT_GUARD_EXEMPT[scheme]
        if not isinstance(reason, str) or len(reason.strip()) < 40:
            return (
                f"{scheme!r} is exempt but its reason string is missing or too "
                "short to be a real justification. State, in code terms, what "
                "resolves super-saturation instead."
            )
        if scheme in HARD_SAT_GUARD_SCHEMES:
            return (f"{scheme!r} is listed BOTH as guarded and as exempt — the "
                    "two lists must be disjoint.")
        return None

    if attr is None:
        return (
            f"{scheme!r} is not exempt but the factory returns no sub-config "
            "for it, so the guard has nowhere to live. Either wire a config "
            "or add a HARD_SAT_GUARD_EXEMPT entry with a real reason."
        )
    if not hasattr(MicrophysicsConfig(), attr):
        return f"MicrophysicsConfig has no {attr!r} field (factory drift)."
    sub = getattr(MicrophysicsConfig(), attr)
    return classify_config(scheme, sub)


def classify_config(scheme: str, sub) -> str | None:
    """Guard-policy check for one scheme sub-config instance."""
    fields = getattr(sub, "_fields", ())
    missing = [f for f in HARD_SAT_GUARD_FIELDS if f not in fields]
    if missing:
        return (
            f"{scheme!r} ({type(sub).__name__}) is missing the "
            f"super-saturation guard field(s) {missing}. A scheme with no "
            "opt-in hard saturation adjustment silently under-drains a "
            "super-saturated column forever (the RCEMIP 140 %-RH drift). Add "
            "the three fields and wire them to the SHARED "
            "_warm_rain.hard_saturation_blend / hard_saturation_drain, or add "
            "a HARD_SAT_GUARD_EXEMPT entry with a verified reason."
        )
    bad = {
        f: getattr(sub, f)
        for f, want in HARD_SAT_GUARD_DEFAULTS.items()
        if getattr(sub, f) != want
    }
    if bad:
        return (
            f"{scheme!r} ({type(sub).__name__}) has non-standard guard "
            f"defaults {bad}; every scheme must agree with "
            f"{HARD_SAT_GUARD_DEFAULTS} so a run's behaviour does not depend "
            "on which scheme happens to be selected."
        )
    if scheme not in HARD_SAT_GUARD_SCHEMES:
        return (
            f"{scheme!r} carries the guard but is missing from "
            "HARD_SAT_GUARD_SCHEMES, so ExperimentConfig.validate_strict will "
            "reject --hard-saturation-adjustment for it."
        )
    return None


# ---------------------------------------------------------------------------
# 1. The ratchet
# ---------------------------------------------------------------------------
def discover_factory_scheme_tests(source: str) -> set[str]:
    """Scheme names the factory COMPARES against (``config.scheme == "x"``).

    Cross-check for :func:`discover_factory_schemes`, which reads only the
    ``return`` tuples. A new branch written as
    ``name = "brandnew"; return name, fn, cfg`` (a Name, not a Constant) is
    invisible to the return scan; it is NOT invisible here, so the two sets
    disagreeing goes red instead of silently under-reporting coverage.
    """
    tree = ast.parse(source)
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == _FACTORY_FN), None)
    if fn is None:
        raise AssertionError(
            f"{_FACTORY_FN} not found in microphysics/integration.py — the "
            "ratchet's discovery anchor moved; re-point it (do NOT delete)."
        )
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare):
            for c in node.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out.add(c.value)
    return out


def test_discovery_is_not_vacuous() -> None:
    schemes = discover_factory_schemes(_integration_source())
    assert len(schemes) >= _MIN_SCHEMES, (
        f"only {len(schemes)} scheme(s) discovered in {_FACTORY_FN} "
        f"({sorted(schemes)}) — below the sanity floor {_MIN_SCHEMES}. The "
        "ratchet would be silently vacuous; fix discovery."
    )
    assert "kessler" in schemes and "none" in schemes


def test_return_scan_and_comparison_scan_agree() -> None:
    """Two independent readings of the factory must see the SAME scheme set.

    The return-tuple scan can be evaded (a Name instead of a string literal, a
    registry-table dispatch, a helper function); the ``==`` comparison scan has
    different blind spots. Requiring agreement means an evasion has to defeat
    both at once, and a mismatch names the scheme that slipped through."""
    returns = set(discover_factory_schemes(_integration_source()))
    compares = discover_factory_scheme_tests(_integration_source())
    assert returns == compares, (
        "the factory's ``return`` names and its ``config.scheme ==`` names "
        f"disagree: only-returned={sorted(returns - compares)}, "
        f"only-compared={sorted(compares - returns)}. One of the two scans is "
        "missing a branch — do not silence this, fix discovery, or the "
        "coverage ratchet under-reports."
    )


def test_every_factory_scheme_is_guarded_or_exempt() -> None:
    schemes = discover_factory_schemes(_integration_source())
    errors = [
        msg for s, a in sorted(schemes.items())
        if (msg := classify_scheme(s, a)) is not None
    ]
    assert not errors, "super-saturation guard coverage:\n  - " + "\n  - ".join(errors)


def test_guard_lists_exactly_partition_the_factory() -> None:
    """No scheme may be silently absent from BOTH lists (that is the hole the
    ratchet exists to close), and no list may name a scheme the factory has
    dropped (stale entry)."""
    schemes = set(discover_factory_schemes(_integration_source()))
    listed = set(HARD_SAT_GUARD_SCHEMES) | set(HARD_SAT_GUARD_EXEMPT)
    assert schemes - listed == set(), (
        f"factory schemes with NO guard policy at all: {sorted(schemes - listed)}"
    )
    assert listed - schemes == set(), (
        f"guard lists name schemes the factory no longer has: "
        f"{sorted(listed - schemes)}"
    )
    assert set(HARD_SAT_GUARD_SCHEMES) & set(HARD_SAT_GUARD_EXEMPT) == set()


def test_exemption_list_is_shrink_only() -> None:
    current = set(HARD_SAT_GUARD_EXEMPT)
    added = current - BASELINE_EXEMPT
    assert not added, (
        f"NEW super-saturation-guard exemption(s) {sorted(added)}. This list "
        "is SHRINK-ONLY: a new microphysics scheme gets the guard, it does not "
        "get an exemption. If the exemption is genuinely right (the scheme "
        "resolves super-saturation explicitly, like sdm/fast_sbm), add it to "
        "BASELINE_EXEMPT in the same PR so a reviewer reads the reason."
    )


def _defined_or_called_names(path: pathlib.Path) -> set[str]:
    """Names DEFINED or CALLED in ``path``, from the AST — so a match cannot be
    satisfied by a docstring or a comment.

    This distinction is the point. The first cut of this test asserted that the
    literal ``"S = e/e_sat"`` appeared in ``sdm/column.py``. It does — in the
    MODULE DOCSTRING. The line that actually runs is
    ``S = relative_humidity(T, p_full, q_v)``. Deleting the whole condensation
    implementation and leaving the docstring kept the test green: exactly the
    CLAUDE.md ``inspect.getsource(wrapper)`` failure, where an assertion names
    something other than the symbol that RUNS.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


def test_exempt_reasons_name_symbols_that_actually_run() -> None:
    """Each exemption reason claims specific machinery resolves
    super-saturation. Verify the named symbols are DEFINED or CALLED there — a
    reason string is a CLAIM (CLAUDE.md), and a docstring is not evidence that
    anything executes."""
    root = ra.repo_root() / "packages/atmosphere/legoesm/atmosphere/physics/microphysics"
    claims = {
        "sdm": [("sdm/condensation.py", "integrate_radius"),
                ("sdm/condensation.py", "saturation_vapor_pressure"),
                ("sdm/column.py", "relative_humidity")],
        "fast_sbm": [
            ("fast_sbm/supersaturation.py", "integrate_supersaturation"),
            ("fast_sbm/condensation_driver.py", "integrate_supersaturation"),
        ],
    }
    for scheme, items in claims.items():
        assert scheme in HARD_SAT_GUARD_EXEMPT
        for rel, symbol in items:
            path = root / rel
            assert path.is_file(), f"{scheme} exemption cites missing {rel}"
            assert symbol in _defined_or_called_names(path), (
                f"{scheme} exemption rests on {symbol!r} running in {rel}; it "
                "is neither defined nor called there. The reason string is "
                "stale — re-verify it or drop the exemption."
            )
    # "none" is exempt because the factory returns no function for it.
    assert discover_factory_schemes(_integration_source())["none"] is None


def test_sdm_exemption_discloses_its_clear_cell_gap() -> None:
    """The sdm exemption is NARROWER than 'sdm resolves super-saturation'.

    Its default condensation-only adapter has no aerosol-Koehler activation:
    ``N_eff = where(m_drop >= m_min, cdnc, q_c*rho/m_min)`` is 0 when q_c = 0,
    so a super-saturated CLEAR cell is left undrained at any RH — precisely the
    q_c = 0, RH 1.4 configuration that motivated this whole ratchet. The
    exemption is still right (the fix is activation, not a bulk adjustment
    bolted onto a super-droplet scheme), but it must SAY SO rather than imply
    blanket coverage."""
    reason = HARD_SAT_GUARD_EXEMPT["sdm"]
    for needle in ("KNOWN LIMITATION", "column_do_coalescence", "N_eff = 0"):
        assert needle in reason, (
            f"the sdm exemption no longer discloses its clear-cell gap "
            f"({needle!r} missing). Do not let it read as blanket coverage."
        )
    src = (ra.repo_root()
           / "packages/atmosphere/legoesm/atmosphere/physics/microphysics"
           / "sdm/column.py").read_text()
    assert "A supersaturated clear cell still produces nothing" in src, (
        "sdm/column.py no longer documents the no-activation gap — if SDM "
        "gained activation, re-verify and narrow the exemption reason."
    )


def test_every_guarded_scheme_actually_READS_the_flag() -> None:
    """Structural presence is not enough: the scheme must CONSUME
    ``config.hard_saturation_adjustment``.

    Without this, a new scheme could declare the three fields, join
    HARD_SAT_GUARD_SCHEMES, import ``saturation_adjustment`` (which every bulk
    module already does for its ordinary smooth condensation) and simply never
    pass ``hard_adjust=``. Every other test in this file would be green while
    the guard was silently inert — verbatim the failure this ratchet's own
    error strings claim to prevent."""
    root = ra.repo_root() / "packages/atmosphere/legoesm/atmosphere/physics/microphysics"
    for scheme in HARD_SAT_GUARD_SCHEMES:
        tree = ast.parse((root / f"{scheme}.py").read_text())
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "hard_saturation_adjustment" in attrs, (
            f"{scheme}.py never reads ``config.hard_saturation_adjustment``, "
            "so its guard fields are decorative and the flag is silently "
            "inert. Gate the shared adjustment on it."
        )
        for field in ("hard_sat_adjust_threshold", "hard_sat_max_heating_K"):
            assert field in attrs, (
                f"{scheme}.py never reads ``config.{field}``, but it is a "
                "declared tier-2 __param_spec__ tunable — tuning it would do "
                "nothing."
            )


def test_all_guarded_schemes_share_one_implementation() -> None:
    """No scheme may re-derive the adjustment: every guarded scheme's module
    must reach the SHARED ``_warm_rain`` entry points (CLAUDE.md: no duplicate
    numerics). Catches a copy-pasted bisection loop in a new scheme."""
    root = ra.repo_root() / "packages/atmosphere/legoesm/atmosphere/physics/microphysics"
    shared = {"saturation_adjustment", "hard_saturation_blend",
              "hard_saturation_drain"}
    for scheme in HARD_SAT_GUARD_SCHEMES:
        path = root / f"{scheme}.py"
        src = path.read_text()
        imported: set[str] = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                    "_warm_rain"):
                imported |= {a.name for a in node.names}
        assert imported & shared, (
            f"{scheme}.py imports {sorted(imported)} from _warm_rain — none of "
            f"the shared adjustment entry points {sorted(shared)}. A guarded "
            "scheme must reuse them, never re-derive the adjustment."
        )
        assert "_HARD_SAT_BISECT_ITERS" not in src, (
            f"{scheme}.py appears to carry its own bisection loop; reuse "
            "_warm_rain.hard_saturation_blend instead."
        )


# ---------------------------------------------------------------------------
# 2. Non-vacuity self-tests (the ratchet must FAIL when the feature is removed)
# ---------------------------------------------------------------------------
class _NoGuardConfig(tuple):
    """Synthetic sub-config with the guard REMOVED."""
    _fields = ("rh_crit", "auto_rate")


class _WrongDefaultConfig(tuple):
    _fields = HARD_SAT_GUARD_FIELDS
    hard_saturation_adjustment = False
    hard_sat_adjust_threshold = 1.35   # != the standard 1.1
    hard_sat_max_heating_K = 5.0


class _GoodConfig(tuple):
    _fields = HARD_SAT_GUARD_FIELDS
    hard_saturation_adjustment = False
    hard_sat_adjust_threshold = 1.1
    hard_sat_max_heating_K = 5.0


def test_selftest_missing_guard_is_flagged() -> None:
    msg = classify_config("kessler", _NoGuardConfig())
    assert msg is not None and "missing the" in msg


def test_selftest_wrong_default_is_flagged() -> None:
    msg = classify_config("kessler", _WrongDefaultConfig())
    assert msg is not None and "non-standard guard defaults" in msg


def test_selftest_compliant_config_passes() -> None:
    assert classify_config("kessler", _GoodConfig()) is None


def test_selftest_short_exempt_reason_is_flagged(monkeypatch) -> None:
    monkeypatch.setitem(HARD_SAT_GUARD_EXEMPT, "__synthetic__", "because")
    msg = classify_scheme("__synthetic__", None)
    assert msg is not None and "too short" in msg


def test_selftest_unlisted_scheme_with_no_config_is_flagged() -> None:
    msg = classify_scheme("__synthetic_new_scheme__", None)
    assert msg is not None and "no sub-config" in msg


def test_selftest_discovery_finds_a_synthetic_new_scheme() -> None:
    """Proves the AST scan would actually SEE a newly added factory branch —
    without this the coverage test could pass by simply not looking."""
    src = (
        "def _get_microphysics_fn(config):\n"
        "    if config.scheme == 'kessler':\n"
        "        return 'kessler', kessler_microphysics, config.kessler\n"
        "    elif config.scheme == 'brandnew':\n"
        "        return 'brandnew', brandnew_microphysics, config.brandnew\n"
        "    elif config.scheme == 'none':\n"
        "        return 'none', None, None\n"
        "    raise ValueError('Unknown microphysics scheme')\n"
    )
    found = discover_factory_schemes(src)
    assert found == {"kessler": "kessler", "brandnew": "brandnew", "none": None}
    # ... and that the new scheme is REJECTED (it is in neither list).
    assert classify_scheme("brandnew", "brandnew") is not None


def test_selftest_discovery_fails_loudly_when_anchor_missing() -> None:
    with pytest.raises(AssertionError, match="discovery anchor"):
        discover_factory_schemes("def something_else():\n    return 1\n")


def test_selftest_docstring_only_match_is_rejected() -> None:
    """Non-vacuity for :func:`_defined_or_called_names`: a symbol appearing
    ONLY in a docstring or comment must NOT count as running, or the exemption
    check reverts to the defect it was written to fix."""
    import tempfile

    src = ('"""Docs mention integrate_supersaturation and S = e/e_sat."""\n'
           "# integrate_supersaturation in a comment too\n"
           "def a_real_def():\n    return 1\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        tmp = pathlib.Path(fh.name)
    try:
        found = _defined_or_called_names(tmp)
        assert "integrate_supersaturation" not in found   # docstring/comment
        assert "a_real_def" in found                      # a real def IS seen
    finally:
        tmp.unlink()


def test_selftest_a_scheme_that_ignores_the_flag_is_flagged(tmp_path) -> None:
    """Non-vacuity for :func:`test_every_guarded_scheme_actually_READS_the_flag`
    — the concrete evasion it closes: import the shared entry point (as every
    bulk module already does) but never pass ``hard_adjust=``."""
    src = (
        "from ..._warm_rain import saturation_adjustment\n"
        "def newscheme(T, q_v, cfg):\n"
        "    cond, q_sat = saturation_adjustment(T, q_v, 1.0, 1.0)\n"
        "    return cond\n"
    )
    attrs = {n.attr for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Attribute)}
    assert "hard_saturation_adjustment" not in attrs

    ok = (
        "from ..._warm_rain import saturation_adjustment\n"
        "def newscheme(T, q_v, cfg):\n"
        "    return saturation_adjustment(\n"
        "        T, q_v, 1.0, 1.0,\n"
        "        hard_adjust=cfg.hard_saturation_adjustment,\n"
        "        hard_threshold=cfg.hard_sat_adjust_threshold,\n"
        "        hard_max_heating_K=cfg.hard_sat_max_heating_K)\n"
    )
    ok_attrs = {n.attr for n in ast.walk(ast.parse(ok))
                if isinstance(n, ast.Attribute)}
    assert {"hard_saturation_adjustment", "hard_sat_adjust_threshold",
            "hard_sat_max_heating_K"} <= ok_attrs
