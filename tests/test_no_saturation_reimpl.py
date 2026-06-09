"""Ratchet: no from-scratch saturation-vapor-pressure curve outside ``thermo.py``.

CLAUDE.md records a concrete invisible failure: a re-derived
``e_sat = 611.2 * exp(17.67 * T_c / (T_c + 243.5))`` in a plotter diverged from
the model's own curve and produced *false supersaturation* in CI. The fix is to
always call :func:`legoesm.thermo.saturation_vapor_pressure` /
``saturation_mixing_ratio`` rather than re-typing the formula.

Two complementary fingerprints, **deduplicated by source line** (one formula =
one site):
  * **Structural** — a call to ``exp``/``jnp.exp``/``np.exp`` whose argument
    subtree carries the Tetens *slope* coefficients ``17.67`` and ``243.5`` (the
    coefficients are *folded*, so ``240 + 3.5`` is recognised too). This is the
    primary, unit-agnostic signal: it catches an hPa-form re-impl or a prefactor
    split onto another line, and does NOT fire on the analytic derivative in
    ``convection/kuo.py`` (slope coefficients but no ``exp`` wrapping them, and it
    calls the canonical ``saturation_vapor_pressure`` — Codex review #6 sanctioned
    it).
  * **Prefactor** — a literal equal to the Pa-form e_sat prefactor family
    (``611.2`` and Bolton/Buck variants), as a backstop.

**Tripwire, not proof, with two documented gaps:** (1) a re-impl whose slope
coefficients are *variable aliases* (``A=17.67; ... exp(A*...)``) needs dataflow
and slips; (2) two distinct re-impls crammed onto one physical line count as one
site. Both are contrived and backstopped by the canonical-source discipline, the
banned-constant gate, and the PreToolUse edit hook.

Shrink-only ratchet over ``tests/_ratchet_audit.py`` discovery. Current hits are
test cross-checks (PERMANENT; ratchet-down candidates — prefer asserting against
``thermo.saturation_vapor_pressure``). ``thermo.py`` (resolved canonical path)
and this guard file are exempt. Self-tests prove non-vacuity.
"""

from __future__ import annotations

import ast

import pytest

from tests import _ratchet_audit as ra

# Pa-form saturation-vapor-pressure prefactors (canonical 611.2 + Bolton/Buck).
_PREFACTORS: frozenset[float] = frozenset({611.2, 611.21, 610.78, 610.94})
# Tetens slope coefficients of ``thermo.saturation_vapor_pressure``.
_SLOPE_A, _SLOPE_B = 17.67, 243.5
_EXP_NAMES = frozenset({"exp"})  # matches ``exp`` and ``jnp.exp``/``np.exp``
_EXEMPT_TAG = "satcurve"

_CANONICAL_THERMO = ra.canonical_source_path("packages/core/legoesm/thermo.py")
_GUARD_FILE = ra.canonical_source_path("tests/test_no_saturation_reimpl.py")

# Per-file budget = deduplicated re-impl site count. All current hits are
# PERMANENT test cross-checks (iter 2026-06-09 baseline).
SATURATION_BUDGET: dict[str, int] = {
    "tests/land/test_land_stability.py": 1,
    "tests/unit/test_physics_radiation.py": 1,
    "tests/unit/test_zm_dilute_parcel.py": 1,
}


def _exp_call_has_tetens_slopes(call: ast.Call) -> bool:
    fn = call.func
    name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
    if name not in _EXP_NAMES:
        return False
    folded = ra.fold_set(call)
    return _SLOPE_A in folded and _SLOPE_B in folded


def saturation_reimpl_lines(src: str) -> list[int]:
    """Sorted, line-deduplicated line numbers carrying a saturation-curve re-impl
    fingerprint (structural exp-call OR prefactor literal), minus ``# satcurve-ok``
    lines. Raises ``SyntaxError`` on an unparseable file."""
    tree = ast.parse(src)
    exempt = ra.comment_tagged_lines(src, _EXEMPT_TAG)
    lines: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and not isinstance(n.value, bool):
            if isinstance(n.value, (int, float)) and float(n.value) in _PREFACTORS:
                lines.add(n.lineno)
        elif isinstance(n, ast.Call) and _exp_call_has_tetens_slopes(n):
            lines.add(n.lineno)
    return sorted(ln for ln in lines if ln not in exempt)


_FILES = ra.discover_py_files()


def test_discovery_sane() -> None:
    ra.assert_discovery_sane(_FILES)


def test_no_stale_budget_entries() -> None:
    missing = [rel for rel in SATURATION_BUDGET if not (ra.repo_root() / rel).is_file()]
    assert not missing, f"SATURATION_BUDGET names non-existent files: {missing}"


@pytest.mark.parametrize("path", _FILES, ids=ra.rel)
def test_no_saturation_curve_reimpl(path) -> None:
    resolved = path.resolve()
    if resolved == _CANONICAL_THERMO or resolved == _GUARD_FILE:
        pytest.skip("definition site / guard file")
    rel = ra.rel(path)
    hits = saturation_reimpl_lines(path.read_text())
    budget = SATURATION_BUDGET.get(rel, 0)
    assert len(hits) <= budget, (
        f"{rel} has {len(hits)} saturation-vapor-pressure re-impl site(s) "
        f"(budget {budget}) — a re-derived e_sat curve risks diverging from the "
        f"model (false supersaturation, CLAUDE.md). Call "
        f"``legoesm.thermo.saturation_vapor_pressure`` instead, or annotate a "
        f"genuine non-saturation use with ``# satcurve-ok: <reason>``. "
        f"Offending lines: {hits}"
    )


# ---------------------------------------------------------------------------
# Non-vacuity / behaviour self-tests
# ---------------------------------------------------------------------------
def test_detector_flags_reimplemented_curve() -> None:
    assert saturation_reimpl_lines("e = 611.2 * jnp.exp(17.67 * Tc / (Tc + 243.5))\n") == [1]


def test_detector_flags_hpa_form_via_structure() -> None:
    """hPa-form re-impl (no 611.2) caught structurally."""
    assert saturation_reimpl_lines("e = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5)) * 100\n") == [1]


def test_detector_flags_folded_slope_coeffs() -> None:
    """Slope coefficients written as folded arithmetic (``240 + 3.5`` == 243.5)
    are still recognised."""
    assert saturation_reimpl_lines("e = np.exp((10 + 7.67) * T / (T + (240 + 3.5)))\n") == [1]


def test_detector_dedups_prefactor_and_structure_on_one_line() -> None:
    assert saturation_reimpl_lines("e = 611.2 * jnp.exp(17.67 * Tc / (Tc + 243.5))\n") == [1]


def test_detector_ignores_canonical_call_and_derivative() -> None:
    src = (
        "from legoesm.thermo import saturation_vapor_pressure\n"
        "e_sat = saturation_vapor_pressure(T)\n"
        "desat_dT = e_sat * (17.67 * 243.5) / (T_c + 243.5) ** 2\n"
    )
    assert saturation_reimpl_lines(src) == []


def test_inline_exemption_requires_real_comment() -> None:
    assert saturation_reimpl_lines(
        "e = 611.2 * jnp.exp(17.67*Tc/(Tc+243.5))  # satcurve-ok: oracle cross-check\n"
    ) == []
