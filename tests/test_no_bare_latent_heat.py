"""Ratchet: no NEW bare ``constants.L_v`` / ``L_s`` / ``L_f`` in production code.

The latent heats are temperature dependent (Kirchhoff) and live as ONE family in
``legoesm.thermo`` (``latent_heat_vaporization`` / ``latent_heat_sublimation`` /
``latent_heat_fusion`` / ``surface_latent_heat``).  A bare constant at a surface
flux, coupler tile or budget ledger re-derives water from heat with the wrong
value (the 2026-09 coupler defect: ~2 % of evaporation lost at warm SST).

Scheme-INTERNAL thermodynamics of ported parameterizations (CLUBB, Bechtold,
CAM6 convection, Morrison/Thompson/P3, ...) keep their oracle's constant by
decision (user 2026-09-28, choice 5a): each such site carries
``# latent-ok: <scheme> oracle constant``.  Every other existing site is in the
shrink-only ``LATENT_BUDGET`` below and is to be migrated; the budget may only
go DOWN.  ``constants.py`` (definition) and ``thermo.py`` (canonical formulas)
are exempt.  Self-tests prove the detector is non-vacuous.
"""
from __future__ import annotations

import ast

import pytest

from tests import _ratchet_audit as ra
from tests._latent_heat_baseline import LATENT_BUDGET

_NAMES = frozenset({"L_v", "L_s", "L_f"})
_EXEMPT_TAG = "latent"
_EXEMPT_FILES = {
    ra.canonical_source_path("packages/core/legoesm/constants.py"),
    ra.canonical_source_path("packages/core/legoesm/thermo.py"),
    ra.canonical_source_path("tests/test_no_bare_latent_heat.py"),
}


def bare_latent_heat_lines(src: str) -> list[int]:
    """Sorted, deduplicated line numbers reading ``constants.L_v``/``L_s``/``L_f``
    (attribute access on a name ending in ``constants``), minus lines tagged
    ``# latent-ok: <reason>``.  Raises ``SyntaxError`` on an unparseable file."""
    tree = ast.parse(src)
    exempt = ra.comment_tagged_lines(src, _EXEMPT_TAG)
    # Every local name bound to the constants module: ``constants``, a dotted
    # ``legoesm.constants`` tail, and any ``import ... constants as <alias>``.
    aliases = {"constants"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name.split(".")[-1] == "constants" and a.asname:
                    aliases.add(a.asname)
    lines: set[int] = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Attribute) and n.attr in _NAMES
                and isinstance(n.value, ast.Name) and n.value.id in aliases):
            lines.add(n.lineno)
    return sorted(ln for ln in lines if ln not in exempt)


_FILES = [p for p in ra.discover_py_files() if not ra.rel(p).startswith("tests/")]


def test_discovery_sane() -> None:
    ra.assert_discovery_sane(ra.discover_py_files())


def test_no_stale_budget_entries() -> None:
    missing = [rel for rel in LATENT_BUDGET if not (ra.repo_root() / rel).is_file()]
    assert not missing, f"LATENT_BUDGET names non-existent files: {missing}"


@pytest.mark.parametrize("path", _FILES, ids=ra.rel)
def test_no_new_bare_latent_heat(path) -> None:
    if path.resolve() in _EXEMPT_FILES:
        pytest.skip("definition / canonical / guard file")
    rel = ra.rel(path)
    hits = bare_latent_heat_lines(path.read_text())
    budget = LATENT_BUDGET.get(rel, 0)
    assert len(hits) <= budget, (
        f"{rel} reads a bare latent-heat constant on {len(hits)} line(s) "
        f"(budget {budget}). Use legoesm.thermo.latent_heat_vaporization / "
        f"latent_heat_sublimation / latent_heat_fusion / surface_latent_heat at "
        f"the temperature of the phase change, or tag a scheme-internal oracle "
        f"constant with ``# latent-ok: <scheme> oracle constant``. Lines: {hits}"
    )


@pytest.mark.parametrize("path", _FILES, ids=ra.rel)
def test_budget_is_tight(path) -> None:
    """A migrated file must also shrink its budget entry (shrink-only ratchet)."""
    rel = ra.rel(path)
    if rel not in LATENT_BUDGET or path.resolve() in _EXEMPT_FILES:
        pytest.skip("no budget entry")
    hits = bare_latent_heat_lines(path.read_text())
    assert len(hits) == LATENT_BUDGET[rel], (
        f"{rel}: {len(hits)} bare site(s) but LATENT_BUDGET says {LATENT_BUDGET[rel]}; "
        f"lower the entry (or delete it at 0)")


def test_detector_flags_each_constant() -> None:
    src = "a = constants.L_v\nb = 2 * constants.L_s\nc = x / constants.L_f\n"
    assert bare_latent_heat_lines(src) == [1, 2, 3]


def test_detector_flags_aliased_module() -> None:
    assert bare_latent_heat_lines("from legoesm import constants as _c\nq = h / _c.L_v\n") == [2]


def test_detector_ignores_the_canonical_family_and_other_attrs() -> None:
    src = ("from legoesm.thermo import latent_heat_vaporization\n"
           "L = latent_heat_vaporization(T)\ng = constants.g\nx = cfg.L_v\n")
    assert bare_latent_heat_lines(src) == []


def test_inline_exemption_requires_real_comment() -> None:
    assert bare_latent_heat_lines("q = h / constants.L_v  # latent-ok: CLUBB oracle constant\n") == []
    assert bare_latent_heat_lines("q = h / constants.L_v; s = \"latent-ok: in a string\"\n") == [1]
