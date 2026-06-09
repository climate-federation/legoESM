"""Generalized ratchet: canonical formulas must not be re-derived inline.

CLAUDE.md "Shared utilities — never re-derive" applies to *every* canonical
formula, not just the saturation curve (which has its own specialist gate,
``test_no_saturation_reimpl.py``). A re-derived formula is an invisible-failure
vector: the inline copy silently drifts from the canonical one (different
clipping, a dropped term, a sign), and norms stay green while the physics rots.

This is a registry-based, extensible version of that idea: ``CANONICAL_FORMULAS``
maps a formula to (a) a high-signal detector, (b) its canonical home (excluded
from the scan), and (c) a shrink-only per-file budget of currently-tolerated
re-derivations. Add a formula by adding a registry entry + a non-vacuity
self-test. Each gate is a TRIPWIRE, not a proof.

Seeded formulas:
  * **buoyancy term ``g/θ``** (Brunt-Väisälä N²=``g/θ·∂θ/∂z`` and anelastic
    buoyancy ``g·θ'/θ₀``) — re-derived inline at 33 sites across 18
    dycore/turbulence/GWD modules with NO shared helper. This entry is a
    **duplication-debt ratchet** (``canonical=None``): the frozen budget must not
    grow; ratchet it to zero by factoring a shared ``brunt_vaisala``/buoyancy
    helper (e.g. ``atmosphere/physics/_shared``) and reusing it. The Exner /
    hydrostatic integrand ``g/(c_p·θ)`` is deliberately excluded (different term).
  * **Monin-Obukhov stability functions** (``psi_m``/``psi_h``/``phi_m``/
    ``stability_function``…) — canonical in ``core/bulk_flux.py``; defined nowhere
    else. Empty budget → a *regression guard* against a new stability-function
    *definition* outside the canonical module. (Known gap: inline Businger-Dyer
    algebra with bare ``16``/``5`` coefficients is too common to fingerprint
    cleanly — the def-name guard catches the obvious copied helper.)

Scope boundaries (documented, not regressions): this detector targets the
``g/θ`` form only — it does NOT catch the E3SM temperature–pressure N² form
(``gravit``/``cpair`` in ``gravity_wave_drag/e3sm_cam.py``); that is a separate
fingerprint to add when/if a shared helper absorbs all N² formulations. Also NOT
gated: the Exner/potential-temperature ``T(p₀/p)^κ`` (≈45 sites — too
fundamental/pervasive to fingerprint without noise) and the virtual-temperature
``0.608`` coefficient (only incidental literals today). The constants/saturation
gates remain the backstop.
"""

from __future__ import annotations

import ast
import re

import pytest

from tests import _ratchet_audit as ra


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def _subtree_has_g(node: ast.AST) -> bool:
    """Gravity in the numerator: ``constants.g`` OR a local alias ``g`` (several
    modules do ``g = constants.g`` then ``g / theta``)."""
    for d in ast.walk(node):
        if (
            isinstance(d, ast.Attribute)
            and d.attr == "g"
            and isinstance(d.value, ast.Name)
            and d.value.id == "constants"
        ):
            return True
        if isinstance(d, ast.Name) and d.id == "g":
            return True
    return False


# Potential-temperature-ish symbol (θ, θv, reference/half variants).
_THETA_RE = re.compile(r"theta|thetav|thv|tht|th_ref|th_v", re.I)
# Specific-heat symbol — distinguishes the Exner/hydrostatic integrand
# ``g/(c_p·θ)`` (NOT buoyancy) from the buoyancy/N² term ``g/θ``.
_CP_RE = re.compile(r"^(c_p|c_pd|cp|cpd|cpair)$", re.I)


def _subtree_has_theta_name(node: ast.AST) -> bool:
    for d in ast.walk(node):
        if isinstance(d, ast.Name) and _THETA_RE.search(d.id):
            return True
        # θ often appears as an attribute (``state.theta_v``, ``cfg.theta_ref0``).
        if isinstance(d, ast.Attribute) and _THETA_RE.search(d.attr):
            return True
    return False


def _subtree_has_cp(node: ast.AST) -> bool:
    for d in ast.walk(node):
        if isinstance(d, ast.Name) and _CP_RE.match(d.id):
            return True
        if isinstance(d, ast.Attribute) and _CP_RE.match(d.attr):
            return True
    return False


def detect_buoyancy_term(src: str) -> list[int]:
    """Line numbers of an inline buoyancy/Brunt-Väisälä term — a division with
    gravity (``constants.g`` or a ``g`` alias) in the numerator and a potential
    temperature in the denominator (``g/θ·∂θ/∂z`` for N²; ``g·θ'/θ₀`` for
    anelastic buoyancy). Excludes the Exner/hydrostatic integrand ``g/(c_p·θ)``
    (c_p in the denominator), which is a different thermodynamic term."""
    tree = ast.parse(src)
    out: list[int] = []
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Div)
            and _subtree_has_g(n.left)
            and _subtree_has_theta_name(n.right)
            and not _subtree_has_cp(n.right)
        ):
            out.append(n.lineno)
    return out


_PSI_RE = re.compile(r"^(psi_[mhcq]|psimhu|psixhu|phi_[mh]|stability_function)$")


def detect_monin_obukhov_stability_defs(src: str) -> list[int]:
    """Line numbers of a Monin-Obukhov stability-function *definition*
    (``def psi_m``/``psi_h``/…) — these belong only in ``core/bulk_flux.py``."""
    tree = ast.parse(src)
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _PSI_RE.match(n.name)
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CANONICAL_FORMULAS = {
    "buoyancy_term_g_over_theta": {
        "detect": detect_buoyancy_term,
        "canonical": None,  # no shared helper yet — every site below is DEBT
        "fix": (
            "factor ``g/θ·∂θ/∂z`` into a shared buoyancy/Brunt-Väisälä helper "
            "(e.g. atmosphere/physics/_shared) and reuse it; ratchet this budget down"
        ),
        # Per-file count, seeded from the measured baseline (iter 2026-06-09):
        # 33 inline buoyancy/N² re-derivations across 18 modules.
        "budget": {
            "packages/atmosphere/legoesm/atmosphere/dynamics/compressible_euler.py": 5,
            "packages/atmosphere/legoesm/atmosphere/dynamics/compressible_euler_mpas.py": 1,
            "packages/atmosphere/legoesm/atmosphere/dynamics/compressible_euler_plane.py": 2,
            "packages/atmosphere/legoesm/atmosphere/dynamics/spectral_les_plane.py": 1,
            "packages/atmosphere/legoesm/atmosphere/dynamics/spectral_nh.py": 2,
            "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/hines.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb_lite.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/edmf.py": 3,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/holtslag_boville.py": 4,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/louis.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/mynn25.py": 3,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/pbl_height.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/smagorinsky.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/tke.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/ysu.py": 3,
        },
    },
    "monin_obukhov_stability_fn": {
        "detect": detect_monin_obukhov_stability_defs,
        "canonical": "packages/core/legoesm/core/bulk_flux.py",
        "fix": (
            "import ``psi_m``/``psi_h`` from legoesm.core.bulk_flux; do not "
            "redefine Monin-Obukhov stability functions"
        ),
        "budget": {},  # none allowed outside the canonical module (regression guard)
    },
}


def _production_files():
    return [
        f for f in ra.discover_py_files()
        if not ra.rel(f).startswith(("tests/", "scripts/"))
    ]


_FILES = _production_files()


def test_discovery_sane() -> None:
    ra.assert_discovery_sane(ra.discover_py_files())


@pytest.mark.parametrize("formula", sorted(CANONICAL_FORMULAS))
def test_no_canonical_formula_reimpl(formula: str) -> None:
    spec = CANONICAL_FORMULAS[formula]
    detect = spec["detect"]
    canonical = spec["canonical"]
    budget = spec["budget"]
    errors: list[str] = []
    for f in _FILES:
        rel = ra.rel(f)
        if canonical is not None and rel == canonical:
            continue
        hits = detect(f.read_text())
        allowed = budget.get(rel, 0)
        if len(hits) > allowed:
            errors.append(f"{rel}: {len(hits)} re-derivation(s) (budget {allowed}) at lines {hits}")
    assert not errors, (
        f"canonical formula '{formula}' re-derived beyond budget. {spec['fix']}.\n  "
        + "\n  ".join(errors)
    )


@pytest.mark.parametrize("formula", sorted(CANONICAL_FORMULAS))
def test_formula_budget_is_not_stale(formula: str) -> None:
    """Every budgeted file must still exist and still trip the detector — so the
    allow-list cannot rot and a cleaned-up site must drop its budget (ratchet)."""
    spec = CANONICAL_FORMULAS[formula]
    detect = spec["detect"]
    problems: list[str] = []
    for rel, allowed in spec["budget"].items():
        p = ra.repo_root() / rel
        if not p.is_file():
            problems.append(f"{rel}: budgeted file missing")
            continue
        actual = len(detect(p.read_text()))
        if actual < allowed:
            problems.append(f"{rel}: budget {allowed} but only {actual} now — lower it")
    assert not problems, f"'{formula}' budget needs maintenance:\n  " + "\n  ".join(problems)


# ---------------------------------------------------------------------------
# Non-vacuity self-tests
# ---------------------------------------------------------------------------
def test_buoyancy_detector_flags_inline_n2() -> None:
    src = "N2 = (constants.g / jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz\n"
    assert len(detect_buoyancy_term(src)) == 1


def test_buoyancy_detector_ignores_shared_helper_call() -> None:
    src = "from legoesm.atmosphere.physics import _shared\nN2 = _shared.brunt_vaisala(theta, dtheta_dz)\n"
    assert detect_buoyancy_term(src) == []


def test_buoyancy_detector_ignores_unrelated_division() -> None:
    src = "x = constants.g / rho\ny = a / theta\n"  # g/rho and a/theta separately
    assert detect_buoyancy_term(src) == []


def test_monin_obukhov_detector_flags_psi_def() -> None:
    src = "def psi_m(zeta):\n    return zeta\ndef psi_h(zeta):\n    return zeta\n"
    assert len(detect_monin_obukhov_stability_defs(src)) == 2


def test_monin_obukhov_detector_ignores_import_and_other_defs() -> None:
    src = "from legoesm.core.bulk_flux import psi_m\ndef compute_flux(psi_m):\n    return psi_m\n"
    assert detect_monin_obukhov_stability_defs(src) == []
