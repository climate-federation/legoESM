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
  * **buoyancy term ``g/θ``** (Brunt-Väisälä N²=``g/θ·∂θ/∂z``, anelastic buoyancy
    ``g·θ'/θ₀``) — 33 sites / 18 modules, NO shared helper → **debt ratchet**
    (``canonical=None``; ratchet to zero via a shared buoyancy helper). Excludes
    the Exner integrand ``g/(c_p·θ)``.
  * **E3SM temperature–pressure N²** (``g²/(c_p·T)``, ``gravit*gravit``) — the
    formulation the ``g/θ`` detector cannot see; 4 sites → debt ratchet (companion
    to the buoyancy helper).
  * **Exner / potential temperature** (``(p/p₀)^κ``) — 42 sites / 29 modules, NO
    canonical helper → debt ratchet (ratchet down by creating a shared
    Exner/potential-temperature helper).
  * **virtual temperature** (``0.608``/``0.61`` = R_v/R_d−1) — canonical is
    ``physics._shared.virtual_temperature``; 3 inline ``T·(1+0.608·q)`` sites →
    debt; replace with the helper.
  * **Monin-Obukhov stability functions** (``psi_m``/``psi_h``/``phi_m``/
    ``stability_function``…) — canonical in ``core/bulk_flux.py``; empty budget →
    *regression guard* against a new stability-function *definition* elsewhere.

Documented gaps (not regressions): inline Businger-Dyer algebra (bare ``16``/``5``
coefficients) is too common to fingerprint, so the M-O guard catches a copied
*helper* but not raw inline stability algebra; the Exner gate keys on the ``κ``
exponent (a θ written without ``**κ`` — e.g. via ``log``/table — is not caught).
The constants/saturation gates remain the backstop.
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


_KAPPA_RE = re.compile(r"kappa")


def _is_constants_kappa(d: ast.AST) -> bool:
    return (
        isinstance(d, ast.Attribute)
        and d.attr == "kappa"
        and isinstance(d.value, ast.Name)
        and d.value.id == "constants"
    )


def _file_aliases_poisson_kappa(tree: ast.AST) -> bool:
    """True if the file binds a ``kappa``-named symbol to ``constants.kappa``
    (the Poisson alias). Distinguishes the Exner exponent from an unrelated local
    ``kappa`` (e.g. ``kappa = R_d*lapse/g`` in standard_atmosphere)."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and n.targets:
            tgt = n.targets[0]
            if isinstance(tgt, ast.Name) and _KAPPA_RE.search(tgt.id) and _is_constants_kappa(n.value):
                return True
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and _KAPPA_RE.search(n.target.id):
            if n.value is not None and _is_constants_kappa(n.value):
                return True
    return False


def detect_exner_potential_temperature(src: str) -> list[int]:
    """Lines with a power whose EXPONENT is the *Poisson* constant κ — the Exner /
    potential-temperature / pressure-power family ``(p/p₀)^κ`` (and ``**(1/κ)`` /
    raw ``p^κ``). The exponent must reference ``constants.kappa`` directly, OR a
    local ``kappa``-named symbol when the file aliases ``kappa = constants.kappa``
    — so an unrelated local ``kappa`` (e.g. ``R_d*lapse/g``) is NOT matched.
    Deduplicated by line.

    The alias proof is file-level (a conservative tripwire): a file that aliases
    the Poisson ``kappa`` AND *also* shadows ``kappa`` with a non-Poisson local in
    some function would flag that shadowed power too. No such case exists today;
    if one appears, annotate/budget it (no current false positive)."""
    tree = ast.parse(src)
    aliased = _file_aliases_poisson_kappa(tree)
    lines: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            for d in ast.walk(n.right):
                if _is_constants_kappa(d) or (
                    aliased and isinstance(d, ast.Name) and _KAPPA_RE.search(d.id)
                ):
                    lines.add(n.lineno)
                    break
    return sorted(lines)


def detect_virtual_temperature(src: str) -> list[int]:
    """Lines with the virtual-temperature coefficient ``0.608``/``0.61``
    (= R_v/R_d − 1) — an inline ``T·(1 + 0.608·q)`` instead of the canonical
    ``physics._shared.virtual_temperature``. Deduplicated by line."""
    tree = ast.parse(src)
    lines: set[int] = set()
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Constant)
            and not isinstance(n.value, bool)
            and isinstance(n.value, (int, float))
            and float(n.value) in (0.608, 0.61)
        ):
            lines.add(n.lineno)
    return sorted(lines)


def detect_brunt_vaisala_tp_form(src: str) -> list[int]:
    """Lines with a ``g·g`` (``gravit*gravit`` / ``g*g``) product — the E3SM
    temperature-pressure N² form ``g²/(c_p·T)`` that the ``g/θ`` detector misses.
    Deduplicated by line."""
    tree = ast.parse(src)
    lines: set[int] = set()
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.BinOp)
            and isinstance(n.op, ast.Mult)
            and isinstance(n.left, ast.Name)
            and isinstance(n.right, ast.Name)
            and n.left.id == n.right.id
            and n.left.id in ("gravit", "g")
        ):
            lines.add(n.lineno)
    return sorted(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CANONICAL_FORMULAS = {
    "buoyancy_term_g_over_theta": {
        "detect": detect_buoyancy_term,
        # Canonical home: ``physics._shared.buoyancy_coefficient(θ) = g/θ`` (the
        # CLUBB port factored its ``g/θ`` sites into it; remaining files below are
        # DEBT to migrate to the helper).
        "canonical": "packages/atmosphere/legoesm/atmosphere/physics/_shared.py",
        "fix": (
            "use ``physics._shared.buoyancy_coefficient(θ)`` (= g/θ) instead of an "
            "inline ``g/θ·(flux or ∂θ/∂z)``; ratchet this budget down"
        ),
        # Per-file count, seeded from the measured baseline (iter 2026-06-09):
        # 33 inline buoyancy/N² re-derivations across 18 modules.
        "budget": {
            "packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler.py": 5,
            "packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_mpas.py": 1,
            "packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py": 2,
            # 1 -> 2: main grew a second inline N² site while this branch was
            # in flight (rebase 2026-06-12 baseline re-seed, not branch debt).
            "packages/atmosphere/legoesm/atmosphere/dynamics/les/spectral_les_plane.py": 2,
            "packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py": 2,
            # GWD hines/lindzen/mcfarlane/prognostic_spectral migrated their
            # inline g/θ·∂θ/∂z N² to physics._shared.brunt_vaisala_n_full
            # (ponytail dedup 2026-06-17) → budget ratcheted to 0 (entries removed).
            # Turbulence dedup 2026-07-07: the N²/bulk-Ri g/θ sites in
            # clubb_lite / louis / pbl_height / smagorinsky / tke and edmf's
            # N²_half migrated to physics._shared.buoyancy_coefficient →
            # budgets ratcheted (entries removed / edmf 3 → 2).  edmf keeps
            # its two anelastic parcel-buoyancy g·θ'/θ sites (updraft buoy +
            # mf_buoyancy), a genuinely different (non-N²) form.
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/edmf.py": 2,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/holtslag_boville.py": 4,
            "packages/atmosphere/legoesm/atmosphere/physics/turbulence/mynn25.py": 3,
            # ysu.py: 4 inline g/θ buoyancy/N² re-derivations migrated to
            # physics._shared.buoyancy_coefficient (audit item 7) → budget 0
            # (entry removed; was 3, which had already gone stale at 4 actual).
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
    "exner_potential_temperature": {
        "detect": detect_exner_potential_temperature,
        # Canonical home: ``physics._shared.exner_function(p) = (p/p_ref)^κ`` (the
        # CLUBB port factored its sites into it; remaining files below are DEBT to
        # migrate to the helper).
        "canonical": "packages/atmosphere/legoesm/atmosphere/physics/_shared.py",
        "fix": (
            "use ``physics._shared.exner_function(p)`` (= (p/p_ref)^κ) instead of "
            "an inline power; ratchet this debt to zero"
        ),
        # 42 inline (p/p0)^κ sites across 29 modules (iter 2026-06-09).
        "budget": {
            "packages/atmosphere/legoesm/atmosphere/dynamics/gcm/_fv3_lin_pgf.py": 2,
            "packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane.py": 1,
            "packages/atmosphere/legoesm/atmosphere/forcing/idealized/held_suarez.py": 1,
            "packages/atmosphere/legoesm/atmosphere/idealized/rcemip_initial_conditions.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/convection/dca.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/convection/kain_fritsch.py": 1,
            # GWD hines/lindzen/mcfarlane/prognostic_spectral dropped their inline
            # θ=T·(p_ref/p)^κ (folded into physics._shared.brunt_vaisala_n_full,
            # ponytail dedup 2026-06-17) → entries removed (budget 0). integration.py
            # keeps its separate exner site.
            "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/integration.py": 1,
            "packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py": 3,
            "packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py": 3,
            "packages/atmosphere/legoesm/atmosphere/physics/thermodynamics.py": 3,
            # New on main while this branch was in flight (rebase 2026-06-12
            # baseline seed, not branch debt): 3 inline (p/p0)^κ sites.
            "packages/atmosphere/legoesm/atmosphere/dynamics/les/spectral_les_moist.py": 3,
            # Turbulence dedup 2026-07-07: every turbulence-module inline
            # (p/p0)^κ / (p0/p)^κ site (edmf, holtslag_boville, integration,
            # louis, mynn25, pbl_height, smagorinsky, tke, vertical_diffusion,
            # ysu) migrated to physics._shared.exner_function (forward Π or
            # 1/Π for the inverse direction) → budgets ratcheted to 0
            # (entries removed).
            "packages/atmosphere/legoesm/atmosphere/forcing/sam_case_forcing.py": 2,
            "packages/atmosphere/legoesm/atmosphere/forcing/scm/scm_forcing.py": 1,
            "packages/core/legoesm/grids/vertical.py": 4,
        },
    },
    "virtual_temperature": {
        "detect": detect_virtual_temperature,
        "canonical": "packages/atmosphere/legoesm/atmosphere/physics/_shared.py",
        "fix": (
            "use legoesm.atmosphere.physics._shared.virtual_temperature(T, q_v) "
            "instead of an inline ``T*(1 + 0.608*q)``"
        ),
        "budget": {
            "packages/core/legoesm/grids/cubed_sphere.py": 3,
        },
    },
    "brunt_vaisala_tp_form": {
        "detect": detect_brunt_vaisala_tp_form,
        "canonical": None,  # buoyancy debt — E3SM temperature-pressure N² (g²/(c_p·T))
        "fix": (
            "factor the E3SM ``g²/(c_p·T)`` N² into the shared buoyancy helper "
            "(companion to the ``g/θ`` debt above); ratchet down"
        ),
        "budget": {
            "packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/e3sm_cam.py": 2,
            "packages/core/legoesm/grids/vertical.py": 2,
        },
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


def test_exner_detector_flags_constants_kappa_power() -> None:
    assert detect_exner_potential_temperature("theta = T * (constants.p_ref / p) ** constants.kappa\n") == [1]


def test_exner_detector_flags_aliased_kappa_power() -> None:
    # bare ``kappa`` counts only when the file aliases it from constants.kappa
    src = "kappa = constants.kappa\np = p_ref * exner ** (1.0 / kappa)\n"
    assert detect_exner_potential_temperature(src) == [2]


def test_exner_detector_ignores_unrelated_local_kappa() -> None:
    # standard_atmosphere shape: a non-Poisson local kappa used as an exponent.
    src = "kappa = R_d * lapse / g\nu_geo = sigma ** kappa\n"
    assert detect_exner_potential_temperature(src) == []


def test_exner_detector_ignores_other_powers() -> None:
    assert detect_exner_potential_temperature("a = x ** 2\nb = rho ** gamma\n") == []


def test_exner_detector_file_level_taint_is_conservative() -> None:
    # DOCUMENTED conservative behavior: the alias proof is file-level, so a module
    # that aliases the Poisson kappa AND separately shadows kappa with a
    # non-Poisson local still flags the shadowed power. No such case exists today;
    # pinned so the conservative behavior is explicit (annotate/budget if it ever
    # appears).
    src = "kappa = constants.kappa\ndef f():\n    kappa = R_d * lapse / g\n    return sigma ** kappa\n"
    assert detect_exner_potential_temperature(src) == [4]


def test_virtual_temperature_detector_flags_coefficient() -> None:
    assert detect_virtual_temperature("Tv = T * (1.0 + 0.608 * q_v)\n") == [1]


def test_virtual_temperature_detector_ignores_other_floats() -> None:
    assert detect_virtual_temperature("x = 0.6\ny = constants.epsilon * q\n") == []


def test_brunt_vaisala_tp_detector_flags_g_squared() -> None:
    assert detect_brunt_vaisala_tp_form("n2 = gravit * gravit / (cpair * ti)\n") == [1]
    assert detect_brunt_vaisala_tp_form("ni = g * g / (c_p * T)\n") == [1]


def test_brunt_vaisala_tp_detector_ignores_unrelated_products() -> None:
    assert detect_brunt_vaisala_tp_form("a = g * h\nb = u * u\n") == []
