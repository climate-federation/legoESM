"""Repo-wide ratchet: no inline empirical coefficients in physics function bodies.

The user's request that motivates this gate: a physics body like
``k_dry = (0.135 * rho_b + 64.7) / (2700.0 - 0.947 * rho_b)`` buries four tunable
empirical coefficients inside an expression. They cannot be discovered, documented,
swapped per-parameterization, or handed to a training loop. The target end state is
that such coefficients are declared *at the top of the file* — as fields of a
scheme ``*Config`` NamedTuple (tunable) or as a named module-level provenance block
(fixed published constants/tables) — and the body reads ``cfg.<name>`` only.

This gate is the *Syntax-Engine* tripwire for that rule (see
``docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md``): it flags every
float literal that lives **inside a function scope** in the physics rosters, with a
small, precise allowlist so genuine math (``0.5 * (a + b)``, ``x ** 2``) and
numerics floors are not false-positives. It composes with — and does not overlap —
``test_no_hardcoded_constants`` (bans a *value set*) and ``test_no_saturation_reimpl``
(bans re-derived *expressions*); this one bans a *context*.

**Tripwire, not a proof.** Coefficients written as a bare ``int`` are also caught
when ``|v| > _INT_INDEX_MAX`` (so ``1350``/``2700`` do not escape via the integer
spelling); small ints stay exempt as indices/counts. Remaining documented gaps: a
published exponent ``x ** 1.5`` exempts the exponent (its prefactor is still
caught); a small (<=16) integer coefficient; a value aliased through a
module-level name is compliant *by design* (module level is the target home).
Review + the physics contract/conservation gates cover the rest.

Detection (reuses ``tests/_ratchet_audit.py`` verbatim):
  * AST: only ``float`` ``Constant`` nodes lexically inside a ``FunctionDef`` /
    ``AsyncFunctionDef`` body (any depth — closures, comprehensions, methods),
    their signature defaults, and ``Lambda`` bodies. Module-level and class-level
    (NamedTuple/dataclass field) defaults are never visited — that IS the target.
  * Exempt: a tiny math allowlist, exact unit conversions, safety floors
    (``|v| <= 1e-6``), overflow guards (``|v| >= 1e20``), subscript indices, math
    ``Pow`` exponents, and a tokenised ``# coeff-ok: <reason>`` comment (the reason
    is REQUIRED — a bare ``# coeff-ok`` does not exempt).
  * The ratchet keys on the **normalised source line** (``fingerprint_subset_errors``
    in exact mode): a same-value relocation gets a new fingerprint with no
    allowance and goes red; a removed site must drop its allowance.

Self-tests exercise the acceptance cases (Johansen line -> 4, KK2000 -> 3, a
signature default -> 1, math/indices/floors/module-level -> 0), the reason-required
escape, and the relocation ratchet; ``test_discovery_sane`` proves non-vacuity.
"""

from __future__ import annotations

import ast
import collections
import io
import re
import tokenize

import pytest

from tests import _ratchet_audit as ra
from tests._inline_coeff_baseline import COEFF_BUDGET, _SEED_FILES

# --- roster: which files carry the rule -----------------------------------
# Physics scheme trees across all model components. Keyed prefix over
# ``ra.discover_py_files()`` (so ``__pycache__``/dedup/scope logic is inherited).
ROSTERS: dict[str, str] = {
    "atmosphere": "packages/atmosphere/legoesm/atmosphere/physics/",
    "ocean": "packages/ocean/legoesm/ocean/physics/",
    "land": "packages/land/legoesm/land/",
    "ice": "packages/ice/legoesm/ice/",
    "coupler": "packages/coupler/legoesm/coupler/",
}

# One known production file per roster + a floor on the file count, so a
# mis-resolved namespace that silently drops a whole component goes red instead
# of vacuously passing (mirrors ``_ratchet_audit`` sentinels, roster-local).
_ROSTER_SENTINELS: dict[str, str] = {
    "atmosphere": "packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py",
    "ocean": "packages/ocean/legoesm/ocean/physics/shortwave_penetration.py",
    "land": "packages/land/legoesm/land/soil_thermal.py",
    "ice": "packages/ice/legoesm/ice/dynamics.py",
    "coupler": "packages/coupler/legoesm/coupler/coupler.py",
}
_ROSTER_MIN_FILES: dict[str, int] = {
    "atmosphere": 100,
    "ocean": 40,
    "land": 15,
    "ice": 12,
    "coupler": 8,
}

# --- detector rules --------------------------------------------------------
# Pure math constants that routinely appear in correct formulas (means, halves,
# squares-as-coeffs); never an empirical tuning knob on their own.
_MATH_ALLOW = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 0.5, 0.25})
# Exact unit/calendar conversions — mathematically fixed, cannot "drift by
# retuning" (s/day, deg/circle, %, ...).
_CONVERSION_ALLOW = frozenset(
    {10.0, 100.0, 1000.0, 1.0e6, 60.0, 3600.0, 86400.0, 180.0, 360.0}
)
# Safety floors / AD-guards (``maximum(x, 1e-12)``) and overflow ceilings — these
# are numerics, not physics (CLAUDE.md exempt category).
_FLOOR_MAX = 1e-6
_GUARD_MIN = 1e20
# Integer constants with |v| <= this are indices/axes/small counts/dims, not
# empirical coefficients; above it an int literal in a body is treated like a
# float coefficient (closes the "write 1350 not 1350.0" escape hatch).
_INT_INDEX_MAX = 16
# Math exponents that legitimately appear as literals in ``x ** e``.
_MATH_EXPONENTS = (2.0, 3.0, 4.0, 0.5, 0.25, 1.5, 1.0 / 3.0, 2.0 / 3.0)
# ``# coeff-ok: <reason>`` — the colon and a non-space reason are REQUIRED.
_COEFF_OK_RE = re.compile(r"coeff-ok:\s*\S")


def _coeff_ok_lines(src: str) -> set[int]:
    """Lines carrying a real ``# coeff-ok: <reason>`` comment (tokenised, so the
    text inside a string literal does not exempt; a bare ``# coeff-ok`` without a
    reason does not exempt either)."""
    out: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT and _COEFF_OK_RE.search(tok.string):
                out.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return out
    return out


def _is_exempt_exponent(fv: float | None) -> bool:
    return fv is not None and any(abs(abs(fv) - e) < 1e-9 for e in _MATH_EXPONENTS)


def coeff_hits(src: str) -> list[tuple[int, float, str]]:
    """(lineno, value, normalized_line) for every inline empirical-float site.

    Raises ``SyntaxError`` to the caller on an unparseable file (surfaced as a
    test failure, never silently "clean")."""
    tree = ast.parse(src)
    lines = src.splitlines()
    ok_lines = _coeff_ok_lines(src)

    # Pre-pass: node ids that are exempt by structural context (subscript
    # indices; math-exponent operands of ``**``).
    exempt_ids: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript):
            for d in ast.walk(n.slice):
                exempt_ids.add(id(d))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            if _is_exempt_exponent(ra.fold_numeric(n.right)):
                for d in ast.walk(n.right):
                    exempt_ids.add(id(d))

    # Function scopes: every FunctionDef/AsyncFunctionDef body (descended into,
    # so nested defs/lambdas/comprehensions are covered), its signature defaults,
    # and any module-level lambda body.
    roots: list[ast.AST] = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            roots.extend(fn.body)
            roots.extend(d for d in fn.args.defaults if d is not None)
            roots.extend(d for d in fn.args.kw_defaults if d is not None)
        elif isinstance(fn, ast.Lambda):
            roots.append(fn.body)

    hits: list[tuple[int, float, str]] = []
    seen: set[int] = set()
    for root in roots:
        for node in ast.walk(root):
            if id(node) in seen or id(node) in exempt_ids:
                continue
            seen.add(id(node))
            if not isinstance(node, ast.Constant):
                continue
            kind = type(node.value)  # exact type: bool is not int here
            if kind is float:
                v = node.value
                if v in _MATH_ALLOW or v in _CONVERSION_ALLOW:
                    continue
                if abs(v) <= _FLOOR_MAX or abs(v) >= _GUARD_MIN:
                    continue
            elif kind is int:
                # Close the "write the coefficient as an int" escape hatch
                # (1350 instead of 1350.0). Small ints are indices/counts/axes/
                # small dims, not empirical coefficients -> exempt below the
                # threshold; exact conversions and overflow guards stay exempt.
                iv = node.value
                if abs(iv) <= _INT_INDEX_MAX:
                    continue
                v = float(iv)
                if v in _CONVERSION_ALLOW or abs(v) >= _GUARD_MIN:
                    continue
            else:
                continue
            if node.lineno in ok_lines:
                continue
            hits.append((node.lineno, v, ra.normalized_line(lines, node.lineno)))
    return hits


# --- file selection --------------------------------------------------------
_ALL_FILES = ra.discover_py_files()


def _roster_of(rel: str) -> str | None:
    for name, prefix in ROSTERS.items():
        if rel.startswith(prefix):
            return name
    return None


_ROSTER_FILES = sorted(
    (ra.rel(f) for f in _ALL_FILES if _roster_of(ra.rel(f)) is not None)
)


# --- tests -----------------------------------------------------------------
def test_discovery_sane() -> None:
    ra.assert_discovery_sane(_ALL_FILES)
    rels = set(_ROSTER_FILES)
    for name, sentinel in _ROSTER_SENTINELS.items():
        assert sentinel in rels, (
            f"roster {name!r} sentinel {sentinel} not discovered — the "
            f"{name} physics tree resolved wrong; scan would be vacuous."
        )
    counts = collections.Counter(_roster_of(r) for r in _ROSTER_FILES)
    for name, minimum in _ROSTER_MIN_FILES.items():
        assert counts[name] >= minimum, (
            f"roster {name!r} has {counts[name]} files (expected >= {minimum})."
        )


def test_budget_files_exist_and_are_in_roster() -> None:
    rels = set(_ROSTER_FILES)
    missing = [r for r in COEFF_BUDGET if not (ra.repo_root() / r).is_file()]
    assert not missing, f"COEFF_BUDGET names non-existent files: {missing}"
    off_roster = sorted(set(COEFF_BUDGET) - rels)
    assert not off_roster, f"COEFF_BUDGET names files outside any roster: {off_roster}"


def test_budget_only_shrinks() -> None:
    """``COEFF_BUDGET`` may only ever be a subset of the seed file set — a new
    file cannot be granted an allowance (it must be clean)."""
    grown = sorted(set(COEFF_BUDGET) - _SEED_FILES)
    assert not grown, (
        "COEFF_BUDGET grew new files (must be clean, not budgeted): "
        + ", ".join(grown)
    )


@pytest.mark.parametrize("rel", _ROSTER_FILES)
def test_no_inline_physics_coefficients(rel: str) -> None:
    observed: collections.Counter[str] = collections.Counter()
    for _ln, _val, fp in coeff_hits((ra.repo_root() / rel).read_text()):
        observed[fp] += 1
    allowed = COEFF_BUDGET.get(rel, {})
    errors = ra.fingerprint_subset_errors(dict(observed), allowed, permanent=False)
    assert not errors, (
        f"{rel}: inline physics-coefficient ratchet failed. Move the coefficient "
        f"to a scheme ``*Config`` field (tunable) or a named module-level "
        f"provenance block (fixed published constant/table), or annotate a genuine "
        f"non-physics one-off with ``# coeff-ok: <reason>``:\n  "
        + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# Non-vacuity / behaviour self-tests
# ---------------------------------------------------------------------------
def _vals(src: str) -> list[float]:
    return sorted(v for _ln, v, _fp in coeff_hits(src))


def test_detector_flags_johansen_dry_conductivity() -> None:
    src = "def f(rho_b):\n    return (0.135 * rho_b + 64.7) / (2700.0 - 0.947 * rho_b)\n"
    assert _vals(src) == [0.135, 0.947, 64.7, 2700.0]


def test_detector_flags_kk2000_coefficients() -> None:
    src = "def f(q, n):\n    return 1350.0 * q ** 2.47 * n ** -1.79\n"
    assert _vals(src) == [1.79, 2.47, 1350.0]


def test_detector_flags_signature_default() -> None:
    assert _vals("def f(u, P_star=2.75e4):\n    return P_star * u\n") == [2.75e4]


def test_detector_ignores_math_and_indices() -> None:
    assert coeff_hits("def f(a, b, x):\n    return 0.5 * (a + b) + x[0] + x ** 2\n") == []
    assert coeff_hits("def f(x):\n    return x ** (1.0 / 3.0)\n") == []
    assert coeff_hits("def f(a, b):\n    return 2.0 * a - 0.25 * b\n") == []


def test_detector_flags_large_int_but_not_small() -> None:
    # large int coefficient (escape-hatch spelling) is caught
    assert _vals("def f(q):\n    return 1350 * q\n") == [1350.0]
    # small ints (indices/counts/axes/dims) are exempt
    assert coeff_hits("def f(x):\n    return x.reshape(6, 12).sum(axis=1)[0]\n") == []
    # int exponent and small int loop count stay exempt
    assert coeff_hits("def f(x):\n    return sum(x ** 2 for _ in range(3))\n") == []


def test_detector_ignores_module_and_class_level() -> None:
    # Module-level constant and a NamedTuple class field default are the TARGET
    # home, never flagged.
    src = (
        "_AM_S = 0.069\n"
        "class C(NamedTuple):\n"
        "    tau_c: float = 7200.0\n"
    )
    assert coeff_hits(src) == []


def test_detector_ignores_docstrings_floors_and_guards() -> None:
    src = (
        "def f(x):\n"
        '    "0.135 is documented here"\n'
        "    y = jnp.maximum(x, 1e-12)\n"
        "    z = jnp.minimum(y, 1e30)\n"
        "    return y + z + 86400.0\n"
    )
    assert coeff_hits(src) == []


def test_coeff_ok_requires_reason() -> None:
    assert coeff_hits("def f():\n    return 0.135  # coeff-ok: FD probe width\n") == []
    # bare tag (no reason) does NOT exempt; same text in a string does NOT exempt.
    assert _vals("def f():\n    return 0.135  # coeff-ok\n") == [0.135]
    assert _vals("def f():\n    return 0.135  # coeff-ok no colon here\n") == [0.135]


def test_fingerprint_ratchet_catches_relocation() -> None:
    errs = ra.fingerprint_subset_errors(
        {"k = 0.135 * rho": 1}, {"k_dry = 0.135 * rho_b": 1}, permanent=False
    )
    assert errs


def test_budget_is_exact_everywhere() -> None:
    """A leftover allowance after a cleanup goes red (ratchet-down)."""
    errs = ra.fingerprint_subset_errors({}, {"k_dry = 0.135 * rho_b": 1}, permanent=False)
    assert errs
