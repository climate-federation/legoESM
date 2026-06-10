"""Source-audit guardrails for the distributed architecture (issue #359).

These tests prevent regressions of the #356 bug class — rank-dependent code
branching in the lat-lon C-grid dynamics, where a fold-descriptor
``is_active`` gate that does NOT also check fold locality (``fold_j >= 0`` /
``fold_is_local``) produces different MPI call counts or rank-local-only
values at partition cuts.  See ``docs/DISTRIBUTED_ARCHITECTURE.md``.

The audit is AST-based (not a line regex), so it is robust to multiline
boolean expressions, simple ``f = grid.fold`` aliases, and
``getattr(fold, "is_active", ...)`` forms.  It is a static scan — no MPI
runtime needed; the runtime rank-vs-serial value tests live in
``tests/distributed/test_latlon_mpi_tripole.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Repo root: tests/distributed/<this file> -> parents[2].
_ROOT = Path(__file__).resolve().parents[2]

# Lat-lon C-grid dynamics modules whose fold-dependent branches must route
# through ``fold_is_local`` (the ``fold_j >= 0`` locality check).
_AUDITED = [
    "packages/core/legoesm/grids/operators_latlon_cgrid.py",
    "packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py",
    "packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py",
    "packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py",
    "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py",
]

# Reading the raw flag is legitimate only inside these canonical helpers.
_HELPER_DEFS = {"is_tripolar", "fold_is_local"}


def _is_active_access(node: ast.AST) -> bool:
    """``X.is_active`` attribute access or ``getattr(X, "is_active", ...)``."""
    if isinstance(node, ast.Attribute) and node.attr == "is_active":
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "is_active"
    ):
        return True
    return False


def _object_of_is_active(node: ast.AST) -> ast.AST | None:
    if isinstance(node, ast.Attribute):
        return node.value
    if isinstance(node, ast.Call):
        return node.args[0]
    return None


def _root_name(expr: ast.AST) -> str:
    """Left-most identifier of an attribute chain (``grid.fold`` -> ``grid``;
    ``fold`` -> ``fold``)."""
    while isinstance(expr, ast.Attribute):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else ""


def _expr_mentions_fold(expr: ast.AST) -> bool:
    """Does the attribute chain reference a ``fold`` somewhere?"""
    cur = expr
    while isinstance(cur, ast.Attribute):
        if "fold" in cur.attr:
            return True
        cur = cur.value
    if isinstance(cur, ast.Name) and "fold" in cur.id:
        return True
    return False


def audit_source(text: str) -> list[str]:
    """Return a list of offending ``lineno: text`` for bare fold gates.

    An offender is an ``is_active`` access on a *fold descriptor* that is
    NOT inside the ``is_tripolar`` / ``fold_is_local`` definitions and whose
    enclosing statement does not also carry a locality check (``fold_j`` or a
    ``fold_is_local(...)`` call).
    """
    tree = ast.parse(text)
    src_lines = text.splitlines()

    # Set parent links for ancestor walks.
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]

    # Names bound to a ``....fold`` attribute (``f = grid.fold``) or
    # ``getattr(x, "fold", ...)`` — fold-descriptor aliases.  Handles plain
    # assign, annotated assign (``f: X = grid.fold``), and tuple/list
    # destructuring (``(f, g) = (grid.fold, y)``).
    def _is_fold_value(val: ast.AST | None) -> bool:
        if isinstance(val, ast.Attribute) and val.attr == "fold":
            return True
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Name)
            and val.func.id == "getattr"
            and len(val.args) >= 2
            and isinstance(val.args[1], ast.Constant)
            and val.args[1].value == "fold"
        ):
            return True
        return False

    fold_aliases: set[str] = set()

    def _bind(target: ast.AST, val: ast.AST | None) -> None:
        # Destructuring: pair targets with values positionally.
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            val, (ast.Tuple, ast.List)
        ) and len(target.elts) == len(val.elts):
            for t, v in zip(target.elts, val.elts):
                _bind(t, v)
            return
        if isinstance(target, ast.Name) and _is_fold_value(val):
            fold_aliases.add(target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                _bind(tgt, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            _bind(node.target, node.value)

    def _is_fold_object(obj: ast.AST | None) -> bool:
        if obj is None:
            return False
        if _expr_mentions_fold(obj):
            return True
        return _root_name(obj) in fold_aliases

    def _enclosing_function(node: ast.AST) -> str | None:
        cur = getattr(node, "_parent", None)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
            cur = getattr(cur, "_parent", None)
        return None

    def _guard_expr(node: ast.AST) -> ast.AST:
        # Climb to the LARGEST enclosing *expression* that contains the
        # is_active access (the controlling test of an ``if``/``ifexp``, or
        # the value of an assign/return), stopping at the statement boundary.
        # This deliberately excludes sibling statement bodies, so a locality
        # check that lives in the branch body (``if fold.is_active: j =
        # fold.fold_j``) does NOT count — only a check in the SAME boolean
        # expression (possibly multiline) is accepted.
        cur = node
        while True:
            parent = getattr(cur, "_parent", None)
            if parent is None or isinstance(parent, ast.stmt):
                return cur
            cur = parent

    def _calls_fold_is_local(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "fold_is_local"
            for n in ast.walk(node)
        )

    def _foldj_compare(node: ast.AST, *, positive: bool) -> bool:
        # ``positive``: owner predicate ``<fold>.fold_j >= 0`` (or ``> -1``).
        # not positive: inverse predicate ``<fold>.fold_j < 0`` (or ``<= -1``).
        # The compared object must be fold-like, so an unrelated
        # ``other.fold_j`` does not count.
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
            return False
        left, op, right = node.left, node.ops[0], node.comparators[0]
        if not (isinstance(left, ast.Attribute) and left.attr == "fold_j"):
            return False
        if not _is_fold_object(left.value) or not isinstance(right, ast.Constant):
            return False
        if positive:
            return (isinstance(op, ast.GtE) and right.value == 0) or (
                isinstance(op, ast.Gt) and right.value == -1
            )
        return (isinstance(op, ast.Lt) and right.value == 0) or (
            isinstance(op, ast.LtE) and right.value == -1
        )

    def _bool_operands(node: ast.AST, kind) -> list[ast.AST]:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, kind):
            out: list[ast.AST] = []
            for v in node.values:
                out.extend(_bool_operands(v, kind))
            return out
        return [node]

    def _negated_within(guard: ast.AST, target: ast.AST) -> bool:
        # True if ``target`` sits under a ``not`` somewhere between it and guard.
        cur = target
        while cur is not None and cur is not guard:
            parent = getattr(cur, "_parent", None)
            if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
                return True
            cur = parent
        return False

    def _has_locality_check(guard: ast.AST, is_active_node: ast.AST) -> bool:
        # Canonical helper call (``fold_is_local(...)`` or its negation).
        if _calls_fold_is_local(guard):
            return True
        negated = _negated_within(guard, is_active_node)
        if not negated:
            # Positive owner predicate: ``is_active AND fold_j >= 0``.  Both
            # must be conjunctive siblings (rejects ``or``, ``fold_j < 0``,
            # unrelated ``other.fold_j``).
            ops = _bool_operands(guard, ast.And)
            active_conjunct = any(
                (op is is_active_node) or (is_active_node in set(ast.walk(op)))
                for op in ops
                if not (isinstance(op, ast.BoolOp) and isinstance(op.op, ast.Or))
                and not (isinstance(op, ast.UnaryOp) and isinstance(op.op, ast.Not))
            )
            has_foldj = any(
                any(_foldj_compare(n, positive=True) for n in ast.walk(op))
                for op in ops
            )
            return active_conjunct and has_foldj
        # Inverse / early-return predicate: ``fold is None or not is_active
        # or fold_j < 0`` (== ``not fold_is_local``).
        ops = _bool_operands(guard, ast.Or)
        has_neg_foldj = any(
            any(_foldj_compare(n, positive=False) for n in ast.walk(op))
            for op in ops
        )
        return has_neg_foldj

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not _is_active_access(node):
            continue
        if not _is_fold_object(_object_of_is_active(node)):
            continue
        if _enclosing_function(node) in _HELPER_DEFS:
            continue
        if _has_locality_check(_guard_expr(node), node):
            continue
        ln = getattr(node, "lineno", 0)
        text_line = src_lines[ln - 1].strip() if 0 < ln <= len(src_lines) else ""
        offenders.append(f"{ln}: {text_line}")
    return offenders


@pytest.mark.parametrize("rel", _AUDITED)
def test_no_bare_is_active_fold_gate(rel):
    path = _ROOT / rel
    assert path.exists(), f"audited module missing: {rel}"
    offenders = audit_source(path.read_text())
    assert not offenders, (
        "Bare fold `.is_active` gate without a `fold_j`/`fold_is_local` "
        "locality check (rank-dependent branching risk; see "
        f"docs/DISTRIBUTED_ARCHITECTURE.md, issue #359) in {rel}:\n  "
        + "\n  ".join(offenders)
    )


# --- audit self-tests: prove flag / no-flag behaviour on fixtures ----------

_FLAGGED = [
    "if fold.is_active:\n    x = 1\n",
    'if getattr(fold, "is_active", False):\n    x = 1\n',
    "f = grid.fold\nif f.is_active:\n    x = 1\n",
    "fold = getattr(grid, 'fold', None)\nif fold.is_active:\n    x = 1\n",
    # multiline bare gate (no fold_j anywhere in the expression)
    "if (fold.is_active\n        and something_else):\n    x = 1\n",
    # fold_j in the BODY, not the controlling condition — still a bare gate
    "if fold.is_active:\n    j = fold.fold_j\n",
    # typed (annotated) alias
    "f: object = grid.fold\nif f.is_active:\n    x = 1\n",
    # tuple-destructured alias
    "(f, other) = (grid.fold, y)\nif f.is_active:\n    x = 1\n",
    # disjunction, not conjunction — locality not enforced
    "if fold.is_active or fold.fold_j >= 0:\n    x = 1\n",
    # wrong sign — not the owner predicate
    "if fold.is_active and fold.fold_j < 0:\n    x = 1\n",
    # fold_j on an unrelated object
    "if fold.is_active and other.fold_j >= 0:\n    x = 1\n",
]

_NOT_FLAGGED = [
    "if fold.is_active and fold.fold_j >= 0:\n    x = 1\n",
    "if fold_is_local(grid):\n    x = 1\n",
    # multiline locality check on the same expression
    "if (fold.is_active\n        and fold.fold_j >= 0):\n    x = 1\n",
    # canonical helper definitions
    "def is_tripolar(grid):\n    fold = grid.fold\n    return fold is not None and bool(fold.is_active)\n",
    "def fold_is_local(grid):\n    fold = grid.fold\n    return fold.is_active and fold.fold_j >= 0\n",
    # inverse / early-return owner predicate (== not fold_is_local)
    "if fold is None or not fold.is_active or fold.fold_j < 0:\n    return simple\n",
    "if not fold_is_local(grid):\n    return simple\n",
    # unrelated active mask (partial-cell coordinate)
    "v = compute(z_coord.is_active, grid)\n",
    "if state.is_active:\n    x = 1\n",
    # docstring mentioning fold.is_active is not code
    'def f():\n    """On tripolar (fold.is_active): north folds."""\n    return 0\n',
]


@pytest.mark.parametrize("snippet", _FLAGGED)
def test_audit_flags_bare_gates(snippet):
    assert audit_source(snippet), f"audit should flag:\n{snippet}"


@pytest.mark.parametrize("snippet", _NOT_FLAGGED)
def test_audit_ignores_safe_and_unrelated(snippet):
    assert not audit_source(snippet), f"audit should NOT flag:\n{snippet}"


def test_fold_is_local_helper_exists_and_is_single_source():
    core = _ROOT / "packages/core/legoesm/grids/operators_latlon_cgrid.py"
    text = core.read_text()
    assert "def fold_is_local(grid)" in text
    assert "fold.fold_j >= 0" in text


def test_architecture_doc_present():
    doc = _ROOT / "docs/DISTRIBUTED_ARCHITECTURE.md"
    assert doc.exists(), "docs/DISTRIBUTED_ARCHITECTURE.md (issue #359) missing"
    body = doc.read_text()
    for anchor in ("Pre-pad-then-operate", "fold_is_local", "is_tripolar"):
        assert anchor in body, f"architecture doc missing section: {anchor}"
