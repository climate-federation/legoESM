"""Ratchet: a function-scope import must not shadow a module-level name that
the same function reads *earlier*.

THE INCIDENT THIS GATE EXISTS FOR (2026-09-20).  The harmonized MPAS OMIP run
died on its first integration step, twice over, each time only after a
multi-hour GPU queue wait::

    UnboundLocalError: cannot access local variable 'compute_layer_thickness'
    where it is not associated with a value
    packages/ocean/legoesm/ocean/physics/mpas_physics.py:376, in physics_fn

``compute_layer_thickness`` was imported at module level.  A later commit added
a *redundant* function-scope import of the same symbol from the same module,
inside the ``sweeney_2band`` arm of an if/elif chain.  Python binds a name for
the whole function body if it is assigned anywhere in that body, and an
``import`` statement is an assignment -- so that one line made the module-level
import invisible to the entire function, and the ``rgb_chl`` arm, which runs
first and never mentions the import, died reading an unbound local.

The two arms are mutually exclusive: they can never both execute.  That is
precisely why the defect survived review.  The commit that introduced it is
correct in the arm it touches; the arm it breaks is a different one.  Neither
commit is wrong alone, no test exercised that pairing, and reading either
region in isolation shows nothing.  Only the whole-function scoping rule makes
it visible -- which is what an AST pass is for and eyes are not.

Mechanism:
  * For every production ``.py`` (``_ratchet_audit.production_roots()``) plus
    ``scripts/`` (the MIP drivers are production code here), collect the names
    bound by **module-level** imports -- including those nested in module-level
    ``if TYPE_CHECKING:`` / ``try: ... except ImportError:`` blocks, which bind
    at module scope just the same.
  * For each function, collect names bound by imports in its **own** scope,
    keeping the *first* such line.  Nested functions and classes are separate
    scopes and are visited on their own, never folded into the parent.
  * Flag any ``Name`` load of such a name at a line *before* that first binding
    line.  That is the exact condition that raises ``UnboundLocalError``.

Deliberately NOT flagged -- each of these is a legitimate, common pattern:
  * A function-scope import of a name that is *not* bound at module level.
    This is the repo's blessed deferred-import pattern for breaking the
    ``core`` -> ``runtime`` circular-import cycle; it is not shadowing.
  * A function-scope import whose every use is *after* the import line.
    Redundant, but harmless, and outside this gate's remit.
  * ``import public_name as _alias`` -- the alias is a different name.
  * A name the function declares ``global`` / ``nonlocal``: the import then
    rebinds the module global rather than creating a local, so no
    ``UnboundLocalError`` is possible.

Baseline is EMPTY and stays empty: a repo-wide scan at the time this landed
found 142 function-scope imports that shadow a module-level name, of which
exactly ZERO are read before their import line.  The gate therefore costs
nothing today and refuses the one arrangement that has actually broken a
production run.

A tripwire, not a proof -- the self-tests below show the scanner flags the real
incident's shape and stays silent on every sanctioned form.
"""

from __future__ import annotations

import ast
import pathlib

from tests import _ratchet_audit as ra

# Shrink-only. Every entry must be a real violation; a stale entry fails.
ALLOWLIST: frozenset[str] = frozenset()


def _module_level_bound_names(tree: ast.Module) -> set[str]:
    """Names bound by imports at module scope.

    Descends into module-level ``if`` / ``try`` / ``with`` blocks (those bind at
    module scope) but never into a function or class body (separate scopes).
    """
    names: set[str] = set()
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        for child in ast.iter_child_nodes(node):
            stack.append(child)
    return names


def _own_scope_nodes(fn: ast.AST) -> list[ast.AST]:
    """Nodes belonging to ``fn``'s own scope, excluding nested scopes."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = [
        b
        for b in fn.body  # type: ignore[attr-defined]
        if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    while stack:
        node = stack.pop()
        out.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
            ):
                continue
            stack.append(child)
    return out


def scan_source(src: str) -> list[tuple[str, str, int, int]]:
    """``(function, name, use_lineno, import_lineno)`` for each live shadowing.

    "Live" means the name is read at a line strictly before the function-scope
    import that binds it -- i.e. the read raises ``UnboundLocalError``.
    """
    tree = ast.parse(src)
    module_names = _module_level_bound_names(tree)
    if not module_names:
        return []

    found: set[tuple[str, str, int, int]] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        nodes = _own_scope_nodes(fn)

        declared_global: set[str] = set()
        for node in nodes:
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                declared_global.update(node.names)

        first_import: dict[str, int] = {}
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    if name in module_names and name not in declared_global:
                        first_import[name] = min(
                            first_import.get(name, node.lineno), node.lineno
                        )
        if not first_import:
            continue

        for node in nodes:
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in first_import
                and node.lineno < first_import[node.id]
            ):
                found.add((fn.name, node.id, node.lineno, first_import[node.id]))
    return sorted(found)


def _audited_files() -> list[pathlib.Path]:
    files = ra.discover_py_files()
    ra.assert_discovery_sane(files)
    roots = [r.resolve() for r in ra.production_roots()] + [
        (ra.repo_root() / "scripts").resolve()
    ]
    keep: list[pathlib.Path] = []
    for f in files:
        rf = f.resolve()
        if any(rf.is_relative_to(r) for r in roots):
            keep.append(f)
    assert len(keep) >= 400, (
        f"only {len(keep)} production/script .py files selected; the namespace "
        f"roots or scripts/ resolved wrong and this audit would be vacuous"
    )
    return keep


def test_no_use_before_function_scope_import():
    violations: list[str] = []
    for path in _audited_files():
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover
            raise AssertionError(f"cannot read {ra.rel(path)}: {exc}") from exc
        try:
            hits = scan_source(src)
        except SyntaxError as exc:  # pragma: no cover
            raise AssertionError(f"cannot parse {ra.rel(path)}: {exc}") from exc
        for fn_name, name, use_line, imp_line in hits:
            entry = f"{ra.rel(path)}::{fn_name}::{name}"
            if entry in ALLOWLIST:
                continue
            violations.append(
                f"{ra.rel(path)}:{use_line} reads {name!r} but {fn_name}() imports "
                f"it at line {imp_line}, which makes it local for the whole "
                f"function -> UnboundLocalError. Delete the function-scope "
                f"import (the module-level one already binds it), or rename it."
            )

    assert not violations, (
        "function-scope import shadows a module-level name that is read "
        "earlier in the same function:\n  " + "\n  ".join(sorted(violations))
    )


def test_allowlist_entries_are_still_violations():
    """Shrink-only: a stale allowlist entry fails so the list cannot rot."""
    if not ALLOWLIST:
        return
    live: set[str] = set()
    for path in _audited_files():
        for fn_name, name, _use, _imp in scan_source(
            path.read_text(encoding="utf-8")
        ):
            live.add(f"{ra.rel(path)}::{fn_name}::{name}")
    stale = sorted(ALLOWLIST - live)
    assert not stale, f"allowlist entries are no longer violations: {stale}"


def test_allowlist_is_empty():
    assert ALLOWLIST == frozenset(), (
        "the shadowed-import allowlist is shrink-only and currently empty; "
        f"do not grow it: {sorted(ALLOWLIST)}"
    )


# --------------------------------------------------------------------------
# Non-vacuity self-tests: the scanner must flag the real incident's shape and
# stay silent on every sanctioned form.
# --------------------------------------------------------------------------

_INCIDENT = '''
from legoesm.ocean.vertical import compute_layer_thickness

def make_physics(cfg):
    def physics_fn(state):
        if cfg.scheme == "rgb_chl":
            dz = compute_layer_thickness(state.eta, state.H, state.z)
            return dz
        elif cfg.scheme == "sweeney_2band":
            from legoesm.ocean.vertical import compute_layer_thickness
            return compute_layer_thickness(state.eta, state.H, state.z)
    return physics_fn
'''


def test_scanner_flags_the_incident_shape():
    hits = scan_source(_INCIDENT)
    assert hits, "scanner missed the exact defect that killed the MPAS run"
    fns = {h[0] for h in hits}
    names = {h[1] for h in hits}
    assert names == {"compute_layer_thickness"}
    assert "physics_fn" in fns
    # The outer factory is a separate scope and owns no such import.
    assert "make_physics" not in fns


def test_scanner_ignores_deferred_import_of_non_module_name():
    """The blessed circular-import workaround: not shadowing, not flagged."""
    src = (
        "import os\n"
        "def f():\n"
        "    from legoesm.runtime.backend import get_backend\n"
        "    return get_backend(os.getcwd())\n"
    )
    assert scan_source(src) == []


def test_scanner_ignores_import_before_use():
    src = (
        "from legoesm.ocean.vertical import compute_layer_thickness\n"
        "def f(s):\n"
        "    from legoesm.ocean.vertical import compute_layer_thickness\n"
        "    return compute_layer_thickness(s)\n"
    )
    assert scan_source(src) == []


def test_scanner_ignores_aliased_import():
    src = (
        "from legoesm.ocean.vertical import compute_layer_thickness\n"
        "def f(s):\n"
        "    x = compute_layer_thickness(s)\n"
        "    from legoesm.ocean.vertical import compute_layer_thickness as _clt\n"
        "    return x, _clt\n"
    )
    assert scan_source(src) == []


def test_scanner_ignores_global_declaration():
    src = (
        "import os\n"
        "def f():\n"
        "    global os\n"
        "    p = os.sep\n"
        "    import os\n"
        "    return p, os\n"
    )
    assert scan_source(src) == []


def test_scanner_sees_module_level_import_inside_try():
    """A module-level import nested in try/except still binds at module scope."""
    src = (
        "try:\n"
        "    from legoesm.ocean.vertical import compute_layer_thickness\n"
        "except ImportError:\n"
        "    compute_layer_thickness = None\n"
        "def f(s):\n"
        "    if s:\n"
        "        return compute_layer_thickness(s)\n"
        "    from legoesm.ocean.vertical import compute_layer_thickness\n"
        "    return compute_layer_thickness(s)\n"
    )
    hits = scan_source(src)
    assert [h[1] for h in hits] == ["compute_layer_thickness"]


def test_scanner_keeps_nested_function_scopes_separate():
    """An inner function's import must not be charged to the outer function."""
    src = (
        "import os\n"
        "def outer():\n"
        "    p = os.sep\n"
        "    def inner():\n"
        "        import os\n"
        "        return os.sep\n"
        "    return p, inner\n"
    )
    assert scan_source(src) == []
