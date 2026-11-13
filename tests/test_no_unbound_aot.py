"""Ratchet: every ahead-of-time program in production code is bound to its numerics mode.

A jitted function re-traces when the numerics mode changes; a ``Compiled`` or
``Exported`` object does not: it runs the program baked when it was made, under
any later mode.  Production code (``packages/``, ``src/``, non-test) must
therefore build each one inside the argument of
``legoesm.core.numerics_mode.bind_aot(...)``, in one expression, so the stamp
is the mode the program was made under.

Flagged: ``.lower(<args>)`` and ``.trace(...)`` (except numpy's) outside such
an argument; ``.compile(...)`` (except ``re.compile``) unless it is called
directly on a ``.lower(...)`` inside one; any ``jax.export`` reference (also
through ``import jax as j``) except ``jax.export.export`` inside one; and every
import of ``jax.export`` under any alias.  So a saved ``Lowered`` compiled later,
under another mode, is flagged at its ``.compile()``.

Not flagged, by construction: a zero-argument ``.lower()``, which the AST
cannot tell from ``str.lower()`` (its result only runs after a flagged
``.compile()``), and dynamic access (``getattr(jax, "export")``,
``importlib``).
"""

from __future__ import annotations

import ast
import pathlib

from tests import _ratchet_audit as ra

#: (repo-relative file, line) -> reason.  Shrink-only; empty at introduction.
ALLOWED: dict[tuple[str, int], str] = {}
#: The binder itself names ``jax.export.Exported`` for its type check.
_BINDER = "packages/core/legoesm/core/numerics_mode.py"

_NUMPY = {"np", "jnp", "numpy", "jax.numpy"}
_REGEX = {"re", "regex"}


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_jax_export(module: str | None) -> bool:
    return module is not None and (module == "jax.export"
                                   or module.startswith("jax.export."))


def _is_aot(node: ast.AST, jax_names: set[str]) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        attr, receiver = node.func.attr, _dotted(node.func.value)
        if attr == "lower":
            return bool(node.args or node.keywords)
        if attr == "trace":
            return receiver not in _NUMPY
        if attr == "compile":
            return receiver not in _REGEX
    if isinstance(node, ast.Attribute):
        return _dotted(node) in {f"{name}.export" for name in jax_names}
    if isinstance(node, ast.Import):
        return any(_is_jax_export(alias.name) for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return _is_jax_export(node.module) or (
            node.module == "jax" and any(a.name == "export" for a in node.names))
    return False


def _routable(node: ast.AST, parent: dict[int, ast.AST]) -> bool:
    """Whether this use may sit inside a ``bind_aot(...)`` argument."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile"):
        receiver = node.func.value
        return (isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Attribute)
                and receiver.func.attr == "lower")
    if isinstance(node, ast.Attribute):
        above = parent.get(id(node))
        return isinstance(above, ast.Attribute) and above.attr == "export"
    return isinstance(node, ast.Call)


def unbound_aot_lines(source: str) -> list[int]:
    """Lines holding an AOT/export use not routed through ``bind_aot(...)``."""
    tree = ast.parse(source)
    jax_names = {"jax"} | {
        alias.asname for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names if alias.name == "jax" and alias.asname}
    parent = {id(child): node for node in ast.walk(tree)
              for child in ast.iter_child_nodes(node)}
    bound: set[int] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and (_dotted(node.func) or "").rsplit(".", 1)[-1] == "bind_aot"):
            for arg in (*node.args, *(k.value for k in node.keywords)):
                bound.update(id(n) for n in ast.walk(arg))
    return sorted({
        node.lineno for node in ast.walk(tree)
        if _is_aot(node, jax_names)
        and not (id(node) in bound and _routable(node, parent))})


def _production_files() -> list[pathlib.Path]:
    files = ra.discover_py_files()
    ra.assert_discovery_sane(files)
    roots = ra.production_roots()
    return [
        f for f in files
        if any(f.is_relative_to(r) for r in roots)
        and "tests" not in ra.rel(f).split("/")
        and not f.name.startswith("test_") and f.name != "conftest.py"
        and ra.rel(f) != _BINDER
    ]


def test_no_unbound_aot_in_production_code() -> None:
    files = _production_files()
    assert len(files) > 500, len(files)
    found = {(ra.rel(f), line)
             for f in files for line in unbound_aot_lines(f.read_text())}
    assert found == set(ALLOWED), (
        "AOT/export use not routed through numerics_mode.bind_aot (or a stale "
        f"ALLOWED entry): new={sorted(found - set(ALLOWED))} "
        f"stale={sorted(set(ALLOWED) - found)}")


def test_scanner_flags_an_unwrapped_lower_compile(tmp_path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import re\n"
        "import jax\n"
        "from jax import export\n"
        "import jax.numpy as jnp\n"
        "from legoesm.core.numerics_mode import bind_aot\n"
        "compiled = jax.jit(f).lower(x).compile()\n"
        "ok = bind_aot(jax.jit(f).lower(x).compile())\n"
        "ok2 = bind_aot(jax.export.export(jax.jit(f))(x))\n"
        "bad = jax.export.export(jax.jit(f))(x)\n"
        "name = 'ABC'.lower(); pattern = re.compile('a'); t = jnp.trace(m)\n"
        "traced = jax.jit(f).trace(x)\n"
    )
    assert unbound_aot_lines(planted.read_text()) == [3, 6, 9, 11]


def test_scanner_flags_aliases_and_a_saved_lowering(tmp_path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import jax as j\n"
        "import jax.export as je\n"
        "from jax import export as ex\n"
        "from jax.export import deserialize\n"
        "from legoesm.core.numerics_mode import bind_aot\n"
        "e = j.export.export(j.jit(f))(x)\n"
        "ok = bind_aot(j.export.export(j.jit(f))(x))\n"
        "lowered = j.jit(f).lower()\n"
        "laundered = bind_aot(lowered.compile())\n"
        "blob = bind_aot(j.export.deserialize(b))\n"
    )
    assert unbound_aot_lines(planted.read_text()) == [2, 3, 4, 6, 9, 10]
