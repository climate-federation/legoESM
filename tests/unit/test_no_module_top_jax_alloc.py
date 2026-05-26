"""Regression test: no module-top JAX array allocations on `import legoesm`.

iter-93: ``grids/vertical.py`` had ``_A60 = jnp.asarray([...])`` and
``_B60 = jnp.asarray([...])`` at MODULE TOP. At import time these
eagerly dispatch ``lax.convert_element_type`` to whatever JAX
default platform is initialized — on macOS that is ``METAL``,
which currently raises ``UNIMPLEMENTED: default_memory_space is
not supported``.  Result: ``import legoesm`` bricks on Apple
Silicon before ``conftest.py`` gets a chance to call
``ensure_metal_or_fallback()`` (load order: legoesm-import
THEN fallback). Pure-Python unit tests (regex parsers, log
parsers, validators) all fail with the Metal stack-trace.

This test pins the structural invariant: hot module-top
expressions that eagerly allocate on the JAX default device must
live INSIDE a function body so they fire only when the function
is called (after fallback is in place).

Detection strategy: AST-walk legoesm source modules and flag any
top-level assignment whose RHS resolves to a ``jnp.<array_ctor>(`` /
``jax.numpy.<array_ctor>(`` call. Scalar ops (``jnp.pi``) and
metadata reads (``jnp.finfo(...).tiny``) are NOT array allocations
and are explicitly allow-listed.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


# JAX numpy / jax functions that ALLOCATE an array (vs scalar ops
# or pure-metadata reads). Adding ``arange`` here would catch a
# similar future-bug; ``finfo`` / ``iinfo`` / ``pi`` are NOT here
# because they don't trigger XLA dispatch.
_ARRAY_CONSTRUCTORS = frozenset({
    "array",
    "asarray",
    "zeros",
    "ones",
    "full",
    "arange",
    "linspace",
    "eye",
})


def _is_jax_alloc_call(node: ast.AST) -> bool:
    """True if ``node`` is a call to a JAX array constructor.

    Matches ``jnp.array(...)``, ``jnp.asarray(...)``,
    ``jax.numpy.array(...)``, etc. Does NOT match
    ``jnp.finfo(...).tiny`` (attribute access on the call result is
    not an alloc call itself unless the outer call is a constructor).
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in _ARRAY_CONSTRUCTORS:
        return False
    val = func.value
    if isinstance(val, ast.Name) and val.id in ("jnp", "np"):
        return val.id == "jnp"
    if isinstance(val, ast.Attribute):
        if val.attr == "numpy" and isinstance(val.value, ast.Name) and val.value.id == "jax":
            return True
    return False


def _walk_for_top_level_jax_allocs(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(line, code_snippet)`` for each top-level JAX alloc."""
    hits: list[tuple[int, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            for sub in ast.walk(value):
                if _is_jax_alloc_call(sub):
                    snippet = ast.unparse(node).splitlines()[0][:80]
                    hits.append((node.lineno, snippet))
                    break
    return hits


# Files we know must stay free of module-top JAX allocs because
# they sit on the ``import legoesm`` critical path. Anything in
# ``src/legoesm/grids/`` is loaded by ``grids/__init__.py``.
_PROTECTED_MODULES = [
    "src/legoesm/grids/vertical.py",
    "src/legoesm/grids/__init__.py",
    "src/legoesm/__init__.py",
    "src/legoesm/core/__init__.py",
    "src/legoesm/runtime/__init__.py",
    "src/legoesm/parallel/__init__.py",
    "src/legoesm/constants.py",
]


@pytest.mark.parametrize("rel_path", _PROTECTED_MODULES)
def test_no_module_top_jax_array_alloc(rel_path: str) -> None:
    """No module-top ``jnp.asarray``/``jnp.array``/etc in protected modules.

    Why: import-time eager dispatch races
    ``ensure_metal_or_fallback()`` and crashes on Metal. See iter-93
    docstring at the top of this file.
    """
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / rel_path
    if not path.exists():
        pytest.skip(f"protected module missing: {rel_path}")
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = _walk_for_top_level_jax_allocs(tree)
    if hits:
        details = "\n".join(f"  line {ln}: {snip}" for ln, snip in hits)
        raise AssertionError(
            f"{rel_path}: module-top JAX array allocation found:\n"
            f"{details}\n"
            f"Move the alloc INSIDE a function body so it only fires "
            f"after `ensure_metal_or_fallback()` has run."
        )


def test_legoesm_imports_without_jax_dispatch_crash() -> None:
    """``import legoesm`` succeeds on whatever the default JAX
    platform is at conftest time.

    This is a smoke test — it only validates that the static AST
    invariant above translates to a working import. Failure here
    means a NEW module-top JAX alloc has slipped in to a module
    not yet in ``_PROTECTED_MODULES``.
    """
    import importlib

    mod = importlib.import_module("legoesm")
    assert mod is not None
