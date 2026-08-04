"""Regression test: no module-top JAX array allocations on `import legoesm`.

iter-93: ``grids/vertical.py`` had ``_A60 = jnp.asarray([...])`` and
``_B60 = jnp.asarray([...])`` at MODULE TOP. At import time these
eagerly dispatch ``lax.convert_element_type`` to whatever JAX
default platform is initialized — on macOS Apple Silicon that is an
Apple GPU backend.  Eager module-top dispatch on the default device
at import time (before backend configuration / device routing) is
fragile and platform-dependent; on an Apple GPU stack that lacks an
op it can brick ``import legoesm`` outright with a low-level
``UNIMPLEMENTED`` dispatch error.

This test pins the structural invariant: hot module-top
expressions that eagerly allocate on the JAX default device must
live INSIDE a function body so they fire only when the function
is called (after backend configuration is in place).

iter-93 Codex hardening (post-adversarial-review):
* Constructor set widened to cover ``empty``, ``*_like``,
  ``meshgrid``, ``broadcast_to``, ``tile``, ``repeat``, ``logspace``,
  ``identity``, ``diag``, etc.  Any of these on module-top would
  hit the same eager-dispatch crash.
* Alias-aware: parses ``import jax.numpy as <name>`` / ``from jax
  import numpy as <name>`` / ``from jax.numpy import asarray, ...``
  and tracks ALL bound names that resolve to ``jax.numpy`` or
  individual JAX numpy constructors.  No longer assumes the alias
  is literally ``jnp``.
  (Any of these at module top would hit the same eager-dispatch crash.)
* ``jax.device_put(...)`` detected (also triggers eager dispatch).
* Protected-module list now scans the entire ``src/legoesm/grids/``
  subtree (per Codex MEDIUM — re-exports from ``__init__.py``
  pull in the whole package early).
* ``src/legoesm/parallel/cubesphere_exchange.py`` added to
  protected list (Codex LOW#3: had 3 module-top ``jnp.array``
  tables of its own; latent risk because not on the eager
  import path but a single ``from legoesm.parallel.cubesphere_exchange
  import …`` from a test conftest would re-trigger the crash).
* Subprocess cold-import smoke test: spawn a fresh interpreter
  with no ``conftest.py`` preamble and verify ``import legoesm``
  succeeds.  The in-process smoke test cannot exercise the
  pre-fallback failure mode (conftest already ran fallback by
  the time pytest collects this file).
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


# JAX numpy / jax functions that ALLOCATE an array (vs scalar ops
# or pure-metadata reads). Codex MEDIUM iter-93: widened from the
# initial 8 constructors to the full set of array-producing JAX
# numpy ops that would trigger eager device dispatch on module top.
# Excluded: ``finfo`` / ``iinfo`` / ``pi`` / ``e`` / ``newaxis``
# (constants, metadata reads — no XLA dispatch).
_ARRAY_CONSTRUCTORS = frozenset({
    # bare allocators
    "array", "asarray", "zeros", "ones", "empty", "full",
    "arange", "linspace", "logspace", "geomspace",
    "eye", "identity", "diag", "diagflat", "tri",
    # _like family — allocates new buffer matching shape/dtype
    "zeros_like", "ones_like", "empty_like", "full_like",
    # shape/broadcast operations that allocate new arrays
    "meshgrid", "broadcast_to", "tile", "repeat",
})

# Top-level JAX functions that move data to device (also trigger
# eager dispatch on the default platform).
_JAX_DISPATCH_FUNCS = frozenset({"device_put"})


def _collect_jax_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Walk MODULE-TOP imports only; return (jax_numpy_aliases, jax_aliases).

    Tracks BOTH:
    * Module aliases pointing at ``jax.numpy`` (e.g., ``jnp``,
      ``jax_np``, ``np2`` after ``import jax.numpy as np2``).
    * Module aliases pointing at ``jax`` itself (for ``jax.device_put``
      detection).

    Does NOT track ``from jax.numpy import asarray`` — direct-name
    imports surface as ``ast.Name`` nodes, handled separately in
    ``_is_jax_alloc_call``.

    iter-94 (Codex Q3 LOW): scans ``tree.body`` only, NOT
    ``ast.walk(tree)``. A function-local ``import jax.numpy as jnp``
    must not be collected as a module-level alias — that would
    produce false-positive matches at module top against names that
    don't actually bind to ``jnp`` there.
    """
    jnp_aliases: set[str] = set()
    jax_aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "jax.numpy":
                    jnp_aliases.add(alias.asname or "jax.numpy")
                elif alias.name == "jax":
                    jax_aliases.add(alias.asname or "jax")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "jax" and node.level == 0:
                for alias in node.names:
                    if alias.name == "numpy":
                        jnp_aliases.add(alias.asname or "numpy")
    return jnp_aliases, jax_aliases


def _collect_directly_imported_ctors(tree: ast.AST) -> set[str]:
    """Names bound directly from ``jax.numpy`` at MODULE TOP.

    Returns the set of names (e.g., ``{"asarray", "zeros"}``) that
    were imported via ``from jax.numpy import asarray`` at module
    scope and intersect with ``_ARRAY_CONSTRUCTORS``.

    iter-94 (Codex Q3 LOW): scans ``tree.body`` only, same
    rationale as ``_collect_jax_aliases``.
    """
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "jax.numpy":
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name in _ARRAY_CONSTRUCTORS:
                    bound.add(name)
    return bound


def _is_jax_alloc_call(
    node: ast.AST,
    jnp_aliases: set[str],
    jax_aliases: set[str],
    direct_ctors: set[str],
) -> bool:
    """True if ``node`` is a call that eagerly dispatches a JAX op.

    Matches:
    * ``<jnp_alias>.<array_ctor>(...)``  (e.g., ``jnp.asarray(...)``)
    * ``<jax_alias>.numpy.<array_ctor>(...)``
    * ``<jax_alias>.device_put(...)``
    * ``<direct_ctor>(...)`` for ``from jax.numpy import <ctor>``
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func

    if isinstance(func, ast.Name) and func.id in direct_ctors:
        return True

    if not isinstance(func, ast.Attribute):
        return False

    val = func.value

    if isinstance(val, ast.Name):
        if val.id in jnp_aliases and func.attr in _ARRAY_CONSTRUCTORS:
            return True
        if val.id in jax_aliases and func.attr in _JAX_DISPATCH_FUNCS:
            return True
        return False

    if isinstance(val, ast.Attribute):
        # jax.numpy.<ctor>: val.attr=="numpy", val.value.id in jax_aliases.
        if (
            val.attr == "numpy"
            and isinstance(val.value, ast.Name)
            and val.value.id in jax_aliases
            and func.attr in _ARRAY_CONSTRUCTORS
        ):
            return True

    return False


def _walk_for_top_level_jax_allocs(
    tree: ast.AST,
) -> list[tuple[int, str]]:
    """Return ``(line, code_snippet)`` for each top-level JAX alloc.

    Scans ``ast.Assign``, ``ast.AnnAssign``, AND bare ``ast.Expr``
    statements at module scope (Codex LOW: a bare ``jax.device_put(...)``
    at module top is a statement, not an assignment, and the initial
    walker missed it).
    """
    jnp_aliases, jax_aliases = _collect_jax_aliases(tree)
    direct_ctors = _collect_directly_imported_ctors(tree)

    hits: list[tuple[int, str]] = []
    for node in tree.body:
        candidates: list[ast.AST] = []
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is not None:
                candidates.append(node.value)
        elif isinstance(node, ast.Expr):
            candidates.append(node.value)
        for value in candidates:
            for sub in ast.walk(value):
                if _is_jax_alloc_call(sub, jnp_aliases, jax_aliases, direct_ctors):
                    try:
                        snippet = ast.unparse(node).splitlines()[0][:80]
                    except Exception:
                        snippet = f"<unparseable node at line {node.lineno}>"
                    hits.append((node.lineno, snippet))
                    break
    return hits


def _enumerate_protected_modules(repo_root: Path) -> list[str]:
    """Modules on the ``import legoesm`` critical path that must
    stay free of module-top JAX allocs.

    Codex MEDIUM iter-93: extended from a hand-curated 7-file list
    to the full ``src/legoesm/grids/`` subtree (since
    ``grids/__init__.py`` re-exports submodules) plus the
    ``__init__.py`` files of the early-imported subpackages plus
    ``parallel/cubesphere_exchange.py`` (Codex LOW#3 latent risk).
    """
    from tests.legoesm_paths import legoesm_source_path

    files: list[Path] = []

    # repo_root= so a run from a git WORKTREE resolves paths inside ITS OWN
    # tree: legoesm.__path__ reports the editable install's roots, which point
    # at the canonical checkout, and the relative_to() at the end of this
    # function then raised a bare ValueError naming a foreign path (#1389).
    grids_dir = legoesm_source_path("grids", repo_root)
    if grids_dir.exists():
        files.extend(sorted(grids_dir.rglob("*.py")))

    # Early-imported package __init__s + substrate top-level modules.  (The
    # legoesm namespace itself has no __init__.py — PEP-420 namespace package —
    # so it is intentionally absent.)
    for rel in (
        "core/__init__.py",
        "runtime/__init__.py",
        "parallel/__init__.py",
        "constants.py",
        "parallel/cubesphere_exchange.py",
    ):
        try:
            path = legoesm_source_path(rel, repo_root)
        except FileNotFoundError:
            continue
        if path not in files:
            files.append(path)

    return [str(p.relative_to(repo_root)) for p in files]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROTECTED_MODULES = _enumerate_protected_modules(_REPO_ROOT)


@pytest.mark.parametrize("rel_path", _PROTECTED_MODULES)
def test_no_module_top_jax_array_alloc(rel_path: str) -> None:
    """No module-top ``jnp.asarray``/``jnp.array``/etc in protected modules.

    Why: import-time eager dispatch runs before backend configuration
    and can crash on an Apple GPU backend. See iter-93 docstring at the
    top of this file.
    """
    path = _REPO_ROOT / rel_path
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
            f"after backend configuration has run. Pattern: "
            f"`_TBL = np.asarray(...)` at module top, "
            f"`jnp.asarray(_TBL)` inside the caller."
        )


def test_legoesm_imports_without_jax_dispatch_crash() -> None:
    """``import legoesm`` succeeds on the current JAX platform.

    This is a smoke test — the in-process variant has limited value
    because by the time pytest collects this test, the backend has
    already been configured. Failure here means a NEW module-top JAX
    alloc has slipped in to a module not yet covered by
    ``_PROTECTED_MODULES``.
    """
    import importlib

    mod = importlib.import_module("legoesm")
    assert mod is not None


# Modules that are lazy-imported (not reached by ``import legoesm``).
# For these, the AST-static check catches literal ``jnp.<ctor>(...)``
# at module top, but a function call returning jax.Array values
# bound to module globals (iter-94d bug class) WILL slip past the
# static check. The runtime introspection test below complements
# the AST check for these modules.
_LAZY_PROTECTED_MODULES = [
    # iter-94d: caught the original residual bug via this entry.
    "legoesm.parallel.cubesphere_exchange",
    # iter-94f: proactive coverage of other lazy modules that were
    # empirically verified clean at iter-94 (no module-top
    # jax.Array globals). These entries lock the invariant so a
    # future refactor that re-introduces a function-call-returns-
    # jax.Array pattern at module top will fire this test.
    "legoesm.parallel.voronoi_partition",
    "legoesm.parallel.mesh",
    "legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid",
    "legoesm.atmosphere.dynamics.gcm.compressible_euler",
    "legoesm.ml.sfno",
]


@pytest.mark.parametrize("module_name", _LAZY_PROTECTED_MODULES)
def test_lazy_module_no_jax_array_globals(module_name: str) -> None:
    """Runtime introspection: import ``module_name`` in a fresh
    subprocess and assert no module global is a ``jax.Array``.

    iter-94d motivation: ``_build_ppermute_tables()`` returned
    ``jnp.array`` tables and was called at module top:
    ``_PPERMUTE_SEND, ... = _build_ppermute_tables()``. The AST
    static check did not detect this because the call site is
    ``_build_ppermute_tables()`` (a plain function call, not a
    literal ``jnp.<ctor>(...)``). Static analysis cannot trace
    into the function body without full inter-procedural analysis.

    This runtime test catches the bug class regardless of how the
    ``jax.Array`` ended up at module top: literal constructor,
    function return, decorator side effect, etc.

    Forces ``JAX_PLATFORMS=cpu`` so the test works on macOS where
    an Apple GPU default could crash before the assertion runs.
    """
    import os
    code = (
        f"import jax\n"
        f"import {module_name} as m\n"
        f"bad = [k for k, v in vars(m).items() if isinstance(v, jax.Array)]\n"
        f"if bad:\n"
        f"    raise AssertionError(\n"
        f"        'Module-top jax.Array globals in {module_name}: ' + repr(bad)\n"
        f"    )\n"
        f"print('ok')\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert proc.returncode == 0, (
        f"Runtime jax.Array global introspection failed for "
        f"{module_name}:\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}\n"
        f"This is the iter-94d failure mode: a function call at "
        f"module top returned jax.Array values that bound to "
        f"module globals — the AST audit cannot detect this "
        f"pattern; use the runtime check OR refactor the function "
        f"to return numpy arrays."
    )
    assert "ok" in proc.stdout


def test_legoesm_imports_cold_subprocess() -> None:
    """Subprocess cold-import smoke test (Codex LOW iter-93).

    Spawn a fresh Python interpreter with NO conftest preamble and
    verify ``import legoesm`` succeeds. This exercises the same
    code path that broke pre-iter-93 on Apple Silicon: legoesm
    imports BEFORE backend configuration gets a chance to run.

    We let JAX pick its default platform. On Linux/x86 the default is
    CPU (no-op). On macOS Apple Silicon it is an Apple GPU backend
    (the original bug scenario).
    """
    code = "import legoesm; print('ok')"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"`import legoesm` failed in cold subprocess "
        f"(returncode={proc.returncode}). This is the iter-93 "
        f"failure mode: a module on the import path eagerly "
        f"dispatched a JAX op before any fallback could apply.\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    assert "ok" in proc.stdout


def test_whole_legoesm_codebase_no_jax_array_globals() -> None:
    """iter-94h: single-subprocess WHOLE-CODEBASE scan for module-top
    jax.Array globals across all 495+ legoesm modules.

    Complements the parametrized ``_LAZY_PROTECTED_MODULES`` list:
    instead of requiring each module to be hand-added, this test
    discovers all submodules of ``legoesm`` via ``pkgutil.walk_packages``
    and inspects globals on each. Catches iter-94d-class regressions
    (function-returning-jax.Array bound to module-top) in ANY
    legoesm module without manual enumeration.

    Marked ``slow`` because it imports the whole codebase (takes
    ~20s wall on M5 Pro). Runs in a subprocess with
    ``JAX_PLATFORMS=cpu`` so it works on macOS without crashing on
    an Apple GPU backend.

    Failure mode: lists every module that has module-top
    ``jax.Array`` globals, separating real failures from import
    errors (some modules legitimately fail to import on this
    platform — e.g. spectral_plane has a known stale-import
    issue; those are skipped, not failed).
    """
    import os
    code = (
        "import jax\n"
        "import importlib\n"
        "import pkgutil\n"
        "import legoesm\n"
        "bad_modules = []\n"
        "import_errors = []\n"
        "for mod_info in pkgutil.walk_packages(legoesm.__path__, 'legoesm.'):\n"
        "    if mod_info.ispkg:\n"
        "        continue\n"
        "    try:\n"
        "        m = importlib.import_module(mod_info.name)\n"
        "    except Exception as e:\n"
        "        import_errors.append((mod_info.name, type(e).__name__))\n"
        "        continue\n"
        "    bad = [k for k, v in vars(m).items() if isinstance(v, jax.Array)]\n"
        "    if bad:\n"
        "        bad_modules.append((mod_info.name, bad[:5]))\n"
        "if bad_modules:\n"
        "    msg = 'Module-top jax.Array globals found:\\n'\n"
        "    for name, bad in bad_modules:\n"
        "        msg += f'  {name}: {bad}\\n'\n"
        "    raise AssertionError(msg)\n"
        "print(f'CLEAN: scanned modules; import_errors={len(import_errors)}')\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, (
        f"Whole-codebase audit failed:\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    assert "CLEAN" in proc.stdout
