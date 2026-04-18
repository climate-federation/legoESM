"""Regression tests for issue #188.

The ``scripts/run_amip.py`` entry point (and every other top-level
script under ``scripts/``) must be importable/runnable *without*
the repository ``tests/`` directory on ``sys.path``. In issue #188,
``src/legoesm/driver/model_driver.py`` and several scripts were
importing Held-Suarez initialization/forcing from
``tests.test_cases.held_suarez``; running ``python scripts/run_amip.py``
from the repo root then crashed with
``ModuleNotFoundError: No module named 'tests'`` because Python puts
the *script directory* (``scripts/``) on ``sys.path``, not the CWD,
so ``tests/`` is not discoverable.

These tests enforce two contracts:

1. Held-Suarez lives in the installed package at
   ``legoesm.atmosphere.held_suarez``.
2. No file under ``src/legoesm/`` or ``scripts/`` imports it from
   ``tests.test_cases.held_suarez``. The shim in
   ``tests/test_cases/held_suarez.py`` exists only so existing
   *test* code that uses the old path keeps working.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Only forbid the import in production code paths. The shim itself
# (tests/test_cases/held_suarez.py) and other existing test modules
# legitimately reference the old location.
FORBIDDEN_ROOTS = ("src/legoesm", "scripts")
FORBIDDEN_PATTERN = re.compile(
    r"^\s*from\s+tests\.test_cases\.held_suarez\s+import",
    re.MULTILINE,
)


def test_held_suarez_is_installed_package():
    """The canonical location for Held-Suarez is the installed package."""
    import legoesm.atmosphere.held_suarez as hs

    for name in (
        "held_suarez_init",
        "held_suarez_init_latlon",
        "held_suarez_init_mpas",
        "held_suarez_forcing",
        "held_suarez_forcing_latlon",
        "held_suarez_forcing_mpas",
        "held_suarez_forcing_spectral",
        "held_suarez_equilibrium_temperature",
        "K_A",
        "K_S",
        "K_F",
        "SIGMA_B",
    ):
        assert hasattr(hs, name), f"missing public symbol {name!r}"


def test_tests_shim_reexports_same_objects():
    """Back-compat shim must expose the same objects, not copies."""
    import legoesm.atmosphere.held_suarez as canonical
    import tests.test_cases.held_suarez as shim

    for name in (
        "held_suarez_init",
        "held_suarez_forcing_mpas",
        "held_suarez_equilibrium_temperature",
        "SIGMA_B",
    ):
        assert getattr(shim, name) is getattr(canonical, name), (
            f"{name} in shim differs from canonical location"
        )


def _walk_py_files(root: pathlib.Path):
    for p in root.rglob("*.py"):
        yield p


@pytest.mark.parametrize("subdir", FORBIDDEN_ROOTS)
def test_no_tests_held_suarez_import_in_production(subdir: str):
    """Production code must import from ``legoesm.atmosphere.held_suarez``.

    Running any script under ``scripts/`` from the repo root puts the
    script's own directory on ``sys.path`` — *not* the repo root — so
    ``tests/`` is not importable. Any ``from tests.test_cases.held_suarez``
    line in those files reintroduces issue #188.
    """
    offenders: list[str] = []
    for path in _walk_py_files(REPO_ROOT / subdir):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_PATTERN.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "The following production files import Held-Suarez from the "
        "tests tree, which breaks installed/script execution (see #188):\n"
        + "\n".join(f"  - {p}" for p in offenders)
    )


def _ast_find_tests_imports(source: str) -> list[tuple[int, str]]:
    """Return every ``from tests.test_cases.held_suarez import ...`` in
    ``source``, including those nested inside functions, classes, or
    conditional branches.

    A plain regex on the source text would miss some cases (e.g. where
    an ``import`` node is constructed dynamically), and more importantly
    AST-level scanning explicitly walks into function bodies so it
    catches lazy-imports that only fire when a method is called — which
    is exactly the original #188 bug pattern.
    """
    import ast

    tree = ast.parse(source)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "tests.test_cases.held_suarez" or mod.startswith(
                "tests.test_cases.held_suarez."
            ):
                names = ", ".join(a.name for a in node.names)
                hits.append((node.lineno, f"from {mod} import {names}"))
    return hits


@pytest.mark.parametrize("subdir", FORBIDDEN_ROOTS)
def test_no_nested_tests_held_suarez_import_via_ast(subdir: str):
    """AST-level scan: catch lazy imports inside functions.

    The file-level regex test above scans every line, but a future
    refactor could hide the same import inside an `exec(...)` block or
    build the module name dynamically. AST walking is the canonical way
    to enumerate every static ``import`` node, including those nested
    inside function bodies (the original #188 bug was exactly a
    function-local import), so this test guards against silent
    reintroduction regardless of indentation style.
    """
    offenders: list[str] = []
    for path in _walk_py_files(REPO_ROOT / subdir):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, stmt in _ast_find_tests_imports(text):
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"  - {rel}:{lineno}: {stmt}")
    assert not offenders, (
        "AST scan found imports of Held-Suarez from the tests tree in "
        "production code (see #188):\n" + "\n".join(offenders)
    )


def _run_subprocess_without_repo_root(code: str) -> subprocess.CompletedProcess:
    """Spawn a fresh Python process with the repo root scrubbed from
    ``PYTHONPATH`` and a neutral CWD, then run ``code``.

    This reproduces ``python scripts/<foo>.py`` launched from a
    directory where ``tests/`` is not discoverable. Any residual
    ``import tests`` that fires along the startup path will raise
    ``ModuleNotFoundError`` in the subprocess.
    """
    env = dict(os.environ)
    pp = env.get("PYTHONPATH", "")
    filtered = [
        p for p in pp.split(os.pathsep)
        if p and pathlib.Path(p).resolve() != REPO_ROOT
    ]
    env["PYTHONPATH"] = os.pathsep.join(filtered)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_run_amip_help_starts_without_tests_on_path():
    """End-to-end: ``run_amip.py --help`` must complete successfully
    when ``tests/`` is not on ``sys.path``.

    This is the exact user-visible failure in #188. ``--help`` is a
    cheap invocation — argparse exits before any heavy JAX/dycore work
    — but Python still walks every module-level import in
    ``run_amip.py`` and its import closure. Any stale ``tests.*``
    reference at module scope along that graph will surface here.
    """
    script = REPO_ROOT / "scripts" / "run_amip.py"
    assert script.is_file(), f"script not found: {script}"
    result = _run_subprocess_without_repo_root(
        "import runpy, sys\n"
        f"sys.argv = [{str(script)!r}, '--help']\n"
        f"try:\n"
        f"    runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    # argparse --help exits 0; anything else is a real failure.\n"
        "    sys.exit(int(exc.code) if exc.code is not None else 0)\n"
    )
    assert "ModuleNotFoundError: No module named 'tests'" not in result.stderr, (
        "run_amip.py still depends on tests/ during startup (#188):\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.returncode == 0, (
        "run_amip.py --help failed to start without tests/ on sys.path:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_held_suarez_resolves_without_tests_on_path():
    """Canonical Held-Suarez module must resolve in an installed env."""
    result = _run_subprocess_without_repo_root(
        "import importlib.util\n"
        "assert importlib.util.find_spec('tests') is None\n"
        "from legoesm.atmosphere.held_suarez import (\n"
        "    held_suarez_init, held_suarez_init_latlon,\n"
        "    held_suarez_init_mpas, held_suarez_forcing_mpas,\n"
        "    K_A, K_S, SIGMA_B,\n"
        ")\n"
    )
    assert result.returncode == 0, (
        f"Canonical Held-Suarez import failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_tests_shim_works_from_raw_repo_checkout():
    """``from tests.test_cases.held_suarez import ...`` must resolve when
    the package has not been pip-installed in editable mode — i.e. only
    the repo root is on ``sys.path`` and ``legoesm`` is not yet
    importable. The shim is expected to bootstrap itself by locating
    ``<repo>/src`` on disk and prepending it to ``sys.path`` before
    delegating to the canonical module.

    Reproducing this exactly requires a venv in which ``legoesm`` is
    *not* installed; since our CI venv uses ``pip install -e .``, we
    simulate the raw-checkout layout in a subprocess by:

    1. Setting ``PYTHONPATH`` to the repo root only.
    2. Removing every editable-install hook (``.pth``-injected entries
       pointing at ``src/``) from ``sys.path`` and deleting any cached
       ``legoesm`` entries from ``sys.modules`` *before* the shim
       imports. At that point, ``import legoesm`` raises
       ``ModuleNotFoundError`` and the shim's fallback path fires.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            # Strip src/ from sys.path so legoesm is not importable via
            # the editable install, forcing the shim to self-bootstrap.
            "import sys, os\n"
            "repo_src = os.path.join(os.environ['PYTHONPATH'], 'src')\n"
            "sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(repo_src)]\n"
            "for mod in [m for m in list(sys.modules) if m == 'legoesm' or m.startswith('legoesm.')]:\n"
            "    del sys.modules[mod]\n"
            "# Sanity check: legoesm must be unresolvable *before* the shim runs.\n"
            "import importlib.util\n"
            "assert importlib.util.find_spec('legoesm') is None, "
            "    'precondition failed: legoesm is already on path'\n"
            "# Now import the shim; it must self-bootstrap src/ onto sys.path.\n"
            "import tests.test_cases.held_suarez as shim\n"
            "assert callable(shim.held_suarez_init)\n"
            "assert shim.SIGMA_B == 0.7\n",
        ],
        cwd=str(REPO_ROOT.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        "tests/ shim failed from raw repo checkout (no editable "
        "install):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
