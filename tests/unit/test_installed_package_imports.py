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
    # NOTE: do not assert ``find_spec('tests') is None`` here — an
    # unrelated site-packages ``tests`` package (which some CI images
    # ship) would make that precondition flaky. The real contract is
    # that the *canonical* Held-Suarez resolves without relying on
    # this repo's tests tree, which the import below verifies directly.
    result = _run_subprocess_without_repo_root(
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


def test_tests_shim_recovers_from_stale_legoesm_install():
    """Shim must self-bootstrap even when a stale ``legoesm`` package is
    already importable but *lacks* the ``atmosphere.held_suarez``
    submodule — e.g. an older wheel/editable install from before the
    #188 move.

    Reproduction: inject a dummy ``legoesm`` namespace package into a
    throwaway staging directory (with no ``atmosphere`` submodule),
    prepend that directory to ``sys.path`` so ``import legoesm``
    succeeds but ``import legoesm.atmosphere.held_suarez`` would fail,
    and then import the shim. The shim is expected to notice the
    missing submodule via ``importlib.util.find_spec`` and prepend
    ``<repo>/src`` so the delegating import resolves to the checkout
    copy rather than the stale stand-in.
    """
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as stale_root:
        stale_pkg = pathlib.Path(stale_root) / "legoesm"
        stale_pkg.mkdir()
        (stale_pkg / "__init__.py").write_text(
            textwrap.dedent(
                """
                # Deliberately stale: no ``atmosphere`` submodule.
                STALE_SENTINEL = True
                """
            ).lstrip()
        )

        env = dict(os.environ)
        # Put the stale package directory *ahead* of the repo root so
        # ``import legoesm`` resolves to the stale stand-in first.
        env["PYTHONPATH"] = os.pathsep.join([str(stale_root), str(REPO_ROOT)])
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                # Scrub editable-install src/ so the stale stand-in wins.
                "import sys, os\n"
                "repo_src = os.path.join(os.environ['PYTHONPATH'].split(os.pathsep)[-1], 'src')\n"
                "sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(repo_src)]\n"
                "for mod in [m for m in list(sys.modules) if m == 'legoesm' or m.startswith('legoesm.')]:\n"
                "    del sys.modules[mod]\n"
                # Prime the stale install: top-level import must succeed,\n"
                # but the submodule must not be reachable yet.\n"
                "import legoesm\n"
                "assert getattr(legoesm, 'STALE_SENTINEL', False), 'stale stand-in not active'\n"
                "import importlib.util\n"
                "assert importlib.util.find_spec('legoesm.atmosphere') is None, "
                "    'stale stand-in unexpectedly exposes atmosphere/'\n"
                # Now import the shim; it must detect the missing\n"
                # submodule, prepend <repo>/src, evict the stale\n"
                # top-level module, and resolve against the checkout.\n"
                "import tests.test_cases.held_suarez as shim\n"
                "assert callable(shim.held_suarez_init)\n"
                "assert shim.SIGMA_B == 0.7\n"
                "# After the shim runs, the canonical module must come\n"
                "# from the checkout, not the stale stand-in.\n"
                "import legoesm.atmosphere.held_suarez as canonical\n"
                "assert shim.held_suarez_init is canonical.held_suarez_init\n",
            ],
            cwd=str(REPO_ROOT.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, (
        "tests/ shim failed to recover from a stale legoesm install:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_tests_shim_recovers_when_src_is_on_path_but_shadowed():
    """Mixed case: checkout ``<repo>/src`` is already on ``sys.path``,
    but a stale ``legoesm`` package appears *earlier* on the path and
    wins the initial ``import legoesm``. The shim must still recover
    and resolve the delegating import against the checkout.

    This is the realistic editable-install scenario that round-4
    Codex review flagged: pip install -e already put ``src/`` on the
    path, yet the developer's environment also has a different
    ``legoesm`` earlier (e.g. a site-packages wheel or a leftover
    staging directory). If the shim only inserts ``src/`` when
    missing, it leaves the stale package in control and crashes on
    the star-import of ``legoesm.atmosphere.held_suarez``.
    """
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as stale_root:
        stale_pkg = pathlib.Path(stale_root) / "legoesm"
        stale_pkg.mkdir()
        (stale_pkg / "__init__.py").write_text(
            textwrap.dedent(
                """
                # Deliberately stale: no ``atmosphere`` submodule.
                STALE_SENTINEL = True
                """
            ).lstrip()
        )

        env = dict(os.environ)
        # Put the stale stand-in *earlier* than any ``src/`` hint; we
        # rely on the in-process ``sys.path`` manipulation to emulate
        # the editable install having ``src/`` already discoverable.
        env["PYTHONPATH"] = os.pathsep.join([str(stale_root), str(REPO_ROOT)])
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                # Emulate a mixed editable-install state: keep
                # ``<repo>/src`` on sys.path, but *after* the stale
                # stand-in so ``import legoesm`` resolves there first.
                "import sys, os\n"
                "repo_root = os.environ['PYTHONPATH'].split(os.pathsep)[-1]\n"
                "repo_src = os.path.join(repo_root, 'src')\n"
                "if repo_src not in sys.path:\n"
                "    sys.path.append(repo_src)\n"
                # Clear any cached legoesm modules and prime the stale\n"
                # stand-in so its incomplete top-level __init__ wins.\n"
                "for mod in [m for m in list(sys.modules) if m == 'legoesm' or m.startswith('legoesm.')]:\n"
                "    del sys.modules[mod]\n"
                "import legoesm\n"
                "assert getattr(legoesm, 'STALE_SENTINEL', False), "
                "    'stale stand-in is not active'\n"
                "import importlib.util\n"
                "try:\n"
                "    sub = importlib.util.find_spec('legoesm.atmosphere.held_suarez')\n"
                "except ModuleNotFoundError:\n"
                "    sub = None\n"
                "assert sub is None, 'precondition failed: submodule already reachable'\n"
                # Now import the shim. The fix must: move <repo>/src to\n"
                # the front, evict stale legoesm from sys.modules, and\n"
                # re-resolve the delegating import against the checkout.\n"
                "import tests.test_cases.held_suarez as shim\n"
                "assert callable(shim.held_suarez_init)\n"
                "import legoesm.atmosphere.held_suarez as canonical\n"
                "assert shim.held_suarez_init is canonical.held_suarez_init\n"
                "import legoesm as reloaded\n"
                "assert not getattr(reloaded, 'STALE_SENTINEL', False), "
                "    'shim did not evict the stale top-level package'\n",
            ],
            cwd=str(REPO_ROOT.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, (
        "tests/ shim failed to recover from a stale install shadowing "
        "an already-on-path src/:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_tests_shim_rejects_stale_install_that_includes_submodule():
    """Stop-gate regression: a stale install that *does* ship
    ``legoesm.atmosphere.held_suarez`` must still not silently bind
    the shim to it.

    Previous shim logic bootstrapped ``<repo>/src`` only when the
    target submodule was unreachable. If a user had an older
    ``legoesm`` earlier on ``sys.path`` that *included* the submodule
    (e.g. a wheel published between two versions of this fix), the
    shim would detect a valid spec, skip the bootstrap, and re-export
    the stale copy — silently running tests against the wrong code.

    The hardened shim must compare the resolved submodule origin
    against ``<repo>/src`` and force the checkout to win whenever the
    two disagree.
    """
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as stale_root:
        stale_pkg = pathlib.Path(stale_root) / "legoesm"
        atmosphere_pkg = stale_pkg / "atmosphere"
        atmosphere_pkg.mkdir(parents=True)
        (stale_pkg / "__init__.py").write_text("STALE_SENTINEL = True\n")
        (atmosphere_pkg / "__init__.py").write_text("")
        # Stale submodule exposes a marker attribute and a sentinel
        # function object. If the shim binds to this copy, the test
        # will see the sentinels instead of the checkout symbols.
        (atmosphere_pkg / "held_suarez.py").write_text(
            textwrap.dedent(
                """
                STALE_HELD_SUAREZ = True

                def held_suarez_init(*args, **kwargs):
                    return 'stale-init'

                SIGMA_B = -1.0
                K_A = -1.0
                K_S = -1.0
                K_F = -1.0
                """
            ).lstrip()
        )

        env = dict(os.environ)
        # Put the stale install *earlier* than the repo root so that
        # ``import legoesm`` and ``import legoesm.atmosphere.held_suarez``
        # both resolve to it before the shim runs.
        env["PYTHONPATH"] = os.pathsep.join([str(stale_root), str(REPO_ROOT)])
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, os\n"
                "repo_root = os.environ['PYTHONPATH'].split(os.pathsep)[-1]\n"
                "repo_src = os.path.join(repo_root, 'src')\n"
                "if repo_src not in sys.path:\n"
                "    sys.path.append(repo_src)\n"
                "for mod in [m for m in list(sys.modules) if m == 'legoesm' or m.startswith('legoesm.')]:\n"
                "    del sys.modules[mod]\n"
                # Precondition: the stale submodule is fully reachable\n"
                # before the shim runs — the critical difference from\n"
                # the earlier stale-install test.\n"
                "import legoesm.atmosphere.held_suarez as pre\n"
                "assert getattr(pre, 'STALE_HELD_SUAREZ', False), "
                "    'precondition failed: stale submodule not active'\n"
                "assert pre.held_suarez_init() == 'stale-init'\n"
                # Now import the shim. It must detect that the found\n"
                # submodule lives outside <repo>/src, prepend the\n"
                # checkout, evict cached modules, and re-resolve.\n"
                "import tests.test_cases.held_suarez as shim\n"
                "assert not getattr(shim, 'STALE_HELD_SUAREZ', False), "
                "    'shim silently bound to the stale installed module'\n"
                "assert shim.SIGMA_B == 0.7, "
                "    f'SIGMA_B came from stale module: {shim.SIGMA_B!r}'\n"
                # After the shim runs, a fresh import of the canonical\n"
                # name must also resolve to the checkout, not the stale\n"
                # copy — confirming sys.path + sys.modules were both\n"
                # repaired, not just the shim's own references.\n"
                "import legoesm.atmosphere.held_suarez as canonical\n"
                "assert not getattr(canonical, 'STALE_HELD_SUAREZ', False), "
                "    'canonical re-import still returned the stale module'\n"
                "assert shim.held_suarez_init is canonical.held_suarez_init\n",
            ],
            cwd=str(REPO_ROOT.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, (
        "tests/ shim silently bound to a stale installed Held-Suarez "
        "module that happened to ship the submodule (#188 stop-gate):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
