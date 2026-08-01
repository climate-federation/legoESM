"""Regression coverage for issue #188 and the generalised
production-imports-from-tests anti-pattern.

Background
----------
Issue #188: ``scripts/run/run_amip.py`` (and other top-level scripts under
``scripts/``) crashed at startup with
``ModuleNotFoundError: No module named 'tests'`` because
``src/legoesm/driver/model_driver.py`` was importing Held-Suarez from
``tests.test_cases.held_suarez``. Running a script puts the *script
directory* on ``sys.path``, not the repo root, so ``tests/`` is not
discoverable in that context.

The original fix moved the Held-Suarez implementation into the
installed package (``legoesm.atmosphere.forcing.idealized.held_suarez``) and left a
back-compat shim at ``tests/test_cases/held_suarez.py``. The shim was
later removed; existing test callers were migrated to import the
canonical location directly.

Why this file still exists
--------------------------
Removing the shim makes ``from tests.test_cases.held_suarez import ...``
fail at parse time, so the original held_suarez-specific narrow guard
is now structurally enforced. But the underlying anti-pattern —
production code importing from the ``tests/`` tree at all — is broader
than held_suarez and is not enforced anywhere else. ``src/legoesm/cli.
py``, ``src/legoesm/atmosphere/dynamics/gcm/spectral_nh.py``, and a number
of scripts currently exhibit the same pattern for ``williamson``,
``baroclinic_wave``, ``dcmip2025``, ``cosine_bell``, ``dcmip_transport``,
and ``test_williamson2_cdgrid``. Each of those is a latent #188-style
bug waiting to surface in any context where ``tests/`` is not on
``sys.path``.

This file enforces three layers of protection:

1. **Narrow held_suarez guard** (``test_no_held_suarez_imports_from_tests_in_production``):
   reproduces the exact #188 condition. Even though Python now raises
   ``ModuleNotFoundError`` at parse time, this AST scan runs at pytest
   collection with a clearer failure message naming the file, line
   number, and the canonical replacement.

2. **Per-file module-set allowlist** (``test_no_new_tests_imports_in_production``
   plus ``test_grandfather_list_is_minimal``): scans every ``.py``
   under ``src/legoesm/`` and ``scripts/`` for any ``from tests.*`` or
   ``import tests.*``. The allowlist
   ``GRANDFATHERED_TESTS_IMPORTS_BY_FILE`` is a dict mapping each
   currently-offending file to the **exact set of ``tests.X`` modules
   it is permitted to import**. A new module — even one added to a
   file already on the allowlist — fails the test. Importantly this
   means an allowlisted file does *not* receive a blank check; it can
   only continue to import the modules it was importing when the
   allowlist was created. The minimality check then refuses to let
   stale entries linger: when a file stops importing one of its
   allowlisted modules (or stops existing), the entry must be removed.

3. **End-to-end subprocess verification**
   (``test_run_amip_help_starts_without_tests_on_path`` and
   ``test_held_suarez_resolves_without_tests_on_path``): launches a
   fresh interpreter with the repo root scrubbed from ``PYTHONPATH``
   and confirms ``run_amip.py --help`` and the canonical Held-Suarez
   import both succeed without ``tests/`` on the path. This is the
   exact #188 user-visible failure.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# Carve-aware: every legoesm namespace root (src/legoesm today; also
# packages/<member>/src/legoesm after the carve), reported repo-relative so the
# scan + the parametrize IDs stay stable, plus scripts.
from tests.legoesm_paths import legoesm_root_paths as _legoesm_root_paths
# repo_root= keeps this working from a git WORKTREE: legoesm.__path__ reports
# the EDITABLE INSTALL's paths, which point at the canonical checkout, and the
# bare relative_to() below then raised ValueError naming a foreign path (#1389).
PRODUCTION_ROOTS = tuple(
    str(p.relative_to(REPO_ROOT)) for p in _legoesm_root_paths(REPO_ROOT)
) + ("scripts",)


# Per-file allowlist of production files (under src/legoesm/ and scripts/)
# that import from the tests/ tree, mapped to the EXACT tests.* modules each
# imports today.  Baseline reset 2026-06-30: re-measured against the current
# tree after the dycore iteration archives (scripts/tmp/dycore_iter_archive/,
# retained — referenced by tests/test_matrix_nh_cube_parity_ast_guard.py and
# the regression MANIFEST) and several production drivers were added without
# updating this list.  NOT a blank check: test_grandfather_list_is_minimal
# forces it to ratchet DOWN (any unused/stale entry fails).  The preferred
# long-term fix per issue #188 is to move the canonical IC/diagnostic harnesses
# (tests.test_cases.*, tests.williamson_diagnostic, tests.legoesm_paths) into
# the installed legoesm package so production code never imports from tests/.
# 2026-07-31 (#1389): 49 stale entries removed (files deleted from
# scripts/tmp/) and 8 live ones added below, all of them the SAME three IC
# modules (baroclinic_wave, williamson, colliding_modons) that the entries
# above already carry — net 62 -> 21 entries, so the list still ratchets DOWN.
# The real fix is unchanged and unblocked by these entries: move those three
# modules into the installed package.  52 files import them, so that migration
# is its own PR, not a rider on a red-main fix.
GRANDFATHERED_TESTS_IMPORTS_BY_FILE: dict[str, frozenset[str]] = {
    'scripts/bench/bench_cube_shardmap_halo.py': frozenset({
        'tests.test_cases.baroclinic_wave',
    }),
    'scripts/bench/bench_fv3_sw_fb_vs_production.py': frozenset({
        'tests.test_cases.williamson',
    }),
    'scripts/bench/bench_mpas_spmd_scaling.py': frozenset({
        'tests.test_cases.baroclinic_wave',
    }),
    'scripts/bench/run_cpu_mpi_scaling.py': frozenset({
        'tests.test_cases.baroclinic_wave',
    }),
    'scripts/validate/fv3_native/extvec_stage_dump.py': frozenset({
        'tests.test_cases.colliding_modons',
    }),
    'scripts/validate/fv3_native/run_duo_stepper_modon.py': frozenset({
        'tests.test_cases.colliding_modons',
    }),
    'scripts/validate/fv3_native/stage_ke_budget.py': frozenset({
        'tests.test_cases.colliding_modons',
    }),
    'scripts/validate/fv3_native/twin_block1_zoom.py': frozenset({
        'tests.test_cases.colliding_modons',
    }),
    'scripts/validate/fv3_native/wedge_sharp_compare.py': frozenset({
        'tests.test_cases.colliding_modons',
    }),
    'scripts/bench/run_levante_gpu_scaling.py': frozenset({
        'tests.test_cases.baroclinic_wave',
    }),
    'scripts/bench/run_scaling_diagnosis.py': frozenset({
        'tests.test_cases.baroclinic_wave',
    }),
    'scripts/matrix/check_conservation_all.py': frozenset({
        'tests.atmosphere.shallow_water.test_cases.williamson',
        'tests.atmosphere.shallow_water.test_cases.williamson_mpas',
    }),
    'scripts/matrix/run_atmosphere_test_matrix.py': frozenset({
        'tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1_mpas',
        'tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_2_mpas',
        'tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_3_mpas',
        'tests.atmosphere.shallow_water.test_cases.williamson_mpas',
        'tests.test_cases.baroclinic_wave',
        'tests.test_cases.colliding_modons',
        'tests.test_cases.cosine_bell',
        'tests.test_cases.dcmip2008',
        'tests.test_cases.dcmip2008.jablonowski_rotated',
        'tests.test_cases.dcmip2012.rest_state_topography',
        'tests.test_cases.dcmip2025',
        'tests.test_cases.dcmip_transport',
        'tests.test_cases.williamson',
        'tests.test_cases.williamson_extended',
    }),
    'scripts/run/run_baroclinic_wave_benchmark.py': frozenset({
        'tests.test_cases.baroclinic_wave',
    }),
    'scripts/run/run_w2_mpas_convergence.py': frozenset({
        'tests.atmosphere.shallow_water.test_cases.williamson_mpas',
    }),
    'scripts/run/run_w2_w5_cosine_bell_iter1030.py': frozenset({
        'tests.atmosphere.shallow_water.test_cases.williamson',
        'tests.test_cases.cosine_bell',
        'tests.test_iter921_w2_v_vs_h_pareto_sentinel',
    }),
    'scripts/validate/run_colliding_modons.py': frozenset({
        'tests.test_cases.colliding_modons',
    }),
    'scripts/validate/validate_cubed_sphere_fv3_atmos.py': frozenset({
        'tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1',
        'tests.test_cases.williamson',
    }),
    'scripts/validate/validate_tpu_emulation_sharding.py': frozenset({
        'tests.test_cases.baroclinic_wave',
    }),
    'scripts/validate/visual_regression.py': frozenset({
        'tests.williamson_diagnostic',
    }),
    'src/legoesm/cli.py': frozenset({
        'tests.test_cases.williamson',
    }),
}


def _walk_py_files(root: pathlib.Path):
    yield from root.rglob("*.py")


def _ast_find_tests_imports(source: str) -> list[tuple[int, str, str]]:
    """Return ``[(lineno, module, statement), ...]`` for every
    ``from tests.*`` or ``import tests.*`` in ``source``, including
    imports nested inside function bodies, class bodies, or
    conditional branches.

    ``module`` is the dotted module path (e.g.
    ``tests.test_cases.williamson``) — used for allowlist matching.
    ``statement`` is the human-readable form (e.g. ``from tests.X
    import a, b``) — used in failure messages.

    A plain regex on the source text would miss some cases (dynamically
    constructed import nodes, exec'd code), but more importantly
    AST-level scanning explicitly walks into function bodies so it
    catches lazy-imports that only fire when a method is called —
    which is the original #188 bug pattern.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "tests" or mod.startswith("tests."):
                names = ", ".join(a.name for a in node.names)
                hits.append((node.lineno, mod, f"from {mod} import {names}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tests" or alias.name.startswith("tests."):
                    hits.append((
                        node.lineno, alias.name, f"import {alias.name}"
                    ))
    return hits


def test_held_suarez_is_installed_package():
    """The canonical location for Held-Suarez is the installed package."""
    import legoesm.atmosphere.forcing.idealized.held_suarez as hs

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


def test_no_held_suarez_imports_from_tests_in_production():
    """Issue #188 narrow guard — production code must import
    Held-Suarez from ``legoesm.atmosphere.forcing.idealized.held_suarez``, never from
    ``tests.test_cases.held_suarez``.

    The shim has been removed, so any such import would also raise
    ``ModuleNotFoundError`` at parse time. This static scan runs at
    pytest collection (no JAX/dycore startup cost) and produces a
    clear, file-and-line failure message that names the canonical
    replacement.
    """
    offenders: list[str] = []
    for subdir in PRODUCTION_ROOTS:
        for path in _walk_py_files(REPO_ROOT / subdir):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, _module, stmt in _ast_find_tests_imports(text):
                if "tests.test_cases.held_suarez" in stmt:
                    rel = path.relative_to(REPO_ROOT)
                    offenders.append(f"  - {rel}:{lineno}: {stmt}")
    assert not offenders, (
        "Production code must import Held-Suarez from "
        "legoesm.atmosphere.forcing.idealized.held_suarez (issue #188):\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("subdir", PRODUCTION_ROOTS)
def test_no_new_tests_imports_in_production(subdir: str):
    """Generalised #188 guard — production code under
    ``src/legoesm/`` and ``scripts/`` must not import from the
    ``tests/`` tree, except for files explicitly listed in
    ``GRANDFATHERED_TESTS_IMPORTS_BY_FILE`` AND only for the modules
    each file is recorded as importing today.

    This catches the same anti-pattern that triggered #188 for any
    other test-case module (``williamson``, ``baroclinic_wave``,
    ``dcmip2025``, ...). Two failure modes:

    1. A file not in the allowlist that imports any ``tests.X``.
    2. A file in the allowlist that imports a ``tests.X`` not in
       its recorded set (i.e. someone added a new violation to an
       already-grandfathered file — the allowlist is per-module,
       not a blank check).
    """
    new_offenders: dict[str, list[tuple[int, str]]] = {}
    for path in _walk_py_files(REPO_ROOT / subdir):
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = _ast_find_tests_imports(text)
        if not hits:
            continue
        allowed = GRANDFATHERED_TESTS_IMPORTS_BY_FILE.get(rel, frozenset())
        for lineno, module, stmt in hits:
            if module not in allowed:
                new_offenders.setdefault(rel, []).append((lineno, stmt))
    assert not new_offenders, (
        "New production-code import from tests/ detected — this is the "
        "issue #188 anti-pattern. Either move the imported module into "
        "the installed legoesm package (preferred), or — only as a last "
        "resort — add the module to the file's frozenset in "
        "GRANDFATHERED_TESTS_IMPORTS_BY_FILE in this test file:\n"
        + "\n".join(
            f"  - {p}:\n"
            + "\n".join(f"      {ln}: {s}" for ln, s in hits)
            for p, hits in sorted(new_offenders.items())
        )
    )


def test_grandfather_list_is_minimal():
    """The allowlist must ratchet DOWN: if an allowlisted file no
    longer imports one (or any) of its allowlisted ``tests.*``
    modules, that entry is stale. Stale entries fail this test,
    forcing the tech debt to monotonically shrink rather than
    accumulate.

    Specifically detected:
      - File no longer exists (entry must be removed).
      - File exists but no longer imports any ``tests.*`` (entry
        must be removed).
      - File imports a strict subset of its allowed modules (the
        unused module names must be removed from the frozenset).
    """
    stale: list[str] = []
    for rel, allowed in sorted(GRANDFATHERED_TESTS_IMPORTS_BY_FILE.items()):
        path = REPO_ROOT / rel
        if not path.is_file():
            stale.append(f"  - {rel}: file no longer exists; remove entry")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        actual = {module for _, module, _ in _ast_find_tests_imports(text)}
        if not actual:
            stale.append(
                f"  - {rel}: no remaining tests.* imports; "
                "remove the file entry from "
                "GRANDFATHERED_TESTS_IMPORTS_BY_FILE"
            )
            continue
        unused = allowed - actual
        if unused:
            stale.append(
                f"  - {rel}: allowlisted but no longer imported: "
                + ", ".join(sorted(unused))
                + " — remove from this file's frozenset"
            )
    assert not stale, (
        "GRANDFATHERED_TESTS_IMPORTS_BY_FILE contains stale entries "
        "(files cleaned up but allowlist not updated):\n"
        + "\n".join(stale)
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
    script = REPO_ROOT / "scripts" / "run" / "run_amip.py"
    assert script.is_file(), f"script not found: {script}"
    result = _run_subprocess_without_repo_root(
        "import runpy, sys\n"
        f"sys.argv = [{str(script)!r}, '--help']\n"
        "try:\n"
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
    """Canonical Held-Suarez module must resolve in an installed env.

    Companion to ``test_run_amip_help_starts_without_tests_on_path``:
    that test verifies the script's full import closure is clean; this
    one verifies the specific submodule the closure depends on
    (``legoesm.atmosphere.forcing.idealized.held_suarez``) loads without any ``tests/``
    reference. If a future refactor moved Held-Suarez back into the
    tests tree (reversing the #188 fix), this test would fail with
    ``ModuleNotFoundError`` even before the script-level test ran.
    """
    # NOTE: do not assert ``find_spec('tests') is None`` here — an
    # unrelated site-packages ``tests`` package (which some CI images
    # ship) would make that precondition flaky. The real contract is
    # that the *canonical* Held-Suarez resolves without relying on
    # this repo's tests tree, which the import below verifies directly.
    result = _run_subprocess_without_repo_root(
        "from legoesm.atmosphere.forcing.idealized.held_suarez import (\n"
        "    held_suarez_init, held_suarez_init_latlon,\n"
        "    held_suarez_init_mpas, held_suarez_forcing_mpas,\n"
        "    K_A, K_S, SIGMA_B,\n"
        ")\n"
    )
    assert result.returncode == 0, (
        f"Canonical Held-Suarez import failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
