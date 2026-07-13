"""Static import-resolution guard for the compare-reanalysis HPC CLIs.

Their ``main()`` (and several helper) functions do FUNCTION-SCOPE imports of
``legoesm.*`` / cross-``scripts.*`` symbols and are ``# pragma: no cover`` (heavy
I/O), so a stale import — a symbol that a module no longer exports, or a wrong
module path — is INVISIBLE until an HPC launch crashes at import time.  That is
exactly the bug iter 103 found in ``run_perfect_model_osse`` (``from
rce_diagnostics import run_forced_les`` — ``run_forced_les`` lives in
``column_les``, not ``rce_diagnostics``).

This test parses each CLI's AST, collects every ABSOLUTE ``from X import Y``
(module-level AND function-scope), and asserts each ``Y`` resolves on ``X`` — WITHOUT running
the heavy code paths.  In-repo modules (``legoesm.*`` / ``scripts.*``) MUST both
import and expose the symbol (a wrong path or a stale symbol fails LOUDLY);
optional third-party deps that are simply absent in the test env (e.g. ``mpi4py``)
are skipped, not failed.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# The compare-reanalysis entry points (the HPC user's run/validate CLIs).
_CLIS = (
    "scripts/run/run_correction_campaign.py",
    "scripts/run/run_column_les.py",
    "scripts/validate/run_perfect_model_osse.py",
    "scripts/validate/compare_amip_era5.py",
    "scripts/experiment/check_cross_grid_deploy.py",
    "scripts/experiment/check_real_era5_full_loop.py",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _from_imports(path: Path):
    """Every absolute ``from X import Y`` (module-level + nested) in the file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                yield node.lineno, node.module, alias.name


def _collect():
    cases = []
    for rel in _CLIS:
        p = _REPO_ROOT / rel
        for lineno, module, name in _from_imports(p):
            cases.append(pytest.param(rel, lineno, module, name,
                                      id=f"{rel}:{lineno}:{module}.{name}"))
    return cases


def _resolves_attr(module, name) -> bool:
    """Does ``name`` resolve on ``module`` — as an attribute OR a submodule (e.g.
    ``from mpi4py import MPI`` imports the MPI SUBMODULE, not an attribute of the
    package until imported)?  A non-package module (no ``__path__``) cannot have a
    submodule, so the symbol must be an attribute or it does not resolve.

    For the submodule case actually IMPORT it (not just ``find_spec``) so a
    discoverable-but-broken submodule — whose body raises — is NOT reported as
    resolving when ``from pkg import name`` would crash at runtime (Codex iter 104).
    A ModuleNotFoundError naming the candidate (or an ancestor) means the submodule
    is genuinely absent → unresolved; an UNRELATED missing module is a transitive
    dependency of an existing submodule → resolved (an env issue, not a stale import).
    """
    if hasattr(module, name):
        return True
    if not hasattr(module, "__path__"):          # a module, not a package
        return False
    candidate = f"{module.__name__}.{name}"
    try:
        importlib.import_module(candidate)
        return True
    except ModuleNotFoundError as exc:
        absent = bool(exc.name) and (
            exc.name == candidate or candidate.startswith(exc.name + "."))
        return not absent


@pytest.mark.parametrize("rel, lineno, module_name, symbol", _collect())
def test_cli_from_imports_resolve(rel, lineno, module_name, symbol):
    """Each ``from <module> import <symbol>`` in the CLI resolves (iter 104)."""
    in_repo = module_name.split(".")[0] in ("legoesm", "scripts")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if in_repo:
            pytest.fail(
                f"{rel}:{lineno}: in-repo module {module_name!r} does not import "
                f"({exc}) — a wrong module path (HPC launch would crash).")
        pytest.skip(f"optional dependency {module_name!r} absent: {exc}")
        return
    assert _resolves_attr(module, symbol), (
        f"{rel}:{lineno}: {module_name!r} does not export {symbol!r} — a stale "
        "import that would crash the CLI at launch (cf. the iter-103 OSSE bug).")


# The compare-reanalysis WORKFLOW CLIs the operator runs AS SCRIPTS (``python scripts/.../X.py``,
# per the sbatch + runbook), NOT via ``-m`` / pytest. A direct script run puts the script's OWN
# dir on sys.path, not the repo root, so a ``from scripts.* import`` needs an in-file sys.path
# bootstrap. The AST guard above runs UNDER pytest (repo root already on path) so it CANNOT catch
# a missing bootstrap — only a subprocess replicating the script invocation can (iter 321: the
# campaign + OSSE were MISSING the bootstrap that smoke + deploy-check had → ModuleNotFoundError).
_SCRIPT_CLIS = (
    "scripts/run/run_correction_campaign.py",
    "scripts/validate/run_perfect_model_osse.py",
    "scripts/experiment/smoke_compare_reanalysis.py",
    "scripts/experiment/check_campaign_deploy.py",
    "scripts/experiment/check_cross_grid_deploy.py",
    "scripts/experiment/check_real_era5_full_loop.py",
)


@pytest.mark.parametrize("rel", _SCRIPT_CLIS)
def test_workflow_cli_runs_as_a_script(rel):
    """Run each workflow CLI AS A SCRIPT (``python scripts/.../X.py --help``) in a CLEAN env (no
    repo root on PYTHONPATH), so ONLY the in-file sys.path bootstrap can make ``scripts.*``
    resolvable — exactly the iter-321 production-launch bug the AST test (pytest path) masks.
    ``--help`` exits 0 only if the module loads + the ``from scripts...`` import resolves (no
    ``ModuleNotFoundError: No module named 'scripts'``)."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, rel, "--help"], cwd=str(_REPO_ROOT),
                       env=env, capture_output=True, text=True, timeout=180)
    assert "No module named 'scripts'" not in r.stderr, (
        f"{rel} run as a script cannot resolve `scripts.*` — add a sys.path bootstrap "
        f"(parents[2] → repo root) BEFORE the `from scripts...` import:\n{r.stderr[-500:]}")
    assert r.returncode == 0, f"{rel} --help failed (rc={r.returncode}):\n{r.stderr[-500:]}"


def test_guard_detects_a_stale_import():
    """Non-vacuity: the resolver REJECTS a name a module does not export, so the
    guard above genuinely catches a regression rather than passing blindly. (The
    iter-103 OSSE bug was ``from rce_diagnostics import run_forced_les`` — it lives
    in ``column_les``.) A synthetic sentinel name is used for the negative checks so
    a future legitimate re-export cannot cause maintenance churn."""
    column_les = importlib.import_module(
        "legoesm.atmosphere.dynamics.les.column_les")
    assert _resolves_attr(column_les, "run_forced_les")           # a real attribute
    # rejection on a non-package MODULE (the OSSE class — column_les is a .py module).
    assert not _resolves_attr(column_les, "_no_such_symbol_xyzzy")
    # rejection on a PACKAGE (the submodule branch): importlib is a stdlib package.
    assert not _resolves_attr(importlib, "_no_such_submodule_xyzzy")
