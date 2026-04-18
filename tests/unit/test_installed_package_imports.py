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

import pathlib
import re

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
