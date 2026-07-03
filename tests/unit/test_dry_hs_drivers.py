"""Smoke tests for the dry Held-Suarez driver scripts.

These ``scripts/run/run_dry_held_suarez_*.py`` drivers isolate the dycore
(no physics) on each grid to diagnose wind-collapse vs. parameterization
issues.  A full run is too heavy for unit testing, so we pin the lightweight
contract every new ``scripts/run`` driver must satisfy:

* the module imports cleanly on the current tree (all top-level
  ``from legoesm...`` imports resolve), and
* it exposes a ``main()`` whose argument parser builds and responds to
  ``--help`` with a clean ``SystemExit(0)`` (i.e. the CLI is wired, not a stub).
"""

from __future__ import annotations

import importlib

import pytest

_DRIVERS = [
    "scripts.run.run_dry_held_suarez_cube",
    "scripts.run.run_dry_held_suarez_latlon",
    "scripts.run.run_dry_held_suarez_mpas",
    "scripts.run.run_dry_held_suarez_spectral",
]


@pytest.mark.parametrize("module_name", _DRIVERS)
def test_driver_imports_and_exposes_main(module_name):
    mod = importlib.import_module(module_name)
    assert callable(getattr(mod, "main", None)), f"{module_name} has no main()"


@pytest.mark.parametrize("module_name", _DRIVERS)
def test_driver_help_builds_parser(module_name, monkeypatch):
    mod = importlib.import_module(module_name)
    monkeypatch.setattr("sys.argv", [module_name, "--help"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    # argparse exits 0 on --help; anything else means the parser is broken.
    assert exc.value.code == 0
