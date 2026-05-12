"""FV3_3D iter 535: smoke test for example_fv3_clip_helper script.

Verifies the user-facing example script in ``scripts/`` imports
without error.  Does not run the full demo (that takes ~60s).

Tests
-----

1. ``test_example_script_importable``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_example_script_importable():
    """Load scripts/example_fv3_clip_helper.py as a module."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "example_fv3_clip_helper.py"
    assert script_path.exists(), f"example script missing: {script_path}"
    spec = importlib.util.spec_from_file_location(
        "example_fv3_clip_helper", str(script_path),
    )
    mod = importlib.util.module_from_spec(spec)
    # Run the module loader (just import, don't call main())
    spec.loader.exec_module(mod)
    # main + build_sbr_state should be exposed
    assert callable(mod.main)
    assert callable(mod.build_sbr_state)
