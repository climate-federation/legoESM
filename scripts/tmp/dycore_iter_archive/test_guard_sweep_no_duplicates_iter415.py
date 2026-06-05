"""FV3_3D iter 415: ensure iter-383 guard-sweep inventory has
no duplicate entries.

Bug-class catch: if a future extension to ``_GUARD_MODULES``
accidentally duplicates a module name, the importability test
passes (duplicates don't break import) but the sweep is
silently weaker.

Tests
-----

1. ``test_guard_sweep_no_duplicate_modules``
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fv3_fidelity_guard_sweep_iter383 import _GUARD_MODULES


def test_guard_sweep_no_duplicate_modules():
    seen = set()
    duplicates = []
    for mod in _GUARD_MODULES:
        if mod in seen:
            duplicates.append(mod)
        seen.add(mod)
    assert not duplicates, (
        f"iter-383 _GUARD_MODULES has duplicate entries: "
        f"{duplicates}"
    )
