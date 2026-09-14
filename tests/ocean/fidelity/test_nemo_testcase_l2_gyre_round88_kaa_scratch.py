"""Direct controls for the Round-88 carried-Kaa proof."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
          / "nemo_testcase_l2_gyre_round88_kaa_scratch.py")
SPEC = importlib.util.spec_from_file_location("round88_kaa_scratch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROUND88 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUND88)


def test_confirmation_requires_every_registered_boundary_to_be_bit_exact():
    rows = {"scratch": {"bit_exact": True}, "w": {"bit_exact": True}}
    assert ROUND88._confirmed(rows)
    rows["w"]["bit_exact"] = False
    assert not ROUND88._confirmed(rows)


def test_empty_boundary_set_cannot_confirm_vacuously():
    assert not ROUND88._confirmed({})
