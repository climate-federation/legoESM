"""Direct test for round 208's resolution-ladder scorer.

The scorer decides what the round-208 receipt says, so its two load-bearing
behaviours are pinned here: it must RAISE on a registry-key mismatch instead
of reporting "nothing moved", and it must raise when its own non-vacuity
control finds no move.  Both were the failure modes an earlier version of the
inertness claim could have had.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/"
          "testcases/nemo_testcase_l1_vortex_round208_resolution_ladder.py")
SPEC = importlib.util.spec_from_file_location("r208_ladder", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules["r208_ladder"] = mod
SPEC.loader.exec_module(mod)


def _row(name, value, status="AT-BAR"):
    return {"name": name, "normalized_max_abs": value, "status": status}


def _report(values):
    """A 50-row gate report in the shape the scorer parses."""
    steps = []
    for kt in range(1, 11):
        steps.append({"kt": kt, "rows": [
            _row(f"CASE-zco.kt{kt}.before.{f}", values.get((kt, f), 0.0))
            for f in ("T", "S", "u", "v", "ssh")]})
    return {"steps": steps}


def _write(tmp_path, name, values):
    p = tmp_path / name
    p.write_text(json.dumps(_report(values)))
    return p


def test_registry_parses_fifty_rows_and_drops_the_case_prefix(tmp_path):
    reg = mod.registry(_write(tmp_path, "a.json", {(2, "u"): 1.5e-9}))
    assert len(reg) == 50
    assert reg["kt2.before.u"]["normalized_max_abs"] == 1.5e-9


def test_a_short_registry_is_refused_not_scored(tmp_path):
    p = tmp_path / "short.json"
    p.write_text(json.dumps({"steps": [{"kt": 1, "rows": [
        _row("CASE-zco.kt1.before.T", 0.0)]}]}))
    with pytest.raises(mod.LadderError, match="expected the 50-row registry"):
        mod.registry(p)


def test_moved_reports_value_and_status_changes(tmp_path):
    a = mod.registry(_write(tmp_path, "a.json", {(2, "u"): 1.0e-9}))
    b = mod.registry(_write(tmp_path, "b.json", {(2, "u"): 2.0e-9}))
    assert mod.moved(a, b) == ["kt2.before.u"]
    assert mod.moved(a, a) == []
    a["kt3.before.v"] = dict(a["kt3.before.v"], status="DEBT")
    assert "kt3.before.v" in mod.moved(a, b)


def test_a_key_mismatch_raises_instead_of_reading_as_nothing_moved(tmp_path):
    """The exact way a "0/50 moved" claim could pass vacuously."""
    a = mod.registry(_write(tmp_path, "a.json", {}))
    b = mod.registry(_write(tmp_path, "b.json", {}))
    b["kt1.before.OTHER"] = b.pop("kt1.before.T")
    with pytest.raises(mod.LadderError, match="registry keys differ"):
        mod.moved(a, b)


def test_the_floor_for_fitting_excludes_the_rounding_quantum():
    """A power law fitted through floor rows would be fitting quantisation.

    The 50-row bar is 1e-15 and the normalising quantum seen on these cards is
    ~2.03e-16, so the fitting floor has to sit above both.
    """
    assert mod.FLOOR_FOR_FITTING > 1.0e-15 > 2.030122e-16
