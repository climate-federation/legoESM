"""The offline before/after comparison must be the SAME gate, and must fail.

Round 34 compares three arms of three cards.  Running the shared
``--compare-to`` path would mean re-running every AFTER arm a second time --
on OVERFLOW, half an hour of identical arithmetic -- because that path
compares the report a run has just produced.  Everything it needs is already
durable, so the offline driver reads two saved report/sidecar pairs and calls
the same ``compare_gate_reports``.

These arms prove it is not a second implementation with its own opinions: the
plants that must turn the shared gate red turn this red too, the sidecar hash
check still fails closed, and a report compared against itself is a PASS with
zero movement.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_offline_compare.py"
)
SPEC = importlib.util.spec_from_file_location("offline_compare", PATH)
assert SPEC and SPEC.loader
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)

ROUND34 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round34")
PAIR = (ROUND34 / "before_traj_LOCK_EXCHANGE-zco.json",
        ROUND34 / "a2_traj_LOCK_EXCHANGE-zco.json")


def _skip_unless_measured():
    for path in PAIR:
        if not path.is_file():
            pytest.skip(f"{path} has not been measured")


def test_a_report_compared_against_itself_moves_nothing(tmp_path):
    _skip_unless_measured()
    out = tmp_path / "self.json"
    code = driver.main([str(PAIR[0]), str(PAIR[0]), "--output", str(out)])
    result = json.loads(out.read_text())
    assert code == 0 and result["status"] == "PASS"
    assert result["worktree"] == json.loads(PAIR[0].read_text())["worktree"]
    assert result["largest_oracle_residual_worsening_ulps"] == 0
    assert (result["first_over_bar_reference"]
            == result["first_over_bar_candidate"])


@pytest.mark.parametrize("plant", ["worsen-3ulp", "at-bar-to-debt"])
def test_every_plant_the_shared_gate_refuses_is_refused_here(tmp_path, plant):
    _skip_unless_measured()
    out = tmp_path / f"{plant}.json"
    code = driver.main([str(PAIR[0]), str(PAIR[0]), "--plant", plant,
                        "--output", str(out)])
    assert code != 0, f"the {plant} plant did not turn the comparison red"
    assert json.loads(out.read_text())["status"] == "FAIL"


def test_an_improving_plant_still_passes(tmp_path):
    """The gate is one-sided: improvement is never a violation."""
    _skip_unless_measured()
    out = tmp_path / "improve.json"
    assert driver.main([str(PAIR[0]), str(PAIR[0]), "--plant", "improve",
                        "--output", str(out)]) == 0
    assert json.loads(out.read_text())["status"] == "PASS"


def test_a_sidecar_that_drifted_from_its_report_is_REFUSED(tmp_path):
    """Fail closed: the sidecar's SHA-256 is checked against the report."""
    _skip_unless_measured()
    report = json.loads(PAIR[0].read_text())
    broken = tmp_path / PAIR[0].name
    key = "oracle_relative_residual_artifact"
    report[key] = dict(report[key])
    report[key]["sha256"] = "0" * 64
    broken.write_text(json.dumps(report))
    (tmp_path / report[key]["path"]).write_bytes(
        (PAIR[0].with_suffix("").with_suffix("")
         .parent / report[key]["path"]).read_bytes())
    with pytest.raises(Exception):
        driver.main([str(broken), str(PAIR[1])])
