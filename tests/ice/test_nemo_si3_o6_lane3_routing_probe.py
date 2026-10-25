"""Direct tests for the lane-3 routing probe's row diff."""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROBE = (
    Path(__file__).parents[2]
    / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o6_lane3_routing_probe.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_o6_lane3_routing_probe_test", PROBE)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _report(v_i_status="AT-BAR", v_i_max=1e-16):
    return {
        "worktree": {"commit": "abc"},
        "rows": [
            {"name": "step1.a_i", "status": "AT-BAR", "max_abs": 0.0},
            {"name": "step1.v_i", "status": v_i_status, "max_abs": v_i_max},
        ],
        "first_divergence": {
            "owner": {"name": "step2.stress1_i", "status": "DEBT", "max_abs": 3e-13}
        },
    }


def test_identical_reports_have_no_change_even_with_different_stamps():
    a, b = _report(), _report()
    b["worktree"] = {"commit": "def"}
    d = probe.diff_rows(probe.rows_of(a), probe.rows_of(b))
    assert d["rows"] == 3 and d["rows_with_any_change"] == 0 and d["rows_missing_or_added"] == []


def test_number_and_status_changes_are_reported_in_both_directions():
    base = probe.rows_of(_report())
    worse = probe.diff_rows(base, probe.rows_of(_report("DEBT", 1e-12)))
    assert worse["rows_with_any_change"] == 1 and len(worse["status_worsened"]) == 1
    assert worse["status_improved"] == []
    better = probe.diff_rows(probe.rows_of(_report("DEBT", 1e-12)), base)
    assert len(better["status_improved"]) == 1 and better["status_worsened"] == []
    quiet = probe.diff_rows(base, probe.rows_of(_report("AT-BAR", 2e-16)))
    assert (
        quiet["rows_with_any_change"] == 1
        and not quiet["status_worsened"]
        and not quiet["status_improved"]
    )


def test_counted_counts_calls():
    count = {}
    f = probe.counted(lambda x: x + 1, count, "k")
    assert f(1) == 2 and f(2) == 3 and count == {"k": 2}


def _summary(changes, calls=1, cor_calls=1, plant_changes=None):
    s = {
        "lane3": {
            "g": {"o6_kernel_calls": {}, "rows_with_any_change": 0, "rows_missing_or_added": []}
        }
    }
    s["o6_rdgrft"] = {
        "g": {
            "o6_kernel_calls": {"o6_rdgrft": calls},
            "rows_with_any_change": changes,
            "rows_missing_or_added": [],
        }
    }
    s["o6_rdgrft_cor"] = {
        "g": {
            "o6_kernel_calls": {"o6_rdgrft": calls, "o6_cor": cor_calls},
            "rows_with_any_change": 0,
            "rows_missing_or_added": [],
        }
    }
    if plant_changes is not None:
        s["o6_rdgrft_planted"] = {
            "g": {
                "o6_kernel_calls": {"o6_rdgrft": calls},
                "rows_with_any_change": plant_changes,
                "rows_missing_or_added": [],
            }
        }
    return {"plant": plant_changes is not None, "summary": s}


def test_verdict_requires_reach_no_change_and_a_binding_plant():
    assert probe.verdict(_summary(0))
    assert not probe.verdict(_summary(1))  # a row moved
    assert not probe.verdict(_summary(0, calls=0))  # patch never reached the gate
    assert not probe.verdict(_summary(0, cor_calls=0))
    assert probe.verdict(_summary(0, plant_changes=3))
    assert not probe.verdict(_summary(0, plant_changes=0))  # plant did not bind
