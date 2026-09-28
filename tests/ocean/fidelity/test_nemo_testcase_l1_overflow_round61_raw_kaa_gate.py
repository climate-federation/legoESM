"""Controls for round 61's completed OVERFLOW stage-boundary walk."""

import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round61_raw_kaa_gate as GATE  # noqa: E402


def _report(tmp_path, name, arrays, commit):
    path = tmp_path / f"{name}.json"
    report = {
        "format": GATE.FORMAT,
        "worktree": {"commit": commit},
        "sidecar": GATE.R60._write_sidecar(path, arrays),
    }
    if name == "base":
        path.write_text(json.dumps(report))
    return path, report


def _synthetic_arrays(raw_candidate):
    oracle = np.zeros(3, dtype=np.float64)
    base = {}
    candidate = {}
    for boundary in GATE.SOURCE_ORDER:
        base[f"oracle::{boundary}"] = oracle
        candidate[f"oracle::{boundary}"] = oracle
        base[boundary] = np.array([3.0, 2.0, 1.0])
        if boundary == "kt3.entry.u":
            candidate[boundary] = base[boundary].copy()
        elif boundary in ("s2.after_adv.u", "s3.after_adv.u",
                          "s3.after_ldf.u", "s3.pre_zdf.u"):
            candidate[boundary] = np.array([2.0, 1.0, 0.5])
        elif boundary == "s3.raw_kaa.u":
            candidate[boundary] = raw_candidate
        else:
            candidate[boundary] = np.array([2.0, 3.0, 1.0])
    return base, candidate


def test_raw_kaa_away_is_the_strict_compensating_owner(tmp_path):
    base, candidate = _synthetic_arrays(np.array([4.0, 3.0, 2.0]))
    base_path, _ = _report(tmp_path, "base", base, "base")
    _, candidate_report = _report(tmp_path, "candidate", candidate, "arm")
    result = GATE.compare(base_path, candidate_report)
    assert result["first_direction_reversal"]["name"] == "s3.raw_kaa.u"
    assert result["compensating_owner"] == "s3.raw_kaa.u"


def test_raw_kaa_toward_then_mixed_does_not_invent_an_owner(tmp_path):
    base, candidate = _synthetic_arrays(np.array([2.0, 1.0, 0.5]))
    base_path, _ = _report(tmp_path, "base", base, "base")
    _, candidate_report = _report(tmp_path, "candidate", candidate, "arm")
    result = GATE.compare(base_path, candidate_report)
    assert result["first_direction_reversal"] is None
    assert result["compensating_owner"] is None


def test_round61_has_no_unmeasured_source_boundary():
    assert "s3.raw_kaa.u" in GATE.SOURCE_ORDER
    assert len(GATE.SOURCE_ORDER) == 11
