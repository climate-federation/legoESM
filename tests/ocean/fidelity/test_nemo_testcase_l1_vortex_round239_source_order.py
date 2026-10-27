"""Round-239 controls for candidate-bound 100-day scoring."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = (Path(__file__).parents[3] / "scripts" / "validate" /
         "ocean_fidelity" / "testcases")
sys.path.insert(0, str(TOOLS))

import nemo_testcase_l1_vortex_round210_100day_comparison as scorer  # noqa: E402


def _ladder(value: float) -> dict:
    return {
        "first_over_bar": {"kt": 2, "fields": ["T"]},
        "steps": [{
            "kt": kt,
            "rows": [{"name": f"kt{kt}.T", "normalized_max_abs": value,
                      "status": "DEBT"}],
        } for kt in range(1, 11)],
    }


def test_candidate_ladder_reference_is_checked_not_silently_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    reference = tmp_path / "candidate.json"
    reference.write_text(json.dumps(_ladder(2.0)))
    measured = _ladder(2.0)
    monkeypatch.setattr(scorer, "trajectory_run", lambda *args, **kwargs: measured)
    assert scorer.sanity_check_kt1_10(
        "smt4", "VORTEX_SMT4_VEC-zps", tmp_path,
        ladder_reference=reference)["status"] == "REPRODUCED"

    measured["steps"][4]["rows"][0]["normalized_max_abs"] = 3.0
    result = scorer.sanity_check_kt1_10(
        "smt4", "VORTEX_SMT4_VEC-zps", tmp_path,
        ladder_reference=reference)
    assert result["status"] == "MISMATCH"
    assert result["mismatches"] == [("kt5.T", 2.0, 3.0)]
