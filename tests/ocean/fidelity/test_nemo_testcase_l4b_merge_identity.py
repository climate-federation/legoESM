import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]
SCRIPT = (
    ROOT / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l4b_merge_identity.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("l4b_merge_identity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_case_census_covers_both_certified_lanes_exactly():
    probe = _module()
    assert set(probe.LANE_A_CASES) == {
        *(f"ORCA2G{i}-{suffix}" for i, suffix in enumerate(
            ("flat", "zps", "ts", "sbc", "ice", "vec", "drg", "ldf", "tadv"),
            start=1)),
        "ORCA2L4-bbc",
        "ORCA2L4b-iso-bbc",
        "ORCA2L6-dmp",
    }
    assert set(probe.LANE_B_CASES) == {
        "ORCA2V1-tke", "ORCA2V2-evd", "ORCA2V3-ddm", "ORCA2V4-iwm",
        "ORCA2V5-havtb", "ORCA2L1-bbl", "ORCA2L2-iso", "ORCA2L3-eiv",
    }
    assert set(probe.LANE_A_CASES).isdisjoint(probe.LANE_B_CASES)
    assert set(probe.ROOTS) == set(probe.LANE_A_CASES) | set(probe.LANE_B_CASES)


def test_compare_is_exact_and_one_bit_plant_fires(tmp_path):
    probe = _module()
    snapshot = {
        "case": "ORCA2G1-flat",
        "git_sha": "a" * 40,
        "step_program": {
            "sha256": "c" * 64, "nbytes": 10, "lines": 1,
        },
        "leaves": {
            "kt1:.T": {
                "dtype": "<f8", "shape": [1], "nbytes": 8,
                "sha256": "b" * 64,
            }
        },
    }
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    text = json.dumps(snapshot)
    reference.write_text(text)
    candidate.write_text(text)

    report = probe.compare(reference, candidate, plant=False)
    assert report["bit_identical"] is True
    assert report["leaf_count"] == 1
    assert report["different_count"] == 0
    assert report["step_program_identical"] is True
    with pytest.raises(RuntimeError, match='"different_count": 1'):
        probe.compare(reference, candidate, plant=True)
    changed = json.loads(text)
    changed["step_program"]["sha256"] = "d" * 64
    candidate.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match='"step_program_identical": false'):
        probe.compare(reference, candidate, plant=False)
