"""Controls for the round-60 downstream UP3 walk."""

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round60_downstream_walk_gate as GATE  # noqa: E402


def test_direction_classifier_names_only_strict_aggregate_directions():
    oracle = np.zeros(3, dtype=np.float64)
    base = np.array([3.0, 2.0, 1.0])
    toward = GATE.classify_direction(base, np.array([2.0, 1.0, 0.5]), oracle)
    away = GATE.classify_direction(base, np.array([4.0, 3.0, 2.0]), oracle)
    mixed = GATE.classify_direction(base, np.array([2.0, 3.0, 1.0]), oracle)
    assert toward["direction"] == "TOWARD" and toward["n_toward"] == 3
    assert away["direction"] == "AWAY" and away["n_away"] == 3
    assert mixed["direction"] == "MIXED"


def test_source_order_keeps_unmeasured_stage3_raw_boundary_explicit():
    assert GATE.SOURCE_ORDER == (
        "kt3.entry.u", "s2.after_adv.u", "s2.pre_zdf.u",
        "s2.raw_kaa.u", "s2.postbar_kaa.u", "s3.after_adv.u",
        "s3.after_ldf.u", "s3.pre_zdf.u", "s3.raw_kaa.u",
        "s3.postbar_kaa.u", "kt4.entry.u",
    )


def test_sidecar_hash_and_field_order_are_fail_closed(tmp_path):
    report_path = tmp_path / "walk.json"
    meta = GATE._write_sidecar(report_path, {"a": np.arange(3.0)})
    report = {"sidecar": meta}
    assert list(GATE._read_sidecar(report)) == ["a"]
    with Path(meta["path"]).open("ab") as handle:
        handle.write(b"plant")
    try:
        GATE._read_sidecar(report)
    except RuntimeError as error:
        assert "hash drift" in str(error)
    else:
        raise AssertionError("corrupt sidecar was admitted")
