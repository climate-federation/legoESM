"""Round-348 fast-libm and arm-comparison controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round348_transcendentals_compare as compare,
)


def _snapshot(path: Path, step: int, *, negative_zero: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "T": np.zeros((2, 3, 2), dtype=np.float64),
        "S": np.ones((2, 3, 2), dtype=np.float64),
        "u": np.zeros((2, 3, 2), dtype=np.float64),
        "v": np.zeros((2, 3, 2), dtype=np.float64),
        "ssh": np.zeros((2, 3), dtype=np.float64),
        "fr_i": np.zeros((2, 3, 2), dtype=np.float64),
        "ice_cover": np.zeros((2, 3), dtype=np.float64),
        "kt": np.asarray(step, dtype=np.int64),
        "day": np.asarray(step // 8, dtype=np.int64),
    }
    if negative_zero:
        arrays["T"][0, 0, 0] = -0.0
    np.savez(path / f"candidate_step{step:08d}.npz", **arrays)


def test_all_field_bitwise_comparison_and_signed_zero_policy(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for step in (8, 10):
        _snapshot(left, step)
        _snapshot(right, step)
    exact = compare.compare_snapshot_data(left, right, (8, 10))
    assert exact["status"] == "EXACT"
    assert exact["signed_zero_policy"] == "DISTINCT_RAW_BITS"
    assert exact["compared_arrays"] == 18
    assert set(exact["fields"]) == {
        "T", "S", "u", "v", "ssh", "fr_i", "ice_cover", "kt", "day"
    }

    _snapshot(right, 8, negative_zero=True)
    signed_zero = compare.compare_snapshot_data(left, right, (8, 10))
    assert signed_zero["status"] == "DIFFERENT"
    assert signed_zero["first_difference"]["field"] == "T"

    _snapshot(right, 8)
    planted = compare.compare_snapshot_data(
        left, right, (8, 10), plant="flip-first-bit"
    )
    assert planted["status"] == "DIFFERENT"


def test_daily_comparison_reuses_area_weights_and_wet_3d_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (148, 180, 30)
    wet = np.ones(shape, dtype=np.float64)
    wet[0, 0, :] = 0.0
    area = np.ones(shape[:2], dtype=np.float64)
    area[0, 1] = 3.0
    reference_t = np.zeros(shape, dtype=np.float64)
    candidate_t = np.zeros(shape, dtype=np.float64)
    candidate_t[0, 0, :] = 99.0  # dry and must not enter either metric
    candidate_t[0, 1, 0] = 2.0

    monkeypatch.setattr(
        compare.pilot_metrics,
        "static_geometry",
        lambda _root: {"tmask": wet, "area_t": area},
    )
    calls = iter(({"T": reference_t}, {"T": candidate_t}))
    monkeypatch.setattr(compare.pilot_metrics, "read_candidate_day", lambda *_: next(calls))
    rows = compare.compare_daily_temperature(Path("left"), Path("right"), Path("geo"), 1)

    surface_weight = area * wet[..., 0]
    expected_sst = np.sqrt(3.0 * 4.0 / surface_weight.sum())
    expected_t3d = np.sqrt(4.0 / wet.astype(bool).sum())
    assert rows[0]["sst_area_weighted_rms_k"] == pytest.approx(expected_sst)
    assert rows[0]["t3d_wet_rms_k"] == pytest.approx(expected_t3d)


def test_metric_controls_fire_and_json_png_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    controls = compare.planted_metric_controls()
    assert controls["area_weighted_sst_rms_plant"] != controls[
        "unweighted_sst_rms_control"
    ]
    assert controls["wet_t3d_rms_plant"] != controls[
        "all_cell_t3d_rms_control"
    ]

    rows = [
        {"day": 1, "kt": 8, "sst_area_weighted_rms_k": 0.0, "t3d_wet_rms_k": 0.0},
        {"day": 2, "kt": 16, "sst_area_weighted_rms_k": 1.0e-12, "t3d_wet_rms_k": 2.0e-12},
    ]
    monkeypatch.setattr(compare, "compare_daily_temperature", lambda *_args: rows)
    monkeypatch.setattr(compare, "planted_metric_controls", lambda: controls)
    monkeypatch.setattr(compare, "worktree_stamp", lambda **_kwargs: "test-clean")
    monkeypatch.setattr(
        compare,
        "compare_snapshot_data",
        lambda *_args: {"status": "EXACT", "snapshot_count": 2, "compared_arrays": 18},
    )
    assert compare.main(
        [
            "--reference-root", str(tmp_path / "libm"),
            "--candidate-root", str(tmp_path / "native"),
            "--geometry-root", str(tmp_path / "geometry"),
            "--output-dir", str(tmp_path / "out"),
            "--days", "2",
            "--bitwise-steps", "8,10",
            "--require-bitwise-exact",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "PASS_R348_BITWISE_EXACT snapshots=2 arrays=18" in output
    assert "PASS_R348_ARM_COMPARISON days=2" in output
    report_path = tmp_path / "out" / "transcendentals_arm_temperature_rms.json"
    report = json.loads(report_path.read_text())
    assert report["daily"] == rows
    assert (tmp_path / "out" / "transcendentals_arm_temperature_rms.png").is_file()


def test_step_parser_refuses_unknown_or_unsorted_values() -> None:
    assert compare.parse_steps("8,10,16") == (8, 10, 16)
    with pytest.raises(Exception, match="unique and increasing"):
        compare.parse_steps("10,8")


def test_manifest_timing_uses_total_wall_seconds_per_completed_step(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(json.dumps({"steps": 2920, "wall_seconds": 70284.4}))
    candidate.write_text(json.dumps({"steps": 80, "wall_seconds": 800.0}))
    timing = compare.compare_manifest_timing(reference, candidate)
    assert timing["reference_seconds_per_step"] == pytest.approx(24.07)
    assert timing["candidate_seconds_per_step"] == 10.0
    assert timing["speedup"] == pytest.approx(2.407)
