"""Round-348 fast-libm and arm-comparison controls."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from legoesm.core import transcendentals as tx

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round348_transcendentals_compare as compare,
)

_REAL_R342_MANIFEST = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/"
    "legoesm_year1_2920step/run_manifest.json"
)

_TESTCASE_SCRIPTS = (
    Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/testcases"
)
sys.path.insert(0, str(_TESTCASE_SCRIPTS))
try:
    import nemo_testcase_l2_gyre_round16_discriminator as r16_discriminator
finally:
    sys.path.pop(0)


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


def _assert_plant_changes_bits(clean: np.ndarray, planted: np.ndarray) -> None:
    word = np.uint32 if clean.dtype == np.float32 else np.uint64
    assert not np.array_equal(clean.view(word), planted.view(word))


def test_round16_routing_plant_changes_compiled_libm_result_and_noop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.linspace(-2.0, 2.0, 257, dtype=np.float64)
    clean = tx._scalar_libm("exp", values)
    poison = r16_discriminator._PoisonFastFunction(tx._fast_function)
    monkeypatch.setattr(tx, "_fast_function", poison)
    planted = tx._scalar_libm("exp", values)
    assert poison.counts["exp"] == values.size
    _assert_plant_changes_bits(clean, planted)
    with pytest.raises(AssertionError):
        _assert_plant_changes_bits(clean, clean.copy())


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
    monkeypatch.setattr(
        compare.pilot_metrics,
        "read_candidate_day",
        lambda root, _day: {"T": reference_t if root == Path("left") else candidate_t},
    )
    rows = compare.compare_daily_temperature(Path("left"), Path("right"), Path("geo"), 1)

    surface_weight = area * wet[..., 0]
    expected_sst = np.sqrt(3.0 * 4.0 / surface_weight.sum())
    expected_t3d = np.sqrt(4.0 / wet.astype(bool).sum())
    assert rows[0]["sst_area_weighted_rms_k"] == pytest.approx(expected_sst)
    assert rows[0]["t3d_wet_rms_k"] == pytest.approx(expected_t3d)
    control = compare.planted_daily_temperature_control(
        Path("left"), Path("right"), Path("geo"), rows[0]
    )
    assert control["daily_sst_rms_plant"] != rows[0]["sst_area_weighted_rms_k"]
    assert control["daily_t3d_rms_plant"] != rows[0]["t3d_wet_rms_k"]

    monkeypatch.setattr(compare, "compare_daily_temperature", lambda *_args, **_kwargs: rows)
    with pytest.raises(RuntimeError, match="plant did not move"):
        compare.planted_daily_temperature_control(
            Path("left"), Path("right"), Path("geo"), rows[0]
        )


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
    monkeypatch.setattr(
        compare,
        "planted_daily_temperature_control",
        lambda *_args: {"daily_sst_rms_plant": 1.0, "daily_t3d_rms_plant": 1.0},
    )
    monkeypatch.setattr(compare, "worktree_stamp", lambda **_kwargs: "test-clean")
    monkeypatch.setattr(
        compare,
        "compare_snapshot_data",
        lambda *_args: {"status": "EXACT", "snapshot_count": 2, "compared_arrays": 18},
    )
    reference_root = tmp_path / "libm"
    candidate_root = tmp_path / "native"
    reference_root.mkdir()
    candidate_root.mkdir()
    reference_manifest = reference_root / "run_manifest.json"
    candidate_manifest = candidate_root / "run_manifest.json"
    reference_manifest.write_text(
        json.dumps(
            {
                "steps": 2,
                "wall_seconds": 4.0,
                "jax_default_backend": "cpu",
                "transcendentals": "libm",
            }
        )
    )
    candidate_manifest.write_text(
        json.dumps(
            {
                "steps": 2,
                "wall_seconds": 2.0,
                "jax_default_backend": "cpu",
                "transcendentals": "native",
            }
        )
    )
    assert compare.main(
        [
            "--reference-root", str(reference_root),
            "--candidate-root", str(candidate_root),
            "--geometry-root", str(tmp_path / "geometry"),
            "--output-dir", str(tmp_path / "out"),
            "--days", "2",
            "--bitwise-steps", "8,10",
            "--require-bitwise-exact",
            "--reference-manifest", str(reference_manifest),
            "--candidate-manifest", str(candidate_manifest),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "PASS_R348_BITWISE_EXACT snapshots=2 arrays=18" in output
    assert "PASS_R348_ARM_COMPARISON days=2" in output
    assert "reference_backend=cpu reference_transcendentals=libm" in output
    assert "candidate_backend=cpu candidate_transcendentals=native" in output
    report_path = tmp_path / "out" / "transcendentals_arm_temperature_rms.json"
    report = json.loads(report_path.read_text())
    assert report["daily"] == rows
    assert (tmp_path / "out" / "transcendentals_arm_temperature_rms.png").is_file()


def test_step_parser_refuses_unknown_or_unsorted_values() -> None:
    assert compare.parse_steps("8,10,16") == (8, 10, 16)
    with pytest.raises(Exception, match="unique and increasing"):
        compare.parse_steps("10,8")


@pytest.mark.skipif(
    not _REAL_R342_MANIFEST.is_file(),
    reason="certified round-342 run manifest is not present on this host",
)
def test_manifest_timing_accepts_copy_of_real_pre_field_r342_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    reference_root.mkdir()
    candidate_root.mkdir()
    reference = reference_root / "run_manifest.json"
    candidate = candidate_root / "run_manifest.json"
    shutil.copyfile(_REAL_R342_MANIFEST, reference)
    reference_payload = json.loads(reference.read_text())
    candidate_payload = dict(reference_payload)
    candidate_payload.update(
        {"steps": 80, "wall_seconds": 800.0, "transcendentals": "native"}
    )
    candidate.write_text(json.dumps(candidate_payload))
    timing = compare.compare_manifest_timing(
        reference,
        candidate,
        reference_root=reference_root,
        candidate_root=candidate_root,
    )
    assert timing["reference_seconds_per_step"] == pytest.approx(
        reference_payload["wall_seconds"] / reference_payload["steps"]
    )
    assert timing["candidate_seconds_per_step"] == 10.0
    assert timing["speedup"] == pytest.approx(
        timing["reference_seconds_per_step"] / 10.0
    )
    assert timing["reference_backend"] == "cpu"
    assert timing["reference_transcendentals"] == "libm"
    assert timing["candidate_transcendentals"] == "native"
    assert (
        "rule=certified-pre-r348-round342-lane-used-libm"
        in capsys.readouterr().out
    )

    bad = json.loads(candidate.read_text())
    bad["jax_default_backend"] = "gpu"
    candidate.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="backend must be 'cpu'"):
        compare.compare_manifest_timing(
            reference,
            candidate,
            reference_root=reference_root,
            candidate_root=candidate_root,
        )


def test_manifest_timing_refuses_identical_paths_and_wrong_roots(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    wrong_root = tmp_path / "wrong"
    for root in (reference_root, candidate_root, wrong_root):
        root.mkdir()
    reference = reference_root / "run_manifest.json"
    candidate = candidate_root / "run_manifest.json"
    payload = {
        "status": "PASS_R342_S4_K0_LEGOESM_RUN",
        "round": "orca2_r342",
        "steps": 1,
        "wall_seconds": 1.0,
        "jax_default_backend": "cpu",
        "transcendentals": "libm",
    }
    reference.write_text(json.dumps(payload))
    candidate.write_text(json.dumps({**payload, "transcendentals": "native"}))
    with pytest.raises(RuntimeError, match="manifest paths must differ"):
        compare.compare_manifest_timing(
            reference,
            reference,
            reference_root=reference_root,
            candidate_root=reference_root,
        )
    with pytest.raises(RuntimeError, match="manifest must belong to run root"):
        compare.compare_manifest_timing(
            reference,
            candidate,
            reference_root=wrong_root,
            candidate_root=candidate_root,
        )


def test_comparison_refuses_identical_roots(tmp_path: Path) -> None:
    root = tmp_path / "same"
    with pytest.raises(RuntimeError, match="reference_root and candidate_root must differ"):
        compare.main(
            [
                "--reference-root", str(root),
                "--candidate-root", str(root),
                "--geometry-root", str(tmp_path / "geometry"),
                "--output-dir", str(tmp_path / "out"),
                "--days", "1",
            ]
        )
