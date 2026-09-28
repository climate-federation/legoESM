"""Controls for the round-67 pinned ORCA2 month-record gate."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_round67_month_record_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round67_month_record", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, name: str, files: tuple[str, ...]) -> None:
    (root / name).write_text("".join(f"{_sha(root / file)}  {file}\n" for file in files))


def _restart(path: Path, *, kt: int, ocean: bool) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 2)
        dataset.createVariable("kt", "f8").assignValue(float(kt))
        if ocean:
            for offset, name in enumerate(gate.RESTART_FIELDS):
                variable = dataset.createVariable(name, "f8", ("x",))
                variable[:] = np.array([offset, offset + 0.5], dtype=np.float64)
        else:
            dataset.createVariable("ice", "f8", ("x",))[:] = (1.0, 2.0)


def _namelist(step: int) -> str:
    return (
        "&namrun\n"
        "   nn_it000 = 1\n"
        f"   nn_itend = {step}\n"
        f"   nn_stock = {step}\n"
        "   rn_dt = 10800.0\n"
        "/\n"
    )


def _logs(root: Path, step: int) -> None:
    (root / "run.user.stdout.log").write_text("STOP 0\n")
    (root / "run.user.time.log").write_text("RUN_DONE\n")
    (root / "ocean.output").write_text(
        f"number of the last time step    nn_itend = {step}\n"
        f"frequency of restart file       nn_stock = {step}\n"
        "number of ice  categories       jpl = 1\n"
        "start from rest\n"
        "No icebergs used\n"
    )


@pytest.fixture
def roots(tmp_path, monkeypatch):
    pinned = tmp_path / "pinned"
    calibration = tmp_path / "calibration"
    month = tmp_path / "month"
    for root in (pinned, calibration, month):
        root.mkdir()

    for root, producer in ((calibration, "calibration-commit"), (month, "month-commit")):
        (root / "nemo").write_bytes(b"scalar-math-nemo")
        (root / "producer_commit.txt").write_text(f"{producer}\n")
        (root / "input.dat").write_bytes(b"input")
        _manifest(root, "input_files.sha256", ("input.dat",))
        (root / "namelist_ice_cfg").write_text("&nampar\n jpl=1\n/\n")
        (root / "fixed.deck").write_bytes(b"fixed")

    (calibration / "namelist_cfg").write_text(_namelist(10))
    (month / "namelist_cfg").write_text(_namelist(240))
    for root in (calibration, month):
        _manifest(
            root,
            "deck_files.sha256",
            ("fixed.deck", "namelist_cfg", "namelist_ice_cfg"),
        )

    for root in (pinned, calibration):
        for suffix in ("", "_ice"):
            for rank in (0, 1):
                _restart(
                    root / f"ORCA2_00000010_restart{suffix}_{rank:04d}.nc",
                    kt=10,
                    ocean=not suffix,
                )
    for suffix in ("", "_ice"):
        for rank in (0, 1):
            _restart(
                month / f"ORCA2_00000240_restart{suffix}_{rank:04d}.nc",
                kt=240,
                ocean=not suffix,
            )
    _logs(calibration, 10)
    _logs(month, 240)

    monkeypatch.setattr(gate, "EXPECTED_BINARY_SHA256", _sha(calibration / "nemo"))
    monkeypatch.setattr(
        gate, "EXPECTED_PINNED_DECK_SHA256", _sha(calibration / "deck_files.sha256")
    )
    monkeypatch.setattr(
        gate, "EXPECTED_INPUT_SHA256", _sha(calibration / "input_files.sha256")
    )
    monkeypatch.setattr(
        gate, "EXPECTED_ICE_CFG_SHA256", _sha(calibration / "namelist_ice_cfg")
    )
    return pinned, calibration, month


def test_admits_calibrated_month_record(roots):
    pinned, calibration, month = roots
    report = gate.validate_record(
        pinned,
        calibration,
        month,
        expect_calibration_commit="calibration-commit",
        expect_month_commit="month-commit",
    )
    assert report["status"] == "PASS_MONTH_RECORD"
    assert report["calibration"]["status"] == "BIT_EXACT"
    assert report["month"]["status"] == "FINITE_FP64"
    assert report["deck_delta"]["changed_assignments"] == [
        "namrun.nn_itend",
        "namrun.nn_stock",
    ]


@pytest.mark.parametrize(
    "plant", ("calibration-ulp", "missing-shard", "hidden-deck-delta")
)
def test_plants_refuse(roots, plant):
    pinned, calibration, month = roots
    with pytest.raises((gate.GateError, gate.phase1.GateError)):
        gate.validate_record(
            pinned,
            calibration,
            month,
            expect_calibration_commit="calibration-commit",
            expect_month_commit="month-commit",
            plant=plant,
        )


def test_calibration_mode_does_not_require_month(roots):
    pinned, calibration, _ = roots
    report = gate.validate_record(
        pinned,
        calibration,
        None,
        expect_calibration_commit="calibration-commit",
        mode="calibration",
    )
    assert report["status"] == "PASS_MONTH_CALIBRATION"


def test_nonfinite_ocean_payload_refuses(roots):
    pinned, calibration, month = roots
    path = month / "ORCA2_00000240_restart_0000.nc"
    with Dataset(path, "r+") as dataset:
        dataset["tn"][0] = np.nan
    with pytest.raises(gate.GateError, match="non-finite"):
        gate.validate_record(
            pinned,
            calibration,
            month,
            expect_calibration_commit="calibration-commit",
            expect_month_commit="month-commit",
        )


@pytest.mark.parametrize(
    ("calibration_commit", "month_commit", "root"),
    (
        ("wrong-calibration", "month-commit", "calibration"),
        ("calibration-commit", "wrong-month", "month"),
    ),
)
def test_each_producer_commit_is_checked(roots, calibration_commit, month_commit, root):
    pinned, calibration, month = roots
    with pytest.raises(gate.GateError, match=f"producer commit mismatch: .*{root}"):
        gate.validate_record(
            pinned,
            calibration,
            month,
            expect_calibration_commit=calibration_commit,
            expect_month_commit=month_commit,
        )
