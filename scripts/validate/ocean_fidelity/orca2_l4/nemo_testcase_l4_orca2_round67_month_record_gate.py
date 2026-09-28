#!/usr/bin/env python3
"""Admit the pinned ORCA1ICE ten-step calibration and 30-day record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


EXPECTED_BINARY_SHA256 = (
    "c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343"
)
EXPECTED_PINNED_DECK_SHA256 = (
    "51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9"
)
EXPECTED_INPUT_SHA256 = (
    "3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5"
)
EXPECTED_ICE_CFG_SHA256 = (
    "6b647863137b518b95ff97f83975d9afcb3944b7f6e8d63e45a05f494f5edc89"
)
RESTART_FIELDS = ("sshn", "un", "vn", "tn", "sn")
PLANTS = ("none", "calibration-ulp", "missing-shard", "hidden-deck-delta")


class GateError(RuntimeError):
    """The month record violated a frozen round-67 predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(root: Path, name: str) -> dict[str, str]:
    path = root / name
    require(path.is_file(), f"missing manifest {path}")
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None,
                f"malformed manifest row in {path}: {line!r}")
        filename = fields[1].lstrip("* ")
        require(filename == Path(filename).name and filename not in rows,
                f"unsafe or duplicate manifest name: {filename!r}")
        rows[filename] = fields[0]
    require(bool(rows), f"empty manifest {path}")
    for filename, expected in rows.items():
        target = root / filename
        require(target.is_file(), f"manifest target missing: {target}")
        require(sha256(target) == expected, f"manifest digest mismatch: {target}")
    return rows


def _integer(value: str, label: str) -> int:
    match = re.match(r"\s*(-?\d+)", value)
    require(match is not None, f"{label}: not an integer assignment: {value!r}")
    return int(match.group(1))


def validate_deck_delta(calibration: Path, month: Path, *, plant: str = "none") -> dict:
    """Require month and calibration decks to differ only in run length."""
    cal_deck = _manifest(calibration, "deck_files.sha256")
    month_deck = _manifest(month, "deck_files.sha256")
    require(set(cal_deck) == set(month_deck), "deck file inventory changed")
    require(cal_deck["namelist_ice_cfg"] == EXPECTED_ICE_CFG_SHA256,
            "calibration does not use the pinned ORCA1ICE namelist")
    require(month_deck["namelist_ice_cfg"] == EXPECTED_ICE_CFG_SHA256,
            "month does not use the pinned ORCA1ICE namelist")
    for name in cal_deck:
        if name != "namelist_cfg":
            require(cal_deck[name] == month_deck[name],
                    f"unauthorized deck-file delta: {name}")

    before = namelist_values(calibration / "namelist_cfg")
    after = namelist_values(month / "namelist_cfg")
    require(set(before) == set(after), "namelist assignment inventory changed")
    if plant == "hidden-deck-delta":
        after["namrun.rn_dt"] = "10801.0"
    changed = sorted(key for key in before if before[key] != after[key])
    expected = ["namrun.nn_itend", "namrun.nn_stock"]
    require(changed == expected,
            f"unauthorized namelist delta: changed={changed}, expected={expected}")
    require(_integer(before["namrun.nn_itend"], "calibration nn_itend") == 10,
            "calibration nn_itend is not 10")
    require(_integer(before["namrun.nn_stock"], "calibration nn_stock") == 10,
            "calibration nn_stock is not 10")
    require(_integer(after["namrun.nn_itend"], "month nn_itend") == 240,
            "month nn_itend is not 240")
    require(_integer(after["namrun.nn_stock"], "month nn_stock") == 240,
            "month nn_stock is not 240")
    return {"changed_assignments": changed, "from": 10, "to": 240}


def _mutated_copy(path: Path, target: Path) -> None:
    shutil.copy2(path, target)
    with Dataset(target, "r+") as dataset:
        for variable in dataset.variables.values():
            if variable.dtype == np.dtype("float64") and variable.size:
                variable.set_auto_maskandscale(False)
                values = np.asarray(variable[:]).copy()
                flat = values.reshape(-1)
                flat[0] = np.nextafter(flat[0], np.inf)
                variable[:] = values
                return
    raise GateError(f"no mutable float64 payload in {path}")


def validate_calibration(pinned: Path, calibration: Path, *, plant: str = "none") -> dict:
    """Compare every ocean and ice restart payload to the admitted pin."""
    names = [
        f"ORCA2_00000010_restart{suffix}_{rank:04d}.nc"
        for suffix in ("", "_ice") for rank in (0, 1)
    ]
    exact = []
    for index, name in enumerate(names):
        left = pinned / name
        right = calibration / name
        require(left.is_file() and right.is_file(), f"missing calibration pair: {name}")
        if plant == "calibration-ulp" and index == 0:
            with tempfile.TemporaryDirectory(prefix="orca2-r67-plant-") as temporary:
                planted = Path(temporary) / name
                _mutated_copy(right, planted)
                phase1._netcdf_equal_except_timestamp(left, planted)
        else:
            phase1._netcdf_equal_except_timestamp(left, right)
        exact.append({"file": name, "variables": _variable_count(right)})
    return {"status": "BIT_EXACT", "restart_shards": exact}


def _variable_count(path: Path) -> int:
    with Dataset(path) as dataset:
        return len(dataset.variables)


def _require_run_log(root: Path, step: int) -> None:
    stdout = (root / "run.user.stdout.log").read_text(errors="strict")
    timing = (root / "run.user.time.log").read_text(errors="strict")
    ocean = (root / "ocean.output").read_text(errors="strict")
    require("STOP 0" in stdout, f"{root}: NEMO did not report STOP 0")
    require("RUN_DONE" in timing, f"{root}: launcher did not report RUN_DONE")
    require(re.search(rf"number of the last time step\s+nn_itend\s+=\s+{step}\b", ocean) is not None,
            f"{root}: resolved nn_itend is not {step}")
    require(re.search(rf"frequency of restart file\s+nn_stock\s+=\s+{step}\b", ocean) is not None,
            f"{root}: resolved nn_stock is not {step}")
    require("number of ice  categories" in ocean and
            re.search(r"number of ice\s+categories\s+jpl\s+=\s+1\b", ocean) is not None,
            f"{root}: resolved ice category count is not one")
    require("start from rest" in ocean, f"{root}: no from-rest provenance")
    require("No icebergs used" in ocean, f"{root}: icebergs are not disabled")


def validate_month(month: Path, *, plant: str = "none") -> dict:
    """Validate the self-describing terminal month restart."""
    names = [
        f"ORCA2_00000240_restart{suffix}_{rank:04d}.nc"
        for suffix in ("", "_ice") for rank in (0, 1)
    ]
    if plant == "missing-shard":
        names[-1] = "ORCA2_00000240_restart_ice_9999.nc"
    for name in names:
        require((month / name).is_file(), f"missing terminal restart shard: {name}")

    rows = []
    for rank in (0, 1):
        path = month / f"ORCA2_00000240_restart_{rank:04d}.nc"
        with Dataset(path) as dataset:
            require(set(RESTART_FIELDS).issubset(dataset.variables),
                    f"{path.name}: required ocean restart fields missing")
            kt = dataset["kt"]
            kt.set_auto_maskandscale(False)
            require(float(np.asarray(kt[:])) == 240.0, f"{path.name}: kt is not 240")
            fields = {}
            for name in RESTART_FIELDS:
                variable = dataset[name]
                require(variable.dtype == np.dtype("float64"),
                        f"{path.name}: {name} is not float64")
                variable.set_auto_maskandscale(False)
                values = np.asarray(variable[:])
                require(bool(np.isfinite(values).all()),
                        f"{path.name}: {name} contains non-finite values")
                fields[name] = {"dtype": str(variable.dtype), "shape": list(values.shape)}
        rows.append({"file": path.name, "fields": fields})
    return {"status": "FINITE_FP64", "step": 240, "ocean_shards": rows}


def validate_record(pinned: Path, calibration: Path, month: Path | None, *,
                    expect_commit: str, mode: str = "full", plant: str = "none") -> dict:
    """Validate calibration alone or the complete acquired record."""
    require(mode in ("calibration", "full"), f"unknown mode {mode}")
    require(plant in PLANTS, f"unknown plant {plant}")
    for root in (calibration,) if month is None else (calibration, month):
        require((root / "nemo").is_file(), f"missing executable: {root}/nemo")
        require(sha256(root / "nemo") == EXPECTED_BINARY_SHA256,
                f"unexpected scalar-math executable: {root}/nemo")
        require((root / "producer_commit.txt").read_text().strip() == expect_commit,
                f"producer commit mismatch: {root}")
        require(sha256(root / "input_files.sha256") == EXPECTED_INPUT_SHA256,
                f"input manifest pin changed: {root}")
        _manifest(root, "input_files.sha256")

    require(sha256(calibration / "deck_files.sha256") == EXPECTED_PINNED_DECK_SHA256,
            "calibration deck manifest is not the pinned ORCA1ICE deck")
    _manifest(calibration, "deck_files.sha256")
    _require_run_log(calibration, 10)
    calibration_report = validate_calibration(pinned, calibration, plant=plant)
    if mode == "calibration":
        require(plant in ("none", "calibration-ulp"),
                f"plant {plant} requires full mode")
        return {
            "format": "nemo-testcase-l4-orca2-round67-month-record-v1",
            "status": "PASS_MONTH_CALIBRATION",
            "claim_label": "independent",
            "calibration": calibration_report,
        }

    require(month is not None, "full admission requires --month")
    deck = validate_deck_delta(calibration, month, plant=plant)
    _require_run_log(month, 240)
    month_report = validate_month(month, plant=plant)
    return {
        "format": "nemo-testcase-l4-orca2-round67-month-record-v1",
        "status": "PASS_MONTH_RECORD",
        "claim_label": "independent",
        "binary_sha256": EXPECTED_BINARY_SHA256,
        "calibration": calibration_report,
        "deck_delta": deck,
        "month": month_report,
        "next": "ORCA2_INDEPENDENT_MONTH_MAGNITUDE_RANKING",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--month", type=Path)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--mode", choices=("calibration", "full"), default="full")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = validate_record(
            args.pinned,
            args.calibration,
            args.month,
            expect_commit=args.expect_commit,
            mode=args.mode,
            plant=args.plant,
        )
        report["worktree"] = worktree_stamp()
    except (GateError, phase1.GateError, KeyError, OSError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
