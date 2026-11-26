#!/usr/bin/env python3
"""Admit the additions-only OMT-2 kt=1..10 frames and month boundary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round199_omt0_record_gate as protocol,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_frame_record_gate as omt1_record,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round218_omt2_deck_gate as omt2_deck,
)


BASE_BINARY_SHA256 = protocol.BINARY_SHA256
INSTRUMENT_BINARY_SHA256 = omt1_record.INSTRUMENT_BINARY_SHA256
STEPS = tuple(range(1, 11))
STAGES = tuple(range(4))
RANKS = (0, 1)
FIELDS = omt1_record.FIELDS
MONTH_ITEND = 96
MONTH_STEPS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
PLANTS = (
    "none", "cadence", "header", "field-name", "truncation", "nonfinite",
    "missing-frame", "twin-ulp", "terminal-byte", "changed-binary",
    "early-month", "stop-line",
)

STP_CTL_LINE = (
    "stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s  or  S <= 0  or  "
    "S >= 100  or  NaN encounter in the tests"
)


class GateError(RuntimeError):
    """The OMT-2 record violated a frozen admission predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return omt1_record.sha256(path)


def _wanted_frames() -> set[str]:
    return {
        f"oracle_r84_frame_rank{rank:04d}_kt{step:08d}_s{stage}.bin"
        for rank in RANKS for step in STEPS for stage in STAGES
    }


def preflight() -> dict:
    inherited = omt1_record.preflight()
    return {
        **inherited,
        "format": "nemo-testcase-l4-orca2-round218-omt2-frame-preflight-v1",
        "status": "PASS_R218_OMT2_FRAME_PREFLIGHT",
        "steps": list(STEPS),
        "expected_frames_per_twin": len(_wanted_frames()),
        "month_itend": MONTH_ITEND,
        "month_steps": list(MONTH_STEPS),
    }


def _validate_inventory(root: Path, plant: str) -> list[str]:
    wanted = _wanted_frames()
    actual = {path.name for path in root.glob("oracle_r84_frame_*.bin")}
    if plant == "missing-frame":
        actual.discard(sorted(wanted)[0])
    require(actual == wanted,
            f"frame inventory mismatch missing={sorted(wanted-actual)} "
            f"extra={sorted(actual-wanted)}")
    return sorted(wanted)


def _validate_run(root: Path, canonical: Path, binary_sha: str, *, itend: int,
                  stock: int, restart_steps: tuple[int, ...],
                  plant: str = "none") -> dict:
    expected = "0" * 64 if plant == "changed-binary" else binary_sha
    require(sha256(root / "nemo") == expected, f"{root}: binary changed")
    stdout = (root / "run.user.stdout.log").read_text()
    timing = (root / "run.user.time.log").read_text()
    require("STOP 0" in stdout and "RUN_DONE" in timing,
            f"{root}: incomplete normal run")
    deck = omt2_deck.validate_run_deck(
        canonical, root, itend=itend, stock=stock,
        restart_steps=restart_steps,
    )
    return {"root": str(root), "binary_sha256": expected, "deck": deck}


def _frame_arrays(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    return omt1_record._frame_arrays(path)


def _month_boundary(root: Path, canonical: Path, plant: str) -> dict:
    require(sha256(root / "nemo") == BASE_BINARY_SHA256,
            f"{root}: binary changed")
    stdout = (root / "run.user.stdout.log").read_text()
    timing = (root / "run.user.time.log").read_text()
    ocean = (root / "ocean.output").read_text()
    deck = omt2_deck.validate_run_deck(
        canonical, root, itend=MONTH_ITEND, stock=MONTH_ITEND,
        restart_steps=MONTH_STEPS,
    )
    if "STOP 0" in stdout and "RUN_DONE" in timing:
        disposition = "COMPLETED"
        boundary_step = MONTH_ITEND
        available = MONTH_STEPS
    else:
        # Round 218's launcher reached the compiled stop, but its shorter
        # literal grep rejected the real source line before it could append
        # RUN_EXPECTED_STP_CTL.  The raw MPI/stpctl/abort-state evidence is
        # the authority; the launcher's derived marker is informational.
        require("RUN_STARTED_UTC=" in timing,
                "month lacks its run-start provenance stamp")
        require("MPI_ABORT was invoked" in stdout and "Errorcode: 123" in stdout,
                "month stp_ctl lacks exact MPI abort evidence")
        if plant == "stop-line":
            ocean = ocean.replace(STP_CTL_LINE, "different compiled stop")
        require(STP_CTL_LINE in ocean,
                "month did not stop through compiled stp_ctl")
        steps = tuple(int(value) for value in re.findall(
            r"(?m)^\s*kt\s+(\d+)\s", ocean,
        ))
        require(steps, "month boundary has no printed kt")
        boundary_step = max(steps)
        if plant == "early-month":
            boundary_step = 9
        require(10 < boundary_step <= MONTH_ITEND,
                f"OMT-2 boundary is not later than OMT-1: kt={boundary_step}")
        require((root / "output.abort_0000.nc").is_file(),
                "month boundary lacks rank-0 abort state")
        disposition = "STP_CTL"
        available = tuple(step for step in MONTH_STEPS if step < boundary_step)
    require(available and available[0] == 10,
            "month record does not retain the ten-step calibration")
    headers = []
    for step in available:
        for rank in RANKS:
            _, header = protocol._payload(
                root / f"ORCA2_{step:08d}_restart_{rank:04d}.nc", step,
            )
            headers.append(header)
    return {
        "disposition": disposition,
        "last_step": boundary_step,
        "available_restart_steps": list(available),
        "restart_headers": headers,
        "deck": deck,
    }


def admit(canonical: Path, calibration: Path, twin_a: Path, twin_b: Path,
          month: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = preflight()
    if plant == "cadence":
        protocol.render_frequency_run_deck(canonical.read_text(), itend=10)
        raise GateError("cadence plant stayed green")

    calibration_row = _validate_run(
        calibration, canonical, BASE_BINARY_SHA256, itend=10, stock=10,
        restart_steps=(10,),
        plant="changed-binary" if plant == "changed-binary" else "none",
    )
    twin_binary = sha256(twin_a / "nemo")
    require(twin_binary == INSTRUMENT_BINARY_SHA256,
            "instrumented binary changed")
    require(sha256(twin_b / "nemo") == twin_binary,
            "instrumented twin binaries differ")
    twin_rows = [
        _validate_run(root, canonical, twin_binary, itend=10, stock=10,
                      restart_steps=(10,))
        for root in (twin_a, twin_b)
    ]

    names_a = _validate_inventory(twin_a, plant)
    names_b = _validate_inventory(twin_b, "none")
    require(names_a == names_b, "twin frame inventories differ")
    parser_plant = plant if plant in {
        "header", "field-name", "truncation", "nonfinite",
    } else None
    frame_headers = []
    comparisons = 0
    for index, name in enumerate(names_a):
        left_record = omt1_record._validate_frame(
            twin_a, name, parser_plant if index == 0 else None,
        )
        right_record = omt1_record._validate_frame(twin_b, name)
        require(left_record["header"] == right_record["header"],
                f"{name}: twin headers differ")
        _, left = _frame_arrays(twin_a / name)
        _, right = _frame_arrays(twin_b / name)
        for field in FIELDS:
            candidate = right[field]
            if plant == "twin-ulp" and index == 0 and field == FIELDS[0]:
                candidate = candidate.copy()
                candidate[0] = np.nextafter(candidate[0], np.inf)
            require(np.array_equal(left[field], candidate),
                    f"{name}: twin field {field} differs")
            comparisons += 1
        frame_headers.append({
            "name": name,
            "header": left_record["header"],
            "sha256_a": left_record["sha256"],
            "sha256_b": right_record["sha256"],
        })

    terminal_comparisons = 0
    for rank in RANKS:
        baseline = (calibration / f"ORCA2_00000010_restart_{rank:04d}.nc").read_bytes()
        for label, root in (("twin_a", twin_a), ("twin_b", twin_b)):
            candidate = (root / f"ORCA2_00000010_restart_{rank:04d}.nc").read_bytes()
            if plant == "terminal-byte" and rank == 0 and label == "twin_a":
                candidate = candidate[:-1] + bytes([candidate[-1] ^ 1])
            require(candidate == baseline,
                    f"{label}: instrument changed terminal restart rank {rank}")
            terminal_comparisons += 1

    month_boundary = _month_boundary(month, canonical, plant)
    report.update({
        "status": "PASS_R218_OMT2_ENTRY_STAGE_AND_MONTH_RECORD",
        "calibration": calibration_row,
        "twins": twin_rows,
        "frame_records_per_twin": len(names_a),
        "frame_field_comparisons": comparisons,
        "terminal_restart_byte_comparisons": terminal_comparisons,
        "month_boundary": month_boundary,
        "frames": frame_headers,
        "instrument_binary_sha256": twin_binary,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--month", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight_only:
            require(args.plant == "none", "preflight does not accept a plant")
            result = preflight()
        else:
            require(all((args.candidate, args.calibration, args.twin_a,
                         args.twin_b, args.month)),
                    "admission requires candidate, calibration, twins and month")
            result = admit(
                args.candidate, args.calibration, args.twin_a, args.twin_b,
                args.month, args.plant,
            )
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, omt1_record.GateError, omt1_record.frame_gate.GateError,
            omt2_deck.GateError,
            protocol.GateError, OSError, UnicodeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
