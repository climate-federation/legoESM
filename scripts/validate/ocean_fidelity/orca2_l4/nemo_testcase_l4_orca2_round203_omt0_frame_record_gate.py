#!/usr/bin/env python3
"""Admit the additions-only OMT-0 kt=1..10 entry/stage frame record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round84_rung0_frame_gate as frame_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round199_omt0_record_gate as omt0_gate,
)


BASE_BINARY_SHA256 = omt0_gate.BINARY_SHA256
INSTRUMENT_BINARY_SHA256 = "b54b37788058697e6f649d16bf9f3843bc9bb672c05bdf06404c946c659ac6a9"
FRAME_SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_OMIP_L4_R90FRAMES/MY_SRC"
)
FRAME_SOURCE_SHA256 = {
    "stprk3.F90": "31f9d62f7ac06b84bc3ef6b5ec94da19695014663c671c31bfc21496e42d1e93",
    "l4_r84_frames.F90": "231b17f2a4b5ca28110efad413ea17af6ab2c3bb1f4875b5cbb0d6b6faa740f5",
    "traadv.F90": "ddd33bdec420246ba43419599da9c33e89148cf032ad7f9ba3fab3d9542625ff",
}
STEPS = tuple(range(1, 11))
STAGES = tuple(range(4))
RANKS = (0, 1)
FIELDS = frame_gate.FIELDS
PLANTS = (
    "none", "cadence", "header", "field-name", "truncation", "nonfinite",
    "missing-frame", "twin-ulp", "terminal-byte", "changed-binary",
)


class GateError(RuntimeError):
    """The OMT-0 frame record violated a frozen admission predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wanted_frames() -> set[str]:
    return {
        f"oracle_r84_frame_rank{rank:04d}_kt{step:08d}_s{stage}.bin"
        for rank in RANKS for step in STEPS for stage in STAGES
    }


def _frame_arrays(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    record = frame_gate.read_frame(path)
    raw = path.read_bytes()
    arrays = {
        name: np.frombuffer(
            raw, dtype="=f8", count=record["fields"][name]["count"],
            offset=record["offsets"][name],
        ).copy()
        for name in FIELDS
    }
    return record, arrays


def _validate_inventory(root: Path, plant: str) -> list[str]:
    wanted = _wanted_frames()
    actual = {path.name for path in root.glob("oracle_r84_frame_*.bin")}
    if plant == "missing-frame":
        actual.discard(sorted(wanted)[0])
    require(actual == wanted,
            f"frame inventory mismatch missing={sorted(wanted-actual)} "
            f"extra={sorted(actual-wanted)}")
    return sorted(wanted)


def _validate_frame(root: Path, name: str, plant: str | None = None) -> dict:
    record = frame_gate.read_frame(root / name, plant)
    header = record["header"]
    expected = (
        f"oracle_r84_frame_rank{header['rank']:04d}_"
        f"kt{header['kt']:08d}_s{header['stage']}.bin"
    )
    require(name == expected, f"{name}: header/name mismatch")
    levels = ({0: 1, 1: 3, 2: 2, 3: 3}
              if header["kt"] % 2 else
              {0: 3, 1: 1, 2: 2, 3: 1})
    require(header["level"] == levels[header["stage"]],
            f"{name}: wrong RK3 level {header['level']}")
    return record


def preflight() -> dict:
    for name, expected in FRAME_SOURCE_SHA256.items():
        path = FRAME_SOURCE / name
        require(path.is_file(), f"missing admitted frame source {path}")
        require(sha256(path) == expected, f"admitted frame source changed: {name}")
    base = frame_gate.preflight()
    source = Path(
        "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
        "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90"
    ).read_text()
    require("MOD( nn_stock, nn_fsbc) /= 0" in source,
            "compiled restart-cadence guard disappeared")
    try:
        omt0_gate.render_frequency_run_deck("", itend=10)
    except omt0_gate.GateError as error:
        require("nn_fsbc=2" in str(error), "frequency retraction changed")
    else:
        raise GateError("round-202 cadence plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round203-frame-preflight-v1",
        "status": "PASS_R203_OMT0_FRAME_PREFLIGHT",
        "steps": list(STEPS),
        "stages": list(STAGES),
        "ranks": list(RANKS),
        "fields": list(FIELDS),
        "expected_frames_per_twin": len(_wanted_frames()),
        "frame_source_sha256": FRAME_SOURCE_SHA256,
        "writer_preflight": base,
        "retracted_protocol": "nn_stock=1 with nn_fsbc=2",
    }


def _validate_run(root: Path, canonical: Path, binary_sha: str,
                  plant: str = "none") -> dict:
    expected_binary = "0" * 64 if plant == "changed-binary" else binary_sha
    require(sha256(root / "nemo") == expected_binary,
            f"{root}: binary changed")
    omt0_gate._run_provenance(
        root, itend=10, restart_steps=(10,), opened_steps=(10,), plant="none",
    )
    deck = omt0_gate.validate_run_deck(
        canonical, root, itend=10, stock=10, restart_steps=(10,), plant="none",
    )
    return {"root": str(root), "binary_sha256": expected_binary, "deck": deck}


def _terminal(root: Path, rank: int) -> Path:
    return root / f"ORCA2_00000010_restart_{rank:04d}.nc"


def admit(canonical: Path, calibration: Path, twin_a: Path, twin_b: Path,
          month: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = preflight()
    if plant == "cadence":
        omt0_gate.render_frequency_run_deck(canonical.read_text(), itend=10)
        raise GateError("cadence plant stayed green")

    calibration_row = _validate_run(
        calibration, canonical, BASE_BINARY_SHA256,
        "changed-binary" if plant == "changed-binary" else "none",
    )
    twin_binary = sha256(twin_a / "nemo")
    require(twin_binary == INSTRUMENT_BINARY_SHA256,
            "instrumented binary changed")
    require(sha256(twin_b / "nemo") == twin_binary,
            "instrumented twin binaries differ")
    twin_rows = [
        _validate_run(twin_a, canonical, twin_binary),
        _validate_run(twin_b, canonical, twin_binary),
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
        left_record = _validate_frame(
            twin_a, name, parser_plant if index == 0 else None,
        )
        right_record = _validate_frame(twin_b, name)
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
        baseline = _terminal(calibration, rank).read_bytes()
        for label, root in (("twin_a", twin_a), ("twin_b", twin_b)):
            candidate = _terminal(root, rank).read_bytes()
            if plant == "terminal-byte" and rank == 0 and label == "twin_a":
                candidate = candidate[:-1] + bytes([candidate[-1] ^ 1])
            require(candidate == baseline,
                    f"{label}: instrument changed terminal restart rank {rank}")
            terminal_comparisons += 1

    boundary = omt0_gate._run_provenance(
        month, itend=omt0_gate.MONTH_ITEND,
        restart_steps=omt0_gate.MONTH_STEPS,
        opened_steps=omt0_gate.AVAILABLE_MONTH_STEPS,
        expected_oracle_stop=True,
    )
    omt0_gate.validate_run_deck(
        canonical, month, itend=omt0_gate.MONTH_ITEND,
        stock=omt0_gate.MONTH_ITEND,
        restart_steps=omt0_gate.MONTH_STEPS,
    )

    report.update({
        "status": "PASS_R203_OMT0_ENTRY_STAGE_RECORD__STOP_MONTH_AT_KT11",
        "calibration": calibration_row,
        "twins": twin_rows,
        "frame_records_per_twin": len(names_a),
        "frame_field_comparisons": comparisons,
        "terminal_restart_byte_comparisons": terminal_comparisons,
        "month_boundary": boundary,
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
    except (GateError, omt0_gate.GateError, frame_gate.GateError,
            OSError, UnicodeError, ValueError) as error:
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
