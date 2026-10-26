#!/usr/bin/env python3
"""Census the admitted rung-0 entry/stage frames in compiled stage order."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FRAME_GATE_PATH = HERE / "nemo_testcase_l4_orca2_round84_rung0_frame_gate.py"
SPEC = importlib.util.spec_from_file_location("orca2_r84_frame_gate", FRAME_GATE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise RuntimeError(f"cannot load {FRAME_GATE_PATH}")
frame_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frame_gate)


class CensusError(RuntimeError):
    """The admitted frames violate the preregistered stage census."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CensusError(message)


def load_payloads(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Reuse the admission parser, then expose its already-validated payloads."""
    record = frame_gate.read_frame(path)
    raw = path.read_bytes()
    arrays = {
        name: np.frombuffer(
            raw,
            dtype="=f8",
            count=field["count"],
            offset=record["offsets"][name],
        ).copy()
        for name, field in record["fields"].items()
    }
    return record["header"], arrays


def census(root: Path, plant: str | None = None) -> dict:
    require(root.is_dir(), f"missing record root {root}")
    frames: dict[int, dict[int, dict[str, np.ndarray]]] = {0: {}, 1: {}}
    headers: dict[int, dict[int, dict]] = {0: {}, 1: {}}
    for rank in (0, 1):
        for stage in range(4):
            path = root / f"oracle_r84_frame_rank{rank:04d}_kt00000001_s{stage}.bin"
            require(path.is_file(), f"missing {path.name}")
            header, arrays = load_payloads(path)
            require(header["kt"] == 1 and header["rank"] == rank
                    and header["stage"] == stage,
                    f"{path.name}: header identity mismatch")
            headers[rank][stage] = header
            frames[rank][stage] = arrays

    if plant == "stage0-motion":
        frames[0][0]["u"][0] = np.nextafter(0.0, 1.0)
    elif plant == "stage1-equal":
        for rank in (0, 1):
            frames[rank][1] = {
                name: value.copy() for name, value in frames[rank][0].items()
            }

    for rank in (0, 1):
        for name in ("u", "v", "ssh"):
            require(bool(np.all(frames[rank][0][name] == 0.0)),
                    f"rank {rank} stage-0 {name} is not initialized rest")
        require(bool(np.any(frames[rank][0]["T"] != 0.0)),
                f"rank {rank} stage-0 T is vacuously zero")
        require(bool(np.any(frames[rank][0]["S"] != 0.0)),
                f"rank {rank} stage-0 S is vacuously zero")

    transitions = []
    first_changed_stage: int | None = None
    for stage in range(1, 4):
        fields = {}
        stage_changed = False
        for name in frame_gate.FIELDS:
            unequal_bits = 0
            max_abs = 0.0
            for rank in (0, 1):
                before = frames[rank][stage - 1][name]
                after = frames[rank][stage][name]
                require(before.shape == after.shape,
                        f"rank {rank} {name}: shape changed at stage {stage}")
                unequal_bits += int(np.count_nonzero(
                    before.view(np.uint64) != after.view(np.uint64)
                ))
                max_abs = max(max_abs, float(np.max(np.abs(after - before))))
            fields[name] = {
                "unequal_bits": unequal_bits,
                "max_abs": max_abs,
            }
            stage_changed = stage_changed or unequal_bits > 0
        if stage_changed and first_changed_stage is None:
            first_changed_stage = stage
        transitions.append({
            "from_stage": stage - 1,
            "to_stage": stage,
            "fields": fields,
        })

    require(first_changed_stage == 1,
            f"first changed stage is {first_changed_stage}, expected 1")
    return {
        "status": "PASS_RUNG0_STAGE_CENSUS",
        "claim_label": "independent",
        "kt": 1,
        "ranks": [0, 1],
        "stage_levels": {
            str(rank): [headers[rank][stage]["level"] for stage in range(4)]
            for rank in (0, 1)
        },
        "stage0_initialized_rest": True,
        "first_changed_stage": first_changed_stage,
        "transitions": transitions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plant", choices=("stage0-motion", "stage1-equal"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = census(args.root, args.plant)
        require(args.plant is None, f"{args.plant} plant stayed green")
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"STATUS {report['status']}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (CensusError, OSError, ValueError) as error:
        if args.plant:
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
