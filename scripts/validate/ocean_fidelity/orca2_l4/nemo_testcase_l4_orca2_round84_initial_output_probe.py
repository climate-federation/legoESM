#!/usr/bin/env python3
"""Classify rung-0 ``output.init`` without treating it as step-entry state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


class ProbeError(RuntimeError):
    pass


VARIABLES = {"u": "vozocrtx", "v": "vomecrty", "ssh": "sossheig"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def run(root: Path, plant: str | None = None) -> dict:
    rows = []
    for rank in (0, 1):
        path = root / f"output.init_{rank:04d}.nc"
        require(path.is_file(), f"missing {path}")
        with Dataset(path) as dataset:
            require("time_counter" in dataset.variables,
                    f"{path.name}: missing time_counter")
            time = np.asarray(dataset.variables["time_counter"][:], dtype=np.float64)
            require(time.size == 1 and time[0] == 0.0,
                    f"{path.name}: unexpected diagnostic timestamp {time.tolist()}")
            for field, variable in VARIABLES.items():
                values = np.asarray(dataset.variables[variable][0], dtype=np.float64)
                require(bool(np.isfinite(values).all()),
                        f"{path.name}: {field} is non-finite")
                if plant == field:
                    values = np.zeros_like(values)
                count = int(np.count_nonzero(values))
                require(count > 0,
                        f"{path.name}: {field} zero-state plant was not detected")
                rows.append({
                    "rank": rank,
                    "field": field,
                    "nonzero": count,
                    "maximum_absolute": float(np.max(np.abs(values))),
                })
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows[-1]["shard_sha256"] = digest
    return {
        "status": "PASS_INITIAL_OUTPUT_ROLE",
        "disposition": "NOT_A_STEP_ENTRY_OPERAND",
        "reason": (
            "time-zero diagnostic output contains evolved nonzero u/v/ssh; "
            "the required pre-sbc Nbb entry frames are separate streams"
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plant", choices=tuple(VARIABLES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.root, args.plant)
        require(args.plant is None, f"{args.plant} plant stayed green")
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"STATUS {report['status']}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ProbeError, OSError, KeyError, ValueError) as error:
        if args.plant:
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
