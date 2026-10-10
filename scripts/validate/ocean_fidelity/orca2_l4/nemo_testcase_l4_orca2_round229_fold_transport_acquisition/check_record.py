#!/usr/bin/env python3
"""Admit the rank-complete round-229 tracer-consumer operand record."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def read(path: Path, plant: str = "none") -> dict:
    raw = bytearray(path.read_bytes())
    if plant == "truncation":
        del raw[-8:]
    offset = 0
    magic = bytes(raw[offset:offset + 16]).decode("ascii").rstrip()
    offset += 16
    header = list(struct.unpack_from("=15i", raw, offset))
    offset += 60
    if plant == "rank":
        header[3] = 7
    version, kt, stage, rank, nx, ny, nz, origin_x, origin_y, i0, j0, i1, j1, bits, count = header
    require(magic == "NEMO_L4_R229FLD1", f"{path.name}: bad magic")
    require((version, kt, stage, bits, count) == (1, 1, 1, 64, 5),
            f"{path.name}: bad provenance header")
    require(rank in (0, 1), f"{path.name}: bad rank {rank}")
    require((nx, ny, nz) == (94, 152, 31), f"{path.name}: bad shape")
    require((i0, j0, i1, j1) == (3, 3, 92, 150),
            f"{path.name}: bad owned bounds")
    expected = ["zFv_after_trp", "T_Kmm", "S_Kmm", "e3t_Kmm", "tmask"]
    fields = {}
    for index, expected_name in enumerate(expected):
        require(offset + 32 <= len(raw), f"{path.name}: truncated field header")
        name = bytes(raw[offset:offset + 16]).decode("ascii").rstrip()
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        if plant == "field-name" and index == 0:
            name = "wrong"
        require((name, ndim, n1, n2, n3) ==
                (expected_name, 3, nx, ny, nz),
                f"{path.name}: bad field {index} {name!r}")
        size = n1 * n2 * n3
        require(offset + 8 * size <= len(raw), f"{path.name}: truncated {name}")
        values = np.frombuffer(raw, dtype=np.float64, count=size, offset=offset)
        offset += 8 * size
        require(np.all(np.isfinite(values)), f"{path.name}: nonfinite {name}")
        fields[name] = {
            "shape": [n1, n2, n3],
            "min": float(values.min()),
            "max": float(values.max()),
        }
    require(offset == len(raw), f"{path.name}: trailing payload")
    require(fields["e3t_Kmm"]["min"] > 0.0,
            f"{path.name}: non-positive live thickness")
    return {"rank": rank, "origin": [origin_x, origin_y], "fields": fields}


def validate(root: Path, plant: str) -> dict:
    paths = sorted(root.glob("oracle_r229_fold_rank????_kt00000001_s1.bin"))
    require(len(paths) == 2, "expected exactly two rank records")
    rows = [read(path, plant=plant) for path in paths]
    require([row["rank"] for row in rows] == [0, 1], "rank coverage moved")
    require([row["origin"] for row in rows] == [[1, 1], [91, 1]],
            "rank origins moved")
    return {
        "format": "nemo-testcase-l4-orca2-round229-fold-record-v1",
        "status": "PASS_R229_FOLD_RECORD",
        "records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plant", choices=("none", "rank", "field-name", "truncation"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.root, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, UnicodeError, struct.error, GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R229_FOLD_RECORD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
