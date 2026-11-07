#!/usr/bin/env python3
"""Fail-closed admission gate for the round-71 kt=2 tracer-stage record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

DIMS = (36, 26, 31)
LEVELS = {
    1: {1: (1, 1, 3, 3), 2: (1, 3, 2, 2)},
    2: {1: (3, 3, 1, 1), 2: (3, 1, 2, 2)},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"REFUSE: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(
    path: Path, stage: int, *, expected_kt: int = 2,
    truncate: bool = False,
) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    if truncate:
        values = values[:-1]
    require(expected_kt in (1, 2), f"{path}: unsupported kt {expected_kt}")
    expected = (1, expected_kt, stage, *LEVELS[expected_kt][stage], *DIMS, 64)
    require(magic == "NEMO_L2_RKTRA_1", f"{path}: bad magic {magic!r}")
    require(header == expected, f"{path}: header {header} != {expected}")
    nx, ny, nz = DIMS
    n3, n2 = nx * ny * nz, nx * ny
    require(values.size == 15 * n3 + 3 * n2, f"{path}: bad payload size")
    names = (
        "zero_T", "zero_S", "zFu", "zFv", "zFw",
        "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S", "Kbb_T", "Kbb_S",
        "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S",
    )
    fields = {}
    for index, name in enumerate(names):
        raw = values[index * n3 : (index + 1) * n3]
        fields[name] = raw.reshape((nx, ny, nz), order="F")[2:-2, 2:-2, :]
        require(np.all(np.isfinite(fields[name])), f"{path}: non-finite {name}")
    offset = len(names) * n3
    for index, name in enumerate(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")):
        raw = values[offset + index * n2 : offset + (index + 1) * n2]
        fields[name] = raw.reshape((nx, ny), order="F")[2:-2, 2:-2]
        require(np.all(np.isfinite(fields[name])), f"{path}: non-finite {name}")
    return {"header": header, "fields": fields, "sha256": sha256(path)}


def check_stamp(path: Path, expected_commit: str) -> None:
    record = path.with_name(path.name.removesuffix(".stamp"))
    parts = path.read_text(encoding="utf-8").strip().split()
    require(len(parts) == 3, f"{path}: malformed stamp")
    require(parts[0] == sha256(record), f"{path}: digest mismatch")
    require(parts[1] == expected_commit, f"{path}: producer commit mismatch")
    require(parts[2] == record.name, f"{path}: record name mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plant", choices=("none", "stamp", "bridge-ulp", "truncation"),
        default="none",
    )
    args = parser.parse_args()

    producer = (args.root / "producer_commit.txt").read_text().strip()
    require(producer == args.expect_commit, "producer_commit.txt mismatch")
    paths = {
        stage: args.root / f"oracle_rktracer_operands_kt00000002_s{stage}.bin"
        for stage in (1, 2)
    }
    for path in paths.values():
        require(path.is_file(), f"missing {path}")
        expected = "planted-wrong-commit" if args.plant == "stamp" else args.expect_commit
        check_stamp(path.with_name(path.name + ".stamp"), expected)

    records = {
        stage: read_record(
            path, stage,
            truncate=args.plant == "truncation" and stage == 2,
        )
        for stage, path in paths.items()
    }
    if args.plant == "bridge-ulp":
        planted = records[2]["fields"]["Kmm_T"]
        planted[0, 0, 0] = np.nextafter(planted[0, 0, 0], np.inf)

    bridges = {}
    for tracer in ("T", "S"):
        bridges[f"Kbb_{tracer}"] = bool(np.array_equal(
            records[1]["fields"][f"Kbb_{tracer}"],
            records[2]["fields"][f"Kbb_{tracer}"],
        ))
        bridges[f"Kaa1_to_Kmm2_{tracer}"] = bool(np.array_equal(
            records[1]["fields"][f"Kaa_{tracer}"],
            records[2]["fields"][f"Kmm_{tracer}"],
        ))
    require(all(bridges.values()), f"stage bridge is not bit-exact: {bridges}")

    report = {
        "format": "nemo-l2-round71-fct-stage2-v1",
        "worktree": worktree_stamp(),
        "producer_commit": producer,
        "records": {
            str(stage): {"header": list(row["header"]), "sha256": row["sha256"]}
            for stage, row in records.items()
        },
        "bit_exact_bridges": bridges,
        "plant": args.plant,
        "status": "AT-BAR",
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ROUND71_FCT_STAGE2_RECORD_AT_BAR")


if __name__ == "__main__":
    main()
