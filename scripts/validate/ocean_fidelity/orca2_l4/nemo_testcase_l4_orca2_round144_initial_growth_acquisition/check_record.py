#!/usr/bin/env python3
"""Admit the rank-complete ORCA2 rung-0 step-1..10 growth record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = "NEMO_L4_R144G1"
HEADER = struct.Struct("=14i")
GROUP = struct.Struct("=5i")
STEPS = tuple(range(1, 11))
FIELDS_2D = (
    "ssh_entry", "r3t_entry", "uub_entry", "vvb_entry",
    "ssh_after", "r3t_after", "uub_after", "vvb_after", "un_adv", "vn_adv",
    "r3t_stage1",
)
FIELDS_3D = ("zFu_stage1", "zFv_stage1", "zFw_stage1")
FIELDS = FIELDS_2D + FIELDS_3D
PLANTS = (
    "none", "header", "field-name", "field-rank", "field-dims", "truncation",
    "missing-step", "nonfinite", "swapped-rank", "restart-byte",
)


class Refusal(RuntimeError):
    """The record violated a frozen acquisition predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_record(path: Path, plant: str = "none") -> dict[str, object]:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 16 + HEADER.size, f"{path.name}: record too short")
    magic = raw[:16].decode("ascii", "replace").rstrip(" \x00")
    if plant == "header":
        magic = "X" + magic[1:]
    require(magic == MAGIC, f"{path.name}: bad magic {magic!r}")
    header = list(HEADER.unpack_from(raw, 16))
    (version, kt, rank, jpi, jpj, jpk, nimpp, njmpp, ntsi, ntsj,
     ntei, ntej, bits, nfields) = header
    if plant == "swapped-rank":
        rank = 1 - rank
    require(version == 1 and kt in STEPS and rank in (0, 1),
            f"{path.name}: bad identity {header}")
    require(bits == 64 and nfields == len(FIELDS),
            f"{path.name}: bad precision/field count {bits}/{nfields}")
    require(1 <= ntsi <= ntei <= jpi and 1 <= ntsj <= ntej <= jpj and jpk > 1,
            f"{path.name}: invalid local/owned dimensions")
    owned = (ntei - ntsi + 1, ntej - ntsj + 1)
    offset = 16 + HEADER.size
    groups: dict[str, np.ndarray] = {}
    for index in range(nfields):
        require(offset + 16 + GROUP.size <= len(raw),
                f"{path.name}: truncated field header {index}")
        name = raw[offset:offset + 16].decode("ascii", "replace").rstrip(" \x00")
        offset += 16
        field_rank, ndim, n1, n2, n3 = GROUP.unpack_from(raw, offset)
        offset += GROUP.size
        if plant == "field-name" and index == 0:
            name = "wrong"
        if plant == "field-rank" and index == 0:
            field_rank = 1 - field_rank
        if plant == "field-dims" and index == 0:
            n1 += 1
        require(name and name not in groups,
                f"{path.name}: duplicate/empty field {name!r}")
        require(field_rank == rank,
                f"{path.name}: field {name!r} rank {field_rank} != {rank}")
        expected = (2, *owned, 1) if name in FIELDS_2D else (3, *owned, jpk)
        require((ndim, n1, n2, n3) == expected,
                f"{path.name}: bad field {name!r} dimensions {(ndim, n1, n2, n3)}")
        count = n1 * n2 * n3
        end = offset + 8 * count
        require(end <= len(raw), f"{path.name}: truncated payload {name!r}")
        shape = (n1, n2) if ndim == 2 else (n1, n2, n3)
        values = np.frombuffer(raw, dtype="=f8", count=count, offset=offset).copy()
        values = values.reshape(shape, order="F")
        if plant == "nonfinite" and index == 0:
            values.flat[0] = np.nan
        require(bool(np.isfinite(values).all()), f"{path.name}: nonfinite field {name!r}")
        groups[name] = values
        offset = end
    require(offset == len(raw), f"{path.name}: trailing bytes")
    require(tuple(groups) == FIELDS,
            f"{path.name}: field registry {tuple(groups)} != {FIELDS}")
    return {
        "kt": kt, "rank": rank, "shape": [jpi, jpj, jpk],
        "origin": [nimpp, njmpp], "owned": [ntsi, ntsj, ntei, ntej],
        "bytes": len(raw), "sha256": digest(raw),
        "field_shapes": {name: list(values.shape) for name, values in groups.items()},
    }


def compare_restarts(calibration: Path, baseline: Path, plant: str) -> list[dict[str, str]]:
    sources = sorted(baseline.glob("ORCA2_000000??_restart_????.nc"))
    require(len(sources) == 20, "baseline does not contain 20 ten-step restart shards")
    rows = []
    for index, source in enumerate(sources):
        target = calibration / source.name
        require(target.is_file(), f"missing calibration restart {source.name}")
        left, right = source.read_bytes(), target.read_bytes()
        if plant == "restart-byte" and index == 0:
            right = right[:-1] + bytes([right[-1] ^ 1])
        require(left == right, f"recorder changed calibration restart {source.name}")
        rows.append({"name": source.name, "sha256": digest(right)})
    return rows


def run(record_root: Path, calibration_root: Path, baseline: Path,
        plant: str) -> dict[str, object]:
    coverage = {step: np.zeros((148, 180), dtype=np.int8) for step in STEPS}
    rows = []
    for step in STEPS:
        for expected_rank in (0, 1):
            if plant == "missing-step" and step == STEPS[-1] and expected_rank == 1:
                continue
            path = record_root / f"oracle_r144_growth_rank{expected_rank:04d}_kt{step:08d}.bin"
            applied = plant if step == STEPS[0] and expected_rank == 0 and plant not in (
                "none", "missing-step", "restart-byte") else "none"
            row = read_record(path, applied)
            require(row["kt"] == step and row["rank"] == expected_rank,
                    f"{path.name}: filename/header identity mismatch")
            nimpp, njmpp = row["origin"]
            ntsi, ntsj, ntei, ntej = row["owned"]
            i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
            i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
            require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                    f"{path.name}: slab outside global domain")
            coverage[step][j0:j1, i0:i1] += 1
            rows.append(row)
    require(len(rows) == 2 * len(STEPS), "growth record is not rank/step complete")
    for step, mask in coverage.items():
        require(bool(np.all(mask == 1)), f"kt={step}: rank slabs do not cover domain exactly once")
    restarts = compare_restarts(calibration_root, baseline, plant)
    terminal = record_root / "ORCA2_00000010_restart_0000.nc"
    require(terminal.is_file(), "growth run lacks kt=10 terminal restart")
    return {
        "status": "PASS_R144_INITIAL_GROWTH_RECORD",
        "claim_label": "independent",
        "steps": list(STEPS),
        "records": rows,
        "rank_coverage": "exactly-once per step",
        "terminal_step": 10,
        "terminal_rank0_sha256": digest(terminal.read_bytes()),
        "calibration_restart_comparisons": restarts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.record_root, args.calibration_root, args.baseline, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, UnicodeDecodeError, ValueError, Refusal) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R144_INITIAL_GROWTH_RECORD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
