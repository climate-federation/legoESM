#!/usr/bin/env python3
"""Admit the reused rank-complete stage-1 schema on the OMT-4 deck."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round175_stage1_acquisition import (  # noqa: E402,E501
    check_record as schema,
)

PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation",
    "swapped-rank", "coverage", "frame-byte", "restart-byte",
)


class Refusal(RuntimeError):
    """The OMT-4 record or additions-only proof is inadmissible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_family(root: Path, baseline: Path, pattern: str,
                   expected: int, plant: bool) -> list[dict[str, object]]:
    sources = sorted(baseline.glob(pattern))
    require(len(sources) == expected,
            f"baseline {pattern} count {len(sources)} != {expected}")
    rows = []
    for index, source in enumerate(sources):
        target = root / source.name
        require(target.is_file(), f"missing target {source.name}")
        left, right = source.read_bytes(), target.read_bytes()
        if plant and index == 0:
            right = right[:-1] + bytes([right[-1] ^ 1])
        require(left == right,
                f"additions-only acquisition changed {source.name}")
        rows.append({"name": source.name,
                     "sha256": hashlib.sha256(right).hexdigest()})
    return rows


def run(root: Path, baseline: Path, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    schema_plants = {
        "header", "field-name", "field-dims", "truncation", "swapped-rank",
    }
    for expected_rank in (0, 1):
        path = root / (
            f"oracle_r175_stage1_rank{expected_rank:04d}_kt00000001.bin")
        local_plant = plant if expected_rank == 0 and plant in schema_plants else "none"
        record = schema.read_record(path, local_plant)
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        if plant == "coverage" and expected_rank == 1:
            i0 -= 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: owned slab outside global domain")
        coverage[j0:j1, i0:i1] += 1
        records.append(record)
    require(bool(np.all(coverage == 1)), "rank slabs are not exactly-once")

    frames = compare_family(
        root, baseline, "oracle_r84_frame_*.bin", 80,
        plant == "frame-byte")
    restarts = compare_family(
        root, baseline, "ORCA2_00000010_restart_????.nc", 2,
        plant == "restart-byte")
    return {
        "status": "PASS_R280_OMT4_STAGE1_ADMISSION",
        "schema": "NEMO_L4_R175S1",
        "rank_coverage": "exactly-once",
        "records": records,
        "existing_frame_comparisons": frames,
        "terminal_restart_comparisons": restarts,
        "baseline": str(baseline),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.root, args.baseline, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, UnicodeDecodeError, ValueError, schema.Refusal, Refusal) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R280_OMT4_STAGE1_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
