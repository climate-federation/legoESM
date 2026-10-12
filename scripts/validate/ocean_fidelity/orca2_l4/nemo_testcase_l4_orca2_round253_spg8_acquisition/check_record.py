#!/usr/bin/env python3
"""Admit the rank-complete OMT-4 kt=8 external-substep record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round95_spgts_acquisition import (  # noqa: E402
    check_record as base,
)

PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation",
    "missing-frame", "swapped-rank", "restart-byte",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise base.Refusal(message)


def run(root: Path, baseline: Path, plant: str) -> dict[str, object]:
    coverage = __import__("numpy").zeros((148, 180), dtype="int8")
    records = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r166_spg_rank{expected_rank:04d}_kt00000008.bin"
        applied = plant if expected_rank == 0 and plant not in (
            "restart-byte", "none") else "none"
        record = base.read_record(
            path, applied, expected_kt=8, expected_magic="NEMO_L4_R166SPG")
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: slab outside global domain")
        coverage[j0:j1, i0:i1] += 1
        records.append(record)
    require(bool((coverage == 1).all()),
            "rank slabs do not cover the domain exactly once")
    require(records[0]["icycle"] == records[1]["icycle"] == 65,
            "rank external-substep counts moved")

    restarts = sorted(baseline.glob("ORCA2_00000010_restart_????.nc"))
    require(len(restarts) == 2, "baseline lacks the two kt=10 ocean restarts")
    restart_rows = []
    for index, source in enumerate(restarts):
        target = root / source.name
        require(target.is_file(), f"missing target restart {source.name}")
        left, right = source.read_bytes(), target.read_bytes()
        if plant == "restart-byte" and index == 0:
            right = right[:-1] + bytes([right[-1] ^ 1])
        require(left == right, f"write-only acquisition changed {source.name}")
        restart_rows.append({
            "name": source.name,
            "sha256": hashlib.sha256(right).hexdigest(),
        })
    return {
        "status": "PASS_R253_OMT4_SPG8_ADMISSION",
        "rank_coverage": "exactly-once",
        "records": records,
        "terminal_restart_comparisons": restart_rows,
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
    except (OSError, UnicodeDecodeError, ValueError, base.Refusal) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R253_OMT4_SPG8_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
