#!/usr/bin/env python3
"""Rank the registered round-160 rung-7 V-maximum regression."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


EXPECTED_SHA256 = "e44b8a5d099c7d511f409192d261a04313a96a037595ecaa35da29a733c618c5"
TARGET = (10, "stage3", "v")
PLANTS = ("none", "target-delta")


class GateError(RuntimeError):
    """The registered rung-7 comparison cannot support the frozen ranking."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_v_maximum_regressions(
    document: dict[str, object], *, plant: str = "none",
) -> dict[str, object]:
    """Rank only same-unit away-moving V maximum rows."""

    require(plant in PLANTS, f"unknown plant {plant!r}")
    document = copy.deepcopy(document)
    require(document.get("status") == "PASS_R111_ORCA2_LADDER_COMPARE",
            "round-160 comparison status moved")
    rows = document["rung7"]["moved_rows"]
    target = next((row for row in rows if (
        row["kt"], row["checkpoint"], row["field"]) == TARGET), None)
    require(target is not None, "registered kt=10 stage-3 V row is missing")
    if plant == "target-delta":
        target["metrics"]["max_abs"]["delta"] = 0.0

    ranked = []
    for row in rows:
        metric = row["metrics"]["max_abs"]
        if row["field"] != "v" or metric["direction"] != "away":
            continue
        delta = float(metric["delta"])
        require(delta > 0.0, "away-moving V maximum has non-positive delta")
        ranked.append({
            "kt": int(row["kt"]),
            "checkpoint": str(row["checkpoint"]),
            "before": float(metric["before"]),
            "after": float(metric["after"]),
            "delta": delta,
        })
    ranked.sort(key=lambda row: (-row["delta"], row["kt"], row["checkpoint"]))
    target_rows = [
        (index, row) for index, row in enumerate(ranked, start=1)
        if (row["kt"], row["checkpoint"], "v") == TARGET
    ]
    require(len(target_rows) == 1, "registered V target is not ranked exactly once")
    rank, target_row = target_rows[0]
    require(target_row == {
        "kt": 10,
        "checkpoint": "stage3",
        "before": 0.4363833806287545,
        "after": 1.2990882244341995,
        "delta": 0.862704843805445,
    }, "registered rung-7 V maximum row moved")
    require(rank == 1, f"registered rung-7 V maximum ranks {rank}, expected 1")
    return {
        "status": "PASS_R161_RUNG7_V_MAXIMUM_RANKING",
        "claim_label": "given NEMO's entry",
        "metric": "max_abs delta within V rows",
        "units": "m/s",
        "away_v_maximum_rows": len(ranked),
        "target_rank": rank,
        "target": target_row,
        "ranking": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        require(args.comparison.is_file(), "round-160 comparison is missing")
        observed = sha256(args.comparison)
        require(observed == EXPECTED_SHA256,
                "round-160 comparison digest changed")
        result = rank_v_maximum_regressions(
            json.loads(args.comparison.read_text()), plant=args.plant)
        result["source"] = {
            "path": str(args.comparison), "sha256": observed,
        }
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, KeyError, TypeError, ValueError, OSError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
