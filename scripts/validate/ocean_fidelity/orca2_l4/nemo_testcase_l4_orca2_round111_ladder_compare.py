#!/usr/bin/env python3
"""Register every ORCA2 row moved by the round-111 EEN mask statement."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


class GateError(RuntimeError):
    """The candidate ladder violates the registered landing predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _rows(document: dict) -> tuple[list[dict], object]:
    if "rows" in document:
        rows = document["rows"]
        first = (document.get("first_non_bit_checkpoint"),
                 document.get("first_non_bit_source_statement"))
    else:
        trajectory = document["candidate_trajectory"]
        rows = [
            {"kt": checkpoint["kt"], "checkpoint": checkpoint["checkpoint"],
             "field": field, **score}
            for checkpoint in trajectory["checkpoints"]
            for field, score in checkpoint["rows"].items()
        ]
        first = trajectory.get("first_non_bit_statement")
    return rows, first


def compare(base: dict, candidate: dict) -> dict:
    base_rows, base_first = _rows(base)
    candidate_rows, candidate_first = _rows(candidate)
    require(len(base_rows) == 200 and len(candidate_rows) == 200,
            "ORCA2 ladder must contain exactly 200 rows")
    key = lambda row: (row["kt"], row["checkpoint"], row["field"])
    before = {key(row): row for row in base_rows}
    after = {key(row): row for row in candidate_rows}
    require(before.keys() == after.keys(), "candidate row key set moved")
    require(base_first == candidate_first,
            "candidate moved the first non-bit statement")

    moved = []
    bit_losses = []
    for row_key in sorted(before):
        old, new = before[row_key], after[row_key]
        if old == new:
            continue
        if old["bit_identical"] and not new["bit_identical"]:
            bit_losses.append(row_key)
        metrics = {}
        for name in ("max_abs", "rms", "mean_abs_over_unequal"):
            delta = float(new[name]) - float(old[name])
            metrics[name] = {
                "before": old[name], "after": new[name], "delta": delta,
                "direction": "toward" if delta < 0 else "away" if delta > 0 else "equal",
            }
        moved.append({
            "kt": row_key[0], "checkpoint": row_key[1], "field": row_key[2],
            "bit_identical_before": old["bit_identical"],
            "bit_identical_after": new["bit_identical"],
            "unequal_before": old["unequal"], "unequal_after": new["unequal"],
            "first_unequal_index_before": old["first_unequal_index"],
            "first_unequal_index_after": new["first_unequal_index"],
            "metrics": metrics,
        })
    require(not bit_losses, f"previously bit-identical rows left the bar: {bit_losses}")
    return {
        "row_count": len(before),
        "moved_row_count": len(moved),
        "unchanged_row_count": len(before) - len(moved),
        "first_non_bit_before": base_first,
        "first_non_bit_after": candidate_first,
        "bit_identical_losses": bit_losses,
        "moved_rows": moved,
    }


def require_salinity_veto(base: dict, candidate: dict) -> dict:
    """Require the kt=10 stage-3 salinity error not to increase."""

    base_rows, _ = _rows(base)
    candidate_rows, _ = _rows(candidate)
    key = lambda row: (row["kt"], row["checkpoint"], row["field"])
    before = {key(row): row for row in base_rows}
    after = {key(row): row for row in candidate_rows}
    target = (10, "stage3", "S")
    require(target in before and target in after,
            "kt=10 stage-3 salinity row is missing")
    base_max = float(before[target]["max_abs"])
    candidate_max = float(after[target]["max_abs"])
    require(candidate_max <= base_max,
            "kt=10 stage-3 salinity maximum increased: "
            f"{base_max} -> {candidate_max}")
    return {"before_max_abs": base_max, "after_max_abs": candidate_max}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung0-base", type=Path, required=True)
    parser.add_argument("--rung0-candidate", type=Path, required=True)
    parser.add_argument("--rung7-base", type=Path, required=True)
    parser.add_argument("--rung7-candidate", type=Path, required=True)
    parser.add_argument("--plant", choices=("none", "bit-loss", "first-earlier"), default="none")
    parser.add_argument("--salinity-veto", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in (
        args.rung0_base, args.rung0_candidate, args.rung7_base, args.rung7_candidate)]
    if args.plant == "bit-loss":
        documents[1] = copy.deepcopy(documents[1])
        documents[1]["rows"][0]["bit_identical"] = False
        documents[1]["rows"][0]["unequal"] = 1
    elif args.plant == "first-earlier":
        documents[3] = copy.deepcopy(documents[3])
        documents[3]["candidate_trajectory"]["first_non_bit_statement"] = {"kt": 0}
    try:
        result = {
            "status": "PASS_R111_ORCA2_LADDER_COMPARE",
            "rung0": compare(documents[0], documents[1]),
            "rung7": compare(documents[2], documents[3]),
        }
        if args.salinity_veto:
            result["salinity_veto"] = {
                "rung0": require_salinity_veto(documents[0], documents[1]),
                "rung7": require_salinity_veto(documents[2], documents[3]),
            }
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, KeyError, TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
