#!/usr/bin/env python3
"""Register every ladder row moved by Decision 115's filter alpha."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round215_decision96_gate as shared,
)


class GateError(RuntimeError):
    """The before/after trajectory violates the frozen landing predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _label(document: dict, name: str) -> dict:
    require(name in document, f"missing {name} trajectory")
    return document[name]


def compare(before: dict, after: dict, label: str) -> dict:
    """Apply the shared 200-row comparison and Decision-96 vote semantics."""

    result = shared._comparison(_label(before, label), _label(after, label))
    counts = {}
    for metric in ("rms", "max_abs"):
        census = Counter(
            row["metrics"][metric]["direction"]
            for row in result["moved_rows"]
        )
        counts[metric] = {
            name: int(census[name]) for name in ("toward", "away", "equal")
        }
    rms_votes = counts["rms"]["toward"] + counts["rms"]["away"]
    require(rms_votes > 0, f"{label}: alpha statement moved no scored RMS row")
    require(counts["rms"]["toward"] * 2 > rms_votes,
            f"{label}: no strict majority of RMS-moved rows toward NEMO")

    first = (1, "stage1", "T")
    by_key = lambda rows: {
        (row["kt"], row["checkpoint"], row["field"]): row for row in rows
    }
    old = by_key(_label(before, label)["rows"])
    new = by_key(_label(after, label)["rows"])
    first_delta = float(new[first]["rms"]) - float(old[first]["rms"])
    require(first_delta <= 0.0,
            f"{label}: first-over-bar row moved away by {first_delta}")
    return {
        **result,
        "claim_label": label,
        "rms_direction_census": counts["rms"],
        "max_direction_census": counts["max_abs"],
        "strict_majority_toward": True,
        "first_over_bar_rms_direction": (
            "toward" if first_delta < 0.0 else "equal"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--plant", choices=("none", "exact-loss"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        before = json.loads(args.before.read_text(encoding="utf-8"))
        after = json.loads(args.after.read_text(encoding="utf-8"))
        if args.plant == "exact-loss":
            planted = next(
                index for index, row in enumerate(before["independent"]["rows"])
                if row["bit_identical"]
            )
            after["independent"]["rows"][planted]["bit_identical"] = False
            after["independent"]["rows"][planted]["unequal"] = 1
        result = {
            "format": "nemo-testcase-l4-orca2-round237-ladder-compare-v1",
            "status": "PASS_R237_ALPHA_LADDER_COMPARE",
            "before": str(args.before),
            "after": str(args.after),
            "independent": compare(before, after, "independent"),
            "given_nemo_entry": compare(before, after, "given_nemo_entry"),
        }
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, shared.GateError,
            GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
