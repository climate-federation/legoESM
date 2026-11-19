#!/usr/bin/env python3
"""Register every aggregate SMT-3 -> SMT-4 ladder movement.

The two cards have different physical decks and different NEMO references, so
"toward" means only that each card's own normalized residual is smaller.  It
is not a candidate/control claim.  All 50 keyed rows must be present; the
``--plant-missing`` control removes one and deliberately exits nonzero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def keyed(report: dict) -> dict[tuple[int, str], dict]:
    rows = {}
    for step in report["steps"]:
        kt = int(step["kt"])
        for row in step["rows"]:
            field = row["name"].rsplit(".", 1)[-1]
            key = (kt, field)
            if key in rows:
                raise ValueError(f"duplicate row {key}")
            rows[key] = row
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant-missing", action="store_true")
    args = parser.parse_args(argv)
    before = keyed(json.loads(args.before.read_text()))
    after = keyed(json.loads(args.after.read_text()))
    if args.plant_missing:
        after.pop(sorted(after)[-1], None)
    expected = {(kt, field) for kt in range(1, 11)
                for field in ("T", "S", "u", "v", "ssh")}
    if set(before) != expected or set(after) != expected:
        print("STATUS PLANT-FIRED" if args.plant_missing else "REFUSE")
        print("row-key mismatch", sorted(expected - set(before)),
              sorted(expected - set(after)))
        return 1
    rows = []
    for kt, field in sorted(expected):
        old = float(before[(kt, field)]["normalized_max_abs"])
        new = float(after[(kt, field)]["normalized_max_abs"])
        rows.append({
            "kt": kt, "field": field,
            "smt3_normalized_max_abs": old,
            "smt4_normalized_max_abs": new,
            "change": new - old,
            "direction": "equal" if new == old else (
                "toward-own-oracle" if new < old else "away-from-own-oracle"),
            "smt3_status": before[(kt, field)]["status"],
            "smt4_status": after[(kt, field)]["status"],
        })
    report = {
        "format": "nemo-testcase-l1-vortex-smt-round238-registry-compare-v1",
        "before": str(args.before), "after": str(args.after),
        "rows": rows,
        "moved_rows": sum(row["change"] != 0.0 for row in rows),
        "toward_rows": sum(row["direction"] == "toward-own-oracle"
                           for row in rows),
        "away_rows": sum(row["direction"] == "away-from-own-oracle"
                         for row in rows),
        "status_changes": [row for row in rows
                           if row["smt3_status"] != row["smt4_status"]],
        "status": "PASS",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"REGISTERED {report['moved_rows']}/50 moved; "
          f"{report['toward_rows']} toward own oracle, "
          f"{report['away_rows']} away; "
          f"{len(report['status_changes'])} status changes")
    print("STATUS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
