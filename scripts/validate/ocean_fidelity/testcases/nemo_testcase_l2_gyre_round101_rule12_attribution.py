#!/usr/bin/env python3
"""Compare the independently registered Round-97 and Round-99 Rule-12 arms.

This is an attribution report, not an acceptance gate.  It proves exactly
which registered rows changed when the direct-W/ratio/clock members were
added to the full-RHS member.  Both inputs are immutable, digest-pinned Rule-12
reports, and every one of their 954 rows must be present in the same order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp


LEFT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round97/"
    "ladder_rule12.json")
RIGHT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round99/"
    "ladder_rule12_b01e550a.json")
LEFT_SHA256 = "1d2bdd640ffcdf5e12b056913ad8d9a972910376736dfc419de28ab8e66c45d8"
RIGHT_SHA256 = "78ef102e2d4f584225e238fef66cf64ada837f9c3cd0a206193a3254d259a1e0"
ROW_RE = re.compile(r"^GYRE-zco\.kt(?P<kt>\d+)\..*\.(?P<field>[^.]+)$")


class AttributionError(RuntimeError):
    """A pinned input or row-set invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttributionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(report: dict) -> dict:
    rows = report["field_moves"]
    moved = [
        row["row"] for row in rows
        if row["n_improved_cells"] or row["n_worsened_cells"]
    ]
    violating = [
        row["row"] for row in rows if row["n_cells_worse_than_bar"]
    ]
    return {
        "status": report["status"],
        "certified_rows": report["n_certified_rows_compared"],
        "moved_rows": len(moved),
        "violating_rows": len(violating),
        "moved_row_names": moved,
        "violating_row_names": violating,
        "row_status_changes": len(report["row_status_changes"]),
        "first_over_bar_reference": report["first_over_bar_reference"],
        "first_over_bar_candidate": report["first_over_bar_candidate"],
    }


def compare(left: dict, right: dict, *, plant: str | None = None) -> dict:
    require(left.get("format") == "legoesm-ocean-oracle-relative-move-gate-v3",
            "left input has the wrong Rule-12 schema")
    require(right.get("format") == "legoesm-ocean-oracle-relative-move-gate-v3",
            "right input has the wrong Rule-12 schema")
    left_rows = list(left["field_moves"])
    right_rows = list(right["field_moves"])
    if plant == "row-drop":
        right_rows = right_rows[:-1]
    require(len(left_rows) == len(right_rows) == 954,
            "the comparison does not contain exactly 954 rows per arm")
    left_names = [row["row"] for row in left_rows]
    right_names = [row["row"] for row in right_rows]
    require(left_names == right_names,
            "the two Rule-12 reports do not have the same ordered row set")
    require(len(set(left_names)) == 954, "the Rule-12 row set is not unique")

    left_summary = _summary({**left, "field_moves": left_rows})
    right_summary = _summary({**right, "field_moves": right_rows})
    require(left_summary["status"] == right_summary["status"] == "FAIL",
            "both immutable arms must retain their Rule-12 refusal")
    require(left_summary["moved_rows"] == right_summary["moved_rows"] == 85,
            "the registered 85-row movement count changed")
    require(left_summary["violating_rows"]
            == right_summary["violating_rows"] == 56,
            "the registered 56-row violation count changed")
    require(left_summary["moved_row_names"]
            == right_summary["moved_row_names"],
            "adding W/ratio/clock changed which rows moved")
    require(left_summary["violating_row_names"]
            == right_summary["violating_row_names"],
            "adding W/ratio/clock changed which rows violate Rule 12")
    require(left_summary["row_status_changes"]
            == right_summary["row_status_changes"] == 0,
            "a report contains an unregistered row-status change")
    require(left_summary["first_over_bar_candidate"]
            == right_summary["first_over_bar_candidate"]
            == {"fields": ["u", "v"], "kt": 2},
            "first-over-bar moved from kt2 U/V")

    differences = []
    by_kt: dict[str, int] = {}
    by_field: dict[str, int] = {}
    for before, after in zip(left_rows, right_rows, strict=True):
        changed = {
            key: {"full_rhs_only": before.get(key),
                  "full_rhs_w_ratio_clock": after.get(key)}
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        }
        if not changed:
            continue
        match = ROW_RE.fullmatch(before["row"])
        require(match is not None, f"cannot parse row name {before['row']!r}")
        kt = match.group("kt")
        field = match.group("field")
        by_kt[kt] = by_kt.get(kt, 0) + 1
        by_field[field] = by_field.get(field, 0) + 1
        differences.append({"row": before["row"], "changed_metrics": changed})

    if plant == "metric":
        differences[0]["changed_metrics"].clear()
    require(len(differences) == 21,
            "the registered W/ratio/clock attribution no longer has 21 rows")
    require(all(row["changed_metrics"] for row in differences),
            "a changed-row entry has no changed metric")
    require(by_kt == {"7": 1, "8": 6, "9": 7, "10": 7},
            f"changed rows moved outside the registered late-step set: {by_kt}")
    return {
        "format": "nemo-testcase-l2-gyre-round101-rule12-attribution-v1",
        "worktree": worktree_stamp(),
        "left_label": "Round 97 full RHS only",
        "right_label": "Round 99 full RHS plus W/ratio/clock",
        "left_summary": left_summary,
        "right_summary": right_summary,
        "same_ordered_954_row_set": True,
        "same_moved_row_set": True,
        "same_violating_row_set": True,
        "changed_row_count": len(differences),
        "changed_rows_by_kt": by_kt,
        "changed_rows_by_field": by_field,
        "changed_rows": differences,
        "interpretation": (
            "W/ratio/clock changes metrics in 21 late rows but does not change "
            "the moved-row set, violating-row set, classification, or kt2 U/V "
            "first-over-bar established by full RHS alone."),
        "status": "PASS",
    }


def run(*, plant: str | None = None) -> dict:
    require(LEFT.is_file(), f"missing pinned left input {LEFT}")
    require(RIGHT.is_file(), f"missing pinned right input {RIGHT}")
    require(sha256(LEFT) == LEFT_SHA256, "Round-97 input digest moved")
    require(sha256(RIGHT) == RIGHT_SHA256, "Round-99 input digest moved")
    left = json.loads(LEFT.read_text(encoding="utf-8"))
    right = json.loads(RIGHT.read_text(encoding="utf-8"))
    report = compare(left, right, plant=plant)
    report["inputs"] = {
        "left": {"path": str(LEFT), "sha256": LEFT_SHA256},
        "right": {"path": str(RIGHT), "sha256": RIGHT_SHA256},
    }
    report["worktree"] = worktree_stamp()
    report["plant"] = plant
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("row-drop", "metric"))
    args = parser.parse_args(argv)
    try:
        report = run(plant=args.plant)
    except (AttributionError, KeyError, TypeError, ValueError) as error:
        report = {
            "format": "nemo-testcase-l2-gyre-round101-rule12-attribution-v1",
            "plant": args.plant,
            "status": "FAIL",
            "error": str(error),
            "worktree": worktree_stamp(),
        }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "PASS" and args.plant is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
