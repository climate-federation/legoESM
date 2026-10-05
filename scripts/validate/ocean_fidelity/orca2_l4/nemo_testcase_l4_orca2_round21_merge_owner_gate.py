#!/usr/bin/env python3
"""Classify the preregistered round-21 ORCA2 merge-owner ladders."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp

FIELDS = ("T", "S", "u", "v", "ssh")
CHECKPOINTS = ("entry", "stage1", "stage2", "stage3")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _rows(document: dict) -> dict[tuple[int, str, str], dict]:
    checkpoints = document["candidate_trajectory"]["checkpoints"]
    rows = {
        (int(item["kt"]), item["checkpoint"], field): item["rows"][field]
        for item in checkpoints
        for field in FIELDS
    }
    require(len(checkpoints) == 40, "a ladder does not contain 40 checkpoints")
    require(len(rows) == 200, "a ladder does not contain 200 scored rows")
    return rows


def _row_differences(left: dict, right: dict) -> list[tuple[int, str, str]]:
    return [key for key in left if left[key] != right[key]]


def _ordered(keys):
    order = {name: index for index, name in enumerate(CHECKPOINTS)}
    field_order = {name: index for index, name in enumerate(FIELDS)}
    return sorted(keys, key=lambda key: (key[0], order[key[1]], field_order[key[2]]))


def _row_summary(key, reference, candidate) -> dict:
    return {
        "kt": key[0],
        "checkpoint": key[1],
        "field": key[2],
        "round20": reference[key],
        "candidate": candidate[key],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(documents: dict[str, dict], *, packages_changed=()) -> dict:
    required = {"round20", "merge", "baseline", "fold", "e3f", "both", "plant"}
    require(set(documents) == required, "wrong document set")
    for name, document in documents.items():
        require(document["status"] == "LADDER_MEASURED",
                f"{name}: ladder status is not LADDER_MEASURED")

    rows = {name: _rows(document) for name, document in documents.items()}
    round20 = rows["round20"]
    baseline = rows["baseline"]
    moved = {
        name: _ordered(_row_differences(candidate, round20))
        for name, candidate in rows.items()
        if name not in ("round20", "merge")
    }
    changed_from_baseline = {
        name: _ordered(_row_differences(candidate, baseline))
        for name, candidate in rows.items()
        if name in ("fold", "e3f", "both", "plant")
    }

    def target(name, kt, checkpoint, field):
        return rows[name][(kt, checkpoint, field)]["max_abs"]

    target_values = {
        name: {
            "kt1_stage2_u": target(name, 1, "stage2", "u"),
            "kt1_stage2_v": target(name, 1, "stage2", "v"),
            "kt10_entry_T": target(name, 10, "entry", "T"),
        }
        for name in ("round20", "baseline", "fold", "e3f", "both", "plant")
    }

    baseline_moved = moved["baseline"]
    direction = {}
    for name in ("fold", "e3f", "both"):
        counts = {"toward": 0, "farther": 0, "unchanged_distance": 0,
                  "restored_exact_row": 0}
        for key in baseline_moved:
            old = round20[key]
            initial = abs(baseline[key]["max_abs"] - old["max_abs"])
            after = abs(rows[name][key]["max_abs"] - old["max_abs"])
            if rows[name][key] == old:
                counts["restored_exact_row"] += 1
            elif after < initial:
                counts["toward"] += 1
            elif after > initial:
                counts["farther"] += 1
            else:
                counts["unchanged_distance"] += 1
        direction[name] = counts

    combined_residual = moved["both"]
    first_residual = (
        _row_summary(combined_residual[0], round20, rows["both"])
        if combined_residual else None
    )
    largest_key = (
        max(combined_residual, key=lambda key: abs(
            rows["both"][key]["max_abs"] - round20[key]["max_abs"]))
        if combined_residual else None
    )
    largest_residual = (
        _row_summary(largest_key, round20, rows["both"])
        if largest_key else None
    )

    def context(document):
        return {key: value for key, value in document.items()
                if key not in ("candidate_trajectory", "worktree")}

    context_equal = all(
        context(documents[name]) == context(documents["baseline"])
        for name in ("fold", "e3f", "both", "plant")
    )
    baseline_matches_merge = (
        documents["baseline"]["candidate_trajectory"]
        == documents["merge"]["candidate_trajectory"]
    )
    plant_fires = bool(_row_differences(rows["plant"], rows["both"]))

    predictions = {
        "R21-P1": baseline_matches_merge and target_values["baseline"] == {
            "kt1_stage2_u": 0.06463349988292608,
            "kt1_stage2_v": 0.03401471577804818,
            "kt10_entry_T": 3.947126188631776,
        },
        "R21-P2": (
            format(target_values["fold"]["kt1_stage2_u"], ".14g")
            == "0.064633491838392" and bool(moved["fold"])
        ),
        "R21-P3": (
            bool(changed_from_baseline["e3f"])
            and direction["e3f"]["toward"] > 0
            and all(not (key[0] == 1 and key[1] in ("entry", "stage1"))
                    for key in changed_from_baseline["e3f"])
        ),
        "R21-P4": not combined_residual,
        "R21-P5": context_equal and plant_fires,
        "R21-P6": not packages_changed,
    }
    verdicts = {
        name: ("CONFIRMED" if value else "REFUTED")
        for name, value in predictions.items()
    }
    recertified = predictions["R21-P4"]
    return {
        "status": "RECERTIFIED" if recertified else "HELD_THIRD_OWNER",
        "claim_label": "INDEPENDENT_WITH_DECISION52_SSH",
        "rows_moved_from_round20": {
            name: len(keys) for name, keys in moved.items()
        },
        "rows_changed_from_merged_baseline": {
            name: len(keys) for name, keys in changed_from_baseline.items()
        },
        "target_values": target_values,
        "direction_on_185_registered_rows": direction,
        "first_combined_residual": first_residual,
        "largest_combined_residual_by_max_abs_change": largest_residual,
        "baseline_matches_merge_candidate_trajectory": baseline_matches_merge,
        "control_contexts_equal": context_equal,
        "plant_rows_changed_from_combined": len(
            _row_differences(rows["plant"], rows["both"])),
        "packages_changed": list(packages_changed),
        "predictions": verdicts,
        "decision58_and_54_eligible": recertified,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("round20", "merge", "baseline", "fold", "e3f", "both", "plant"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        documents = {
            name: json.loads(getattr(args, name).read_text())
            for name in ("round20", "merge", "baseline", "fold", "e3f", "both", "plant")
        }
        changed = subprocess.run(
            ["git", "diff", "--name-only", f"{args.base_commit}..HEAD", "--", "packages"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        result = analyze(documents, packages_changed=changed)
        result["input_documents"] = {
            name: {"path": str(getattr(args, name)),
                   "sha256": _sha256(getattr(args, name))}
            for name in ("round20", "merge", "baseline", "fold", "e3f", "both", "plant")
        }
        result["worktree"] = worktree_stamp()
    except (GateError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"REFUSE: {exc}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 0 if result["status"] == "RECERTIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
