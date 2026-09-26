#!/usr/bin/env python3
"""Gate Decision 58's ORCA2 second-continuity-solve card change."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round21_merge_owner_gate as ladder_gate,
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered(keys):
    checkpoint = {name: index for index, name in enumerate(ladder_gate.CHECKPOINTS)}
    field = {name: index for index, name in enumerate(ladder_gate.FIELDS)}
    return sorted(keys, key=lambda key: (key[0], checkpoint[key[1]], field[key[2]]))


def _residuals_equal(before: Path, after: Path) -> tuple[bool, int]:
    with np.load(before, allow_pickle=False) as left, np.load(
            after, allow_pickle=False) as right:
        require(left.files == right.files, "GYRE residual key order changed")
        unequal = sum(
            not np.array_equal(left[name], right[name]) for name in left.files)
        return unequal == 0, len(left.files)


def _daily_equal(before: Path, after: Path) -> tuple[bool, int]:
    left = sorted(before.glob("day*.npz"))
    right = sorted(after.glob("day*.npz"))
    require(len(left) == len(right) == 30, "expected 30 daily snapshots per arm")
    require([path.name for path in left] == [path.name for path in right],
            "GYRE daily snapshot names changed")
    equal = all(_sha256(a) == _sha256(b) for a, b in zip(left, right))
    return equal, len(left)


def analyze(
    before: dict,
    after: dict,
    gyre_comparison: dict,
    *,
    residuals_equal: bool,
    residual_arrays: int,
    daily_equal: bool,
    daily_snapshots: int,
    package_diff_exact: bool,
) -> dict:
    for name, document in (("before", before), ("after", after)):
        require(document["status"] == "LADDER_MEASURED",
                f"{name}: ORCA2 ladder is not measured")
        require(document["trajectory_claim"] ==
                "MEASURED_INDEPENDENT_WITH_DECISION52_SSH",
                f"{name}: wrong claim label")

    before_rows = ladder_gate._rows(before)
    after_rows = ladder_gate._rows(after)
    moved = _ordered(ladder_gate._row_differences(before_rows, after_rows))
    require(moved, "Decision 58 is inert on all 200 ORCA2 rows")

    kt1_early_moved = [
        key for key in moved
        if key[0] == 1 and key[1] in ("entry", "stage1")
    ]
    protected_losses = [
        key for key in moved
        if key[1] in ("entry", "stage1")
        and before_rows[key]["bit_identical"]
        and not after_rows[key]["bit_identical"]
    ]
    first = moved[0]
    direction = {"toward_nemo": 0, "away_from_nemo": 0, "same_max": 0}
    for key in moved:
        old = before_rows[key]["max_abs"]
        new = after_rows[key]["max_abs"]
        if new < old:
            direction["toward_nemo"] += 1
        elif new > old:
            direction["away_from_nemo"] += 1
        else:
            direction["same_max"] += 1

    before_first = before["candidate_trajectory"]["first_non_bit_statement"]
    after_first = after["candidate_trajectory"]["first_non_bit_statement"]
    compare_pass = (
        gyre_comparison.get("status") == "PASS"
        and gyre_comparison.get("violations") == []
        and gyre_comparison.get("n_certified_rows_compared") == 70
        and gyre_comparison.get("largest_oracle_residual_worsening_ulps") == 0
    )
    predictions = {
        "R23-P1": package_diff_exact,
        "R23-P2": (
            not kt1_early_moved
            and first[0] == 1
            and first[1] == "stage2"
            and first[2] in ("u", "v")
        ),
        "R23-P3": before_first == after_first,
        "R23-P4": bool(moved),
        "R23-P5": (
            compare_pass and residuals_equal and residual_arrays == 210
            and daily_equal and daily_snapshots == 30
        ),
    }
    binding = predictions["R23-P1"] and predictions["R23-P4"] and predictions["R23-P5"]
    safe_boundary = (
        not kt1_early_moved
        and not protected_losses
        and before_first == after_first
    )
    status = "LANDED" if binding and safe_boundary else "HELD"
    return {
        "status": status,
        "claim_label": "INDEPENDENT_WITH_DECISION52_SSH",
        "orca2_rows_moved": len(moved),
        "first_moved_row": {
            "kt": first[0], "checkpoint": first[1], "field": first[2],
            "before": before_rows[first], "after": after_rows[first],
        },
        "kt1_entry_or_stage1_rows_moved": [
            list(key) for key in kt1_early_moved],
        "formerly_bit_identical_entry_or_stage1_rows_lost": [
            list(key) for key in protected_losses],
        "direction_by_max_abs": direction,
        "first_non_bit_statement_unchanged": before_first == after_first,
        "first_non_bit_statement": after_first,
        "gyre": {
            "offline_compare_pass": compare_pass,
            "residual_arrays_equal": residuals_equal,
            "residual_arrays": residual_arrays,
            "daily_snapshots_equal": daily_equal,
            "daily_snapshots": daily_snapshots,
        },
        "predictions": {
            key: ("CONFIRMED" if value else "REFUTED")
            for key, value in predictions.items()
        },
    }


def _package_diff_is_exact(base_commit: str) -> bool:
    result = subprocess.run(
        ["git", "diff", "--unified=0", f"{base_commit}..HEAD", "--",
         "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py"],
        check=True, capture_output=True, text=True,
    )
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{base_commit}..HEAD", "--", "packages"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    code_lines = []
    for line in result.stdout.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        body = line[1:].strip()
        if body and not body.startswith("#"):
            code_lines.append(line)
    return changed == [
        "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py",
        "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py",
        "packages/ocean/legoesm/ocean/state.py",
    ] and code_lines == [
        "-            nemo_stage_momentum_wzv_split=False,",
        "+            nemo_stage_momentum_wzv_split=True,",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--gyre-comparison", type=Path, required=True)
    parser.add_argument("--before-residuals", type=Path, required=True)
    parser.add_argument("--after-residuals", type=Path, required=True)
    parser.add_argument("--before-daily", type=Path, required=True)
    parser.add_argument("--after-daily", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        before = json.loads(args.before.read_text())
        after = json.loads(args.after.read_text())
        if args.plant:
            row = after["candidate_trajectory"]["checkpoints"][0]["rows"]["T"]
            row["max_abs"] = float(np.nextafter(row["max_abs"], np.inf))
        residual_equal, residual_count = _residuals_equal(
            args.before_residuals, args.after_residuals)
        daily_equal, daily_count = _daily_equal(args.before_daily, args.after_daily)
        result = analyze(
            before, after, json.loads(args.gyre_comparison.read_text()),
            residuals_equal=residual_equal,
            residual_arrays=residual_count,
            daily_equal=daily_equal,
            daily_snapshots=daily_count,
            package_diff_exact=_package_diff_is_exact(args.base_commit),
        )
        result["worktree"] = worktree_stamp()
    except (GateError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"REFUSE: {exc}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 0 if result["status"] == "LANDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
