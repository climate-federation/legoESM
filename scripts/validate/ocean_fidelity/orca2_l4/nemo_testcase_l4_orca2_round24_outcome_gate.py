#!/usr/bin/env python3
"""Fail-closed outcome gate for the held ORCA2 Decision-54 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text())


def _row_key(checkpoint: dict, field: str) -> str:
    return f"kt={checkpoint['kt']}:{checkpoint['checkpoint']}:{field}"


def run(args: argparse.Namespace) -> dict:
    before = _read(args.before)
    after = _read(args.after)
    operator = _read(args.operator)
    plant = _read(args.operator_plant)
    require(args.failure_log.is_file(), f"missing artifact: {args.failure_log}")
    failure = args.failure_log.read_text()

    bcheck = before["candidate_trajectory"]["checkpoints"]
    acheck = after["candidate_trajectory"]["checkpoints"]
    require(len(bcheck) == 40, "baseline is not the admitted ten-step/40-row ladder")
    require(len(acheck) == 12, "bounded experiment is not exactly kt=1..3/12 checkpoints")
    require(operator["status"] == "AT-BAR", "compiled operator replay is not AT-BAR")
    require(operator["u_momentum"]["unequal"] == 0, "operator U replay is non-bit")
    require(operator["v_momentum"]["unequal"] == 0, "operator V replay is non-bit")
    require(plant["status"] == "DEBT", "operator plant did not fire")
    require(plant["u_momentum"]["unequal"] == 1, "operator plant moved != 1 U cell")
    refusal = "raw-mesh e3w_int must contain only finite values > 0"
    require(refusal in failure, "ten-step log lacks the registered step-4 refusal")
    require("REFUSE: the production step raised an unregistered refusal" in failure,
            "ladder did not fail closed on the step-4 refusal")

    fields = ("T", "S", "u", "v", "ssh")
    moved: list[dict] = []
    at_bar_left: list[str] = []
    for b, a in zip(bcheck, acheck, strict=False):
        require((b["kt"], b["checkpoint"]) == (a["kt"], a["checkpoint"]),
                "baseline/experiment checkpoint ordering differs")
        for field in fields:
            brow = b["rows"][field]
            arow = a["rows"][field]
            if brow == arow:
                continue
            key = _row_key(a, field)
            if brow["bit_identical"] and not arow["bit_identical"]:
                at_bar_left.append(key)
            before_error = float(brow["max_abs"])
            after_error = float(arow["max_abs"])
            direction = ("toward" if after_error < before_error else
                         "away" if after_error > before_error else "same-maximum")
            moved.append({
                "row": key,
                "direction": direction,
                "before": brow,
                "after": arow,
            })

    require(moved, "Decision-54 experiment is vacuous")
    first_moved = moved[0]["row"]
    first_before = before["candidate_trajectory"]["first_non_bit_statement"]
    first_after = after["candidate_trajectory"]["first_non_bit_statement"]
    same_owner = first_before == first_after
    predictions = {
        "R24-P2": operator["status"] == "AT-BAR",
        "R24-P3": first_moved in ("kt=2:stage3:u", "kt=2:stage3:v"),
        "R24-P4": same_owner,
        "R24-P7": plant["status"] == "DEBT",
    }
    require(not at_bar_left, f"formerly AT-BAR rows left the bar: {at_bar_left}")
    require(same_owner, "first non-bit statement changed")

    # A three-step artifact plus a fail-closed step-4 refusal can only be HELD.
    return {
        "status": "HELD",
        "claim_label": "independent with Decision-52 SSH",
        "base_commit": args.base_commit,
        "experiment_commit": operator["worktree"]["commit"],
        "operator_replay": {
            "claim_label": operator["claim_label"],
            "status": operator["status"],
            "u_unequal": operator["u_momentum"]["unequal"],
            "v_unequal": operator["v_momentum"]["unequal"],
        },
        "observable_ladder_extent": {"steps": 3, "checkpoints": len(acheck)},
        "ten_step_refusal": {"kt": 4, "message": refusal},
        "first_moved_row": first_moved,
        "first_non_bit_statement_unchanged": same_owner,
        "at_bar_rows_left": at_bar_left,
        "moved_rows": moved,
        "moved_row_count": len(moved),
        "predictions": {
            key: "CONFIRMED" if value else "REFUTED"
            for key, value in predictions.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--operator-plant", type=Path, required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
    except (GateError, KeyError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
