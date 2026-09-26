#!/usr/bin/env python3
"""Fail-closed comparison of the three ORCA2 Decision-54 component arms."""

from __future__ import annotations

import argparse
import copy
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


def _key(checkpoint: dict, field: str) -> str:
    return f"kt={checkpoint['kt']}:{checkpoint['checkpoint']}:{field}"


def _compare(baseline: dict, arm: dict, name: str) -> dict:
    bcheck = baseline["candidate_trajectory"]["checkpoints"]
    acheck = arm["candidate_trajectory"]["checkpoints"]
    require(len(acheck) in (12, 40), f"{name}: unexpected checkpoint count")
    moved = []
    at_bar_left = []
    for before, after in zip(bcheck, acheck, strict=False):
        require((before["kt"], before["checkpoint"]) ==
                (after["kt"], after["checkpoint"]),
                f"{name}: checkpoint ordering differs")
        for field in ("T", "S", "u", "v", "ssh"):
            brow = before["rows"][field]
            arow = after["rows"][field]
            if brow == arow:
                continue
            row = _key(after, field)
            if brow["bit_identical"] and not arow["bit_identical"]:
                at_bar_left.append(row)
            before_max = float(brow["max_abs"])
            after_max = float(arow["max_abs"])
            direction = ("toward" if after_max < before_max else
                         "away" if after_max > before_max else "same-maximum")
            moved.append({"row": row, "direction": direction,
                          "before": brow, "after": arow})
    same_owner = (
        baseline["candidate_trajectory"]["first_non_bit_statement"] ==
        arm["candidate_trajectory"]["first_non_bit_statement"]
    )
    return {
        "commit": arm["worktree"]["commit"],
        "checkpoint_count": len(acheck),
        "first_moved_row": moved[0]["row"] if moved else None,
        "first_non_bit_statement_unchanged": same_owner,
        "at_bar_rows_left": at_bar_left,
        "moved_row_count": len(moved),
        "toward": sum(row["direction"] == "toward" for row in moved),
        "away": sum(row["direction"] == "away" for row in moved),
        "same_maximum": sum(row["direction"] == "same-maximum" for row in moved),
        "moved_rows": moved,
    }


def run(args: argparse.Namespace) -> dict:
    baseline = _read(args.baseline)
    single_mask = _read(args.single_mask)
    live_thickness = _read(args.live_thickness)
    native_f_metrics = _read(args.native_f_metrics)
    require(args.refusal_log.is_file(), f"missing artifact: {args.refusal_log}")
    refusal_log = args.refusal_log.read_text()

    require(baseline["status"] == "LADDER_MEASURED", "baseline is not measured")
    require(len(baseline["candidate_trajectory"]["checkpoints"]) == 40,
            "baseline is not the ten-step/40-checkpoint ladder")
    require(single_mask["status"] == "LADDER_MEASURED",
            "single-mask arm did not reach ten steps")
    require(native_f_metrics["status"] == "LADDER_MEASURED",
            "native-F-metric arm did not reach ten steps")
    require(live_thickness["status"] == "LADDER_MEASURED",
            "bounded live-thickness artifact did not complete")
    require(len(live_thickness["candidate_trajectory"]["checkpoints"]) == 12,
            "live-thickness artifact is not exactly kt=1..3")
    refusal = "raw-mesh e3w_int must contain only finite values > 0"
    require(refusal in refusal_log, "live-thickness step-4 refusal is absent")
    require("REFUSE: the production step raised an unregistered refusal" in refusal_log,
            "live-thickness log did not fail closed")

    if args.plant:
        single_mask = copy.deepcopy(single_mask)
        row = single_mask["candidate_trajectory"]["checkpoints"][0]["rows"]["T"]
        row.update(bit_identical=False, unequal=1,
                   max_abs=float.fromhex("0x1p-1074"),
                   mean_abs_over_unequal=float.fromhex("0x1p-1074"),
                   first_unequal_index=[0, 0, 0])

    arms = {
        "single_mask": _compare(baseline, single_mask, "single_mask"),
        "live_thickness": _compare(baseline, live_thickness, "live_thickness"),
        "native_f_metrics": _compare(baseline, native_f_metrics,
                                      "native_f_metrics"),
    }
    for name, arm in arms.items():
        require(not arm["at_bar_rows_left"],
                f"{name}: formerly AT-BAR rows left: {arm['at_bar_rows_left']}")
        require(arm["first_non_bit_statement_unchanged"],
                f"{name}: first non-bit statement changed")

    predictions = {
        "R25-P1": all(arm["moved_row_count"] > 0 for arm in arms.values()),
        "R25-P2": all(arm["first_moved_row"] in
                       ("kt=1:stage2:u", "kt=1:stage2:v")
                       for arm in arms.values()),
        "R25-P3": refusal in refusal_log,
        "R25-P4": (arms["single_mask"]["checkpoint_count"] == 40 and
                    arms["native_f_metrics"]["checkpoint_count"] == 40),
        "R25-P5": all(not arm["at_bar_rows_left"] for arm in arms.values()),
        "R25-P6": not args.plant,
    }
    return {
        "status": "HELD",
        "claim_label": "independent with Decision-52 SSH",
        "baseline_commit": baseline["worktree"]["commit"],
        "isolated_instability_owner": "live_thickness",
        "live_thickness_step4_refusal": refusal,
        "arms": arms,
        "predictions": {key: "CONFIRMED" if value else "REFUTED"
                        for key, value in predictions.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--single-mask", type=Path, required=True)
    parser.add_argument("--live-thickness", type=Path, required=True)
    parser.add_argument("--native-f-metrics", type=Path, required=True)
    parser.add_argument("--refusal-log", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
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
