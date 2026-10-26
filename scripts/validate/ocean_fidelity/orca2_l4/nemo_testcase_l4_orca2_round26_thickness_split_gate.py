#!/usr/bin/env python3
"""Fail-closed outcome gate for the round-26 ORCA2 LDF thickness split."""

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


def _row_key(checkpoint: dict, field: str) -> str:
    return f"kt={checkpoint['kt']}:{checkpoint['checkpoint']}:{field}"


def _compare(baseline: dict, arm: dict, name: str) -> dict:
    before = baseline["candidate_trajectory"]["checkpoints"]
    after = arm["candidate_trajectory"]["checkpoints"]
    moved = []
    at_bar_left = []
    for bcp, acp in zip(before, after, strict=False):
        require((bcp["kt"], bcp["checkpoint"]) ==
                (acp["kt"], acp["checkpoint"]),
                f"{name}: checkpoint ordering differs")
        for field in ("T", "S", "u", "v", "ssh"):
            brow = bcp["rows"][field]
            arow = acp["rows"][field]
            if brow == arow:
                continue
            key = _row_key(acp, field)
            if brow["bit_identical"] and not arow["bit_identical"]:
                at_bar_left.append(key)
            bmax = float(brow["max_abs"])
            amax = float(arow["max_abs"])
            direction = ("toward" if amax < bmax else
                         "away" if amax > bmax else "same-maximum")
            moved.append({"row": key, "direction": direction,
                          "before": brow, "after": arow})
    return {
        "commit": arm["worktree"]["commit"],
        "checkpoint_count": len(after),
        "first_moved_row": moved[0]["row"] if moved else None,
        "first_non_bit_statement_unchanged": (
            baseline["candidate_trajectory"]["first_non_bit_statement"] ==
            arm["candidate_trajectory"]["first_non_bit_statement"]),
        "at_bar_rows_left": at_bar_left,
        "moved_row_count": len(moved),
        "toward": sum(row["direction"] == "toward" for row in moved),
        "away": sum(row["direction"] == "away" for row in moved),
        "same_maximum": sum(row["direction"] == "same-maximum"
                            for row in moved),
        "moved_rows": moved,
    }


def _score(document: dict, kt: int, checkpoint: str, field: str) -> dict:
    for row in document["candidate_trajectory"]["checkpoints"]:
        if row["kt"] == kt and row["checkpoint"] == checkpoint:
            return row["rows"][field]
    raise GateError(f"missing score kt={kt}:{checkpoint}:{field}")


def run(args: argparse.Namespace) -> dict:
    baseline = _read(args.baseline)
    f_curl = _read(args.f_curl)
    kbb = _read(args.kbb_divergence)
    kmm = _read(args.kmm_divisor)
    combined = _read(args.combined)
    require(args.f_curl_refusal.is_file(),
            f"missing artifact: {args.f_curl_refusal}")
    refusal_log = args.f_curl_refusal.read_text()

    for name, document, checkpoints in (
        ("baseline", baseline, 40), ("f_curl", f_curl, 12),
        ("kbb_divergence", kbb, 40), ("kmm_divisor", kmm, 40),
    ):
        require(document["status"] == "LADDER_MEASURED",
                f"{name}: ladder is not measured")
        require(len(document["candidate_trajectory"]["checkpoints"]) == checkpoints,
                f"{name}: expected {checkpoints} checkpoints")
        require(document["worktree"]["clean"],
                f"{name}: worktree stamp is dirty")
        require(document["card"] == baseline["card"],
                f"{name}: resolved card differs from baseline")

    refusal = "raw-mesh e3w_int must contain only finite values > 0"
    require(refusal in refusal_log, "F-curl kt=4 refusal is absent")
    require("REFUSE: the production step raised an unregistered refusal" in
            refusal_log, "F-curl run did not fail closed")

    if args.plant:
        kbb = copy.deepcopy(kbb)
        row = kbb["candidate_trajectory"]["checkpoints"][0]["rows"]["T"]
        row.update(bit_identical=False, unequal=1,
                   max_abs=float.fromhex("0x1p-1074"),
                   mean_abs_over_unequal=float.fromhex("0x1p-1074"),
                   first_unequal_index=[0, 0, 0])

    arms = {
        "f_curl": _compare(baseline, f_curl, "f_curl"),
        "kbb_divergence": _compare(baseline, kbb, "kbb_divergence"),
        "kmm_divisor": _compare(baseline, kmm, "kmm_divisor"),
    }
    for name, arm in arms.items():
        require(not arm["at_bar_rows_left"],
                f"{name}: formerly AT-BAR rows left: {arm['at_bar_rows_left']}")
        require(arm["first_non_bit_statement_unchanged"],
                f"{name}: first non-bit statement changed")

    combined_u = _score(combined, 1, "stage2", "u")
    combined_v = _score(combined, 1, "stage2", "v")
    recomposes = (
        float(combined_u["max_abs"]) == 0.10080009966621735 and
        float(combined_v["max_abs"]) == 0.11456680400114852
    )
    predictions = {
        "R26-P1": all(document["card"] == baseline["card"]
                       for document in (f_curl, kbb, kmm)),
        "R26-P2": all(arm["first_moved_row"] in
                       ("kt=1:stage2:u", "kt=1:stage2:v")
                       for arm in arms.values()),
        "R26-P3": False,
        "R26-P4": all(not arm["at_bar_rows_left"] for arm in arms.values()),
        "R26-P5": recomposes,
        "R26-P6": not args.plant,
    }
    return {
        "status": "HELD",
        "claim_label": "independent with Decision-52 SSH",
        "isolated_refusal_owner": "f_curl",
        "f_curl_step4_refusal": refusal,
        "kbb_kmm_score_documents_identical": (
            kbb["candidate_trajectory"] == kmm["candidate_trajectory"]),
        "combined_round25_first_movement": {"u": combined_u, "v": combined_v},
        "arms": arms,
        "predictions": {key: "CONFIRMED" if value else "REFUTED"
                        for key, value in predictions.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--f-curl", type=Path, required=True)
    parser.add_argument("--f-curl-refusal", type=Path, required=True)
    parser.add_argument("--kbb-divergence", type=Path, required=True)
    parser.add_argument("--kmm-divisor", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
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
