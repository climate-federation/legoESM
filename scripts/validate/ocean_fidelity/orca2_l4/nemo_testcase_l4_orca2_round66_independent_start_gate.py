#!/usr/bin/env python3
"""Classify Decision 52's independent-start ORCA2 ten-step ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp


FIELD_ORDER = ("T", "S", "u", "v", "ssh")
CHECKPOINT_ORDER = ("entry", "stage1", "stage2", "stage3")
EXPECTED_UNMEASURED = (
    "staged_gm_eiv",
    "linear_implicit_bottom_drag",
    "spatial_lateral_viscosity",
    "freshwater_budget_carry",
    "si3_jpl5_layered_prather_state",
)
SSH_CITATION = (
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/"
    "iceistate.f90:440-465"
)
EXPECTED_ENTRY_SSH = {
    "unequal": 16433,
    "count": 26640,
    "max_abs": 0.015479333813968585,
}


class GateError(RuntimeError):
    """The independent-start result violated a frozen round-66 predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_row(row: dict[str, object], label: str) -> None:
    require(int(row["count"]) > 0, f"{label}: empty row")
    require(0 <= int(row["unequal"]) <= int(row["count"]),
            f"{label}: invalid unequal count")
    for metric in ("max_abs", "mean_abs_over_unequal", "rms"):
        require(math.isfinite(float(row[metric])),
                f"{label}: non-finite {metric}")
    exact = int(row["unequal"]) == 0
    require(bool(row["bit_identical"]) == exact,
            f"{label}: bit flag/count disagreement")
    if exact:
        require(float(row["max_abs"]) == 0.0 and float(row["rms"]) == 0.0,
                f"{label}: exact row has a nonzero metric")


def classify(ladder: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    """Validate a full ladder result and return its compact headline rows."""
    ladder = json.loads(json.dumps(ladder))
    trajectory = ladder["candidate_trajectory"]
    if plant == "bridge":
        trajectory["initial_mode"] = "decision52_ssh_bridge"
        trajectory["decision52_bridge"] = {"planted": True}
    elif plant == "entry-count":
        trajectory["executed_initial_state_vs_nemo"]["rows"]["ssh"][
            "unequal"
        ] += 1
    elif plant == "checkpoint-drop":
        trajectory["checkpoints"].pop()

    require(ladder.get("status") == "LADDER_MEASURED",
            "independent ladder did not complete")
    require(ladder.get("trajectory_claim") ==
            "MEASURED_INDEPENDENT_CARD_OWN_STATE",
            "trajectory claim is not independent card-own-state")
    require(tuple(ladder.get("unmeasured_features", ())) == EXPECTED_UNMEASURED,
            "sea-ice/unmeasured feature registry changed")
    require(trajectory.get("claim_label") == "INDEPENDENT",
            "trajectory label is not INDEPENDENT")
    require(trajectory.get("initial_mode") == "card_own_state",
            "trajectory did not execute the card's own initial state")
    require(trajectory.get("decision52_bridge") is None,
            "independent trajectory applied the Decision-52 SSH bridge")
    require(trajectory.get("execution") ==
            "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")

    initial = trajectory["executed_initial_state_vs_nemo"]
    for field in FIELD_ORDER[:-1]:
        row = initial["rows"][field]
        _finite_row(row, f"initial {field}")
        require(bool(row["bit_identical"]) and int(row["unequal"]) == 0,
                f"independent initial {field} is non-bit")
    ssh = initial["rows"]["ssh"]
    _finite_row(ssh, "initial ssh")
    for key, expected in EXPECTED_ENTRY_SSH.items():
        require(ssh[key] == expected,
                f"initial ssh {key} changed: {ssh[key]} != {expected}")

    first = trajectory["first_non_bit_statement"]
    require((first.get("kt"), first.get("checkpoint"), first.get("field")) ==
            (1, "entry", "ssh"),
            "first non-bit checkpoint is not kt=1 entry SSH")
    require(first.get("source_citation") == SSH_CITATION,
            "first non-bit SSH citation changed")

    checkpoints = trajectory["checkpoints"]
    expected_positions = [
        (kt, checkpoint)
        for kt in range(1, 11)
        for checkpoint in CHECKPOINT_ORDER
    ]
    require(len(checkpoints) == 40, "independent ladder is not 40 checkpoints")
    require([(int(row["kt"]), row["checkpoint"]) for row in checkpoints]
            == expected_positions,
            "independent checkpoint order changed")

    rows_scored = 0
    for checkpoint in checkpoints:
        require(len(checkpoint["rows"]) == len(FIELD_ORDER) and
                set(checkpoint["rows"]) == set(FIELD_ORDER),
                "checkpoint field registry changed")
        for field in FIELD_ORDER:
            _finite_row(
                checkpoint["rows"][field],
                f"kt{checkpoint['kt']} {checkpoint['checkpoint']} {field}",
            )
            rows_scored += 1
    require(rows_scored == 200, "independent ladder did not score 200 rows")

    kt10 = {
        checkpoint["checkpoint"]: checkpoint["rows"]
        for checkpoint in checkpoints
        if int(checkpoint["kt"]) == 10
    }
    require(tuple(kt10) == CHECKPOINT_ORDER, "kt=10 checkpoint set changed")
    ranking = sorted(
        (
            {
                "checkpoint": checkpoint,
                "field": field,
                **kt10[checkpoint][field],
            }
            for checkpoint in CHECKPOINT_ORDER
            for field in FIELD_ORDER
        ),
        key=lambda row: (
            -float(row["max_abs"]),
            CHECKPOINT_ORDER.index(str(row["checkpoint"])),
            FIELD_ORDER.index(str(row["field"])),
        ),
    )

    return {
        "format": "nemo-testcase-l4-orca2-round66-independent-start-v1",
        "status": "PASS_INDEPENDENT_LADDER",
        "claim_label": "independent",
        "execution": trajectory["execution"],
        "initial_state": initial,
        "first_non_bit_statement": first,
        "checkpoints": 40,
        "field_rows": rows_scored,
        "kt10_rows": kt10,
        "kt10_ranking_by_max_abs": ranking,
        "unmeasured_features": list(EXPECTED_UNMEASURED),
        "next": "ORCA2_MONTH_SCALE_MAGNITUDE_RANKING",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-json", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plant",
        choices=("none", "bridge", "entry-count", "checkpoint-drop"),
        default="none",
    )
    args = parser.parse_args()
    try:
        report = classify(
            json.loads(args.ladder_json.read_text()), plant=args.plant
        )
        report["input"] = {
            "path": str(args.ladder_json),
            "sha256": sha256(args.ladder_json),
        }
        report["worktree"] = worktree_stamp()
    except (GateError, KeyError, OSError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_INDEPENDENT_LADDER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
