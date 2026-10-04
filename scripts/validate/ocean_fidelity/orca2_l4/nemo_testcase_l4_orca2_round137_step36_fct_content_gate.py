#!/usr/bin/env python3
"""Walk the level-3 FCT value that feeds rung-0's step-36 ZDF failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round131_step16_walk_gate as state_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round134_step36_fct_gate as registry,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
)


TARGET = (86, 159, 3)
RETURNED_TARGET = (86, 159, 0)
FIELDS = state_gate.FIELDS
TRACE_FIELD_ORDER = passive.TRACE_FIELD_ORDER
GROUP_ORDER = passive.GROUP_ORDER
TRACE_GROUPS = registry.TRACE_GROUPS
PLANTS = ("none", "source-order", "passivity", "support", "zdf-link")


class GateError(RuntimeError):
    """The level-3 FCT/content walk violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _first_target_field(report: dict[str, object]) -> str | None:
    for group, fields in TRACE_GROUPS[:-1]:
        details = report["groups"][group]["details"]["T"]
        for field in fields:
            if int(details[field]["target_nonfinite"]):
                return field
    if int(report["groups"]["caller_advection_content"]
           ["target_nonfinite"]["T"]):
        return "caller_content"
    return None


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "source-order":
        report["first_target_nonfinite_field"] = "first_u"
    elif plant == "passivity":
        report["observer_state_equal"]["T"] = False
    elif plant == "support":
        report["groups"]["antidiffusive_flux"]["active_count"] += 1
    elif plant == "zdf-link":
        report["round136_pre_zdf"]["first_nonfinite"] = [0, 0, 0]

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "FCT/content walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "walk did not start from step 35")
    require(tuple(report.get("target", ())) == TARGET,
            "level-3 target changed")
    require(tuple(report.get("trace_field_order", ())) == TRACE_FIELD_ORDER,
            "FCT trace field order changed")
    require(tuple(report.get("group_order", ())) == GROUP_ORDER,
            "FCT group order changed")
    require(report.get("ordinary_repeat_state_equal") == {
        name: True for name in FIELDS},
        "ordinary step-36 repeat changed bits")
    require(all(report.get("observer_state_equal", {}).values()),
            "passive FCT side output moved an ordinary state leaf")
    require(report.get("returned_first_nonfinite") == {
        "field": "T", "index": list(RETURNED_TARGET), "value": "nan"},
        "step-36 returned failure changed")
    require(report.get("side_output_type") == "_NEMOWSFCTInputTrace"
            and report.get("side_output_field") == "mass_flux_w",
            "FCT side-output scope changed")

    for group, fields in TRACE_GROUPS:
        row = report["groups"][group]
        require(tuple(row["fields"]) == fields,
                f"{group}: field registry changed")
        require(int(row["nonfinite_total"]) == sum(
            int(value) for value in row["nonfinite"].values()),
            f"{group}: non-finite census disagrees")
        require(int(row["target_nonfinite_total"]) == sum(
            int(value) for value in row["target_nonfinite"].values()),
            f"{group}: target census disagrees")
        expected_active = sum(int(row["support_count"][name]) for name in fields)
        require(int(row["active_count"]) == expected_active,
                f"{group}: active support census disagrees")

    first_field = _first_target_field(report)
    require(first_field == report.get("first_target_nonfinite_field"),
            "first target non-finite field is not source ordered")
    require(first_field is not None,
            "FCT trace did not reach the level-3 non-finite boundary")
    require(report["groups"]["final_rhs"]["target_nonfinite"]["T"] > 0,
            "FCT final RHS is finite at the level-3 target")
    require(report["groups"]["caller_advection_content"]
            ["target_nonfinite"]["T"] > 0,
            "caller advection content is finite at the level-3 target")
    require(report.get("round136_pre_zdf", {}).get("first_nonfinite")
            == list(TARGET),
            "round-136 pre-ZDF boundary no longer starts at the target")

    early = set(TRACE_FIELD_ORDER[:10])
    p2_confirmed = first_field not in early
    report["prediction_ledger"] = {
        "R137-P1": {
            "status": "CONFIRMED",
            "observed": "all ordinary state leaves bit-identical",
        },
        "R137-P2": {
            "status": "CONFIRMED" if p2_confirmed else "REFUTED",
            "predicted": "after rhs_after_up",
            "observed": first_field,
        },
        "R137-P3": {
            "status": "CONFIRMED",
            "observed": "FCT final RHS -> caller content -> pre-ZDF level 3",
        },
        "R137-P4": {
            "status": "UNMEASURED",
            "observed": "shared gates pending",
        },
    }
    return {**report, "status": "PASS_ROUND137_STEP36_FCT_CONTENT_WALK"}


def measure(deck_root: Path, expect_commit: str,
            round136_json: Path) -> dict[str, object]:
    old_target = passive.TARGET
    passive.TARGET = TARGET
    try:
        report = passive.measure(deck_root, expect_commit)
    finally:
        passive.TARGET = old_target

    previous = json.loads(round136_json.read_text())
    content_row = previous["rows"]["content_T"]
    report.update({
        "format": "nemo-testcase-l4-orca2-round137-step36-fct-content-v1",
        "target": list(TARGET),
        "first_target_nonfinite_field": _first_target_field(report),
        "round136_pre_zdf": {
            "source": str(round136_json),
            "nonfinite": int(content_row["nonfinite"]),
            "first_nonfinite": content_row["first_nonfinite"],
        },
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--round136-json", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.expect_commit is None
                    and args.round136_json is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root is not None and args.expect_commit
                    and args.round136_json is not None,
                    "runtime mode requires deck, commit, and round-136 record")
            raw = measure(args.deck_root, args.expect_commit, args.round136_json)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, passive.GateError, OSError, KeyError, TypeError,
            ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND137_STEP36_FCT_CONTENT_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
