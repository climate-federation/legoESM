#!/usr/bin/env python3
"""Walk the source-ordered FCT predictor that makes round-139 paft infinite."""

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
    nemo_testcase_l4_orca2_round134_step36_fct_gate as registry,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
)


TARGET = (87, 159, 4)
RETURNED_TARGET = (86, 159, 0)
SOURCE_FIELDS = tuple(
    field for group, fields in registry.TRACE_GROUPS[:6] for field in fields)
PLANTS = (
    "none", "registry", "passivity", "source-order", "paft-link",
    "finite-prefix",
)


class GateError(RuntimeError):
    """The upstream-predictor record violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _target_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for group, fields in registry.TRACE_GROUPS[:6]:
        details = report["groups"][group]["details"]["T"]
        for field in fields:
            row = details[field]
            rows.append({
                "group": group,
                "field": field,
                "indices": row["target_indices"],
                "values": row["target_values"],
                "nonfinite": int(row["target_nonfinite"]),
            })
    return rows


def _first_nonfinite(rows: list[dict[str, object]]) -> str | None:
    return next((str(row["field"]) for row in rows if row["nonfinite"]), None)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "registry":
        report["source_field_order"][0] = "wrong"
    elif plant == "passivity":
        report["observer_state_equal"]["T"] = False
    elif plant == "source-order":
        report["first_target_nonfinite_field"] = "wrong"
    elif plant == "paft-link":
        report["round139_link"]["paft_nonfinite"] = False
    elif plant == "finite-prefix":
        report["target_rows"][0]["nonfinite"] = 1

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "predictor walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("steps_completed_before_walk") == 35,
            "walk did not start from step 35")
    require(tuple(report.get("target", ())) == TARGET,
            "predictor target changed")
    require(tuple(report.get("source_field_order", ())) == SOURCE_FIELDS,
            "source field registry changed")
    require(report.get("ordinary_repeat_state_equal") == {
        name: True for name in passive.FIELDS},
        "ordinary step-36 repeat changed bits")
    require(all(report.get("observer_state_equal", {}).values()),
            "passive side output moved ordinary state")
    require(report.get("returned_first_nonfinite") == {
        "field": "T", "index": list(RETURNED_TARGET), "value": "nan"},
        "returned step-36 boundary changed")
    require(report.get("round139_link") == {
        "cell": list(TARGET), "pbef_nonfinite": False,
        "paft_nonfinite": True, "paft": "inf"},
        "round-139 paft boundary changed")

    rows = report.get("target_rows", [])
    require([row.get("field") for row in rows] == list(SOURCE_FIELDS),
            "target rows are not source ordered")
    first = _first_nonfinite(rows)
    require(first == report.get("first_target_nonfinite_field"),
            "first non-finite field is not source ordered")
    require(first is not None, "all predictor intermediates stayed finite")
    first_index = SOURCE_FIELDS.index(first)
    require(all(not row["nonfinite"] for row in rows[:first_index]),
            "a source field before the named owner is non-finite")

    averaged = first in ("average_u", "average_v", "average_w")
    report["prediction_ledger"] = {
        "R140-P1": {"status": "CONFIRMED",
                     "observed": "ordinary state exact; controls fired"},
        "R140-P2": {"status": "CONFIRMED" if averaged else "REFUTED",
                     "predicted": "incident averaged horizontal face flux",
                     "observed": first},
        "R140-P3": {"status": "UNMEASURED",
                     "observed": "numerator/divisor split follows the owner"},
        "R140-P4": {"status": "UNMEASURED", "observed": "EVD plant pending"},
        "R140-P5": {"status": "UNMEASURED", "observed": "shared gates pending"},
    }
    return {**report, "status": "PASS_ROUND140_UPSTREAM_PREDICTOR_WALK"}


def measure(deck_root: Path, expect_commit: str,
            round139_json: Path) -> dict[str, object]:
    old_target = passive.TARGET
    passive.TARGET = TARGET
    try:
        report = passive.measure(deck_root, expect_commit)
    finally:
        passive.TARGET = old_target
    previous = json.loads(round139_json.read_text())
    selected = previous["selected_sources"]
    rows = _target_rows(report)
    report.update({
        "format": "nemo-testcase-l4-orca2-round140-upstream-predictor-v1",
        "target": list(TARGET),
        "source_field_order": list(SOURCE_FIELDS),
        "target_rows": rows,
        "first_target_nonfinite_field": _first_nonfinite(rows),
        "round139_link": {
            "cell": selected["index"],
            "pbef_nonfinite": selected["pbef_nonfinite"],
            "paft_nonfinite": selected["paft_nonfinite"],
            "paft": selected["paft"],
        },
        "compiled_citations": {
            "first_fluxes": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:500-526",
            "midpoint": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:528-539",
            "averaged_fluxes": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:562-596",
            "after_update": "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:598-610",
        },
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--round139-json", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.expect_commit is None
                    and args.round139_json is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root is not None and args.expect_commit
                    and args.round139_json is not None,
                    "runtime mode requires deck, commit, and round-139 record")
            raw = measure(args.deck_root, args.expect_commit, args.round139_json)
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
    print("STATUS PASS_ROUND140_UPSTREAM_PREDICTOR_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
