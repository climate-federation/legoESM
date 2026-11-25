#!/usr/bin/env python3
"""Round 324: DECISION 150, the rung-10 comparison runs NEMO's implicit
ice-ocean top drag with the recorded coefficient, under Decision 122.

Arms: control (the shipped card, top drag off as DECISION 133 keeps it) and
drag (rung 10: the ladder's ``comparison_card``, which selects the existing
G5 path; the step reads the recorded rCdU_ice, iceupdate.F90:391).  Rung 0
and OMT-1..4 carry no replayed ice record: their drag arm is the shipped
card, stated, never silently skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round278_d122_landing_gate as r278,
    nemo_testcase_l4_orca2_round285_g4_d122_gate as r285,
    nemo_testcase_l4_orca2_round318_fold_d122_gate as r318,
)

GateError = r276.GateError
require = r276.require
RUNGS = ("rung0", "omt1", "omt2", "omt3", "omt4", "rung10")
ARMS = ("control", "drag")
FIELDS = ("ssh", "T", "S", "u", "v")


def selected_card(card, *, arm: str, rung: str):
    require(arm in ARMS and rung in RUNGS, f"{rung}: arm {arm} is not defined")
    require(not card.recipe.model_config.implicit_ice_ocean_top_drag,
            f"{rung}: shipped card selects the ice-ocean top drag")
    if arm == "drag" and rung == "rung10":
        return rung10.comparison_card(card)
    return card


def kt10(paths: list[Path]) -> dict:
    """kt-10 entry (both ranks) and stage-3 (rank 0) rows of each arm file."""
    out = {}
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = {(r["checkpoint"], r["field"]): r for r in doc["ladder"]["rows"]
                if r["kt"] == 10}
        require(len(rows) == 20, f"{path.name}: kt-10 rows incomplete")
        out[path.name] = {
            f"{cp}_{f}": {"max_abs": rows[(cp, f)]["max_abs"],
                          "rms": rows[(cp, f)]["rms"],
                          "count": rows[(cp, f)]["count"]}
            for cp in ("entry", "stage3") for f in FIELDS}
    return {"format": "nemo-testcase-l4-orca2-round324-kt10-v1",
            "status": "PASS_KT10_ROWS", "arms": out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=ARMS)
    parser.add_argument("--rung", choices=RUNGS)
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--decide", nargs="+", type=Path)
    parser.add_argument("--kt10", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.label, args.deck_root, args.twin_a,
                         args.expect_commit)), "measurement arguments incomplete")

            def select(base, *, candidate, rung):
                return selected_card(base, arm=args.measure, rung=rung)

            original, r285.selected_card = r285.selected_card, select
            try:
                result = r285.measure(args.rung, args.label, args.deck_root,
                                      args.twin_a, args.measure != "control",
                                      args.expect_commit)
            finally:
                r285.selected_card = original
            result.update(
                format="nemo-testcase-l4-orca2-round324-top-drag-v1",
                arm=args.measure,
                selected_statements=(
                    ["implicit_ice_ocean_top_drag"]
                    if args.measure == "drag" and args.rung == "rung10"
                    else []))
        elif args.classify:
            result = r318.classify_later_first(*args.classify, plant="none")
        elif args.kt10:
            result = kt10(args.kt10)
        else:
            require(args.decide is not None, "no gate operation selected")
            result = r278.decide(args.decide)
    except (GateError, r276.decision96.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0 if result.get("d122_c_and_d", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
