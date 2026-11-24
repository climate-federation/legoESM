#!/usr/bin/env python3
"""Round 321: lane B's L3 (eddy-induced velocity) under Decision 122.

Arms (applied only on the cards whose NEMO record runs ln_ldfeiv = T --
rung 10 alone; rung 0 and OMT-1..4 record ln_ldfeiv = F -- on the others an
arm is the shipped card, stated, never silently skipped):
  control  the card as shipped before this round's selection (rung 10: L2).
  L3       L3's certified selection (orca2_l3_eiv_gm_redi).
  D122     the card as shipped after this round's selection.
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
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round278_d122_landing_gate as r278,
    nemo_testcase_l4_orca2_round285_g4_d122_gate as r285,
    nemo_testcase_l4_orca2_round318_fold_d122_gate as r318,
)

GateError = r276.GateError
require = r276.require
RUNGS = ("rung0", "omt1", "omt2", "omt3", "omt4", "rung10")
DECK_PIECES = {rung: () for rung in RUNGS} | {"rung10": ("L3",)}
ARM_PIECES = {"control": (), "L3": ("L3",)}
ARMS = (*ARM_PIECES, "D122")
# This round's selection, frozen after measuring (receipt).
D122_SELECTED = {rung: () for rung in RUNGS} | {"rung10": ("L3",)}


def selected_card(card, *, arm: str, rung: str):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        orca2_l2_isoneutral_gm_redi,
        orca2_l3_eiv_gm_redi,
    )
    require(arm in ARMS and rung in RUNGS, f"{rung}: arm {arm} is not defined")
    cfg = card.recipe.model_config
    l3 = cfg.gm_redi == orca2_l3_eiv_gm_redi()
    shipped = ("L3",) if l3 else ()
    require(shipped in ((), D122_SELECTED[rung]),
            f"{rung}: shipped card carries {shipped}")
    if arm == "D122":
        require(shipped == D122_SELECTED[rung],
                f"{rung}: D122 needs the selected card, found {shipped}")
        return card
    if shipped:
        # "control" is the card before round 321's selection: round 320's L2.
        cfg = cfg._replace(gm_redi=orca2_l2_isoneutral_gm_redi())
    pieces = tuple(p for p in ARM_PIECES[arm] if p in DECK_PIECES[rung])
    new = cfg._replace(gm_redi=orca2_l3_eiv_gm_redi()) if pieces else cfg
    if not pieces:
        require(new == cfg, f"{rung}: arm {arm} moved a card whose deck "
                            "sets none of its pieces")
        if not shipped:
            require(new == card.recipe.model_config,
                    f"{rung}: arm {arm} moved the pre-selection card")
    return card._replace(recipe=card.recipe._replace(model_config=new))


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
                format="nemo-testcase-l4-orca2-round321-l3-v1",
                arm=args.measure,
                selected_statements=list(
                    D122_SELECTED[args.rung] if args.measure == "D122" else
                    (p for p in ARM_PIECES[args.measure]
                     if p in DECK_PIECES[args.rung])))
        elif args.classify:
            result = r318.classify_later_first(*args.classify, plant="none")
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
