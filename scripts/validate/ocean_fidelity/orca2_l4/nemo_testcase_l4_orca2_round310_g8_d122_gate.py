#!/usr/bin/env python3
"""Round 310: rung G8's certified dyn_ldf_lev statements under Decision 122.

Every arm applies to all seven card/label pairs (same tree, all relative to
the shipped card):
  control  the card as shipped.
  AREA     stored F reciprocal r1_e1e2f in the curl term.
  FOLDR    lbc-complete V and F fold rows in dyn_ldf_lev.
  ORDER    stp_2D's stage-1 Krhs order hpg -> ldf -> vor -> keg -> zad.
  R3F      stage 3's F ratio = mean of the two stored ratios.
  G8       all four together.
  SEL      AREA + R3F (the candidates after the first batch; see addendum).
  D122     the card as shipped after this round's selection.
OMT-1 and OMT-2 run no lateral momentum diffusion (their decks set
ln_dynldf_OFF), so NEMO never reaches dynldf_lev there: D122 (b) fails and
the statements stay off on those two cards whatever (c)/(d) say.
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
)

GateError = r276.GateError
require = r276.require
STATEMENTS = {
    "AREA": {"nemo_ldf_stored_f_area": True},
    "FOLDR": {"nemo_ldf_fold_rows": True},
    "ORDER": {"rk3_stage1_ldf_source_order": True},
    "R3F": {"nemo_ldf_stage_r3f": True},
}
STATEMENTS["G8"] = {k: v for arm in tuple(STATEMENTS) for k, v in
                    STATEMENTS[arm].items()}
# FOLDR leans away and grows rung 10's kt-10 sea-level error; ORDER's
# stage-1 block replaces G1b's KEG contraction guard on vector cards.
STATEMENTS["SEL"] = {**STATEMENTS["AREA"], **STATEMENTS["R3F"]}
RUNGS = ("rung0", "omt1", "omt2", "omt3", "omt4", "rung10")
ARMS = ("control", "D122", *STATEMENTS)
# This round's selection: SEL where the deck runs dynldf_lev; nothing on
# OMT-1/OMT-2 (ln_dynldf_OFF, D122 (b) fails).
D122_SELECTED: dict[str, dict] = {
    rung: ({} if rung in ("omt1", "omt2") else dict(STATEMENTS["SEL"]))
    for rung in RUNGS}


def selected_card(card, *, arm: str, rung: str, plant: str = "none"):
    require(arm in ARMS and rung in RUNGS, f"{rung}: arm {arm} is not defined")
    cfg = card.recipe.model_config
    if arm == "D122":
        moved = {k: getattr(cfg, k) for k in STATEMENTS["G8"]
                 if getattr(cfg, k)}
        require(moved == D122_SELECTED[rung],
                f"{rung}: shipped card carries {moved}, not round 310's")
        return card
    want = {} if arm == "control" else STATEMENTS[arm]
    if plant == "retained-selector" and arm == "control":
        want = STATEMENTS["FOLDR"]
    new = cfg._replace(**want)
    for key in STATEMENTS["G8"]:
        expect = want.get(key, getattr(cfg, key))
        require(getattr(new, key) == expect,
                f"{rung}: arm {arm} carries {key}={getattr(new, key)!r}")
    if arm == "control" or plant != "none":
        require(new == cfg, f"{rung}: control arm moved the card")
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
    parser.add_argument("--plant", choices=("none", "exact-loss",
                                             "retained-selector"),
                        default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.label, args.deck_root, args.twin_a,
                         args.expect_commit)), "measurement arguments incomplete")

            def select(base, *, candidate, rung):
                return selected_card(base, arm=args.measure, rung=rung,
                                     plant=args.plant)

            original, r285.selected_card = r285.selected_card, select
            try:
                result = r285.measure(args.rung, args.label, args.deck_root,
                                      args.twin_a, args.measure != "control",
                                      args.expect_commit)
            finally:
                r285.selected_card = original
            result.update(format="nemo-testcase-l4-orca2-round310-g8-v1",
                          arm=args.measure,
                          selected_statements=(
                              D122_SELECTED[args.rung]
                              if args.measure == "D122"
                              else STATEMENTS.get(args.measure, {})))
        elif args.classify:
            result = r276.classify(*args.classify, plant=args.plant)
        else:
            require(args.decide is not None, "no gate operation selected")
            result = r278.decide(args.decide)
    except (GateError, r276.decision96.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    if args.plant != "none":
        print(f"STATUS PLANT-GREEN: {args.plant} plant did not fire")
        return 3
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0 if result.get("d122_c_and_d", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
