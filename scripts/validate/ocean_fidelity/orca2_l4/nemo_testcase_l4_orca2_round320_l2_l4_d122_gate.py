#!/usr/bin/env python3
"""Round 320: lane B's L2 (isoneutral) and lane A's L4 (geothermal) under Decision 122.

Arms (applied only on the cards whose NEMO deck sets the piece; on the
others an arm is the shipped card, stated, never silently skipped):
  control  the card as shipped.
  ISO      L2's certified isoneutral selection (orca2_l2_isoneutral_gm_redi).
           Decks: rung 0 (namelist_cfg:319-326) and rung 10
           (namelist_cfg:301-308); OMT-1..4 state ln_traldf_off.
  GEO      L4's stage-3 geothermal statement (rk3_stage3_geothermal_nemo_written).
           Deck: rung 10 only (ln_trabbc, nn_geoflx = 2, namelist_cfg:257-264).
  BOTH     ISO and GEO.
  D122     the card as shipped after this round's selection.
"control" and the arms always start from the card before the selection;
on the selected tree rung 10's isoneutral block is rebuilt for them
(pre_round320_gm_redi).
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
DECK_PIECES = {"rung0": ("ISO",), "omt1": (), "omt2": (), "omt3": (),
               "omt4": (), "rung10": ("ISO", "GEO")}
ARM_PIECES = {"control": (), "ISO": ("ISO",), "GEO": ("GEO",),
              "BOTH": ("ISO", "GEO")}
ARMS = (*ARM_PIECES, "D122")
# This round's selection, frozen after measuring (receipt): ISO on rung 10.
D122_SELECTED = {rung: () for rung in RUNGS} | {"rung10": ("ISO",)}


def _apply(cfg, pieces):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        orca2_l2_isoneutral_gm_redi,
    )
    if "ISO" in pieces:
        cfg = cfg._replace(gm_redi=orca2_l2_isoneutral_gm_redi())
    if "GEO" in pieces:
        cfg = cfg._replace(rk3_stage3_geothermal_nemo_written=True)
    return cfg


def pre_round320_gm_redi():
    """Rung 10's isoneutral block before round 320 (the generic card block).

    Default GMRediConfig plus these twelve fields, read off the card at
    246f73429; the control ladder rerun on the selected tree reproduces the
    pre-selection control row for row (receipt)."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    return GMRediConfig()._replace(
        kappa_GM=0.0, slope_scheme="nemo_iso_lap", slope_density="neutral",
        slope_positions="nemo_native", nemo_mld_slope_ramp=True,
        mld_criterion="n2_integral", slope_metric_evaluation="nemo_reciprocal",
        slope_face_thickness_evaluation="nemo_qco_live",
        slope_depth_evaluation="nemo_qco_live_literal", nemo_slope_shapiro=True,
        redi_a33_evaluation="nemo_literal", implicit_K33=True)


def selected_card(card, *, arm: str, rung: str):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        orca2_l2_isoneutral_gm_redi,
    )
    require(arm in ARMS and rung in RUNGS, f"{rung}: arm {arm} is not defined")
    cfg = card.recipe.model_config
    iso = cfg.gm_redi == orca2_l2_isoneutral_gm_redi()
    geo = cfg.rk3_stage3_geothermal_nemo_written
    shipped = tuple(p for p, on in (("ISO", iso), ("GEO", geo)) if on)
    # Tolerates the tree before the selection (3da2f6835: nothing shipped)
    # and after it (D122_SELECTED shipped); anything else refuses.
    require(shipped in ((), D122_SELECTED[rung]),
            f"{rung}: shipped card carries {shipped}")
    if arm == "D122":
        require(shipped == D122_SELECTED[rung],
                f"{rung}: D122 needs the selected card, found {shipped}")
        return card
    if shipped:
        # "control" is the card before round 320's selection.
        cfg = cfg._replace(gm_redi=pre_round320_gm_redi())
    pieces = tuple(p for p in ARM_PIECES[arm] if p in DECK_PIECES[rung])
    new = _apply(cfg, pieces)
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
                format="nemo-testcase-l4-orca2-round320-l2-l4-v1",
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
