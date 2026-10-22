#!/usr/bin/env python3
"""Round 323: lane A's L8 stage-3 Krhs route, then L3 and L5, under Decision 122.

Pieces (each set only by rung 10's record: ln_trabbl, ln_trabbc, ln_tradmp,
ln_ldfeiv and ln_mle are T there and F on rung 0 and OMT-1..4; on those
cards an arm is the shipped card, stated, never silently skipped):
  L8   G9's stage-3 Krhs walk carrying the diffusive BBL, isoneutral
       diffusion, geothermal and interior damping (gridladder ce9636bc1f).
  L3   lane B's eddy-induced velocity (orca2_l3_eiv_gm_redi; needs L8).
  L5   lane B's mixed-layer eddies (orca2_l5_mle; lane B certified it on G9).
Arms: control (the card before this round's selection), L8, L8+L3, L8+L5,
L8+L3+L5, and D122 (the card as shipped after the selection).
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
DECK_PIECES = {rung: () for rung in RUNGS} | {"rung10": ("L8", "L3", "L5")}
ARM_PIECES = {"control": (), "L8": ("L8",), "L8L3": ("L8", "L3"),
              "L8L5": ("L8", "L5"), "L8L3L5": ("L8", "L3", "L5")}
ARMS = (*ARM_PIECES, "D122")
# This round's selection, frozen after measuring (receipt).
D122_SELECTED = {rung: () for rung in RUNGS} | {"rung10": ("L8", "L5")}


def with_pieces(cfg, pieces):
    """The shipped rung-10 config with the named pieces selected."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        orca2_l3_eiv_gm_redi,
        orca2_l5_mle,
        orca2_l8_krhs_selection,
    )
    if "L8" in pieces:
        cfg = cfg._replace(**orca2_l8_krhs_selection())
    if "L3" in pieces:
        cfg = cfg._replace(gm_redi=orca2_l3_eiv_gm_redi())
    if "L5" in pieces:
        cfg = cfg._replace(physics=cfg.physics._replace(mle=orca2_l5_mle()))
    return cfg


# The rung-10 values before this round's selection (round 322's card).
PRE_SELECTION = {
    "rk3_stage3_tracer_krhs_nemo_order": False,
    "rk3_stage3_geothermal_nemo_written": False,
    "rk3_stage3_geothermal_after_ldf_nemo_order": False,
    "rk3_stage3_damping_after_geothermal_nemo_order": False,
    "rk3_stage3_l8_nemo_order": False,
    "bbl_nemo_stage3_krhs": False,
    "bbl_nemo_pivot_ghost": False,
    "rk3_end_of_stage_pivot_fold": True,
    "bbl_adv_option": 0, "bbl_diffusive_option": 1,
    "bbl_aht_m2_s": 1000.0, "bbl_gamma_s": 0.0,
}


def without_selection(card):
    """Rung 10 as round 322 shipped it: this round's selectors reverted."""
    cfg = card.recipe.model_config
    cfg = cfg._replace(**PRE_SELECTION,
                       physics=cfg.physics._replace(mle=None))
    return card._replace(
        recipe=card.recipe._replace(model_config=cfg, physics_config=cfg.physics,
                                    tracer_damping=None),
        bbl_gamma_s=0.0)


def shipped_pieces(cfg):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        orca2_l3_eiv_gm_redi,
        orca2_l5_mle,
        orca2_l8_krhs_selection,
    )
    l8 = all(getattr(cfg, k) == v for k, v in orca2_l8_krhs_selection().items())
    return tuple(p for p, on in (
        ("L8", l8), ("L3", cfg.gm_redi == orca2_l3_eiv_gm_redi()),
        ("L5", cfg.physics.mle == orca2_l5_mle())) if on)


def selected_card(card, *, arm: str, rung: str):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        orca2_l2_isoneutral_gm_redi,
    )
    require(arm in ARMS and rung in RUNGS, f"{rung}: arm {arm} is not defined")
    cfg = card.recipe.model_config
    shipped = shipped_pieces(cfg)
    require(shipped in ((), D122_SELECTED[rung]),
            f"{rung}: shipped card carries {shipped}")
    if arm == "D122":
        require(shipped == D122_SELECTED[rung],
                f"{rung}: D122 needs the selected card, found {shipped}")
        return card
    if shipped:
        card = without_selection(card)
        cfg = card.recipe.model_config
        require(shipped_pieces(cfg) == (), f"{rung}: control rebuild kept pieces")
    require(cfg.gm_redi == orca2_l2_isoneutral_gm_redi() or rung != "rung10",
            "rung 10 control must carry round 320's L2")
    pieces = tuple(p for p in ARM_PIECES[arm] if p in DECK_PIECES[rung])
    new = with_pieces(cfg, pieces)
    if not pieces:
        require(new == card.recipe.model_config,
                f"{rung}: arm {arm} moved a card whose deck sets none of its "
                "pieces")
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
                format="nemo-testcase-l4-orca2-round323-l8-l3-l5-v1",
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
