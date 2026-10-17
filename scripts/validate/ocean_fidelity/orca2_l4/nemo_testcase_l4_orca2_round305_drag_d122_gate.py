#!/usr/bin/env python3
"""Round 305: Decision-122 arms for the G5 top drag + G7 bottom-drag statements.

Arms (same tree, one card per arm):
  control  the card as shipped.
  B        G7's bottom residual + dyn_zdf bottom RHS without the ice top drag
           (NEMO's nn_ice=0 program, zdfdrg.f90:380); needs the linear bundle.
  L        NEMO's ln_lin linear implicit bundle (the card's drag LAW changes).
  LB       L + B.
  LBT      L + G5's ice-ocean top drag, which also selects B (G7's composition).
OMT-2..4 already run the linear bundle; rung 0 and rung 10 do not, so every
rung-0 / rung-10 arm except control changes the drag law.  OMT-1 is ln_drg_OFF.
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
BOTTOM = "nemo_bottom_drag_written_statements"
TOP = "implicit_ice_ocean_top_drag"
CARDS = r278.CARDS
ARMS = {
    "omt2": ("control", "B"), "omt3": ("control", "B"), "omt4": ("control", "B"),
    "omt1": ("control",),
    "rung0": ("control", "L", "LB"),
    "rung10": ("control", "L", "LB", "LBT"),
}


def _linear(cfg):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        _nemo_linear_bottom_drag,
    )
    return _nemo_linear_bottom_drag(cfg)


def selected_card(card, *, arm: str, rung: str, plant: str = "none"):
    require(arm in ARMS[rung], f"{rung}: arm {arm} is not defined")
    cfg = card.recipe.model_config
    require(not getattr(cfg, BOTTOM) and not getattr(cfg, TOP),
            f"{rung}: shipped card already selects a round-305 statement")
    if "L" in arm:
        cfg = _linear(cfg)
    if "B" in arm:
        cfg = cfg._replace(**{BOTTOM: True})
    if "T" in arm:
        cfg = cfg._replace(**{TOP: True})
    if plant == "retained-selector" and arm == "control":
        cfg = cfg._replace(**{BOTTOM: True})
    if arm == "control":
        require(not getattr(cfg, BOTTOM) and not getattr(cfg, TOP),
                f"{rung}: control retains a round-305 selector")
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", choices=("control", "B", "L", "LB", "LBT"))
    parser.add_argument("--rung", choices=tuple(sorted(ARMS)))
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    # arm-vs-arm (one variable): the first arm stands in as the control.
    parser.add_argument("--classify-arms", nargs=2, type=Path)
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
            result.update(format="nemo-testcase-l4-orca2-round305-drag-v1",
                          arm=args.measure, selected_statements=[
                              n for n, on in ((BOTTOM, "B" in args.measure),
                                              (TOP, "T" in args.measure),
                                              ("ln_lin bundle", "L" in args.measure))
                              if on])
        elif args.classify:
            result = r276.classify(*args.classify, plant=args.plant)
        elif args.classify_arms:
            base, cand = (json.loads(p.read_text(encoding="utf-8"))
                          for p in args.classify_arms)
            require(base["arm"] != "control" and cand["arm"] != base["arm"],
                    "--classify-arms compares two non-control arms")
            result = r276.classify_documents(
                dict(base, arm="control"), cand, plant=args.plant)
            result["baseline_arm"] = base["arm"]
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
