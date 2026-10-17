#!/usr/bin/env python3
"""Round 307: the V1/V2 vertical-mixing statements measured under Decision 122.

Rung 10 is the only ORCA2 card that runs TKE.  Arms (same tree):
  control  every V1 selector off (the card before round 307's selection).
  N2 SH2E SOLVE FOLD RHS R3  one statement (R3 = the r3u/r3v pair).
  ALL      all six.
Non-TKE cards (rung 0, OMT-1..4) have only ``shipped``.
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
    nemo_testcase_l4_orca2_round305_drag_d122_gate as r305,
)

GateError = r276.GateError
require = r276.require
STATEMENTS = {
    "N2": ("nemo_bn2_source_rounded",),
    "SH2E": ("nemo_sh2_step_entry_ssh",),
    "SOLVE": ("nemo_tke_solve_source_rounded",),
    "FOLD": ("nemo_sh2_fold_top_v",),
    "RHS": ("nemo_tke_matrix_rhs_source_rounded",),
    "R3": ("nemo_sh2_r3_source_rounded", "nemo_sh2_r3_stored_reciprocal"),
}
STATEMENTS["ALL"] = tuple(f for flags in STATEMENTS.values() for f in flags)
V1_FLAGS = STATEMENTS["ALL"]
ARMS = {"rung10": ("control", *STATEMENTS)}
ARMS.update({rung: ("shipped",) for rung in
             ("rung0", "omt1", "omt2", "omt3", "omt4")})


def selected_card(card, *, arm: str, rung: str, plant: str = "none"):
    require(arm in ARMS[rung], f"{rung}: arm {arm} is not defined")
    cfg = card.recipe.model_config
    vm = cfg.physics.vertical_mixing
    if arm == "shipped":
        return card
    require(vm.scheme == "tke", f"{rung}: arm {arm} needs a TKE card")
    want = set(STATEMENTS.get(arm, ()))
    if plant == "retained-selector" and arm == "control":
        want = {V1_FLAGS[0]}
    tke = vm.tke._replace(**{f: (f in want) for f in V1_FLAGS})
    cfg = cfg._replace(physics=cfg.physics._replace(
        vertical_mixing=vm._replace(tke=tke)))
    on = {f for f in V1_FLAGS if getattr(tke, f)}
    require(on == set(STATEMENTS.get(arm, ())),
            f"{rung}: arm {arm} carries {sorted(on)}")
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure")
    parser.add_argument("--rung", choices=tuple(sorted(ARMS)))
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
    parser.add_argument("--resolved-config", nargs="+")
    args = parser.parse_args()
    try:
        if args.resolved_config:
            original, r305.selected_card = (
                r305.selected_card,
                lambda card, *, arm, rung, plant="none": card)
            try:
                result = dict(r305.resolved_configs(
                    {k: Path(v) for k, v in
                     (item.split("=", 1) for item in args.resolved_config)}),
                    status="RESOLVED_CONFIG")
            finally:
                r305.selected_card = original
        elif args.measure:
            require(all((args.rung, args.label, args.deck_root, args.twin_a,
                         args.expect_commit)), "measurement arguments incomplete")

            def select(base, *, candidate, rung):
                return selected_card(base, arm=args.measure, rung=rung,
                                     plant=args.plant)

            original, r285.selected_card = r285.selected_card, select
            try:
                result = r285.measure(args.rung, args.label, args.deck_root,
                                      args.twin_a,
                                      args.measure not in ("control", "shipped"),
                                      args.expect_commit)
            finally:
                r285.selected_card = original
            result.update(format="nemo-testcase-l4-orca2-round307-vmix-v1",
                          arm=args.measure,
                          selected_statements=list(
                              STATEMENTS.get(args.measure, ())))
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
